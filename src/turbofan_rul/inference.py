"""
Inference: turns recent raw sensor readings for one engine into a RUL
prediction with a confidence interval and a risk flag.

This module is the single scoring function, used by `api/main.py`, and by
the Streamlit dashboard only in its embedded (live-demo) mode; in HTTP mode
the dashboard calls the API instead. Both paths therefore produce identical
predictions. See `export_deployment_artifacts.py` for how the artifacts
this module loads are produced.

Random Survival Forest supplies the point estimate and risk ranking (best
concordance and NASA score across the model ladder, Phase 6). The 90%
interval combines both models (Phase 24, notebooks/15_interval_fix.ipynb):
its lower bound is the more cautious (smaller) of RSF's and Weibull AFT's
lower bounds, and its upper bound is Weibull's. Weibull alone (the Phase 6
choice) covered only ~55% of engines within 25 cycles of failure, always
promising more life than they had; RSF alone can't reach past its 125-cycle
training ceiling. The combination keeps Weibull's reach above the cap and
RSF's caution near failure. The bundle's `interval` setting
(`interval_calibration.IntervalCalibration`) records which interval is
served, so a future export can change it without a code change.

Multi-subset support (Phase 12d): `load_deployment_bundle` takes a
subset-specific directory (`models/fd001/`, `models/fd004/`). FD004's
bundle additionally carries a fitted regime-clustering KMeans model and
normalization stats (see `features.py`); `predict` applies regime
normalization first when a bundle has them, before any feature engineering
-- readings for a multi-regime subset must be regime-normalized the same
way training data was, or the engineered features would be dominated by
which regime the engine happens to be in, not its degradation (see
`notebooks/10_fd004_full_pipeline.ipynb`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import conformal, drift, features, interval_calibration, manifest, models

# Risk-band thresholds on the RSF point estimate, in cycles. These are an
# illustrative default policy, not a value learned from data -- a real
# deployment would set these from the maintenance lead time actually
# available (how long it takes to schedule and perform a replacement),
# which this project has no data on. Documented here so they're easy to
# find and override rather than buried as magic numbers in a request handler.
RISK_THRESHOLDS = {"critical": 20, "warning": 50}

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# The interval served when a bundle's metadata doesn't name one (Phase 24).
DEFAULT_INTERVAL = interval_calibration.IntervalCalibration(base="hybrid", mode="none")


@dataclass
class DeploymentBundle:
    subset: str
    kept_sensors: list[str]
    curated_feature_cols: list[str]
    full_feature_cols: list[str]
    raw_passthrough_cols: list[str]
    cap: int
    duration_eps: float
    rsf_model: object
    weibull_model: object
    weibull_scaler: object
    regime_kmeans: object = None
    regime_norm_stats: object = None
    rsf_interval_correction: float = 0.0
    # Bundles exported before Phase 24 have no "interval" entry in their
    # metadata; they get the Phase 24 interval, which needs no fitted
    # correction (hybrid base, no calibration), so old exports serve it too.
    interval: interval_calibration.IntervalCalibration = None
    # Content-derived version from manifest.json (see manifest.py), or
    # "unversioned" when there is no manifest or the artifacts no longer
    # match it. Reported with every prediction.
    model_version: str = manifest.UNVERSIONED
    manifest: dict | None = None
    # drift_reference.json (drift.py): used for the per-request input check
    # and by the API's drift monitor. Optional; not part of the model version.
    drift_reference: dict | None = None
    # Optional second point model (notebook 18): when present, the point
    # estimate is the average of RSF's RMST and this regressor's prediction.
    gbm_model: object = None

    def __post_init__(self):
        if self.interval is None:
            self.interval = DEFAULT_INTERVAL

    @property
    def uses_regime_clustering(self) -> bool:
        return self.regime_kmeans is not None


def load_deployment_bundle(models_dir: Path | str, verify_manifest: bool = True) -> DeploymentBundle:
    """Load one subset's deployment bundle. `models_dir` must point at a
    subset-specific directory (e.g. `models/fd001/`, `models/fd004/`), not
    the shared `models/` parent -- each subset's artifacts live in their
    own subdirectory so they never collide (see export_deployment_artifacts.py).

    With `verify_manifest=True` (the default) the artifacts are re-hashed
    and the bundle reports the manifest's version only if they still match
    it (about a second for FD001's ~226 MB forest); otherwise "unversioned"."""
    models_dir = Path(models_dir)
    with open(models_dir / "deployment_metadata.json") as f:
        meta = json.load(f)

    regime_kmeans = None
    regime_norm_stats = None
    if meta.get("uses_regime_clustering"):
        regime_kmeans = models.load_model(models_dir / "regime_kmeans.joblib")
        regime_norm_stats = models.load_model(models_dir / "regime_norm_stats.joblib")

    bundle = DeploymentBundle(
        subset=meta["subset"],
        kept_sensors=meta["kept_sensors"],
        curated_feature_cols=meta["curated_feature_cols"],
        full_feature_cols=meta["full_feature_cols"],
        raw_passthrough_cols=meta["raw_passthrough_cols"],
        cap=meta["cap"],
        duration_eps=meta["duration_eps"],
        rsf_model=models.load_model(models_dir / "rsf.joblib"),
        weibull_model=models.load_model(models_dir / "weibull_aft.joblib"),
        weibull_scaler=models.load_model(models_dir / "weibull_scaler.joblib"),
        gbm_model=(models.load_model(models_dir / "gbm.joblib")
                   if meta.get("point_model", {}).get("type") == "blend" else None),
        regime_kmeans=regime_kmeans,
        regime_norm_stats=regime_norm_stats,
        rsf_interval_correction=meta.get("rsf_interval_correction", 0.0),
        interval=(interval_calibration.IntervalCalibration.from_dict(meta["interval"])
                  if "interval" in meta else None),
    )
    bundle.drift_reference = drift.load_reference(models_dir)
    if verify_manifest:
        bundle_manifest = manifest.load_manifest(models_dir)
        bundle.model_version, _ = manifest.verified_version(models_dir, bundle_manifest)
        bundle.manifest = bundle_manifest
    return bundle


class InvalidReadingsError(ValueError):
    """Raised when the supplied sensor readings can't be scored -- missing
    columns, missing values, empty history, or non-increasing cycles. Kept as
    its own exception type so the API layer can map it to a 422 instead of a 500."""


def _validate_readings(readings: pd.DataFrame, bundle: DeploymentBundle) -> None:
    if readings.empty:
        raise InvalidReadingsError("readings must contain at least one cycle")
    # RSF was trained on the raw value of every sensor and operating setting,
    # not just the sensors Phase 1/12 flagged as informative for engineered
    # features (see export_deployment_artifacts.py for why) -- so the
    # request needs to supply the full raw telemetry schema.
    required = ["time_cycles", *bundle.raw_passthrough_cols]
    missing = [c for c in required if c not in readings.columns]
    if missing:
        raise InvalidReadingsError(f"missing required columns: {missing}")
    nan_cols = [c for c in required if readings[c].isna().any()]
    if nan_cols:
        raise InvalidReadingsError(f"missing values (NaN) in columns: {nan_cols}")
    if readings["time_cycles"].duplicated().any():
        raise InvalidReadingsError("time_cycles must be unique per reading")
    if not readings["time_cycles"].is_monotonic_increasing:
        raise InvalidReadingsError("readings must be sorted by ascending time_cycles")


def featurize(bundle: DeploymentBundle, readings: pd.DataFrame) -> pd.DataFrame:
    """Engineered features for one or more engines (needs `unit_number`),
    preprocessed exactly as the bundle's models expect: regime-normalized
    first for multi-regime subsets. Used by `predict` for a single request
    and by drift monitoring for whole fleets."""
    working = readings
    if bundle.uses_regime_clustering:
        regime_labels = features.assign_regime_clusters(working, bundle.regime_kmeans)
        working = features.apply_regime_normalization(
            working, SENSOR_COLS, regime_labels, bundle.regime_norm_stats
        )
    return features.build_feature_matrix(working, kept_sensors=bundle.kept_sensors)


def predict(bundle: DeploymentBundle, readings: pd.DataFrame) -> dict:
    """Score one engine's recent sensor history; see `predict_with_features`."""
    return predict_with_features(bundle, readings)[0]


def predict_with_features(bundle: DeploymentBundle, readings: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Score one engine's recent sensor history. Returns (response dict, the
    one-row feature frame the models scored) -- the API passes the feature
    row to drift monitoring (`drift.py`).

    `readings` is a dataframe with `time_cycles` plus every column in
    `bundle.raw_passthrough_cols` (all 21 raw sensors + 3 operating
    settings), one row per cycle, in ascending cycle order. A single
    synthetic `unit_number` is assigned internally since the feature
    pipeline groups by engine and this function always scores one engine
    per call.

    Rolling-window features are computed over rows, not cycle numbers, so
    readings should be consecutive cycles: a gap (e.g. cycles 10, 11, 15)
    is accepted, but the windows will then span more cycles than they did
    in training.
    """
    _validate_readings(readings, bundle)

    # Fresh 0..n-1 index: the feature pipeline requires unique index labels,
    # and a caller's dataframe might not have them.
    working = readings.reset_index(drop=True)
    working["unit_number"] = 1  # single synthetic engine id for this request

    feature_row = featurize(bundle, working).iloc[[-1]]

    # --- RSF: point estimate + risk ranking ---
    rsf_out = models.predict_rul_from_survival(bundle.rsf_model, feature_row[bundle.full_feature_cols])
    rsf_point = float(rsf_out["pred_rul"].iloc[0])
    if bundle.gbm_model is not None:
        gbm_point = max(float(bundle.gbm_model.predict(feature_row[bundle.full_feature_cols])[0]), 0.0)
        predicted_rul = (rsf_point + gbm_point) / 2
    else:
        gbm_point = None
        predicted_rul = rsf_point
    rsf_ci_low_raw = float(rsf_out["pred_ci_low"].iloc[0])
    rsf_ci_high_raw = float(rsf_out["pred_ci_high"].iloc[0])

    # --- Weibull AFT interval, on the curated/scaled feature subset ---
    w_low, w_high = interval_calibration.weibull_interval(
        bundle.weibull_model, bundle.weibull_scaler, feature_row[bundle.curated_feature_cols]
    )
    weibull_low, weibull_high = float(w_low[0]), float(w_high[0])

    # --- The served 90% interval (Phase 24): combines RSF and Weibull ---
    low, high = bundle.interval.apply(
        [predicted_rul], [rsf_ci_low_raw], [rsf_ci_high_raw], [weibull_low], [weibull_high]
    )
    ci_low, ci_high = float(low[0]), float(high[0])

    # --- RSF's OWN interval, conformally calibrated (Phase 14) -- supplementary only.
    # Calibration (notebooks/12_rsf_interval_calibration.ipynb) reached ~90% coverage
    # in aggregate on the official test set, but coverage collapsed to ~27% specifically
    # on engines above RSF's training time ceiling (~cap cycles) -- a uniform additive
    # correction cannot fix a structural support problem (55% on FD004, notebook 14).
    # The served interval takes its upper bound from Weibull for exactly that reason,
    # so this field is exposed for transparency/comparison, never as the interval a
    # caller should rely on for risk decisions.
    rsf_cal_low, rsf_cal_high = conformal.apply_conformal_correction(
        np.array([rsf_ci_low_raw]), np.array([rsf_ci_high_raw]), bundle.rsf_interval_correction
    )

    if predicted_rul < RISK_THRESHOLDS["critical"]:
        risk_level = "critical"
    elif predicted_rul < RISK_THRESHOLDS["warning"]:
        risk_level = "warning"
    else:
        risk_level = "healthy"

    # Round once, and compute the outside-interval flag from the same rounded
    # numbers the response returns -- otherwise a value within 0.05 of a bound
    # could be flagged one way while the displayed numbers say the other.
    predicted_rul_out = round(predicted_rul, 1)
    ci_low_out, ci_high_out = round(ci_low, 1), round(ci_high, 1)

    # The interval's upper bound comes from Weibull AFT, an independent model
    # scoring the same input, so nothing forces the RSF point estimate below
    # it. Surfaced explicitly rather than left for a viewer to notice unaided.
    point_outside_interval = not (ci_low_out <= predicted_rul_out <= ci_high_out)

    result = {
        "subset": bundle.subset,
        "model_version": bundle.model_version,
        "predicted_rul_cycles": predicted_rul_out,
        "confidence_interval_90": [ci_low_out, ci_high_out],
        "point_estimate_outside_interval": point_outside_interval,
        "risk_level": risk_level,
        "cycles_observed": int(readings["time_cycles"].iloc[-1]),
        "model_source": {
            "point_estimate": ("blend: mean of random_survival_forest and gradient_boosting"
                               if bundle.gbm_model is not None else "random_survival_forest"),
            "interval": f"{bundle.interval.name}: lower = min(RSF, Weibull AFT), upper = Weibull AFT"
                        if bundle.interval.base == "hybrid" else bundle.interval.name,
        },
        "weibull_interval_90": [round(weibull_low, 1), round(weibull_high, 1)],
        "point_estimate_components": {"random_survival_forest": round(rsf_point, 1),
                                      "gradient_boosting": None if gbm_point is None else round(gbm_point, 1)},
        # Are this engine's inputs within what the model was trained on?
        # (drift.familiarity; None if the bundle has no drift reference)
        "input_check": drift.familiarity(bundle.drift_reference, feature_row),
        "rsf_calibrated_interval_supplementary": {
            "interval_90": [round(float(rsf_cal_low[0]), 1), round(float(rsf_cal_high[0]), 1)],
            "caveat": "Conformally calibrated for aggregate coverage, but unreliable for "
                      "engines near/beyond RSF's training time ceiling -- see README. "
                      "Informational only; confidence_interval_90 is authoritative.",
        },
    }
    return result, feature_row
