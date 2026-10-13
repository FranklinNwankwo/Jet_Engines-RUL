"""
Inference: turns recent raw sensor readings for one engine into a RUL
prediction with a confidence interval and a risk flag.

This module is the single scoring function, used by `api/main.py`. The
Streamlit dashboard (`app/streamlit_app.py`) does not import it: since the
Phase 15 refactor the dashboard is a pure HTTP client that calls the API's
`/predict` endpoint, so this module runs only inside the API service. See
`export_deployment_artifacts.py` for how the artifacts this module loads
are produced.

Per the Phase 6 deployment decision: Random Survival Forest supplies the
point estimate and risk ranking (best concordance and NASA score across the
model ladder), Weibull AFT supplies the confidence interval (RSF's own
intervals under-cover and it cannot predict past its training time
ceiling). Serving two models per request is a direct consequence of the
evaluation in Phase 6, not an implementation convenience.

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

from . import conformal, features, models

# Risk-band thresholds on the RSF point estimate, in cycles. These are an
# illustrative default policy, not a value learned from data -- a real
# deployment would set these from the maintenance lead time actually
# available (how long it takes to schedule and perform a replacement),
# which this project has no data on. Documented here so they're easy to
# find and override rather than buried as magic numbers in a request handler.
RISK_THRESHOLDS = {"critical": 20, "warning": 50}

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


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

    @property
    def uses_regime_clustering(self) -> bool:
        return self.regime_kmeans is not None


def load_deployment_bundle(models_dir: Path | str) -> DeploymentBundle:
    """Load one subset's deployment bundle. `models_dir` must point at a
    subset-specific directory (e.g. `models/fd001/`, `models/fd004/`), not
    the shared `models/` parent -- each subset's artifacts live in their
    own subdirectory so they never collide (see export_deployment_artifacts.py)."""
    models_dir = Path(models_dir)
    with open(models_dir / "deployment_metadata.json") as f:
        meta = json.load(f)

    regime_kmeans = None
    regime_norm_stats = None
    if meta.get("uses_regime_clustering"):
        regime_kmeans = models.load_model(models_dir / "regime_kmeans.joblib")
        regime_norm_stats = models.load_model(models_dir / "regime_norm_stats.joblib")

    return DeploymentBundle(
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
        regime_kmeans=regime_kmeans,
        regime_norm_stats=regime_norm_stats,
        rsf_interval_correction=meta.get("rsf_interval_correction", 0.0),
    )


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


def predict(bundle: DeploymentBundle, readings: pd.DataFrame) -> dict:
    """Score one engine's recent sensor history.

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

    if bundle.uses_regime_clustering:
        regime_labels = features.assign_regime_clusters(working, bundle.regime_kmeans)
        working = features.apply_regime_normalization(
            working, SENSOR_COLS, regime_labels, bundle.regime_norm_stats
        )

    feature_row = features.build_feature_matrix(working, kept_sensors=bundle.kept_sensors).iloc[[-1]]

    # --- RSF: point estimate + risk ranking ---
    rsf_out = models.predict_rul_from_survival(bundle.rsf_model, feature_row[bundle.full_feature_cols])
    predicted_rul = float(rsf_out["pred_rul"].iloc[0])
    rsf_ci_low_raw = float(rsf_out["pred_ci_low"].iloc[0])
    rsf_ci_high_raw = float(rsf_out["pred_ci_high"].iloc[0])

    # --- Weibull AFT: PRIMARY confidence interval, on the curated/scaled feature subset ---
    curated = feature_row[bundle.curated_feature_cols].copy()
    curated[bundle.curated_feature_cols] = bundle.weibull_scaler.transform(curated)
    ci_low = float(bundle.weibull_model.predict_percentile(curated, p=0.95).iloc[0])
    ci_high = float(bundle.weibull_model.predict_percentile(curated, p=0.05).iloc[0])

    # --- RSF's OWN interval, conformally calibrated (Phase 14) -- supplementary only.
    # Calibration (notebooks/12_rsf_interval_calibration.ipynb) reached ~90% coverage
    # in aggregate on the official test set, but coverage collapsed to ~27% specifically
    # on engines above RSF's training time ceiling (~cap cycles) -- a uniform additive
    # correction cannot fix a structural support problem. That's exactly the failure
    # mode Weibull's unbounded parametric interval was kept as PRIMARY to cover, so this
    # field is exposed for transparency/comparison, never as the interval a caller should
    # rely on for risk decisions.
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

    # RSF and Weibull AFT are independent models scoring the same input --
    # nothing forces the RSF point estimate to fall inside the Weibull
    # interval. Surfaced explicitly rather than left for a viewer to notice
    # unaided.
    point_outside_interval = not (ci_low_out <= predicted_rul_out <= ci_high_out)

    return {
        "subset": bundle.subset,
        "predicted_rul_cycles": predicted_rul_out,
        "confidence_interval_90": [ci_low_out, ci_high_out],
        "point_estimate_outside_interval": point_outside_interval,
        "risk_level": risk_level,
        "cycles_observed": int(readings["time_cycles"].iloc[-1]),
        "model_source": {"point_estimate": "random_survival_forest", "interval": "weibull_aft"},
        "rsf_calibrated_interval_supplementary": {
            "interval_90": [round(float(rsf_cal_low[0]), 1), round(float(rsf_cal_high[0]), 1)],
            "caveat": "Conformally calibrated for aggregate coverage, but unreliable for "
                      "engines near/beyond RSF's training time ceiling -- see README. "
                      "Informational only; confidence_interval_90 (Weibull) is authoritative.",
        },
    }