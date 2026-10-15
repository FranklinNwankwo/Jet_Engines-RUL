"""
Calibrating the deployed 90% interval (Phase 24, notebooks/15_interval_fix.ipynb).

Why this module exists
----------------------
Notebook 14 checked the deployed intervals on the official test sets and
found that they fail on opposite ends of the remaining-life range:

- the Weibull AFT interval (deployed) covers only ~55% of engines within 25
  cycles of failure, and every one of those misses promises MORE life than
  the engine has -- the dangerous direction;
- the conformally calibrated RSF interval covers all of those engines, but
  collapses above the 125-cycle training cap (27% on FD001, 55% on FD004).

Notebook 12's calibration could not see the first problem: it scored each
held-out engine once, at a random point between 30% and 90% of its life, so
almost no calibration point was close to failure. This module produces
calibration data that spans the whole life of each engine, builds the
candidate intervals, and calibrates them -- including per-band ("Mondrian")
conformal corrections, so a correction learned where an interval is fine
isn't forced onto the region where it fails.

Everything here runs on TRAINING engines only (out-of-fold). The official
test sets are used once, afterwards, to confirm the selected interval.

Candidate intervals
-------------------
Each candidate is a base interval plus a calibration mode.

Bases (all 90% nominal):
- ``weibull``: Weibull AFT 5th-95th percentiles (the previous deployment).
- ``rsf``: RSF survival-curve quantiles (bounded above by the training cap).
- ``hybrid``: lower bound = the more cautious (smaller) of the two lower
  bounds; upper bound = Weibull's (which has support above the cap).

Calibration modes:
- ``none``: the base interval as it is.
- ``global``: one additive conformal correction for all engines
  (`conformal.conformal_correction`, the Phase 14 method).
- ``mondrian``: a separate correction per band of the RSF point estimate
  (`PRED_BANDS`). Bands are defined on the *prediction*, never the true RUL,
  so the band is known at serving time.

Corrections are clipped at zero (see `conformal.nonconformity_scores`), so
calibration only ever widens an interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import conformal, cross_validation, features, labeling, models

BASES = ("weibull", "rsf", "hybrid")
MODES = ("none", "global", "mondrian")

# Bands of the RSF point estimate (cycles) for Mondrian calibration. Chosen
# before looking at any calibration result: "close to failure", "mid-life",
# and "near the training cap" (RSF's RMST can't exceed 125).
PRED_BANDS = (0.0, 40.0, 90.0, np.inf)
PRED_BAND_LABELS = ("pred < 40", "pred 40-90", "pred >= 90")

# A Mondrian band with fewer calibration points than this falls back to the
# global correction: below ~10 points a 90% conformal quantile is just the max.
MIN_POINTS_PER_BAND = 30

META_COLS = {"unit_number", "time_cycles", "RUL", "duration", "event"}


def curated_feature_cols(kept_sensors: list[str]) -> list[str]:
    """The Weibull model's 1 + 2*n_sensors curated features (Phase 4)."""
    cols = ["cycle_count"]
    for sensor in kept_sensors:
        cols += [f"{sensor}_roll15_mean", f"{sensor}_roll15_slope"]
    return cols


def fit_weibull_aft(train_survival: pd.DataFrame, curated_cols: list[str],
                    penalizer: float = 0.1, duration_eps: float = 1e-3):
    """Fit the deployed Weibull AFT recipe: standardize the curated features,
    shift durations off zero, fit with an L2 penalizer. Returns (scaler, model)."""
    scaler = StandardScaler().fit(train_survival[curated_cols])
    fit_df = train_survival[curated_cols + ["duration", "event"]].copy()
    fit_df[curated_cols] = scaler.transform(train_survival[curated_cols])
    fit_df["duration"] = fit_df["duration"] + duration_eps
    model = WeibullAFTFitter(penalizer=penalizer)
    model.fit(fit_df, duration_col="duration", event_col="event")
    return scaler, model


def weibull_interval(model, scaler, X_curated: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Weibull AFT 90% interval: S(t)=0.95 gives the lower bound, S(t)=0.05 the upper."""
    scaled = X_curated.copy()
    scaled[X_curated.columns] = scaler.transform(X_curated)
    low = np.asarray(model.predict_percentile(scaled, p=0.95), dtype=float).ravel()
    high = np.asarray(model.predict_percentile(scaled, p=0.05), dtype=float).ravel()
    return low, high


def out_of_fold_intervals(train_for_labeling: pd.DataFrame, kept_sensors: list[str], *,
                          min_samples_leaf: int, row_stride: int, cap: int = 125,
                          duration_eps: float = 1e-3, penalizer: float = 0.1,
                          n_splits: int = 5, n_truncations: int = 10,
                          min_frac: float = 0.3, max_frac: float = 0.99,
                          random_state: int = 42) -> pd.DataFrame:
    """Score every training engine with RSF and Weibull models that never saw it.

    Engines are split into `n_splits` folds by engine. For each fold, both
    models are fit on the other folds with the deployed recipe, and each
    held-out engine is scored at `n_truncations` random points between
    `min_frac` and `max_frac` of its life. `max_frac=0.99` (rather than the
    0.9 used by Phase 14) is what puts calibration points close to failure.

    Returns one row per (engine, truncation point) with the true remaining
    life, RSF's point estimate and interval, and Weibull's interval.
    Duplicate truncation cycles for an engine are kept once.
    """
    units = train_for_labeling["unit_number"].unique()
    folds = GroupKFold(n_splits=n_splits).split(units, groups=units)
    curated = curated_feature_cols(kept_sensors)
    out = []
    for fold, (train_pos, test_pos) in enumerate(folds):
        train_units, test_units = units[train_pos], units[test_pos]
        fold_train = train_for_labeling[train_for_labeling.unit_number.isin(train_units)]
        fold_test = train_for_labeling[train_for_labeling.unit_number.isin(test_units)]

        train_survival = labeling.build_train_survival_frame(fold_train, cap=cap)
        train_survival = features.build_feature_matrix(train_survival, kept_sensors=kept_sensors)
        full_cols = [c for c in train_survival.columns if c not in META_COLS]

        rsf = models.fit_random_survival_forest(
            train_survival, full_cols, row_stride=row_stride, min_samples_leaf=min_samples_leaf,
            duration_eps=duration_eps, random_state=random_state,
        )
        scaler, weibull = fit_weibull_aft(train_survival, curated, penalizer, duration_eps)

        test_featured = features.build_feature_matrix(fold_test, kept_sensors=kept_sensors)
        frames = []
        for k in range(n_truncations):
            trunc = cross_validation.sample_truncation_points(
                fold_test, min_frac=min_frac, max_frac=max_frac,
                random_state=random_state + 1000 * fold + k,
            )
            frames.append(cross_validation.build_truncated_eval_frame(test_featured, trunc))
        eval_frame = (pd.concat(frames, ignore_index=True)
                        .drop_duplicates(["unit_number", "time_cycles"])
                        .reset_index(drop=True))
        eval_frame = eval_frame[eval_frame.true_rul >= 1].reset_index(drop=True)

        rsf_pred = models.predict_rul_from_survival(rsf, eval_frame[full_cols])
        w_low, w_high = weibull_interval(weibull, scaler, eval_frame[curated])
        out.append(pd.DataFrame({
            "fold": fold,
            "unit_number": eval_frame.unit_number.to_numpy(),
            "time_cycles": eval_frame.time_cycles.to_numpy(),
            "true_rul": eval_frame.true_rul.to_numpy(float),
            "rsf_pred": rsf_pred.pred_rul.to_numpy(),
            "rsf_low": rsf_pred.pred_ci_low.to_numpy(),
            "rsf_high": rsf_pred.pred_ci_high.to_numpy(),
            "weibull_low": w_low,
            "weibull_high": w_high,
        }))
    return pd.concat(out, ignore_index=True)


def base_interval(base: str, rsf_low, rsf_high, weibull_low, weibull_high) -> tuple[np.ndarray, np.ndarray]:
    """The uncalibrated interval for one base (see module docstring)."""
    rsf_low, rsf_high = np.asarray(rsf_low, float), np.asarray(rsf_high, float)
    weibull_low, weibull_high = np.asarray(weibull_low, float), np.asarray(weibull_high, float)
    if base == "weibull":
        return weibull_low, weibull_high
    if base == "rsf":
        return rsf_low, rsf_high
    if base == "hybrid":
        return np.minimum(rsf_low, weibull_low), weibull_high
    raise ValueError(f"unknown interval base {base!r}; expected one of {BASES}")


def pred_band(rsf_pred) -> np.ndarray:
    """Index (0, 1, 2) of the RSF-prediction band each estimate falls in."""
    return np.digitize(np.asarray(rsf_pred, float), PRED_BANDS[1:-1], right=False)


@dataclass
class IntervalCalibration:
    """A fitted interval: which base, which mode, and the correction(s).

    `corrections` has one entry per prediction band (identical entries for
    ``global``, zeros for ``none``), so applying it is the same code path
    for every mode. Serialised into deployment_metadata.json by the export
    script via `to_dict` / `from_dict`.
    """
    base: str
    mode: str
    corrections: list[float] = field(default_factory=lambda: [0.0] * (len(PRED_BANDS) - 1))
    alpha: float = 0.10

    @property
    def name(self) -> str:
        return f"{self.base}+{self.mode}"

    def apply(self, rsf_pred, rsf_low, rsf_high, weibull_low, weibull_high) -> tuple[np.ndarray, np.ndarray]:
        low, high = base_interval(self.base, rsf_low, rsf_high, weibull_low, weibull_high)
        q = np.asarray(self.corrections, float)[pred_band(rsf_pred)]
        return np.maximum(low - q, 0.0), high + q

    def to_dict(self) -> dict:
        return {"base": self.base, "mode": self.mode, "alpha": self.alpha,
                "corrections": [round(float(q), 4) for q in self.corrections],
                "pred_bands": [b if np.isfinite(b) else None for b in PRED_BANDS]}

    @classmethod
    def from_dict(cls, d: dict) -> "IntervalCalibration":
        return cls(base=d["base"], mode=d["mode"], corrections=list(d["corrections"]),
                   alpha=d.get("alpha", 0.10))


def calibrate(frame: pd.DataFrame, base: str, mode: str, alpha: float = 0.10) -> IntervalCalibration:
    """Fit an interval's correction(s) on out-of-fold rows from `out_of_fold_intervals`."""
    if mode not in MODES:
        raise ValueError(f"unknown calibration mode {mode!r}; expected one of {MODES}")
    n_bands = len(PRED_BANDS) - 1
    if mode == "none":
        return IntervalCalibration(base, mode, [0.0] * n_bands, alpha)

    low, high = base_interval(base, frame.rsf_low, frame.rsf_high, frame.weibull_low, frame.weibull_high)
    scores = conformal.nonconformity_scores(frame.true_rul.to_numpy(float), low, high)
    q_global = conformal.conformal_correction(scores, alpha=alpha)
    if mode == "global":
        return IntervalCalibration(base, mode, [q_global] * n_bands, alpha)

    bands = pred_band(frame.rsf_pred)
    corrections = []
    for b in range(n_bands):
        band_scores = scores[bands == b]
        corrections.append(conformal.conformal_correction(band_scores, alpha=alpha)
                           if len(band_scores) >= MIN_POINTS_PER_BAND else q_global)
    return IntervalCalibration(base, mode, corrections, alpha)


def candidates() -> list[tuple[str, str]]:
    """The six (base, mode) pairs compared in notebook 15, fixed in advance."""
    return [("weibull", "none"), ("rsf", "global"), ("weibull", "mondrian"),
            ("hybrid", "none"), ("hybrid", "global"), ("hybrid", "mondrian")]
