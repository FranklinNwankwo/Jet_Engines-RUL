"""
Build the artifacts the deployment API actually loads.

Why this is a script, not something read out of a notebook
--------------------------------------------------------------
Phases 3-5 fit models on an 80/20 engine-level split specifically to get an
honest held-out evaluation (Phase 6). That split's purpose is now served --
model selection is done. Refitting on *all* available training engines for
the deployed model is standard practice once evaluation is complete, since
there's no more reason to hold 20% back. Rebuilding that here, in a script
that runs independently of any notebook's kernel state, also means the API
doesn't depend on "did the right notebook cell run in the right order" --
it depends on this script, which is deterministic (fixed random_state) and
re-runnable from a clean checkout.

Per the Phase 6 deployment decision: Random Survival Forest supplies the
point estimate and risk ranking, Weibull AFT supplies the confidence
interval (RSF's own intervals under-cover and it cannot predict past its
training time ceiling -- see notebooks 05 and 06). Cox PH is not part of
deployment: RSF beat it on every Phase 6 metric, and Cox produces no natural
prediction interval. (Cox did beat Weibull on concordance, RMSE and MAE --
Weibull is deployed for its interval, not for its point accuracy.)

Training labels come from `labeling.build_train_survival_frame` with its
default `censor_above_cap=False`: every row is event=1 at its remaining life
capped at CAP. Censoring rows above the cap instead was tested and rejected
(it wrecks Weibull AFT's point estimates and widens its interval -- see the
`labeling.py` docstring). Every model below, including the cross-conformal
calibration folds, is fit on those (duration, event) pairs.

Multi-subset support (Phase 12d)
-----------------------------------
FD001 uses the original, unchanged recipe: no regime clustering (single
operating condition), sensor selection read from the Phase 1 EDA artifact
(`fd001_sensor_selection.json`). This is deliberately NOT retrofitted with
the regime-clustering machinery added for FD002/FD004 -- doing so would
silently change the model behind every FD001 number already reported in
Phase 6/11, which is exactly the mistake this project's "don't silently
change an already-evaluated model" precedent (see README's bugs-caught
section) exists to avoid.

FD004 uses the Phase 12 recipe: regime clustering + within-cluster
normalization (`features.fit_regime_clusters` / `*_normalization_stats`),
with sensor selection computed on the normalized data (see
`notebooks/10_fd004_full_pipeline.ipynb` for why pre-normalization
selection would be dominated by regime, not informativeness). The fitted
KMeans model and normalization stats are saved as artifacts alongside the
models, since inference needs to reproduce the exact same normalization at
request time.

Each subset's artifacts are saved to their own subdirectory
(`models/fd001/`, `models/fd004/`) so the two never collide.

Run with: python -m turbofan_rul.export_deployment_artifacts [--subset FD001|FD004]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from lifelines import WeibullAFTFitter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import conformal, cross_validation, data_loading, features, labeling, models

logger = logging.getLogger(__name__)

CAP = 125
DURATION_EPS = 1e-3
PENALIZER = 0.1
RANDOM_STATE = 42
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# Subsets needing regime clustering (more than one operating condition) and
# the row_stride used for their RSF fit -- FD004 is ~3x FD001's row count,
# so a coarser stride keeps a full export runnable in reasonable time.
REGIME_CLUSTERED_SUBSETS = {"FD002": 6, "FD004": 6}
ROW_STRIDE_BY_SUBSET = {"FD001": 3, "FD004": 15}

# Phase 13 (notebooks/11_hyperparameter_tuning.ipynb): min_samples_leaf=5
# beat the original default of 30 by mean concordance across a full 5-fold
# CV confirmation (0.860 vs. 0.847, winning 4/5 folds) -- at the cost of
# roughly tripling RSF's fold-to-fold variance (std 0.033 vs. 0.010), a
# real tradeoff stated in that notebook's Verdict, not hidden here. Tuning
# was done on FD001 only; FD004 inherits the FD001-tuned value without an
# independent search -- a stated scope gap (see README), not an assumption
# that the FD001-optimal leaf size is also FD004-optimal.
RSF_MIN_SAMPLES_LEAF_BY_SUBSET = {"FD001": 5, "FD004": 5}


def _prepare_fd001_style(subset: str, cmapss, processed_dir: Path):
    """Single-regime recipe: sensor selection from the saved Phase 1 EDA
    artifact, no clustering, no normalization -- exactly the original
    FD001 pipeline, parameterized only by subset name so it can (in
    principle) serve any single-regime subset with its own EDA artifact."""
    sensor_selection_path = processed_dir / f"{subset.lower()}_sensor_selection.json"
    if not sensor_selection_path.exists():
        raise FileNotFoundError(
            f"{sensor_selection_path} not found -- run the Phase-1-style EDA notebook "
            f"for {subset} first to produce it."
        )
    kept_sensors = features.load_kept_sensors(sensor_selection_path)
    return kept_sensors, cmapss.train, None, None


def _prepare_regime_clustered(subset: str, cmapss):
    """Multi-regime recipe (FD002/FD004): fit clusters + normalization on
    train, select sensors on the NORMALIZED data -- see module docstring."""
    n_clusters = REGIME_CLUSTERED_SUBSETS[subset]
    kmeans_model = features.fit_regime_clusters(cmapss.train, n_clusters=n_clusters, random_state=RANDOM_STATE)
    train_labels = features.assign_regime_clusters(cmapss.train, kmeans_model)
    norm_stats = features.fit_regime_normalization_stats(cmapss.train, SENSOR_COLS, train_labels)
    train_norm = features.apply_regime_normalization(cmapss.train, SENSOR_COLS, train_labels, norm_stats)

    # After within-regime z-scoring, any sensor that varies at all within its
    # regime ends up with std close to 1, so this threshold only removes
    # sensors that are constant within every regime -- it is not a test of
    # how informative a sensor is about degradation.
    std = train_norm[SENSOR_COLS].std()
    kept_sensors = std[std > 0.05].index.tolist()
    return kept_sensors, train_norm, kmeans_model, norm_stats


def _calibrate_rsf_interval(train_for_labeling, kept_sensors: list[str], min_samples_leaf: int,
                             row_stride: int, alpha: float = 0.10, n_splits: int = 5) -> float:
    """Cross-conformal calibration of RSF's confidence interval (Phase 14,
    notebooks/12_rsf_interval_calibration.ipynb): each of the training
    engines is scored by a model fit on the other folds (never saw it),
    pooling calibration data across all engines without holding any out of
    the final deployed model. Returns the additive correction `q` -- see
    `conformal.py` for the math and why this specific approach.

    Takes `train_for_labeling` -- the same (already regime-normalized, for
    FD002/FD004) frame `build_and_save` derives features from -- rather
    than re-deriving it from raw data, so the calibration folds see
    exactly the same preprocessing the deployed model itself was trained
    on. Using raw, un-normalized data here for a multi-regime subset would
    silently recreate the exact regime-vs-degradation confound Phase 12's
    clustering step exists to remove.

    Uses the same `row_stride` and `min_samples_leaf` as the deployed RSF, so
    the fold models match the deployed model's settings (the exploratory
    calibration in notebook 12 used `row_stride=8`, so its `q` can differ
    slightly from the value computed here). The fold models still see 80% of
    the engines and are scored at random truncation points, so the
    calibrated coverage is an empirical target, not a guarantee -- see
    `conformal.py`.

    Runs for every subset this script exports, FD004 included.

    This adds real time to a full export (another `n_splits` RSF fits on
    top of the final model's own fit) -- a deliberate cost for a
    reproducible-from-clean-checkout guarantee, not an oversight.
    """
    units = train_for_labeling["unit_number"].unique()
    fold_splits = list(GroupKFold(n_splits=n_splits).split(units, groups=units))
    meta_cols = {"unit_number", "time_cycles", "RUL", "duration", "event"}

    all_true, all_low, all_high = [], [], []
    for train_pos, test_pos in fold_splits:
        train_units, test_units = units[train_pos], units[test_pos]
        fold_train_raw = train_for_labeling[train_for_labeling.unit_number.isin(train_units)]
        fold_test_raw = train_for_labeling[train_for_labeling.unit_number.isin(test_units)]

        fold_train_survival = labeling.build_train_survival_frame(fold_train_raw, cap=CAP)
        fold_train_survival = features.build_feature_matrix(fold_train_survival, kept_sensors=kept_sensors)
        full_cols = [c for c in fold_train_survival.columns if c not in meta_cols]

        fold_test_featured = features.build_feature_matrix(fold_test_raw, kept_sensors=kept_sensors)
        trunc = cross_validation.sample_truncation_points(fold_test_raw, random_state=RANDOM_STATE)
        eval_frame = cross_validation.build_truncated_eval_frame(fold_test_featured, trunc)

        rsf = models.fit_random_survival_forest(
            fold_train_survival, full_cols, row_stride=row_stride,
            min_samples_leaf=min_samples_leaf, duration_eps=DURATION_EPS, random_state=RANDOM_STATE,
        )
        pred = models.predict_rul_from_survival(rsf, eval_frame[full_cols])
        all_true.append(eval_frame["true_rul"].values)
        all_low.append(pred["pred_ci_low"].values)
        all_high.append(pred["pred_ci_high"].values)

    scores = conformal.nonconformity_scores(np.concatenate(all_true), np.concatenate(all_low), np.concatenate(all_high))
    return conformal.conformal_correction(scores, alpha=alpha)


def build_and_save(subset: str, data_dir: Path, processed_dir: Path, models_dir: Path) -> None:
    subset_models_dir = models_dir / subset.lower()
    subset_models_dir.mkdir(parents=True, exist_ok=True)

    data_loading.download_cmapss(subset, data_dir=data_dir)
    cmapss = data_loading.load_subset(subset, data_dir=data_dir)

    uses_clustering = subset in REGIME_CLUSTERED_SUBSETS
    if uses_clustering:
        kept_sensors, train_for_labeling, kmeans_model, norm_stats = _prepare_regime_clustered(subset, cmapss)
        models.save_model(kmeans_model, subset_models_dir / "regime_kmeans.joblib")
        models.save_model(norm_stats, subset_models_dir / "regime_norm_stats.joblib")
    else:
        kept_sensors, train_for_labeling, _, _ = _prepare_fd001_style(subset, cmapss, processed_dir)

    train_survival = labeling.build_train_survival_frame(train_for_labeling, cap=CAP)
    train_survival = features.build_feature_matrix(train_survival, kept_sensors=kept_sensors)

    curated_feature_cols = ["cycle_count"]
    for sensor in kept_sensors:
        curated_feature_cols += [f"{sensor}_roll15_mean", f"{sensor}_roll15_slope"]
    meta_cols = {"unit_number", "time_cycles", "RUL", "duration", "event"}
    full_feature_cols = [c for c in train_survival.columns if c not in meta_cols]
    # See README bugs-caught section: full_feature_cols includes raw
    # passthrough columns (every raw sensor + operating setting), not just
    # engineered ones -- kept as-is for consistency with how RSF was trained
    # and evaluated in Phase 6/12c. Phase 17's ablation (RSF refit on the
    # 205 engineered-only columns) moved every metric by less than ordinary
    # run-to-run variance, so this is a documented scope quirk, not a
    # measurable accuracy problem.
    raw_passthrough_cols = [c for c in full_feature_cols if c in cmapss.train.columns]

    scaler = StandardScaler().fit(train_survival[curated_feature_cols])
    models.save_model(scaler, subset_models_dir / "weibull_scaler.joblib")

    weibull_fit_df = train_survival.copy()
    weibull_fit_df[curated_feature_cols] = scaler.transform(train_survival[curated_feature_cols])
    weibull_fit_df["duration"] = weibull_fit_df["duration"] + DURATION_EPS
    weibull_model = WeibullAFTFitter(penalizer=PENALIZER)
    weibull_model.fit(weibull_fit_df[curated_feature_cols + ["duration", "event"]],
                       duration_col="duration", event_col="event")
    models.save_model(weibull_model, subset_models_dir / "weibull_aft.joblib")

    row_stride = ROW_STRIDE_BY_SUBSET.get(subset, 3)
    min_samples_leaf = RSF_MIN_SAMPLES_LEAF_BY_SUBSET.get(subset, 30)
    rsf_model = models.fit_random_survival_forest(
        train_survival, full_feature_cols, row_stride=row_stride,
        min_samples_leaf=min_samples_leaf, duration_eps=DURATION_EPS, random_state=RANDOM_STATE,
    )
    models.save_model(rsf_model, subset_models_dir / "rsf.joblib")

    logger.info("Calibrating RSF interval via cross-conformal prediction (5 extra RSF fits)...")
    rsf_interval_correction = _calibrate_rsf_interval(
        train_for_labeling, kept_sensors, min_samples_leaf, row_stride,
    )
    logger.info("RSF interval correction q = %.2f cycles", rsf_interval_correction)

    metadata = {
        "subset": subset,
        "cap": CAP,
        "duration_eps": DURATION_EPS,
        "kept_sensors": kept_sensors,
        "curated_feature_cols": curated_feature_cols,
        "full_feature_cols": full_feature_cols,
        "raw_passthrough_cols": raw_passthrough_cols,
        "uses_regime_clustering": uses_clustering,
        "rsf_min_samples_leaf": min_samples_leaf,
        "rsf_row_stride": row_stride,
        "rsf_interval_correction": rsf_interval_correction,
        "recommended_min_history_cycles": 30,
    }
    with open(subset_models_dir / "deployment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved artifacts for %s to %s", subset, subset_models_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD004"])
    args = parser.parse_args()

    build_and_save(
        subset=args.subset,
        data_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
        models_dir=Path("models"),
    )
