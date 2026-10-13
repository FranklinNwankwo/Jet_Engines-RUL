"""
Thin wrappers around lifelines / scikit-survival model fitting, plus the
engine-level train/validation split every survival model in this project
needs.

Why a dedicated split function
-------------------------------
`train_survival.parquet` has one row per *cycle*, but rows from the same
engine are correlated (they're the same underlying trajectory sampled at
different points). A row-level random train/val split would put cycles from
the same engine on both sides of the split -- the model would effectively be
tested on engines it already partially saw during training, inflating every
validation metric. `train_val_split_by_engine` splits on `unit_number`
first, then takes all rows belonging to each unit -- the split boundary is
an engine, never a row.

This module intentionally stays thin: it does not contain model-specific
tuning logic or evaluation metrics (those belong in the individual model
notebooks and in `evaluation.py` respectively) -- it exists so that fitting,
saving, and loading a model is one function call, not boilerplate repeated
in every notebook.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_val_split_by_engine(df: pd.DataFrame, unit_col: str = "unit_number",
                               val_fraction: float = 0.2,
                               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a row-level dataframe into train/val by engine, not by row.

    Returns (train_df, val_df), each containing every row for the engines
    assigned to that split.
    """
    units = df[unit_col].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])

    train_df = df[df[unit_col].isin(train_units)].reset_index(drop=True)
    val_df = df[df[unit_col].isin(val_units)].reset_index(drop=True)
    return train_df, val_df


def standardize_features(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          feature_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Z-score `feature_cols`, fitting the scaler on train only.

    Fitting on train only (never on val, and never on train+val combined) is
    what keeps this a legitimate held-out evaluation -- scaling with
    statistics that saw the validation engines would leak information about
    them into the "unseen" split.

    Standardizing is also what makes Cox PH hazard ratios comparable across
    features of very different raw scales (a pressure sensor in the
    thousands vs. a rolling slope near zero) -- see docs/project_brief.md.
    """
    scaler = StandardScaler().fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    return train_scaled, val_scaled, scaler


def fit_cox_ph(df: pd.DataFrame, feature_cols: Sequence[str],
               duration_col: str = "duration", event_col: str = "event",
               penalizer: float = 0.1, cluster_col: str | None = None):
    """Fit a Cox Proportional Hazards model.

    `penalizer` applies ridge regularization to the partial likelihood --
    not a default left at zero, but a deliberate choice: several engineered
    features here are correlated by construction (e.g. a sensor's roll15
    mean and roll30 mean), and Cox PH's coefficient estimates become
    unstable under collinearity. The penalizer trades a small amount of
    coefficient bias for materially more stable, more trustworthy hazard
    ratios -- worth stating explicitly rather than fitting unregularized and
    hoping the optimizer converges cleanly.

    `cluster_col` (e.g. "unit_number") makes lifelines compute robust
    (sandwich) standard errors clustered by that column. Every training row
    is one cycle of an engine, and rows from the same engine are strongly
    correlated, so the default standard errors -- which treat ~17k rows from
    ~80 engines as independent -- are too small and the p-values too
    optimistic. Coefficients and predictions are the same either way; only
    standard errors, confidence intervals and p-values change. Default None
    keeps the original behaviour.

    lifelines' routine advisory warnings are silenced, but a
    `ConvergenceWarning` is still shown: a fit that didn't converge should
    never pass silently.
    """
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    cols = list(feature_cols) + [duration_col, event_col]
    if cluster_col is not None and cluster_col not in cols:
        cols.append(cluster_col)

    model = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(df[cols], duration_col=duration_col, event_col=event_col,
                  cluster_col=cluster_col)
    return model


def fit_random_survival_forest(df: pd.DataFrame, feature_cols: Sequence[str],
                                duration_col: str = "duration", event_col: str = "event",
                                n_estimators: int = 100, min_samples_leaf: int = 30,
                                max_features: str = "sqrt", row_stride: int = 3,
                                duration_eps: float = 1e-3, random_state: int = 42,
                                n_jobs: int = -1):
    """Fit a Random Survival Forest.

    `row_stride` subsamples every Nth row of the training frame. This is a
    modeling decision, not just a speed hack: consecutive cycles from the
    same engine are extremely redundant (a rolling-window feature at cycle
    t and cycle t+1 share almost all their underlying data), so keeping
    every row inflates the effective sample size without adding
    proportionate information. Empirically on FD001, stride=3 reaches
    essentially the same held-out concordance as stride=1 (~0.90) at a
    fraction of the fit cost -- see notebooks/05_random_survival_forest.ipynb
    for that comparison.

    The `event_col` values are passed through as-is: with the project's
    default labeling every training row is event=1, and if
    `labeling.build_train_survival_frame(..., censor_above_cap=True)` is used
    instead, rows marked event=0 are treated as censored by the forest.
    `duration_eps` shifts every duration slightly above zero, since the final
    cycle of each engine has duration 0.

    Returns the fitted model; use `predict_rul_from_survival` for RUL
    point estimates and intervals.
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    fit_df = df.iloc[::row_stride]
    y = Surv.from_arrays(event=fit_df[event_col].astype(bool),
                          time=fit_df[duration_col] + duration_eps)

    model = RandomSurvivalForest(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=random_state, n_jobs=n_jobs,
    )
    model.fit(fit_df[list(feature_cols)], y)
    return model


def predict_rul_from_survival(model, X: pd.DataFrame,
                               lower_p: float = 0.95, upper_p: float = 0.05) -> pd.DataFrame:
    """Turn a fitted survival model's predicted curves into RUL point estimates + intervals.

    The point estimate is **restricted mean survival time** (RMST) -- the
    area under the predicted survival curve -- rather than the median. This
    matters for a tree-based model specifically: an RSF's survival curve is
    non-parametric and defined only over the time support seen in training,
    so for a healthy-looking engine the curve may never cross S(t)=0.5
    within that support and the median silently saturates at the maximum
    training duration. On FD001 the median saturates for roughly a quarter
    of test engines; RMST integrates the whole curve instead and degrades
    gracefully. (A parametric model like Weibull AFT has no such ceiling,
    which is exactly why the project brief keeps both.)

    RMST here is the trapezoid-rule area under S(t) evaluated at the model's
    unique training times (roughly one per cycle on C-MAPSS), from the first
    training time up to the last. RSF's survival curve is a step function,
    so this differs from the exact step-function area by at most about half
    a cycle. Because the curve stops at the largest training time (the cap,
    plus `duration_eps`), RMST can never exceed it.

    Interval bounds come from survival quantiles: the first training time at
    which S(t) drops to or below the given probability, or the largest
    training time if it never does. Because S(t) is decreasing, the *lower*
    RUL bound corresponds to the *larger* survival probability -- an easy
    convention to invert, so it's asserted below.
    """
    times = model.unique_times_
    surv_fns = model.predict_survival_function(X)
    S = np.vstack([fn(times) for fn in surv_fns])

    rmst = np.trapezoid(S, times, axis=1)

    def _quantile(s_row: np.ndarray, p: float) -> float:
        idx = np.searchsorted(-s_row, -p)
        return float(times[min(idx, len(times) - 1)])

    lower = np.array([_quantile(s, lower_p) for s in S])
    upper = np.array([_quantile(s, upper_p) for s in S])
    assert (lower <= upper).all(), "interval bounds inverted -- check lower_p/upper_p convention"

    return pd.DataFrame({
        "pred_rul": rmst,
        "pred_ci_low": lower,
        "pred_ci_high": upper,
    }, index=X.index)


def save_model(model, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str):
    return joblib.load(path)
