"""
Evaluation metrics for RUL / survival models.

The headline metric here is NOT RMSE. See docs/project_brief.md Section 6:
RMSE treats a 20-cycle over-prediction and a 20-cycle under-prediction as
equally bad, which is wrong for maintenance. Over-predicting remaining life
means an engine fails in service; under-predicting means a part gets
replaced earlier than strictly necessary. Those outcomes don't cost the same,
and the metric should reflect that.

`nasa_score` is implemented directly from the C-MAPSS literature rather
than imported from a package -- the asymmetry constants are the whole point
of the metric, so they're written out explicitly below where they can be
read and checked.

Concordance is not computed here; the notebooks use
`lifelines.utils.concordance_index` (engine-level, on point predictions) or
the scikit-survival model's own `.score` (row-level). Those are different
quantities -- compare concordance values only when they were computed the
same way on the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Asymmetry constants from the PHM08 / C-MAPSS prognostics challenge scoring
# function (Saxena et al., 2008). Late predictions (predicted RUL exceeds
# true RUL -- i.e. the model thinks there is more life left than there is)
# decay with the smaller constant, which makes them grow FASTER. This is the
# encoded domain judgment: a missed failure is worse than an early replacement.
LATE_DECAY = 10.0   # applied when d > 0 (over-prediction; dangerous)
EARLY_DECAY = 13.0  # applied when d <= 0 (under-prediction; merely costly; both branches are 0 at d = 0)


def _as_matching_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty set of predictions")
    return y_true, y_pred


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-MAPSS asymmetric prognostics score. Lower is better; 0 is perfect.

    For each engine, let d = predicted RUL - true RUL. The per-engine
    penalty is:

        exp(d / 10) - 1   if d >  0   (late / over-prediction)
        exp(-d / 13) - 1  if d <= 0   (early / under-prediction)

    Both branches are zero at d = 0 and grow exponentially with error
    magnitude, but the late branch grows faster because it divides by the
    smaller constant. A 20-cycle over-prediction scores ~6.4; a 20-cycle
    under-prediction scores ~3.7 -- roughly 1.75x worse for being late, and
    the gap widens with error size.

    Note this is a SUM over engines, not a mean, which is the convention in
    the C-MAPSS literature -- so it is only comparable between models
    evaluated on the same number of units. `nasa_score_mean` is provided
    for cross-dataset comparisons.
    """
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    d = y_pred - y_true
    penalties = np.where(d > 0, np.exp(d / LATE_DECAY) - 1.0, np.exp(-d / EARLY_DECAY) - 1.0)
    return float(np.sum(penalties))


def nasa_score_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-engine mean of the NASA score, for comparing across differently sized sets."""
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    return nasa_score(y_true, y_pred) / y_true.size


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. Reported as a SECONDARY, diagnostic metric only.

    Included for comparability with the wider C-MAPSS literature (where it
    is commonly reported), not as a decision criterion -- it is symmetric
    and therefore blind to the cost asymmetry that `nasa_score` captures.
    """
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. Secondary/diagnostic, same caveat as `rmse`."""
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def late_prediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of engines where predicted RUL exceeded true RUL (strictly).

    The operationally meaningful failure mode, reported separately because
    a model can post a decent aggregate score while being late on a
    dangerous fraction of units.
    """
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    return float(np.mean(y_pred > y_true))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical fraction of true values falling inside the predicted interval
    (bounds inclusive).

    For a nominal 90% interval this should come out near 0.90. Materially
    below means the intervals are overconfident; materially above means
    they are wider than they need to be.
    """
    y_true, lower = _as_matching_arrays(y_true, lower)
    _, upper = _as_matching_arrays(y_true, upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                              lower: np.ndarray | None = None,
                              upper: np.ndarray | None = None,
                              model_name: str = "model") -> dict:
    """Run the full metric suite for one model, ordered by importance.

    The ordering of keys in the returned dict is deliberate: the
    domain-appropriate metrics come first, RMSE/MAE last, so that a results
    table built from these rows reads in the same priority order the
    project brief argues for. `interval_coverage` is added only when both
    `lower` and `upper` are given.
    """
    if (lower is None) != (upper is None):
        raise ValueError("pass both `lower` and `upper` to compute interval coverage, or neither")
    results = {
        "model": model_name,
        "nasa_score": nasa_score(y_true, y_pred),
        "nasa_score_mean": nasa_score_mean(y_true, y_pred),
        "late_prediction_rate": late_prediction_rate(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    if lower is not None and upper is not None:
        results["interval_coverage"] = interval_coverage(y_true, lower, upper)
    return results


def calibration_by_risk_bin(y_true: np.ndarray, y_pred: np.ndarray,
                             n_bins: int = 5) -> pd.DataFrame:
    """Bin engines by predicted RUL and compare mean predicted vs. mean observed.

    Bins are quantiles of the predicted RUL, so bin 0 holds the engines with
    the LOWEST predicted RUL (highest predicted risk). A well-calibrated
    model's points lie on the diagonal: engines it thinks have ~30 cycles
    left should on average have ~30 cycles left. Deviation from the diagonal
    shows *where* in the risk range a model is biased -- which aggregate
    metrics like RMSE cannot show, since a model can be unbiased on average
    while systematically over-predicting the highest-risk units (the ones
    that matter most).

    This checks the point estimates only. It is not a survival-curve
    calibration check (e.g. predicted curves vs. Kaplan-Meier), and it says
    nothing about whether the intervals are calibrated -- use
    `interval_coverage` for that.

    If many predictions are tied (e.g. RSF predictions piling up near its
    training ceiling), duplicate quantile edges are dropped and fewer than
    `n_bins` bins may be returned.
    """
    y_true, y_pred = _as_matching_arrays(y_true, y_pred)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bin"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")

    grouped = df.groupby("bin").agg(
        n=("y_true", "size"),
        mean_predicted=("y_pred", "mean"),
        mean_observed=("y_true", "mean"),
    ).reset_index()
    grouped["bias"] = grouped["mean_predicted"] - grouped["mean_observed"]
    return grouped