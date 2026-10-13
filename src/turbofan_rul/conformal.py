"""
Conformal calibration for RSF's RUL confidence intervals.

Why this, specifically, for under-coverage
---------------------------------------------
Phase 6 found RSF's 90% intervals achieve ~86-87% empirical coverage on
the real test set -- under-covering, i.e. overconfident. The underlying
cause is that RSF's predicted survival-curve quantiles are not, in
general, well-calibrated probabilities; nothing in how the forest is fit
forces them to be. Rather than trying to fix that indirectly (more trees,
different leaf size, etc. -- Phase 13's tuning already showed those trades
come with their own costs), conformal prediction fixes coverage directly:
widen the interval by a single correction learned on held-out data, chosen
so that at least the target fraction of held-out points fall inside. The
method is model-agnostic -- it would work identically on Weibull's or the
LSTM's intervals.

What is and isn't guaranteed
-------------------------------
Split conformal prediction gives a finite-sample marginal coverage guarantee
of at least `1 - alpha` when the calibration points and the future test
points are exchangeable AND the calibration scores come from the same
fitted model that is later used for prediction. This project's setup
departs from that in two ways, so the guarantee should be read as a
motivation, not a proof:

1. Cross-conformal pooling. Scores are pooled across the Phase 11 5-fold CV
   splits (each engine scored by a model fit on the other 4 folds), then
   the correction is applied to a model refit on all 100 engines. Pooling
   out-of-fold scores this way has, at best, a weaker worst-case guarantee
   (around `1 - 2*alpha` for CV+-style methods), and none in this exact
   refit-on-everything form.
2. Calibration vs. test conditions. Calibration engines are scored at
   random mid-life truncation points by fold models; the deployed model is
   scored at NASA's official truncation points. The two are not strictly
   exchangeable.

The coverage reached on the official test set is therefore an empirical
result, checked directly in `notebooks/12_rsf_interval_calibration.ipynb`,
not a guaranteed one.

Cross-conformal calibration
-------------------------------
A single held-out calibration split would cost real training data and
only be as large as that one split. Pooling out-of-fold scores gives
calibration scores for all 100 training engines without holding any of
them out of the final deployed model. See
`notebooks/12_rsf_interval_calibration.ipynb` for how the scores are
produced; this module only implements the calibration math itself, which
is deliberately decoupled from any one way of generating scores.
"""

from __future__ import annotations

import numpy as np


def nonconformity_scores(true_rul: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    """How far outside its own interval each true value falls.

    Zero if the true value is already inside [ci_low, ci_high]; otherwise
    the (positive) distance to the nearer bound.

    This is the CQR (conformalized quantile regression) score
    `max(ci_low - y, y - ci_high)` clipped at zero. Standard CQR leaves it
    signed, so a negative correction can *shrink* intervals that are
    already too wide. Clipping is a deliberate, conservative variant: the
    resulting correction is always >= 0, so calibration can only widen RSF's
    intervals, never narrow them. That matches the goal here (fixing
    under-coverage) and guarantees the calibrated interval always contains
    the raw one.
    """
    true_rul = np.asarray(true_rul, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    return np.maximum(np.maximum(ci_low - true_rul, true_rul - ci_high), 0.0)


def conformal_correction(scores: np.ndarray, alpha: float = 0.10) -> float:
    """The single additive correction `q` that is applied to both interval
    bounds to target `1 - alpha` coverage.

    Uses the standard finite-sample-corrected quantile level
    ceil((n+1)(1-alpha))/n, not the naive `1-alpha` quantile -- the naive
    version under-covers by construction on finite samples, which would be
    a strange bug to introduce into a module whose entire purpose is
    fixing under-coverage.

    When n is too small for the requested alpha (ceil((n+1)(1-alpha)) > n,
    e.g. n < 9 for alpha=0.10), exact split conformal would return an
    infinite correction. This function caps the level at 1.0 and returns
    the largest score instead, so in that regime the coverage target is not
    guaranteed. The project's calibration set (100 engines) is well above
    that threshold.

    See the module docstring for the conditions under which the target
    coverage is actually guaranteed versus only empirically checked.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("cannot calibrate from zero calibration scores")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal_correction(ci_low: np.ndarray, ci_high: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Widen both bounds by the calibrated correction `q`."""
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    return ci_low - q, ci_high + q