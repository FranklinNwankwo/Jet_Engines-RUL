import numpy as np
import pytest

from turbofan_rul.conformal import (
    apply_conformal_correction,
    conformal_correction,
    nonconformity_scores,
)


def test_nonconformity_score_zero_when_inside_interval():
    scores = nonconformity_scores(np.array([50.0]), np.array([40.0]), np.array([60.0]))
    assert scores[0] == 0.0


def test_nonconformity_score_positive_when_below_interval():
    scores = nonconformity_scores(np.array([30.0]), np.array([40.0]), np.array([60.0]))
    assert scores[0] == pytest.approx(10.0)


def test_nonconformity_score_positive_when_above_interval():
    scores = nonconformity_scores(np.array([70.0]), np.array([40.0]), np.array([60.0]))
    assert scores[0] == pytest.approx(10.0)


def test_nonconformity_score_never_negative():
    # deeply inside the interval should still score exactly 0, not negative
    scores = nonconformity_scores(np.array([50.0]), np.array([0.0]), np.array([100.0]))
    assert scores[0] == 0.0


def test_conformal_correction_is_nonnegative():
    scores = np.array([0.0, 2.0, 5.0, 10.0, 1.0])
    q = conformal_correction(scores, alpha=0.1)
    assert q >= 0


def test_conformal_correction_zero_when_all_scores_zero():
    scores = np.zeros(20)
    q = conformal_correction(scores, alpha=0.1)
    assert q == 0.0


def test_conformal_correction_uses_finite_sample_level_not_naive_quantile():
    """The whole point of the finite-sample correction: it should be >= the
    naive (uncorrected) 1-alpha quantile, since ceil((n+1)(1-alpha))/n >=
    (1-alpha) for any finite n -- an under-corrected q would silently
    reintroduce the under-coverage this module exists to fix."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 20, size=37)
    q_corrected = conformal_correction(scores, alpha=0.1)
    q_naive = np.quantile(scores, 0.9)
    assert q_corrected >= q_naive


def test_conformal_correction_raises_on_empty_scores():
    with pytest.raises(ValueError):
        conformal_correction(np.array([]))


def test_apply_conformal_correction_widens_symmetrically():
    low, high = apply_conformal_correction(np.array([40.0]), np.array([60.0]), q=5.0)
    assert low[0] == pytest.approx(35.0)
    assert high[0] == pytest.approx(65.0)


def test_apply_conformal_correction_zero_q_is_noop():
    low, high = apply_conformal_correction(np.array([40.0]), np.array([60.0]), q=0.0)
    assert low[0] == pytest.approx(40.0)
    assert high[0] == pytest.approx(60.0)


def test_calibration_achieves_at_least_nominal_coverage_on_calibration_set_itself():
    """End-to-end sanity check: calibrating on a set and then re-scoring
    THAT SAME SET (not held-out data -- this is checking the math, not
    claiming generalization) should hit at least the nominal coverage,
    since that's what conformal_correction is defined to guarantee on the
    calibration data itself."""
    rng = np.random.default_rng(1)
    n = 200
    true_rul = rng.uniform(0, 100, n)
    # deliberately under-covering intervals: only +/-5 around a noisy estimate
    point_est = true_rul + rng.normal(0, 15, n)
    ci_low, ci_high = point_est - 5, point_est + 5

    raw_coverage = np.mean((true_rul >= ci_low) & (true_rul <= ci_high))
    assert raw_coverage < 0.85  # confirm the premise: these intervals under-cover

    scores = nonconformity_scores(true_rul, ci_low, ci_high)
    q = conformal_correction(scores, alpha=0.1)
    new_low, new_high = apply_conformal_correction(ci_low, ci_high, q)
    new_coverage = np.mean((true_rul >= new_low) & (true_rul <= new_high))
    assert new_coverage >= 0.90
