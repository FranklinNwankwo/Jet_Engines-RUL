import numpy as np
import pytest

from turbofan_rul.evaluation import (
    calibration_by_risk_bin,
    evaluate_rul_predictions,
    interval_coverage,
    late_prediction_rate,
    mae,
    nasa_score,
    nasa_score_mean,
    rmse,
)


def test_nasa_score_is_zero_for_perfect_predictions():
    y = np.array([10.0, 50.0, 100.0])
    assert nasa_score(y, y) == pytest.approx(0.0)


def test_nasa_score_penalizes_late_more_than_early():
    """The defining property of this metric -- if this test fails, the
    asymmetry constants are swapped and the metric is actively misleading."""
    y_true = np.array([50.0])
    late = np.array([70.0])   # predicted MORE life than exists -> dangerous
    early = np.array([30.0])  # predicted LESS life than exists -> merely costly

    assert nasa_score(y_true, late) > nasa_score(y_true, early)


def test_nasa_score_known_values():
    # d = +20 -> exp(20/10) - 1 = exp(2) - 1
    assert nasa_score(np.array([50.0]), np.array([70.0])) == pytest.approx(np.exp(2) - 1)
    # d = -20 -> exp(20/13) - 1
    assert nasa_score(np.array([50.0]), np.array([30.0])) == pytest.approx(np.exp(20 / 13) - 1)


def test_nasa_score_grows_exponentially_not_linearly():
    y_true = np.array([100.0])
    err_10 = nasa_score(y_true, np.array([110.0]))
    err_20 = nasa_score(y_true, np.array([120.0]))
    # doubling the error should more than double the penalty
    assert err_20 > 2 * err_10


def test_nasa_score_sums_over_engines():
    y_true = np.array([50.0, 50.0])
    y_pred = np.array([70.0, 70.0])
    single = nasa_score(np.array([50.0]), np.array([70.0]))
    assert nasa_score(y_true, y_pred) == pytest.approx(2 * single)


def test_nasa_score_mean_divides_by_count():
    y_true = np.array([50.0, 50.0])
    y_pred = np.array([70.0, 70.0])
    assert nasa_score_mean(y_true, y_pred) == pytest.approx(nasa_score(y_true, y_pred) / 2)


def test_nasa_score_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        nasa_score(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("metric", [nasa_score, nasa_score_mean, rmse, mae, late_prediction_rate])
def test_every_metric_rejects_shape_mismatch(metric):
    with pytest.raises(ValueError, match="shape mismatch"):
        metric(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("metric", [nasa_score, nasa_score_mean, rmse, mae, late_prediction_rate])
def test_every_metric_rejects_empty_input(metric):
    with pytest.raises(ValueError, match="empty"):
        metric(np.array([]), np.array([]))


def test_interval_coverage_rejects_mismatched_bounds():
    with pytest.raises(ValueError, match="shape mismatch"):
        interval_coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0, 3.0]))


def test_evaluate_rul_predictions_rejects_only_one_bound():
    y = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="both"):
        evaluate_rul_predictions(y, y, lower=np.array([5.0, 15.0]))
    with pytest.raises(ValueError, match="both"):
        evaluate_rul_predictions(y, y, upper=np.array([15.0, 25.0]))


def test_late_prediction_rate():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([15.0, 15.0, 5.0, 10.0])  # 2 late, 1 early, 1 exact
    assert late_prediction_rate(y_true, y_pred) == pytest.approx(0.5)


def test_rmse_and_mae_are_symmetric():
    """Contrast with the NASA score -- these treat over/under equally, which
    is exactly why they are secondary metrics here."""
    y_true = np.array([50.0])
    over = np.array([60.0])
    under = np.array([40.0])
    assert rmse(y_true, over) == pytest.approx(rmse(y_true, under))
    assert mae(y_true, over) == pytest.approx(mae(y_true, under))


def test_interval_coverage():
    y_true = np.array([5.0, 15.0, 25.0, 35.0])
    lower = np.array([0.0, 10.0, 30.0, 30.0])
    upper = np.array([10.0, 20.0, 40.0, 32.0])
    # 5 in [0,10] -> inside; 15 in [10,20] -> inside;
    # 25 not in [30,40] -> outside; 35 not in [30,32] -> outside  => 2/4
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.5)


def test_interval_coverage_bounds_are_inclusive():
    y_true = np.array([10.0, 20.0])
    assert interval_coverage(y_true, np.array([10.0, 0.0]), np.array([15.0, 20.0])) == pytest.approx(1.0)


def test_evaluate_rul_predictions_includes_coverage_only_when_given():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    without = evaluate_rul_predictions(y_true, y_pred, model_name="m")
    assert "interval_coverage" not in without
    assert without["model"] == "m"

    with_ci = evaluate_rul_predictions(
        y_true, y_pred, lower=np.array([5.0, 15.0, 25.0]),
        upper=np.array([15.0, 25.0, 35.0]), model_name="m"
    )
    assert with_ci["interval_coverage"] == pytest.approx(1.0)


def test_calibration_bins_detect_systematic_bias():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 100, 200)
    y_pred = y_true + 15.0  # uniformly over-predicting by 15

    cal = calibration_by_risk_bin(y_true, y_pred, n_bins=4)
    assert len(cal) == 4
    # every bin should show ~+15 bias, not just the average
    assert np.allclose(cal["bias"].values, 15.0, atol=1.0)


def test_calibration_unbiased_model_sits_near_diagonal():
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0, 100, 300)
    y_pred = y_true + rng.normal(0, 2, 300)

    cal = calibration_by_risk_bin(y_true, y_pred, n_bins=5)
    assert np.abs(cal["bias"]).max() < 5.0