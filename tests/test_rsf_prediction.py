"""Tests for predict_rul_from_survival, using a stub model rather than a real
RSF fit -- the logic being tested (RMST integration, quantile lookup,
interval ordering) is independent of how the curves were produced, and a
stub keeps the test fast and deterministic."""

import numpy as np
import pandas as pd
import pytest

from turbofan_rul.models import predict_rul_from_survival


class _StubSurvivalModel:
    """Minimal stand-in exposing the two attributes predict_rul_from_survival uses."""

    def __init__(self, times, curves):
        self.unique_times_ = np.asarray(times, dtype=float)
        self._curves = curves

    def predict_survival_function(self, X):
        # sksurv returns callables that take an array of times
        return [lambda t, c=c: np.interp(t, self.unique_times_, c) for c in self._curves]


def _healthy_and_degraded():
    times = np.linspace(0, 100, 101)
    # healthy engine: survival stays high across the whole support
    healthy = np.clip(1.0 - times / 400.0, 0, 1)
    # degraded engine: drops to near zero early
    degraded = np.clip(1.0 - times / 40.0, 0, 1)
    return times, [healthy, degraded]


def test_rmst_larger_for_healthier_engine():
    times, curves = _healthy_and_degraded()
    model = _StubSurvivalModel(times, curves)
    X = pd.DataFrame(index=[0, 1])

    out = predict_rul_from_survival(model, X)
    assert out.loc[0, "pred_rul"] > out.loc[1, "pred_rul"]


def test_interval_bounds_are_ordered():
    """Only ordering is checked: the RMST point estimate is NOT guaranteed to
    fall inside the quantile interval, so that isn't asserted."""
    times, curves = _healthy_and_degraded()
    model = _StubSurvivalModel(times, curves)
    X = pd.DataFrame(index=[0, 1])

    out = predict_rul_from_survival(model, X)
    assert (out["pred_ci_low"] <= out["pred_ci_high"]).all()


def test_interval_bounds_are_first_time_survival_reaches_the_level():
    """Bounds are the first training time at which S(t) <= the level.

    Curves are chosen so each level is crossed BETWEEN grid points, so the
    expected answer doesn't hinge on floating-point equality at a boundary:
    - S(t) = 1 - t/50 crosses 0.95 at t=2.5 and 0.05 at t=47.5, so on an
      integer grid the bounds are t=3 and t=48.
    - S(t) = 1 - t/250 crosses 0.95 at t=12.5 (-> 13) and never reaches
      0.05 within the 0-100 support, so its upper bound pins to the last
      training time -- the RSF ceiling effect.
    """
    times = np.linspace(0, 100, 101)
    curves = [np.clip(1.0 - times / 250.0, 0, 1), np.clip(1.0 - times / 50.0, 0, 1)]
    model = _StubSurvivalModel(times, curves)
    out = predict_rul_from_survival(model, pd.DataFrame(index=[0, 1]))

    assert out.loc[1, "pred_ci_low"] == pytest.approx(3.0)
    assert out.loc[1, "pred_ci_high"] == pytest.approx(48.0)
    assert out.loc[0, "pred_ci_low"] == pytest.approx(13.0)
    assert out.loc[0, "pred_ci_high"] == pytest.approx(100.0)  # never reaches 0.05 -> ceiling


def test_rmst_matches_area_under_known_curve():
    # degraded curve: linear from 1 at t=0 to 0 at t=40, then 0 -> area = 20
    times, curves = _healthy_and_degraded()
    model = _StubSurvivalModel(times, curves)
    out = predict_rul_from_survival(model, pd.DataFrame(index=[0, 1]))
    assert out.loc[1, "pred_rul"] == pytest.approx(20.0)


def test_rmst_of_flat_survival_equals_time_support():
    # S(t) = 1 everywhere -> RMST should equal the full observed time range,
    # which is the saturation ceiling documented in the function's docstring.
    times = np.linspace(0, 50, 51)
    model = _StubSurvivalModel(times, [np.ones_like(times)])
    X = pd.DataFrame(index=[0])

    out = predict_rul_from_survival(model, X)
    assert out.loc[0, "pred_rul"] == pytest.approx(50.0)


def test_inverted_percentile_arguments_raise():
    times, curves = _healthy_and_degraded()
    model = _StubSurvivalModel(times, curves)
    X = pd.DataFrame(index=[0, 1])

    # swapping the convention should trip the internal assertion rather than
    # silently returning reversed intervals
    with pytest.raises(AssertionError):
        predict_rul_from_survival(model, X, lower_p=0.05, upper_p=0.95)


def test_output_index_matches_input():
    times, curves = _healthy_and_degraded()
    model = _StubSurvivalModel(times, curves)
    X = pd.DataFrame(index=[7, 42])

    out = predict_rul_from_survival(model, X)
    assert list(out.index) == [7, 42]