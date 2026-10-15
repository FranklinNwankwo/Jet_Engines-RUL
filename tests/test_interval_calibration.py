"""Tests for interval_calibration.py: the candidate intervals and their
conformal calibration (Phase 24). Pure-numpy logic, no fitted models."""

import numpy as np
import pandas as pd
import pytest

from turbofan_rul import interval_calibration as ic


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(1, 200, n)
    pred = np.clip(true + rng.normal(0, 15, n), 0, 125)
    return pd.DataFrame({
        "true_rul": true, "rsf_pred": pred,
        "rsf_low": np.clip(pred - 20, 0, None), "rsf_high": np.minimum(pred + 20, 125),
        "weibull_low": pred - 5, "weibull_high": pred + 60,
    })


def test_hybrid_takes_cautious_lower_and_weibull_upper():
    low, high = ic.base_interval("hybrid", [10, 30], [50, 60], [20, 25], [90, 95])
    np.testing.assert_array_equal(low, [10, 25])
    np.testing.assert_array_equal(high, [90, 95])


@pytest.mark.parametrize("base", ["weibull", "rsf"])
def test_single_model_bases_pass_through(base):
    low, high = ic.base_interval(base, [10], [50], [20], [90])
    assert (low[0], high[0]) == ((20, 90) if base == "weibull" else (10, 50))


def test_unknown_base_and_mode_raise():
    with pytest.raises(ValueError, match="unknown interval base"):
        ic.base_interval("lstm", [1], [2], [1], [2])
    with pytest.raises(ValueError, match="unknown calibration mode"):
        ic.calibrate(_frame(), "hybrid", "isotonic")


def test_pred_band_edges():
    np.testing.assert_array_equal(ic.pred_band([0, 39.9, 40, 89.9, 90, 125]), [0, 0, 1, 1, 2, 2])


def test_none_mode_has_zero_corrections_and_clips_at_zero():
    cal = ic.calibrate(_frame(), "weibull", "none")
    assert cal.corrections == [0.0, 0.0, 0.0]
    low, high = cal.apply([2], [0], [10], [-3], [40])
    assert (low[0], high[0]) == (0.0, 40.0)


def test_global_mode_reaches_target_coverage_in_sample():
    frame = _frame(n=1000)
    cal = ic.calibrate(frame, "weibull", "global")
    assert len(set(cal.corrections)) == 1 and cal.corrections[0] >= 0
    low, high = cal.apply(frame.rsf_pred, frame.rsf_low, frame.rsf_high, frame.weibull_low, frame.weibull_high)
    assert ((frame.true_rul >= low) & (frame.true_rul <= high)).mean() >= 0.90


def test_mondrian_corrections_only_widen():
    frame = _frame(n=1000)
    cal = ic.calibrate(frame, "hybrid", "mondrian")
    assert all(q >= 0 for q in cal.corrections)
    base_low, base_high = ic.base_interval("hybrid", frame.rsf_low, frame.rsf_high,
                                           frame.weibull_low, frame.weibull_high)
    low, high = cal.apply(frame.rsf_pred, frame.rsf_low, frame.rsf_high, frame.weibull_low, frame.weibull_high)
    assert (low <= np.maximum(base_low, 0) + 1e-9).all() and (high >= base_high - 1e-9).all()


def test_mondrian_small_band_falls_back_to_global():
    frame = _frame(n=1000)
    frame.loc[frame.rsf_pred < 40, "rsf_pred"] = 50  # empty the first band
    frame = pd.concat([frame, frame.iloc[:5].assign(rsf_pred=10.0)], ignore_index=True)  # 5 points < MIN
    mondrian = ic.calibrate(frame, "hybrid", "mondrian")
    global_q = ic.calibrate(frame, "hybrid", "global").corrections[0]
    assert mondrian.corrections[0] == pytest.approx(global_q)


def test_to_dict_round_trip_is_json_safe():
    import json
    cal = ic.IntervalCalibration("hybrid", "mondrian", [1.5, 0.0, 3.25])
    d = json.loads(json.dumps(cal.to_dict()))
    assert d["pred_bands"][-1] is None
    back = ic.IntervalCalibration.from_dict(d)
    assert (back.base, back.mode, back.corrections) == ("hybrid", "mondrian", [1.5, 0.0, 3.25])
    assert back.name == "hybrid+mondrian"


def test_candidates_fixed_list():
    assert ("hybrid", "none") in ic.candidates() and len(ic.candidates()) == 6
