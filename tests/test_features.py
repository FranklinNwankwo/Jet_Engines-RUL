import numpy as np
import pandas as pd
import pytest

from turbofan_rul.features import (
    _rolling_slope,
    add_cycle_features,
    add_ewm_features,
    add_rolling_features,
    build_feature_matrix,
)


def _two_engine_df() -> pd.DataFrame:
    # engine 1: sensor rises then jumps; engine 2: constant.
    # The jump at the start of engine 2 must never appear in engine 1's
    # trailing-window features, and vice versa -- that's the leakage this
    # test suite exists to catch.
    e1 = pd.DataFrame({
        "unit_number": 1,
        "time_cycles": range(1, 11),
        "sensor_2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],  # spike only on last row
    })
    e2 = pd.DataFrame({
        "unit_number": 2,
        "time_cycles": range(1, 11),
        "sensor_2": [5.0] * 10,
    })
    return pd.concat([e1, e2], ignore_index=True)


def test_rolling_slope_positive_for_increasing_series():
    assert _rolling_slope(np.array([1.0, 2.0, 3.0, 4.0])) > 0


def test_rolling_slope_zero_for_flat_series():
    assert _rolling_slope(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)


def test_rolling_features_do_not_leak_across_engines():
    df = _two_engine_df()
    out = add_rolling_features(df, ["sensor_2"], windows=(5,))
    # engine 2's rolling mean should never be pulled toward engine 1's spike (100)
    engine2_means = out[out.unit_number == 2]["sensor_2_roll5_mean"]
    assert engine2_means.max() < 10


def test_rolling_features_use_min_periods_one_no_nans_at_start():
    df = _two_engine_df()
    out = add_rolling_features(df, ["sensor_2"], windows=(5,))
    first_row = out[out.unit_number == 1].iloc[0]
    assert not pd.isna(first_row["sensor_2_roll5_mean"])
    assert first_row["sensor_2_roll5_mean"] == 1.0  # only one point available yet


def test_ewm_features_do_not_leak_across_engines():
    df = _two_engine_df()
    out = add_ewm_features(df, ["sensor_2"], spans=(5,))
    engine2_ewm = out[out.unit_number == 2]["sensor_2_ewm5_mean"]
    assert engine2_ewm.max() < 10


def test_cycle_features_no_leakage_of_future_life_length():
    df = _two_engine_df()
    out = add_cycle_features(df)
    # cycle_count must just mirror observed cycle, never involve total trajectory length
    assert (out["cycle_count"] == out["time_cycles"]).all()


def test_build_feature_matrix_runs_end_to_end_and_preserves_row_count():
    df = _two_engine_df()
    out = build_feature_matrix(df, kept_sensors=["sensor_2"], windows=(5,), ewm_spans=(5,))
    assert len(out) == len(df)
    assert "sensor_2_roll5_slope" in out.columns
    assert "sensor_2_ewm5_mean" in out.columns
    assert out.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Index alignment. Every test above uses a 0..n-1 index, which hid a real bug:
# the old EWM code replaced the frame's index labels with 0..n-1, so on any
# frame whose index wasn't already 0..n-1 -- e.g. one CV fold's engines
# filtered out of the full training set -- EWM values landed on the wrong rows
# or became NaN. These tests use exactly that kind of filtered frame.
# ---------------------------------------------------------------------------

def _three_engine_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    parts = [
        pd.DataFrame({"unit_number": u, "time_cycles": range(1, n + 1),
                      "sensor_2": rng.normal(size=n).cumsum(),
                      "sensor_3": rng.normal(size=n).cumsum()})
        for u, n in [(1, 12), (2, 9), (3, 15)]
    ]
    return pd.concat(parts, ignore_index=True)


def test_features_on_filtered_frame_match_full_frame():
    """Featurizing a subset of engines (non-contiguous index, like a CV fold)
    must give exactly the same values as featurizing the full frame and then
    selecting those rows."""
    full = _three_engine_df()
    fold = full[full.unit_number.isin([2, 3])]  # index starts at 12, not 0
    assert fold.index[0] != 0

    kwargs = dict(kept_sensors=["sensor_2", "sensor_3"], windows=(3, 5), ewm_spans=(4,))
    feats_full = build_feature_matrix(full, **kwargs)
    feats_fold = build_feature_matrix(fold, **kwargs)

    assert feats_fold.index.equals(fold.index)
    assert feats_fold.isna().sum().sum() == 0
    pd.testing.assert_frame_equal(feats_fold, feats_full.loc[fold.index])


def test_ewm_on_filtered_frame_matches_per_engine_ewm():
    full = _three_engine_df()
    fold = full[full.unit_number.isin([1, 3])]  # gap in the index where engine 2 was
    out = add_ewm_features(fold, ["sensor_2"], spans=(4,))
    for unit, group in fold.groupby("unit_number"):
        expected = group["sensor_2"].ewm(span=4, adjust=False).mean()
        assert np.allclose(out.loc[group.index, "sensor_2_ewm4_mean"], expected)


def test_build_feature_matrix_rejects_duplicate_index():
    df = _two_engine_df()
    df.index = [0] * len(df)
    with pytest.raises(ValueError, match="unique index"):
        build_feature_matrix(df, kept_sensors=["sensor_2"], windows=(5,), ewm_spans=(5,))


def test_build_feature_matrix_rejects_rows_out_of_cycle_order():
    df = _two_engine_df()
    shuffled = df.iloc[[1, 0] + list(range(2, len(df)))]  # swap engine 1's first two cycles
    with pytest.raises(ValueError, match="increasing time_cycles"):
        build_feature_matrix(shuffled, kept_sensors=["sensor_2"], windows=(5,), ewm_spans=(5,))


def test_health_index_is_optional_and_finite():
    df = _three_engine_df()
    out = build_feature_matrix(df, kept_sensors=["sensor_2", "sensor_3"], windows=(5,),
                               ewm_spans=(5,), include_health_index=True)
    assert "health_index" in out.columns
    assert np.isfinite(out["health_index"]).all()
    assert "health_index" not in build_feature_matrix(df, kept_sensors=["sensor_2"],
                                                      windows=(5,), ewm_spans=(5,)).columns