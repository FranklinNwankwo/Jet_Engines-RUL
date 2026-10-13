import numpy as np
import pandas as pd
import pytest

from turbofan_rul import features, labeling
from turbofan_rul.cross_validation import (
    build_censored_train_frame,
    build_truncated_eval_frame,
    sample_administrative_censoring,
    sample_truncation_points,
)


def _synthetic_engines():
    e1 = pd.DataFrame({"unit_number": 1, "time_cycles": range(1, 101), "x": range(100)})
    e2 = pd.DataFrame({"unit_number": 2, "time_cycles": range(1, 51), "x": range(50)})
    return pd.concat([e1, e2], ignore_index=True)


def test_sample_truncation_points_within_bounds():
    df = _synthetic_engines()
    points = sample_truncation_points(df, min_frac=0.3, max_frac=0.9, random_state=0)
    assert points[1] >= 30 and points[1] <= 90
    assert points[2] >= 15 and points[2] <= 45


def test_sample_truncation_points_never_zero():
    tiny = pd.DataFrame({"unit_number": 3, "time_cycles": [1, 2], "x": [0, 1]})
    points = sample_truncation_points(tiny, min_frac=0.3, max_frac=0.9, random_state=0)
    assert points[3] >= 1


def test_sample_truncation_points_reproducible_with_same_seed():
    df = _synthetic_engines()
    p1 = sample_truncation_points(df, random_state=7)
    p2 = sample_truncation_points(df, random_state=7)
    assert p1 == p2


def test_sample_truncation_points_one_per_engine():
    df = _synthetic_engines()
    points = sample_truncation_points(df, random_state=1)
    assert set(points.keys()) == {1, 2}


def test_build_truncated_eval_frame_true_rul_is_remaining_life():
    df = _synthetic_engines()
    # engine 1 has 100 cycles total; truncate at cycle 60 -> true_rul should be 40
    truncation_points = {1: 60, 2: 30}
    eval_frame = build_truncated_eval_frame(df, truncation_points)

    row1 = eval_frame[eval_frame.unit_number == 1].iloc[0]
    assert row1["time_cycles"] == 60
    assert row1["true_rul"] == 100 - 60

    row2 = eval_frame[eval_frame.unit_number == 2].iloc[0]
    assert row2["true_rul"] == 50 - 30


def test_build_truncated_eval_frame_never_selects_final_cycle_by_construction():
    """The whole point of this module -- confirm the selected row is never
    an engine's last cycle (which would make true_rul=0, the degenerate
    case this module exists to avoid)."""
    df = _synthetic_engines()
    points = sample_truncation_points(df, min_frac=0.3, max_frac=0.9, random_state=0)
    eval_frame = build_truncated_eval_frame(df, points)
    assert (eval_frame["true_rul"] > 0).all()


def test_build_truncated_eval_frame_one_row_per_engine():
    df = _synthetic_engines()
    points = {1: 60, 2: 30}
    eval_frame = build_truncated_eval_frame(df, points)
    assert len(eval_frame) == 2
    assert set(eval_frame["unit_number"]) == {1, 2}


def test_build_truncated_eval_frame_raises_when_truncation_row_missing():
    """An engine with no row at its truncation cycle must raise, not be
    silently dropped -- dropping it would shrink the fold's evaluation set
    without any warning."""
    df = _synthetic_engines()
    # engine 2 only has cycles 1-50, so cycle 80 doesn't exist
    with pytest.raises(ValueError, match="no row at their truncation cycle"):
        build_truncated_eval_frame(df, {1: 60, 2: 80})


def test_build_truncated_eval_frame_raises_when_engine_absent():
    df = _synthetic_engines()
    with pytest.raises(ValueError, match="no row at their truncation cycle"):
        build_truncated_eval_frame(df, {1: 60, 99: 10})


# ---------------------------------------------------------------------------
# Simulated fleet censoring
# ---------------------------------------------------------------------------

def _fleet(n_engines=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(130, 300))
        frames.append(pd.DataFrame({
            "unit_number": unit,
            "time_cycles": np.arange(1, life + 1),
            "sensor_a": rng.normal(size=life).cumsum(),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_administrative_censoring_hits_exact_fraction(fraction):
    df = _fleet()
    cens = sample_administrative_censoring(df, fraction, random_state=1)
    assert len(cens) == round(fraction * 40)


def test_administrative_censoring_cycles_are_before_failure():
    df = _fleet()
    lives = df.groupby("unit_number")["time_cycles"].max()
    cens = sample_administrative_censoring(df, 0.5, random_state=2)
    for unit, cycle in cens.items():
        assert 1 <= cycle < lives[unit]


def test_administrative_censoring_reproducible_and_seed_dependent():
    df = _fleet()
    a = sample_administrative_censoring(df, 0.5, random_state=3)
    b = sample_administrative_censoring(df, 0.5, random_state=3)
    c = sample_administrative_censoring(df, 0.5, random_state=4)
    assert a == b
    assert a != c


def test_administrative_censoring_rejects_bad_fraction():
    with pytest.raises(ValueError):
        sample_administrative_censoring(_fleet(), 1.5)


def _featured_fleet(cap=125):
    df = _fleet(n_engines=10)
    surv = labeling.build_train_survival_frame(df, cap=cap)
    return df, features.build_feature_matrix(surv, kept_sensors=["sensor_a"])


def test_censored_frame_labels_censored_engines_correctly():
    cap = 125
    _, featured = _featured_fleet(cap)
    cens = {1: 60, 2: 200}
    out = build_censored_train_frame(featured, cens, cap=cap)

    e1 = out[out.unit_number == 1]
    assert e1.time_cycles.max() == 60                    # rows after the cut are gone
    assert (e1.event == 0).all()                         # 60 - cycle < 125 everywhere
    assert np.allclose(e1.duration, 60 - e1.time_cycles)
    assert e1.RUL.isna().all()                           # true RUL hidden
    assert e1.engine_censored.all()

    e2 = out[out.unit_number == 2]
    long_window = e2[200 - e2.time_cycles >= cap]
    short_window = e2[200 - e2.time_cycles < cap]
    assert (long_window.event == 1).all() and (long_window.duration == cap).all()
    assert (short_window.event == 0).all()


def test_censored_frame_leaves_uncensored_engines_unchanged():
    _, featured = _featured_fleet()
    out = build_censored_train_frame(featured, {1: 60})
    before = featured[featured.unit_number == 3]
    after = out[out.unit_number == 3].drop(columns="engine_censored")
    # RUL becomes float once censored engines' values are NaN; the values
    # themselves must be identical.
    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_censored_frame_features_match_truncate_then_featurize():
    """Features are backward-looking, so dropping rows after the cut must
    give the same feature values as building features on the truncated
    trajectory -- the assumption build_censored_train_frame relies on."""
    raw, featured = _featured_fleet()
    out = build_censored_train_frame(featured, {1: 60})
    truncated_raw = raw[(raw.unit_number != 1) | (raw.time_cycles <= 60)]
    refeatured = features.build_feature_matrix(
        labeling.build_train_survival_frame(truncated_raw), kept_sensors=["sensor_a"])
    feat_cols = [c for c in refeatured.columns if c.startswith("sensor_a_") or c == "cycle_count"]
    a = out[out.unit_number == 1][feat_cols].reset_index(drop=True)
    b = refeatured[refeatured.unit_number == 1][feat_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_censored_frame_with_no_censoring_is_identity():
    _, featured = _featured_fleet()
    out = build_censored_train_frame(featured, {})
    pd.testing.assert_frame_equal(featured, out.drop(columns="engine_censored"))