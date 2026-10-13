import numpy as np
import pandas as pd
import pytest

from turbofan_rul.cross_validation import build_truncated_eval_frame, sample_truncation_points


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