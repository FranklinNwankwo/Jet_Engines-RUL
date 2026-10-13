import pandas as pd
import pytest

from turbofan_rul.labeling import (
    add_piecewise_rul,
    build_test_survival_frame,
    build_train_survival_frame,
)


def _synthetic_train_engine(unit: int, n_cycles: int) -> pd.DataFrame:
    return pd.DataFrame({
        "unit_number": unit,
        "time_cycles": range(1, n_cycles + 1),
        "sensor_2": range(n_cycles),
    })


def test_piecewise_rul_counts_down_to_zero_at_last_cycle():
    df = _synthetic_train_engine(1, n_cycles=50)
    labeled = add_piecewise_rul(df, cap=125)
    assert labeled["RUL"].iloc[-1] == 0
    assert labeled["RUL"].iloc[0] == 49


def test_piecewise_rul_is_capped():
    df = _synthetic_train_engine(1, n_cycles=300)
    labeled = add_piecewise_rul(df, cap=125)
    assert labeled["RUL"].max() == 125
    # first many rows should be flat at the cap, not still counting down linearly
    assert (labeled["RUL"].iloc[:100] == 125).all()


def test_piecewise_rul_is_per_engine_not_global():
    df = pd.concat([
        _synthetic_train_engine(1, n_cycles=20),
        _synthetic_train_engine(2, n_cycles=50),
    ], ignore_index=True)
    labeled = add_piecewise_rul(df, cap=125)
    unit1_last = labeled[labeled.unit_number == 1]["RUL"].iloc[-1]
    unit2_last = labeled[labeled.unit_number == 2]["RUL"].iloc[-1]
    assert unit1_last == 0
    assert unit2_last == 0  # each engine's own last cycle should hit 0, not leak across engines


def test_train_survival_frame_default_marks_every_row_as_event():
    """Default (censor_above_cap=False): every training row is event=1 at its
    capped remaining life -- including rows with more than `cap` cycles left,
    which sit at duration=cap. This is the project's chosen convention; see
    the labeling module docstring for why."""
    df = pd.concat([
        _synthetic_train_engine(1, n_cycles=200),
        _synthetic_train_engine(2, n_cycles=50),
    ], ignore_index=True)
    frame = build_train_survival_frame(df, cap=125)
    assert (frame["event"] == 1).all()
    assert (frame["duration"] == frame["RUL"]).all()
    assert frame["duration"].max() == 125
    assert (frame.loc[frame.unit_number == 1, "duration"] == 125).sum() == 75  # true RUL 199..125


def test_train_survival_frame_censor_above_cap_censors_rows_beyond_cap():
    """censor_above_cap=True: rows with more than `cap` cycles of true
    remaining life are right-censored at the cap (duration=cap, event=0)."""
    df = _synthetic_train_engine(1, n_cycles=200)  # true RUL runs 199 down to 0
    frame = build_train_survival_frame(df, cap=125, censor_above_cap=True)
    true_rul = 200 - frame["time_cycles"]

    censored = frame[true_rul > 125]
    observed = frame[true_rul <= 125]
    assert len(censored) == 74  # true RUL 199..126
    assert (censored["event"] == 0).all()
    assert (censored["duration"] == 125).all()
    assert (observed["event"] == 1).all()
    assert (observed["duration"] == true_rul[true_rul <= 125]).all()


def test_train_survival_frame_row_at_exactly_cap_is_an_event_either_way():
    """True RUL == cap means the failure time is known and inside the modeled
    window, so it's event=1 under both settings."""
    df = _synthetic_train_engine(1, n_cycles=200)
    for censor in (False, True):
        frame = build_train_survival_frame(df, cap=125, censor_above_cap=censor)
        row = frame[frame["time_cycles"] == 200 - 125].iloc[0]
        assert row["duration"] == 125
        assert row["event"] == 1


def test_train_survival_frame_settings_only_differ_above_cap():
    """The switch must only change rows beyond the cap: durations are always
    identical, and events differ exactly where true RUL > cap."""
    df = pd.concat([
        _synthetic_train_engine(1, n_cycles=200),
        _synthetic_train_engine(2, n_cycles=60),
    ], ignore_index=True)
    default = build_train_survival_frame(df, cap=125)
    censored = build_train_survival_frame(df, cap=125, censor_above_cap=True)
    true_rul = df.groupby("unit_number")["time_cycles"].transform("max") - df["time_cycles"]

    assert (default["duration"] == censored["duration"]).all()
    assert ((default["event"] != censored["event"]) == (true_rul > 125)).all()
    # every engine's final cycle is an observed failure at duration 0 either way
    for frame in (default, censored):
        last_rows = frame.groupby("unit_number").tail(1)
        assert (last_rows["duration"] == 0).all() and (last_rows["event"] == 1).all()


def test_test_survival_frame_is_censored_and_one_row_per_engine():
    test_df = pd.concat([
        _synthetic_train_engine(1, n_cycles=20),
        _synthetic_train_engine(2, n_cycles=50),
    ], ignore_index=True)
    rul_df = pd.DataFrame({"unit_number": [1, 2], "RUL": [80, 15]})

    frame = build_test_survival_frame(test_df, rul_df)

    assert len(frame) == 2  # one row per engine, not one per cycle
    assert (frame["event"] == 0).all()
    row1 = frame[frame.unit_number == 1].iloc[0]
    assert row1["duration"] == 20
    assert row1["true_rul"] == 80


def test_test_survival_frame_true_rul_not_used_to_derive_duration():
    # duration must come purely from observed cycles, never from the (unknown-
    # at-deployment-time) true_rul ground truth -- this guards against a
    # label-leakage regression.
    test_df = _synthetic_train_engine(1, n_cycles=20)
    rul_df = pd.DataFrame({"unit_number": [1], "RUL": [999]})
    frame = build_test_survival_frame(test_df, rul_df)
    assert frame.iloc[0]["duration"] == 20
