"""
RUL labeling and survival (duration, event) table construction for C-MAPSS.

Two distinct things happen in this module, and they answer two different
questions -- keeping them as separate functions is deliberate, since
conflating them is an easy way to quietly leak information:

1. `add_piecewise_rul` answers "what is the target label for row-level /
   regression-style models?" -- the standard piecewise-linear RUL convention
   from the C-MAPSS literature (true remaining life, clipped at a cap).

2. `build_train_survival_frame` / `build_test_survival_frame` answer "what is
   the (duration, event) pair for fitting a survival model?" -- see
   docs/project_brief.md Section 3.

How the cap is encoded in the training set
--------------------------------------------
Every training engine is observed all the way to failure, so each training
row's true remaining life is known exactly. The cap (125 cycles by default)
limits the model to the last `cap` cycles before failure, where degradation
is actually visible. There are two ways to encode rows whose true remaining
life is above the cap, selected by `censor_above_cap`:

- `censor_above_cap=False` (DEFAULT, used throughout this project):
  duration = cap, event = 1. This is the standard C-MAPSS piecewise-linear
  target used as a survival label. Strictly speaking it's not true survival
  semantics -- those engines didn't fail at `cap` -- but it deliberately
  tells every model "remaining life beyond `cap` isn't distinguished", which
  matches how the task is evaluated: every official test engine has a true
  remaining life within roughly the same range.

- `censor_above_cap=True`: duration = cap, event = 0 (right-censored --
  "survived at least `cap` more cycles"). This is the statistically exact
  encoding. It was tested and rejected for this project: on the official
  FD001 test set, Cox PH improved slightly and RSF was essentially
  unchanged, but Weibull AFT -- whose parametric tail can extrapolate past
  the cap -- predicted 250-300 cycles of remaining life for healthy-looking
  engines whose true remaining life is at most 145, so its NASA score went
  from ~1.3e3 to ~2e10 and its 90% interval nearly doubled in width. Since
  Weibull supplies the deployed confidence interval, the default stays
  False. The option is kept so that comparison can be reproduced.

Rows with true remaining life <= cap are identical under both settings:
duration = true remaining life, event = 1.

The test set is separate: each test engine is truncated before failure, so
its true remaining life is unknown at prediction time (event = 0). Treating
each cycle of a training engine as a pseudo-independent survival "subject"
is a known simplification (rows within one engine are correlated, not
i.i.d.) -- standard in the C-MAPSS survival-modeling literature, and called
out here rather than glossed over.
"""

from __future__ import annotations

import pandas as pd


def _true_rul(df: pd.DataFrame, unit_col: str, cycle_col: str) -> pd.Series:
    """Uncapped remaining life per row of a full run-to-failure trajectory."""
    max_cycle_per_unit = df.groupby(unit_col)[cycle_col].transform("max")
    return max_cycle_per_unit - df[cycle_col]


def add_piecewise_rul(df: pd.DataFrame, cap: int = 125, unit_col: str = "unit_number",
                       cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Add a piecewise-linear RUL column to a *training* (full run-to-failure) dataframe.

    For each engine, RUL at a given cycle = (final observed cycle - current cycle),
    then clipped at `cap`. The cap encodes the standard C-MAPSS assumption that an
    engine's early-life cycles carry no meaningful degradation signal, so it's not
    useful (and can hurt training) to force a model to fit an uncapped linear
    countdown from cycle 1 -- see docs/project_brief.md Section 4.

    Parameters
    ----------
    df : dataframe containing at least [unit_col, cycle_col], one row per cycle
    cap : maximum RUL value; commonly 125 in the C-MAPSS literature

    Returns
    -------
    A copy of df with an added "RUL" column.
    """
    df = df.copy()
    df["RUL"] = _true_rul(df, unit_col, cycle_col).clip(upper=cap)
    return df


def build_train_survival_frame(df: pd.DataFrame, cap: int = 125, unit_col: str = "unit_number",
                                cycle_col: str = "time_cycles",
                                censor_above_cap: bool = False) -> pd.DataFrame:
    """Build the row-level (duration, event) survival table for training engines.

    - duration = true remaining life, clipped at `cap`
    - event = 1 for every row when `censor_above_cap=False` (the default);
      when `censor_above_cap=True`, rows whose true remaining life exceeds
      `cap` get event = 0 (right-censored at `cap`) instead.

    See the module docstring for why the default is False and what was
    measured when it was set to True.

    Returns
    -------
    A copy of df with added "RUL", "duration" and "event" columns.
    """
    labeled = df.copy()
    true_rul = _true_rul(labeled, unit_col, cycle_col)
    labeled["RUL"] = true_rul.clip(upper=cap)
    labeled["duration"] = labeled["RUL"]
    if censor_above_cap:
        labeled["event"] = (true_rul <= cap).astype(int)
    else:
        labeled["event"] = 1
    return labeled


def build_test_survival_frame(test_df: pd.DataFrame, rul_df: pd.DataFrame,
                               unit_col: str = "unit_number",
                               cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Build the engine-level (duration, event) survival table for the test set.

    Each test engine is right-censored at its last observed cycle: we take the
    *final* observed row per engine (the most recent, most informative snapshot)
    as the survival "subject," with duration = cycles observed and event = 0.

    `rul_df` (from RUL_FD00X.txt) supplies the ground-truth remaining life at
    truncation. It is attached as `true_rul` for evaluation purposes only --
    it must never be used as a model input, since in a real deployment it is
    exactly the unknown quantity being predicted. Keeping it in a clearly
    separate, clearly-named column (rather than folding it into `duration`)
    makes that boundary hard to cross by accident.

    Note: `duration` here is the engine's age at truncation, not a remaining-life
    value like the training frame's `duration`. No model is fit or scored on it;
    evaluation uses `true_rul`.
    """
    last_rows = (
        test_df.sort_values(cycle_col)
        .groupby(unit_col, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    last_rows["duration"] = last_rows[cycle_col]
    last_rows["event"] = 0
    merged = last_rows.merge(rul_df[[unit_col, "RUL"]].rename(columns={"RUL": "true_rul"}),
                              on=unit_col, how="left")
    return merged