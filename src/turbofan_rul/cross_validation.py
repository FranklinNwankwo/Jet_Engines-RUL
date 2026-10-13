"""
Helpers for k-fold cross-validation that mirrors the *structure* of the
official C-MAPSS train/test split, not just a generic held-out row split.

Why this isn't just "hold out some rows"
------------------------------------------
Every training engine runs to failure, so a held-out training engine's last
row always has RUL=0 by construction -- evaluating there would be
degenerate. NASA's actual test engines are each truncated at some point
*before* failure, with the true remaining life at that truncation point
used as ground truth (`RUL_FD001.txt`). To get a fold evaluation with the
same structure as the Phase 6 test-set evaluation, each held-out engine in
a fold is given a single random truncation point (not its final cycle) and
scored against its true remaining life from there.

The structure matches; the distribution of truncation points does not
necessarily. NASA's official truncation points are fixed and were not
drawn from the [min_frac, max_frac] range used here, so absolute CV scores
are not directly comparable to official-test-set scores -- only rankings
and fold-to-fold stability are.

Feature engineering itself needs no special handling for this: every
engineered feature is a backward-looking rolling statistic (see
`features.py`), so computing them over an engine's full trajectory and then
selecting the row at the sampled truncation cycle gives the identical
result to computing them on a trajectory that had been truncated there in
the first place. Only the row selection simulates truncation, not the
feature computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_truncation_points(df: pd.DataFrame, unit_col: str = "unit_number",
                              cycle_col: str = "time_cycles",
                              min_frac: float = 0.3, max_frac: float = 0.9,
                              random_state: int = 42) -> dict[int, int]:
    """For each engine in `df`, sample one truncation cycle uniformly between
    `min_frac` and `max_frac` of its observed lifetime.

    The [0.3, 0.9] default range avoids truncating too early (near-empty
    history, uninformative features) or right at failure (the degenerate
    RUL=0 case this function exists to avoid) -- deliberately not [0, 1].

    Assumes each engine's cycles are numbered contiguously from 1, as in the
    raw C-MAPSS files, so every sampled cycle exists in the data.
    `build_truncated_eval_frame` raises if that assumption is ever broken.
    """
    rng = np.random.default_rng(random_state)
    max_cycles = df.groupby(unit_col)[cycle_col].max()
    truncation_points = {}
    for unit, max_cycle in max_cycles.items():
        frac = rng.uniform(min_frac, max_frac)
        truncation_points[unit] = max(1, int(round(max_cycle * frac)))
    return truncation_points


def build_truncated_eval_frame(featured_df: pd.DataFrame, truncation_points: dict[int, int],
                                unit_col: str = "unit_number",
                                cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Select each engine's row at its sampled truncation cycle, and attach
    the true (uncapped) remaining life from that point -- the eval-frame
    analogue of `labeling.build_test_survival_frame`, but for held-out
    training engines instead of the official test set.

    `featured_df` must already have engineered features computed on the
    FULL trajectory (see module docstring for why truncating first is
    unnecessary).

    Raises
    ------
    ValueError
        If any engine has no row at its truncation cycle. Silently skipping
        that engine would shrink the evaluation set without warning and bias
        fold metrics toward whichever engines happened to survive the filter.
    """
    max_cycles = featured_df.groupby(unit_col)[cycle_col].max()
    rows = []
    missing = []
    for unit, trunc_cycle in truncation_points.items():
        unit_rows = featured_df[featured_df[unit_col] == unit]
        row = unit_rows[unit_rows[cycle_col] == trunc_cycle]
        if row.empty:
            missing.append((unit, trunc_cycle))
            continue
        rows.append(row)
    if missing:
        raise ValueError(
            f"{len(missing)} engine(s) have no row at their truncation cycle "
            f"(unit, cycle): {missing[:10]}. Check that `featured_df` contains "
            f"the same engines as the frame passed to `sample_truncation_points`, "
            f"with contiguous cycle numbering."
        )
    eval_frame = pd.concat(rows, ignore_index=True)
    eval_frame["true_rul"] = eval_frame[unit_col].map(max_cycles) - eval_frame[cycle_col]
    return eval_frame


# ---------------------------------------------------------------------------
# Simulated fleet censoring (notebooks/13_censoring_experiment.ipynb)
# ---------------------------------------------------------------------------
#
# Every C-MAPSS training engine runs to failure, so the training data never
# contains the situation survival analysis exists for: engines that are
# still running when the model is built. These two helpers create it, so
# survival models and regression baselines can be compared on data that
# actually has censoring.
#
# The mechanism is administrative censoring, as in a real fleet observed up
# to a fixed date: each engine's observation window is C_i = s * u_i, with
# u_i ~ Uniform(0.5, 1.5) drawn independently of the engine's lifetime and
# a single fleet-wide scale s. An engine is censored when C_i < its
# lifetime. Because u_i is independent of lifetime, censoring is
# non-informative for each engine -- the assumption survival models rely
# on. (Censoring each engine at a fixed *fraction* of its own life, as
# `sample_truncation_points` does for evaluation, would make the censoring
# time depend on the lifetime, which survival models are not built for.)
# s is chosen so that exactly round(censored_fraction * n_engines) engines
# are censored, which keeps the censoring level controlled across runs.

def sample_administrative_censoring(df: pd.DataFrame, censored_fraction: float,
                                    unit_col: str = "unit_number",
                                    cycle_col: str = "time_cycles",
                                    spread: tuple[float, float] = (0.5, 1.5),
                                    random_state: int = 0) -> dict[int, int]:
    """Choose which training engines are censored, and when.

    Returns {unit: last observed cycle} for the censored engines only;
    engines not in the dict are observed through to failure. Every returned
    cycle is >= 1 and strictly below that engine's failure cycle.
    """
    if not 0.0 <= censored_fraction <= 1.0:
        raise ValueError(f"censored_fraction must be in [0, 1], got {censored_fraction}")
    lives = df.groupby(unit_col)[cycle_col].max()
    n_censor = int(round(censored_fraction * len(lives)))
    if n_censor == 0:
        return {}

    rng = np.random.default_rng(random_state)
    u = pd.Series(rng.uniform(spread[0], spread[1], size=len(lives)), index=lives.index)
    # Engine i is censored iff s * u_i < life_i, i.e. iff s < life_i / u_i.
    # Placing s between the n_censor-th and (n_censor+1)-th largest ratio
    # censors exactly n_censor engines.
    ratios = (lives / u).sort_values(ascending=False)
    upper = ratios.iloc[n_censor - 1]
    lower = ratios.iloc[n_censor] if n_censor < len(ratios) else 0.0
    s = (upper + lower) / 2.0

    censor_cycles = {}
    for unit in ratios.index[:n_censor]:
        cycle = int(np.floor(s * u[unit]))
        censor_cycles[int(unit)] = int(min(max(cycle, 1), lives[unit] - 1))
    return censor_cycles


def build_censored_train_frame(featured_df: pd.DataFrame, censor_cycles: dict[int, int],
                               cap: int = 125, unit_col: str = "unit_number",
                               cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Apply `censor_cycles` to a featured training survival frame.

    `featured_df` is the output of `labeling.build_train_survival_frame`
    (default labeling) passed through `features.build_feature_matrix`.
    Features are backward-looking, so computing them on the full trajectory
    and then dropping later rows equals computing them on the truncated
    trajectory.

    For each censored engine, rows after its censor cycle are dropped, and
    each remaining row's time-to-censoring t = censor_cycle - cycle becomes:
      * t >= cap: duration = cap, event = 1 -- the true capped RUL is known
        to be exactly `cap`, matching the project's default labeling;
      * t <  cap: duration = t, event = 0 -- right-censored: the engine is
        only known to survive at least t more cycles.
    The true `RUL` of censored engines' rows is set to NaN so it can't leak
    into a model, and an `engine_censored` column marks those engines.
    Engines not in `censor_cycles` are unchanged.
    """
    out = featured_df.copy()
    out["engine_censored"] = out[unit_col].isin(list(censor_cycles))
    if not censor_cycles:
        return out

    cut = out[unit_col].map(censor_cycles)
    keep = cut.isna() | (out[cycle_col] <= cut)
    out = out[keep].copy()
    cut = cut[keep]

    censored = out["engine_censored"]
    t = cut[censored] - out.loc[censored, cycle_col]
    out.loc[censored, "duration"] = t.clip(upper=cap)
    out.loc[censored, "event"] = (t >= cap).astype(int)
    out.loc[censored, "RUL"] = np.nan
    return out