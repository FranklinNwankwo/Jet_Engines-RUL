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