"""
Feature engineering for C-MAPSS sensor trajectories.

The raw sensor level at a single cycle is noisy and, on its own, a weak
predictor of remaining life -- the degradation signal lives in how a sensor's
*level and rate of change* evolve over the recent history of an engine, which
is why every feature here is computed over a rolling window rather than from
a single row. See docs/project_brief.md Section 5 for the layer-by-layer
rationale. Sensor selection (layer 1) is read from Phase 1's output rather
than redone here (`load_kept_sensors`); this module implements rolling
statistics, trend slopes, EWMA and cycle count (layers 2-5), regime
clustering + within-regime normalization for multi-regime subsets (layer 6),
and the optional composite health index (layer 7, `add_health_index`).

Everything here operates **per engine** (grouped by unit_number) -- rolling
windows must never cross an engine boundary, or a feature at the start of
engine B's life would quietly include trailing cycles from engine A. Every
window operation groups by unit_number, and every per-engine result is
written back aligned to the input's own index labels, so the input does NOT
need a 0..n-1 index (e.g. a frame filtered to one CV fold's engines works).
The input does need a unique index and, within each engine, rows in
increasing cycle order -- `build_feature_matrix` checks both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (5, 15, 30)
DEFAULT_EWM_SPANS = (10, 30)


def fit_regime_clusters(df: pd.DataFrame, op_setting_cols: list[str] = ("op_setting_1", "op_setting_2", "op_setting_3"),
                         n_clusters: int = 6, random_state: int = 42):
    """Fit a KMeans model identifying an engine's discrete operating regime
    from its operating settings.

    FD001/FD003 have a single operating regime, so this step isn't used
    there. FD002/FD004 cycle through 6 discrete regimes -- confirmed
    empirically (not assumed) by within-cluster std on the operating
    settings dropping to ~0 once clustered, in the notebook that calls this
    function. Fit on TRAINING data only, same discipline as every other
    scaler/model in this project.
    """
    from sklearn.cluster import KMeans

    op_settings = df[list(op_setting_cols)].to_numpy()
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(op_settings)


def assign_regime_clusters(df: pd.DataFrame, kmeans_model,
                            op_setting_cols: list[str] = ("op_setting_1", "op_setting_2", "op_setting_3")) -> np.ndarray:
    """Assign each row's regime cluster using an already-fitted KMeans model."""
    return kmeans_model.predict(df[list(op_setting_cols)].to_numpy())


def fit_regime_normalization_stats(df: pd.DataFrame, sensor_cols: list[str],
                                    regime_labels: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-regime (mean, std) for each sensor, fit on training data.

    This is the actual point of clustering: a sensor's absolute level shifts
    with operating regime independent of degradation, so a rolling-window
    feature computed on raw values would conflate "the regime changed" with
    "the engine degraded." Normalizing within each regime -- subtracting
    that regime's mean, dividing by that regime's std -- removes the
    regime-driven component before any rolling/EWMA feature is computed,
    so what's left should track degradation only.
    """
    stats = {}
    working = df[sensor_cols].copy()
    working["_regime"] = regime_labels
    for regime, group in working.groupby("_regime"):
        mean = group[sensor_cols].mean().to_numpy()
        std = group[sensor_cols].std().to_numpy() + 1e-8
        stats[int(regime)] = (mean, std)
    return stats


def apply_regime_normalization(df: pd.DataFrame, sensor_cols: list[str], regime_labels: np.ndarray,
                                stats: dict[int, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Apply per-regime normalization fit by `fit_regime_normalization_stats`.

    When labels come from `assign_regime_clusters` with the same fitted
    KMeans model used to build `stats`, every label is guaranteed to have
    stats (KMeans.predict only returns labels it was fit with). The fallback
    below -- normalizing an unknown regime with its own on-the-fly mean/std --
    only triggers if the labels and stats come from different fits, and in
    that case the output is not comparable to training-time features.
    """
    out = df.copy()
    regime_labels = np.asarray(regime_labels)
    normalized = np.zeros((len(df), len(sensor_cols)), dtype=float)
    raw = df[sensor_cols].to_numpy()

    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if int(regime) in stats:
            mean, std = stats[int(regime)]
        else:
            mean = raw[mask].mean(axis=0)
            std = raw[mask].std(axis=0) + 1e-8
        normalized[mask] = (raw[mask] - mean) / std

    out[sensor_cols] = normalized
    return out


def load_kept_sensors(path: Path | str) -> list[str]:
    """Load the sensor list produced by the Phase 1 EDA notebook.

    Reading this from disk (rather than re-deriving a sensor list here) is
    deliberate: the EDA decision about which sensors carry signal and the
    feature pipeline that consumes it should not be able to silently drift
    apart from each other.
    """
    with open(path) as f:
        selection = json.load(f)
    return selection["kept_sensors"]


def _check_frame(df: pd.DataFrame, unit_col: str, cycle_col: str) -> None:
    """Raise if `df` can't be featurized correctly: duplicate index labels
    (per-engine results are aligned back by label) or rows out of cycle order
    within an engine (rolling/EWMA windows follow row order)."""
    if not df.index.is_unique:
        raise ValueError("df must have a unique index; call df.reset_index(drop=True) first.")
    if cycle_col in df.columns:
        steps = df.groupby(unit_col, sort=False)[cycle_col].diff()
        if (steps <= 0).any():
            bad = df.loc[steps <= 0, unit_col].unique()[:10].tolist()
            raise ValueError(
                f"rows must be in strictly increasing {cycle_col} order within each engine; "
                f"out of order for {unit_col}(s) {bad}. Sort by [{unit_col}, {cycle_col}] first."
            )


def _rolling_slope(values: np.ndarray) -> float:
    """Slope of an ordinary least-squares fit of `values` against a 0..n-1 index.

    Used as a rolling-window trend feature: a direct estimate of a sensor's
    instantaneous rate of change, which is often more predictive of imminent
    failure than the sensor's raw level.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str],
                          windows: tuple[int, ...] = DEFAULT_WINDOWS,
                          unit_col: str = "unit_number") -> pd.DataFrame:
    """Add rolling mean/std/min/max/slope per sensor, per window size, per engine.

    `min_periods=1` is used deliberately rather than requiring a full window:
    the alternative (dropping the first `window` rows of every engine) would
    throw away exactly the early-life, "still looks healthy" rows that give
    the model its contrast against the degrading rows later in the
    trajectory -- and at inference time, a newly-deployed sensor stream won't
    have a full window of history yet either. The tradeoff is that
    early-window statistics are noisier (computed on fewer points); this is
    an explicit, documented choice, not an oversight.

    `groupby(...).rolling(...)` returns a (unit, original index) MultiIndex;
    dropping the unit level leaves the original index labels, which
    `pd.DataFrame(new_cols, index=df.index)` then aligns on.
    """
    grouped = df.groupby(unit_col, group_keys=False)
    new_cols = {}

    for sensor in sensor_cols:
        for window in windows:
            roll = grouped[sensor].rolling(window=window, min_periods=1)
            new_cols[f"{sensor}_roll{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{sensor}_roll{window}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            new_cols[f"{sensor}_roll{window}_min"] = roll.min().reset_index(level=0, drop=True)
            new_cols[f"{sensor}_roll{window}_max"] = roll.max().reset_index(level=0, drop=True)
            new_cols[f"{sensor}_roll{window}_slope"] = (
                grouped[sensor]
                .rolling(window=window, min_periods=2)
                .apply(_rolling_slope, raw=True)
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

    # Built as a dict and concatenated once, rather than assigning columns
    # one at a time onto `df` -- with ~200+ engineered columns, repeated
    # single-column assignment fragments the DataFrame's internal block
    # structure badly enough that pandas warns about it, and it's
    # meaningfully slower. Concatenating once keeps the result identical
    # while avoiding that.
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_ewm_features(df: pd.DataFrame, sensor_cols: list[str],
                      spans: tuple[int, ...] = DEFAULT_EWM_SPANS,
                      unit_col: str = "unit_number") -> pd.DataFrame:
    """Add exponentially-weighted moving averages per sensor, per span, per engine.

    Benchmarked as an alternative smoothing/trend-extraction approach to the
    simple rolling window above -- EWMA weights recent cycles more heavily
    without a hard window cutoff, which can track an accelerating degradation
    trend more responsively. Both are kept so the modeling notebooks can
    compare which representation each model actually benefits from, rather
    than assuming one is strictly better.

    Uses `groupby(...).transform`, which always returns a result indexed
    exactly like `df`. (An earlier version used `groupby(...).apply(...)`
    followed by `reset_index(level=0, drop=True)`; because that apply returns
    a single-level index, the reset replaced the real labels with 0..n-1, so
    on any frame whose index wasn't already 0..n-1 -- e.g. one CV fold's
    engines filtered out of the full training set -- the EWM columns were
    misaligned or NaN.)
    """
    grouped = df.groupby(unit_col, group_keys=False)
    new_cols = {}
    for sensor in sensor_cols:
        for span in spans:
            new_cols[f"{sensor}_ewm{span}_mean"] = grouped[sensor].transform(
                lambda s, sp=span: s.ewm(span=sp, adjust=False).mean()
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_cycle_features(df: pd.DataFrame, unit_col: str = "unit_number",
                        cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Add cycle-count-based features.

    Just the raw cycle count for now. Deliberately does *not* include a
    "fraction of life elapsed" feature (current_cycle / total_life) -- that
    would require knowing an engine's eventual failure cycle, which is
    exactly the unknown quantity at inference time. Including it here would
    silently leak the label into the features.
    """
    df = df.copy()
    df["cycle_count"] = df[cycle_col]
    return df


def add_health_index(df: pd.DataFrame, sensor_cols: list[str], n_components: int = 1,
                      healthy_cycles: int = 20, unit_col: str = "unit_number") -> pd.DataFrame:
    """(Stretch, optional) Add a composite "distance from healthy" health index.

    Fits a StandardScaler + PCA on the first `healthy_cycles` rows of every
    engine in `df` (pooled; assumed close to a healthy baseline), then scores
    every row's reconstruction error against that baseline -- a single scalar
    that should trend upward as an engine degrades. This is intentionally the
    last feature added, and optional: benchmark it against the rolling/EWMA
    features before assuming it helps, rather than including it by default.

    Limitation: the scaler and PCA are fit on whatever `df` is passed in and
    are not returned. Calling this separately on training data and on test
    or inference data therefore fits two different baselines, so the
    resulting health_index values are not on a comparable scale. Fit-once,
    apply-everywhere support would be needed before using this in the
    deployed pipeline.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    healthy_rows = df.groupby(unit_col, group_keys=False).head(healthy_cycles)[sensor_cols]

    scaler = StandardScaler().fit(healthy_rows)
    pca = PCA(n_components=n_components).fit(scaler.transform(healthy_rows))

    X = scaler.transform(df[sensor_cols])
    reconstructed = pca.inverse_transform(pca.transform(X))
    reconstruction_error = np.sum((X - reconstructed) ** 2, axis=1)

    df = df.copy()
    df["health_index"] = reconstruction_error
    return df


def build_feature_matrix(df: pd.DataFrame, kept_sensors: list[str],
                          windows: tuple[int, ...] = DEFAULT_WINDOWS,
                          ewm_spans: tuple[int, ...] = DEFAULT_EWM_SPANS,
                          include_health_index: bool = False,
                          unit_col: str = "unit_number",
                          cycle_col: str = "time_cycles") -> pd.DataFrame:
    """Run the full feature-engineering pipeline in the fixed order used by both
    the training notebooks and the deployed API -- this is the single function
    that should be called from anywhere features are needed, so the two paths
    can't drift apart.

    Raises ValueError if `df` has duplicate index labels or rows out of cycle
    order within an engine (see module docstring).
    """
    _check_frame(df, unit_col, cycle_col)
    out = add_cycle_features(df, unit_col=unit_col, cycle_col=cycle_col)
    out = add_rolling_features(out, kept_sensors, windows=windows, unit_col=unit_col)
    out = add_ewm_features(out, kept_sensors, spans=ewm_spans, unit_col=unit_col)
    if include_health_index:
        out = add_health_index(out, kept_sensors, unit_col=unit_col)
    return out