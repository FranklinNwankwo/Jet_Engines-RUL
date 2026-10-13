import numpy as np
import pandas as pd
import pytest

from turbofan_rul.features import (
    apply_regime_normalization,
    assign_regime_clusters,
    fit_regime_clusters,
    fit_regime_normalization_stats,
)


def _two_regime_df():
    # regime A: op_settings near (0, 0, 100); regime B: near (40, 0.8, 100)
    rng = np.random.default_rng(0)
    n = 100
    regime_a = pd.DataFrame({
        "op_setting_1": rng.normal(0, 0.01, n),
        "op_setting_2": rng.normal(0, 0.01, n),
        "op_setting_3": 100.0,
        "sensor_1": rng.normal(500, 1, n),   # regime A baseline level
    })
    regime_b = pd.DataFrame({
        "op_setting_1": rng.normal(40, 0.01, n),
        "op_setting_2": rng.normal(0.8, 0.01, n),
        "op_setting_3": 100.0,
        "sensor_1": rng.normal(600, 1, n),   # regime B baseline level -- shifted
    })
    return pd.concat([regime_a, regime_b], ignore_index=True)


def test_fit_regime_clusters_separates_two_distinct_regimes():
    df = _two_regime_df()
    km = fit_regime_clusters(df, n_clusters=2)
    labels = assign_regime_clusters(df, km)
    # first 100 rows (regime A) should all share one label, last 100 (regime B) the other
    assert len(set(labels[:100])) == 1
    assert len(set(labels[100:])) == 1
    assert labels[0] != labels[100]


def test_regime_normalization_removes_between_regime_level_shift():
    df = _two_regime_df()
    km = fit_regime_clusters(df, n_clusters=2)
    labels = assign_regime_clusters(df, km)
    stats = fit_regime_normalization_stats(df, ["sensor_1"], labels)
    normalized = apply_regime_normalization(df, ["sensor_1"], labels, stats)

    # before normalization, regime B's sensor_1 mean is ~100 higher than regime A's
    raw_gap = df["sensor_1"][100:].mean() - df["sensor_1"][:100].mean()
    assert raw_gap > 50

    # after normalization, both regimes should be centered near 0 -- the
    # level shift (which is regime, not degradation) should be gone
    norm_gap = normalized["sensor_1"][100:].mean() - normalized["sensor_1"][:100].mean()
    assert abs(norm_gap) < 0.5


def test_regime_normalization_stats_fit_on_train_only():
    df = _two_regime_df()
    km = fit_regime_clusters(df, n_clusters=2)
    labels = assign_regime_clusters(df, km)
    stats = fit_regime_normalization_stats(df, ["sensor_1"], labels)

    # a shifted "test" set using the SAME fitted stats should not silently
    # refit -- applying train-fitted stats to shifted data should NOT
    # recenter it to 0 (that would indicate stats were refit on the new data)
    shifted = df.copy()
    shifted["sensor_1"] = shifted["sensor_1"] + 1000
    normalized = apply_regime_normalization(shifted, ["sensor_1"], labels, stats)
    assert normalized["sensor_1"].mean() > 100  # still far from 0 -- stats weren't refit


def test_apply_regime_normalization_handles_unseen_regime_gracefully():
    """A label with no fitted stats can't come from `assign_regime_clusters`
    with the same KMeans model (it only predicts labels it was fit with), so
    this only happens if labels and stats come from different fits. It must
    not crash or produce NaN -- but those rows are normalized with their own
    on-the-fly stats, so they're not comparable to training-time features."""
    df = _two_regime_df()
    km = fit_regime_clusters(df, n_clusters=2)
    labels = assign_regime_clusters(df, km)
    stats = fit_regime_normalization_stats(df, ["sensor_1"], labels)

    # simulate an unseen regime label at apply-time
    fake_labels = labels.copy()
    fake_labels[0] = 99
    result = apply_regime_normalization(df, ["sensor_1"], fake_labels, stats)
    assert not result["sensor_1"].isna().any()


def test_assign_regime_clusters_only_returns_fitted_labels():
    """Why the unseen-regime fallback above is unreachable in normal use:
    even far-out operating settings get mapped to one of the fitted clusters."""
    df = _two_regime_df()
    km = fit_regime_clusters(df, n_clusters=2)
    stats = fit_regime_normalization_stats(df, ["sensor_1"], assign_regime_clusters(df, km))

    odd = pd.DataFrame({"op_setting_1": [1e6, -1e6], "op_setting_2": [5.0, -5.0],
                        "op_setting_3": [0.0, 1e4]})
    labels = assign_regime_clusters(odd, km)
    assert set(labels) <= set(stats.keys())


def test_single_regime_data_still_works():
    """FD001/FD003-like data: only one operating regime. Clustering with
    n_clusters=1 should be a well-defined no-op, not an error."""
    df = pd.DataFrame({
        "op_setting_1": [0.0] * 20, "op_setting_2": [0.0] * 20, "op_setting_3": [100.0] * 20,
        "sensor_1": np.linspace(500, 520, 20),
    })
    km = fit_regime_clusters(df, n_clusters=1)
    labels = assign_regime_clusters(df, km)
    assert len(set(labels)) == 1
    stats = fit_regime_normalization_stats(df, ["sensor_1"], labels)
    normalized = apply_regime_normalization(df, ["sensor_1"], labels, stats)
    assert abs(normalized["sensor_1"].mean()) < 1e-6