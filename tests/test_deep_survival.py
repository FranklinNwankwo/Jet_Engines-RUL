import numpy as np
import pandas as pd
import pytest

from turbofan_rul.deep_survival import (
    DiscreteHazardMLP,
    assign_bins,
    build_mask_and_targets,
    build_sequence_windows,
    gradient_check,
    make_bin_edges,
    predict_rul_from_discrete_survival,
)


def test_make_bin_edges_spans_zero_to_cap():
    edges = make_bin_edges(cap=125, n_bins=25)
    assert edges[0] == 0
    assert edges[-1] == 125
    assert len(edges) == 26


def test_assign_bins_endpoints():
    edges = make_bin_edges(cap=100, n_bins=10)  # bins of width 10
    bins = assign_bins(np.array([0.0, 5.0, 10.0, 95.0, 100.0]), edges)
    assert bins[0] == 1     # duration 0 -> first bin
    assert bins[2] == 1     # duration exactly at edge 10 -> still bin 1 (right-closed)
    assert bins[-1] == 10   # duration at cap -> last bin
    assert (bins >= 1).all() and (bins <= 10).all()


def test_build_mask_and_targets_event_row():
    # one observation, event in bin 3 of 5
    bins = np.array([3])
    events = np.array([1])
    M, Y = build_mask_and_targets(bins, events, n_bins=5)
    assert M.tolist() == [[1, 1, 1, 0, 0]]  # at risk through bin 3
    assert Y.tolist() == [[0, 0, 1, 0, 0]]  # event exactly at bin 3


def test_build_mask_and_targets_censored_row():
    # censored (event=0) at bin 3 -> at risk through bin 3, no target=1 anywhere
    bins = np.array([3])
    events = np.array([0])
    M, Y = build_mask_and_targets(bins, events, n_bins=5)
    assert M.tolist() == [[1, 1, 1, 0, 0]]
    assert Y.tolist() == [[0, 0, 0, 0, 0]]


def test_build_sequence_windows_no_leakage_across_engines():
    e1 = pd.DataFrame({"unit_number": 1, "time_cycles": range(1, 6), "s": [1, 1, 1, 1, 1]})
    e2 = pd.DataFrame({"unit_number": 2, "time_cycles": range(1, 6), "s": [9, 9, 9, 9, 9]})
    df = pd.concat([e1, e2], ignore_index=True)

    windows = build_sequence_windows(df, sensor_cols=["s"], window=3)
    # engine 2's first row's window should be all 9s (padded with its own first value),
    # never containing engine 1's trailing 1s
    engine2_first_row_window = windows[5]  # index 5 = first row of engine 2
    assert np.all(engine2_first_row_window == 9)


def test_build_sequence_windows_pads_early_rows_by_repeating_first_value():
    df = pd.DataFrame({"unit_number": 1, "time_cycles": range(1, 4), "s": [10.0, 20.0, 30.0]})
    windows = build_sequence_windows(df, sensor_cols=["s"], window=5)
    # first row: only 1 real observation (10), padded with 4 copies of 10
    assert windows[0].tolist() == [10.0, 10.0, 10.0, 10.0, 10.0]
    # third row: 3 real observations, padded with 2 copies of the first (10)
    assert windows[2].tolist() == [10.0, 10.0, 10.0, 20.0, 30.0]


def test_build_sequence_windows_shape():
    df = pd.DataFrame({"unit_number": [1] * 4, "time_cycles": range(1, 5),
                        "s1": range(4), "s2": range(4)})
    windows = build_sequence_windows(df, sensor_cols=["s1", "s2"], window=3)
    assert windows.shape == (4, 3 * 2)


def _toy_model_and_data(random_state=0):
    rng = np.random.default_rng(random_state)
    n, d, n_bins = 12, 6, 5
    X = rng.normal(size=(n, d))
    durations = rng.uniform(0, 100, size=n)
    events = rng.integers(0, 2, size=n)
    edges = make_bin_edges(cap=100, n_bins=n_bins)
    bins = assign_bins(durations, edges)
    M, Y = build_mask_and_targets(bins, events, n_bins)
    model = DiscreteHazardMLP(n_features=d, n_bins=n_bins, hidden_size=4, l2=1e-2, random_state=random_state)
    return model, X, M, Y, edges


def test_gradient_check_passes():
    model, X, M, Y, _ = _toy_model_and_data()
    errors = gradient_check(model, X, M, Y, n_checks_per_param=6)
    for name, err in errors.items():
        assert err < 1e-4, f"{name} gradient relative error too high: {err:.2e}"


def test_forward_outputs_valid_probabilities():
    model, X, M, Y, _ = _toy_model_and_data()
    H = model.predict_hazard(X)
    assert H.shape == (X.shape[0], model.n_bins)
    assert np.all(H > 0) and np.all(H < 1)


def test_survival_is_monotonically_nonincreasing():
    model, X, M, Y, _ = _toy_model_and_data()
    S = model.predict_survival(X)
    diffs = np.diff(S, axis=1)
    assert np.all(diffs <= 1e-12)


def test_training_reduces_loss():
    model, X, M, Y, _ = _toy_model_and_data()
    loss_before, _ = model.loss_and_grad(X, M, Y)
    history = model.fit(X, M, Y, epochs=200, lr=0.05, batch_size=None)
    assert history[-1] < loss_before


def test_predict_rul_from_discrete_survival_orders_intervals():
    model, X, M, Y, edges = _toy_model_and_data()
    model.fit(X, M, Y, epochs=50, lr=0.05, batch_size=None)
    out = predict_rul_from_discrete_survival(model, X, edges)
    assert (out["pred_ci_low"] <= out["pred_ci_high"]).all()
    assert (out["pred_rul"] >= 0).all()


def test_predict_rul_inverted_percentiles_raise():
    model, X, M, Y, edges = _toy_model_and_data()
    with pytest.raises(AssertionError):
        predict_rul_from_discrete_survival(model, X, edges, lower_p=0.05, upper_p=0.95)


def test_window_scaler_fit_on_train_only():
    from turbofan_rul.deep_survival import apply_window_scaler, fit_window_scaler

    train_X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    other_X = np.array([[100.0, 100.0]])  # wildly different scale

    mu, sigma = fit_window_scaler(train_X)
    assert np.isclose(mu[0], 2.0)

    scaled_train = apply_window_scaler(train_X, mu, sigma)
    assert abs(scaled_train.mean()) < 1e-6  # train roughly standardized

    scaled_other = apply_window_scaler(other_X, mu, sigma)
    assert scaled_other[0, 0] > 10  # far out of range under train's scaler, as expected


def test_build_sequence_windows_3d_shape_and_no_leakage():
    from turbofan_rul.deep_survival import build_sequence_windows_3d

    e1 = pd.DataFrame({"unit_number": 1, "time_cycles": range(1, 6), "s": [1, 1, 1, 1, 1]})
    e2 = pd.DataFrame({"unit_number": 2, "time_cycles": range(1, 6), "s": [9, 9, 9, 9, 9]})
    df = pd.concat([e1, e2], ignore_index=True)

    windows = build_sequence_windows_3d(df, sensor_cols=["s"], window=3)
    assert windows.shape == (10, 3, 1)
    # engine 2's first window should be all 9s, never engine 1's trailing 1s
    assert np.all(windows[5] == 9)


def test_build_sequence_windows_3d_matches_flattened_2d_version():
    """The 3D (Keras) and 2D (from-scratch MLP) windowers must agree once the
    3D version is flattened -- they exist for different consumers, not
    different logic. If a fix ever lands in one and not the other, this
    test should catch the drift."""
    from turbofan_rul.deep_survival import build_sequence_windows, build_sequence_windows_3d

    df = pd.DataFrame({"unit_number": [1] * 4 + [2] * 3, "time_cycles": [1, 2, 3, 4, 1, 2, 3],
                        "s1": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
                        "s2": [5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0]})
    flat = build_sequence_windows(df, sensor_cols=["s1", "s2"], window=3)
    cube = build_sequence_windows_3d(df, sensor_cols=["s1", "s2"], window=3)
    assert np.allclose(cube.reshape(len(df), -1), flat)


def test_keras_loss_matches_numpy_loss_on_identical_inputs():
    """Cross-validates the Keras masked-BCE implementation against the
    from-scratch NumPy one on the same predicted hazards, mask, and target --
    both are supposed to compute exactly the same quantity."""
    tf = pytest.importorskip("tensorflow")
    from turbofan_rul.deep_survival import make_keras_masked_bce_loss, pack_mask_and_targets_for_keras

    rng = np.random.default_rng(0)
    n, K = 10, 6
    H = rng.uniform(0.05, 0.95, size=(n, K)).astype(np.float32)
    bins = rng.integers(1, K + 1, size=n)
    events = rng.integers(0, 2, size=n)
    M, Y = build_mask_and_targets(bins, events, K)

    # numpy per-sample masked-BCE, computed the same way loss_and_grad does
    eps = 1e-9
    numpy_per_sample = -np.sum(M * (Y * np.log(H + eps) + (1 - Y) * np.log(1 - H + eps)), axis=1)

    keras_loss_fn = make_keras_masked_bce_loss(K)
    y_true = pack_mask_and_targets_for_keras(M, Y)
    keras_per_sample = keras_loss_fn(tf.constant(y_true), tf.constant(H)).numpy()

    assert np.allclose(numpy_per_sample, keras_per_sample, atol=1e-4)


def test_pack_mask_and_targets_shape():
    from turbofan_rul.deep_survival import pack_mask_and_targets_for_keras

    M = np.zeros((5, 4))
    Y = np.ones((5, 4))
    packed = pack_mask_and_targets_for_keras(M, Y)
    assert packed.shape == (5, 8)
    assert np.all(packed[:, :4] == 0) and np.all(packed[:, 4:] == 1)


def test_build_mask_and_targets_censored_at_cap_survives_every_bin():
    """With `labeling.build_train_survival_frame(..., censor_above_cap=True)`,
    censoring only happens at exactly the cap -- which lands in the last bin.
    That row must be at risk in every bin and have no failure target anywhere.
    (The project's default labeling has no censored training rows at all.)"""
    edges = make_bin_edges(cap=125, n_bins=25)
    bins = assign_bins(np.array([125.0]), edges)
    M, Y = build_mask_and_targets(bins, np.array([0]), n_bins=25)
    assert bins[0] == 25
    assert M.sum() == 25
    assert Y.sum() == 0


def test_discrete_rmst_known_curves():
    """RMST is the area under the survival curve (trapezoid rule)."""
    from turbofan_rul.deep_survival import discrete_rmst

    edges = make_bin_edges(cap=100, n_bins=10)  # bins of width 10
    always_alive = np.ones((1, 10))
    dies_in_first_bin = np.zeros((1, 10))
    linear_decline = np.linspace(0.9, 0.0, 10)[None, :]  # S falls 1 -> 0 linearly over 0..100

    assert discrete_rmst(always_alive, edges)[0] == pytest.approx(100.0)
    # failure somewhere in (0, 10]: survival falls from 1 to 0 across the first bin -> area 5
    assert discrete_rmst(dies_in_first_bin, edges)[0] == pytest.approx(5.0)
    assert discrete_rmst(linear_decline, edges)[0] == pytest.approx(50.0)


def test_predict_rul_point_estimate_uses_discrete_rmst():
    from turbofan_rul.deep_survival import discrete_rmst

    model, X, M, Y, edges = _toy_model_and_data()
    out = predict_rul_from_discrete_survival(model, X, edges)
    expected = discrete_rmst(model.predict_survival(X), edges)
    assert np.allclose(out["pred_rul"].to_numpy(), expected)


def test_build_sequence_windows_independent_of_row_order():
    """Windows follow each engine's cycle order even if df's rows are
    shuffled; output row i still corresponds to df's row i."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"unit_number": [1] * 6 + [2] * 4,
                        "time_cycles": list(range(1, 7)) + list(range(1, 5)),
                        "s": rng.normal(size=10)})
    sorted_windows = build_sequence_windows(df, sensor_cols=["s"], window=3)

    shuffled = df.sample(frac=1.0, random_state=1)
    shuffled_windows = build_sequence_windows(shuffled, sensor_cols=["s"], window=3)

    positions = pd.Series(np.arange(len(df)), index=df.index)
    assert np.array_equal(shuffled_windows, sorted_windows[positions.loc[shuffled.index].to_numpy()])


def test_build_sequence_windows_rejects_duplicate_index():
    df = pd.DataFrame({"unit_number": [1, 1], "time_cycles": [1, 2], "s": [1.0, 2.0]}, index=[0, 0])
    with pytest.raises(ValueError, match="unique index"):
        build_sequence_windows(df, sensor_cols=["s"], window=2)


def test_hazards_stay_finite_for_extreme_inputs():
    """The sigmoid must not overflow (or warn) for very large activations."""
    import warnings

    model, X, M, Y, _ = _toy_model_and_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow RuntimeWarning fails the test
        H = model.predict_hazard(X * 1e4)
    assert np.all(np.isfinite(H))
    assert np.all((H >= 0) & (H <= 1))
