"""
Discrete-time hazard survival model: a from-scratch NumPy implementation,
plus the windowing, binning and loss helpers shared with the Keras models.

Role of this module in Phase 7
--------------------------------
The Phase 7 models actually trained and reported (a Keras LSTM and a
Conv1D->LSTM, in notebooks/07_deep_survival_stretch.ipynb) use the Keras
loss built by `make_keras_masked_bce_loss` and the window/bin/mask helpers
below. The NumPy `DiscreteHazardMLP` is the validation layer: forward pass,
hand-derived backpropagation and a from-scratch Adam optimizer, with the
analytic gradient checked against numerical finite differences
(`gradient_check`, `tests/test_deep_survival.py`) and the loss checked to
match the Keras loss on identical inputs -- the same "derive it, then prove
the implementation matches" standard applied to the Weibull AFT in Phase 4.

Why not PyTorch / pycox: this project's sandboxed build environment could
not reach a CPU-only torch wheel (PyPI's default `torch` package bundles
several GB of CUDA dependencies that exceeded available disk, and the
official CPU wheel index was outside the network allowlist), so
`tensorflow-cpu` was used for the trained models.

Why discrete-time hazards, and why raw sequence windows
---------------------------------------------------------
Phases 3-5 all operate on hand-engineered features (rolling stats, EWMA)
computed from sensor readings. The stated point of trying a "deep" model at
all (see docs/project_brief.md Section 4) is to test whether a model can
learn useful temporal structure directly from a window of raw sensor
readings, without that hand-engineering step -- so the input here is a
window of the last W raw sensor readings per engine
(`build_sequence_windows` / `build_sequence_windows_3d`), not the Phase 2
feature matrix.

Time is binned into K intervals and the network predicts a per-bin hazard
probability. The discrete-time hazard likelihood is mathematically
equivalent to a *masked binary cross-entropy* over bins -- for observation i
with (capped) duration bin b_i and event indicator delta_i, every bin
k <= b_i is "at risk," with target 1 only at k = b_i and only if delta_i=1,
and 0 everywhere else at risk. A censored observation (delta_i=0) is
therefore treated as having survived its own bin b_i. The project's default
labeling marks every training row event=1; with
`labeling.build_train_survival_frame(..., censor_above_cap=True)`, censoring
only happens at exactly the cap, i.e. at the end of the last bin, so the
convention is exact in that case too. This equivalence (derived in
notebooks/07_deep_survival_stretch.ipynb) is what makes a from-scratch
implementation tractable: no risk-set bookkeeping across the whole batch
(unlike Cox's partial likelihood), just a masked elementwise loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def fit_window_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-column mean/std on the training window matrix only.

    Kept as a standalone (mu, sigma) pair rather than a stateful object so
    it's trivial to serialize alongside the model for deployment -- same
    "fit on train, apply everywhere" discipline as
    `models.standardize_features`.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return mu, sigma


def apply_window_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _build_windows(df: pd.DataFrame, sensor_cols: list[str], window: int,
                   unit_col: str, cycle_col: str, dtype) -> np.ndarray:
    """Shared windowing logic for the 2D and 3D builders.

    Returns shape (n_rows, window, n_sensors), with output row i
    corresponding to row i of `df` (in `df`'s current order). Within each
    engine, windows are built in cycle order regardless of how `df` is
    sorted, and early-life rows are left-padded by repeating the engine's
    first observed cycle.
    """
    if not df.index.is_unique:
        raise ValueError(
            "df must have a unique index (each row's window is written back by index label); "
            "call df.reset_index(drop=True) first."
        )
    n_sensors = len(sensor_cols)
    out = np.zeros((len(df), window, n_sensors), dtype=dtype)
    positions = pd.Series(np.arange(len(df)), index=df.index)

    for _, group in df.groupby(unit_col, sort=False):
        group = group.sort_values(cycle_col)
        out_pos = positions.loc[group.index].to_numpy()
        values = group[sensor_cols].to_numpy()  # (n_cycles, n_sensors)
        for row_pos in range(len(values)):
            start = row_pos - window + 1
            if start < 0:
                pad = np.repeat(values[0:1], -start, axis=0)
                seq = np.vstack([pad, values[: row_pos + 1]])
            else:
                seq = values[start: row_pos + 1]
            out[out_pos[row_pos]] = seq
    return out


def build_sequence_windows(df: pd.DataFrame, sensor_cols: list[str], window: int = 15,
                            unit_col: str = "unit_number", cycle_col: str = "time_cycles") -> np.ndarray:
    """For each row, build a flattened window of the last `window` cycles' raw sensor
    readings for that engine, ending at that row's own cycle.

    Early-life rows (fewer than `window` prior cycles available) are
    left-padded by repeating the engine's first observed cycle -- the same
    "don't discard early-life rows" reasoning as `features.add_rolling_features`
    (min_periods=1): those rows are exactly the ones that give the model
    contrast between healthy and degrading behaviour, and at inference time a
    newly-deployed engine won't have a full window of history yet either.

    Returns an array of shape (n_rows, window * len(sensor_cols)), with rows
    in the same order as `df`. `df` must have a unique index.
    """
    windows = _build_windows(df, sensor_cols, window, unit_col, cycle_col, dtype=float)
    return windows.reshape(len(df), window * len(sensor_cols))


def build_sequence_windows_3d(df: pd.DataFrame, sensor_cols: list[str], window: int = 15,
                               unit_col: str = "unit_number", cycle_col: str = "time_cycles") -> np.ndarray:
    """Same windowing as `build_sequence_windows`, but returns a 3D float32 array
    of shape (n_rows, window, n_sensors) instead of a flattened 2D one --
    the shape a Keras `LSTM`/`Conv1D` layer expects (batch, timesteps,
    features), rather than the flattened input the from-scratch MLP uses.
    Padding and per-engine grouping follow identical rules to
    `build_sequence_windows` -- see that function's docstring.
    """
    return _build_windows(df, sensor_cols, window, unit_col, cycle_col, dtype=np.float32)


def make_keras_masked_bce_loss(n_bins: int):
    """Build the Keras-compatible version of the masked binary cross-entropy
    loss derived in the module docstring and implemented in NumPy below
    (`DiscreteHazardMLP.loss_and_grad`).

    Keras loss functions take only `(y_true, y_pred)`, with no separate
    channel for a mask, so the mask M and target Y are packed together into
    one `y_true` tensor of shape (n, 2*n_bins) -- the first half is M, the
    second half is Y -- and unpacked inside the loss. Packing is a Keras API
    accommodation, not a different formula.

    Differences from `DiscreteHazardMLP.loss_and_grad`, both deliberate:
    - no L2 penalty term (Keras models add regularization through layers or
      callbacks, not inside the loss);
    - predictions are clipped to [1e-7, 1 - 1e-7] before the log, where the
      NumPy version adds 1e-9 inside the log. These only differ for hazards
      extremely close to 0 or 1.
    `tests/test_deep_survival.py` checks the two agree numerically on
    identical, non-extreme inputs.
    """
    import tensorflow as tf

    def masked_bce_loss(y_true, y_pred):
        mask = y_true[:, :n_bins]
        target = y_true[:, n_bins:]
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        per_element = mask * (target * tf.math.log(p) + (1 - target) * tf.math.log(1 - p))
        return -tf.reduce_sum(per_element, axis=1)

    return masked_bce_loss


def pack_mask_and_targets_for_keras(M: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Concatenate (M, Y) into the single tensor `make_keras_masked_bce_loss` expects."""
    return np.concatenate([M, Y], axis=1).astype(np.float32)


def make_bin_edges(cap: float, n_bins: int) -> np.ndarray:
    """Evenly spaced bin edges from 0 to `cap`. Bin k covers (edges[k-1], edges[k]]."""
    return np.linspace(0.0, cap, n_bins + 1)


def assign_bins(durations: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous durations to a 1-indexed bin number in [1, n_bins].

    Bin k covers (edges[k-1], edges[k]]. A duration of exactly 0 goes to bin 1,
    and anything above the last edge is clipped into the last bin.
    """
    n_bins = len(bin_edges) - 1
    # np.digitize(x, inner_edges, right=True) returns i such that
    # inner_edges[i-1] < x <= inner_edges[i]; +1 makes the bin number 1-indexed.
    bins = np.digitize(durations, bin_edges[1:-1], right=True) + 1
    return np.clip(bins, 1, n_bins)


def build_mask_and_targets(bins: np.ndarray, events: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the (n, n_bins) at-risk mask M and target Y described in the module docstring.

    `events` are the event indicators (1 = failure observed, 0 = censored),
    e.g. the `event` column from `labeling.build_train_survival_frame`. Use
    that column rather than hard-coding all ones, so the targets stay correct
    if the labeling is ever switched to `censor_above_cap=True`.
    """
    k = np.arange(1, n_bins + 1)
    events = np.asarray(events)
    M = (k[None, :] <= bins[:, None]).astype(float)
    Y = ((k[None, :] == bins[:, None]) & (events[:, None] == 1)).astype(float)
    return M, Y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow warnings for large |z|)."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class DiscreteHazardMLP:
    """1-hidden-layer MLP mapping an input vector to per-bin hazard probabilities.

    Architecture is deliberately small (one hidden layer, ReLU) -- this is
    a from-scratch implementation used to validate the loss and gradients,
    not a production deep learning system, and a small network is faster to
    gradient-check and less likely to silently overfit on a dataset this size.
    """
    n_features: int
    n_bins: int
    hidden_size: int = 64
    l2: float = 1e-3
    random_state: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.random_state)
        # He initialization, appropriate for ReLU hidden units
        self.W1 = rng.normal(size=(self.n_features, self.hidden_size)) * np.sqrt(2.0 / self.n_features)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(size=(self.hidden_size, self.n_bins)) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.n_bins)
        self._adam_state = None

    def _forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ self.W2 + self.b2
        H = _sigmoid(Z2)
        return Z1, A1, Z2, H

    def predict_hazard(self, X: np.ndarray) -> np.ndarray:
        """Per-bin hazard probabilities, shape (n, n_bins)."""
        _, _, _, H = self._forward(X)
        return H

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """Discrete survival S(k) = prod_{j<=k}(1 - h_j), shape (n, n_bins)."""
        H = self.predict_hazard(X)
        return np.cumprod(1.0 - H, axis=1)

    def loss_and_grad(self, X: np.ndarray, M: np.ndarray, Y: np.ndarray):
        """Masked binary cross-entropy loss (mean over samples) plus L2 penalty,
        and its gradients.

        This is the function validated by `gradient_check` -- see module
        docstring for the mask/target construction and why this equals the
        discrete-time hazard negative log-likelihood.
        """
        n = X.shape[0]
        Z1, A1, Z2, H = self._forward(X)
        eps = 1e-9
        per_element = M * (Y * np.log(H + eps) + (1 - Y) * np.log(1 - H + eps))
        loss = -np.sum(per_element) / n + self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

        dZ2 = M * (H - Y) / n
        dW2 = A1.T @ dZ2 + 2 * self.l2 * self.W2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1 + 2 * self.l2 * self.W1
        db1 = dZ1.sum(axis=0)
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _init_adam(self):
        self._adam_state = {
            name: {"m": np.zeros_like(getattr(self, name)), "v": np.zeros_like(getattr(self, name)), "t": 0}
            for name in ("W1", "b1", "W2", "b2")
        }

    def _adam_step(self, grads: dict, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        for name, grad in grads.items():
            state = self._adam_state[name]
            state["t"] += 1
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
            m_hat = state["m"] / (1 - beta1 ** state["t"])
            v_hat = state["v"] / (1 - beta2 ** state["t"])
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)

    def fit(self, X: np.ndarray, M: np.ndarray, Y: np.ndarray, epochs: int = 300,
            lr: float = 1e-3, batch_size: int | None = 512, verbose_every: int = 0) -> list[float]:
        """Full-batch or mini-batch Adam training. Returns the per-epoch loss history
        (useful for confirming the optimizer is actually converging, not just running)."""
        self._init_adam()
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        history = []

        for epoch in range(epochs):
            if batch_size is None or batch_size >= n:
                loss, grads = self.loss_and_grad(X, M, Y)
                self._adam_step(grads, lr)
                history.append(loss)
            else:
                perm = rng.permutation(n)
                epoch_losses = []
                for start in range(0, n, batch_size):
                    batch_idx = perm[start:start + batch_size]
                    loss, grads = self.loss_and_grad(X[batch_idx], M[batch_idx], Y[batch_idx])
                    self._adam_step(grads, lr)
                    epoch_losses.append(loss)
                history.append(float(np.mean(epoch_losses)))
            if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {history[-1]:.4f}")
        return history


def gradient_check(model: DiscreteHazardMLP, X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                    eps: float = 1e-5, n_checks_per_param: int = 8, random_state: int = 0) -> dict[str, float]:
    """Compare the analytic gradients from `loss_and_grad` against numerical
    finite-difference gradients on a handful of random entries per parameter.

    Returns the maximum relative error found per parameter -- this is the
    correctness check that stands in for "validate against a library", since
    no survival deep-learning library was available in this environment (see
    module docstring).
    """
    rng = np.random.default_rng(random_state)
    _, analytic = model.loss_and_grad(X, M, Y)

    def loss_only():
        return model.loss_and_grad(X, M, Y)[0]

    max_rel_errors = {}
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(model, name)
        flat = param.reshape(-1)
        n_checks = min(n_checks_per_param, flat.size)
        indices = rng.choice(flat.size, size=n_checks, replace=False)
        rel_errors = []
        for idx in indices:
            multi_idx = np.unravel_index(idx, param.shape)
            orig = param[multi_idx]
            param[multi_idx] = orig + eps
            loss_plus = loss_only()
            param[multi_idx] = orig - eps
            loss_minus = loss_only()
            param[multi_idx] = orig
            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            analytic_grad = analytic[name][multi_idx]
            rel_error = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad) + 1e-8)
            rel_errors.append(rel_error)
        max_rel_errors[name] = float(max(rel_errors))
    return max_rel_errors


def discrete_rmst(S: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Restricted mean survival time (up to the last bin edge) from discrete
    survival curves S of shape (n, n_bins), where S[:, k-1] is survival at the
    END of bin k.

    Uses the trapezoid rule -- survival assumed to fall linearly within each
    bin, matching the labeling convention that a failure in bin k happened
    somewhere in (edges[k-1], edges[k]]. Summing only the end-of-bin values
    (`S.sum(axis=1) * width`) would treat every failure as happening at the
    START of its bin and bias every estimate low by up to one bin width.
    """
    widths = np.diff(bin_edges)
    S_start = np.hstack([np.ones((S.shape[0], 1)), S[:, :-1]])  # survival at the start of each bin
    return np.sum(0.5 * (S_start + S) * widths, axis=1)


def predict_rul_from_discrete_survival(model: DiscreteHazardMLP, X: np.ndarray, bin_edges: np.ndarray,
                                        lower_p: float = 0.95, upper_p: float = 0.05) -> pd.DataFrame:
    """Turn predicted discrete survival curves into RUL point estimates + intervals.

    Point estimate is the restricted mean survival time (`discrete_rmst`), for
    the same reason as `models.predict_rul_from_survival`: it degrades
    gracefully instead of saturating at the time ceiling for the healthiest
    engines.

    Interval bounds are the end of the first bin where survival drops to or
    below `lower_p` / `upper_p` (or the last bin edge if it never does), so
    they're resolved only to the bin width.
    """
    S = model.predict_survival(X)  # (n, n_bins)
    bin_upper = bin_edges[1:]  # time at the END of each bin

    rmst = discrete_rmst(S, bin_edges)

    def _quantile(s_row: np.ndarray, p: float) -> float:
        # first bin where survival drops to or below p
        below = np.where(s_row <= p)[0]
        idx = below[0] if len(below) > 0 else len(s_row) - 1
        return float(bin_upper[idx])

    lower = np.array([_quantile(s, lower_p) for s in S])
    upper = np.array([_quantile(s, upper_p) for s in S])
    assert (lower <= upper).all(), "interval bounds inverted -- check lower_p/upper_p convention"

    return pd.DataFrame({"pred_rul": rmst, "pred_ci_low": lower, "pred_ci_high": upper})
