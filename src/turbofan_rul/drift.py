"""
Input drift monitoring (operations layer, part 2).

Why this module exists
----------------------
A model is only as good as the match between the data it was trained on and
the data it is asked about. Nothing in the deployment checked that match:
an engine from a fleet running in different operating conditions, or with a
fault the model never saw, would be scored just as confidently as a familiar
one. This module compares the distribution of recent requests with a
reference distribution saved at export time, and reports drift.

What is compared
----------------
The monitored inputs are the three operating settings and the 15-cycle
rolling mean of every sensor the model uses, taken from the same feature row
`inference.predict` builds (so FD004's are regime-normalized, exactly as the
model sees them). Rolling slopes and the cycle count are left out on purpose:
they mostly encode where an engine is in its life, so a fleet of younger
engines would look "drifted" without anything being wrong.

The reference has to look like real traffic
-------------------------------------------
A reference built from every training row (full lives, failure included)
made FD001's own test engines look drifted on every sensor (PSI up to
0.57): test engines are observed partway through life, so they are
healthier on average, and sensor readings track wear. The reference is
therefore built the way the test set was: each training engine is cut at a
random point between 30% and 99% of its life (5 draws), and snapshots are
taken every 10 cycles from cycle 30 up to the cut, as a monitored fleet would
send them. See notebooks/17_drift_monitoring.ipynb.

The statistic
-------------
Population Stability Index per feature, over 10 bins at the reference
deciles:  PSI = sum_i (p_i - q_i) * ln(p_i / q_i),  with p the recent-request
share and q the reference share per bin (both floored at 1e-4). The
monitor's status is set by the largest PSI across features, over a sliding
window of `DEFAULT_WINDOW` requests ("insufficient data" below
`MIN_REQUESTS`: with 100 requests, pure chance gives a PSI around 0.1).

Thresholds are calibrated, not textbook
---------------------------------------
The usual reading (< 0.1 stable, > 0.25 shifted) assumes independent
samples and a single feature. Here neither holds: each engine sends many
correlated requests, and the status takes the worst of 15-20 features, so
on perfectly normal traffic the maximum PSI sits around 0.12. With the
textbook lines, 75-86% of normal windows were "warnings".

`calibrate_thresholds` therefore measures what normal traffic produces,
using training engines only: the reference is built from half the engines,
the other half is replayed as a steady-state fleet (`steady_state_stream`),
and this is repeated over several random splits. The warning and drift
thresholds are the 95th and 99th percentiles of the window maxima, never
below 0.1 and 0.25. They are stored in the reference file.

Per-engine input check (the "familiar inputs?" badge)
-----------------------------------------------------
PSI needs hundreds of requests, so a single engine is judged differently:
how far each monitored input lies outside the range the model saw in
training, over all training rows from cycle 30, failure included (the model
was trained on full lives). For the operating settings, single raw readings
that are pure measurement noise on FD001, the range is the full min-max seen
in training; for the sensors' 15-cycle rolling means, already smooth, it is
the 0.5th-99.5th percentile band. The distance is measured in units of that
range, and the worst feature counts. Two kinds of
evidence are kept apart, because they mean different things:

- operating settings outside the range: conditions the model never saw ->
  "unfamiliar", the prediction should not be trusted;
- sensor readings outside the range: a fault or state the model rarely saw
  -> "caution". This also happens in an engine's last few cycles, when
  readings reach extremes few training engines had.

Both thresholds are calibrated like the drift thresholds, on training
engines only, from held-out engines' scores across every cycle: the 99.9th
percentile for the strong "unfamiliar" verdict and the 99th for "caution",
with a floor of 0.05 so harmless rounding in discrete settings never counts.

What it does not do
-------------------
It detects changes in the *inputs*. It does not measure accuracy (true
remaining life is only known after failure), and a change that leaves the
inputs looking familiar would pass unnoticed. Notebook 17 checks how input
drift relates to model error on the C-MAPSS subsets.

Usage
-----
Export writes `models/<subset>/drift_reference.json`. For models exported
before this module existed, write one without retraining:

    python -m turbofan_rul.drift --subset FD001 --subset FD004
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_NAME = "drift_reference.json"
N_BINS = 10
DEFAULT_WINDOW = 500
MIN_REQUESTS = 200
WARNING_PSI = 0.10
DRIFT_PSI = 0.25
OP_SETTINGS = ("op_setting_1", "op_setting_2", "op_setting_3")
FLOOR = 1e-4
FAMILIARITY_TAILS = (0.005, 0.995)
FAMILIARITY_FLOOR = 0.05


def monitored_columns(bundle) -> list[str]:
    return list(OP_SETTINGS) + [f"{s}_roll15_mean" for s in bundle.kept_sensors]


def fleet_snapshots(featured: pd.DataFrame, truncation: dict[int, int],
                    every: int = 10, min_cycle: int = 30) -> pd.DataFrame:
    """Rows a monitored fleet would send: every `every` cycles from
    `min_cycle` up to each engine's truncation cycle."""
    limit = featured.unit_number.map(truncation)
    keep = (featured.time_cycles >= min_cycle) & (featured.time_cycles <= limit) & \
           (featured.time_cycles % every == 0)
    return featured[keep]


def bin_edges(values: np.ndarray, n_bins: int = N_BINS) -> list[float]:
    """Interior bin edges at the reference quantiles (duplicates removed, so a
    near-constant feature gets fewer bins rather than empty ones)."""
    qs = np.quantile(np.asarray(values, float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(e) for e in np.unique(qs)]


def bin_shares(values, edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges, float), np.asarray(values, float), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(observed: np.ndarray, expected: np.ndarray, floor: float = FLOOR) -> float:
    p = np.clip(np.asarray(observed, float), floor, None)
    q = np.clip(np.asarray(expected, float), floor, None)
    return float(((p - q) * np.log(p / q)).sum())


def steady_state_stream(snapshots: pd.DataFrame, period: float = 300.0, seed: int = 0) -> pd.DataFrame:
    """Order fleet snapshots as a running fleet would send them: each engine
    entered service at a random date, and the calendar wraps around, so at
    any moment the fleet holds engines of every age. (Replaying from an empty
    fleet instead makes the first windows all-young and the last all-old,
    which the monitor rightly flags.)"""
    rng = np.random.default_rng(seed)
    start = {u: rng.uniform(0, period) for u in snapshots.unit_number.unique()}
    out = snapshots.assign(_calendar=(snapshots.unit_number.map(start) + snapshots.time_cycles) % period)
    return out.sort_values("_calendar").drop(columns="_calendar").reset_index(drop=True)


def window_maxima(reference: dict, stream: pd.DataFrame, window: int = DEFAULT_WINDOW,
                  step: int = 50) -> np.ndarray:
    """Max PSI over features for every sliding window of a (circular) stream."""
    cols = list(reference["features"])
    if len(stream) < window:
        return np.array([])
    ring = pd.concat([stream, stream.iloc[:window]])[cols]
    return np.array([max(compare(reference, ring.iloc[i:i + window]).values())
                     for i in range(0, len(stream), step)])


def _snapshots(featured, train_raw, units, n_draws, every, min_cycle, random_state):
    from . import cross_validation

    raw = train_raw[train_raw.unit_number.isin(units)]
    feats = featured[featured.unit_number.isin(units)]
    parts = []
    for k in range(n_draws):
        trunc = cross_validation.sample_truncation_points(raw, min_frac=0.3, max_frac=0.99,
                                                          random_state=random_state + k)
        parts.append(fleet_snapshots(feats, trunc, every=every, min_cycle=min_cycle))
    return pd.concat(parts)


def _feature_reference(snapshots: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for c in columns:
        edges = bin_edges(snapshots[c].to_numpy())
        out[c] = {"edges": edges, "expected": [round(float(s), 6) for s in bin_shares(snapshots[c], edges)]}
    return out


def calibrate_thresholds(featured: pd.DataFrame, train_raw: pd.DataFrame, columns: list[str],
                         n_splits: int = 5, n_draws: int = 5, every: int = 10, min_cycle: int = 30,
                         window: int = DEFAULT_WINDOW, random_state: int = 100) -> dict:
    """Warning / drift thresholds from normal traffic, training engines only
    (see module docstring)."""
    units = np.sort(train_raw.unit_number.unique())
    rng = np.random.default_rng(random_state)
    maxima = []
    for split in range(n_splits):
        shuffled = rng.permutation(units)
        ref_units, stream_units = shuffled[: len(units) // 2], shuffled[len(units) // 2:]
        ref = {"features": _feature_reference(
            _snapshots(featured, train_raw, ref_units, n_draws, every, min_cycle, random_state + 10 * split),
            columns)}
        stream_rows = _snapshots(featured, train_raw, stream_units, 1, every, min_cycle,
                                 random_state + 10 * split + 7)
        stream = steady_state_stream(stream_rows, seed=split)
        if len(stream) >= window:
            maxima.append(window_maxima(ref, stream, window=window))
        elif len(stream) >= MIN_REQUESTS:
            # A small fleet (e.g. a challenger trained on a fraction of the
            # engines) can't fill a window; judge its whole stream once.
            maxima.append(np.array([max(compare(ref, stream[columns]).values())]))
    if not maxima:
        return {"warning": WARNING_PSI, "drift": DRIFT_PSI,
                "calibration": {"method": "textbook -- too few training engines to calibrate",
                                "n_windows": 0, "window": window}}
    maxima = np.concatenate(maxima)
    return {
        "warning": round(max(WARNING_PSI, float(np.quantile(maxima, 0.95))), 4),
        "drift": round(max(DRIFT_PSI, float(np.quantile(maxima, 0.99))), 4),
        "calibration": {"method": "half-split replay of training engines", "n_splits": n_splits,
                        "n_windows": int(len(maxima)), "window": window,
                        "median_max_psi": round(float(np.median(maxima)), 4),
                        "p95": round(float(np.quantile(maxima, 0.95)), 4),
                        "p99": round(float(np.quantile(maxima, 0.99)), 4)},
    }


def _range_scores(rows: pd.DataFrame, low: dict, high: dict) -> pd.DataFrame:
    """Per-feature distance outside [low, high], in units of the range width."""
    cols = list(low)
    lo, hi = pd.Series(low)[cols], pd.Series(high)[cols]
    span = (hi - lo).where(hi > lo, 1e-9)
    x = rows[cols].astype(float)
    return np.maximum(np.maximum(x - hi, lo - x), 0) / span


def _familiar_range(rows: pd.DataFrame, columns: list[str]) -> tuple[dict, dict]:
    """[low, high] per feature: full min-max for operating settings, the
    FAMILIARITY_TAILS percentile band for sensor features."""
    low, high = {}, {}
    for c in columns:
        q = (0.0, 1.0) if c in OP_SETTINGS else FAMILIARITY_TAILS
        low[c], high[c] = float(rows[c].quantile(q[0])), float(rows[c].quantile(q[1]))
    return low, high


def calibrate_familiarity(featured: pd.DataFrame, columns: list[str], n_splits: int = 5,
                          min_cycle: int = 30, random_state: int = 0) -> dict:
    """Ranges from all training rows, and thresholds from held-out training
    engines (see module docstring)."""
    rows = featured[featured.time_cycles >= min_cycle]
    rows = rows.assign(_rul=rows.groupby("unit_number").time_cycles.transform("max") - rows.time_cycles)
    ops = [c for c in columns if c in OP_SETTINGS]
    sensors = [c for c in columns if c not in OP_SETTINGS]
    units = np.sort(rows.unit_number.unique())
    rng = np.random.default_rng(random_state)
    held_scores = []
    for _ in range(n_splits):
        shuffled = rng.permutation(units)
        ref, held = rows[rows.unit_number.isin(shuffled[: len(units) // 2])], \
            rows[rows.unit_number.isin(shuffled[len(units) // 2:])]
        low, high = _familiar_range(ref, columns)
        d = _range_scores(held, low, high)
        held_scores.append(pd.DataFrame({"operating": d[ops].max(axis=1).values,
                                         "sensors": d[sensors].max(axis=1).values,
                                         "rul": held._rul.values}))
    held_scores = pd.concat(held_scores)
    # The red "unfamiliar" verdict is strong, so it targets 0.1% false alarms
    # on held-out training engines; the amber "caution" targets 1%.
    quantiles = {"operating": 0.999, "sensors": 0.99}
    thresholds = {k: round(max(FAMILIARITY_FLOOR, float(np.quantile(held_scores[k], q))), 4)
                  for k, q in quantiles.items()}
    bands = pd.cut(held_scores.rul, [-1, 10, 25, 50, np.inf], labels=["<=10", "11-25", "26-50", ">50"])
    flag_rates = {k: {str(b): round(float((g[k] > thresholds[k]).mean()), 4)
                      for b, g in held_scores.groupby(bands, observed=True)} for k in thresholds}
    low, high = _familiar_range(rows, columns)
    return {
        "low": low,
        "high": high,
        "operating_columns": ops,
        "sensor_columns": sensors,
        "thresholds": thresholds,
        "calibration": {"method": "half-split of training engines, every cycle >= %d" % min_cycle,
                        "quantiles": quantiles,
                        "n_splits": n_splits, "held_out_flag_rate_by_remaining_life": flag_rates},
    }


def familiarity(reference: dict | None, feature_row) -> dict | None:
    """Per-engine input check for one feature row (see module docstring).
    Returns None for references written before the check existed."""
    fam = (reference or {}).get("familiarity")
    if fam is None:
        return None
    row = feature_row if isinstance(feature_row, pd.DataFrame) else feature_row.to_frame().T
    d = _range_scores(row, fam["low"], fam["high"]).iloc[0]
    op_score = float(d[fam["operating_columns"]].max())
    sensor_score = float(d[fam["sensor_columns"]].max())
    t = fam["thresholds"]
    if op_score > t["operating"]:
        status = "unfamiliar"
        message = ("Operating settings outside anything the model was trained on. "
                   "The prediction should not be trusted for this engine.")
    elif sensor_score > t["sensors"]:
        status = "caution"
        message = ("Some sensor readings are outside the range seen in training. This happens in an "
                   "engine's last few cycles; if the predicted life isn't low, treat the prediction with caution.")
    else:
        status = "familiar"
        message = "All inputs are within the range the model was trained on."
    outside = d[d > 0].sort_values(ascending=False).head(5)
    return {
        "status": status,
        "message": message,
        "operating_score": round(op_score, 4),
        "sensor_score": round(sensor_score, 4),
        "thresholds": t,
        "outside_training_range": [
            {"feature": f, "value": round(float(row[f].iloc[0]), 4),
             "training_range": [round(fam["low"][f], 4), round(fam["high"][f], 4)]}
            for f in outside.index if outside[f] > 0
        ],
    }


def build_reference(bundle, train_raw: pd.DataFrame, n_draws: int = 5, every: int = 10,
                    min_cycle: int = 30, random_state: int = 100) -> dict:
    """Reference distribution of the monitored inputs on fleet-like snapshots
    of all training engines, with thresholds calibrated on held-out training
    engines (see module docstring)."""
    from . import inference

    featured = inference.featurize(bundle, train_raw)
    columns = monitored_columns(bundle)
    ref = _snapshots(featured, train_raw, train_raw.unit_number.unique(), n_draws, every, min_cycle, random_state)
    features_ref = _feature_reference(ref, columns)
    thresholds = calibrate_thresholds(featured, train_raw, columns, n_draws=n_draws, every=every,
                                      min_cycle=min_cycle, random_state=random_state)
    return {
        "subset": bundle.subset,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_reference_rows": int(len(ref)),
        "construction": {"n_draws": n_draws, "every_cycles": every, "min_cycle": min_cycle,
                         "truncation_fraction": [0.3, 0.99], "random_state": random_state},
        "thresholds": thresholds,
        "features": features_ref,
        "familiarity": calibrate_familiarity(featured, columns, min_cycle=min_cycle),
    }


def write_reference(bundle, train_raw: pd.DataFrame, subset_dir: Path | str) -> dict:
    ref = build_reference(bundle, train_raw)
    (Path(subset_dir) / REFERENCE_NAME).write_text(json.dumps(ref, indent=1) + "\n")
    return ref


def load_reference(subset_dir: Path | str) -> dict | None:
    path = Path(subset_dir) / REFERENCE_NAME
    return json.loads(path.read_text()) if path.exists() else None


def status_for(max_psi: float | None, n: int, min_requests: int = MIN_REQUESTS,
               thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"warning": WARNING_PSI, "drift": DRIFT_PSI}
    if n < min_requests or max_psi is None:
        return "insufficient data"
    if max_psi > thresholds["drift"]:
        return "drift"
    if max_psi > thresholds["warning"]:
        return "warning"
    return "stable"


def compare(reference: dict, rows: pd.DataFrame) -> dict[str, float]:
    """PSI of each monitored feature for a batch of feature rows."""
    return {c: psi(bin_shares(rows[c].to_numpy(), f["edges"]), f["expected"])
            for c, f in reference["features"].items()}


class DriftMonitor:
    """Sliding window of recent requests for one subset, compared with its
    reference on demand. Thread-safe (FastAPI runs sync endpoints in a
    thread pool). In-memory: the window starts empty after a restart."""

    def __init__(self, reference: dict, window: int = DEFAULT_WINDOW, min_requests: int = MIN_REQUESTS):
        self.reference = reference
        self.columns = list(reference["features"])
        self.thresholds = {k: reference.get("thresholds", {}).get(k, d)
                           for k, d in (("warning", WARNING_PSI), ("drift", DRIFT_PSI))}
        self.window = window
        self.min_requests = min(min_requests, window)
        self._rows: deque = deque(maxlen=window)
        self._total = 0
        self._lock = threading.Lock()

    def update(self, feature_row) -> None:
        """Record one request's feature row (a one-row DataFrame or a Series)."""
        row = feature_row.iloc[0] if isinstance(feature_row, pd.DataFrame) else feature_row
        values = tuple(float(row[c]) for c in self.columns)
        with self._lock:
            self._rows.append(values)
            self._total += 1

    def report(self) -> dict:
        with self._lock:
            rows = pd.DataFrame(list(self._rows), columns=self.columns)
            total = self._total
        n = len(rows)
        # Below the minimum, PSI is dominated by sampling noise (a single
        # request scores ~8), so no number is reported at all.
        per_feature = compare(self.reference, rows) if n >= self.min_requests else {}
        max_psi = max(per_feature.values()) if per_feature else None
        ranked = sorted(per_feature.items(), key=lambda kv: -kv[1])
        return {
            "status": status_for(max_psi, n, self.min_requests, self.thresholds),
            "requests_in_window": n,
            "requests_seen": total,
            "window": self.window,
            "min_requests": self.min_requests,
            "max_psi": None if max_psi is None else round(max_psi, 4),
            "top_features": [{"feature": f, "psi": round(v, 4)} for f, v in ranked[:5]],
            "thresholds": self.thresholds,
        }


def main(argv: list[str] | None = None) -> None:
    from . import data_loading, inference

    parser = argparse.ArgumentParser(description="Write drift_reference.json for exported bundles (no retraining).")
    parser.add_argument("--subset", action="append", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args(argv)
    for subset in (s.upper() for s in args.subset):
        subset_dir = Path(args.models_dir) / subset.lower()
        bundle = inference.load_deployment_bundle(subset_dir, verify_manifest=False)
        train = data_loading.load_subset(subset, data_dir=args.data_dir).train
        ref = write_reference(bundle, train, subset_dir)
        t = ref["thresholds"]
        ft = ref["familiarity"]["thresholds"]
        print(f"{subset}: drift reference with {len(ref['features'])} features from "
              f"{ref['n_reference_rows']} fleet-like training snapshots; calibrated thresholds "
              f"warning {t['warning']}, drift {t['drift']}; input check: operating {ft['operating']}, "
              f"sensors {ft['sensors']} -> {subset_dir / REFERENCE_NAME}")


if __name__ == "__main__":
    main()
