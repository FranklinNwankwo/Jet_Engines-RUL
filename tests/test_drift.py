"""Tests for drift.py: the PSI statistic, the monitor's window and verdicts,
fleet-like snapshots, and -- when the data is present -- the headline claim:
FD001's own test fleet reads as stable, FD002's as drift."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbofan_rul import drift

REPO_ROOT = Path(__file__).parent.parent
RNG = np.random.default_rng(0)


def _reference(values: dict[str, np.ndarray]) -> dict:
    feats = {}
    for c, v in values.items():
        edges = drift.bin_edges(v)
        feats[c] = {"edges": edges, "expected": list(drift.bin_shares(v, edges))}
    return {"features": feats}


def test_bin_shares_sum_to_one_and_cover_out_of_range_values():
    edges = drift.bin_edges(np.arange(100))
    shares = drift.bin_shares([-1e6, 50, 1e6], edges)
    assert shares.sum() == pytest.approx(1.0) and shares[0] > 0 and shares[-1] > 0


def test_psi_is_zero_for_identical_and_grows_with_shift():
    q = np.full(10, 0.1)
    assert drift.psi(q, q) == pytest.approx(0.0)
    small = drift.psi(np.r_[0.12, 0.08, np.full(8, 0.1)], q)
    large = drift.psi(np.r_[0.5, 0.0, np.full(8, 0.0625)], q)
    assert 0 < small < 0.1 < large


def test_constant_feature_gets_one_edge_not_empty_bins():
    assert len(drift.bin_edges(np.full(50, 3.0))) == 1


@pytest.mark.parametrize("value,n,expected", [
    (0.05, 500, "stable"), (0.15, 500, "warning"), (0.40, 500, "drift"), (0.40, 10, "insufficient data"),
])
def test_status_thresholds(value, n, expected):
    assert drift.status_for(value, n) == expected


def test_monitor_reports_nothing_until_minimum_then_judges_window():
    ref = _reference({"x": RNG.normal(0, 1, 5000)})
    m = drift.DriftMonitor(ref, window=300, min_requests=100)
    for v in RNG.normal(0, 1, 50):
        m.update(pd.Series({"x": v}))
    early = m.report()
    assert early["status"] == "insufficient data" and early["max_psi"] is None and early["top_features"] == []
    for v in RNG.normal(0, 1, 400):
        m.update(pd.Series({"x": v}))
    r = m.report()
    assert r["requests_in_window"] == 300 and r["requests_seen"] == 450 and r["status"] == "stable"


def test_monitor_flags_a_shifted_stream_and_names_the_feature():
    ref = _reference({"a": RNG.normal(0, 1, 5000), "b": RNG.normal(0, 1, 5000)})
    m = drift.DriftMonitor(ref, window=400, min_requests=200)
    for a, b in zip(RNG.normal(0, 1, 400), RNG.normal(1.5, 1, 400)):
        m.update(pd.DataFrame({"a": [a], "b": [b]}))
    r = m.report()
    assert r["status"] == "drift" and r["top_features"][0]["feature"] == "b"


def test_fleet_snapshots_respect_truncation_cadence_and_minimum_cycle():
    df = pd.DataFrame({"unit_number": [1] * 100 + [2] * 100, "time_cycles": list(range(1, 101)) * 2})
    snaps = drift.fleet_snapshots(df, {1: 75, 2: 40}, every=10, min_cycle=30)
    assert snaps[snaps.unit_number == 1].time_cycles.tolist() == [30, 40, 50, 60, 70]
    assert snaps[snaps.unit_number == 2].time_cycles.tolist() == [30, 40]


_HAVE = all((REPO_ROOT / p).exists() for p in ["models/fd001/rsf.joblib", "models/fd001/drift_reference.json",
                                                "data/raw/test_FD001.txt", "data/raw/test_FD002.txt"])


@pytest.mark.skipif(not _HAVE, reason="needs the FD001 bundle, its drift reference and FD001/FD002 test data")
@pytest.mark.parametrize("subset,expected", [("FD001", "stable"), ("FD002", "drift")])
def test_fd001_reference_separates_its_own_fleet_from_fd002(subset, expected):
    from turbofan_rul import data_loading, inference
    bundle = inference.load_deployment_bundle(REPO_ROOT / "models" / "fd001", verify_manifest=False)
    ref = drift.load_reference(REPO_ROOT / "models" / "fd001")
    test = data_loading.load_subset(subset, data_dir=REPO_ROOT / "data" / "raw").test
    snaps = inference.featurize(bundle, test)
    snaps = snaps[(snaps.time_cycles >= 30) & (snaps.time_cycles % 10 == 0)]
    psis = drift.compare(ref, snaps)
    assert drift.status_for(max(psis.values()), len(snaps), thresholds=ref["thresholds"]) == expected


@pytest.mark.skipif(not _HAVE, reason="needs the FD001 drift reference")
def test_reference_thresholds_are_calibrated_and_never_below_textbook():
    t = drift.load_reference(REPO_ROOT / "models" / "fd001")["thresholds"]
    assert t["drift"] >= max(t["warning"], drift.DRIFT_PSI) and t["warning"] >= drift.WARNING_PSI
    assert t["calibration"]["n_windows"] > 0


def test_steady_state_stream_mixes_engine_ages_in_every_window():
    df = pd.DataFrame([(u, c) for u in range(1, 61) for c in range(30, 300, 10)],
                      columns=["unit_number", "time_cycles"])
    stream = drift.steady_state_stream(df, seed=0)
    for start in range(0, len(stream) - 400, 400):
        window = stream.iloc[start:start + 400]
        assert window.time_cycles.min() <= 60 and window.time_cycles.max() >= 250


def test_monitor_uses_reference_thresholds():
    ref = _reference({"x": RNG.normal(0, 1, 5000)})
    ref["thresholds"] = {"warning": 5.0, "drift": 9.0}
    m = drift.DriftMonitor(ref, window=300, min_requests=100)
    for v in RNG.normal(1.5, 1, 300):
        m.update(pd.Series({"x": v}))
    assert m.report()["status"] == "stable" and m.report()["thresholds"] == {"warning": 5.0, "drift": 9.0}
