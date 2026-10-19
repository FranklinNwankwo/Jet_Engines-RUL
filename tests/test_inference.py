"""Tests for inference.py, run against the real exported deployment
artifacts (models/fd001/) rather than mocks -- the thing worth testing here
is whether the actual saved models and metadata compose correctly end to
end, which a mock would hide. FD004's bundle (regime clustering path) is
exercised separately below, skipped if its artifacts aren't built."""

from pathlib import Path

import pandas as pd
import pytest

from turbofan_rul import data_loading, inference

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models" / "fd001"
FD004_MODELS_DIR = REPO_ROOT / "models" / "fd004"
DATA_DIR = REPO_ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "rsf.joblib").exists(),
    reason="FD001 deployment artifacts not built -- run "
           "`python -m turbofan_rul.export_deployment_artifacts --subset FD001` first. "
           "(deployment_metadata.json is tracked in git and may exist even on a fresh "
           "clone; rsf.joblib is gitignored -- too large to version -- so its presence "
           "is the real signal that a full export has actually been run.)",
)


@pytest.fixture(scope="module")
def bundle():
    return inference.load_deployment_bundle(MODELS_DIR)


@pytest.fixture(scope="module")
def sample_readings(bundle):
    data_loading.download_cmapss("FD001", data_dir=DATA_DIR)
    cmapss = data_loading.load_subset("FD001", data_dir=DATA_DIR)
    engine = cmapss.test[cmapss.test.unit_number == 9].sort_values("time_cycles")
    return engine[["time_cycles"] + bundle.raw_passthrough_cols].reset_index(drop=True)


def test_predict_returns_expected_keys(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    assert set(result.keys()) == {
        "subset", "model_version", "predicted_rul_cycles", "confidence_interval_90", "point_estimate_outside_interval",
        "risk_level", "cycles_observed", "model_source", "weibull_interval_90", "input_check",
        "point_estimate_components",
        "rsf_calibrated_interval_supplementary",
    }


def test_served_interval_combines_rsf_and_weibull(bundle, sample_readings):
    """Phase 24: lower bound = the more cautious of RSF's raw and Weibull's
    lower bounds; upper bound = Weibull's."""
    from turbofan_rul import features, models as models_module

    result = inference.predict(bundle, sample_readings)
    low, high = result["confidence_interval_90"]
    w_low, w_high = result["weibull_interval_90"]

    working = sample_readings.copy()
    working["unit_number"] = 1
    feature_row = features.build_feature_matrix(working, kept_sensors=bundle.kept_sensors).iloc[[-1]]
    raw = models_module.predict_rul_from_survival(bundle.rsf_model, feature_row[bundle.full_feature_cols])
    rsf_low = float(raw["pred_ci_low"].iloc[0])

    tol = 0.05 + 1e-6  # outputs are rounded to 1 decimal
    assert low == pytest.approx(max(min(rsf_low, w_low), 0.0), abs=tol)
    assert high == pytest.approx(w_high, abs=tol)


def test_bundle_without_interval_metadata_serves_default(bundle):
    """Bundles exported before Phase 24 have no "interval" entry; they must
    still serve the Phase 24 interval."""
    meta = __import__("json").loads((MODELS_DIR / "deployment_metadata.json").read_text())
    expected = meta.get("interval", inference.DEFAULT_INTERVAL.to_dict())
    assert bundle.interval.to_dict() == expected
    assert inference.DEFAULT_INTERVAL.base == "hybrid" and inference.DEFAULT_INTERVAL.mode == "none"


def test_predict_rsf_calibrated_interval_is_ordered(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    low, high = result["rsf_calibrated_interval_supplementary"]["interval_90"]
    assert low <= high


def test_predict_rsf_calibrated_interval_wider_than_or_equal_to_raw(bundle, sample_readings):
    """The conformal correction only ever widens (q >= 0) -- the calibrated
    interval should never be narrower than what predict_rul_from_survival
    would give uncorrected."""
    from turbofan_rul import models as models_module

    result = inference.predict(bundle, sample_readings)
    cal_low, cal_high = result["rsf_calibrated_interval_supplementary"]["interval_90"]

    working = sample_readings.copy()
    working["unit_number"] = 1
    from turbofan_rul import features
    feature_row = features.build_feature_matrix(working, kept_sensors=bundle.kept_sensors).iloc[[-1]]
    raw_out = models_module.predict_rul_from_survival(bundle.rsf_model, feature_row[bundle.full_feature_cols])
    raw_low, raw_high = float(raw_out["pred_ci_low"].iloc[0]), float(raw_out["pred_ci_high"].iloc[0])

    # The API rounds to 1 decimal, so the calibrated bounds can sit up to 0.05
    # on the "wrong" side of the unrounded raw bounds when q is ~0 -- allow
    # for that rather than letting rounding make this test flaky.
    tol = 0.05 + 1e-6
    assert cal_low <= raw_low + tol
    assert cal_high >= raw_high - tol


def test_predict_subset_matches_bundle(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    assert result["subset"] == "FD001"


def test_predict_rul_is_positive_and_plausible(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    assert 0 < result["predicted_rul_cycles"] < 300  # generous plausibility bound


def test_predict_interval_is_ordered(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    low, high = result["confidence_interval_90"]
    assert low <= high


def test_predict_works_with_single_cycle(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings.iloc[:1])
    assert result["cycles_observed"] == int(sample_readings["time_cycles"].iloc[0])


def test_predict_risk_level_is_valid(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    assert result["risk_level"] in {"critical", "warning", "healthy"}


def test_predict_risk_level_matches_returned_numbers(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    expected = inference.risk_level_for(result["predicted_rul_cycles"], result["confidence_interval_90"][0])
    assert result["risk_level"] == expected


def test_predict_flags_point_outside_interval_when_true(bundle, sample_readings):
    result = inference.predict(bundle, sample_readings)
    low, high = result["confidence_interval_90"]
    expected = not (low <= result["predicted_rul_cycles"] <= high)
    assert result["point_estimate_outside_interval"] == expected


def test_predict_rejects_empty_readings(bundle):
    with pytest.raises(inference.InvalidReadingsError, match="at least one cycle"):
        inference.predict(bundle, pd.DataFrame())


def test_predict_rejects_missing_columns(bundle, sample_readings):
    incomplete = sample_readings.drop(columns=["sensor_2"])
    with pytest.raises(inference.InvalidReadingsError, match="missing required columns"):
        inference.predict(bundle, incomplete)


def test_predict_rejects_unsorted_cycles(bundle, sample_readings):
    unsorted = sample_readings.iloc[::-1].reset_index(drop=True)
    with pytest.raises(inference.InvalidReadingsError, match="sorted"):
        inference.predict(bundle, unsorted)


def test_predict_rejects_duplicate_cycles(bundle, sample_readings):
    dup = pd.concat([sample_readings, sample_readings.iloc[[-1]]], ignore_index=True)
    with pytest.raises(inference.InvalidReadingsError, match="unique"):
        inference.predict(bundle, dup)


def test_predict_rejects_nan_values(bundle, sample_readings):
    with_nan = sample_readings.copy()
    with_nan.loc[3, "sensor_4"] = float("nan")
    with pytest.raises(inference.InvalidReadingsError, match="NaN"):
        inference.predict(bundle, with_nan)


def test_predict_ignores_caller_index(bundle, sample_readings):
    """A caller's dataframe may have any index -- shifted, gapped or even
    duplicated labels. The prediction must be identical to the 0..n-1 case,
    and must never surface as a non-InvalidReadingsError (which the API would
    turn into a 500)."""
    expected = inference.predict(bundle, sample_readings)

    shifted = sample_readings.copy()
    shifted.index = shifted.index + 1000
    duplicated = sample_readings.copy()
    duplicated.index = [7] * len(duplicated)

    assert inference.predict(bundle, shifted) == expected
    assert inference.predict(bundle, duplicated) == expected


def test_low_true_rul_engine_flags_critical(bundle):
    """An engine near end-of-life should reliably trigger the critical risk
    band -- this is a coarse sanity check, not a precision claim (Phase 6
    already covers accuracy in detail)."""
    data_loading.download_cmapss("FD001", data_dir=DATA_DIR)
    cmapss = data_loading.load_subset("FD001", data_dir=DATA_DIR)
    true_rul = cmapss.rul.set_index("unit_number")["RUL"]
    lowest_unit = true_rul.idxmin()
    assert true_rul.loc[lowest_unit] < 15  # sanity-check the test's own premise

    engine = cmapss.test[cmapss.test.unit_number == lowest_unit].sort_values("time_cycles")
    readings = engine[["time_cycles"] + bundle.raw_passthrough_cols].reset_index(drop=True)
    result = inference.predict(bundle, readings)
    assert result["risk_level"] == "critical"


# ---------------------------------------------------------------------------
# FD004 (regime-clustering path) -- separate fixtures, skipped independently
# of the FD001 tests above if FD004's artifacts aren't built.
# ---------------------------------------------------------------------------

fd004_skip = pytest.mark.skipif(
    not (FD004_MODELS_DIR / "rsf.joblib").exists(),
    reason="FD004 deployment artifacts not built -- run "
           "`python -m turbofan_rul.export_deployment_artifacts --subset FD004` first. "
           "(same gitignored-artifact reasoning as the FD001 skip above)",
)


@pytest.fixture(scope="module")
def fd004_bundle():
    return inference.load_deployment_bundle(FD004_MODELS_DIR)


@pytest.fixture(scope="module")
def fd004_sample_readings(fd004_bundle):
    data_loading.download_cmapss("FD004", data_dir=DATA_DIR)
    cmapss = data_loading.load_subset("FD004", data_dir=DATA_DIR)
    engine = cmapss.test[cmapss.test.unit_number == 9].sort_values("time_cycles")
    return engine[["time_cycles"] + fd004_bundle.raw_passthrough_cols].reset_index(drop=True)


@fd004_skip
def test_fd004_bundle_uses_regime_clustering(fd004_bundle):
    assert fd004_bundle.uses_regime_clustering
    assert fd004_bundle.regime_kmeans is not None
    assert fd004_bundle.regime_norm_stats is not None


@fd004_skip
def test_fd004_predict_returns_correct_subset(fd004_bundle, fd004_sample_readings):
    result = inference.predict(fd004_bundle, fd004_sample_readings)
    assert result["subset"] == "FD004"


@fd004_skip
def test_fd004_predict_rul_is_positive_and_plausible(fd004_bundle, fd004_sample_readings):
    result = inference.predict(fd004_bundle, fd004_sample_readings)
    assert 0 < result["predicted_rul_cycles"] < 300


@fd004_skip
def test_fd004_predict_works_with_single_cycle(fd004_bundle, fd004_sample_readings):
    """Regime assignment (KMeans.predict) must work on a single-row request,
    not just a full trajectory -- this is the FD004-specific edge case
    FD001's bundle never exercises."""
    result = inference.predict(fd004_bundle, fd004_sample_readings.iloc[:1])
    assert result["cycles_observed"] == int(fd004_sample_readings["time_cycles"].iloc[0])


@fd004_skip
def test_fd001_and_fd004_bundles_are_independent(bundle, fd004_bundle):
    """The two subsets' artifacts must never be cross-wired -- different
    sensor sets, different feature columns, different models."""
    assert bundle.subset != fd004_bundle.subset
    assert bundle.rsf_model is not fd004_bundle.rsf_model
    assert bundle.uses_regime_clustering != fd004_bundle.uses_regime_clustering
