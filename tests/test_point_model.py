"""Tests for the optional blended point estimate (notebook 18): the average
of RSF's RMST and a gradient-boosted regressor. Uses the real FD001 bundle
with a stand-in regressor, so no extra export is needed."""

import copy
from pathlib import Path

import numpy as np
import pytest

from turbofan_rul import data_loading, export_deployment_artifacts as export, inference, manifest

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models" / "fd001"

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "rsf.joblib").exists(), reason="FD001 artifacts not built")


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture(scope="module")
def bundle():
    return inference.load_deployment_bundle(MODELS_DIR, verify_manifest=False)


@pytest.fixture(scope="module")
def readings(bundle):
    test = data_loading.load_subset("FD001", data_dir=REPO_ROOT / "data" / "raw").test
    engine = test[test.unit_number == 1].sort_values("time_cycles")
    return engine[["time_cycles"] + bundle.raw_passthrough_cols]


def _rsf_only(bundle):
    b = copy.copy(bundle)
    b.gbm_model = None
    return b


def test_rsf_bundle_reports_rsf_only(bundle, readings):
    r = inference.predict(_rsf_only(bundle), readings)
    assert r["model_source"]["point_estimate"] == "random_survival_forest"
    assert r["point_estimate_components"]["gradient_boosting"] is None
    assert r["predicted_rul_cycles"] == pytest.approx(r["point_estimate_components"]["random_survival_forest"], abs=0.06)


def test_blend_is_the_average_and_leaves_the_interval_alone(bundle, readings):
    plain = inference.predict(_rsf_only(bundle), readings)
    blended_bundle = copy.copy(bundle)
    blended_bundle.gbm_model = ConstantRegressor(40.0)
    r = inference.predict(blended_bundle, readings)
    rsf = r["point_estimate_components"]["random_survival_forest"]
    assert r["point_estimate_components"]["gradient_boosting"] == 40.0
    assert r["predicted_rul_cycles"] == pytest.approx((rsf + 40.0) / 2, abs=0.06)
    assert r["confidence_interval_90"] == plain["confidence_interval_90"]
    assert r["model_source"]["point_estimate"].startswith("blend")


def test_negative_regressor_output_is_clipped(bundle, readings):
    b = copy.copy(bundle)
    b.gbm_model = ConstantRegressor(-30.0)
    assert inference.predict(b, readings)["point_estimate_components"]["gradient_boosting"] == 0.0


def test_unknown_point_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="point_model"):
        export.build_and_save("FD001", tmp_path, tmp_path, tmp_path, point_model="lstm")


def test_gbm_file_is_part_of_the_model_version():
    assert "gbm.joblib" in manifest.PREDICTION_ARTIFACTS


def test_deployed_recipe_blends_fd001_only():
    assert export.POINT_MODEL_BY_SUBSET == {"FD001": "blend", "FD004": "rsf"}


def test_deployed_fd001_bundle_matches_its_recipe(bundle, readings):
    """Whatever the export wrote, the bundle and its metadata must agree, and
    a blend's point estimate must be the mean of its two reported members."""
    import json
    meta = json.loads((MODELS_DIR / "deployment_metadata.json").read_text())
    is_blend = meta.get("point_model", {}).get("type") == "blend"
    assert (bundle.gbm_model is not None) == is_blend
    r = inference.predict(bundle, readings)
    parts = r["point_estimate_components"]
    if is_blend:
        assert r["predicted_rul_cycles"] == pytest.approx(
            (parts["random_survival_forest"] + parts["gradient_boosting"]) / 2, abs=0.06)
