"""Tests for manifest.py: content-derived model versions and their
verification. The pure tests use small fake artifact files; the last two run
against the real exported models when they exist."""

import json
from pathlib import Path

import pytest

from turbofan_rul import manifest

REPO_ROOT = Path(__file__).parent.parent


def _fake_bundle(tmp_path, rsf=b"forest", weibull=b"weibull"):
    d = tmp_path / "fd001"
    d.mkdir()
    (d / "rsf.joblib").write_bytes(rsf)
    (d / "weibull_aft.joblib").write_bytes(weibull)
    (d / "weibull_scaler.joblib").write_bytes(b"scaler")
    (d / "deployment_metadata.json").write_text(json.dumps({"subset": "FD001", "cap": 125}))
    return d


def _write_manifest(d):
    hashes = manifest.artifact_hashes(d)
    m = {"model_version": manifest.compute_model_version("FD001", hashes), "artifacts": hashes}
    (d / manifest.MANIFEST_NAME).write_text(json.dumps(m))
    return m


def test_artifact_hashes_cover_only_prediction_files(tmp_path):
    d = _fake_bundle(tmp_path)
    (d / "notes.txt").write_text("not a model")
    (d / manifest.MANIFEST_NAME).write_text("{}")
    assert set(manifest.artifact_hashes(d)) == {"rsf.joblib", "weibull_aft.joblib", "weibull_scaler.joblib",
                                                "deployment_metadata.json"}


def test_version_is_deterministic_and_content_derived(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = manifest.artifact_hashes(_fake_bundle(tmp_path / "a"))
    b = manifest.artifact_hashes(_fake_bundle(tmp_path / "b"))
    assert manifest.compute_model_version("FD001", a) == manifest.compute_model_version("FD001", b)
    assert manifest.compute_model_version("FD001", a).startswith("fd001-")
    assert len(manifest.compute_model_version("FD001", a)) == len("fd001-") + 12


def test_version_changes_when_any_artifact_changes(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = manifest.artifact_hashes(_fake_bundle(tmp_path / "a"))
    b = manifest.artifact_hashes(_fake_bundle(tmp_path / "b", weibull=b"retrained weibull"))
    assert manifest.compute_model_version("FD001", a) != manifest.compute_model_version("FD001", b)


def test_version_does_not_depend_on_dict_order():
    h = {"rsf.joblib": "1", "weibull_aft.joblib": "2"}
    assert manifest.compute_model_version("FD004", h) == manifest.compute_model_version("FD004", dict(reversed(h.items())))


def test_version_requires_rsf():
    with pytest.raises(ValueError, match="rsf.joblib"):
        manifest.compute_model_version("FD001", {"weibull_aft.joblib": "x"})


def test_verified_version_without_manifest_is_unversioned(tmp_path):
    d = _fake_bundle(tmp_path)
    assert manifest.verified_version(d) == (manifest.UNVERSIONED, [manifest.MANIFEST_NAME])


def test_verified_version_matches_then_detects_changed_artifact(tmp_path):
    d = _fake_bundle(tmp_path)
    m = _write_manifest(d)
    assert manifest.verified_version(d) == (m["model_version"], [])
    (d / "rsf.joblib").write_bytes(b"re-exported without updating the manifest")
    version, mismatched = manifest.verified_version(d)
    assert version == manifest.UNVERSIONED and mismatched == ["rsf.joblib"]


def test_verified_version_detects_added_artifact(tmp_path):
    d = _fake_bundle(tmp_path)
    _write_manifest(d)
    (d / "regime_kmeans.joblib").write_bytes(b"new file")
    assert manifest.verified_version(d) == (manifest.UNVERSIONED, ["regime_kmeans.joblib"])


def test_git_info_prefers_environment(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    assert manifest.git_info()["commit"] == "abc123"


def test_environment_info_lists_pickle_relevant_packages():
    env = manifest.environment_info()
    assert env["python"] and set(manifest.TRACKED_PACKAGES) == set(env["packages"])


@pytest.mark.skipif(not (REPO_ROOT / "models" / "fd001" / "rsf.joblib").exists(),
                    reason="FD001 deployment artifacts not built")
def test_real_bundle_version_matches_its_manifest_or_is_unversioned():
    from turbofan_rul import inference
    d = REPO_ROOT / "models" / "fd001"
    bundle = inference.load_deployment_bundle(d)
    m = manifest.load_manifest(d)
    if m is None:
        assert bundle.model_version == manifest.UNVERSIONED
    else:
        assert bundle.model_version in (m["model_version"], manifest.UNVERSIONED)
        assert bundle.model_version == manifest.compute_model_version("FD001", manifest.artifact_hashes(d)) \
            or bundle.model_version == manifest.UNVERSIONED


@pytest.mark.skipif(not (REPO_ROOT / "models" / "fd001" / manifest.MANIFEST_NAME).exists(),
                    reason="FD001 manifest not written -- run `python -m turbofan_rul.manifest --subset FD001`")
def test_real_manifest_records_plausible_test_metrics():
    t = manifest.load_manifest(REPO_ROOT / "models" / "fd001")["test_metrics"]
    assert t["n_engines"] == 100
    assert 0.80 < t["c_index"] < 0.95
    assert 0.85 <= t["interval_coverage"] <= 1.0
