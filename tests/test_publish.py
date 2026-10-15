"""Tests for publish.py's decisions -- what gets uploaded, and when it refuses
-- against a fake Hugging Face client, so no network or token is needed."""

import json
from types import SimpleNamespace

import pytest

from turbofan_rul import manifest, publish


def _bundle(models_dir, subset="FD001", rsf=b"forest"):
    d = models_dir / subset.lower()
    d.mkdir(parents=True)
    (d / "rsf.joblib").write_bytes(rsf)
    (d / "weibull_aft.joblib").write_bytes(b"weibull")
    (d / "weibull_scaler.joblib").write_bytes(b"scaler")
    (d / "deployment_metadata.json").write_text(json.dumps({"subset": subset}))
    hashes = manifest.artifact_hashes(d)
    (d / "manifest.json").write_text(json.dumps(
        {"model_version": manifest.compute_model_version(subset, hashes), "artifacts": hashes}))
    return d


class FakeApi:
    def __init__(self, remote: dict[str, dict[str, str]]):
        self.remote, self.commits, self.tags = remote, [], []

    def list_repo_tree(self, repo_id, path_in_repo=None, **kw):
        if path_in_repo not in self.remote:
            raise type("RemoteEntryNotFoundError", (Exception,), {})()
        return [SimpleNamespace(path=f"{path_in_repo}/{n}", lfs=SimpleNamespace(sha256=h))
                for n, h in self.remote[path_in_repo].items()]

    def create_commit(self, repo_id, operations, **kw):
        self.commits.append([op.path_in_repo for op in operations])
        return SimpleNamespace(oid="c0ffee")

    def create_tag(self, repo_id, *, tag, revision=None, **kw):
        self.tags.append((tag, revision))


def _remote_same_as(d):
    return {n: h for n, h in manifest.artifact_hashes(d).items() if n.endswith(".joblib")}


def test_identical_models_upload_only_the_small_files(tmp_path):
    d = _bundle(tmp_path)
    api = FakeApi({"fd001": _remote_same_as(d)})
    out = publish.publish(api, "u/repo", ["FD001"], tmp_path)
    assert api.commits == [["fd001/deployment_metadata.json", "fd001/manifest.json"]]
    version = manifest.load_manifest(d)["model_version"]
    assert api.tags == [(version, "c0ffee")] and out["tags"] == [version]


def test_changed_model_file_is_refused_without_full(tmp_path):
    d = _bundle(tmp_path)
    remote = _remote_same_as(d); remote["rsf.joblib"] = "0" * 64
    api = FakeApi({"fd001": remote})
    with pytest.raises(SystemExit, match="--full"):
        publish.publish(api, "u/repo", ["FD001"], tmp_path)
    assert api.commits == [] and api.tags == []


def test_full_uploads_model_files_too(tmp_path):
    _bundle(tmp_path)
    api = FakeApi({})                      # nothing on Hugging Face yet
    publish.publish(api, "u/repo", ["FD001"], tmp_path, full=True)
    assert sorted(api.commits[0]) == sorted(["fd001/deployment_metadata.json", "fd001/manifest.json",
                                             "fd001/rsf.joblib", "fd001/weibull_aft.joblib",
                                             "fd001/weibull_scaler.joblib"])


def test_unversioned_model_is_refused(tmp_path):
    d = _bundle(tmp_path)
    (d / "rsf.joblib").write_bytes(b"re-exported, manifest not updated")
    with pytest.raises(SystemExit, match="python -m turbofan_rul.manifest"):
        publish.publish(FakeApi({}), "u/repo", ["FD001"], tmp_path, full=True)


def test_two_subsets_go_in_one_commit_with_one_tag_each(tmp_path):
    d1, d4 = _bundle(tmp_path, "FD001"), _bundle(tmp_path, "FD004", rsf=b"other forest")
    api = FakeApi({"fd001": _remote_same_as(d1), "fd004": _remote_same_as(d4)})
    publish.publish(api, "u/repo", ["FD001", "FD004"], tmp_path)
    assert len(api.commits) == 1 and len(api.commits[0]) == 4
    assert [t for t, _ in api.tags] == [manifest.load_manifest(d)["model_version"] for d in (d1, d4)]


def test_dry_run_uploads_nothing(tmp_path):
    d = _bundle(tmp_path)
    api = FakeApi({"fd001": _remote_same_as(d)})
    publish.publish(api, "u/repo", ["FD001"], tmp_path, dry_run=True)
    assert api.commits == [] and api.tags == []
