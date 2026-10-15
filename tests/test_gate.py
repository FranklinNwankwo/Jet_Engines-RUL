"""Tests for gate.py: each rule, the overall decision, and the CLI's exit
code, using synthetic manifests (no models or network needed)."""

import json

import pytest

from turbofan_rul import gate, manifest

BASE = {"nasa_score_per_engine": 4.0, "c_index": 0.864, "interval_coverage": 0.97,
        "interval_coverage_near_failure": 1.0}


def _m(version="fd001-aaaa", subset="FD001", **overrides):
    return {"model_version": version, "subset": subset, "test_metrics": {**BASE, **overrides},
            "training_config": {"training_engines": 100, "rsf_min_samples_leaf": 5}}


def test_identical_challenger_is_promoted():
    r = gate.decide(_m(), _m())
    assert r["decision"] == "promote" and r["identical_models"] and all(c["passed"] for c in r["checks"])


@pytest.mark.parametrize("overrides,failed", [
    ({"nasa_score_per_engine": 4.25}, "NASA score per engine"),       # 6% worse
    ({"c_index": 0.85}, "Test C-index"),                               # 0.014 lower
    ({"interval_coverage": 0.84}, "Interval coverage"),                # below the floor
    ({"interval_coverage_near_failure": 0.94}, "Coverage near failure"),  # 6 points lower
])
def test_each_rule_can_reject_on_its_own(overrides, failed):
    r = gate.decide(_m(), _m("fd001-bbbb", **overrides))
    assert r["decision"] == "reject"
    assert [c["name"] for c in r["checks"] if not c["passed"]] == [failed]


def test_small_noise_within_tolerance_is_promoted():
    r = gate.decide(_m(), _m("fd001-bbbb", nasa_score_per_engine=4.19, c_index=0.856,
                             interval_coverage=0.86, interval_coverage_near_failure=0.96))
    assert r["decision"] == "promote" and not r["identical_models"]


def test_better_challenger_is_promoted():
    r = gate.decide(_m(), _m("fd001-cccc", nasa_score_per_engine=3.5, c_index=0.88))
    assert r["decision"] == "promote"


def test_different_subsets_are_refused():
    with pytest.raises(ValueError, match="cannot compare"):
        gate.decide(_m(), _m(subset="FD004"))


def test_markdown_shows_verdict_and_every_check():
    text = gate.to_markdown(gate.decide(_m(), _m("fd001-bbbb", c_index=0.80)))
    assert "REJECT" in text and text.count("| ") >= 4 and "**FAIL**" in text


def test_cli_exit_code_and_files(tmp_path):
    for name, metrics in [("champ", {}), ("chal", {"nasa_score_per_engine": 9.0})]:
        d = tmp_path / name / "fd001"; d.mkdir(parents=True)
        (d / "rsf.joblib").write_bytes(name.encode())
        hashes = manifest.artifact_hashes(d)
        m = _m(manifest.compute_model_version("FD001", hashes), **metrics); m["artifacts"] = hashes
        (d / "manifest.json").write_text(json.dumps(m))
    summary, decision = tmp_path / "summary.md", tmp_path / "decision.json"
    code = gate.main(["--subset", "FD001", "--champion-dir", str(tmp_path / "champ"),
                      "--challenger-dir", str(tmp_path / "chal"),
                      "--summary-file", str(summary), "--decision-file", str(decision)])
    assert code == 1 and "REJECT" in summary.read_text(encoding="utf-8")
    assert json.loads(decision.read_text(encoding="utf-8"))["decision"] == "reject"


def test_unversioned_challenger_is_refused(tmp_path):
    d = tmp_path / "chal" / "fd001"; d.mkdir(parents=True)
    (d / "rsf.joblib").write_bytes(b"no manifest")
    with pytest.raises(SystemExit, match="no up-to-date manifest"):
        gate.load_challenger("FD001", tmp_path / "chal")