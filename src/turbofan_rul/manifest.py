"""
Model versioning (operations layer, part 1): a manifest per exported model.

Why this module exists
----------------------
Until now, nothing tied a running model to how it was made. The API, the
live demo and the Docker image each load "the FD001 model", but which one?
Built from which commit, which data, which library versions, and how good was
it on the official test set? This module answers that with a `manifest.json`
written next to each subset's artifacts (`models/fd001/manifest.json`, ...).

The version is derived from content, not assigned
-------------------------------------------------
`model_version` is a hash of the files that determine predictions (the
pickled models plus `deployment_metadata.json`), e.g. `fd001-3f9a1c0b2e4d`.
Two exports that produce byte-identical models get the same version; any
change to a model file or to the served-interval setting gives a new one. So
a version can't drift away from what it names, and anyone can recompute it.

`inference.load_deployment_bundle` re-hashes the artifacts at load time and
reports the manifest's version only if they still match it, so a model that
was rebuilt without updating its manifest shows up as "unversioned" rather
than borrowing a version that no longer describes it.

What a manifest records
-----------------------
- `model_version`, `subset`, `created_at` (UTC);
- `artifacts`: SHA-256 of every prediction-relevant file;
- `training_data`: SHA-256 of the raw C-MAPSS train/test/RUL files used;
- `code`: git commit and whether the working tree had uncommitted changes
  (from `GIT_COMMIT` / `GITHUB_SHA` if set, e.g. in Docker or CI, otherwise
  `git rev-parse`; `null` when neither is available);
- `environment`: Python and library versions (pickled models are tied to them);
- `training_config`: RSF leaf size and row stride, cap, served interval;
- `test_metrics`: the bundle scored on the official test set through
  `inference.predict` -- the exact serving path -- so the numbers describe
  the deployed behaviour, interval included.

Usage
-----
`export_deployment_artifacts.py` writes the manifest at the end of every
export. For models exported before this module existed, write one without
retraining:

    python -m turbofan_rul.manifest --subset FD001 --subset FD004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
UNVERSIONED = "unversioned"

# Files whose bytes determine a prediction. deployment_metadata.json is
# included because it carries the feature lists and the served-interval
# setting; manifest.json itself is not (it describes the others).
PREDICTION_ARTIFACTS = (
    "deployment_metadata.json",
    "rsf.joblib",
    "weibull_aft.joblib",
    "weibull_scaler.joblib",
    "regime_kmeans.joblib",
    "regime_norm_stats.joblib",
    "gbm.joblib",
)

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "lifelines", "joblib")


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(subset_dir: Path | str) -> dict[str, str]:
    """SHA-256 of each prediction-relevant file present in `subset_dir`."""
    subset_dir = Path(subset_dir)
    return {name: sha256_file(subset_dir / name)
            for name in PREDICTION_ARTIFACTS if (subset_dir / name).exists()}


def compute_model_version(subset: str, hashes: dict[str, str]) -> str:
    """Content-derived version: `<subset>-<first 12 hex of a hash over the artifact hashes>`."""
    if "rsf.joblib" not in hashes:
        raise ValueError("cannot version a bundle without rsf.joblib")
    digest = hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()
    return f"{subset.lower()}-{digest[:12]}"


def git_info(repo_dir: Path | str = ".") -> dict:
    commit = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    dirty = None
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                                    text=True, check=True, timeout=10).stdout.strip() or None
            status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
                                    capture_output=True, text=True, check=True, timeout=10).stdout
            dirty = bool(status.strip())
        except (OSError, subprocess.SubprocessError):
            commit = None
    return {"commit": commit, "uncommitted_changes": dirty}


def environment_info() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": platform.python_version(), "platform": platform.platform(terse=True), "packages": versions}


def training_data_hashes(subset: str, data_dir: Path | str) -> dict[str, str]:
    data_dir = Path(data_dir)
    names = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    return {n: sha256_file(data_dir / n) for n in names if (data_dir / n).exists()}


def evaluate_on_test(bundle, cmapss, cap: int = 125, near_failure: int = 25) -> dict:
    """Score every official test engine through `inference.predict`, the
    serving path, and summarise point accuracy and interval coverage."""
    from sksurv.metrics import concordance_index_censored

    from . import evaluation, inference

    true_rul = cmapss.rul.set_index("unit_number")["RUL"]
    y, pred, low, high = [], [], [], []
    for unit, engine in cmapss.test.groupby("unit_number"):
        engine = engine.sort_values("time_cycles")
        r = inference.predict(bundle, engine[["time_cycles"] + bundle.raw_passthrough_cols])
        y.append(float(true_rul.loc[unit]))
        pred.append(r["predicted_rul_cycles"])
        low.append(r["confidence_interval_90"][0])
        high.append(r["confidence_interval_90"][1])
    y, pred, low, high = map(np.asarray, (y, pred, low, high))
    covered = (y >= low) & (y <= high)
    return {
        "n_engines": int(len(y)),
        "c_index": round(float(concordance_index_censored(np.ones(len(y), bool), y, -pred)[0]), 4),
        "nasa_score_per_engine": round(float(evaluation.nasa_score_mean(y, pred)), 3),
        "rmse": round(float(evaluation.rmse(y, pred)), 3),
        "mae": round(float(evaluation.mae(y, pred)), 3),
        "late_rate": round(float(evaluation.late_prediction_rate(y, pred)), 4),
        "interval_coverage": round(float(covered.mean()), 4),
        "interval_coverage_near_failure": round(float(covered[y <= near_failure].mean()), 4),
        "interval_coverage_above_cap": round(float(covered[y > cap].mean()), 4),
        "interval_mean_width": round(float((high - low).mean()), 2),
        "dangerous_misses": int((y < low).sum()),
    }


def build_manifest(subset: str, subset_dir: Path | str, data_dir: Path | str,
                   repo_dir: Path | str = ".", test_metrics: dict | None = None) -> dict:
    """Assemble (but don't write) the manifest for an exported subset."""
    from . import data_loading, inference

    subset_dir = Path(subset_dir)
    hashes = artifact_hashes(subset_dir)
    meta = json.loads((subset_dir / "deployment_metadata.json").read_text())
    if test_metrics is None:
        bundle = inference.load_deployment_bundle(subset_dir, verify_manifest=False)
        cmapss = data_loading.load_subset(subset, data_dir=data_dir)
        test_metrics = evaluate_on_test(bundle, cmapss, cap=meta.get("cap", 125))
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": compute_model_version(subset, hashes),
        "subset": subset,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": hashes,
        "training_data": training_data_hashes(subset, data_dir),
        "code": git_info(repo_dir),
        "environment": environment_info(),
        "training_config": {
            "cap": meta.get("cap"),
            "rsf_min_samples_leaf": meta.get("rsf_min_samples_leaf"),
            "rsf_row_stride": meta.get("rsf_row_stride"),
            "training_engines": meta.get("training_engines"),
            "point_model": meta.get("point_model", {"type": "rsf"}).get("type"),
            "train_fraction": meta.get("train_fraction", 1.0),
            "uses_regime_clustering": meta.get("uses_regime_clustering"),
            "rsf_interval_correction": meta.get("rsf_interval_correction"),
            "interval": meta.get("interval", {"base": "hybrid", "mode": "none", "note": "inference default"}),
        },
        "test_metrics": test_metrics,
    }


def write_manifest(subset: str, models_dir: Path | str, data_dir: Path | str,
                   repo_dir: Path | str = ".") -> dict:
    subset_dir = Path(models_dir) / subset.lower()
    manifest = build_manifest(subset, subset_dir, data_dir, repo_dir)
    (subset_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(subset_dir: Path | str) -> dict | None:
    path = Path(subset_dir) / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else None


def verified_version(subset_dir: Path | str, manifest: dict | None = None) -> tuple[str, list[str]]:
    """The manifest's `model_version` if every listed artifact still has the
    recorded hash, else `UNVERSIONED`. Also returns the mismatched file names
    (missing manifest -> ["manifest.json"])."""
    manifest = manifest if manifest is not None else load_manifest(subset_dir)
    if manifest is None:
        return UNVERSIONED, [MANIFEST_NAME]
    current = artifact_hashes(subset_dir)
    recorded = manifest.get("artifacts", {})
    mismatched = sorted(name for name in set(current) | set(recorded) if current.get(name) != recorded.get(name))
    return (manifest["model_version"], []) if not mismatched else (UNVERSIONED, mismatched)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write manifest.json for exported model bundles (no retraining).")
    parser.add_argument("--subset", action="append", required=True, help="FD001 or FD004; repeat for several.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args(argv)
    for subset in args.subset:
        m = write_manifest(subset.upper(), args.models_dir, args.data_dir)
        t = m["test_metrics"]
        print(f"{m['model_version']}: C-index {t['c_index']}, RMSE {t['rmse']}, "
              f"coverage {t['interval_coverage']:.3f} (near failure {t['interval_coverage_near_failure']:.3f}), "
              f"commit {m['code']['commit'] or 'unknown'}", file=sys.stdout)


if __name__ == "__main__":
    main()
