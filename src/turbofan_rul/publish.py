"""
Publish versioned models to the Hugging Face model repo the live demo reads.

    python -m turbofan_rul.publish --repo Chinonso11/turbofan-rul-models --dry-run
    python -m turbofan_rul.publish --repo Chinonso11/turbofan-rul-models
    python -m turbofan_rul.publish --repo Chinonso11/turbofan-rul-models --full

What it does
------------
1. Checks each local subset folder has a manifest that matches its files
   (`manifest.verified_version`); refuses to publish an unversioned model.
2. Compares the large model files with what is already on Hugging Face,
   using the SHA-256 Hugging Face stores for every LFS file. When they are
   byte-identical, only the small JSON files (`deployment_metadata.json`,
   `manifest.json`) are uploaded: a few kilobytes instead of ~350 MB. When a
   model file differs, it stops and asks for `--full`, so a large upload
   never happens by accident.
3. Uploads everything in a single commit and tags that commit with each
   subset's `model_version` (e.g. `fd001-ec87c2b68240`). Setting the live
   demo's `MODEL_REVISION` to that tag pins it to exactly these models.

Requires `huggingface_hub` and a logged-in token with write access
(`hf auth login`).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import manifest

SMALL_FILES = ("deployment_metadata.json", manifest.MANIFEST_NAME)


@dataclass
class SubsetPlan:
    subset: str
    model_version: str
    uploads: list[tuple[Path, str]] = field(default_factory=list)   # (local path, path in repo)
    changed_models: list[str] = field(default_factory=list)          # large files that differ remotely
    upload_bytes: int = 0


def remote_lfs_hashes(api, repo_id: str, folder: str) -> dict[str, str]:
    """{file name: sha256} for LFS files under `folder` in the remote repo
    (empty if the folder doesn't exist yet)."""
    try:
        entries = api.list_repo_tree(repo_id, path_in_repo=folder, repo_type="model", expand=True)
        hashes = {}
        for e in entries:
            lfs = getattr(e, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
            if sha:
                hashes[Path(e.path).name] = sha
        return hashes
    except Exception as e:  # missing folder / repo -> treat as empty
        if "NotFound" in type(e).__name__:
            return {}
        raise


def plan_subset(api, repo_id: str, subset: str, models_dir: Path, full: bool) -> SubsetPlan:
    subset_dir = Path(models_dir) / subset.lower()
    version, mismatched = manifest.verified_version(subset_dir)
    if version == manifest.UNVERSIONED:
        raise SystemExit(
            f"{subset}: no up-to-date manifest ({', '.join(mismatched)} differ). Run\n"
            f"  python -m turbofan_rul.manifest --subset {subset}\nfirst.")
    local = manifest.artifact_hashes(subset_dir)
    remote = remote_lfs_hashes(api, repo_id, subset.lower())
    large = [name for name in local if name.endswith(".joblib")]
    plan = SubsetPlan(subset, version, changed_models=[n for n in large if remote.get(n) != local[n]])
    if plan.changed_models and not full:
        return plan
    names = list(SMALL_FILES) + (large if full else [])
    for name in names:
        path = subset_dir / name
        if path.exists():
            plan.uploads.append((path, f"{subset.lower()}/{name}"))
            plan.upload_bytes += path.stat().st_size
    return plan


def publish(api, repo_id: str, subsets: list[str], models_dir: Path | str = "models",
            full: bool = False, dry_run: bool = False) -> dict:
    plans = [plan_subset(api, repo_id, s, Path(models_dir), full) for s in subsets]
    blocked = [p for p in plans if p.changed_models and not full]
    for p in plans:
        print(f"{p.subset}: {p.model_version}")
        if p.changed_models:
            print(f"  model files differ from Hugging Face: {', '.join(p.changed_models)}")
        for local, remote in p.uploads:
            print(f"  upload {remote} ({local.stat().st_size / 1e6:.2f} MB)")
    if blocked:
        sizes = sum((Path(models_dir) / p.subset.lower() / n).stat().st_size
                    for p in blocked for n in p.changed_models) / 1e6
        raise SystemExit(f"\nThe model files above differ from the ones on Hugging Face. Re-run with --full "
                         f"to upload them (about {sizes:.0f} MB).")
    total = sum(p.upload_bytes for p in plans) / 1e6
    print(f"\nTotal upload: {total:.2f} MB")
    if dry_run:
        print("Dry run: nothing uploaded.")
        return {"commit": None, "tags": [p.model_version for p in plans]}

    from huggingface_hub import CommitOperationAdd

    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                  for p in plans for local, remote in p.uploads]
    info = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                             commit_message="Publish " + ", ".join(p.model_version for p in plans))
    commit = getattr(info, "oid", None)
    for p in plans:
        api.create_tag(repo_id, tag=p.model_version, revision=commit, repo_type="model", exist_ok=True,
                       tag_message=f"{p.subset} model {p.model_version}")
    print(f"Committed {commit} and tagged it {', '.join(p.model_version for p in plans)}.")
    print(f"To pin the live demo to these models, set MODEL_REVISION = {plans[0].model_version}")
    return {"commit": commit, "tags": [p.model_version for p in plans]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish versioned model bundles to Hugging Face.")
    parser.add_argument("--repo", default=os.environ.get("MODEL_REPO"),
                        help="Hugging Face model repo, e.g. user/turbofan-rul-models (default: $MODEL_REPO)")
    parser.add_argument("--subset", action="append", help="FD001 / FD004 (default: both)")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--full", action="store_true", help="also upload the large model files")
    parser.add_argument("--dry-run", action="store_true", help="show what would be uploaded, upload nothing")
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo is required (or set MODEL_REPO)")
    from huggingface_hub import HfApi

    publish(HfApi(), args.repo, [s.upper() for s in (args.subset or ["FD001", "FD004"])],
            args.models_dir, full=args.full, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
