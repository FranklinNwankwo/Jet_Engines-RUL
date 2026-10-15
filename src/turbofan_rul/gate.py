"""
Champion/challenger retraining gate (operations layer, part 3).

Why this module exists
----------------------
Retraining can make a model worse without anyone noticing: a data-pipeline
bug, lost training data, a hyperparameter change that looked harmless. This
gate stands between "a new model was trained" and "it gets published". It
compares the *challenger* (the freshly trained bundle) with the *champion*
(the version currently published) and promotes the challenger only if it
passes rules written down in advance, here, before any comparison is run.

What is compared
----------------
Each side's `manifest.json` (manifest.py), which records test-set metrics
measured through `inference.predict`, the serving path, interval included.
The champion's manifest can be read from a local folder or downloaded from
the Hugging Face repo the live demo uses (a few kilobytes: the models
themselves are not needed to judge it).

The rules
---------
1. NASA score per engine (the project's primary metric) no more than 5%
   worse than the champion's: retraining noise is allowed, a real
   regression is not.
2. Test C-index no more than 0.01 below the champion's.
3. 90%-interval coverage at least 0.85 overall (an absolute floor).
4. Coverage within 25 cycles of failure no more than 0.05 below the
   champion's: the near-failure fix of notebook 15 must not regress.

All four must pass. A challenger identical to the champion passes (it is
"not worse"), so a routine retrain of the same recipe can be published.

Caveat: the rules use the official test set, which the project has looked at
many times. That is acceptable for a no-worse-than check, which selects
nothing, but the gate should not be used to *choose* between many
challengers; that would be tuning on the test set.

Usage
-----
    python -m turbofan_rul.gate --subset FD001 --challenger-dir models_challenger \\
        --champion-repo Chinonso11/turbofan-rul-models [--champion-revision TAG]

Exits 0 when the challenger is promoted and 1 when it is rejected, so CI marks
a rejected retrain as failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import manifest

NASA_TOLERANCE = 0.05            # rule 1: at most 5% worse
CINDEX_TOLERANCE = 0.01          # rule 2
COVERAGE_FLOOR = 0.85            # rule 3
NEAR_FAILURE_TOLERANCE = 0.05    # rule 4


@dataclass(frozen=True)
class Check:
    name: str
    metric: str
    rule: str
    champion: float
    challenger: float
    passed: bool


def evaluate(champion: dict, challenger: dict) -> list[Check]:
    """Apply the four rules to two manifests' `test_metrics`."""
    a, b = champion["test_metrics"], challenger["test_metrics"]
    nasa_limit = a["nasa_score_per_engine"] * (1 + NASA_TOLERANCE)
    cindex_limit = a["c_index"] - CINDEX_TOLERANCE
    near_limit = a["interval_coverage_near_failure"] - NEAR_FAILURE_TOLERANCE
    return [
        Check("NASA score per engine", "nasa_score_per_engine", f"<= {nasa_limit:.3f} (champion +5%)",
              a["nasa_score_per_engine"], b["nasa_score_per_engine"], b["nasa_score_per_engine"] <= nasa_limit),
        Check("Test C-index", "c_index", f">= {cindex_limit:.4f} (champion -0.01)",
              a["c_index"], b["c_index"], b["c_index"] >= cindex_limit),
        Check("Interval coverage", "interval_coverage", f">= {COVERAGE_FLOOR:.2f}",
              a["interval_coverage"], b["interval_coverage"], b["interval_coverage"] >= COVERAGE_FLOOR),
        Check("Coverage near failure", "interval_coverage_near_failure", f">= {near_limit:.3f} (champion -0.05)",
              a["interval_coverage_near_failure"], b["interval_coverage_near_failure"],
              b["interval_coverage_near_failure"] >= near_limit),
    ]


def decide(champion: dict, challenger: dict) -> dict:
    if champion.get("subset") != challenger.get("subset"):
        raise ValueError(f"cannot compare {champion.get('subset')} with {challenger.get('subset')}")
    checks = evaluate(champion, challenger)
    return {
        "subset": challenger["subset"],
        "decision": "promote" if all(c.passed for c in checks) else "reject",
        "champion_version": champion["model_version"],
        "challenger_version": challenger["model_version"],
        "identical_models": champion["model_version"] == challenger["model_version"],
        "challenger_training_config": challenger.get("training_config", {}),
        "checks": [c.__dict__ for c in checks],
    }


def to_markdown(result: dict) -> str:
    # Plain text, no emoji: Windows consoles and pipes may not encode them.
    verdict = "**PROMOTE**" if result["decision"] == "promote" else "**REJECT**"
    cfg = result.get("challenger_training_config", {})
    lines = [
        f"### Retraining gate — {result['subset']}: {verdict}",
        "",
        f"Champion `{result['champion_version']}` vs challenger `{result['challenger_version']}`"
        + (" (byte-identical models)" if result["identical_models"] else ""),
        "",
        f"Challenger trained on {cfg.get('training_engines', '?')} engines, "
        f"RSF min_samples_leaf={cfg.get('rsf_min_samples_leaf', '?')}.",
        "",
        "| Check | Champion | Challenger | Rule | Result |",
        "|---|---|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(f"| {c['name']} | {c['champion']:.4g} | {c['challenger']:.4g} | {c['rule']} | "
                     f"{'pass' if c['passed'] else '**FAIL**'} |")
    return "\n".join(lines) + "\n"


def load_champion(subset: str, champion_dir: Path | None = None, repo: str | None = None,
                  revision: str | None = None) -> dict:
    if champion_dir is not None:
        m = manifest.load_manifest(Path(champion_dir) / subset.lower())
        if m is None:
            raise SystemExit(f"no manifest.json in {Path(champion_dir) / subset.lower()}")
        return m
    if repo is None:
        raise SystemExit("give --champion-dir or --champion-repo")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=f"{subset.lower()}/{manifest.MANIFEST_NAME}",
                           revision=revision, repo_type="model")
    return json.loads(Path(path).read_text())


def load_challenger(subset: str, challenger_dir: Path) -> dict:
    subset_dir = Path(challenger_dir) / subset.lower()
    version, mismatched = manifest.verified_version(subset_dir)
    if version == manifest.UNVERSIONED:
        raise SystemExit(f"challenger in {subset_dir} has no up-to-date manifest ({', '.join(mismatched)})")
    return manifest.load_manifest(subset_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champion/challenger retraining gate.")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--challenger-dir", required=True, type=Path)
    parser.add_argument("--champion-dir", type=Path, help="local models folder holding the champion")
    parser.add_argument("--champion-repo", help="Hugging Face repo holding the champion")
    parser.add_argument("--champion-revision", default=None, help="tag or commit (default: main)")
    parser.add_argument("--decision-file", type=Path, help="write the decision as JSON here")
    parser.add_argument("--summary-file", type=Path, help="append the markdown table here ($GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)

    subset = args.subset.upper()
    champion = load_champion(subset, args.champion_dir, args.champion_repo, args.champion_revision)
    challenger = load_challenger(subset, args.challenger_dir)
    result = decide(champion, challenger)
    text = to_markdown(result)
    print(text)
    if args.decision_file:
        args.decision_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if result["decision"] == "promote" else 1


if __name__ == "__main__":
    sys.exit(main())