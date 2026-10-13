"""
Data acquisition and loading for NASA's C-MAPSS turbofan degradation dataset.

Why download from a URL instead of a manual upload
----------------------------------------------------
Pointing this module at a stable, scriptable source (rather than "download the zip
from NASA's data repository by hand and drop it in a folder") makes the whole
pipeline reproducible: a fresh clone of this repo can run `python -m
turbofan_rul.data_loading` and have data/raw/ populated without any manual step,
which matters both for your own reproducibility and for anyone reviewing the repo.

NASA's own prognostics data repository has moved hosts a few times and doesn't
always resolve reliably from arbitrary environments (including sandboxed ones),
so this module defaults to a GitHub-hosted mirror of the original files
(DemetraAS/C-MAPSS). The loaded tables are checked against reference counts and
basic structural invariants after loading (see `load_subset`), specifically to
catch a corrupted or wrong mirror early rather than silently training on bad data.

If you'd rather pull from NASA's official repository or a Kaggle mirror, swap the
`BASE_URL` below and re-run -- nothing else in the pipeline depends on the source.

Subset scope
------------
FD001 is the primary subset for this project (single operating condition, single
fault mode) -- see docs/project_brief.md for the rationale. FD002-FD004 use the
same loader and schema. FD002 and FD004 have six operating regimes, so they need
the regime clustering + within-regime normalization step in `features.py`, which
FD001 and FD003 (single regime) never exercise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/DemetraAS/C-MAPSS/main"

# unit_number, time_cycles, 3 operating settings, 21 sensor measurements = 26 columns.
# The raw .txt files end every line with trailing whitespace, which pandas reads as
# a 27th all-NaN column -- we name it explicitly and check it really is all-NaN
# before dropping it, rather than silently slicing off "the last column" and hoping
# it's the right one.
COLUMN_NAMES = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
    + ["_trailing_nan"]
)

# Reference statistics used as a sanity check after loading. Unit counts,
# operating conditions and fault modes are from the dataset's documentation
# (Saxena & Goebel, 2008); row counts are the line counts of the original NASA
# .txt files. See docs/project_brief.md Section 3.
SUBSET_INFO = {
    "FD001": dict(train_units=100, test_units=100, train_rows=20631, test_rows=13096,
                  operating_conditions=1, fault_modes=1,
                  description="Single operating condition (sea level); single fault mode (HPC degradation)."),
    "FD002": dict(train_units=260, test_units=259, train_rows=53759, test_rows=33991,
                  operating_conditions=6, fault_modes=1,
                  description="Six operating conditions; single fault mode (HPC degradation)."),
    "FD003": dict(train_units=100, test_units=100, train_rows=24720, test_rows=16596,
                  operating_conditions=1, fault_modes=2,
                  description="Single operating condition; two fault modes (HPC + fan degradation)."),
    "FD004": dict(train_units=249, test_units=248, train_rows=61249, test_rows=41214,
                  operating_conditions=6, fault_modes=2,
                  description="Six operating conditions; two fault modes (HPC + fan degradation). Most complex subset."),
}

VALID_SUBSETS = tuple(SUBSET_INFO.keys())


@dataclass
class CMAPSSData:
    """Container for one subset's train/test/RUL tables, plus which subset it is."""
    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.DataFrame


def _file_url(kind: str, subset: str) -> str:
    return f"{BASE_URL}/{kind}_{subset}.txt"


def download_cmapss(subset: str, data_dir: Path | str = "data/raw", force: bool = False) -> None:
    """Download the train/test/RUL files for one C-MAPSS subset if not already present.

    Parameters
    ----------
    subset : one of "FD001", "FD002", "FD003", "FD004"
    data_dir : local directory the raw .txt files are cached in
    force : re-download even if the files already exist locally
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {VALID_SUBSETS}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for kind in ("train", "test", "RUL"):
        dest = data_dir / f"{kind}_{subset}.txt"
        if dest.exists() and not force:
            logger.info("%s already present, skipping download", dest)
            continue

        url = _file_url(kind, subset)
        logger.info("Downloading %s -> %s", url, dest)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)


def _read_sensor_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    if df["_trailing_nan"].notna().any():
        raise ValueError(
            f"{path.name}: expected 26 columns per line plus trailing whitespace, but found "
            f"values in a 27th column -- the file layout doesn't match C-MAPSS."
        )
    return df.drop(columns=["_trailing_nan"])


def _read_rul_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    # RUL files are one row per test unit, in unit-number order starting at 1 --
    # made explicit here since nothing in the raw file states it.
    df.insert(0, "unit_number", range(1, len(df) + 1))
    return df


def _non_contiguous_units(df: pd.DataFrame) -> list:
    """Units whose cycles aren't exactly 1, 2, ..., n in file order.

    Downstream code relies on this: `cross_validation.sample_truncation_points`
    samples cycle numbers by fraction of the max cycle and expects every one to
    exist, and per-engine rolling features assume rows are in cycle order.
    """
    bad = []
    for unit, cycles in df.groupby("unit_number", sort=False)["time_cycles"]:
        if not (cycles.to_numpy() == range(1, len(cycles) + 1)).all():
            bad.append(unit)
    return bad


def load_subset(subset: str, data_dir: Path | str = "data/raw", validate: bool = True) -> CMAPSSData:
    """Load one C-MAPSS subset's train, test, and RUL tables into labeled DataFrames.

    Assumes `download_cmapss(subset, data_dir)` has already been run (it is not
    called automatically here, so loading is fast and side-effect-free once cached).

    With `validate=True` (the default), checks row and unit counts against
    `SUBSET_INFO`, that every engine's cycles run contiguously from 1, and that
    the RUL file covers exactly the test engines with non-negative values.
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {VALID_SUBSETS}")

    data_dir = Path(data_dir)
    train = _read_sensor_file(data_dir / f"train_{subset}.txt")
    test = _read_sensor_file(data_dir / f"test_{subset}.txt")
    rul = _read_rul_file(data_dir / f"RUL_{subset}.txt")

    if validate:
        expected = SUBSET_INFO[subset]
        checks = {
            "train rows": (len(train), expected["train_rows"]),
            "test rows": (len(test), expected["test_rows"]),
            "train units": (train["unit_number"].nunique(), expected["train_units"]),
            "test units": (test["unit_number"].nunique(), expected["test_units"]),
            "RUL rows": (len(rul), expected["test_units"]),
        }
        mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
        if mismatches:
            raise ValueError(
                f"{subset}: loaded data doesn't match reference dataset statistics "
                f"(got, expected): {mismatches}. The mirror may be stale or corrupted -- "
                f"re-download with force=True, or check BASE_URL."
            )

        problems = []
        for name, frame in (("train", train), ("test", test)):
            bad_units = _non_contiguous_units(frame)
            if bad_units:
                problems.append(f"{name} units with non-contiguous cycles (first 10): {bad_units[:10]}")
        if set(rul["unit_number"]) != set(test["unit_number"]):
            problems.append("RUL file unit numbers don't match the test set's unit numbers")
        if (rul["RUL"] < 0).any():
            problems.append("RUL file contains negative values")
        if problems:
            raise ValueError(f"{subset}: structural check failed -- " + "; ".join(problems))

    return CMAPSSData(subset=subset, train=train, test=test, rul=rul)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for subset in VALID_SUBSETS:
        download_cmapss(subset)
        data = load_subset(subset)
        logger.info("%s OK: train=%s test=%s rul=%s", subset, data.train.shape, data.test.shape, data.rul.shape)