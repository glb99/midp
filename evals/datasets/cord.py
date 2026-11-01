"""Download, load and freeze the CORD receipt dataset for evaluation.

Run by hand. Tests and CI never call this module's download path.

    uv run python -m evals.datasets.cord download --revision <full commit SHA>
    uv run python -m evals.datasets.cord freeze
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DATASET_ID = "naver-clova-ix/cord-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cord"
SPLITS_DIR = REPO_ROOT / "evals" / "splits"
REVISION_FILE = DATA_DIR / "REVISION"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def download(revision: str) -> None:
    """Fetch the dataset at an exact commit into data/cord/."""
    if not SHA_PATTERN.match(revision):
        raise SystemExit("--revision must be the full 40-character commit SHA")

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=revision, cache_dir=str(DATA_DIR))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REVISION_FILE.write_text(revision + "\n", encoding="utf-8")
    for name, split in dataset.items():
        print(f"{name}: {len(split)} documents")


def load(split: str):
    """Return one split of the downloaded dataset. Never downloads."""
    if not REVISION_FILE.exists():
        raise FileNotFoundError(
            "CORD is not downloaded. Run: "
            "uv run python -m evals.datasets.cord download --revision <SHA>"
        )
    from datasets import load_dataset

    revision = REVISION_FILE.read_text(encoding="utf-8").strip()
    dataset = load_dataset(DATASET_ID, revision=revision, cache_dir=str(DATA_DIR))
    return dataset[split]


def freeze() -> None:
    """Write the dev and test id lists to evals/splits/.

    Dev is CORD's train plus validation. Test is CORD's test split. Ids are
    the split name and the row index, so the lists do not depend on the
    order of the rows in a later download.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    dev_ids = [f"train-{i:05d}" for i in range(len(load("train")))]
    dev_ids += [f"validation-{i:05d}" for i in range(len(load("validation")))]
    test_ids = [f"test-{i:05d}" for i in range(len(load("test")))]
    (SPLITS_DIR / "cord_dev.txt").write_text("\n".join(dev_ids) + "\n", encoding="utf-8")
    (SPLITS_DIR / "cord_test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    print(f"dev: {len(dev_ids)} ids, test: {len(test_ids)} ids -> {SPLITS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.datasets.cord")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="download CORD at a pinned revision")
    fetch.add_argument("--revision", required=True, help="full commit SHA on Hugging Face")
    commands.add_parser("freeze", help="write dev and test id lists")
    args = parser.parse_args()

    if args.command == "download":
        download(args.revision)
    elif args.command == "freeze":
        freeze()


if __name__ == "__main__":
    main()
