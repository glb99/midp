"""Download, load and freeze the CORD receipt dataset for evaluation.

Run by hand. Tests and CI never call this module's download path.

    uv run python -m evals.datasets.cord download --revision <full commit SHA>
    uv run python -m evals.datasets.cord freeze
    uv run python -m evals.datasets.cord export-images

Images are read from the parquet files. export-images writes them once to
data/cord_images/ as <id>.png, so they can be opened by hand. The parquet
stays the source of truth; the folder is a cache, outside git.
"""

from __future__ import annotations

import argparse
import functools
import re
from decimal import Decimal
from pathlib import Path

from evals.metrics import UNLABELLED

DATASET_ID = "naver-clova-ix/cord-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cord"
SPLITS_DIR = REPO_ROOT / "evals" / "splits"
REVISION_FILE = DATA_DIR / "REVISION"
IMAGES_DIR = REPO_ROOT / "data" / "cord_images"
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


def image_path(row_id: str) -> Path:
    """Where the exported image of a row lives, such as data/cord_images/train-00019.png."""
    return IMAGES_DIR / f"{row_id}.png"


def export_images() -> int:
    """Write every row's image to data/cord_images/, skipping the ones already there."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for split in ("train", "validation", "test"):
        for index, row in enumerate(load(split)):
            path = image_path(f"{split}-{index:05d}")
            if not path.exists():
                row["image"].save(path)
                written += 1
    return written


def get_row(row_id: str) -> dict:
    """Return the row for an id such as "train-00005"."""
    split, _, index = row_id.rpartition("-")
    return load(split)[int(index)]


@functools.cache
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


def to_reference(row: dict) -> dict:
    """Map one CORD row to the field names in configs/receipts.yaml.

    CORD's labels keep the receipt's own wording (menu, sub_total, total).
    Everything here is a rename or a parse; the product never sees CORD's
    names. Fields CORD does not label come back as None.
    """
    import json

    from midp.values import parse_quantity

    parse = json.loads(row["ground_truth"])["gt_parse"]
    menu = parse.get("menu") or []
    if isinstance(menu, dict):  # CORD stores a single line as an object
        menu = [menu]
    sub_total = _merge_blocks(parse.get("sub_total"))
    total = _merge_blocks(parse.get("total"))

    lines = []
    discounts = []
    for item in menu:
        amount = _amount(item.get("price"))
        if amount is not None and amount is not UNLABELLED and amount < 0:
            discounts.append(amount)  # a discount is an adjustment, not a product
            continue
        lines.append(
            {
                "description": _as_text(item.get("nm")),
                "quantity": _safe(parse_quantity, _as_text(item.get("cnt"))),
                "amount": amount,
            }
        )
    # One unreadable line makes the whole line list unlabelled: we cannot
    # tell which predicted lines are right without it.
    if any(line["amount"] is UNLABELLED for line in lines):
        lines = UNLABELLED

    return {
        "lines": lines,
        "subtotal": _amount(sub_total.get("subtotal_price")),
        "service": _amount(sub_total.get("service_price")),
        "tax": _amount(sub_total.get("tax_price")),
        "rounding": _amount(sub_total.get("etc")),
        "total": _amount(total.get("total_price")),
        "discount": _discount(sub_total.get("discount_price"), discounts),
    }


def _safe(parse, value):
    try:
        return parse(value)
    except ValueError:
        return UNLABELLED


def _amount(value: object):
    """An amount label, None when the label is absent, or UNLABELLED.

    A list whose values are all equal means the same amount was printed
    twice (TOTAL and DPP on one receipt). A list with different values is
    a breakdown, not an amount. Text that cannot be read is UNLABELLED too.
    """
    from midp.values import parse_amount

    if isinstance(value, list):
        distinct = {str(part).strip() for part in value if part not in (None, "")}
        if len(distinct) != 1:
            return UNLABELLED
        value = distinct.pop()
    return _safe(parse_amount, value)


def _discount(label: object, negative_menu_items: list) -> Decimal | None:
    """Total discount as a negative number, or None if the receipt has none.

    CORD writes the discount in sub_total as a positive number (its
    "discount_price"); some receipts list it as a negative menu item instead.
    Both count, and the result is always negative.
    """
    total = Decimal(0)
    found = False
    amount = _amount(label)
    if amount is not None and amount is not UNLABELLED:
        total -= amount
        found = True
    for item in negative_menu_items:
        total += item
        found = True
    return total if found else None


def _merge_blocks(value: object) -> dict:
    """One label block, or a list of them (one receipt has two sub_total blocks).

    The first block wins when two blocks give the same field. That one
    receipt is ambiguous, and this rule is a choice, not a fact.
    """
    if not value:
        return {}
    blocks = value if isinstance(value, list) else [value]
    merged: dict = {}
    for block in blocks:
        for key, field in block.items():
            merged.setdefault(key, field)
    return merged


def _as_text(value: object) -> str | None:
    """CORD sometimes splits one label into parts, e.g. ["-", "Gyro Platter"].

    Join the parts with spaces. For a quantity, the caller keeps only the
    numeric part, so a stray word in the list is dropped there instead.
    """
    if isinstance(value, list):
        parts = [str(part) for part in value if part not in (None, "")]
        numeric = [part for part in parts if part.strip().isdigit()]
        return numeric[0] if numeric else " ".join(parts) or None
    return value


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
    commands.add_parser("export-images", help="write every image to data/cord_images/ once")
    args = parser.parse_args()

    if args.command == "download":
        download(args.revision)
    elif args.command == "freeze":
        freeze()
    elif args.command == "export-images":
        print(f"exported {export_images()} images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
