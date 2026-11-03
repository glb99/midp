"""Run a predictor over the dev split and print field-level scores.

    uv run python -m evals.run

Results are also saved to runs/ as JSON, so runs can be compared later.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from evals.datasets.cord import get_row, to_reference
from evals.metrics import UNLABELLED, aggregate, score_document
from midp.config import field_names, load_config
from midp.gemini import gemini_model
from midp.predictors import EmptyPredictor, MultimodalPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = REPO_ROOT / "evals" / "splits" / "cord_dev.txt"
CONFIG_FILE = REPO_ROOT / "configs" / "receipts.yaml"
RUNS_DIR = REPO_ROOT / "runs"


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def build_predictor(name: str, model_name: str | None):
    """Return the predictor and, for Gemini, the model call with its usage totals."""
    if name == "empty":
        return EmptyPredictor(), None
    if name == "gemini":
        fields = field_names(load_config(CONFIG_FILE))
        model = gemini_model(model_name) if model_name else gemini_model()
        return MultimodalPredictor(model=model, fields=fields), model
    raise SystemExit(f"unknown predictor: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.run")
    parser.add_argument("--predictor", default="empty", choices=["empty", "gemini"])
    parser.add_argument("--model", default=None, help="model name, e.g. gemini-3.8-flash-lite")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents")
    args = parser.parse_args()

    predictor, model = build_predictor(args.predictor, args.model)
    ids = read_ids(SPLIT_FILE)
    if args.limit is not None:
        ids = ids[: args.limit]

    per_document = []
    unlabelled: Counter[str] = Counter()
    for row_id in ids:
        row = get_row(row_id)
        predicted = predictor.predict(row["image"])
        reference = to_reference(row)
        unlabelled.update(field for field, value in reference.items() if value is UNLABELLED)
        per_document.append(score_document(predicted, reference))
    totals = aggregate(per_document)

    print(f"predictor: {predictor.name}, split: dev, documents: {len(ids)}\n")
    print(f"{'field':<10}{'tp':>6}{'fp':>6}{'fn':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for field, score in totals.items():
        print(
            f"{field:<10}{score.tp:>6}{score.fp:>6}{score.fn:>6}  "
            f"{fmt(score.precision):>9}  {fmt(score.recall):>6}  {fmt(score.f1):>6}"
        )

    if unlabelled:
        print("\nunlabelled in the reference (not scored):")
        for field, count in unlabelled.items():
            print(f"  {field}: {count}")

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {
        "predictor": predictor.name,
        "model": args.model or (model.model if model else None),
        "split": "dev",
        "documents": len(ids),
        "fields": {
            field: {
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
            }
            for field, s in totals.items()
        },
    }
    if model is not None:
        usage = dict(model.usage)
        result["usage"] = usage
        per = {k: round(v / max(usage.get("calls", 1), 1), 1) for k, v in usage.items()}
        print("\nusage (totals / per document):")
        for key, total in usage.items():
            print(f"  {key}: {total} / {per[key]}")
    out = RUNS_DIR / f"{stamp}-{predictor.name}-dev.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
