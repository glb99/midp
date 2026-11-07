"""Measure the document type classifier the app uses (TypeClassifier).

    uv run python -m evals.classify

The test set is fixed: 40 titled delivery notes, 40 untitled delivery notes,
40 invoices (synthetic, with known types) and 40 receipts (CORD dev). Each
image goes through the same TypeClassifier as POST /receipts/extract. The
predictions, per-type precision, recall and F1, the confusion matrix and the
review rate (documents the classifier does not type) are saved to runs/.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from PIL import Image

from evals.common import REPO_ROOT, RUNS_DIR, fmt
from evals.metrics import Score

TYPES_FILE = REPO_ROOT / "configs" / "document_types.yaml"
CORD_DEV = REPO_ROOT / "evals" / "splits" / "cord_dev.txt"
CORD_EXAMPLES = REPO_ROOT / "evals" / "splits" / "cord_examples.txt"
CLASSES = ["receipt", "invoice", "delivery_note"]
NONE = "none"
TRANSCRIBE = "Transcribe all the text in this document, in reading order, as plain text. Output only the text, nothing else."


def example_set() -> list[tuple[str, Image.Image]]:
    """Typed examples for the similarity method: untitled delivery notes, titled invoices and receipts.

    Delivery notes and invoices are generated with seeds the test set does not use.
    Receipts come from CORD train, outside the first 40 dev ids that the test set uses.
    """
    import random

    from evals.datasets.synthetic_invoices import _invoice, _render

    examples: list[tuple[str, Image.Image]] = []
    rng = random.Random(3)
    for i in range(20):
        labels, _ = _invoice(rng, i)
        for key in ("subtotal", "tax", "total"):
            labels[key] = None
        examples.append(("delivery_note", _render(labels, rng, title=None, prices=False)))
    rng = random.Random(4)
    for i in range(20):
        labels, _ = _invoice(rng, i)
        examples.append(("invoice", _render(labels, rng, title="FACTURA", prices=True)))
    from evals.datasets.cord import get_row

    ids = CORD_EXAMPLES.read_text(encoding="utf-8").split()
    examples += [("receipt", get_row(row_id)["image"]) for row_id in ids]
    return examples


def class_scores(pairs: list[tuple[str, str | None]], classes: list[str]) -> dict[str, Score]:
    """Per-type TP, FP and FN from (true type, predicted type) pairs.

    A prediction of None is not a false positive of any type; it is a false
    negative of the true type. A wrong type counts as a false positive of the
    predicted type and a false negative of the true one.
    """
    scores: dict[str, Score] = {}
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        scores[c] = Score(tp=tp, fp=fp, fn=fn)
    return scores


def confusion(pairs: list[tuple[str, str | None]], classes: list[str]) -> dict[tuple[str, str], int]:
    counts = Counter((t, p if p is not None else NONE) for t, p in pairs)
    return {(t, p): counts.get((t, p), 0) for t in classes for p in [*classes, NONE]}


def review_rate(pairs: list[tuple[str, str | None]]) -> float:
    return sum(1 for _, p in pairs if p is None) / len(pairs) if pairs else 0.0


def _labelled_images(folder: str, true_type: str, n: int = 40) -> list[tuple[str, str, Image.Image]]:
    root = REPO_ROOT / folder
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    return [
        (f"{folder}/{name}", true_type, Image.open(root / name).convert("RGB"))
        for name in sorted(labels)[:n]
    ]


def _cord_images(n: int = 40) -> list[tuple[str, str, Image.Image]]:
    from evals.datasets.cord import get_row

    ids = CORD_DEV.read_text(encoding="utf-8").split()[:n]
    return [(f"cord/{row_id}", "receipt", get_row(row_id)["image"]) for row_id in ids]


def test_set(base: str = "data") -> list[tuple[str, str, Image.Image]]:
    """The test set. base="data" uses the clean synthetic images; base="data/synthetic_scanned" the scan-like copies.

    CORD receipts are real photos and are the same in both.
    """
    return (
        _labelled_images(f"{base}/synthetic_delivery_notes", "delivery_note")
        + _labelled_images(f"{base}/synthetic_delivery_notes_untitled", "delivery_note")
        + _labelled_images(f"{base}/synthetic_invoices", "invoice")
        + _cord_images()
    )


def main() -> None:
    import argparse

    from midp.classifier import TypeClassifier
    from midp.config import load_document_types
    from midp.gemini import DEFAULT_MODEL, gemini_model

    parser = argparse.ArgumentParser(prog="evals.classify")
    parser.add_argument("--method", default="model", choices=["model", "similarity"],
                        help="model: the app's TypeClassifier; similarity: nearest typed examples (k=3)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name")
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N documents of the test set")
    parser.add_argument("--set", default="clean", choices=["clean", "scanned"], help="clean synthetic images, or scan-like copies")
    args = parser.parse_args()

    if args.method == "model":
        types = load_document_types(TYPES_FILE)
        classifier = TypeClassifier(model=gemini_model(args.model), types=types)
    else:
        from midp.predictors import _png_bytes
        from midp.similarity import (
            SimilarityClassifier,
            gemini_embedder,
            load_similarity_config,
        )

        model = gemini_model(args.model)
        transcribe = lambda image: model(_png_bytes(image), TRANSCRIBE)
        settings = load_similarity_config(REPO_ROOT / "configs" / "similarity.yaml")
        classifier = SimilarityClassifier(
            transcribe=transcribe, embed=gemini_embedder(), examples=example_set(), k=settings["k"]
        )
    items = test_set("data/synthetic_scanned" if args.set == "scanned" else "data")
    if args.limit is not None:
        items = items[: args.limit]

    details: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        if args.method == "similarity":
            details = list(pool.map(lambda item: classifier.inspect(item[2]), items))
            outcomes = [type("Outcome", (), {"document_type": d["document_type"], "confidence": d["confidence"]})() for d in details]
        else:
            outcomes = list(pool.map(lambda item: classifier.classify(item[2]), items))

    rows = [
        {"id": item[0], "true": item[1], "predicted": o.document_type, "confidence": o.confidence}
        for item, o in zip(items, outcomes)
    ]
    if details:
        for row, d in zip(rows, details):
            row.update({"text": d["text"], "vector": d["vector"], "best_per_type": d["best_per_type"], "margin": d["margin"]})
    pairs = [(r["true"], r["predicted"]) for r in rows]
    scores = class_scores(pairs, CLASSES)
    matrix = confusion(pairs, CLASSES)

    print(f"set: {args.set}, method: {args.method}, model: {args.model}, documents: {len(rows)}, accuracy: {sum(t == p for t, p in pairs) / len(pairs):.3f}")
    print(f"review rate (not typed): {review_rate(pairs):.3f}\n")
    print(f"{'type':15} {'tp':>4} {'fp':>4} {'fn':>4}  precision  recall      f1")
    for c in CLASSES:
        s = scores[c]
        print(f"{c:15} {s.tp:>4} {s.fp:>4} {s.fn:>4}  {fmt(s.precision):>9}  {fmt(s.recall):>6}  {fmt(s.f1):>6}")
    print("\nconfusion (rows = real, columns = predicted):")
    print(f"{'':15}" + "".join(f"{p:>15}" for p in [*CLASSES, NONE]))
    for t in CLASSES:
        print(f"{t:15}" + "".join(f"{matrix[(t, p)]:>15}" for p in [*CLASSES, NONE]))

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"{stamp}-classify-{args.set}-{args.method}-{args.model}.json"
    out.write_text(
        json.dumps(
            {
                "task": "classify",
                "set": args.set,
                "method": args.method,
                "model": args.model,
                "documents": len(rows),
                "review_rate": review_rate(pairs),
                "classes": {
                    c: {"tp": scores[c].tp, "fp": scores[c].fp, "fn": scores[c].fn,
                        "precision": scores[c].precision, "recall": scores[c].recall, "f1": scores[c].f1}
                    for c in CLASSES
                },
                "confusion": {f"{t}->{p}": n for (t, p), n in matrix.items()},
                "predictions": rows,
                "examples": (
                    [{"type": label, "vector": vec} for label, vec in zip(classifier._labels, classifier._vectors.tolist())]
                    if args.method == "similarity" else []
                ),
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
