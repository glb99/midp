"""Field-level precision, recall and F1 for one document.

A scalar field is correct when the predicted value equals the reference.
A line item is correct when its description (case and spacing ignored)
and amount match an unused reference line. Each reference line can be
matched once, so repeated names pair up in order.

A missing prediction lowers recall. An extra or wrong prediction lowers
precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


class _Unlabelled:
    """Marks a reference field that could not be read. It is not scored."""

    def __repr__(self) -> str:
        return "UNLABELLED"


UNLABELLED = _Unlabelled()


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def __add__(self, other: Score) -> Score:
        return Score(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def score_scalar(predicted: Any, reference: Any) -> Score:
    if reference is None:
        return Score(tp=0, fp=0 if predicted is None else 1, fn=0)
    if predicted is None:
        return Score(tp=0, fp=0, fn=1)
    if predicted == reference:
        return Score(tp=1, fp=0, fn=0)
    return Score(tp=0, fp=1, fn=1)


def _name(text: str | None) -> str:
    """Lowercase, without a leading item code and without punctuation.

    "1069-ROTI KEJU" and "ROTI KEJU" are the same item; so are "- TICKET CP"
    and "TICKET CP". Only the letters and digits of the name are compared.
    """
    cleaned = re.sub(r"^\s*\d+\s*[-.]\s*", "", text or "")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", cleaned)
    return " ".join(cleaned.lower().split())


def _same_item(reference: str | None, predicted: str | None) -> bool:
    """Same name, or one name contains the other as whole words.

    The shorter name must have at least two words, so a single word such
    as "Tahu" cannot match a longer line by accident. Callers also compare
    amounts, which is what keeps "Ice Tea" and "Ice Tea Tawar" apart when
    their prices differ.
    """
    a, b = _name(reference), _name(predicted)
    if a == b:
        return True
    if SequenceMatcher(None, a, b).ratio() >= 0.85:  # typos such as CUSTAFD for CUSTARD
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter.split()) < 2:
        return False
    return f" {shorter} " in f" {longer} "


def score_lines(predicted: list[dict], reference: list[dict]) -> Score:
    unmatched = list(reference)
    tp = fp = 0
    for line in predicted:
        match = next(
            (
                ref
                for ref in unmatched
                if _same_item(ref["description"], line["description"])
                and ref["amount"] == line["amount"]
            ),
            None,
        )
        if match is None:
            fp += 1
        else:
            tp += 1
            unmatched.remove(match)
    return Score(tp=tp, fp=fp, fn=len(unmatched))


def score_document(predicted: dict, reference: dict) -> dict[str, Score]:
    """Score every field whose reference is readable. Unlabelled fields are skipped."""
    scores: dict[str, Score] = {}
    for field, value in reference.items():
        if value is UNLABELLED:
            continue
        if field == "lines":
            scores[field] = score_lines(predicted.get("lines") or [], value)
        else:
            scores[field] = score_scalar(predicted.get(field), value)
    return scores


def aggregate(documents: list[dict[str, Score]]) -> dict[str, Score]:
    totals: dict[str, Score] = {}
    for scores in documents:
        for field, score in scores.items():
            totals[field] = totals.get(field, Score(0, 0, 0)) + score
    return totals
