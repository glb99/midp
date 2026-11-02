"""Field-level precision, recall and F1 for one document.

A scalar field is correct when the predicted value equals the reference.
A line item is correct when its description (case and spacing ignored)
and amount match an unused reference line. Each reference line can be
matched once, so repeated names pair up in order.

A missing prediction lowers recall. An extra or wrong prediction lowers
precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    return " ".join((text or "").lower().split())


def score_lines(predicted: list[dict], reference: list[dict]) -> Score:
    unmatched = list(reference)
    tp = fp = 0
    for line in predicted:
        match = next(
            (
                ref
                for ref in unmatched
                if _name(ref["description"]) == _name(line["description"])
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
    scores = {"lines": score_lines(predicted.get("lines") or [], reference.get("lines") or [])}
    for field, value in reference.items():
        if field != "lines":
            scores[field] = score_scalar(predicted.get(field), value)
    return scores


def aggregate(documents: list[dict[str, Score]]) -> dict[str, Score]:
    totals: dict[str, Score] = {}
    for scores in documents:
        for field, score in scores.items():
            totals[field] = totals.get(field, Score(0, 0, 0)) + score
    return totals
