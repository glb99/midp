"""Typed examples and their nearest-neighbour search, against Postgres with pgvector."""

import math

import pytest
from sqlmodel import Session

from midp.db import EMBEDDING_DIMENSIONS
from midp.repository import ExampleRepository


def along(*axes: int) -> list[float]:
    """A unit vector with equal weight on the given axes, zero elsewhere."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for axis in axes:
        vector[axis] = 1.0 / math.sqrt(len(axes))
    return vector


def test_nearest_returns_the_most_similar_examples_first(session: Session) -> None:
    examples = ExampleRepository(session)
    examples.add(document_type="receipt", text="TOTAL 61.799", embedding=along(0))
    examples.add(document_type="invoice", text="Invoice F1", embedding=along(1))
    examples.add(document_type="delivery_note", text="Albarán", embedding=along(0, 1))

    nearest = examples.nearest(along(0), k=2)

    assert [document_type for document_type, _ in nearest] == ["receipt", "delivery_note"]
    assert nearest[0][1] == pytest.approx(1.0)
    assert nearest[1][1] == pytest.approx(1 / math.sqrt(2))


def test_nearest_returns_at_most_k(session: Session) -> None:
    examples = ExampleRepository(session)
    for axis, document_type in enumerate(["receipt", "invoice", "delivery_note"]):
        examples.add(document_type=document_type, text=document_type, embedding=along(axis))

    assert len(examples.nearest(along(0), k=3)) == 3
    assert len(examples.nearest(along(0), k=1)) == 1


def test_nearest_on_an_empty_table_is_empty(session: Session) -> None:
    assert ExampleRepository(session).nearest(along(0), k=3) == []
