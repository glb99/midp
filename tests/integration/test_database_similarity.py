"""The similarity classifier reads its nearest examples from the examples table in Postgres."""

import numpy as np
import pytest
from sqlmodel import Session

from midp.db import EMBEDDING_DIMENSIONS
from midp.repository import ExampleRepository
from midp.similarity import SimilarityClassifier


def _vector(*axes: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for axis in axes:
        vector[axis] = 1.0 / len(axes) ** 0.5
    return vector


def _embed(texts: list[str]) -> np.ndarray:
    # a text such as "0,1" stands for the vector with weight on those axes
    return np.array([_vector(*(int(axis) for axis in text.split(","))) for text in texts])


def test_the_nearest_stored_examples_decide_the_type(session: Session) -> None:
    repository = ExampleRepository(session)
    for document_type, text in [("receipt", "0"), ("receipt", "0,1"), ("invoice", "1"), ("delivery_note", "2")]:
        repository.add(document_type=document_type, text=text, embedding=_vector(*(int(a) for a in text.split(","))))

    classifier = SimilarityClassifier(
        transcribe=lambda image: image,
        embed=_embed,
        nearest=lambda vector, k: ExampleRepository(session).nearest(vector, k),
        k=2,
    )

    result = classifier.classify("0")

    assert result.document_type == "receipt"
    assert result.confidence == pytest.approx(1.0, abs=1e-5)
