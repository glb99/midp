import numpy as np

from midp.similarity import SimilarityClassifier


def test_the_most_common_type_among_the_nearest_examples_wins() -> None:
    neighbours = [("delivery_note", 1.0), ("delivery_note", 0.9), ("invoice", 0.5)]
    classifier = SimilarityClassifier(
        transcribe=lambda image: image,
        embed=lambda texts: np.array([[1.0, 0.0]] * len(texts)),
        nearest=lambda vector, k: neighbours[:k],
        k=3,
    )

    result = classifier.classify("a9")

    assert result.document_type == "delivery_note"
    assert result.confidence == 1.0
