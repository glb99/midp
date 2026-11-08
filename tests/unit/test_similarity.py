import numpy as np

from midp.similarity import SimilarityClassifier


def _fake_embed(texts: list[str]) -> np.ndarray:
    # each text is a 2-d vector given by its first two characters: "a" or "b" then a digit
    return np.array([[1.0, 0.0] if t.startswith("a") else [0.0, 1.0] for t in texts])


def test_the_most_common_type_among_the_nearest_examples_wins() -> None:
    examples = [("delivery_note", "a1"), ("delivery_note", "a2"), ("invoice", "b1"), ("invoice", "b2")]
    classifier = SimilarityClassifier(transcribe=lambda image: image, embed=_fake_embed, examples=examples, k=3)

    result = classifier.classify("a9")

    assert result.document_type == "delivery_note"
    assert result.confidence == 1.0
