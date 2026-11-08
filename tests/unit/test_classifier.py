from pathlib import Path

from PIL import Image

from midp.classifier import (
    Classification,
    TypeClassifier,
    build_type_prompt,
    parse_classification,
)
from midp.config import load_document_types

TYPES = load_document_types(Path(__file__).parents[2] / "configs" / "document_types.yaml")
NAMES = [entry["name"] for entry in TYPES]


def test_the_prompt_names_every_registered_type_and_its_description() -> None:
    prompt = build_type_prompt(TYPES)

    for entry in TYPES:
        assert entry["name"] in prompt
        assert entry["description"] in prompt


def test_a_known_type_with_its_confidence_is_read() -> None:
    assert parse_classification('{"type": "invoice", "confidence": 0.95}', NAMES) == ("invoice", 0.95)
    assert parse_classification('```json\n{"type": "delivery_note", "confidence": 0.9}\n```', NAMES) == (
        "delivery_note",
        0.9,
    )


def test_an_unknown_type_or_a_broken_answer_gives_no_type() -> None:
    assert parse_classification('{"type": "contract", "confidence": 0.99}', NAMES) == (None, 0.99)
    assert parse_classification("not json", NAMES) == (None, None)


def test_a_confident_answer_is_classified() -> None:
    classifier = TypeClassifier(model=lambda image, prompt: '{"type": "receipt", "confidence": 0.97}', types=TYPES)

    result = classifier.classify(Image.new("RGB", (4, 4)))

    assert result == Classification(document_type="receipt", confidence=0.97)


def test_an_unsure_answer_is_not_classified() -> None:
    classifier = TypeClassifier(model=lambda image, prompt: '{"type": "invoice", "confidence": 0.55}', types=TYPES)

    result = classifier.classify(Image.new("RGB", (4, 4)))

    assert result == Classification(document_type=None, confidence=0.55)
