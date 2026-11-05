from pathlib import Path

from PIL import Image

from midp.classifier import TypeClassifier, build_type_prompt, parse_type
from midp.config import load_document_types

TYPES = load_document_types(Path(__file__).parents[2] / "configs" / "document_types.yaml")
NAMES = [entry["name"] for entry in TYPES]


def test_the_prompt_names_every_registered_type_and_its_description() -> None:
    prompt = build_type_prompt(TYPES)

    for entry in TYPES:
        assert entry["name"] in prompt
        assert entry["description"] in prompt


def test_an_answer_naming_a_known_type_is_read() -> None:
    assert parse_type('{"type": "invoice"}', NAMES) == "invoice"
    assert parse_type('```json\n{"type": "delivery_note"}\n```', NAMES) == "delivery_note"


def test_an_unknown_type_or_a_broken_answer_is_none() -> None:
    assert parse_type('{"type": "contract"}', NAMES) is None
    assert parse_type("not json", NAMES) is None


def test_the_classifier_returns_the_type_the_model_chose() -> None:
    classifier = TypeClassifier(model=lambda image, prompt: '{"type": "receipt"}', types=TYPES)

    assert classifier.classify(Image.new("RGB", (4, 4))) == "receipt"
