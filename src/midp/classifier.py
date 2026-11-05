"""Decide which document type an image is, from the types in the registry.

The prompt is built from configs/document_types.yaml, so adding a type means
adding a registry entry and its config, not editing this module.

The model also says how sure it is. A type chosen with less confidence than
the threshold is not returned: the document goes to review instead of being
extracted with the wrong type's fields.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

ModelCall = Callable[[bytes, str], str]
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class Classification:
    """document_type is None when the type is unknown or not sure enough."""

    document_type: str | None
    confidence: float | None


def build_type_prompt(types: list[dict]) -> str:
    """Ask for one type name, chosen from the registry, with a confidence from 0 to 1."""
    lines = [f"- {entry['name']}: {entry['description']}" for entry in types]
    names = " or ".join(f'"{entry["name"]}"' for entry in types)
    return (
        "Look at this document and decide its type. The possible types are:\n"
        + "\n".join(lines)
        + "\nSay how sure you are."
        + f'\nReturn only JSON: {{"type": {names}, "confidence": a number from 0 to 1}}.'
    )


def parse_classification(text: str, names: list[str]) -> tuple[str | None, float | None]:
    """The type and confidence in the model's answer. Unknown types and broken answers give None."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        answer = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(answer, dict):
        return None, None
    document_type = answer.get("type") if answer.get("type") in names else None
    confidence = answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        confidence = None
    return document_type, confidence


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TypeClassifier:
    """Chooses one registry type per image, or none if the model is not sure enough."""

    def __init__(self, model: ModelCall, types: list[dict], threshold: float = DEFAULT_THRESHOLD) -> None:
        self._model = model
        self._names = [entry["name"] for entry in types]
        self._prompt = build_type_prompt(types)
        self._threshold = threshold

    def classify(self, image: Image.Image) -> Classification:
        document_type, confidence = parse_classification(
            self._model(_png_bytes(image), self._prompt), self._names
        )
        if document_type is None or confidence is None or confidence < self._threshold:
            return Classification(document_type=None, confidence=confidence)
        return Classification(document_type=document_type, confidence=confidence)
