"""Decide which document type an image is, from the types in the registry.

The prompt is built from configs/document_types.yaml, so adding a type means
adding a registry entry and its config, not editing this module.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from typing import Any

from PIL import Image

ModelCall = Callable[[bytes, str], str]


def build_type_prompt(types: list[dict]) -> str:
    """Ask for one type name, chosen from the registry and nothing else."""
    lines = [f"- {entry['name']}: {entry['description']}" for entry in types]
    names = " or ".join(f'{{"type": "{entry["name"]}"}}' for entry in types)
    return (
        "Look at this document and decide its type. The possible types are:\n"
        + "\n".join(lines)
        + f"\nReturn only JSON: {names}."
    )


def parse_type(text: str, names: list[str]) -> str | None:
    """The type named in the model's answer, or None if it names no known type."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned).get("type")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if value in names else None


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TypeClassifier:
    """Chooses one registry type per image. None means the model named no known type."""

    def __init__(self, model: ModelCall, types: list[dict]) -> None:
        self._model = model
        self._names = [entry["name"] for entry in types]
        self._prompt = build_type_prompt(types)

    def classify(self, image: Image.Image) -> str | None:
        return parse_type(self._model(_png_bytes(image), self._prompt), self._names)
