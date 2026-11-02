"""Systems that read a receipt image and return values in the config's shape.

A predictor takes the image and returns a dict whose keys are the field
names in configs/receipts.yaml. Fields it cannot read are None; lines it
cannot read are simply absent.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from typing import Any

from midp.values import parse_amount, parse_quantity

# Sends the image bytes and the prompt to a model, returns the model's text.
ModelCall = Callable[[bytes, str], str]


class EmptyPredictor:
    """Predicts nothing. Every field is missing.

    It is the floor every real system has to beat: it scores zero recall
    everywhere and no precision at all.
    """

    name = "empty"

    def predict(self, image: Any) -> dict:
        return {
            "lines": [],
            "subtotal": None,
            "service": None,
            "tax": None,
            "rounding": None,
            "total": None,
        }


def build_prompt(fields: list[str]) -> str:
    """Ask for JSON with exactly the configured fields, nothing else."""
    return (
        "Read this receipt. Return only a JSON object with these keys: "
        + ", ".join(fields)
        + ". 'lines' is a list of objects with 'description', 'quantity' and "
        "'amount'. Amounts are whole numbers without separators or currency. "
        "Use null for any value that is not on the receipt."
    )


def parse_response(text: str) -> dict:
    """Turn the model's text into a dict in the config's shape.

    Models sometimes wrap JSON in a code fence; that is removed first.
    A value that cannot be read becomes None, so it scores as a miss.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    raw = json.loads(cleaned)

    lines = []
    for item in raw.get("lines") or []:
        lines.append(
            {
                "description": item.get("description"),
                "quantity": _safe(parse_quantity, item.get("quantity")),
                "amount": _safe(parse_amount, item.get("amount")),
            }
        )
    result = {"lines": lines}
    for field, value in raw.items():
        if field != "lines":
            result[field] = _safe(parse_amount, value)
    return result


def _safe(parse, value):
    if value is None:
        return None
    try:
        return parse(str(value))
    except ValueError:
        return None


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MultimodalPredictor:
    """Sends the receipt image to a multimodal model and reads its JSON answer.

    The model call is passed in, so the logic can be tested without a
    network and the provider can be swapped without touching this class.
    """

    name = "multimodal"

    def __init__(self, model: ModelCall, fields: list[str]) -> None:
        self._model = model
        self._prompt = build_prompt(fields)

    def predict(self, image: Any) -> dict:
        return parse_response(self._model(_png_bytes(image), self._prompt))
