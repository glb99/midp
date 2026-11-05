"""Systems that read a receipt image and return values in the config's shape.

A predictor takes the image and returns a dict whose keys are the field
names in configs/receipts.yaml. Fields it cannot read are None; lines it
cannot read are simply absent.
"""

from __future__ import annotations

import io
import json
import re
import time
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


class FakePredictor:
    """Waits a fixed time and predicts nothing, like a network call of known length.

    Used to measure the script itself: with the latency fixed, any change in
    run time comes from our code, not from the model or the network.
    """

    name = "fake"

    def __init__(self, latency: float) -> None:
        self._latency = latency

    def predict(self, image: Any) -> dict:
        time.sleep(self._latency)  # releases the GIL, like waiting for a response
        return EmptyPredictor().predict(image)


def build_prompt(fields: list[str]) -> str:
    """Ask for JSON with exactly the configured fields, nothing else."""
    return (
        "Read this receipt. Return only a JSON object with these keys: "
        + ", ".join(fields)
        + ". 'lines' is a list of objects with 'description', 'quantity' and "
        "'amount'. Discounts are not lines: report their total as a negative 'discount'. "
        "Amounts are whole numbers without separators or currency. "
        "Use null for any value that is not on the receipt."
    )


def parse_response(text: str, config: dict) -> dict:
    """Turn the model's text into a dict in the config's shape.

    Each field is read according to its type in the config: amounts become
    numbers, text stays text. Models sometimes wrap JSON in a code fence,
    which is removed first. A value that cannot be read becomes None.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    raw = json.loads(cleaned)
    specs = {field["name"]: field for field in config["fields"]}

    result: dict[str, Any] = {}
    for name, value in raw.items():
        spec = specs.get(name)
        if spec is None:
            continue
        if spec.get("kind") == "list":
            item_types = {item["name"]: item["type"] for item in spec.get("item", [])}
            result[name] = [_parse_item(item, item_types) for item in (value or [])]
        else:
            result[name] = _parse_scalar(value, spec["type"])
    return result


def _parse_item(item: Any, item_types: dict[str, str]) -> dict:
    if not isinstance(item, dict):
        return {}
    return {key: _parse_scalar(value, item_types[key]) for key, value in item.items() if key in item_types}


def _parse_scalar(value: Any, kind: str) -> Any:
    if kind == "amount":
        return _safe(parse_amount, value)
    if kind == "integer":
        return _safe(parse_quantity, value)
    return None if value is None else str(value)


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

    def __init__(self, model: ModelCall, config: dict, model_name: str) -> None:
        self._model = model
        self._config = config
        self._prompt = build_prompt([field["name"] for field in config["fields"]])
        self.model_name = model_name  # recorded with each stored result

    def predict(self, image: Any) -> dict:
        return parse_response(self._model(_png_bytes(image), self._prompt), self._config)
