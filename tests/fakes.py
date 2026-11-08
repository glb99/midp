"""Stand-ins for the model, used wherever a test must not call Gemini."""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO

from PIL import Image

from midp.classifier import Classification


class FixedPredictor:
    name = "fixed"
    model_name = "test-model"

    def predict(self, image):
        return {"lines": [], "total": Decimal(61799), "discount": Decimal(-19000)}


class FixedClassifier:
    """Always answers the same type, with a fixed confidence."""

    def __init__(self, document_type: str | None = "receipt") -> None:
        self._classification = Classification(document_type=document_type, confidence=0.95)

    def classify(self, image):
        return self._classification


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()
