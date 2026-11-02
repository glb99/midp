"""Systems that read a receipt image and return values in the config's shape.

A predictor takes the image and returns a dict whose keys are the field
names in configs/receipts.yaml. Fields it cannot read are None; lines it
cannot read are simply absent.
"""

from __future__ import annotations

from typing import Any


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
