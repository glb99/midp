from decimal import Decimal
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from midp.api import app, get_predictor


class FixedPredictor:
    def predict(self, image):
        return {"lines": [], "total": Decimal(61799), "discount": Decimal(-19000)}


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_returns_fields_as_json_numbers() -> None:
    app.dependency_overrides[get_predictor] = lambda: FixedPredictor()
    client = TestClient(app)

    response = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"lines": [], "total": 61799, "discount": -19000}


def test_extract_refuses_files_that_are_not_images() -> None:
    app.dependency_overrides[get_predictor] = lambda: FixedPredictor()
    client = TestClient(app)

    wrong_type = client.post("/receipts/extract", files={"file": ("r.pdf", b"%PDF", "application/pdf")})
    broken = client.post("/receipts/extract", files={"file": ("r.png", b"not a png", "image/png")})

    app.dependency_overrides.clear()
    assert wrong_type.status_code == 415
    assert broken.status_code == 400
