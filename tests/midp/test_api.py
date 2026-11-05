from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from midp.api import app, get_classifier, get_predictors, get_repository
from midp.classifier import Classification
from midp.repository import ReceiptRepository


class FixedPredictor:
    name = "fixed"
    model_name = "test-model"

    def predict(self, image):
        return {"lines": [], "total": Decimal(61799), "discount": Decimal(-19000)}


class FixedClassifier:
    def __init__(self, document_type):
        self._classification = Classification(document_type=document_type, confidence=0.95)

    def classify(self, image):
        return self._classification


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path: Path):
    app.dependency_overrides[get_classifier] = lambda: FixedClassifier("receipt")
    app.dependency_overrides[get_predictors] = lambda: {
        name: FixedPredictor() for name in ("receipt", "invoice", "delivery_note")
    }
    app.dependency_overrides[get_repository] = lambda: ReceiptRepository(tmp_path / "test.sqlite3")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_extract_classifies_then_returns_the_type_and_its_fields(client: TestClient) -> None:
    response = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == {
        "document_type": "receipt",
        "confidence": 0.95,
        "lines": [],
        "total": 61799,
        "discount": -19000,
    }
    assert len(body["id"]) == 32


def test_an_image_of_an_unknown_type_is_stored_without_fields(tmp_path: Path) -> None:
    app.dependency_overrides[get_classifier] = lambda: FixedClassifier(None)
    app.dependency_overrides[get_repository] = lambda: ReceiptRepository(tmp_path / "unknown.sqlite3")
    client = TestClient(app)

    response = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["result"] == {"document_type": None, "confidence": 0.95}


def test_a_stored_receipt_can_be_read_again(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    response = client.get(f"/receipts/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == created["result"]
    assert body["model"] == "test-model"
    assert "created_at" in body and "seconds" in body


def test_unknown_receipt_is_not_found(client: TestClient) -> None:
    assert client.get("/receipts/does-not-exist").status_code == 404


def test_extract_refuses_files_that_are_not_images(client: TestClient) -> None:
    wrong_type = client.post("/receipts/extract", files={"file": ("r.pdf", b"%PDF", "application/pdf")})
    broken = client.post("/receipts/extract", files={"file": ("r.png", b"not a png", "image/png")})

    assert wrong_type.status_code == 415
    assert broken.status_code == 400


def test_a_correction_is_kept_apart_from_the_original_answer(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    corrected = client.post(f"/receipts/{created['id']}/corrections", json={"total": 61800})
    body = client.get(f"/receipts/{created['id']}").json()

    assert corrected.status_code == 200
    assert body["result"] == created["result"]
    assert body["corrections"] == [{"total": 61800}]
    assert body["final"]["total"] == 61800
    assert body["final"]["lines"] == created["result"]["lines"]


def test_a_later_correction_replaces_the_one_shown(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()
    client.post(f"/receipts/{created['id']}/corrections", json={"total": 1})
    client.post(f"/receipts/{created['id']}/corrections", json={"total": 2})

    body = client.get(f"/receipts/{created['id']}").json()
    assert body["corrections"] == [{"total": 1}, {"total": 2}]
    assert body["final"]["total"] == 2


def test_correcting_an_unknown_receipt_is_not_found(client: TestClient) -> None:
    response = client.post("/receipts/does-not-exist/corrections", json={"total": 1})

    assert response.status_code == 404


def test_a_correction_with_an_unknown_field_is_refused(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    response = client.post(f"/receipts/{created['id']}/corrections", json={"banana": 3})

    assert response.status_code == 422
    assert "unknown field: banana" in response.json()["detail"]


def test_a_correction_with_a_text_amount_is_refused(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    response = client.post(f"/receipts/{created['id']}/corrections", json={"total": "1.591.600"})

    assert response.status_code == 422


def test_a_correction_to_a_line_is_checked_too(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    good = client.post(
        f"/receipts/{created['id']}/corrections",
        json={"lines": [{"description": "Nasi", "quantity": 1, "amount": 75000}]},
    )
    bad = client.post(
        f"/receipts/{created['id']}/corrections",
        json={"lines": [{"description": "Nasi", "amount": "75.000"}]},
    )

    assert good.status_code == 200
    assert bad.status_code == 422


def test_config_is_found_whatever_the_working_directory(tmp_path, monkeypatch) -> None:
    from midp.api import TYPES_FILE

    monkeypatch.chdir(tmp_path)

    assert TYPES_FILE.is_file()
