from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from midp.api import app, get_predictor, get_repository
from midp.repository import ReceiptRepository


class FixedPredictor:
    name = "fixed"
    model_name = "test-model"

    def predict(self, image):
        return {"lines": [], "total": Decimal(61799), "discount": Decimal(-19000)}


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path: Path):
    app.dependency_overrides[get_predictor] = lambda: FixedPredictor()
    app.dependency_overrides[get_repository] = lambda: ReceiptRepository(tmp_path / "test.sqlite3")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_extract_returns_id_and_fields_as_json_numbers(client: TestClient) -> None:
    response = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == {"lines": [], "total": 61799, "discount": -19000}
    assert len(body["id"]) == 32


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


def test_config_is_found_whatever_the_working_directory(tmp_path, monkeypatch) -> None:
    from midp.api import CONFIG_FILE

    monkeypatch.chdir(tmp_path)

    assert CONFIG_FILE.is_file()


def test_a_correction_is_kept_apart_from_the_original_answer(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()

    corrected = client.post(f"/receipts/{created['id']}/corrections", json={"total": 61800})
    body = client.get(f"/receipts/{created['id']}").json()

    assert corrected.status_code == 200
    assert body["result"] == created["result"]
    assert body["correction"] == {"total": 61800}


def test_a_later_correction_replaces_the_one_shown(client: TestClient) -> None:
    created = client.post("/receipts/extract", files={"file": ("r.png", _png(), "image/png")}).json()
    client.post(f"/receipts/{created['id']}/corrections", json={"total": 1})
    client.post(f"/receipts/{created['id']}/corrections", json={"total": 2})

    assert client.get(f"/receipts/{created['id']}").json()["correction"] == {"total": 2}


def test_correcting_an_unknown_receipt_is_not_found(client: TestClient) -> None:
    response = client.post("/receipts/does-not-exist/corrections", json={"total": 1})

    assert response.status_code == 404
