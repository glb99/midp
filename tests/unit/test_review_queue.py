from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from midp.api import app, get_classifier, get_predictors, get_repository
from midp.db import make_engine
from midp.repository import DocumentRepository
from tests.fakes import FixedClassifier, FixedPredictor, png_bytes


class InvoiceWithoutTax(FixedPredictor):
    def predict(self, image):
        return {"number": "F1", "date": "d", "supplier": "s", "lines": [{"description": "x", "quantity": 3, "amount": 144}], "subtotal": 144, "tax": None, "total": 144}


class ReceiptThatAddsUp(FixedPredictor):
    def predict(self, image):
        return {"lines": [], "total": Decimal(61799), "discount": Decimal(-19000)}


@pytest.fixture
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    app.dependency_overrides[get_repository] = lambda: DocumentRepository(session)
    yield TestClient(app)
    app.dependency_overrides.clear()
    session.close()


def _extract(client: TestClient, document_type: str, predictor) -> str:
    app.dependency_overrides[get_classifier] = lambda: FixedClassifier(document_type)
    app.dependency_overrides[get_predictors] = lambda: {document_type: predictor}
    response = client.post("/documents/extract", files={"file": ("r.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    return response.json()["id"]


def test_a_document_needing_review_is_listed_until_it_is_reviewed(client: TestClient) -> None:
    document_id = _extract(client, "invoice", InvoiceWithoutTax())

    queue = client.get("/documents/review").json()
    assert [item["id"] for item in queue] == [document_id]
    assert "missing tax" in queue[0]["problems"]

    reviewed = client.post(f"/documents/{document_id}/review")
    assert reviewed.status_code == 200
    assert client.get("/documents/review").json() == []


def test_a_document_that_passes_its_checks_is_not_queued(client: TestClient) -> None:
    _extract(client, "receipt", ReceiptThatAddsUp())

    assert client.get("/documents/review").json() == []


def test_reviewing_an_unknown_document_is_not_found(client: TestClient) -> None:
    assert client.post("/documents/does-not-exist/review").status_code == 404
