"""End to end: an uploaded image goes through the API and ends up in Postgres.

The model is replaced by fixed answers, and the session the API uses is the
test's session inside its transaction, so nothing from this test survives.
"""

from fastapi.testclient import TestClient
from sqlmodel import Session

from midp.api import app, get_classifier, get_predictors, get_session
from midp.db import DocumentRow
from tests.fakes import FixedClassifier, FixedPredictor, png_bytes


def test_an_uploaded_document_is_stored_in_postgres(session: Session) -> None:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_classifier] = lambda: FixedClassifier("receipt")
    app.dependency_overrides[get_predictors] = lambda: {"receipt": FixedPredictor()}
    try:
        response = TestClient(app).post("/documents/extract", files={"file": ("r.png", png_bytes(), "image/png")})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    document_id = response.json()["id"]
    stored = session.get(DocumentRow, document_id)
    assert stored is not None
    assert stored.model == "test-model"
    assert stored.result["total"] == 61799
    assert stored.result["document_type"] == "receipt"
