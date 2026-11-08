"""The documents repository against Postgres, with the schema from the Alembic migrations."""

from sqlmodel import Session

from midp.repository import DocumentRepository


def test_a_saved_document_and_its_correction_round_trip(session: Session) -> None:
    repository = DocumentRepository(session)
    document_id = repository.save(
        image=b"\x89PNG-test",
        content_type="image/png",
        model="test-model",
        seconds=1.5,
        result={"document_type": "receipt", "total": 61799},
    )
    repository.add_correction(document_id, {"total": 61800})

    stored = repository.get(document_id)

    assert stored is not None
    assert stored.model == "test-model"
    assert stored.result["total"] == 61799
    assert stored.corrections == [{"total": 61800}]
    assert repository.exists(document_id)


def test_a_document_needing_review_leaves_the_queue_when_reviewed(session: Session) -> None:
    repository = DocumentRepository(session)
    flagged = repository.save(
        image=b"\x89PNG-test",
        content_type="image/png",
        model="test-model",
        seconds=1.0,
        result={"document_type": "invoice", "needs_review": True, "problems": ["missing tax"]},
    )
    repository.save(
        image=b"\x89PNG-test",
        content_type="image/png",
        model="test-model",
        seconds=1.0,
        result={"document_type": "receipt", "needs_review": False, "problems": []},
    )

    assert [row.id for row in repository.pending_review()] == [flagged]
    assert repository.mark_reviewed(flagged) is not None
    assert repository.pending_review() == []
    assert repository.mark_reviewed("no-such-id") is None


def test_an_unknown_document_is_not_there(session: Session) -> None:
    repository = DocumentRepository(session)

    assert repository.get("no-such-id") is None
    assert repository.exists("no-such-id") is False
