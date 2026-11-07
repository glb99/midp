"""The receipts repository against Postgres, with the schema from the Alembic migrations."""

from sqlmodel import Session

from midp.repository import ReceiptRepository


def test_a_saved_receipt_and_its_correction_round_trip(session: Session) -> None:
    repository = ReceiptRepository(session)
    receipt_id = repository.save(
        image=b"\x89PNG-test",
        content_type="image/png",
        model="test-model",
        seconds=1.5,
        result={"document_type": "receipt", "total": 61799},
    )
    repository.add_correction(receipt_id, {"total": 61800})

    stored = repository.get(receipt_id)

    assert stored is not None
    assert stored.model == "test-model"
    assert stored.result["total"] == 61799
    assert stored.corrections == [{"total": 61800}]
    assert repository.exists(receipt_id)


def test_an_unknown_receipt_is_not_there(session: Session) -> None:
    repository = ReceiptRepository(session)

    assert repository.get("no-such-id") is None
    assert repository.exists("no-such-id") is False
