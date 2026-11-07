"""Receipts repository: the image as uploaded, what the model returned, and corrections.

Two tables. `receipts` keeps one row per receipt and never changes after it is
saved. `corrections` keeps one row per correction, so the history is kept.

The repository works on a session it is given, as orkid's repositories do.
Writes commit on that session. The caller decides which session it is: the
app gives one per request, and the tests give one inside a transaction that
is rolled back at the end.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from midp.db import CorrectionRow, ReceiptRow


@dataclass(frozen=True)
class StoredReceipt:
    id: str
    created_at: str
    model: str
    seconds: float
    result: dict
    corrections: list[dict]


class ReceiptRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, *, image: bytes, content_type: str, model: str, seconds: float, result: dict[str, Any]) -> str:
        receipt_id = uuid.uuid4().hex
        self._session.add(
            ReceiptRow(
                id=receipt_id,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                model=model,
                seconds=seconds,
                content_type=content_type,
                image=image,
                result=result,
            )
        )
        self._session.commit()
        return receipt_id

    def exists(self, receipt_id: str) -> bool:
        return self._session.get(ReceiptRow, receipt_id) is not None

    def add_correction(self, receipt_id: str, fields: dict[str, Any]) -> None:
        self._session.add(
            CorrectionRow(
                receipt_id=receipt_id,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                fields=fields,
            )
        )
        self._session.commit()

    def get(self, receipt_id: str) -> StoredReceipt | None:
        row = self._session.get(ReceiptRow, receipt_id)
        if row is None:
            return None
        corrections = self._session.exec(
            select(CorrectionRow).where(CorrectionRow.receipt_id == receipt_id).order_by(CorrectionRow.id)
        ).all()
        return StoredReceipt(
            id=row.id,
            created_at=row.created_at,
            model=row.model,
            seconds=row.seconds,
            result=row.result,
            corrections=[c.fields for c in corrections],
        )


def apply_corrections(result: dict, corrections: list[dict]) -> dict:
    """The model's answer with every correction applied in order.

    A correction changes only the fields it names; a later correction to the
    same field wins. The stored answer and the corrections are not changed.
    """
    final = dict(result)
    for correction in corrections:
        final.update(correction)
    return final
