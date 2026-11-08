"""Documents repository: the image as uploaded, what the model returned, and corrections.

Two tables. `documents` keeps one row per document and never changes after it is
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

from midp.db import CorrectionRow, DocumentRow, ExampleRow


@dataclass(frozen=True)
class StoredDocument:
    id: str
    created_at: str
    model: str
    seconds: float
    result: dict
    corrections: list[dict]


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, *, image: bytes, content_type: str, model: str, seconds: float, result: dict[str, Any]) -> str:
        document_id = uuid.uuid4().hex
        self._session.add(
            DocumentRow(
                id=document_id,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                model=model,
                seconds=seconds,
                content_type=content_type,
                image=image,
                result=result,
                needs_review=bool(result.get("needs_review", False)),
            )
        )
        self._session.commit()
        return document_id

    def pending_review(self) -> list[DocumentRow]:
        """Documents marked for review that nobody has reviewed yet, oldest first."""
        return list(
            self._session.exec(
                select(DocumentRow)
                .where(DocumentRow.needs_review.is_(True), DocumentRow.reviewed_at.is_(None))
                .order_by(DocumentRow.created_at)
            ).all()
        )

    def mark_reviewed(self, document_id: str, decision: str) -> str | None:
        """Close the review with a decision ("APPROVED" or "REJECTED"). Returns when it was reviewed, or None if there is no such document."""
        row = self._session.get(DocumentRow, document_id)
        if row is None:
            return None
        row.reviewed_at = datetime.now(UTC).isoformat(timespec="seconds")
        row.review_decision = decision
        self._session.commit()
        return row.reviewed_at

    def exists(self, document_id: str) -> bool:
        return self._session.get(DocumentRow, document_id) is not None

    def add_correction(self, document_id: str, fields: dict[str, Any]) -> None:
        self._session.add(
            CorrectionRow(
                document_id=document_id,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                fields=fields,
            )
        )
        self._session.commit()

    def get(self, document_id: str) -> StoredDocument | None:
        row = self._session.get(DocumentRow, document_id)
        if row is None:
            return None
        corrections = self._session.exec(
            select(CorrectionRow).where(CorrectionRow.document_id == document_id).order_by(CorrectionRow.id)
        ).all()
        return StoredDocument(
            id=row.id,
            created_at=row.created_at,
            model=row.model,
            seconds=row.seconds,
            result=row.result,
            corrections=[c.fields for c in corrections],
        )


class ExampleRepository:
    """Typed examples and their embeddings, searched by cosine similarity with pgvector."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, document_type: str, text: str, embedding: list[float]) -> int:
        row = ExampleRow(document_type=document_type, text=text, embedding=embedding)
        self._session.add(row)
        self._session.commit()
        return row.id

    def nearest(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        """The k examples most similar to the vector, as (document_type, cosine similarity), most similar first."""
        distance = ExampleRow.embedding.cosine_distance(embedding)
        rows = self._session.exec(select(ExampleRow.document_type, distance).order_by(distance).limit(k)).all()
        return [(document_type, 1.0 - float(d)) for document_type, d in rows]


def apply_corrections(result: dict, corrections: list[dict]) -> dict:
    """The model's answer with every correction applied in order.

    A correction changes only the fields it names; a later correction to the
    same field wins. The stored answer and the corrections are not changed.
    """
    final = dict(result)
    for correction in corrections:
        final.update(correction)
    return final
