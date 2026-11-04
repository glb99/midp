"""SQLite repository for receipts: the image as uploaded, what the model returned,
and any human corrections.

Two tables. `receipts` keeps one row per receipt and never changes after
it is saved. `corrections` keeps one row per correction, so the history
is kept. Both are created on start-up, so there are no migrations yet.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    model        TEXT NOT NULL,
    seconds      REAL NOT NULL,
    content_type TEXT NOT NULL,
    image        BLOB NOT NULL,
    result       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id  TEXT NOT NULL REFERENCES receipts(id),
    created_at  TEXT NOT NULL,
    fields      TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StoredReceipt:
    id: str
    created_at: str
    model: str
    seconds: float
    result: dict
    corrections: list[dict]


class ReceiptRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def save(self, *, image: bytes, content_type: str, model: str, seconds: float, result: dict[str, Any]) -> str:
        receipt_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as db:
            db.execute(
                "INSERT INTO receipts (id, created_at, model, seconds, content_type, image, result)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (receipt_id, created_at, model, seconds, content_type, image,
                 json.dumps(result, ensure_ascii=False, default=str)),
            )
        return receipt_id

    def exists(self, receipt_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM receipts WHERE id = ?", (receipt_id,)).fetchone() is not None

    def add_correction(self, receipt_id: str, fields: dict[str, Any]) -> None:
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as db:
            db.execute(
                "INSERT INTO corrections (receipt_id, created_at, fields) VALUES (?, ?, ?)",
                (receipt_id, created_at, json.dumps(fields, ensure_ascii=False, default=str)),
            )

    def get(self, receipt_id: str) -> StoredReceipt | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, created_at, model, seconds, result FROM receipts WHERE id = ?",
                (receipt_id,),
            ).fetchone()
            if row is None:
                return None
            corrections = db.execute(
                "SELECT fields FROM corrections WHERE receipt_id = ? ORDER BY id",
                (receipt_id,),
            ).fetchall()
        return StoredReceipt(
            id=row[0],
            created_at=row[1],
            model=row[2],
            seconds=row[3],
            result=json.loads(row[4]),
            corrections=[json.loads(c[0]) for c in corrections],
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
