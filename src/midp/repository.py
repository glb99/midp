"""SQLite repository for receipts: the image as uploaded, and what the model returned.

One table, one row per receipt. The result is kept as JSON text so the
stored answer is exactly what the model produced, even if the field
configuration changes later.
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
)
"""


@dataclass(frozen=True)
class StoredReceipt:
    id: str
    created_at: str
    model: str
    seconds: float
    result: dict


class ReceiptRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connect() as db:
            db.execute(SCHEMA)

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

    def get(self, receipt_id: str) -> StoredReceipt | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, created_at, model, seconds, result FROM receipts WHERE id = ?",
                (receipt_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredReceipt(id=row[0], created_at=row[1], model=row[2], seconds=row[3], result=json.loads(row[4]))
