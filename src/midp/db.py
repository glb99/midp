"""Database tables for receipts and their corrections, shared by the repository and Alembic.

The schema is changed only through Alembic migrations (see alembic/). The
tests create the tables directly on a throwaway SQLite file with create_all.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Column, LargeBinary
from sqlmodel import Field, SQLModel, create_engine


class ReceiptRow(SQLModel, table=True):
    __tablename__ = "receipts"

    id: str = Field(primary_key=True)
    created_at: str
    model: str
    seconds: float
    content_type: str
    image: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    result: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class CorrectionRow(SQLModel, table=True):
    __tablename__ = "corrections"

    id: int | None = Field(default=None, primary_key=True)
    receipt_id: str = Field(foreign_key="receipts.id", index=True)
    created_at: str
    fields: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


def make_engine(url: str):
    return create_engine(url, pool_pre_ping=True)
