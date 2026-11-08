"""Rename receipts to documents: the table and the corrections' foreign key column.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("receipts", "documents")
    op.alter_column("corrections", "receipt_id", new_column_name="document_id")
    op.drop_index(op.f("ix_corrections_receipt_id"), table_name="corrections")
    op.create_index(op.f("ix_corrections_document_id"), "corrections", ["document_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_corrections_document_id"), table_name="corrections")
    op.create_index(op.f("ix_corrections_receipt_id"), "corrections", ["receipt_id"], unique=False)
    op.alter_column("corrections", "document_id", new_column_name="receipt_id")
    op.rename_table("documents", "receipts")
