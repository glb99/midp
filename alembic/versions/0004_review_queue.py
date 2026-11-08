"""The review queue: documents flagged for review, and when a person reviewed them.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("documents", sa.Column("reviewed_at", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    # documents stored before this migration keep their flag from the result they were saved with
    op.execute("UPDATE documents SET needs_review = COALESCE((result->>'needs_review')::boolean, false)")
    op.create_index(op.f("ix_documents_needs_review"), "documents", ["needs_review"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_needs_review"), table_name="documents")
    op.drop_column("documents", "reviewed_at")
    op.drop_column("documents", "needs_review")
