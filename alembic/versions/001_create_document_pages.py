"""create document_pages table

Revision ID: 001_create_document_pages
Revises: 
Create Date: 2026-01-28

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "001_create_document_pages"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_pages",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("source_file", sa.String(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("company_name", sa.String(), nullable=True),
        sa.Column("doc_type", sa.String(), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("fiscal_quarter", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
    )
    op.create_index("ix_document_pages_file_hash", "document_pages", ["file_hash"])
    op.create_index("ix_document_pages_company_name", "document_pages", ["company_name"])
    op.create_index("ix_document_pages_doc_type", "document_pages", ["doc_type"])
    op.create_index("ix_document_pages_fiscal_year", "document_pages", ["fiscal_year"])
    op.create_index("ix_document_pages_fiscal_quarter", "document_pages", ["fiscal_quarter"])

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_pages_embedding "
        "ON document_pages USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_pages_embedding")
    op.drop_index("ix_document_pages_fiscal_quarter", table_name="document_pages")
    op.drop_index("ix_document_pages_fiscal_year", table_name="document_pages")
    op.drop_index("ix_document_pages_doc_type", table_name="document_pages")
    op.drop_index("ix_document_pages_company_name", table_name="document_pages")
    op.drop_index("ix_document_pages_file_hash", table_name="document_pages")
    op.drop_table("document_pages")
