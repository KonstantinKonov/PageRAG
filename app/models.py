import uuid
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.db import Base
from app.config import settings


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String, nullable=False, index=True)
    source_file = Column(String, nullable=False)
    page = Column(Integer, nullable=False)

    company_name = Column(String, nullable=True, index=True)
    doc_type = Column(String, nullable=True, index=True)
    fiscal_year = Column(Integer, nullable=True, index=True)
    fiscal_quarter = Column(String, nullable=True, index=True)

    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=False)
