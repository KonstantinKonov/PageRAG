import hashlib
import os
import time
from pathlib import Path
from typing import List, Tuple

from docling.document_converter import DocumentConverter
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import DocumentPage
from app.ollama_embed import embed_texts
from app.logger import setup_logger


logger = setup_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_metadata_from_filename(filename: str) -> dict:
    name = filename.replace(".pdf", "")
    parts = name.split()

    metadata = {}
    quarter = None
    year = None
    if len(parts) >= 4 and parts[2].lower().startswith("q"):
        quarter = parts[2].lower()
        if parts[3].isdigit():
            year = int(parts[3])
    elif len(parts) >= 3 and parts[2].isdigit():
        year = int(parts[2])
    elif len(parts) >= 4 and parts[3].isdigit():
        year = int(parts[3])

    metadata["fiscal_quarter"] = quarter
    metadata["fiscal_year"] = year

    metadata["company_name"] = parts[0] if len(parts) > 0 else None
    metadata["doc_type"] = parts[1] if len(parts) > 1 else None

    return metadata


def _extract_pdf_pages_docling(pdf_path: str) -> List[str]:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    page_break = "<!-- page break -->"
    markdown_text = result.document.export_to_markdown(
        page_break_placeholder=page_break
    )
    pages = markdown_text.split(page_break)
    return pages


def _extract_pdf_pages_pypdf(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def extract_pdf_pages(pdf_path: str) -> List[str]:
    try:
        return _extract_pdf_pages_docling(pdf_path)
    except Exception:
        # Fallback for offline environments without model downloads.
        return _extract_pdf_pages_pypdf(pdf_path)


async def ingest_pdf_file(
    session: AsyncSession, pdf_path: str
) -> Tuple[bool, str]:
    start_time = time.time()
    logger.info(f"[ingest_pdf_file] Starting ingestion for: {pdf_path}")
    
    file_hash = compute_file_hash(pdf_path)
    logger.debug(f"[ingest_pdf_file] Computed file hash: {file_hash}")
    
    existing = await session.execute(
        select(DocumentPage.id).where(DocumentPage.file_hash == file_hash)
    )
    if existing.scalar_one_or_none():
        logger.info(f"[ingest_pdf_file] File already exists in DB: {file_hash}")
        return False, file_hash

    file_metadata = extract_metadata_from_filename(Path(pdf_path).name)
    logger.info(f"[ingest_pdf_file] Extracted metadata: {file_metadata}")

    extract_start = time.time()
    pages = extract_pdf_pages(pdf_path)
    extract_elapsed = time.time() - extract_start
    logger.info(f"[ingest_pdf_file] Extracted {len(pages)} pages in {extract_elapsed:.2f}s")

    embed_start = time.time()
    page_embeddings = embed_texts(pages)
    embed_elapsed = time.time() - embed_start
    logger.info(f"[ingest_pdf_file] Generated embeddings for {len(pages)} pages in {embed_elapsed:.2f}s")

    for page_num, page_text in enumerate(pages, start=1):
        metadata = dict(file_metadata)
        doc = DocumentPage(
            file_hash=file_hash,
            source_file=Path(pdf_path).name,
            page=page_num,
            company_name=metadata.get("company_name"),
            doc_type=metadata.get("doc_type"),
            fiscal_year=metadata.get("fiscal_year"),
            fiscal_quarter=metadata.get("fiscal_quarter"),
            content=page_text,
            embedding=page_embeddings[page_num - 1],
        )
        session.add(doc)

    commit_start = time.time()
    await session.commit()
    commit_elapsed = time.time() - commit_start
    logger.debug(f"[ingest_pdf_file] DB commit took {commit_elapsed:.2f}s")
    
    total_elapsed = time.time() - start_time
    logger.info(f"[ingest_pdf_file] Successfully ingested {len(pages)} pages in {total_elapsed:.2f}s")
    
    return True, file_hash


def ensure_upload_dir() -> None:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
