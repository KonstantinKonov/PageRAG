import os
import re
import time
import numpy as np
from typing import List, Dict, Any

from rank_bm25 import BM25Plus
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from app.config import settings
from app.models import DocumentPage
from app.ollama_embed import embed_query
from app.logger import setup_logger


logger = setup_logger(__name__)


def extract_headings_with_content(text: str) -> List[str]:
    chunks = []
    sections = text.split("\n\n")
    i = 0
    while i < len(sections):
        section = sections[i].strip()
        if re.match(r"^#+\s+", section):
            heading = section
            if i + 1 < len(sections):
                next_content = sections[i + 1].strip()
                chunk = f"{heading}\n\n{next_content}"
                i += 2
            else:
                chunk = heading
                i += 1
            chunks.append(chunk)
        else:
            i += 1
    return chunks


def rank_documents_by_keywords(
    docs: List[DocumentPage], keywords: List[str], k: int
) -> List[DocumentPage]:
    if not docs or not keywords:
        return docs

    query_tokens = " ".join(keywords).lower().split(" ")
    doc_chunks = []
    for doc in docs:
        chunks = extract_headings_with_content(doc.content)
        combined = " ".join(chunks) if chunks else doc.content
        doc_chunks.append(combined.lower().split(" "))

    bm25 = BM25Plus(doc_chunks)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked_indices[:k]]


def mmr(
    query_vec: List[float],
    doc_vecs: List[List[float]],
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    if not doc_vecs:
        return []
    return maximal_marginal_relevance(
        query_embedding=np.array(query_vec, dtype=np.float32),
        embedding_list=doc_vecs,
        k=k,
        lambda_mult=lambda_mult,
    )


async def build_filters(
    filters: Dict[str, Any], ranking_keywords: List[str]
) -> List[Any]:
    logger.debug(f"[build_filters] Input filters: {filters}, keywords: {ranking_keywords}")
    
    conditions = []
    if filters:
        if filters.get("company_name"):
            conditions.append(DocumentPage.company_name == filters["company_name"])
        if filters.get("doc_type"):
            conditions.append(DocumentPage.doc_type == filters["doc_type"])
        if filters.get("fiscal_year"):
            conditions.append(DocumentPage.fiscal_year == filters["fiscal_year"])
        if filters.get("fiscal_quarter"):
            conditions.append(DocumentPage.fiscal_quarter == filters["fiscal_quarter"])

    if ranking_keywords:
        keyword_conditions = [
            DocumentPage.content.ilike(f"%{kw}%") for kw in ranking_keywords
        ]
        conditions.append(or_(*keyword_conditions))

    logger.debug(f"[build_filters] Built {len(conditions)} SQL conditions")
    return conditions


async def search_docs(
    session: AsyncSession,
    query: str,
    filters: Dict[str, Any],
    ranking_keywords: List[str],
    k: int,
    fetch_k: int,
) -> List[DocumentPage]:
    start_time = time.time()
    logger.debug(f"[search_docs] Query: {query}, k={k}, fetch_k={fetch_k}")
    
    embed_start = time.time()
    query_vec = embed_query(query)
    embed_elapsed = time.time() - embed_start
    logger.debug(f"[search_docs] Query embedding took {embed_elapsed:.2f}s")

    conditions = await build_filters(filters, ranking_keywords)
    stmt = select(DocumentPage).where(*conditions).order_by(
        DocumentPage.embedding.cosine_distance(query_vec)
    ).limit(fetch_k)

    db_start = time.time()
    rows = (await session.execute(stmt)).scalars().all()
    db_elapsed = time.time() - db_start
    logger.info(f"[search_docs] DB query returned {len(rows)} rows in {db_elapsed:.2f}s")
    
    if not rows:
        logger.warning("[search_docs] No documents found matching filters/keywords")
        return []

    mmr_start = time.time()
    doc_vecs = [doc.embedding for doc in rows]
    selected = mmr(query_vec, doc_vecs, k=k)
    mmr_docs = [rows[i] for i in selected]
    mmr_elapsed = time.time() - mmr_start
    logger.debug(f"[search_docs] MMR selected {len(mmr_docs)} docs in {mmr_elapsed:.2f}s")

    rerank_start = time.time()
    reranked = rank_documents_by_keywords(mmr_docs, ranking_keywords, k=k)
    rerank_elapsed = time.time() - rerank_start
    logger.debug(f"[search_docs] BM25 reranking took {rerank_elapsed:.2f}s")
    
    total_elapsed = time.time() - start_time
    logger.info(f"[search_docs] Total search took {total_elapsed:.2f}s, returning {len(reranked)} docs")
    
    return reranked


def write_debug_log(docs: List[DocumentPage]) -> None:
    os.makedirs(settings.DEBUG_LOG_DIR, exist_ok=True)
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"--- Document {i} ---")
        lines.append(f"company_name: {doc.company_name}")
        lines.append(f"doc_type: {doc.doc_type}")
        lines.append(f"fiscal_year: {doc.fiscal_year}")
        lines.append(f"fiscal_quarter: {doc.fiscal_quarter}")
        lines.append(f"page: {doc.page}")
        lines.append(f"source_file: {doc.source_file}")
        lines.append(f"file_hash: {doc.file_hash}")
        lines.append("")
        lines.append("Content:")
        lines.append(doc.content)
        lines.append("")

    log_path = os.path.join(settings.DEBUG_LOG_DIR, "retrieved_reranked_docs.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
