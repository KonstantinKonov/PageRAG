import os
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import init_db_extensions, get_session
from app.ingest import ingest_pdf_file, ensure_upload_dir
from app.schemas import IngestResponse, QueryRequest, QueryResponse
from app.agent import (
    extract_filters,
    generate_ranking_keywords,
    decompose_query,
    generate_answer,
    classify_query_scope,
    grade_documents,
    rewrite_query,
)
from app.retrieval import search_docs, write_debug_log
from app.web_search import web_search
from app.logger import setup_logger


logger = setup_logger(__name__)
app = FastAPI(title=settings.APP_NAME)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db_extensions()
    ensure_upload_dir()
    os.makedirs(settings.DEBUG_LOG_DIR, exist_ok=True)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    start_time = time.time()
    logger.info(f"[/ingest] Starting ingest for {len(files)} files")
    
    ingested = []
    skipped = []
    for idx, uploaded_file in enumerate(files, 1):
        logger.info(f"[/ingest] Processing file {idx}/{len(files)}: {uploaded_file.filename}")
        
        if not uploaded_file.filename.lower().endswith(".pdf"):
            logger.error(f"[/ingest] Rejected non-PDF file: {uploaded_file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_path = Path(settings.UPLOAD_DIR) / uploaded_file.filename
        logger.debug(f"[/ingest] Saving to {file_path}")
        
        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())

        created, _ = await ingest_pdf_file(session, str(file_path))
        if created:
            logger.info(f"[/ingest] Successfully ingested: {uploaded_file.filename}")
            ingested.append(uploaded_file.filename)
        else:
            logger.warning(f"[/ingest] Skipped (already exists): {uploaded_file.filename}")
            skipped.append(uploaded_file.filename)

    elapsed = time.time() - start_time
    logger.info(f"[/ingest] Completed in {elapsed:.2f}s. Ingested: {len(ingested)}, Skipped: {len(skipped)}")
    
    return IngestResponse(ingested_files=ingested, skipped_files=skipped)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, session: AsyncSession = Depends(get_session)
):
    start_time = time.time()
    logger.info(f"[/query] Starting query: {request.query[:100]}... k={request.k}")
    
    k = request.k or settings.DEFAULT_TOP_K
    logger.debug(f"[/query] Using k={k} (default={settings.DEFAULT_TOP_K})")
    
    scope = classify_query_scope(request.query)
    if not scope.in_scope:
        logger.info(f"[/query] Out-of-scope query blocked: {scope.reason}")
        return QueryResponse(
            answer="Запрос не относится к финансовым данным из SEC-отчётности. "
            "Пожалуйста, уточните компанию, период и финансовую метрику."
        )

    queries = decompose_query(request.query)
    logger.info(f"[/query] Decomposed into {len(queries)} queries: {queries}")

    all_retrieved_text = []
    for idx, q in enumerate(queries, 1):
        logger.info(f"[/query] Processing sub-query {idx}/{len(queries)}: {q}")
        
        def format_docs(docs):
            doc_text = []
            for i, doc in enumerate(docs, 1):
                doc_text.append(f"--- Document {i} ---")
                doc_text.append(f"company_name: {doc.company_name}")
                doc_text.append(f"doc_type: {doc.doc_type}")
                doc_text.append(f"fiscal_year: {doc.fiscal_year}")
                doc_text.append(f"fiscal_quarter: {doc.fiscal_quarter}")
                doc_text.append(f"page: {doc.page}")
                doc_text.append(f"source_file: {doc.source_file}")
                doc_text.append(f"file_hash: {doc.file_hash}")
                doc_text.append("")
                doc_text.append("Content:")
                doc_text.append(doc.content)
                doc_text.append("")
            return "\n".join(doc_text)

        filters = extract_filters(q)
        logger.debug(f"[/query] Extracted filters: {filters}")
        
        keywords = generate_ranking_keywords(q)
        logger.debug(f"[/query] Generated keywords: {keywords}")
        
        docs = await search_docs(
            session,
            q,
            filters=filters,
            ranking_keywords=keywords,
            k=k,
            fetch_k=settings.DEFAULT_FETCH_K,
        )
        logger.info(f"[/query] Retrieved {len(docs)} documents for sub-query {idx}")
        
        write_debug_log(docs)
        chunk_text = format_docs(docs)
        logger.debug(f"[/query] Retrieved chunk text for sub-query {idx}:\n{chunk_text}")

        if docs:
            grade = grade_documents(q, chunk_text)
            is_relevant = grade.is_relevant
        else:
            logger.info(f"[/query] No documents retrieved for sub-query {idx}")
            is_relevant = False

        if not is_relevant:
            rewritten = rewrite_query(q)
            logger.info(f"[/query] Rewritten sub-query {idx}: {rewritten}")

            rewrite_filters = extract_filters(rewritten)
            rewrite_keywords = generate_ranking_keywords(rewritten)

            docs = await search_docs(
                session,
                rewritten,
                filters=rewrite_filters,
                ranking_keywords=rewrite_keywords,
                k=k,
                fetch_k=settings.DEFAULT_FETCH_K,
            )
            logger.info(
                f"[/query] Retrieved {len(docs)} documents after rewrite for sub-query {idx}"
            )
            write_debug_log(docs)
            chunk_text = format_docs(docs)
            logger.debug(
                f"[/query] Retrieved chunk text after rewrite for sub-query {idx}:\n{chunk_text}"
            )

            if docs:
                grade = grade_documents(rewritten, chunk_text)
                is_relevant = grade.is_relevant
            else:
                is_relevant = False

            if not is_relevant:
                web_text = web_search(rewritten)
                if web_text:
                    logger.info(f"[/query] Using web search fallback for sub-query {idx}")
                    if chunk_text:
                        chunk_text = f"{chunk_text}\n\n[WEB_SEARCH]\n{web_text}"
                    else:
                        chunk_text = f"[WEB_SEARCH]\n{web_text}"
                else:
                    logger.info(
                        f"[/query] Web search fallback returned no results for sub-query {idx}"
                    )

        all_retrieved_text.append(chunk_text)

    combined_retrieved = "\n\n".join(all_retrieved_text)
    logger.debug(f"[/query] Combined retrieved text length: {len(combined_retrieved)} chars")
    logger.debug(f"[/query] Combined retrieved text:\n{combined_retrieved}")
    
    answer = generate_answer(request.query, combined_retrieved)
    
    elapsed = time.time() - start_time
    logger.info(f"[/query] Completed in {elapsed:.2f}s, answer length: {len(answer)} chars")
    
    return QueryResponse(answer=answer)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
