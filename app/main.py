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
from app.agent import classify_query_scope
from app.graph import get_graph
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

    graph = get_graph()
    initial_state = {"query": request.query, "session": session, "k": k}
    final_state = None
    async for state in graph.astream(initial_state, stream_mode="values"):
        final_state = state
        logger.debug(f"[/query] Graph state keys: {list(state.keys())}")

    answer = (final_state or {}).get("final_answer", "")
    
    elapsed = time.time() - start_time
    logger.info(f"[/query] Completed in {elapsed:.2f}s, answer length: {len(answer)} chars")
    
    return QueryResponse(answer=answer)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
