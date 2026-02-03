import time
from typing import List
import httpx

from app.config import settings
from app.logger import setup_logger


logger = setup_logger(__name__)


def _base_url() -> str:
    return settings.OLLAMA_BASE_URL.rstrip("/")


def embed_texts(texts: List[str]) -> List[List[float]]:
    start_time = time.time()
    logger.debug(f"[embed_texts] Embedding {len(texts)} texts")
    
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "input": texts,
        "keep_alive": "5m",
    }
    
    try:
        resp = httpx.post(f"{_base_url()}/api/embed", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        
        elapsed = time.time() - start_time
        logger.info(f"[embed_texts] Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        return embeddings
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[embed_texts] Failed after {elapsed:.2f}s: {e}")
        raise


def embed_query(text: str) -> List[float]:
    logger.debug(f"[embed_query] Embedding query: {text[:100]}...")
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []
