import time
from typing import Any, Dict, List

import requests

from app.config import settings
from app.logger import setup_logger


logger = setup_logger(__name__)


def _extract_results(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    results = payload.get("results")
    return results if isinstance(results, list) else []


def web_search(query: str) -> str:
    start_time = time.time()
    if not settings.WEB_SEARCH_ENDPOINT or not settings.WEB_SEARCH_API_KEY:
        logger.warning("[web_search] Missing endpoint or API key, skipping web search")
        return ""

    headers = {
        "Authorization": f"Bearer {settings.WEB_SEARCH_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "query": query,
        "search_depth": "advanced",
        "max_results": settings.WEB_SEARCH_MAX_RESULTS,
    }

    try:
        response = requests.post(
            settings.WEB_SEARCH_ENDPOINT,
            headers=headers,
            json=body,
            timeout=settings.WEB_SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        results = _extract_results(payload)

        snippets = []
        for item in results[: settings.WEB_SEARCH_MAX_RESULTS]:
            title = item.get("title") or ""
            snippet = item.get("content") or ""
            link = item.get("url") or ""
            if not (title or snippet):
                continue
            snippets.append(f"- {title}\n{snippet}\n{link}".strip())

        elapsed = time.time() - start_time
        logger.info(f"[web_search] Retrieved {len(snippets)} results in {elapsed:.2f}s")
        return "\n\n".join(snippets)
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"[web_search] Failed in {elapsed:.2f}s: {e}")
        return ""
