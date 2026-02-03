# PageRAG

FastAPI сервис для RAG по финансовой отчетности (10-K/10-Q/8-K) с векторным поиском в Postgres (pgvector), Ollama для LLM/эмбеддингов и corrective‑циклом (grade → rewrite → retry → web‑fallback).

## Возможности
- Загрузка PDF и нарезка на страницы/чанки с сохранением в Postgres
- Векторный поиск + MMR + BM25 переранжирование
- Corrective RAG: проверка релевантности, переписывание запроса, один ретрай
- Web‑fallback через Tavily API (если локальные документы нерелевантны)
- Markdown‑ответы на русском

## Архитектура
```mermaid
flowchart LR
    User[User] -->|"POST /query"| API[FastAPI]
    API --> Scope[ScopeCheck]
    Scope --> Decompose[DecomposeQuery]
    Decompose --> Retrieve[VectorSearch]
    Retrieve --> Grade[RelevanceGrader]
    Grade -->|Relevant| Answer[GenerateAnswer]
    Grade -->|NotRelevant| Rewrite[RewriteQuery]
    Rewrite --> Retry[VectorSearchRetry]
    Retry --> Grade2[RelevanceGrader]
    Grade2 -->|Relevant| Answer
    Grade2 -->|NotRelevant| Web[TavilySearch]
    Web --> Answer
    Answer --> Response[Response]

    User -->|"POST /ingest"| Ingest[IngestPDF]
    Ingest --> Pg[(Postgres+pgvector)]
    Retrieve --> Pg
    Retry --> Pg
```

## Быстрый старт
1) Подготовьте `.env` на основе `.example.env`
2) Запустите сервисы:
```
docker compose up --build
```
3) Проверьте здоровье:
```
curl http://localhost:8000/health
```

## Конфигурация
- Все переменные задаются в `.env`
- Основные:
  - `DATABASE_URL` — подключение к Postgres
  - `OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBED_MODEL`
  - `WEB_SEARCH_ENDPOINT`, `WEB_SEARCH_API_KEY` (Tavily)

## API
- `POST /ingest` — загрузка PDF (multipart/form-data, поле `files`)
- `POST /query` — запрос к RAG:
```
{
  "query": "Какая была выручка у Google в 2025 году?",
  "k": 5
}
```
- `GET /health` — healthcheck

## Логи
Логи пайплайна пишутся в `debug_logs/pipeline.log`.
