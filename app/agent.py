import time
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from app.config import settings
from app.llm_schemas import (
    ChunkMetadata,
    RankingKeywords,
    SearchQueries,
    QueryScope,
    GradeDecision,
    RewriteQuery,
    ReflexionAnswer,
)
from app.logger import setup_logger


logger = setup_logger(__name__)
llm = ChatOllama(
    model=settings.OLLAMA_LLM_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    reasoning=settings.OLLAMA_REASONING,
)


def extract_filters(user_query: str) -> Dict[str, Any]:
    start_time = time.time()
    logger.debug(f"[extract_filters] Input query: {user_query}")
    
    llm_structured = llm.with_structured_output(ChunkMetadata)
    prompt = f"""Извлеки метаданные из запроса. Для неупомянутых полей верни None.

    ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}

    СООТВЕТСТВИЯ КОМПАНИЙ:
    - Amazon/AMZN -> amazon
    - Google/Alphabet/GOOGL/GOOG -> google
    - Apple/AAPL -> apple
    - Microsoft/MSFT -> microsoft
    - Tesla/TSLA -> tesla
    - Nvidia/NVDA -> nvidia
    - Meta/Facebook/FB -> meta

    ТИП ДОКУМЕНТА:
    - Annual report -> 10-k
    - Quarterly report -> 10-q
    - Current report -> 8-k

    ПРИМЕРЫ:
    "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
    "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
    "Tesla profitability" -> {{"company_name": "tesla"}}

    Извлеки метаданные:
    """

    try:
        metadata = llm_structured.invoke(
            prompt,
            options={
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        result = metadata.model_dump(exclude_none=True)
        elapsed = time.time() - start_time
        logger.info(f"[extract_filters] Extracted filters in {elapsed:.2f}s: {result}")
        return result
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[extract_filters] Failed in {elapsed:.2f}s: {e}")
        return {}


def classify_query_scope(user_query: str) -> QueryScope:
    start_time = time.time()
    logger.debug(f"[classify_query_scope] Input query: {user_query}")

    llm_structured = llm.with_structured_output(QueryScope)
    prompt = f"""Определи, относится ли запрос к финансовым данным компаний из SEC-отчётности (10-K/10-Q/8-K).
Ответь ТОЛЬКО в JSON со схемой:
{{"in_scope": boolean, "reason": string|null}}

Примеры in_scope:
- "Выручка Google в 2024 году"
- "Amazon operating income Q3 2023"
- "cash flows from operating activities Microsoft 2022"

Примеры out_of_scope:
- "Привет как дела"
- "Погода в Москве"
- "Сколько будет 2+2"

ЗАПРОС: {user_query}
"""

    try:
        result = llm_structured.invoke(
            prompt,
            options={
                "num_predict": 64,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        elapsed = time.time() - start_time
        logger.info(
            f"[classify_query_scope] in_scope={result.in_scope} in {elapsed:.2f}s reason={result.reason}"
        )
        return result
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[classify_query_scope] Failed in {elapsed:.2f}s: {e}")
        return QueryScope(in_scope=True, reason="fallback_allow")


def generate_ranking_keywords(user_query: str) -> List[str]:
    start_time = time.time()
    logger.debug(f"[generate_ranking_keywords] Input query: {user_query}")
    
    prompt = f"""
    Сгенерируй РОВНО 5 финансовых ключевых фраз на терминологии отчётности SEC.

    ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}

    ИСПОЛЬЗУЙ ТОЧНЫЕ ТЕРМИНЫ ИЗ ОТЧЁТОВ 10-K/10-Q:

    ЗАГОЛОВКИ ОТЧЁТНОСТИ:
    "consolidated statements of operations", "consolidated balance sheets", "consolidated statements of cash flows", "consolidated statements of stockholders equity"

    ОТЧЁТ О ПРИБЫЛЯХ И УБЫТКАХ:
    "revenue", "net revenue", "cost of revenue", "gross profit", "operating income", "net income", "earnings per share"

    БАЛАНС:
    "total assets", "cash and cash equivalents", "total liabilities", "stockholders equity", "working capital", "long-term debt"

    ДЕНЕЖНЫЕ ПОТОКИ:
    "cash flows from operating activities", "net cash provided by operating activities", "cash flows from investing activities", "free cash flow", "capital expenditures"

    ПРАВИЛА:
    - Верни РОВНО 5 ключевых фраз
    - Используй точные формулировки из отчётности SEC
    - Соответствуй теме запроса (revenue -> термины выручки, cash -> термины cash flow)
    - Используй "cash flows" (во мн. числе) и "stockholders equity"

    ПРИМЕРЫ:
    "revenue analysis" -> ["revenue", "net revenue", "total revenue", "consolidated statements of operations", "net sales"]
    "cash flow performance" -> ["consolidated statements of cash flows", "cash flows from operating activities", "net cash provided by operating activities", "free cash flow", "operating activities"]
    "balance sheet strength" -> ["consolidated balance sheets", "total assets", "stockholders equity", "cash and cash equivalents", "long-term debt"]

    Сгенерируй РОВНО 5 ключевых фраз:
    """

    llm_structured = llm.with_structured_output(RankingKeywords)
    try:
        result = llm_structured.invoke(
            prompt,
            options={
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        elapsed = time.time() - start_time
        logger.info(f"[generate_ranking_keywords] Generated {len(result.keywords)} keywords in {elapsed:.2f}s: {result.keywords}")
        return result.keywords
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[generate_ranking_keywords] Failed in {elapsed:.2f}s: {e}")
        return []


def decompose_query(user_query: str) -> List[str]:
    start_time = time.time()
    logger.debug(f"[decompose_query] Input query: {user_query}")
    
    llm_structured = llm.with_structured_output(SearchQueries)
    system_prompt = """
    Ты редактор запросов, который разбивает сложные запросы на фокусированные поисковые запросы для векторного хранилища.

    СТРАТЕГИЯ ДЕКОМПОЗИЦИИ:
    Разбей исходный запрос на 1–3 конкретных и фокусированных запроса, где каждый запрос нацелен на:
    - Одну компанию
    - Конкретный период времени
    - Конкретную метрику или аспект

    ПРАВИЛА:
    - Раскрывай сокращения (например, "rev" -> "revenue", "GOOGL" -> "Google")
    - Делай каждый запрос самодостаточным и конкретным
    - Держи запросы краткими (5–10 слов)
    """

    messages = [SystemMessage(system_prompt), HumanMessage(f"Исходный запрос: {user_query}")]
    try:
        response = llm_structured.invoke(
            messages,
            options={
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        queries = response.search_queries[: settings.MAX_SUB_QUERIES]
        result = queries if queries else [user_query]
        elapsed = time.time() - start_time
        logger.info(f"[decompose_query] Decomposed into {len(result)} queries in {elapsed:.2f}s: {result}")
        return result
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[decompose_query] Failed in {elapsed:.2f}s, returning original query: {e}")
        return [user_query]


def generate_answer(user_query: str, retrieved_docs: str) -> str:
    start_time = time.time()
    logger.debug(f"[generate_answer] Input query: {user_query}, docs length: {len(retrieved_docs)} chars")
    
    system_prompt = f"""Ты финансовый аналитик. Отвечай строго на основе предоставленных документов.

    Требования:
    - Ответ на русском языке
    - Используй Markdown
    - Для сравнений выводи таблицы
    - Если данных нет, честно скажи об этом
    """

    user_prompt = f"Вопрос пользователя: {user_query}\n\nДокументы:\n{retrieved_docs}"
    logger.debug(f"[generate_answer] Prompt length: {len(user_prompt)} chars")
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    response = llm.invoke(
        messages,
        options={
            "num_predict": settings.OLLAMA_NUM_PREDICT,
            "temperature": 0.2,
            "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
        },
    )
    content = response.content or ""
    logger.debug(f"[generate_answer] Response content repr: {content!r}")
    logger.debug(
        f"[generate_answer] Response metadata: {getattr(response, 'response_metadata', {})}"
    )
    logger.debug(
        f"[generate_answer] Response additional_kwargs: {getattr(response, 'additional_kwargs', {})}"
    )
    logger.debug(f"[generate_answer] Response tool_calls: {getattr(response, 'tool_calls', None)}")
    logger.debug(
        f"[generate_answer] Response usage_metadata: {getattr(response, 'usage_metadata', {})}"
    )
    if not content.strip():
        logger.warning("[generate_answer] LLM returned empty content")
        return "Не удалось сформировать ответ. Попробуйте уточнить запрос."
    
    elapsed = time.time() - start_time
    logger.info(f"[generate_answer] Generated answer in {elapsed:.2f}s, length: {len(content)} chars")
    logger.debug(f"[generate_answer] Answer content:\n{content}")
    return content


def grade_documents(user_query: str, retrieved_docs: str) -> GradeDecision:
    start_time = time.time()
    logger.debug(
        f"[grade_documents] Input query: {user_query}, docs length: {len(retrieved_docs)} chars"
    )

    llm_structured = llm.with_structured_output(GradeDecision)
    prompt = f"""You are a document relevance grader.

TASK: Evaluate if the retrieved documents are relevant to answer the user's question.

USER QUESTION: {user_query}

RETRIEVED DOCUMENTS:
{retrieved_docs}

CRITERIA:
- is_relevant = True: Documents contain information that can answer the question
- is_relevant = False: Documents are completely irrelevant, off-topic, or empty

Respond ONLY in JSON with this schema:
{{"is_relevant": true/false, "reasoning": "brief explanation"}}
"""
    try:
        result = llm_structured.invoke(
            prompt,
            options={
                "num_predict": 128,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        elapsed = time.time() - start_time
        logger.info(
            f"[grade_documents] is_relevant={result.is_relevant} in {elapsed:.2f}s reason={result.reasoning}"
        )
        return result
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[grade_documents] Failed in {elapsed:.2f}s: {e}")
        return GradeDecision(is_relevant=True, reasoning="fallback_allow")


def rewrite_query(user_query: str) -> str:
    start_time = time.time()
    logger.debug(f"[rewrite_query] Input query: {user_query}")

    llm_structured = llm.with_structured_output(RewriteQuery)
    prompt = f"""You are a query rewriting expert.

TASK: Rewrite the user's question to make it more specific and targeted for document retrieval.

ORIGINAL QUESTION: {user_query}

INSTRUCTIONS:
- Make the query more specific with keywords
- Add relevant financial terms (revenue, profit, earnings, cash flow, etc.)
- Preserve the original intent
- Keep it concise (5-12 words)

Return ONLY in JSON with this schema:
{{"rewritten_query": "..."}}"""
    try:
        response = llm_structured.invoke(
            prompt,
            options={
                "num_predict": 64,
                "temperature": 0,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        rewritten = response.rewritten_query.strip()
        elapsed = time.time() - start_time
        logger.info(f"[rewrite_query] Rewritten in {elapsed:.2f}s: {rewritten}")
        return rewritten or user_query
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[rewrite_query] Failed in {elapsed:.2f}s: {e}")
        return user_query


def draft_reflexion_answer(user_query: str, retrieved_docs: str) -> ReflexionAnswer:
    start_time = time.time()
    logger.debug(
        f"[draft_reflexion_answer] Input query: {user_query}, docs length: {len(retrieved_docs)} chars"
    )

    llm_structured = llm.with_structured_output(ReflexionAnswer)
    system_prompt = """Ты финансовый аналитик.

ЗАДАЧА:
1) Дай подробный ответ (~200-300 слов) на вопрос.
2) Используй Markdown (заголовки, списки, таблицы при необходимости).
3) Добавь inline ссылки [1], [2] если в документах есть факты для цитирования, иначе укажи, какие данные нужны.
4) Проведи критическую рефлексию: что отсутствует и что лишнее.
5) Если данных недостаточно, сгенерируй 1-3 уточняющих поисковых запроса.

Верни JSON по схеме:
{
  "answer": "...",
  "reflection": {"missing": "...", "superfluous": "..."},
  "search_queries": ["..."],
  "is_complete": false
}
"""
    user_prompt = f"Вопрос: {user_query}\n\nДокументы:\n{retrieved_docs}"
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    try:
        response = llm_structured.invoke(
            messages,
            options={
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": 0.2,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        elapsed = time.time() - start_time
        logger.info(
            f"[draft_reflexion_answer] complete={response.is_complete} in {elapsed:.2f}s queries={len(response.search_queries)}"
        )
        return response
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[draft_reflexion_answer] Failed in {elapsed:.2f}s: {e}")
        return ReflexionAnswer(
            answer="Не удалось сформировать ответ. Попробуйте уточнить запрос.",
            reflection={"missing": "unknown", "superfluous": "unknown"},
            search_queries=[],
            is_complete=True,
        )


def revise_reflexion_answer(
    user_query: str, retrieved_docs: str, prior_answer: str
) -> ReflexionAnswer:
    start_time = time.time()
    logger.debug(
        f"[revise_reflexion_answer] Input query: {user_query}, docs length: {len(retrieved_docs)} chars"
    )

    llm_structured = llm.with_structured_output(ReflexionAnswer)
    system_prompt = """Ты финансовый аналитик.

ЗАДАЧА:
1) Перепиши ответ, используя новые документы.
2) Добавь inline ссылки [1], [2] и список источников, если есть данные.
3) Проведи рефлексию: что отсутствует и что лишнее.
4) Если данных всё еще недостаточно, сгенерируй 1-3 уточняющих поисковых запроса.

Если данных достаточно, верни is_complete=true и пустой список search_queries.

ВАЖНО: Верни СТРОГО JSON c ключами:
{
  "answer": "...",
  "reflection": {"missing": "...", "superfluous": "..."},
  "search_queries": ["..."],
  "is_complete": false
}
Никаких других ключей (например, response/reflections/redundant) не используй.
"""
    user_prompt = (
        f"Вопрос: {user_query}\n\n"
        f"Предыдущий ответ:\n{prior_answer}\n\n"
        f"Новые документы:\n{retrieved_docs}"
    )
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    try:
        response = llm_structured.invoke(
            messages,
            options={
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": 0.2,
                "num_ctx": settings.OLLAMA_CONTEXT_LENGTH,
            },
            format="json",
        )
        elapsed = time.time() - start_time
        logger.info(
            f"[revise_reflexion_answer] complete={response.is_complete} in {elapsed:.2f}s queries={len(response.search_queries)}"
        )
        return response
    except (OutputParserException, Exception) as e:
        elapsed = time.time() - start_time
        logger.warning(f"[revise_reflexion_answer] Failed in {elapsed:.2f}s: {e}")
        return ReflexionAnswer(
            answer=prior_answer,
            reflection={"missing": "unknown", "superfluous": "unknown"},
            search_queries=[],
            is_complete=True,
        )
