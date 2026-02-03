from typing import List, TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

from app.agent import (
    decompose_query,
    extract_filters,
    generate_ranking_keywords,
    grade_documents,
    rewrite_query,
    generate_answer,
)
from app.retrieval import search_docs, write_debug_log
from app.web_search import web_search
from app.logger import setup_logger
from app.config import settings


logger = setup_logger(__name__)


class QueryState(TypedDict, total=False):
    query: str
    session: Any
    k: int
    sub_queries: List[str]
    sub_index: int
    current_query: str
    rewrite_attempted: bool
    retrieved_docs_text: str
    is_relevant: bool
    web_text: str
    combined_context: List[str]
    final_answer: str


def _format_docs(docs) -> str:
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


async def decompose_node(state: QueryState) -> QueryState:
    user_query = state["query"]
    queries = decompose_query(user_query)
    logger.info(f"[graph.decompose] Sub-queries: {queries}")
    current = queries[0] if queries else user_query
    return {
        "sub_queries": queries,
        "sub_index": 0,
        "current_query": current,
        "rewrite_attempted": False,
        "combined_context": [],
    }


async def retrieve_node(state: QueryState) -> QueryState:
    query = state["current_query"]
    logger.info(f"[graph.retrieve] Query: {query}")

    filters = extract_filters(query)
    keywords = generate_ranking_keywords(query)
    docs = await search_docs(
        state["session"],
        query,
        filters=filters,
        ranking_keywords=keywords,
        k=state["k"],
        fetch_k=settings.DEFAULT_FETCH_K,
    )
    write_debug_log(docs)
    chunk_text = _format_docs(docs)
    logger.debug(f"[graph.retrieve] Retrieved docs length: {len(chunk_text)}")
    return {"retrieved_docs_text": chunk_text}


async def grade_node(state: QueryState) -> QueryState:
    retrieved_docs = state.get("retrieved_docs_text", "")
    if not retrieved_docs:
        logger.info("[graph.grade] No documents to grade")
        return {"is_relevant": False}

    decision = grade_documents(state["current_query"], retrieved_docs)
    return {"is_relevant": decision.is_relevant}


async def rewrite_node(state: QueryState) -> QueryState:
    rewritten = rewrite_query(state["current_query"])
    logger.info(f"[graph.rewrite] Rewritten query: {rewritten}")
    return {
        "current_query": rewritten,
        "rewrite_attempted": True,
    }


async def web_search_node(state: QueryState) -> QueryState:
    query = state["current_query"]
    result = web_search(query)
    return {"web_text": result}


async def append_context_node(state: QueryState) -> QueryState:
    combined = state.get("combined_context", [])
    chunk_text = state.get("retrieved_docs_text", "")
    web_text = state.get("web_text", "")
    if web_text:
        if chunk_text:
            chunk_text = f"{chunk_text}\n\n[WEB_SEARCH]\n{web_text}"
        else:
            chunk_text = f"[WEB_SEARCH]\n{web_text}"

    if chunk_text:
        combined.append(chunk_text)

    sub_queries = state.get("sub_queries", [])
    sub_index = state.get("sub_index", 0) + 1

    next_query = None
    if sub_index < len(sub_queries):
        next_query = sub_queries[sub_index]

    return {
        "combined_context": combined,
        "sub_index": sub_index,
        "current_query": next_query,
        "rewrite_attempted": False,
        "retrieved_docs_text": "",
        "web_text": "",
        "is_relevant": True,
    }


async def answer_node(state: QueryState) -> QueryState:
    combined_retrieved = "\n\n".join(state.get("combined_context", []))
    answer = generate_answer(state["query"], combined_retrieved)
    return {"final_answer": answer}


def _route_after_grade(state: QueryState) -> str:
    if state.get("is_relevant", False):
        return "append_context"
    if state.get("rewrite_attempted", False):
        return "web_search"
    return "rewrite"


def _route_after_append(state: QueryState) -> str:
    current_query = state.get("current_query")
    return "answer" if not current_query else "retrieve"


def build_graph():
    builder = StateGraph(QueryState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("append_context", append_context_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", _route_after_grade, ["append_context", "rewrite", "web_search"]
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("web_search", "append_context")
    builder.add_conditional_edges("append_context", _route_after_append, ["retrieve", "answer"])
    builder.add_edge("answer", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
