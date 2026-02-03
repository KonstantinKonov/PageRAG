from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, AliasChoices


class DocType(Enum):
    TEN_K = "10-k"
    TEN_Q = "10-q"
    EIGHT_K = "8-k"
    OTHER = "other"


class FiscalQuarter(Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class ChunkMetadata(BaseModel):
    company_name: Optional[str] = Field(
        default=None,
        description="Company name (lowercase, e.g. 'amazon', 'apple', 'google')",
    )
    doc_type: Optional[DocType] = Field(
        default=None, description="Document type (10-k, 10-q, 8-k, etc.)"
    )
    fiscal_year: Optional[int] = Field(
        default=None, ge=1950, le=2050, description="Fiscal year"
    )
    fiscal_quarter: Optional[FiscalQuarter] = Field(
        default=None, description="Fiscal quarter (q1-q4) if applicable"
    )

    model_config = {"use_enum_values": True}


class RankingKeywords(BaseModel):
    keywords: List[str] = Field(
        ..., description="Generate exactly 5 financial keywords", min_length=5, max_length=5
    )


class SearchQueries(BaseModel):
    search_queries: List[str] = Field(
        validation_alias=AliasChoices("search_queries", "queries"),
        description="1-3 search queries to retrieve missing information.",
    )

    model_config = {"populate_by_name": True}


class QueryScope(BaseModel):
    in_scope: bool = Field(
        ...,
        description="Whether the user query is in-scope for financial RAG on SEC filings.",
    )
    reason: Optional[str] = Field(default=None, description="Short reason for the decision.")


class GradeDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if documents are relevant to answer the question, False otherwise.",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why documents are relevant or not.",
    )


class RewriteQuery(BaseModel):
    rewritten_query: str = Field(
        ..., description="Rewritten query optimized for document retrieval."
    )
