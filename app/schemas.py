from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    ingested_files: List[str]
    skipped_files: List[str]


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
