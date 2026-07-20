from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    strategy_used: str
    heal_attempts: int
    confidence: float


class HealthResponse(BaseModel):
    status: str


class DocumentInfo(BaseModel):
    name: str
    chunk_count: int
    size_bytes: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int


class AnalyticsResponse(BaseModel):
    total_failures: int
    strategy_counts: Dict[str, int]
    reason_counts: Dict[str, int]
    query_counts: Dict[str, int]


class TopQueriesResponse(BaseModel):
    top_queries: List[Dict[str, Any]]


class ChatResetResponse(BaseModel):
    status: str
    message: str


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    total_messages: int


class SystemStatusResponse(BaseModel):
    status: str
    total_chunks: int
    bm25_ready: bool
    vectorstore_ready: bool
    llm_model: str
    embedding_model: str
    documents_ingested: int