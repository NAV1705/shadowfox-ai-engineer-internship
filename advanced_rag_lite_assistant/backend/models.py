from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    ready = "ready"
    failed = "failed"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    status: DocumentStatus


class UploadResponse(BaseModel):
    documents: List[DocumentInfo]


class QueryRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=1, description="Documents to search over")
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    groundedness_score: float
    confidence_label: str
    retrieved_chunks: List[RetrievedChunk]
