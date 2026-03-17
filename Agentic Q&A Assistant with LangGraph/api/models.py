"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


# ============================================================
# Enums
# ============================================================

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class RetrievalMode(str, Enum):
    SIMILARITY = "similarity"
    HYBRID = "hybrid"


# ============================================================
# Document Models
# ============================================================

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    chunks_indexed: int = Field(default=0, description="Number of chunks indexed")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    document_id: str = Field(..., description="Unique identifier")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    chunks_count: int = Field(default=0, description="Number of chunks")
    indexed_at: Optional[datetime] = Field(None, description="Indexing timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    total: int = Field(..., description="Total number of documents")
    documents: List[DocumentInfo] = Field(..., description="List of documents")


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    document_id: str = Field(..., description="Deleted document ID")
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")


# ============================================================
# Query Models
# ============================================================

class QueryRequest(BaseModel):
    """Request for Q&A query."""
    query: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.SIMILARITY, description="Retrieval strategy")
    similarity_cutoff: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    stream: bool = Field(default=False, description="Enable streaming response")


class SourceDocument(BaseModel):
    """A source document used to generate the answer."""
    text: str = Field(..., description="Document text snippet")
    source: Optional[str] = Field(None, description="Source filename")
    page: Optional[int] = Field(None, description="Page number")
    doc_id: Optional[str] = Field(None, description="Document ID")
    score: Optional[float] = Field(None, description="Relevance score")


class QueryResponse(BaseModel):
    """Response for Q&A query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents")
    retrieval_mode: RetrievalMode = Field(..., description="Retrieval mode used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SearchRequest(BaseModel):
    """Request for similarity search (retrieval only, no LLM)."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.SIMILARITY, description="Retrieval strategy")


class SearchResponse(BaseModel):
    """Response for similarity search."""
    query: str = Field(..., description="Original query")
    results: List[SourceDocument] = Field(..., description="Retrieved documents")
    total_results: int = Field(..., description="Number of results returned")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================
# Health & Monitoring Models
# ============================================================

class ServiceHealth(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")


class ComponentStatus(BaseModel):
    """Status of a service component."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Status (up/down)")
    latency_ms: Optional[float] = Field(None, description="Latency in milliseconds")
    details: Optional[str] = Field(None, description="Additional details")


class DetailedHealth(BaseModel):
    """Detailed health check with component statuses."""
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
    components: List[ComponentStatus] = Field(..., description="Component statuses")
    timestamp: datetime = Field(..., description="Current timestamp")


class MetricsResponse(BaseModel):
    """Service metrics."""
    total_documents: int = Field(..., description="Total indexed documents")
    total_chunks: int = Field(..., description="Total indexed chunks")
    total_queries: int = Field(..., description="Total queries processed")
    avg_query_time_ms: float = Field(..., description="Average query time")
    uptime_seconds: float = Field(..., description="Service uptime")


# ============================================================
# Error Models
# ============================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: List[Dict[str, Any]] = Field(..., description="Validation errors")
