"""
Pydantic models for the RAG Knowledge Assistant API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    filename: str = Field(..., description="Name of the uploaded file")
    content: str = Field(..., description="Text content of the document")
    content_type: str = Field(..., description="MIME type of the document")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Name of the uploaded file")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")


class QueryRequest(BaseModel):
    """Request model for knowledge base queries."""
    question: str = Field(..., description="User's question", min_length=1, max_length=1000)
    max_chunks: int = Field(default=5, description="Maximum number of chunks to retrieve", ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score", ge=0.0, le=1.0)


class SourceReference(BaseModel):
    """Reference to a source document chunk."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    chunk_index: int = Field(..., description="Chunk index within the document")
    similarity_score: float = Field(..., description="Similarity score for this chunk")
    content_preview: str = Field(..., description="Preview of the chunk content")


class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""
    answer: str = Field(..., description="Generated answer based on retrieved context")
    sources: List[SourceReference] = Field(..., description="Source references used to generate the answer")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class DocumentChunk(BaseModel):
    """Internal model for document chunks."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk index within the document")
    start_char: int = Field(..., description="Start character position in original document")
    end_char: int = Field(..., description="End character position in original document")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class SearchResult(BaseModel):
    """Internal model for search results."""
    chunk: DocumentChunk = Field(..., description="Retrieved chunk")
    similarity_score: float = Field(..., description="Similarity score")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    vector_db_status: str = Field(..., description="Vector database status")
    embeddings_status: str = Field(..., description="Embeddings service status")
