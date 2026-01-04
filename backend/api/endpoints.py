"""
FastAPI endpoints for RAG Knowledge Assistant.
"""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from ..models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse, 
    HealthCheckResponse, ErrorResponse
)
from ..rag.pipeline import RAGPipeline
from ..models.config import Settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global pipeline instance (will be initialized in main.py)
pipeline: RAGPipeline = None


def get_pipeline() -> RAGPipeline:
    """Dependency to get pipeline instance."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    return pipeline


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> DocumentUploadResponse:
    """
    Upload and process a document for the knowledge base.
    
    Accepts text, markdown, or PDF files and processes them into
    searchable chunks with vector embeddings.
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content_bytes = await file.read()
        
        # Handle different file types
        if file.content_type == "application/pdf":
            # For PDF, we'd need to extract text first
            # For now, return an error suggesting text extraction
            raise HTTPException(
                status_code=400, 
                detail="PDF support requires text extraction. Please convert to text first."
            )
        else:
            # Assume text-based file
            try:
                content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8.")
        
        # Validate content
        if not content.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        # Process document
        result = await rag_pipeline.ingest_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type or "text/plain"
        )
        
        logger.info(f"Successfully uploaded document: {file.filename}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error during upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during document upload")


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> QueryResponse:
    """
    Query the knowledge base and get an AI-generated answer.
    
    Retrieves relevant document chunks and generates a response
    based only on the uploaded content.
    """
    try:
        # Validate query
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process query
        result = await rag_pipeline.query_knowledge_base(
            question=request.question,
            max_chunks=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        logger.info(f"Processed query: '{request.question[:50]}...'")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error during query: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> HealthCheckResponse:
    """
    Check the health status of all RAG pipeline components.
    """
    try:
        # Perform comprehensive health check
        health_status = await rag_pipeline.health_check()
        
        # Determine overall status
        overall_status = "healthy" if health_status.get("pipeline") == "healthy" else "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            vector_db_status=health_status.get("vector_db", "unknown"),
            embeddings_status=health_status.get("embeddings", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            vector_db_status="unknown",
            embeddings_status="unknown"
        )


@router.get("/stats")
async def get_stats(
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.
    """
    try:
        stats = rag_pipeline.get_document_stats()
        return {"status": "success", "data": stats}
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> Dict[str, Any]:
    """
    Delete a specific document from the knowledge base.
    """
    try:
        result = rag_pipeline.delete_document(document_id)
        
        if result.get("status") == "success":
            return result
        else:
            raise HTTPException(status_code=404, detail=result.get("message", "Document not found"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting document")


@router.delete("/clear")
async def clear_knowledge_base(
    rag_pipeline: RAGPipeline = Depends(get_pipeline)
) -> Dict[str, str]:
    """
    Clear all documents from the knowledge base.
    
    WARNING: This action cannot be undone.
    """
    try:
        result = rag_pipeline.clear_knowledge_base()
        
        if result.get("status") == "success":
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Error clearing knowledge base"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Error clearing knowledge base")


# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc)
        ).dict()
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).dict()
    )
