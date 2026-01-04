"""
FastAPI main application for RAG Knowledge Assistant.
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models.config import Settings
from .rag.pipeline import RAGPipeline
from .api import endpoints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
settings = None
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global settings, pipeline
    
    # Startup
    logger.info("Starting RAG Knowledge Assistant...")
    
    try:
        # Load settings
        settings = Settings()
        logger.info("Settings loaded successfully")
        
        # Initialize RAG pipeline
        pipeline = RAGPipeline(settings)
        
        # Set pipeline in endpoints module
        endpoints.pipeline = pipeline
        
        logger.info("RAG pipeline initialized successfully")
        logger.info("Application startup completed")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down RAG Knowledge Assistant...")
    
    try:
        # Persist vector database
        if pipeline and pipeline.vector_store:
            pipeline.vector_store.persist_to_disk()
            logger.info("Vector database persisted")
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="RAG Knowledge Assistant",
    description="A production-ready RAG system for document-based question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(endpoints.router, prefix="/api/v1", tags=["RAG"])

# Serve static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend application."""
        index_path = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {"message": "RAG Knowledge Assistant API", "docs_url": "/docs"}
else:
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "RAG Knowledge Assistant API",
            "docs_url": "/docs",
            "version": "1.0.0"
        }


@app.get("/info")
async def get_info():
    """Get application information."""
    return {
        "name": "RAG Knowledge Assistant",
        "version": "1.0.0",
        "description": "Production-ready RAG system for document-based Q&A",
        "features": [
            "Document upload (text, markdown)",
            "Vector-based similarity search",
            "OpenAI embeddings and chat completion",
            "FAISS vector database",
            "Chunk-based retrieval with sources"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
