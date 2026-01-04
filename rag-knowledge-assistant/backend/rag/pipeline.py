"""
Main RAG pipeline orchestrating ingestion, retrieval, and generation.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.schemas import (
    DocumentChunk, SearchResult, QueryResponse, SourceReference,
    DocumentUploadResponse
)
from ..models.config import Settings
from ..vectordb.faiss_store import FAISVectorStore, VectorDBConfig
from .ingest import DocumentProcessor
from .embeddings import EmbeddingService
from .retrieve import DocumentRetriever, RetrievalConfig
from .prompt import RAGGenerator, LLMProvider, PromptBuilder

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for document ingestion and question answering.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize RAG pipeline with all components.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize vector database
        vector_config = VectorDBConfig(persist_path=settings.vector_db_path)
        self.vector_store = FAISVectorStore(
            config=vector_config,
            dimension=settings.embedding_dimension
        )
        
        # Initialize services
        self.embedding_service = EmbeddingService(settings)
        self.document_processor = DocumentProcessor(settings)
        self.retriever = DocumentRetriever(self.vector_store, self.embedding_service)
        
        # Initialize LLM components
        llm_provider = LLMProvider(settings)
        prompt_builder = PromptBuilder()
        self.generator = RAGGenerator(llm_provider, prompt_builder)
        
        logger.info("RAG pipeline initialized successfully")
    
    async def ingest_document(
        self,
        filename: str,
        content: str,
        content_type: str
    ) -> DocumentUploadResponse:
        """
        Ingest a document into the knowledge base.
        
        Args:
            filename: Original filename
            content: Document content
            content_type: MIME type of the document
            
        Returns:
            Upload response with document ID and status
        """
        try:
            start_time = time.time()
            
            # Validate file
            if not self.document_processor.validate_file_type(filename, content_type):
                raise ValueError(f"Unsupported file type: {content_type}")
            
            if not self.document_processor.validate_file_size(content):
                raise ValueError(f"File too large (max {self.settings.max_file_size_mb}MB)")
            
            # Process document into chunks
            document_id, chunks = self.document_processor.process_document(
                filename, content, content_type
            )
            
            if not chunks:
                raise ValueError("No content could be extracted from the document")
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store chunks in vector database
            self.vector_store.add_chunks(chunks)
            
            # Persist to disk
            self.vector_store.persist_to_disk()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Successfully ingested document '{filename}' with {len(chunks)} chunks in {processing_time}ms")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                chunks_created=len(chunks),
                status="success",
                message=f"Document processed into {len(chunks)} chunks"
            )
            
        except Exception as e:
            logger.error(f"Error ingesting document '{filename}': {e}")
            raise ValueError(f"Failed to ingest document: {str(e)}")
    
    async def query_knowledge_base(
        self,
        question: str,
        max_chunks: int = None,
        similarity_threshold: float = None
    ) -> QueryResponse:
        """
        Query the knowledge base and generate an answer.
        
        Args:
            question: User's question
            max_chunks: Maximum number of chunks to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            
        Returns:
            Query response with answer and sources
        """
        try:
            start_time = time.time()
            
            # Set defaults from settings
            max_chunks = max_chunks or self.settings.default_max_chunks
            similarity_threshold = similarity_threshold or self.settings.default_similarity_threshold
            
            # Configure retrieval
            retrieval_config = RetrievalConfig(
                max_chunks=max_chunks,
                similarity_threshold=similarity_threshold,
                rerank=True,  # Enable reranking for better results
                diversity_threshold=0.8  # Ensure diverse results
            )
            
            # Retrieve relevant chunks
            search_results = await self.retriever.retrieve_relevant_chunks(
                query=question,
                config=retrieval_config
            )
            
            if not search_results:
                # No relevant chunks found
                answer = "I don't have any relevant information in the knowledge base to answer this question."
                sources = []
            else:
                # Generate answer using retrieved chunks
                answer, sources = await self.generator.generate_answer(
                    question=question,
                    search_results=search_results,
                    max_context_length=4000
                )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Answered query with {len(sources)} sources in {processing_time}ms")
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                retrieved_chunks=len(search_results),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise ValueError(f"Failed to process query: {str(e)}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            vector_stats = self.vector_store.get_stats()
            retrieval_stats = self.retriever.get_retrieval_stats()
            
            return {
                "total_documents": retrieval_stats["total_documents"],
                "total_chunks": retrieval_stats["total_chunks"],
                "vector_dimension": vector_stats["dimension"],
                "index_size": vector_stats["total_vectors"],
                "storage_path": vector_stats["persist_path"],
                "embedding_model": self.settings.openai_embedding_model,
                "chat_model": self.settings.openai_chat_model
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def clear_knowledge_base(self) -> Dict[str, str]:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            Status message
        """
        try:
            self.vector_store.clear()
            self.vector_store.persist_to_disk()
            
            logger.info("Knowledge base cleared successfully")
            return {"status": "success", "message": "Knowledge base cleared"}
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return {"status": "error", "message": f"Failed to clear knowledge base: {str(e)}"}
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a specific document from the knowledge base.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Status with number of chunks deleted
        """
        try:
            deleted_count = self.vector_store.delete_document(document_id)
            
            if deleted_count > 0:
                self.vector_store.persist_to_disk()
                logger.info(f"Deleted document {document_id} ({deleted_count} chunks)")
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_deleted": deleted_count,
                "message": f"Deleted {deleted_count} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete document: {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all pipeline components.
        
        Returns:
            Health status for each component
        """
        health_status = {
            "pipeline": "healthy",
            "vector_db": "unknown",
            "embeddings": "unknown",
            "llm": "unknown"
        }
        
        try:
            # Check vector database
            vector_stats = self.vector_store.get_stats()
            health_status["vector_db"] = "healthy" if vector_stats else "unhealthy"
            
            # Check embedding service
            embedding_healthy = await self.embedding_service.health_check()
            health_status["embeddings"] = "healthy" if embedding_healthy else "unhealthy"
            
            # Check LLM service
            llm_healthy = await self.generator.health_check()
            health_status["llm"] = "healthy" if llm_healthy else "unhealthy"
            
            # Overall pipeline health
            all_healthy = all(
                status == "healthy" 
                for status in health_status.values() 
                if status != "unknown"
            )
            health_status["pipeline"] = "healthy" if all_healthy else "unhealthy"
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status["pipeline"] = "unhealthy"
        
        return health_status
