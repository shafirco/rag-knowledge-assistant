"""
Document retrieval for RAG pipeline.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..models.schemas import SearchResult, DocumentChunk
from ..vectordb.faiss_store import FAISVectorStore
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    max_chunks: int = 5
    similarity_threshold: float = 0.7
    rerank: bool = False
    diversity_threshold: float = 0.8


class DocumentRetriever:
    """
    Handles document retrieval using vector similarity search.
    """
    
    def __init__(self, vector_store: FAISVectorStore, embedding_service: EmbeddingService):
        """
        Initialize document retriever.
        
        Args:
            vector_store: FAISS vector database
            embedding_service: Embedding generation service
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[SearchResult]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query string
            config: Retrieval configuration
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_single_text(query)
            
            # Search vector database
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=config.max_chunks * 2,  # Get more candidates for potential reranking
                similarity_threshold=config.similarity_threshold
            )
            
            # Apply post-processing if enabled
            if config.rerank:
                search_results = self._rerank_results(query, search_results)
            
            # Apply diversity filtering if enabled
            if config.diversity_threshold < 1.0:
                search_results = self._apply_diversity_filter(
                    search_results, config.diversity_threshold
                )
            
            # Limit to requested number of chunks
            search_results = search_results[:config.max_chunks]
            
            logger.info(f"Retrieved {len(search_results)} relevant chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise ValueError(f"Retrieval failed: {str(e)}")
    
    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results using additional scoring methods.
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked search results
        """
        # Simple keyword-based reranking
        query_tokens = set(query.lower().split())
        
        for result in results:
            chunk_tokens = set(result.chunk.content.lower().split())
            
            # Calculate keyword overlap bonus
            overlap = len(query_tokens.intersection(chunk_tokens))
            overlap_ratio = overlap / len(query_tokens) if query_tokens else 0
            
            # Boost similarity score based on keyword overlap
            boost = 0.1 * overlap_ratio
            result.similarity_score = min(1.0, result.similarity_score + boost)
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    def _apply_diversity_filter(
        self,
        results: List[SearchResult],
        diversity_threshold: float
    ) -> List[SearchResult]:
        """
        Filter results to maintain diversity (avoid too similar chunks).
        
        Args:
            results: Search results to filter
            diversity_threshold: Similarity threshold for diversity filtering
            
        Returns:
            Filtered search results
        """
        if not results:
            return results
        
        filtered_results = [results[0]]  # Always include the best result
        
        for candidate in results[1:]:
            # Check similarity with already selected results
            too_similar = False
            
            for selected in filtered_results:
                # Simple content similarity check (could be improved with embeddings)
                content_similarity = self._calculate_content_similarity(
                    candidate.chunk.content,
                    selected.chunk.content
                )
                
                if content_similarity > diversity_threshold:
                    too_similar = True
                    break
            
            if not too_similar:
                filtered_results.append(candidate)
        
        return filtered_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two text contents using Jaccard similarity.
        
        Args:
            content1: First text content
            content2: Second text content
            
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize and create sets
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_chunk_by_document(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks belonging to the document
        """
        chunks = []
        for chunk in self.vector_store.metadata_store.values():
            if chunk.document_id == document_id:
                chunks.append(chunk)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.chunk_index)
        return chunks
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with retrieval statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        # Calculate document statistics
        document_ids = set()
        total_chunks = 0
        
        for chunk in self.vector_store.metadata_store.values():
            document_ids.add(chunk.document_id)
            total_chunks += 1
        
        return {
            "total_documents": len(document_ids),
            "total_chunks": total_chunks,
            "vector_dimension": vector_stats["dimension"],
            "index_type": vector_stats["index_type"],
            "vector_store_size": vector_stats["total_vectors"]
        }
