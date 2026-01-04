"""
FAISS vector database implementation for document embeddings.
"""
import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from ..models.schemas import DocumentChunk, SearchResult
from ..models.config import VectorDBConfig

logger = logging.getLogger(__name__)


class FAISVectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    Provides functionality to store, search, and persist vector embeddings
    with associated document metadata.
    """
    
    def __init__(self, config: VectorDBConfig, dimension: int = 1536):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Vector database configuration
            dimension: Embedding vector dimension
        """
        self.config = config
        self.dimension = dimension
        self.index = None
        self.metadata_store: Dict[int, DocumentChunk] = {}
        self.next_id = 0
        
        # Ensure persist directory exists
        Path(self.config.persist_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize or load existing index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index or load from disk if exists."""
        index_path = Path(self.config.persist_path) / "faiss.index"
        metadata_path = Path(self.config.persist_path) / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            logger.info("Loading existing FAISS index and metadata...")
            self._load_from_disk()
        else:
            logger.info("Creating new FAISS index...")
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use Inner Product index (equivalent to cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.next_id = 0
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _load_from_disk(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            index_path = Path(self.config.persist_path) / "faiss.index"
            metadata_path = Path(self.config.persist_path) / "metadata.pkl"
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata']
                self.next_id = data['next_id']
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            logger.info("Creating new index...")
            self._create_new_index()
    
    def persist_to_disk(self) -> None:
        """Persist FAISS index and metadata to disk."""
        try:
            index_path = Path(self.config.persist_path) / "faiss.index"
            metadata_path = Path(self.config.persist_path) / "metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata_store,
                    'next_id': self.next_id
                }, f)
            
            logger.info(f"Persisted FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error persisting FAISS index: {e}")
            raise
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[int]:
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            List of assigned vector IDs
        """
        if not chunks:
            return []
        
        # Prepare embeddings matrix
        embeddings = []
        chunk_ids = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            # Normalize embedding for cosine similarity
            embedding = np.array(chunk.embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
            chunk_ids.append(self.next_id)
            
            # Store metadata
            self.metadata_store[self.next_id] = chunk
            self.next_id += 1
        
        # Add to FAISS index
        embeddings_matrix = np.vstack(embeddings)
        self.index.add(embeddings_matrix)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return chunk_ids
    
    def search(self, query_embedding: List[float], k: int = 5, 
               similarity_threshold: float = 0.7) -> List[SearchResult]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with similarity scores
        """
        if self.index.ntotal == 0:
            logger.warning("No vectors in index for search")
            return []
        
        # Normalize query embedding
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices
            if idx == -1:
                continue
                
            # Convert inner product back to cosine similarity (they're the same for normalized vectors)
            similarity = float(score)
            
            # Apply similarity threshold
            if similarity < similarity_threshold:
                continue
            
            # Get chunk metadata
            if idx in self.metadata_store:
                chunk = self.metadata_store[idx]
                results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=similarity
                ))
        
        logger.info(f"Found {len(results)} chunks above threshold {similarity_threshold}")
        return results
    
    def get_chunk_by_id(self, vector_id: int) -> Optional[DocumentChunk]:
        """Get chunk metadata by vector ID."""
        return self.metadata_store.get(vector_id)
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        # Find chunks to delete
        ids_to_delete = [
            vid for vid, chunk in self.metadata_store.items()
            if chunk.document_id == document_id
        ]
        
        if not ids_to_delete:
            return 0
        
        # Remove from metadata store
        for vid in ids_to_delete:
            del self.metadata_store[vid]
        
        # Note: FAISS doesn't support deletion, so we'd need to rebuild the index
        # For simplicity, we'll mark this as a limitation
        logger.warning(f"Deleted {len(ids_to_delete)} chunks from metadata (FAISS index rebuild required)")
        
        return len(ids_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "metadata_count": len(self.metadata_store),
            "persist_path": str(self.config.persist_path)
        }
    
    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self._create_new_index()
        logger.info("Cleared vector store")
