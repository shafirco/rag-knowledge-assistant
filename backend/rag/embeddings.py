"""
Embedding generation for RAG pipeline.
"""
import asyncio
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

try:
    import openai
except ImportError:
    openai = None

from ..models.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-ada-002.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            settings: Application settings containing OpenAI configuration
        """
        if openai is None:
            raise ImportError("openai package not installed")
        
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = settings.embedding_dimension
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # OpenAI client is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._generate_embeddings_sync, 
                texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            logger.info(f"Generated {len(embeddings)} embeddings using {self.model}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}")
    
    def _generate_embeddings_sync(self, texts: List[str]):
        """Synchronous wrapper for OpenAI embeddings API."""
        return self.client.embeddings.create(
            input=texts,
            model=self.model
        )
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


class EmbeddingService:
    """
    Service for generating embeddings with provider abstraction.
    """
    
    def __init__(self, settings: Settings, provider: Optional[EmbeddingProvider] = None):
        """
        Initialize embedding service.
        
        Args:
            settings: Application settings
            provider: Optional embedding provider (defaults to OpenAI)
        """
        self.settings = settings
        self.provider = provider or OpenAIEmbeddingProvider(settings)
    
    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for texts with batching support.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = await self.provider.generate_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    async def embed_single_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.provider.get_dimension()
    
    async def health_check(self) -> bool:
        """
        Check if embedding service is healthy.
        
        Returns:
            True if service is responsive
        """
        try:
            # Test with a simple text
            test_embedding = await self.embed_single_text("test")
            return len(test_embedding) == self.get_dimension()
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
