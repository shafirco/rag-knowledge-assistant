"""
Configuration models for the RAG Knowledge Assistant.
"""
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    api_title: str = Field(default="RAG Knowledge Assistant", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # OpenAI Settings
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", description="OpenAI embedding model")
    openai_chat_model: str = Field(default="gpt-3.5-turbo", description="OpenAI chat model")
    openai_max_tokens: int = Field(default=1000, description="Maximum tokens for chat completion")
    openai_temperature: float = Field(default=0.1, description="Temperature for chat completion")
    
    # Vector Database Settings
    vector_db_path: str = Field(default="./data/vector_db", description="Path to FAISS vector database")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    
    # Document Processing Settings
    chunk_size: int = Field(default=1000, description="Default chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters")
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    supported_file_types: list = Field(default=["txt", "md", "pdf"], description="Supported file extensions")
    
    # RAG Settings
    default_max_chunks: int = Field(default=5, description="Default maximum chunks to retrieve")
    default_similarity_threshold: float = Field(default=0.7, description="Default similarity threshold")
    
    # CORS Settings
    cors_origins: list = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"], description="CORS origins")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding providers."""
    
    provider: str = Field(default="openai", description="Embedding provider (openai, huggingface)")
    model_name: str = Field(default="text-embedding-ada-002", description="Model name")
    dimension: int = Field(default=1536, description="Embedding dimension")
    batch_size: int = Field(default=100, description="Batch size for embedding generation")
    
    class Config:
        env_prefix = "EMBEDDING_"


class VectorDBConfig(BaseSettings):
    """Configuration for vector database."""
    
    index_type: str = Field(default="IndexFlatIP", description="FAISS index type")
    metric_type: str = Field(default="cosine", description="Distance metric")
    persist_path: str = Field(default="./data/vector_db", description="Path to persist vector database")
    
    class Config:
        env_prefix = "VECTOR_DB_"
