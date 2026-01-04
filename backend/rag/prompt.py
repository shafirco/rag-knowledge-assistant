"""
Prompt engineering and LLM interaction for RAG pipeline.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

from ..models.schemas import SearchResult, SourceReference
from ..models.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template for RAG prompts."""
    
    system_message: str = """You are a helpful AI assistant that answers questions based solely on the provided context. 

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Be specific and cite the sources when possible
4. Do not use any external knowledge beyond the provided context
5. Maintain a helpful and professional tone

Context will be provided in the format:
[Source: filename - chunk X] content"""

    user_template: str = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""


class LLMProvider:
    """
    LLM provider for generating responses using OpenAI Chat Completion.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize LLM provider.
        
        Args:
            settings: Application settings
        """
        if openai is None:
            raise ImportError("openai package not installed")
        
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response using OpenAI Chat Completion.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            import asyncio
            
            # OpenAI client is synchronous, run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_response_sync,
                messages,
                max_tokens or self.max_tokens,
                temperature or self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")
    
    def _generate_response_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ):
        """Synchronous wrapper for OpenAI Chat Completion."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )


class PromptBuilder:
    """
    Builds prompts for RAG queries.
    """
    
    def __init__(self, template: Optional[PromptTemplate] = None):
        """
        Initialize prompt builder.
        
        Args:
            template: Optional custom prompt template
        """
        self.template = template or PromptTemplate()
    
    def build_rag_prompt(
        self,
        question: str,
        search_results: List[SearchResult],
        max_context_length: int = 4000
    ) -> List[Dict[str, str]]:
        """
        Build RAG prompt from question and search results.
        
        Args:
            question: User's question
            search_results: Retrieved document chunks
            max_context_length: Maximum context length in characters
            
        Returns:
            List of chat messages for LLM
        """
        # Build context from search results
        context = self._build_context(search_results, max_context_length)
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": self.template.system_message
            },
            {
                "role": "user",
                "content": self.template.user_template.format(
                    context=context,
                    question=question
                )
            }
        ]
        
        return messages
    
    def _build_context(
        self,
        search_results: List[SearchResult],
        max_length: int
    ) -> str:
        """
        Build context string from search results.
        
        Args:
            search_results: Retrieved document chunks
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for result in search_results:
            chunk = result.chunk
            
            # Format source reference
            source_ref = f"[Source: {chunk.filename} - chunk {chunk.chunk_index + 1}]"
            chunk_text = f"{source_ref}\n{chunk.content}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_length and context_parts:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def extract_source_references(
        self,
        search_results: List[SearchResult]
    ) -> List[SourceReference]:
        """
        Extract source references from search results.
        
        Args:
            search_results: Retrieved document chunks
            
        Returns:
            List of source references
        """
        sources = []
        
        for result in search_results:
            chunk = result.chunk
            
            # Create preview (first 150 characters)
            preview = chunk.content[:150]
            if len(chunk.content) > 150:
                preview += "..."
            
            source = SourceReference(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                similarity_score=result.similarity_score,
                content_preview=preview
            )
            
            sources.append(source)
        
        return sources


class RAGGenerator:
    """
    Main RAG generation service combining retrieval and generation.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize RAG generator.
        
        Args:
            llm_provider: LLM provider for response generation
            prompt_builder: Optional custom prompt builder
        """
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    async def generate_answer(
        self,
        question: str,
        search_results: List[SearchResult],
        max_context_length: int = 4000
    ) -> tuple[str, List[SourceReference]]:
        """
        Generate answer from question and retrieved chunks.
        
        Args:
            question: User's question
            search_results: Retrieved relevant chunks
            max_context_length: Maximum context length for prompt
            
        Returns:
            Tuple of (generated_answer, source_references)
        """
        try:
            # Build prompt
            messages = self.prompt_builder.build_rag_prompt(
                question=question,
                search_results=search_results,
                max_context_length=max_context_length
            )
            
            # Generate response
            answer = await self.llm_provider.generate_response(messages)
            
            # Extract source references
            sources = self.prompt_builder.extract_source_references(search_results)
            
            logger.info(f"Generated answer with {len(sources)} source references")
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise ValueError(f"Failed to generate answer: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if RAG generator is healthy.
        
        Returns:
            True if service is responsive
        """
        try:
            # Test with a simple message
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OK' if you can respond."}
            ]
            
            response = await self.llm_provider.generate_response(messages)
            return "ok" in response.lower()
            
        except Exception as e:
            logger.error(f"RAG generator health check failed: {e}")
            return False
