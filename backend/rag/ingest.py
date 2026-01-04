"""
Document ingestion and chunking for RAG pipeline.
"""
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from ..models.schemas import DocumentChunk
from ..models.config import Settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents for RAG pipeline.
    
    Handles text extraction, chunking with overlap, and metadata creation.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize document processor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def process_document(self, filename: str, content: str, 
                        content_type: str) -> tuple[str, List[DocumentChunk]]:
        """
        Process a document into chunks.
        
        Args:
            filename: Original filename
            content: Document content (already extracted)
            content_type: MIME type of the document
            
        Returns:
            Tuple of (document_id, list_of_chunks)
        """
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Clean and normalize text
        cleaned_content = self._clean_text(content)
        
        # Create chunks with overlap
        chunks = self._create_chunks(
            document_id=document_id,
            filename=filename,
            content=cleaned_content,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        logger.info(f"Processed document '{filename}' into {len(chunks)} chunks")
        return document_id, chunks
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content.
        
        Args:
            pdf_content: Raw PDF bytes
            
        Returns:
            Extracted text content
        """
        if PyPDF2 is None:
            raise ValueError("PyPDF2 not installed. Cannot process PDF files.")
        
        try:
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_chunks(self, document_id: str, filename: str, content: str,
                      chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """
        Create overlapping chunks from document content.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            content: Document content
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        if not content.strip():
            return []
        
        chunks = []
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(content):
            # Calculate end position
            end_pos = min(start_pos + chunk_size, len(content))
            
            # If this isn't the last chunk, try to break at a sentence or word boundary
            if end_pos < len(content):
                # Look for sentence boundary (. ! ?) within the last 100 characters
                sentence_end = self._find_sentence_boundary(content, end_pos - 100, end_pos)
                if sentence_end != -1:
                    end_pos = sentence_end
                else:
                    # Look for word boundary
                    word_end = self._find_word_boundary(content, end_pos - 50, end_pos)
                    if word_end != -1:
                        end_pos = word_end
            
            # Extract chunk content
            chunk_content = content[start_pos:end_pos].strip()
            
            # Skip empty chunks
            if not chunk_content:
                start_pos = end_pos
                continue
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                filename=filename,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "content_length": len(chunk_content),
                    "word_count": len(chunk_content.split()),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position with overlap
            start_pos = max(start_pos + chunk_size - overlap, end_pos)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the last sentence boundary within a range.
        
        Args:
            text: Text content
            start: Start position to search
            end: End position to search
            
        Returns:
            Position of sentence boundary, or -1 if not found
        """
        # Look for sentence endings
        for i in range(end - 1, max(start - 1, -1), -1):
            if text[i] in '.!?':
                # Make sure there's a space or end of text after the punctuation
                if i + 1 >= len(text) or text[i + 1].isspace():
                    return i + 1
        return -1
    
    def _find_word_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the last word boundary within a range.
        
        Args:
            text: Text content
            start: Start position to search
            end: End position to search
            
        Returns:
            Position of word boundary, or -1 if not found
        """
        # Look for whitespace
        for i in range(end - 1, max(start - 1, -1), -1):
            if text[i].isspace():
                return i + 1
        return -1
    
    def validate_file_type(self, filename: str, content_type: str) -> bool:
        """
        Validate if file type is supported.
        
        Args:
            filename: Original filename
            content_type: MIME type
            
        Returns:
            True if file type is supported
        """
        # Extract file extension
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Check against supported types
        return extension in self.settings.supported_file_types
    
    def validate_file_size(self, content: str) -> bool:
        """
        Validate if file size is within limits.
        
        Args:
            content: File content
            
        Returns:
            True if file size is acceptable
        """
        # Calculate size in MB
        size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        return size_mb <= self.settings.max_file_size_mb
