from pydoc import text
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class LoaderException(Exception):
    """Base exception for loader errors."""
    pass

class PDFLoadException(LoaderException):
    """Raised when a PDF cannot be loaded."""
    pass

@dataclass
class Document:
    """Represents as single document chunk."""
    id: str
    content: str
    source_file: str
    page_num: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate document after initalization."""
        if not self.content.strip():
            raise ValueError(f"Document {self.id} has empty content")
        

    
class DocumentLoader:
    """Load and chunk cancer reaserach PDFs."""
    def __init__(self,chunk_size:int = 512,overlap:int = 50) -> None:
        """
        Initialize loader.
        
        Args:
            chunk_size: Characters per chunk
            overlap: Overlap between consecutive chunks (for context)
        
        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size
        """
        if chunk_size <=0:
            raise ValueError("chunk_size must be positive")
        
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk size ")
        
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        logger.info(f"DocumentLoader initialized with chunk_size={chunk_size}, overlap={overlap}")  

    def load_pdf(self, filepath:str) -> List[Document]:
        """
        Load a PDF and return list of Document chunks.
        
        Args:
            filepath: Path to the PDF file"""
        
        try:
            import os
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            logger.info(f"Loading PDF: {filepath}")

            return []
        except Exception as e:
            logger.error(f"Error loading PDF {filepath}: {e}")
            raise PDFLoadException(f"Failed to load PDF {filepath}: {e}") from e
        
    def chunk_text(self, text: str,source_id: str) -> List[Document]:
        """
        Chunk text into Document objects.
        
        Args:
            text: The text to chunk
            source_id: Identifier for the source document
        
        Returns:
            List of Document chunks
        """
        logger.debug(f"Chunking {len(text)} chars into {self.chunk_size}-sized chunks")
        chunks: List[Document] = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            # Extract one chunk starting at position i
            chunk_text = text[i : i + self.chunk_size]
            
            # Skip tiny chunks (just whitespace)
            # These aren't useful for retrieval
            if len(chunk_text.strip()) < 10:
                logger.debug(f"Skipping tiny chunk at position {i}")
                continue
            
            # Create unique chunk ID
            # Example: "cancer_paper_chunk_0", "cancer_paper_chunk_1", etc.
            chunk_id = f"{source_id}_chunk_{len(chunks)}"
            
            # Create Document object
            doc = Document(
                id=chunk_id,
                content=chunk_text,
                source_file=source_id,
                page_num=None  # Will add page numbers in future
            )
            
            # Add to list
            chunks.append(doc)
        
        # Log completion
        logger.info(f"Created {len(chunks)} chunks from {source_id}")
        
        # Return all chunks
        return chunks
    
    
    










