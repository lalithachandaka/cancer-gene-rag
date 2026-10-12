"""
RAG Pipeline: Connects loader → embeddings → search → claude_api

Single entry point for the full RAG workflow.
Caches embeddings so PDF is only processed once.
"""

from typing import List, Dict, Optional
import logging
import os

from src.retriever.loader import DocumentLoader, Document
from src.retriever.embeddings import EmbeddingGenerator
from src.retriever.search import VectorSearch
from src.generator.claude_api import ClaudeAPI
from src.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class PipelineException(Exception):
    """Base exception for pipeline module."""
    pass


class PDFNotLoadedException(PipelineException):
    """Raised when query called before PDF loaded."""
    pass


class PDFAlreadyLoadedException(PipelineException):
    """Raised when PDF already loaded."""
    pass


# ============================================================================
# CLASS: RAGPipeline
# ============================================================================

class RAGPipeline:
    """
    Full RAG pipeline connecting all components.
    
    Workflow:
        1. load_pdf(): Load PDF → chunks → embeddings → index
        2. query(): Search index → retrieve chunks → generate answer
    
    Caching:
        PDF is embedded ONCE and cached in memory.
        Multiple queries reuse cached embeddings.
    
    Example:
        >>> pipeline = RAGPipeline()
        >>> pipeline.load_pdf("cancer_paper.pdf")
        >>> result = pipeline.query("How does TP53 affect therapy?")
        >>> result["answer"]
        "Based on the research, TP53 mutations..."
        >>> result["cost_usd"]
        0.003
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        top_k: int = 5,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize RAG pipeline with all components.
        
        Args:
            chunk_size: Characters per chunk for loader
            overlap: Overlap between chunks
            top_k: Number of chunks to retrieve per query
            max_tokens: Max tokens in Claude response
        
        Example:
            >>> pipeline = RAGPipeline(chunk_size=512, top_k=5)
        """
        # Initialize all components
        self.loader = DocumentLoader(
            chunk_size=chunk_size,
            overlap=overlap
        )
        self.embedder = EmbeddingGenerator()
        self.search = VectorSearch()
        self.claude = ClaudeAPI(
            api_key=ANTHROPIC_API_KEY,
            max_tokens=max_tokens
        )
        
        # Pipeline state
        self.is_indexed: bool = False
        self.current_pdf: Optional[str] = None
        self.documents: List[Document] = []
        self.top_k: int = top_k
        
        logger.info("RAGPipeline initialized")
    
    def load_pdf(self, filepath: str) -> Dict:
        """
        Load PDF, create embeddings, index for search.
        
        This is the EXPENSIVE step - run once per PDF.
        Embeddings are cached in self.search for reuse.
        
        Args:
            filepath: Path to PDF file
        
        Returns:
            Dict with:
                chunks_created: Number of chunks
                filepath: Path to loaded PDF
        
        Raises:
            PipelineException: If PDF loading fails
        
        Example:
            >>> result = pipeline.load_pdf("cancer_paper.pdf")
            >>> result["chunks_created"]
            45
        """
        try:
            logger.info(f"Loading PDF: {filepath}")
            
            # Step 1: Load PDF → chunks
            logger.info("Step 1/3: Loading and chunking PDF...")
            self.documents = self.loader.load_pdf(filepath)
            
            if not self.documents:
                raise PipelineException(f"No chunks created from {filepath}")
            
            # Step 2: chunks → embeddings
            logger.info(f"Step 2/3: Embedding {len(self.documents)} chunks...")
            embeddings_with_metadata = self.embedder.embed_documents_with_metadata(
                self.documents
            )
            
            # Step 3: Store embeddings in search index
            logger.info("Step 3/3: Indexing embeddings...")
            self.search.clear()  # Clear any previous PDF
            self.search.add_documents(embeddings_with_metadata)
            
            # Update state
            self.is_indexed = True
            self.current_pdf = filepath
            
            logger.info(
                f"PDF loaded successfully: "
                f"{len(self.documents)} chunks indexed"
            )
            
            return {
                "chunks_created": len(self.documents),
                "filepath": filepath,
            }
        
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise PipelineException(f"PDF loading failed: {e}") from e
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG pipeline with a question.
        
        Uses cached embeddings - no re-embedding needed.
        
        Args:
            question: User question about the PDF
        
        Returns:
            Dict with:
                answer: Claude's answer
                sources: List of source chunks used
                cost_usd: Cost of this query
                input_tokens: Tokens in prompt
                output_tokens: Tokens in response
        
        Raises:
            PDFNotLoadedException: If no PDF loaded yet
            PipelineException: If query fails
        
        Example:
            >>> result = pipeline.query("How does TP53 affect therapy?")
            >>> result["answer"]
            "Based on the research..."
            >>> result["sources"]
            ["cancer_paper chunk 0", "cancer_paper chunk 2"]
        """
        # Check PDF is loaded
        if not self.is_indexed:
            raise PDFNotLoadedException(
                "No PDF loaded. Call load_pdf() first."
            )
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            
            # Step 1: Embed query
            logger.debug("Step 1/3: Embedding query...")
            query_embedding = self.embedder.encode(question)
            
            # Step 2: Search for relevant chunks
            logger.debug(f"Step 2/3: Searching top-{self.top_k} chunks...")
            results = self.search.query_with_content(
                query_embedding,
                self.documents,
                top_k=self.top_k
            )
            
            # Step 3: Format context for Claude
            context = self.search.format_context(results, max_chars=3000)
            
            # Step 4: Generate answer with Claude
            logger.debug("Step 3/3: Generating answer with Claude...")
            response = self.claude.generate(
                query=question,
                context=context
            )
            
            # Build sources list
            sources = [
                {
                    "id": r["id"],
                    "source_file": r["source_file"],
                    "page_num": r["page_num"],
                    "similarity": round(r["similarity"], 3),
                }
                for r in results
            ]
            
            logger.info(
                f"Query complete. "
                f"Cost: ${response['cost_usd']:.4f}"
            )
            
            return {
                "answer": response["text"],
                "sources": sources,
                "cost_usd": response["cost_usd"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
            }
        
        except PDFNotLoadedException:
            raise
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise PipelineException(f"Query failed: {e}") from e
    
    def get_stats(self) -> Dict:
        """
        Return pipeline statistics.
        
        Returns:
            Dict with usage stats and pipeline state
        
        Example:
            >>> pipeline.get_stats()
            {
                "is_indexed": True,
                "current_pdf": "cancer_paper.pdf",
                "chunks_indexed": 45,
                "total_queries": 10,
                "total_cost_usd": 0.045,
            }
        """
        usage = self.claude.get_usage_stats()
        
        return {
            "is_indexed": self.is_indexed,
            "current_pdf": self.current_pdf,
            "chunks_indexed": self.search.count(),
            "total_queries": usage["total_queries"],
            "total_input_tokens": usage["total_input_tokens"],
            "total_output_tokens": usage["total_output_tokens"],
            "total_cost_usd": usage["total_cost_usd"],
            "avg_cost_per_query": usage["avg_cost_per_query"],
        }