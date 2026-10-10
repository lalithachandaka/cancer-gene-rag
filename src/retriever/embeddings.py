"""
Embedding Generator: Convert text chunks to vector embeddings.

Uses sentence-transformers to convert text to semantic embeddings.
Embeddings capture meaning, allowing semantic search.
"""

from pydoc import text
from typing import List, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from src.retriever.loader import Document

logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class EmbeddingException(Exception):
    """Base exception for embedding module."""
    pass


class ModelLoadException(EmbeddingException):
    """Raised when embedding model fails to load."""
    pass


class EncodingException(EmbeddingException):
    """Raised when text encoding fails."""
    pass


# ============================================================================
# CLASS: EmbeddingGenerator
# ============================================================================

class EmbeddingGenerator:
    """
    Convert text chunks to vector embeddings.
    
    Uses pre-trained sentence-transformers model to convert text to embeddings.
    
    Embeddings capture semantic meaning:
    - Similar texts have similar embeddings
    - Enables semantic search (not just keyword matching)
    
    Attributes:
        model_name: Name of sentence-transformers model to use
        model: Loaded SentenceTransformer model
        embedding_dim: Dimension of embeddings (e.g., 384 for MiniLM)
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embedding = generator.encode("TP53 mutations in cancer")
        >>> embedding.shape
        (384,)
        >>> embeddings = generator.encode_batch(["text1", "text2"])
        >>> embeddings.shape
        (2, 384)
    """
    
    # Common pre-trained models
    MODELS = {
        "mini": "all-MiniLM-L6-v2",      # Small, fast (384 dim)
        "base": "all-mpnet-base-v2",     # Larger, more accurate (768 dim)
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize EmbeddingGenerator with specified model.
        
        Args:
            model_name: Key for pre-trained model to use ("mini" or "base")
        
        Raises:
            ModelLoadException: If model fails to load
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            
            # Determine embedding dimension
            dummy_embedding = self.model.encode("test")
            self.embedding_dim = len(dummy_embedding)
            
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadException(f"Cannot load model {model_name}") from e
        
        def encode(self, text: str) -> np.ndarray:

            try:
                if not text or not text.strip():
                    raise ValueError("Text cannot be empty")
                
                logger.debug(f"Encoding single text: {len(text)} chars")
                embedding = self.model.encode(text)
                
                return embedding
            
            except Exception as e:
                logger.error(f"Encoding failed: {e}")
                raise EncodingException(f"Failed to encode text") from e
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings (batch processing).
        
        Batch processing is MUCH faster than encoding individually.
        
        Args:
            texts: List of text strings to encode
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
            Example: array([[0.2, 0.8, ...], [0.1, 0.9, ...]])
        
        Raises:
            EncodingException: If encoding fails
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["TP53 mutations", "BRCA1 analysis", "Therapy response"]
            >>> embeddings = generator.encode_batch(texts)
            >>> embeddings.shape
            (3, 384)
            >>> embeddings[0].shape
            (384,)
        
        Performance:
            - Encoding 100 texts individually: ~2 seconds
            - Encoding 100 texts as batch: ~0.3 seconds (6x faster)
        """
        try:
            if not texts:
                raise ValueError("Text list cannot be empty")
            
            # Filter out empty strings
            non_empty_texts = [t for t in texts if t and t.strip()]
            
            if not non_empty_texts:
                raise ValueError("All texts are empty")
            
            logger.debug(f"Encoding batch: {len(non_empty_texts)} texts")
            embeddings = self.model.encode(non_empty_texts)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            raise EncodingException(f"Failed to encode batch") from e
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Convert Document chunks to embeddings.
        
        Takes Document objects, extracts content, encodes to embeddings.
        
        Args:
            documents: List of Document objects to embed
        
        Returns:
            List of numpy arrays (embeddings)
            Each array has shape (embedding_dim,)
        
        Raises:
            EncodingException: If encoding fails
        
        Example:
            >>> from src.retriever.loader import Document
            >>> docs = [
            ...     Document(id="1", content="TP53 mutations", source_file="paper1"),
            ...     Document(id="2", content="BRCA1 analysis", source_file="paper1"),
            ... ]
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.embed_documents(docs)
            >>> len(embeddings)
            2
            >>> embeddings[0].shape
            (384,)
        """
        try:
            if not documents:
                raise ValueError("Documents list cannot be empty")
            
            logger.info(f"Embedding {len(documents)} documents")
            
            # Extract text from documents
            texts = [doc.content for doc in documents]
            
            # Encode in batch (faster than individual)
            embeddings = self.encode_batch(texts)
            
            # Convert to list of individual embeddings
            embedding_list = [embeddings[i] for i in range(len(documents))]
            
            logger.info(f"Embedded {len(embedding_list)} documents successfully")
            
            return embedding_list
        
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise EncodingException(f"Failed to embed documents") from e
    
    def embed_documents_with_metadata(
        self, 
        documents: List[Document]
    ) -> List[dict]:
        """
        Convert Documents to embeddings WITH metadata.
        
        Returns dict with both embedding and source info.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of dicts: [
                {
                    "id": "chunk_0",
                    "source_file": "paper1.pdf",
                    "embedding": array([...]),
                },
                ...
            ]
        
        Example:
            >>> docs = [Document(id="1", content="text", source_file="paper1")]
            >>> results = generator.embed_documents_with_metadata(docs)
            >>> results[0]["id"]
            '1'
            >>> results[0]["embedding"].shape
            (384,)
        """
        try:
            embeddings = self.embed_documents(documents)
            
            results = []
            for doc, embedding in zip(documents, embeddings):
                results.append({
                    "id": doc.id,
                    "source_file": doc.source_file,
                    "embedding": embedding,
                    "page_num": doc.page_num,
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Metadata embedding failed: {e}")
            raise EncodingException(f"Failed to embed with metadata") from e
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Similarity measures how "close" two texts are semantically.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score between 0 and 1
            0.0 = completely different
            1.0 = identical
        
        Example:
            >>> embedding1 = generator.encode("TP53 mutations")
            >>> embedding2 = generator.encode("p53 mutations")
            >>> similarity = generator.similarity(embedding1, embedding2)
            >>> similarity
            0.95  (very similar)
        
        Math:
            cosine_similarity = dot_product(A, B) / (norm(A) * norm(B))
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate dot product (cosine similarity for normalized vectors)
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    