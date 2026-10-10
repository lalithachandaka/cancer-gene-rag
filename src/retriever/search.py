"""
Search: Query Embedding → Most Relevant Chunks

Takes a query embedding, compares to all stored chunk embeddings,
returns top-K most relevant chunks sorted by similarity.
"""

from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class SearchException(Exception):
    """Base exception for search module."""
    pass


class EmptyIndexException(SearchException):
    """Raised when searching an empty index."""
    pass


class InvalidQueryException(SearchException):
    """Raised when query embedding is invalid."""
    pass


# ============================================================================
# CLASS: VectorSearch
# ============================================================================

class VectorSearch:
    """
    Search over stored chunk embeddings using cosine similarity.
    
    Workflow:
        1. Index: Store chunk embeddings with metadata
        2. Search: Given query embedding, find top-K similar chunks
        3. Return: Ranked list of chunks with similarity scores
    
    Example:
        >>> search = VectorSearch()
        >>> search.index(embeddings_with_metadata)
        >>> results = search.query(query_embedding, top_k=5)
        >>> results[0]["similarity"]
        0.95
    """
    
    def __init__(self) -> None:
        """Initialize empty vector index."""
        # Stores all chunk embeddings with metadata
        self.index: List[Dict] = []
        
        logger.info("VectorSearch initialized (empty index)")
    
    def add_documents(self, embeddings_with_metadata: List[Dict]) -> None:
        """
        Add chunk embeddings to the index.
        
        Args:
            embeddings_with_metadata: List of dicts from
                EmbeddingGenerator.embed_documents_with_metadata()
                Each dict has: id, source_file, embedding, page_num
        
        Example:
            >>> search = VectorSearch()
            >>> search.add_documents(embeddings_with_metadata)
            >>> search.count()
            25  # 25 chunks indexed
        """
        if not embeddings_with_metadata:
            raise SearchException("Cannot index empty list")
        
        # Add each chunk to index
        for item in embeddings_with_metadata:
            self.index.append({
                "id": item["id"],
                "source_file": item["source_file"],
                "embedding": item["embedding"],
                "page_num": item["page_num"],
            })
        
        logger.info(f"Indexed {len(embeddings_with_metadata)} chunks. Total: {len(self.index)}")
    
    def count(self) -> int:
        """Return number of chunks in index."""
        return len(self.index)
    
    def clear(self) -> None:
        """Clear all chunks from index."""
        self.index = []
        logger.info("Index cleared")
    
    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embedding: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between query and document.
        
        Math:
            similarity = dot(A, B) / (norm(A) * norm(B))
        
        Args:
            query_embedding: Query vector (384,)
            doc_embedding: Document vector (384,)
        
        Returns:
            Similarity score between -1 and 1
            Higher = more similar
        
        Example:
            query  = [0.23, 0.45, -0.12, ...]
            doc    = [0.21, 0.44, -0.11, ...]
            result = 0.95  (very similar!)
        """
        # Normalize both vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
        
        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(query_norm, doc_norm)
        
        return float(similarity)
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find top-K most similar chunks to query embedding.
        
        Args:
            query_embedding: Query vector shape (384,)
            top_k: Number of results to return (default 5)
        
        Returns:
            List of dicts sorted by similarity (highest first):
            [
                {
                    "id": "chunk_2",
                    "source_file": "cancer_paper",
                    "page_num": 5,
                    "similarity": 0.98,
                },
                ...
            ]
        
        Raises:
            EmptyIndexException: If no documents indexed
            InvalidQueryException: If query embedding invalid
        
        Example:
            >>> query_emb = generator.encode("TP53 therapy")
            >>> results = search.query(query_emb, top_k=3)
            >>> results[0]["similarity"]
            0.95
        """
        # Validate index
        if not self.index:
            raise EmptyIndexException("No documents indexed. Call add_documents() first")
        
        # Validate query
        if query_embedding is None:
            raise InvalidQueryException("Query embedding cannot be None")
        
        if len(query_embedding) == 0:
            raise InvalidQueryException("Query embedding cannot be empty")
        
        logger.debug(f"Searching {len(self.index)} chunks for top-{top_k} results")
        
        # Step 1: Calculate similarity to EVERY chunk
        results = []
        for chunk in self.index:
            similarity = self._cosine_similarity(
                query_embedding,
                chunk["embedding"]
            )
            
            results.append({
                "id": chunk["id"],
                "source_file": chunk["source_file"],
                "page_num": chunk["page_num"],
                "similarity": similarity,
            })
        
        # Step 2: Sort by similarity (highest first)
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        # Step 3: Return top-K
        top_results = results[:top_k]
        
        logger.info(f"Found top-{top_k} results. Best similarity: {top_results[0]['similarity']:.3f}")
        
        return top_results
    
    def query_with_content(
        self,
        query_embedding: np.ndarray,
        documents: List,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find top-K chunks AND return their content.
        
        Args:
            query_embedding: Query vector shape (384,)
            documents: Original Document objects
            top_k: Number of results to return
        
        Returns:
            List of dicts with content included:
            [
                {
                    "id": "chunk_2",
                    "source_file": "cancer_paper",
                    "similarity": 0.98,
                    "content": "TP53 mutations affect therapy..."
                },
                ...
            ]
        
        Example:
            >>> results = search.query_with_content(query_emb, chunks, top_k=5)
            >>> results[0]["content"]
            "TP53 mutations affect therapy response..."
        """
        # Get top results
        top_results = self.query(query_embedding, top_k=top_k)
        
        # Create lookup: id → document content
        doc_lookup = {doc.id: doc.content for doc in documents}
        
        # Add content to results
        for result in top_results:
            chunk_id = result["id"]
            result["content"] = doc_lookup.get(chunk_id, "Content not found")
        
        return top_results
    
    def format_context(
        self,
        results: List[Dict],
        max_chars: int = 2000
    ) -> str:
        """
        Format search results into context string for LLM.
        
        This is the final step before passing to Claude API.
        
        Args:
            results: List of dicts from query_with_content()
            max_chars: Maximum characters for LLM context
        
        Returns:
            Formatted string ready for Claude API
        
        Example:
            >>> context = search.format_context(results)
            >>> print(context)
            [Source: cancer_paper, Page: 5, Similarity: 0.98]
            TP53 mutations affect therapy response...
            
            [Source: cancer_paper, Page: 3, Similarity: 0.95]
            Tumor suppressor genes regulate...
        """
        if not results:
            return "No relevant content found."
        
        # Build context string
        context_parts = []
        
        for result in results:
            # Format each chunk with metadata
            chunk_text = (
                f"[Source: {result['source_file']}, "
                f"Page: {result.get('page_num', 'N/A')}, "
                f"Similarity: {result['similarity']:.2f}]\n"
                f"{result.get('content', '')}"
            )
            context_parts.append(chunk_text)
        
        # Join all chunks
        context = "\n\n".join(context_parts)
        
        # Limit to max_chars for LLM input
        if len(context) > max_chars:
            context = context[:max_chars]
            logger.debug(f"Context truncated to {max_chars} chars")
        
        # Clean up whitespace
        context = context.replace("\n\n\n", "\n\n").strip()
        
        return context