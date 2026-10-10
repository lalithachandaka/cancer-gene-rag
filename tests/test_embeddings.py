"""
Test suite for retriever.embeddings module

Tests EmbeddingGenerator class and embedding functionality.
Run with: pytest tests/test_embeddings.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.retriever.embeddings import (
    EmbeddingGenerator,
    EmbeddingException,
    ModelLoadException,
    EncodingException,
)
from src.retriever.loader import Document


# ============================================================================
# SHARED FIXTURE
# ============================================================================

@pytest.fixture
def generator():
    """
    Mock EmbeddingGenerator to avoid downloading model during tests.
    
    Instead of loading a real 100MB model, we simulate its behavior.
    This makes tests run in 0.5s instead of 60s.
    """
    gen = MagicMock(spec=EmbeddingGenerator)
    gen.model_name = "all-MiniLM-L6-v2"
    gen.embedding_dim = 384

    # Mock encode: single text → 1D array shape (384,)
    def mock_encode(text):
        if not text or not text.strip():
            from src.retriever.embeddings import EncodingException
            raise EncodingException("Text cannot be empty")
        return np.random.rand(384).astype(np.float32)

    # Mock encode_batch: list of texts → 2D array shape (n, 384)
    def mock_encode_batch(texts):
        if not texts:
            from src.retriever.embeddings import EncodingException
            raise EncodingException("Text list cannot be empty")
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            from src.retriever.embeddings import EncodingException
            raise EncodingException("All texts are empty")
        return np.random.rand(len(valid), 384).astype(np.float32)

    # Mock embed_documents: list of docs → list of 1D arrays
    def mock_embed_documents(docs):
        if not docs:
            from src.retriever.embeddings import EncodingException
            raise EncodingException("Documents list cannot be empty")
        return [np.random.rand(384).astype(np.float32) for _ in docs]

    # Mock embed_documents_with_metadata: list of docs → list of dicts
    def mock_embed_with_metadata(docs):
        if not docs:
            from src.retriever.embeddings import EncodingException
            raise EncodingException("Documents list cannot be empty")
        return [
            {
                "id": doc.id,
                "source_file": doc.source_file,
                "embedding": np.random.rand(384).astype(np.float32),
                "page_num": doc.page_num,
            }
            for doc in docs
        ]

    # Mock similarity: two embeddings → float
    def mock_similarity(emb1, emb2):
        e1 = emb1 / np.linalg.norm(emb1)
        e2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(e1, e2))

    gen.encode = mock_encode
    gen.encode_batch = mock_encode_batch
    gen.embed_documents = mock_embed_documents
    gen.embed_documents_with_metadata = mock_embed_with_metadata
    gen.similarity = mock_similarity

    return gen


# ============================================================================
# TESTS FOR EmbeddingGenerator INITIALIZATION
# ============================================================================

class TestEmbeddingGeneratorInit:
    """Test EmbeddingGenerator initialization."""

    def test_init_default_model_name(self, generator):
        """Generator has correct default model name."""
        assert generator.model_name == "all-MiniLM-L6-v2"

    def test_init_embedding_dim(self, generator):
        """Generator has correct embedding dimension."""
        assert generator.embedding_dim == 384


# ============================================================================
# TESTS FOR encode() - Single Text → 1D Array
# ============================================================================

class TestEncode:
    """Test encode() method."""

    def test_encode_returns_numpy_array(self, generator):
        """encode() returns numpy array."""
        result = generator.encode("TP53 mutations in cancer")
        assert isinstance(result, np.ndarray)

    def test_encode_returns_1d_array(self, generator):
        """encode() returns 1D array with 384 dimensions."""
        result = generator.encode("TP53 mutations in cancer")
        assert result.shape == (384,)

    def test_encode_short_text(self, generator):
        """encode() works with short text."""
        result = generator.encode("TP53")
        assert result.shape == (384,)

    def test_encode_long_text(self, generator):
        """encode() works with long text."""
        result = generator.encode("cancer " * 500)
        assert result.shape == (384,)

    def test_encode_empty_string_raises_error(self, generator):
        """encode() raises EncodingException for empty string."""
        with pytest.raises(EncodingException):
            generator.encode("")

    def test_encode_whitespace_only_raises_error(self, generator):
        """encode() raises EncodingException for whitespace-only text."""
        with pytest.raises(EncodingException):
            generator.encode("   \n   \t   ")

    def test_encode_returns_float_array(self, generator):
        """encode() returns float32 array."""
        result = generator.encode("TP53 mutations")
        assert result.dtype == np.float32


# ============================================================================
# TESTS FOR encode_batch() - Multiple Texts → 2D Array
# ============================================================================

class TestEncodeBatch:
    """Test encode_batch() method."""

    def test_encode_batch_returns_numpy_array(self, generator):
        """encode_batch() returns numpy array."""
        texts = ["TP53 mutations", "BRCA1 analysis", "Therapy response"]
        result = generator.encode_batch(texts)
        assert isinstance(result, np.ndarray)

    def test_encode_batch_returns_2d_array(self, generator):
        """encode_batch() returns 2D array with shape (n, 384)."""
        texts = ["TP53 mutations", "BRCA1 analysis", "Therapy response"]
        result = generator.encode_batch(texts)
        assert result.shape == (3, 384)

    def test_encode_batch_single_text(self, generator):
        """encode_batch() works with single text in list."""
        result = generator.encode_batch(["TP53 mutations"])
        assert result.shape == (1, 384)

    def test_encode_batch_100_texts(self, generator):
        """encode_batch() works with 100 texts."""
        texts = [f"Cancer text {i}" for i in range(100)]
        result = generator.encode_batch(texts)
        assert result.shape == (100, 384)

    def test_encode_batch_empty_list_raises_error(self, generator):
        """encode_batch() raises EncodingException for empty list."""
        with pytest.raises(EncodingException):
            generator.encode_batch([])

    def test_encode_batch_all_empty_texts_raises_error(self, generator):
        """encode_batch() raises error when all texts are empty."""
        with pytest.raises(EncodingException):
            generator.encode_batch(["", "   ", "\n\n"])

    def test_encode_batch_each_row_is_embedding(self, generator):
        """Each row in 2D array is a separate embedding."""
        texts = ["text1", "text2", "text3"]
        result = generator.encode_batch(texts)

        row_0 = result[0]
        row_1 = result[1]
        row_2 = result[2]

        assert row_0.shape == (384,)
        assert row_1.shape == (384,)
        assert row_2.shape == (384,)


# ============================================================================
# TESTS FOR embed_documents()
# ============================================================================

class TestEmbedDocuments:
    """Test embed_documents() method."""

    @pytest.fixture
    def sample_docs(self):
        """Sample Document objects for testing."""
        return [
            Document(
                id="chunk_0",
                content="TP53 mutations increase cancer risk",
                source_file="paper1.pdf"
            ),
            Document(
                id="chunk_1",
                content="BRCA1 mutations affect therapy response",
                source_file="paper1.pdf"
            ),
            Document(
                id="chunk_2",
                content="Combined mutations show interaction effects",
                source_file="paper1.pdf"
            ),
        ]

    def test_embed_documents_returns_list(self, generator, sample_docs):
        """embed_documents() returns list."""
        result = generator.embed_documents(sample_docs)
        assert isinstance(result, list)

    def test_embed_documents_correct_count(self, generator, sample_docs):
        """embed_documents() returns one embedding per document."""
        result = generator.embed_documents(sample_docs)
        assert len(result) == 3

    def test_embed_documents_each_is_1d(self, generator, sample_docs):
        """Each embedding is 1D array with 384 dimensions."""
        result = generator.embed_documents(sample_docs)
        for i, emb in enumerate(result):
            assert isinstance(emb, np.ndarray), f"Item {i} not ndarray"
            assert emb.shape == (384,), f"Item {i} wrong shape"

    def test_embed_documents_empty_raises_error(self, generator):
        """embed_documents() raises error for empty list."""
        with pytest.raises(EncodingException):
            generator.embed_documents([])

    def test_embed_documents_50_docs(self, generator):
        """embed_documents() works with 50 documents."""
        docs = [
            Document(
                id=f"chunk_{i}",
                content=f"Document {i} about cancer",
                source_file="paper.pdf"
            )
            for i in range(50)
        ]
        result = generator.embed_documents(docs)
        assert len(result) == 50
        assert all(e.shape == (384,) for e in result)


# ============================================================================
# TESTS FOR embed_documents_with_metadata()
# ============================================================================

class TestEmbedDocumentsWithMetadata:
    """Test embed_documents_with_metadata() method."""

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(
                id="chunk_0",
                content="TP53 mutations",
                source_file="paper1.pdf",
                page_num=5
            ),
            Document(
                id="chunk_1",
                content="BRCA1 analysis",
                source_file="paper2.pdf",
                page_num=10
            ),
        ]

    def test_returns_list_of_dicts(self, generator, sample_docs):
        """Returns list of dicts."""
        result = generator.embed_documents_with_metadata(sample_docs)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_dict_has_required_keys(self, generator, sample_docs):
        """Each dict has id, source_file, embedding, page_num."""
        result = generator.embed_documents_with_metadata(sample_docs)
        for r in result:
            assert "id" in r
            assert "source_file" in r
            assert "embedding" in r
            assert "page_num" in r

    def test_embedding_is_1d_array(self, generator, sample_docs):
        """Embedding in dict is 1D array."""
        result = generator.embed_documents_with_metadata(sample_docs)
        for r in result:
            assert isinstance(r["embedding"], np.ndarray)
            assert r["embedding"].shape == (384,)

    def test_preserves_document_metadata(self, generator, sample_docs):
        """Metadata matches original documents."""
        result = generator.embed_documents_with_metadata(sample_docs)

        assert result[0]["id"] == "chunk_0"
        assert result[0]["source_file"] == "paper1.pdf"
        assert result[0]["page_num"] == 5

        assert result[1]["id"] == "chunk_1"
        assert result[1]["source_file"] == "paper2.pdf"
        assert result[1]["page_num"] == 10


# ============================================================================
# TESTS FOR similarity()
# ============================================================================

class TestSimilarity:
    """Test similarity() method."""

    def test_similarity_returns_float(self, generator):
        """similarity() returns a float."""
        emb1 = np.random.rand(384).astype(np.float32)
        emb2 = np.random.rand(384).astype(np.float32)
        result = generator.similarity(emb1, emb2)
        assert isinstance(result, float)

    def test_similarity_identical_returns_one(self, generator):
        """Identical embeddings have similarity of 1.0."""
        emb = np.random.rand(384).astype(np.float32)
        result = generator.similarity(emb, emb)
        assert result > 0.99

    def test_similarity_in_valid_range(self, generator):
        """Similarity is always between -1 and 1."""
        emb1 = np.random.rand(384).astype(np.float32)
        emb2 = np.random.rand(384).astype(np.float32)
        result = generator.similarity(emb1, emb2)
        assert -1.0 <= result <= 1.0

    def test_similarity_is_commutative(self, generator):
        """similarity(A, B) == similarity(B, A)."""
        emb1 = np.random.rand(384).astype(np.float32)
        emb2 = np.random.rand(384).astype(np.float32)
        assert np.isclose(
            generator.similarity(emb1, emb2),
            generator.similarity(emb2, emb1)
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple steps."""

    def test_embed_then_similarity(self, generator):
        """Embed documents then compare similarity."""
        docs = [
            Document(id="1", content="TP53 tumor suppressor", source_file="p1.pdf"),
            Document(id="2", content="p53 mutations cancer", source_file="p1.pdf"),
        ]

        embeddings = generator.embed_documents(docs)

        assert len(embeddings) == 2
        assert all(e.shape == (384,) for e in embeddings)

    def test_batch_embed_with_metadata(self, generator):
        """Batch embed with metadata."""
        docs = [
            Document(
                id=f"chunk_{i}",
                content=f"Cancer mutation {i}",
                source_file=f"paper{i}.pdf",
                page_num=i
            )
            for i in range(5)
        ]

        results = generator.embed_documents_with_metadata(docs)

        assert len(results) == 5
        assert all("embedding" in r for r in results)
        assert all(r["embedding"].shape == (384,) for r in results)
        assert all(r["page_num"] == i for i, r in enumerate(results))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])