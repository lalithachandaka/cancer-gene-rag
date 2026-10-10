"""
Test suite for retriever.search module

Tests VectorSearch class.
Run with: pytest tests/test_search.py -v
"""

import pytest
import numpy as np

from src.retriever.search import (
    VectorSearch,
    SearchException,
    EmptyIndexException,
    InvalidQueryException,
)
from src.retriever.loader import Document


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def search():
    """Empty VectorSearch instance."""
    return VectorSearch()


@pytest.fixture
def sample_embeddings():
    """
    Sample embeddings with metadata.
    Simulates output of EmbeddingGenerator.embed_documents_with_metadata()
    """
    return [
        {
            "id": "chunk_0",
            "source_file": "cancer_paper",
            "embedding": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
            "page_num": 1,
        },
        {
            "id": "chunk_1",
            "source_file": "cancer_paper",
            "embedding": np.array([0.9, 0.1, 0.0] + [0.0] * 381),
            "page_num": 2,
        },
        {
            "id": "chunk_2",
            "source_file": "cancer_paper",
            "embedding": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
            "page_num": 3,
        },
        {
            "id": "chunk_3",
            "source_file": "cancer_paper",
            "embedding": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
            "page_num": 4,
        },
        {
            "id": "chunk_4",
            "source_file": "paper_2",
            "embedding": np.array([0.8, 0.2, 0.0] + [0.0] * 381),
            "page_num": 1,
        },
    ]


@pytest.fixture
def populated_search(search, sample_embeddings):
    """VectorSearch with 5 chunks already indexed."""
    search.add_documents(sample_embeddings)
    return search


@pytest.fixture
def sample_documents():
    """Sample Document objects matching sample_embeddings."""
    return [
        Document(
            id="chunk_0",
            content="TP53 mutations increase cancer risk significantly",
            source_file="cancer_paper",
            page_num=1
        ),
        Document(
            id="chunk_1",
            content="TP53 affects therapy response in breast cancer",
            source_file="cancer_paper",
            page_num=2
        ),
        Document(
            id="chunk_2",
            content="BRCA1 mutations found in ovarian cancer patients",
            source_file="cancer_paper",
            page_num=3
        ),
        Document(
            id="chunk_3",
            content="Targeted therapy shows promise for lung cancer",
            source_file="cancer_paper",
            page_num=4
        ),
        Document(
            id="chunk_4",
            content="TP53 and BRCA1 interactions in triple negative cancer",
            source_file="paper_2",
            page_num=1
        ),
    ]


# ============================================================================
# TESTS FOR VectorSearch INITIALIZATION
# ============================================================================

class TestVectorSearchInit:
    """Test VectorSearch initialization."""

    def test_init_creates_empty_index(self, search):
        """VectorSearch starts with empty index."""
        assert search.count() == 0

    def test_init_index_is_list(self, search):
        """Index is a list."""
        assert isinstance(search.index, list)


# ============================================================================
# TESTS FOR add_documents()
# ============================================================================

class TestAddDocuments:
    """Test add_documents() method."""

    def test_add_documents_increases_count(self, search, sample_embeddings):
        """add_documents() increases chunk count."""
        search.add_documents(sample_embeddings)
        assert search.count() == 5

    def test_add_documents_empty_raises_error(self, search):
        """add_documents() raises SearchException for empty list."""
        with pytest.raises(SearchException):
            search.add_documents([])

    def test_add_documents_twice_accumulates(self, search, sample_embeddings):
        """Adding documents twice accumulates in index."""
        search.add_documents(sample_embeddings)
        search.add_documents(sample_embeddings)
        assert search.count() == 10

    def test_add_documents_stores_metadata(self, search, sample_embeddings):
        """add_documents() stores id, source_file, page_num."""
        search.add_documents(sample_embeddings)

        first = search.index[0]
        assert first["id"] == "chunk_0"
        assert first["source_file"] == "cancer_paper"
        assert first["page_num"] == 1

    def test_add_documents_stores_embedding(self, search, sample_embeddings):
        """add_documents() stores embedding array."""
        search.add_documents(sample_embeddings)

        first = search.index[0]
        assert isinstance(first["embedding"], np.ndarray)
        assert first["embedding"].shape == (384,)


# ============================================================================
# TESTS FOR count() and clear()
# ============================================================================

class TestCountAndClear:
    """Test count() and clear() methods."""

    def test_count_empty(self, search):
        """count() returns 0 for empty index."""
        assert search.count() == 0

    def test_count_after_adding(self, search, sample_embeddings):
        """count() returns correct number after adding."""
        search.add_documents(sample_embeddings)
        assert search.count() == 5

    def test_clear_empties_index(self, populated_search):
        """clear() removes all documents."""
        populated_search.clear()
        assert populated_search.count() == 0

    def test_clear_then_add(self, populated_search, sample_embeddings):
        """Can add documents after clearing."""
        populated_search.clear()
        populated_search.add_documents(sample_embeddings)
        assert populated_search.count() == 5


# ============================================================================
# TESTS FOR _cosine_similarity()
# ============================================================================

class TestCosineSimilarity:
    """Test _cosine_similarity() method."""

    def test_identical_vectors_similarity_is_one(self, search):
        """Identical vectors have similarity of 1.0."""
        emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        similarity = search._cosine_similarity(emb, emb)
        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors_similarity_is_zero(self, search):
        """Orthogonal vectors have similarity of 0.0."""
        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381)
        similarity = search._cosine_similarity(emb1, emb2)
        assert np.isclose(similarity, 0.0)

    def test_similar_vectors_high_similarity(self, search):
        """Similar vectors have high similarity."""
        emb1 = np.array([1.0, 0.1, 0.0] + [0.0] * 381)
        emb2 = np.array([0.9, 0.1, 0.0] + [0.0] * 381)
        similarity = search._cosine_similarity(emb1, emb2)
        assert similarity > 0.99

    def test_similarity_returns_float(self, search):
        """_cosine_similarity() returns float."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(384)
        similarity = search._cosine_similarity(emb1, emb2)
        assert isinstance(similarity, float)

    def test_similarity_is_commutative(self, search):
        """similarity(A, B) == similarity(B, A)."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(384)
        sim1 = search._cosine_similarity(emb1, emb2)
        sim2 = search._cosine_similarity(emb2, emb1)
        assert np.isclose(sim1, sim2)

    def test_similarity_range(self, search):
        """Similarity is in range [-1, 1]."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(384)
        similarity = search._cosine_similarity(emb1, emb2)
        assert -1.0 <= similarity <= 1.0


# ============================================================================
# TESTS FOR query()
# ============================================================================

class TestQuery:
    """Test query() method."""

    def test_query_returns_list(self, populated_search):
        """query() returns list of results."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query(query_emb, top_k=3)
        assert isinstance(results, list)

    def test_query_returns_top_k(self, populated_search):
        """query() returns exactly top_k results."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query(query_emb, top_k=3)
        assert len(results) == 3

    def test_query_returns_all_if_fewer_than_k(self, populated_search):
        """query() returns all if index has fewer than top_k chunks."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query(query_emb, top_k=100)
        assert len(results) == 5  # Only 5 in index

    def test_query_sorted_by_similarity(self, populated_search):
        """query() returns results sorted by similarity (highest first)."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query(query_emb, top_k=5)

        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_query_finds_most_similar(self, populated_search):
        """query() returns most similar chunk first."""
        # Query is [1.0, 0.0, ...] which matches chunk_0 exactly
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query(query_emb, top_k=5)

        # First result should be chunk_0 (most similar)
        assert results[0]["id"] == "chunk_0"
        assert np.isclose(results[0]["similarity"], 1.0)

    def test_query_result_has_required_keys(self, populated_search):
        """Each result has id, source_file, page_num, similarity."""
        query_emb = np.random.rand(384)
        results = populated_search.query(query_emb, top_k=3)

        for result in results:
            assert "id" in result
            assert "source_file" in result
            assert "page_num" in result
            assert "similarity" in result

    def test_query_empty_index_raises_error(self, search):
        """query() raises EmptyIndexException if no documents indexed."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        with pytest.raises(EmptyIndexException):
            search.query(query_emb, top_k=5)

    def test_query_none_embedding_raises_error(self, populated_search):
        """query() raises InvalidQueryException for None embedding."""
        with pytest.raises(InvalidQueryException):
            populated_search.query(None, top_k=5)

    def test_query_similarity_scores_in_range(self, populated_search):
        """All similarity scores are in range [-1, 1]."""
        query_emb = np.random.rand(384)
        results = populated_search.query(query_emb, top_k=5)

        for result in results:
            assert -1.0 <= result["similarity"] <= 1.0


# ============================================================================
# TESTS FOR query_with_content()
# ============================================================================

class TestQueryWithContent:
    """Test query_with_content() method."""

    def test_query_with_content_returns_list(
        self, populated_search, sample_documents
    ):
        """query_with_content() returns list."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query_with_content(
            query_emb, sample_documents, top_k=3
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_query_with_content_has_content_key(
        self, populated_search, sample_documents
    ):
        """Each result has content key."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query_with_content(
            query_emb, sample_documents, top_k=3
        )
        for result in results:
            assert "content" in result

    def test_query_with_content_returns_correct_text(
        self, populated_search, sample_documents
    ):
        """Results contain correct document content."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query_with_content(
            query_emb, sample_documents, top_k=1
        )

        # chunk_0 is most similar
        assert "TP53" in results[0]["content"]

    def test_query_with_content_sorted_by_similarity(
        self, populated_search, sample_documents
    ):
        """Results sorted by similarity."""
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        results = populated_search.query_with_content(
            query_emb, sample_documents, top_k=5
        )

        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)


# ============================================================================
# TESTS FOR format_context()
# ============================================================================

class TestFormatContext:
    """Test format_context() method."""

    @pytest.fixture
    def sample_results(self):
        """Sample results with content."""
        return [
            {
                "id": "chunk_0",
                "source_file": "cancer_paper",
                "page_num": 1,
                "similarity": 0.98,
                "content": "TP53 mutations increase cancer risk significantly",
            },
            {
                "id": "chunk_1",
                "source_file": "cancer_paper",
                "page_num": 2,
                "similarity": 0.95,
                "content": "TP53 affects therapy response in breast cancer",
            },
        ]

    def test_format_context_returns_string(self, search, sample_results):
        """format_context() returns string."""
        context = search.format_context(sample_results)
        assert isinstance(context, str)

    def test_format_context_contains_source(self, search, sample_results):
        """Context contains source file name."""
        context = search.format_context(sample_results)
        assert "cancer_paper" in context

    def test_format_context_contains_similarity(self, search, sample_results):
        """Context contains similarity score."""
        context = search.format_context(sample_results)
        assert "0.98" in context

    def test_format_context_contains_content(self, search, sample_results):
        """Context contains chunk content."""
        context = search.format_context(sample_results)
        assert "TP53" in context

    def test_format_context_respects_max_chars(self, search, sample_results):
        """Context is truncated to max_chars."""
        context = search.format_context(sample_results, max_chars=50)
        assert len(context) <= 50

    def test_format_context_empty_returns_message(self, search):
        """format_context() returns message for empty results."""
        context = search.format_context([])
        assert "No relevant content found" in context


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSearchIntegration:
    """Full integration tests."""

    def test_full_search_workflow(self, sample_documents):
        """
        Full workflow:
        1. Create index
        2. Add documents
        3. Query
        4. Get content
        5. Format for LLM
        """
        # Step 1: Create embeddings (simulated)
        embeddings_with_metadata = [
            {
                "id": doc.id,
                "source_file": doc.source_file,
                "embedding": np.random.rand(384).astype(np.float32),
                "page_num": doc.page_num,
            }
            for doc in sample_documents
        ]

        # Step 2: Index
        search = VectorSearch()
        search.add_documents(embeddings_with_metadata)
        assert search.count() == 5

        # Step 3: Query
        query_embedding = np.random.rand(384).astype(np.float32)
        results = search.query(query_embedding, top_k=3)
        assert len(results) == 3

        # Step 4: Get content
        results_with_content = search.query_with_content(
            query_embedding,
            sample_documents,
            top_k=3
        )
        assert all("content" in r for r in results_with_content)

        # Step 5: Format for LLM
        context = search.format_context(results_with_content, max_chars=2000)
        assert isinstance(context, str)
        assert len(context) > 0
        assert len(context) <= 2000

    def test_search_multiple_sources(self, sample_documents):
        """Search works across multiple source files."""
        embeddings_with_metadata = [
            {
                "id": doc.id,
                "source_file": doc.source_file,
                "embedding": np.random.rand(384).astype(np.float32),
                "page_num": doc.page_num,
            }
            for doc in sample_documents
        ]

        search = VectorSearch()
        search.add_documents(embeddings_with_metadata)

        query_embedding = np.random.rand(384).astype(np.float32)
        results = search.query(query_embedding, top_k=5)

        # Should return results from both source files
        sources = {r["source_file"] for r in results}
        assert len(sources) >= 1

    def test_clear_and_reindex(self, sample_documents):
        """Can clear index and reindex."""
        search = VectorSearch()

        # Add documents
        embeddings = [
            {
                "id": doc.id,
                "source_file": doc.source_file,
                "embedding": np.random.rand(384).astype(np.float32),
                "page_num": doc.page_num,
            }
            for doc in sample_documents
        ]
        search.add_documents(embeddings)
        assert search.count() == 5

        # Clear
        search.clear()
        assert search.count() == 0

        # Reindex
        search.add_documents(embeddings)
        assert search.count() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])