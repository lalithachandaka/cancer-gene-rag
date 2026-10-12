"""
Test suite for pipeline.rag module

Tests RAGPipeline class with mocked components.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.pipeline.rag import (
    RAGPipeline,
    PipelineException,
    PDFNotLoadedException,
)
from src.retriever.loader import Document


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def mock_documents():
    """Sample Document objects simulating PDF chunks."""
    return [
        Document(
            id="chunk_0",
            content="TP53 mutations reduce apoptosis in cancer cells",
            source_file="cancer_paper",
            page_num=1
        ),
        Document(
            id="chunk_1",
            content="BRCA1 mutations affect therapy response",
            source_file="cancer_paper",
            page_num=2
        ),
        Document(
            id="chunk_2",
            content="Targeted therapy shows promise for TP53 mutations",
            source_file="cancer_paper",
            page_num=3
        ),
    ]


@pytest.fixture
def mock_embeddings_with_metadata(mock_documents):
    """Sample embeddings with metadata."""
    return [
        {
            "id": doc.id,
            "source_file": doc.source_file,
            "embedding": np.random.rand(384).astype(np.float32),
            "page_num": doc.page_num,
        }
        for doc in mock_documents
    ]


@pytest.fixture
def mock_search_results():
    """Sample search results with content."""
    return [
        {
            "id": "chunk_0",
            "source_file": "cancer_paper",
            "page_num": 1,
            "similarity": 0.98,
            "content": "TP53 mutations reduce apoptosis in cancer cells",
        },
        {
            "id": "chunk_2",
            "source_file": "cancer_paper",
            "page_num": 3,
            "similarity": 0.95,
            "content": "Targeted therapy shows promise for TP53 mutations",
        },
    ]


@pytest.fixture
def mock_claude_response():
    """Sample Claude API response."""
    return {
        "text": "Based on the research, TP53 mutations reduce apoptosis...",
        "input_tokens": 500,
        "output_tokens": 200,
        "cost_usd": 0.0045,
        "model": "claude-3-5-sonnet-20241022",
    }


@pytest.fixture
def pipeline(
    mock_documents,
    mock_embeddings_with_metadata,
    mock_search_results,
    mock_claude_response,
):
    """
    RAGPipeline with all components mocked.
    Avoids real API calls and model downloads.
    """
    with patch("src.pipeline.rag.DocumentLoader") as mock_loader_cls, \
         patch("src.pipeline.rag.EmbeddingGenerator") as mock_embedder_cls, \
         patch("src.pipeline.rag.VectorSearch") as mock_search_cls, \
         patch("src.pipeline.rag.ClaudeAPI") as mock_claude_cls, \
         patch("src.pipeline.rag.ANTHROPIC_API_KEY", "fake-key"):

        # Setup loader mock
        mock_loader = MagicMock()
        mock_loader.load_pdf.return_value = mock_documents
        mock_loader_cls.return_value = mock_loader

        # Setup embedder mock
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_embedder.embed_documents_with_metadata.return_value = (
            mock_embeddings_with_metadata
        )
        mock_embedder_cls.return_value = mock_embedder

        # Setup search mock
        mock_search = MagicMock()
        mock_search.count.return_value = 3
        mock_search.query_with_content.return_value = mock_search_results
        mock_search.format_context.return_value = (
            "[Source: cancer_paper, Page: 1, Similarity: 0.98]\n"
            "TP53 mutations reduce apoptosis..."
        )
        mock_search_cls.return_value = mock_search

        # Setup Claude mock
        mock_claude = MagicMock()
        mock_claude.generate.return_value = mock_claude_response
        mock_claude.get_usage_stats.return_value = {
            "total_queries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_query": 0.0,
        }
        mock_claude_cls.return_value = mock_claude

        # Create pipeline
        p = RAGPipeline()

        # Store mocks for test access
        p._mock_loader = mock_loader
        p._mock_embedder = mock_embedder
        p._mock_search = mock_search
        p._mock_claude = mock_claude

        yield p


# ============================================================================
# TESTS FOR RAGPipeline INITIALIZATION
# ============================================================================

class TestRAGPipelineInit:
    """Test RAGPipeline initialization."""

    def test_init_not_indexed(self, pipeline):
        """Pipeline starts not indexed."""
        assert pipeline.is_indexed is False

    def test_init_no_current_pdf(self, pipeline):
        """Pipeline starts with no PDF loaded."""
        assert pipeline.current_pdf is None

    def test_init_empty_documents(self, pipeline):
        """Pipeline starts with empty document list."""
        assert pipeline.documents == []

    def test_init_default_top_k(self, pipeline):
        """Pipeline starts with default top_k."""
        assert pipeline.top_k == 5


# ============================================================================
# TESTS FOR load_pdf()
# ============================================================================

class TestLoadPDF:
    """Test load_pdf() method."""

    def test_load_pdf_returns_dict(self, pipeline):
        """load_pdf() returns dict."""
        result = pipeline.load_pdf("cancer_paper.pdf")
        assert isinstance(result, dict)

    def test_load_pdf_returns_chunk_count(self, pipeline):
        """load_pdf() returns number of chunks created."""
        result = pipeline.load_pdf("cancer_paper.pdf")
        assert "chunks_created" in result
        assert result["chunks_created"] == 3

    def test_load_pdf_returns_filepath(self, pipeline):
        """load_pdf() returns filepath."""
        result = pipeline.load_pdf("cancer_paper.pdf")
        assert "filepath" in result
        assert result["filepath"] == "cancer_paper.pdf"

    def test_load_pdf_sets_is_indexed(self, pipeline):
        """load_pdf() sets is_indexed to True."""
        pipeline.load_pdf("cancer_paper.pdf")
        assert pipeline.is_indexed is True

    def test_load_pdf_sets_current_pdf(self, pipeline):
        """load_pdf() sets current_pdf."""
        pipeline.load_pdf("cancer_paper.pdf")
        assert pipeline.current_pdf == "cancer_paper.pdf"

    def test_load_pdf_stores_documents(self, pipeline):
        """load_pdf() stores document chunks."""
        pipeline.load_pdf("cancer_paper.pdf")
        assert len(pipeline.documents) == 3

    def test_load_pdf_calls_loader(self, pipeline):
        """load_pdf() calls DocumentLoader.load_pdf()."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline._mock_loader.load_pdf.assert_called_once_with("cancer_paper.pdf")

    def test_load_pdf_calls_embedder(self, pipeline):
        """load_pdf() calls EmbeddingGenerator.embed_documents_with_metadata()."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline._mock_embedder.embed_documents_with_metadata.assert_called_once()

    def test_load_pdf_calls_search_index(self, pipeline):
        """load_pdf() calls VectorSearch.add_documents()."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline._mock_search.add_documents.assert_called_once()

    def test_load_pdf_clears_previous_index(self, pipeline):
        """load_pdf() clears previous index before loading."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline._mock_search.clear.assert_called_once()


# ============================================================================
# TESTS FOR query()
# ============================================================================

class TestQuery:
    """Test query() method."""

    def test_query_without_pdf_raises_error(self, pipeline):
        """query() raises PDFNotLoadedException if no PDF loaded."""
        with pytest.raises(PDFNotLoadedException):
            pipeline.query("How does TP53 affect therapy?")

    def test_query_returns_dict(self, pipeline):
        """query() returns dict."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")
        assert isinstance(result, dict)

    def test_query_returns_answer(self, pipeline):
        """query() returns answer text."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, pipeline):
        """query() returns source chunks used."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_returns_cost(self, pipeline):
        """query() returns cost."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")
        assert "cost_usd" in result
        assert result["cost_usd"] == 0.0045

    def test_query_returns_token_counts(self, pipeline):
        """query() returns token counts."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 200

    def test_query_empty_question_raises_error(self, pipeline):
        """query() raises ValueError for empty question."""
        pipeline.load_pdf("cancer_paper.pdf")
        with pytest.raises(ValueError):
            pipeline.query("")

    def test_query_whitespace_question_raises_error(self, pipeline):
        """query() raises ValueError for whitespace question."""
        pipeline.load_pdf("cancer_paper.pdf")
        with pytest.raises(ValueError):
            pipeline.query("   ")

    def test_query_calls_embedder(self, pipeline):
        """query() calls embedder to encode question."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline.query("How does TP53 affect therapy?")
        pipeline._mock_embedder.encode.assert_called_once_with(
            "How does TP53 affect therapy?"
        )

    def test_query_calls_search(self, pipeline):
        """query() calls search to find relevant chunks."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline.query("How does TP53 affect therapy?")
        pipeline._mock_search.query_with_content.assert_called_once()

    def test_query_calls_claude(self, pipeline):
        """query() calls Claude API to generate answer."""
        pipeline.load_pdf("cancer_paper.pdf")
        pipeline.query("How does TP53 affect therapy?")
        pipeline._mock_claude.generate.assert_called_once()

    def test_query_sources_have_required_keys(self, pipeline):
        """Each source has id, source_file, page_num, similarity."""
        pipeline.load_pdf("cancer_paper.pdf")
        result = pipeline.query("How does TP53 affect therapy?")

        for source in result["sources"]:
            assert "id" in source
            assert "source_file" in source
            assert "page_num" in source
            assert "similarity" in source

    def test_query_multiple_times_reuses_index(self, pipeline):
        """Multiple queries reuse cached embeddings."""
        pipeline.load_pdf("cancer_paper.pdf")

        # Ask 3 questions
        pipeline.query("Question 1?")
        pipeline.query("Question 2?")
        pipeline.query("Question 3?")

        # loader.load_pdf should only be called ONCE (not 3 times)
        pipeline._mock_loader.load_pdf.assert_called_once()

        # But embedder.encode called 3 times (once per query)
        assert pipeline._mock_embedder.encode.call_count == 3


# ============================================================================
# TESTS FOR get_stats()
# ============================================================================

class TestGetStats:
    """Test get_stats() method."""

    def test_get_stats_returns_dict(self, pipeline):
        """get_stats() returns dict."""
        stats = pipeline.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_required_keys(self, pipeline):
        """get_stats() has all required keys."""
        stats = pipeline.get_stats()
        assert "is_indexed" in stats
        assert "current_pdf" in stats
        assert "chunks_indexed" in stats
        assert "total_queries" in stats
        assert "total_cost_usd" in stats
        assert "avg_cost_per_query" in stats

    def test_get_stats_before_load(self, pipeline):
        """Stats show not indexed before load_pdf."""
        stats = pipeline.get_stats()
        assert stats["is_indexed"] is False
        assert stats["current_pdf"] is None

    def test_get_stats_after_load(self, pipeline):
        """Stats show indexed after load_pdf."""
        pipeline.load_pdf("cancer_paper.pdf")
        stats = pipeline.get_stats()
        assert stats["is_indexed"] is True
        assert stats["current_pdf"] == "cancer_paper.pdf"
        assert stats["chunks_indexed"] == 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for full pipeline workflow."""

    def test_full_pipeline_workflow(self, pipeline):
        """
        Full workflow:
        1. Load PDF
        2. Query multiple times
        3. Check stats
        """
        # Step 1: Load PDF
        load_result = pipeline.load_pdf("cancer_paper.pdf")
        assert load_result["chunks_created"] == 3
        assert pipeline.is_indexed is True

        # Step 2: Query
        result = pipeline.query("How does TP53 affect therapy?")
        assert "answer" in result
        assert "sources" in result
        assert "cost_usd" in result

        # Step 3: Check stats
        stats = pipeline.get_stats()
        assert stats["is_indexed"] is True
        assert stats["current_pdf"] == "cancer_paper.pdf"

    def test_pipeline_multiple_queries_same_pdf(self, pipeline):
        """Multiple queries on same PDF reuse index."""
        pipeline.load_pdf("cancer_paper.pdf")

        results = []
        questions = [
            "How does TP53 affect therapy?",
            "What is BRCA1 role in cancer?",
            "Which mutations cause resistance?",
        ]

        for q in questions:
            result = pipeline.query(q)
            results.append(result)

        # All queries should succeed
        assert len(results) == 3
        assert all("answer" in r for r in results)
        assert all("cost_usd" in r for r in results)

        # PDF loaded only once
        pipeline._mock_loader.load_pdf.assert_called_once()

    def test_pipeline_query_before_load_fails(self, pipeline):
        """Querying before loading PDF fails gracefully."""
        with pytest.raises(PDFNotLoadedException) as exc_info:
            pipeline.query("How does TP53 affect therapy?")

        assert "load_pdf" in str(exc_info.value).lower() or \
               "no pdf" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])