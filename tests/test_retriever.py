"""
Test suite for retriever.loader module

Tests Document class and DocumentLoader class.
Run with: pytest tests/test_retriever.py -v
"""

import pytest
from src.retriever.loader import Document, DocumentLoader, LoaderException, PDFLoadException


# ============================================================================
# TESTS FOR Document CLASS
# ============================================================================

class TestDocument:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Document can be created with valid data."""
        doc = Document(
            id="paper_1",
            content="TP53 mutations increase cancer risk",
            source_file="paper_1.pdf"
        )
        assert doc.id == "paper_1"
        assert doc.source_file == "paper_1.pdf"
        assert doc.page_num is None
    
    def test_document_rejects_empty_content(self):
        """Document rejects empty or whitespace-only content."""
        with pytest.raises(ValueError, match="empty content"):
            Document(id="empty", content="   ", source_file="test.pdf")
    
    def test_document_rejects_newline_only(self):
        """Document rejects content with only newlines."""
        with pytest.raises(ValueError, match="empty content"):
            Document(id="newline", content="\n\n\n", source_file="test.pdf")
    
    def test_document_with_page_number(self):
        """Document can store page number."""
        doc = Document(
            id="p1_page2",
            content="Some content here",
            source_file="paper.pdf",
            page_num=2
        )
        assert doc.page_num == 2
    
    def test_document_page_number_optional(self):
        """Document page_num defaults to None if not provided."""
        doc = Document(
            id="p1",
            content="Content",
            source_file="paper.pdf"
        )
        assert doc.page_num is None


# ============================================================================
# TESTS FOR DocumentLoader CLASS
# ============================================================================

class TestDocumentLoaderInit:
    """Test DocumentLoader initialization."""
    
    def test_loader_initialization_default(self):
        """Loader initializes with default parameters."""
        loader = DocumentLoader()
        assert loader.chunk_size == 512
        assert loader.overlap == 50
    
    def test_loader_initialization_custom(self):
        """Loader initializes with custom parameters."""
        loader = DocumentLoader(chunk_size=256, overlap=30)
        assert loader.chunk_size == 256
        assert loader.overlap == 30
    
    def test_loader_rejects_zero_chunk_size(self):
        """Loader rejects chunk_size of 0."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentLoader(chunk_size=0)
    
    def test_loader_rejects_negative_chunk_size(self):
        """Loader rejects negative chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentLoader(chunk_size=-1)
    
    def test_loader_rejects_overlap_equal_chunk_size(self):
        """Loader rejects overlap equal to chunk_size."""
        with pytest.raises(ValueError, match="overlap must be"):
            DocumentLoader(chunk_size=100, overlap=100)
    
    def test_loader_rejects_overlap_greater_chunk_size(self):
        """Loader rejects overlap greater than chunk_size."""
        with pytest.raises(ValueError, match="overlap must be"):
            DocumentLoader(chunk_size=100, overlap=101)


class TestDocumentLoaderChunking:
    """Test chunk_text method."""
    
    def test_chunk_text_basic(self):
        """chunk_text splits text into multiple chunks."""
        loader = DocumentLoader(chunk_size=20, overlap=5)
        text = "a" * 100  # 100 character string
        chunks = loader.chunk_text(text, source_id="test_doc")
        
        # Should create multiple chunks
        assert len(chunks) > 1, "Should create multiple chunks for 100 char text"
        
        # All chunks should be from same source
        assert all(c.source_file == "test_doc" for c in chunks)
        
        # All chunks should be <= chunk_size
        assert all(len(c.content) <= 20 for c in chunks)
    
    def test_chunk_text_with_real_content(self):
        """chunk_text preserves actual document content."""
        loader = DocumentLoader(chunk_size=50, overlap=10)
        text = "TP53 mutations affect cancer progression in breast tissue samples analysis"
        chunks = loader.chunk_text(text, source_id="cancer_paper")
        
        # Should create chunks
        assert len(chunks) >= 1
        
        # First chunk should have content from start
        assert "TP53" in chunks[0].content
        
        # All chunks should be from correct source
        assert all(c.source_file == "cancer_paper" for c in chunks)
    
    def test_chunk_text_skips_tiny_chunks(self):
        """chunk_text skips chunks smaller than 10 characters."""
        loader = DocumentLoader(chunk_size=15, overlap=2)
        text = "short"  # Only 5 characters
        chunks = loader.chunk_text(text, source_id="tiny")
        
        # Should skip this tiny chunk
        assert len(chunks) == 0, "Should skip chunks < 10 chars"
    
    def test_chunk_text_creates_unique_ids(self):
        """chunk_text creates unique IDs for each chunk."""
        loader = DocumentLoader(chunk_size=30, overlap=5)
        text = "a" * 100
        chunks = loader.chunk_text(text, source_id="paper1")
        
        # Extract all IDs
        ids = [c.id for c in chunks]
        
        # Should be unique
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"
        
        # Should follow pattern: source_id_chunk_N
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"paper1_chunk_{i}"
    
    def test_chunk_text_overlap_mechanism(self):
        """chunk_text creates overlapping chunks correctly."""
        loader = DocumentLoader(chunk_size=10, overlap=3)
        text = "ABCDEFGHIJKLMNOPQRST"  # 20 chars
        chunks = loader.chunk_text(text, source_id="overlap_test")
        
        # With chunk_size=10, overlap=3, step=7:
        # Chunk 0: chars 0-9   (ABCDEFGHIJ)
        # Chunk 1: chars 7-16  (HIJKLMNOPQ)
        # Chunk 2: chars 14-19 (OPQRST + padding)
        
        # First chunk should start with A
        assert chunks[0].content[0] == "A"
        
        # Chunks should overlap
        # Last 3 chars of chunk[0] should match first 3 chars of chunk[1]
        if len(chunks) > 1:
            last_chars_chunk0 = chunks[0].content[-3:]
            first_chars_chunk1 = chunks[1].content[:3]
            assert last_chars_chunk0 == first_chars_chunk1, "Chunks should overlap"
    
    def test_chunk_text_empty_string(self):
        """chunk_text handles empty string."""
        loader = DocumentLoader(chunk_size=50, overlap=10)
        chunks = loader.chunk_text("", source_id="empty")
        
        # Empty string should produce no chunks
        assert len(chunks) == 0
    
    def test_chunk_text_whitespace_only(self):
        """chunk_text skips whitespace-only content."""
        loader = DocumentLoader(chunk_size=50, overlap=10)
        text = "   \n   \t   "
        chunks = loader.chunk_text(text, source_id="whitespace")
        
        # Should skip whitespace-only chunks
        assert len(chunks) == 0


class TestDocumentLoaderLoadPDF:
    """Test load_pdf method."""
    
    def test_load_pdf_missing_file(self):
        """load_pdf raises PDFLoadException for missing file."""
        loader = DocumentLoader()
        
        with pytest.raises(PDFLoadException):
            loader.load_pdf("/nonexistent/file.pdf")
    
    def test_load_pdf_error_contains_path(self):
        """PDFLoadException message contains the file path."""
        loader = DocumentLoader()
        
        try:
            loader.load_pdf("/nonexistent/cancer_paper.pdf")
        except PDFLoadException as e:
            assert "cancer_paper.pdf" in str(e)
    
    def test_load_pdf_returns_list(self):
        """load_pdf returns a list (even if empty for now)."""
        loader = DocumentLoader()
        
        # For now, load_pdf returns empty list (TODO implementation)
        # So we can't test with real PDF, but we can test it returns correct type
        # This will be tested properly when pypdf integration is done
        pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test integration between components."""
    
    def test_create_loader_and_chunk(self):
        """Full workflow: create loader, chunk text, access results."""
        # Create loader
        loader = DocumentLoader(chunk_size=100, overlap=15)
        
        # Chunk some text
        text = "TP53 is a tumor suppressor gene. Mutations in TP53 increase cancer risk. " * 3
        chunks = loader.chunk_text(text, source_id="study_2023")
        
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        
        # Verify we can access chunk data
        for chunk in chunks:
            assert len(chunk.id) > 0
            assert len(chunk.content) > 0
            assert chunk.source_file == "study_2023"
    
    def test_multiple_sources(self):
        """DocumentLoader can chunk multiple sources without collision."""
        loader = DocumentLoader(chunk_size=50, overlap=10)
        
        # Chunk two different papers
        chunks1 = loader.chunk_text("Paper 1 content here " * 5, source_id="paper1")
        chunks2 = loader.chunk_text("Paper 2 content here " * 5, source_id="paper2")
        
        # IDs should be unique across both
        all_ids = [c.id for c in chunks1 + chunks2]
        assert len(all_ids) == len(set(all_ids)), "IDs should be unique across sources"
        
        # Each should have correct source
        for c in chunks1:
            assert c.source_file == "paper1"
        for c in chunks2:
            assert c.source_file == "paper2"


if __name__ == "__main__":
    # Run tests with: pytest tests/test_retriever.py -v
    pytest.main([__file__, "-v"])