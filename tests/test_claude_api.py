"""
Test suite for generator.claude_api module

Tests ClaudeAPI class with mocked Anthropic client.
Run with: pytest tests/test_claude_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from src.generator.claude_api import (
    ClaudeAPI,
    ClaudeAPIException,
    TokenLimitException,
    APIConnectionException,
    COST_PER_MILLION_INPUT_TOKENS,
    COST_PER_MILLION_OUTPUT_TOKENS,
)


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def mock_anthropic_response():
    """Mock successful Claude API response."""
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].text = "TP53 mutations reduce apoptosis, affecting therapy response."
    response.usage.input_tokens = 500
    response.usage.output_tokens = 200
    return response


@pytest.fixture
def api(mock_anthropic_response):
    """
    ClaudeAPI instance with mocked Anthropic client.
    Avoids real API calls during tests.
    """
    with patch("src.generator.claude_api.anthropic.Anthropic") as mock_client:
        # Setup mock to return fake response
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = mock_anthropic_response
        mock_client.return_value = mock_instance

        # Create API instance
        claude = ClaudeAPI(api_key="sk-ant-fake-key-for-testing")
        claude.client = mock_instance

        yield claude


@pytest.fixture
def sample_context():
    """Sample retrieved context from search.py."""
    return """[Source: cancer_paper, Page: 1, Similarity: 0.98]
TP53 mutations reduce apoptosis in cancer cells.

[Source: cancer_paper, Page: 2, Similarity: 0.95]
Therapy resistance is linked to TP53 loss of function."""


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "How does TP53 affect therapy response?"


# ============================================================================
# TESTS FOR ClaudeAPI INITIALIZATION
# ============================================================================

class TestClaudeAPIInit:
    """Test ClaudeAPI initialization."""

    def test_init_default_model(self, api):
        """API initializes with correct default model."""
        assert api.model == "claude-3-5-sonnet-20241022"

    def test_init_default_max_tokens(self, api):
        """API initializes with correct default max_tokens."""
        assert api.max_tokens == 1024

    def test_init_custom_model(self):
        """API accepts custom model name."""
        with patch("src.generator.claude_api.anthropic.Anthropic"):
            api = ClaudeAPI(
                api_key="fake-key",
                model="claude-3-haiku-20240307"
            )
            assert api.model == "claude-3-haiku-20240307"

    def test_init_custom_max_tokens(self):
        """API accepts custom max_tokens."""
        with patch("src.generator.claude_api.anthropic.Anthropic"):
            api = ClaudeAPI(api_key="fake-key", max_tokens=2048)
            assert api.max_tokens == 2048

    def test_init_empty_api_key_raises_error(self):
        """Empty API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            ClaudeAPI(api_key="")

    def test_init_zero_max_tokens_raises_error(self):
        """Zero max_tokens raises ValueError."""
        with patch("src.generator.claude_api.anthropic.Anthropic"):
            with pytest.raises(ValueError, match="max_tokens must be positive"):
                ClaudeAPI(api_key="fake-key", max_tokens=0)

    def test_init_starts_with_zero_usage(self, api):
        """API starts with zero usage stats."""
        assert api.total_queries == 0
        assert api.total_input_tokens == 0
        assert api.total_output_tokens == 0
        assert api.total_cost_usd == 0.0


# ============================================================================
# TESTS FOR _build_prompt()
# ============================================================================

class TestBuildPrompt:
    """Test _build_prompt() method."""

    def test_build_prompt_contains_query(self, api, sample_query, sample_context):
        """Prompt contains the user query."""
        prompt = api._build_prompt(sample_query, sample_context)
        assert sample_query in prompt

    def test_build_prompt_contains_context(self, api, sample_query, sample_context):
        """Prompt contains the context."""
        prompt = api._build_prompt(sample_query, sample_context)
        assert "TP53 mutations reduce apoptosis" in prompt

    def test_build_prompt_returns_string(self, api, sample_query, sample_context):
        """_build_prompt() returns a string."""
        prompt = api._build_prompt(sample_query, sample_context)
        assert isinstance(prompt, str)

    def test_build_prompt_context_before_query(self, api, sample_query, sample_context):
        """Context appears before query in prompt."""
        prompt = api._build_prompt(sample_query, sample_context)

        context_pos = prompt.find("TP53 mutations reduce apoptosis")
        query_pos = prompt.find(sample_query)

        assert context_pos < query_pos


# ============================================================================
# TESTS FOR _calculate_cost()
# ============================================================================

class TestCalculateCost:
    """Test _calculate_cost() method."""

    def test_calculate_cost_returns_float(self, api):
        """_calculate_cost() returns float."""
        cost = api._calculate_cost(500, 200)
        assert isinstance(cost, float)

    def test_calculate_cost_zero_tokens(self, api):
        """Zero tokens = zero cost."""
        cost = api._calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_input_only(self, api):
        """Cost with only input tokens."""
        # 1,000,000 input tokens = $3.00
        cost = api._calculate_cost(1_000_000, 0)
        assert cost == pytest.approx(3.00, rel=1e-3)

    def test_calculate_cost_output_only(self, api):
        """Cost with only output tokens."""
        # 1,000,000 output tokens = $15.00
        cost = api._calculate_cost(0, 1_000_000)
        assert cost == pytest.approx(15.00, rel=1e-3)

    def test_calculate_cost_numerical_example(self, api):
        """Verify cost calculation with known values.

        Input:  500 tokens × ($3.00  / 1,000,000) = $0.0015
        Output: 200 tokens × ($15.00 / 1,000,000) = $0.0030
        Total:                                       $0.0045
        """
        cost = api._calculate_cost(500, 200)
        assert cost == pytest.approx(0.0045, rel=1e-3)

    def test_calculate_cost_increases_with_tokens(self, api):
        """More tokens = higher cost."""
        cost_small = api._calculate_cost(100, 50)
        cost_large = api._calculate_cost(1000, 500)
        assert cost_large > cost_small


# ============================================================================
# TESTS FOR generate()
# ============================================================================

class TestGenerate:
    """Test generate() method."""

    def test_generate_returns_dict(self, api, sample_query, sample_context):
        """generate() returns a dict."""
        result = api.generate(sample_query, sample_context)
        assert isinstance(result, dict)

    def test_generate_returns_text(self, api, sample_query, sample_context):
        """generate() result contains text key."""
        result = api.generate(sample_query, sample_context)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_generate_returns_token_counts(self, api, sample_query, sample_context):
        """generate() result contains token counts."""
        result = api.generate(sample_query, sample_context)
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["input_tokens"] == 500   # From mock
        assert result["output_tokens"] == 200  # From mock

    def test_generate_returns_cost(self, api, sample_query, sample_context):
        """generate() result contains cost."""
        result = api.generate(sample_query, sample_context)
        assert "cost_usd" in result
        assert isinstance(result["cost_usd"], float)
        assert result["cost_usd"] > 0

    def test_generate_returns_model(self, api, sample_query, sample_context):
        """generate() result contains model name."""
        result = api.generate(sample_query, sample_context)
        assert "model" in result
        assert result["model"] == "claude-3-5-sonnet-20241022"

    def test_generate_empty_query_raises_error(self, api, sample_context):
        """generate() raises ValueError for empty query."""
        with pytest.raises(ValueError):
            api.generate("", sample_context)

    def test_generate_empty_context_raises_error(self, api, sample_query):
        """generate() raises ValueError for empty context."""
        with pytest.raises(ValueError):
            api.generate(sample_query, "")

    def test_generate_updates_usage_stats(self, api, sample_query, sample_context):
        """generate() updates cumulative usage stats."""
        api.generate(sample_query, sample_context)

        assert api.total_queries == 1
        assert api.total_input_tokens == 500
        assert api.total_output_tokens == 200
        assert api.total_cost_usd > 0

    def test_generate_accumulates_multiple_queries(
        self, api, sample_query, sample_context
    ):
        """Multiple generate() calls accumulate stats."""
        api.generate(sample_query, sample_context)
        api.generate(sample_query, sample_context)
        api.generate(sample_query, sample_context)

        assert api.total_queries == 3
        assert api.total_input_tokens == 1500   # 500 × 3
        assert api.total_output_tokens == 600   # 200 × 3

    def test_generate_api_connection_error(self, api, sample_query, sample_context):
        """generate() raises APIConnectionException on connection error."""
        import anthropic

        api.client.messages.create.side_effect = (
            anthropic.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(APIConnectionException):
            api.generate(sample_query, sample_context)

    def test_generate_bad_request_error(self, api, sample_query, sample_context):
        """generate() raises TokenLimitException on bad request."""
        import anthropic

        api.client.messages.create.side_effect = (
            anthropic.BadRequestError(
                message="too many tokens",
                response=MagicMock(),
                body={}
            )
        )

        with pytest.raises(TokenLimitException):
            api.generate(sample_query, sample_context)


# ============================================================================
# TESTS FOR get_usage_stats()
# ============================================================================

class TestGetUsageStats:
    """Test get_usage_stats() method."""

    def test_get_usage_stats_returns_dict(self, api):
        """get_usage_stats() returns dict."""
        stats = api.get_usage_stats()
        assert isinstance(stats, dict)

    def test_get_usage_stats_has_required_keys(self, api):
        """get_usage_stats() has all required keys."""
        stats = api.get_usage_stats()

        assert "total_queries" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost_usd" in stats
        assert "avg_cost_per_query" in stats

    def test_get_usage_stats_zero_queries(self, api):
        """Stats are zero before any queries."""
        stats = api.get_usage_stats()

        assert stats["total_queries"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert stats["avg_cost_per_query"] == 0.0

    def test_get_usage_stats_after_queries(
        self, api, sample_query, sample_context
    ):
        """Stats update after queries."""
        api.generate(sample_query, sample_context)
        api.generate(sample_query, sample_context)

        stats = api.get_usage_stats()

        assert stats["total_queries"] == 2
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 400
        assert stats["total_cost_usd"] > 0
        assert stats["avg_cost_per_query"] > 0

    def test_avg_cost_calculation(
        self, api, sample_query, sample_context
    ):
        """Average cost per query is calculated correctly."""
        api.generate(sample_query, sample_context)

        stats = api.get_usage_stats()

        assert stats["avg_cost_per_query"] == pytest.approx(
            stats["total_cost_usd"] / stats["total_queries"],
            rel=1e-3
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestClaudeAPIIntegration:
    """Integration tests."""

    def test_full_rag_query_workflow(self, api):
        """
        Full RAG workflow:
        1. Build context from search results
        2. Send query + context to Claude
        3. Get answer with cost tracking
        """
        # Step 1: Simulate search results
        context = """[Source: cancer_paper, Page: 1, Similarity: 0.98]
TP53 mutations reduce apoptosis in cancer cells, leading to
therapy resistance in breast cancer patients.

[Source: cancer_paper, Page: 3, Similarity: 0.95]
Clinical studies show TP53 loss correlates with poor
response to chemotherapy in triple-negative breast cancer."""

        # Step 2: Send query
        query = "How does TP53 affect chemotherapy response?"
        result = api.generate(query, context)

        # Step 3: Verify result
        assert "text" in result
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 200
        assert result["cost_usd"] > 0
        assert result["model"] == "claude-3-5-sonnet-20241022"

        # Step 4: Check stats
        stats = api.get_usage_stats()
        assert stats["total_queries"] == 1

    def test_cost_tracking_multiple_queries(self, api):
        """Cost accumulates correctly across multiple queries."""
        context = "TP53 mutations affect cancer therapy response."
        query = "Explain TP53 role in therapy."

        # Run 5 queries
        for _ in range(5):
            api.generate(query, context)

        stats = api.get_usage_stats()

        # Verify accumulation
        assert stats["total_queries"] == 5
        assert stats["total_input_tokens"] == 2500   # 500 × 5
        assert stats["total_output_tokens"] == 1000  # 200 × 5

        # Verify average cost
        expected_avg = stats["total_cost_usd"] / 5
        assert stats["avg_cost_per_query"] == pytest.approx(expected_avg, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])