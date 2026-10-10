"""
Claude API Wrapper: Context + Query → Answer

Takes retrieved chunks from search.py and user query,
sends to Claude API, returns answer with cost tracking.
"""

from typing import List, Dict, Optional
import logging
import anthropic
# In any script or test:

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ClaudeAPIException(Exception):
    """Base exception for Claude API module."""
    pass


class TokenLimitException(ClaudeAPIException):
    """Raised when token limit exceeded."""
    pass


class APIConnectionException(ClaudeAPIException):
    """Raised when API connection fails."""
    pass


# ============================================================================
# COST CONSTANTS
# ============================================================================

# Claude pricing per million tokens (as of 2024)
# Source: https://anthropic.com/pricing
COST_PER_MILLION_INPUT_TOKENS = 3.00   # $3 per 1M input tokens
COST_PER_MILLION_OUTPUT_TOKENS = 15.00  # $15 per 1M output tokens


# ============================================================================
# CLASS: ClaudeAPI
# ============================================================================

class ClaudeAPI:
    """
    Wrapper for Claude API with token tracking and cost monitoring.
    
    Workflow:
        1. Build prompt from context + query
        2. Send to Claude API
        3. Return answer + usage stats
    
    Example:
        >>> api = ClaudeAPI(api_key="your-key")
        >>> answer = api.generate(
        ...     query="How does TP53 affect therapy?",
        ...     context="TP53 mutations reduce apoptosis..."
        ... )
        >>> answer["text"]
        "Based on the research, TP53 mutations..."
        >>> answer["cost_usd"]
        0.003
    """
    
    # System prompt for cancer genomics RAG
    SYSTEM_PROMPT = """You are an expert cancer genomics researcher and clinical oncologist.

Your role:
- Answer questions about cancer genetics, mutations, and therapy
- Base answers ONLY on the provided research context
- If context doesn't contain enough information, say so clearly
- Cite sources when possible (paper name, page number)
- Be precise and scientific in your language

Important:
- Do NOT make up information not in the context
- Do NOT use general knowledge outside the provided context
- Always mention confidence level in your answer"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize Claude API wrapper.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum tokens in response
        
        Example:
            >>> api = ClaudeAPI(api_key="sk-ant-...")
            >>> api.model
            'claude-3-5-sonnet-20241022'
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Track cumulative usage
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_queries: int = 0
        
        logger.info(f"ClaudeAPI initialized: model={model}, max_tokens={max_tokens}")
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt from query and retrieved context.
        
        Args:
            query: User question
            context: Retrieved chunks from search.py
        
        Returns:
            Formatted prompt string
        
        Example:
            >>> prompt = api._build_prompt(
            ...     query="How does TP53 affect therapy?",
            ...     context="[Source: paper1]\nTP53 mutations..."
            ... )
            >>> print(prompt)
            Context from cancer research papers:
            [Source: paper1]
            TP53 mutations...
            
            Question: How does TP53 affect therapy?
        """
        prompt = f"""Context from cancer research papers:

{context}

Question: {query}

Please answer the question based ONLY on the context provided above."""
        
        return prompt
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a single API call.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        
        Returns:
            Cost in USD
        
        Example:
            Input tokens: 500
            Output tokens: 200
            
            Input cost:  500 / 1,000,000 × $3.00  = $0.0015
            Output cost: 200 / 1,000,000 × $15.00 = $0.0030
            Total:                                   $0.0045
        """
        input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS
        output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT_TOKENS
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Generate answer from query and context.
        
        Args:
            query: User question
            context: Retrieved chunks from search.py
            system_prompt: Override default system prompt
        
        Returns:
            Dict with:
                text: Claude's answer
                input_tokens: Tokens in prompt
                output_tokens: Tokens in response
                cost_usd: Cost of this query
                model: Model used
        
        Raises:
            ClaudeAPIException: If API call fails
            TokenLimitException: If context too large
        
        Example:
            >>> result = api.generate(
            ...     query="How does TP53 affect therapy?",
            ...     context="TP53 mutations reduce apoptosis..."
            ... )
            >>> result["text"]
            "Based on the research..."
            >>> result["cost_usd"]
            0.003
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")
        
        try:
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            # Use default or custom system prompt
            system = system_prompt or self.SYSTEM_PROMPT
            
            logger.info(f"Sending query to Claude: {query[:50]}...")
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract answer
            answer_text = response.content[0].text
            
            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update cumulative stats
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
            self.total_queries += 1
            
            logger.info(
                f"Query complete. "
                f"Tokens: {input_tokens}+{output_tokens}, "
                f"Cost: ${cost:.4f}"
            )
            
            return {
                "text": answer_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "model": self.model,
            }
        
        except anthropic.APIConnectionError as e:
            logger.error(f"API connection failed: {e}")
            raise APIConnectionException("Cannot connect to Claude API") from e
        
        except anthropic.BadRequestError as e:
            logger.error(f"Bad request (possibly token limit): {e}")
            raise TokenLimitException("Request too large for model") from e
        
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            raise ClaudeAPIException(f"API call failed: {e}") from e
    
    def get_usage_stats(self) -> Dict:
        """
        Return cumulative usage statistics.
        
        Returns:
            Dict with total tokens and cost across all queries
        
        Example:
            >>> api.get_usage_stats()
            {
                "total_queries": 10,
                "total_input_tokens": 5000,
                "total_output_tokens": 2000,
                "total_cost_usd": 0.045,
                "avg_cost_per_query": 0.0045,
            }
        """
        avg_cost = (
            self.total_cost_usd / self.total_queries
            if self.total_queries > 0
            else 0.0
        )
        
        return {
            "total_queries": self.total_queries,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_query": round(avg_cost, 6),
        }