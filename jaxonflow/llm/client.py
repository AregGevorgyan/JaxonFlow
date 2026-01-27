"""LLM client abstraction for JaxonFlow."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: The generated text content.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        model: The model that generated the response.
        latency_ms: Request latency in milliseconds.
        stop_reason: Why generation stopped (e.g., "end_turn", "max_tokens").
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class TokenUsage:
    """Tracks cumulative token usage and estimated costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0

    # Pricing per million tokens (default: Claude Sonnet pricing)
    input_price_per_mtok: float = 3.0
    output_price_per_mtok: float = 15.0

    def add(self, response: LLMResponse) -> None:
        """Add a response's usage to cumulative totals."""
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_requests += 1
        self.total_latency_ms += response.latency_ms

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_price_per_mtok
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price_per_mtok
        return input_cost + output_cost

    @property
    def average_latency_ms(self) -> float:
        """Average latency per request."""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Defines the interface that all LLM providers must implement.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM client.

        Args:
            config: LLM configuration including API key and model settings.
        """
        self.config = config
        self.usage = TokenUsage()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (overrides config if provided).
            max_tokens: Maximum tokens to generate (overrides config if provided).

        Returns:
            LLMResponse with generated content and usage statistics.

        Raises:
            LLMAPIError: If the API call fails.
            LLMRateLimitError: If rate limit is exceeded.
        """
        pass

    @abstractmethod
    def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from a multi-turn conversation.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content and usage statistics.
        """
        pass

    def extract_code(self, response: LLMResponse, language: str = "python") -> str | None:
        """Extract code block from response content.

        Args:
            response: The LLM response.
            language: The expected language (for code fence matching).

        Returns:
            Extracted code or None if no code block found.
        """
        content = response.content

        # Try to find fenced code block
        fence_start = f"```{language}"
        fence_end = "```"

        start_idx = content.find(fence_start)
        if start_idx == -1:
            # Try without language specifier
            start_idx = content.find("```")
            if start_idx == -1:
                return None
            start_idx += 3
        else:
            start_idx += len(fence_start)

        # Skip whitespace after opening fence
        while start_idx < len(content) and content[start_idx] in "\n\r":
            start_idx += 1

        end_idx = content.find(fence_end, start_idx)
        if end_idx == -1:
            return None

        return content[start_idx:end_idx].strip()

    def extract_yaml(self, response: LLMResponse) -> str | None:
        """Extract YAML block from response content.

        Args:
            response: The LLM response.

        Returns:
            Extracted YAML or None if no YAML block found.
        """
        content = response.content

        # Try to find fenced YAML block
        for fence in ["```yaml", "```yml"]:
            start_idx = content.find(fence)
            if start_idx != -1:
                start_idx += len(fence)
                while start_idx < len(content) and content[start_idx] in "\n\r":
                    start_idx += 1
                end_idx = content.find("```", start_idx)
                if end_idx != -1:
                    return content[start_idx:end_idx].strip()

        return None


def create_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: LLM configuration specifying provider and settings.

    Returns:
        Appropriate LLMClient implementation.

    Raises:
        ValueError: If provider is not supported.
    """
    from ..config import LLMProvider
    from .providers import AnthropicProvider, OpenAIProvider

    if config.provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == LLMProvider.OPENAI:
        return OpenAIProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
