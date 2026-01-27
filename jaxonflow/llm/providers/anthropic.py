"""Anthropic/Claude LLM provider for JaxonFlow."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError, LLMRateLimitError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMClient):
    """LLM client for Anthropic's Claude models.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Anthropic models.
    Uses the official Anthropic Python SDK.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the Anthropic client.

        Args:
            config: LLM configuration with API key and model settings.
        """
        super().__init__(config)
        self._client: "anthropic.Anthropic | None" = None

    @property
    def client(self) -> "anthropic.Anthropic":
        """Lazily initialize and return the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package is required for Anthropic provider. "
                    "Install with: pip install anthropic"
                ) from e

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from Claude.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content and usage statistics.

        Raises:
            LLMAPIError: If the API call fails.
            LLMRateLimitError: If rate limit is exceeded.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
        import anthropic

        start_time = time.perf_counter()

        try:
            kwargs: dict = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens or 4096,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if temperature is not None:
                kwargs["temperature"] = temperature

            response = self.client.messages.create(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content from response
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            result = LLMResponse(
                content=content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                latency_ms=latency_ms,
                stop_reason=response.stop_reason,
            )

            # Track usage
            self.usage.add(result)

            logger.debug(
                f"Anthropic response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {e}",
                retry_after=getattr(e, "retry_after", None),
            ) from e

        except anthropic.APIError as e:
            raise LLMAPIError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            ) from e

        except Exception as e:
            raise LLMAPIError(
                f"Unexpected error calling Anthropic API: {e}",
                provider="anthropic",
            ) from e
