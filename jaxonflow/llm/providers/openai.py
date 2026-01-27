"""OpenAI LLM provider for JaxonFlow."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError, LLMRateLimitError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMClient):
    """LLM client for OpenAI's models.

    Supports GPT-4, GPT-4 Turbo, GPT-4o, and other OpenAI models.
    Uses the official OpenAI Python SDK.
    """

    # Default model mappings
    MODEL_ALIASES = {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the OpenAI client.

        Args:
            config: LLM configuration with API key and model settings.
        """
        super().__init__(config)
        self._client: "openai.OpenAI | None" = None

        # Adjust pricing for OpenAI models
        self.usage.input_price_per_mtok = 5.0  # GPT-4o pricing
        self.usage.output_price_per_mtok = 15.0

    @property
    def client(self) -> "openai.OpenAI":
        """Lazily initialize and return the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAI provider. "
                    "Install with: pip install openai"
                ) from e

            kwargs: dict = {"api_key": self.config.api_key}

            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url

            if self.config.timeout:
                kwargs["timeout"] = self.config.timeout

            if self.config.max_retries:
                kwargs["max_retries"] = self.config.max_retries

            self._client = openai.OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from OpenAI.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0 to 2.0).
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
        import openai

        start_time = time.perf_counter()

        try:
            # Build messages list with system prompt
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            kwargs: dict = {
                "model": self.config.model,
                "messages": full_messages,
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            if temperature is not None:
                kwargs["temperature"] = temperature

            response = self.client.chat.completions.create(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content from response
            content = response.choices[0].message.content or ""
            stop_reason = response.choices[0].finish_reason

            # Get token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            result = LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=response.model,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
            )

            # Track usage
            self.usage.add(result)

            logger.debug(
                f"OpenAI response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {e}",
                retry_after=None,
            ) from e

        except openai.APIError as e:
            raise LLMAPIError(
                f"OpenAI API error: {e}",
                provider="openai",
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            ) from e

        except Exception as e:
            raise LLMAPIError(
                f"Unexpected error calling OpenAI API: {e}",
                provider="openai",
            ) from e
