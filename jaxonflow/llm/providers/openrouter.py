"""OpenRouter LLM provider for JaxonFlow.

OpenRouter provides a unified API (OpenAI-compatible) to access models from
Anthropic, Google, Meta, Mistral, and others through a single endpoint.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError, LLMRateLimitError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(LLMClient):
    """LLM client for OpenRouter's multi-model API.

    OpenRouter exposes an OpenAI-compatible API, so this provider uses the
    OpenAI SDK with a custom base URL.  Any model available on OpenRouter
    can be used (e.g. "anthropic/claude-sonnet-4-20250514",
    "google/gemini-2.0-flash-001", "meta-llama/llama-3-70b-instruct").
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the OpenRouter client.

        Args:
            config: LLM configuration with API key and model settings.
                    base_url defaults to OpenRouter if not set.
        """
        super().__init__(config)
        self._client = None

        # OpenRouter pricing varies per model; use a rough average
        self.usage.input_price_per_mtok = 3.0
        self.usage.output_price_per_mtok = 15.0

    @property
    def client(self):
        """Lazily initialize and return the OpenAI client pointed at OpenRouter."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenRouter provider. "
                    "Install with: pip install openai"
                ) from e

            base_url = self.config.base_url or OPENROUTER_BASE_URL

            kwargs: dict = {
                "api_key": self.config.api_key,
                "base_url": base_url,
            }

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
        """Generate a response via OpenRouter.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated content and usage statistics.
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

            content = response.choices[0].message.content or ""
            stop_reason = response.choices[0].finish_reason

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            result = LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=response.model or self.config.model,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
            )

            self.usage.add(result)

            logger.debug(
                f"OpenRouter response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except openai.RateLimitError as e:
            raise LLMRateLimitError(
                f"OpenRouter rate limit exceeded: {e}",
                retry_after=None,
            ) from e

        except openai.APIError as e:
            raise LLMAPIError(
                f"OpenRouter API error: {e}",
                provider="openrouter",
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            ) from e

        except Exception as e:
            raise LLMAPIError(
                f"Unexpected error calling OpenRouter API: {e}",
                provider="openrouter",
            ) from e
