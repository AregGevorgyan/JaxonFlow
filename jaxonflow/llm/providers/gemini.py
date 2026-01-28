"""Google Gemini LLM provider for JaxonFlow."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError, LLMRateLimitError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)


class GeminiProvider(LLMClient):
    """LLM client for Google's Gemini models.

    Supports Gemini 2.0 Flash, Gemini 1.5 Pro, and other Gemini models.
    Uses the official Google GenAI Python SDK.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the Gemini client.

        Args:
            config: LLM configuration with API key and model settings.
        """
        super().__init__(config)
        self._client = None

        # Gemini pricing (Gemini 1.5 Pro defaults)
        self.usage.input_price_per_mtok = 1.25
        self.usage.output_price_per_mtok = 5.0

    @property
    def client(self):
        """Lazily initialize and return the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai package is required for Gemini provider. "
                    "Install with: pip install google-genai"
                ) from e

            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
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
        from google import genai
        from google.genai import types

        start_time = time.perf_counter()

        try:
            # Build contents list for Gemini
            contents = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])],
                    )
                )

            # Build generation config
            gen_config_kwargs = {}
            if temperature is not None:
                gen_config_kwargs["temperature"] = temperature
            if max_tokens is not None:
                gen_config_kwargs["max_output_tokens"] = max_tokens

            gen_config = types.GenerateContentConfig(
                **gen_config_kwargs,
                system_instruction=system_prompt if system_prompt else None,
            )

            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=gen_config,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.text or ""

            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0

            result = LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.config.model,
                latency_ms=latency_ms,
                stop_reason=None,
            )

            self.usage.add(result)

            logger.debug(
                f"Gemini response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str and "limit" in err_str:
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    retry_after=None,
                ) from e

            if "quota" in err_str or "429" in err_str:
                raise LLMRateLimitError(
                    f"Gemini quota exceeded: {e}",
                    retry_after=None,
                ) from e

            raise LLMAPIError(
                f"Gemini API error: {e}",
                provider="gemini",
            ) from e
