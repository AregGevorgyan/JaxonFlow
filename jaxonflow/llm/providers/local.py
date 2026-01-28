"""Local / self-hosted LLM provider for JaxonFlow.

Thin wrapper around the OpenAI-compatible API exposed by local inference
servers such as Ollama, vLLM, llama.cpp (llama-server), LM Studio,
LocalAI, and TGI.

Usage:
    config = LLMConfig(
        provider=LLMProvider.LOCAL,
        model="llama3",                        # model name on the server
        api_key="not-needed",                  # most local servers ignore this
        base_url="http://localhost:11434/v1",   # Ollama default
    )
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"  # Ollama default


class LocalProvider(LLMClient):
    """LLM client for locally-hosted OpenAI-compatible servers.

    Works with any server exposing an OpenAI-compatible chat/completions
    endpoint: Ollama, vLLM, llama.cpp, LM Studio, LocalAI, TGI, etc.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the local model client.

        Args:
            config: LLM configuration. ``base_url`` should point at the
                    local server (defaults to ``http://localhost:11434/v1``).
        """
        super().__init__(config)
        self._client = None

        # Local models are free
        self.usage.input_price_per_mtok = 0.0
        self.usage.output_price_per_mtok = 0.0

    @property
    def client(self):
        """Lazily initialize the OpenAI client pointed at the local server."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package is required for the local provider. "
                    "Install with: pip install openai"
                ) from e

            base_url = self.config.base_url or DEFAULT_LOCAL_BASE_URL

            self._client = openai.OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=base_url,
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
        """Generate a response from the local model.

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
                f"Local response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except ConnectionError as e:
            raise LLMAPIError(
                f"Cannot connect to local server at "
                f"{self.config.base_url or DEFAULT_LOCAL_BASE_URL}: {e}",
                provider="local",
            ) from e

        except Exception as e:
            raise LLMAPIError(
                f"Local model error: {e}",
                provider="local",
            ) from e
