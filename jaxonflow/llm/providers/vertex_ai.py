"""Google Vertex AI LLM provider for JaxonFlow."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ..client import LLMClient, LLMResponse
from ...exceptions import LLMAPIError, LLMRateLimitError

if TYPE_CHECKING:
    from ...config import LLMConfig

logger = logging.getLogger(__name__)


class VertexAIProvider(LLMClient):
    """LLM client for Google Vertex AI.

    Provides enterprise access to Gemini models through Google Cloud Platform.
    Uses Application Default Credentials (ADC) for authentication — no API key
    required. Set up credentials via ``gcloud auth application-default login``
    or by setting the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.

    Environment variables:
        GOOGLE_CLOUD_PROJECT: GCP project ID (required).
        GOOGLE_CLOUD_LOCATION: GCP region (default: ``us-central1``).
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the Vertex AI client.

        Args:
            config: LLM configuration with model settings.
                    ``api_key`` is not used — authentication is via ADC.
        """
        super().__init__(config)
        self._model = None

        self._project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Gemini pricing (same as public API)
        self.usage.input_price_per_mtok = 1.25
        self.usage.output_price_per_mtok = 5.0

    @property
    def model(self):
        """Lazily initialize and return the Vertex AI GenerativeModel."""
        if self._model is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
            except ImportError as e:
                raise ImportError(
                    "google-cloud-aiplatform package is required for Vertex AI provider. "
                    "Install with: pip install google-cloud-aiplatform"
                ) from e

            vertexai.init(project=self._project, location=self._location)
            self._model = GenerativeModel(self.config.model)
        return self._model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from Vertex AI.

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
        from vertexai.generative_models import Content, GenerationConfig, Part

        start_time = time.perf_counter()

        try:
            # Build contents list
            contents = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    Content(role=role, parts=[Part.from_text(msg["content"])])
                )

            # Build generation config
            gen_config_kwargs: dict = {}
            if temperature is not None:
                gen_config_kwargs["temperature"] = temperature
            if max_tokens is not None:
                gen_config_kwargs["max_output_tokens"] = max_tokens

            gen_config = GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None

            # If system prompt, re-create model with system instruction
            model = self.model
            if system_prompt:
                from vertexai.generative_models import GenerativeModel

                model = GenerativeModel(
                    self.config.model,
                    system_instruction=system_prompt,
                )

            response = model.generate_content(
                contents=contents,
                generation_config=gen_config,
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
                f"Vertex AI response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                raise LLMRateLimitError(
                    f"Vertex AI rate limit exceeded: {e}",
                    retry_after=None,
                ) from e

            if "quota" in err_str:
                raise LLMRateLimitError(
                    f"Vertex AI quota exceeded: {e}",
                    retry_after=None,
                ) from e

            raise LLMAPIError(
                f"Vertex AI API error: {e}",
                provider="vertex_ai",
            ) from e
