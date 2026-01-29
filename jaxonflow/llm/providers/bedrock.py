"""AWS Bedrock LLM provider for JaxonFlow."""

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


class BedrockProvider(LLMClient):
    """LLM client for AWS Bedrock.

    Provides access to Claude, Llama, Mistral, and other models via AWS Bedrock.
    Uses the standard AWS credential chain for authentication (environment
    variables, IAM roles, or ``~/.aws/credentials``).

    Environment variables:
        AWS_DEFAULT_REGION: AWS region for the Bedrock endpoint
            (default: ``us-east-1``).
        AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: Optional explicit credentials.
            If not set, ``boto3`` falls back to IAM roles or shared credential files.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the Bedrock client.

        Args:
            config: LLM configuration with model settings.
                    ``api_key`` is not used — authentication is via AWS credentials.
        """
        super().__init__(config)
        self._client = None

        self._region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        # Default pricing (Claude Sonnet on Bedrock)
        self.usage.input_price_per_mtok = 3.0
        self.usage.output_price_per_mtok = 15.0

    @property
    def client(self):
        """Lazily initialize and return the Bedrock Runtime client."""
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 package is required for Bedrock provider. "
                    "Install with: pip install boto3"
                ) from e

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from Bedrock.

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

        Uses the Bedrock ``Converse`` API which provides a unified message
        format across all supported model providers.

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
            # Convert messages to Bedrock Converse format
            bedrock_messages = []
            for msg in messages:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

            kwargs: dict = {
                "modelId": self.config.model,
                "messages": bedrock_messages,
            }

            if system_prompt:
                kwargs["system"] = [{"text": system_prompt}]

            # Build inference config
            inference_config: dict = {}
            if temperature is not None:
                inference_config["temperature"] = temperature
            if max_tokens is not None:
                inference_config["maxTokens"] = max_tokens
            else:
                inference_config["maxTokens"] = 4096

            kwargs["inferenceConfig"] = inference_config

            response = self.client.converse(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content from response
            content = ""
            output_message = response.get("output", {}).get("message", {})
            for block in output_message.get("content", []):
                if "text" in block:
                    content += block["text"]

            # Extract token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            # Extract stop reason
            stop_reason = response.get("stopReason")

            result = LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.config.model,
                latency_ms=latency_ms,
                stop_reason=stop_reason,
            )

            self.usage.add(result)

            logger.debug(
                f"Bedrock response: {result.input_tokens} in, "
                f"{result.output_tokens} out, {latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            # Check for boto3-specific throttling errors
            error_code = ""
            if hasattr(e, "response"):
                error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "ThrottlingException" or "throttl" in str(e).lower():
                raise LLMRateLimitError(
                    f"Bedrock rate limit exceeded: {e}",
                    retry_after=None,
                ) from e

            if error_code == "ServiceQuotaExceededException":
                raise LLMRateLimitError(
                    f"Bedrock quota exceeded: {e}",
                    retry_after=None,
                ) from e

            raise LLMAPIError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                status_code=int(error_code) if error_code.isdigit() else None,
            ) from e
