"""Tests for JaxonFlow LLM client."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from jaxonflow.config import LLMConfig, LLMProvider
from jaxonflow.llm.client import LLMClient, LLMResponse, TokenUsage, create_client


class TestLLMResponse:
    def test_total_tokens(self):
        response = LLMResponse(
            content="hello",
            input_tokens=100,
            output_tokens=200,
            model="test-model",
            latency_ms=500.0,
        )
        assert response.total_tokens == 300

    def test_stop_reason(self):
        response = LLMResponse(
            content="hello",
            input_tokens=10,
            output_tokens=20,
            model="test-model",
            latency_ms=100.0,
            stop_reason="end_turn",
        )
        assert response.stop_reason == "end_turn"


class TestTokenUsage:
    def test_initial_state(self):
        usage = TokenUsage()
        assert usage.total_input_tokens == 0
        assert usage.total_output_tokens == 0
        assert usage.total_requests == 0
        assert usage.estimated_cost_usd == 0.0
        assert usage.average_latency_ms == 0.0

    def test_add_response(self):
        usage = TokenUsage()
        response = LLMResponse(
            content="hello",
            input_tokens=1000,
            output_tokens=500,
            model="test-model",
            latency_ms=200.0,
        )
        usage.add(response)

        assert usage.total_input_tokens == 1000
        assert usage.total_output_tokens == 500
        assert usage.total_requests == 1
        assert usage.total_latency_ms == 200.0

    def test_cumulative_tracking(self):
        usage = TokenUsage()
        for i in range(3):
            usage.add(LLMResponse(
                content="",
                input_tokens=100,
                output_tokens=50,
                model="test",
                latency_ms=100.0,
            ))

        assert usage.total_input_tokens == 300
        assert usage.total_output_tokens == 150
        assert usage.total_requests == 3
        assert usage.average_latency_ms == pytest.approx(100.0)

    def test_cost_estimation(self):
        usage = TokenUsage(
            total_input_tokens=1_000_000,
            total_output_tokens=1_000_000,
            input_price_per_mtok=3.0,
            output_price_per_mtok=15.0,
        )
        # 1M * $3/M + 1M * $15/M = $18
        assert usage.estimated_cost_usd == pytest.approx(18.0)


class TestExtractCode:
    """Test code extraction using a concrete mock client."""

    def _make_client(self):
        """Create a minimal concrete LLMClient for testing extraction."""
        config = LLMConfig(api_key="test")

        class TestClient(LLMClient):
            def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
                return LLMResponse(content=prompt, input_tokens=0, output_tokens=0,
                                   model="test", latency_ms=0)

            def generate_with_messages(self, messages, system_prompt=None, temperature=None, max_tokens=None):
                return LLMResponse(content="", input_tokens=0, output_tokens=0,
                                   model="test", latency_ms=0)

        return TestClient(config)

    def test_extract_python_code(self):
        client = self._make_client()
        response = LLMResponse(
            content='Some text\n```python\ndef hello():\n    return "world"\n```\nMore text',
            input_tokens=0, output_tokens=0, model="test", latency_ms=0,
        )
        code = client.extract_code(response, language="python")
        assert code is not None
        assert "def hello():" in code

    def test_extract_code_no_fence(self):
        client = self._make_client()
        response = LLMResponse(
            content="Just some text without code blocks",
            input_tokens=0, output_tokens=0, model="test", latency_ms=0,
        )
        code = client.extract_code(response, language="python")
        assert code is None

    def test_extract_generic_code_block(self):
        client = self._make_client()
        response = LLMResponse(
            content='```\nx = 1\n```',
            input_tokens=0, output_tokens=0, model="test", latency_ms=0,
        )
        code = client.extract_code(response, language="python")
        assert code is not None
        assert "x = 1" in code

    def test_extract_yaml(self):
        client = self._make_client()
        response = LLMResponse(
            content='```yaml\nkey: value\n```',
            input_tokens=0, output_tokens=0, model="test", latency_ms=0,
        )
        yaml = client.extract_yaml(response)
        assert yaml is not None
        assert "key: value" in yaml


class TestCreateClient:
    def test_create_anthropic_client(self):
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-key")
        client = create_client(config)
        from jaxonflow.llm.providers.anthropic import AnthropicProvider
        assert isinstance(client, AnthropicProvider)

    def test_create_openai_client(self):
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key="test-key")
        client = create_client(config)
        from jaxonflow.llm.providers.openai import OpenAIProvider
        assert isinstance(client, OpenAIProvider)

    def test_create_gemini_client(self):
        config = LLMConfig(provider=LLMProvider.GEMINI, api_key="test-key",
                           model="gemini-2.0-flash")
        client = create_client(config)
        from jaxonflow.llm.providers.gemini import GeminiProvider
        assert isinstance(client, GeminiProvider)
        assert client.usage.input_price_per_mtok == 1.25

    def test_create_openrouter_client(self):
        config = LLMConfig(provider=LLMProvider.OPENROUTER, api_key="test-key",
                           model="anthropic/claude-sonnet-4-20250514")
        client = create_client(config)
        from jaxonflow.llm.providers.openrouter import OpenRouterProvider
        assert isinstance(client, OpenRouterProvider)

    def test_create_local_client(self):
        config = LLMConfig(provider=LLMProvider.LOCAL, api_key="not-needed",
                           model="llama3",
                           base_url="http://localhost:11434/v1")
        client = create_client(config)
        from jaxonflow.llm.providers.local import LocalProvider
        assert isinstance(client, LocalProvider)
        assert client.usage.input_price_per_mtok == 0.0
        assert client.usage.output_price_per_mtok == 0.0

    def test_create_vertex_ai_client(self):
        config = LLMConfig(provider=LLMProvider.VERTEX_AI,
                           model="gemini-2.0-flash")
        client = create_client(config)
        from jaxonflow.llm.providers.vertex_ai import VertexAIProvider
        assert isinstance(client, VertexAIProvider)
        assert client.usage.input_price_per_mtok == 1.25
        assert client.usage.output_price_per_mtok == 5.0

    def test_create_bedrock_client(self):
        config = LLMConfig(provider=LLMProvider.BEDROCK,
                           model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        client = create_client(config)
        from jaxonflow.llm.providers.bedrock import BedrockProvider
        assert isinstance(client, BedrockProvider)
        assert client.usage.input_price_per_mtok == 3.0
        assert client.usage.output_price_per_mtok == 15.0
