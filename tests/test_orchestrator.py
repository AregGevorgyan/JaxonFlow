"""Tests for JaxonFlow multi-agent orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jaxonflow.agents.base import AgentConfig, AgentRole
from jaxonflow.agents.orchestrator import LLMAgent, MultiAgentOrchestrator
from jaxonflow.config import AgentBackendConfig, CacheConfig, LLMConfig, LLMProvider
from jaxonflow.llm.client import LLMResponse
from jaxonflow.spec import CompiledKernel, KernelSpec


@pytest.fixture
def mock_config(tmp_path):
    """Config that won't try to connect to a real LLM."""
    return AgentBackendConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="test-model",
            api_key="test-key",
        ),
        cache=CacheConfig(enabled=False),
        hardware_target="A100-80GB",
    )


@pytest.fixture
def mock_orchestrator(mock_config, mock_llm_client):
    """Orchestrator with mocked LLM client."""
    orch = MultiAgentOrchestrator(mock_config)
    # Inject mock client to avoid real API calls
    orch._llm_client = mock_llm_client
    return orch


class TestLLMAgent:
    def test_run_calls_llm(self, mock_llm_client):
        config = AgentConfig(
            role=AgentRole.CODER,
            model="test-model",
            temperature=0.2,
            max_tokens=4096,
            system_prompt="You are a coder.",
        )
        agent = LLMAgent(config, llm_client=mock_llm_client)

        response = agent.run("Generate a kernel")
        assert isinstance(response, LLMResponse)
        mock_llm_client.generate.assert_called_once_with(
            prompt="Generate a kernel",
            system_prompt="You are a coder.",
            temperature=0.2,
            max_tokens=4096,
        )

    def test_run_no_args_raises(self, mock_llm_client):
        config = AgentConfig(role=AgentRole.CODER, model="test-model")
        agent = LLMAgent(config, llm_client=mock_llm_client)

        with pytest.raises(ValueError, match="requires at least a prompt"):
            agent.run()

    def test_run_temperature_override(self, mock_llm_client):
        config = AgentConfig(
            role=AgentRole.CODER,
            model="test-model",
            temperature=0.2,
        )
        agent = LLMAgent(config, llm_client=mock_llm_client)

        agent.run("prompt", temperature=0.9)
        mock_llm_client.generate.assert_called_once_with(
            prompt="prompt",
            system_prompt="",
            temperature=0.9,
            max_tokens=4096,
        )


class TestMultiAgentOrchestrator:
    def test_initialization(self, mock_config):
        orch = MultiAgentOrchestrator(mock_config)
        assert orch.config is mock_config
        assert orch._llm_client is None
        assert orch._agents is None

    def test_lazy_agent_init(self, mock_orchestrator):
        agents = mock_orchestrator.agents
        assert AgentRole.PLANNER in agents
        assert AgentRole.CODER in agents
        assert AgentRole.DEBUGGER in agents
        assert AgentRole.PROFILER in agents
        assert AgentRole.VERIFIER in agents
        assert len(agents) == 5

    def test_planner_is_llm_agent(self, mock_orchestrator):
        planner = mock_orchestrator.agents[AgentRole.PLANNER]
        assert isinstance(planner, LLMAgent)
        assert planner.config.role == AgentRole.PLANNER

    def test_verifier_is_kernel_verifier(self, mock_orchestrator):
        from jaxonflow.agents.verification import KernelVerifier

        verifier = mock_orchestrator.agents[AgentRole.VERIFIER]
        assert isinstance(verifier, KernelVerifier)

    def test_generate_kernel_with_ref(self, mock_orchestrator, matmul_spec_with_ref):
        """Test kernel generation with a reference implementation."""
        # Override mock to return matmul-compatible code
        mock_orchestrator._llm_client.extract_code.return_value = (
            "def wrapper(a, b):\n    return a @ b\n"
        )
        kernel = mock_orchestrator.generate_kernel(matmul_spec_with_ref)

        assert kernel is not None
        assert isinstance(kernel, CompiledKernel)
        assert kernel.spec.operation == "matmul"
        assert kernel.iterations_to_generate >= 1

    def test_generate_kernel_syntax_error(self, mock_orchestrator, mock_llm_client):
        """Test that syntax errors in generated code trigger debugger."""
        # Make the coder return syntactically invalid code
        bad_code_response = LLMResponse(
            content='```python\ndef broken(\n```',
            input_tokens=100,
            output_tokens=50,
            model="test",
            latency_ms=100,
        )
        mock_llm_client.generate.return_value = bad_code_response
        mock_llm_client.extract_code.return_value = "def broken("

        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y,
        )

        # Set max_iterations to 1 so it gives up after one failed attempt
        mock_orchestrator.config.agent.max_iterations = 1
        kernel = mock_orchestrator.generate_kernel(spec)
        assert kernel is None  # Should fail with invalid code

    def test_compile_kernel_empty_code(self, mock_orchestrator, matmul_spec):
        result = mock_orchestrator._compile_kernel("", matmul_spec)
        assert result.success is False
        assert "Empty" in result.error

    def test_compile_kernel_syntax_error(self, mock_orchestrator, matmul_spec):
        result = mock_orchestrator._compile_kernel("def foo(:", matmul_spec)
        assert result.success is False
        assert "Syntax error" in result.error

    def test_compile_kernel_valid_code(self, mock_orchestrator, matmul_spec_with_ref):
        code = "def wrapper(a, b):\n    return a @ b\n"
        result = mock_orchestrator._compile_kernel(code, matmul_spec_with_ref)
        assert result.success is True
        assert result.kernel is not None

    def test_get_usage_stats_no_client(self, mock_config):
        orch = MultiAgentOrchestrator(mock_config)
        stats = orch.get_usage_stats()
        assert stats["total_input_tokens"] == 0
        assert stats["estimated_cost_usd"] == 0.0

    def test_get_usage_stats_with_client(self, mock_orchestrator):
        stats = mock_orchestrator.get_usage_stats()
        assert "total_input_tokens" in stats
        assert "estimated_cost_usd" in stats
