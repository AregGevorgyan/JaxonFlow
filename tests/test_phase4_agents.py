"""Tests for Phase 4: Advanced Agents.

Tests the specialized agents (Planner, Coder, Debugger, ProfilerAgent)
using mock LLM clients.
"""

import pytest
from unittest.mock import MagicMock, patch

from jaxonflow.agents.base import AgentConfig, AgentRole, LLMAgent
from jaxonflow.agents.planner import PlannerAgent
from jaxonflow.agents.coder import CoderAgent
from jaxonflow.agents.debugger import DebuggerAgent
from jaxonflow.agents.profiler_agent import ProfilerAgent
from jaxonflow.agents.verification import KernelVerifier
from jaxonflow.agents.prompts import get_system_prompt
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec, ProfileResult
from jaxonflow.llm.client import LLMResponse


# ── Test Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    client = MagicMock()
    response = LLMResponse(
        content="mock response",
        input_tokens=100,
        output_tokens=200,
        model="test-model",
        latency_ms=500.0,
        stop_reason="end_turn",
    )
    client.generate.return_value = response
    return client


@pytest.fixture
def hardware():
    return HardwareContext.from_name("A100-80GB")


@pytest.fixture
def matmul_spec(hardware):
    return KernelSpec(
        operation="matmul",
        input_shapes=[(128, 64), (64, 256)],
        input_dtypes=["float32", "float32"],
        output_shape=(128, 256),
        output_dtype="float32",
        hardware=hardware,
    )


def make_agent_config(role: AgentRole) -> AgentConfig:
    return AgentConfig(
        role=role,
        model="test-model",
        temperature=0.0,
        max_tokens=2048,
        system_prompt=get_system_prompt(role),
    )


# ── PlannerAgent ────────────────────────────────────────────────────


class TestPlannerAgent:
    def test_create_plan(self, mock_llm, matmul_spec):
        mock_llm.generate.return_value = LLMResponse(
            content="strategy: tile_128x128",
            input_tokens=100,
            output_tokens=50,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        agent = PlannerAgent(make_agent_config(AgentRole.PLANNER), llm_client=mock_llm)
        plan = agent.create_plan(matmul_spec)
        assert isinstance(plan, str)
        assert "tile_128x128" in plan
        mock_llm.generate.assert_called_once()

    def test_revise_plan(self, mock_llm, matmul_spec):
        mock_llm.generate.return_value = LLMResponse(
            content="revised: larger_tiles",
            input_tokens=200,
            output_tokens=100,
            model="test",
            latency_ms=150,
            stop_reason="end_turn",
        )
        agent = PlannerAgent(make_agent_config(AgentRole.PLANNER), llm_client=mock_llm)
        plan = agent.revise_plan(matmul_spec, "old plan", "memory bottleneck")
        assert "larger_tiles" in plan
        # Check prompt contains feedback
        call_args = mock_llm.generate.call_args
        assert "memory bottleneck" in call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")

    def test_create_plan_non_response_object(self, mock_llm, matmul_spec):
        """Test fallback when LLM returns a non-LLMResponse object."""
        mock_llm.generate.return_value = "plain string plan"
        agent = PlannerAgent(make_agent_config(AgentRole.PLANNER), llm_client=mock_llm)
        plan = agent.create_plan(matmul_spec)
        assert plan == "plain string plan"


# ── CoderAgent ──────────────────────────────────────────────────────


class TestCoderAgent:
    def test_generate_kernel_with_code_extraction(self, mock_llm, matmul_spec):
        mock_llm.generate.return_value = LLMResponse(
            content='```python\nimport triton\n@triton.jit\ndef kernel(): pass\n```',
            input_tokens=100,
            output_tokens=300,
            model="test",
            latency_ms=200,
            stop_reason="end_turn",
        )
        mock_llm.extract_code.return_value = "@triton.jit\ndef kernel(): pass"
        agent = CoderAgent(make_agent_config(AgentRole.CODER), llm_client=mock_llm)
        code = agent.generate_kernel(matmul_spec, "plan: tile 128x128", [])
        assert "kernel" in code

    def test_generate_kernel_extracts_python_block(self, mock_llm, matmul_spec):
        """When extract_code is not available, falls back to markdown extraction."""
        mock_llm.generate.return_value = LLMResponse(
            content='Here is the code:\n```python\ndef my_kernel(): return 42\n```\nDone.',
            input_tokens=100,
            output_tokens=200,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        # Remove extract_code so the agent falls back to _extract_code
        del mock_llm.extract_code
        agent = CoderAgent(make_agent_config(AgentRole.CODER), llm_client=mock_llm)
        code = agent.generate_kernel(matmul_spec, "plan", [])
        assert "def my_kernel" in code

    def test_generate_kernel_extracts_generic_code_block(self, mock_llm, matmul_spec):
        mock_llm.generate.return_value = LLMResponse(
            content='```triton\nimport triton\n```',
            input_tokens=50,
            output_tokens=50,
            model="test",
            latency_ms=50,
            stop_reason="end_turn",
        )
        del mock_llm.extract_code
        agent = CoderAgent(make_agent_config(AgentRole.CODER), llm_client=mock_llm)
        code = agent.generate_kernel(matmul_spec, "plan", [])
        assert "import triton" in code

    def test_generate_kernel_with_history(self, mock_llm, matmul_spec):
        mock_llm.generate.return_value = LLMResponse(
            content="def kernel(): pass",
            input_tokens=200,
            output_tokens=100,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        del mock_llm.extract_code
        agent = CoderAgent(make_agent_config(AgentRole.CODER), llm_client=mock_llm)
        history = [
            {"iteration": 0, "phase": "compilation_failed", "error": "syntax error", "debug_feedback": "fix line 5"},
            {"iteration": 1, "phase": "verification_failed", "error": "mismatch", "debug_feedback": "check boundary"},
        ]
        code = agent.generate_kernel(matmul_spec, "plan", history)
        assert isinstance(code, str)
        # Verify history was included in prompt
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
        assert "Iteration 0" in prompt or "Previous Attempts" in prompt

    def test_extract_code_no_fences(self):
        """When there are no code fences, return the raw text."""
        agent = CoderAgent(
            make_agent_config(AgentRole.CODER),
            llm_client=MagicMock(),
        )
        text = "import triton\ndef kernel(): pass"
        assert agent._extract_code(text) == text


# ── DebuggerAgent ───────────────────────────────────────────────────


class TestDebuggerAgent:
    def test_analyze_error(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            content="diagnosis: syntax error on line 5",
            input_tokens=100,
            output_tokens=100,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        agent = DebuggerAgent(make_agent_config(AgentRole.DEBUGGER), llm_client=mock_llm)
        result = agent.analyze_error("bad code", "SyntaxError line 5", "compilation")
        assert "syntax error" in result.lower()

    def test_analyze_mismatch(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            content="fix: boundary handling issue",
            input_tokens=100,
            output_tokens=100,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        agent = DebuggerAgent(make_agent_config(AgentRole.DEBUGGER), llm_client=mock_llm)
        result = agent.analyze_mismatch("code", "expected_array", "actual_array")
        assert isinstance(result, str)
        # analyze_mismatch delegates to analyze_error
        mock_llm.generate.assert_called_once()


# ── ProfilerAgent ───────────────────────────────────────────────────


class TestProfilerAgent:
    def test_analyze_performance(self, mock_llm):
        mock_llm.generate.return_value = LLMResponse(
            content="bottleneck: memory_bound, suggestion: increase tile size",
            input_tokens=200,
            output_tokens=150,
            model="test",
            latency_ms=200,
            stop_reason="end_turn",
        )
        agent = ProfilerAgent(make_agent_config(AgentRole.PROFILER), llm_client=mock_llm)
        profile = ProfileResult(
            execution_time_us=150.0,
            speedup_vs_reference=1.1,
            memory_bandwidth_utilization=0.85,
            compute_utilization=0.3,
            achieved_occupancy=0.7,
            primary_bottleneck="memory",
        )
        hardware = HardwareContext.from_name("A100-80GB")
        result = agent.analyze_performance("kernel code", profile, hardware)
        assert "memory_bound" in result or "tile" in result

    def test_analyze_performance_no_optional_fields(self, mock_llm):
        """Test with minimal ProfileResult (no optional fields)."""
        mock_llm.generate.return_value = LLMResponse(
            content="analysis result",
            input_tokens=100,
            output_tokens=50,
            model="test",
            latency_ms=100,
            stop_reason="end_turn",
        )
        agent = ProfilerAgent(make_agent_config(AgentRole.PROFILER), llm_client=mock_llm)
        profile = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=0.9,
        )
        hardware = HardwareContext.from_name("A100-80GB")
        result = agent.analyze_performance("code", profile, hardware)
        assert isinstance(result, str)


# ── KernelVerifier (Phase 4 agent) ─────────────────────────────────


class TestKernelVerifierAgent:
    """Test KernelVerifier as used by the orchestrator."""

    @pytest.fixture
    def verifier(self):
        config = AgentConfig(
            role=AgentRole.VERIFIER,
            model="deterministic",
            system_prompt=get_system_prompt(AgentRole.VERIFIER),
        )
        return KernelVerifier(config)

    def test_run_correct_kernel(self, verifier):
        import numpy as np
        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y,
        )
        # CompiledKernel-like object
        kernel = MagicMock()
        kernel.side_effect = lambda *args: args[0] + args[1]
        kernel.__call__ = kernel
        result = verifier.run(kernel, spec)
        assert result.correct is True

    def test_run_no_reference_uses_default(self, verifier):
        """Verifier has default references for common ops."""
        import numpy as np
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 8), (8, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
            reference_impl=None,
        )
        # Correct matmul
        kernel = lambda a, b: a @ b
        result = verifier.run(kernel, spec)
        assert result.correct is True

    def test_run_unknown_op_no_reference(self, verifier):
        spec = KernelSpec(
            operation="custom_exotic_op",
            input_shapes=[(10,)],
            input_dtypes=["float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=None,
        )
        kernel = lambda x: x
        result = verifier.run(kernel, spec)
        # No reference and no default → passes with note
        assert result.correct is True
        assert "No reference" in (result.test_case_description or "")


# ── Agent Prompts ───────────────────────────────────────────────────


class TestAgentPrompts:
    def test_all_roles_have_prompts(self):
        for role in AgentRole:
            prompt = get_system_prompt(role)
            assert isinstance(prompt, str)
            assert len(prompt) > 50  # Non-trivial prompt

    def test_planner_prompt_mentions_yaml(self):
        prompt = get_system_prompt(AgentRole.PLANNER)
        assert "yaml" in prompt.lower() or "YAML" in prompt

    def test_coder_prompt_mentions_triton(self):
        prompt = get_system_prompt(AgentRole.CODER)
        assert "triton" in prompt.lower() or "Triton" in prompt

    def test_debugger_prompt_mentions_fixes(self):
        prompt = get_system_prompt(AgentRole.DEBUGGER)
        assert "fix" in prompt.lower()

    def test_profiler_prompt_mentions_bottleneck(self):
        prompt = get_system_prompt(AgentRole.PROFILER)
        assert "bottleneck" in prompt.lower()
