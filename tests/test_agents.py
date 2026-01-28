"""Tests for JaxonFlow agents."""

from __future__ import annotations

import numpy as np
import pytest

from jaxonflow.agents.base import Agent, AgentConfig, AgentRole
from jaxonflow.agents.prompts import get_system_prompt
from jaxonflow.agents.verification import KernelVerifier
from jaxonflow.spec import KernelSpec, VerificationResult


class TestAgentRole:
    def test_all_roles_exist(self):
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.CODER.value == "coder"
        assert AgentRole.DEBUGGER.value == "debugger"
        assert AgentRole.PROFILER.value == "profiler"
        assert AgentRole.VERIFIER.value == "verifier"

    def test_role_count(self):
        assert len(AgentRole) == 5


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig(role=AgentRole.CODER, model="test-model")
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.system_prompt == ""

    def test_custom_values(self):
        config = AgentConfig(
            role=AgentRole.PLANNER,
            model="claude-sonnet-4-20250514",
            temperature=0.8,
            max_tokens=2048,
            system_prompt="You are a planner.",
        )
        assert config.temperature == 0.8
        assert config.max_tokens == 2048


class TestAgentBase:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Agent(AgentConfig(role=AgentRole.CODER, model="test"))


class TestSystemPrompts:
    def test_all_roles_have_prompts(self):
        for role in AgentRole:
            prompt = get_system_prompt(role)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Prompts should be substantial

    def test_planner_prompt_content(self):
        prompt = get_system_prompt(AgentRole.PLANNER)
        assert "optimization" in prompt.lower()
        assert "YAML" in prompt
        assert "strategy" in prompt.lower()

    def test_coder_prompt_content(self):
        prompt = get_system_prompt(AgentRole.CODER)
        assert "Triton" in prompt
        assert "kernel" in prompt.lower()
        assert "COMPLETE" in prompt

    def test_debugger_prompt_content(self):
        prompt = get_system_prompt(AgentRole.DEBUGGER)
        assert "debug" in prompt.lower()
        assert "fix" in prompt.lower()

    def test_profiler_prompt_content(self):
        prompt = get_system_prompt(AgentRole.PROFILER)
        assert "profil" in prompt.lower()
        assert "bottleneck" in prompt.lower()


class TestKernelVerifier:
    def test_correct_add_kernel(self, add_spec):
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        result = verifier.run(lambda x, y: x + y, add_spec)
        assert result.correct is True

    def test_incorrect_kernel(self, add_spec):
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        # Subtraction instead of addition
        result = verifier.run(lambda x, y: x - y, add_spec)
        assert result.correct is False
        assert result.max_abs_error is not None
        assert result.max_abs_error > 0

    def test_matmul_verification(self, matmul_spec_with_ref):
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        result = verifier.run(lambda a, b: a @ b, matmul_spec_with_ref)
        assert result.correct is True

    def test_softmax_verification(self, softmax_spec):
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        result = verifier.run(softmax_spec.reference_impl, softmax_spec)
        assert result.correct is True

    def test_default_reference_matmul(self):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(8, 4), (4, 8)],
            input_dtypes=["float32", "float32"],
            output_shape=(8, 8),
            output_dtype="float32",
        )
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        result = verifier.run(lambda a, b: a @ b, spec)
        assert result.correct is True

    def test_unknown_operation_no_ref(self):
        spec = KernelSpec(
            operation="custom_fused_op",
            input_shapes=[(10,)],
            input_dtypes=["float32"],
            output_shape=(10,),
            output_dtype="float32",
        )
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )
        # Should return pass with note about no reference
        result = verifier.run(lambda x: x, spec)
        assert result.correct is True
        assert "No reference" in (result.test_case_description or "")

    def test_kernel_execution_error(self, add_spec):
        verifier = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic")
        )

        def failing_kernel(*args):
            raise RuntimeError("GPU error")

        result = verifier.run(failing_kernel, add_spec)
        assert result.correct is False
        assert "execution error" in (result.test_case_description or "")

    def test_custom_tolerances(self):
        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y,
        )

        # Kernel with small error
        def noisy_add(x, y):
            return x + y + np.random.normal(0, 1e-6, size=x.shape).astype(x.dtype)

        # Should pass with loose tolerance
        verifier_loose = KernelVerifier(
            AgentConfig(role=AgentRole.VERIFIER, model="deterministic"),
            rtol=1e-3,
            atol=1e-3,
        )
        result = verifier_loose.run(noisy_add, spec)
        assert result.correct is True
