"""Integration tests for JaxonFlow (no GPU/LLM required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxonflow import AgentBackend, AgentBackendConfig
from jaxonflow.config import CacheConfig, LLMConfig, LLMProvider
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import CompiledKernel, KernelSpec


class TestEndToEnd:
    """End-to-end tests with mocked LLM client."""

    def test_full_flow_matmul(self, tmp_path):
        """Test full flow: config -> backend -> generate -> cache -> reuse."""
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test-key"),
            cache=CacheConfig(
                enabled=True,
                cache_dir=tmp_path / "e2e_cache",
            ),
            hardware_target="A100-80GB",
        )
        backend = AgentBackend(config)

        # Mock the agent system
        ref_impl = lambda a, b: a @ b
        mock_kernel = CompiledKernel(
            spec=KernelSpec(
                operation="matmul",
                input_shapes=[(16, 8), (8, 32)],
                input_dtypes=["float32", "float32"],
                output_shape=(16, 32),
                output_dtype="float32",
            ),
            code="def wrapper(a, b):\n    return a @ b\n",
            callable=ref_impl,
        )
        backend.agent_system = MagicMock()
        backend.agent_system.generate_kernel.return_value = mock_kernel
        backend.agent_system.get_usage_stats.return_value = {
            "total_input_tokens": 500,
            "total_output_tokens": 300,
            "total_requests": 2,
            "estimated_cost_usd": 0.006,
            "average_latency_ms": 1500.0,
        }

        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(16, 8), (8, 32)],
            input_dtypes=["float32", "float32"],
            output_shape=(16, 32),
            output_dtype="float32",
            reference_impl=ref_impl,
        )

        # Generate kernel
        kernel = backend.get_or_generate_kernel(spec)
        assert kernel is not None
        assert kernel.code == "def wrapper(a, b):\n    return a @ b\n"

        # Verify it works
        a = np.random.randn(16, 8).astype(np.float32)
        b = np.random.randn(8, 32).astype(np.float32)
        result = kernel(a, b)
        np.testing.assert_allclose(result, a @ b)

        # Verify cache hit on second call
        kernel2 = backend.get_or_generate_kernel(spec)
        assert kernel2 is not None
        backend.agent_system.generate_kernel.assert_called_once()

        # Verify stats
        assert backend.cache_stats["hits"] == 1
        assert backend.cache_stats["misses"] == 1  # First call was a miss

    def test_multiple_operations(self, tmp_path):
        """Test generating kernels for different operations."""
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test-key"),
            cache=CacheConfig(
                enabled=True,
                cache_dir=tmp_path / "multi_cache",
            ),
            hardware_target="A100-80GB",
        )
        backend = AgentBackend(config)

        ops = [
            ("add", [(10,), (10,)], ["float32", "float32"], (10,), lambda x, y: x + y),
            ("mul", [(10,), (10,)], ["float32", "float32"], (10,), lambda x, y: x * y),
        ]

        for op_name, shapes, dtypes, out_shape, ref_impl in ops:
            spec = KernelSpec(
                operation=op_name,
                input_shapes=shapes,
                input_dtypes=dtypes,
                output_shape=out_shape,
                output_dtype="float32",
                reference_impl=ref_impl,
            )

            mock_kernel = CompiledKernel(
                spec=spec,
                code=f"# {op_name} kernel",
                callable=ref_impl,
            )
            backend.agent_system = MagicMock()
            backend.agent_system.generate_kernel.return_value = mock_kernel
            backend.agent_system.get_usage_stats.return_value = {
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_requests": 0, "estimated_cost_usd": 0.0,
                "average_latency_ms": 0.0,
            }

            kernel = backend.get_or_generate_kernel(spec)
            assert kernel is not None
            assert kernel.spec.operation == op_name

    def test_disabled_cache(self, tmp_path):
        """Test that disabled cache always triggers generation."""
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test-key"),
            cache=CacheConfig(enabled=False),
            hardware_target="A100-80GB",
        )
        backend = AgentBackend(config)

        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y,
        )

        mock_kernel = CompiledKernel(
            spec=spec, code="# code", callable=lambda x, y: x + y,
        )
        backend.agent_system = MagicMock()
        backend.agent_system.generate_kernel.return_value = mock_kernel
        backend.agent_system.get_usage_stats.return_value = {
            "total_input_tokens": 0, "total_output_tokens": 0,
            "total_requests": 0, "estimated_cost_usd": 0.0,
            "average_latency_ms": 0.0,
        }

        # Call twice - both should trigger generation
        backend.get_or_generate_kernel(spec)
        backend.get_or_generate_kernel(spec)
        assert backend.agent_system.generate_kernel.call_count == 2


class TestPackageImports:
    """Test that the public API imports correctly."""

    def test_top_level_imports(self):
        from jaxonflow import (
            AgentBackend,
            AgentBackendConfig,
            AgentConfig,
            CacheConfig,
            CompiledKernel,
            HardwareContext,
            KernelSpec,
            LLMConfig,
            LLMProvider,
        )

        assert AgentBackend is not None
        assert AgentBackendConfig is not None
        assert KernelSpec is not None

    def test_version(self):
        import jaxonflow
        assert hasattr(jaxonflow, "__version__")
        assert jaxonflow.__version__ == "0.1.0"

    def test_llm_imports(self):
        from jaxonflow.llm import LLMClient, LLMResponse, create_client
        assert LLMClient is not None

    def test_provider_imports(self):
        from jaxonflow.llm.providers import (
            AnthropicProvider, GeminiProvider, LocalProvider,
            OpenAIProvider, OpenRouterProvider,
        )
        assert AnthropicProvider is not None
        assert GeminiProvider is not None
        assert OpenAIProvider is not None
        assert OpenRouterProvider is not None
        assert LocalProvider is not None

    def test_agent_imports(self):
        from jaxonflow.agents import Agent, AgentRole, MultiAgentOrchestrator
        assert AgentRole.PLANNER is not None

    def test_hardware_profiles(self):
        profiles = HardwareContext.available_profiles()
        assert len(profiles) >= 4
        for name in profiles:
            hw = HardwareContext.from_name(name)
            ctx = hw.to_prompt_context()
            assert len(ctx) > 100
