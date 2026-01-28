"""Tests for JaxonFlow dispatch/backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxonflow.cache import CacheEntry
from jaxonflow.config import AgentBackendConfig, CacheConfig, LLMConfig, LLMProvider
from jaxonflow.dispatch import AgentBackend
from jaxonflow.exceptions import FallbackError, KernelGenerationError
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import CompiledKernel, KernelSpec


@pytest.fixture
def backend(tmp_path):
    """Create an AgentBackend with mocked LLM."""
    config = AgentBackendConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",
        ),
        cache=CacheConfig(
            enabled=True,
            cache_dir=tmp_path / "test_cache",
        ),
        hardware_target="A100-80GB",
    )
    backend = AgentBackend(config)

    # Mock the agent system to avoid real LLM calls
    mock_kernel = CompiledKernel(
        spec=KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
        ),
        code="# generated code",
        callable=lambda a, b: a @ b,
    )
    backend.agent_system = MagicMock()
    backend.agent_system.generate_kernel.return_value = mock_kernel
    backend.agent_system.get_usage_stats.return_value = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_requests": 0,
        "estimated_cost_usd": 0.0,
        "average_latency_ms": 0.0,
    }

    return backend


class TestAgentBackend:
    def test_initialization(self, backend):
        assert isinstance(backend.hardware_context, HardwareContext)
        assert backend.hardware_context.name == "NVIDIA A100-80GB"
        assert backend.kernel_cache is not None

    def test_hardware_auto_detect(self, tmp_path):
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test"),
            cache=CacheConfig(enabled=False),
            hardware_target="auto",
        )
        backend = AgentBackend(config)
        assert backend.hardware_context is not None

    def test_hardware_explicit(self, tmp_path):
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test"),
            cache=CacheConfig(enabled=False),
            hardware_target="H100-SXM",
        )
        backend = AgentBackend(config)
        assert backend.hardware_context.name == "NVIDIA H100-SXM"

    def test_get_or_generate_kernel(self, backend, matmul_spec_with_ref):
        kernel = backend.get_or_generate_kernel(matmul_spec_with_ref)

        assert kernel is not None
        assert isinstance(kernel, CompiledKernel)
        assert kernel.spec.operation == "matmul"

    def test_cache_hit(self, backend, matmul_spec_with_ref):
        # First call generates
        kernel1 = backend.get_or_generate_kernel(matmul_spec_with_ref)
        assert kernel1 is not None

        # Second call should hit cache
        kernel2 = backend.get_or_generate_kernel(matmul_spec_with_ref)
        assert kernel2 is not None
        assert kernel2.code == kernel1.code

        # Agent system should only be called once
        backend.agent_system.generate_kernel.assert_called_once()

    def test_hardware_injection(self, backend):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            hardware=None,
        )
        kernel = backend.get_or_generate_kernel(spec)
        assert spec.hardware is not None
        assert spec.hardware.name == "NVIDIA A100-80GB"

    def test_fallback_kernel_with_ref(self, backend):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
            reference_impl=lambda a, b: a @ b,
        )
        kernel = backend._fallback_kernel(spec)
        assert kernel is not None
        assert kernel.callable is not None

        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        result = kernel(a, b)
        np.testing.assert_allclose(result, a @ b)

    def test_fallback_kernel_no_ref_raises(self, backend):
        spec = KernelSpec(
            operation="custom_op",
            input_shapes=[(4, 4)],
            input_dtypes=["float32"],
            output_shape=(4, 4),
            output_dtype="float32",
        )
        with pytest.raises(FallbackError):
            backend._fallback_kernel(spec)

    def test_generation_failure_with_fallback(self, backend):
        backend.agent_system.generate_kernel.return_value = None

        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
            reference_impl=lambda a, b: a @ b,
        )
        kernel = backend.get_or_generate_kernel(spec)
        assert kernel is not None

    def test_generation_failure_no_fallback(self, tmp_path):
        config = AgentBackendConfig(
            llm=LLMConfig(api_key="test"),
            cache=CacheConfig(enabled=False),
            hardware_target="A100-80GB",
            fallback_to_xla=False,
            fallback_to_inductor=False,
        )
        be = AgentBackend(config)
        be.agent_system = MagicMock()
        be.agent_system.generate_kernel.return_value = None

        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
        )
        with pytest.raises(KernelGenerationError):
            be.get_or_generate_kernel(spec)

    def test_cache_stats(self, backend, matmul_spec_with_ref):
        stats = backend.cache_stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_llm_usage(self, backend):
        usage = backend.llm_usage
        assert "total_input_tokens" in usage
        assert "estimated_cost_usd" in usage


class TestRehydration:
    def test_make_callable_with_ref(self, backend, matmul_spec_with_ref):
        code = "import triton  # This will fail without GPU"
        callable_fn = backend._make_callable_from_code(code, matmul_spec_with_ref)
        assert callable_fn is matmul_spec_with_ref.reference_impl

    def test_make_callable_with_wrapper(self, backend, matmul_spec):
        code = "def wrapper(a, b):\n    return a + b\n"
        callable_fn = backend._make_callable_from_code(code, matmul_spec)
        assert callable_fn is not None
        assert callable_fn(1, 2) == 3

    def test_make_callable_noop_fallback(self, backend):
        spec = KernelSpec(
            operation="custom_op",
            input_shapes=[(10,)],
            input_dtypes=["float32"],
            output_shape=(10,),
            output_dtype="float32",
        )
        code = "import nonexistent_module"
        callable_fn = backend._make_callable_from_code(code, spec)
        assert callable_fn is not None
        assert callable_fn("test") == "test"
