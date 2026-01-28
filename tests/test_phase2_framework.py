"""Tests for Phase 2: Framework Integration.

Tests JAX and PyTorch integration modules without requiring actual
JAX/PyTorch GPU execution - focuses on module structure, imports,
and API surface.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from jaxonflow.config import AgentBackendConfig, LLMConfig, LLMProvider, CacheConfig
from jaxonflow.dispatch import AgentBackend
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec


# ── JAX subpackage ──────────────────────────────────────────────────


class TestJaxPackageImports:
    """Verify jaxonflow.jax subpackage is importable and has correct exports."""

    def test_jax_package_importable(self):
        from jaxonflow.jax import JaxDispatcher
        assert JaxDispatcher is not None

    def test_jax_dispatch_importable(self):
        from jaxonflow.jax.dispatch import JaxDispatcher
        assert JaxDispatcher is not None

    def test_jax_lowering_importable(self):
        from jaxonflow.jax.lowering import register_agent_lowerings
        assert callable(register_agent_lowerings)

    def test_jax_pallas_importable(self):
        from jaxonflow.jax.pallas_backend import PallasAgentBackend
        assert PallasAgentBackend is not None


class TestJaxDispatcher:
    """Test the JaxDispatcher class."""

    @pytest.fixture
    def backend(self, tmp_path):
        config = AgentBackendConfig(
            llm=LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="test-model",
                api_key="test-key",
            ),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
            hardware_target="A100-80GB",
        )
        return AgentBackend(config)

    @pytest.fixture
    def dispatcher(self, backend):
        from jaxonflow.jax.dispatch import JaxDispatcher
        return JaxDispatcher(backend)

    def test_get_kernel_spec(self, dispatcher):
        spec = dispatcher.get_kernel_spec(
            primitive_name="dot_general",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            params={"dimension_numbers": (((1,), (0,)), ((), ()))},
        )
        assert isinstance(spec, KernelSpec)
        assert spec.operation == "dot_general"
        assert spec.input_shapes == [(128, 64), (64, 256)]
        assert spec.output_shape == (128, 256)
        assert spec.hardware is not None
        assert spec.hardware.name == "NVIDIA A100-80GB"

    def test_get_kernel_spec_preserves_params(self, dispatcher):
        params = {"stride": (1, 1), "padding": "SAME"}
        spec = dispatcher.get_kernel_spec(
            primitive_name="conv2d",
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=["float32"],
            output_shape=(1, 64, 112, 112),
            output_dtype="float32",
            params=params,
        )
        assert spec.parameters == params


# ── PyTorch subpackage ──────────────────────────────────────────────


class TestPyTorchPackageImports:
    """Verify jaxonflow.pytorch subpackage is importable and has correct exports."""

    def test_pytorch_package_importable(self):
        from jaxonflow.pytorch import AgentCompilerBackend
        assert AgentCompilerBackend is not None

    def test_pytorch_compiler_backend_importable(self):
        from jaxonflow.pytorch.compiler_backend import AgentCompilerBackend
        assert AgentCompilerBackend is not None

    def test_pytorch_custom_ops_importable(self):
        from jaxonflow.pytorch.custom_ops import AgentOpRegistry
        assert AgentOpRegistry is not None

    def test_pytorch_triton_wrapper_importable(self):
        from jaxonflow.pytorch.triton_wrapper import TritonKernelWrapper, agent_matmul
        assert TritonKernelWrapper is not None
        assert callable(agent_matmul)

    def test_pytorch_inductor_importable(self):
        from jaxonflow.pytorch.inductor_extension import InductorAgentExtension
        assert InductorAgentExtension is not None


class TestAgentCompilerBackend:
    """Test the PyTorch AgentCompilerBackend class."""

    def test_init_default_config(self):
        from jaxonflow.pytorch.compiler_backend import AgentCompilerBackend
        backend = AgentCompilerBackend()
        assert backend.config is not None
        assert backend.backend is not None

    def test_init_custom_config(self, tmp_path):
        from jaxonflow.pytorch.compiler_backend import AgentCompilerBackend
        config = AgentBackendConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="test", api_key="test"),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
        )
        backend = AgentCompilerBackend(config)
        assert backend.config is config

    def test_call_with_mock_graph(self):
        """Test that __call__ handles a mock FX graph module."""
        import torch
        from jaxonflow.pytorch.compiler_backend import AgentCompilerBackend

        backend = AgentCompilerBackend()

        # Create a simple FX graph module
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        example_input = torch.randn(4, 4)

        # Should return a callable (passthrough in Phase 2)
        result = backend(gm, [example_input])
        assert callable(result)

        # Verify passthrough works
        out = result(example_input)
        expected = example_input + 1
        assert torch.allclose(out, expected)


class TestTritonKernelWrapper:
    """Test the TritonKernelWrapper class."""

    def test_init(self, tmp_path):
        from jaxonflow.pytorch.triton_wrapper import TritonKernelWrapper
        config = AgentBackendConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="test", api_key="test"),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
        )
        backend = AgentBackend(config)
        wrapper = TritonKernelWrapper(backend)
        assert wrapper.backend is backend
        assert wrapper.kernel_cache == {}

    def test_create_matmul_returns_autograd_class(self, tmp_path):
        from jaxonflow.pytorch.triton_wrapper import TritonKernelWrapper
        import torch

        config = AgentBackendConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="test", api_key="test"),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
        )
        backend = AgentBackend(config)
        wrapper = TritonKernelWrapper(backend)
        matmul_cls = wrapper.create_matmul()

        # Should be a subclass of torch.autograd.Function
        assert issubclass(matmul_cls, torch.autograd.Function)


class TestInductorAgentExtension:
    """Test the InductorAgentExtension class."""

    def test_init(self, tmp_path):
        from jaxonflow.pytorch.inductor_extension import InductorAgentExtension
        config = AgentBackendConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="test", api_key="test"),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
        )
        backend = AgentBackend(config)
        ext = InductorAgentExtension(backend)
        assert ext.backend is backend
        assert ext.registered is False

    def test_should_use_agent(self, tmp_path):
        from jaxonflow.pytorch.inductor_extension import InductorAgentExtension
        config = AgentBackendConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="test", api_key="test"),
            cache=CacheConfig(enabled=True, cache_dir=tmp_path / "cache"),
        )
        backend = AgentBackend(config)
        ext = InductorAgentExtension(backend)
        assert ext._should_use_agent() is True
