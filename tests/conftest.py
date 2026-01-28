"""Shared test fixtures for JaxonFlow tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jaxonflow.cache import KernelCache
from jaxonflow.config import (
    AgentBackendConfig,
    AgentConfig,
    CacheConfig,
    LLMConfig,
    LLMProvider,
)
from jaxonflow.hardware import HardwareContext
from jaxonflow.llm.client import LLMResponse
from jaxonflow.spec import KernelSpec


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary directory for cache testing."""
    return tmp_path / "cache"


@pytest.fixture
def cache_config(tmp_cache_dir):
    """Provide a CacheConfig pointing to a temp directory."""
    return CacheConfig(enabled=True, cache_dir=tmp_cache_dir, max_entries=100)


@pytest.fixture
def kernel_cache(cache_config):
    """Provide a KernelCache with a temp directory."""
    cache = KernelCache(cache_config)
    yield cache
    cache.close()


@pytest.fixture
def disabled_cache():
    """Provide a disabled KernelCache."""
    return KernelCache(CacheConfig(enabled=False))


@pytest.fixture
def a100_hardware():
    """Provide an A100-80GB HardwareContext."""
    return HardwareContext.from_name("A100-80GB")


@pytest.fixture
def h100_hardware():
    """Provide an H100-SXM HardwareContext."""
    return HardwareContext.from_name("H100-SXM")


@pytest.fixture
def matmul_spec(a100_hardware):
    """Provide a matmul KernelSpec."""
    return KernelSpec(
        operation="matmul",
        input_shapes=[(128, 64), (64, 256)],
        input_dtypes=["float32", "float32"],
        output_shape=(128, 256),
        output_dtype="float32",
        hardware=a100_hardware,
    )


@pytest.fixture
def matmul_spec_with_ref(a100_hardware):
    """Provide a matmul KernelSpec with reference implementation."""
    import numpy as np

    return KernelSpec(
        operation="matmul",
        input_shapes=[(128, 64), (64, 256)],
        input_dtypes=["float32", "float32"],
        output_shape=(128, 256),
        output_dtype="float32",
        hardware=a100_hardware,
        reference_impl=lambda a, b: a @ b,
    )


@pytest.fixture
def add_spec():
    """Provide an elementwise add KernelSpec."""
    return KernelSpec(
        operation="add",
        input_shapes=[(10,), (10,)],
        input_dtypes=["float32", "float32"],
        output_shape=(10,),
        output_dtype="float32",
        reference_impl=lambda x, y: x + y,
    )


@pytest.fixture
def softmax_spec():
    """Provide a softmax KernelSpec."""
    import numpy as np

    def softmax_ref(x):
        shift = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shift)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    return KernelSpec(
        operation="softmax",
        input_shapes=[(32, 128)],
        input_dtypes=["float32"],
        output_shape=(32, 128),
        output_dtype="float32",
        reference_impl=softmax_ref,
    )


@pytest.fixture
def mock_llm_response():
    """Provide a factory for mock LLM responses."""

    def _make_response(
        content: str = "mock response",
        input_tokens: int = 100,
        output_tokens: int = 200,
        model: str = "mock-model",
        latency_ms: float = 500.0,
        stop_reason: str = "end_turn",
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )

    return _make_response


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Provide a mock LLM client."""
    client = MagicMock()
    client.generate.return_value = mock_llm_response(
        content='```python\ndef wrapper(a, b):\n    return a + b\n```'
    )
    client.generate_with_messages.return_value = mock_llm_response()
    client.extract_code.return_value = "def wrapper(a, b):\n    return a + b\n"
    client.usage = MagicMock(
        total_input_tokens=0,
        total_output_tokens=0,
        total_requests=0,
        estimated_cost_usd=0.0,
        average_latency_ms=0.0,
    )
    return client


@pytest.fixture
def backend_config(tmp_cache_dir):
    """Provide an AgentBackendConfig for testing."""
    return AgentBackendConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        ),
        cache=CacheConfig(
            enabled=True,
            cache_dir=tmp_cache_dir,
        ),
        hardware_target="A100-80GB",
    )
