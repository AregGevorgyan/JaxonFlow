"""Performance tests for JaxonFlow — speedup measurement vs baselines.

Measures JaxonFlow-generated kernels against:
  - NumPy reference implementations (always available)
  - JAX/XLA compiled kernels (when JAX is available with GPU)
  - PyTorch/TorchInductor kernels (when PyTorch CUDA is available)

Designed to run on GPU machines.  On CPU-only hosts the XLA and Inductor
baselines are skipped and only the NumPy reference comparison is performed
(which still exercises the full agent pipeline).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from jaxonflow.config import AgentBackendConfig
from jaxonflow.dispatch import AgentBackend
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_jax_gpu() -> bool:
    try:
        import jax
        return any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in jax.devices())
    except Exception:
        return False


def _has_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BaselineTiming:
    """Execution time of a baseline implementation.

    Attributes:
        name: Baseline name (e.g. "numpy", "xla", "inductor").
        execution_time_us: Median execution time in microseconds.
        warmup_runs: Number of warmup iterations used.
        timed_runs: Number of timed iterations used.
    """

    name: str
    execution_time_us: float
    warmup_runs: int = 0
    timed_runs: int = 0


@dataclass
class PerfTestResult:
    """Result of a single performance comparison.

    Attributes:
        task_name: Name of the benchmark task.
        agent_time_us: Agent-generated kernel execution time.
        baselines: Timing data for each baseline.
        speedups: Speedup of the agent kernel vs each baseline.
        generation_time_s: Time to generate the kernel.
    """

    task_name: str
    agent_time_us: float = 0.0
    baselines: list[BaselineTiming] = field(default_factory=list)
    speedups: dict[str, float] = field(default_factory=dict)
    generation_time_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "agent_time_us": round(self.agent_time_us, 2),
            "baselines": {b.name: round(b.execution_time_us, 2) for b in self.baselines},
            "speedups": {k: round(v, 4) for k, v in self.speedups.items()},
            "generation_time_s": round(self.generation_time_s, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Performance tester
# ---------------------------------------------------------------------------

class PerfTester:
    """Measures speedup of JaxonFlow kernels against framework baselines.

    Usage::

        tester = PerfTester()
        result = tester.compare_matmul(m=1024, k=1024, n=1024, dtype="float32")
        print(result.speedups)

    Args:
        config: AgentBackendConfig (uses defaults if not provided).
        warmup_runs: Number of warmup iterations before timing.
        timed_runs: Number of timed iterations for median calculation.
    """

    def __init__(
        self,
        config: AgentBackendConfig | None = None,
        warmup_runs: int = 5,
        timed_runs: int = 20,
    ) -> None:
        self.config = config or AgentBackendConfig()
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self._backend: AgentBackend | None = None

    @property
    def backend(self) -> AgentBackend:
        if self._backend is None:
            self._backend = AgentBackend(self.config)
        return self._backend

    # ----- core timing utility ------------------------------------------------

    def _time_fn(
        self,
        fn: Callable[..., Any],
        args: list[Any],
        warmup: int | None = None,
        runs: int | None = None,
    ) -> float:
        """Time a callable and return the median execution time in microseconds.

        Args:
            fn: The function to time.
            args: Positional arguments to pass to *fn*.
            warmup: Override number of warmup iterations.
            runs: Override number of timed iterations.

        Returns:
            Median execution time in microseconds.
        """
        warmup = warmup if warmup is not None else self.warmup_runs
        runs = runs if runs is not None else self.timed_runs

        # Warmup
        for _ in range(warmup):
            fn(*args)

        # Timed runs
        times: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(*args)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1e6)  # to microseconds

        return float(np.median(times))

    # ----- baseline runners ---------------------------------------------------

    def _numpy_baseline(
        self, ref_impl: Callable[..., Any], inputs: list[np.ndarray]
    ) -> BaselineTiming:
        """Time the NumPy reference implementation."""
        t = self._time_fn(ref_impl, inputs)
        return BaselineTiming(
            name="numpy",
            execution_time_us=t,
            warmup_runs=self.warmup_runs,
            timed_runs=self.timed_runs,
        )

    def _xla_baseline(
        self, spec: KernelSpec, inputs: list[np.ndarray]
    ) -> BaselineTiming | None:
        """Time the JAX/XLA compiled baseline (requires JAX + GPU)."""
        if not _has_jax_gpu():
            return None

        import jax
        import jax.numpy as jnp

        jax_inputs = [jnp.array(x) for x in inputs]

        if spec.operation == "matmul":
            @jax.jit
            def _jax_fn(a: Any, b: Any) -> Any:
                return a @ b
        elif spec.operation == "softmax":
            @jax.jit
            def _jax_fn(x: Any) -> Any:
                return jax.nn.softmax(x, axis=-1)
        elif spec.operation == "relu":
            @jax.jit
            def _jax_fn(x: Any) -> Any:
                return jax.nn.relu(x)
        elif spec.operation == "add":
            @jax.jit
            def _jax_fn(a: Any, b: Any) -> Any:
                return a + b
        elif spec.operation in ("reduce_sum",):
            axis = spec.parameters.get("axis", -1)
            @jax.jit
            def _jax_fn(x: Any) -> Any:
                return jnp.sum(x, axis=axis)
        elif spec.operation in ("reduce_max",):
            axis = spec.parameters.get("axis", -1)
            @jax.jit
            def _jax_fn(x: Any) -> Any:
                return jnp.max(x, axis=axis)
        else:
            return None

        # Warm-up (triggers compilation)
        _jax_fn(*jax_inputs)

        t = self._time_fn(_jax_fn, jax_inputs)
        return BaselineTiming(
            name="xla",
            execution_time_us=t,
            warmup_runs=self.warmup_runs,
            timed_runs=self.timed_runs,
        )

    def _inductor_baseline(
        self, spec: KernelSpec, inputs: list[np.ndarray]
    ) -> BaselineTiming | None:
        """Time the PyTorch/TorchInductor compiled baseline (requires CUDA)."""
        if not _has_torch_cuda():
            return None

        import torch

        torch_inputs = [torch.from_numpy(x).cuda() for x in inputs]

        if spec.operation == "matmul":
            @torch.compile
            def _torch_fn(a: Any, b: Any) -> Any:
                return a @ b
        elif spec.operation == "softmax":
            @torch.compile
            def _torch_fn(x: Any) -> Any:
                return torch.nn.functional.softmax(x, dim=-1)
        elif spec.operation == "relu":
            @torch.compile
            def _torch_fn(x: Any) -> Any:
                return torch.nn.functional.relu(x)
        elif spec.operation == "add":
            @torch.compile
            def _torch_fn(a: Any, b: Any) -> Any:
                return a + b
        elif spec.operation in ("reduce_sum",):
            axis = spec.parameters.get("axis", -1)
            @torch.compile
            def _torch_fn(x: Any) -> Any:
                return torch.sum(x, dim=axis)
        elif spec.operation in ("reduce_max",):
            axis = spec.parameters.get("axis", -1)
            @torch.compile
            def _torch_fn(x: Any) -> Any:
                return torch.max(x, dim=axis).values
        else:
            return None

        # Warm-up (triggers compilation)
        _torch_fn(*torch_inputs)

        t = self._time_fn(_torch_fn, torch_inputs)
        return BaselineTiming(
            name="inductor",
            execution_time_us=t,
            warmup_runs=self.warmup_runs,
            timed_runs=self.timed_runs,
        )

    # ----- public comparison methods ------------------------------------------

    def compare(self, spec: KernelSpec) -> PerfTestResult:
        """Compare agent-generated kernel against all available baselines.

        Args:
            spec: Kernel specification.

        Returns:
            PerfTestResult with timing and speedup data.
        """
        result = PerfTestResult(task_name=f"{spec.operation}_{_shape_tag(spec)}")

        # Generate inputs
        inputs = _generate_inputs(spec)

        # 1. Generate kernel via agent backend
        gen_start = time.monotonic()
        try:
            kernel = self.backend.get_or_generate_kernel(spec)
        except Exception as e:
            result.error = f"Generation failed: {e}"
            result.generation_time_s = time.monotonic() - gen_start
            return result
        result.generation_time_s = time.monotonic() - gen_start

        # 2. Time agent kernel
        try:
            result.agent_time_us = self._time_fn(kernel, inputs)
        except Exception as e:
            result.error = f"Agent kernel execution failed: {e}"
            return result

        # 3. Baselines
        if spec.reference_impl is not None:
            numpy_bl = self._numpy_baseline(spec.reference_impl, inputs)
            result.baselines.append(numpy_bl)
            if numpy_bl.execution_time_us > 0:
                result.speedups["numpy"] = numpy_bl.execution_time_us / result.agent_time_us

        xla_bl = self._xla_baseline(spec, inputs)
        if xla_bl is not None:
            result.baselines.append(xla_bl)
            if xla_bl.execution_time_us > 0:
                result.speedups["xla"] = xla_bl.execution_time_us / result.agent_time_us

        inductor_bl = self._inductor_baseline(spec, inputs)
        if inductor_bl is not None:
            result.baselines.append(inductor_bl)
            if inductor_bl.execution_time_us > 0:
                result.speedups["inductor"] = inductor_bl.execution_time_us / result.agent_time_us

        return result

    def compare_matmul(
        self,
        m: int = 1024,
        k: int = 1024,
        n: int = 1024,
        dtype: str = "float32",
    ) -> PerfTestResult:
        """Convenience method for matrix multiplication comparison."""
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(m, k), (k, n)],
            input_dtypes=[dtype, dtype],
            output_shape=(m, n),
            output_dtype=dtype,
            reference_impl=lambda a, b: a @ b,
        )
        return self.compare(spec)

    def compare_softmax(
        self,
        rows: int = 128,
        cols: int = 1024,
        dtype: str = "float32",
    ) -> PerfTestResult:
        """Convenience method for softmax comparison."""

        def _softmax_ref(x: np.ndarray) -> np.ndarray:
            shift = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(shift)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        spec = KernelSpec(
            operation="softmax",
            input_shapes=[(rows, cols)],
            input_dtypes=[dtype],
            output_shape=(rows, cols),
            output_dtype=dtype,
            reference_impl=_softmax_ref,
        )
        return self.compare(spec)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _generate_inputs(spec: KernelSpec) -> list[np.ndarray]:
    """Generate random inputs for a spec."""
    inputs = []
    for shape, dtype_str in zip(spec.input_shapes, spec.input_dtypes):
        dtype = _parse_dtype(dtype_str)
        inputs.append(np.random.randn(*shape).astype(dtype))
    return inputs


def _parse_dtype(dtype_str: str) -> np.dtype[Any]:
    mapping: dict[str, Any] = {
        "float32": np.float32,
        "float16": np.float16,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    return mapping.get(dtype_str, np.float32)


def _shape_tag(spec: KernelSpec) -> str:
    """Create a short shape tag for display."""
    parts = []
    for s in spec.input_shapes:
        parts.append("x".join(str(d) for d in s))
    return "_".join(parts)
