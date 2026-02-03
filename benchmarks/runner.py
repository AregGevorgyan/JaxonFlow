"""Benchmark harness for JaxonFlow kernel generation.

Provides BenchmarkTask, BenchmarkSuite, BenchmarkResult, and BenchmarkRunner
for systematically evaluating kernel generation quality, correctness, and
performance across different operations, shapes, dtypes, and hardware targets.

Supports mock mode (JAXONFLOW_MOCK_GPU=1) for CPU-only environments, and
real GPU profiling when hardware is available.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from jaxonflow.config import AgentBackendConfig
from jaxonflow.dispatch import AgentBackend
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec, ProfileResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """A single benchmark task that evaluates one kernel specification.

    Attributes:
        name: Human-readable task name (e.g. "matmul_1024x1024_fp32").
        spec: The kernel specification to generate and evaluate.
        tags: Optional tags for filtering (e.g. "matmul", "large", "fp16").
    """

    name: str
    spec: KernelSpec
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of running a single benchmark task.

    Attributes:
        task_name: Name of the task.
        compilation_success: Whether the generated code compiled.
        correctness: Whether the kernel output matches the reference.
        speedup_vs_reference: Speedup relative to the reference implementation.
        execution_time_us: Kernel execution time in microseconds.
        iterations_to_success: Number of agent iterations before a correct
            kernel was produced (0 if compilation or correctness failed).
        generation_time_s: Wall-clock time for the full generation loop.
        total_tokens_used: LLM tokens consumed.
        error: Error message if the task failed.
        profile: Full profiling result (when available).
    """

    task_name: str
    compilation_success: bool = False
    correctness: bool = False
    speedup_vs_reference: float = 0.0
    execution_time_us: float = 0.0
    iterations_to_success: int = 0
    generation_time_s: float = 0.0
    total_tokens_used: int = 0
    error: str | None = None
    profile: ProfileResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        d: dict[str, Any] = {
            "task_name": self.task_name,
            "compilation_success": self.compilation_success,
            "correctness": self.correctness,
            "speedup_vs_reference": self.speedup_vs_reference,
            "execution_time_us": self.execution_time_us,
            "iterations_to_success": self.iterations_to_success,
            "generation_time_s": self.generation_time_s,
            "total_tokens_used": self.total_tokens_used,
            "error": self.error,
        }
        if self.profile is not None:
            d["profile"] = {
                "memory_bandwidth_utilization": self.profile.memory_bandwidth_utilization,
                "compute_utilization": self.profile.compute_utilization,
                "achieved_occupancy": self.profile.achieved_occupancy,
                "primary_bottleneck": self.profile.primary_bottleneck,
            }
        else:
            d["profile"] = None
        return d


@dataclass
class BenchmarkResult:
    """Aggregated result from running an entire benchmark suite.

    Attributes:
        suite_name: Name of the suite that was run.
        hardware: Hardware profile used.
        task_results: Per-task results.
        wall_clock_time_s: Total wall-clock time for the full suite.
    """

    suite_name: str
    hardware: str
    task_results: list[TaskResult] = field(default_factory=list)
    wall_clock_time_s: float = 0.0

    # -- aggregate properties --------------------------------------------------

    @property
    def total_tasks(self) -> int:
        return len(self.task_results)

    @property
    def compilation_success_rate(self) -> float:
        """Fraction of tasks where compilation succeeded."""
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.compilation_success) / len(self.task_results)

    @property
    def correctness_rate(self) -> float:
        """Fraction of tasks that produced correct output."""
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.correctness) / len(self.task_results)

    @property
    def geometric_mean_speedup(self) -> float:
        """Geometric mean of speedups across correct tasks."""
        speedups = [r.speedup_vs_reference for r in self.task_results if r.correctness and r.speedup_vs_reference > 0]
        if not speedups:
            return 0.0
        log_sum = sum(math.log(s) for s in speedups)
        return math.exp(log_sum / len(speedups))

    @property
    def average_iterations(self) -> float:
        """Average iterations to success across correct tasks."""
        iters = [r.iterations_to_success for r in self.task_results if r.correctness]
        if not iters:
            return 0.0
        return sum(iters) / len(iters)

    @property
    def total_tokens(self) -> int:
        """Total LLM tokens consumed across all tasks."""
        return sum(r.total_tokens_used for r in self.task_results)

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary."""
        return {
            "suite_name": self.suite_name,
            "hardware": self.hardware,
            "total_tasks": self.total_tasks,
            "compilation_success_rate": round(self.compilation_success_rate, 4),
            "correctness_rate": round(self.correctness_rate, 4),
            "geometric_mean_speedup": round(self.geometric_mean_speedup, 4),
            "average_iterations": round(self.average_iterations, 2),
            "total_tokens": self.total_tokens,
            "wall_clock_time_s": round(self.wall_clock_time_s, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full result to JSON."""
        return json.dumps(
            {
                "summary": self.summary(),
                "tasks": [r.to_dict() for r in self.task_results],
            },
            indent=indent,
        )

    def save(self, path: str | Path) -> None:
        """Save the result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info(f"Benchmark result saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Load a result from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        summary = data["summary"]

        task_results = []
        for td in data.get("tasks", []):
            profile_data = td.pop("profile", None)
            profile = None
            if profile_data:
                profile = ProfileResult(
                    execution_time_us=td.get("execution_time_us", 0.0),
                    speedup_vs_reference=td.get("speedup_vs_reference", 0.0),
                    memory_bandwidth_utilization=profile_data.get("memory_bandwidth_utilization"),
                    compute_utilization=profile_data.get("compute_utilization"),
                    achieved_occupancy=profile_data.get("achieved_occupancy"),
                    primary_bottleneck=profile_data.get("primary_bottleneck"),
                )
            task_results.append(TaskResult(**td, profile=profile))

        return cls(
            suite_name=summary["suite_name"],
            hardware=summary["hardware"],
            task_results=task_results,
            wall_clock_time_s=summary.get("wall_clock_time_s", 0.0),
        )


# ---------------------------------------------------------------------------
# Built-in benchmark suites
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """A collection of benchmark tasks.

    Provides factory methods for common benchmark suites and supports
    filtering by tags.
    """

    def __init__(self, name: str, tasks: list[BenchmarkTask] | None = None) -> None:
        self.name = name
        self.tasks: list[BenchmarkTask] = tasks or []

    def add_task(self, task: BenchmarkTask) -> None:
        self.tasks.append(task)

    def filter_by_tags(self, *tags: str) -> BenchmarkSuite:
        """Return a new suite containing only tasks matching *all* given tags."""
        filtered = [t for t in self.tasks if all(tag in t.tags for tag in tags)]
        return BenchmarkSuite(name=f"{self.name}[{','.join(tags)}]", tasks=filtered)

    def __len__(self) -> int:
        return len(self.tasks)

    # ---- factory methods for built-in suites --------------------------------

    @classmethod
    def matmul_sweep(cls, hardware: HardwareContext | None = None) -> BenchmarkSuite:
        """Matrix multiplication across typical shapes and dtypes."""
        suite = cls(name="matmul_sweep")
        shapes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            # Non-square
            (1024, 512, 2048),
            (2048, 768, 1024),
            # Small (latency-bound)
            (32, 32, 32),
            (64, 64, 64),
        ]
        dtypes = ["float32", "float16"]

        for m, k, n in shapes:
            for dtype in dtypes:
                name = f"matmul_{m}x{k}x{n}_{dtype}"
                tags = ["matmul", dtype]
                if m * k * n > 1024**3:
                    tags.append("large")
                elif m * k * n < 256**3:
                    tags.append("small")
                else:
                    tags.append("medium")

                suite.add_task(BenchmarkTask(
                    name=name,
                    spec=KernelSpec(
                        operation="matmul",
                        input_shapes=[(m, k), (k, n)],
                        input_dtypes=[dtype, dtype],
                        output_shape=(m, n),
                        output_dtype=dtype,
                        hardware=hardware,
                        reference_impl=lambda a, b: a @ b,
                    ),
                    tags=tags,
                ))
        return suite

    @classmethod
    def elementwise_sweep(cls, hardware: HardwareContext | None = None) -> BenchmarkSuite:
        """Elementwise operations (add, mul, relu, gelu, softmax)."""
        suite = cls(name="elementwise_sweep")
        sizes = [1024, 4096, 65536, 1048576]

        for n in sizes:
            # Add
            suite.add_task(BenchmarkTask(
                name=f"add_{n}",
                spec=KernelSpec(
                    operation="add",
                    input_shapes=[(n,), (n,)],
                    input_dtypes=["float32", "float32"],
                    output_shape=(n,),
                    output_dtype="float32",
                    hardware=hardware,
                    reference_impl=lambda a, b: a + b,
                ),
                tags=["elementwise", "add", "float32"],
            ))

            # ReLU
            suite.add_task(BenchmarkTask(
                name=f"relu_{n}",
                spec=KernelSpec(
                    operation="relu",
                    input_shapes=[(n,)],
                    input_dtypes=["float32"],
                    output_shape=(n,),
                    output_dtype="float32",
                    hardware=hardware,
                    reference_impl=lambda x: np.maximum(x, 0),
                ),
                tags=["elementwise", "relu", "float32"],
            ))

        # Softmax (2D)
        for rows, cols in [(32, 128), (64, 512), (128, 1024)]:
            def _softmax_ref(x: np.ndarray) -> np.ndarray:
                shift = x - np.max(x, axis=-1, keepdims=True)
                exp_x = np.exp(shift)
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

            suite.add_task(BenchmarkTask(
                name=f"softmax_{rows}x{cols}",
                spec=KernelSpec(
                    operation="softmax",
                    input_shapes=[(rows, cols)],
                    input_dtypes=["float32"],
                    output_shape=(rows, cols),
                    output_dtype="float32",
                    hardware=hardware,
                    reference_impl=_softmax_ref,
                ),
                tags=["elementwise", "softmax", "float32"],
            ))

        return suite

    @classmethod
    def reduction_sweep(cls, hardware: HardwareContext | None = None) -> BenchmarkSuite:
        """Reduction operations (sum, max)."""
        suite = cls(name="reduction_sweep")
        shapes_2d = [(128, 1024), (512, 2048), (1024, 4096)]

        for rows, cols in shapes_2d:
            # Sum reduction along last axis
            suite.add_task(BenchmarkTask(
                name=f"reduce_sum_{rows}x{cols}",
                spec=KernelSpec(
                    operation="reduce_sum",
                    input_shapes=[(rows, cols)],
                    input_dtypes=["float32"],
                    output_shape=(rows,),
                    output_dtype="float32",
                    parameters={"axis": -1},
                    hardware=hardware,
                    reference_impl=lambda x: np.sum(x, axis=-1),
                ),
                tags=["reduction", "reduce_sum", "float32"],
            ))

            # Max reduction
            suite.add_task(BenchmarkTask(
                name=f"reduce_max_{rows}x{cols}",
                spec=KernelSpec(
                    operation="reduce_max",
                    input_shapes=[(rows, cols)],
                    input_dtypes=["float32"],
                    output_shape=(rows,),
                    output_dtype="float32",
                    parameters={"axis": -1},
                    hardware=hardware,
                    reference_impl=lambda x: np.max(x, axis=-1),
                ),
                tags=["reduction", "reduce_max", "float32"],
            ))

        return suite

    @classmethod
    def full(cls, hardware: HardwareContext | None = None) -> BenchmarkSuite:
        """Comprehensive suite combining matmul, elementwise, and reductions."""
        suite = cls(name="full")
        for sub in (cls.matmul_sweep(hardware), cls.elementwise_sweep(hardware), cls.reduction_sweep(hardware)):
            suite.tasks.extend(sub.tasks)
        return suite


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs a BenchmarkSuite against a configured AgentBackend.

    Handles:
    - Running each task through the agent backend
    - Collecting correctness, performance, and generation metrics
    - Aggregating results into a BenchmarkResult
    - Optional profiling via KernelProfiler

    Args:
        config: Agent backend configuration.
        profile: Whether to run GPU profiling on correct kernels.
        output_dir: Directory for saving results (default: benchmarks/results/).
    """

    def __init__(
        self,
        config: AgentBackendConfig | None = None,
        profile: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or AgentBackendConfig()
        self.profile = profile
        self.output_dir = Path(output_dir) if output_dir else Path("benchmarks/results")
        self._backend: AgentBackend | None = None

    @property
    def backend(self) -> AgentBackend:
        """Lazily initialize the agent backend."""
        if self._backend is None:
            self._backend = AgentBackend(self.config)
        return self._backend

    def run_suite(self, suite: BenchmarkSuite) -> BenchmarkResult:
        """Run all tasks in a benchmark suite.

        Args:
            suite: The benchmark suite to run.

        Returns:
            BenchmarkResult with per-task and aggregate metrics.
        """
        hardware_name = self.backend.hardware_context.name
        logger.info(
            f"Starting benchmark suite '{suite.name}' "
            f"({len(suite)} tasks) on {hardware_name}"
        )

        result = BenchmarkResult(
            suite_name=suite.name,
            hardware=hardware_name,
        )
        suite_start = time.monotonic()

        for i, task in enumerate(suite.tasks):
            logger.info(f"[{i + 1}/{len(suite)}] Running task: {task.name}")
            task_result = self.run_task(task)
            result.task_results.append(task_result)

            status = "PASS" if task_result.correctness else "FAIL"
            logger.info(
                f"  {status} | speedup={task_result.speedup_vs_reference:.2f}x | "
                f"iters={task_result.iterations_to_success} | "
                f"gen_time={task_result.generation_time_s:.1f}s"
            )

        result.wall_clock_time_s = time.monotonic() - suite_start

        logger.info(
            f"Suite '{suite.name}' complete: "
            f"correctness={result.correctness_rate:.1%}, "
            f"geomean_speedup={result.geometric_mean_speedup:.2f}x, "
            f"wall_clock={result.wall_clock_time_s:.1f}s"
        )
        return result

    def run_task(self, task: BenchmarkTask) -> TaskResult:
        """Run a single benchmark task.

        Args:
            task: The benchmark task to run.

        Returns:
            TaskResult with metrics for this task.
        """
        task_result = TaskResult(task_name=task.name)
        start = time.monotonic()

        try:
            kernel = self.backend.get_or_generate_kernel(task.spec)
        except Exception as e:
            task_result.error = str(e)
            task_result.generation_time_s = time.monotonic() - start
            return task_result

        task_result.compilation_success = True
        task_result.iterations_to_success = kernel.iterations_to_generate
        task_result.total_tokens_used = kernel.total_tokens_used
        task_result.generation_time_s = kernel.generation_time_s or (time.monotonic() - start)

        # Verify correctness
        if task.spec.reference_impl is not None:
            try:
                correct = self._verify(kernel, task.spec)
                task_result.correctness = correct
            except Exception as e:
                task_result.error = f"Verification error: {e}"
                return task_result
        else:
            # No reference impl -> assume correct (cannot verify)
            task_result.correctness = True

        # Profile if requested and kernel is correct
        if self.profile and task_result.correctness:
            try:
                profile_result = self._profile(kernel, task.spec)
                task_result.speedup_vs_reference = profile_result.speedup_vs_reference
                task_result.execution_time_us = profile_result.execution_time_us
                task_result.profile = profile_result
            except Exception as e:
                logger.warning(f"Profiling failed for {task.name}: {e}")

        return task_result

    def _verify(self, kernel: Any, spec: KernelSpec) -> bool:
        """Verify kernel correctness."""
        from jaxonflow.verification import KernelVerifier

        verifier = KernelVerifier()
        result = verifier.verify(kernel, spec)
        return result.correct

    def _profile(self, kernel: Any, spec: KernelSpec) -> ProfileResult:
        """Profile kernel performance."""
        from jaxonflow.profiler import KernelProfiler

        hardware = spec.hardware or self.backend.hardware_context
        profiler = KernelProfiler(hardware)
        return profiler.profile(kernel, spec)

    def run_and_save(
        self,
        suite: BenchmarkSuite,
        filename: str | None = None,
    ) -> BenchmarkResult:
        """Run a suite and save the result to disk.

        Args:
            suite: The benchmark suite.
            filename: Output filename (default: {suite_name}_{timestamp}.json).

        Returns:
            BenchmarkResult.
        """
        result = self.run_suite(suite)
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{suite.name}_{timestamp}.json"
        result.save(self.output_dir / filename)
        return result
