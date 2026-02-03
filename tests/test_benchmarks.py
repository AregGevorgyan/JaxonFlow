"""Tests for the benchmarks package (Phase 6).

All tests run without a GPU by relying on mock mode
(JAXONFLOW_MOCK_GPU=1) and reference implementations.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxonflow.config import AgentBackendConfig, CacheConfig, LLMConfig, LLMProvider
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec, ProfileResult

# Ensure mock mode is active for tests
os.environ.setdefault("JAXONFLOW_MOCK_GPU", "1")

from benchmarks.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    TaskResult,
)
from benchmarks.perf_tests import (
    BaselineTiming,
    PerfTestResult,
    PerfTester,
    _generate_inputs,
    _shape_tag,
)
from benchmarks.regression import (
    ComparisonReport,
    RegressionTracker,
    TaskComparison,
    _get_git_commit,
    _get_git_branch,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def a100_hw():
    return HardwareContext.from_name("A100-80GB")


@pytest.fixture
def matmul_spec(a100_hw):
    return KernelSpec(
        operation="matmul",
        input_shapes=[(64, 32), (32, 128)],
        input_dtypes=["float32", "float32"],
        output_shape=(64, 128),
        output_dtype="float32",
        hardware=a100_hw,
        reference_impl=lambda a, b: a @ b,
    )


@pytest.fixture
def add_spec(a100_hw):
    return KernelSpec(
        operation="add",
        input_shapes=[(256,), (256,)],
        input_dtypes=["float32", "float32"],
        output_shape=(256,),
        output_dtype="float32",
        hardware=a100_hw,
        reference_impl=lambda a, b: a + b,
    )


@pytest.fixture
def backend_config(tmp_path):
    return AgentBackendConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        ),
        cache=CacheConfig(enabled=True, cache_dir=tmp_path / "bench_cache"),
        hardware_target="A100-80GB",
    )


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "benchmark_results"


# ============================================================================
# TaskResult tests
# ============================================================================

class TestTaskResult:
    def test_to_dict_basic(self):
        tr = TaskResult(
            task_name="test_task",
            compilation_success=True,
            correctness=True,
            speedup_vs_reference=1.5,
            execution_time_us=100.0,
            iterations_to_success=3,
        )
        d = tr.to_dict()
        assert d["task_name"] == "test_task"
        assert d["compilation_success"] is True
        assert d["correctness"] is True
        assert d["speedup_vs_reference"] == 1.5
        assert d["profile"] is None  # no profile data

    def test_to_dict_with_profile(self):
        profile = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=1.5,
            memory_bandwidth_utilization=0.85,
            compute_utilization=0.6,
            achieved_occupancy=0.75,
            primary_bottleneck="memory",
        )
        tr = TaskResult(task_name="profiled_task", profile=profile)
        d = tr.to_dict()
        assert d["profile"] is not None
        assert d["profile"]["memory_bandwidth_utilization"] == 0.85
        assert d["profile"]["primary_bottleneck"] == "memory"

    def test_to_dict_with_error(self):
        tr = TaskResult(task_name="error_task", error="Something went wrong")
        d = tr.to_dict()
        assert d["error"] == "Something went wrong"
        assert d["compilation_success"] is False


# ============================================================================
# BenchmarkResult tests
# ============================================================================

class TestBenchmarkResult:
    def test_empty_result(self):
        result = BenchmarkResult(suite_name="empty", hardware="A100")
        assert result.total_tasks == 0
        assert result.compilation_success_rate == 0.0
        assert result.correctness_rate == 0.0
        assert result.geometric_mean_speedup == 0.0
        assert result.average_iterations == 0.0
        assert result.total_tokens == 0

    def test_aggregate_metrics(self):
        tasks = [
            TaskResult(task_name="a", compilation_success=True, correctness=True, speedup_vs_reference=2.0, iterations_to_success=3, total_tokens_used=500),
            TaskResult(task_name="b", compilation_success=True, correctness=True, speedup_vs_reference=0.5, iterations_to_success=7, total_tokens_used=800),
            TaskResult(task_name="c", compilation_success=True, correctness=False, speedup_vs_reference=0.0, total_tokens_used=300),
            TaskResult(task_name="d", compilation_success=False, correctness=False, total_tokens_used=100),
        ]
        result = BenchmarkResult(suite_name="test", hardware="A100", task_results=tasks)

        assert result.total_tasks == 4
        assert result.compilation_success_rate == 0.75
        assert result.correctness_rate == 0.5
        assert result.total_tokens == 1700
        assert result.average_iterations == 5.0  # (3 + 7) / 2

        # Geometric mean of 2.0 and 0.5 = sqrt(1.0) = 1.0
        assert abs(result.geometric_mean_speedup - 1.0) < 1e-6

    def test_summary(self):
        result = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="a", correctness=True, speedup_vs_reference=1.5),
            ],
            wall_clock_time_s=10.5,
        )
        s = result.summary()
        assert s["suite_name"] == "test"
        assert s["hardware"] == "A100"
        assert s["total_tasks"] == 1
        assert s["wall_clock_time_s"] == 10.5

    def test_json_serialization(self):
        result = BenchmarkResult(
            suite_name="ser_test",
            hardware="H100",
            task_results=[
                TaskResult(task_name="t1", compilation_success=True, correctness=True, speedup_vs_reference=1.2),
            ],
            wall_clock_time_s=5.0,
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "summary" in data
        assert "tasks" in data
        assert data["summary"]["suite_name"] == "ser_test"
        assert len(data["tasks"]) == 1

    def test_save_and_load(self, tmp_path):
        result = BenchmarkResult(
            suite_name="io_test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", compilation_success=True, correctness=True, speedup_vs_reference=1.3, execution_time_us=50.0),
                TaskResult(task_name="t2", compilation_success=False, error="compile error"),
            ],
            wall_clock_time_s=20.0,
        )
        path = tmp_path / "result.json"
        result.save(path)

        loaded = BenchmarkResult.load(path)
        assert loaded.suite_name == "io_test"
        assert loaded.hardware == "A100"
        assert loaded.total_tasks == 2
        assert loaded.task_results[0].task_name == "t1"
        assert loaded.task_results[0].correctness is True
        assert loaded.task_results[1].error == "compile error"

    def test_geometric_mean_with_single_task(self):
        result = BenchmarkResult(
            suite_name="single",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=3.0),
            ],
        )
        assert abs(result.geometric_mean_speedup - 3.0) < 1e-6


# ============================================================================
# BenchmarkSuite tests
# ============================================================================

class TestBenchmarkSuite:
    def test_empty_suite(self):
        suite = BenchmarkSuite("empty")
        assert len(suite) == 0

    def test_add_task(self, matmul_spec):
        suite = BenchmarkSuite("test")
        task = BenchmarkTask(name="mm_test", spec=matmul_spec, tags=["matmul"])
        suite.add_task(task)
        assert len(suite) == 1
        assert suite.tasks[0].name == "mm_test"

    def test_filter_by_tags(self, matmul_spec, add_spec):
        suite = BenchmarkSuite("test")
        suite.add_task(BenchmarkTask(name="mm", spec=matmul_spec, tags=["matmul", "float32"]))
        suite.add_task(BenchmarkTask(name="add", spec=add_spec, tags=["elementwise", "float32"]))

        matmul_suite = suite.filter_by_tags("matmul")
        assert len(matmul_suite) == 1
        assert matmul_suite.tasks[0].name == "mm"

        float32_suite = suite.filter_by_tags("float32")
        assert len(float32_suite) == 2

    def test_matmul_sweep_factory(self, a100_hw):
        suite = BenchmarkSuite.matmul_sweep(a100_hw)
        assert suite.name == "matmul_sweep"
        # 10 shapes * 2 dtypes = 20 tasks
        assert len(suite) == 20
        # Every task should have the "matmul" tag
        for task in suite.tasks:
            assert "matmul" in task.tags

    def test_elementwise_sweep_factory(self, a100_hw):
        suite = BenchmarkSuite.elementwise_sweep(a100_hw)
        assert suite.name == "elementwise_sweep"
        assert len(suite) > 0
        for task in suite.tasks:
            assert "elementwise" in task.tags

    def test_reduction_sweep_factory(self, a100_hw):
        suite = BenchmarkSuite.reduction_sweep(a100_hw)
        assert suite.name == "reduction_sweep"
        assert len(suite) > 0
        for task in suite.tasks:
            assert "reduction" in task.tags

    def test_full_factory(self, a100_hw):
        suite = BenchmarkSuite.full(a100_hw)
        assert suite.name == "full"
        matmul_count = len(BenchmarkSuite.matmul_sweep(a100_hw))
        elem_count = len(BenchmarkSuite.elementwise_sweep(a100_hw))
        red_count = len(BenchmarkSuite.reduction_sweep(a100_hw))
        assert len(suite) == matmul_count + elem_count + red_count


# ============================================================================
# BenchmarkRunner tests
# ============================================================================

class TestBenchmarkRunner:
    def test_run_task_with_mock_backend(self, matmul_spec, backend_config, tmp_path):
        """Run a single task through the mock pipeline."""
        runner = BenchmarkRunner(config=backend_config, output_dir=tmp_path / "results")

        task = BenchmarkTask(name="mm_test", spec=matmul_spec)

        # Mock the backend to avoid LLM calls
        with patch.object(runner, '_backend') as mock_backend:
            # Return a fake compiled kernel that uses the reference impl
            from jaxonflow.spec import CompiledKernel
            fake_kernel = CompiledKernel(
                spec=matmul_spec,
                code="# mock code",
                callable=matmul_spec.reference_impl,
                iterations_to_generate=2,
                total_tokens_used=500,
                generation_time_s=1.5,
            )
            mock_backend.get_or_generate_kernel.return_value = fake_kernel
            mock_backend.hardware_context = HardwareContext.from_name("A100-80GB")

            result = runner.run_task(task)

        assert result.task_name == "mm_test"
        assert result.compilation_success is True
        assert result.correctness is True
        assert result.iterations_to_success == 2
        assert result.total_tokens_used == 500

    def test_run_task_handles_generation_error(self, matmul_spec, backend_config, tmp_path):
        runner = BenchmarkRunner(config=backend_config, output_dir=tmp_path / "results")

        task = BenchmarkTask(name="mm_fail", spec=matmul_spec)

        with patch.object(runner, '_backend') as mock_backend:
            mock_backend.get_or_generate_kernel.side_effect = RuntimeError("LLM API down")
            mock_backend.hardware_context = HardwareContext.from_name("A100-80GB")

            result = runner.run_task(task)

        assert result.task_name == "mm_fail"
        assert result.compilation_success is False
        assert result.correctness is False
        assert "LLM API down" in result.error

    def test_run_suite(self, matmul_spec, add_spec, backend_config, tmp_path):
        runner = BenchmarkRunner(config=backend_config, output_dir=tmp_path / "results")

        suite = BenchmarkSuite("mini")
        suite.add_task(BenchmarkTask(name="mm", spec=matmul_spec))
        suite.add_task(BenchmarkTask(name="add", spec=add_spec))

        with patch.object(runner, '_backend') as mock_backend:
            from jaxonflow.spec import CompiledKernel
            mock_backend.hardware_context = HardwareContext.from_name("A100-80GB")

            def _mock_generate(spec):
                return CompiledKernel(
                    spec=spec,
                    code="# mock",
                    callable=spec.reference_impl,
                    iterations_to_generate=1,
                    generation_time_s=0.5,
                )

            mock_backend.get_or_generate_kernel.side_effect = _mock_generate

            result = runner.run_suite(suite)

        assert result.suite_name == "mini"
        assert result.total_tasks == 2
        assert result.compilation_success_rate == 1.0
        assert result.correctness_rate == 1.0
        assert result.wall_clock_time_s > 0

    def test_run_and_save(self, matmul_spec, backend_config, tmp_path):
        runner = BenchmarkRunner(config=backend_config, output_dir=tmp_path / "results")
        suite = BenchmarkSuite("save_test")
        suite.add_task(BenchmarkTask(name="mm", spec=matmul_spec))

        with patch.object(runner, '_backend') as mock_backend:
            from jaxonflow.spec import CompiledKernel
            mock_backend.hardware_context = HardwareContext.from_name("A100-80GB")
            mock_backend.get_or_generate_kernel.return_value = CompiledKernel(
                spec=matmul_spec,
                code="# mock",
                callable=matmul_spec.reference_impl,
                iterations_to_generate=1,
            )

            result = runner.run_and_save(suite, filename="test_result.json")

        saved_path = tmp_path / "results" / "test_result.json"
        assert saved_path.exists()
        loaded = BenchmarkResult.load(saved_path)
        assert loaded.suite_name == "save_test"


# ============================================================================
# PerfTester tests
# ============================================================================

class TestPerfTester:
    def test_generate_inputs(self):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 3), (3, 5)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 5),
            output_dtype="float32",
        )
        inputs = _generate_inputs(spec)
        assert len(inputs) == 2
        assert inputs[0].shape == (4, 3)
        assert inputs[1].shape == (3, 5)
        assert inputs[0].dtype == np.float32

    def test_generate_inputs_float16(self):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(8, 8), (8, 8)],
            input_dtypes=["float16", "float16"],
            output_shape=(8, 8),
            output_dtype="float16",
        )
        inputs = _generate_inputs(spec)
        assert inputs[0].dtype == np.float16

    def test_shape_tag(self):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(1024, 512), (512, 2048)],
            input_dtypes=["float32", "float32"],
            output_shape=(1024, 2048),
            output_dtype="float32",
        )
        tag = _shape_tag(spec)
        assert tag == "1024x512_512x2048"

    def test_baseline_timing_dataclass(self):
        bt = BaselineTiming(name="numpy", execution_time_us=150.0, warmup_runs=5, timed_runs=20)
        assert bt.name == "numpy"
        assert bt.execution_time_us == 150.0

    def test_perf_test_result_to_dict(self):
        r = PerfTestResult(
            task_name="matmul_test",
            agent_time_us=100.0,
            baselines=[BaselineTiming(name="numpy", execution_time_us=200.0)],
            speedups={"numpy": 2.0},
            generation_time_s=5.0,
        )
        d = r.to_dict()
        assert d["task_name"] == "matmul_test"
        assert d["agent_time_us"] == 100.0
        assert d["baselines"]["numpy"] == 200.0
        assert d["speedups"]["numpy"] == 2.0

    def test_time_fn(self, backend_config):
        tester = PerfTester(config=backend_config, warmup_runs=1, timed_runs=3)

        def dummy_fn(x):
            return x + 1

        t = tester._time_fn(dummy_fn, [np.array(1.0)])
        assert t > 0  # should be some positive number of microseconds

    def test_numpy_baseline(self, backend_config):
        tester = PerfTester(config=backend_config, warmup_runs=1, timed_runs=3)
        ref = lambda a, b: a @ b
        inputs = [np.random.randn(16, 16).astype(np.float32), np.random.randn(16, 16).astype(np.float32)]
        bt = tester._numpy_baseline(ref, inputs)
        assert bt.name == "numpy"
        assert bt.execution_time_us > 0


# ============================================================================
# RegressionTracker tests
# ============================================================================

class TestRegressionTracker:
    def test_record(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir)

        result = BenchmarkResult(
            suite_name="test_suite",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.5),
            ],
            wall_clock_time_s=10.0,
        )

        path = tracker.record(result)
        assert path.exists()

        # Index should have one entry
        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0]["suite_name"] == "test_suite"

    def test_compare_latest_not_enough_runs(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir)

        result = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[TaskResult(task_name="t1")],
        )
        tracker.record(result)

        # Only one run - should return None
        comparison = tracker.compare_latest()
        assert comparison is None

    def test_compare_latest_two_runs(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir)

        # Run 1
        r1 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.0),
                TaskResult(task_name="t2", correctness=True, speedup_vs_reference=2.0),
            ],
            wall_clock_time_s=10.0,
        )
        tracker.record(r1)

        # Run 2 (with improvement on t1 and regression on t2)
        r2 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.5),
                TaskResult(task_name="t2", correctness=True, speedup_vs_reference=1.5),
            ],
            wall_clock_time_s=8.0,
        )
        tracker.record(r2)

        comparison = tracker.compare_latest()
        assert comparison is not None
        assert comparison.suite_name == "test"
        assert len(comparison.task_comparisons) == 2

    def test_check_regressions(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir, regression_threshold=0.10)

        r1 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=2.0),
            ],
        )
        tracker.record(r1)

        # Big regression: 2.0 -> 1.0 = -50%
        r2 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.0),
            ],
        )
        tracker.record(r2)

        regressions = tracker.check_regressions()
        assert len(regressions) == 1
        assert regressions[0].task_name == "t1"
        assert regressions[0].regressed is True

    def test_no_regressions_when_improving(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir, regression_threshold=0.05)

        r1 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.0),
            ],
        )
        tracker.record(r1)

        r2 = BenchmarkResult(
            suite_name="test",
            hardware="A100",
            task_results=[
                TaskResult(task_name="t1", correctness=True, speedup_vs_reference=1.5),
            ],
        )
        tracker.record(r2)

        regressions = tracker.check_regressions()
        assert len(regressions) == 0

    def test_list_runs_with_filter(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir)

        r1 = BenchmarkResult(suite_name="suite_a", hardware="A100")
        r2 = BenchmarkResult(suite_name="suite_b", hardware="A100")
        tracker.record(r1)
        tracker.record(r2)

        all_runs = tracker.list_runs()
        assert len(all_runs) == 2

        a_runs = tracker.list_runs(suite_name="suite_a")
        assert len(a_runs) == 1
        assert a_runs[0]["suite_name"] == "suite_a"

    def test_get_history(self, results_dir):
        tracker = RegressionTracker(results_dir=results_dir)

        for speedup in [1.0, 1.2, 1.5]:
            r = BenchmarkResult(
                suite_name="test",
                hardware="A100",
                task_results=[
                    TaskResult(task_name="t1", correctness=True, speedup_vs_reference=speedup),
                ],
            )
            tracker.record(r)

        history = tracker.get_history("t1", suite_name="test")
        assert len(history) == 3
        speedups = [h["speedup"] for h in history]
        assert speedups == [1.0, 1.2, 1.5]


# ============================================================================
# TaskComparison and ComparisonReport tests
# ============================================================================

class TestComparisonReport:
    def test_regressions_and_improvements(self):
        comps = [
            TaskComparison(task_name="improved", old_speedup=1.0, new_speedup=1.5, delta=0.5, delta_pct=50.0),
            TaskComparison(task_name="regressed", old_speedup=2.0, new_speedup=1.5, delta=-0.5, delta_pct=-25.0, regressed=True),
            TaskComparison(task_name="same", old_speedup=1.0, new_speedup=1.0, delta=0.0, delta_pct=0.0),
        ]
        report = ComparisonReport(
            baseline_commit="abc123",
            current_commit="def456",
            suite_name="test",
            task_comparisons=comps,
        )

        assert len(report.regressions) == 1
        assert report.regressions[0].task_name == "regressed"

        assert len(report.improvements) == 1
        assert report.improvements[0].task_name == "improved"

    def test_correctness_changes(self):
        comps = [
            TaskComparison(task_name="broke", old_correct=True, new_correct=False),
            TaskComparison(task_name="fixed", old_correct=False, new_correct=True),
            TaskComparison(task_name="still_good", old_correct=True, new_correct=True),
        ]
        report = ComparisonReport(
            baseline_commit="a", current_commit="b", suite_name="test",
            task_comparisons=comps,
        )
        changes = report.correctness_changes
        assert len(changes) == 2
        names = {c.task_name for c in changes}
        assert names == {"broke", "fixed"}

    def test_summary(self):
        report = ComparisonReport(
            baseline_commit="abc123",
            current_commit="def456",
            suite_name="test",
            task_comparisons=[
                TaskComparison(task_name="t1", regressed=True),
                TaskComparison(task_name="t2", delta=0.1),
            ],
            baseline_summary={"correctness_rate": 0.9, "geometric_mean_speedup": 1.5},
            current_summary={"correctness_rate": 0.95, "geometric_mean_speedup": 1.7},
        )
        s = report.summary()
        assert s["total_tasks"] == 2
        assert s["regressions"] == 1
        assert s["baseline_correctness_rate"] == 0.9
        assert s["current_geomean_speedup"] == 1.7

    def test_to_markdown(self):
        report = ComparisonReport(
            baseline_commit="abc12345",
            current_commit="def45678",
            suite_name="test",
            task_comparisons=[
                TaskComparison(task_name="regressed", old_speedup=2.0, new_speedup=1.5, delta=-0.5, delta_pct=-25.0, regressed=True),
            ],
            baseline_summary={"correctness_rate": 1.0, "geometric_mean_speedup": 2.0, "total_tokens": 1000},
            current_summary={"correctness_rate": 1.0, "geometric_mean_speedup": 1.5, "total_tokens": 900},
        )
        md = report.to_markdown()
        assert "## Benchmark Comparison: test" in md
        assert "abc12345" in md
        assert "Regressions" in md
        assert "regressed" in md


# ============================================================================
# Git helper tests
# ============================================================================

class TestGitHelpers:
    def test_get_git_commit_returns_string(self):
        commit = _get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0  # should be "unknown" or a hash

    def test_get_git_branch_returns_string(self):
        branch = _get_git_branch()
        assert isinstance(branch, str)
        assert len(branch) > 0
