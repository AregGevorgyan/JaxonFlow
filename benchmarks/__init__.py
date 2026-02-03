"""JaxonFlow benchmark suite.

Provides tools for benchmarking kernel generation quality, measuring speedup
against framework baselines (XLA, TorchInductor), and tracking performance
across git commits.
"""

from .perf_tests import BaselineTiming, PerfTestResult, PerfTester
from .regression import ComparisonReport, RegressionTracker, TaskComparison
from .runner import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    TaskResult,
)

__all__ = [
    "BaselineTiming",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "ComparisonReport",
    "PerfTestResult",
    "PerfTester",
    "RegressionTracker",
    "TaskComparison",
    "TaskResult",
]
