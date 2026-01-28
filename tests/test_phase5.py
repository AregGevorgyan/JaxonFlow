"""Tests for Phase 5: Production Features.

Tests async backend, warmup, cost tracking, and telemetry.
"""

import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import Future

from jaxonflow.async_backend import AsyncKernelGenerator
from jaxonflow.warmup import WarmupSystem
from jaxonflow.cost import CostTracker, DEFAULT_PRICING, UsageStats
from jaxonflow.telemetry import TelemetryLogger, TelemetryEvent
from jaxonflow.spec import KernelSpec, CompiledKernel


# ── AsyncKernelGenerator ────────────────────────────────────────────


class TestAsyncKernelGenerator:
    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        backend.kernel_cache = MagicMock()
        backend.kernel_cache.get.return_value = None
        return backend

    def test_init(self, mock_backend):
        gen = AsyncKernelGenerator(mock_backend)
        assert gen.backend is mock_backend
        gen.shutdown()

    def test_submit_returns_future(self, mock_backend):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
        )
        mock_kernel = CompiledKernel(spec=spec, code="", callable=lambda a, b: a)
        mock_backend.get_or_generate_kernel.return_value = mock_kernel

        gen = AsyncKernelGenerator(mock_backend)
        future = gen.submit(spec)
        assert isinstance(future, Future)
        result = future.result(timeout=10)
        assert result is mock_kernel
        gen.shutdown()

    def test_submit_deduplicates(self, mock_backend):
        """Submitting the same spec twice returns the same future."""
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
        )
        # Make generation slow so the future stays pending
        import time
        def slow_generate(s):
            time.sleep(0.5)
            return CompiledKernel(spec=s, code="", callable=lambda a, b: a)
        mock_backend.get_or_generate_kernel.side_effect = slow_generate

        gen = AsyncKernelGenerator(mock_backend)
        f1 = gen.submit(spec)
        f2 = gen.submit(spec)
        assert f1 is f2
        gen.shutdown(wait=True)

    def test_get_result_if_ready_none_when_pending(self, mock_backend):
        import time
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
        )
        def slow_generate(s):
            time.sleep(2)
            return CompiledKernel(spec=s, code="", callable=lambda a, b: a)
        mock_backend.get_or_generate_kernel.side_effect = slow_generate

        gen = AsyncKernelGenerator(mock_backend)
        gen.submit(spec)
        result = gen.get_result_if_ready(spec)
        assert result is None
        gen.shutdown(wait=True)

    def test_shutdown(self, mock_backend):
        gen = AsyncKernelGenerator(mock_backend)
        gen.shutdown()  # Should not raise


# ── WarmupSystem ────────────────────────────────────────────────────


class TestWarmupSystem:
    def test_warmup_calls_backend(self):
        mock_backend = MagicMock()
        warmup = WarmupSystem(mock_backend)
        warmup.warmup(verbose=False)
        assert mock_backend.get_or_generate_kernel.called
        # Should have been called multiple times (matmul + softmax combos)
        assert mock_backend.get_or_generate_kernel.call_count > 5

    def test_warmup_handles_failures(self):
        mock_backend = MagicMock()
        mock_backend.get_or_generate_kernel.side_effect = RuntimeError("fail")
        warmup = WarmupSystem(mock_backend)
        # Should not raise despite all failures
        warmup.warmup(verbose=False)

    def test_get_common_specs(self):
        mock_backend = MagicMock()
        warmup = WarmupSystem(mock_backend)
        specs = warmup._get_common_specs()
        assert len(specs) > 0
        assert all(isinstance(s, KernelSpec) for s in specs)
        # Check we have matmul and softmax specs
        ops = {s.operation for s in specs}
        assert "matmul" in ops
        assert "softmax" in ops


# ── CostTracker ─────────────────────────────────────────────────────


class TestCostTracker:
    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset singleton state between tests."""
        tracker = CostTracker()
        tracker.reset()
        yield
        tracker.reset()

    def test_singleton(self):
        t1 = CostTracker()
        t2 = CostTracker()
        assert t1 is t2

    def test_track_usage(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", 1000, 500)
        usage = tracker.get_model_usage("gpt-4o")
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.request_count == 1
        assert usage.total_tokens == 1500

    def test_cost_calculation(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", 1_000_000, 500_000)
        cost = tracker.get_total_cost()
        # gpt-4o: $5/M input + $15/M output
        expected = (1_000_000 / 1e6 * 5.0) + (500_000 / 1e6 * 15.0)
        assert abs(cost - expected) < 0.001

    def test_cost_unknown_model(self):
        tracker = CostTracker()
        tracker.track("unknown-model", 1000, 500)
        cost = tracker.get_total_cost()
        assert cost == 0.0  # No pricing info

    def test_budget_tracking(self):
        tracker = CostTracker()
        tracker.set_budget(0.01)  # $0.01 budget
        # This should trigger budget alert (80% threshold)
        tracker.track("gpt-4o", 1_000_000, 1_000_000)
        # Cost is $5 + $15 = $20, which exceeds $0.01

    def test_multiple_models(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", 1000, 500)
        tracker.track("claude-3-sonnet", 2000, 1000)
        all_usage = tracker.get_all_usage()
        assert len(all_usage) == 2
        assert "gpt-4o" in all_usage
        assert "claude-3-sonnet" in all_usage

    def test_prefix_matching(self):
        tracker = CostTracker()
        # "gpt-4o-2024-05-13" should match "gpt-4o" pricing
        tracker.track("gpt-4o-2024-05-13", 1_000_000, 0)
        cost = tracker.get_total_cost()
        assert cost == 5.0  # $5/M input tokens for gpt-4o


# ── TelemetryLogger ────────────────────────────────────────────────


class TestTelemetryLogger:
    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset singleton state between tests."""
        logger = TelemetryLogger()
        logger.reset()
        yield
        logger.reset()

    def test_singleton(self):
        l1 = TelemetryLogger()
        l2 = TelemetryLogger()
        assert l1 is l2

    def test_log_event(self):
        logger = TelemetryLogger()
        logger.log(TelemetryEvent.SYSTEM_INIT, version="0.1.0")
        metrics = logger.get_metrics()
        assert metrics[TelemetryEvent.SYSTEM_INIT.value] == 1

    def test_multiple_events(self):
        logger = TelemetryLogger()
        logger.log(TelemetryEvent.GENERATION_START, op="matmul")
        logger.log(TelemetryEvent.GENERATION_SUCCESS, op="matmul")
        logger.log(TelemetryEvent.CACHE_HIT, key="abc123")
        logger.log(TelemetryEvent.CACHE_MISS, key="def456")
        metrics = logger.get_metrics()
        assert metrics[TelemetryEvent.GENERATION_START.value] == 1
        assert metrics[TelemetryEvent.GENERATION_SUCCESS.value] == 1
        assert metrics[TelemetryEvent.CACHE_HIT.value] == 1
        assert metrics[TelemetryEvent.CACHE_MISS.value] == 1

    def test_disabled_logging(self):
        logger = TelemetryLogger()
        logger.configure(enabled=False)
        logger.log(TelemetryEvent.SYSTEM_INIT)
        metrics = logger.get_metrics()
        # When disabled, no events are recorded in the buffer
        assert metrics[TelemetryEvent.SYSTEM_INIT.value] == 0

    def test_file_logging(self, tmp_path):
        logger = TelemetryLogger()
        logger.configure(log_dir=tmp_path, enabled=True)
        logger.log(TelemetryEvent.SYSTEM_INIT, version="0.1.0")
        logger.log(TelemetryEvent.GENERATION_START, op="matmul")
        metrics = logger.get_metrics()
        assert metrics[TelemetryEvent.SYSTEM_INIT.value] == 1
        assert metrics[TelemetryEvent.GENERATION_START.value] == 1

    def test_all_event_types_exist(self):
        expected_events = [
            "system_init", "generation_start", "generation_success",
            "generation_failure", "cache_hit", "cache_miss",
            "fallback_used", "warmup_start", "warmup_complete",
        ]
        for event_name in expected_events:
            assert hasattr(TelemetryEvent, event_name.upper())
