"""Profiler module for JaxonFlow.

Handles performance profiling of generated kernels.
Supports a mock mode for environments without GPUs.
"""

import os
import time
import logging
import random
from typing import Literal

from .spec import CompiledKernel, KernelSpec, ProfileResult
from .hardware import HardwareContext

logger = logging.getLogger(__name__)

# Check for mock mode
MOCK_MODE = os.environ.get("JAXONFLOW_MOCK_GPU", "1") == "1"


class KernelProfiler:
    """Profiles the performance of generated kernels."""

    def __init__(self, hardware: HardwareContext):
        self.hardware = hardware
        self.mock_mode = MOCK_MODE

    def profile(self, kernel: CompiledKernel, spec: KernelSpec) -> ProfileResult:
        """Profile kernel performance.

        Args:
            kernel: The compiled kernel.
            spec: The kernel specification.

        Returns:
            ProfileResult containing timing and utilization metrics.
        """
        if self.mock_mode:
            return self._profile_mock(kernel, spec)
        
        return self._profile_real(kernel, spec)

    def _profile_mock(self, kernel: CompiledKernel, spec: KernelSpec) -> ProfileResult:
        """Mock profiling for no-GPU environments."""
        logger.info(f"Mock profiling kernel for {spec.operation}")

        # Simulate plausible execution time based on estimated FLOPs
        flops = spec.estimated_flops()
        if flops is None:
            flops = 1e9  # Default to 1 GFLOP work

        # Assume 50% of peak FP32 throughput for base "execution time"
        peak_throughput = self.hardware.fp32_tflops * 1e12  # TFLOPS to FLOPS
        theoretical_time_s = flops / (peak_throughput * 0.5)
        
        # Add some random variance
        variance = random.uniform(0.8, 1.2)
        execution_time_us = (theoretical_time_s * 1e6) * variance

        # Determine if cache hit/good optimization happened
        # Randomly assign a speedup value.
        # Most of the time let's give a positive speedup to allow agent progress
        speedup = random.uniform(0.9, 1.3)

        # Random bottleneck analysis
        bottlenecks: list[Literal["memory", "compute", "latency"]] = ["memory", "compute", "latency"]
        primary_bottleneck = random.choice(bottlenecks)
        
        # Calculate derived metrics
        mem_util = random.uniform(0.4, 0.9)
        compute_util = random.uniform(0.4, 0.9)
        if primary_bottleneck == "memory":
            mem_util = random.uniform(0.85, 0.99)
            compute_util = random.uniform(0.3, 0.6)
        elif primary_bottleneck == "compute":
            compute_util = random.uniform(0.85, 0.99)
            mem_util = random.uniform(0.3, 0.6)

        return ProfileResult(
            execution_time_us=execution_time_us,
            speedup_vs_reference=speedup,
            memory_bandwidth_utilization=mem_util,
            compute_utilization=compute_util,
            achieved_occupancy=random.uniform(0.6, 0.95),
            theoretical_occupancy=random.uniform(0.8, 1.0),
            registers_per_thread=random.randint(32, 128),
            shared_memory_bytes=random.randint(4096, 49152),
            primary_bottleneck=primary_bottleneck
        )

    def _profile_real(self, kernel: CompiledKernel, spec: KernelSpec) -> ProfileResult:
        """Real profiling implementation (placeholder)."""
        logger.warning("Real profiling not implemented.")
        return ProfileResult(
            execution_time_us=0.0,
            speedup_vs_reference=0.0
        )
