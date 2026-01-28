"""Profiler agent for JaxonFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..spec import ProfileResult
from ..hardware import HardwareContext
from .base import LLMAgent

if TYPE_CHECKING:
    from ..llm.client import LLMResponse


class ProfilerAgent(LLMAgent):
    """Agent responsible for analyzing performance and suggesting optimizations."""

    def analyze_performance(
        self, code: str, profile: ProfileResult, hardware: HardwareContext
    ) -> str:
        """Analyze performance metrics and provide feedback.

        Args:
            code: The kernel code.
            profile: Profiling results.
            hardware: Hardware context.

        Returns:
            Performance feedback string.
        """
        # Format profile data for prompt using actual ProfileResult fields
        latency_ms = profile.execution_time_us / 1000.0
        parts = [
            f"Speedup vs Reference: {profile.speedup_vs_reference:.2f}x",
            f"Latency: {latency_ms:.4f} ms",
        ]
        if profile.memory_bandwidth_utilization is not None:
            parts.append(f"Memory Bandwidth Utilization: {profile.memory_bandwidth_utilization:.1%}")
        if profile.compute_utilization is not None:
            parts.append(f"Compute Utilization: {profile.compute_utilization:.1%}")
        if profile.achieved_occupancy is not None:
            parts.append(f"Achieved Occupancy: {profile.achieved_occupancy:.1%}")
        if profile.primary_bottleneck is not None:
            parts.append(f"Primary Bottleneck: {profile.primary_bottleneck}")
        profile_summary = "\n".join(parts)
        
        prompt = (
            f"Analyze the performance of the following Triton kernel:\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Hardware Context:\n{hardware.to_prompt_context()}\n\n"
            f"Profiling Results:\n{profile_summary}\n\n"
            f"Identify bottlenecks and suggest optimizations in YAML format."
        )

        response = self.run(prompt)
        return response.content if hasattr(response, "content") else str(response)
