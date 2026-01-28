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
        # Format profile data for prompt
        # In a real implementation we would format this nicely
        profile_summary = (
            f"Speedup vs Reference: {profile.speedup_vs_reference:.2f}x\n"
            f"Latency: {profile.latency_ms:.4f} ms\n"
            f"Throughput: {profile.flops / 1e12:.2f} TFLOPS\n"
            f"Memory Bandwidth: {profile.memory_bandwidth_gbps:.2f} GB/s\n"
        )
        
        prompt = (
            f"Analyze the performance of the following Triton kernel:\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Hardware Context:\n{hardware.to_prompt_context()}\n\n"
            f"Profiling Results:\n{profile_summary}\n\n"
            f"Identify bottlenecks and suggest optimizations in YAML format."
        )

        response = self.run(prompt)
        return response.content if hasattr(response, "content") else str(response)
