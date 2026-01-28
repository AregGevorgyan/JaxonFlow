"""Feedback module for JaxonFlow.

Translates profiling metrics into natural language feedback for agents.
"""

from .spec import ProfileResult
from .hardware import HardwareContext

class FeedbackTranslator:
    """Translates profiling results into natural language prompts."""

    def translate(self, profile: ProfileResult, hardware: HardwareContext) -> str:
        """Convert profile metrics to actionable feedback text.

        Args:
            profile: The profiling results.
            hardware: The hardware context.

        Returns:
            A string containing the feedback for the agent.
        """
        lines = ["## Performance Feedback", ""]
        
        lines.append(f"**Speedup vs Reference**: {profile.speedup_vs_reference:.2f}x")
        lines.append(f"**Execution Time**: {profile.execution_time_us:.2f} us")
        
        if profile.primary_bottleneck:
            lines.append(f"**Primary Bottleneck**: {profile.primary_bottleneck.upper()}")
        
        if profile.memory_bandwidth_utilization:
            lines.append(f"**Memory Bandwidth Utilization**: {profile.memory_bandwidth_utilization:.1%}")
            
        if profile.compute_utilization:
            lines.append(f"**Compute Utilization**: {profile.compute_utilization:.1%}")

        if profile.achieved_occupancy:
             lines.append(f"**Achieved Occupancy**: {profile.achieved_occupancy:.1%}")

        lines.append("")
        lines.append("### Recommendations based on bottleneck:")
        
        if profile.primary_bottleneck == "memory":
            lines.append("- The kernel is memory bound.")
            lines.append("- Suggest adjusting tile sizes to better fit in L2/L1 cache.")
            lines.append("- Consider reordering loops to improve memory coalescing.")
            lines.append("- Try to increase arithmetic intensity if possible (e.g., fusion).")
            
        elif profile.primary_bottleneck == "compute":
            lines.append("- The kernel is compute bound.")
            lines.append("- Suggest using tensor cores equivalents (dot ops) if not already.")
            lines.append("- Check if tile sizes allow for enough parallelism.")
            
        elif profile.primary_bottleneck == "latency":
            lines.append("- The kernel has high latency overhead.")
            lines.append("- If kernel is very small, overhead might dwarf execution.")
            lines.append("- Check syncthreads usage.")

        return "\n".join(lines)
