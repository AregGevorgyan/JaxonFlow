"""Planner agent for JaxonFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..spec import KernelSpec
from .base import LLMAgent
from .prompts import get_system_prompt

if TYPE_CHECKING:
    from ..llm.client import LLMResponse


class PlannerAgent(LLMAgent):
    """Agent responsible for creating optimization strategies."""

    def create_plan(self, spec: KernelSpec) -> str:
        """Create an optimization plan for the given kernel specification.

        Args:
            spec: The kernel specification to plan for.

        Returns:
            The optimization plan as a string (block of text/YAML).
        """
        prompt = (
            f"Create an optimization plan for the following kernel:\n\n"
            f"{spec.to_prompt()}\n\n"
            f"Consider the hardware constraints and suggest tile sizes, "
            f"grid configuration, and memory access patterns."
        )
        response = self.run(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def revise_plan(
        self, spec: KernelSpec, current_plan: str, performance_feedback: str
    ) -> str:
        """Revise the optimization plan based on performance feedback.

        Args:
            spec: The kernel specification.
            current_plan: The current plan that was executed.
            performance_feedback: Feedback from the profiler agent.

        Returns:
            A revised optimization plan.
        """
        prompt = (
            f"Revise the optimization plan for the following kernel:\n\n"
            f"{spec.to_prompt()}\n\n"
            f"Current Plan:\n{current_plan}\n\n"
            f"Performance Feedback:\n{performance_feedback}\n\n"
            f"Suggest a new plan that addresses the performance bottlenecks associated with the current plan."
        )
        response = self.run(prompt)
        return response.content if hasattr(response, "content") else str(response)
