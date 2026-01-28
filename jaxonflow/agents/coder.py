"""Coder agent for JaxonFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..spec import KernelSpec
from .base import LLMAgent

if TYPE_CHECKING:
    from ..llm.client import LLMResponse


class CoderAgent(LLMAgent):
    """Agent responsible for generating Triton kernel code."""

    def generate_kernel(
        self, spec: KernelSpec, plan: str, history: list[dict[str, Any]]
    ) -> str:
        """Generate Triton kernel code based on the specification and plan.

        Args:
            spec: The kernel specification.
            plan: The optimization plan.
            history: Previous iteration history for context.

        Returns:
            Generated Triton code string.
        """
        prompt_parts = [
            f"Generate a Triton kernel for the following specification:\n\n",
            f"{spec.to_prompt()}\n\n",
            f"Optimization Plan:\n{plan}\n\n",
        ]

        if history:
            prompt_parts.append("Previous Attempts:\n")
            # Look at the last few attempts for context
            for entry in history[-3:]:
                prompt_parts.append(
                    f"- Iteration {entry['iteration']}: {entry['phase']}\n"
                    f"  Error: {entry.get('error', 'N/A')}\n"
                    f"  Feedback: {entry.get('debug_feedback', 'N/A')}\n"
                )
            prompt_parts.append(
                "\nPlease fix the issues from previous attempts.\n"
            )

        prompt_parts.append(
            "Generate the complete, executable Triton kernel code. "
            "Include a wrapper function that handles grid computation and kernel launch."
        )

        prompt = "".join(prompt_parts)
        response = self.run(prompt)
        
        if hasattr(response, "content"):
            # If it's an LLMResponse or similar object
            if hasattr(self.llm_client, "extract_code"):
                code = self.llm_client.extract_code(response, language="python")
                if code:
                    return code
            
            # Fallback if extract_code returns None or client doesn't have it
            return self._extract_code(response.content)
        
        return self._extract_code(str(response))

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks if present."""
        if "```python" in text:
            try:
                start = text.index("```python") + 9
                end = text.index("```", start)
                return text[start:end].strip()
            except ValueError:
                pass
        
        if "```" in text:
             try:
                start = text.index("```") + 3
                end = text.index("```", start)
                # handle if language was specified but not python (e.g. ```triton)
                first_line_end = text.find('\n', start)
                if first_line_end != -1 and first_line_end < end:
                     # Check if the first line is just a language identifier
                     lang_line = text[start:first_line_end].strip()
                     if ' ' not in lang_line:
                          start = first_line_end + 1
                return text[start:end].strip()
             except ValueError:
                pass

        return text
