"""Debugger agent for JaxonFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import LLMAgent

if TYPE_CHECKING:
    from ..llm.client import LLMResponse


class DebuggerAgent(LLMAgent):
    """Agent responsible for analyzing errors and suggesting fixes."""

    def analyze_error(
        self, code: str, error: str, error_type: str = "compilation"
    ) -> str:
        """Analyze an error and provide fixes.

        Args:
            code: The code that failed.
            error: The error message.
            error_type: Type of error ("compilation", "correctness", "runtime").

        Returns:
            Debug feedback string (YAML formatted).
        """
        prompt = (
            f"Analyze the following {error_type} error in a Triton kernel:\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            f"Provide specific fixes in YAML format."
        )

        response = self.run(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def analyze_mismatch(
        self, code: str, expected: Any, actual: Any
    ) -> str:
        """Analyze a correctness mismatch.

        Args:
            code: The code that produced incorrect results.
            expected: Expected output (summary or small sample).
            actual: Actual output (summary or small sample).
        
        Returns:
            Debug feedback string.
        """
        # Format expected/actual for prompt - keeping them concise
        # In a real scenario we might summarize statistics
        
        error_msg = (
            f"Output mismatch.\n"
            f"Expected summary: {expected}\n"
            f"Actual summary: {actual}"
        )
        
        return self.analyze_error(code, error_msg, error_type="correctness")
