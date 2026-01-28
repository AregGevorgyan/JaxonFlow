
"""Base classes for agents."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.client import LLMClient, LLMResponse


class AgentRole(Enum):
    """Role of an agent in the system."""

    PLANNER = "planner"
    CODER = "coder"
    DEBUGGER = "debugger"
    PROFILER = "profiler"
    VERIFIER = "verifier"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    role: AgentRole
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    system_prompt: str = ""
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


class Agent(abc.ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration.
        """
        self.config = config

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent's main logic.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Agent's output.
        """
        pass


class LLMAgent(Agent):
    """An agent that uses an LLM client to generate responses.

    Each agent has a specialized role with its own system prompt,
    temperature setting, and token limit.
    """

    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        """Initialize an LLM-based agent.

        Args:
            config: Agent configuration with role, temperature, etc.
            llm_client: The LLM client to use for generation.
        """
        super().__init__(config)
        self.llm_client = llm_client

    def run(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """Run the agent with a formatted prompt.

        Expects the first positional argument to be the user prompt string.

        Args:
            *args: First arg should be the prompt string.
            **kwargs: Optional overrides for temperature, max_tokens.

        Returns:
            LLMResponse from the LLM client.
        """
        if not args:
            raise ValueError("LLMAgent.run() requires at least a prompt argument")

        prompt = str(args[0])
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # We assume LLMClient.generate is available and returns LLMResponse
        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
