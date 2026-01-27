"""Base classes for agents."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
