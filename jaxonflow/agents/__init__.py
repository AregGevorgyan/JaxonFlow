"""Multi-agent system for kernel generation."""

from .base import Agent, AgentRole
from .orchestrator import MultiAgentOrchestrator
from .prompts import get_system_prompt

__all__ = [
    "Agent",
    "AgentRole",
    "MultiAgentOrchestrator",
    "get_system_prompt",
]
