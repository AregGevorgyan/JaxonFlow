"""Configuration classes for JaxonFlow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY")

    def validate(self) -> None:
        """Validate the configuration."""
        if self.api_key is None:
            env_var = "ANTHROPIC_API_KEY" if self.provider == LLMProvider.ANTHROPIC else "OPENAI_API_KEY"
            raise ConfigurationError(
                f"API key not provided and {env_var} environment variable not set"
            )


@dataclass
class CacheConfig:
    """Configuration for kernel cache."""

    enabled: bool = True
    cache_dir: Path | None = None
    max_entries: int = 10000
    eviction_ratio: float = 0.1  # Evict 10% when full

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            env_cache_dir = os.environ.get("JAXONFLOW_CACHE_DIR")
            if env_cache_dir:
                self.cache_dir = Path(env_cache_dir)
            else:
                self.cache_dir = Path.home() / ".jaxonflow" / "cache"


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system."""

    max_iterations: int = 10
    target_speedup: float = 1.0  # Relative to reference implementation
    timeout_per_iteration: float = 60.0  # Seconds
    total_timeout: float = 600.0  # 10 minutes max

    # Temperature settings for different agent roles
    planner_temperature: float = 0.8
    coder_temperature: float = 0.2
    debugger_temperature: float = 0.5
    profiler_temperature: float = 0.3
    verifier_temperature: float = 0.1

    # Token limits
    planner_max_tokens: int = 2048
    coder_max_tokens: int = 4096
    debugger_max_tokens: int = 2048
    profiler_max_tokens: int = 2048
    verifier_max_tokens: int = 1024


@dataclass
class AgentBackendConfig:
    """Main configuration for the agent backend."""

    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Hardware settings
    hardware_target: str = "auto"  # "auto", "A100-80GB", "H100-SXM", etc.

    # Fallback settings
    fallback_to_xla: bool = True
    fallback_to_inductor: bool = True

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_generated_code: bool = False

    # Cost tracking
    track_costs: bool = True
    cost_budget_usd: float | None = None  # None = unlimited

    @classmethod
    def from_env(cls) -> AgentBackendConfig:
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables
        if max_iter := os.environ.get("JAXONFLOW_MAX_ITERATIONS"):
            config.agent.max_iterations = int(max_iter)

        if target := os.environ.get("JAXONFLOW_TARGET_SPEEDUP"):
            config.agent.target_speedup = float(target)

        if fallback := os.environ.get("JAXONFLOW_FALLBACK"):
            config.fallback_to_xla = fallback.lower() in ("true", "1", "yes")
            config.fallback_to_inductor = fallback.lower() in ("true", "1", "yes")

        if log_level := os.environ.get("JAXONFLOW_LOG_LEVEL"):
            if log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
                config.log_level = log_level.upper()  # type: ignore

        if hardware := os.environ.get("JAXONFLOW_HARDWARE_TARGET"):
            config.hardware_target = hardware

        if budget := os.environ.get("JAXONFLOW_COST_BUDGET"):
            config.cost_budget_usd = float(budget)

        return config

    def validate(self) -> None:
        """Validate the entire configuration."""
        self.llm.validate()

        if self.agent.max_iterations < 1:
            raise ConfigurationError("max_iterations must be at least 1")

        if self.agent.target_speedup <= 0:
            raise ConfigurationError("target_speedup must be positive")

        if self.cache.max_entries < 1:
            raise ConfigurationError("cache max_entries must be at least 1")

        if not 0 < self.cache.eviction_ratio < 1:
            raise ConfigurationError("cache eviction_ratio must be between 0 and 1")
