"""Cost tracking for JaxonFlow LLM usage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing for a specific model."""
    input_cost_per_m: float  # $ per 1M tokens
    output_cost_per_m: float  # $ per 1M tokens


# Default pricing (approximate as of 2024/2025)
DEFAULT_PRICING = {
    # Anthropic
    "claude-3-opus": ModelPricing(15.0, 75.0),
    "claude-3-sonnet": ModelPricing(3.0, 15.0),
    "claude-3-haiku": ModelPricing(0.25, 1.25),
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0), # Hypothetical future model
    
    # OpenAI
    "gpt-4o": ModelPricing(5.0, 15.0),
    "gpt-4-turbo": ModelPricing(10.0, 30.0),
    "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
    
    # Google
    "gemini-1.5-pro": ModelPricing(3.5, 10.5),
    "gemini-1.5-flash": ModelPricing(0.35, 1.05),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40), # Hypothetical
}


@dataclass
class UsageStats:
    """Usage statistics for a model."""
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """Tracks token usage and estimates costs across different models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CostTracker, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self) -> None:
        """Reset all statistics."""
        self._usage: dict[str, UsageStats] = defaultdict(UsageStats)
        self._budget_limit: float | None = None
        self._alert_threshold: float = 0.8
        self._alert_triggered: bool = False

    def set_budget(self, limit_usd: float) -> None:
        """Set a spending limit in USD."""
        self._budget_limit = limit_usd
        self._alert_triggered = False

    def track(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record usage for a request."""
        stats = self._usage[model]
        stats.input_tokens += input_tokens
        stats.output_tokens += output_tokens
        stats.request_count += 1
        
        self._check_budget()

    def get_total_cost(self) -> float:
        """Calculate total estimated cost in USD."""
        total = 0.0
        for model, stats in self._usage.items():
            pricing = self._get_pricing(model)
            if pricing:
                input_cost = (stats.input_tokens / 1_000_000) * pricing.input_cost_per_m
                output_cost = (stats.output_tokens / 1_000_000) * pricing.output_cost_per_m
                total += input_cost + output_cost
        return total

    def get_model_usage(self, model: str) -> UsageStats:
        """Get usage stats for a specific model."""
        return self._usage[model]
    
    def get_all_usage(self) -> dict[str, UsageStats]:
        """Get usage stats for all models."""
        return dict(self._usage)

    def _get_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a model, handling partial matches."""
        # Exact match
        if model in DEFAULT_PRICING:
            return DEFAULT_PRICING[model]
        
        # Prefix match (e.g. gpt-4o-2024-05-13 -> gpt-4o)
        for key, pricing in DEFAULT_PRICING.items():
            if model.startswith(key):
                return pricing
        
        return None

    def _check_budget(self) -> None:
        """Check if budget or alert threshold is exceeded."""
        if self._budget_limit is None:
            return

        current_cost = self.get_total_cost()
        
        if current_cost > self._budget_limit:
            logger.error(f"BUDGET EXCEEDED: ${current_cost:.4f} > ${self._budget_limit:.4f}")
            # In a real system, might raise an exception or stop generation
            
        elif not self._alert_triggered and current_cost > (self._budget_limit * self._alert_threshold):
            logger.warning(f"Budget alert: ${current_cost:.4f} is > {self._alert_threshold*100}% of ${self._budget_limit:.4f}")
            self._alert_triggered = True
