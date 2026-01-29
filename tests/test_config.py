"""Tests for JaxonFlow configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from jaxonflow.config import (
    AgentBackendConfig,
    AgentConfig,
    CacheConfig,
    LLMConfig,
    LLMProvider,
)
from jaxonflow.exceptions import ConfigurationError


class TestLLMConfig:
    def test_default_provider(self):
        config = LLMConfig()
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-sonnet-4-20250514"

    def test_env_var_loading(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
            config = LLMConfig()
            assert config.api_key == "sk-test-123"

    def test_openai_env_var(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-456"}, clear=False):
            config = LLMConfig(provider=LLMProvider.OPENAI)
            assert config.api_key == "sk-openai-456"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}):
            config = LLMConfig(api_key="explicit-key")
            assert config.api_key == "explicit-key"

    def test_validate_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = LLMConfig(api_key=None)
            # Force re-init
            config.api_key = None
            with pytest.raises(ConfigurationError, match="API key"):
                config.validate()

    def test_validate_with_key(self):
        config = LLMConfig(api_key="valid-key")
        config.validate()  # Should not raise

    def test_gemini_env_var(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=False):
            config = LLMConfig(provider=LLMProvider.GEMINI)
            assert config.api_key == "gemini-key"

    def test_openrouter_env_var(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            config = LLMConfig(provider=LLMProvider.OPENROUTER)
            assert config.api_key == "or-key"

    def test_local_provider_no_key_needed(self):
        config = LLMConfig(provider=LLMProvider.LOCAL, model="llama3")
        # No env var set, api_key stays None - that's fine for local
        assert config.provider == LLMProvider.LOCAL

    def test_all_providers_exist(self):
        assert len(LLMProvider) == 7


class TestCacheConfig:
    def test_defaults(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.max_entries == 10000
        assert config.eviction_ratio == 0.1
        assert config.cache_dir is not None

    def test_env_var_cache_dir(self):
        with patch.dict(os.environ, {"JAXONFLOW_CACHE_DIR": "/custom/cache"}):
            config = CacheConfig(cache_dir=None)
            # Re-trigger __post_init__ by creating new instance
            config2 = CacheConfig()
            # The env var should be picked up
            assert str(config2.cache_dir) == "/custom/cache"

    def test_disabled_cache(self):
        config = CacheConfig(enabled=False)
        assert config.enabled is False


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.target_speedup == 1.0
        assert config.planner_temperature == 0.8
        assert config.coder_temperature == 0.2
        assert config.debugger_temperature == 0.5
        assert config.profiler_temperature == 0.3
        assert config.verifier_temperature == 0.1

    def test_token_limits(self):
        config = AgentConfig()
        assert config.coder_max_tokens == 4096
        assert config.planner_max_tokens == 2048


class TestAgentBackendConfig:
    def test_defaults(self):
        config = AgentBackendConfig()
        assert config.hardware_target == "auto"
        assert config.fallback_to_xla is True
        assert config.fallback_to_inductor is True
        assert config.log_level == "INFO"

    def test_from_env(self):
        env = {
            "JAXONFLOW_MAX_ITERATIONS": "5",
            "JAXONFLOW_TARGET_SPEEDUP": "2.0",
            "JAXONFLOW_LOG_LEVEL": "DEBUG",
            "JAXONFLOW_HARDWARE_TARGET": "H100-SXM",
            "JAXONFLOW_COST_BUDGET": "10.0",
        }
        with patch.dict(os.environ, env, clear=False):
            config = AgentBackendConfig.from_env()
            assert config.agent.max_iterations == 5
            assert config.agent.target_speedup == 2.0
            assert config.log_level == "DEBUG"
            assert config.hardware_target == "H100-SXM"
            assert config.cost_budget_usd == 10.0

    def test_validate_invalid_iterations(self):
        config = AgentBackendConfig(llm=LLMConfig(api_key="key"))
        config.agent.max_iterations = 0
        with pytest.raises(ConfigurationError, match="max_iterations"):
            config.validate()

    def test_validate_invalid_speedup(self):
        config = AgentBackendConfig(llm=LLMConfig(api_key="key"))
        config.agent.target_speedup = -1.0
        with pytest.raises(ConfigurationError, match="target_speedup"):
            config.validate()

    def test_validate_invalid_eviction_ratio(self):
        config = AgentBackendConfig(llm=LLMConfig(api_key="key"))
        config.cache.eviction_ratio = 1.5
        with pytest.raises(ConfigurationError, match="eviction_ratio"):
            config.validate()
