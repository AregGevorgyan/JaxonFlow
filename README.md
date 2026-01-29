# JaxonFlow

AI-agent-based backend for JAX and PyTorch that generates optimized GPU kernels using LLMs.

Instead of traditional compiler pipelines (XLA/StableHLO for JAX, TorchInductor for PyTorch), JaxonFlow uses a multi-agent LLM system to generate, verify, and iteratively optimize Triton GPU kernels.

## Features

- **Framework-agnostic core** — shared agent system works with both JAX and PyTorch
- **Multi-agent collaboration** — specialized agents for planning, coding, debugging, profiling, and verification
- **Hardware-aware generation** — GPU specs are injected into prompts for architecture-specific optimization
- **Iterative refinement** — profile-guided feedback loops that translate metrics to natural language
- **Aggressive caching** — SQLite-backed kernel cache keyed by operation signature and hardware target
- **Multiple LLM providers** — Anthropic, OpenAI, Gemini, Vertex AI, AWS Bedrock, OpenRouter, and local models

## Requirements

- Python 3.10+
- CUDA-capable GPU (for kernel execution)

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### LLM Provider Extras

Install the provider(s) you plan to use:

```bash
# Individual providers
uv pip install -e ".[anthropic]"     # Anthropic Claude
uv pip install -e ".[openai]"        # OpenAI GPT-4
uv pip install -e ".[gemini]"        # Google Gemini (public API)
uv pip install -e ".[vertex-ai]"     # Google Vertex AI (GCP)
uv pip install -e ".[bedrock]"       # AWS Bedrock

# All LLM providers
uv pip install -e ".[llm]"

# Everything (LLM + frameworks + dev tools)
uv pip install -e ".[all]"
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for all available environment variables.

### LLM Provider Setup

**Anthropic** (default):
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**OpenAI**:
```bash
export OPENAI_API_KEY="sk-..."
```

**Google Gemini** (public API):
```bash
export GEMINI_API_KEY="..."
```

**Google Vertex AI** (GCP — no API key, uses Application Default Credentials):
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"  # optional, defaults to us-central1
```

**AWS Bedrock** (uses standard AWS credential chain):
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"  # optional, defaults to us-east-1
```

**OpenRouter**:
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Local** (Ollama, vLLM, etc. — no credentials needed):
```python
LLMConfig(provider=LLMProvider.LOCAL, model="llama3", base_url="http://localhost:11434/v1")
```

## Quick Start

### JAX

```python
from jaxonflow import AgentBackend, AgentBackendConfig, LLMConfig, LLMProvider

config = AgentBackendConfig(
    llm=LLMConfig(provider=LLMProvider.ANTHROPIC),
)
backend = AgentBackend(config)

import jax.numpy as jnp

a = jnp.ones((1024, 1024))
b = jnp.ones((1024, 1024))
c = backend.run_spec_from_arrays("matmul", [a, b])
```

### PyTorch

```python
import torch
from jaxonflow.pytorch.compiler_backend import AgentCompilerBackend
from jaxonflow import AgentBackendConfig

backend = AgentCompilerBackend(AgentBackendConfig())

@torch.compile(backend=backend)
def my_model(x, weight):
    return x @ weight
```

### Programmatic Configuration

```python
from jaxonflow import (
    AgentBackendConfig,
    AgentConfig,
    CacheConfig,
    LLMConfig,
    LLMProvider,
)

config = AgentBackendConfig(
    llm=LLMConfig(
        provider=LLMProvider.BEDROCK,
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    ),
    agent=AgentConfig(
        max_iterations=10,
        target_speedup=1.5,
    ),
    cache=CacheConfig(
        enabled=True,
        max_entries=10000,
    ),
    hardware_target="auto",
    fallback_to_xla=True,
)
```

## Project Structure

```
jaxonflow/
    __init__.py          # Public API
    config.py            # Configuration classes
    dispatch.py          # Agent backend dispatcher
    spec.py              # Kernel specification format
    hardware.py          # GPU hardware context
    cache.py             # SQLite-backed kernel cache
    compiler.py          # Kernel compilation
    verification.py      # Correctness verification
    profiler.py          # Performance profiling
    feedback.py          # Profile-to-language translation
    async_backend.py     # Async kernel generation
    warmup.py            # Pre-generation of common kernels
    cost.py              # LLM cost tracking
    telemetry.py         # Event logging
    agents/              # Multi-agent system
        orchestrator.py  # Agent coordination loop
        base.py          # LLM agent base class
        prompts.py       # Role-specific system prompts
    llm/                 # LLM provider abstraction
        client.py        # Abstract client + factory
        providers/       # Provider implementations
    jax/                 # JAX integration layer
    pytorch/             # PyTorch integration layer
tests/                   # Test suite
```

## Running Tests

```bash
uv run pytest -v
```

## Architecture

See [AGENTS.md](AGENTS.md) for the full architecture document covering:

- Multi-agent system design (planner, coder, debugger, profiler, verifier)
- JAX and PyTorch integration layers
- Kernel specification format
- Feedback loop design
- Verification and correctness
- Caching and deployment considerations

## License

Apache 2.0
