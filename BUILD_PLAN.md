# JaxonFlow Build Plan

## Phase 1: Core Infrastructure (COMPLETE)

Everything below is built and tested.

- **Configuration** (`config.py`) - `AgentBackendConfig`, `LLMConfig`, `CacheConfig`, `AgentConfig`, env var loading
- **Hardware** (`hardware.py`) - GPU profiles (A100, H100, V100, RTX4090), auto-detection, prompt formatting
- **Kernel Spec** (`spec.py`) - `KernelSpec`, `CompiledKernel`, cache keys, FLOP estimation
- **Cache** (`cache.py`) - SQLite-backed persistent cache, LRU eviction, thread-safe
- **LLM Providers** (`llm/`) - Anthropic, OpenAI, Gemini, OpenRouter, Local (Ollama/vLLM/llama.cpp)
- **Agents** (`agents/`) - 5 roles (Planner, Coder, Debugger, Profiler, Verifier), system prompts
- **Orchestrator** (`agents/orchestrator.py`) - Multi-agent generate-evaluate-refine loop
- **Backend** (`dispatch.py`) - `AgentBackend` with cache integration and fallback chain

### Testing (no GPU required)

```bash
uv run pytest tests/ -v
```

---

## Phase 2: Framework Integration

Wire JaxonFlow into JAX and PyTorch so it can intercept operations and replace them with generated kernels.

| Component | File | What |
|-----------|------|------|
| JAX Lowering | `jaxonflow/jax/lowering.py` | Custom lowering rules for JAX primitives |
| JAX Pallas | `jaxonflow/jax/pallas_backend.py` | Pallas integration for autodiff support |
| JAX Dispatch | `jaxonflow/jax/dispatch.py` | JAX-specific dispatch logic |
| PyTorch Backend | `jaxonflow/pytorch/compiler_backend.py` | `torch.compile` custom backend |
| PyTorch Custom Ops | `jaxonflow/pytorch/custom_ops.py` | `torch.library` operator registration |
| PyTorch Triton | `jaxonflow/pytorch/triton_wrapper.py` | Triton wrapper with autograd support |
| PyTorch Inductor | `jaxonflow/pytorch/inductor_extension.py` | TorchInductor extension point |

---

## Phase 3: Compilation & Execution (requires GPU)

Actually compile generated Triton code and run it on hardware.

| Component | File | What |
|-----------|------|------|
| Kernel Compiler | `jaxonflow/compiler.py` | Compile Triton source to executable kernels |
| Kernel Verifier | `jaxonflow/verification.py` | Correctness verification with real GPU tensors |
| Kernel Profiler | `jaxonflow/profiler.py` | GPU profiling (memory, compute, latency) |
| Feedback Translator | `jaxonflow/feedback.py` | Convert profile metrics to natural language for agents |

---

## Phase 4: Advanced Agents

Dedicated agent implementations with specialized logic beyond basic LLM prompting.

| Component | File | What |
|-----------|------|------|
| Planner Agent | `jaxonflow/agents/planner.py` | Optimization strategy with tile size search |
| Coder Agent | `jaxonflow/agents/coder.py` | Triton code generation with template library |
| Debugger Agent | `jaxonflow/agents/debugger.py` | Error analysis with common fix patterns |
| Profiler Agent | `jaxonflow/agents/profiler_agent.py` | Performance analysis and bottleneck detection |
| Verifier Agent | `jaxonflow/agents/verifier.py` | Extended correctness checking (numerical stability, edge cases) |

---

## Phase 5: Production Features

| Component | File | What |
|-----------|------|------|
| Async Generation | `jaxonflow/async_backend.py` | Background kernel generation, non-blocking API |
| Warm-up System | `jaxonflow/warmup.py` | Pre-generate kernels for common shapes/ops |
| Cost Tracker | `jaxonflow/cost.py` | LLM API cost budgets and alerts |
| Logging/Telemetry | `jaxonflow/telemetry.py` | Structured logging and metrics export |

---

## Phase 6: Benchmarking (COMPLETE)

Built and tested (264 tests total, all passing). Supports mock mode on CPU; real GPU profiling when hardware is available.

| Component | File | What |
|-----------|------|------|
| Benchmark Harness | `benchmarks/runner.py` | `BenchmarkRunner`, `BenchmarkSuite`, `BenchmarkTask`, `BenchmarkResult`, `TaskResult`. Built-in suites: matmul_sweep, elementwise_sweep, reduction_sweep, full. |
| Perf Tests | `benchmarks/perf_tests.py` | `PerfTester` with NumPy/XLA/Inductor baselines, median timing, per-op comparison helpers. |
| Regression Tracking | `benchmarks/regression.py` | `RegressionTracker` with git-aware recording, commit comparison, regression detection, history tracking, Markdown report generation. |
| Package Init | `benchmarks/__init__.py` | Re-exports all public classes. |
| Tests | `tests/test_benchmarks.py` | 40 tests covering runner, suites, perf, regression, serialization. |

---

## Current File Layout

```
jaxonflow/
├── __init__.py
├── config.py
├── exceptions.py
├── hardware.py
├── spec.py
├── cache.py
├── dispatch.py
├── compiler.py
├── verification.py
├── profiler.py
├── feedback.py
├── async_backend.py
├── warmup.py
├── cost.py
├── telemetry.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   └── providers/
│       ├── __init__.py
│       ├── anthropic.py
│       ├── openai.py
│       ├── gemini.py
│       ├── vertex_ai.py
│       ├── bedrock.py
│       ├── openrouter.py
│       └── local.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── orchestrator.py
│   ├── planner.py
│   ├── coder.py
│   ├── debugger.py
│   ├── profiler_agent.py
│   ├── verification.py
│   └── prompts.py
├── jax/
│   ├── __init__.py
│   ├── dispatch.py
│   ├── lowering.py
│   └── pallas_backend.py
└── pytorch/
    ├── __init__.py
    ├── compiler_backend.py
    ├── custom_ops.py
    ├── inductor_extension.py
    └── triton_wrapper.py

benchmarks/
├── __init__.py
├── runner.py
├── perf_tests.py
└── regression.py

tests/
├── conftest.py
├── test_agents.py
├── test_benchmarks.py
├── test_cache.py
├── test_config.py
├── test_dispatch.py
├── test_hardware.py
├── test_integration.py
├── test_llm_client.py
├── test_orchestrator.py
├── test_spec.py
├── test_phase2_framework.py
├── test_phase3_compiler.py
├── test_phase3_mock.py
├── test_phase4_agents.py
└── test_phase5.py
```

## Supported LLM Providers

| Provider | Env Var | Example Model |
|----------|---------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4-20250514` |
| Local | (none needed) | `llama3` (via Ollama, vLLM, etc.) |
