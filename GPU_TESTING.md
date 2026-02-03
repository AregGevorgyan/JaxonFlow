# GPU Testing Guide — Dual RTX 3090 + Vertex AI

Instructions for running JaxonFlow on a machine with 2x NVIDIA RTX 3090 GPUs, using Google Vertex AI as the LLM provider.

## Prerequisites

- 2x NVIDIA RTX 3090 (24 GB each)
- CUDA 12.x drivers installed
- Python 3.10+
- A Google Cloud project with Vertex AI API enabled
- `gcloud` CLI installed and authenticated

Verify your GPUs are visible:

```bash
nvidia-smi
```

You should see both 3090s listed.

## 1. Clone and install

```bash
git clone <repo-url> JaxonFlow
cd JaxonFlow

# Install with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[vertex-ai,dev]"

# Or with pip
pip install -e ".[vertex-ai,dev]"
```

## 2. Authenticate with Google Cloud

Vertex AI uses Application Default Credentials (no API key needed):

```bash
gcloud auth application-default login
```

Set your GCP project:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"   # or your preferred region
```

## 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Vertex AI
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# JaxonFlow settings
JAXONFLOW_HARDWARE_TARGET=auto          # will auto-detect RTX 3090
JAXONFLOW_MOCK_GPU=0                    # IMPORTANT: disable mock mode for real GPU
JAXONFLOW_MAX_ITERATIONS=10
JAXONFLOW_TARGET_SPEEDUP=1.0
JAXONFLOW_FALLBACK=true
JAXONFLOW_LOG_LEVEL=INFO
```

The critical setting is `JAXONFLOW_MOCK_GPU=0` — this enables real Triton compilation and GPU profiling instead of mock stubs.

## 4. Run the existing test suite

First confirm all tests pass in your environment:

```bash
# With mock mode (no GPU needed) — should match the 264 passing tests
JAXONFLOW_MOCK_GPU=1 uv run pytest tests/ -v

# With real GPU — compiler and profiler tests will use actual Triton
JAXONFLOW_MOCK_GPU=0 uv run pytest tests/ -v
```

## 5. Run benchmarks

### Quick smoke test

```python
#!/usr/bin/env python3
"""smoke_test.py — Verify GPU detection and basic kernel generation."""

import os
os.environ["JAXONFLOW_MOCK_GPU"] = "0"

from jaxonflow import AgentBackendConfig, AgentBackend, LLMConfig, LLMProvider, HardwareContext

# Check hardware detection
hw = HardwareContext.detect()
print(f"Detected hardware: {hw.name}")
print(f"  Compute capability: {hw.sm_arch}")
print(f"  SMs: {hw.sm_count}")
print(f"  Memory: {hw.global_memory_gb:.0f} GB")
print(f"  Bandwidth: {hw.memory_bandwidth_gbps} GB/s")
print()

# Configure with Vertex AI
config = AgentBackendConfig(
    llm=LLMConfig(
        provider=LLMProvider.VERTEX_AI,
        model="gemini-2.0-flash",     # fast and cheap for testing
    ),
    hardware_target="auto",
)

backend = AgentBackend(config)
print(f"Backend initialized on: {backend.hardware_context.name}")
print(f"Cache location: {config.cache.cache_dir}")
```

```bash
uv run python smoke_test.py
```

### Run the matmul benchmark suite

```python
#!/usr/bin/env python3
"""run_matmul_bench.py — Run matmul benchmarks on GPU."""

import os
os.environ["JAXONFLOW_MOCK_GPU"] = "0"

from jaxonflow import AgentBackendConfig, LLMConfig, LLMProvider
from benchmarks import BenchmarkRunner, BenchmarkSuite

config = AgentBackendConfig(
    llm=LLMConfig(
        provider=LLMProvider.VERTEX_AI,
        model="gemini-2.0-flash",
    ),
    hardware_target="auto",
)

runner = BenchmarkRunner(config=config, profile=True)

# Start with a small matmul sweep filtered to small/medium sizes
suite = BenchmarkSuite.matmul_sweep()
small_suite = suite.filter_by_tags("small")

result = runner.run_and_save(small_suite)
print(result.summary())
```

```bash
uv run python run_matmul_bench.py
```

### Run the full benchmark suite

```python
#!/usr/bin/env python3
"""run_full_bench.py — Full benchmark suite."""

import os
os.environ["JAXONFLOW_MOCK_GPU"] = "0"

from jaxonflow import AgentBackendConfig, LLMConfig, LLMProvider
from benchmarks import BenchmarkRunner, BenchmarkSuite

config = AgentBackendConfig(
    llm=LLMConfig(
        provider=LLMProvider.VERTEX_AI,
        model="gemini-2.0-flash",
    ),
    hardware_target="auto",
)

runner = BenchmarkRunner(config=config, profile=True)
suite = BenchmarkSuite.full()

result = runner.run_and_save(suite)

# Print summary
summary = result.summary()
for k, v in summary.items():
    print(f"  {k}: {v}")

# Print per-task results
print("\nPer-task results:")
for tr in result.task_results:
    status = "PASS" if tr.correctness else "FAIL"
    print(f"  [{status}] {tr.task_name}: speedup={tr.speedup_vs_reference:.2f}x, "
          f"exec={tr.execution_time_us:.1f}us, iters={tr.iterations_to_success}")
```

### Performance comparison against baselines

```python
#!/usr/bin/env python3
"""run_perf_comparison.py — Compare against NumPy, XLA, and Inductor."""

import os
os.environ["JAXONFLOW_MOCK_GPU"] = "0"

from jaxonflow import AgentBackendConfig, LLMConfig, LLMProvider
from benchmarks import PerfTester

config = AgentBackendConfig(
    llm=LLMConfig(
        provider=LLMProvider.VERTEX_AI,
        model="gemini-2.0-flash",
    ),
    hardware_target="auto",
)

tester = PerfTester(config=config, warmup_runs=10, timed_runs=50)

# Matmul comparisons
for size in [256, 512, 1024, 2048]:
    result = tester.compare_matmul(m=size, k=size, n=size)
    print(f"\nmatmul {size}x{size}x{size}:")
    print(f"  Agent kernel:  {result.agent_time_us:.1f} us")
    for bl in result.baselines:
        speedup = result.speedups.get(bl.name, 0)
        print(f"  vs {bl.name}: {bl.execution_time_us:.1f} us (speedup: {speedup:.2f}x)")

# Softmax comparison
result = tester.compare_softmax(rows=128, cols=1024)
print(f"\nsoftmax 128x1024:")
print(f"  Agent kernel:  {result.agent_time_us:.1f} us")
for bl in result.baselines:
    speedup = result.speedups.get(bl.name, 0)
    print(f"  vs {bl.name}: {bl.execution_time_us:.1f} us (speedup: {speedup:.2f}x)")
```

## 6. Regression tracking

After running benchmarks, record and compare across commits:

```python
#!/usr/bin/env python3
"""track_regression.py"""

from benchmarks import RegressionTracker

tracker = RegressionTracker(results_dir="benchmarks/results")

# List all recorded runs
runs = tracker.list_runs()
for run in runs:
    print(f"  [{run['commit'][:8]}] {run['suite_name']} — {run['timestamp']}")

# Compare the two most recent runs
report = tracker.compare_latest()
if report:
    print(report.to_markdown())

# Check for regressions (> 5% slowdown)
regressions = tracker.check_regressions(threshold=0.05)
if regressions:
    print(f"\n{len(regressions)} regression(s) detected:")
    for r in regressions:
        print(f"  {r.task_name}: {r.old_speedup:.2f}x -> {r.new_speedup:.2f}x ({r.delta_pct:+.1f}%)")
else:
    print("\nNo regressions detected.")
```

## 7. Multi-GPU notes

JaxonFlow currently generates kernels for a single GPU. With 2x RTX 3090:

- **Default behavior**: Uses GPU 0. Auto-detection will pick the first device.
- **To select a specific GPU**: Set `CUDA_VISIBLE_DEVICES` before running:

```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 uv run python run_full_bench.py

# Use only GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python run_full_bench.py
```

- **To benchmark both GPUs in parallel**: Run two separate benchmark processes, each pinned to a different GPU. Results can be compared via the regression tracker.

## 8. Vertex AI model options

| Model | Speed | Quality | Cost | Recommended for |
|-------|-------|---------|------|----------------|
| `gemini-2.0-flash` | Fast | Good | Low | Iterative development, quick benchmarks |
| `gemini-1.5-flash` | Fast | Good | Low | Budget-friendly alternative |
| `gemini-1.5-pro` | Medium | High | Medium | Production kernel generation |

Change the model in `LLMConfig`:

```python
LLMConfig(
    provider=LLMProvider.VERTEX_AI,
    model="gemini-1.5-pro",  # higher quality kernels
)
```

## 9. Troubleshooting

### "Triton not found"

```bash
uv pip install triton>=3.6.0
```

Triton requires a CUDA-capable GPU at runtime.

### "CUDA not available"

Check that CUDA drivers are installed and visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### "Vertex AI authentication failed"

```bash
gcloud auth application-default login
echo $GOOGLE_CLOUD_PROJECT   # must be set
```

### Mock mode still active

Ensure `JAXONFLOW_MOCK_GPU=0` is set. The default is `1` (mock mode enabled). You can set it in `.env` or export it:

```bash
export JAXONFLOW_MOCK_GPU=0
```

### Kernel generation produces no-op fallback

This means the LLM-generated code couldn't be compiled by Triton. Check:

1. `JAXONFLOW_LOG_LEVEL=DEBUG` for detailed output
2. The generated code in the logs
3. That `JAXONFLOW_MOCK_GPU=0` is set

### Out of memory on 24 GB

Large matrix sizes (4096+) with float32 may exceed 24 GB. Filter benchmarks to smaller sizes:

```python
suite = BenchmarkSuite.matmul_sweep()
small = suite.filter_by_tags("small")
medium = suite.filter_by_tags("medium")
```

## 10. Expected output structure

After running benchmarks, results are saved to `benchmarks/results/`:

```
benchmarks/results/
├── index.json                              # Index of all recorded runs
├── matmul_sweep_abc1234_20260202_143022_123456.json
├── full_abc1234_20260202_150000_654321.json
└── ...
```

Each result JSON contains a summary and per-task data:

```json
{
  "summary": {
    "suite_name": "matmul_sweep",
    "hardware": "NVIDIA RTX 3090",
    "total_tasks": 20,
    "compilation_success_rate": 0.95,
    "correctness_rate": 0.90,
    "geometric_mean_speedup": 1.12,
    "average_iterations": 3.5,
    "total_tokens": 45000,
    "wall_clock_time_s": 180.5
  },
  "tasks": [...]
}
```
