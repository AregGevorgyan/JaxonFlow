# JaxonFlow: Building an AI-Agent-Based High-Performance Backend for JAX and PyTorch

> A comprehensive guide to replacing XLA/StableHLO and TorchInductor with LLM-powered agents for kernel generation
This project is called JaxonFlow.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation](#motivation)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Multi-Agent System Design](#multi-agent-system-design)
6. [JAX Integration Layer](#jax-integration-layer)
7. [PyTorch Integration Layer](#pytorch-integration-layer)
8. [Kernel Specification Format](#kernel-specification-format)
9. [Feedback Loop Design](#feedback-loop-design)
10. [Verification and Correctness](#verification-and-correctness)
11. [Memory and Caching](#memory-and-caching)
12. [Benchmarking](#benchmarking)
13. [Deployment Considerations](#deployment-considerations)
14. [References](#references)

---

## Executive Summary

This document describes the architecture for an alternative backend for **both JAX and PyTorch** that uses LLM-powered AI agents to generate optimized GPU kernels. For JAX, this replaces the traditional StableHLO → XLA → LLVM pipeline. For PyTorch, this replaces or augments TorchInductor's Triton code generation. Instead of rule-based compiler optimizations, the system leverages large language models in an iterative generate-evaluate-refine loop to produce high-performance Triton kernels.

### Key Design Principles

1. **Framework-agnostic core**: Shared agent system works with both JAX and PyTorch
2. **Multi-agent collaboration**: Specialized agents for planning, coding, debugging, and verification
3. **Hardware-aware generation**: Inject GPU specifications into prompts for architecture-specific optimization
4. **Iterative refinement**: Profile-guided feedback loops that translate metrics to natural language
5. **Deterministic orchestration**: Python controls workflow; LLMs only generate kernel code
6. **Aggressive caching**: Memoize generated kernels by operation signature and hardware target

---

## Motivation

### Limitations of Traditional Compilers

Traditional ML compilers like XLA and TorchInductor face fundamental challenges:

| Challenge | XLA (JAX) | TorchInductor (PyTorch) |
|-----------|-----------|------------------------|
| **Fixed optimization rules** | Cannot discover novel algorithms | Template-based Triton generation is rigid |
| **Long development cycles** | New hardware requires compiler work | Inductor patterns require manual addition |
| **Limited fusion heuristics** | Rule-based fusion misses opportunities | Fusion decisions are heuristic-based |
| **Cross-platform burden** | Separate backends per accelerator | Limited non-NVIDIA support |
| **Dynamic shapes** | Recompilation overhead | Guard overhead and specialization |

### Why LLM Agents?

LLMs offer a paradigm shift:

- **Compressed expert knowledge**: Training on vast code repositories captures optimization patterns
- **Flexible reasoning**: Can adapt strategies based on hardware and workload characteristics
- **Iterative improvement**: Feedback-driven refinement navigates irregular optimization landscapes
- **Rapid adaptation**: New hardware support via prompt engineering, not compiler rewrites

### Research Foundation

This architecture draws from recent advances in LLM-driven kernel generation:

- **STARK**: Multi-agent collaboration with planning/coding/debugging separation
- **Astra**: Specialized agents for testing, profiling, and planning
- **CUDA-LLM**: Hardware-aware prompting with GPU specifications
- **KernelFalcon**: Deterministic orchestration with parallel worker pools
- **TritonRL**: Reinforcement learning with hierarchical reward decomposition

---

## Architecture Overview

```
┌────────────────────────────────────┐    ┌────────────────────────────────────┐
│          JAX User Code             │    │        PyTorch User Code           │
│  (jax.numpy, jit, vmap, grad)      │    │    (torch.nn, torch.compile)       │
└────────────────────────────────────┘    └────────────────────────────────────┘
                  │                                        │
                  ▼                                        ▼
┌────────────────────────────────────┐    ┌────────────────────────────────────┐
│        JAX Trace System            │    │      TorchDynamo + FX Graph        │
│   (Captures JAX primitives)        │    │    (Captures PyTorch operations)   │
└────────────────────────────────────┘    └────────────────────────────────────┘
                  │                                        │
                  │         ┌──────────────────┐           │
                  └────────▶│  Unified Kernel  │◀─────────┘
                            │  Specification   │
                            └──────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Agent Backend (Framework-Agnostic)                     │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Primitive/Op   │    │   KernelSpec    │    │     Cache       │        │
│  │  Interception   │───▶│   Extraction    │───▶│     Lookup      │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                        │                   │
│                              ┌─────────────────────────┼───────────────┐   │
│                              │ Cache Miss              │ Cache Hit     │   │
│                              ▼                         ▼               │   │
│                    ┌─────────────────┐       ┌─────────────────┐      │   │
│                    │  Agent System   │       │  Return Cached  │      │   │
│                    └─────────────────┘       │     Kernel      │      │   │
│                              │               └─────────────────┘      │   │
└──────────────────────────────┼────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Agent Orchestrator                             │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │ Planning │    │  Coding  │    │ Debugging│    │Profiling │            │
│  │  Agent   │───▶│  Agent   │───▶│  Agent   │───▶│  Agent   │            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│       │                                               │                    │
│       │              Feedback Loop                    │                    │
│       └───────────────────────────────────────────────┘                    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                    Verification Agent                            │     │
│  │  (Correctness checking against reference implementation)         │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Generated Triton Kernels                               │
│                  (Shared format for both frameworks)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   JAX Pallas     │ │  PyTorch Custom  │ │   Direct Triton  │
│   Integration    │ │     Operator     │ │    Execution     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Hardware Execution                                  │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  NVIDIA GPU     │    │    AMD GPU      │    │   Google TPU    │        │
│  │  (PTX/SASS)     │    │   (HSACO)       │    │   (Mosaic)      │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Unified KernelSpec**: Both JAX primitives and PyTorch operations map to the same specification format
2. **Shared Agent System**: The multi-agent orchestrator is framework-agnostic
3. **Triton as Common Target**: Generated kernels are Triton code, usable by both frameworks
4. **Framework-Specific Wrappers**: Thin integration layers adapt kernels to each framework's calling conventions

---

## Core Components

### 1. Primitive Interception Layer

The entry point intercepts JAX's primitive operations before they reach the standard XLA lowering:

```python
# jax_agent_backend/dispatch.py

from jax._src import core
from jax._src.interpreters import mlir
from jax import lax
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class AgentBackendConfig:
    """Configuration for the agent backend."""
    enable_caching: bool = True
    max_iterations: int = 10
    target_speedup: float = 1.0  # vs reference implementation
    hardware_target: str = "auto"  # auto-detect or specify
    fallback_to_xla: bool = True  # use XLA if agent fails

class AgentBackend:
    """Main backend class that intercepts JAX primitives."""
    
    def __init__(self, config: AgentBackendConfig):
        self.config = config
        self.kernel_cache = KernelCache()
        self.agent_system = MultiAgentOrchestrator(config)
        self.hardware_context = HardwareContext.detect()
        
    def register_lowerings(self):
        """Register custom lowering rules for JAX primitives."""
        # High-value primitives to intercept
        primitives_to_intercept = [
            lax.dot_general_p,           # Matrix multiplication
            lax.conv_general_dilated_p,  # Convolution
            lax.reduce_sum_p,            # Reductions
            lax.reduce_max_p,
            # Add more as needed
        ]
        
        for primitive in primitives_to_intercept:
            self._register_primitive(primitive)
    
    def _extract_kernel_spec(self, primitive, ctx, args, params) -> 'KernelSpec':
        """Convert JAX primitive to KernelSpec."""
        return KernelSpec(
            operation=primitive.name,
            input_shapes=[arg.shape for arg in args],
            input_dtypes=[str(arg.dtype) for arg in args],
            output_shape=ctx.avals_out[0].shape,
            output_dtype=str(ctx.avals_out[0].dtype),
            parameters=params,
            hardware=self.hardware_context
        )
    
    def _get_or_generate_kernel(self, spec: 'KernelSpec') -> 'CompiledKernel':
        """Check cache or generate new kernel."""
        cache_key = spec.cache_key()
        
        if self.config.enable_caching:
            cached = self.kernel_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Generate via agent system
        kernel = self.agent_system.generate_kernel(spec)
        
        if kernel is not None:
            self.kernel_cache.put(cache_key, kernel)
            return kernel
        
        # Fallback to XLA if enabled
        if self.config.fallback_to_xla:
            return self._xla_fallback(spec)
        
        raise RuntimeError(f"Failed to generate kernel for {spec.operation}")
```

### 2. Hardware Context

Captures GPU specifications for hardware-aware prompting:

```python
# jax_agent_backend/hardware.py

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class HardwareContext:
    """Hardware specifications for kernel optimization."""
    
    # Device identification
    name: str
    compute_capability: Tuple[int, int]  # e.g., (9, 0) for SM90
    vendor: str  # "nvidia", "amd", "google"
    
    # Compute resources
    sm_count: int
    cores_per_sm: int
    warp_size: int
    max_threads_per_block: int
    max_blocks_per_sm: int
    
    # Memory hierarchy
    global_memory_gb: float
    shared_memory_per_block: int  # bytes
    shared_memory_per_sm: int     # bytes
    l2_cache_size: int            # bytes
    registers_per_block: int
    
    # Bandwidth
    memory_bandwidth_gbps: float
    
    # Compute throughput (TFLOPS)
    fp32_tflops: float
    fp16_tflops: float
    tf32_tflops: float
    fp8_tflops: Optional[float] = None
    
    @classmethod
    def detect(cls) -> 'HardwareContext':
        """Auto-detect hardware from current device."""
        # Implementation uses nvidia-smi, rocm-smi, or JAX device query
        import jax
        devices = jax.devices()
        if devices and 'gpu' in str(devices[0]).lower():
            # Query GPU properties
            return cls.from_name("A100-80GB")  # Default fallback
        return cls.from_name("A100-80GB")
    
    @classmethod
    def from_name(cls, name: str) -> 'HardwareContext':
        """Load known hardware profile by name."""
        profiles = {
            "A100-80GB": cls(
                name="NVIDIA A100-80GB",
                compute_capability=(8, 0),
                vendor="nvidia",
                sm_count=108,
                cores_per_sm=64,
                warp_size=32,
                max_threads_per_block=1024,
                max_blocks_per_sm=32,
                global_memory_gb=80,
                shared_memory_per_block=163840,
                shared_memory_per_sm=167936,
                l2_cache_size=41943040,
                registers_per_block=65536,
                memory_bandwidth_gbps=2039,
                fp32_tflops=19.5,
                fp16_tflops=312,
                tf32_tflops=156,
            ),
            "H100-SXM": cls(
                name="NVIDIA H100-SXM",
                compute_capability=(9, 0),
                vendor="nvidia",
                sm_count=132,
                cores_per_sm=128,
                warp_size=32,
                max_threads_per_block=1024,
                max_blocks_per_sm=32,
                global_memory_gb=80,
                shared_memory_per_block=232448,
                shared_memory_per_sm=233472,
                l2_cache_size=52428800,
                registers_per_block=65536,
                memory_bandwidth_gbps=3352,
                fp32_tflops=67,
                fp16_tflops=1979,
                tf32_tflops=989,
                fp8_tflops=3958
            ),
        }
        return profiles.get(name, profiles["A100-80GB"])
    
    def to_prompt_context(self) -> str:
        """Format hardware specs for LLM prompt."""
        return f"""Target Hardware: {self.name}
Vendor: {self.vendor}
Compute Capability: SM{self.compute_capability[0]}{self.compute_capability[1]}

Compute Resources:
- SMs: {self.sm_count}
- Cores per SM: {self.cores_per_sm}
- Warp Size: {self.warp_size}
- Max Threads per Block: {self.max_threads_per_block}
- Max Blocks per SM: {self.max_blocks_per_sm}

Memory Hierarchy:
- Global Memory: {self.global_memory_gb} GB
- Shared Memory per Block: {self.shared_memory_per_block:,} bytes
- L2 Cache: {self.l2_cache_size // 1024 // 1024} MB
- Registers per Block: {self.registers_per_block:,}

Bandwidth:
- Memory Bandwidth: {self.memory_bandwidth_gbps} GB/s

Compute Throughput:
- FP32: {self.fp32_tflops} TFLOPS
- FP16: {self.fp16_tflops} TFLOPS
- TF32: {self.tf32_tflops} TFLOPS"""
```

---

## Multi-Agent System Design

### Agent Architecture

The system uses specialized agents, each optimized for a specific subtask:

```python
# jax_agent_backend/agents/orchestrator.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

class AgentRole(Enum):
    PLANNER = "planner"
    CODER = "coder"
    DEBUGGER = "debugger"
    PROFILER = "profiler"
    VERIFIER = "verifier"

@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    role: AgentRole
    model: str  # e.g., "claude-sonnet-4-20250514", "gpt-4o"
    temperature: float
    max_tokens: int
    system_prompt: str

class MultiAgentOrchestrator:
    """
    Coordinates multiple specialized agents for kernel generation.
    
    Key insight from STARK: Different subtasks require different LLM behaviors.
    - Planning: High temperature (0.8-0.9) for creative strategy exploration
    - Coding: Low temperature (0.1-0.3) for precise, correct code
    - Debugging: Medium temperature (0.4-0.6) for analytical reasoning
    """
    
    def __init__(self, config: AgentBackendConfig):
        self.config = config
        self.agents = self._initialize_agents()
        self.memory = AgentMemory()
        
    def _initialize_agents(self) -> Dict[AgentRole, 'Agent']:
        """Initialize specialized agents with role-appropriate configurations."""
        return {
            AgentRole.PLANNER: Agent(AgentConfig(
                role=AgentRole.PLANNER,
                model="claude-sonnet-4-20250514",
                temperature=0.8,
                max_tokens=2048,
                system_prompt=PLANNER_SYSTEM_PROMPT
            )),
            AgentRole.CODER: Agent(AgentConfig(
                role=AgentRole.CODER,
                model="claude-sonnet-4-20250514",
                temperature=0.2,
                max_tokens=4096,
                system_prompt=CODER_SYSTEM_PROMPT
            )),
            AgentRole.DEBUGGER: Agent(AgentConfig(
                role=AgentRole.DEBUGGER,
                model="claude-sonnet-4-20250514",
                temperature=0.5,
                max_tokens=2048,
                system_prompt=DEBUGGER_SYSTEM_PROMPT
            )),
            AgentRole.PROFILER: Agent(AgentConfig(
                role=AgentRole.PROFILER,
                model="claude-sonnet-4-20250514",
                temperature=0.3,
                max_tokens=2048,
                system_prompt=PROFILER_SYSTEM_PROMPT
            )),
            AgentRole.VERIFIER: Agent(AgentConfig(
                role=AgentRole.VERIFIER,
                model="claude-sonnet-4-20250514",
                temperature=0.1,
                max_tokens=1024,
                system_prompt=VERIFIER_SYSTEM_PROMPT
            )),
        }
    
    def generate_kernel(self, spec: 'KernelSpec') -> Optional['CompiledKernel']:
        """
        Main kernel generation loop.
        
        Flow:
        1. Planner creates optimization strategy
        2. Coder generates Triton kernel
        3. Verifier checks correctness
        4. If incorrect: Debugger analyzes and feeds back to Coder
        5. If correct: Profiler analyzes performance
        6. If slow: Profiler feeds back to Planner for new strategy
        7. Repeat until target met or max iterations
        """
        best_kernel = None
        best_speedup = 0.0
        
        # Phase 1: Planning
        plan = self.agents[AgentRole.PLANNER].create_plan(spec)
        
        for iteration in range(self.config.max_iterations):
            # Phase 2: Code Generation
            code = self.agents[AgentRole.CODER].generate_kernel(
                spec=spec,
                plan=plan,
                history=self.memory.get_iteration_history()
            )
            
            # Phase 3: Compilation
            compile_result = self._compile_kernel(code)
            if not compile_result.success:
                # Debugging feedback loop
                debug_feedback = self.agents[AgentRole.DEBUGGER].analyze_error(
                    code=code,
                    error=compile_result.error
                )
                self.memory.add_iteration(code, compile_result, debug_feedback)
                continue
            
            # Phase 4: Correctness Verification
            verify_result = self._verify_correctness(compile_result.kernel, spec)
            if not verify_result.correct:
                debug_feedback = self.agents[AgentRole.DEBUGGER].analyze_mismatch(
                    code=code,
                    expected=verify_result.expected,
                    actual=verify_result.actual
                )
                self.memory.add_iteration(code, verify_result, debug_feedback)
                continue
            
            # Phase 5: Performance Profiling
            profile_result = self._profile_kernel(compile_result.kernel, spec)
            speedup = profile_result.speedup_vs_reference
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_kernel = compile_result.kernel
            
            # Check if target met
            if speedup >= self.config.target_speedup:
                return best_kernel
            
            # Phase 6: Performance Feedback
            perf_feedback = self.agents[AgentRole.PROFILER].analyze_performance(
                code=code,
                profile=profile_result,
                hardware=spec.hardware
            )
            
            # Update plan based on profiling insights
            plan = self.agents[AgentRole.PLANNER].revise_plan(
                spec=spec,
                current_plan=plan,
                performance_feedback=perf_feedback
            )
            
            self.memory.add_iteration(code, profile_result, perf_feedback)
        
        return best_kernel
    
    def _compile_kernel(self, code: str) -> 'CompilationResult':
        """Compile the generated kernel code."""
        compiler = KernelCompiler()
        return compiler.compile(code, self.hardware_context)
    
    def _verify_correctness(self, kernel, spec) -> 'VerificationResult':
        """Verify kernel correctness."""
        verifier = KernelVerifier()
        return verifier.verify(kernel, spec)
    
    def _profile_kernel(self, kernel, spec) -> 'ProfileResult':
        """Profile kernel performance."""
        profiler = KernelProfiler(spec.hardware)
        return profiler.profile(kernel, spec)
```

### Agent System Prompts

Each agent has a specialized system prompt:

```python
# jax_agent_backend/agents/prompts.py

PLANNER_SYSTEM_PROMPT = """You are an expert GPU kernel optimization planner. 
Your role is to devise high-level optimization strategies for GPU kernels based 
on the operation semantics and target hardware.

Your plans should be:
1. GROUNDED: Reference specific code locations and data structures
2. HARDWARE-AWARE: Consider memory hierarchy, compute throughput, and occupancy
3. ACTIONABLE: Provide concrete steps the coding agent can implement

Output format (YAML):
```yaml
strategy:
  name: "descriptive name"
  rationale: "why this approach suits this workload and hardware"
  
optimizations:
  - type: "memory_tiling"
    target: "input matrices"
    tile_size: [128, 128]
    reason: "fits in shared memory with double buffering"

grid_config:
  block_size: [128, 4]
  reason: "balance occupancy with register pressure"

expected_bottleneck: "memory_bound" | "compute_bound" | "latency_bound"
```

Do NOT generate code. Focus only on strategy."""

CODER_SYSTEM_PROMPT = """You are an expert Triton kernel programmer. Your role 
is to implement GPU kernels following the optimization plan provided.

Requirements:
1. Generate COMPLETE, EXECUTABLE Triton code
2. Follow the plan's tile sizes, grid configuration, and optimization strategies
3. Include proper boundary handling for arbitrary input sizes
4. Use descriptive variable names and add comments for complex sections

Output only the complete kernel code, no explanations."""

DEBUGGER_SYSTEM_PROMPT = """You are an expert at debugging GPU kernels. Your 
role is to analyze compilation errors or correctness mismatches and provide 
specific fixes.

For compilation errors:
1. Identify the root cause (syntax, type mismatch, invalid Triton operation)
2. Provide the EXACT fix with before/after code snippets
3. Explain why the original code was wrong

Output format (YAML):
```yaml
diagnosis:
  error_type: "compilation" | "correctness"
  root_cause: "brief description"
  
fixes:
  - location: "line number or code section"
    original: "problematic code"
    replacement: "fixed code"
    explanation: "why this fixes the issue"
```

Be SPECIFIC. Vague suggestions waste iterations."""

PROFILER_SYSTEM_PROMPT = """You are an expert at interpreting GPU profiling data 
and providing actionable optimization suggestions.

Given profiling metrics and hardware specifications, identify:
1. The primary bottleneck (memory, compute, or latency)
2. Specific inefficiencies in the current implementation
3. Concrete optimizations to address each issue

Output format (YAML):
```yaml
bottleneck_analysis:
  primary: "memory_bound" | "compute_bound" | "latency_bound"
  utilization:
    memory_bandwidth: 75%
    compute_throughput: 30%
  
recommendations:
  - optimization: "transpose K dimension access pattern"
    expected_improvement: "20-30% memory bandwidth"
    implementation_hint: "swap loop order"
```

Translate metrics to NATURAL LANGUAGE suggestions the coder can act on."""

VERIFIER_SYSTEM_PROMPT = """You are a verification agent that checks kernel 
correctness. You do NOT fix errors. You only report them precisely.

Output format (YAML):
```yaml
verification_result:
  status: "pass" | "fail"
  
  failure_info:  # If fail
    test_case: "description of failing input"
    max_abs_error: 0.005
    error_pattern: "systematic offset" | "random noise" | "NaN/Inf"
```"""
```

---

## JAX Integration Layer

### Option 1: Custom Lowering Rules (Recommended)

Hook into JAX's MLIR lowering system:

```python
# jax_agent_backend/lowering.py

from jax._src.interpreters import mlir
from jax import lax

def register_agent_lowerings(backend: AgentBackend):
    """Register agent-based lowerings for high-value primitives."""
    
    @mlir.register_lowering(lax.dot_general_p, platform="cuda")
    def dot_general_agent_lowering(ctx, lhs, rhs, *, dimension_numbers, 
                                    precision, preferred_element_type):
        spec = KernelSpec(
            operation="dot_general",
            input_shapes=[ctx.avals_in[0].shape, ctx.avals_in[1].shape],
            input_dtypes=[str(ctx.avals_in[0].dtype), str(ctx.avals_in[1].dtype)],
            output_shape=ctx.avals_out[0].shape,
            output_dtype=str(ctx.avals_out[0].dtype),
            parameters={
                "dimension_numbers": dimension_numbers,
                "precision": precision,
            },
            hardware=backend.hardware_context
        )
        
        kernel = backend.get_or_generate_kernel(spec)
        return emit_triton_kernel_call(ctx, kernel, [lhs, rhs])
```

### Option 2: Pallas-Based Integration

Use JAX's Pallas extension for cleaner integration with automatic autodiff:

```python
# jax_agent_backend/pallas_backend.py

from jax.experimental import pallas as pl
import jax
import jax.numpy as jnp

class PallasAgentBackend:
    """Generate Pallas kernels via LLM agents."""
    
    def __init__(self, agent_system: MultiAgentOrchestrator):
        self.agent_system = agent_system
        self.kernel_cache = {}
    
    def create_matmul(self, m: int, n: int, k: int, dtype) -> callable:
        """Create an agent-optimized matmul operation."""
        
        cache_key = ("matmul", m, n, k, dtype)
        if cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]
        
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(m, k), (k, n)],
            input_dtypes=[str(dtype), str(dtype)],
            output_shape=(m, n),
            output_dtype=str(dtype),
            parameters={},
            hardware=HardwareContext.detect()
        )
        
        # Agent generates Pallas kernel
        pallas_code = self.agent_system.generate_pallas_kernel(spec)
        kernel_fn = self._compile_pallas_code(pallas_code)
        
        block_m, block_n, block_k = 128, 128, 32  # From spec
        
        @jax.jit
        def optimized_matmul(a, b):
            return pl.pallas_call(
                kernel_fn,
                out_shape=jax.ShapeDtypeStruct((m, n), dtype),
                grid=(m // block_m, n // block_n),
                in_specs=[
                    pl.BlockSpec((block_m, block_k), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_k, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            )(a, b)
        
        self.kernel_cache[cache_key] = optimized_matmul
        return optimized_matmul
```

---

## PyTorch Integration Layer

PyTorch offers multiple integration points for custom backends. We support four approaches:

### Option 1: Custom torch.compile Backend (Recommended)

Replace or augment TorchInductor with agent-generated kernels:

```python
# agent_backend/pytorch/compiler_backend.py

import torch
from torch._dynamo.backends.common import aot_autograd
from typing import List, Callable

class AgentCompilerBackend:
    """
    Custom torch.compile backend that uses AI agents for kernel generation.
    
    Usage:
        @torch.compile(backend=AgentCompilerBackend())
        def my_model(x):
            return x @ x.T
    """
    
    def __init__(self, config: AgentBackendConfig):
        self.config = config
        self.agent_system = MultiAgentOrchestrator(config)
        self.kernel_cache = KernelCache()
        self.hardware = HardwareContext.detect()
    
    def __call__(self, gm: torch.fx.GraphModule, 
                 example_inputs: List[torch.Tensor]) -> Callable:
        """
        Compile a GraphModule using agent-generated kernels.
        
        Args:
            gm: FX GraphModule captured by TorchDynamo
            example_inputs: Example inputs for shape inference
            
        Returns:
            Compiled callable
        """
        # Analyze the graph for fusible subgraphs
        subgraphs = self._identify_subgraphs(gm)
        
        # Generate kernels for each subgraph
        kernel_map = {}
        for subgraph_id, subgraph in subgraphs.items():
            spec = self._subgraph_to_spec(subgraph, example_inputs)
            kernel = self._get_or_generate_kernel(spec)
            kernel_map[subgraph_id] = kernel
        
        # Create optimized callable
        return self._create_optimized_callable(gm, kernel_map, example_inputs)
    
    def _identify_subgraphs(self, gm: torch.fx.GraphModule) -> dict:
        """
        Identify fusible subgraphs in the FX graph.
        
        Looks for patterns like:
        - Linear + Activation
        - Attention (Q @ K.T, softmax, @ V)
        - LayerNorm + Linear
        - Conv + BatchNorm + Activation
        """
        subgraphs = {}
        
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                # Check for high-value operations
                if self._is_matmul(node):
                    subgraphs[node.name] = self._extract_matmul_subgraph(gm, node)
                elif self._is_attention_pattern(node, gm):
                    subgraphs[node.name] = self._extract_attention_subgraph(gm, node)
        
        return subgraphs
    
    def _subgraph_to_spec(self, subgraph: dict, 
                          example_inputs: List[torch.Tensor]) -> KernelSpec:
        """Convert FX subgraph to KernelSpec."""
        
        input_shapes = []
        input_dtypes = []
        
        for inp in subgraph['inputs']:
            if isinstance(inp, torch.Tensor):
                input_shapes.append(tuple(inp.shape))
                input_dtypes.append(str(inp.dtype).replace('torch.', ''))
        
        return KernelSpec(
            operation=subgraph['operation'],
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_shape=subgraph['output_shape'],
            output_dtype=subgraph['output_dtype'],
            parameters=subgraph.get('parameters', {}),
            hardware=self.hardware,
            framework="pytorch"
        )


# Register as a torch.compile backend
from torch._dynamo import register_backend

@register_backend
def agent_backend(gm, example_inputs):
    """Registered backend for torch.compile(backend='agent_backend')"""
    config = AgentBackendConfig()
    backend = AgentCompilerBackend(config)
    return backend(gm, example_inputs)
```

### Option 2: Custom Operators via torch.library

Register agent-generated kernels as custom operators:

```python
# agent_backend/pytorch/custom_ops.py

import torch
from torch.library import Library, impl

# Create a library for agent-generated ops
agent_lib = Library("agent_ops", "DEF")

class AgentOpRegistry:
    """Registry for agent-generated custom operators."""
    
    def __init__(self, agent_system: MultiAgentOrchestrator):
        self.agent_system = agent_system
        self.registered_ops = {}
    
    def register_matmul(self, name: str = "agent_matmul"):
        """Register an agent-optimized matmul."""
        
        # Define the operator schema
        agent_lib.define(f"{name}(Tensor a, Tensor b) -> Tensor")
        
        @impl(agent_lib, name, "CUDA")
        def agent_matmul_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            spec = KernelSpec(
                operation="matmul",
                input_shapes=[tuple(a.shape), tuple(b.shape)],
                input_dtypes=[str(a.dtype).replace('torch.', ''), 
                              str(b.dtype).replace('torch.', '')],
                output_shape=(a.shape[0], b.shape[1]),
                output_dtype=str(a.dtype).replace('torch.', ''),
                hardware=HardwareContext.detect(),
                framework="pytorch"
            )
            
            kernel = self._get_kernel(spec)
            return kernel(a, b)
        
        # CPU fallback
        @impl(agent_lib, name, "CPU")
        def agent_matmul_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.mm(a, b)
        
        self.registered_ops[name] = True
        return getattr(torch.ops.agent_ops, name)
    
    def _get_kernel(self, spec: KernelSpec):
        """Get or generate kernel for spec."""
        cache_key = spec.cache_key()
        kernel = self.agent_system.kernel_cache.get(cache_key)
        
        if kernel is None:
            kernel = self.agent_system.generate_kernel(spec)
            self.agent_system.kernel_cache.put(cache_key, kernel)
        
        return kernel
```

### Option 3: Triton Kernel Wrapper with Autograd

Directly wrap agent-generated Triton kernels for PyTorch with gradient support:

```python
# agent_backend/pytorch/triton_wrapper.py

import torch
import triton
from typing import Tuple, Optional

class TritonKernelWrapper:
    """
    Wrapper to make agent-generated Triton kernels usable in PyTorch.
    
    Handles:
    - Autograd integration (forward + backward)
    - torch.compile compatibility
    - Device/dtype validation
    """
    
    def __init__(self, agent_system: MultiAgentOrchestrator):
        self.agent_system = agent_system
        self.kernel_cache = {}
    
    def create_matmul(self) -> torch.autograd.Function:
        """Create a PyTorch autograd function for agent-optimized matmul."""
        
        agent_system = self.agent_system
        kernel_cache = self.kernel_cache
        
        class AgentMatmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
                assert a.shape[1] == b.shape[0], "Shape mismatch"
                
                cache_key = (
                    "matmul", 
                    tuple(a.shape), tuple(b.shape), 
                    str(a.dtype), str(b.dtype)
                )
                
                if cache_key not in kernel_cache:
                    spec = KernelSpec(
                        operation="matmul",
                        input_shapes=[tuple(a.shape), tuple(b.shape)],
                        input_dtypes=[str(a.dtype).replace('torch.', ''), 
                                      str(b.dtype).replace('torch.', '')],
                        output_shape=(a.shape[0], b.shape[1]),
                        output_dtype=str(a.dtype).replace('torch.', ''),
                        hardware=HardwareContext.detect(),
                        framework="pytorch"
                    )
                    kernel_cache[cache_key] = agent_system.generate_kernel(spec)
                
                kernel = kernel_cache[cache_key]
                ctx.save_for_backward(a, b)
                
                return kernel(a, b)
            
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                a, b = ctx.saved_tensors
                grad_a = grad_b = None
                
                if ctx.needs_input_grad[0]:
                    grad_a = AgentMatmul.apply(grad_output, b.t())
                if ctx.needs_input_grad[1]:
                    grad_b = AgentMatmul.apply(a.t(), grad_output)
                
                return grad_a, grad_b
        
        return AgentMatmul


# Convenience function
def agent_matmul(a: torch.Tensor, b: torch.Tensor, 
                 agent_system: Optional[MultiAgentOrchestrator] = None) -> torch.Tensor:
    """Drop-in replacement for torch.mm with agent optimization."""
    if agent_system is None:
        agent_system = get_global_agent_system()
    
    wrapper = TritonKernelWrapper(agent_system)
    AgentMatmul = wrapper.create_matmul()
    return AgentMatmul.apply(a, b)
```

### Option 4: TorchInductor Extension

Extend TorchInductor to use agent-generated kernels for specific patterns:

```python
# agent_backend/pytorch/inductor_extension.py

from torch._inductor.lowering import lowerings, register_lowering

class InductorAgentExtension:
    """
    Extend TorchInductor with agent-generated kernels.
    
    Hooks into Inductor's lowering system to replace specific
    operations with agent-generated implementations.
    """
    
    def __init__(self, agent_system: MultiAgentOrchestrator):
        self.agent_system = agent_system
        self.registered = False
    
    def register_lowerings(self):
        """Register custom lowerings that use agent-generated kernels."""
        
        if self.registered:
            return
        
        original_mm = lowerings.get(torch.ops.aten.mm.default)
        
        @register_lowering(torch.ops.aten.mm.default, override=True)
        def agent_mm_lowering(a, b):
            if self._should_use_agent(a, b):
                return self._agent_mm(a, b)
            return original_mm(a, b)
        
        self.registered = True
    
    def _should_use_agent(self, *tensors) -> bool:
        """Use agent for large enough tensors."""
        total_elements = sum(t.get_numel() for t in tensors)
        return total_elements > 1024 * 1024  # 1M elements threshold
```

### PyTorch Integration Summary

| Approach | Best For | Complexity | torch.compile Compatible |
|----------|----------|------------|-------------------------|
| **Custom Backend** | Full model optimization | High | Yes (is the backend) |
| **torch.library** | Specific ops | Medium | Yes |
| **Triton Wrapper** | Quick integration | Low | Yes |
| **Inductor Extension** | Augmenting existing compiler | High | Yes |

### PyTorch Usage Examples

```python
# Example 1: Using torch.compile with agent backend
import torch
from agent_backend.pytorch import agent_backend

@torch.compile(backend="agent_backend")
def transformer_block(x, w_q, w_k, w_v, w_o):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    attn = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1) @ v
    return attn @ w_o

# Example 2: Using custom ops directly
from agent_backend.pytorch.custom_ops import AgentOpRegistry

registry = AgentOpRegistry(agent_system)
agent_matmul = registry.register_matmul()

class MyModel(torch.nn.Module):
    def forward(self, x, weight):
        return torch.ops.agent_ops.agent_matmul(x, weight)

# Example 3: Drop-in replacement
from agent_backend.pytorch import agent_matmul

c = agent_matmul(a, b)  # Uses agent-generated kernel
```

---

## Kernel Specification Format

```python
# agent_backend/spec.py

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Dict, Literal
import hashlib
import json

@dataclass
class KernelSpec:
    """Complete specification of a kernel to generate (framework-agnostic)."""
    
    operation: str  # e.g., "matmul", "conv2d", "softmax"
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hardware: Optional['HardwareContext'] = None
    reference_impl: Optional[callable] = None
    framework: Literal["jax", "pytorch", "generic"] = "generic"
    
    def cache_key(self) -> str:
        """Generate unique cache key."""
        key_data = {
            "op": self.operation,
            "in_shapes": [list(s) for s in self.input_shapes],
            "in_dtypes": self.input_dtypes,
            "out_shape": list(self.output_shape),
            "out_dtype": self.output_dtype,
            "hw": self.hardware.name if self.hardware else "generic",
            # Note: framework not in cache key - same kernel works for both
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def to_prompt(self) -> str:
        """Convert spec to natural language for LLM prompt."""
        lines = [f"Operation: {self.operation}", "", "Inputs:"]
        
        for i, (shape, dtype) in enumerate(zip(self.input_shapes, self.input_dtypes)):
            lines.append(f"  - input_{i}: shape={shape}, dtype={dtype}")
        
        lines.extend(["", "Output:", f"  - shape={self.output_shape}, dtype={self.output_dtype}"])
        
        if self.parameters:
            lines.extend(["", "Parameters:"])
            for key, value in self.parameters.items():
                lines.append(f"  - {key}: {value}")
        
        if self.hardware:
            lines.extend(["", "Hardware:", self.hardware.to_prompt_context()])
        
        lines.extend(["", f"Target Framework: {self.framework}"])
        
        return "\n".join(lines)
```

---

## Feedback Loop Design

### Profile-to-Language Translation

Key insight from PRAGMA: LLMs perform better with human-readable suggestions rather than raw metrics.

```python
# jax_agent_backend/feedback.py

class FeedbackTranslator:
    """Translates raw profiling data to actionable natural language."""
    
    def __init__(self, hardware: HardwareContext):
        self.hardware = hardware
    
    def translate(self, profile: 'ProfileResult') -> str:
        """Generate natural language feedback from profile."""
        
        findings = []
        recommendations = []
        
        # Memory bandwidth analysis
        if profile.memory_bandwidth_utilization < 0.7:
            findings.append(
                f"Memory bandwidth utilization is low "
                f"({profile.memory_bandwidth_utilization*100:.0f}% of peak)."
            )
            recommendations.append(
                "Ensure memory accesses are coalesced: consecutive threads "
                "should access consecutive memory addresses."
            )
        
        # Compute utilization analysis
        if profile.compute_utilization < 0.5:
            findings.append(
                f"Compute utilization is low "
                f"({profile.compute_utilization*100:.0f}% of peak). "
                "This kernel is memory-bound."
            )
            recommendations.append(
                "Consider kernel fusion to reuse loaded data for multiple "
                "operations before writing back to global memory."
            )
        
        # Occupancy analysis
        if profile.achieved_occupancy < 0.5 * profile.theoretical_occupancy:
            findings.append(
                f"Achieved occupancy ({profile.achieved_occupancy*100:.0f}%) "
                f"is below theoretical max ({profile.theoretical_occupancy*100:.0f}%)."
            )
            if profile.registers_per_thread > 64:
                recommendations.append(
                    f"High register usage ({profile.registers_per_thread} per thread) "
                    "limits occupancy. Consider reducing tile sizes."
                )
        
        # Format output
        output = "## Performance Analysis\n\n"
        if findings:
            output += "### Findings\n"
            for i, finding in enumerate(findings, 1):
                output += f"{i}. {finding}\n"
        if recommendations:
            output += "\n### Recommendations\n"
            for i, rec in enumerate(recommendations, 1):
                output += f"{i}. {rec}\n"
        
        return output
```

---

## Verification and Correctness

```python
# jax_agent_backend/verification.py

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class VerificationResult:
    """Result of correctness verification."""
    correct: bool
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    test_case_description: Optional[str] = None

class KernelVerifier:
    """Verifies kernel correctness against reference implementation."""
    
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-5):
        self.rtol = rtol
        self.atol = atol
    
    def verify(self, kernel: callable, spec: 'KernelSpec') -> VerificationResult:
        """Run verification test suite."""
        
        test_cases = [
            ("random_normal", self._random_normal_inputs(spec)),
            ("zeros", self._zeros_inputs(spec)),
            ("ones", self._ones_inputs(spec)),
            ("large_values", self._large_inputs(spec)),
        ]
        
        for test_name, inputs in test_cases:
            result = self._run_single_test(kernel, spec, inputs, test_name)
            if not result.correct:
                return result
        
        return VerificationResult(correct=True)
    
    def _run_single_test(self, kernel, spec, inputs, test_name) -> VerificationResult:
        """Run a single test case."""
        
        # Get reference output
        if spec.reference_impl:
            expected = spec.reference_impl(*inputs)
        else:
            expected = self._default_reference(spec, inputs)
        
        # Run kernel
        try:
            actual = kernel(*[jnp.array(x) for x in inputs])
            actual = np.array(actual)
        except Exception as e:
            return VerificationResult(
                correct=False,
                test_case_description=f"{test_name}: execution error: {str(e)}"
            )
        
        # Compare
        if not np.allclose(expected, actual, rtol=self.rtol, atol=self.atol):
            diff = np.abs(expected - actual)
            return VerificationResult(
                correct=False,
                max_abs_error=float(np.max(diff)),
                test_case_description=test_name
            )
        
        return VerificationResult(correct=True)
    
    def _random_normal_inputs(self, spec):
        return [np.random.randn(*shape).astype(dtype) 
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)]
    
    def _zeros_inputs(self, spec):
        return [np.zeros(shape, dtype=dtype) 
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)]
    
    def _ones_inputs(self, spec):
        return [np.ones(shape, dtype=dtype) 
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)]
    
    def _large_inputs(self, spec):
        return [np.full(shape, 1000.0, dtype=dtype) 
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)]
    
    def _default_reference(self, spec, inputs):
        """Default reference for common operations."""
        if spec.operation == "matmul":
            return inputs[0] @ inputs[1]
        elif spec.operation == "softmax":
            x = inputs[0]
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        raise ValueError(f"No reference for {spec.operation}")
```

---

## Memory and Caching

```python
# jax_agent_backend/cache.py

import sqlite3
import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime
import threading

class KernelCache:
    """Persistent cache for generated kernels (SQLite-backed)."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_entries: int = 10000):
        self.cache_dir = cache_dir or Path.home() / ".jax_agent_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "kernels.db"
        self.max_entries = max_entries
        self._local = threading.local()
        self._init_db()
    
    @property
    def _conn(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS kernels (
                    cache_key TEXT PRIMARY KEY,
                    operation TEXT,
                    code TEXT,
                    compiled_data BLOB,
                    speedup REAL,
                    created_at TEXT,
                    last_used TEXT,
                    use_count INTEGER DEFAULT 0
                )
            """)
    
    def get(self, cache_key: str) -> Optional[callable]:
        """Retrieve kernel from cache."""
        cursor = self._conn.execute(
            "SELECT compiled_data FROM kernels WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        
        if row:
            self._conn.execute(
                "UPDATE kernels SET last_used = ?, use_count = use_count + 1 WHERE cache_key = ?",
                (datetime.now().isoformat(), cache_key)
            )
            self._conn.commit()
            return pickle.loads(row[0])
        return None
    
    def put(self, cache_key: str, kernel: callable):
        """Store kernel in cache."""
        self._evict_if_needed()
        
        self._conn.execute(
            """INSERT OR REPLACE INTO kernels 
               (cache_key, compiled_data, created_at, last_used, use_count)
               VALUES (?, ?, ?, ?, 1)""",
            (cache_key, pickle.dumps(kernel), 
             datetime.now().isoformat(), datetime.now().isoformat())
        )
        self._conn.commit()
    
    def _evict_if_needed(self):
        """LRU eviction when cache is full."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM kernels")
        if cursor.fetchone()[0] >= self.max_entries:
            self._conn.execute(
                """DELETE FROM kernels WHERE cache_key IN (
                     SELECT cache_key FROM kernels ORDER BY last_used ASC LIMIT ?
                   )""",
                (self.max_entries // 10,)
            )
```

---

## Benchmarking

### Target Benchmarks

Test your implementation against established benchmarks:

| Benchmark | Focus | Source |
|-----------|-------|--------|
| **KernelBench** | PyTorch→CUDA translation, 250 tasks | Stanford |
| **TritonBench** | Production Triton kernels, 350 tasks | GitHub/PyTorch |
| **FlashInfer-Bench** | LLM serving workloads, 1600 tasks | FlashInfer |

### Evaluation Metrics

```python
@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    
    # Correctness
    compilation_success_rate: float  # % that compile
    correctness_rate: float          # % that produce correct output
    
    # Performance
    speedup_vs_reference: float      # Geometric mean
    speedup_vs_pytorch: float
    speedup_vs_xla: float
    
    # Efficiency
    iterations_to_success: float     # Average iterations needed
    llm_tokens_used: int
    wall_clock_time_s: float
```

---

## Deployment Considerations

### Latency Management

The main challenge is agent generation latency (seconds to minutes):

1. **Aggressive Caching**: Cache by (op, shapes, dtypes, hardware)
2. **Async Generation**: Generate kernels in background while using fallback
3. **Warm-up Phase**: Pre-generate kernels for common operations at startup
4. **Tiered Approach**: Use fast heuristics first, agent for optimization

### Cost Management

LLM API calls are expensive:

```python
@dataclass  
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    @property
    def estimated_cost_usd(self) -> float:
        # Claude Sonnet pricing (example)
        return (self.total_input_tokens * 3 + self.total_output_tokens * 15) / 1_000_000
```

### Fallback Strategy

Always have XLA as fallback:

```python
def get_kernel_with_fallback(self, spec: KernelSpec):
    try:
        return self.agent_system.generate_kernel(spec)
    except (TimeoutError, APIError, GenerationError):
        logger.warning(f"Agent failed for {spec.operation}, using XLA")
        return self.xla_backend.compile(spec)
```

---

## References

### Papers

1. **STARK** (2025): "Strategic Team of Agents for Refining Kernels" - Multi-agent collaboration
2. **Astra** (2025): "A Multi-Agent System for GPU Kernel Performance Optimization"
3. **CUDA-LLM** (2025): "LLMs Can Write Efficient CUDA Kernels" - FSR feedback loop
4. **KernelFalcon** (2026): Meta's deep agent architecture for kernel generation
5. **TritonRL** (2025): Reinforcement learning for Triton kernel generation
6. **LLM4Kernel Survey** (2026): "Towards Automated Kernel Generation in the Era of LLMs"
7. **TorchDynamo** (2022): "TorchDynamo: Python Frame Evaluation"
8. **TorchInductor** (2023): "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation"

### Benchmarks

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) - Stanford (250 tasks)
- [TritonBench](https://github.com/triton-lang/tritonbench) - Triton team (350 tasks)
- [KernelAgent](https://github.com/meta-pytorch/KernelAgent) - Meta
- [TorchBench](https://github.com/pytorch/benchmark) - PyTorch performance benchmarks

### Documentation

- [JAX Pallas Documentation](https://jax.readthedocs.io/en/latest/pallas/)
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
- [TorchInductor Deep Dive](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

---

## Appendix: Quick Start

### Installation

```bash
pip install agent-kernel-backend

# For JAX support
pip install jax[cuda12]

# For PyTorch support
pip install torch triton
```

### JAX Usage

```python
from agent_backend import AgentBackend, AgentBackendConfig
from agent_backend.jax import register_jax_lowerings

# Initialize backend
config = AgentBackendConfig(
    enable_caching=True,
    max_iterations=10,
    target_speedup=1.0,
    fallback_to_xla=True
)
backend = AgentBackend(config)
register_jax_lowerings(backend)

# Now JAX operations use agent-generated kernels
import jax.numpy as jnp

a = jnp.ones((1024, 1024))
b = jnp.ones((1024, 1024))
c = a @ b  # Uses agent-generated matmul kernel
```

### PyTorch Usage

```python
import torch
from agent_backend import AgentBackend, AgentBackendConfig

# Option 1: Use as torch.compile backend
from agent_backend.pytorch import register_compile_backend
register_compile_backend()

@torch.compile(backend="agent_backend")
def my_model(x, weight):
    return x @ weight

# Option 2: Use custom ops directly
from agent_backend.pytorch import AgentOpRegistry

config = AgentBackendConfig()
backend = AgentBackend(config)
registry = AgentOpRegistry(backend.agent_system)

agent_matmul = registry.register_matmul()
c = torch.ops.agent_ops.agent_matmul(a, b)

# Option 3: Drop-in replacement
from agent_backend.pytorch import agent_matmul
c = agent_matmul(a, b)
```

### Unified API (Both Frameworks)

```python
from agent_backend import create_kernel, KernelSpec, HardwareContext

# Create a kernel spec
spec = KernelSpec(
    operation="matmul",
    input_shapes=[(1024, 512), (512, 1024)],
    input_dtypes=["float16", "float16"],
    output_shape=(1024, 1024),
    output_dtype="float16",
    hardware=HardwareContext.from_name("H100-SXM")
)

# Generate kernel (works for both frameworks)
kernel = create_kernel(spec)

# Use with JAX
import jax.numpy as jnp
a_jax = jnp.ones((1024, 512), dtype=jnp.float16)
b_jax = jnp.ones((512, 1024), dtype=jnp.float16)
c_jax = kernel.call_jax(a_jax, b_jax)

# Use with PyTorch
import torch
a_torch = torch.ones(1024, 512, dtype=torch.float16, device='cuda')
b_torch = torch.ones(512, 1024, dtype=torch.float16, device='cuda')
c_torch = kernel.call_pytorch(a_torch, b_torch)
```

### Environment Variables

```bash
# LLM API configuration
export ANTHROPIC_API_KEY="your-key"  # For Claude
export OPENAI_API_KEY="your-key"     # For GPT-4

# Backend configuration
export AGENT_BACKEND_CACHE_DIR="~/.agent_kernel_cache"
export AGENT_BACKEND_MAX_ITERATIONS=10
export AGENT_BACKEND_FALLBACK=true
export AGENT_BACKEND_LOG_LEVEL=INFO
```

---

*Document Version: 2.0*  
*Last Updated: January 2026*  
*Supports: JAX, PyTorch*
