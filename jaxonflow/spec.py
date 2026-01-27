"""Kernel specification and compiled kernel wrapper for JaxonFlow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from .hardware import HardwareContext


@dataclass
class KernelSpec:
    """Complete specification of a kernel to generate.

    This is the framework-agnostic specification that captures everything
    needed to generate an optimized GPU kernel. Both JAX primitives and
    PyTorch operations map to this format.

    Attributes:
        operation: The operation type (e.g., "matmul", "conv2d", "softmax").
        input_shapes: List of input tensor shapes.
        input_dtypes: List of input tensor dtypes as strings.
        output_shape: Output tensor shape.
        output_dtype: Output tensor dtype as string.
        parameters: Operation-specific parameters (e.g., stride for conv).
        hardware: Target hardware context for optimization.
        reference_impl: Optional reference implementation for verification.
        framework: Source framework ("jax", "pytorch", or "generic").
    """

    operation: str
    input_shapes: list[tuple[int, ...]]
    input_dtypes: list[str]
    output_shape: tuple[int, ...]
    output_dtype: str
    parameters: dict[str, Any] = field(default_factory=dict)
    hardware: HardwareContext | None = None
    reference_impl: Callable[..., Any] | None = None
    framework: Literal["jax", "pytorch", "generic"] = "generic"

    def cache_key(self) -> str:
        """Generate unique cache key for this kernel specification.

        The cache key is a SHA256 hash of the operation signature,
        input/output shapes and dtypes, and hardware target. The framework
        is intentionally excluded since the same Triton kernel works for both.

        Returns:
            16-character hexadecimal cache key.
        """
        key_data = {
            "op": self.operation,
            "in_shapes": [list(s) for s in self.input_shapes],
            "in_dtypes": self.input_dtypes,
            "out_shape": list(self.output_shape),
            "out_dtype": self.output_dtype,
            "params": self._serialize_parameters(),
            "hw": self.hardware.name if self.hardware else "generic",
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]

    def _serialize_parameters(self) -> dict[str, Any]:
        """Serialize parameters for hashing, handling non-JSON types."""
        result = {}
        for key, value in self.parameters.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            elif isinstance(value, dict):
                result[key] = value
            else:
                # Convert to string representation for other types
                result[key] = str(value)
        return result

    def to_prompt(self) -> str:
        """Convert spec to natural language for LLM prompt.

        Returns:
            Multi-line string describing the kernel specification
            in a format suitable for LLM prompts.
        """
        lines = [
            f"Operation: {self.operation}",
            "",
            "Inputs:",
        ]

        for i, (shape, dtype) in enumerate(zip(self.input_shapes, self.input_dtypes)):
            lines.append(f"  - input_{i}: shape={shape}, dtype={dtype}")

        lines.extend([
            "",
            "Output:",
            f"  - shape={self.output_shape}, dtype={self.output_dtype}",
        ])

        if self.parameters:
            lines.extend(["", "Parameters:"])
            for key, value in self.parameters.items():
                lines.append(f"  - {key}: {value}")

        if self.hardware:
            lines.extend(["", "Hardware Context:", self.hardware.to_prompt_context()])

        lines.extend(["", f"Target Framework: {self.framework}"])

        return "\n".join(lines)

    def total_input_elements(self) -> int:
        """Calculate total number of input elements."""
        total = 0
        for shape in self.input_shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total += elements
        return total

    def total_output_elements(self) -> int:
        """Calculate total number of output elements."""
        elements = 1
        for dim in self.output_shape:
            elements *= dim
        return elements

    def estimated_flops(self) -> int | None:
        """Estimate FLOPs for common operations.

        Returns:
            Estimated FLOPs or None if unknown operation.
        """
        if self.operation == "matmul" and len(self.input_shapes) == 2:
            # C[M,N] = A[M,K] @ B[K,N] -> 2*M*N*K FLOPs
            m, k = self.input_shapes[0]
            _, n = self.input_shapes[1]
            return 2 * m * n * k

        if self.operation == "softmax":
            # Softmax: exp + sum + div per element
            return self.total_output_elements() * 5

        if self.operation in ("relu", "gelu", "silu"):
            return self.total_output_elements()

        if self.operation in ("add", "mul", "sub", "div"):
            return self.total_output_elements()

        return None

    def is_memory_bound_estimate(self) -> bool:
        """Estimate if this operation is likely memory-bound.

        Uses arithmetic intensity (FLOPs/byte) to estimate.
        Operations with low arithmetic intensity are memory-bound.

        Returns:
            True if likely memory-bound, False if likely compute-bound.
        """
        flops = self.estimated_flops()
        if flops is None:
            return True  # Assume memory-bound if unknown

        # Estimate bytes moved (simplified)
        bytes_per_element = 4  # Assume fp32
        if "float16" in self.input_dtypes or "fp16" in self.input_dtypes:
            bytes_per_element = 2
        elif "bfloat16" in self.input_dtypes or "bf16" in self.input_dtypes:
            bytes_per_element = 2

        total_bytes = (self.total_input_elements() + self.total_output_elements()) * bytes_per_element
        arithmetic_intensity = flops / total_bytes

        # Rough threshold: < 50 FLOPs/byte is typically memory-bound on modern GPUs
        return arithmetic_intensity < 50


@dataclass
class CompiledKernel:
    """Wrapper for a compiled Triton kernel.

    Holds the generated code, compiled callable, and metadata
    about the kernel's performance and generation history.
    """

    spec: KernelSpec
    code: str
    callable: Callable[..., Any] | None = None

    # Performance metadata
    speedup_vs_reference: float | None = None
    execution_time_us: float | None = None

    # Generation metadata
    iterations_to_generate: int = 0
    total_tokens_used: int = 0
    generation_time_s: float = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled kernel."""
        if self.callable is None:
            raise RuntimeError("Kernel has not been compiled yet")
        return self.callable(*args, **kwargs)

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for cache storage (excluding callable)."""
        return {
            "spec_cache_key": self.spec.cache_key(),
            "operation": self.spec.operation,
            "code": self.code,
            "speedup_vs_reference": self.speedup_vs_reference,
            "execution_time_us": self.execution_time_us,
            "iterations_to_generate": self.iterations_to_generate,
            "total_tokens_used": self.total_tokens_used,
            "generation_time_s": self.generation_time_s,
        }


@dataclass
class CompilationResult:
    """Result of kernel compilation."""

    success: bool
    kernel: CompiledKernel | None = None
    error: str | None = None
    error_line: int | None = None


@dataclass
class VerificationResult:
    """Result of kernel correctness verification."""

    correct: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    test_case_description: str | None = None
    expected: Any | None = None
    actual: Any | None = None


@dataclass
class ProfileResult:
    """Result of kernel performance profiling."""

    # Timing
    execution_time_us: float
    speedup_vs_reference: float

    # Utilization (0.0 to 1.0)
    memory_bandwidth_utilization: float | None = None
    compute_utilization: float | None = None

    # Occupancy
    achieved_occupancy: float | None = None
    theoretical_occupancy: float | None = None

    # Resource usage
    registers_per_thread: int | None = None
    shared_memory_bytes: int | None = None

    # Bottleneck analysis
    primary_bottleneck: Literal["memory", "compute", "latency"] | None = None
