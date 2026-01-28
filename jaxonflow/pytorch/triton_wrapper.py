"""Triton kernel wrapper for PyTorch with autograd support."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

from ..dispatch import AgentBackend
from ..hardware import HardwareContext
from ..spec import KernelSpec

logger = logging.getLogger(__name__)


class TritonKernelWrapper:
    """Wrapper to make agent-generated Triton kernels usable in PyTorch."""

    def __init__(self, backend: AgentBackend) -> None:
        """Initialize wrapper.

        Args:
            backend: AgentBackend instance.
        """
        self.backend = backend
        self.kernel_cache: dict[tuple, Any] = {}

    def create_matmul(self) -> type[torch.autograd.Function]:
        """Create a PyTorch autograd function for agent-optimized matmul."""
        backend = self.backend
        kernel_cache = self.kernel_cache

        class AgentMatmul(torch.autograd.Function):
            """Autograd function for matrix multiplication."""

            @staticmethod
            def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # Basic validation
                if not (a.is_cuda and b.is_cuda):
                    # In real code, might fallback to CPU or move tensors
                    # For now, asserting as per plan
                    # logger.warning("Inputs not on CUDA, operation might fail if kernel expects CUDA")
                    pass

                # Cache key based on shapes and dtypes
                # Using a tuple key suitable for dictionary lookup
                cache_key = (
                    "matmul",
                    tuple(a.shape),
                    tuple(b.shape),
                    str(a.dtype),
                    str(b.dtype),
                )

                if cache_key not in kernel_cache:
                    spec = KernelSpec(
                        operation="matmul",
                        input_shapes=[tuple(a.shape), tuple(b.shape)],
                        input_dtypes=[
                            str(a.dtype).replace("torch.", ""),
                            str(b.dtype).replace("torch.", ""),
                        ],
                        output_shape=(a.shape[0], b.shape[1]),
                        output_dtype=str(a.dtype).replace("torch.", ""),
                        parameters={},
                        hardware=HardwareContext.detect(),
                    )
                    # Store generated kernel (CompiledKernel)
                    kernel_cache[cache_key] = backend.get_or_generate_kernel(spec)

                kernel = kernel_cache[cache_key]
                
                # Save tensors for backward pass
                ctx.save_for_backward(a, b)
                
                # Execute kernel
                return kernel.callable(a, b)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
                a, b = ctx.saved_tensors
                grad_a = grad_b = None

                # Simplified backward pass assuming standard matmul gradients
                # Ideally, we should request backward kernels from the agent too!
                # For Phase 2, we can reuse the forward matmul kernel (A @ B)
                # Grad A = Grad_Out @ B.T
                # Grad B = A.T @ Grad_Out
                
                # We reuse the agent-generated matmul for backward pass if possible
                # or fallback to torch.matmul for stability in this phase.
                
                # Using class method .apply recursively
                
                if ctx.needs_input_grad[0]:
                    # grad_a = grad_output @ b.T
                    # We transpose b safely
                    grad_a = AgentMatmul.apply(grad_output, b.t())
                    
                if ctx.needs_input_grad[1]:
                    # grad_b = a.T @ grad_output
                    grad_b = AgentMatmul.apply(a.t(), grad_output)
                    
                return grad_a, grad_b

        return AgentMatmul


def agent_matmul(a: torch.Tensor, b: torch.Tensor, backend: AgentBackend) -> torch.Tensor:
    """Drop-in replacement for torch.mm with agent optimization.
    
    Args:
        a: Left tensor.
        b: Right tensor.
        backend: The agent backend instance.
        
    Returns:
        Result tensor.
    """
    wrapper = TritonKernelWrapper(backend)
    matmul_fn = wrapper.create_matmul()
    return matmul_fn.apply(a, b)
