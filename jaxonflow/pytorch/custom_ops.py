"""Custom PyTorch operators for JaxonFlow."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import library

from ..dispatch import AgentBackend
from ..hardware import HardwareContext
from ..spec import KernelSpec

logger = logging.getLogger(__name__)

# Create a library for agent-generated ops
# Using "jaxonflow" as the namespace
agent_lib = library.Library("jaxonflow", "DEF")


class AgentOpRegistry:
    """Registry for agent-generated custom operators."""

    def __init__(self, backend: AgentBackend) -> None:
        """Initialize registry.

        Args:
            backend: AgentBackend instance.
        """
        self.backend = backend
        self.registered_ops: dict[str, bool] = {}

    def register_matmul(self, name: str = "agent_matmul") -> Any:
        """Register an agent-optimized matmul.

        Args:
            name: Name of the operator.

        Returns:
            The registered operator callable.
        """
        # Define the operator schema
        # jaxonflow::agent_matmul(Tensor a, Tensor b) -> Tensor
        full_name = f"jaxonflow::{name}"
        if name in self.registered_ops:
             return getattr(torch.ops.jaxonflow, name)

        agent_lib.define(f"{name}(Tensor a, Tensor b) -> Tensor")

        @library.impl(agent_lib, name, "CUDA")
        def agent_matmul_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # Create spec derived from input tensors
            spec = KernelSpec(
                operation="matmul",
                input_shapes=[tuple(a.shape), tuple(b.shape)],
                input_dtypes=[str(a.dtype).replace("torch.", ""), 
                             str(b.dtype).replace("torch.", "")],
                output_shape=(a.shape[0], b.shape[1]),
                output_dtype=str(a.dtype).replace("torch.", ""),
                parameters={},
                hardware=HardwareContext.detect(),
                # framework="pytorch", # Not strictly needed if generic, but good for context
            )

            # Get kernel from backend
            kernel = self.backend.get_or_generate_kernel(spec)
            
            # Execute kernel safely
            # Note: kernel.callable might need specific args matching Triton kernel
            # but here we assume the backend returns a callable that accepts tensors
            # or the generic wrapper adapter handles it.
            return kernel.callable(a, b)

        @library.impl(agent_lib, name, "CPU")
        def agent_matmul_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
             # Fallback to standard PyTorch implementation
            return torch.mm(a, b)

        self.registered_ops[name] = True
        logger.info(f"Registered custom op: {full_name}")
        
        # Return the op so user can use it
        return getattr(torch.ops.jaxonflow, name)
