"""Warmup system for JaxonFlow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .spec import KernelSpec

if TYPE_CHECKING:
    from .dispatch import AgentBackend

logger = logging.getLogger(__name__)


class WarmupSystem:
    """Pre-generates kernels for common operations and shapes.

    This reduces latency specifically for the first inference run by 
    ensuring common kernels are already in the cache.
    """

    def __init__(self, backend: AgentBackend) -> None:
        """Initialize the warmup system.

        Args:
            backend: The agent backend to warm up.
        """
        self.backend = backend

    def warmup(self, verbose: bool = True) -> None:
        """Run the standard warmup routine.

        Triggers generation for a set of common operations and shapes
        typical in transformer models.

        Args:
            verbose: Whether to log progress.
        """
        if verbose:
            logger.info("Starting JaxonFlow warmup...")

        specs = self._get_common_specs()
        
        success_count = 0
        for i, spec in enumerate(specs):
            try:
                if verbose:
                    logger.info(f"Warming up [{i+1}/{len(specs)}]: {spec.operation} {spec.input_shapes}")
                
                # We use get_or_generate_kernel which handles caching
                self.backend.get_or_generate_kernel(spec)
                success_count += 1
            except Exception as e:
                logger.warning(f"Warmup failed for {spec.operation}: {e}")

        if verbose:
            logger.info(f"Warmup complete. {success_count}/{len(specs)} kernels ready.")

    def _get_common_specs(self) -> list[KernelSpec]:
        """Generate a list of specs to warm up."""
        specs = []
        
        # Common hidden sizes in LLMs (BERT, GPT, Llama)
        hidden_sizes = [768, 1024, 2048, 4096]
        batch_sizes = [1, 4, 8, 32]
        dtypes = ["float32", "float16"]
        
        # 1. Matrix Multiplication (Linear layers)
        for b in batch_sizes:
            for h in hidden_sizes:
                for dtype in dtypes:
                    # Generic linear layer: [B, H] @ [H, H] -> [B, H]
                    if b * h * h < 1e9: # Skip huge ones to save time/cost during warmup
                        specs.append(KernelSpec(
                            operation="matmul",
                            input_shapes=[(b, h), (h, h)],
                            input_dtypes=[dtype, dtype],
                            output_shape=(b, h),
                            output_dtype=dtype
                        ))
        
        # 2. Softmax (Attention scores)
        # [B, H, S, S] is typical, usually last dim is relevant
        seq_lens = [128, 512, 1024]
        for s in seq_lens:
            for dtype in dtypes:
                specs.append(KernelSpec(
                    operation="softmax",
                    input_shapes=[(1, 12, s, s)], # Example attention scores shape
                    input_dtypes=[dtype],
                    output_shape=(1, 12, s, s),
                    output_dtype=dtype,
                    parameters={"axis": -1}
                ))

        # 3. LayerNorm / RMSNorm (simplified as generic reduction/elementwise)
        # Often implemented as custom kernels, but here we just warmup generic ops
        # if they were supported as primitives.
        
        return specs
