"""Custom lowering rules for JaxonFlow JAX integration."""

from __future__ import annotations

import logging
from typing import Any

import jax
from jax import lax
from jax._src.interpreters import mlir

from .dispatch import JaxDispatcher

logger = logging.getLogger(__name__)


def register_agent_lowerings(dispatcher: JaxDispatcher) -> None:
    """Register agent-based lowerings for high-value primitives.
    
    Args:
        dispatcher: The JAX dispatcher instance.
    """
    
    # We register for the 'cuda' platform specifically as we target GPUs
    platform = "cuda"
    
    # Example for dot_general
    if hasattr(lax, 'dot_general_p'):
        @mlir.register_lowering(lax.dot_general_p, platform=platform)
        def dot_general_agent_lowering(ctx, lhs, rhs, *, dimension_numbers, 
                                        precision, preferred_element_type):
            
            # This implementation assumes we are in an MLIR lowering context.
            # To properly call out to our agent system, we need to extract info
            # from the context 'ctx'.
            
            # Note: `ctx.avals_in` and `ctx.avals_out` provide shape/dtype info.
            
            # input_shapes = [val.type.shape for val in [lhs, rhs]] # Approx
            # input_dtypes = [val.type.element_type for val in [lhs, rhs]] # Approx
            
            # Using our dispatcher to get/generate the kernel
            # Since lowering happens at compile time, we can afford the agent latency 
            # (or cache lookup).
            
            # However, inserting the call into the MLIR module is complex without
            # specific MLIR builder tools (ir.Module, etc.) or Pallas.
            
            # For this phase, we will log the interception and fall back to 
            # standard XLA lowering or a mock behavior for implementation completeness.
            
            logger.info("Intercepted dot_general in JaxonFlow lowering")

            # Fallback to standard lowering for now to ensure we don't break execution
            # In a full implementation, we would emit a custom_call to our kernel.
            return mlir.lower_fun(lax.dot_general_p, platform)(
                ctx, lhs, rhs, 
                dimension_numbers=dimension_numbers, 
                precision=precision, 
                preferred_element_type=preferred_element_type
            )

    logger.info(f"Registered JaxonFlow lowerings for platform: {platform}")
