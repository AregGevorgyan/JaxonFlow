"""Custom lowering rules for JaxonFlow JAX integration."""

from __future__ import annotations

import logging
from typing import Any

import jax
from jax import lax
from jax._src.interpreters import mlir

from .dispatch import JaxDispatcher

logger = logging.getLogger(__name__)

# Store original lowerings so we can fall back to them
_original_lowerings: dict[str, Any] = {}


def register_agent_lowerings(dispatcher: JaxDispatcher) -> None:
    """Register agent-based lowerings for high-value primitives.

    Args:
        dispatcher: The JAX dispatcher instance.
    """

    platform = "cuda"

    # For dot_general: capture the original lowering before overriding
    if hasattr(lax, 'dot_general_p'):
        # Capture original lowering rule for fallback
        original_rule = mlir._lowerings.get(lax.dot_general_p, {}).get(platform)
        _original_lowerings['dot_general'] = original_rule

        @mlir.register_lowering(lax.dot_general_p, platform=platform)
        def dot_general_agent_lowering(ctx, lhs, rhs, *, dimension_numbers,
                                        precision, preferred_element_type):

            logger.info("Intercepted dot_general in JaxonFlow lowering")

            # In a full implementation, we would:
            # 1. Extract shapes/dtypes from ctx.avals_in / ctx.avals_out
            # 2. Create a KernelSpec via dispatcher.get_kernel_spec(...)
            # 3. Generate/retrieve a kernel via dispatcher.dispatch(...)
            # 4. Emit a custom_call to our Triton kernel

            # For now, fall back to the original XLA lowering
            original = _original_lowerings.get('dot_general')
            if original is not None:
                return original(
                    ctx, lhs, rhs,
                    dimension_numbers=dimension_numbers,
                    precision=precision,
                    preferred_element_type=preferred_element_type
                )

            # If we couldn't capture the original, use lax.dot_general via lower_fun
            # lower_fun expects a Python function, not a primitive
            fallback = mlir.lower_fun(
                lambda lhs, rhs: lax.dot_general(
                    lhs, rhs,
                    dimension_numbers=dimension_numbers,
                    precision=precision,
                    preferred_element_type=preferred_element_type,
                ),
                multiple_results=False,
            )
            return fallback(ctx, lhs, rhs)

    logger.info(f"Registered JaxonFlow lowerings for platform: {platform}")
