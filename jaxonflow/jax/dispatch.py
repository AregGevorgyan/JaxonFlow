"""JAX-specific dispatch logic for JaxonFlow."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from jax import core
from jax._src import dtypes

from ..dispatch import AgentBackend
from ..hardware import HardwareContext
from ..spec import KernelSpec

logger = logging.getLogger(__name__)


class JaxDispatcher:
    """Handles logic for converting JAX primitives to KernelSpecs and dispatching."""

    def __init__(self, backend: AgentBackend) -> None:
        """Initialize the JAX dispatcher.

        Args:
            backend: The generic agent backend instance.
        """
        self.backend = backend

    def get_kernel_spec(
        self,
        primitive_name: str,
        input_shapes: Sequence[tuple[int, ...]],
        input_dtypes: Sequence[str],
        output_shape: tuple[int, ...],
        output_dtype: str,
        params: dict[str, Any],
    ) -> KernelSpec:
        """Create a KernelSpec from JAX primitive info.

        Args:
            primitive_name: Name of the JAX primitive (e.g., 'dot_general').
            input_shapes: List of input shapes.
            input_dtypes: List of input dtype strings.
            output_shape: Output shape.
            output_dtype: Output dtype string.
            params: Dictionary of primitive parameters.

        Returns:
            A populated KernelSpec.
        """
        return KernelSpec(
            operation=primitive_name,
            input_shapes=list(input_shapes),
            input_dtypes=list(input_dtypes),
            output_shape=output_shape,
            output_dtype=output_dtype,
            parameters=params,
            hardware=self.backend.hardware_context,
            # We can populate reference_impl if we have a way to get it from JAX
            # For now we leave it None or handle it in specific lowerings
        )

    def dispatch(
        self,
        primitive_name: str,
        input_shapes: Sequence[tuple[int, ...]],
        input_dtypes: Sequence[str],
        output_shape: tuple[int, ...],
        output_dtype: str,
        params: dict[str, Any],
    ) -> Any:
        """Main dispatch method.

        Args:
            primitive_name: Name of the primitive.
            input_shapes: Input shapes.
            input_dtypes: Input dtypes.
            output_shape: Output shape.
            output_dtype: Output dtype.
            params: Parameters.

        Returns:
            The compiled kernel (callable).
        """
        spec = self.get_kernel_spec(
            primitive_name,
            input_shapes,
            input_dtypes,
            output_shape,
            output_dtype,
            params,
        )
        
        return self.backend.get_or_generate_kernel(spec)
