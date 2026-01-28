"""TorchInductor extension for JaxonFlow."""

from __future__ import annotations

import logging
from typing import Any

import torch

# Try to import Inductor internals
# Note: These are private APIs and might change across PyTorch versions
try:
    from torch._inductor import lowerings
    from torch._inductor.lowering import register_lowering
    HAS_INDUCTOR = True
except ImportError:
    HAS_INDUCTOR = False
    lowerings = None
    register_lowering = lambda *args, **kwargs: lambda f: f

from ..dispatch import AgentBackend
from ..spec import KernelSpec

logger = logging.getLogger(__name__)


class InductorAgentExtension:
    """Extension to TorchInductor using agent-generated kernels."""

    def __init__(self, backend: AgentBackend) -> None:
        """Initialize extension.

        Args:
            backend: AgentBackend instance.
        """
        self.backend = backend
        self.registered = False

    def register_lowerings(self) -> None:
        """Register custom lowerings that use agent-generated kernels."""
        if not HAS_INDUCTOR:
            logger.warning("TorchInductor not available, skipping registration")
            return

        if self.registered:
            return

        # Capture original lowering BEFORE we override it to avoid infinite recursion
        original_mm = lowerings.lowerings.get(torch.ops.aten.mm.default) if lowerings else None

        if original_mm is None:
            logger.warning("Could not capture original mm lowering, skipping registration")
            return

        @register_lowering(torch.ops.aten.mm.default, override=True)
        def agent_mm_lowering(a: Any, b: Any, layout: Any = None) -> Any:
            # 'a' and 'b' here are Inductor IR nodes (TensorBox/StorageBox), not real tensors
            # So we can't easily check values, but we can check shapes if static

            # For this Phase 2 implementation, we log and defer to the original lowering.
            # In a full implementation, we would generate Inductor IR that calls our kernel.

            logger.debug("JaxonFlow intercepted mm in Inductor, delegating to original")

            # Delegate to the captured original lowering (avoids infinite recursion)
            return original_mm(a, b)

        self.registered = True
        logger.info("Registered JaxonFlow Inductor lowerings")

    def _should_use_agent(self, *tensors: Any) -> bool:
        """Heuristic to decide whether to use agent optimization."""
        # This would inspect IR node metadata
        return True
