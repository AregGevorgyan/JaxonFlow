"""Main entry point for JaxonFlow agent backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from .cache import KernelCache
from .hardware import HardwareContext
from .agents.orchestrator import MultiAgentOrchestrator
from .config import CacheConfig
from .spec import CompiledKernel, KernelSpec

if TYPE_CHECKING:
    import jax
    import torch


logger = logging.getLogger(__name__)


@dataclass
class AgentBackendConfig:
    """Configuration for the agent backend."""

    enable_caching: bool = True
    max_iterations: int = 10
    target_speedup: float = 1.0  # vs reference implementation
    hardware_target: str = "auto"  # auto-detect or specify
    fallback_to_xla: bool = True  # use XLA if agent fails
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    # Initialize from file or env vars could go here


class AgentBackend:
    """Main backend class that intercepts operations and manages kernel generation."""

    def __init__(self, config: AgentBackendConfig = AgentBackendConfig()) -> None:
        """Initialize the agent backend.

        Args:
            config: Backend configuration.
        """
        self.config = config
        
        # Initialize sub-components
        self.hardware_context = self._init_hardware()
        self.kernel_cache = KernelCache(self.config.cache_config)
        self.agent_system = MultiAgentOrchestrator(self.config)
        
        logger.info(f"AgentBackend initialized for {self.hardware_context.name}")

    def _init_hardware(self) -> HardwareContext:
        """Initialize hardware context."""
        if self.config.hardware_target == "auto":
            return HardwareContext.detect()
        else:
            return HardwareContext.from_name(self.config.hardware_target)

    def get_or_generate_kernel(self, spec: KernelSpec) -> CompiledKernel:
        """Check cache or generate new kernel for the given specification.

        Args:
            spec: Kernel specification.

        Returns:
            Compiled kernel.

        Raises:
            RuntimeError: If kernel generation fails and no fallback is available.
        """
        # 1. Update spec with hardware context if missing
        if spec.hardware is None:
            spec.hardware = self.hardware_context
            
        cache_key = spec.cache_key()

        # 2. Check cache
        if self.config.enable_caching:
            cached_entry = self.kernel_cache.get(cache_key)
            if cached_entry:
                logger.debug(f"Cache hit for {spec.operation}")
                # Reconstruct CompiledKernel from cache entry
                # Note: The callable needs to be re-compiled/loaded from code
                # For now, we simulate this or assume we can exec() the code
                return self._rehydrate_kernel(cached_entry, spec)

        # 3. Generate via agent system
        logger.info(f"Generating kernel for {spec.operation}...")
        kernel = self.agent_system.generate_kernel(spec)

        if kernel is not None:
            # Cache the result
            if self.config.enable_caching:
                self.kernel_cache.put_kernel(kernel)
            return kernel

        # 4. Fallback (Stubbed)
        # In a real system, we might return a wrapper that calls the original framework native op
        raise RuntimeError(f"Failed to generate kernel for {spec.operation}")

    def _rehydrate_kernel(self, entry: Any, spec: KernelSpec) -> CompiledKernel:
        """Reconstruct a CompiledKernel from a cache entry."""
        # This requires compiling the code.
        # For our mock/no-GPU environment:
        
        def dummy_callable(*args, **kwargs):
             if spec.reference_impl:
                 return spec.reference_impl(*args, **kwargs)
             return args[0] # Dummy

        return CompiledKernel(
            spec=spec,
            code=entry.code,
            callable=dummy_callable,
            # Restore metadata from entry.metadata dict
        )
