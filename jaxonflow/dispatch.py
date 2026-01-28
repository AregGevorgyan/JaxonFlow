"""Main entry point for JaxonFlow agent backend."""

from __future__ import annotations

import logging
from typing import Any

from .cache import CacheEntry, KernelCache
from .config import AgentBackendConfig
from .hardware import HardwareContext
from .agents.orchestrator import MultiAgentOrchestrator
from .exceptions import FallbackError, KernelGenerationError
from .spec import CompiledKernel, KernelSpec

logger = logging.getLogger(__name__)


class AgentBackend:
    """Main backend class that intercepts operations and manages kernel generation.

    This is the primary entry point for JaxonFlow. It manages hardware detection,
    kernel caching, and delegates kernel generation to the multi-agent orchestrator.

    Example:
        config = AgentBackendConfig()
        backend = AgentBackend(config)

        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(1024, 512), (512, 1024)],
            input_dtypes=["float32", "float32"],
            output_shape=(1024, 1024),
            output_dtype="float32",
        )

        kernel = backend.get_or_generate_kernel(spec)
    """

    def __init__(self, config: AgentBackendConfig | None = None) -> None:
        """Initialize the agent backend.

        Args:
            config: Backend configuration. Uses defaults if not provided.
        """
        self.config = config or AgentBackendConfig()

        # Initialize sub-components
        self.hardware_context = self._init_hardware()
        self.kernel_cache = KernelCache(self.config.cache)
        self.agent_system = MultiAgentOrchestrator(self.config)

        logger.info(f"AgentBackend initialized for {self.hardware_context.name}")

    def _init_hardware(self) -> HardwareContext:
        """Initialize hardware context based on configuration."""
        if self.config.hardware_target == "auto":
            return HardwareContext.detect()
        return HardwareContext.from_name(self.config.hardware_target)

    def get_or_generate_kernel(self, spec: KernelSpec) -> CompiledKernel:
        """Check cache or generate new kernel for the given specification.

        Args:
            spec: Kernel specification.

        Returns:
            Compiled kernel.

        Raises:
            KernelGenerationError: If kernel generation fails and no fallback
                is available.
        """
        # Inject hardware context if missing
        if spec.hardware is None:
            spec.hardware = self.hardware_context

        cache_key = spec.cache_key()

        # Check cache first
        if self.config.cache.enabled:
            cached_entry = self.kernel_cache.get(cache_key)
            if cached_entry is not None:
                logger.debug(f"Cache hit for {spec.operation} ({cache_key})")
                return self._rehydrate_kernel(cached_entry, spec)

        # Generate via agent system
        logger.info(f"Cache miss for {spec.operation}, generating kernel...")
        kernel = self.agent_system.generate_kernel(spec)

        if kernel is not None:
            # Cache using the request spec's cache_key (with hardware injected)
            # to ensure lookup/store consistency
            if self.config.cache.enabled:
                self.kernel_cache.put(
                    cache_key=cache_key,
                    operation=spec.operation,
                    code=kernel.code,
                    metadata=kernel.to_cache_dict(),
                )
            return kernel

        # Fallback
        if self.config.fallback_to_xla or self.config.fallback_to_inductor:
            logger.warning(
                f"Agent generation failed for {spec.operation}, "
                f"would fall back to native compiler"
            )
            return self._fallback_kernel(spec)

        raise KernelGenerationError(
            f"Failed to generate kernel for {spec.operation}",
            operation=spec.operation,
            iterations=self.config.agent.max_iterations,
        )

    def _rehydrate_kernel(self, entry: CacheEntry, spec: KernelSpec) -> CompiledKernel:
        """Reconstruct a CompiledKernel from a cache entry.

        In a GPU environment, this would recompile the cached Triton code.
        Without a GPU, it wraps the code with a reference implementation fallback.

        Args:
            entry: The cache entry with stored code and metadata.
            spec: The kernel specification.

        Returns:
            A CompiledKernel with a callable.
        """
        callable_fn = self._make_callable_from_code(entry.code, spec)

        return CompiledKernel(
            spec=spec,
            code=entry.code,
            callable=callable_fn,
            speedup_vs_reference=entry.metadata.get("speedup_vs_reference"),
            execution_time_us=entry.metadata.get("execution_time_us"),
            iterations_to_generate=entry.metadata.get("iterations_to_generate", 0),
            total_tokens_used=entry.metadata.get("total_tokens_used", 0),
            generation_time_s=entry.metadata.get("generation_time_s", 0.0),
        )

    def _make_callable_from_code(
        self, code: str, spec: KernelSpec
    ) -> Any:
        """Attempt to create a callable from generated Triton code.

        This tries to compile the code in a sandboxed namespace. If Triton
        is not available or compilation fails, it falls back to the
        reference implementation.

        Args:
            code: The generated kernel code string.
            spec: The kernel specification (used for reference fallback).

        Returns:
            A callable that executes the kernel.
        """
        # Try to exec the code and extract the wrapper function
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace)  # noqa: S102

            # Look for a wrapper function (typically named after the operation)
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name != "triton":
                    # Check if it looks like a wrapper (not the @triton.jit kernel itself)
                    if hasattr(obj, "__wrapped__") or "wrapper" in name.lower():
                        return obj

            # If we found a triton kernel but no wrapper, the code needs GPU
            # to actually run. Fall through to reference impl.
        except Exception as e:
            logger.debug(f"Could not exec cached kernel code: {e}")

        # Fallback: use reference implementation if available
        if spec.reference_impl is not None:
            return spec.reference_impl

        # Last resort: identity-like function
        def _noop_kernel(*args: Any, **kwargs: Any) -> Any:
            logger.warning(
                f"Using no-op kernel for {spec.operation} "
                f"(no GPU/Triton and no reference impl)"
            )
            return args[0] if args else None

        return _noop_kernel

    def _fallback_kernel(self, spec: KernelSpec) -> CompiledKernel:
        """Create a fallback kernel using reference implementation.

        Args:
            spec: The kernel specification.

        Returns:
            A CompiledKernel that uses the reference implementation.

        Raises:
            FallbackError: If no reference implementation is available.
        """
        if spec.reference_impl is not None:
            return CompiledKernel(
                spec=spec,
                code="# Fallback to reference implementation",
                callable=spec.reference_impl,
            )

        raise FallbackError(
            f"No fallback available for {spec.operation}: "
            f"native compiler integration not yet implemented"
        )

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = self.kernel_cache.stats
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "total_entries": stats.total_entries,
            "evictions": stats.evictions,
        }

    @property
    def llm_usage(self) -> dict[str, Any]:
        """Get LLM usage statistics from the orchestrator."""
        return self.agent_system.get_usage_stats()
