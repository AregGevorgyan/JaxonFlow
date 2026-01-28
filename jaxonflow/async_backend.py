"""Async kernel generation backend for JaxonFlow."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dispatch import AgentBackend
    from .spec import CompiledKernel, KernelSpec

logger = logging.getLogger(__name__)


class AsyncKernelGenerator:
    """Handles asynchronous kernel generation.

    This class provides a non-blocking interface for kernel generation,
    allowing the application to continue using fallback implementations
    while the optimized kernel is being generated in the background.
    """

    def __init__(self, backend: AgentBackend, max_workers: int = 1) -> None:
        """Initialize the async generator.

        Args:
            backend: The synchronous agent backend to use for generation.
            max_workers: Number of background threads for generation.
        """
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jaxonflow_gen")
        self._pending_futures: dict[str, Future[CompiledKernel]] = {}

    def submit(self, spec: KernelSpec) -> Future[CompiledKernel]:
        """Submit a kernel generation task.

        If a task for the same kernel (same cache key) is already pending,
        returns the existing future.

        Args:
            spec: The kernel specification to generate.

        Returns:
            A Future object representing the pending generation.
        """
        cache_key = spec.cache_key()

        # Check if already pending
        if cache_key in self._pending_futures:
            future = self._pending_futures[cache_key]
            if not future.done():
                return future
            # If done, remove from pending and potentially re-submit if failed?
            # ideally calling backend.get_or_generate_kernel would hit cache if it succeeded.
            # let's just clean up completed futures.
            del self._pending_futures[cache_key]

        # Submit new task
        future = self._executor.submit(self._generate_task, spec)
        self._pending_futures[cache_key] = future
        
        # Add cleanup callback
        future.add_done_callback(lambda f: self._cleanup_future(cache_key))
        
        return future

    def _generate_task(self, spec: KernelSpec) -> CompiledKernel:
        """The actual generation task running in the thread."""
        try:
            # This calls the synchronous backend, which handles caching, agents, etc.
            return self.backend.get_or_generate_kernel(spec)
        except Exception as e:
            logger.error(f"Async generation failed for {spec.operation}: {e}")
            raise

    def _cleanup_future(self, cache_key: str) -> None:
        """Remove future from pending dict when done."""
        # pop if it's still there (might have been replaced?)
        # safe enough to just check existence
        if cache_key in self._pending_futures and self._pending_futures[cache_key].done():
            del self._pending_futures[cache_key]

    def get_result_if_ready(self, spec: KernelSpec) -> CompiledKernel | None:
        """Check if generation is complete and return result if so.

        Args:
            spec: The kernel specification.

        Returns:
            The CompiledKernel if ready, None otherwise.
        """
        cache_key = spec.cache_key()
        
        # Method 1: Check synchronous cache (fastest if already done and persisted)
        # We can try to get from backend's cache directly without triggering generation
        cached = self.backend.kernel_cache.get(cache_key)
        if cached is not None:
             return self.backend._rehydrate_kernel(cached, spec)

        # Method 2: Check pending futures
        if cache_key in self._pending_futures:
            future = self._pending_futures[cache_key]
            if future.done():
                try:
                    return future.result()
                except Exception:
                    # Logged in _generate_task
                    return None
        
        return None

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)
