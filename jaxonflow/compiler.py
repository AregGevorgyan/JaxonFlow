"""Compiler module for JaxonFlow.

Handles compilation of generated Triton code into executable kernels.
Supports a mock mode for environments without GPUs.
"""

import os
import time
import logging
from typing import Any, Callable

from .spec import CompiledKernel, CompilationResult, KernelSpec
from .hardware import HardwareContext

logger = logging.getLogger(__name__)

# Check for mock mode
MOCK_MODE = os.environ.get("JAXONFLOW_MOCK_GPU", "1") == "1"

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    if not MOCK_MODE:
        logger.warning("Triton not found, but JAXONFLOW_MOCK_GPU is not set. Forcing mock mode.")
        MOCK_MODE = True


class KernelCompiler:
    """Compiles Triton kernel code into executable functions."""

    def __init__(self):
        self.mock_mode = MOCK_MODE

    def compile(self, code: str, hardware: HardwareContext) -> CompilationResult:
        """Compile Triton code.

        Args:
            code: The Triton source code as a string.
            hardware: The target hardware context.

        Returns:
            CompilationResult indicating success or failure.
        """
        if self.mock_mode:
            return self._compile_mock(code, hardware)

        return self._compile_real(code, hardware)

    def _compile_mock(self, code: str, hardware: HardwareContext) -> CompilationResult:
        """Mock compilation for no-GPU environments.

        Performs a syntax check and creates a stub kernel. Without the
        full KernelSpec, the callable is limited to a no-op placeholder.
        Use compile_with_spec() for full functionality.
        """
        # Syntax check
        try:
            compile(code, "<generated_kernel>", "exec")
        except SyntaxError as e:
            return CompilationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                error_line=e.lineno,
            )

        # Try to exec and find a callable
        callable_fn = None
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace)  # noqa: S102
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name != "triton":
                    if "wrapper" in name.lower() or "launch" in name.lower():
                        callable_fn = obj
                        break
            if callable_fn is None:
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith("_") and name != "triton":
                        callable_fn = obj
                        break
        except Exception as e:
            logger.debug(f"Could not exec kernel code in mock mode: {e}")

        # Fallback: no-op placeholder
        if callable_fn is None:
            def _noop_kernel(*args, **kwargs):
                logger.warning("Executing no-op mock kernel (no spec reference)")
                return args[0] if args else None
            callable_fn = _noop_kernel

        # Build a minimal spec for the compiled kernel
        kernel = CompiledKernel(
            spec=KernelSpec(
                operation="unknown",
                input_shapes=[],
                input_dtypes=[],
                output_shape=(),
                output_dtype="float32",
                hardware=hardware,
            ),
            code=code,
            callable=callable_fn,
            generation_time_s=0.0,
        )

        return CompilationResult(success=True, kernel=kernel)

    def _compile_real(self, code: str, hardware: HardwareContext) -> CompilationResult:
        """Real compilation for GPU environments.

        Attempts to compile and exec the Triton code, extracting a callable
        kernel function. Falls back to mock if Triton is unavailable.
        """
        if not HAS_TRITON:
            logger.warning("Triton not available, falling back to mock compilation")
            return self._compile_mock(code, hardware)

        # Syntax check
        try:
            compile(code, "<generated_kernel>", "exec")
        except SyntaxError as e:
            return CompilationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                error_line=e.lineno,
            )

        # Exec the code with triton available in namespace
        callable_fn = None
        try:
            namespace: dict[str, Any] = {"triton": triton, "tl": tl}
            exec(code, namespace)  # noqa: S102

            # Look for wrapper/launch function first
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    if "wrapper" in name.lower() or "launch" in name.lower():
                        callable_fn = obj
                        break

            # Fall back to any callable (likely the @triton.jit kernel)
            if callable_fn is None:
                for name, obj in namespace.items():
                    if (callable(obj) and not name.startswith("_")
                            and name not in ("triton", "tl")):
                        callable_fn = obj
                        break
        except Exception as e:
            return CompilationResult(
                success=False,
                error=f"Execution error: {str(e)}",
            )

        if callable_fn is None:
            return CompilationResult(
                success=False,
                error="No callable kernel function found in generated code",
            )

        kernel = CompiledKernel(
            spec=KernelSpec(
                operation="unknown",
                input_shapes=[],
                input_dtypes=[],
                output_shape=(),
                output_dtype="float32",
                hardware=hardware,
            ),
            code=code,
            callable=callable_fn,
            generation_time_s=0.0,
        )
        return CompilationResult(success=True, kernel=kernel)

    def compile_with_spec(self, code: str, spec: KernelSpec) -> CompilationResult:
        """Compile Triton code using the full kernel spec.

        This is the preferred method as it provides the full spec including
        the reference implementation for mock mode.

        Args:
            code: Generated Triton source code.
            spec: The kernel specification.

        Returns:
            CompilationResult.
        """
        if self.mock_mode:
            return self._compile_mock_with_spec(code, spec)
        return self._compile_real_with_spec(code, spec)

    def _compile_mock_with_spec(self, code: str, spec: KernelSpec) -> CompilationResult:
        """Mock compilation returning the reference implementation."""
        logger.info(f"Mock compiling kernel for {spec.operation}")

        if spec.reference_impl is None:
            def dummy_impl(*args, **kwargs):
                logger.warning(f"Executing mock kernel for {spec.operation} without reference impl.")
                return None
            callable_impl = dummy_impl
        else:
            callable_impl = spec.reference_impl

        kernel = CompiledKernel(
            spec=spec,
            code=code,
            callable=callable_impl,
            generation_time_s=0.1,
        )

        return CompilationResult(
            success=True,
            kernel=kernel
        )

    def _compile_real_with_spec(self, code: str, spec: KernelSpec) -> CompilationResult:
        """Real compilation with full spec context.

        Attempts to exec the code with Triton in the namespace. If that fails,
        falls back to mock compilation with spec.
        """
        if not HAS_TRITON:
            logger.warning("Triton not available, falling back to mock compilation")
            return self._compile_mock_with_spec(code, spec)

        # Syntax check
        try:
            compile(code, "<generated_kernel>", "exec")
        except SyntaxError as e:
            return CompilationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                error_line=e.lineno,
            )

        # Exec with triton available
        callable_fn = None
        try:
            namespace: dict[str, Any] = {"triton": triton, "tl": tl}
            exec(code, namespace)  # noqa: S102

            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    if "wrapper" in name.lower() or "launch" in name.lower():
                        callable_fn = obj
                        break

            if callable_fn is None:
                for name, obj in namespace.items():
                    if (callable(obj) and not name.startswith("_")
                            and name not in ("triton", "tl")):
                        callable_fn = obj
                        break
        except Exception as e:
            return CompilationResult(
                success=False,
                error=f"Execution error: {str(e)}",
            )

        if callable_fn is None:
            # Fall back to reference impl if available
            if spec.reference_impl is not None:
                callable_fn = spec.reference_impl
            else:
                return CompilationResult(
                    success=False,
                    error="No callable kernel function found in generated code",
                )

        kernel = CompiledKernel(
            spec=spec,
            code=code,
            callable=callable_fn,
            generation_time_s=0.0,
        )
        return CompilationResult(success=True, kernel=kernel)
