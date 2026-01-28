"""Verification module for JaxonFlow.

Handles correctness verification of generated kernels against reference implementations.
"""

import time
import logging
import numpy as np
from typing import Any, List, Optional

from .spec import CompiledKernel, KernelSpec, VerificationResult

logger = logging.getLogger(__name__)


class KernelVerifier:
    """Verifies the correctness of generated kernels."""

    def __init__(self):
        pass

    def verify(self, kernel: CompiledKernel, spec: KernelSpec) -> VerificationResult:
        """Verify kernel correctness against the reference implementation.

        Args:
            kernel: The compiled kernel to test.
            spec: The kernel specification containing input details and reference impl.

        Returns:
            VerificationResult with pass/fail status and error metrics.
        """
        if spec.reference_impl is None:
            return VerificationResult(
                correct=False,
                test_case_description="No reference implementation provided",
            )

        try:
            # 1. Generate test inputs
            inputs = self._generate_inputs(spec)

            # 2. Run kernel
            # Note: In mock mode, this might just run the reference impl
            # but we treat it as the "actual" result.
            start_time = time.time()
            actual_output = kernel(*inputs)
            kernel_time = time.time() - start_time
            
            # If the kernel wrapper returns None (e.g. dummy mock), fail
            if actual_output is None:
                return VerificationResult(
                    correct=False,
                    test_case_description="Kernel execution returned None (Mock dummy)"
                )

            # 3. Run reference implementation
            # We re-run it even if the kernel IS the reference impl wrapper
            # to be structurally correct for the real system.
            expected_output = spec.reference_impl(*inputs)

            # 4. Compare results
            is_correct, max_abs, max_rel = self._compare_outputs(
                actual_output, expected_output, spec.output_dtype
            )

            return VerificationResult(
                correct=is_correct,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                test_case_description="Random inputs",
                expected=expected_output if not is_correct else None,
                actual=actual_output if not is_correct else None,
            )

        except Exception as e:
            logger.exception("Verification failed with exception")
            return VerificationResult(
                correct=False,
                test_case_description=f"Exception during verification: {str(e)}"
            )

    def _generate_inputs(self, spec: KernelSpec) -> List[np.ndarray]:
        """Generate random inputs based on the spec."""
        inputs = []
        for shape, dtype_str in zip(spec.input_shapes, spec.input_dtypes):
            dtype = self._parse_dtype(dtype_str)
            
            if np.issubdtype(dtype, np.integer):
                # Random integers
                data = np.random.randint(-10, 10, size=shape).astype(dtype)
            elif np.issubdtype(dtype, np.floating):
                # Random floats
                data = np.random.randn(*shape).astype(dtype)
            elif dtype == np.bool_:
                data = np.random.choice([True, False], size=shape)
            else:
                # Fallback
                data = np.zeros(shape, dtype=dtype)
                
            inputs.append(data)
        return inputs

    def _parse_dtype(self, dtype_str: str) -> Any:
        """Convert string dtype to numpy dtype."""
        # Simple mapping, can be expanded
        mapping = {
            "float32": np.float32,
            "float16": np.float16,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
            "int8": np.int8,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        # Handle JAX/Torch style strings like 'dtype(\'float32\')' if necessary
        clean_str = dtype_str.replace("dtype('", "").replace("')", "")
        return mapping.get(clean_str, np.float32)

    def _compare_outputs(self, actual: Any, expected: Any, dtype_str: str) -> tuple[bool, float, float]:
        """Compare actual and expected outputs."""
        # Check for JAX/Torch tensors and convert to numpy
        if hasattr(actual, "cpu"): # PyTorch
            actual = actual.detach().cpu().numpy()
        elif hasattr(actual, "__array__"): # JAX/Numpy
            actual = np.array(actual)
            
        if hasattr(expected, "cpu"):
            expected = expected.detach().cpu().numpy()
        elif hasattr(expected, "__array__"):
            expected = np.array(expected)

        if actual.shape != expected.shape:
            return False, 0.0, 0.0

        # Tolerances
        atol = 1e-5
        rtol = 1e-5
        if "float16" in dtype_str:
            atol = 1e-3
            rtol = 1e-3
            
        try:
            is_close = np.allclose(actual, expected, atol=atol, rtol=rtol)
            
            diff = np.abs(actual - expected)
            max_abs = float(np.max(diff))
            
            # Avoid divide by zero
            denom = np.abs(expected) + 1e-8
            max_rel = float(np.max(diff / denom))
            
            return is_close, max_abs, max_rel
        except Exception:
            return False, 999.0, 999.0
