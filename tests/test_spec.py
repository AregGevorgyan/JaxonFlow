"""Tests for JaxonFlow kernel specification."""

from __future__ import annotations

import pytest

from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import (
    CompiledKernel,
    CompilationResult,
    KernelSpec,
    ProfileResult,
    VerificationResult,
)


class TestKernelSpec:
    def test_basic_creation(self, matmul_spec):
        assert matmul_spec.operation == "matmul"
        assert matmul_spec.input_shapes == [(128, 64), (64, 256)]
        assert matmul_spec.output_shape == (128, 256)

    def test_cache_key_deterministic(self, matmul_spec):
        key1 = matmul_spec.cache_key()
        key2 = matmul_spec.cache_key()
        assert key1 == key2
        assert len(key1) == 16

    def test_cache_key_differs_by_shape(self, a100_hardware):
        spec1 = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            hardware=a100_hardware,
        )
        spec2 = KernelSpec(
            operation="matmul",
            input_shapes=[(256, 128), (128, 512)],
            input_dtypes=["float32", "float32"],
            output_shape=(256, 512),
            output_dtype="float32",
            hardware=a100_hardware,
        )
        assert spec1.cache_key() != spec2.cache_key()

    def test_cache_key_differs_by_dtype(self, a100_hardware):
        spec_fp32 = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            hardware=a100_hardware,
        )
        spec_fp16 = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float16", "float16"],
            output_shape=(128, 256),
            output_dtype="float16",
            hardware=a100_hardware,
        )
        assert spec_fp32.cache_key() != spec_fp16.cache_key()

    def test_cache_key_same_across_frameworks(self, a100_hardware):
        spec_jax = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            hardware=a100_hardware,
            framework="jax",
        )
        spec_pytorch = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 64), (64, 256)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 256),
            output_dtype="float32",
            hardware=a100_hardware,
            framework="pytorch",
        )
        assert spec_jax.cache_key() == spec_pytorch.cache_key()

    def test_to_prompt(self, matmul_spec):
        prompt = matmul_spec.to_prompt()
        assert "matmul" in prompt
        assert "(128, 64)" in prompt
        assert "float32" in prompt
        assert "NVIDIA A100" in prompt

    def test_to_prompt_no_hardware(self):
        spec = KernelSpec(
            operation="relu",
            input_shapes=[(32, 128)],
            input_dtypes=["float32"],
            output_shape=(32, 128),
            output_dtype="float32",
        )
        prompt = spec.to_prompt()
        assert "relu" in prompt
        assert "NVIDIA" not in prompt

    def test_total_input_elements(self, matmul_spec):
        # (128*64) + (64*256) = 8192 + 16384 = 24576
        assert matmul_spec.total_input_elements() == 24576

    def test_total_output_elements(self, matmul_spec):
        # 128 * 256 = 32768
        assert matmul_spec.total_output_elements() == 32768

    def test_estimated_flops_matmul(self, matmul_spec):
        # 2 * M * N * K = 2 * 128 * 256 * 64 = 4194304
        assert matmul_spec.estimated_flops() == 2 * 128 * 256 * 64

    def test_estimated_flops_unknown(self):
        spec = KernelSpec(
            operation="custom_op",
            input_shapes=[(10,)],
            input_dtypes=["float32"],
            output_shape=(10,),
            output_dtype="float32",
        )
        assert spec.estimated_flops() is None

    def test_is_memory_bound_estimate(self):
        # Softmax is typically memory-bound
        spec = KernelSpec(
            operation="softmax",
            input_shapes=[(1024, 1024)],
            input_dtypes=["float32"],
            output_shape=(1024, 1024),
            output_dtype="float32",
        )
        assert spec.is_memory_bound_estimate() is True

    def test_matmul_compute_bound(self):
        # Large matmul is typically compute-bound
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4096, 4096), (4096, 4096)],
            input_dtypes=["float32", "float32"],
            output_shape=(4096, 4096),
            output_dtype="float32",
        )
        assert spec.is_memory_bound_estimate() is False

    def test_parameters_in_cache_key(self, a100_hardware):
        spec1 = KernelSpec(
            operation="conv2d",
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=["float32"],
            output_shape=(1, 64, 112, 112),
            output_dtype="float32",
            parameters={"stride": 2, "padding": 1},
            hardware=a100_hardware,
        )
        spec2 = KernelSpec(
            operation="conv2d",
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=["float32"],
            output_shape=(1, 64, 112, 112),
            output_dtype="float32",
            parameters={"stride": 1, "padding": 0},
            hardware=a100_hardware,
        )
        assert spec1.cache_key() != spec2.cache_key()


class TestCompiledKernel:
    def test_callable(self, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec,
            code="# kernel code",
            callable=lambda a, b: a + b,
        )
        assert kernel(1, 2) == 3

    def test_not_compiled_raises(self, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec,
            code="# kernel code",
            callable=None,
        )
        with pytest.raises(RuntimeError, match="not been compiled"):
            kernel(1, 2)

    def test_to_cache_dict(self, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec,
            code="# kernel code",
            callable=lambda: None,
            speedup_vs_reference=1.5,
            iterations_to_generate=3,
        )
        cache_dict = kernel.to_cache_dict()
        assert cache_dict["operation"] == "matmul"
        assert cache_dict["code"] == "# kernel code"
        assert cache_dict["speedup_vs_reference"] == 1.5
        assert cache_dict["iterations_to_generate"] == 3


class TestCompilationResult:
    def test_success(self, matmul_spec):
        kernel = CompiledKernel(spec=matmul_spec, code="# code")
        result = CompilationResult(success=True, kernel=kernel)
        assert result.success is True
        assert result.kernel is not None

    def test_failure(self):
        result = CompilationResult(success=False, error="Syntax error")
        assert result.success is False
        assert result.error == "Syntax error"


class TestVerificationResult:
    def test_correct(self):
        result = VerificationResult(correct=True)
        assert result.correct is True

    def test_incorrect(self):
        result = VerificationResult(
            correct=False,
            max_abs_error=0.01,
            test_case_description="random_normal",
        )
        assert result.correct is False
        assert result.max_abs_error == 0.01


class TestProfileResult:
    def test_creation(self):
        result = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=2.0,
            memory_bandwidth_utilization=0.85,
            compute_utilization=0.6,
            primary_bottleneck="compute",
        )
        assert result.speedup_vs_reference == 2.0
        assert result.primary_bottleneck == "compute"
