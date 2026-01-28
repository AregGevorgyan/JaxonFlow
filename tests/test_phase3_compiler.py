"""Tests for Phase 3: Compilation & Execution.

Tests the compiler, verifier, profiler, and feedback modules.
These run in mock mode (no GPU required).
"""

import os
import pytest
import numpy as np

from jaxonflow.compiler import KernelCompiler
from jaxonflow.verification import KernelVerifier
from jaxonflow.profiler import KernelProfiler
from jaxonflow.feedback import FeedbackTranslator
from jaxonflow.hardware import HardwareContext
from jaxonflow.spec import KernelSpec, CompiledKernel, CompilationResult, ProfileResult


# Ensure mock mode
os.environ["JAXONFLOW_MOCK_GPU"] = "1"


# ── KernelCompiler ─────────────────────────────────────────────────


class TestKernelCompiler:
    """Test the KernelCompiler class."""

    @pytest.fixture
    def compiler(self):
        return KernelCompiler()

    @pytest.fixture
    def hardware(self):
        return HardwareContext.from_name("A100-80GB")

    @pytest.fixture
    def matmul_spec(self, hardware):
        return KernelSpec(
            operation="matmul",
            input_shapes=[(32, 64), (64, 32)],
            input_dtypes=["float32", "float32"],
            output_shape=(32, 32),
            output_dtype="float32",
            hardware=hardware,
            reference_impl=lambda a, b: np.dot(a, b),
        )

    def test_compile_with_spec_mock_success(self, compiler, matmul_spec):
        code = "@triton.jit\ndef matmul_kernel(): pass"
        result = compiler.compile_with_spec(code, matmul_spec)
        assert result.success is True
        assert result.kernel is not None
        assert result.kernel.callable is not None

    def test_compile_with_spec_mock_no_ref(self, compiler, hardware):
        spec = KernelSpec(
            operation="unknown",
            input_shapes=[(10,)],
            input_dtypes=["float32"],
            output_shape=(10,),
            output_dtype="float32",
            hardware=hardware,
            reference_impl=None,
        )
        result = compiler.compile_with_spec("dummy", spec)
        assert result.success is True
        # Callable should be a dummy that returns None
        assert result.kernel.callable is not None

    def test_compile_without_spec_returns_result(self, compiler, hardware):
        """Test the compile() method (without spec) now returns a proper result."""
        code = "def wrapper(a, b): return a + b"
        result = compiler.compile(code, hardware)
        assert isinstance(result, CompilationResult)
        assert result.success is True
        assert result.kernel is not None

    def test_compile_syntax_error(self, compiler, hardware):
        """compile() should detect syntax errors."""
        code = "def broken(:\n  pass"
        result = compiler.compile(code, hardware)
        assert result.success is False
        assert "Syntax error" in result.error

    def test_compile_finds_wrapper_function(self, compiler, hardware):
        code = """
def my_wrapper(a, b):
    return a + b
"""
        result = compiler.compile(code, hardware)
        assert result.success is True
        assert result.kernel.callable is not None
        # The callable should be my_wrapper
        assert result.kernel.callable(2, 3) == 5

    def test_compile_finds_any_callable(self, compiler, hardware):
        code = """
def compute(x):
    return x * 2
"""
        result = compiler.compile(code, hardware)
        assert result.success is True
        assert result.kernel.callable(5) == 10


# ── KernelVerifier (Phase 3 standalone) ─────────────────────────────


class TestPhase3Verifier:
    """Test the Phase 3 standalone KernelVerifier."""

    @pytest.fixture
    def verifier(self):
        return KernelVerifier()

    def test_verify_correct_kernel(self, verifier):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(16, 32), (32, 16)],
            input_dtypes=["float32", "float32"],
            output_shape=(16, 16),
            output_dtype="float32",
            reference_impl=lambda a, b: np.dot(a, b),
        )
        kernel = CompiledKernel(
            spec=spec,
            code="",
            callable=lambda a, b: np.dot(a, b),
        )
        result = verifier.verify(kernel, spec)
        assert result.correct is True

    def test_verify_incorrect_kernel(self, verifier):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(16, 32), (32, 16)],
            input_dtypes=["float32", "float32"],
            output_shape=(16, 16),
            output_dtype="float32",
            reference_impl=lambda a, b: np.dot(a, b),
        )
        # Wrong kernel: returns zeros instead of matmul
        kernel = CompiledKernel(
            spec=spec,
            code="",
            callable=lambda a, b: np.zeros((16, 16), dtype=np.float32),
        )
        result = verifier.verify(kernel, spec)
        assert result.correct is False

    def test_verify_no_reference(self, verifier):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(16, 32), (32, 16)],
            input_dtypes=["float32", "float32"],
            output_shape=(16, 16),
            output_dtype="float32",
            reference_impl=None,
        )
        kernel = CompiledKernel(
            spec=spec,
            code="",
            callable=lambda a, b: np.dot(a, b),
        )
        result = verifier.verify(kernel, spec)
        assert result.correct is False
        assert "No reference" in result.test_case_description

    def test_verify_kernel_raises(self, verifier):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(4, 4), (4, 4)],
            input_dtypes=["float32", "float32"],
            output_shape=(4, 4),
            output_dtype="float32",
            reference_impl=lambda a, b: np.dot(a, b),
        )

        def broken_kernel(a, b):
            raise RuntimeError("GPU exploded")

        kernel = CompiledKernel(
            spec=spec,
            code="",
            callable=broken_kernel,
        )
        result = verifier.verify(kernel, spec)
        assert result.correct is False
        assert "Exception" in result.test_case_description

    def test_dtype_mapping(self, verifier):
        assert verifier._parse_dtype("float32") == np.float32
        assert verifier._parse_dtype("float16") == np.float16
        assert verifier._parse_dtype("int32") == np.int32
        assert verifier._parse_dtype("unknown") == np.float32  # fallback

    def test_tolerances_float16(self, verifier):
        is_close, max_abs, max_rel = verifier._compare_outputs(
            np.array([1.0], dtype=np.float32),
            np.array([1.001], dtype=np.float32),
            "float16",
        )
        # float16 has looser tolerance (1e-3)
        assert is_close is True


# ── KernelProfiler ──────────────────────────────────────────────────


class TestKernelProfiler:
    """Test the KernelProfiler class."""

    @pytest.fixture
    def hardware(self):
        return HardwareContext.from_name("A100-80GB")

    @pytest.fixture
    def profiler(self, hardware):
        return KernelProfiler(hardware)

    @pytest.fixture
    def matmul_spec(self, hardware):
        return KernelSpec(
            operation="matmul",
            input_shapes=[(1024, 512), (512, 1024)],
            input_dtypes=["float32", "float32"],
            output_shape=(1024, 1024),
            output_dtype="float32",
            hardware=hardware,
        )

    def test_mock_profile_returns_result(self, profiler, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec, code="", callable=lambda a, b: a
        )
        result = profiler.profile(kernel, matmul_spec)
        assert isinstance(result, ProfileResult)
        assert result.execution_time_us > 0
        assert result.speedup_vs_reference > 0
        assert result.primary_bottleneck in ("memory", "compute", "latency")

    def test_mock_profile_has_utilization_metrics(self, profiler, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec, code="", callable=lambda a, b: a
        )
        result = profiler.profile(kernel, matmul_spec)
        assert result.memory_bandwidth_utilization is not None
        assert 0 <= result.memory_bandwidth_utilization <= 1.0
        assert result.compute_utilization is not None
        assert 0 <= result.compute_utilization <= 1.0
        assert result.achieved_occupancy is not None
        assert result.registers_per_thread is not None
        assert result.shared_memory_bytes is not None

    def test_mock_profile_unknown_flops(self, profiler):
        """Profile with an operation that has no estimated FLOPs."""
        hardware = HardwareContext.from_name("A100-80GB")
        spec = KernelSpec(
            operation="custom_op",
            input_shapes=[(100,)],
            input_dtypes=["float32"],
            output_shape=(100,),
            output_dtype="float32",
            hardware=hardware,
        )
        kernel = CompiledKernel(spec=spec, code="", callable=lambda x: x)
        result = profiler.profile(kernel, spec)
        # Should still work with default 1 GFLOP
        assert result.execution_time_us > 0


# ── FeedbackTranslator ──────────────────────────────────────────────


class TestFeedbackTranslator:
    """Test the FeedbackTranslator class."""

    @pytest.fixture
    def translator(self):
        return FeedbackTranslator()

    @pytest.fixture
    def hardware(self):
        return HardwareContext.from_name("A100-80GB")

    def test_translate_memory_bound(self, translator, hardware):
        profile = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=1.1,
            memory_bandwidth_utilization=0.9,
            compute_utilization=0.3,
            achieved_occupancy=0.8,
            primary_bottleneck="memory",
        )
        feedback = translator.translate(profile, hardware)
        assert "memory bound" in feedback.lower()
        assert "Speedup vs Reference" in feedback

    def test_translate_compute_bound(self, translator, hardware):
        profile = ProfileResult(
            execution_time_us=50.0,
            speedup_vs_reference=0.9,
            memory_bandwidth_utilization=0.3,
            compute_utilization=0.95,
            primary_bottleneck="compute",
        )
        feedback = translator.translate(profile, hardware)
        assert "compute bound" in feedback.lower()

    def test_translate_latency_bound(self, translator, hardware):
        profile = ProfileResult(
            execution_time_us=5.0,
            speedup_vs_reference=0.5,
            primary_bottleneck="latency",
        )
        feedback = translator.translate(profile, hardware)
        assert "latency" in feedback.lower()

    def test_translate_includes_recommendations(self, translator, hardware):
        profile = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=1.0,
            primary_bottleneck="memory",
        )
        feedback = translator.translate(profile, hardware)
        assert "Recommendations" in feedback

    def test_translate_no_bottleneck(self, translator, hardware):
        profile = ProfileResult(
            execution_time_us=100.0,
            speedup_vs_reference=2.0,
            primary_bottleneck=None,
        )
        feedback = translator.translate(profile, hardware)
        assert "Speedup vs Reference" in feedback


# ── End-to-end Pipeline ────────────────────────────────────────────


class TestPhase3Pipeline:
    """Test the full Phase 3 pipeline: spec -> compile -> verify -> profile -> feedback."""

    def test_full_pipeline(self):
        hardware = HardwareContext.from_name("H100-SXM")

        def ref_matmul(a, b):
            return np.dot(a, b)

        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(64, 128), (128, 64)],
            input_dtypes=["float32", "float32"],
            output_shape=(64, 64),
            output_dtype="float32",
            hardware=hardware,
            reference_impl=ref_matmul,
        )

        # 1. Compile
        compiler = KernelCompiler()
        compile_result = compiler.compile_with_spec(
            "@triton.jit\ndef kernel(): pass", spec
        )
        assert compile_result.success

        # 2. Verify
        verifier = KernelVerifier()
        verify_result = verifier.verify(compile_result.kernel, spec)
        assert verify_result.correct

        # 3. Profile
        profiler = KernelProfiler(hardware)
        profile_result = profiler.profile(compile_result.kernel, spec)
        assert profile_result.execution_time_us > 0

        # 4. Feedback
        translator = FeedbackTranslator()
        feedback = translator.translate(profile_result, hardware)
        assert isinstance(feedback, str)
        assert len(feedback) > 0
