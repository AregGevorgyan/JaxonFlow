"""Tests for JaxonFlow hardware detection."""

from __future__ import annotations

import pytest

from jaxonflow.hardware import HardwareContext


class TestHardwareProfiles:
    def test_a100_profile(self):
        hw = HardwareContext.from_name("A100-80GB")
        assert hw.name == "NVIDIA A100-80GB"
        assert hw.compute_capability == (8, 0)
        assert hw.vendor == "nvidia"
        assert hw.sm_count == 108
        assert hw.warp_size == 32
        assert hw.global_memory_gb == pytest.approx(80.0, abs=1.0)

    def test_h100_profile(self):
        hw = HardwareContext.from_name("H100-SXM")
        assert hw.name == "NVIDIA H100-SXM"
        assert hw.compute_capability == (9, 0)
        assert hw.sm_count == 132
        assert hw.fp8_tflops == 3958

    def test_v100_profile(self):
        hw = HardwareContext.from_name("V100-32GB")
        assert hw.name == "NVIDIA V100-32GB"
        assert hw.compute_capability == (7, 0)

    def test_rtx4090_profile(self):
        hw = HardwareContext.from_name("RTX4090")
        assert hw.name == "NVIDIA RTX 4090"
        assert hw.compute_capability == (8, 9)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown hardware profile"):
            HardwareContext.from_name("NonexistentGPU")

    def test_available_profiles(self):
        profiles = HardwareContext.available_profiles()
        assert "A100-80GB" in profiles
        assert "H100-SXM" in profiles
        assert "V100-32GB" in profiles
        assert "RTX4090" in profiles
        assert len(profiles) >= 4


class TestHardwareProperties:
    def test_sm_arch(self):
        hw = HardwareContext.from_name("A100-80GB")
        assert hw.sm_arch == "SM80"

        hw_h100 = HardwareContext.from_name("H100-SXM")
        assert hw_h100.sm_arch == "SM90"

    def test_global_memory_gb(self):
        hw = HardwareContext.from_name("A100-80GB")
        assert hw.global_memory_gb == pytest.approx(80.0, abs=1.0)

    def test_frozen_dataclass(self):
        hw = HardwareContext.from_name("A100-80GB")
        with pytest.raises(AttributeError):
            hw.name = "Modified"  # type: ignore


class TestPromptContext:
    def test_to_prompt_context_format(self):
        hw = HardwareContext.from_name("A100-80GB")
        ctx = hw.to_prompt_context()

        assert "NVIDIA A100-80GB" in ctx
        assert "nvidia" in ctx
        assert "SM80" in ctx
        assert "SMs: 108" in ctx
        assert "Warp Size: 32" in ctx
        assert "FP32:" in ctx
        assert "FP16:" in ctx
        assert "TF32:" in ctx

    def test_h100_includes_fp8(self):
        hw = HardwareContext.from_name("H100-SXM")
        ctx = hw.to_prompt_context()
        assert "FP8:" in ctx

    def test_a100_excludes_fp8(self):
        hw = HardwareContext.from_name("A100-80GB")
        ctx = hw.to_prompt_context()
        assert "FP8:" not in ctx


class TestHardwareDetection:
    def test_detect_fallback(self):
        """Without a GPU, detect() should fall back to A100-80GB."""
        hw = HardwareContext.detect()
        assert hw is not None
        assert hw.vendor == "nvidia"
        # Should be the fallback profile
        assert hw.name in ("NVIDIA A100-80GB", "NVIDIA H100-SXM", "NVIDIA RTX 4090", "NVIDIA V100-32GB")
