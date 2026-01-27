"""Hardware context detection and GPU profiles for JaxonFlow."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Literal

from .exceptions import HardwareDetectionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareContext:
    """Hardware specifications for kernel optimization.

    This dataclass captures GPU specifications that are essential for
    hardware-aware kernel generation. The specs are injected into LLM
    prompts to enable architecture-specific optimizations.
    """

    # Device identification
    name: str
    compute_capability: tuple[int, int]  # e.g., (9, 0) for SM90
    vendor: Literal["nvidia", "amd", "google", "unknown"]

    # Compute resources
    sm_count: int
    cores_per_sm: int
    warp_size: int
    max_threads_per_block: int
    max_blocks_per_sm: int

    # Memory hierarchy (all sizes in bytes unless specified)
    global_memory_bytes: int
    shared_memory_per_block: int
    shared_memory_per_sm: int
    l2_cache_size: int
    registers_per_block: int

    # Bandwidth (GB/s)
    memory_bandwidth_gbps: float

    # Compute throughput (TFLOPS)
    fp32_tflops: float
    fp16_tflops: float
    tf32_tflops: float
    fp8_tflops: float | None = None

    @property
    def global_memory_gb(self) -> float:
        """Global memory in GB."""
        return self.global_memory_bytes / (1024**3)

    @property
    def sm_arch(self) -> str:
        """SM architecture string (e.g., 'SM90')."""
        major, minor = self.compute_capability
        return f"SM{major}{minor}"

    @classmethod
    def detect(cls) -> HardwareContext:
        """Auto-detect hardware from current device.

        Attempts detection in order:
        1. JAX device query (if JAX is available)
        2. PyTorch device query (if PyTorch is available)
        3. nvidia-smi (if available)
        4. Fallback to A100-80GB profile

        Returns:
            HardwareContext for detected or fallback hardware.
        """
        # Try JAX detection
        try:
            return cls._detect_jax()
        except Exception as e:
            logger.debug(f"JAX detection failed: {e}")

        # Try PyTorch detection
        try:
            return cls._detect_pytorch()
        except Exception as e:
            logger.debug(f"PyTorch detection failed: {e}")

        # Try nvidia-smi
        try:
            return cls._detect_nvidia_smi()
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

        # Fallback to A100
        logger.warning("Hardware detection failed, using A100-80GB profile as fallback")
        return cls.from_name("A100-80GB")

    @classmethod
    def _detect_jax(cls) -> HardwareContext:
        """Detect hardware using JAX."""
        import jax

        devices = jax.devices()
        if not devices:
            raise HardwareDetectionError("No JAX devices found")

        device = devices[0]
        device_kind = str(device.device_kind).lower()

        if "gpu" not in device_kind and "cuda" not in device_kind:
            raise HardwareDetectionError(f"No GPU device found (device_kind: {device_kind})")

        # JAX doesn't expose detailed GPU info directly
        # Try to match by name
        platform = str(device.platform).lower()
        if "h100" in platform:
            return cls.from_name("H100-SXM")
        elif "a100" in platform:
            return cls.from_name("A100-80GB")

        # Default to A100 for unknown NVIDIA GPUs
        return cls.from_name("A100-80GB")

    @classmethod
    def _detect_pytorch(cls) -> HardwareContext:
        """Detect hardware using PyTorch."""
        import torch

        if not torch.cuda.is_available():
            raise HardwareDetectionError("CUDA not available in PyTorch")

        device_name = torch.cuda.get_device_name(0).lower()
        props = torch.cuda.get_device_properties(0)

        # Match by name
        if "h100" in device_name:
            return cls.from_name("H100-SXM")
        elif "a100" in device_name:
            if props.total_memory > 70 * (1024**3):
                return cls.from_name("A100-80GB")
            else:
                return cls.from_name("A100-40GB")
        elif "v100" in device_name:
            return cls.from_name("V100-32GB")
        elif "4090" in device_name:
            return cls.from_name("RTX4090")

        # Build from PyTorch properties for unknown GPUs
        cc_major, cc_minor = props.major, props.minor
        return cls(
            name=torch.cuda.get_device_name(0),
            compute_capability=(cc_major, cc_minor),
            vendor="nvidia",
            sm_count=props.multi_processor_count,
            cores_per_sm=64 if cc_major >= 8 else 32,  # Approximate
            warp_size=32,
            max_threads_per_block=props.max_threads_per_block,
            max_blocks_per_sm=32,  # Approximate
            global_memory_bytes=props.total_memory,
            shared_memory_per_block=props.max_shared_memory_per_block,
            shared_memory_per_sm=props.max_shared_memory_per_block * 2,  # Approximate
            l2_cache_size=40 * 1024 * 1024,  # Approximate
            registers_per_block=props.regs_per_block,
            memory_bandwidth_gbps=1000.0,  # Approximate
            fp32_tflops=20.0,  # Approximate
            fp16_tflops=100.0,  # Approximate
            tf32_tflops=50.0,  # Approximate
        )

    @classmethod
    def _detect_nvidia_smi(cls) -> HardwareContext:
        """Detect hardware using nvidia-smi."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise HardwareDetectionError(f"nvidia-smi failed: {result.stderr}")

        output = result.stdout.strip()
        if not output:
            raise HardwareDetectionError("nvidia-smi returned empty output")

        name, memory_mb = output.split(",")
        name = name.strip().lower()
        memory_gb = float(memory_mb.strip()) / 1024

        # Match by name
        if "h100" in name:
            return cls.from_name("H100-SXM")
        elif "a100" in name:
            if memory_gb > 70:
                return cls.from_name("A100-80GB")
            else:
                return cls.from_name("A100-40GB")
        elif "v100" in name:
            return cls.from_name("V100-32GB")
        elif "4090" in name:
            return cls.from_name("RTX4090")

        # Default to A100 for unknown
        logger.warning(f"Unknown GPU '{name}', using A100-80GB profile")
        return cls.from_name("A100-80GB")

    @classmethod
    def from_name(cls, name: str) -> HardwareContext:
        """Load known hardware profile by name.

        Args:
            name: Hardware profile name (e.g., "A100-80GB", "H100-SXM")

        Returns:
            HardwareContext for the specified hardware.

        Raises:
            ValueError: If profile name is unknown.
        """
        profiles = _get_hardware_profiles()
        if name not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(f"Unknown hardware profile '{name}'. Available: {available}")
        return profiles[name]

    @classmethod
    def available_profiles(cls) -> list[str]:
        """List available hardware profile names."""
        return sorted(_get_hardware_profiles().keys())

    def to_prompt_context(self) -> str:
        """Format hardware specs for LLM prompt injection.

        Returns:
            Multi-line string with hardware specifications formatted
            for inclusion in LLM prompts.
        """
        lines = [
            f"Target Hardware: {self.name}",
            f"Vendor: {self.vendor}",
            f"Compute Capability: {self.sm_arch}",
            "",
            "Compute Resources:",
            f"- SMs: {self.sm_count}",
            f"- Cores per SM: {self.cores_per_sm}",
            f"- Warp Size: {self.warp_size}",
            f"- Max Threads per Block: {self.max_threads_per_block}",
            f"- Max Blocks per SM: {self.max_blocks_per_sm}",
            "",
            "Memory Hierarchy:",
            f"- Global Memory: {self.global_memory_gb:.1f} GB",
            f"- Shared Memory per Block: {self.shared_memory_per_block:,} bytes",
            f"- Shared Memory per SM: {self.shared_memory_per_sm:,} bytes",
            f"- L2 Cache: {self.l2_cache_size // (1024 * 1024)} MB",
            f"- Registers per Block: {self.registers_per_block:,}",
            "",
            "Bandwidth:",
            f"- Memory Bandwidth: {self.memory_bandwidth_gbps:.0f} GB/s",
            "",
            "Compute Throughput:",
            f"- FP32: {self.fp32_tflops} TFLOPS",
            f"- FP16: {self.fp16_tflops} TFLOPS",
            f"- TF32: {self.tf32_tflops} TFLOPS",
        ]

        if self.fp8_tflops is not None:
            lines.append(f"- FP8: {self.fp8_tflops} TFLOPS")

        return "\n".join(lines)


def _get_hardware_profiles() -> dict[str, HardwareContext]:
    """Get all known hardware profiles.

    Returns:
        Dictionary mapping profile names to HardwareContext instances.
    """
    return {
        "A100-80GB": HardwareContext(
            name="NVIDIA A100-80GB",
            compute_capability=(8, 0),
            vendor="nvidia",
            sm_count=108,
            cores_per_sm=64,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            global_memory_bytes=80 * 1024**3,
            shared_memory_per_block=163840,
            shared_memory_per_sm=167936,
            l2_cache_size=40 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=2039,
            fp32_tflops=19.5,
            fp16_tflops=312,
            tf32_tflops=156,
        ),
        "A100-40GB": HardwareContext(
            name="NVIDIA A100-40GB",
            compute_capability=(8, 0),
            vendor="nvidia",
            sm_count=108,
            cores_per_sm=64,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            global_memory_bytes=40 * 1024**3,
            shared_memory_per_block=163840,
            shared_memory_per_sm=167936,
            l2_cache_size=40 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=1555,
            fp32_tflops=19.5,
            fp16_tflops=312,
            tf32_tflops=156,
        ),
        "H100-SXM": HardwareContext(
            name="NVIDIA H100-SXM",
            compute_capability=(9, 0),
            vendor="nvidia",
            sm_count=132,
            cores_per_sm=128,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            global_memory_bytes=80 * 1024**3,
            shared_memory_per_block=232448,
            shared_memory_per_sm=233472,
            l2_cache_size=50 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=3352,
            fp32_tflops=67,
            fp16_tflops=1979,
            tf32_tflops=989,
            fp8_tflops=3958,
        ),
        "H100-PCIe": HardwareContext(
            name="NVIDIA H100-PCIe",
            compute_capability=(9, 0),
            vendor="nvidia",
            sm_count=114,
            cores_per_sm=128,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            global_memory_bytes=80 * 1024**3,
            shared_memory_per_block=232448,
            shared_memory_per_sm=233472,
            l2_cache_size=50 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=2000,
            fp32_tflops=51,
            fp16_tflops=1513,
            tf32_tflops=756,
            fp8_tflops=3026,
        ),
        "V100-32GB": HardwareContext(
            name="NVIDIA V100-32GB",
            compute_capability=(7, 0),
            vendor="nvidia",
            sm_count=80,
            cores_per_sm=64,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            global_memory_bytes=32 * 1024**3,
            shared_memory_per_block=98304,
            shared_memory_per_sm=98304,
            l2_cache_size=6 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=900,
            fp32_tflops=15.7,
            fp16_tflops=125,
            tf32_tflops=15.7,  # V100 doesn't have TF32
        ),
        "RTX4090": HardwareContext(
            name="NVIDIA RTX 4090",
            compute_capability=(8, 9),
            vendor="nvidia",
            sm_count=128,
            cores_per_sm=128,
            warp_size=32,
            max_threads_per_block=1024,
            max_blocks_per_sm=24,
            global_memory_bytes=24 * 1024**3,
            shared_memory_per_block=102400,
            shared_memory_per_sm=102400,
            l2_cache_size=72 * 1024 * 1024,
            registers_per_block=65536,
            memory_bandwidth_gbps=1008,
            fp32_tflops=82.6,
            fp16_tflops=330,
            tf32_tflops=165,
        ),
    }
