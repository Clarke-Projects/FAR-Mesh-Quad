"""
Accelerator/device types for FAR Mesh Quad 3.
Currently a placeholder – no actual GPU code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComputeDeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    EXTERNAL = "external"


@dataclass(slots=True)
class ComputeCapability:
    provider_name: str
    device_kind: ComputeDeviceKind
    available: bool
    memory_bytes: int | None = None
    compute_units: int | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionHints:
    prefer_device: ComputeDeviceKind | None = None
    allow_fallback: bool = True
    max_memory_bytes: int | None = None
    latency_sensitive: bool = False
    throughput_sensitive: bool = False
    preview_safe: bool = True
