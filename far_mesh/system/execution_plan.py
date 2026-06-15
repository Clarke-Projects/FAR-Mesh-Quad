"""
Execution plan types for FAR MESH Quad 3.

Pure data only:
- no Qt
- no viewport
- no GUI
- no dependency on future accelerator/provider packages
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    """How a task should be executed."""

    INLINE = "inline"
    THREAD = "thread"
    PROCESS = "process"
    SUBPROCESS = "subprocess"
    ACCELERATOR = "accelerator"  # future socket only


class ComputeDeviceKind(str, Enum):
    """
    Minimal device-kind socket.

    This intentionally lives here for now so Phase 1.5 does not depend on a
    not-yet-required far_mesh.compute package.
    """

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ExecutionPlan:
    """Pure routing result produced by task_router.py."""

    mode: ExecutionMode
    device_kind: ComputeDeviceKind = ComputeDeviceKind.CPU
    provider_name: str | None = None
    reason: str = ""
    max_workers: int | None = None
    isolate_memory: bool = False
