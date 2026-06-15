"""
Stable task request/result types for FAR Mesh Quad 3.
These are pure data – no Qt, no viewport, no GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskKind(str, Enum):
    """Kinds of work the execution layer must schedule."""

    MESH_PREVIEW = "mesh_preview"
    MESH_REPAIR = "mesh_repair"
    MESH_REDUCE = "mesh_reduce"

    GROUP_CLEANUP = "group_cleanup"
    GROUP_REDUCE = "group_reduce"

    HOLE_FILL_PREVIEW = "hole_fill_preview"
    BORE_REGION_EXTRACT = "bore_region_extract"
    BORE_CLEAN_PREVIEW = "bore_clean_preview"
    BORE_REBUILD_CANDIDATE = "bore_rebuild_candidate"

    EXTERNAL_INSTANT_MESHES = "external_instant_meshes"
    EXTERNAL_QUADWILD = "external_quadwild"

    NEURAL_SURFACE_REPAIR = "neural_surface_repair"  # future socket


@dataclass(slots=True)
class TaskRequest:
    """What the controller asks the execution layer to do."""
    kind: TaskKind
    payload: dict[str, Any] = field(default_factory=dict)
    hints: dict[str, Any] = field(default_factory=dict)
    source_mesh_ref: str | None = None
    description: str = ""


@dataclass(slots=True)
class TaskResult:
    """Structured result from any execution backend."""
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
