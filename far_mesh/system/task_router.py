"""
Task routing policy for FAR MESH Quad 3.

This module decides *how* a task should run. It does not execute the task.

Rules:
- no Qt imports
- no viewport imports
- no MainWindow dependency
- no accelerator framework yet
- launcher-neutral resource interpretation
"""

from __future__ import annotations

from typing import Any

from far_mesh.system.execution_plan import ExecutionMode, ExecutionPlan
from far_mesh.system.resource_probe import SystemResources
from far_mesh.system.task_protocol import TaskKind, TaskRequest


# Mesh repair methods that may touch native/Open3D-backed repair paths.
# "trimesh" remains THREAD-safe by default unless the task is large.
NATIVE_MESH_REPAIR_METHODS = {
    "open3d",
    "hybrid",
    "cad_workflow",
    "pymeshfix",
    "cad_safe",
    "cad_safe_pymeshlab",
    "cad_preserve_features",
    "light_normalize",
    "topology_cleanup",
    "scan_closing",
}


# Mesh reduce backends that may load Open3D/native code.
NATIVE_MESH_REDUCE_BACKENDS = {
    "open3d",
}


def plan_task(
    request: TaskRequest,
    resources: SystemResources,
    hints: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """
    Return an ExecutionPlan for a task request.

    Conservative policy:
    - external executables use SUBPROCESS
    - native/Open3D-backed geometry work uses PROCESS
    - small pure-core repair/reduce paths may use THREAD
    - explicit force_mode remains an override for debugging/tests
    """

    hints = hints or {}
    kind = request.kind

    forced_mode = hints.get("force_mode") or request.hints.get("force_mode")
    if forced_mode:
        try:
            mode = ExecutionMode(str(forced_mode))
            return ExecutionPlan(
                mode=mode,
                reason=f"forced execution mode: {mode.value}",
                max_workers=_recommended_workers(resources),
                isolate_memory=mode == ExecutionMode.PROCESS,
            )
        except ValueError:
            return ExecutionPlan(
                mode=ExecutionMode.THREAD,
                reason=f"invalid forced execution mode ignored: {forced_mode!r}",
                max_workers=_recommended_workers(resources),
            )

    if kind in {
        TaskKind.EXTERNAL_INSTANT_MESHES,
        TaskKind.EXTERNAL_QUADWILD,
    }:
        return ExecutionPlan(
            mode=ExecutionMode.SUBPROCESS,
            reason="external backend must run as a subprocess",
            isolate_memory=True,
        )

    if kind == TaskKind.NEURAL_SURFACE_REPAIR:
        return ExecutionPlan(
            mode=ExecutionMode.PROCESS,
            reason="future neural/accelerator task; isolated process fallback",
            max_workers=1,
            isolate_memory=True,
        )

    if kind in {
        TaskKind.BORE_REGION_EXTRACT,
        TaskKind.BORE_CLEAN_PREVIEW,
        TaskKind.BORE_REBUILD_CANDIDATE,
        TaskKind.HOLE_FILL_PREVIEW,
    }:
        return ExecutionPlan(
            mode=ExecutionMode.PROCESS,
            reason="heavy topology or bore-aware geometry task",
            max_workers=1,
            isolate_memory=True,
        )

    if kind == TaskKind.MESH_PREVIEW:
        # manual_edit_pipeline converts to Open3D tensor mesh before operation
        # dispatch, so even delete_faces/delete_vertices can load Open3D.
        return ExecutionPlan(
            mode=ExecutionMode.PROCESS,
            reason="manual preview task uses native/Open3D pipeline and requires process isolation",
            max_workers=1,
            isolate_memory=True,
        )

    if kind in {
        TaskKind.GROUP_CLEANUP,
        TaskKind.GROUP_REDUCE,
    }:
        # quad_group_adapter uses Open3D tensor meshes for group processing.
        # Always isolate this path; do not special-case small meshes.
        return ExecutionPlan(
            mode=ExecutionMode.PROCESS,
            reason="group processing uses native/Open3D pipeline and requires process isolation",
            max_workers=1,
            isolate_memory=True,
        )

    if kind == TaskKind.MESH_REPAIR:
        if _mesh_repair_requires_process(request):
            return ExecutionPlan(
                mode=ExecutionMode.PROCESS,
                reason="native mesh repair task requires process isolation",
                max_workers=1,
                isolate_memory=True,
            )

        if _request_is_large(request, resources):
            return ExecutionPlan(
                mode=ExecutionMode.PROCESS,
                reason="large mesh repair task",
                max_workers=1,
                isolate_memory=True,
            )

        return ExecutionPlan(
            mode=ExecutionMode.THREAD,
            reason="mesh repair task",
            max_workers=_recommended_workers(resources),
        )

    if kind == TaskKind.MESH_REDUCE:
        if _mesh_reduce_requires_process(request):
            return ExecutionPlan(
                mode=ExecutionMode.PROCESS,
                reason="native mesh reduction task requires process isolation",
                max_workers=1,
                isolate_memory=True,
            )

        if _request_is_large(request, resources):
            return ExecutionPlan(
                mode=ExecutionMode.PROCESS,
                reason="large mesh reduction task",
                max_workers=1,
                isolate_memory=True,
            )

        return ExecutionPlan(
            mode=ExecutionMode.THREAD,
            reason="mesh reduction task",
            max_workers=_recommended_workers(resources),
        )

    return ExecutionPlan(
        mode=ExecutionMode.THREAD,
        reason="default conservative threaded execution",
        max_workers=_recommended_workers(resources),
    )


def _mesh_reduce_requires_process(request: TaskRequest) -> bool:
    backend = _normalized_payload_string(request, "backend")

    if backend in NATIVE_MESH_REDUCE_BACKENDS:
        return True

    return False


def _mesh_repair_requires_process(request: TaskRequest) -> bool:
    method = _normalized_payload_string(request, "method")

    if method in NATIVE_MESH_REPAIR_METHODS:
        return True

    requested_method = _normalized_payload_string(request, "requested_method")
    if requested_method in NATIVE_MESH_REPAIR_METHODS:
        return True

    return False


def _normalized_payload_string(request: TaskRequest, name: str) -> str:
    return _normalized_string(request.payload.get(name))


def _normalized_string(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _recommended_workers(resources: SystemResources) -> int:
    cpu = getattr(resources, "cpu", None)
    value = getattr(cpu, "recommended_worker_count", None)

    if isinstance(value, int) and value > 0:
        return value

    return 1


def _request_is_large(request: TaskRequest, resources: SystemResources) -> bool:
    raw = (
        request.hints.get("large")
        or request.hints.get("is_large")
        or request.payload.get("large")
        or request.payload.get("is_large")
    )

    if isinstance(raw, bool):
        return raw

    face_count = request.hints.get("face_count") or request.payload.get("face_count")
    vertex_count = request.hints.get("vertex_count") or request.payload.get("vertex_count")

    try:
        if face_count is not None and int(face_count) >= 250_000:
            return True
        if vertex_count is not None and int(vertex_count) >= 250_000:
            return True
    except (TypeError, ValueError):
        return False

    memory = getattr(resources, "memory", None)
    available = getattr(memory, "effective_available_bytes", None)

    if isinstance(available, int) and available > 0:
        constrained_limit = 2 * 1024 * 1024 * 1024
        if available < constrained_limit:
            return True

    return False
