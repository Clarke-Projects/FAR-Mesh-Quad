"""
Core task registration for FAR MESH Quad 3 Phase 1.5.

This module bridges pure core task handlers into far_mesh.system.

Rules:
- no Qt imports
- no viewport imports
- no MainWindow imports
- no live MeshProcessor state ownership
- handlers receive TaskRequest.payload only
- handlers return plain dictionaries for TaskResult.payload

MeshProcessor remains the mesh authority for normal application state.
These handlers are for explicit payload-based execution.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import trimesh

from far_mesh.core.hole_fill_preview import build_hole_fill_preview_mesh
from far_mesh.core.manual_edit_pipeline import (
    ManualEditRequest,
    ManualSelection,
    build_manual_edit_preview,
)
from far_mesh.core.mesh_reducer import MeshReducer
from far_mesh.core.mesh_repairer import MeshRepairer, MeshRepairResult
from far_mesh.core.quad_group_adapter import (
    QuadGroupProcessOptions,
    build_quadwild_proxy_from_mesh,
    process_selected_groups_locally,
)
from far_mesh.core.selection_topology import HoleCandidate, find_hole_candidates
from far_mesh.core.tool_preview_state import (
    HOLE_FILL_PREVIEW_MARKER,
    LOCAL_REGION_OPERATION_MARKER,
    OPEN3D_FILL_PREVIEW_MARKER,
    REGION_KIND_HOLE_BOUNDARY,
    REGION_KIND_HOLE_PATCH,
    SNAPSHOT_ROLE_BASE,
    SNAPSHOT_ROLE_PATCH,
    SNAPSHOT_ROLE_PREVIEW,
    MeshSnapshot,
    ToolPreviewState,
    ToolRegion,
)
from far_mesh.system.execution_manager import get_lifecycle_manager, register_task
from far_mesh.system.task_protocol import TaskKind


TaskPayload = dict[str, Any]


def register_core_tasks() -> None:
    """
    Register core-safe Phase 1.5 task handlers.

    Call once during application/core startup.
    """

    register_task(TaskKind.MESH_PREVIEW, _handle_mesh_preview)
    register_task(TaskKind.MESH_REDUCE, _handle_mesh_reduce)
    register_task(TaskKind.MESH_REPAIR, _handle_mesh_repair)
    register_task(TaskKind.GROUP_CLEANUP, _handle_group_cleanup)
    register_task(TaskKind.GROUP_REDUCE, _handle_group_reduce)

    # Phase 1.5G external remesher sockets.
    # The router chooses ExecutionMode.SUBPROCESS for these task kinds.
    # The handlers below launch the lifecycle-aware external runners and
    # return path/result metadata. MeshProcessor remains responsible for
    # loading, validating, and committing the returned mesh.
    register_task(TaskKind.EXTERNAL_INSTANT_MESHES, _handle_external_instant_meshes)
    register_task(TaskKind.EXTERNAL_QUADWILD, _handle_external_quadwild)

    # Phase 2 / future topology, bore, and neural sockets.
    register_task(TaskKind.HOLE_FILL_PREVIEW, _handle_hole_fill_preview)
    register_task(TaskKind.BORE_REGION_EXTRACT, _handle_bore_region_extract)
    register_task(TaskKind.BORE_CLEAN_PREVIEW, _handle_bore_clean_preview)
    register_task(TaskKind.BORE_REBUILD_CANDIDATE, _handle_bore_rebuild_candidate)
    register_task(TaskKind.NEURAL_SURFACE_REPAIR, _handle_neural_surface_repair)


def _handle_mesh_preview(payload: TaskPayload) -> TaskPayload:
    mesh = _require_mesh(payload)
    req = _manual_request_from_payload(payload)

    preview = build_manual_edit_preview(mesh, req)

    return {
        "operation": preview.operation,
        "preview_mesh": preview.preview_mesh,
        "base_mesh": preview.base_mesh,
        "selection_summary": dict(preview.selection_summary),
        "notes": list(preview.notes),
    }


def _handle_mesh_reduce(payload: TaskPayload) -> TaskPayload:
    """
    Run full-mesh reduction from explicit payload data.

    Expected payload:
        {
            "mesh": trimesh.Trimesh,
            "backend": optional str,
            "target_faces": optional int,
            "boundary_weight": optional float,
            "cleanup": optional bool,
        }
    """

    mesh = _require_mesh(payload)

    backend = str(payload.get("backend") or "open3d")
    target_faces = int(payload.get("target_faces", 50_000))
    boundary_weight = float(payload.get("boundary_weight", 5.0))
    cleanup = bool(payload.get("cleanup", True))

    reducer = MeshReducer()

    reduced_mesh, reduction = reducer.reduce(
        mesh=mesh,
        backend=backend,
        target_faces=target_faces,
        boundary_weight=boundary_weight,
        cleanup=cleanup,
    )

    reduced_mesh = _require_mesh({"mesh": reduced_mesh})
    reduced_mesh = reduced_mesh.copy()
    reduced_mesh.remove_unreferenced_vertices()

    return {
        "operation": "mesh_reduce",
        "mesh": reduced_mesh,
        "base_mesh": mesh.copy(),
        "backend": backend,
        "target_faces": target_faces,
        "boundary_weight": boundary_weight,
        "cleanup": cleanup,
        "reduction": {
            "before_vertices": reduction.before_vertices,
            "before_faces": reduction.before_faces,
            "after_vertices": reduction.after_vertices,
            "after_faces": reduction.after_faces,
            "reduction_ratio": reduction.reduction_ratio,
            "elapsed_seconds": reduction.elapsed_seconds,
            "note": reduction.note,
        },
        "notes": [
            "Full mesh reduction completed through Phase 1.5 execution.",
            str(reduction.note or ""),
        ],
    }


def _handle_mesh_repair(payload: TaskPayload) -> TaskPayload:
    """
    Run full-mesh repair from explicit payload data.

    Expected payload:
        {
            "mesh": trimesh.Trimesh,
            "method": optional str,
            "join_comp": optional bool,
            "fill_holes": optional bool,
            "collect_inspection": optional bool,
            "repair_options": optional dict,
            "workflow_options": optional dict,
        }
    """

    mesh = _require_mesh(payload)

    method = str(payload.get("method") or "hybrid")
    join_comp = bool(payload.get("join_comp", True))
    fill_holes = bool(payload.get("fill_holes", True))
    collect_inspection = bool(payload.get("collect_inspection", True))

    repair_options = payload.get("repair_options")
    if repair_options is None:
        repair_options = {}
    if not isinstance(repair_options, dict):
        raise TypeError("payload['repair_options'] must be a dict when provided.")

    workflow_options = payload.get("workflow_options")
    if workflow_options is None:
        workflow_options = {}
    if not isinstance(workflow_options, dict):
        raise TypeError("payload['workflow_options'] must be a dict when provided.")

    repairer = MeshRepairer()

    report = repairer.clean_with_report(
        mesh,
        method=method,
        join_comp=join_comp,
        fill_holes=fill_holes,
        collect_inspection=collect_inspection,
        repair_options=repair_options,
        workflow_options=workflow_options,
    )

    if not isinstance(report, MeshRepairResult):
        raise TypeError("MeshRepairer returned an unexpected report object.")

    repaired_mesh = _require_mesh({"mesh": report.mesh})
    repaired_mesh = repaired_mesh.copy()
    repaired_mesh.remove_unreferenced_vertices()

    return {
        "operation": "mesh_repair",
        "mesh": repaired_mesh,
        "base_mesh": mesh.copy(),
        "method": method,
        "requested_method": method,
        "executed_method": report.executed_method,
        "join_comp": join_comp,
        "fill_holes": fill_holes,
        "collect_inspection": collect_inspection,
        "backend_chain": list(report.backend_chain),
        "elapsed_seconds": report.elapsed_seconds,
        "inspection_before": report.inspection_before,
        "inspection_after": report.inspection_after,
        "stats_before": report.stats_before,
        "stats_after": report.stats_after,
        "notes": list(report.notes),
    }


def _handle_group_cleanup(payload: TaskPayload) -> TaskPayload:
    return _handle_group_processing(payload, reduce=False)


def _handle_group_reduce(payload: TaskPayload) -> TaskPayload:
    return _handle_group_processing(payload, reduce=True)


def _handle_group_processing(payload: TaskPayload, *, reduce: bool) -> TaskPayload:
    mesh = _require_mesh(payload)
    face_ids = _int_array(payload.get("face_ids"))

    if face_ids.size == 0:
        raise ValueError("Grouped processing requires non-empty face_ids.")

    opts = _group_options_from_payload(payload)
    opts.reduce = bool(reduce)

    proxy = build_quadwild_proxy_from_mesh(mesh, opts)

    processed = process_selected_groups_locally(
        proxy,
        selected_face_ids=face_ids,
        opts=opts,
    )

    return {
        "operation": "group_reduce" if reduce else "group_cleanup",
        "preview_mesh": processed.proxy_trimesh.copy(),
        "base_mesh": mesh.copy(),
        "selection_summary": {
            "mode": "faces",
            "selected_faces": int(face_ids.size),
            "selected_vertices": 0,
        },
        "notes": [
            "Preview shows the full merged mesh after patch-aware group processing.",
            f"Group decode mode: {opts.decode_mode}",
        ],
    }


def _handle_external_instant_meshes(payload: TaskPayload) -> TaskPayload:
    """
    Run Instant Meshes as a lifecycle-aware external task.

    Expected payload is path-based, not live MeshProcessor-state-based.
    Common accepted keys:
        input_path / source_path
        output_path / target_path
        executable_path / executable / binary_path
        timeout
        target_faces / target_vertex_count / target_vertices
        options / runner_options

    The returned payload contains the output path and runner metadata.
    MeshProcessor must load, validate, and commit the result mesh.
    """

    from far_mesh.core.remesher_wrapper import InstantMeshesRunner

    lifecycle = get_lifecycle_manager()
    input_path = _require_existing_path(
        payload,
        "input_path",
        "source_path",
        "source_mesh_path",
        "mesh_path",
    )
    output_path = _require_output_path(
        payload,
        "output_path",
        "target_path",
        "result_path",
        "output_mesh_path",
    )

    executable_path = _first_value(
        payload,
        "executable_path",
        "executable",
        "binary_path",
        "instant_meshes_path",
    )

    runner = _construct_runner(
        InstantMeshesRunner,
        {
            "executable_path": executable_path,
            "executable": executable_path,
            "binary_path": executable_path,
            "instant_meshes_path": executable_path,
            **_dict_from_payload(payload, "runner_kwargs", "constructor_kwargs"),
        },
    )

    options = _external_options(payload)
    run_kwargs = {
        "input_path": str(input_path),
        "source_path": str(input_path),
        "input_file": str(input_path),
        "input_mesh_path": str(input_path),
        "output_path": str(output_path),
        "target_path": str(output_path),
        "output_file": str(output_path),
        "output_mesh_path": str(output_path),
        "target_faces": _first_value(payload, "target_faces", "target_face_count"),
        "target_face_count": _first_value(payload, "target_face_count", "target_faces"),
        "target_vertices": _first_value(payload, "target_vertices", "target_vertex_count"),
        "target_vertex_count": _first_value(payload, "target_vertex_count", "target_vertices"),
        "timeout": payload.get("timeout"),
        "lifecycle": lifecycle,
        **options,
    }

    runner_result = _call_with_supported_kwargs(runner.run, run_kwargs)
    runner_payload = _plain_result(runner_result)
    _raise_if_runner_failed("Instant Meshes", runner_payload)

    return {
        "operation": "external_instant_meshes",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "runner_result": runner_payload,
        "notes": [
            "Instant Meshes completed through Phase 1.5 EXTERNAL_INSTANT_MESHES routing.",
            "MeshProcessor must validate and commit the output mesh.",
        ],
    }


def _handle_external_quadwild(payload: TaskPayload) -> TaskPayload:
    """
    Run QuadWild-BiMDF as a lifecycle-aware external task.

    Expected payload is path-based.
    Common accepted keys:
        input_path / source_path
        output_dir / work_dir / target_dir
        root_dir / quadwild_root / repo_root
        stage1_config / config_path
        timeout
        options / runner_options

    The returned payload contains output directory and runner metadata.
    MeshProcessor must load, validate, and commit the selected result mesh.
    """

    from far_mesh.core.quadwild_bimdf_runner import QuadWildBiMDFRunner

    lifecycle = get_lifecycle_manager()
    input_path = _require_existing_path(
        payload,
        "input_path",
        "source_path",
        "source_mesh_path",
        "mesh_path",
    )
    output_dir = _require_output_path(
        payload,
        "output_dir",
        "work_dir",
        "target_dir",
        "result_dir",
    )

    root_dir = _first_value(payload, "root_dir", "quadwild_root", "repo_root", "project_root")
    build_dir = _first_value(payload, "build_dir", "quadwild_build_dir")
    config_dir = _first_value(payload, "config_dir", "quadwild_config_dir")

    runner = _construct_runner(
        QuadWildBiMDFRunner,
        {
            # Current runner constructor uses repo_root. The aliases remain for
            # compatibility if older/newer runner signatures are present.
            "repo_root": root_dir,
            "root_dir": root_dir,
            "quadwild_root": root_dir,
            "project_root": root_dir,
            "build_dir": build_dir,
            "config_dir": config_dir,
            **_dict_from_payload(payload, "runner_kwargs", "constructor_kwargs"),
        },
    )

    options = _external_options(payload)
    run_kwargs = {
        "input_path": str(input_path),
        "source_path": str(input_path),
        "input_file": str(input_path),
        "input_mesh_path": str(input_path),
        "output_dir": str(output_dir),
        "work_dir": str(output_dir),
        "target_dir": str(output_dir),
        "result_dir": str(output_dir),
        "stage1_config": _first_value(payload, "stage1_config", "config_path", "stage1_preset"),
        "stage1_config_rel": _first_value(payload, "stage1_config_rel", "stage1_config", "config_path", "stage1_preset"),
        "stage2_config_rel": _first_value(payload, "stage2_config_rel", "stage2_config", "stage2_preset"),
        "config_path": _first_value(payload, "config_path", "stage1_config", "stage1_preset"),
        "preset": _first_value(payload, "preset", "stage1_preset"),
        "timeout": payload.get("timeout"),
        "timeout_stage1": _first_value(payload, "timeout_stage1", "timeout"),
        "timeout_stage2": _first_value(payload, "timeout_stage2", "timeout"),
        "lifecycle": lifecycle,
        **options,
    }

    runner_result = _call_with_supported_kwargs(runner.run, run_kwargs)
    runner_payload = _plain_result(runner_result)
    _raise_if_runner_failed("QuadWild-BiMDF", runner_payload)

    return {
        "operation": "external_quadwild",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "runner_result": runner_payload,
        "notes": [
            "QuadWild-BiMDF completed through Phase 1.5 EXTERNAL_QUADWILD routing.",
            "MeshProcessor must validate and commit the selected output mesh.",
        ],
    }




# Stable metadata copied from hole-fill preview builders into routed task payloads.
# Keep this list deliberately broad: preview builders attach diagnostic values to
# trimesh.metadata, while the GUI reads them from selection_summary,
# ToolPreviewState.metadata, or the preview mesh metadata.
HOLE_FILL_PREVIEW_METADATA_KEYS: tuple[str, ...] = (
    "adaptive_controller",
    "adaptive_controller_schema_version",
    "adaptive_public_method",
    "adaptive_selected_strategy",
    "adaptive_attempted_strategies",
    "adaptive_fallback_used",
    "adaptive_fallback_reason",
    "adaptive_fallback_error",
    "adaptive_primary_seam_decision",
    "adaptive_fallback_seam_decision",
    "adaptive_controller_notes",
    "adaptive_score_delta",
    "adaptive_score_decision",
    "adaptive_selected_score_breakdown",
    "adaptive_selected_score",
    "adaptive_conservative_g1_score_breakdown",
    "adaptive_conservative_g1_score",
    "adaptive_primary_score_breakdown",
    "adaptive_primary_score",
    "adaptive_selected_surface_weight",
    "adaptive_selected_relaxation_strength",
    "adaptive_selected_relaxation_iterations",
    "adaptive_conservative_g1_decision",
    "adaptive_conservative_g1_reason",
    "adaptive_conservative_g1_used",
    "adaptive_conservative_g1_attempted",
    "adaptive_primary_g1_decision",
    "adaptive_g1_policy_reasons",
    "adaptive_g1_relaxation_policy",
    "adaptive_diagnostics",
    "adaptive_context_kind",
    "adaptive_context_confidence",
    "selected_seed_strategy",
    "seam_status",
    "seam_coverage_ratio",
    "seam_missing_edge_count",
    "seam_overused_edge_count",
    "seam_weak_edge_count",
    "seam_problem_edge_count",
    "seam_recovery_required",
    "seam_recovery_strategy",
    "seam_problem_edges",
    "seam_problem_edge_runs",
    "quality_status",
    "relaxation_status",
    "density_status",
    "commit_allowed",
    "commit_blocking_reasons",
    "commit_warnings",
    "adaptive_end_layer_reasons",
    "adaptive_end_layer_selected_patch_source",
    "adaptive_end_layer_rerun_reason",
    "adaptive_end_layer_rerun_allowed",
    "adaptive_end_layer_refinement_recommended",
    "adaptive_end_layer_problem_region_count",
    "adaptive_end_layer_geometry_deviation_max",
    "adaptive_end_layer_geometry_deviation_mean",
    "adaptive_end_layer_curvature_relative_deviation_mean",
    "adaptive_end_layer_curvature_deviation_max",
    "adaptive_end_layer_curvature_deviation_mean",
    "adaptive_end_layer_patch_vertex_count",
    "adaptive_end_layer_support_vertex_count",
    "adaptive_end_layer_support_ring_count",
    "adaptive_end_layer_reference_patch_available",
    "adaptive_end_layer_patch_available",
    "adaptive_end_layer_local_region_available",
    "adaptive_end_layer_action",
    "adaptive_end_layer_status",
    "adaptive_curvature_fairing_trial_movement_to_context_edge_ratio",
    "adaptive_curvature_fairing_trial_mean_displacement",
    "adaptive_curvature_fairing_trial_max_displacement",
    "adaptive_curvature_fairing_trial_error",
    "adaptive_curvature_fairing_trial_notes",
    "adaptive_curvature_fairing_trial_reasons",
    "adaptive_curvature_fairing_trial_mode",
    "adaptive_curvature_fairing_trial_accepted",
    "adaptive_curvature_fairing_trial_applied",
    "adaptive_curvature_fairing_trial_attempted",
    "adaptive_curvature_fairing_trial_action",
    "adaptive_curvature_fairing_trial_status",
    "adaptive_curvature_fairing_reasons",
    "adaptive_curvature_fairing_max_displacement_factor",
    "adaptive_curvature_fairing_iterations",
    "adaptive_curvature_fairing_strength",
    "adaptive_curvature_fairing_needed",
    "adaptive_curvature_fairing_eligible",
    "adaptive_curvature_fairing_action",
    "adaptive_curvature_fairing_status",
    "adaptive_curvature_reasons",
    "adaptive_curvature_patch_sample_count",
    "adaptive_curvature_support_sample_count",
    "adaptive_curvature_sign_consistency",
    "adaptive_curvature_relative_delta_mean",
    "adaptive_curvature_delta_max",
    "adaptive_curvature_delta_mean",
    "adaptive_patch_curvature_std",
    "adaptive_patch_curvature_max",
    "adaptive_patch_curvature_mean",
    "adaptive_support_curvature_std",
    "adaptive_support_curvature_max",
    "adaptive_support_curvature_mean",
    "adaptive_curvature_estimator",
    "adaptive_curvature_context_kind",
    "adaptive_curvature_status",
    "adaptive_feature_support_normal_spread_degrees",
    "adaptive_feature_boundary_vertex_count",
    "adaptive_feature_smooth_context_valid",
    "adaptive_feature_density_mode",
    "adaptive_feature_recommended_action",
    "adaptive_feature_policy_reasons",
    "adaptive_feature_preservation_mode",
    "adaptive_feature_context_kind",
    "g1_gate_support_normal_spread_degrees",
    "g1_gate_boundary_normal_max_deviation_degrees",
    "g1_gate_boundary_normal_mean_deviation_degrees",
    "g1_gate_reasons",
    "g1_gate_status",
    "topology_before",
    "topology_after",
    "quality_before",
    "quality_after",
    "relaxation",
    "dynamic_diagnostics",
    "seam_constraint_report",
    "seam_recovery_decision",
    "constrained_edge_recovery_report",
    "boundary_loop_diagnostic",
    "surface_context_decision",
    "local_density_budget_decision",
    "local_density_refinement_budget",
    "preseed_local_density_budget_decision",
    "preseed_local_density_refinement_budget",
    "seed_max_interior_points",
    "new_face_ids",
    "new_vertex_ids",
    "delaunay_triangle_count",
    "interior_uv_count",
    "biharmonic_fairing_decision",
    "biharmonic_fairing",
    "fit_report",
    "seed_surface_error",
    "seed_surface_kind",
    "preseed_relaxation_confidence_profile",
    "support_context",
    "surface_guidance",
    "seed_backend",
    "relaxation_iterations",
    "relaxation_strength",
    "relaxation_surface_weight",
    "relaxed_max_displacement",
    "relaxed_mean_displacement",
)

ADAPTIVE_SURFACE_V2_METHOD = "adaptive_surface_v2"

ADAPTIVE_SURFACE_METHOD_ALIASES = {
    "adaptive",
    "adaptive_surface",
    "adaptive_surface_fill",
    "adaptive_uvdelaunay_relaxed",
}

ADAPTIVE_SURFACE_V2_METHOD_ALIASES = {
    "adaptive_v2",
    "adaptive_surface_v2",
    "adaptive_surface_fill_v2",
}



def _normalize_hole_fill_method_key(method: object) -> str:
    method_key = str(method or "fan").strip().lower().replace("-", "_")
    if method_key == "triangulate_boundary_fan":
        return "fan"
    if method_key == "open3d_fill":
        return "open3d"
    if method_key in ADAPTIVE_SURFACE_V2_METHOD_ALIASES:
        return ADAPTIVE_SURFACE_V2_METHOD
    if method_key in ADAPTIVE_SURFACE_METHOD_ALIASES:
        return ADAPTIVE_SURFACE_V2_METHOD
    return method_key


def _as_plain_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return dict(result)
        except Exception:
            return {}

    return {}


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _first_mapping_value(mapping: dict[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return default


def _collect_preview_builder_metadata(preview_mesh: trimesh.Trimesh) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_metadata = getattr(preview_mesh, "metadata", None)
    if not isinstance(raw_metadata, dict):
        return metadata

    metadata.update(raw_metadata)
    for nested_key in (
        "hole_fill_preview",
        "preview_builder_metadata",
        "adaptive_diagnostics",
    ):
        nested = raw_metadata.get(nested_key)
        if isinstance(nested, dict):
            metadata.update(nested)

    return metadata


def _adaptive_diagnostics_from_preview_metadata(
    *,
    method: object,
    backend: object,
    metadata: dict[str, Any],
) -> dict[str, object]:
    """Build a small stable diagnostic summary from preview-builder metadata."""

    method_key = _normalize_hole_fill_method_key(method)
    backend_key = str(backend or metadata.get("adaptive_selected_strategy") or "fan")
    public_method = ADAPTIVE_SURFACE_V2_METHOD if method_key == ADAPTIVE_SURFACE_V2_METHOD else method_key

    seam_report = _as_plain_mapping(metadata.get("seam_constraint_report"))
    seam_decision = _as_plain_mapping(metadata.get("seam_recovery_decision"))
    topology_after = _as_plain_mapping(metadata.get("topology_after"))
    quality_after = _as_plain_mapping(metadata.get("quality_after"))
    relaxation = _as_plain_mapping(metadata.get("relaxation"))
    surface_context = _as_plain_mapping(metadata.get("surface_context_decision"))
    boundary_diagnostic = _as_plain_mapping(metadata.get("boundary_loop_diagnostic"))

    missing = _int_or_zero(
        _first_mapping_value(
            seam_decision,
            "missing_seam_edge_count",
            default=_first_mapping_value(
                seam_report,
                "missing_seam_edge_count",
                default=_first_mapping_value(topology_after, "missing_seam_edge_count", default=0),
            ),
        )
    )
    overused = _int_or_zero(
        _first_mapping_value(
            seam_decision,
            "overused_seam_edge_count",
            default=_first_mapping_value(seam_report, "overused_seam_edge_count", default=0),
        )
    )
    weak = _int_or_zero(
        _first_mapping_value(
            seam_decision,
            "weak_seam_edge_count",
            default=_first_mapping_value(seam_report, "weak_seam_edge_count", default=0),
        )
    )
    coverage = _float_or_none(
        _first_mapping_value(
            seam_decision,
            "seam_coverage_ratio",
            default=_first_mapping_value(
                seam_report,
                "seam_coverage_ratio",
                default=_first_mapping_value(topology_after, "seam_coverage_ratio", default=None),
            ),
        )
    )
    recovery_required = bool(
        _first_mapping_value(
            seam_decision,
            "recovery_required",
            default=bool(missing or overused or (coverage is not None and coverage < 0.999)),
        )
    )
    problem_count = _int_or_zero(
        _first_mapping_value(
            seam_decision,
            "problem_edge_count",
            default=missing + overused + weak,
        )
    )

    if missing or overused or (coverage is not None and coverage < 0.999):
        seam_status = "blocked"
    elif weak:
        seam_status = "warning"
    elif seam_report or seam_decision or topology_after:
        seam_status = "ok"
    else:
        seam_status = "unknown"

    degenerate_faces = _int_or_zero(quality_after.get("degenerate_face_count"))
    if degenerate_faces:
        quality_status = "blocked"
    elif quality_after:
        quality_status = "ok"
    else:
        quality_status = "unknown"

    if relaxation:
        relaxation_status = "applied"
    elif metadata.get("relaxation_iterations") is not None:
        relaxation_status = "reported"
    elif public_method == ADAPTIVE_SURFACE_V2_METHOD:
        relaxation_status = "unknown"
    else:
        relaxation_status = "not_applicable"

    density_report = _as_plain_mapping(
        metadata.get("local_density_budget_decision")
        or metadata.get("local_density_refinement_budget")
        or metadata.get("preseed_local_density_budget_decision")
    )
    if density_report:
        density_status = str(
            density_report.get("status")
            or density_report.get("decision")
            or ("ok" if density_report.get("allowed") is not False else "limited")
        )
    else:
        density_status = "unknown"

    context_kind = str(
        _first_mapping_value(
            surface_context,
            "kind",
            "context_kind",
            "surface_kind",
            "decision",
            default=_first_mapping_value(
                boundary_diagnostic,
                "kind",
                "diagnostic_kind",
                default="unknown",
            ),
        )
    )
    context_confidence = _float_or_none(
        _first_mapping_value(
            surface_context,
            "confidence",
            "score",
            "smooth_confidence",
            default=_first_mapping_value(boundary_diagnostic, "confidence", default=None),
        )
    )

    commit_blocking_reasons: list[str] = []
    if missing:
        commit_blocking_reasons.append(f"missing seam edges: {missing}")
    if overused:
        commit_blocking_reasons.append(f"overused seam edges: {overused}")
    if degenerate_faces:
        commit_blocking_reasons.append(f"degenerate patch faces: {degenerate_faces}")

    return {
        "schema_version": 1,
        "public_method": public_method,
        "method": public_method,
        "backend": backend_key,
        "adaptive_stage": backend_key,
        "context_kind": context_kind,
        "context_confidence": context_confidence,
        "selected_seed_strategy": str(metadata.get("adaptive_selected_strategy") or metadata.get("seed_backend") or backend_key),
        "adaptive_controller": metadata.get("adaptive_controller", "-"),
        "adaptive_controller_schema_version": metadata.get("adaptive_controller_schema_version", "-"),
        "adaptive_fallback_used": metadata.get("adaptive_fallback_used", False),
        "adaptive_fallback_reason": metadata.get("adaptive_fallback_reason", "-"),
        "adaptive_fallback_error": metadata.get("adaptive_fallback_error", ""),
        "adaptive_attempted_strategies": metadata.get("adaptive_attempted_strategies", ()),
        "adaptive_primary_seam_decision": metadata.get("adaptive_primary_seam_decision", {}),
        "adaptive_fallback_seam_decision": metadata.get("adaptive_fallback_seam_decision", {}),
        "seam_status": seam_status,
        "seam_missing_edge_count": missing,
        "seam_overused_edge_count": overused,
        "seam_weak_edge_count": weak,
        "seam_problem_edge_count": problem_count,
        "seam_coverage_ratio": coverage,
        "seam_recovery_required": recovery_required,
        "seam_recovery_strategy": str(seam_decision.get("strategy", "none" if not recovery_required else "unknown")),
        "seam_problem_edges": seam_decision.get("problem_edges", seam_report.get("missing_edges", ())),
        "seam_problem_edge_runs": seam_decision.get("problem_edge_runs", ()),
        "quality_status": quality_status,
        "relaxation_status": relaxation_status,
        "density_status": density_status,
        "commit_allowed": not commit_blocking_reasons,
        "commit_blocking_reasons": tuple(commit_blocking_reasons),
        "commit_warnings": tuple(),
        "metadata_sources": tuple(str(key) for key in metadata.keys() if key in HOLE_FILL_PREVIEW_METADATA_KEYS),
    }


def _hole_fill_backend_for_method(method: object) -> str:
    method_key = _normalize_hole_fill_method_key(method)
    if method_key == ADAPTIVE_SURFACE_V2_METHOD:
        return "adaptive_surface_v2_curvature_normal_seed"
    if method_key in {
        "open3d",
        "curvature_sphere",
        "curvature_sphere_refined",
        "curvature_sphere_grid8",
        "curvature_sphere_uvgrid",
        "curvature_sphere_uvdelaunay",
        "curvature_sphere_uvdelaunay_relaxed",
        "surface_uvdelaunay_relaxed",
        "surface_uvdelaunay_sealed_relaxed",
        "surface_uvdelaunay_sealed_dense_relaxed",
    }:
        return method_key
    return "fan"


def _normalize_hole_fill_selection_summary(
    summary: dict[str, object],
    *,
    requested_method: object,
) -> dict[str, object]:
    normalized = dict(summary)
    method_key = _normalize_hole_fill_method_key(
        normalized.get("method") or requested_method or "fan"
    )
    normalized["method"] = method_key
    normalized.setdefault("public_method", method_key)
    normalized["backend"] = str(normalized.get("backend") or _hole_fill_backend_for_method(method_key))


    if method_key == ADAPTIVE_SURFACE_V2_METHOD:
        selected_strategy = str(
            normalized.get("adaptive_selected_strategy")
            or normalized.get("selected_seed_strategy")
            or normalized.get("backend")
            or "adaptive_surface_v2_curvature_normal_seed"
        )
        normalized["method"] = ADAPTIVE_SURFACE_V2_METHOD
        normalized["public_method"] = ADAPTIVE_SURFACE_V2_METHOD
        normalized["backend"] = selected_strategy
        normalized["adaptive_stage"] = selected_strategy
        normalized["selected_seed_strategy"] = selected_strategy

    return normalized


def _handle_hole_fill_preview(payload: TaskPayload) -> TaskPayload:
    """
    Build a non-destructive Phase 2 hole-fill preview from explicit payload data.

    Expected payload:
        {
            "mesh": trimesh.Trimesh,
            "candidate": optional HoleCandidate,
            "hole_candidate": optional HoleCandidate,
            "candidate_index": optional int,
            "index": optional int,
            "method": optional str,
            "fill_method": optional str,
            "face_ids": optional list[int],
            "max_area_hint": optional float,
            "max_perimeter": optional float,
            "storage_dir": optional path,
            "preview_dir": optional path,
            "operation_id": optional str,
            "write_snapshots": optional bool,
        }

    Rules:
    - does not mutate source mesh
    - does not mutate MeshProcessor.mesh
    - does not import GUI or viewport code
    - returns preview payload for MeshProcessor to validate and expose
    - writes ToolPreviewState only when storage_dir/preview_dir is provided
    """

    mesh = _require_mesh(payload)

    if bool(payload.get("low_memory_hole_fill", False)):
        return _handle_low_memory_hole_fill_preview(payload, mesh)

    base_mesh = mesh.copy()

    candidate, candidate_index = _hole_candidate_from_payload(payload, mesh)
    method = str(_first_value(payload, "method", "fill_method") or "fan")
    normalized_method = _normalize_hole_fill_method_key(method)

    all_candidates = None
    if normalized_method in {
        "open3d",
        ADAPTIVE_SURFACE_V2_METHOD,
    }:
        all_candidates = _hole_candidates_from_payload(payload, mesh)

    raw_preview_mesh = build_hole_fill_preview_mesh(
        mesh,
        candidate,
        method=method,
        all_candidates=all_candidates,
    )

    preview_builder_metadata: dict[str, Any] = _collect_preview_builder_metadata(raw_preview_mesh)

    preview_mesh = _require_mesh({"mesh": raw_preview_mesh})
    preview_mesh = preview_mesh.copy()
    if preview_builder_metadata:
        preview_mesh.metadata.update(preview_builder_metadata)
    preview_mesh.remove_unreferenced_vertices()

    patch_mesh, patch_face_ids, patch_vertex_ids, new_face_ids, new_vertex_ids = (
        _extract_appended_patch_mesh(base_mesh, preview_mesh)
    )

    operation_id = str(
        _first_value(payload, "operation_id", "preview_id")
        or f"hole_fill_preview_{candidate_index}"
    )

    backend = str(
        preview_builder_metadata.get("adaptive_selected_strategy")
        or preview_builder_metadata.get("backend")
        or _hole_fill_backend_for_method(normalized_method)
    )

    selection_summary = {
        "mode": "hole_boundary",
        "candidate_index": int(candidate_index),
        "method": method,
        "backend": backend,
        "selected_faces": 0,
        "selected_vertices": int(len(candidate.boundary_vertices)),
        "boundary_vertices": int(len(candidate.boundary_vertices)),
        "boundary_edges": int(len(candidate.boundary_edges)),
        "patch_faces": int(len(patch_face_ids)),
        "patch_vertices": int(len(patch_vertex_ids)),
        "new_faces": int(len(new_face_ids)),
        "new_vertices": int(len(new_vertex_ids)),
    }
    if normalized_method == ADAPTIVE_SURFACE_V2_METHOD:
        selection_summary["method"] = ADAPTIVE_SURFACE_V2_METHOD
        selection_summary["public_method"] = ADAPTIVE_SURFACE_V2_METHOD
        selection_summary["backend"] = backend
        selection_summary["adaptive_stage"] = backend
        selection_summary["selected_seed_strategy"] = str(
            preview_builder_metadata.get("adaptive_selected_strategy") or backend
        )

    diagnostics = _adaptive_diagnostics_from_preview_metadata(
        method=normalized_method,
        backend=backend,
        metadata=preview_builder_metadata,
    )
    preview_builder_metadata.setdefault("adaptive_diagnostics", diagnostics)

    for key in HOLE_FILL_PREVIEW_METADATA_KEYS:
        if key in preview_builder_metadata:
            selection_summary[key] = _plain_result(preview_builder_metadata[key])

    for key, value in preview_builder_metadata.items():
        key_text = str(key)
        if key_text.startswith("adaptive_"):
            selection_summary[key_text] = _plain_result(value)

    selection_summary.update(
        {
            "adaptive_diagnostics": _plain_result(diagnostics),
            "adaptive_context_kind": diagnostics.get("context_kind"),
            "adaptive_context_confidence": diagnostics.get("context_confidence"),
            "selected_seed_strategy": diagnostics.get("selected_seed_strategy"),
            "adaptive_controller": diagnostics.get("adaptive_controller"),
            "adaptive_controller_schema_version": diagnostics.get("adaptive_controller_schema_version"),
            "adaptive_fallback_used": diagnostics.get("adaptive_fallback_used"),
            "adaptive_fallback_reason": diagnostics.get("adaptive_fallback_reason"),
            "adaptive_fallback_error": diagnostics.get("adaptive_fallback_error"),
            "adaptive_attempted_strategies": _plain_result(diagnostics.get("adaptive_attempted_strategies")),
            "adaptive_primary_seam_decision": _plain_result(diagnostics.get("adaptive_primary_seam_decision")),
            "adaptive_fallback_seam_decision": _plain_result(diagnostics.get("adaptive_fallback_seam_decision")),
            "seam_status": diagnostics.get("seam_status"),
            "seam_coverage_ratio": diagnostics.get("seam_coverage_ratio"),
            "seam_missing_edge_count": diagnostics.get("seam_missing_edge_count"),
            "seam_overused_edge_count": diagnostics.get("seam_overused_edge_count"),
            "seam_weak_edge_count": diagnostics.get("seam_weak_edge_count"),
            "seam_problem_edge_count": diagnostics.get("seam_problem_edge_count"),
            "seam_recovery_required": diagnostics.get("seam_recovery_required"),
            "seam_recovery_strategy": diagnostics.get("seam_recovery_strategy"),
            "seam_problem_edges": _plain_result(diagnostics.get("seam_problem_edges")),
            "seam_problem_edge_runs": _plain_result(diagnostics.get("seam_problem_edge_runs")),
            "quality_status": diagnostics.get("quality_status"),
            "relaxation_status": diagnostics.get("relaxation_status"),
            "density_status": diagnostics.get("density_status"),
            "commit_allowed": diagnostics.get("commit_allowed"),
            "commit_blocking_reasons": _plain_result(diagnostics.get("commit_blocking_reasons")),
            "commit_warnings": _plain_result(diagnostics.get("commit_warnings")),
        }
    )


    selection_summary = _normalize_hole_fill_selection_summary(
        selection_summary,
        requested_method=method,
    )

    selection_summary = _strict_adaptive_surface_commit_policy(selection_summary, preview_builder_metadata)

    notes = [
        "Hole fill preview created through Phase 1.5 HOLE_FILL_PREVIEW routing.",
        "Preview is non-destructive; MeshProcessor remains responsible for commit.",
    ]
    if backend == "open3d":
        notes.append("Open3D tensor TriangleMesh.fill_holes backend used for preview.")
    if normalized_method == ADAPTIVE_SURFACE_V2_METHOD:
        notes.append(
            "Adaptive Surface Fill v2 controller used: "
            f"{selection_summary.get('adaptive_controller', '-')} | "
            f"selected_strategy={selection_summary.get('selected_seed_strategy', '-')} | "
            f"fallback_used={selection_summary.get('adaptive_fallback_used', '-')}"
        )

    tool_preview_state: ToolPreviewState | None = None
    preview_state_path: str | None = None
    tool_preview_state_dict: dict[str, Any] | None = None

    storage_dir_value = _first_value(payload, "storage_dir", "preview_dir", "tool_state_dir")
    write_snapshots_requested = bool(payload.get("write_snapshots", False))

    if storage_dir_value is None and write_snapshots_requested:
        raise ValueError("write_snapshots=True requires storage_dir or preview_dir.")

    if storage_dir_value is not None:
        storage_dir = Path(str(storage_dir_value)).expanduser().resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)

        base_snapshot = MeshSnapshot.capture(
            base_mesh,
            storage_dir,
            role=SNAPSHOT_ROLE_BASE,
            name="base_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": operation_id,
            },
        )
        preview_snapshot = MeshSnapshot.capture(
            preview_mesh,
            storage_dir,
            role=SNAPSHOT_ROLE_PREVIEW,
            name="preview_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": operation_id,
                "method": method,
                "backend": backend,
            },
        )
        patch_snapshot = MeshSnapshot.capture(
            patch_mesh,
            storage_dir,
            role=SNAPSHOT_ROLE_PATCH,
            name="patch_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": operation_id,
                "method": method,
                "backend": backend,
            },
        )

        boundary_region = ToolRegion(
            name="Hole boundary",
            kind=REGION_KIND_HOLE_BOUNDARY,
            mesh_snapshot=None,
            face_ids=(),
            vertex_ids=_candidate_boundary_vertices(candidate),
            edge_ids=_candidate_boundary_edges(candidate),
            source="find_hole_candidates",
            metadata=_candidate_metadata(candidate),
        )

        patch_region = ToolRegion(
            name="Generated hole-fill patch",
            kind=REGION_KIND_HOLE_PATCH,
            mesh_snapshot=patch_snapshot,
            face_ids=patch_face_ids,
            vertex_ids=patch_vertex_ids,
            new_face_ids=new_face_ids,
            new_vertex_ids=new_vertex_ids,
            source="hole_fill_preview",
            metadata={
                "method": selection_summary.get("method", method),
                "public_method": selection_summary.get("public_method", selection_summary.get("method", method)),
                "backend": backend,
                "adaptive_stage": selection_summary.get("adaptive_stage", backend),
                "adaptive_controller": selection_summary.get("adaptive_controller"),
                "adaptive_selected_strategy": selection_summary.get("selected_seed_strategy"),
            },
        )

        markers = (HOLE_FILL_PREVIEW_MARKER, LOCAL_REGION_OPERATION_MARKER)
        if backend == "open3d":
            markers = markers + (OPEN3D_FILL_PREVIEW_MARKER,)

        tool_state_metadata = {
            "candidate_index": int(candidate_index),
            "requested_method": method,
            "public_method": selection_summary.get("public_method", selection_summary.get("method", method)),
            "method": selection_summary.get("method", method),
            "backend": backend,
            "adaptive_stage": selection_summary.get("adaptive_stage", backend),
            "commit_allowed": bool(selection_summary.get("commit_allowed", True)),
            "commit_blocking_reasons": _plain_result(selection_summary.get("commit_blocking_reasons", ())),
            "commit_warnings": _plain_result(selection_summary.get("commit_warnings", ())),
        }

        for key in HOLE_FILL_PREVIEW_METADATA_KEYS:
            if key in preview_builder_metadata:
                tool_state_metadata[key] = _plain_result(preview_builder_metadata[key])
            elif key in selection_summary:
                tool_state_metadata[key] = _plain_result(selection_summary[key])

        for key, value in preview_builder_metadata.items():
            key_text = str(key)
            if key_text.startswith("adaptive_"):
                tool_state_metadata[key_text] = _plain_result(value)

        tool_preview_state = ToolPreviewState(
            operation_id=operation_id,
            operation="hole_fill_preview",
            base_snapshot=base_snapshot,
            preview_snapshot=preview_snapshot,
            input_regions=(boundary_region,),
            output_regions=(patch_region,),
            committable=bool(selection_summary.get("commit_allowed", True)),
            markers=markers,
            notes=tuple(notes),
            metadata=tool_state_metadata,
        )

        json_path = storage_dir / "preview_state.json"
        tool_preview_state.write_json(json_path, base_dir=storage_dir)
        preview_state_path = str(json_path)
        tool_preview_state_dict = tool_preview_state.to_dict(base_dir=storage_dir)
        notes.append(f"ToolPreviewState written: {json_path}")

    return {
        "implemented": True,
        "operation": "hole_fill_preview",
        "preview_mesh": preview_mesh,
        "base_mesh": base_mesh,
        "patch_mesh": patch_mesh,
        "selection_summary": selection_summary,
        "notes": notes,
        "candidate": candidate,
        "candidate_index": int(candidate_index),
        "method": method,
        "backend": backend,
        "patch_face_ids": list(patch_face_ids),
        "patch_vertex_ids": list(patch_vertex_ids),
        "new_face_ids": list(new_face_ids),
        "new_vertex_ids": list(new_vertex_ids),
        "tool_preview_state": tool_preview_state,
        "tool_preview_state_dict": tool_preview_state_dict,
        "preview_state_path": preview_state_path,
        "adaptive_diagnostics": _plain_result(diagnostics),
        "adaptive_controller": _plain_result(selection_summary.get("adaptive_controller")),
        "adaptive_selected_strategy": _plain_result(selection_summary.get("selected_seed_strategy")),
        "adaptive_attempted_strategies": _plain_result(selection_summary.get("adaptive_attempted_strategies")),
        "adaptive_fallback_used": _plain_result(selection_summary.get("adaptive_fallback_used")),
        "adaptive_fallback_reason": _plain_result(selection_summary.get("adaptive_fallback_reason")),
        "adaptive_fallback_error": _plain_result(selection_summary.get("adaptive_fallback_error")),
        "adaptive_primary_seam_decision": _plain_result(selection_summary.get("adaptive_primary_seam_decision")),
        "adaptive_fallback_seam_decision": _plain_result(selection_summary.get("adaptive_fallback_seam_decision")),
        "commit_allowed": bool(selection_summary.get("commit_allowed", True)),
        "commit_blocking_reasons": _plain_result(selection_summary.get("commit_blocking_reasons", ())),
        "commit_warnings": _plain_result(selection_summary.get("commit_warnings", ())),
        "preview_builder_metadata": _plain_result(preview_builder_metadata),
    }



def _ordered_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(v) for v in np.asarray(value, dtype=np.int64).reshape(-1).tolist())


def _closed_edges_from_ordered_vertices(vertex_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(vertex_ids) < 2:
        return ()

    edges: list[tuple[int, int]] = []
    for index, a in enumerate(vertex_ids):
        b = vertex_ids[(index + 1) % len(vertex_ids)]
        if int(a) == int(b):
            continue
        edges.append((int(a), int(b)))
    return tuple(edges)


def _low_memory_target_boundary_candidate(payload: TaskPayload) -> tuple[Any, int, tuple[Any, ...]]:
    """Return a synthetic local candidate for the original target boundary.

    Low-memory work units introduce artificial cut boundaries around the support
    band.  Re-running ``find_hole_candidates`` on the local context can therefore
    discover several local holes and may fail to produce a candidate whose
    boundary set exactly equals the original hole boundary.  The target boundary
    is already known and ordered by MeshProcessor, so the safe low-memory path is
    to build a small candidate proxy from that explicit boundary instead of
    rediscovering it from the cut work-unit mesh.
    """

    target_ids = _ordered_int_tuple(payload.get("target_boundary_local_vertex_ids"))
    if len(target_ids) < 3:
        raise ValueError("low-memory hole fill requires at least three target boundary vertices")

    # Preserve the ordered target loop.  The preview builders prefer
    # candidate.loop.vertices when present; keep it open, matching the normal
    # HoleCandidate boundary_vertices convention.
    loop = SimpleNamespace(vertices=target_ids)

    area_hint = payload.get("target_area_hint")
    perimeter = payload.get("target_perimeter")
    fill_priority = payload.get("target_fill_priority")
    centroid = payload.get("target_centroid")

    try:
        area_hint_value = None if area_hint is None else float(area_hint)
    except Exception:
        area_hint_value = None
    try:
        perimeter_value = float(perimeter) if perimeter is not None else 0.0
    except Exception:
        perimeter_value = 0.0
    try:
        fill_priority_value = float(fill_priority) if fill_priority is not None else 0.0
    except Exception:
        fill_priority_value = 0.0

    centroid_value = None
    if centroid is not None:
        try:
            centroid_array = np.asarray(centroid, dtype=float).reshape(-1)
            if centroid_array.size >= 3:
                centroid_value = centroid_array[:3].copy()
        except Exception:
            centroid_value = None

    candidate = SimpleNamespace(
        boundary_vertices=target_ids,
        boundary_edges=_closed_edges_from_ordered_vertices(target_ids),
        loop=loop,
        classified_loop=None,
        perimeter=perimeter_value,
        area_hint=area_hint_value,
        fill_priority=fill_priority_value,
        centroid=centroid_value,
    )

    return candidate, 0, (candidate,)


def _encode_low_memory_patch_faces_for_commit(
    *,
    local_preview_mesh: trimesh.Trimesh,
    local_base_mesh: trimesh.Trimesh,
    local_to_source_vertex_ids: tuple[int, ...],
) -> tuple[list[list[int]], list[list[float]]]:
    """Encode local patch faces for source-mesh commit.

    Existing source vertices are encoded as their global source vertex id.
    Generated preview vertices are encoded as negative placeholders:

        -1 -> generated_vertices[0]
        -2 -> generated_vertices[1]

    MeshProcessor later resolves those placeholders to appended source-mesh
    vertex ids during commit.  This keeps the worker result patch-only and
    avoids returning a full merged mesh for large inputs.
    """

    base_vertex_count = int(len(local_base_mesh.vertices))
    base_face_count = int(len(local_base_mesh.faces))
    preview_vertices = np.asarray(local_preview_mesh.vertices, dtype=float)
    preview_faces = np.asarray(local_preview_mesh.faces, dtype=np.int64)

    if preview_faces.ndim != 2 or preview_faces.shape[1] < 3:
        raise ValueError("low-memory preview mesh does not contain triangular faces")
    if int(len(preview_faces)) <= base_face_count:
        raise ValueError("low-memory hole fill preview did not append patch faces")

    patch_faces_local = preview_faces[base_face_count:, :3]
    generated_ids = sorted(
        int(v)
        for v in np.unique(patch_faces_local.reshape(-1)).tolist()
        if int(v) >= base_vertex_count
    )
    generated_ordinal = {vertex_id: index for index, vertex_id in enumerate(generated_ids)}
    generated_vertices = [
        [float(x) for x in preview_vertices[vertex_id, :3].tolist()]
        for vertex_id in generated_ids
    ]

    encoded_faces: list[list[int]] = []
    for raw_face in patch_faces_local:
        encoded: list[int] = []
        for raw_vertex_id in raw_face[:3]:
            vertex_id = int(raw_vertex_id)
            if vertex_id < base_vertex_count:
                if vertex_id < 0 or vertex_id >= len(local_to_source_vertex_ids):
                    raise ValueError(
                        "low-memory patch references a local source vertex without a source mapping"
                    )
                encoded.append(int(local_to_source_vertex_ids[vertex_id]))
            else:
                encoded.append(-(generated_ordinal[vertex_id] + 1))
        encoded_faces.append(encoded)

    return encoded_faces, generated_vertices


def _handle_low_memory_hole_fill_preview(payload: TaskPayload, mesh: trimesh.Trimesh) -> TaskPayload:
    """Build a patch-only low-memory hole-fill preview on a local work unit.

    The payload mesh is expected to be a local N-ring work-unit mesh, not the
    full application mesh.  The result intentionally returns a patch overlay and
    commit mapping, not a full merged preview mesh.
    """

    base_mesh = mesh.copy()
    candidate, local_candidate_index, all_candidates = _low_memory_target_boundary_candidate(payload)

    method = str(_first_value(payload, "method", "fill_method") or "fan")
    normalized_method = _normalize_hole_fill_method_key(method)

    raw_preview_mesh = build_hole_fill_preview_mesh(
        mesh,
        candidate,
        method=method,
        all_candidates=all_candidates,
    )

    preview_builder_metadata: dict[str, Any] = _collect_preview_builder_metadata(raw_preview_mesh)
    local_preview_mesh = _require_mesh({"mesh": raw_preview_mesh})
    local_preview_mesh = local_preview_mesh.copy()
    if preview_builder_metadata:
        local_preview_mesh.metadata.update(preview_builder_metadata)
    local_preview_mesh.remove_unreferenced_vertices()

    patch_mesh, patch_face_ids, patch_vertex_ids, new_face_ids, new_vertex_ids = (
        _extract_appended_patch_mesh(base_mesh, local_preview_mesh)
    )

    local_to_source = tuple(
        int(v)
        for v in np.asarray(payload.get("local_to_source_vertex_ids"), dtype=np.int64).reshape(-1).tolist()
    )
    if len(local_to_source) != int(len(base_mesh.vertices)):
        raise ValueError("low-memory local_to_source_vertex_ids does not match local mesh vertices")

    patch_faces_source, generated_vertices = _encode_low_memory_patch_faces_for_commit(
        local_preview_mesh=local_preview_mesh,
        local_base_mesh=base_mesh,
        local_to_source_vertex_ids=local_to_source,
    )

    operation_id = str(
        _first_value(payload, "operation_id", "preview_id")
        or f"hole_fill_preview_low_memory_{local_candidate_index}"
    )

    backend = str(
        preview_builder_metadata.get("adaptive_selected_strategy")
        or preview_builder_metadata.get("backend")
        or _hole_fill_backend_for_method(normalized_method)
    )

    diagnostics = _adaptive_diagnostics_from_preview_metadata(
        method=normalized_method,
        backend=backend,
        metadata=preview_builder_metadata,
    )
    preview_builder_metadata.setdefault("adaptive_diagnostics", diagnostics)

    source_face_count = int(payload.get("source_mesh_face_count") or 0)
    source_vertex_count = int(payload.get("source_mesh_vertex_count") or 0)
    original_candidate_index = int(payload.get("source_candidate_index") or payload.get("candidate_index") or 0)
    boundary_source = str(payload.get("boundary_source") or "candidate_boundary")
    selected_edge_ids = tuple(
        int(v)
        for v in np.asarray(payload.get("selected_edge_ids") or (), dtype=np.int64).reshape(-1).tolist()
    )

    target_source_boundary = tuple(
        int(v)
        for v in np.asarray(payload.get("target_boundary_source_vertex_ids"), dtype=np.int64).reshape(-1).tolist()
    )
    target_source_edges = tuple(
        tuple(int(x) for x in edge)
        for edge in (payload.get("target_boundary_source_edges") or ())
        if len(edge) == 2
    )

    selection_summary: dict[str, Any] = {
        "mode": "selected_edges_boundary" if boundary_source == "selected_edges" else "hole_boundary",
        "scope": "selected_edges" if boundary_source == "selected_edges" else "low_memory_local_work_unit",
        "boundary_source": boundary_source,
        "selected_edges": int(len(selected_edge_ids)),
        "selected_edge_ids": list(selected_edge_ids),
        "candidate_index": original_candidate_index,
        "local_candidate_index": int(local_candidate_index),
        "method": method,
        "backend": backend,
        "low_memory_patch_only": True,
        "low_memory_work_unit": True,
        "source_mesh_face_count": source_face_count,
        "source_mesh_vertex_count": source_vertex_count,
        "local_context_face_count": int(len(base_mesh.faces)),
        "local_context_vertex_count": int(len(base_mesh.vertices)),
        "selected_faces": 0,
        "selected_vertices": int(len(target_source_boundary)),
        "boundary_vertices": int(len(target_source_boundary)),
        "boundary_edges": int(len(target_source_edges)),
        "target_boundary_source_vertex_ids": list(target_source_boundary),
        "target_boundary_source_edges": [list(edge) for edge in target_source_edges],
        "patch_faces": int(len(patch_face_ids)),
        "patch_vertices": int(len(patch_vertex_ids)),
        "new_faces": int(len(new_face_ids)),
        "new_vertices": int(len(generated_vertices)),
        "low_memory_patch_faces_source": patch_faces_source,
        "low_memory_generated_vertices": generated_vertices,
        "commit_allowed": diagnostics.get("commit_allowed", True),
        "commit_blocking_reasons": _plain_result(diagnostics.get("commit_blocking_reasons", ())),
        "commit_warnings": _plain_result(diagnostics.get("commit_warnings", ())),
    }

    if normalized_method == ADAPTIVE_SURFACE_V2_METHOD:
        selection_summary["method"] = ADAPTIVE_SURFACE_V2_METHOD
        selection_summary["public_method"] = ADAPTIVE_SURFACE_V2_METHOD
        selection_summary["adaptive_stage"] = backend
        selection_summary["selected_seed_strategy"] = str(
            preview_builder_metadata.get("adaptive_selected_strategy") or backend
        )

    for key in HOLE_FILL_PREVIEW_METADATA_KEYS:
        if key in preview_builder_metadata:
            selection_summary[key] = _plain_result(preview_builder_metadata[key])

    for key, value in preview_builder_metadata.items():
        key_text = str(key)
        if key_text.startswith("adaptive_"):
            selection_summary[key_text] = _plain_result(value)

    selection_summary.update(
        {
            "adaptive_diagnostics": _plain_result(diagnostics),
            "adaptive_context_kind": diagnostics.get("context_kind"),
            "adaptive_context_confidence": diagnostics.get("context_confidence"),
            "selected_seed_strategy": diagnostics.get("selected_seed_strategy"),
            "adaptive_controller": diagnostics.get("adaptive_controller"),
            "adaptive_controller_schema_version": diagnostics.get("adaptive_controller_schema_version"),
            "adaptive_fallback_used": diagnostics.get("adaptive_fallback_used"),
            "adaptive_fallback_reason": diagnostics.get("adaptive_fallback_reason"),
            "adaptive_fallback_error": diagnostics.get("adaptive_fallback_error"),
            "adaptive_attempted_strategies": _plain_result(diagnostics.get("adaptive_attempted_strategies")),
            "seam_status": diagnostics.get("seam_status"),
            "seam_coverage_ratio": diagnostics.get("seam_coverage_ratio"),
            "quality_status": diagnostics.get("quality_status"),
            "relaxation_status": diagnostics.get("relaxation_status"),
            "density_status": diagnostics.get("density_status"),
        }
    )

    selection_summary = _normalize_hole_fill_selection_summary(
        selection_summary,
        requested_method=method,
    )
    selection_summary = _strict_adaptive_surface_commit_policy(selection_summary, preview_builder_metadata)
    # Preserve the low-memory commit mapping after normalizers/policy helpers.
    selection_summary["low_memory_patch_only"] = True
    selection_summary["low_memory_work_unit"] = True
    selection_summary["low_memory_patch_faces_source"] = patch_faces_source
    selection_summary["low_memory_generated_vertices"] = generated_vertices
    selection_summary["source_mesh_face_count"] = source_face_count
    selection_summary["source_mesh_vertex_count"] = source_vertex_count
    selection_summary["target_boundary_source_vertex_ids"] = list(target_source_boundary)
    selection_summary["target_boundary_source_edges"] = [list(edge) for edge in target_source_edges]

    notes = [
        "Hole fill preview created through Phase 1.5 HOLE_FILL_PREVIEW routing.",
        (
            "Low-memory selected-edge boundary path used; preview payload is patch-only."
            if boundary_source == "selected_edges"
            else "Low-memory local work-unit path used; preview payload is patch-only."
        ),
        "Preview is non-destructive; MeshProcessor remains responsible for commit.",
    ]
    if normalized_method == ADAPTIVE_SURFACE_V2_METHOD:
        notes.append(
            "Adaptive Surface Fill v2 controller used on local work unit: "
            f"{selection_summary.get('adaptive_controller', '-')} | "
            f"selected_strategy={selection_summary.get('selected_seed_strategy', '-')} | "
            f"fallback_used={selection_summary.get('adaptive_fallback_used', '-')}"
        )

    storage_dir_value = _first_value(payload, "storage_dir", "preview_dir", "tool_state_dir")
    preview_state_path: str | None = None
    if storage_dir_value is not None:
        storage_dir = Path(str(storage_dir_value)).expanduser().resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
        patch_path = storage_dir / "patch_mesh.ply"
        patch_mesh.export(str(patch_path))
        mapping_path = storage_dir / "low_memory_patch_mapping.json"
        mapping_path.write_text(
            _json_dumps_stable(
                {
                    "operation": "hole_fill_preview",
                    "operation_id": operation_id,
                    "low_memory_patch_only": True,
                    "patch_mesh_path": str(patch_path),
                    "patch_faces_source": patch_faces_source,
                    "generated_vertices": generated_vertices,
                    "source_mesh_face_count": source_face_count,
                    "source_mesh_vertex_count": source_vertex_count,
                    "target_boundary_source_vertex_ids": list(target_source_boundary),
                    "target_boundary_source_edges": [list(edge) for edge in target_source_edges],
                }
            ),
            encoding="utf-8",
        )
        selection_summary["patch_mesh_path"] = str(patch_path)
        selection_summary["low_memory_patch_mapping_path"] = str(mapping_path)
        preview_state_path = str(mapping_path)
        notes.append(f"Low-memory patch mapping written: {mapping_path}")

    return {
        "implemented": True,
        "operation": "hole_fill_preview",
        "low_memory_patch_only": True,
        "preview_mesh": patch_mesh,
        "base_mesh": base_mesh,
        "patch_mesh": patch_mesh,
        "selection_summary": selection_summary,
        "notes": notes,
        "candidate": candidate,
        "candidate_index": original_candidate_index,
        "method": method,
        "backend": backend,
        "patch_face_ids": list(patch_face_ids),
        "patch_vertex_ids": list(patch_vertex_ids),
        "new_face_ids": list(new_face_ids),
        "new_vertex_ids": list(new_vertex_ids),
        "tool_preview_state": None,
        "tool_preview_state_dict": None,
        "preview_state_path": preview_state_path,
        "adaptive_diagnostics": _plain_result(diagnostics),
        "commit_allowed": bool(selection_summary.get("commit_allowed", True)),
        "commit_blocking_reasons": _plain_result(selection_summary.get("commit_blocking_reasons", ())),
        "commit_warnings": _plain_result(selection_summary.get("commit_warnings", ())),
        "preview_builder_metadata": _plain_result(preview_builder_metadata),
    }


def _json_dumps_stable(value: Any) -> str:
    import json

    return json.dumps(_plain_result(value), indent=2, sort_keys=True)

def _handle_bore_region_extract(payload: TaskPayload) -> TaskPayload:
    """Run BoreTool RegionData/Recognition candidate extraction in a core task.

    This handler is PROCESS-safe: it receives an explicit mesh copy and selected
    edge IDs, performs the same core BoreTool analysis used by the direct GUI
    path, and returns a plain dictionary payload. It does not mutate
    MeshProcessor state, selection state, project history, or the viewport.
    """

    mesh = _require_mesh(payload).copy()
    selected_edge_ids = _bore_selected_edge_ids_from_payload(payload)

    from far_mesh.core.bore.tool import analyze_bore_candidates

    display_result = analyze_bore_candidates(mesh, selected_edge_ids)
    return _bore_display_result_to_payload(display_result)


def _handle_bore_clean_preview(payload: TaskPayload) -> TaskPayload:
    """Compatibility alias for the old Bore clean-preview socket.

    The modern rebuild execution path is BORE_REBUILD_CANDIDATE. The legacy
    socket remains useful for callers/tests that still request a candidate
    extraction-style Bore preview through the old task kind.
    """

    payload_out = _handle_bore_region_extract(payload)
    payload_out["operation"] = "bore_clean_preview"
    payload_out.setdefault(
        "notes",
        [],
    ).append("BORE_CLEAN_PREVIEW routed through Bore region extraction compatibility path.")
    return payload_out


def _handle_bore_rebuild_candidate(payload: TaskPayload) -> TaskPayload:
    """Run an explicit Bore candidate rebuild in a spawned/core-safe task.

    This handler is a transport/execution adapter only. It must not re-run Bore
    recognition, reselect candidates by index, promote/reject candidates, or
    reinterpret CandidateData. The selected candidate payload was produced by
    the BoreTool facade; ``rebuild_bore_candidate`` owns candidate-view
    normalization and rebuild gating. MeshProcessor remains responsible for
    validating and committing the returned mesh in the parent process.
    """

    mesh = _require_mesh(payload).copy()
    selected_edge_ids = _bore_selected_edge_ids_from_payload(payload)

    raw_candidate = _first_value(payload, "candidate", "candidate_metadata", "candidate_view")
    if not isinstance(raw_candidate, dict):
        raise ValueError(
            "BORE_REBUILD_CANDIDATE requires an explicit CandidateView/CandidateData payload. "
            "The task handler coordinates execution only and does not re-run Bore recognition "
            "or select candidates by index."
        )
    candidate_payload: dict[str, Any] = dict(raw_candidate)

    from far_mesh.core.bore.tool import rebuild_bore_candidate

    quad_density_mode = str(payload.get("quad_density_mode") or payload.get("rebuild_density_mode") or "lean_pi_opening")
    color_rebuilt_faces = bool(payload.get("color_rebuilt_faces", True))
    rebuilt_face_color = _optional_rgba_tuple(payload.get("rebuilt_face_color"))

    rebuild_kwargs: dict[str, Any] = {
        "edge_ids": selected_edge_ids,
        "candidate": candidate_payload,
        "quad_density_mode": quad_density_mode,
        "color_rebuilt_faces": color_rebuilt_faces,
    }
    if rebuilt_face_color is not None:
        rebuild_kwargs["rebuilt_face_color"] = rebuilt_face_color

    rebuild_result = rebuild_bore_candidate(mesh, **rebuild_kwargs)
    diagnostics = dict(getattr(rebuild_result, "diagnostics", {}) or {})
    diagnostics.setdefault("execution_layer", "process_task")
    diagnostics.setdefault("task_kind", "bore_rebuild_candidate")
    diagnostics.setdefault("candidate_authority", "explicit_payload_candidate_view")
    diagnostics.setdefault("candidate_selection_policy", "no_task_handler_reanalysis_no_index_selection")
    diagnostics.setdefault("candidate_id", str(candidate_payload.get("candidate_id") or candidate_payload.get("feature_id") or ""))
    if "source_candidate_index_hint" in payload:
        diagnostics.setdefault("source_candidate_index_hint", int(payload.get("source_candidate_index_hint") or 0))

    normalized_edge_ids = payload.get("normalized_edge_ids")
    if normalized_edge_ids is None:
        normalized_edge_ids = selected_edge_ids

    out: TaskPayload = {
        "implemented": True,
        "operation": "bore_rebuild_candidate",
        "mesh": getattr(rebuild_result, "mesh", None),
        "removed_face_ids": _plain_result(getattr(rebuild_result, "removed_face_ids", ())),
        "added_face_ids": _plain_result(getattr(rebuild_result, "added_face_ids", ())),
        "added_faces": _plain_result(getattr(rebuild_result, "added_faces", ())),
        "loop0_vertices": _plain_result(getattr(rebuild_result, "loop0_vertices", ())),
        "loop1_vertices": _plain_result(getattr(rebuild_result, "loop1_vertices", ())),
        "axis": _plain_result(getattr(rebuild_result, "axis", (0.0, 0.0, 1.0))),
        "radius": float(getattr(rebuild_result, "radius", 0.0) or 0.0),
        "diagnostics": _plain_result(diagnostics),
        "candidate": _plain_result(candidate_payload),
        "candidate_metadata": _plain_result(candidate_payload),
        "selected_edge_ids": list(selected_edge_ids),
        "normalized_edge_ids": _plain_result(normalized_edge_ids),
        "quad_density_mode": quad_density_mode,
        "notes": [
            "Bore candidate rebuild computed through Phase 6 BORE_REBUILD_CANDIDATE task.",
            "Task handler used explicit CandidateView payload; no Bore recognition reanalysis or candidate-index selection was performed.",
            "MeshProcessor must validate and commit the returned mesh in the parent process.",
        ],
    }
    if "source_candidate_index_hint" in payload:
        out["source_candidate_index_hint"] = int(payload.get("source_candidate_index_hint") or 0)
    return out


def _bore_selected_edge_ids_from_payload(payload: TaskPayload) -> tuple[int, ...]:
    raw = _first_value(payload, "selected_edge_ids", "edge_ids", "bore_edge_ids")
    arr = _int_array(raw)
    out: list[int] = []
    seen: set[int] = set()
    for item in arr.tolist():
        value = int(item)
        if value < 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("Bore task payload requires non-empty selected_edge_ids.")
    return tuple(out)


def _bore_display_result_to_payload(display_result: Any) -> TaskPayload:
    candidates = list(getattr(display_result, "candidates", ()) or ())
    diagnostics = dict(getattr(display_result, "diagnostics", {}) or {})
    return {
        "implemented": True,
        "operation": "bore_region_extract",
        "selected_edge_ids": _plain_result(getattr(display_result, "selected_edge_ids", ())),
        "normalized_edge_ids": _plain_result(getattr(display_result, "normalized_edge_ids", ())),
        "region_face_ids": _plain_result(getattr(display_result, "region_face_ids", ())),
        "seed_face_ids": _plain_result(getattr(display_result, "seed_face_ids", ())),
        "region_preview_face_ids": _plain_result(getattr(display_result, "region_preview_face_ids", ())),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "candidate_count": int(len(candidates)),
        "diagnostics": _plain_result(diagnostics),
        "analysis_text": str(getattr(display_result, "analysis_text", "") or ""),
        "preview_text": str(getattr(display_result, "preview_text", "") or ""),
        "status_text": str(getattr(display_result, "status_text", "") or ""),
        "boundary_status_text": str(getattr(display_result, "boundary_status_text", "") or ""),
        "selected_candidate_id": str(getattr(display_result, "selected_candidate_id", "") or ""),
        "notes": [
            "Bore candidate analysis computed through Phase 6 BORE_REGION_EXTRACT task.",
            "Selection and viewport display remain owned by the host process.",
        ],
    }


def _select_bore_candidate_for_rebuild(display_result: Any, payload: TaskPayload) -> tuple[Any, int]:
    candidates = tuple(getattr(display_result, "candidates", ()) or ())
    if not candidates:
        raise ValueError("Bore rebuild task found no candidates for the selected edges.")

    raw_candidate = _first_value(payload, "candidate", "candidate_metadata", "candidate_view")
    candidate_mapping = raw_candidate if isinstance(raw_candidate, dict) else {}

    requested_id = str(
        payload.get("candidate_id")
        or payload.get("selected_candidate_id")
        or candidate_mapping.get("candidate_id")
        or ""
    ).strip()
    if requested_id:
        for index, candidate in enumerate(candidates):
            if str(getattr(candidate, "candidate_id", "")) == requested_id:
                return candidate, index

    requested_rebuild_faces = tuple(
        int(v)
        for v in np.asarray(candidate_mapping.get("rebuild_face_ids", ()), dtype=np.int64).reshape(-1).tolist()
    ) if candidate_mapping else ()
    if requested_rebuild_faces:
        requested_set = set(requested_rebuild_faces)
        for index, candidate in enumerate(candidates):
            candidate_set = set(int(v) for v in getattr(candidate, "rebuild_face_ids", ()) or ())
            if candidate_set == requested_set:
                return candidate, index

    raw_index = _first_value(payload, "candidate_index", "index")
    if raw_index is not None:
        candidate_index = int(raw_index)
        if 0 <= candidate_index < len(candidates):
            return candidates[candidate_index], candidate_index
        raise IndexError(f"Bore candidate_index out of range: {candidate_index}; available candidates: {len(candidates)}")

    return candidates[0], 0


def _optional_rgba_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        vals = [int(v) for v in tuple(value)]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    return tuple(max(0, min(255, v)) for v in vals)  # type: ignore[return-value]


def _handle_neural_surface_repair(payload: TaskPayload) -> TaskPayload:
    del payload
    return {
        "implemented": False,
        "status": "neural_surface_repair is a future execution socket",
        "faces": [],
    }


def _manual_request_from_payload(payload: TaskPayload) -> ManualEditRequest:
    operation = str(payload.get("operation") or "cleanup")
    selection_mode = str(payload.get("selection_mode") or "faces")

    if selection_mode not in {"faces", "vertices"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")

    parameters = payload.get("parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise TypeError("payload['parameters'] must be a dict when provided.")

    selection = ManualSelection(
        mode=selection_mode,  # type: ignore[arg-type]
        face_ids=_int_array(payload.get("face_ids")),
        vertex_ids=_int_array(payload.get("vertex_ids")),
        source_path=payload.get("source_path"),
    )

    return ManualEditRequest(
        operation=operation,  # type: ignore[arg-type]
        selection=selection,
        preview_only=True,
        parameters=parameters,
    )


def _group_options_from_payload(payload: TaskPayload) -> QuadGroupProcessOptions:
    raw = payload.get("group_options") or {}

    if not isinstance(raw, dict):
        raise TypeError("payload['group_options'] must be a dict when provided.")

    opts = QuadGroupProcessOptions()

    for name in (
        "decode_mode",
        "texture_path",
        "cleanup",
        "reduce",
        "target_ratio",
        "boundary_weight",
        "allow_non_manifold_edge_removal",
    ):
        if name in raw:
            setattr(opts, name, raw[name])

    return opts


def _require_mesh(payload: TaskPayload) -> trimesh.Trimesh:
    mesh = payload.get("mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Task payload must contain 'mesh' as trimesh.Trimesh.")

    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("Task payload mesh must be non-empty.")

    return mesh


def _int_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int64)

    return np.asarray(value, dtype=np.int64).reshape(-1)


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(v) for v in np.asarray(value, dtype=np.int64).reshape(-1).tolist())


def _hole_candidates_from_payload(
    payload: TaskPayload,
    mesh: trimesh.Trimesh,
) -> tuple[HoleCandidate, ...]:
    face_ids = _optional_int_tuple(payload.get("face_ids"))
    max_area_hint = payload.get("max_area_hint")
    max_perimeter = payload.get("max_perimeter")

    find_kwargs = {
        "mesh": mesh,
        "face_ids": face_ids,
        "max_area_hint": None if max_area_hint is None else float(max_area_hint),
        "max_perimeter": None if max_perimeter is None else float(max_perimeter),
    }
    candidates = _call_with_supported_kwargs(find_hole_candidates, find_kwargs)

    if not isinstance(candidates, (list, tuple)):
        raise TypeError("find_hole_candidates() returned an unexpected value.")

    for candidate in candidates:
        if not isinstance(candidate, HoleCandidate):
            raise TypeError("find_hole_candidates() returned a non-HoleCandidate item.")

    return tuple(candidates)


def _hole_candidate_from_payload(
    payload: TaskPayload,
    mesh: trimesh.Trimesh,
) -> tuple[HoleCandidate, int]:
    raw_candidate = _first_value(payload, "candidate", "hole_candidate")
    raw_index = _first_value(payload, "candidate_index", "index")
    candidate_index = int(raw_index if raw_index is not None else 0)

    if raw_candidate is not None:
        if not isinstance(raw_candidate, HoleCandidate):
            raise TypeError("payload['candidate'] must be a HoleCandidate when provided.")
        return raw_candidate, candidate_index

    face_ids = _optional_int_tuple(payload.get("face_ids"))
    max_area_hint = payload.get("max_area_hint")
    max_perimeter = payload.get("max_perimeter")

    find_kwargs = {
        "mesh": mesh,
        "face_ids": face_ids,
        "max_area_hint": None if max_area_hint is None else float(max_area_hint),
        "max_perimeter": None if max_perimeter is None else float(max_perimeter),
    }
    candidates = _call_with_supported_kwargs(find_hole_candidates, find_kwargs)

    if not isinstance(candidates, (list, tuple)):
        raise TypeError("find_hole_candidates() returned an unexpected value.")

    if not candidates:
        raise ValueError("No hole candidates found for hole fill preview.")

    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError(
            f"candidate_index out of range: {candidate_index}; "
            f"available candidates: {len(candidates)}"
        )

    candidate = candidates[candidate_index]
    if not isinstance(candidate, HoleCandidate):
        raise TypeError("find_hole_candidates() returned a non-HoleCandidate item.")

    return candidate, candidate_index


def _extract_appended_patch_mesh(
    base_mesh: trimesh.Trimesh,
    preview_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Extract the generated hole-fill patch from the current fan-preview shape.

    Current Phase 2E fan preview appends generated faces to the end of the
    preview mesh. Centralizing this assumption here removes the need for GUI
    code to slice appended faces directly. A later hole_fill_preview.py change
    can replace this helper with explicit patch data.
    """

    base_face_count = int(len(base_mesh.faces))
    base_vertex_count = int(len(base_mesh.vertices))

    preview_vertices = np.asarray(preview_mesh.vertices, dtype=float)
    preview_faces = np.asarray(preview_mesh.faces, dtype=np.int64)

    if preview_faces.shape[0] < base_face_count:
        raise ValueError("Preview mesh has fewer faces than base mesh; cannot extract patch.")

    patch_faces_global = preview_faces[base_face_count:]
    if patch_faces_global.size == 0:
        raise ValueError("Hole fill preview did not append any patch faces.")

    used_vertex_ids = np.unique(patch_faces_global.reshape(-1)).astype(np.int64)
    remap = {int(old_id): new_id for new_id, old_id in enumerate(used_vertex_ids.tolist())}

    patch_vertices = preview_vertices[used_vertex_ids]
    patch_faces = np.asarray(
        [[remap[int(vertex_id)] for vertex_id in face] for face in patch_faces_global],
        dtype=np.int64,
    )

    patch_mesh = trimesh.Trimesh(
        vertices=patch_vertices,
        faces=patch_faces,
        process=False,
    )
    patch_mesh.remove_unreferenced_vertices()

    patch_face_ids = tuple(range(base_face_count, int(len(preview_mesh.faces))))
    patch_vertex_ids = tuple(int(v) for v in used_vertex_ids.tolist())
    new_face_ids = patch_face_ids
    new_vertex_ids = tuple(int(v) for v in used_vertex_ids.tolist() if int(v) >= base_vertex_count)

    return patch_mesh, patch_face_ids, patch_vertex_ids, new_face_ids, new_vertex_ids


def _candidate_boundary_vertices(candidate: HoleCandidate) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(candidate, "boundary_vertices", ()) or ())


def _candidate_boundary_edges(candidate: HoleCandidate) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for item in getattr(candidate, "boundary_edges", ()) or ():
        if len(item) != 2:
            continue
        edges.append((int(item[0]), int(item[1])))
    return tuple(edges)


def _candidate_metadata(candidate: HoleCandidate) -> dict[str, Any]:
    centroid = getattr(candidate, "centroid", None)
    if centroid is not None:
        centroid_value: list[float] | None = [float(v) for v in np.asarray(centroid).reshape(-1).tolist()]
    else:
        centroid_value = None

    area_hint = getattr(candidate, "area_hint", None)

    return {
        "perimeter": float(getattr(candidate, "perimeter", 0.0)),
        "area_hint": None if area_hint is None else float(area_hint),
        "fill_priority": float(getattr(candidate, "fill_priority", 0.0)),
        "centroid": centroid_value,
    }


def _require_existing_path(payload: TaskPayload, *names: str) -> Path:
    value = _first_value(payload, *names)
    if value is None or str(value).strip() == "":
        joined = ", ".join(names)
        raise ValueError(f"Task payload must provide one of: {joined}.")

    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return path


def _require_output_path(payload: TaskPayload, *names: str) -> Path:
    value = _first_value(payload, *names)
    if value is None or str(value).strip() == "":
        joined = ", ".join(names)
        raise ValueError(f"Task payload must provide one of: {joined}.")

    path = Path(str(value)).expanduser().resolve()
    parent = path if _looks_like_directory_key(names) else path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return path


def _looks_like_directory_key(names: tuple[str, ...]) -> bool:
    return any(name.endswith("_dir") or name in {"work_dir", "target_dir", "result_dir"} for name in names)


def _first_value(payload: TaskPayload, *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _dict_from_payload(payload: TaskPayload, *names: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in names:
        raw = payload.get(name)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise TypeError(f"payload['{name}'] must be a dict when provided.")
        merged.update(raw)
    return merged


def _external_options(payload: TaskPayload) -> dict[str, Any]:
    """Merge external runner options while keeping the top-level payload explicit."""

    options = _dict_from_payload(payload, "options", "runner_options")

    # Allow common external-tool knobs to be passed at the top level without
    # forcing MeshProcessor to wrap everything in an options dict.
    passthrough_names = {
        "scale",
        "sharp",
        "alpha",
        "crease_angle",
        "deterministic",
        "dominant",
        "intrinsic",
        "boundaries",
        "smooth_iterations",
        "cleanup_method",
        "repair_method",
        "target_ratio",
        "boundary_weight",
        "allow_non_manifold_edge_removal",
        "stage1_preset",
        "stage2_preset",
    }

    for name in passthrough_names:
        if name in payload and payload[name] is not None:
            options.setdefault(name, payload[name])

    return options


def _construct_runner(cls: type, kwargs: dict[str, Any]) -> Any:
    filtered = _supported_kwargs(cls, kwargs)
    try:
        return cls(**filtered)
    except TypeError:
        # Some existing runners may still use a no-argument constructor with
        # defaults discovered from the project tree. Fall back to that only if
        # the filtered constructor call failed.
        if filtered:
            return cls()
        raise


def _call_with_supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    filtered = _supported_kwargs(func, kwargs)
    return func(**filtered)


def _supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in kwargs.items() if v is not None}

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return clean

    params = sig.parameters.values()
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    if accepts_var_kwargs:
        return clean

    allowed = {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {k: v for k, v in clean.items() if k in allowed}


def _plain_result(value: Any) -> Any:
    if value is None:
        return {}

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if dataclasses.is_dataclass(value):
        return _plain_result(dataclasses.asdict(value))

    if isinstance(value, dict):
        return {str(k): _plain_result(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_plain_result(v) for v in value]

    if hasattr(value, "__dict__"):
        return {str(k): _plain_result(v) for k, v in vars(value).items() if not k.startswith("_")}

    return repr(value)



def _strict_int_value(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "-", {}, [], ()):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _strict_float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "-", {}, [], ()):
            return default
        return float(value)
    except Exception:
        return default


def _strict_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _strict_first_mapping_value(
    *mappings: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    for mapping in mappings:
        if isinstance(mapping, dict) and mapping.get(key) not in (None, "", "-", {}, [], ()):
            return mapping.get(key)
    return default


def _strict_adaptive_surface_commit_policy(
    selection_summary: dict[str, Any],
    preview_builder_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Apply strict commit policy for adaptive/adaptive_surface_v2 previews.

    Geometry is unchanged. This only updates commit policy fields.
    """

    summary = dict(selection_summary)
    metadata = dict(preview_builder_metadata or {})

    method = str(
        summary.get("public_method")
        or summary.get("method")
        or metadata.get("adaptive_public_method")
        or metadata.get("method")
        or ""
    ).strip().lower().replace("-", "_")

    if method not in {"adaptive_surface_v2", "adaptive_surface", "adaptive", "adaptive_surface_fill", "adaptive_surface_fill_v2"}:
        return summary

    adaptive_diagnostics = _strict_mapping(
        summary.get("adaptive_diagnostics") or metadata.get("adaptive_diagnostics")
    )
    seam_report = _strict_mapping(
        summary.get("seam_constraint_report") or metadata.get("seam_constraint_report")
    )
    seam_decision = _strict_mapping(
        summary.get("seam_recovery_decision") or metadata.get("seam_recovery_decision")
    )
    topology_after = _strict_mapping(
        summary.get("topology_after") or metadata.get("topology_after")
    )
    quality_after = _strict_mapping(
        summary.get("quality_after") or metadata.get("quality_after")
    )

    relaxation = _strict_mapping(summary.get("relaxation") or metadata.get("relaxation"))
    if not quality_after and isinstance(relaxation.get("quality_after"), dict):
        quality_after = dict(relaxation["quality_after"])

    missing = _strict_int_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            seam_decision,
            seam_report,
            topology_after,
            key="seam_missing_edge_count",
            default=_strict_first_mapping_value(
                seam_decision,
                seam_report,
                topology_after,
                key="missing_seam_edge_count",
                default=0,
            ),
        )
    )
    overused = _strict_int_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            seam_decision,
            seam_report,
            key="seam_overused_edge_count",
            default=_strict_first_mapping_value(
                seam_decision,
                seam_report,
                key="overused_seam_edge_count",
                default=0,
            ),
        )
    )
    coverage = _strict_float_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            seam_decision,
            seam_report,
            topology_after,
            key="seam_coverage_ratio",
            default=None,
        )
    )

    nonmanifold = _strict_int_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            topology_after,
            key="nonmanifold_patch_edge_count",
            default=0,
        )
    )
    extra_open = _strict_int_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            topology_after,
            key="extra_open_boundary_edge_count",
            default=0,
        )
    )
    degenerate = _strict_int_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            quality_after,
            key="degenerate_face_count",
            default=0,
        )
    )

    reasons: list[str] = []

    if missing > 0:
        reasons.append(f"adaptive_surface strict gate: missing seam edges = {missing}")
    if overused > 0:
        reasons.append(f"adaptive_surface strict gate: overused seam edges = {overused}")
    if coverage is not None and coverage < 0.999999:
        reasons.append(f"adaptive_surface strict gate: seam coverage {coverage:.6g} < 1.0")
    if nonmanifold > 0:
        reasons.append(f"adaptive_surface strict gate: nonmanifold patch edges = {nonmanifold}")
    if extra_open > 0:
        reasons.append(f"adaptive_surface strict gate: extra open patch boundary edges = {extra_open}")
    if degenerate > 0:
        reasons.append(f"adaptive_surface strict gate: degenerate patch faces = {degenerate}")

    existing_reasons = summary.get("commit_blocking_reasons", ())
    if isinstance(existing_reasons, str):
        blocking_reasons = [existing_reasons]
    else:
        try:
            blocking_reasons = [str(item) for item in existing_reasons if str(item)]
        except Exception:
            blocking_reasons = []

    if reasons:
        summary["commit_allowed"] = False
        summary["commit_blocking_reasons"] = tuple([*blocking_reasons, *reasons])
        summary["strict_seam_topology_gate"] = "blocked"
    else:
        summary.setdefault("commit_allowed", True)
        summary["commit_blocking_reasons"] = tuple(blocking_reasons)
        summary["strict_seam_topology_gate"] = "passed"

    summary["strict_gate_seam_missing_edge_count"] = int(missing)
    summary["strict_gate_seam_overused_edge_count"] = int(overused)
    summary["strict_gate_seam_coverage_ratio"] = coverage
    summary["strict_gate_nonmanifold_patch_edge_count"] = int(nonmanifold)
    summary["strict_gate_extra_open_boundary_edge_count"] = int(extra_open)
    summary["strict_gate_degenerate_face_count"] = int(degenerate)

    # H-ADAPT-4C: G1/C1-style warning/blocking policy.
    #
    # Support normal spread alone is not a blocker. It often means the
    # surrounding support context is curved, mixed, or feature-like. Block only
    # when the patch boundary itself has severe normal deviation.
    support_context = _strict_mapping(
        summary.get("support_context") or metadata.get("support_context")
    )
    surface_context = _strict_mapping(
        summary.get("surface_context_decision") or metadata.get("surface_context_decision")
    )
    if not support_context and isinstance(surface_context.get("support_context"), dict):
        support_context = dict(surface_context["support_context"])

    g1_mean = _strict_float_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            quality_after,
            key="g1_boundary_normal_mean_deviation_degrees",
            default=_strict_first_mapping_value(
                quality_after,
                key="boundary_normal_mean_deviation_degrees",
                default=None,
            ),
        )
    )
    g1_max = _strict_float_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            quality_after,
            key="g1_boundary_normal_max_deviation_degrees",
            default=_strict_first_mapping_value(
                quality_after,
                key="boundary_normal_max_deviation_degrees",
                default=None,
            ),
        )
    )
    support_spread = _strict_float_value(
        _strict_first_mapping_value(
            summary,
            adaptive_diagnostics,
            support_context,
            key="g1_support_normal_spread_degrees",
            default=_strict_first_mapping_value(
                support_context,
                key="normal_spread_degrees",
                default=None,
            ),
        )
    )

    g1_blocking_reasons: list[str] = []
    g1_warnings: list[str] = []
    g1_gate_status = "not_reported"

    if g1_mean is None and g1_max is None and support_spread is None:
        g1_gate_status = "not_reported"
    else:
        g1_gate_status = "passed"

        if g1_mean is not None and g1_mean > 35.0:
            g1_blocking_reasons.append(
                f"adaptive_surface G1 gate: boundary normal mean deviation {g1_mean:.6g}° > 35°"
            )
        if g1_max is not None and g1_max > 70.0:
            g1_blocking_reasons.append(
                f"adaptive_surface G1 gate: boundary normal max deviation {g1_max:.6g}° > 70°"
            )

        if g1_blocking_reasons:
            g1_gate_status = "blocked"
        elif support_spread is not None and support_spread >= 75.0:
            g1_gate_status = "feature_like"
            g1_warnings.append(
                f"adaptive_surface G1 gate: feature-like support normal spread {support_spread:.6g}°; smooth G1 was not forced"
            )
        elif support_spread is not None and support_spread >= 45.0:
            g1_gate_status = "warning"
            g1_warnings.append(
                f"adaptive_surface G1 gate: elevated support normal spread {support_spread:.6g}°; conservative relaxation is acceptable"
            )
        elif (g1_mean is not None and g1_mean > 20.0) or (g1_max is not None and g1_max > 45.0):
            g1_gate_status = "warning"
            if g1_mean is not None and g1_mean > 20.0:
                g1_warnings.append(
                    f"adaptive_surface G1 gate: boundary normal mean deviation is elevated ({g1_mean:.6g}°)"
                )
            if g1_max is not None and g1_max > 45.0:
                g1_warnings.append(
                    f"adaptive_surface G1 gate: boundary normal max deviation is elevated ({g1_max:.6g}°)"
                )

    if g1_blocking_reasons:
        summary["commit_allowed"] = False
        existing_blocking = summary.get("commit_blocking_reasons", ())
        try:
            blocking_reasons = [str(item) for item in existing_blocking if str(item)]
        except Exception:
            blocking_reasons = [str(existing_blocking)] if str(existing_blocking) else []
        summary["commit_blocking_reasons"] = tuple([*blocking_reasons, *g1_blocking_reasons])

    if g1_warnings:
        existing_warnings = summary.get("commit_warnings", ())
        try:
            warning_items = [str(item) for item in existing_warnings if str(item)]
        except Exception:
            warning_items = [str(existing_warnings)] if str(existing_warnings) else []
        summary["commit_warnings"] = tuple([*warning_items, *g1_warnings])

    summary["g1_gate_status"] = g1_gate_status
    summary["g1_gate_reasons"] = tuple([*g1_blocking_reasons, *g1_warnings])
    summary["g1_gate_boundary_normal_mean_deviation_degrees"] = g1_mean
    summary["g1_gate_boundary_normal_max_deviation_degrees"] = g1_max
    summary["g1_gate_support_normal_spread_degrees"] = support_spread

    return summary

def _runner_ok_value(result: Any) -> bool:
    if not isinstance(result, dict):
        return True

    for key in ("ok", "success", "succeeded"):
        if key in result:
            return bool(result[key])

    for key in ("return_code", "returncode", "exit_code"):
        if key in result and result[key] is not None:
            try:
                return int(result[key]) == 0
            except Exception:
                return False

    # Some runners only return metadata and rely on exceptions for failure.
    return True


def _raise_if_runner_failed(label: str, result: Any) -> None:
    if _runner_ok_value(result):
        return

    error = None
    if isinstance(result, dict):
        error = result.get("error") or result.get("stderr") or result.get("message")

    raise RuntimeError(f"{label} failed: {error or result!r}")
