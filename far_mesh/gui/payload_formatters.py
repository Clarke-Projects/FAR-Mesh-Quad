from __future__ import annotations

from typing import Any


def format_repair_inspection_lines(label: str, inspection: dict[str, Any] | None) -> list[str]:
    """Return log lines for a repair inspection payload."""
    if not inspection:
        return []

    lines = [
        f"{label}: vertices={inspection.get('vertices')}, "
        f"faces={inspection.get('faces')}, "
        f"watertight={inspection.get('watertight')}, "
        f"boundary_edges={inspection.get('boundary_edge_count')}, "
        f"boundary_loops={inspection.get('boundary_loop_count')}, "
        f"components={inspection.get('connected_components')}"
    ]

    recommended = inspection.get("recommended_workflow")
    if recommended:
        lines.append(f"{label} recommended workflow: {recommended}")

    defects = inspection.get("defects") or {}
    if defects:
        defect_parts = []
        for key in (
            "self_intersections",
            "non_manifold_edge_faces",
            "non_manifold_vertices",
            "boundary_edges_estimate",
            "boundary_loops_estimate",
        ):
            if key in defects:
                defect_parts.append(f"{key}={defects.get(key)}")
        if defect_parts:
            lines.append(f"{label} defects: {', '.join(defect_parts)}")

    return lines


def format_repair_payload_lines(payload: dict[str, Any]) -> list[str]:
    """Return log lines for a repair result payload."""
    requested = payload.get("requested_method", payload.get("method"))
    executed = payload.get("executed_method", payload.get("method"))
    lines = [
        f"Repair requested: {requested}",
        f"Repair executed: {executed}",
        (
            f"Repair options: join_comp={payload.get('join_comp')}, "
            f"fill_holes={payload.get('fill_holes')}"
        ),
        f"Repair elapsed: {payload.get('elapsed_seconds', 0.0):.2f}s",
    ]

    backend_chain = payload.get("backend_chain") or []
    if backend_chain:
        lines.append(f"Repair backend chain: {' -> '.join(str(x) for x in backend_chain)}")

    before_stats = payload.get("stats_before") or {}
    after_stats = payload.get("stats_after") or {}
    if before_stats or after_stats:
        lines.append(
            f"Repair stats: vertices {before_stats.get('vertices')} -> {after_stats.get('vertices')}, "
            f"faces {before_stats.get('faces')} -> {after_stats.get('faces')}, "
            f"watertight {before_stats.get('watertight')} -> {after_stats.get('watertight')}"
        )

    note = payload.get("note")
    if note:
        lines.append(f"Repair note: {note}")

    notes = payload.get("notes") or []
    for extra_note in notes:
        if extra_note and extra_note != note:
            lines.append(f"Repair note: {extra_note}")

    lines.extend(format_repair_inspection_lines("Repair inspection before", payload.get("inspection_before")))
    lines.extend(format_repair_inspection_lines("Repair inspection after", payload.get("inspection_after")))

    steps = payload.get("steps") or []
    for step in steps:
        if isinstance(step, dict):
            lines.append(
                f"Repair step: {step.get('method')} "
                f"({step.get('elapsed_seconds', 0.0):.2f}s)"
            )

    return lines


def format_reduce_payload_lines(payload: dict[str, Any]) -> list[str]:
    """Return log lines for a reduction result payload."""
    lines = [
        f"Reduce backend: {payload.get('backend')}",
        (
            f"Reduce faces: {payload.get('before_faces')} -> {payload.get('after_faces')} "
            f"(target {payload.get('target_faces')})"
        ),
        (
            f"Reduce ratio: {payload.get('reduction_ratio', 0.0):.4f}, "
            f"elapsed {payload.get('elapsed_seconds', 0.0):.2f}s"
        ),
    ]
    note = payload.get("note")
    if note:
        lines.append(f"Reduce note: {note}")
    return lines


def format_remesh_payload_lines(payload: dict[str, Any]) -> list[str]:
    """Return log lines for an Instant Meshes or QuadWild remesh payload."""
    backend = payload.get("backend")
    lines = [
        f"Remesh backend: {backend}",
        f"Remesh final stage: {payload.get('final_stage')}",
    ]
    if payload.get("output_path"):
        lines.append(f"Remesh output: {payload.get('output_path')}")

    if backend == "instant_meshes":
        lines.append(f"Instant Meshes elapsed: {payload.get('elapsed_seconds', 0.0):.2f}s")
        stats = payload.get("stats") or {}
        if stats:
            lines.append(
                f"Result stats: vertices={stats.get('vertices')}, "
                f"faces={stats.get('faces')}, watertight={stats.get('watertight')}"
            )
        return lines

    if backend == "quadwild_bimdf":
        lines.append(
            f"Stage times: "
            f"stage1={payload.get('stage1_elapsed_seconds', 0.0):.2f}s, "
            f"stage2={payload.get('stage2_elapsed_seconds', 0.0):.2f}s, "
            f"pipeline={payload.get('pipeline_total_elapsed_seconds', 0.0):.2f}s"
        )
        lines.append(f"Used original input file: {payload.get('used_original_input_file')}")
        lines.append(f"Source mesh path: {payload.get('source_mesh_path')}")
        lines.append(
            f"QuadWild output counts: vertices={payload.get('quadwild_output_vertices')}, "
            f"faces={payload.get('quadwild_output_faces')}"
        )
        workflow_steps = payload.get("workflow_steps") or []
        for step in workflow_steps:
            lines.append(f"Workflow: {step}")
        auto_reduce_payload = payload.get("auto_reduce_payload")
        if auto_reduce_payload:
            lines.append(
                f"Auto reduction: {auto_reduce_payload.get('before_faces')} -> "
                f"{auto_reduce_payload.get('after_faces')} faces "
                f"using {auto_reduce_payload.get('backend')} "
                f"in {auto_reduce_payload.get('elapsed_seconds', 0.0):.2f}s"
            )
        generated = payload.get("generated_files") or {}
        for key, value in sorted(generated.items()):
            lines.append(f"Generated file [{key}]: {value}")

    return lines


def _format_optional_payload_float(value: object, *, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def _format_optional_payload_bool(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _v3_p20b_payload_value(payload: dict[str, Any], key: str, default: object = None) -> object:
    if key in payload:
        return payload.get(key, default)
    target_key = key.replace("v3_p20b_", "target_surface_p20b_", 1)
    if target_key in payload:
        return payload.get(target_key, default)
    diagnostic_report = payload.get("v3_diagnostic_report")
    if isinstance(diagnostic_report, dict):
        target_surface_p20b = diagnostic_report.get("target_surface_p20b")
        if isinstance(target_surface_p20b, dict):
            short_key = key.removeprefix("v3_p20b_")
            if short_key in target_surface_p20b:
                return target_surface_p20b.get(short_key, default)
        report_key = f"p20b_{key.removeprefix('v3_p20b_')}"
        if report_key in diagnostic_report:
            return diagnostic_report.get(report_key, default)
    return default


def format_v3_p20b_payload_lines(payload: dict[str, Any]) -> list[str]:
    """Return log lines for V3 P20B target-surface construction metadata."""
    checkpoint = _v3_p20b_payload_value(payload, "v3_p20b_checkpoint")
    applied = _v3_p20b_payload_value(payload, "v3_p20b_applied")
    consumed = _v3_p20b_payload_value(
        payload,
        "v3_p20b_consumed_by_target_surface_sampling",
    )
    if checkpoint is None and applied is None and consumed is None:
        return []

    return [
        "V3 P20B principal-line transport:",
        f"  checkpoint: {checkpoint or '-'}",
        (
            "  applied/consumed: "
            f"{_format_optional_payload_bool(applied)} / "
            f"{_format_optional_payload_bool(consumed)}"
        ),
        f"  status: {_v3_p20b_payload_value(payload, 'v3_p20b_status', '-')}",
        f"  policy: {_v3_p20b_payload_value(payload, 'v3_p20b_policy', '-')}",
        f"  source: {_v3_p20b_payload_value(payload, 'v3_p20b_source', '-')}",
        (
            "  nodes/generated/rings/centers: "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_node_count'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_generated_sample_count'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_ring_count'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_center_count'))}"
        ),
        (
            "  movement mean/max: "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_movement_mean'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_movement_max'))}"
        ),
        (
            "  flow/c1/c2 coherence: "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_flow_consistency_mean'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_neighbor_c1_abs_dot_mean'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_neighbor_c2_abs_dot_mean'))}"
        ),
        (
            "  anisotropy/reliability mean: "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_anisotropy_mean'))} / "
            f"{_format_optional_payload_float(_v3_p20b_payload_value(payload, 'v3_p20b_directional_reliability_mean'))}"
        ),
        (
            "  boundary exact / no faces-connectivity: "
            f"{_format_optional_payload_bool(_v3_p20b_payload_value(payload, 'v3_p20b_boundary_exact'))} / "
            f"{_format_optional_payload_bool(_v3_p20b_payload_value(payload, 'v3_p20b_no_faces_no_connectivity'))}"
        ),
    ]

