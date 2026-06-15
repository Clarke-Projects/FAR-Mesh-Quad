from __future__ import annotations

import numpy as np


def topology_scope_label(face_ids: tuple[int, ...] | None) -> str:
    """Return the user-facing topology scope label for whole-mesh or selected-face work."""
    if face_ids is None:
        return "whole mesh"
    return f"selected face region ({len(face_ids)} faces)"


def format_optional_float(value: object, *, digits: int = 6) -> str:
    """Format an optional numeric value for compact GUI/status text."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def format_optional_centroid(value: object) -> str:
    """Format an optional XYZ centroid-like value for GUI/status text."""
    if value is None:
        return "-"
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return str(value)
    if arr.size < 3:
        return str(value)
    return f"({arr[0]:.6g}, {arr[1]:.6g}, {arr[2]:.6g})"


def format_optional_bool(value: object) -> str:
    """Format optional booleans for compact GUI/status text."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _p20b_value(payload: dict[str, object], key: str, default: object = None) -> object:
    """Return a P20B value from flat or nested V3 diagnostic payloads."""
    if key in payload:
        return payload.get(key, default)

    diagnostic_report = payload.get("v3_diagnostic_report")
    if isinstance(diagnostic_report, dict):
        target_surface_p20b = diagnostic_report.get("target_surface_p20b")
        if isinstance(target_surface_p20b, dict):
            short_key = key.removeprefix("v3_p20b_").removeprefix("target_surface_p20b_")
            if short_key in target_surface_p20b:
                return target_surface_p20b.get(short_key, default)
        short_key = key.removeprefix("v3_p20b_")
        report_key = f"p20b_{short_key}"
        if report_key in diagnostic_report:
            return diagnostic_report.get(report_key, default)

    target_key = key.replace("v3_p20b_", "target_surface_p20b_", 1)
    if target_key in payload:
        return payload.get(target_key, default)
    return default


def format_v3_p20b_status_lines(payload: dict[str, object] | None) -> list[str]:
    """Return compact GUI/status lines for P20B target-surface construction metadata."""
    if not payload:
        return []

    checkpoint = _p20b_value(payload, "v3_p20b_checkpoint")
    applied = _p20b_value(payload, "v3_p20b_applied")
    consumed = _p20b_value(payload, "v3_p20b_consumed_by_target_surface_sampling")

    if checkpoint is None and applied is None and consumed is None:
        return []

    lines = ["V3 P20B principal-line transport:"]
    lines.append(f"  checkpoint: {checkpoint or '-'}")
    lines.append(
        "  applied/consumed: "
        f"{format_optional_bool(applied)} / {format_optional_bool(consumed)}"
    )

    status = _p20b_value(payload, "v3_p20b_status")
    if status:
        lines.append(f"  status: {status}")

    policy = _p20b_value(payload, "v3_p20b_policy")
    if policy:
        lines.append(f"  policy: {policy}")

    source = _p20b_value(payload, "v3_p20b_source")
    if source:
        lines.append(f"  source: {source}")

    generated = _p20b_value(payload, "v3_p20b_generated_sample_count")
    node_count = _p20b_value(payload, "v3_p20b_node_count")
    ring_count = _p20b_value(payload, "v3_p20b_ring_count")
    center_count = _p20b_value(payload, "v3_p20b_center_count")
    lines.append(
        "  nodes/generated/rings/centers: "
        f"{format_optional_float(node_count)} / "
        f"{format_optional_float(generated)} / "
        f"{format_optional_float(ring_count)} / "
        f"{format_optional_float(center_count)}"
    )

    lines.append(
        "  movement mean/max: "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_movement_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_movement_max'))}"
    )
    lines.append(
        "  depth mean/abs/max: "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_depth_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_depth_abs_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_depth_max'))}"
    )
    lines.append(
        "  flow/c1/c2 coherence: "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_flow_consistency_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_neighbor_c1_abs_dot_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_neighbor_c2_abs_dot_mean'))}"
    )
    lines.append(
        "  anisotropy/reliability mean: "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_anisotropy_mean'))} / "
        f"{format_optional_float(_p20b_value(payload, 'v3_p20b_directional_reliability_mean'))}"
    )
    lines.append(
        "  boundary exact / no faces-connectivity: "
        f"{format_optional_bool(_p20b_value(payload, 'v3_p20b_boundary_exact'))} / "
        f"{format_optional_bool(_p20b_value(payload, 'v3_p20b_no_faces_no_connectivity'))}"
    )
    return lines

