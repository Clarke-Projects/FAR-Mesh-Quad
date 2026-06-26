"""Validation and trial-acceptance helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It may count boundary edges,
validate mesh inputs, and evaluate trial-rebuild diagnostics, but it must not
interpret CandidateData, authorize deletion, generate replacement geometry, or
commit mesh topology.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import trimesh

from ..topology import boundary_edges_for_face_patch, normalize_edge
from .rebuild_geometry import median_loop_edge_length, unit_vector

EdgeKey = tuple[int, int]

REBUILD_VALIDATION_EXTRACTION_CHECKPOINT_V176F = (
    "v176f_rebuild_validation_extraction_no_behavior_change_trial_acceptance_helpers"
)

REBUILD_VALIDATION_NON_MUTATION_CONTRACT_V176F = (
    "rebuild_validation_may_evaluate_mesh_and_trial_diagnostics_but_must_not_mutate_or_authorize_delete"
)

REBUILD_BOUNDARY_DIAGNOSTIC_EXTRACTION_CHECKPOINT_V176P = (
    "v176p_rebuild_boundary_diagnostic_helpers_extracted_no_behavior_change"
)

REBUILD_BOUNDARY_DIAGNOSTIC_NON_MUTATION_CONTRACT_V176P = (
    "boundary_diagnostics_may_compare_deleted_patch_and_generated_surface_edges_but_must_not_mutate_or_authorize_delete"
)


REBUILD_PLAN_GEOMETRY_QUALITY_EXTRACTION_CHECKPOINT_V177F = (
    "v177f_rebuild_plan_geometry_quality_gate_extracted_no_behavior_change"
)

REBUILD_PLAN_GEOMETRY_QUALITY_NON_MUTATION_CONTRACT_V177F = (
    "plan_geometry_quality_helpers_may_reject_bad_trial_plan_quality_only_no_delete_authorization_no_geometry_generation_no_mutation"
)


def int_preserve_zero(value: object, default: int = -1) -> int:
    """Coerce diagnostics to int without treating 0 as missing.

    Rebuild diagnostics legitimately use zero for success states, for example
    generated_boundary_generated_vertex_edge_count=0.  Do not use
    ``value or fallback`` here because it turns a valid zero into the fallback.
    """

    if value is None:
        return int(default)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def pocket_recess_cup_trial_accepts(*, trial: Mapping[str, object], plan: object) -> bool:
    """Strict local acceptance for a POCKET recessed cup on non-watertight meshes."""

    diag = dict(getattr(plan, "diagnostics", {}) or {})
    boundary_match = bool(diag.get("boundary_match_exact", False))
    missing = int_preserve_zero(diag.get("missing_patch_boundary_edge_count", -1), -1)
    extra = int_preserve_zero(diag.get("extra_generated_boundary_edge_count", -1), -1)
    patch_edges = int_preserve_zero(diag.get("patch_boundary_edge_count", -1), -1)
    generated_edges = int_preserve_zero(diag.get("generated_boundary_edge_count", -1), -1)
    generated_original_boundary = int_preserve_zero(diag.get("generated_boundary_original_vertex_edge_count", -1), -1)
    generated_new_boundary = int_preserve_zero(diag.get("generated_boundary_generated_vertex_edge_count", -1), -1)
    before = int_preserve_zero(trial.get("boundary_edge_count_before", -1), -1)
    after = int_preserve_zero(trial.get("boundary_edge_count_after", -1), -1)
    return bool(
        boundary_match
        and missing == 0
        and extra == 0
        and patch_edges > 0
        and generated_edges == patch_edges
        and generated_original_boundary == patch_edges
        and generated_new_boundary == 0
        and before >= 0
        and after <= before
    )


def trial_accepts_for_context(*, context: object, trial: Mapping[str, object], plan: object) -> bool:
    """Return whether a trial replacement is acceptable for this feature type.

    Global watertightness remains the strongest success condition.  For imported
    meshes that are already globally open, an accepted local feature CandidateData
    may still be valid when the replacement exactly preserves the deleted patch
    boundary and does not add new boundary edges.
    """

    # v85: preserve real zero counts.  The previous code used
    # ``int(value or -1)``, which converted boundary_edge_count_after=0 into
    # -1.  Damaged-bore full-depth trials can legitimately return
    # after=0, watertight_after=True after swallowing internal defect
    # boundary loops, so zero must remain zero for acceptance.
    after_raw = trial.get("boundary_edge_count_after", -1)
    before_raw = trial.get("boundary_edge_count_before", -1)
    after = int(after_raw if after_raw is not None else -1)
    before = int(before_raw if before_raw is not None else -1)
    if after == 0 and bool(trial.get("watertight_after", False)):
        return True

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    if entity in {"chamfer", "borehole", "pocket", "circular_pocket"}:
        diag = dict(getattr(plan, "diagnostics", {}) or {})
        boundary_match = bool(diag.get("boundary_match_exact", False))
        missing = int_preserve_zero(diag.get("missing_patch_boundary_edge_count", 1), 1)
        extra = int_preserve_zero(diag.get("extra_generated_boundary_edge_count", 1), 1)
        patch_edges = int_preserve_zero(diag.get("patch_boundary_edge_count", -1), -1)
        generated_edges = int_preserve_zero(diag.get("generated_boundary_edge_count", -1), -1)
        generated_original_boundary = int_preserve_zero(diag.get("generated_boundary_original_vertex_edge_count", -1), -1)
        generated_new_boundary = int_preserve_zero(diag.get("generated_boundary_generated_vertex_edge_count", -1), -1)
        candidate_owned_patch = bool(getattr(context, "candidate_from_component_engine", False)) and bool(getattr(context, "candidate_has_preview_face_patch", False))
        local_boundary_preserved = bool(
            boundary_match
            and missing == 0
            and extra == 0
            and patch_edges > 0
            and generated_edges == patch_edges
            and before >= 0
            and after <= before
        )
        if entity == "chamfer" and local_boundary_preserved:
            return True
        if entity == "borehole" and candidate_owned_patch and local_boundary_preserved:
            return True
        if entity == "borehole" and candidate_owned_patch and bool(diag.get("damaged_bore_internal_boundary_swallow_acceptance_v173n", False)):
            # v173n: damaged BORE wall CandidateData may have more than two
            # patch boundary loops because wall holes/tears are inside the owned
            # bore wall.  The generated replacement must match the two measured
            # endpoint loops exactly and add no new boundary edges, but it must
            # not be required to reproduce internal defect-hole loops.
            if before >= 0 and after <= before and extra == 0:
                return True
        if entity in {"pocket", "circular_pocket"}:
            # POCKET v98 local acceptance: the pocket side-wall delete patch is
            # a measured two-rim sleeve between the opening loop and the floor
            # perimeter.  On user meshes that are already globally open, the
            # whole mesh may remain non-watertight even when the local replacement
            # is exact.  Accept only the strict local topology contract: the
            # generated surface must reproduce every original patch boundary edge,
            # add no generated-vertex boundary edges, and not increase the global
            # boundary count.  Floor faces are preserved by CandidateData/rebuild
            # input; this does not authorize broad RegionData deletion.
            pocket_boundary_is_original_only = bool(generated_original_boundary == patch_edges and generated_new_boundary == 0)
            if candidate_owned_patch and local_boundary_preserved and pocket_boundary_is_original_only:
                return True
    return False


def boundary_edge_count(faces: np.ndarray) -> int:
    counts: dict[EdgeKey, int] = {}
    arr = np.asarray(faces, dtype=np.int64)
    for face in arr[:, :3]:
        verts = [int(v) for v in face[:3]]
        for edge in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            key = normalize_edge(edge)
            counts[key] = counts.get(key, 0) + 1
    return int(sum(1 for count in counts.values() if int(count) == 1))


def _loop_edges_from_vertices_v176p(vertices: tuple[int, ...]) -> tuple[EdgeKey, ...]:
    ids = tuple(int(v) for v in tuple(vertices or ()))
    if len(ids) < 2:
        return ()
    return tuple(normalize_edge((ids[i], ids[(i + 1) % len(ids)])) for i in range(len(ids)))


def _generated_boundary_counts_v176p(triangles: np.ndarray) -> tuple[dict[EdgeKey, int], int]:
    tri_values = np.asarray(triangles, dtype=np.int64)
    tri_arr = tri_values.reshape((-1, 3)) if tri_values.size else np.zeros((0, 3), dtype=np.int64)
    generated_counts: dict[EdgeKey, int] = {}
    generated_invalid_edges = 0
    for tri in tri_arr[:, :3]:
        verts = (int(tri[0]), int(tri[1]), int(tri[2]))
        for edge in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            a, b = int(edge[0]), int(edge[1])
            if a < 0 or b < 0:
                generated_invalid_edges += 1
                continue
            key = normalize_edge((a, b))
            generated_counts[key] = generated_counts.get(key, 0) + 1
    return generated_counts, int(generated_invalid_edges)


def generated_surface_boundary_match_diagnostics_v176p(
    *,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    triangles: np.ndarray,
    source_vertex_count: int,
) -> dict[str, object]:
    """Compare deleted patch boundary against generated replacement boundary.

    Extracted from rebuild.py in v176p without intended behavior change.  This
    helper is diagnostic-only: it does not authorize deletion, validate topology,
    generate geometry, or mutate a mesh.
    """

    source_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    patch_boundary = {normalize_edge(edge) for edge in boundary_edges_for_face_patch(source_arr, face_ids)}

    generated_counts, generated_invalid_edges = _generated_boundary_counts_v176p(triangles)
    generated_boundary = {edge for edge, count in generated_counts.items() if int(count) == 1}
    generated_boundary_original_only = {
        edge for edge in generated_boundary
        if int(edge[0]) < int(source_vertex_count) and int(edge[1]) < int(source_vertex_count)
    }
    generated_boundary_generated_involved = generated_boundary - generated_boundary_original_only

    shared = patch_boundary & generated_boundary
    missing = patch_boundary - generated_boundary
    extra = generated_boundary - patch_boundary

    patch_edges_generated_count_hist: dict[int, int] = {}
    for edge in patch_boundary:
        count = int(generated_counts.get(edge, 0))
        patch_edges_generated_count_hist[count] = patch_edges_generated_count_hist.get(count, 0) + 1

    return {
        "boundary_match_policy": "generated_surface_boundary_must_equal_deleted_patch_boundary",
        "patch_boundary_edge_count": int(len(patch_boundary)),
        "generated_boundary_edge_count": int(len(generated_boundary)),
        "generated_boundary_original_vertex_edge_count": int(len(generated_boundary_original_only)),
        "generated_boundary_generated_vertex_edge_count": int(len(generated_boundary_generated_involved)),
        "shared_boundary_edge_count": int(len(shared)),
        "missing_patch_boundary_edge_count": int(len(missing)),
        "extra_generated_boundary_edge_count": int(len(extra)),
        "generated_invalid_edge_count": int(generated_invalid_edges),
        "boundary_match_exact": bool(not missing and not extra),
        "patch_edges_generated_count_histogram": tuple(sorted((int(k), int(v)) for k, v in patch_edges_generated_count_hist.items())),
        "sample_missing_patch_boundary_edges": tuple(sorted(missing)[:24]),
        "sample_extra_generated_boundary_edges": tuple(sorted(extra)[:24]),
        "sample_shared_boundary_edges": tuple(sorted(shared)[:12]),
    }


def damaged_bore_internal_boundary_swallow_diagnostics_v176p(
    *,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    triangles: np.ndarray,
    source_vertex_count: int,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    boundary_loop_count: int,
) -> dict[str, object]:
    """Classify damaged-BORE internal patch loops as swallowable defects.

    Extracted from rebuild.py in v176p without intended behavior change.  This
    helper only reports whether generated boundary edges match the protected
    endpoint loops while internal wall-defect loops are swallowed.
    """

    source_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    patch_boundary = {normalize_edge(edge) for edge in boundary_edges_for_face_patch(source_arr, face_ids)}
    protected_edges = set(_loop_edges_from_vertices_v176p(loop0)) | set(_loop_edges_from_vertices_v176p(loop1))

    generated_counts, generated_invalid_edges = _generated_boundary_counts_v176p(triangles)
    generated_boundary = {edge for edge, count in generated_counts.items() if int(count) == 1}
    generated_boundary_original_only = {
        edge for edge in generated_boundary
        if int(edge[0]) < int(source_vertex_count) and int(edge[1]) < int(source_vertex_count)
    }
    generated_boundary_generated_involved = generated_boundary - generated_boundary_original_only

    protected_missing_from_patch = protected_edges - patch_boundary
    protected_missing_from_generated = protected_edges - generated_boundary
    generated_extra_not_protected = generated_boundary - protected_edges
    generated_extra_not_patch = generated_boundary - patch_boundary
    swallowed_defect_edges = patch_boundary - protected_edges

    considered = bool(
        int(boundary_loop_count) > 2
        and len(protected_edges) >= 6
        and len(patch_boundary) > len(protected_edges)
        and len(swallowed_defect_edges) > 0
    )
    valid = bool(
        considered
        and not protected_missing_from_patch
        and not protected_missing_from_generated
        and not generated_extra_not_protected
        and not generated_extra_not_patch
        and not generated_boundary_generated_involved
        and generated_invalid_edges == 0
    )

    return {
        "damaged_bore_internal_boundary_swallow_v173n": bool(considered),
        "damaged_bore_internal_boundary_swallow_acceptance_v173n": bool(valid),
        "damaged_bore_boundary_match_scope_v173n": "protected_bore_endpoint_loops_only_internal_wall_defect_loops_swallowed" if considered else "not_applicable",
        "damaged_bore_protected_boundary_edge_count_v173n": int(len(protected_edges)),
        "damaged_bore_swallowed_defect_boundary_edge_count_v173n": int(len(swallowed_defect_edges)),
        "damaged_bore_patch_boundary_edge_count_v173n": int(len(patch_boundary)),
        "damaged_bore_generated_boundary_edge_count_v173n": int(len(generated_boundary)),
        "damaged_bore_generated_original_boundary_edge_count_v173n": int(len(generated_boundary_original_only)),
        "damaged_bore_generated_new_boundary_edge_count_v173n": int(len(generated_boundary_generated_involved)),
        "damaged_bore_protected_missing_from_patch_count_v173n": int(len(protected_missing_from_patch)),
        "damaged_bore_protected_missing_from_generated_count_v173n": int(len(protected_missing_from_generated)),
        "damaged_bore_generated_extra_not_protected_count_v173n": int(len(generated_extra_not_protected)),
        "damaged_bore_generated_extra_not_patch_count_v173n": int(len(generated_extra_not_patch)),
        "damaged_bore_generated_invalid_edge_count_v173n": int(generated_invalid_edges),
        "damaged_bore_sample_swallowed_defect_boundary_edges_v173n": tuple(sorted(swallowed_defect_edges)[:24]),
        "damaged_bore_sample_generated_extra_not_protected_edges_v173n": tuple(sorted(generated_extra_not_protected)[:24]),
    }


def loop_radius_stats_v177f(vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> dict[str, float]:
    """Return radius median and p95 wobble for one protected loop.

    Extracted from ``rebuild.py`` in v177f without intended behavior change.
    This is a geometry-quality validation helper only: it does not classify a
    feature, authorize deletion, generate replacement geometry, mutate a mesh,
    or emit a ``RebuildResult``.
    """

    ids = [int(v) for v in tuple(loop or ()) if 0 <= int(v) < len(vertices)]
    if not ids:
        return {"median": 0.0, "p95_delta": 0.0}
    pts = vertices[np.asarray(ids, dtype=np.int64), :3]
    rel = pts - np.asarray(center, dtype=float).reshape(1, 3)
    axis = unit_vector(axis)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    radii = np.linalg.norm(radial, axis=1)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return {"median": 0.0, "p95_delta": 0.0}
    median = float(np.median(radii))
    p95_delta = float(np.percentile(np.abs(radii - median), 95.0))
    return {"median": median, "p95_delta": p95_delta}


def validate_plan_geometry_quality_v177f(
    *,
    context: object,
    vertices: np.ndarray,
    attempt: object,
    plan: object,
) -> dict[str, object]:
    """Reject watertight-but-wrong loop pairs before the mesh is committed.

    Extracted from ``rebuild.py`` in v177f without intended behavior change.
    A BOREHOLE replacement should be cylindrical.  This quality gate does not
    classify features; it only prevents a BOREHOLE rebuild from using a loop
    pair that behaves like a chamfer/taper or an over-sealed neighbouring
    surface.  CHAMFER and other transition objects are allowed to taper.
    """

    if context.entity_type != "borehole":
        return {"valid": True, "policy": "not_borehole_no_cylindrical_quality_gate"}

    r0 = loop_radius_stats_v177f(vertices, plan.loop0, plan.center0, plan.axis)
    r1 = loop_radius_stats_v177f(vertices, plan.loop1, plan.center1, plan.axis)
    median0 = float(r0.get("median", 0.0))
    median1 = float(r1.get("median", 0.0))
    avg_radius = max(0.5 * (median0 + median1), 1.0e-12)
    edge_scale = max(
        median_loop_edge_length(vertices, plan.loop0),
        median_loop_edge_length(vertices, plan.loop1),
        1.0e-12,
    )
    radius_delta = abs(median0 - median1)
    max_radius_delta = max(0.18 * avg_radius, 2.5 * edge_scale)

    wobble0 = float(r0.get("p95_delta", 0.0))
    wobble1 = float(r1.get("p95_delta", 0.0))
    max_wobble = max(0.16 * avg_radius, 3.0 * edge_scale)

    diagnostics = {
        "policy": "borehole_cylindrical_loop_pair_quality_gate",
        "normal_orientation_policy": str(plan.diagnostics.get("normal_orientation_policy", "")),
        "normal_flip_count": int(plan.diagnostics.get("normal_flip_count", 0) or 0),
        "normal_alignment_median": float(plan.diagnostics.get("normal_alignment_median", 0.0) or 0.0),
        "normal_alignment_min": float(plan.diagnostics.get("normal_alignment_min", 0.0) or 0.0),
        "loop0_radius_median": float(median0),
        "loop1_radius_median": float(median1),
        "loop_radius_delta": float(radius_delta),
        "max_loop_radius_delta": float(max_radius_delta),
        "loop0_radius_p95_delta": float(wobble0),
        "loop1_radius_p95_delta": float(wobble1),
        "max_loop_radius_p95_delta": float(max_wobble),
        "attempt_source": str(attempt.source),
        "target_source": str(attempt.target_source),
    }

    if radius_delta > max_radius_delta:
        diagnostics["valid"] = False
        diagnostics["reason"] = "borehole_loop_pair_radius_delta_too_large"
        return diagnostics
    if max(wobble0, wobble1) > max_wobble:
        diagnostics["valid"] = False
        diagnostics["reason"] = "borehole_boundary_loop_radius_wobble_too_large"
        return diagnostics

    diagnostics["valid"] = True
    diagnostics["reason"] = "ok"
    return diagnostics


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh is None:
        raise ValueError("No mesh provided.")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError("Mesh must provide vertices and faces.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh is empty.")
