"""POCKET rebuild patch-plan shell helpers.

This module is intentionally non-mutating.  It exists as the first POCKET
family extraction boundary for the v176 rebuild roadmap: future POCKET-specific
geometry planning may move here, but mesh mutation, delete authorization, trial
validation, and final ``RebuildResult`` authority remain in ``rebuild.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import math

import numpy as np

from .rebuild_density import (
    QUAD_DENSITY_MODE_FULL,
    QUAD_DENSITY_MODE_PI,
    normalize_quad_density_mode_v176q as _normalize_quad_density_mode,
)
from .rebuild_geometry import (
    loop_center as _loop_center,
    sample_closed_loop_points as _sample_closed_loop_points,
    unit_vector as _unit_vector,
)
from .rebuild_loops import (
    align_second_loop_to_first_v176g as _align_second_loop_to_first,
    align_unequal_loop_pair_to_angle_samples_v176w as _align_unequal_loop_pair_to_angle_samples,
    band_quads_between_rings_v176n as _band_quads_between_rings,
    densify_transition_count_sequence_v176n as _densify_transition_count_sequence,
    sort_loop_pair_by_angle_v176g as _sort_loop_pair_by_angle,
    transition_count_sequence_v176n as _transition_count_sequence,
)

Vector3 = tuple[float, float, float]

REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I = (
    "v176i_rebuild_pocket_patch_plan_shell_no_behavior_change"
)

REBUILD_POCKET_PATCH_PLAN_SHELL_NON_MUTATION_CONTRACT_V176I = (
    "rebuild_pocket_py_may_return_patch_plans_only_rebuild_py_still_mutates_validates_and_emits_results"
)

REBUILD_POCKET_PATCH_PLAN_FIELDS_V176I: tuple[str, ...] = (
    "removed_face_ids",
    "generated_vertices",
    "triangles",
    "logical_quads",
    "loop0",
    "loop1",
    "center0",
    "center1",
    "axis",
    "radius",
    "diagnostics",
)

REBUILD_POCKET_PATCH_PLAN_BOUNDARY_V176I: dict[str, tuple[str, ...]] = {
    "future_rebuild_pocket_py_may_own": (
        "pocket_recess_cup_patch_geometry_planning",
        "pocket_sidewall_ring_generation",
        "pocket_quad_floor_generation",
        "pocket_radius_frame_authority_diagnostics",
        "pocket_family_local_patch_plan_diagnostics",
    ),
    "future_rebuild_pocket_py_must_not_own": (
        "CandidateData_feature_identity",
        "DeletePatchProposal_authorization",
        "trial_mesh_application",
        "topology_validation_gate",
        "RebuildResult_authority",
        "active_mesh_replacement_or_host_state_mutation",
    ),
}


REBUILD_POCKET_RECESS_CUP_GATE_EXTRACTION_CHECKPOINT_V176J = (
    "v176j_rebuild_pocket_recess_cup_request_gate_extracted_no_behavior_change"
)

REBUILD_POCKET_RECESS_CUP_GATE_NON_MUTATION_CONTRACT_V176J = (
    "pocket_recess_cup_gate_checks_family_action_scope_and_explicit_owned_role_ids_only_no_geometry_no_mutation"
)


REBUILD_POCKET_CAP_GATE_EXTRACTION_CHECKPOINT_V176K = (
    "v176k_rebuild_pocket_cap_request_gate_extracted_no_behavior_change"
)

REBUILD_POCKET_CAP_GATE_NON_MUTATION_CONTRACT_V176K = (
    "pocket_cap_gate_checks_family_action_gate_scope_only_no_geometry_no_mutation_no_delete_authorization"
)


REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_CHECKPOINT_V176L = (
    "v176l_rebuild_pocket_recess_cup_patch_assembly_extracted_no_behavior_change"
)

REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_NON_MUTATION_CONTRACT_V176L = (
    "pocket_recess_cup_patch_assembly_combines_generated_wall_and_floor_arrays_only_no_delete_authorization_no_trial_no_mutation"
)


REBUILD_POCKET_CAP_GEOMETRY_HELPER_EXTRACTION_CHECKPOINT_V176M = (
    "v176m_rebuild_pocket_cap_geometry_helper_extracted_no_behavior_change"
)

REBUILD_POCKET_CAP_GEOMETRY_HELPER_NON_MUTATION_CONTRACT_V176M = (
    "pocket_cap_geometry_helper_builds_cap_fan_triangles_only_no_delete_authorization_no_trial_no_mutation"
)


REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U = (
    "v176u_rebuild_pocket_floor_grid_helpers_extracted_no_behavior_change"
)

REBUILD_POCKET_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176U = (
    "pocket_floor_grid_helpers_generate_floor_vertices_and_triangles_only_no_delete_authorization_no_trial_no_mutation"
)


def tuple_ints_v176j(value: object) -> tuple[int, ...]:
    """Return a tuple of ints from loose CandidateData metadata.

    This helper is intentionally local to the non-mutating POCKET request gate so
    the family module can decide whether a POCKET candidate has explicit owned
    side-wall/floor role ids without importing rebuild.py internals.
    """

    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(int(v) for v in value)  # type: ignore[arg-type]
    except Exception:
        try:
            return (int(value),)
        except Exception:
            return ()


def candidate_requests_pocket_recess_cup_rebuild_v176j(*, context: object, candidate_metadata: Mapping[str, object]) -> bool:
    """Return True for the pocket-native floor+side-wall recessed cup rebuild.

    Extraction boundary: this is a family-local request gate only.  It does not
    authorize deletion, does not build a DeletePatchProposal, does not generate
    geometry, does not validate topology, and does not mutate mesh state.  It
    preserves the former rebuild.py predicate semantics exactly: explicit POCKET
    family/entity plus owned side-wall and floor ids plus a recess/cup/floor
    action/scope signal.
    """

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    action = str(candidate_metadata.get("candidate_action", "") or "").strip().lower()
    gate = str(getattr(context, "rebuild_gate", "") or "").strip().lower()
    scope = str(candidate_metadata.get("pocket_rebuild_enable_scope", "") or "").strip().lower()
    if not (entity in {"pocket", "circular_pocket"} or family in {"pocket", "circular_pocket"}):
        return False
    if "cap" in action or "cap" in gate or "cap" in scope or "flatten" in action or "flatten" in scope:
        return False
    side_ids = tuple_ints_v176j(candidate_metadata.get("pocket_side_wall_face_ids", ()))
    floor_ids = tuple_ints_v176j(candidate_metadata.get("pocket_floor_face_ids", ()))
    return bool(
        side_ids
        and floor_ids
        and (
            "recess" in action
            or "recess" in gate
            or "recess" in scope
            or "cup" in action
            or "cup" in gate
            or "cup" in scope
            or "owned_floor" in scope
            or "floor" in action
        )
    )


def candidate_requests_pocket_cap_rebuild_v176k(*, context: object, candidate_metadata: Mapping[str, object]) -> bool:
    """Return True for the legacy POCKET floor+wall delete/cap path.

    Extraction boundary: this is a family-local request gate only.  It preserves
    the former rebuild.py predicate semantics exactly.  It does not authorize
    deletion, does not build a DeletePatchProposal, does not generate cap
    geometry, does not validate topology, and does not mutate mesh state.
    """

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    action = str(candidate_metadata.get("candidate_action", "") or "").strip().lower()
    gate = str(getattr(context, "rebuild_gate", "") or "").strip().lower()
    scope = str(candidate_metadata.get("pocket_rebuild_enable_scope", "") or "").strip().lower()
    return bool(
        entity in {"pocket", "circular_pocket"}
        or family in {"pocket", "circular_pocket"}
    ) and bool(
        "cap" in action
        or "cap" in gate
        or "cap" in scope
        or "floor_and_sidewall" in action
        or "side_wall_plus_floor" in scope
    )


@dataclass(frozen=True, slots=True)
class GeneratedPatchPlan:
    """Family-local generated replacement geometry plan.

    The plan is data only.  It does not authorize deletion, does not apply the
    patch to a mesh, does not validate topology, and does not return a committed
    rebuild result.  ``rebuild.py`` remains responsible for all mutation and
    final acceptance decisions.
    """

    removed_face_ids: tuple[int, ...]
    generated_vertices: np.ndarray
    triangles: np.ndarray
    logical_quads: tuple[tuple[int, int, int, int], ...]
    loop0: tuple[int, ...]
    loop1: tuple[int, ...]
    center0: np.ndarray
    center1: np.ndarray
    axis: np.ndarray
    radius: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PocketRecessCupPatchPlanShell:
    """Typed shell for a future POCKET recess-cup generated patch plan.

    v176i intentionally does not route active rebuild execution through this
    type yet.  It pins the data boundary that a later patch can use when moving
    family-local POCKET patch generation out of ``rebuild.py``.
    """

    patch_plan: GeneratedPatchPlan
    wall_face_ids: tuple[int, ...]
    floor_face_ids: tuple[int, ...]
    protected_child_loop_vertex_counts: tuple[int, ...] = ()



def assemble_pocket_recess_cup_patch_plan_v176l(
    *,
    removed_face_ids: tuple[int, ...],
    wall_generated_vertices: np.ndarray,
    wall_triangles: np.ndarray,
    wall_logical_quads: tuple[tuple[int, int, int, int], ...],
    wall_loop0: tuple[int, ...],
    wall_loop1: tuple[int, ...],
    wall_center0: np.ndarray,
    wall_center1: np.ndarray,
    wall_axis: np.ndarray,
    wall_diagnostics: Mapping[str, object],
    floor_generated_vertices: np.ndarray,
    floor_triangles: np.ndarray,
    floor_logical_quads: tuple[tuple[int, int, int, int], ...],
    floor_diagnostics: Mapping[str, object],
    cap_loop: tuple[int, ...],
    wall_face_ids: tuple[int, ...],
    floor_face_ids: tuple[int, ...],
    wall_boundary_loop_count: int,
    floor_boundary_loop_count: int,
    combined_boundary_loop_count: int,
    top_loop_source: str,
    bottom_loop_source: str,
    loop_resolution_status: str,
    loop_resolution_source: str,
    protected_floor_hole_vertex_counts: tuple[int, ...] = (),
    pocket_input_loop_phase_trace_v175j: Mapping[str, object] | None = None,
    pocket_planned_loop_phase_trace_v175j: Mapping[str, object] | None = None,
    radius: float = 0.0,
) -> GeneratedPatchPlan:
    """Assemble the generated POCKET recess-cup patch plan data.

    This helper is deliberately non-mutating: it only combines the side-wall
    generated patch, the generated floor patch, and the diagnostics into the
    family-local patch-plan data boundary.  It does not authorize deletion,
    does not apply the patch to a mesh, does not trial/validate topology, and
    does not emit a final rebuild result.  ``rebuild.py`` remains responsible
    for converting this data into its runtime ``QuadPlan``, trialing it, and
    applying it to the mesh only after validation.
    """

    wall_generated_vertices = np.asarray(wall_generated_vertices, dtype=float).reshape((-1, 3))
    floor_generated_vertices = np.asarray(floor_generated_vertices, dtype=float).reshape((-1, 3))
    wall_triangles = np.asarray(wall_triangles, dtype=np.int64).reshape((-1, 3))
    floor_triangles = np.asarray(floor_triangles, dtype=np.int64).reshape((-1, 3))

    generated_vertices = np.vstack([wall_generated_vertices, floor_generated_vertices]).reshape((-1, 3))
    triangles = np.vstack([wall_triangles, floor_triangles]).reshape((-1, 3))

    input_trace = dict(pocket_input_loop_phase_trace_v175j or {})
    planned_trace = dict(pocket_planned_loop_phase_trace_v175j or {})
    floor_kind = str(dict(floor_diagnostics).get("pocket_floor_fill_kind", "quad_grid"))

    diagnostics = {
        **dict(wall_diagnostics),
        **dict(floor_diagnostics),
        "pocket_recess_cup_patch_assembly_checkpoint_v176l": REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_CHECKPOINT_V176L,
        "pocket_recess_cup_patch_assembly_contract_v176l": REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_NON_MUTATION_CONTRACT_V176L,
        "pocket_recess_cup_patch_assembly_extracted_v176l": True,
        "plan_type": "pocket_recess_cup_rebuild_plan_v100_quad_floor",
        "geometry_source": "owned_pocket_wall_loops_plus_owned_floor_perimeter",
        "semantic_rebuild_contract": "POCKET CandidateData -> owned floor+side-wall DeletePatchProposal -> side-wall plus quad-floor generated cup",
        "pocket_rebuild_operation": "restore_recess_cup",
        "pocket_loop_resolution_status_v144": str(loop_resolution_status),
        "pocket_loop_resolution_source_v144": str(loop_resolution_source),
        "pocket_wall_boundary_loop_count_v144": int(wall_boundary_loop_count),
        "pocket_floor_boundary_loop_count_v144": int(floor_boundary_loop_count),
        "pocket_combined_boundary_loop_count_v144": int(combined_boundary_loop_count),
        "pocket_top_loop_source_v144": str(top_loop_source),
        "pocket_bottom_loop_source_v144": str(bottom_loop_source),
        "pocket_rebuild_target_loop_status": "top_and_floor_loops_resolved",
        "pocket_rebuild_floor_loop_fallback_used_v144": bool(str(loop_resolution_source).endswith("v144")),
        "pocket_rebuild_enable_gate_v144": "rebuild_allowed_only_after_top_and_floor_loop_resolution",
        "pocket_delete_patch_meaning": "owned_pocket_floor_plus_owned_pocket_side_wall",
        "pocket_top_opening_policy": "preserve_opening_not_parent_surface_cap",
        "pocket_transition_policy": "transition_faces_excluded_unless_separately_owned",
        "pocket_side_wall_face_count": int(len(wall_face_ids)),
        "pocket_floor_face_count": int(len(floor_face_ids)),
        "pocket_delete_face_count": int(len(removed_face_ids)),
        "pocket_floor_cap_loop_vertex_count": int(len(cap_loop)),
        "pocket_floor_cap_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_cap_generated_center_vertex_count": 0,
        "pocket_floor_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_logical_quad_count": int(len(tuple(floor_logical_quads))),
        "pocket_floor_fill_kind": floor_kind,
        "pocket_floor_protected_child_bore_opening_count": int(len(protected_floor_hole_vertex_counts)),
        "pocket_floor_protected_child_bore_loop_vertex_counts": tuple(int(v) for v in protected_floor_hole_vertex_counts),
        "compound_pocket_bore_semantics": "POCKET rebuild protects child BORE floor opening as relationship metadata; BORE remains separate candidate",
        "pocket_wall_added_triangle_count": int(len(wall_triangles)),
        "pocket_wall_logical_quad_count": int(len(tuple(wall_logical_quads))),
        "pocket_rebuild_frame_trace_checkpoint_v175j": True,
        "pocket_rebuild_frame_trace_semantic_contract_v175j": "diagnostic_only_loop_phase_and_radius_authority_trace_no_geometry_change",
        "pocket_input_loop_phase_trace_v175j": input_trace,
        "pocket_planned_loop_phase_trace_v175j": planned_trace,
        "pocket_input_loop_phase_wave_risk_v175j": bool(input_trace.get("phase_wave_risk", False)),
        "pocket_planned_loop_phase_wave_risk_v175j": bool(planned_trace.get("phase_wave_risk", False)),
        "pocket_input_loop0_count_v175j": int(input_trace.get("loop0_count", 0) or 0),
        "pocket_input_loop1_count_v175j": int(input_trace.get("loop1_count", 0) or 0),
        "pocket_planned_loop0_count_v175j": int(planned_trace.get("loop0_count", 0) or 0),
        "pocket_planned_loop1_count_v175j": int(planned_trace.get("loop1_count", 0) or 0),
        "pocket_planned_loop_best_shift_v175j": int(planned_trace.get("best_cyclic_shift", 0) or 0),
        "pocket_planned_loop_best_shift_degrees_v175j": float(planned_trace.get("best_cyclic_shift_degrees", 0.0) or 0.0),
        "pocket_planned_loop_reversed_v175j": bool(planned_trace.get("best_alignment_reversed_loop1", False)),
        "pocket_planned_loop_mean_chord_error_v175j": float(planned_trace.get("best_alignment_mean_unit_chord_error", 0.0) or 0.0),
        "pocket_planned_loop_max_chord_error_v175j": float(planned_trace.get("best_alignment_max_unit_chord_error", 0.0) or 0.0),
    }

    return GeneratedPatchPlan(
        removed_face_ids=tuple(int(fid) for fid in removed_face_ids),
        generated_vertices=generated_vertices,
        triangles=triangles,
        logical_quads=tuple(wall_logical_quads) + tuple(floor_logical_quads),
        loop0=tuple(int(v) for v in tuple(wall_loop0 or ())),
        loop1=tuple(int(v) for v in tuple(wall_loop1 or ())),
        center0=np.asarray(wall_center0, dtype=float).reshape(3),
        center1=np.asarray(wall_center1, dtype=float).reshape(3),
        axis=np.asarray(wall_axis, dtype=float).reshape(3),
        radius=float(radius),
        diagnostics=diagnostics,
    )



# -----------------------------------------------------------------------------
# POCKET floor-grid helpers (v176u)
# -----------------------------------------------------------------------------


def pocket_floor_quad_grid_from_loop_v176u(
    *,
    vertices: np.ndarray,
    loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    quad_density_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Return a quad-dominant floor grid for the owned POCKET floor role.

    This is the floor companion to the wall-loop quad rebuild.  The boundary
    loop remains locked to the original pocket floor perimeter.  Generated
    inner rings shrink toward the floor center and gradually reduce vertex
    count down to a small even core loop, which is closed with logical quads.
    No center-vertex fan is used for even boundary loops such as the current
    126-vertex circular-pocket test.
    """

    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 4:
        raise ValueError("Pocket quad floor requires at least four boundary vertices.")

    n = int(len(loop))
    if n % 2 == 0 and n >= 6:
        target_count = 6
        core_fill_kind = "two_logical_quads_hex_core"
    elif n % 2 == 0 and n >= 4:
        target_count = 4
        core_fill_kind = "single_logical_quad_core"
    else:
        # A pure all-quad disk with an odd locked boundary is topologically not
        # possible without inserting/splitting boundary vertices.  Keep this
        # fallback explicit and diagnostic; current circular-pocket tests use an
        # even boundary and therefore stay in the quad path.
        return pocket_floor_center_fan_fallback_v176u(
            vertices=vertices,
            loop_vertices=loop,
            generated_vertex_offset=generated_vertex_offset,
            reason="odd_locked_boundary_count_cannot_form_pure_quad_disk",
        )

    pts = vertices[np.asarray(loop, dtype=np.int64), :3]
    center = np.mean(pts, axis=0).reshape(3)
    # Use the existing density vocabulary, but apply it radially across the
    # floor rather than axially along the wall.  The counts still control
    # topology; density controls how many equal-count spacer bands appear.
    base_counts = _transition_count_sequence(int(n), int(target_count), mode=quad_density_mode)
    radial_span = float(np.median(np.linalg.norm(pts - center.reshape(1, 3), axis=1)))
    edge_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1.0e-12)]
    measured_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
    mode_norm = _normalize_quad_density_mode(quad_density_mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL:
        pitch = max(float(measured_edge), 1.0e-12)
        cap = 96
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        pitch = max(2.5 * float(measured_edge), 1.0e-12)
        cap = 64
    else:
        pitch = max(4.0 * float(measured_edge), 1.0e-12)
        cap = 48
    raw_radial_bands = int(math.ceil(radial_span / pitch)) if np.isfinite(radial_span) and radial_span > 1.0e-12 else len(base_counts) - 1
    target_band_count = max(int(len(base_counts) - 1), min(int(cap), max(1, int(raw_radial_bands))))
    counts = _densify_transition_count_sequence(base_counts, target_band_count=target_band_count)
    counts = tuple(int(v) for v in counts if int(v) >= int(target_count))
    if counts[0] != n:
        counts = (int(n),) + tuple(counts)
    if counts[-1] != target_count:
        counts = tuple(counts) + (int(target_count),)

    # Avoid a collapsed center: the final generated ring is a small measured
    # floor core, then the core itself is closed with logical quads.
    inner_scale = 0.08 if int(target_count) >= 6 else 0.12
    band_count = max(1, len(counts) - 1)
    generated_blocks: list[np.ndarray] = []
    ring_ids: list[tuple[int, ...]] = [tuple(int(v) for v in loop)]
    next_generated_id = int(len(vertices) + int(generated_vertex_offset))
    for ring_index, count in enumerate(counts[1:], start=1):
        t = float(ring_index) / float(band_count)
        scale = (1.0 - t) + t * float(inner_scale)
        sample = _sample_closed_loop_points(pts, int(count))
        ring_pts = center.reshape(1, 3) + scale * (sample - center.reshape(1, 3))
        generated_blocks.append(np.asarray(ring_pts, dtype=float).reshape((int(count), 3)))
        ids = tuple(range(next_generated_id, next_generated_id + int(count)))
        next_generated_id += int(count)
        ring_ids.append(ids)

    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    for band in range(len(ring_ids) - 1):
        band_quads = _band_quads_between_rings(tuple(ring_ids[band]), tuple(ring_ids[band + 1]))
        logical_quads.extend(band_quads)
        for quad in band_quads:
            triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
            triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    core = tuple(int(v) for v in ring_ids[-1])
    core_quads: tuple[tuple[int, int, int, int], ...]
    if len(core) == 4:
        core_quads = ((core[0], core[1], core[2], core[3]),)
    elif len(core) == 6:
        core_quads = ((core[0], core[1], core[2], core[3]), (core[0], core[3], core[4], core[5]))
    else:
        # Keep the function honest if the target count is changed later.
        return pocket_floor_center_fan_fallback_v176u(
            vertices=vertices,
            loop_vertices=loop,
            generated_vertex_offset=generated_vertex_offset,
            reason=f"unsupported_core_loop_count_{len(core)}",
        )
    logical_quads.extend(core_quads)
    for quad in core_quads:
        triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
        triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    generated_vertices = np.vstack(generated_blocks).reshape((-1, 3)) if generated_blocks else np.zeros((0, 3), dtype=float)
    diag = {
        "pocket_floor_fill_kind": "quad_grid",
        "pocket_floor_grid_contract_v100": "locked_floor_perimeter_to_concentric_quad_grid_no_center_fan",
        "pocket_floor_boundary_vertex_count": int(n),
        "pocket_floor_inner_core_vertex_count": int(target_count),
        "pocket_floor_core_fill_kind": str(core_fill_kind),
        "pocket_floor_ring_count_sequence": tuple(int(v) for v in counts),
        "pocket_floor_radial_band_count": int(len(ring_ids) - 1),
        "pocket_floor_generated_ring_count": int(max(0, len(ring_ids) - 1)),
        "pocket_floor_generated_vertex_count": int(len(generated_vertices)),
        "pocket_floor_logical_quad_count": int(len(logical_quads)),
        "pocket_floor_triangle_count": int(len(triangles)),
        "pocket_floor_center_fan_used": False,
        "quad_density_mode": str(_normalize_quad_density_mode(quad_density_mode)),
    }
    return (
        np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        tuple((int(a), int(b), int(c), int(d)) for a, b, c, d in logical_quads),
        diag,
    )


def pocket_floor_center_fan_fallback_v176u(
    *,
    vertices: np.ndarray,
    loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    reason: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Fallback only for non-quad-compatible locked boundaries."""

    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 3:
        raise ValueError("Pocket floor fallback requires at least three boundary vertices.")
    pts = vertices[np.asarray(loop, dtype=np.int64), :3]
    center = np.mean(pts, axis=0).reshape(1, 3)
    center_id = int(len(vertices) + int(generated_vertex_offset))
    triangles = [(int(a), int(loop[(i + 1) % len(loop)]), int(center_id)) for i, a in enumerate(loop)]
    return (
        np.asarray(center, dtype=float).reshape((1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        (),
        {
            "pocket_floor_fill_kind": "diagnostic_center_fan_fallback",
            "pocket_floor_center_fan_used": True,
            "pocket_floor_quad_fallback_reason": str(reason),
            "pocket_floor_logical_quad_count": 0,
            "pocket_floor_triangle_count": int(len(triangles)),
            "pocket_floor_generated_vertex_count": 1,
        },
    )


def _unit_vector_v176m(value: object, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Local unit-vector helper for non-mutating POCKET cap fan orientation."""

    try:
        vec = np.asarray(value, dtype=float).reshape(3)
    except Exception:
        vec = np.asarray(fallback, dtype=float).reshape(3)
    length = float(np.linalg.norm(vec))
    if np.isfinite(length) and length > 1.0e-12:
        return vec / length
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fb_len = float(np.linalg.norm(fb))
    if np.isfinite(fb_len) and fb_len > 1.0e-12:
        return fb / fb_len
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _triangle_set_area_normal_v176m(*, vertices: np.ndarray, triangles: np.ndarray, generated_center_index: int) -> np.ndarray:
    """Return the area-weighted normal for a candidate POCKET cap triangle fan."""

    arr = np.asarray(vertices, dtype=float)[:, :3]
    center_point = None
    if int(generated_center_index) >= len(arr):
        # The cap center is the mean of the original boundary vertices used in
        # the fan.  It is reconstructed here only for orientation testing.
        original_ids = sorted({int(v) for tri in np.asarray(triangles, dtype=np.int64).reshape((-1, 3)) for v in tri if 0 <= int(v) < len(arr)})
        center_point = np.mean(arr[np.asarray(original_ids, dtype=np.int64), :3], axis=0) if original_ids else np.zeros(3, dtype=float)
    acc = np.zeros(3, dtype=float)
    for tri in np.asarray(triangles, dtype=np.int64).reshape((-1, 3)):
        pts = []
        for raw in tri[:3]:
            idx = int(raw)
            if 0 <= idx < len(arr):
                pts.append(arr[idx, :3])
            elif idx == int(generated_center_index) and center_point is not None:
                pts.append(center_point)
        if len(pts) == 3:
            acc += np.cross(pts[1] - pts[0], pts[2] - pts[0])
    return _unit_vector_v176m(acc, fallback=(0.0, 0.0, 1.0))


def cap_triangles_for_loop_v176m(*, loop_vertices: tuple[int, ...], center_index: int, vertices: np.ndarray, normal_hint: np.ndarray) -> np.ndarray:
    """Build oriented planar-cap fan triangles for the legacy POCKET cap path.

    This helper is intentionally non-mutating.  It generates only triangle index
    data for the cap patch; rebuild.py still owns boundary matching, trial mesh
    construction, validation, application, and final RebuildResult emission.
    """

    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    tris = np.asarray([[loop[i], loop[(i + 1) % len(loop)], int(center_index)] for i in range(len(loop))], dtype=np.int64)
    normal = _triangle_set_area_normal_v176m(vertices=vertices, triangles=tris, generated_center_index=int(center_index))
    hint = _unit_vector_v176m(normal_hint, fallback=(0.0, 0.0, 1.0))
    if float(np.dot(normal, hint)) < 0.0:
        tris = np.asarray([[loop[(i + 1) % len(loop)], loop[i], int(center_index)] for i in range(len(loop))], dtype=np.int64)
    return tris



REBUILD_POCKET_ADAPTIVE_SEW_COLLAR_HELPER_EXTRACTION_CHECKPOINT_V176V = (
    "v176v_rebuild_pocket_adaptive_sew_collar_helpers_extracted_no_behavior_change"
)

REBUILD_POCKET_ADAPTIVE_SEW_COLLAR_HELPER_NON_MUTATION_CONTRACT_V176V = (
    "pocket_adaptive_sew_collar_helpers_generate_ring_band_faces_only_no_delete_authorization_no_trial_no_mutation"
)


REBUILD_POCKET_ANNULAR_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176W = (
    "v176w_rebuild_pocket_annular_floor_grid_helper_extracted_no_behavior_change"
)

REBUILD_POCKET_ANNULAR_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176W = (
    "pocket_annular_floor_grid_helper_generates_floor_vertices_and_faces_only_no_delete_authorization_no_trial_no_mutation"
)

def pocket_band_faces_between_rings_with_adaptive_sew_collar_v176v(
    ring_a: tuple[int, ...],
    ring_b: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int], ...], dict[str, object]]:
    """Return runtime triangles and logical quads between two locked rings.

    First try the existing all-quad transition band.  If locked loop counts make
    that impossible, fall back to a local zipper-style adaptive sew collar.  The
    fallback preserves all boundary vertices and consumes every boundary edge
    exactly once; it is intentionally used only as the boundary adapter while
    the annular floor interior remains regular quads.
    """

    a = tuple(int(v) for v in tuple(ring_a or ()))
    b = tuple(int(v) for v in tuple(ring_b or ()))
    n_a = int(len(a))
    n_b = int(len(b))
    if n_a < 3 or n_b < 3:
        return (), (), {
            "adapter_kind": "invalid_short_ring",
            "adaptive_sew_collar_used": False,
            "quad_count": 0,
            "triangle_count": 0,
        }

    try:
        quads = _band_quads_between_rings(a, b)
        tris: list[tuple[int, int, int]] = []
        for q in tuple(quads or ()): 
            tris.append((int(q[0]), int(q[1]), int(q[2])))
            tris.append((int(q[0]), int(q[2]), int(q[3])))
        return (
            tuple(tuple(int(v) for v in q) for q in tuple(quads or ())),
            tuple(tuple(int(v) for v in tri) for tri in tris),
            {
                "adapter_kind": "pure_quad_transition_band",
                "adaptive_sew_collar_used": False,
                "quad_count": int(len(quads)),
                "triangle_count": 0,
                "ring_a_count": int(n_a),
                "ring_b_count": int(n_b),
            },
        )
    except Exception as exc:
        quads, tris = pocket_adaptive_zipper_sew_collar_faces_between_rings_v176v(a, b)
        return (
            tuple(tuple(int(v) for v in q) for q in tuple(quads or ())),
            tuple(tuple(int(v) for v in tri) for tri in tuple(tris or ())),
            {
                "adapter_kind": "adaptive_zipper_sew_collar",
                "adaptive_sew_collar_used": True,
                "all_quad_transition_rejection": str(exc),
                "quad_count": int(len(quads)),
                "triangle_count": int(len(tris) - 2 * len(quads)),
                "runtime_triangle_count": int(len(tris)),
                "ring_a_count": int(n_a),
                "ring_b_count": int(n_b),
            },
        )


def pocket_adaptive_zipper_sew_collar_faces_between_rings_v176v(
    ring_a: tuple[int, ...],
    ring_b: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Build a local adaptive collar between arbitrary locked loop counts.

    The output is quad-dominant where the angular steps coincide and uses local
    transition triangles only where count parity/ratio makes an all-quad band
    topologically impossible without splitting locked boundary edges.
    """

    a = tuple(int(v) for v in tuple(ring_a or ()))
    b = tuple(int(v) for v in tuple(ring_b or ()))
    n_a = int(len(a))
    n_b = int(len(b))
    if n_a < 3 or n_b < 3:
        return (), ()

    quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    ia = 0
    ib = 0
    ca = 0
    cb = 0
    eps = 1.0e-9
    # Merge the two cyclic edge streams by normalized perimeter progress.  This
    # consumes every boundary edge exactly once and never creates a split vertex
    # on a locked loop.
    while ca < n_a or cb < n_b:
        next_a = float(ca + 1) / float(n_a) if ca < n_a else float("inf")
        next_b = float(cb + 1) / float(n_b) if cb < n_b else float("inf")
        if ca < n_a and cb < n_b and abs(next_a - next_b) <= eps:
            q = (int(a[ia % n_a]), int(a[(ia + 1) % n_a]), int(b[(ib + 1) % n_b]), int(b[ib % n_b]))
            quads.append(q)
            triangles.append((q[0], q[1], q[2]))
            triangles.append((q[0], q[2], q[3]))
            ia += 1
            ib += 1
            ca += 1
            cb += 1
        elif ca < n_a and (cb >= n_b or next_a < next_b):
            tri = (int(a[ia % n_a]), int(a[(ia + 1) % n_a]), int(b[ib % n_b]))
            if len(set(tri)) == 3:
                triangles.append(tri)
            ia += 1
            ca += 1
        elif cb < n_b:
            tri = (int(a[ia % n_a]), int(b[(ib + 1) % n_b]), int(b[ib % n_b]))
            if len(set(tri)) == 3:
                triangles.append(tri)
            ib += 1
            cb += 1
        else:
            break
    return tuple(quads), tuple(triangles)


def pocket_floor_annular_quad_grid_from_loops_v176w(
    *,
    vertices: np.ndarray,
    outer_loop_vertices: tuple[int, ...],
    inner_loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    floor_axis: np.ndarray,
    quad_density_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Return an adaptive annular floor grid for a POCKET with a child BORE.

    v104 semantic contract:
        POCKET remains the parent feature family.
        BORE remains a separate child feature candidate.
        The floor hole is relationship/protected-boundary metadata.
        Rebuild must sew to the locked outer pocket-floor boundary and the
        locked inner child-BORE opening without changing either boundary.

    This replaces the v103 "fail on odd count delta" policy.  Pure all-quad
    bands are used whenever the loop counts allow them.  When the two locked
    boundaries cannot be connected by a pure all-quad annulus without splitting
    existing boundary edges, the generator creates adaptive sew collars at the
    locked boundary side(s), while keeping the interior floor as a regular
    density-controlled quad grid.  The fallback collar is local, explicit, and
    diagnostic; it does not close the child BORE and does not invent a new
    feature family.
    """

    outer = tuple(int(v) for v in tuple(outer_loop_vertices or ()) if 0 <= int(v) < len(vertices))
    inner = tuple(int(v) for v in tuple(inner_loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(outer) < 4 or len(inner) < 4:
        raise ValueError(
            "POCKET annular floor rebuild requires valid outer and protected inner loops with at least four vertices. "
            f"outer={len(outer)}; inner={len(inner)}; Geometry changed: no."
        )

    axis_vec = _unit_vector(floor_axis)
    outer_center = _loop_center(vertices, outer)
    inner_center = _loop_center(vertices, inner)
    sorted_outer, sorted_inner = _sort_loop_pair_by_angle(vertices, outer, inner, outer_center, inner_center, axis_vec)

    # Align loops by angular phase.  Equal loops can use the exact measured loop
    # alignment.  Unequal loops use angle-sample alignment only; this is still
    # relationship/geometry evidence, not feature-family creation.
    if len(sorted_outer) == len(sorted_inner):
        sorted_inner = _align_second_loop_to_first(
            vertices,
            tuple(int(v) for v in sorted_outer),
            tuple(int(v) for v in sorted_inner),
            outer_center,
            inner_center,
            axis_vec,
        )
        alignment_diag: dict[str, object] = {
            "annular_floor_equal_loop_used": True,
            "annular_floor_unequal_loop_used": False,
            "annular_floor_alignment_source": "equal_loop_cyclic_alignment",
        }
    else:
        aligned_outer, aligned_inner, alignment_diag_raw = _align_unequal_loop_pair_to_angle_samples(
            vertices=vertices,
            loop0=tuple(int(v) for v in sorted_outer),
            loop1=tuple(int(v) for v in sorted_inner),
            center0=outer_center,
            center1=inner_center,
            axis=axis_vec,
        )
        sorted_outer = tuple(int(v) for v in aligned_outer)
        sorted_inner = tuple(int(v) for v in aligned_inner)
        alignment_diag = {
            **dict(alignment_diag_raw),
            "annular_floor_equal_loop_used": False,
            "annular_floor_unequal_loop_used": True,
            "annular_floor_alignment_source": "unequal_loop_angle_sample_alignment",
        }

    outer_pts = vertices[np.asarray(sorted_outer, dtype=np.int64), :3]
    inner_pts = vertices[np.asarray(sorted_inner, dtype=np.int64), :3]
    n_outer = int(len(sorted_outer))
    n_inner = int(len(sorted_inner))

    sample_count = max(int(n_outer), int(n_inner), 8)
    outer_sample = _sample_closed_loop_points(outer_pts, sample_count)
    inner_sample = _sample_closed_loop_points(inner_pts, sample_count)
    radial_gaps = np.linalg.norm(outer_sample - inner_sample, axis=1)
    radial_gaps = radial_gaps[np.isfinite(radial_gaps) & (radial_gaps > 1.0e-12)]
    radial_span = float(np.median(radial_gaps)) if radial_gaps.size else float(np.linalg.norm(outer_center - inner_center))
    outer_edges = np.linalg.norm(np.roll(outer_pts, -1, axis=0) - outer_pts, axis=1)
    inner_edges = np.linalg.norm(np.roll(inner_pts, -1, axis=0) - inner_pts, axis=1)
    edge_lengths = np.concatenate([outer_edges, inner_edges])
    edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1.0e-12)]
    measured_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
    measured_edge = max(float(measured_edge), 1.0e-12)

    mode_norm = _normalize_quad_density_mode(quad_density_mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL:
        pitch = measured_edge
        cap = 128
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        pitch = max(2.5 * measured_edge, radial_span / 32.0 if radial_span > 1.0e-12 else measured_edge)
        cap = 64
    else:
        pitch = max(4.0 * measured_edge, radial_span / 24.0 if radial_span > 1.0e-12 else measured_edge)
        cap = 48
    raw_radial_bands = int(math.ceil(radial_span / max(pitch, 1.0e-12))) if np.isfinite(radial_span) and radial_span > 1.0e-12 else 1

    # Adaptive sew-collar strategy:
    #   locked outer loop -> optional generated common-count grid rings -> locked inner loop
    # The interior rings all use a common count so the floor center area remains
    # an orderly quad grid.  Only the boundary collars adapt to mismatched
    # source geometry.  This avoids failing when loop counts do not meet the
    # strict pure-quad parity rule, while still preserving both locked loops.
    common_count = int(max(n_outer, n_inner, 4))
    count_delta = abs(int(n_outer) - int(n_inner))
    needs_adaptive_collar = bool(count_delta % 2 or count_delta != 0)
    target_band_count = max(1, min(int(cap), int(raw_radial_bands)))
    if needs_adaptive_collar:
        target_band_count = max(2, target_band_count)

    generated_blocks: list[np.ndarray] = []
    ring_ids: list[tuple[int, ...]] = [tuple(int(v) for v in sorted_outer)]
    next_generated_id = int(len(vertices) + int(generated_vertex_offset))
    for ring_index in range(1, int(target_band_count)):
        t = float(ring_index) / float(target_band_count)
        out_sample = _sample_closed_loop_points(outer_pts, common_count)
        in_sample = _sample_closed_loop_points(inner_pts, common_count)
        pts = (1.0 - t) * out_sample + t * in_sample
        generated_blocks.append(np.asarray(pts, dtype=float).reshape((common_count, 3)))
        ids = tuple(range(next_generated_id, next_generated_id + common_count))
        next_generated_id += common_count
        ring_ids.append(ids)
    ring_ids.append(tuple(int(v) for v in sorted_inner))

    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    band_count_pairs: list[tuple[int, int]] = []
    band_quad_counts: list[int] = []
    band_triangle_counts: list[int] = []
    band_adapter_kinds: list[str] = []
    adaptive_collar_band_indices: list[int] = []
    for band_index in range(len(ring_ids) - 1):
        ring_a = tuple(int(v) for v in ring_ids[band_index])
        ring_b = tuple(int(v) for v in ring_ids[band_index + 1])
        band_count_pairs.append((int(len(ring_a)), int(len(ring_b))))
        band_quads, band_tris, band_diag = pocket_band_faces_between_rings_with_adaptive_sew_collar_v176v(ring_a, ring_b)
        logical_quads.extend(band_quads)
        triangles.extend(band_tris)
        band_quad_counts.append(int(len(band_quads)))
        band_triangle_counts.append(int(band_diag.get("triangle_count", 0)))
        adapter_kind = str(band_diag.get("adapter_kind", "unknown"))
        band_adapter_kinds.append(adapter_kind)
        if bool(band_diag.get("adaptive_sew_collar_used", False)):
            adaptive_collar_band_indices.append(int(band_index))

    generated_vertices = np.vstack(generated_blocks).reshape((-1, 3)) if generated_blocks else np.zeros((0, 3), dtype=float)
    diagnostics = {
        **dict(alignment_diag),
        "pocket_floor_fill_kind": "annular_adaptive_sew_collar_quad_grid_protected_child_bore_opening",
        "pocket_floor_center_fan_used": False,
        "pocket_floor_annular_fill_used": True,
        "pocket_floor_annular_concentric_grid_used": True,
        "pocket_floor_adaptive_sew_collar_used": bool(adaptive_collar_band_indices),
        "pocket_floor_adaptive_sew_collar_band_indices": tuple(int(v) for v in adaptive_collar_band_indices),
        "pocket_floor_annular_grid_contract_v104": "locked_outer_floor_perimeter_to_locked_child_bore_opening_with_adaptive_sew_collars_and_density_controlled_quad_interior",
        "pocket_floor_annular_grid_contract_v103_reverted": "no_clean_fail_on_odd_count_delta; use adaptive sew collars instead of rejecting rebuild",
        "pocket_floor_protected_inner_loop_count": 1,
        "pocket_floor_outer_loop_vertex_count": int(n_outer),
        "pocket_floor_inner_bore_loop_vertex_count": int(n_inner),
        "pocket_floor_annular_common_interior_ring_count": int(common_count),
        "pocket_floor_annular_radial_span": float(radial_span),
        "pocket_floor_annular_median_edge_length": float(measured_edge),
        "pocket_floor_annular_target_pitch": float(pitch),
        "pocket_floor_annular_raw_radial_bands": int(raw_radial_bands),
        "pocket_floor_annular_radial_band_count": int(len(ring_ids) - 1),
        "pocket_floor_annular_generated_ring_count": int(max(0, len(ring_ids) - 2)),
        "pocket_floor_annular_ring_count_sequence": tuple(int(len(v)) for v in ring_ids),
        "pocket_floor_annular_band_count_pairs": tuple((int(a), int(b)) for a, b in band_count_pairs),
        "pocket_floor_annular_band_adapter_kinds": tuple(str(v) for v in band_adapter_kinds),
        "pocket_floor_annular_band_quad_counts": tuple(int(v) for v in band_quad_counts),
        "pocket_floor_annular_band_triangle_counts": tuple(int(v) for v in band_triangle_counts),
        "pocket_floor_annular_triangle_count_from_adaptive_collars": int(sum(band_triangle_counts)),
        "pocket_floor_logical_quad_count": int(len(logical_quads)),
        "pocket_floor_triangle_count": int(len(triangles)),
        "pocket_floor_generated_vertex_count": int(len(generated_vertices)),
        "quad_density_mode": str(mode_norm),
        "semantic_floor_contract": "POCKET floor rebuilt as annular quad-grid interior with adaptive sew collars; protected child BORE opening remains open and separate",
    }
    return (
        np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        tuple(tuple(int(v) for v in quad) for quad in tuple(logical_quads or ())),
        diagnostics,
    )

def pocket_patch_plan_shell_inventory_v176i() -> dict[str, object]:
    """Return the no-behavior-change POCKET patch-plan shell inventory."""

    return {
        "checkpoint": REBUILD_POCKET_PATCH_PLAN_SHELL_CHECKPOINT_V176I,
        "non_mutation_contract": REBUILD_POCKET_PATCH_PLAN_SHELL_NON_MUTATION_CONTRACT_V176I,
        "fields": tuple(REBUILD_POCKET_PATCH_PLAN_FIELDS_V176I),
        "boundary": {
            key: tuple(values)
            for key, values in REBUILD_POCKET_PATCH_PLAN_BOUNDARY_V176I.items()
        },
        "active_behavior_change": False,
        "active_rebuild_path_still_in_rebuild_py": True,
        "active_request_gate_checkpoint": REBUILD_POCKET_RECESS_CUP_GATE_EXTRACTION_CHECKPOINT_V176J,
        "active_request_gate_contract": REBUILD_POCKET_RECESS_CUP_GATE_NON_MUTATION_CONTRACT_V176J,
        "active_request_gate_extracted": True,
        "cap_request_gate_checkpoint": REBUILD_POCKET_CAP_GATE_EXTRACTION_CHECKPOINT_V176K,
        "cap_request_gate_contract": REBUILD_POCKET_CAP_GATE_NON_MUTATION_CONTRACT_V176K,
        "cap_request_gate_extracted": True,
        "pocket_recess_cup_patch_assembly_checkpoint_v176l": REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_CHECKPOINT_V176L,
        "pocket_recess_cup_patch_assembly_contract_v176l": REBUILD_POCKET_RECESS_CUP_PATCH_ASSEMBLY_NON_MUTATION_CONTRACT_V176L,
        "pocket_recess_cup_patch_assembly_available_v176l": True,
        "pocket_cap_geometry_helper_checkpoint_v176m": REBUILD_POCKET_CAP_GEOMETRY_HELPER_EXTRACTION_CHECKPOINT_V176M,
        "pocket_cap_geometry_helper_contract_v176m": REBUILD_POCKET_CAP_GEOMETRY_HELPER_NON_MUTATION_CONTRACT_V176M,
        "pocket_cap_geometry_helper_extracted_v176m": True,
        "pocket_floor_grid_helper_checkpoint_v176u": REBUILD_POCKET_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176U,
        "pocket_floor_grid_helper_contract_v176u": REBUILD_POCKET_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176U,
        "pocket_floor_grid_helpers_extracted_v176u": True,
        "pocket_adaptive_sew_collar_helper_checkpoint_v176v": REBUILD_POCKET_ADAPTIVE_SEW_COLLAR_HELPER_EXTRACTION_CHECKPOINT_V176V,
        "pocket_adaptive_sew_collar_helper_contract_v176v": REBUILD_POCKET_ADAPTIVE_SEW_COLLAR_HELPER_NON_MUTATION_CONTRACT_V176V,
        "pocket_adaptive_sew_collar_helpers_extracted_v176v": True,
        "pocket_annular_floor_grid_helper_checkpoint_v176w": REBUILD_POCKET_ANNULAR_FLOOR_GRID_HELPER_EXTRACTION_CHECKPOINT_V176W,
        "pocket_annular_floor_grid_helper_contract_v176w": REBUILD_POCKET_ANNULAR_FLOOR_GRID_HELPER_NON_MUTATION_CONTRACT_V176W,
        "pocket_annular_floor_grid_helper_extracted_v176w": True,
    }
