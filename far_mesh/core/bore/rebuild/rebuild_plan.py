"""Measured quad-plan builders for BoreTool rebuild.

This module is intentionally non-mutating.  It may build in-memory generated
vertex/triangle/logical-quad plans from measured boundary loops and accepted
semantic CandidateData metadata, but it must not authorize deletion, apply
geometry to a mesh, validate final topology, mutate project state, or emit
RebuildResult.  ``rebuild.py`` remains the public rebuild orchestrator and the
single mesh mutation/trial/result authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .rebuild_density import axial_segment_count_v176q as _axial_segment_count
from .rebuild_geometry import (
    boundary_adapter_weight_v165 as _boundary_adapter_weight_v165,
    endpoint_safe_semantic_weight_v166 as _endpoint_safe_semantic_weight_v166,
    loop_center as _loop_center,
    sample_closed_loop_points as _sample_closed_loop_points,
    unit_vector as _unit_vector,
)
from .rebuild_semantic import (
    SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
    chamfer_local_rail_ring_points_v167_v176z as _chamfer_local_rail_ring_points_v167,
    semantic_constraint_quality_summary_v176z as _semantic_constraint_quality_summary,
    semantic_generated_vertex_shaping_policy_v177a as _semantic_generated_vertex_shaping_policy,
    semantic_projected_ring_points_v176z as _semantic_projected_ring_points,
)
from .rebuild_bore import shape_frame_loop_angles_v173z_v176y as _shape_frame_loop_angles_v173z
from .rebuild_loops import (
    align_second_loop_to_first_v176g as _align_second_loop_to_first,
    align_unequal_loop_pair_to_angle_samples_v176w as _align_unequal_loop_pair_to_angle_samples,
    band_quads_between_rings_v176n as _band_quads_between_rings,
    densify_transition_count_sequence_v176n as _densify_transition_count_sequence,
    sort_loop_by_angle_v176g as _sort_loop_by_angle,
    sort_loop_pair_by_angle_v176g as _sort_loop_pair_by_angle,
    target_unequal_transition_band_count_v176n as _target_unequal_transition_band_count,
    transition_count_sequence_v176n as _transition_count_sequence,
)

REBUILD_QUAD_PLAN_EXTRACTION_CHECKPOINT_V177C = (
    "v177c_rebuild_generic_quad_plan_builders_extracted_no_behavior_change"
)

REBUILD_QUAD_PLAN_NON_MUTATION_CONTRACT_V177C = (
    "quad_plan_builders_may_return_generated_vertices_faces_and_diagnostics_only_rebuild_py_remains_mutation_authority"
)

REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_CHECKPOINT_V177D = (
    "v177d_rebuild_quad_plan_dataclass_extraction_hotfix_no_behavior_change"
)

REBUILD_QUAD_PLAN_DATACLASS_HOTFIX_NON_MUTATION_CONTRACT_V177D = (
    "quad_plan_dataclass_hotfix_restores_runtime_constructor_only_no_geometry_no_mutation_change"
)

@dataclass(frozen=True, slots=True)
class QuadPlan:
    """Runtime triangle backing plus logical quad diagnostics."""

    generated_vertices: np.ndarray
    triangles: np.ndarray
    logical_quads: tuple[tuple[int, int, int, int], ...]
    loop0: tuple[int, ...]
    loop1: tuple[int, ...]
    center0: np.ndarray
    center1: np.ndarray
    axis: np.ndarray
    diagnostics: dict[str, object]


def loop_pair_transition_allowed_v177c(
    n0: int,
    n1: int,
    *,
    allow_unequal_loop_transition: bool,
) -> bool:
    n0 = int(n0)
    n1 = int(n1)
    if n0 < 3 or n1 < 3:
        return False
    if n0 == n1:
        return True
    if not allow_unequal_loop_transition:
        return False
    total = n0 + n1
    delta = abs(n0 - n1)
    if total % 2 != 0 or delta % 2 != 0:
        return False
    return (delta // 2) <= min(n0, n1)


def quad_plan_for_attempt_v177c(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    attempt: BoundaryLoopAttempt,
    axis: np.ndarray,
    context: RebuildCandidateContext,
    candidate_metadata: Mapping[str, object],
    quad_density_mode: str,
) -> QuadPlan:
    center_a = _loop_center(vertices, attempt.loop0)
    center_b = _loop_center(vertices, attempt.loop1)
    axis = _unit_vector(axis, fallback=center_b - center_a)
    midpoint = 0.5 * (center_a + center_b)
    axial_a = float(np.dot(center_a - midpoint, axis))
    axial_b = float(np.dot(center_b - midpoint, axis))
    loop_a = attempt.loop0
    loop_b = attempt.loop1
    if axial_b < axial_a:
        loop_a, loop_b = loop_b, loop_a
        center_a, center_b = center_b, center_a
        axial_a, axial_b = axial_b, axial_a
    if abs(axial_b - axial_a) <= 1.0e-12:
        raise ValueError("Measured boundary loops have no usable axial separation.")

    sorted0, sorted1 = _sort_loop_pair_by_angle(vertices, loop_a, loop_b, center_a, center_b, axis)
    if len(sorted0) == len(sorted1):
        aligned1 = _align_second_loop_to_first(vertices, sorted0, sorted1, center_a, center_b, axis)
        plan = equal_loop_quad_plan_v177c(
            vertices=vertices,
            source_faces=source_faces,
            face_ids=attempt.face_ids,
            loop0=tuple(int(v) for v in sorted0),
            loop1=tuple(int(v) for v in aligned1),
            center0=center_a,
            center1=center_b,
            axis=axis,
            context=context,
            candidate_metadata=candidate_metadata,
            quad_density_mode=quad_density_mode,
        )
        return plan

    aligned0, aligned1, alignment_diag = _align_unequal_loop_pair_to_angle_samples(
        vertices=vertices,
        loop0=tuple(int(v) for v in sorted0),
        loop1=tuple(int(v) for v in sorted1),
        center0=center_a,
        center1=center_b,
        axis=axis,
    )
    plan = unequal_loop_quad_plan_v177c(
        vertices=vertices,
        source_faces=source_faces,
        face_ids=attempt.face_ids,
        loop0=aligned0,
        loop1=aligned1,
        center0=center_a,
        center1=center_b,
        axis=axis,
        context=context,
        candidate_metadata=candidate_metadata,
        quad_density_mode=quad_density_mode,
    )
    plan.diagnostics.update(alignment_diag)
    return plan


def equal_loop_quad_plan_v177c(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray | None = None,
    face_ids: Iterable[int] | None = None,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    context: RebuildCandidateContext,
    candidate_metadata: Mapping[str, object],
    quad_density_mode: str,
) -> QuadPlan:
    n = int(len(loop0))
    if n != int(len(loop1)) or n < 3:
        raise ValueError("Equal-loop quad plan requires equal loop sizes >= 3.")

    loop0_pts = vertices[np.asarray(loop0, dtype=np.int64), :3]
    loop1_pts = vertices[np.asarray(loop1, dtype=np.int64), :3]
    axial_segments = _axial_segment_count(
        loop0_pts=loop0_pts,
        loop1_pts=loop1_pts,
        center0=center0,
        center1=center1,
        mode=quad_density_mode,
    )

    shaping = _semantic_generated_vertex_shaping_policy(
        context=context,
        candidate_metadata=candidate_metadata,
        vertices=vertices,
        loop0=loop0,
        loop1=loop1,
        center0=center0,
        center1=center1,
        axis=axis,
        quad_density_mode=quad_density_mode,
        source_faces=source_faces,
        face_ids=face_ids,
    )
    shaping_enabled = bool(shaping.get("enabled", False))
    target_circumferential = int(shaping.get("target_circumferential_segments", n) or n)

    # v173z: when BORE owned-wall evidence provides the full cylinder frame,
    # the protected boundary loops must be ordered and phase-aligned in that
    # same frame before they are sewn to generated rings.  v173x/v173y still
    # allowed old boundary/candidate angular ordering to connect into the new
    # cylinder frame, which visually produced two axes fighting.
    v173z_boundary_loop_frame_reordered = False
    if shaping_enabled and bool(shaping.get("v173z_full_axis_authority", False)):
        try:
            sf_axis = _unit_vector(shaping.get("shape_axis_v173y", axis), fallback=axis)
            sf_center0 = np.asarray(shaping.get("shape_center0_v173x", center0), dtype=float).reshape(3)
            sf_center1 = np.asarray(shaping.get("shape_center1_v173x", center1), dtype=float).reshape(3)
            sf_u = _unit_vector(shaping.get("basis_u"), fallback=(1.0, 0.0, 0.0))
            sf_v = _unit_vector(shaping.get("basis_v"), fallback=np.cross(sf_axis, sf_u))
            reordered0 = _sort_loop_by_angle(vertices, loop0, sf_center0, sf_axis, sf_u, sf_v)
            reordered1 = _sort_loop_by_angle(vertices, loop1, sf_center1, sf_axis, sf_u, sf_v)
            if len(reordered0) == len(loop0) and len(reordered1) == len(loop1):
                aligned1 = _align_second_loop_to_first(vertices, reordered0, reordered1, sf_center0, sf_center1, sf_axis)
                loop0 = tuple(int(v) for v in reordered0)
                loop1 = tuple(int(v) for v in aligned1)
                loop0_pts = vertices[np.asarray(loop0, dtype=np.int64), :3]
                loop1_pts = vertices[np.asarray(loop1, dtype=np.int64), :3]
                v173z_boundary_loop_frame_reordered = True
                if int(target_circumferential) == int(len(loop0)):
                    angle_samples = _shape_frame_loop_angles_v173z(
                        vertices=vertices,
                        loop=loop0,
                        center=sf_center0,
                        axis=sf_axis,
                        basis_u=sf_u,
                        basis_v=sf_v,
                    )
                    shaping = dict(shaping)
                    shaping["shape_angle_samples_v173z"] = tuple(float(a) for a in angle_samples)
                    shaping["shape_phase_offset_v173z"] = float(angle_samples[0]) if angle_samples else 0.0
                else:
                    angle_samples = _shape_frame_loop_angles_v173z(
                        vertices=vertices,
                        loop=loop0,
                        center=sf_center0,
                        axis=sf_axis,
                        basis_u=sf_u,
                        basis_v=sf_v,
                    )
                    shaping = dict(shaping)
                    shaping["shape_phase_offset_v173z"] = float(angle_samples[0]) if angle_samples else 0.0
        except Exception as exc:
            shaping = dict(shaping)
            shaping["v173z_boundary_loop_frame_reorder_error"] = str(exc)

    # v177b: BORE under a chamfer can enter the v173v opening-to-exit projection
    # path without a complete v173z owned-wall frame.  In that case the old
    # projection code used a synthetic uniform 0..tau angle table for internal
    # generated rings, while the protected boundary loops still used measured,
    # sometimes non-uniform/chamfer-adjacent vertex phase.  That recreates the
    # same visual wave class previously fixed for POCKET by reusing locked loop
    # angles.  Keep the boundary vertices locked, but phase generated BORE rings
    # from the measured loop when the circumferential count is unchanged.
    if (
        shaping_enabled
        and str(shaping.get("feature_family_v162", "") or "").strip().lower() == "bore"
        and int(target_circumferential) == int(len(loop0))
        and not tuple(shaping.get("shape_angle_samples_v173z", ()) or ())
    ):
        try:
            sf_axis = _unit_vector(shaping.get("shape_axis_v173y", axis), fallback=axis)
            sf_center0 = np.asarray(shaping.get("shape_center0_v173x", center0), dtype=float).reshape(3)
            sf_u = _unit_vector(shaping.get("basis_u"), fallback=(1.0, 0.0, 0.0))
            sf_v = _unit_vector(shaping.get("basis_v"), fallback=np.cross(sf_axis, sf_u))
            angle_samples = _shape_frame_loop_angles_v173z(
                vertices=vertices,
                loop=loop0,
                center=sf_center0,
                axis=sf_axis,
                basis_u=sf_u,
                basis_v=sf_v,
            )
            if int(len(angle_samples)) == int(target_circumferential):
                shaping = dict(shaping)
                shaping["shape_angle_samples_v173z"] = tuple(float(a) for a in angle_samples)
                shaping["shape_phase_offset_v173z"] = float(angle_samples[0]) if angle_samples else 0.0
                shaping["bore_locked_loop_angle_samples_used_v177b"] = True
                shaping["bore_locked_loop_angle_guard_reason_v177b"] = (
                    "locked_bore_boundary_loop_phase_reused_for_generated_rings_to_prevent_chamfer_adjacent_wave"
                )
        except Exception as exc:
            shaping = dict(shaping)
            shaping["bore_locked_loop_angle_guard_error_v177b"] = str(exc)

    if shaping_enabled:
        # Rebuild must be allowed to change topology, not merely reconnect the old
        # strips.  Even shallow CHAMFER bands get at least one generated ring so
        # there is a real replacement surface that can be projected onto the
        # measured semantic primitive.
        axial_segments = max(int(axial_segments), int(shaping.get("min_axial_segments", 2) or 2))

    generated: list[np.ndarray] = []
    rings: list[tuple[int, ...]] = [tuple(int(v) for v in loop0)]
    v165_adapter_weights: list[float] = []
    v165_adapter_used = False
    v167_chamfer_local_rail_used = False
    for step in range(1, axial_segments):
        t = float(step) / float(axial_segments)
        linear_pts = (1.0 - t) * loop0_pts + t * loop1_pts
        if shaping_enabled:
            if bool(shaping.get("v167_chamfer_local_rail_curvature_adapter_enabled", False)):
                pts = _chamfer_local_rail_ring_points_v167(
                    t=t,
                    count=target_circumferential,
                    loop0_points=loop0_pts,
                    loop1_points=loop1_pts,
                )
                v167_chamfer_local_rail_used = True
                projected_pts = pts
            elif bool(shaping.get("pocket_linear_locked_boundary_projection_v177h", False)):
                # v177h: POCKET side-wall rebuild can be visually destroyed when
                # a coarse / contaminated annular rail exports a very wide
                # radius envelope but the generated internal rings are still
                # forced onto a nominal cylinder.  Keep the topology/refinement
                # plan, but place generated side-wall rings by measured
                # boundary-to-boundary interpolation.  The explicit floor cap is
                # still generated separately from the owned floor perimeter in
                # rebuild.py.
                projected_pts = linear_pts
            else:
                projected_pts = _semantic_projected_ring_points(
                    t=t,
                    count=target_circumferential,
                    center0=np.asarray(shaping.get("shape_center0_v173x", center0), dtype=float).reshape(3),
                    center1=np.asarray(shaping.get("shape_center1_v173x", center1), dtype=float).reshape(3),
                    axis=np.asarray(shaping.get("shape_axis_v173y", axis), dtype=float).reshape(3),
                    basis_u=np.asarray(shaping.get("basis_u"), dtype=float).reshape(3),
                    basis_v=np.asarray(shaping.get("basis_v"), dtype=float).reshape(3),
                    radius0=float(shaping.get("radius0", 0.0) or 0.0),
                    radius1=float(shaping.get("radius1", 0.0) or 0.0),
                    constant_radius=bool(shaping.get("constant_radius", False)),
                    centerline_mode=str(shaping.get("v165_centerline_mode", "interpolate_boundary_centers") or "interpolate_boundary_centers"),
                    phase_offset=float(shaping.get("shape_phase_offset_v173z", 0.0) or 0.0),
                    angle_samples=tuple(shaping.get("shape_angle_samples_v173z", ()) or ()),
                )
            if (
                bool(shaping.get("v166_endpoint_safe_bore_blend_enabled", False))
                and int(len(projected_pts)) == int(len(linear_pts))
            ):
                # v166: use old-code boundary interpolation as the baseline and
                # apply semantic cylinder projection only through a symmetric
                # center window.  This removes the remaining bottom lip caused by
                # anchoring the generated cylinder through only one endpoint.
                weight = _endpoint_safe_semantic_weight_v166(t)
                pts = (1.0 - weight) * linear_pts + weight * projected_pts
                v165_adapter_weights.append(float(weight))
                v165_adapter_used = True
            elif (
                bool(shaping.get("v165_locked_boundary_adapter_enabled", False))
                and int(len(projected_pts)) == int(len(linear_pts))
            ):
                weight = _boundary_adapter_weight_v165(
                    t,
                    float(shaping.get("v165_locked_boundary_adapter_fraction", 0.22) or 0.22),
                )
                pts = (1.0 - weight) * linear_pts + weight * projected_pts
                v165_adapter_weights.append(float(weight))
                v165_adapter_used = True
            else:
                pts = projected_pts
        else:
            pts = linear_pts
        start = len(vertices) + sum(len(item) for item in generated)
        generated.append(np.asarray(pts, dtype=float).reshape((-1, 3)))
        rings.append(tuple(range(start, start + int(len(pts)))))
    rings.append(tuple(int(v) for v in loop1))

    generated_vertices = np.vstack(generated).reshape((-1, 3)) if generated else np.zeros((0, 3), dtype=float)
    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    band_count_pairs: list[tuple[int, int]] = []
    for band in range(len(rings) - 1):
        a = tuple(int(v) for v in rings[band])
        b = tuple(int(v) for v in rings[band + 1])
        band_count_pairs.append((int(len(a)), int(len(b))))
        for quad in _band_quads_between_rings(a, b):
            logical_quads.append(tuple(int(v) for v in quad))
            triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
            triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    quality_diag = _semantic_constraint_quality_summary(
        vertices=vertices,
        loop0=loop0,
        loop1=loop1,
        center0=center0,
        center1=center1,
        axis=axis,
        radius0=float(shaping.get("radius0", 0.0) or 0.0),
        radius1=float(shaping.get("radius1", 0.0) or 0.0),
        constant_radius=bool(shaping.get("constant_radius", False)),
        feature_family=str(shaping.get("feature_family_v162", getattr(context, "entity_type", "") or "unknown")),
    ) if shaping_enabled else {
        "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        "semantic_feature_family_v162": str(getattr(context, "entity_type", "") or "unknown"),
        "semantic_constraint_quality_report_v162": False,
    }

    plan_center0_for_return = np.asarray(center0, dtype=float).reshape(3)
    plan_center1_for_return = np.asarray(center1, dtype=float).reshape(3)
    plan_axis_for_return = _unit_vector(axis)
    if shaping_enabled and bool(shaping.get("v173z_full_axis_authority", False)):
        plan_center0_for_return = np.asarray(shaping.get("shape_center0_v173x", center0), dtype=float).reshape(3)
        plan_center1_for_return = np.asarray(shaping.get("shape_center1_v173x", center1), dtype=float).reshape(3)
        plan_axis_for_return = _unit_vector(shaping.get("shape_axis_v173y", axis), fallback=axis)

    diagnostics = {
        **dict(quality_diag),
        "plan_type": "equal_loop_semantic_geometric_quad_plan" if shaping_enabled else "equal_loop_measured_quad_plan",
        "unequal_loop_transition_used": bool(any(a != b for a, b in band_count_pairs)),
        "loop0_vertex_count": int(len(loop0)),
        "loop1_vertex_count": int(len(loop1)),
        "circumferential_segments": int(target_circumferential if shaping_enabled else n),
        "original_boundary_circumferential_segments": int(n),
        "axial_segments": int(axial_segments),
        "generated_internal_ring_count": int(max(0, axial_segments - 1)),
        "semantic_geometric_rebuild_used_v160": bool(shaping_enabled),
        "semantic_geometric_rebuild_policy_v160": str(shaping.get("policy", "boundary_linear")),
        "semantic_geometric_boundary_vertices_locked_v160": True,
        "semantic_geometric_generated_vertex_count_v160": int(len(generated_vertices)),
        "semantic_geometric_min_axial_segments_v161": int(shaping.get("min_axial_segments", 0) or 0),
        "semantic_geometric_chamfer_refined_surface_v161": bool(shaping.get("chamfer_refined_surface_v161", False)),
        "constraint_bound_feature_remesh_used_v162": bool(shaping_enabled),
        "constraint_bound_feature_remesh_used": bool(shaping_enabled),
        "semantic_feature_remesh_contract_v162": str(shaping.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162)),
        "semantic_feature_remesh_contract": str(shaping.get("semantic_feature_remesh_contract_v162", SEMANTIC_FEATURE_REMESH_CONTRACT_V162)),
        "semantic_feature_family_v162": str(shaping.get("feature_family_v162", getattr(context, "entity_type", "") or "unknown")),
        "semantic_feature_family": str(shaping.get("feature_family_v162", getattr(context, "entity_type", "") or "unknown")),
        "semantic_constraint_model_v162": str(shaping.get("constraint_model_v162", "boundary_linear")),
        "semantic_constraint_model": str(shaping.get("constraint_model_v162", "boundary_linear")),
        "generated_vertices_follow_measured_constraints_v162": bool(shaping.get("generated_vertices_follow_measured_constraints_v162", False)),
        "pocket_sidewall_semantic_geometry_v162": bool(shaping.get("pocket_sidewall_semantic_geometry_v162", False)),
        "pocket_linear_locked_boundary_projection_v177h": bool(shaping.get("pocket_linear_locked_boundary_projection_v177h", False)),
        "pocket_linear_locked_boundary_projection_reason_v177h": str(shaping.get("pocket_linear_locked_boundary_projection_reason_v177h", "")),
        "pocket_linear_locked_boundary_projection_spread_v177h": float(shaping.get("pocket_linear_locked_boundary_projection_spread_v177h", 0.0) or 0.0),
        "semantic_geometric_ring_count_sequence_v160": tuple(int(len(r)) for r in rings),
        "semantic_geometric_band_count_pairs_v160": tuple((int(a), int(b)) for a, b in band_count_pairs),
        "semantic_geometric_radius0_v160": float(shaping.get("radius0", 0.0) or 0.0),
        "semantic_geometric_radius1_v160": float(shaping.get("radius1", 0.0) or 0.0),
        "semantic_geometric_constant_radius_v160": bool(shaping.get("constant_radius", False)),
        "v163_constraint_radius_authority_source": str(shaping.get("v163_constraint_radius_authority_source", "-")),
        "semantic_radius_authority_source": str(shaping.get("v163_constraint_radius_authority_source", "-")),
        "v163_candidate_radius_rejected": bool(shaping.get("v163_candidate_radius_rejected", False)),
        "candidate_radius_rejected_by_authority_guard": bool(shaping.get("v163_candidate_radius_rejected", False)),
        "v163_candidate_radius": float(shaping.get("v163_candidate_radius", 0.0) or 0.0),
        "v163_boundary_radius0": float(shaping.get("v163_boundary_radius0", 0.0) or 0.0),
        "v163_boundary_radius1": float(shaping.get("v163_boundary_radius1", 0.0) or 0.0),
        "v163_selected_opening_radius": float(shaping.get("v163_selected_opening_radius", 0.0) or 0.0),
        "v163_learned_wall_radius": float(shaping.get("v163_learned_wall_radius", 0.0) or 0.0),
        "v163_radius_conflict_reason": str(shaping.get("v163_radius_conflict_reason", "")),
        "v164_requested_circumferential_segments": int(shaping.get("v164_requested_circumferential_segments", target_circumferential) or target_circumferential),
        "v164_boundary_locked_circumferential_segments": int(shaping.get("v164_boundary_locked_circumferential_segments", n) or n),
        "v164_boundary_locked_angular_density_deferred": bool(shaping.get("v164_boundary_locked_angular_density_deferred", False)),
        "locked_boundary_angular_density_guard_used": bool(shaping.get("v164_boundary_locked_angular_density_deferred", False) or shaping.get("v168_chamfer_boundary_locked_angular_density_deferred", False)),
        "v164_boundary_locked_reason": str(shaping.get("v164_boundary_locked_reason", "")),
        "v168_chamfer_locked_boundary_density_guard_enabled": bool(shaping.get("v168_chamfer_locked_boundary_density_guard_enabled", False)),
        "v168_chamfer_boundary_locked_circumferential_segments": int(shaping.get("v168_chamfer_boundary_locked_circumferential_segments", n) or n),
        "v168_chamfer_boundary_locked_angular_density_deferred": bool(shaping.get("v168_chamfer_boundary_locked_angular_density_deferred", False)),
        "chamfer_locked_boundary_density_guard_used": bool(shaping.get("v168_chamfer_boundary_locked_angular_density_deferred", False)),
        "v168_chamfer_boundary_locked_reason": str(shaping.get("v168_chamfer_boundary_locked_reason", "")),
        "v165_locked_boundary_adapter_enabled": bool(shaping.get("v165_locked_boundary_adapter_enabled", False)),
        "v165_locked_boundary_adapter_used": bool(v165_adapter_used),
        "v165_locked_boundary_adapter_fraction": float(shaping.get("v165_locked_boundary_adapter_fraction", 0.0) or 0.0),
        "v165_boundary_adapter_weight_min": float(min(v165_adapter_weights)) if v165_adapter_weights else 0.0,
        "v165_boundary_adapter_weight_max": float(max(v165_adapter_weights)) if v165_adapter_weights else 0.0,
        "v165_boundary_adapter_weights": tuple(float(round(w, 6)) for w in v165_adapter_weights[:24]),
        "v165_boundary_radius_delta0_rel": float(shaping.get("v165_boundary_radius_delta0_rel", 0.0) or 0.0),
        "v165_boundary_radius_delta1_rel": float(shaping.get("v165_boundary_radius_delta1_rel", 0.0) or 0.0),
        "v165_lateral_center_shift": float(shaping.get("v165_lateral_center_shift", 0.0) or 0.0),
        "v165_lateral_center_shift_rel": float(shaping.get("v165_lateral_center_shift_rel", 0.0) or 0.0),
        "v165_centerline_mode": str(shaping.get("v165_centerline_mode", "")),
        "v165_boundary_adapter_policy": str(shaping.get("v165_boundary_adapter_policy", "")),
        "v166_endpoint_safe_bore_blend_enabled": bool(shaping.get("v166_endpoint_safe_bore_blend_enabled", False)),
        "v166_endpoint_safe_bore_blend_used": bool(shaping.get("v166_endpoint_safe_bore_blend_enabled", False) and v165_adapter_used),
        "endpoint_safe_bore_center_blend_used": bool(shaping.get("v166_endpoint_safe_bore_blend_enabled", False) and v165_adapter_used),
        "v166_endpoint_safe_centerline_mode": str(shaping.get("v166_endpoint_safe_centerline_mode", "")),
        "v166_endpoint_safe_blend_policy": str(shaping.get("v166_endpoint_safe_blend_policy", "")),
        "bore_wall_cylinder_shape_authority_audit_v173x": bool(shaping.get("v173x_bore_wall_cylinder_shape_authority_audit", False)),
        "bore_wall_cylinder_shape_authority_valid_v173x": bool(shaping.get("v173x_bore_wall_cylinder_shape_authority_valid", False)),
        "bore_wall_cylinder_shape_authority_used_v173x": bool(shaping.get("v173x_bore_wall_cylinder_shape_authority_used", False)),
        "bore_wall_cylinder_shape_authority_reason_v173x": str(shaping.get("v173x_bore_wall_cylinder_shape_authority_reason", "")),
        "bore_wall_shape_authority_radius_v173x": float(shaping.get("v173x_owned_wall_fitted_radius", 0.0) or 0.0),
        "bore_wall_shape_authority_median_radius_v173x": float(shaping.get("v173x_owned_wall_median_radius", 0.0) or 0.0),
        "bore_wall_shape_authority_radial_mad_v173x": float(shaping.get("v173x_owned_wall_radial_mad", 0.0) or 0.0),
        "bore_wall_shape_authority_radial_error_rel_v173x": float(shaping.get("v173x_owned_wall_radial_error_rel_median", 0.0) or 0.0),
        "bore_wall_shape_authority_normal_axis_abs_median_v173x": float(shaping.get("v173x_owned_wall_normal_axis_abs_median", 0.0) or 0.0),
        "bore_wall_shape_authority_radial_normal_alignment_median_v173x": float(shaping.get("v173x_owned_wall_radial_normal_alignment_median", 0.0) or 0.0),
        "bore_wall_shape_authority_rule_v173x": str(shaping.get("v173x_shape_authority_rule", "")),
        "bore_unified_owned_wall_shape_frame_v173y": bool(shaping.get("v173y_bore_unified_owned_wall_shape_frame", False)),
        "bore_unified_shape_frame_rule_v173y": str(shaping.get("v173y_bore_unified_frame_rule", "")),
        "bore_unified_shape_axis_v173y": tuple(float(v) for v in np.asarray(shaping.get("shape_axis_v173y", axis), dtype=float).reshape(3)),
        "bore_owned_wall_full_axis_authority_v173z": bool(shaping.get("v173z_full_axis_authority", False)),
        "bore_owned_wall_full_cylinder_frame_valid_v173z": bool(shaping.get("v173z_owned_wall_full_cylinder_frame_valid", False)),
        "bore_owned_wall_axis_pca_ratio_v173z": float(shaping.get("v173z_owned_wall_axis_pca_ratio", 0.0) or 0.0),
        "bore_owned_wall_axis_fallback_dot_v173z": float(shaping.get("v173z_owned_wall_axis_fallback_dot", 0.0) or 0.0),
        "bore_owned_wall_boundary_loop_frame_reordered_v173z": bool(v173z_boundary_loop_frame_reordered),
        "bore_owned_wall_shape_phase_offset_v173z": float(shaping.get("shape_phase_offset_v173z", 0.0) or 0.0),
        "bore_owned_wall_shape_angle_sample_count_v173z": int(len(tuple(shaping.get("shape_angle_samples_v173z", ()) or ()))),
        "bore_owned_wall_shape_authority_rule_v173z": str(shaping.get("v173z_shape_authority_rule", "")),
        "bore_locked_loop_angle_samples_used_v177b": bool(shaping.get("bore_locked_loop_angle_samples_used_v177b", False)),
        "bore_locked_loop_angle_sample_count_v177b": int(len(tuple(shaping.get("shape_angle_samples_v173z", ()) or ()))) if str(shaping.get("feature_family_v162", "") or "").strip().lower() == "bore" else 0,
        "bore_locked_loop_angle_guard_reason_v177b": str(shaping.get("bore_locked_loop_angle_guard_reason_v177b", "")),
        "bore_locked_loop_angle_guard_error_v177b": str(shaping.get("bore_locked_loop_angle_guard_error_v177b", "")),
        "bore_v166_endpoint_safe_blend_disabled_for_unified_frame_v173y": bool(shaping.get("v173y_bore_unified_owned_wall_shape_frame", False) and not bool(shaping.get("v166_endpoint_safe_bore_blend_enabled", False))),
        "bore_v165_boundary_adapter_disabled_for_unified_frame_v173y": bool(shaping.get("v173y_bore_unified_owned_wall_shape_frame", False) and not bool(shaping.get("v165_locked_boundary_adapter_enabled", False))),
        "bore_opening_to_exit_rebuild_projection_v173v": bool(shaping.get("v173v_bore_opening_to_exit_rebuild_projection_guard", False)),
        "bore_v166_endpoint_safe_blend_disabled_for_opening_to_exit_wall_v173v": bool(shaping.get("v173v_bore_v166_endpoint_safe_blend_disabled", False)),
        "bore_v165_boundary_adapter_disabled_for_opening_to_exit_wall_v173v": bool(shaping.get("v173v_bore_v165_boundary_adapter_disabled", False)),
        "bore_internal_rings_follow_measured_cylinder_v173v": bool(shaping.get("v173v_bore_internal_rings_follow_measured_cylinder", False)),
        "bore_boundary_loops_locked_as_seams_only_v173v": bool(shaping.get("v173v_bore_boundary_loops_locked_as_seams_only", False)),
        "bore_opening_to_exit_rebuild_projection_rule_v173v": str(shaping.get("v173v_bore_opening_to_exit_rebuild_projection_rule", "")),
        "v167_chamfer_local_rail_curvature_adapter_enabled": bool(shaping.get("v167_chamfer_local_rail_curvature_adapter_enabled", False)),
        "v167_chamfer_local_rail_curvature_adapter_used": bool(v167_chamfer_local_rail_used),
        "chamfer_local_rail_curvature_adapter_used": bool(v167_chamfer_local_rail_used),
        "v167_chamfer_local_rail_curvature_adapter_policy": str(shaping.get("v167_chamfer_local_rail_curvature_adapter_policy", "")),
        "v167_chamfer_global_conical_projection_disabled": bool(shaping.get("v167_chamfer_global_conical_projection_disabled", False)),
        "transition_drop_quad_count": int(sum(abs(a - b) // 2 for a, b in band_count_pairs)),
        "transition_ring_vertex_count": int(target_circumferential if shaping_enabled and target_circumferential != n else 0),
        "geometry_source": "semantic_measured_primitive_projected_generated_rings_v160" if shaping_enabled else "measured_boundary_loop_vertices",
        "interpolation_rule": "generated_rings_projected_to_semantic_cylinder_or_cone_boundary_locked" if shaping_enabled else "linear_between_corresponding_measured_loop_vertices",
        "parameter_fit_used": False,
        "candidate_measured_primitive_used_for_vertex_placement": bool(shaping_enabled),
        "radius_used_for_vertex_placement": bool(shaping_enabled),
        "axis_used_for_vertex_placement": bool(shaping_enabled),
        "existing_boundary_vertices_moved": 0,
    }
    return QuadPlan(
        generated_vertices=np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        triangles=np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        logical_quads=tuple(logical_quads),
        loop0=tuple(int(v) for v in loop0),
        loop1=tuple(int(v) for v in loop1),
        center0=np.asarray(plan_center0_for_return, dtype=float).reshape(3),
        center1=np.asarray(plan_center1_for_return, dtype=float).reshape(3),
        axis=_unit_vector(plan_axis_for_return),
        diagnostics=diagnostics,
    )


def unequal_loop_quad_plan_v177c(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray | None = None,
    face_ids: Iterable[int] | None = None,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    context: RebuildCandidateContext,
    candidate_metadata: Mapping[str, object],
    quad_density_mode: str,
) -> QuadPlan:
    """Build a smooth measured quad surface between unequal boundary loops.

    Large count deltas, such as 126 ↔ 52, should not be collapsed in one band.
    A single large transition is topologically valid but visually poor: it
    creates a dense drop-sector fan over a short axial distance.  This planner
    distributes the count reduction over several generated rings, then orients
    triangles against the original patch normals before validation.
    """

    n0 = int(len(loop0))
    n1 = int(len(loop1))
    if not loop_pair_transition_allowed_v177c(n0, n1, allow_unequal_loop_transition=True):
        raise ValueError(f"Unequal-loop quad plan rejected by topology rule. loop sizes: {n0}, {n1}.")

    loop0_is_big = n0 > n1
    big_loop = tuple(int(v) for v in (loop0 if loop0_is_big else loop1))
    small_loop = tuple(int(v) for v in (loop1 if loop0_is_big else loop0))
    big_center = np.asarray(center0 if loop0_is_big else center1, dtype=float).reshape(3)
    small_center = np.asarray(center1 if loop0_is_big else center0, dtype=float).reshape(3)

    n_big = int(len(big_loop))
    n_small = int(len(small_loop))

    # Measured loop points are needed both for ring generation and for the
    # density planner.  Keep them next to the loop selection so the transition
    # plan reads in data-flow order.
    big_pts = vertices[np.asarray(big_loop, dtype=np.int64), :3]
    small_pts = vertices[np.asarray(small_loop, dtype=np.int64), :3]

    base_count_sequence = _transition_count_sequence(n_big, n_small, mode=quad_density_mode)
    target_band_info = _target_unequal_transition_band_count(
        big_points=big_pts,
        small_points=small_pts,
        big_center=big_center,
        small_center=small_center,
        mode=quad_density_mode,
        base_band_count=max(1, len(base_count_sequence) - 1),
    )
    count_sequence = _densify_transition_count_sequence(
        base_count_sequence,
        target_band_count=int(target_band_info.get("target_band_count", max(1, len(base_count_sequence) - 1))),
    )
    if count_sequence[0] != n_big or count_sequence[-1] != n_small:
        raise ValueError(f"Invalid unequal-loop transition count sequence: {count_sequence!r}")

    band_count = max(1, len(count_sequence) - 1)

    generated_blocks: list[np.ndarray] = []
    ring_ids: list[tuple[int, ...]] = [tuple(int(v) for v in big_loop)]
    next_generated_id = int(len(vertices))

    # Intermediate generated rings gradually move from the larger boundary loop
    # toward the smaller boundary loop while also gradually reducing vertex count.
    for ring_index, count in enumerate(count_sequence[1:-1], start=1):
        t = float(ring_index) / float(band_count)
        big_sample = _sample_closed_loop_points(big_pts, int(count))
        small_sample = _sample_closed_loop_points(small_pts, int(count))
        pts = (1.0 - t) * big_sample + t * small_sample
        generated_blocks.append(np.asarray(pts, dtype=float).reshape((int(count), 3)))
        ids = tuple(range(next_generated_id, next_generated_id + int(count)))
        next_generated_id += int(count)
        ring_ids.append(ids)

    ring_ids.append(tuple(int(v) for v in small_loop))
    generated_vertices = np.vstack(generated_blocks).reshape((-1, 3)) if generated_blocks else np.zeros((0, 3), dtype=float)

    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    band_drop_counts: list[int] = []
    band_count_pairs: list[tuple[int, int]] = []

    for band_index in range(len(ring_ids) - 1):
        ring_a = tuple(int(v) for v in ring_ids[band_index])
        ring_b = tuple(int(v) for v in ring_ids[band_index + 1])
        band_count_pairs.append((int(len(ring_a)), int(len(ring_b))))
        band_quads = _band_quads_between_rings(ring_a, ring_b)
        band_drop_counts.append(int(abs(len(ring_a) - len(ring_b)) // 2))
        logical_quads.extend(band_quads)
        for quad in band_quads:
            triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
            triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    output_loop0 = tuple(int(v) for v in loop0)
    output_loop1 = tuple(int(v) for v in loop1)
    diagnostics = {
        "plan_type": "gradual_unequal_loop_measured_quad_plan",
        "unequal_loop_transition_used": True,
        "unequal_loop_transition_policy": "gradual_topological_drop_sector_bands",
        "transition_base_count_sequence": tuple(int(v) for v in base_count_sequence),
        "transition_count_sequence": tuple(int(v) for v in count_sequence),
        "transition_band_count": int(len(count_sequence) - 1),
        "transition_base_band_count": int(max(1, len(base_count_sequence) - 1)),
        "transition_target_band_count": int(target_band_info.get("target_band_count", max(1, len(count_sequence) - 1))),
        "transition_band_count_pairs": tuple(band_count_pairs),
        "transition_band_drop_counts": tuple(int(v) for v in band_drop_counts),
        "transition_drop_quad_count": int(sum(band_drop_counts)),
        "transition_equal_count_spacer_band_count": int(sum(1 for value in band_drop_counts if int(value) == 0)),
        "transition_ring_vertex_count": int(sum(count_sequence[1:-1])),
        "transition_source_loop_vertex_counts": (int(n0), int(n1)),
        "transition_big_loop_first": bool(loop0_is_big),
        "quad_density_mode": str(quad_density_mode),
        "density_policy": "full_equal_edge_even_drop_distribution",
        "target_axial_edge_length": float(target_band_info.get("target_axial_edge_length", 0.0)),
        "raw_equal_edge_axial_segments": int(target_band_info.get("raw_equal_edge_axial_segments", 0) or 0),
        "axial_segment_cap": int(target_band_info.get("axial_segment_cap", 0) or 0),
        "circumferential_segments": int(max(n0, n1)),
        "axial_segments": int(max(1, len(count_sequence) - 1)),
        "generated_internal_ring_count": int(max(0, len(count_sequence) - 2)),
        "geometry_source": "measured_boundary_loop_vertices_with_gradual_boundary_locked_unequal_transition",
        "interpolation_rule": "multi_band_count_reduction_between_measured_boundary_loops",
        "parameter_fit_used": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
    }
    return QuadPlan(
        generated_vertices=np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        triangles=np.asarray(triangles, dtype=np.int64).reshape((-1, 3)) if triangles else np.zeros((0, 3), dtype=np.int64),
        logical_quads=tuple(logical_quads),
        loop0=output_loop0,
        loop1=output_loop1,
        center0=np.asarray(center0, dtype=float).reshape(3),
        center1=np.asarray(center1, dtype=float).reshape(3),
        axis=_unit_vector(axis),
        diagnostics=diagnostics,
    )
