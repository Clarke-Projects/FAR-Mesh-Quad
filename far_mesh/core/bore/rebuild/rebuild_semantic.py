"""Semantic vertex-placement helpers for BoreTool rebuild.

This module is intentionally non-mutating. It may read accepted CandidateData
metadata, choose measured semantic radius authority, report constraint quality,
and generate projected ring point arrays for patch plans. It must not authorize
deletion, apply generated geometry to a mesh, validate final topology, mutate
project state, or emit RebuildResult. ``rebuild.py`` remains the public rebuild
orchestrator and mutation authority.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math

import numpy as np

from .rebuild_density import (
    QUAD_DENSITY_MODE_FULL,
    QUAD_DENSITY_MODE_PI,
    normalize_quad_density_mode_v176q,
)
from .rebuild_geometry import (
    loop_angle_basis,
    loop_median_radius,
    loop_radius_spread_ratio,
    orthonormal_basis,
    safe_float,
    sample_closed_loop_points,
    unit_vector,
)
from .rebuild_bore import (
    owned_bore_wall_cylinder_shape_authority_v173x_v176y,
    owned_bore_wall_cylinder_shape_authority_v173z_v176y,
    shape_frame_loop_angles_v173z_v176y,
)
from ..types import FeatureFamily

SEMANTIC_FEATURE_REMESH_CONTRACT_V162 = (
    "rebuild_is_constraint_bound_semantic_feature_remeshing_for_every_feature_family"
)

REBUILD_SEMANTIC_HELPER_EXTRACTION_CHECKPOINT_V176Z = (
    "v176z_rebuild_semantic_radius_and_ring_projection_helpers_extracted_no_behavior_change"
)

REBUILD_SEMANTIC_HELPER_NON_MUTATION_CONTRACT_V176Z = (
    "semantic_helpers_may_choose_measured_radius_and_project_ring_points_only_no_delete_authorization_no_trial_no_mutation"
)

REBUILD_SEMANTIC_SHAPING_POLICY_EXTRACTION_CHECKPOINT_V177A = (
    "v177a_rebuild_semantic_generated_vertex_shaping_policy_extracted_no_behavior_change"
)

REBUILD_SEMANTIC_SHAPING_POLICY_NON_MUTATION_CONTRACT_V177A = (
    "semantic_shaping_policy_may_return_generated_vertex_placement_policy_only_no_delete_authorization_no_trial_no_mutation"
)


def loop_pair_transition_allowed_v176z(
    n0: int,
    n1: int,
    *,
    allow_unequal_loop_transition: bool,
) -> bool:
    """Return whether unequal ring counts can be sewn without boundary mutation."""

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

def candidate_rebuild_radius_v176z(candidate_metadata: Mapping[str, object]) -> float:
    for key in (
        "radius",
        "radius_nominal",
        "primitive_radius",
        "feature_radius",
        "selected_opening_radius",
        "opening_radius",
    ):
        if key in candidate_metadata:
            value = safe_float(candidate_metadata.get(key), 0.0)
            if value > 1.0e-9 and np.isfinite(value):
                return float(value)
    primitives = candidate_metadata.get("feature_primitives")
    if isinstance(primitives, (list, tuple)):
        for primitive in primitives:
            if isinstance(primitive, Mapping):
                value = safe_float(primitive.get("radius", primitive.get("radius_nominal", 0.0)), 0.0)
                if value > 1.0e-9 and np.isfinite(value):
                    return float(value)
    return 0.0


def metadata_first_float_v163_v176z(candidate_metadata: Mapping[str, object], keys: tuple[str, ...]) -> float:
    """Find a measured numeric value in CandidateData/diagnostics.

    This is not feature recognition.  It is rebuild-side constraint authority
    plumbing: Recognition already measured these values; Rebuild must avoid
    grabbing the broad display radius when a more specific owned-wall/rim radius
    was provided.
    """

    def _scan(value: object, depth: int = 0) -> float:
        if depth > 4:
            return 0.0
        if isinstance(value, Mapping):
            for key in keys:
                if key in value:
                    v = safe_float(value.get(key), 0.0)
                    if v > 1.0e-9 and np.isfinite(v):
                        return float(v)
            # Prioritize semantically named children before generic recursion.
            for child_key in ("diagnostics", "selected_opening_frame_resolver", "selected_opening_measurement_audit", "opening_measurement_audit", "two_opening_bore_frame"):
                if child_key in value:
                    v = _scan(value.get(child_key), depth + 1)
                    if v > 1.0e-9:
                        return v
            primitives = value.get("feature_primitives")
            if isinstance(primitives, (list, tuple)):
                for primitive in primitives:
                    v = _scan(primitive, depth + 1)
                    if v > 1.0e-9:
                        return v
        return 0.0

    return float(_scan(candidate_metadata, 0))


def metadata_indicates_bore_opening_to_exit_wall_v173v_v176z(candidate_metadata: Mapping[str, object]) -> bool:
    """Return True when Recognition exported v173u BORE wall ownership.

    This is rebuild-side contract plumbing, not feature recognition.  The
    meaning has already been created by Recognition: selected opening rail +
    opposite/exit rail + measured axis + contained cylindrical wall interval.
    Rebuild uses this flag only to choose vertex placement for the accepted
    BORE_WALL CandidateData.
    """

    keys = {
        "bore_opening_to_exit_wall_ownership_valid_v173u",
        "selected_opening_primary_component_weak_overridden_v173u",
        "bore_opening_to_exit_wall_subset_mapping_v173u",
        "bore_opening_to_exit_wall_subset_used_v173u",
    }
    text_keys = {
        "active_candidate_authority",
        "candidate_authority",
        "recognition_rule",
        "feature_ownership_source",
        "bore_opening_to_exit_wall_subset_source_v173u",
    }

    def _truthy(value: object) -> bool:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            try:
                return bool(float(value) != 0.0 and math.isfinite(float(value)))
            except Exception:
                return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "valid", "used"}
        return bool(value)

    def _scan(value: object, depth: int = 0) -> bool:
        if depth > 5:
            return False
        if isinstance(value, Mapping):
            for key in keys:
                if key in value and _truthy(value.get(key)):
                    return True
            for key in text_keys:
                if key in value:
                    text = str(value.get(key, "") or "").strip().lower()
                    if "opening_to_exit" in text or "v108_bore_opening_to_exit" in text:
                        return True
            # Prefer known nested semantic containers before generic recursion.
            for child_key in ("diagnostics", "feature_primitives", "two_opening_bore_frame", "owned_bore_frame"):
                if child_key in value and _scan(value.get(child_key), depth + 1):
                    return True
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)) and _scan(child, depth + 1):
                    return True
        elif isinstance(value, (list, tuple)):
            for child in value:
                if _scan(child, depth + 1):
                    return True
        return False

    return bool(_scan(dict(candidate_metadata or {}), 0))


def semantic_radius_authority_v163_v176z(
    *,
    candidate_metadata: Mapping[str, object],
    feature_family: str,
    candidate_radius: float,
    boundary_radius0: float,
    boundary_radius1: float,
) -> dict[str, object]:
    """Choose vertex-placement radius from semantic evidence agreement.

    v163 guardrail: a rebuild constraint is authoritative only if it is
    consistent with the owned boundary/wall evidence.  In the reported failure,
    the selected rim was clean (~15.16), the owned wall radius was ~15.60, but
    CandidateData.radius was the broad two-opening frame average (~17.36).  v162
    promoted that broad value into vertex placement and deformed a good bore.
    """

    family = str(feature_family or "").strip().lower()
    r0 = float(boundary_radius0) if np.isfinite(float(boundary_radius0)) else 0.0
    r1 = float(boundary_radius1) if np.isfinite(float(boundary_radius1)) else 0.0
    cr = float(candidate_radius) if np.isfinite(float(candidate_radius)) else 0.0
    valid_boundary = tuple(v for v in (r0, r1) if v > 1.0e-9)
    boundary_median = float(np.median(valid_boundary)) if valid_boundary else 0.0

    selected = metadata_first_float_v163_v176z(candidate_metadata, (
        "selected_opening_radius_authority_v150",
        "selected_opening_radius",
        "raw_component_best_radius",
        "raw_best_radius",
        "opening_radius_hint",
        "opening_radius",
    ))
    learned = metadata_first_float_v163_v176z(candidate_metadata, (
        "learned_wall_radius",
        "owned_wall_radius",
        "wall_radius",
    ))

    def _rel_delta(a: float, b: float) -> float:
        if a <= 1.0e-9 or b <= 1.0e-9:
            return 999999.0
        return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0e-9)

    source = "boundary_loop_median"
    reason = "candidate_radius_not_authoritative"
    chosen = boundary_median if boundary_median > 1.0e-9 else cr
    candidate_rejected = False

    if family == "bore":
        # The owned wall/rim evidence has higher placement authority than the
        # two-opening descriptor.  Prefer learned wall radius when it agrees with
        # the selected rim; otherwise prefer the clean selected rim.  Candidate
        # radius can remain authoritative only when it agrees with those role
        # measurements.
        role_values = []
        if selected > 1.0e-9:
            role_values.append(selected)
        if learned > 1.0e-9:
            role_values.append(learned)
        if boundary_median > 1.0e-9:
            role_values.append(boundary_median)
        role_radius = float(np.median(role_values)) if role_values else (boundary_median if boundary_median > 1.0e-9 else cr)
        if cr > 1.0e-9 and role_radius > 1.0e-9 and _rel_delta(cr, role_radius) <= 0.08:
            chosen = cr
            source = "candidate_radius_agrees_with_owned_wall_and_rim_v163"
            reason = "candidate_radius_semantically_consistent"
        else:
            chosen = role_radius
            candidate_rejected = bool(cr > 1.0e-9 and role_radius > 1.0e-9)
            if learned > 1.0e-9 and selected > 1.0e-9 and _rel_delta(learned, selected) <= 0.12:
                source = "learned_wall_plus_selected_rim_radius_v163"
                chosen = float(np.median([learned, selected]))
                reason = "broad_two_opening_radius_conflicts_with_owned_wall_radius"
            elif selected > 1.0e-9:
                source = "selected_opening_radius_v163"
                chosen = selected
                reason = "selected_rim_is_cleaner_than_broad_two_opening_radius"
            elif learned > 1.0e-9:
                source = "learned_wall_radius_v163"
                chosen = learned
                reason = "owned_wall_radius_is_more_specific_than_candidate_radius"
            elif boundary_median > 1.0e-9:
                source = "boundary_loop_median_v163"
                chosen = boundary_median
                reason = "falling_back_to_proven_boundary_loops"
    elif family == "pocket":
        if selected > 1.0e-9:
            chosen = selected
            source = "pocket_selected_rim_radius_v163"
            reason = "pocket_recess_sidewall_uses_selected_rim_radius"
        elif cr > 1.0e-9:
            chosen = cr
            source = "pocket_candidate_radius_v163"
            reason = "pocket_candidate_radius_authority"
    else:
        # Chamfers intentionally interpolate between parent boundaries unless a
        # conical primitive is explicitly measured elsewhere.
        if r0 > 1.0e-9 and r1 > 1.0e-9:
            return {
                "radius0": float(r0),
                "radius1": float(r1),
                "constant_radius": False,
                "source": "chamfer_boundary_radii_v163",
                "candidate_radius_rejected": False,
                "candidate_radius": float(cr),
                "selected_opening_radius": float(selected),
                "learned_wall_radius": float(learned),
                "reason": "chamfer_preserves_measured_parent_boundary_radii",
            }

    if chosen <= 1.0e-9:
        chosen = boundary_median if boundary_median > 1.0e-9 else cr
    return {
        "radius0": float(chosen),
        "radius1": float(chosen),
        "constant_radius": bool(chosen > 1.0e-9 and family in {"bore", "pocket"}),
        "source": str(source),
        "candidate_radius_rejected": bool(candidate_rejected),
        "candidate_radius": float(cr),
        "selected_opening_radius": float(selected),
        "learned_wall_radius": float(learned),
        "reason": str(reason),
    }


def semantic_constraint_quality_summary_v176z(
    *,
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    radius0: float,
    radius1: float,
    constant_radius: bool,
    feature_family: str,
) -> dict[str, object]:
    """Small before/target quality report for constraint-bound remeshing.

    The report does not authorize the rebuild.  It makes the rebuild intent
    visible: what measured constraint was used, and whether the generated
    surface is expected to reduce coarse boundary wobble in the replacement
    interior.
    """

    spread0 = loop_radius_spread_ratio(vertices=vertices, loop=loop0, center=center0, axis=axis)
    spread1 = loop_radius_spread_ratio(vertices=vertices, loop=loop1, center=center1, axis=axis)
    before = max(float(spread0), float(spread1))
    target = 0.0 if bool(constant_radius) else abs(float(radius1) - float(radius0)) / max(float(radius0), float(radius1), 1.0e-12)
    return {
        "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        "semantic_feature_family_v162": str(feature_family or "unknown"),
        "semantic_constraint_quality_report_v162": True,
        "semantic_constraint_boundary_radius_spread_before_v162": float(before),
        "semantic_constraint_target_radius_spread_v162": float(target),
        "semantic_constraint_expected_radial_improvement_v162": bool(before > target + 1.0e-9),
    }


def semantic_target_circumferential_count_v176z(n: int, *, mode: str, entity: str) -> int:
    n = max(3, int(n))
    mode_norm = normalize_quad_density_mode_v176q(mode)
    if entity == "chamfer":
        # Chamfers are short but quality-sensitive; double angular samples and
        # use multiple axial rings so the bevel surface can actually express the
        # measured angle instead of a single coarse transition band.
        desired = n * 2
    elif entity in {"pocket", "circular_pocket"}:
        # Pocket side-walls share the cylindrical wall intent with bores, but
        # the floor grid and protected child openings already add topology.
        desired = n * (3 if mode_norm == QUAD_DENSITY_MODE_FULL else 2)
    elif mode_norm == QUAD_DENSITY_MODE_FULL:
        desired = n * 3
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        desired = n * 2
    else:
        desired = n * 2
    cap = 128 if mode_norm == QUAD_DENSITY_MODE_FULL else 96
    desired = min(int(cap), max(n, int(desired)))
    # _band_quads_between_rings can transition only even count deltas.  Keep the
    # original boundary loop untouched and choose the nearest useful count that
    # can be sewn back into it.
    if (desired - n) % 2:
        desired += 1
    while desired > n and not loop_pair_transition_allowed_v176z(n, desired, allow_unequal_loop_transition=True):
        desired -= 2
    return max(n, int(desired))


def semantic_projected_ring_points_v176z(
    *,
    t: float,
    count: int,
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    radius0: float,
    radius1: float,
    constant_radius: bool,
    centerline_mode: str = "interpolate_boundary_centers",
    phase_offset: float = 0.0,
    angle_samples: Iterable[float] | None = None,
) -> np.ndarray:
    t = max(0.0, min(1.0, float(t)))
    count = max(3, int(count))
    c0 = np.asarray(center0, dtype=float).reshape(3)
    c1 = np.asarray(center1, dtype=float).reshape(3)
    axis_vec = unit_vector(axis, fallback=c1 - c0)

    # v165: for clean BORE remeshing, boundary loops can contain old chamfer/lip
    # offsets.  If generated rings interpolate those boundary centers directly,
    # the new cylinder can drift through the body and leave an asymmetric lip at
    # the endpoint.  The semantic centerline should be anchored to the lower
    # boundary center plus the measured axis; locked boundary mismatch is handled
    # by a local boundary adapter blend, not by bending the whole cylinder.
    centerline_mode_norm = str(centerline_mode or "").strip().lower()
    if centerline_mode_norm == "axis_anchored_from_loop0_v165" or "axis_anchored_from_loop0_v173v" in centerline_mode_norm:
        axial_span = float(np.dot(c1 - c0, axis_vec))
        center = c0 + (t * axial_span) * axis_vec
    elif "unified_owned_wall_axis_centerline_v173y" in centerline_mode_norm:
        # c0/c1 already lie on the owned-wall centerline. Interpolating them
        # keeps generated rings in the same axis/center/radius frame as the
        # owned BORE_WALL shape authority.
        center = (1.0 - t) * c0 + t * c1
    else:
        center = (1.0 - t) * c0 + t * c1

    radius = float(radius0 if constant_radius else ((1.0 - t) * float(radius0) + t * float(radius1)))
    radius = max(radius, 1.0e-12)
    angle_tuple = tuple(float(v) for v in tuple(angle_samples or ()))
    if len(angle_tuple) == count:
        angles = np.asarray(angle_tuple, dtype=float).reshape(count)
    else:
        angles = float(phase_offset) + (np.arange(count, dtype=float) / float(count)) * (2.0 * math.pi)
    u = unit_vector(basis_u)
    v = unit_vector(basis_v)
    pts = center.reshape(1, 3) + radius * (np.cos(angles).reshape(-1, 1) * u.reshape(1, 3) + np.sin(angles).reshape(-1, 1) * v.reshape(1, 3))
    return np.asarray(pts, dtype=float).reshape((count, 3))


def chamfer_local_rail_ring_points_v167_v176z(
    *,
    t: float,
    count: int,
    loop0_points: np.ndarray,
    loop1_points: np.ndarray,
) -> np.ndarray:
    """Generate a chamfer ring from local boundary rails instead of a global cone.

    Chamfer rebuilds run between two parent-surface boundary loops.  On parts
    with curvature in more than one direction, a single global conical/circular
    projection can make some upper-band segments float over the following band.
    The safe semantic constraint for a locked-boundary chamfer is the local rail
    surface spanned by the two measured/protected boundary loops: resample both
    loops to the requested angular count, then interpolate along each local rail.
    That preserves both parent-surface curvatures while still allowing generated
    topology inside the owned chamfer patch.
    """
    t = max(0.0, min(1.0, float(t)))
    count = max(3, int(count))
    pts0 = sample_closed_loop_points(np.asarray(loop0_points, dtype=float).reshape((-1, 3)), count)
    pts1 = sample_closed_loop_points(np.asarray(loop1_points, dtype=float).reshape((-1, 3)), count)
    return np.asarray((1.0 - t) * pts0 + t * pts1, dtype=float).reshape((count, 3))

# v177a extraction aliases: keep the moved policy body byte-for-byte close to
# its rebuild.py source while routing dependencies through semantic/bore/geometry
# helper modules.  These aliases are non-mutating and local to this module.
_candidate_rebuild_radius = candidate_rebuild_radius_v176z
_metadata_indicates_bore_opening_to_exit_wall_v173v = metadata_indicates_bore_opening_to_exit_wall_v173v_v176z
_semantic_radius_authority_v163 = semantic_radius_authority_v163_v176z
_semantic_target_circumferential_count = semantic_target_circumferential_count_v176z
_loop_pair_transition_allowed = loop_pair_transition_allowed_v176z
_unit_vector = unit_vector
_safe_float = safe_float
_loop_angle_basis = loop_angle_basis
_loop_median_radius = loop_median_radius
_orthonormal_basis = orthonormal_basis
_owned_bore_wall_cylinder_shape_authority_v173x = owned_bore_wall_cylinder_shape_authority_v173x_v176y
_owned_bore_wall_cylinder_shape_authority_v173z = owned_bore_wall_cylinder_shape_authority_v173z_v176y
_shape_frame_loop_angles_v173z = shape_frame_loop_angles_v173z_v176y

def semantic_generated_vertex_shaping_policy_v177a(
    *,
    context: RebuildCandidateContext,
    candidate_metadata: Mapping[str, object],
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    quad_density_mode: str,
    source_faces: np.ndarray | None = None,
    face_ids: Iterable[int] | None = None,
) -> dict[str, object]:
    """Return generated-vertex placement policy for semantic feature remeshing.

    v162 contract: Rebuild applies the same intent to every supported feature
    family.  The policy is not feature recognition and not a delete-target
    expansion.  It runs only after CandidateData has owned a bounded patch and
    Rebuild has proven the boundary loops.  Its job is to generate replacement
    topology whose generated vertices express the measured semantic primitive:
    BORE -> cylindrical wall, POCKET -> cylindrical recess side-wall plus
    separate floor grid, CHAMFER -> conical/bevel transition band.
    """

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    gate = str(getattr(context, "rebuild_gate", "") or "").strip().lower()

    # v175l: family identity for rebuild shape authority must not be inferred by
    # substring matches after a stronger CandidateData family/role declaration
    # already exists.  POCKET candidates can legitimately contain words such as
    # "child_bore_openings" in their relationship/protection metadata; that
    # must not make the pocket side-wall generator borrow BORE cylinder authority.
    explicit_pocket_family_v175l = bool(entity in {"pocket", "circular_pocket"} or family in {FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value})
    explicit_chamfer_family_v175l = bool(entity == "chamfer" or family == FeatureFamily.CHAMFER_FORM.value)
    explicit_bore_family_v175l = bool(entity in {"borehole", "bore", "core_bore_cylinder_candidate"} or family == FeatureFamily.BORE.value)

    is_pocket = bool(explicit_pocket_family_v175l or (not explicit_bore_family_v175l and not explicit_chamfer_family_v175l and "pocket" in gate))
    is_chamfer = bool(explicit_chamfer_family_v175l or (not explicit_pocket_family_v175l and not explicit_bore_family_v175l and "chamfer" in gate))
    is_bore = bool(explicit_bore_family_v175l or (not explicit_pocket_family_v175l and not explicit_chamfer_family_v175l and "bore" in gate))
    if is_pocket:
        # POCKET owns a recess side-wall/floor rebuild.  Child BORE text in the
        # gate is relationship metadata and must not activate BORE shape authority.
        is_bore = False
        is_chamfer = False
    if not (is_bore or is_chamfer or is_pocket):
        return {
            "enabled": False,
            "policy": "unsupported_feature_family_keeps_existing_boundary_linear_plan",
            "feature_family_v162": family or entity or "unknown",
            "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        }

    n = int(len(loop0))
    if n < 4 or int(len(loop1)) != n:
        return {
            "enabled": False,
            "policy": "requires_equal_boundary_loop_pair",
            "feature_family_v162": family or entity or "unknown",
            "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        }

    axis_vec = _unit_vector(axis, fallback=np.asarray(center1, dtype=float).reshape(3) - np.asarray(center0, dtype=float).reshape(3))
    basis_u, basis_v = _loop_angle_basis(vertices=vertices, loop=loop0, center=center0, axis=axis_vec)
    r0_boundary = _loop_median_radius(vertices=vertices, loop=loop0, center=center0, axis=axis_vec)
    r1_boundary = _loop_median_radius(vertices=vertices, loop=loop1, center=center1, axis=axis_vec)
    candidate_radius = _candidate_rebuild_radius(candidate_metadata)

    # v163: Constraint authority must come from the owned wall/rim evidence, not
    # from a broad two-opening frame descriptor when the descriptors disagree.
    # The v162 implementation let CandidateData.radius override both boundary
    # loops.  On clean bores with a noisy opposite/opening-ring measurement this
    # inflated the generated cylinder and turned an already straight bore into a
    # deformed one.  Rebuild may improve topology, but it must not let a
    # contradictory descriptor become vertex-placement authority.
    radius_authority = _semantic_radius_authority_v163(
        candidate_metadata=candidate_metadata,
        feature_family="bore" if is_bore else ("pocket" if is_pocket else ("chamfer" if is_chamfer else family or entity or "unknown")),
        candidate_radius=float(candidate_radius),
        boundary_radius0=float(r0_boundary),
        boundary_radius1=float(r1_boundary),
    )
    r0 = float(radius_authority.get("radius0", r0_boundary) or r0_boundary)
    r1 = float(radius_authority.get("radius1", r1_boundary) or r1_boundary)
    constant_radius = bool(radius_authority.get("constant_radius", False))

    # v173x: for accepted BORE_WALL candidates, rebuild shape authority must be
    # traced to owned interior wall evidence.  Selected rims, opposite/endpoint
    # bands, and terminal-continuation evidence are boundary/depth/context clues;
    # they must not silently become cylinder-shape authority when the owned wall
    # faces themselves define a cleaner cylinder.
    owned_wall_shape_authority_v173x: dict[str, object] = {}
    shape_center0_v173x = np.asarray(center0, dtype=float).reshape(3)
    shape_center1_v173x = np.asarray(center1, dtype=float).reshape(3)
    shape_axis_v173y = np.asarray(axis_vec, dtype=float).reshape(3)
    unified_owned_wall_shape_frame_v173y = False
    if is_bore:
        # v173z first tries to derive the *complete* rebuild shape frame from
        # owned BORE_WALL faces, including the axis.  v173x is retained only as
        # a fallback diagnostic path because it solved the radius/center under
        # the incoming axis and could therefore leave two axes fighting.
        owned_wall_shape_authority_v173x = _owned_bore_wall_cylinder_shape_authority_v173z(
            vertices=vertices,
            source_faces=source_faces,
            face_ids=face_ids,
            fallback_axis=axis_vec,
            center0=center0,
            center1=center1,
        )
        if not bool(owned_wall_shape_authority_v173x.get("valid", False)):
            fallback_v173x = _owned_bore_wall_cylinder_shape_authority_v173x(
                vertices=vertices,
                source_faces=source_faces,
                face_ids=face_ids,
                axis=axis_vec,
                center0=center0,
                center1=center1,
            )
            fallback_v173x = dict(fallback_v173x)
            fallback_v173x["v173z_full_axis_authority"] = False
            fallback_v173x["v173z_primary_rejection_reason"] = str(owned_wall_shape_authority_v173x.get("reason", ""))
            owned_wall_shape_authority_v173x = fallback_v173x
        if bool(owned_wall_shape_authority_v173x.get("valid", False)):
            fitted_radius_v173x = _safe_float(owned_wall_shape_authority_v173x.get("fitted_radius", 0.0), 0.0)
            if fitted_radius_v173x > 1.0e-9:
                r0 = float(fitted_radius_v173x)
                r1 = float(fitted_radius_v173x)
                constant_radius = True
                radius_authority = dict(radius_authority)
                radius_authority["source"] = "owned_bore_wall_cylinder_shape_authority_v173x"
                radius_authority["reason"] = "owned_BORE_WALL_face_evidence_defines_rebuild_shape_authority"
                radius_authority["candidate_radius_rejected"] = bool(
                    _safe_float(radius_authority.get("candidate_radius", 0.0), 0.0) > 1.0e-9
                    and abs(_safe_float(radius_authority.get("candidate_radius", 0.0), 0.0) - fitted_radius_v173x) / max(abs(fitted_radius_v173x), 1.0e-9) > 0.04
                )
                shape_center0_candidate = owned_wall_shape_authority_v173x.get("shape_center0")
                shape_center1_candidate = owned_wall_shape_authority_v173x.get("shape_center1")
                shape_axis_candidate = owned_wall_shape_authority_v173x.get("axis")
                shape_basis_u_candidate = owned_wall_shape_authority_v173x.get("basis_u")
                shape_basis_v_candidate = owned_wall_shape_authority_v173x.get("basis_v")
                try:
                    shape_center0_v173x = np.asarray(shape_center0_candidate, dtype=float).reshape(3)
                    shape_center1_v173x = np.asarray(shape_center1_candidate, dtype=float).reshape(3)
                    shape_axis_v173y = _unit_vector(shape_axis_candidate, fallback=axis_vec)
                    if shape_basis_u_candidate is not None and shape_basis_v_candidate is not None:
                        basis_u = _unit_vector(shape_basis_u_candidate, fallback=basis_u)
                        basis_v = _unit_vector(shape_basis_v_candidate, fallback=basis_v)
                    else:
                        basis_u, basis_v = _orthonormal_basis(shape_axis_v173y)
                    # Re-orthogonalize the exported basis to the exact owned-wall axis.
                    basis_v = _unit_vector(np.cross(shape_axis_v173y, basis_u), fallback=basis_v)
                    basis_u = _unit_vector(np.cross(basis_v, shape_axis_v173y), fallback=basis_u)
                    unified_owned_wall_shape_frame_v173y = True
                except Exception:
                    shape_center0_v173x = np.asarray(center0, dtype=float).reshape(3)
                    shape_center1_v173x = np.asarray(center1, dtype=float).reshape(3)
                    shape_axis_v173y = np.asarray(axis_vec, dtype=float).reshape(3)

    if r0 <= 1.0e-9 or r1 <= 1.0e-9:
        return {
            "enabled": False,
            "policy": "invalid_measured_radius",
            "feature_family_v162": family or entity or "unknown",
            "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        }

    feature_key = "bore" if is_bore else ("chamfer" if is_chamfer else "pocket")
    requested_target_count = _semantic_target_circumferential_count(n, mode=quad_density_mode, entity=feature_key)
    target_count = int(requested_target_count)
    v164_boundary_locked_angular_density_deferred = False
    v164_boundary_locked_reason = ""
    v168_chamfer_boundary_locked_angular_density_deferred = False
    v168_chamfer_boundary_locked_reason = ""
    v175l_pocket_boundary_locked_angular_density_deferred = False
    v175l_pocket_boundary_locked_reason = ""
    v175l_pocket_locked_loop_angle_samples: tuple[float, ...] = ()
    pocket_linear_locked_boundary_projection_v177h = False
    pocket_linear_locked_boundary_projection_reason_v177h = ""
    pocket_linear_locked_boundary_projection_spread_v177h = 0.0
    if target_count != n and not _loop_pair_transition_allowed(n, target_count, allow_unequal_loop_transition=True):
        target_count = n

    # v164: with locked boundary vertices, changing circumferential density at
    # the first/last band creates a collar/lip: the boundary loop remains coarse
    # while the generated bore cylinder immediately jumps to a denser ring.
    # That is not a semantic feature improvement; it is a transition artifact.
    # Until Rebuild owns a boundary-splitting / boundary-relaxation stage, BORE
    # remeshing may add axial rings and project them to the measured cylinder,
    # but it must keep the same angular count at the protected openings.
    if is_bore and target_count != n:
        v164_boundary_locked_angular_density_deferred = True
        v164_boundary_locked_reason = (
            "boundary_vertices_locked_no_circumferential_density_jump; "
            "defer angular refinement until protected boundary split/relaxation"
        )
        target_count = n

    # v168: the same locked-boundary density rule must apply to CHAMFER bands.
    # v167 still doubled a 20-vertex protected chamfer boundary to 40 internal
    # samples.  On a short two-axis curved chamfer, the 20->40 transition band
    # can stack upper generated segments over the next lower band even when the
    # local rail loft is otherwise correct.  Until Rebuild owns a safe
    # boundary-splitting/parent-surface relaxation stage, chamfer remeshing may
    # add axial refinement rings but must not change circumferential density at
    # locked parent-boundary loops.
    if is_chamfer and target_count != n:
        v168_chamfer_boundary_locked_angular_density_deferred = True
        v168_chamfer_boundary_locked_reason = (
            "locked_chamfer_parent_boundaries_no_20_to_40_transition_band; "
            "defer chamfer angular refinement until protected boundary split/relaxation"
        )
        target_count = n

    # v175l: blind POCKET recess cups have locked mouth/floor boundary loops in
    # the delete patch.  The generated side-wall rings may add axial refinement,
    # but they must not resample the circumferential count or fall back to a
    # BORE cylinder frame.  Reuse the measured locked mouth-loop angular samples
    # for internal rings so each generated ring follows the same seam/phase as
    # the pocket boundary instead of a uniform synthetic 0..tau angle table.
    if is_pocket and target_count != n:
        v175l_pocket_boundary_locked_angular_density_deferred = True
        v175l_pocket_boundary_locked_reason = (
            "locked_pocket_mouth_floor_boundaries_no_circumferential_density_jump; "
            "defer pocket angular refinement until protected boundary split/relaxation"
        )
        target_count = n
    if is_pocket:
        try:
            v175l_pocket_locked_loop_angle_samples = tuple(float(a) for a in _shape_frame_loop_angles_v173z(
                vertices=vertices,
                loop=loop0,
                center=np.asarray(center0, dtype=float).reshape(3),
                axis=axis_vec,
                basis_u=basis_u,
                basis_v=basis_v,
            ))
        except Exception:
            v175l_pocket_locked_loop_angle_samples = ()

        # v177h: A POCKET side-wall is bounded by two locked, measured loops and
        # an explicit floor cap.  On coarse pockets the selected annular rail can
        # carry a wide radius envelope.  Forcing generated internal rings onto a
        # nominal cylindrical radius can create a watertight but visibly folded
        # pocket.  In that case keep the axial refinement rings, but place them
        # by boundary-to-boundary interpolation instead of nominal cylinder
        # projection.  BORE and CHAMFER keep their existing projection policies.
        try:
            spread0_v177h = float(loop_radius_spread_ratio(vertices=vertices, loop=loop0, center=center0, axis=axis_vec))
            spread1_v177h = float(loop_radius_spread_ratio(vertices=vertices, loop=loop1, center=center1, axis=axis_vec))
            pocket_linear_locked_boundary_projection_spread_v177h = max(spread0_v177h, spread1_v177h)
        except Exception:
            pocket_linear_locked_boundary_projection_spread_v177h = 0.0
        if pocket_linear_locked_boundary_projection_spread_v177h > 0.18:
            pocket_linear_locked_boundary_projection_v177h = True
            pocket_linear_locked_boundary_projection_reason_v177h = (
                "wide_locked_pocket_boundary_radius_spread_use_boundary_linear_sidewall_rings_no_nominal_cylinder_projection"
            )

    if is_chamfer:
        policy = "chamfer_local_rail_curvature_adapter_v168_locked_boundary_density_guard"
        min_axial_segments = 4
        constraint_model = "measured_chamfer_surface_between_parent_boundaries_with_local_two_axis_curvature_adapter"
    elif is_pocket:
        if pocket_linear_locked_boundary_projection_v177h:
            policy = "pocket_locked_boundary_linear_sidewall_rings_v177h_no_nominal_cylinder_projection"
            constraint_model = "measured_pocket_recess_sidewall_between_owned_mouth_and_floor_boundaries_boundary_linear_safe_mode"
        else:
            policy = "pocket_cylindrical_sidewall_generated_rings_v162_constraint_bound_recess_wall"
            constraint_model = "measured_circular_recess_sidewall_between_mouth_and_floor_boundaries"
        min_axial_segments = 3
    else:
        policy = "bore_cylindrical_generated_rings_v162_constraint_bound_wall"
        min_axial_segments = 3
        constraint_model = "measured_cylindrical_wall_between_openings"

    def _rel_delta_v165(a: float, b: float) -> float:
        if a <= 1.0e-9 or b <= 1.0e-9:
            return 0.0
        return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0e-9)

    # v165: boundary-locked feature remeshing must adapt from existing protected
    # loops into the semantic primitive.  If one endpoint loop is older/chamfered
    # or laterally offset, projecting every internal ring directly onto the target
    # cylinder creates a visible lip at that endpoint.  The center region still
    # follows the measured feature; only a small protected-boundary adapter band
    # follows the original loop-to-loop interpolation.
    boundary_radius_delta0_v165 = _rel_delta_v165(float(r0_boundary), float(r0))
    boundary_radius_delta1_v165 = _rel_delta_v165(float(r1_boundary), float(r1))
    lateral_center_shift_v165 = float(np.linalg.norm((np.asarray(center1, dtype=float).reshape(3) - np.asarray(center0, dtype=float).reshape(3)) - float(np.dot(np.asarray(center1, dtype=float).reshape(3) - np.asarray(center0, dtype=float).reshape(3), axis_vec)) * axis_vec))
    avg_boundary_radius_v165 = max(0.5 * (float(r0_boundary) + float(r1_boundary)), 1.0e-9)
    lateral_center_shift_rel_v165 = float(lateral_center_shift_v165 / avg_boundary_radius_v165)
    v165_adapter_needed = bool(is_bore and (max(boundary_radius_delta0_v165, boundary_radius_delta1_v165) > 0.035 or lateral_center_shift_rel_v165 > 0.035))
    v165_adapter_fraction = 0.22 if v165_adapter_needed else 0.0

    # v173v: when Recognition has already proven a BORE_WALL by the v173u
    # opening-to-exit cylinder contract, Rebuild must not let the old locked-
    # boundary blend become wall-shape authority again.  The protected boundary
    # loops remain seam constraints, but the internal generated rings must follow
    # the measured cylindrical wall directly.  This fixes the lower-wall slope
    # left by the old pocket-over-bore mitigation path.
    v173v_opening_to_exit_projection_guard = bool(
        is_bore and _metadata_indicates_bore_opening_to_exit_wall_v173v(candidate_metadata)
    )
    if v173v_opening_to_exit_projection_guard:
        v165_adapter_needed = False
        v165_adapter_fraction = 0.0
        min_axial_segments = max(int(min_axial_segments), 6)

    # v173y: when owned BORE_WALL evidence has produced a complete shape frame,
    # Rebuild must not blend that frame with the old boundary-loop interpolation
    # frame.  The boundary loops remain locked seam constraints; internal rings
    # use one atomic wall-cylinder frame: axis, centerline, angular basis, radius.
    if unified_owned_wall_shape_frame_v173y:
        v165_adapter_needed = False
        v165_adapter_fraction = 0.0
        min_axial_segments = max(int(min_axial_segments), 6)

    # v166: the loop0-anchored centerline guard removed the top lip but could
    # leave the opposite/bottom endpoint visibly offset.  For locked-boundary
    # BORE remeshing, the safe centerline is the old-code center interpolation;
    # the semantic improvement is blended into the middle of the feature, not
    # forced through one endpoint.
    if is_bore and unified_owned_wall_shape_frame_v173y:
        v165_centerline_mode = "unified_owned_wall_axis_centerline_v173y"
    elif is_bore and v173v_opening_to_exit_projection_guard:
        v165_centerline_mode = "axis_anchored_from_loop0_v173v_opening_to_exit_projection"
    elif is_bore:
        v165_centerline_mode = "interpolate_boundary_centers_v166_locked_boundary_safe"
    else:
        v165_centerline_mode = "interpolate_boundary_centers"

    return {
        "enabled": True,
        "policy": policy,
        "feature_family_v162": feature_key,
        "constraint_model_v162": constraint_model,
        "semantic_feature_remesh_contract_v162": SEMANTIC_FEATURE_REMESH_CONTRACT_V162,
        "basis_u": np.asarray(basis_u, dtype=float).reshape(3),
        "basis_v": np.asarray(basis_v, dtype=float).reshape(3),
        "shape_center0_v173x": np.asarray(shape_center0_v173x, dtype=float).reshape(3),
        "shape_center1_v173x": np.asarray(shape_center1_v173x, dtype=float).reshape(3),
        "shape_axis_v173y": np.asarray(shape_axis_v173y, dtype=float).reshape(3),
        "shape_angle_samples_v173z": tuple(float(a) for a in v175l_pocket_locked_loop_angle_samples) if is_pocket and len(v175l_pocket_locked_loop_angle_samples) == int(target_count) else (),
        "shape_phase_offset_v173z": float(v175l_pocket_locked_loop_angle_samples[0]) if is_pocket and v175l_pocket_locked_loop_angle_samples else 0.0,
        "v173y_bore_unified_owned_wall_shape_frame": bool(unified_owned_wall_shape_frame_v173y),
        "v173y_bore_unified_frame_rule": "owned_BORE_WALL_axis_centerline_basis_and_radius_are_used_as_one_rebuild_shape_frame; boundary_loops_are_locked_seams_only",
        "radius0": float(r0),
        "radius1": float(r1),
        "constant_radius": bool(constant_radius),
        "v163_constraint_radius_authority_source": str(radius_authority.get("source", "unknown")),
        "v163_candidate_radius_rejected": bool(radius_authority.get("candidate_radius_rejected", False)),
        "v163_candidate_radius": float(candidate_radius),
        "v163_boundary_radius0": float(r0_boundary),
        "v163_boundary_radius1": float(r1_boundary),
        "v163_selected_opening_radius": float(radius_authority.get("selected_opening_radius", 0.0) or 0.0),
        "v163_learned_wall_radius": float(radius_authority.get("learned_wall_radius", 0.0) or 0.0),
        "v163_radius_conflict_reason": str(radius_authority.get("reason", "")),
        "v173x_bore_wall_cylinder_shape_authority_audit": bool(is_bore),
        "v173x_bore_wall_cylinder_shape_authority_valid": bool(owned_wall_shape_authority_v173x.get("valid", False)),
        "v173x_bore_wall_cylinder_shape_authority_used": bool(owned_wall_shape_authority_v173x.get("used", False)),
        "v173x_bore_wall_cylinder_shape_authority_reason": str(owned_wall_shape_authority_v173x.get("reason", "not_bore") if is_bore else "not_bore"),
        "v173x_owned_wall_face_count": int(owned_wall_shape_authority_v173x.get("owned_face_count", 0) or 0),
        "v173x_owned_wall_sample_count": int(owned_wall_shape_authority_v173x.get("sample_count", 0) or 0),
        "v173x_owned_wall_fitted_radius": float(owned_wall_shape_authority_v173x.get("fitted_radius", 0.0) or 0.0),
        "v173x_owned_wall_median_radius": float(owned_wall_shape_authority_v173x.get("median_radius_from_owned_faces", 0.0) or 0.0),
        "v173x_owned_wall_radial_mad": float(owned_wall_shape_authority_v173x.get("radial_mad", 0.0) or 0.0),
        "v173x_owned_wall_radial_error_rel_median": float(owned_wall_shape_authority_v173x.get("radial_error_rel_median", 0.0) or 0.0),
        "v173x_owned_wall_normal_axis_abs_median": float(owned_wall_shape_authority_v173x.get("normal_axis_abs_median", 0.0) or 0.0),
        "v173x_owned_wall_radial_normal_alignment_median": float(owned_wall_shape_authority_v173x.get("radial_normal_alignment_median", 0.0) or 0.0),
        "v173x_shape_authority_rule": "BORE rebuild shape authority comes from owned BORE_WALL cylinder evidence; rims/endpoints remain boundary/depth evidence",
        "v173z_full_axis_authority": bool(owned_wall_shape_authority_v173x.get("v173z_full_axis_authority", False)),
        "v173z_owned_wall_full_cylinder_frame_valid": bool(owned_wall_shape_authority_v173x.get("v173z_full_axis_authority", False) and owned_wall_shape_authority_v173x.get("valid", False)),
        "v173z_owned_wall_axis_pca_eigenvalues": tuple(owned_wall_shape_authority_v173x.get("pca_eigenvalues", ()) or ()),
        "v173z_owned_wall_axis_pca_ratio": float(owned_wall_shape_authority_v173x.get("pca_axis_ratio", 0.0) or 0.0),
        "v173z_owned_wall_axis_fallback_dot": float(owned_wall_shape_authority_v173x.get("fallback_axis_dot", 0.0) or 0.0),
        "v173z_shape_authority_rule": "owned_BORE_WALL_faces_define_axis_centerline_basis_radius_and_boundary_loop_phase_as_one_frame",
        "target_circumferential_segments": int(target_count),
        "v164_requested_circumferential_segments": int(requested_target_count),
        "v164_boundary_locked_circumferential_segments": int(n),
        "v164_boundary_locked_angular_density_deferred": bool(v164_boundary_locked_angular_density_deferred),
        "v164_boundary_locked_reason": str(v164_boundary_locked_reason),
        "v168_chamfer_boundary_locked_circumferential_segments": int(n),
        "v168_chamfer_boundary_locked_angular_density_deferred": bool(v168_chamfer_boundary_locked_angular_density_deferred),
        "v168_chamfer_boundary_locked_reason": str(v168_chamfer_boundary_locked_reason),
        "v168_chamfer_locked_boundary_density_guard_enabled": bool(is_chamfer),
        "pocket_radius_frame_authority_checkpoint_v175l": bool(is_pocket),
        "pocket_rebuild_shape_authority_contract_v175l": "POCKET rebuild uses pocket rim/sidewall/floor frame authority; child BORE relationship metadata must not activate BORE cylinder authority" if is_pocket else "not_applicable",
        "pocket_bore_gate_text_suppressed_for_family_authority_v175l": bool(is_pocket and "bore" in gate),
        "pocket_radius_authority_source_v175l": str(radius_authority.get("source", "unknown")) if is_pocket else "not_applicable",
        "pocket_radius_authority_family_mismatch_v175l": bool(is_pocket and "bore" in str(radius_authority.get("source", "")).lower()),
        "pocket_boundary_locked_angular_density_deferred_v175l": bool(v175l_pocket_boundary_locked_angular_density_deferred),
        "pocket_boundary_locked_reason_v175l": str(v175l_pocket_boundary_locked_reason),
        "pocket_locked_loop_angle_samples_used_v175l": bool(is_pocket and len(v175l_pocket_locked_loop_angle_samples) == int(target_count) and int(target_count) == int(n)),
        "pocket_locked_loop_angle_sample_count_v175l": int(len(v175l_pocket_locked_loop_angle_samples)) if is_pocket else 0,
        "pocket_circumferential_segments_locked_to_boundary_v175l": int(n) if is_pocket else 0,
        "pocket_requested_circumferential_segments_v175l": int(requested_target_count) if is_pocket else 0,
        "pocket_linear_locked_boundary_projection_v177h": bool(pocket_linear_locked_boundary_projection_v177h),
        "pocket_linear_locked_boundary_projection_reason_v177h": str(pocket_linear_locked_boundary_projection_reason_v177h),
        "pocket_linear_locked_boundary_projection_spread_v177h": float(pocket_linear_locked_boundary_projection_spread_v177h),
        "v165_locked_boundary_adapter_enabled": bool(v165_adapter_needed),
        "v165_locked_boundary_adapter_fraction": float(v165_adapter_fraction),
        "v165_boundary_radius_delta0_rel": float(boundary_radius_delta0_v165),
        "v165_boundary_radius_delta1_rel": float(boundary_radius_delta1_v165),
        "v165_lateral_center_shift": float(lateral_center_shift_v165),
        "v165_lateral_center_shift_rel": float(lateral_center_shift_rel_v165),
        "v165_centerline_mode": str(v165_centerline_mode),
        "v165_boundary_adapter_policy": "blend_locked_boundary_interpolation_to_semantic_cylinder" if v165_adapter_needed else "not_needed",
        "v166_endpoint_safe_bore_blend_enabled": bool(is_bore and not v173v_opening_to_exit_projection_guard and not unified_owned_wall_shape_frame_v173y),
        "v166_endpoint_safe_centerline_mode": (
            "unified_owned_wall_axis_centerline_v173y"
            if unified_owned_wall_shape_frame_v173y
            else (
                "measured_axis_projection_from_opening_to_exit_v173v"
                if v173v_opening_to_exit_projection_guard
                else ("old_code_boundary_center_interpolation" if is_bore else "not_applicable")
            )
        ),
        "v166_endpoint_safe_blend_policy": (
            "disabled_for_v173y_unified_owned_wall_shape_frame"
            if unified_owned_wall_shape_frame_v173y
            else (
                "disabled_for_v173u_opening_to_exit_bore_wall_ownership"
                if v173v_opening_to_exit_projection_guard
                else ("sinusoidal_center_only_semantic_projection_zero_at_locked_boundaries" if is_bore else "not_applicable")
            )
        ),
        "v173v_bore_opening_to_exit_rebuild_projection_guard": bool(v173v_opening_to_exit_projection_guard),
        "v173v_bore_v166_endpoint_safe_blend_disabled": bool(v173v_opening_to_exit_projection_guard),
        "v173v_bore_v165_boundary_adapter_disabled": bool(v173v_opening_to_exit_projection_guard),
        "v173v_bore_internal_rings_follow_measured_cylinder": bool(v173v_opening_to_exit_projection_guard),
        "v173v_bore_boundary_loops_locked_as_seams_only": bool(v173v_opening_to_exit_projection_guard),
        "v173v_bore_opening_to_exit_rebuild_projection_rule": (
            "accepted v173u BORE_WALL CandidateData uses measured cylinder for internal generated rings; protected loops are seams, not wall-shape authority"
            if v173v_opening_to_exit_projection_guard else "not_applicable"
        ),
        "v167_chamfer_local_rail_curvature_adapter_enabled": bool(is_chamfer),
        "v167_chamfer_local_rail_curvature_adapter_policy": (
            "locked_boundary_count_local_rails_no_circumferential_resample_v168"
            if is_chamfer else "not_applicable"
        ),
        "v167_chamfer_global_conical_projection_disabled": bool(is_chamfer),
        "min_axial_segments": int(min_axial_segments),
        "chamfer_refined_surface_v161": bool(is_chamfer),
        "pocket_sidewall_semantic_geometry_v162": bool(is_pocket),
        "generated_vertices_follow_measured_constraints_v162": True,
    }

