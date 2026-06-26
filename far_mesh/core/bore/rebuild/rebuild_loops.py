"""Loop-ordering and phase diagnostics for BoreTool rebuild.

This module is intentionally non-mutating.  It may inspect loop ordering,
orientation, cyclic seam alignment, and mouth/floor phase relationships, but it
must not authorize deletion, generate replacement topology, validate a trial
mesh, or mutate the mesh.

v176c extracts the POCKET loop-phase trace from ``rebuild.py`` as the first
loop-helper extraction step.  v176s removes the temporary density-mode duplicate
introduced by the v176n import hotfix and delegates density normalization to
``rebuild_density.py``.  The public rebuild authority remains ``rebuild.py``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math

import numpy as np

from .rebuild_density import (
    QUAD_DENSITY_MODE_FULL as QUAD_DENSITY_MODE_FULL_V176N,
    QUAD_DENSITY_MODE_LEAN as QUAD_DENSITY_MODE_LEAN_V176N,
    QUAD_DENSITY_MODE_PI as QUAD_DENSITY_MODE_PI_V176N,
    normalize_quad_density_mode_v176q as normalize_quad_density_mode_v176n,
)

REBUILD_LOOPS_EXTRACTION_CHECKPOINT_V176C = (
    "v176c_rebuild_loop_phase_helper_extraction_no_behavior_change"
)

REBUILD_LOOPS_NON_MUTATION_CONTRACT_V176C = (
    "rebuild_loops_may_order_measure_and_trace_boundary_loops_but_must_not_mutate_mesh_or_authorize_rebuild"
)

REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E = (
    "v176e_pocket_recess_loop_role_resolution_extracted_no_behavior_change"
)

REBUILD_POCKET_LOOP_ROLE_NON_MUTATION_CONTRACT_V176E = (
    "pocket_loop_role_resolution_may_label_mouth_floor_and_protected_child_loops_but_must_not_generate_or_mutate_topology"
)

REBUILD_TRANSITION_RING_HELPER_EXTRACTION_CHECKPOINT_V176N = (
    "v176n_rebuild_transition_ring_helpers_extracted_no_behavior_change"
)

REBUILD_TRANSITION_RING_HELPER_NON_MUTATION_CONTRACT_V176N = (
    "transition_ring_helpers_may_compute_count_sequences_and_band_quads_only_no_delete_authorization_no_trial_no_mutation"
)

REBUILD_LOOP_DENSITY_DEDUP_CHECKPOINT_V176S = (
    "v176s_rebuild_loop_density_dedup_no_behavior_change"
)

REBUILD_LOOP_DENSITY_DEDUP_NON_MUTATION_CONTRACT_V176S = (
    "rebuild_loops_uses_rebuild_density_as_single_density_authority_no_geometry_no_mutation_change"
)

REBUILD_UNEQUAL_LOOP_ANGLE_ALIGNMENT_EXTRACTION_CHECKPOINT_V176W = (
    "v176w_rebuild_unequal_loop_angle_alignment_extracted_no_behavior_change"
)

REBUILD_UNEQUAL_LOOP_ANGLE_ALIGNMENT_NON_MUTATION_CONTRACT_V176W = (
    "unequal_loop_angle_alignment_may_cyclically_align_loop_ids_only_no_geometry_generation_no_mesh_mutation"
)

# v176s: density labels and normalization are owned by rebuild_density.py.
# Keep the v176n local names as import aliases so the transition helper call
# sites and default arguments remain bytecode-stable in meaning.


def align_unequal_loop_pair_to_angle_samples_v176w(
    *,
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    """Cyclically align the smaller unequal loop against angular samples of the larger loop.

    The equal-loop path already performs cyclic/reversal alignment.  Unequal
    loops need the same treatment; otherwise the transition band can start at the
    wrong angular phase and form a twisted or zigzag wall even when topology is
    watertight.
    """

    n0 = len(loop0)
    n1 = len(loop1)
    if n0 == n1 or n0 < 3 or n1 < 3:
        return tuple(loop0), tuple(loop1), {"unequal_loop_alignment_used": False}

    axis = _unit_vector(axis)
    loop0_is_big = n0 > n1
    big_loop = tuple(int(v) for v in (loop0 if loop0_is_big else loop1))
    small_loop = tuple(int(v) for v in (loop1 if loop0_is_big else loop0))
    big_center = np.asarray(center0 if loop0_is_big else center1, dtype=float).reshape(3)
    small_center = np.asarray(center1 if loop0_is_big else center0, dtype=float).reshape(3)

    big_radial = _project_radial(vertices[np.asarray(big_loop, dtype=np.int64)], big_center, axis)
    small_base = _project_radial(vertices[np.asarray(small_loop, dtype=np.int64)], small_center, axis)
    n_big = len(big_loop)
    n_small = len(small_loop)
    big_sample_indices = tuple(int(round(float(j) * float(n_big) / float(n_small))) % n_big for j in range(n_small))
    big_samples = big_radial[np.asarray(big_sample_indices, dtype=np.int64), :]

    best_score = float("inf")
    best_small = tuple(small_loop)
    best_reversed = False
    best_shift = 0
    for reversed_flag, candidate in ((False, tuple(small_loop)), (True, tuple(reversed(small_loop)))):
        candidate_radial = _project_radial(vertices[np.asarray(candidate, dtype=np.int64)], small_center, axis)
        for shift in range(n_small):
            shifted_radial = np.roll(candidate_radial, -shift, axis=0)
            score = float(np.mean(np.sum((big_samples - shifted_radial) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best_small = tuple(int(v) for v in np.roll(np.asarray(candidate, dtype=np.int64), -shift).tolist())
                best_reversed = bool(reversed_flag)
                best_shift = int(shift)

    diagnostics = {
        "unequal_loop_alignment_used": True,
        "unequal_loop_alignment_score": float(best_score),
        "unequal_loop_alignment_reversed_smaller_loop": bool(best_reversed),
        "unequal_loop_alignment_shift": int(best_shift),
        "unequal_loop_alignment_big_loop_count": int(n_big),
        "unequal_loop_alignment_small_loop_count": int(n_small),
    }
    if loop0_is_big:
        return big_loop, best_small, diagnostics
    return best_small, big_loop, diagnostics

def target_unequal_transition_band_count_v176n(
    *,
    big_points: np.ndarray,
    small_points: np.ndarray,
    big_center: np.ndarray,
    small_center: np.ndarray,
    mode: str,
    base_band_count: int,
) -> dict[str, object]:
    """Return the axial band target for an unequal-loop transition.

    Extracted from rebuild.py without intended behavior change.  Density
    controls axial spacing only; the larger ring still reduces to the smaller
    ring without moving boundary vertices.
    """

    mode_norm = normalize_quad_density_mode_v176n(mode)
    base_band_count = max(1, int(base_band_count))
    span = float(np.linalg.norm(np.asarray(small_center, dtype=float).reshape(3) - np.asarray(big_center, dtype=float).reshape(3)))

    edge_lengths: list[float] = []
    for pts in (np.asarray(big_points, dtype=float).reshape((-1, 3)), np.asarray(small_points, dtype=float).reshape((-1, 3))):
        if len(pts) >= 2:
            diffs = np.roll(pts, -1, axis=0) - pts
            edge_lengths.extend(float(v) for v in np.linalg.norm(diffs, axis=1) if np.isfinite(float(v)) and float(v) > 1.0e-12)
    measured_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    measured_edge = max(measured_edge, 1.0e-12)

    if mode_norm == QUAD_DENSITY_MODE_FULL_V176N:
        pitch = measured_edge
        cap = 96
        policy = "full_equal_edge"
    elif mode_norm == QUAD_DENSITY_MODE_PI_V176N:
        pitch = max(2.5 * measured_edge, measured_edge)
        cap = 64
        policy = "balanced_edge_spacing"
    else:
        pitch = max(4.0 * measured_edge, measured_edge)
        cap = max(base_band_count, 48)
        policy = "lean_drop_bands"

    raw_segments = int(math.ceil(span / max(pitch, 1.0e-12))) if np.isfinite(span) and span > 1.0e-12 else base_band_count
    target = base_band_count if mode_norm == QUAD_DENSITY_MODE_LEAN_V176N else max(base_band_count, min(int(cap), max(1, raw_segments)))

    return {
        "policy": policy,
        "target_band_count": int(target),
        "base_band_count": int(base_band_count),
        "span": float(span),
        "median_boundary_edge_length": float(measured_edge),
        "target_axial_edge_length": float(pitch),
        "raw_equal_edge_axial_segments": int(raw_segments),
        "axial_segment_cap": int(cap),
    }


def densify_transition_count_sequence_v176n(base_counts: tuple[int, ...], *, target_band_count: int) -> tuple[int, ...]:
    """Return a monotone ring-count sequence with evenly distributed drops."""

    base = tuple(int(v) for v in tuple(base_counts or ()))
    if len(base) <= 1:
        return base

    n_big = int(base[0])
    n_small = int(base[-1])
    if n_big == n_small:
        return (n_big,)
    if n_big < n_small:
        n_big, n_small = n_small, n_big

    delta = n_big - n_small
    if delta % 2:
        raise ValueError(f"Unequal transition count sequence needs even delta. got {n_big}, {n_small}.")

    drop_units = delta // 2
    base_band_count = max(1, len(base) - 1)
    target_band_count = max(base_band_count, int(target_band_count))

    counts: list[int] = []
    previous_drop = -1
    for ring_index in range(target_band_count + 1):
        if ring_index == target_band_count:
            current_drop = drop_units
        else:
            current_drop = int(math.floor(float(ring_index) * float(drop_units) / float(target_band_count)))
        current_drop = max(previous_drop, min(drop_units, current_drop))
        counts.append(int(n_big - 2 * current_drop))
        previous_drop = current_drop

    counts[0] = int(n_big)
    counts[-1] = int(n_small)
    return tuple(int(v) for v in counts)


def transition_count_sequence_v176n(n_big: int, n_small: int, *, mode: str = QUAD_DENSITY_MODE_LEAN_V176N) -> tuple[int, ...]:
    """Return density-controlled gradual ring counts from larger loop to smaller loop."""

    n_big = int(n_big)
    n_small = int(n_small)
    if n_big < n_small:
        n_big, n_small = n_small, n_big
    if n_big == n_small:
        return (n_big,)
    delta = n_big - n_small
    if delta % 2:
        raise ValueError(f"Unequal transition count sequence needs even delta. got {n_big}, {n_small}.")

    mode_norm = normalize_quad_density_mode_v176n(mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL_V176N:
        max_drop_fixed = 1
        ratio = 0.030
    elif mode_norm == QUAD_DENSITY_MODE_PI_V176N:
        max_drop_fixed = 2
        ratio = 0.060
    else:
        max_drop_fixed = 8
        ratio = 0.140

    counts = [n_big]
    current = n_big
    while current > n_small:
        remaining_delta = current - n_small
        remaining_drop = remaining_delta // 2
        max_drop_this_band = max(1, min(int(max_drop_fixed), int(round(ratio * float(current))), remaining_drop))
        next_count = current - 2 * max_drop_this_band
        if next_count < n_small:
            next_count = n_small
        # Avoid a tiny final step if possible by taking it now.
        if next_count - n_small == 2 and len(counts) > 0:
            next_count = n_small
        counts.append(int(next_count))
        current = int(next_count)
    return tuple(int(v) for v in counts)


def distributed_drop_positions_v176n(n_small: int, drop_count: int) -> set[int]:
    """Return distributed small-ring drop positions for unequal ring-band quads."""

    n_small = int(n_small)
    drop_count = int(drop_count)
    if n_small <= 0 or drop_count <= 0:
        return set()
    positions: set[int] = set()
    for k in range(drop_count):
        pos = int(round((float(k) + 0.5) * float(n_small) / float(drop_count))) % n_small
        while pos in positions:
            pos = (pos + 1) % n_small
        positions.add(pos)
    return positions


def band_quads_between_rings_v176n(ring_a: tuple[int, ...], ring_b: tuple[int, ...]) -> tuple[tuple[int, int, int, int], ...]:
    """Build logical quads between two rings with equal or even-different counts."""

    n_a = int(len(ring_a))
    n_b = int(len(ring_b))
    if n_a < 3 or n_b < 3:
        return ()
    if n_a == n_b:
        return tuple(
            (int(ring_a[i]), int(ring_a[(i + 1) % n_a]), int(ring_b[(i + 1) % n_b]), int(ring_b[i]))
            for i in range(n_a)
        )

    a_is_big = n_a > n_b
    big = tuple(int(v) for v in (ring_a if a_is_big else ring_b))
    small = tuple(int(v) for v in (ring_b if a_is_big else ring_a))
    n_big = len(big)
    n_small = len(small)
    if (n_big - n_small) % 2:
        raise ValueError(f"Band count delta must be even. got {n_a}, {n_b}.")
    drop_count = (n_big - n_small) // 2
    if drop_count > n_small:
        raise ValueError(f"Band drop count exceeds smaller ring capacity. got {n_a}, {n_b}.")
    drop_positions = distributed_drop_positions_v176n(n_small, drop_count)

    quads_big_to_small: list[tuple[int, int, int, int]] = []
    i = 0
    for j in range(n_small):
        if j in drop_positions:
            quads_big_to_small.append((
                int(big[i % n_big]),
                int(big[(i + 1) % n_big]),
                int(big[(i + 2) % n_big]),
                int(small[j]),
            ))
            i += 2
        quads_big_to_small.append((
            int(big[i % n_big]),
            int(big[(i + 1) % n_big]),
            int(small[(j + 1) % n_small]),
            int(small[j]),
        ))
        i += 1
    if i != n_big:
        raise ValueError(f"Band transition consumed {i} big edges; expected {n_big}.")

    if a_is_big:
        return tuple(quads_big_to_small)
    # Reverse orientation order when the larger ring is ring_b so the logical
    # strip still travels from ring_a to ring_b.
    return tuple((q[3], q[2], q[1], q[0]) for q in quads_big_to_small)

EdgeKey = tuple[int, int]


def _edge_key(edge: object) -> EdgeKey:
    try:
        a, b = tuple(edge)[:2]
        ia = int(a)
        ib = int(b)
    except Exception:
        return (0, 0)
    return (ia, ib) if ia <= ib else (ib, ia)


def loop_edge_set_v176e(loop_record: Mapping[str, object]) -> set[EdgeKey]:
    """Return normalized edge keys for a loop record.

    Diagnostic/ordering helper only.  It never mutates geometry or decides
    rebuild permission.
    """

    return {_edge_key(edge) for edge in tuple(loop_record.get("edges", ()) or ())}


def loop_vertex_set_v176e(loop_record: Mapping[str, object]) -> set[int]:
    """Return integer vertices for a loop record."""

    return {int(v) for v in tuple(loop_record.get("vertices", ()) or ())}


def loop_vertices_v176e(loop_record: Mapping[str, object], *, vertex_count: int) -> tuple[int, ...]:
    """Return valid ordered loop vertices for a loop record."""

    return tuple(int(v) for v in tuple(loop_record.get("vertices", ()) or ()) if 0 <= int(v) < int(vertex_count))


def loop_signature_v176e(loop_record: Mapping[str, object]) -> frozenset[int]:
    """Return a vertex-set signature used to compare boundary-loop roles."""

    return frozenset(loop_vertex_set_v176e(loop_record))


def resolve_pocket_recess_loop_roles_v176e(
    *,
    vertices: np.ndarray,
    axis: object,
    wall_boundary_loops: Iterable[Mapping[str, object]],
    floor_boundary_loops: Iterable[Mapping[str, object]],
    combined_boundary_loops: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Resolve top/mouth, floor, and protected child loops for POCKET rebuild.

    This is a no-behavior-change extraction from ``rebuild.py``.  The helper
    labels existing boundary-loop records for the POCKET recess-cup path:

        floor_loop_record
            Owned POCKET floor outer perimeter.

        protected_floor_hole_records
            Other owned-floor loops, interpreted as protected child openings.

        top_record
            Mouth/top loop for the recess side-wall.

        bottom_record
            Floor-side loop for the recess side-wall.

    It does not authorize deletion, generate vertices/faces, validate topology,
    or mutate the mesh.
    """

    wall_records = tuple(wall_boundary_loops or ())
    floor_records = tuple(floor_boundary_loops or ())
    combined_records = tuple(combined_boundary_loops or ())
    vertex_count = int(len(vertices))
    if not floor_records:
        return {
            "valid": False,
            "reason": "missing_floor_boundary_loops",
            "wall_boundary_loop_count": int(len(wall_records)),
            "floor_boundary_loop_count": 0,
            "combined_boundary_loop_count": int(len(combined_records)),
        }

    def _floor_wall_overlap_score(floor_record: Mapping[str, object]) -> tuple[int, int, int, int]:
        floor_edges_local = loop_edge_set_v176e(floor_record)
        floor_vertices_local = loop_vertex_set_v176e(floor_record)
        best_edge_overlap = 0
        best_vertex_overlap = 0
        for wall_record in wall_records:
            wall_edges = loop_edge_set_v176e(wall_record)
            wall_vertices = loop_vertex_set_v176e(wall_record)
            best_edge_overlap = max(best_edge_overlap, int(len(floor_edges_local & wall_edges)))
            best_vertex_overlap = max(best_vertex_overlap, int(len(floor_vertices_local & wall_vertices)))
        return (best_edge_overlap, best_vertex_overlap, int(len(floor_edges_local)), int(len(floor_vertices_local)))

    floor_loop_record = max(floor_records, key=_floor_wall_overlap_score)
    floor_edges = loop_edge_set_v176e(floor_loop_record)
    floor_vertices = loop_vertex_set_v176e(floor_loop_record)
    protected_floor_hole_records = tuple(
        record for record in floor_records
        if loop_vertex_set_v176e(record) != floor_vertices and len(loop_vertex_set_v176e(record)) >= 3
    )

    def _wall_floor_overlap(loop_record: Mapping[str, object]) -> tuple[int, int, int]:
        loop_edges = loop_edge_set_v176e(loop_record)
        loop_vertices = loop_vertex_set_v176e(loop_record)
        return (int(len(loop_edges & floor_edges)), int(len(loop_vertices & floor_vertices)), int(len(loop_vertices)))

    axis_vec = _unit_vector(axis)
    floor_loop_vertices = loop_vertices_v176e(floor_loop_record, vertex_count=vertex_count)
    floor_center_for_choice = (
        np.mean(vertices[np.asarray(floor_loop_vertices, dtype=np.int64), :3], axis=0)
        if floor_loop_vertices
        else np.zeros(3, dtype=float)
    )
    floor_sig = loop_signature_v176e(floor_loop_record)

    def _top_loop_score(loop_record: Mapping[str, object]) -> tuple[float, int, int, int]:
        verts = loop_vertices_v176e(loop_record, vertex_count=vertex_count)
        if len(verts) < 3:
            return (-1.0, 0, 0, 0)
        loop_sig = loop_signature_v176e(loop_record)
        overlap_vertices = int(len(loop_sig & floor_sig))
        center = np.mean(vertices[np.asarray(verts, dtype=np.int64), :3], axis=0)
        axial_sep = abs(float(np.dot(center - floor_center_for_choice, axis_vec)))
        return (float(axial_sep), -overlap_vertices, int(len(verts)), int(len(loop_edge_set_v176e(loop_record))))

    def _best_top_record_from(records: Iterable[Mapping[str, object]]) -> Mapping[str, object] | None:
        candidates: list[Mapping[str, object]] = []
        for record in tuple(records or ()):  # tolerate generators and None-like values
            verts = loop_vertices_v176e(record, vertex_count=vertex_count)
            if len(verts) < 3:
                continue
            # Do not reuse the floor perimeter as the top opening unless it is
            # literally the only loop left; a POCKET cup needs two loop roles.
            if loop_signature_v176e(record) == floor_sig:
                continue
            candidates.append(record)
        if not candidates:
            return None
        return max(candidates, key=_top_loop_score)

    loop_resolution_source = "wall_two_loop_pair"
    loop_resolution_status = "sidewall_supplied_top_and_floor_loops"
    bottom_record: Mapping[str, object] = floor_loop_record
    top_record: Mapping[str, object] | None = None

    if len(wall_records) >= 2:
        bottom_record = max(wall_records, key=_wall_floor_overlap)
        bottom_sig = loop_signature_v176e(bottom_record)
        top_candidates = [item for item in wall_records if loop_signature_v176e(item) != bottom_sig]
        top_record = max(top_candidates, key=lambda item: len(tuple(item.get("vertices", ()) or ()))) if top_candidates else None
    else:
        # Coarse/tessellated POCKET ownership can produce a valid owned floor
        # loop and a single side-wall boundary loop.  Use the same fallback
        # policy as the original v144 code path in rebuild.py.
        loop_resolution_source = "floor_loop_plus_single_wall_or_patch_mouth_loop_v144"
        loop_resolution_status = "sidewall_single_loop_floor_fallback_used"
        top_record = _best_top_record_from(wall_records)
        if top_record is None:
            top_record = _best_top_record_from(combined_records)
            loop_resolution_source = "floor_loop_plus_combined_patch_mouth_loop_v144"
        bottom_record = floor_loop_record

    wall_loop_summaries = tuple(
        {
            "index": int(i),
            "vertex_count": int(len(tuple(item.get("vertices", ()) or ()))),
            "floor_overlap_edges": int(_wall_floor_overlap(item)[0]),
            "floor_overlap_vertices": int(_wall_floor_overlap(item)[1]),
        }
        for i, item in enumerate(wall_records[:12])
    )

    if top_record is None:
        return {
            "valid": False,
            "reason": "top_loop_not_resolved",
            "floor_loop_record": floor_loop_record,
            "protected_floor_hole_records": protected_floor_hole_records,
            "loop_resolution_source": str(loop_resolution_source),
            "loop_resolution_status": str(loop_resolution_status),
            "wall_loop_summaries": wall_loop_summaries,
            "wall_boundary_loop_count": int(len(wall_records)),
            "floor_boundary_loop_count": int(len(floor_records)),
            "combined_boundary_loop_count": int(len(combined_records)),
        }

    top_loop = loop_vertices_v176e(top_record, vertex_count=vertex_count)
    bottom_loop = loop_vertices_v176e(bottom_record, vertex_count=vertex_count)
    return {
        "valid": True,
        "floor_loop_record": floor_loop_record,
        "protected_floor_hole_records": protected_floor_hole_records,
        "top_record": top_record,
        "bottom_record": bottom_record,
        "top_loop": top_loop,
        "bottom_loop": bottom_loop,
        "floor_loop_vertices": floor_loop_vertices,
        "floor_loop_vertex_set": set(floor_vertices),
        "loop_resolution_source": str(loop_resolution_source),
        "loop_resolution_status": str(loop_resolution_status),
        "wall_loop_summaries": wall_loop_summaries,
        "wall_boundary_loop_count": int(len(wall_records)),
        "floor_boundary_loop_count": int(len(floor_records)),
        "combined_boundary_loop_count": int(len(combined_records)),
        "checkpoint": REBUILD_POCKET_LOOP_ROLE_EXTRACTION_CHECKPOINT_V176E,
        "non_mutation_contract": REBUILD_POCKET_LOOP_ROLE_NON_MUTATION_CONTRACT_V176E,
    }


def _unit_vector(value: object, fallback: object = (0.0, 0.0, 1.0)) -> np.ndarray:
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


def _orthonormal_basis(axis: object) -> tuple[np.ndarray, np.ndarray]:
    axis_vec = _unit_vector(axis, fallback=(0.0, 0.0, 1.0))
    candidates = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    ref = min(candidates, key=lambda item: abs(float(np.dot(item, axis_vec))))
    basis_u = ref - float(np.dot(ref, axis_vec)) * axis_vec
    basis_u = _unit_vector(basis_u, fallback=(1.0, 0.0, 0.0))
    if abs(float(np.dot(basis_u, axis_vec))) > 0.95:
        basis_u = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = basis_u - float(np.dot(basis_u, axis_vec)) * axis_vec
        basis_u = _unit_vector(basis_u, fallback=(0.0, 1.0, 0.0))
    basis_v = _unit_vector(np.cross(axis_vec, basis_u), fallback=(0.0, 1.0, 0.0))
    basis_u = _unit_vector(np.cross(basis_v, axis_vec), fallback=basis_u)
    return np.asarray(basis_u, dtype=float).reshape(3), np.asarray(basis_v, dtype=float).reshape(3)


def _project_radial(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=float)[:, :3] - np.asarray(center, dtype=float).reshape(1, 3)
    axis = _unit_vector(axis)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    lengths = np.linalg.norm(radial, axis=1)
    out = np.zeros_like(radial)
    valid = lengths > 1.0e-12
    out[valid] = radial[valid] / lengths[valid].reshape(-1, 1)
    return out


def _loop_angle_basis(*, vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_vec = _unit_vector(axis)
    ids = tuple(int(v) for v in loop if 0 <= int(v) < len(vertices))
    center_arr = np.asarray(center, dtype=float).reshape(3)
    basis_u: np.ndarray | None = None
    for vid in ids:
        rel = vertices[int(vid), :3] - center_arr
        radial = rel - float(np.dot(rel, axis_vec)) * axis_vec
        norm = float(np.linalg.norm(radial))
        if norm > 1.0e-12 and np.isfinite(norm):
            basis_u = radial / norm
            break
    if basis_u is None:
        basis_u, _ = _orthonormal_basis(axis_vec)
    basis_v = _unit_vector(np.cross(axis_vec, basis_u), fallback=(0.0, 1.0, 0.0))
    basis_u = _unit_vector(np.cross(basis_v, axis_vec), fallback=basis_u)
    return np.asarray(basis_u, dtype=float).reshape(3), np.asarray(basis_v, dtype=float).reshape(3)


def _loop_radius_stats(vertices: np.ndarray, loop: tuple[int, ...], center: np.ndarray, axis: np.ndarray) -> dict[str, float]:
    ids = [int(v) for v in tuple(loop or ()) if 0 <= int(v) < len(vertices)]
    if not ids:
        return {"median": 0.0, "p95_delta": 0.0}
    pts = vertices[np.asarray(ids, dtype=np.int64), :3]
    rel = pts - np.asarray(center, dtype=float).reshape(1, 3)
    axis = _unit_vector(axis)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    radii = np.linalg.norm(radial, axis=1)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return {"median": 0.0, "p95_delta": 0.0}
    median = float(np.median(radii))
    p95_delta = float(np.percentile(np.abs(radii - median), 95.0))
    return {"median": median, "p95_delta": p95_delta}


def pocket_loop_phase_trace_v175j(
    *,
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
    label: str,
) -> dict[str, object]:
    """Diagnostic-only loop phase/orientation trace for blind POCKET rebuilds.

    v175j contract:
        This helper records how the mouth/floor loop pair is ordered and phase
        aligned before and after planning.  It does not choose loops, modify
        topology, authorize CandidateData, or change generated geometry.  Its
        purpose is to expose the source of the visual "rolling wave" twist in
        blind-pocket rebuilds: radius authority leak, reversed loop orientation,
        cyclic seam offset, or unequal loop phase mapping.
    """

    l0 = tuple(int(v) for v in tuple(loop0 or ()) if 0 <= int(v) < len(vertices))
    l1 = tuple(int(v) for v in tuple(loop1 or ()) if 0 <= int(v) < len(vertices))
    out: dict[str, object] = {
        "trace_label": str(label),
        "diagnostic_only": True,
        "loop0_count": int(len(l0)),
        "loop1_count": int(len(l1)),
        "equal_loop_counts": bool(len(l0) == len(l1) and len(l0) >= 3),
    }
    if len(l0) < 3 or len(l1) < 3:
        out.update({"valid": False, "reason": "insufficient_loop_vertices"})
        return out

    try:
        axis_vec = _unit_vector(axis)
        c0 = np.asarray(center0, dtype=float).reshape(3)
        c1 = np.asarray(center1, dtype=float).reshape(3)
        basis_u, basis_v = _loop_angle_basis(vertices=vertices, loop=l0, center=c0, axis=axis_vec)

        def _angles(loop: tuple[int, ...], center: np.ndarray) -> np.ndarray:
            vals: list[float] = []
            for vid in loop:
                rel = vertices[int(vid), :3] - center
                radial = rel - float(np.dot(rel, axis_vec)) * axis_vec
                angle = float(np.arctan2(float(np.dot(radial, basis_v)), float(np.dot(radial, basis_u))))
                if angle < 0.0:
                    angle += float(2.0 * np.pi)
                vals.append(angle)
            return np.asarray(vals, dtype=float)

        def _wrap_delta(delta: np.ndarray) -> np.ndarray:
            return (delta + np.pi) % (2.0 * np.pi) - np.pi

        def _orientation(angles: np.ndarray) -> dict[str, object]:
            if angles.size < 3:
                return {"signed_turn": 0.0, "orientation": "unknown"}
            deltas = _wrap_delta(np.roll(angles, -1) - angles)
            signed = float(np.sum(deltas))
            if abs(signed) < 1.0e-6:
                orient = "degenerate_or_unordered"
            else:
                orient = "positive" if signed > 0.0 else "negative"
            return {
                "signed_turn": float(signed),
                "signed_turn_abs_over_tau": float(abs(signed) / max(float(2.0 * np.pi), 1.0e-12)),
                "orientation": orient,
                "median_step_abs_degrees": float(np.degrees(np.median(np.abs(deltas)))) if deltas.size else 0.0,
                "max_step_abs_degrees": float(np.degrees(np.max(np.abs(deltas)))) if deltas.size else 0.0,
            }

        a0 = _angles(l0, c0)
        a1 = _angles(l1, c1)
        o0 = _orientation(a0)
        o1 = _orientation(a1)
        stats0 = _loop_radius_stats(vertices, l0, c0, axis_vec)
        stats1 = _loop_radius_stats(vertices, l1, c1, axis_vec)
        centerline_distance = float(np.linalg.norm((c1 - c0) - float(np.dot(c1 - c0, axis_vec)) * axis_vec))
        axial_separation = abs(float(np.dot(c1 - c0, axis_vec)))
        out.update({
            "valid": True,
            "axis": tuple(float(v) for v in axis_vec.reshape(3).tolist()),
            "centerline_distance": float(centerline_distance),
            "axial_separation": float(axial_separation),
            "loop0_orientation": str(o0.get("orientation", "unknown")),
            "loop1_orientation": str(o1.get("orientation", "unknown")),
            "loop_orientation_opposed": bool(str(o0.get("orientation")) != str(o1.get("orientation")) and "unknown" not in {str(o0.get("orientation")), str(o1.get("orientation"))}),
            "loop0_signed_turn": float(o0.get("signed_turn", 0.0) or 0.0),
            "loop1_signed_turn": float(o1.get("signed_turn", 0.0) or 0.0),
            "loop0_median_step_abs_degrees": float(o0.get("median_step_abs_degrees", 0.0) or 0.0),
            "loop1_median_step_abs_degrees": float(o1.get("median_step_abs_degrees", 0.0) or 0.0),
            "loop0_max_step_abs_degrees": float(o0.get("max_step_abs_degrees", 0.0) or 0.0),
            "loop1_max_step_abs_degrees": float(o1.get("max_step_abs_degrees", 0.0) or 0.0),
            "loop0_radius_median": float(stats0.get("median", 0.0) or 0.0),
            "loop1_radius_median": float(stats1.get("median", 0.0) or 0.0),
            "loop0_radius_p95_delta": float(stats0.get("p95_delta", 0.0) or 0.0),
            "loop1_radius_p95_delta": float(stats1.get("p95_delta", 0.0) or 0.0),
        })

        # Equal-count loop pairs are where twist is easiest to diagnose: a
        # large cyclic shift or a required reversal means the side-wall generator
        # may have sewn the two locked boundaries with the wrong phase.
        if len(l0) == len(l1) and len(l0) >= 3:
            radial0 = _project_radial(vertices[np.asarray(l0, dtype=np.int64)], c0, axis_vec)
            best_score = float("inf")
            best_max = float("inf")
            best_shift = 0
            best_reversed = False
            for reversed_flag, candidate in ((False, l1), (True, tuple(reversed(l1)))):
                radial1 = _project_radial(vertices[np.asarray(candidate, dtype=np.int64)], c1, axis_vec)
                for shift in range(len(l1)):
                    shifted = np.roll(radial1, -shift, axis=0)
                    errs = np.linalg.norm(radial0 - shifted, axis=1)
                    score = float(np.mean(errs)) if errs.size else float("inf")
                    max_err = float(np.max(errs)) if errs.size else float("inf")
                    if score < best_score:
                        best_score = score
                        best_max = max_err
                        best_shift = int(shift)
                        best_reversed = bool(reversed_flag)
            shift_fraction = float(best_shift) / max(float(len(l1)), 1.0)
            out.update({
                "equal_count_phase_alignment_available": True,
                "best_cyclic_shift": int(best_shift),
                "best_cyclic_shift_fraction": float(shift_fraction),
                "best_cyclic_shift_degrees": float(360.0 * shift_fraction),
                "best_alignment_reversed_loop1": bool(best_reversed),
                "best_alignment_mean_unit_chord_error": float(best_score),
                "best_alignment_max_unit_chord_error": float(best_max),
                "phase_wave_risk": bool(best_reversed or best_score > 0.20 or best_max > 0.65),
            })
        else:
            out.update({
                "equal_count_phase_alignment_available": False,
                "unequal_loop_phase_trace_required": bool(len(l0) != len(l1)),
                "phase_wave_risk": bool(str(o0.get("orientation")) != str(o1.get("orientation")) and "unknown" not in {str(o0.get("orientation")), str(o1.get("orientation"))}),
            })
    except Exception as exc:  # diagnostics must never break rebuild
        out.update({"valid": False, "reason": "trace_exception", "error": str(exc)})
    return out

REBUILD_GENERIC_LOOP_HELPER_EXTRACTION_CHECKPOINT_V176G = (
    "v176g_generic_rebuild_loop_ordering_helpers_extracted_no_behavior_change"
)

REBUILD_GENERIC_LOOP_HELPER_NON_MUTATION_CONTRACT_V176G = (
    "generic_rebuild_loop_helpers_may_order_align_and_convert_loops_but_must_not_generate_or_mutate_topology"
)


def loop_edges_from_vertices_v176g(loop_vertices: Iterable[int]) -> tuple[EdgeKey, ...]:
    """Return normalized edge keys for an ordered closed vertex loop.

    Extracted from ``rebuild.py`` without behavior change.  This helper only
    converts an already-resolved loop into edge keys; it does not classify,
    authorize, generate, validate, or mutate topology.
    """

    verts = tuple(int(v) for v in tuple(loop_vertices or ()))
    if len(verts) < 2:
        return ()
    edges: set[EdgeKey] = set()
    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        if int(a) == int(b):
            continue
        edges.add(_edge_key((int(a), int(b))))
    return tuple(sorted(edges))


def sort_loop_pair_by_angle_v176g(
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sort two closed loops in a shared angular frame.

    Pure loop-ordering helper extracted from ``rebuild.py`` without behavior
    change.  It returns reordered vertex IDs only.
    """

    axis = _unit_vector(axis)
    basis_u = None
    for vid in loop0:
        if 0 <= int(vid) < len(vertices):
            radial = vertices[int(vid), :3] - center0
            radial = radial - float(np.dot(radial, axis)) * axis
            length = float(np.linalg.norm(radial))
            if np.isfinite(length) and length > 1.0e-12:
                basis_u = radial / length
                break
    if basis_u is None:
        basis_u = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(basis_u, axis))) > 0.90:
            basis_u = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = basis_u - float(np.dot(basis_u, axis)) * axis
        basis_u = _unit_vector(basis_u)
    basis_v = _unit_vector(np.cross(axis, basis_u), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    return (
        sort_loop_by_angle_v176g(vertices, loop0, center0, axis, basis_u, basis_v),
        sort_loop_by_angle_v176g(vertices, loop1, center1, axis, basis_u, basis_v),
    )


def sort_loop_by_angle_v176g(
    vertices: np.ndarray,
    loop: tuple[int, ...],
    center: np.ndarray,
    axis: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[int, ...]:
    """Sort a closed loop around an existing angular basis."""

    items: list[tuple[float, int]] = []
    seen: set[int] = set()
    for raw in loop:
        vid = int(raw)
        if vid in seen or vid < 0 or vid >= len(vertices):
            continue
        seen.add(vid)
        rel = vertices[vid, :3] - center
        radial = rel - float(np.dot(rel, axis)) * axis
        angle = float(np.arctan2(float(np.dot(radial, basis_v)), float(np.dot(radial, basis_u))))
        if angle < 0.0:
            angle += float(2.0 * np.pi)
        items.append((angle, vid))
    items.sort(key=lambda item: item[0])
    return tuple(int(vid) for _, vid in items)


def _project_radial_for_alignment_v176g(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=float)[:, :3] - np.asarray(center, dtype=float).reshape(1, 3)
    axis = _unit_vector(axis)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
    lengths = np.linalg.norm(radial, axis=1)
    out = np.zeros_like(radial)
    valid = lengths > 1.0e-12
    out[valid] = radial[valid] / lengths[valid].reshape(-1, 1)
    return out


def align_second_loop_to_first_v176g(
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
) -> tuple[int, ...]:
    """Cyclically shift/reverse loop1 to best match loop0 radial directions."""

    n = int(len(loop0))
    radial0 = _project_radial_for_alignment_v176g(vertices[np.asarray(loop0, dtype=np.int64)], center0, axis)
    best_score = float("inf")
    best_loop = tuple(loop1)
    for candidate in (tuple(loop1), tuple(reversed(loop1))):
        radial1 = _project_radial_for_alignment_v176g(vertices[np.asarray(candidate, dtype=np.int64)], center1, axis)
        for shift in range(n):
            shifted = np.roll(radial1, -shift, axis=0)
            score = float(np.mean(np.sum((radial0 - shifted) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best_loop = tuple(int(v) for v in np.roll(np.asarray(candidate, dtype=np.int64), -shift).tolist())
    return best_loop


def order_closed_edge_loop_vertices_v176g(edges: set[EdgeKey]) -> dict[str, object]:
    """Order a degree-2 closed edge component into vertices and edge keys."""

    normalized = {_edge_key(edge) for edge in edges}
    if len(normalized) < 3:
        return {"vertices": (), "edges": (), "closed": False}

    adjacency: dict[int, list[int]] = {}
    for a, b in sorted(normalized):
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    for neighbors in adjacency.values():
        neighbors.sort()
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return {"vertices": (), "edges": (), "closed": False}

    start = min(adjacency.keys())
    previous: int | None = None
    current = int(start)
    ordered_vertices: list[int] = [current]
    ordered_edges: list[EdgeKey] = []

    for _ in range(len(normalized) + 2):
        candidates = [int(v) for v in adjacency.get(current, ()) if int(v) != previous]
        if not candidates:
            return {"vertices": (), "edges": (), "closed": False}
        nxt = candidates[0]
        edge = _edge_key((current, nxt))
        ordered_edges.append(edge)
        previous, current = current, nxt
        if current == start:
            break
        ordered_vertices.append(current)

    if current != start or set(ordered_edges) != normalized:
        return {"vertices": (), "edges": (), "closed": False}
    return {
        "vertices": tuple(int(v) for v in ordered_vertices),
        "edges": tuple(_edge_key(edge) for edge in ordered_edges),
        "closed": True,
    }


def loop_vertices_to_edges_v176g(loop: tuple[int, ...]) -> set[EdgeKey]:
    """Return normalized edge keys for a closed vertex loop."""

    if len(loop) < 2:
        return set()
    return {_edge_key((loop[i], loop[(i + 1) % len(loop)])) for i in range(len(loop))}
