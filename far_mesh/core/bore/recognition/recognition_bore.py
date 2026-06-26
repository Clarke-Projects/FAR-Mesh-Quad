"""BORE surface-role ownership helpers for FAR MESH BoreTool.

This module is the first BORE-local split from
``recognition_component_engine.py``.  It owns BORE-local helper logic only.  It
does not emit CandidateData, decide rebuild targets, mutate topology, or perform
host state changes.

Semantic boundary
-----------------
Input meaning:
    measured BORE evidence + already-computed terminal-continuation audit state

Output meaning:
    filtered BORE_WALL ownership state and diagnostics proving that terminal
    continuation remains evidence-only unless its own BORE_WALL role audit passed.

The component engine remains the orchestration and CandidateData emission owner.
v174h moves the already-extracted v173w/v174d terminal-continuation filter
here while preserving its return keys and behavior. v174i continues the BORE-local
split by moving wall-subset validation helpers into this module without moving
CandidateData emission out of the component engine. v174j moves selected-opening
locality and seed-island evidence helpers into this module with delegates left in
the component engine. v174k moves BORE CandidateData construction helpers here,
but final candidate-list assembly remains in recognition_component_engine.py. v174z moves the BORE candidate-row assembly body into this module while keeping the final multi-family merge in recognition_component_engine.py. v175g lets BORE consume SecondaryOpeningRimSearchHeuristic proposals as alternate opening evidence/anchor, but the heuristic still cannot create ownership or CandidateData by itself. v175i adds a derived measured-opening context from a strong secondary rim so BORE wall search can be re-run from the discovered bore mouth rather than the originally selected outer CHAMFER rail; BORE_WALL ownership gates still decide promotion.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math
import numpy as np

from ..geometry import to_vector3
from ..topology import boundary_edges_for_face_patch, connected_face_components, edge_loop_components, face_edges
from ..heuristics import (
    BORE_HEURISTIC_RECIPE,
    HEURISTIC_CONTRACT_VERSION,
    compact_heuristic_summary,
    recipe_contracts,
)
from .recognition_common import (
    face_patch_boundary_semantic_report_v174x as _face_patch_boundary_semantic_report_v174z,
    heuristic_results_for_current_bore_state_v174x as _heuristic_results_for_current_bore_state_v174z,
    unit_rows_v174x as _unit_rows_v174z,
)
from ..types import (
    EvidenceKind,
    FeatureFamily,
    FeaturePrimitiveKind,
    FeatureRelationshipKind,
    RecognitionStage,
    tuple_ints,
)

BORE_RECOGNITION_SPLIT_CHECKPOINT_V174H = (
    "v174h_first_bore_module_split_terminal_continuation_filter_no_behavior_change"
)

BORE_RECOGNITION_OWNERSHIP_HELPER_SPLIT_CHECKPOINT_V174I = (
    "v174i_split_bore_wall_subset_and_post_isolation_validation_helpers_no_behavior_change"
)

BORE_RECOGNITION_OPENING_LOCALITY_HELPER_SPLIT_CHECKPOINT_V174J = (
    "v174j_split_selected_opening_locality_and_seed_island_helpers_no_behavior_change"
)

BORE_SECONDARY_OPENING_RECONTEXTUALIZED_MEASUREMENT_CHECKPOINT_V175I = (
    "v175i_secondary_rim_recontextualized_bore_measurement_context_heuristic_evidence_consumed_by_ownership_gates"
)


def filter_bore_terminal_continuation_ownership_v174h(
    *,
    core_bore_wall_owned_face_ids: Iterable[int],
    bore_wall_owned_face_ids: Iterable[int],
    terminal_wall_face_ids: Iterable[int],
    terminal_wall_completion_used: bool,
    endpoint_extension_added_face_count: int,
    terminal_wall_rejection_reasons: Iterable[str],
    terminal_wall_coaxial_audit: Mapping[str, object],
) -> dict[str, object]:
    """Apply the v173w BORE terminal-continuation ownership filter.

    Terminal continuation may remain diagnostic evidence, but it cannot remain
    in BORE_WALL ownership unless its own coaxial/inside-wall audit passed.  This
    helper does not identify a BORE, emit CandidateData, create a
    DeletePatchProposal, or mutate mesh topology.
    """

    core_ids = tuple_ints(core_bore_wall_owned_face_ids)
    bore_ids = tuple_ints(bore_wall_owned_face_ids)
    terminal_ids = tuple_ints(terminal_wall_face_ids)
    audit = dict(terminal_wall_coaxial_audit or {})
    rejection_reasons = [str(v) for v in tuple(terminal_wall_rejection_reasons or ())]
    removed_by_v173w = False
    removed_count_v173w = 0
    removed_reason_v173w = ""
    completion_used = bool(terminal_wall_completion_used)
    extension_added_count = int(endpoint_extension_added_face_count or 0)

    if (
        bool(completion_used)
        and tuple(terminal_ids or ())
        and not bool(audit.get("terminal_continuation_passed_coaxial_audit", False))
    ):
        reason = str(
            audit.get(
                "terminal_continuation_coaxial_audit_reason",
                "terminal_continuation_failed_bore_wall_role_audit",
            )
        )
        removed_by_v173w = True
        removed_count_v173w = int(len(terminal_ids))
        removed_reason_v173w = reason
        bore_ids = tuple_ints(core_ids)
        terminal_ids = ()
        completion_used = False
        extension_added_count = 0
        rejection_reasons.append(
            "v173w_terminal_continuation_not_promoted_to_bore_wall_ownership:" + reason
        )
        audit.update({
            "terminal_continuation_removed_from_bore_ownership_v173w": True,
            "terminal_continuation_removed_face_count_v173w": int(removed_count_v173w),
            "terminal_continuation_removed_reason_v173w": str(reason),
            "terminal_continuation_ownership_rule_v173w": (
                "Terminal continuation is diagnostic evidence only until it passes its own opening-to-exit BORE_WALL audit; "
                "failed terminal faces are not allowed to enter BORE preview, ownership, CandidateData, or DeletePatchProposal."
            ),
        })

    return {
        "bore_wall_owned_face_ids": tuple_ints(bore_ids),
        "terminal_wall_face_ids": tuple_ints(terminal_ids),
        "terminal_wall_completion_used": bool(completion_used),
        "endpoint_extension_added_face_count": int(extension_added_count),
        "terminal_wall_rejection_reasons": tuple(rejection_reasons),
        "terminal_wall_coaxial_audit": audit,
        "terminal_continuation_removed_by_v173w": bool(removed_by_v173w),
        "terminal_continuation_removed_count_v173w": int(removed_count_v173w),
        "terminal_continuation_removed_reason_v173w": str(removed_reason_v173w),
        "v174d_terminal_filter_helper_extracted": True,
        "v174h_terminal_filter_moved_to_recognition_bore": True,
    }


# -----------------------------------------------------------------------------
# v174i BORE wall ownership/helper split
# -----------------------------------------------------------------------------


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _face_patch_boundary_semantic_report_v174i(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Return BORE-local patch boundary evidence without creating ownership.

    This helper is copied from the component engine as a BORE-local utility for
    v174i.  It reports topology evidence used by BORE wall-subset validation; it
    does not promote candidates, create delete patches, or mutate topology.
    """

    ids = tuple_ints(face_ids)
    if not ids:
        return {
            "face_count": 0,
            "boundary_edge_count": 0,
            "boundary_loop_count": 0,
            "boundary_loop_edge_counts": (),
            "boundary_loop_vertex_counts": (),
            "component_count": 0,
            "component_face_counts": (),
        }
    try:
        boundary = boundary_edges_for_face_patch(faces, ids)
        loops = edge_loop_components(boundary)
        comps = connected_face_components(faces, ids)
        loop_vertex_counts = []
        for loop in loops:
            verts = set()
            for edge in loop:
                try:
                    a, b = tuple(edge)[:2]
                    verts.add(int(a))
                    verts.add(int(b))
                except Exception:
                    continue
            loop_vertex_counts.append(int(len(verts)))
        return {
            "face_count": int(len(ids)),
            "boundary_edge_count": int(len(boundary)),
            "boundary_loop_count": int(len(loops)),
            "boundary_loop_edge_counts": tuple(int(len(loop)) for loop in loops),
            "boundary_loop_vertex_counts": tuple(loop_vertex_counts),
            "component_count": int(len(comps)),
            "component_face_counts": tuple(int(len(comp)) for comp in comps),
        }
    except Exception as exc:
        return {
            "face_count": int(len(ids)),
            "boundary_edge_count": 0,
            "boundary_loop_count": 0,
            "boundary_loop_edge_counts": (),
            "boundary_loop_vertex_counts": (),
            "component_count": 0,
            "component_face_counts": (),
            "boundary_report_error": str(exc),
        }


def select_bore_wall_semantic_owned_subset_v174i(
    *,
    faces: np.ndarray,
    owned_face_ids: Iterable[int],
    core_face_ids: Iterable[int],
    terminal_face_ids: Iterable[int],
    damaged_internal_boundary_evidence: bool,
) -> dict[str, object]:
    """Find a rebuildable semantic BORE wall subset when evidence over-selected.

    This is the BORE-local v174i home for the existing v158 helper.  It preserves
    the exact meaning and return keys: repair ownership to a two-boundary-loop
    BORE_WALL subset when possible; do not emit CandidateData, build targets, or
    mutate mesh topology.
    """

    owned_ids = tuple_ints(owned_face_ids)
    core_ids = tuple_ints(core_face_ids)
    terminal_ids = tuple_ints(terminal_face_ids)

    def report_for(ids: Iterable[int]) -> dict[str, object]:
        return _face_patch_boundary_semantic_report_v174i(faces, tuple_ints(ids))

    def role_valid(report: Mapping[str, object]) -> bool:
        loops = int(report.get("boundary_loop_count", 0) or 0)
        comps = int(report.get("component_count", 0) or 0)
        return bool((loops == 2 or damaged_internal_boundary_evidence) and comps >= 1)

    full_report = report_for(owned_ids)
    if role_valid(full_report):
        return {
            "face_ids": owned_ids,
            "removed_face_ids": (),
            "semantic_subset_used_v158": False,
            "semantic_subset_source_v158": "full_owned_wall_patch_already_proves_boundary_role",
            "semantic_subset_report_v158": dict(full_report),
            "semantic_subset_candidate_reports_v158": (),
            "v174i_bore_wall_subset_helper_split": True,
        }

    candidates: list[tuple[int, int, str, tuple[int, ...], dict[str, object]]] = []
    candidate_reports: list[dict[str, object]] = []

    def add_candidate(source: str, ids: Iterable[int]) -> None:
        cand_ids = tuple_ints(ids)
        if not cand_ids:
            return
        rep = report_for(cand_ids)
        row = {
            "source": source,
            "face_count": int(len(cand_ids)),
            "boundary_loop_count": int(rep.get("boundary_loop_count", 0) or 0),
            "boundary_edge_count": int(rep.get("boundary_edge_count", 0) or 0),
            "component_count": int(rep.get("component_count", 0) or 0),
            "boundary_loop_vertex_counts": tuple(rep.get("boundary_loop_vertex_counts", ()) or ()),
            "role_valid": bool(role_valid(rep)),
        }
        candidate_reports.append(row)
        if bool(row["role_valid"]):
            candidates.append((int(len(cand_ids)), -int(rep.get("component_count", 0) or 0), source, cand_ids, rep))

    add_candidate("core_wall_only", core_ids)

    terminal_components = tuple(connected_face_components(np.asarray(faces, dtype=np.int64), terminal_ids)) if terminal_ids else ()
    # Try core + each connected terminal component.  Also try small combinations
    # because v153 terminal completion can produce one valid continuation island
    # plus one exterior/side strip island.
    if core_ids and terminal_components:
        from itertools import combinations
        limited = tuple(tuple_ints(comp) for comp in terminal_components[:6])
        for r in range(1, min(len(limited), 4) + 1):
            for combo in combinations(range(len(limited)), r):
                merged = set(int(v) for v in core_ids)
                label_parts = []
                for idx in combo:
                    merged.update(int(v) for v in limited[idx])
                    label_parts.append(str(idx))
                add_candidate("core_plus_terminal_components_" + "_".join(label_parts), tuple(sorted(merged)))

    for idx, comp in enumerate(connected_face_components(np.asarray(faces, dtype=np.int64), owned_ids)):
        add_candidate(f"owned_connected_component_{idx}", comp)

    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        _, _, source, selected_ids, selected_report = candidates[0]
        removed = tuple_ints(sorted(set(int(v) for v in owned_ids) - set(int(v) for v in selected_ids)))
        return {
            "face_ids": selected_ids,
            "removed_face_ids": removed,
            "semantic_subset_used_v158": True,
            "semantic_subset_source_v158": source,
            "semantic_subset_report_v158": dict(selected_report),
            "semantic_subset_candidate_reports_v158": tuple(candidate_reports),
            "v174i_bore_wall_subset_helper_split": True,
        }

    return {
        "face_ids": owned_ids,
        "removed_face_ids": (),
        "semantic_subset_used_v158": False,
        "semantic_subset_source_v158": "no_boundary_role_valid_subset_found",
        "semantic_subset_report_v158": dict(full_report),
        "semantic_subset_candidate_reports_v158": tuple(candidate_reports),
        "v174i_bore_wall_subset_helper_split": True,
    }


def post_isolation_bore_wall_validation_v174i(
    st: Mapping[str, object],
    *,
    radial_scale: float,
    edge_scale: float,
) -> dict[str, object]:
    """Validate that an isolated seed island is still a BORE wall.

    v174i moves the existing post-isolation BORE_WALL predicate into the
    BORE-local module.  It remains a validation helper only: it does not create
    CandidateData, alter ownership outside its returned diagnostics, or build a
    rebuild target.
    """

    face_count = _safe_int(st.get("face_count", 0), 0)
    radial_median = _safe_float(st.get("radial_median", 0.0), 0.0)
    radial_mad = _safe_float(st.get("radial_mad", 999999.0), 999999.0)
    radial_rel_mad = _safe_float(st.get("radial_rel_mad", 999999.0), 999999.0)
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    normal_axis = _safe_float(st.get("normal_axis_abs_median", 1.0), 1.0)
    normal_axis_q75 = _safe_float(st.get("normal_axis_abs_q75", 1.0), 1.0)
    radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)
    radial_align_q25 = _safe_float(st.get("radial_normal_alignment_q25", 0.0), 0.0)

    local_edge_scale = min(
        max(float(edge_scale), 1.0e-9),
        max(0.30 * max(radial_median, radial_scale, 1.0), 1.0e-9),
    )
    # A measured BORE wall must have axial continuation.  A strip whose depth is
    # only one or two triangulation rows is transition/rim evidence, not wall
    # ownership, even when it touches the selected opening seed.
    min_wall_axial_span = max(
        0.30 * max(radial_median, radial_scale, 1.0),
        2.25 * local_edge_scale,
        0.12,
    )
    max_wall_radial_mad = max(0.70 * local_edge_scale, 0.065 * max(radial_median, 1.0))
    ok = bool(
        face_count >= 5
        and radial_median > 1.0e-9
        and axial_span >= min_wall_axial_span
        and (radial_rel_mad <= 0.085 or radial_mad <= max_wall_radial_mad)
        and normal_axis <= 0.36
        and normal_axis_q75 <= 0.52
        and radial_align >= 0.58
        and radial_align_q25 >= 0.34
    )
    reason = ""
    if not ok:
        if axial_span < min_wall_axial_span:
            reason = "post_isolation_seed_island_is_shallow_transition_strip_not_bore_wall"
        elif radial_rel_mad > 0.085 and radial_mad > max_wall_radial_mad:
            reason = "post_isolation_seed_island_radius_layer_not_compact_enough"
        elif normal_axis > 0.36 or normal_axis_q75 > 0.52 or radial_align < 0.58 or radial_align_q25 < 0.34:
            reason = "post_isolation_seed_island_normals_not_bore_wall_like"
        else:
            reason = "post_isolation_seed_island_failed_bore_wall_validation"
    return {
        "post_isolation_bore_wall_valid": bool(ok),
        "post_isolation_bore_wall_reject_reason": str(reason),
        "post_isolation_face_count": int(face_count),
        "post_isolation_axial_span": float(axial_span),
        "post_isolation_min_wall_axial_span": float(min_wall_axial_span),
        "post_isolation_radial_median": float(radial_median),
        "post_isolation_radial_mad": float(radial_mad),
        "post_isolation_radial_rel_mad": float(radial_rel_mad),
        "post_isolation_max_wall_radial_mad": float(max_wall_radial_mad),
        "post_isolation_normal_axis_abs_median": float(normal_axis),
        "post_isolation_normal_axis_abs_q75": float(normal_axis_q75),
        "post_isolation_radial_normal_alignment_median": float(radial_align),
        "post_isolation_radial_normal_alignment_q25": float(radial_align_q25),
        "v174i_post_isolation_validation_helper_split": True,
    }


# -----------------------------------------------------------------------------
# v174j BORE selected-opening locality / seed-island helper split
# -----------------------------------------------------------------------------


def _percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, float(q)))


def seed_neighborhood_face_ids_v174j(
    *,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    max_depth: int = 4,
    max_faces: int = 900,
) -> tuple[int, ...]:
    """Return a bounded face neighborhood around selected-opening seed faces.

    This is BORE-local Recognition evidence scoping for damaged BORE review. It
    does not classify a feature, emit CandidateData, authorize a delete patch,
    or mutate topology. It only preserves selected-opening locality before BORE
    ownership is promoted by the component engine.
    """

    seeds = tuple(sorted(int(fid) for fid in seed_face_set if int(fid) in adjacency))
    if not seeds:
        return ()
    seen: set[int] = set(seeds)
    frontier: set[int] = set(seeds)
    for _depth in range(max(0, int(max_depth))):
        nxt: set[int] = set()
        for fid in frontier:
            for nb in adjacency.get(int(fid), ()):  # selected-region adjacency only
                nb_i = int(nb)
                if nb_i not in seen:
                    nxt.add(nb_i)
        if not nxt:
            break
        seen.update(nxt)
        if len(seen) >= int(max_faces):
            break
        frontier = nxt
    if len(seen) > int(max_faces):
        # Deterministic bounded output: keep seed-related local IDs first.
        return tuple(sorted(seen)[: int(max_faces)])
    return tuple(sorted(seen))


def bounded_component_from_seed_faces_v174j(
    *,
    component_face_ids: Iterable[int],
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    max_depth: int = 7,
    max_faces: int = 900,
) -> tuple[int, ...]:
    """Return the selected-opening-local island inside a broader component.

    The raw component measurement can identify the selected opening, but broad
    RegionData may still contain a large connected cylindrical component that
    reaches unrelated mesh areas. Candidate ownership may therefore only use the
    bounded neighborhood grown from measured opening seed faces inside the
    candidate component. Neutral RegionData remains evidence, not ownership.
    """

    comp_set = {int(fid) for fid in component_face_ids}
    seeds = tuple(sorted(int(fid) for fid in seed_face_set if int(fid) in comp_set))
    if not seeds:
        return ()
    seen: set[int] = set(seeds)
    frontier: set[int] = set(seeds)
    for _depth in range(max(0, int(max_depth))):
        nxt: set[int] = set()
        for fid in frontier:
            for nb in adjacency.get(int(fid), ()):  # selected-region adjacency only
                nb_i = int(nb)
                if nb_i in comp_set and nb_i not in seen:
                    nxt.add(nb_i)
        if not nxt:
            break
        seen.update(nxt)
        if len(seen) >= int(max_faces):
            break
        frontier = nxt
    if len(seen) > int(max_faces):
        return tuple(sorted(seen)[: int(max_faces)])
    return tuple(sorted(seen))


def cylindrical_seed_island_from_seed_faces_v174j(
    *,
    component_face_ids: Iterable[int],
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    fid_to_local: Mapping[int, int],
    radial: np.ndarray,
    axial: np.ndarray,
    finite: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    opening_anchor: Mapping[str, object],
    edge_scale: float,
    max_depth: int = 9,
    max_faces: int = 520,
) -> tuple[int, ...]:
    """Grow the selected-opening wall island with geometry gating.

    Plain adjacency BFS can walk from the selected bore mouth into neighboring
    or unrelated mesh areas. This helper only crosses faces that stay on the
    measured bore cylinder band and remain plausible wall/defect continuation.
    The measured primary seed island is the authority; broad RegionData and
    expanded measurement fragments are not.
    """

    comp_set = {int(fid) for fid in component_face_ids}
    seeds = tuple(sorted(int(fid) for fid in seed_face_set if int(fid) in comp_set))
    if not seeds:
        return ()

    radius = _safe_float(opening_anchor.get("opening_frame_radius", 0.0), 0.0)
    radius_tol = _safe_float(opening_anchor.get("opening_radius_tolerance", 0.0), 0.0)
    if radius <= 1.0e-9:
        radius = _safe_float(opening_anchor.get("opening_seed_radial_median", 0.0), 0.0)
    # Be deliberately tighter than the generic anchor tolerance. We need a
    # candidate tied to this cylinder, not every face in the neutral cutout.
    radius_tol = min(
        max(radius_tol, 0.0) if radius_tol > 0.0 else max(0.24 * max(radius, 1.0), 3.0 * max(edge_scale, 1.0e-9), 0.45),
        max(0.24 * max(radius, 1.0), 3.5 * max(edge_scale, 1.0e-9), 1.0),
    )
    seed_center = _safe_float(opening_anchor.get("opening_seed_axial_center", 0.0), 0.0)
    hypothesis_depth = _safe_float(opening_anchor.get("opening_hypothesis_depth", 0.0), 0.0)
    axial_limit = max(
        abs(hypothesis_depth) + 0.85 * max(radius, 1.0),
        3.0 * max(radius, 1.0),
        18.0 * max(edge_scale, 1.0e-9),
    )

    def ok(fid: int) -> bool:
        if fid in seeds:
            return True
        local = fid_to_local.get(int(fid))
        if local is None or local < 0 or local >= len(finite) or not bool(finite[local]):
            return False
        radial_error = abs(float(radial[local]) - float(radius)) if radius > 1.0e-9 else 0.0
        if radial_error > max(radius_tol, 1.0e-9):
            return False
        # Bore wall/defect continuation: avoid walking across broad flat/chamfer
        # caps into a neighboring feature. Keep this permissive enough for noisy
        # imported mesh damage.
        if float(normal_axis_abs[local]) > 0.90 and float(radial_normal_alignment[local]) < 0.08:
            return False
        if abs(float(axial[local]) - float(seed_center)) > float(axial_limit):
            return False
        return True

    seen: set[int] = set(seeds)
    frontier: set[int] = set(seeds)
    for _depth in range(max(0, int(max_depth))):
        nxt: set[int] = set()
        for fid in frontier:
            for nb in adjacency.get(int(fid), ()):  # selected-region adjacency only
                nb_i = int(nb)
                if nb_i in comp_set and nb_i not in seen and ok(nb_i):
                    nxt.add(nb_i)
        if not nxt:
            break
        seen.update(nxt)
        if len(seen) >= int(max_faces):
            break
        frontier = nxt
    if len(seen) > int(max_faces):
        return tuple(sorted(seen)[: int(max_faces)])
    return tuple(sorted(seen))


def interval_gap_v174j(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Return zero for overlapping intervals, otherwise the positive gap."""

    lo0, hi0 = (float(a_min), float(a_max)) if float(a_min) <= float(a_max) else (float(a_max), float(a_min))
    lo1, hi1 = (float(b_min), float(b_max)) if float(b_min) <= float(b_max) else (float(b_max), float(b_min))
    if hi0 < lo1:
        return float(lo1 - hi0)
    if hi1 < lo0:
        return float(lo0 - hi1)
    return 0.0


def opening_anchor_reference_v174j(
    *,
    frame: object,
    valid_face_ids: tuple[int, ...],
    seed_face_set: set[int],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    edge_scale: float,
    axial_span_all: float,
    region_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return selected-opening measurement authority for damaged BORE review.

    The selected/measured opening frame is the authority for damaged-bore
    hypothesis construction. RegionData may be broad, but a damaged BORE review
    candidate must remain tied to this opening frame by radius and axial contact;
    otherwise a random cylinder-like fragment in the cutout can masquerade as
    the selected bore. This helper returns evidence/diagnostics only.
    """

    seed_idx = np.asarray([fid_to_local[int(fid)] for fid in seed_face_set if int(fid) in fid_to_local], dtype=np.int64)
    if seed_idx.size:
        seed_axial = axial[seed_idx]
        seed_radial = radial[seed_idx]
        seed_axial = seed_axial[np.isfinite(seed_axial)]
        seed_radial = seed_radial[np.isfinite(seed_radial)]
    else:
        seed_axial = np.asarray((0.0,), dtype=float)
        seed_radial = np.asarray((float(getattr(frame, "radius", 0.0)),), dtype=float)

    frame_radius = float(getattr(frame, "radius", 0.0))
    opening_radius = frame_radius if np.isfinite(frame_radius) and frame_radius > 1.0e-9 else _percentile(radial, 50.0, 1.0)
    median_seed_radial = float(np.median(seed_radial)) if seed_radial.size else float(opening_radius)
    seed_min = float(np.min(seed_axial)) if seed_axial.size else 0.0
    seed_max = float(np.max(seed_axial)) if seed_axial.size else 0.0
    seed_center = float(np.median(seed_axial)) if seed_axial.size else 0.0

    projection_half_depth = _safe_float(region_diagnostics.get("projection_axial_half_depth", 0.0), 0.0)
    projected_depth = float(2.0 * projection_half_depth) if projection_half_depth > 1.0e-9 else 0.0
    # v49: for damaged-bore review the selected opening frame is a hypothesis
    # descriptor, not a rebuild instruction. Prefer the measured RegionData
    # axial span over the projection depth, because projection depth can be a
    # deliberately broad search volume and produced misleading 400+ depth labels
    # on damaged meshes.
    hypothesis_depth = float(axial_span_all) if float(axial_span_all) > 1.0e-9 else float(projected_depth)

    radius_tolerance = max(4.0 * max(float(edge_scale), 1.0e-9), 0.32 * max(float(opening_radius), 1.0), 0.35)
    axial_tolerance = max(6.0 * max(float(edge_scale), 1.0e-9), 0.45 * max(float(opening_radius), 1.0), 0.75)
    return {
        "selected_opening_anchor_available": bool(seed_idx.size > 0),
        "selected_opening_seed_face_count": int(seed_idx.size),
        "opening_frame_center": to_vector3(getattr(frame, "center")),
        "opening_frame_axis": to_vector3(getattr(frame, "axis")),
        "opening_frame_radius": float(opening_radius),
        "opening_frame_diameter": float(2.0 * opening_radius),
        "opening_seed_radial_median": float(median_seed_radial),
        "opening_seed_axial_min": float(seed_min),
        "opening_seed_axial_max": float(seed_max),
        "opening_seed_axial_center": float(seed_center),
        "opening_radius_tolerance": float(radius_tolerance),
        "opening_axial_contact_tolerance": float(axial_tolerance),
        "opening_hypothesis_depth": float(hypothesis_depth),
        "opening_hypothesis_depth_source": "region_projection_axial_depth_or_region_span",
        "v174j_opening_anchor_reference_helper_split": True,
    }


def opening_frame_anchor_metrics_v174j(
    st: Mapping[str, object],
    *,
    opening_anchor: Mapping[str, object],
) -> dict[str, object]:
    """Measure whether a component belongs to the selected opening frame.

    This is a Recognition-stage guard for damaged BORE review. It does not
    create ownership and it does not authorize rebuild. It only prevents the
    damaged path from turning remote cylinder-like fragments into the selected
    bore candidate.
    """

    opening_radius = _safe_float(opening_anchor.get("opening_frame_radius", 0.0), 0.0)
    radius_tolerance = _safe_float(opening_anchor.get("opening_radius_tolerance", 0.0), 0.0)
    axial_tolerance = _safe_float(opening_anchor.get("opening_axial_contact_tolerance", 0.0), 0.0)
    seed_axial_min = _safe_float(opening_anchor.get("opening_seed_axial_min", 0.0), 0.0)
    seed_axial_max = _safe_float(opening_anchor.get("opening_seed_axial_max", 0.0), 0.0)
    radial_median = _safe_float(st.get("radial_median", 0.0), 0.0)
    axial_min = _safe_float(st.get("axial_min", 0.0), 0.0)
    axial_max = _safe_float(st.get("axial_max", 0.0), 0.0)
    radius_error = abs(float(radial_median) - float(opening_radius)) if opening_radius > 1.0e-9 else 0.0
    axial_gap = interval_gap_v174j(axial_min, axial_max, seed_axial_min, seed_axial_max)
    seed_related = bool(st.get("seed_related", False))
    radius_match = bool(opening_radius <= 1.0e-9 or radius_error <= max(radius_tolerance, 1.0e-9))
    axial_contact = bool(seed_related or axial_gap <= max(axial_tolerance, 1.0e-9))
    anchored = bool(radius_match and axial_contact)
    return {
        **dict(opening_anchor),
        "selected_opening_radius_error": float(radius_error),
        "selected_opening_axial_gap": float(axial_gap),
        "selected_opening_radius_match": bool(radius_match),
        "selected_opening_axial_contact": bool(axial_contact),
        "selected_opening_frame_anchored": bool(anchored),
        "candidate_locality_source": "selected_opening_frame" if anchored else "rejected_remote_cylindrical_fragment",
        "v174j_opening_frame_anchor_metrics_helper_split": True,
    }


def selected_opening_evidence_face_ids_v174j(
    *,
    valid_face_ids: tuple[int, ...],
    seed_face_set: set[int],
    seed_neighborhood_ids: tuple[int, ...],
    fid_to_local: Mapping[int, int],
    radial: np.ndarray,
    finite: np.ndarray,
    opening_anchor: Mapping[str, object],
    max_faces: int = 320,
) -> tuple[int, ...]:
    """Return display evidence tied directly to the selected opening.

    Damaged-bore review preview must be physically located at the selected
    opening. Broad RegionData, raw selected-edge pollution, and remote
    cylindrical fragments may remain diagnostics, but they must not become the
    visible damaged BORE candidate. This returns normalized opening seed faces
    first, then only the seed-connected neighborhood that matches the opening
    radius band.
    """

    opening_radius = _safe_float(opening_anchor.get("opening_frame_radius", 0.0), 0.0)
    radius_tol = _safe_float(opening_anchor.get("opening_radius_tolerance", 0.0), 0.0)
    if radius_tol <= 0.0:
        radius_tol = max(0.35 * max(opening_radius, 1.0), 0.35)

    out: list[int] = []
    seen: set[int] = set()

    def add(fid: int) -> None:
        if fid in seen or fid not in fid_to_local:
            return
        local = int(fid_to_local[fid])
        if local < 0 or local >= len(finite) or not bool(finite[local]):
            return
        if opening_radius > 1.0e-9 and abs(float(radial[local]) - float(opening_radius)) > float(radius_tol):
            return
        seen.add(int(fid))
        out.append(int(fid))

    # Seed faces are the only directly selected-opening-local face evidence.
    for fid in sorted(int(v) for v in seed_face_set):
        add(fid)
        if len(out) >= int(max_faces):
            return tuple(out)

    # Add a compact first/near neighborhood only after the selected seed faces.
    # This is display evidence, not completed wall ownership.
    for fid in tuple_ints(seed_neighborhood_ids):
        add(int(fid))
        if len(out) >= int(max_faces):
            break

    return tuple(out)


# -----------------------------------------------------------------------------
# v174k BORE CandidateData construction helpers
# -----------------------------------------------------------------------------

BORE_RECOGNITION_CANDIDATE_CONTRACT_SPLIT_CHECKPOINT_V174K = (
    "v174k_split_bore_candidate_contract_helpers_no_behavior_change"
)

def damaged_bore_reason_flags_v174k(st: Mapping[str, object]) -> tuple[str, ...]:
    """Return human-readable damage/review reasons for bore-like evidence.

    This is not feature identity.  It records why a BORE hypothesis stayed in a
    review/preview state instead of becoming accepted bore-wall ownership.
    """

    reasons: list[str] = []
    face_count = int(st.get("face_count", 0) or 0)
    min_faces = int(st.get("min_bore_faces", 0) or 0)
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    min_axial = _safe_float(st.get("min_bore_axial_span", 0.0), 0.0)
    radial_rel_mad = _safe_float(st.get("radial_rel_mad", 999999.0), 999999.0)
    radial_mad = _safe_float(st.get("radial_mad", 999999.0), 999999.0)
    max_radial_mad = _safe_float(st.get("max_bore_radial_mad", 0.0), 0.0)
    normal_axis_q75 = _safe_float(st.get("normal_axis_abs_q75", 1.0), 1.0)
    radial_align_q25 = _safe_float(st.get("radial_normal_alignment_q25", 0.0), 0.0)

    if min_faces and face_count < min_faces:
        reasons.append("fragmented_or_partial_wall_face_support")
    if min_axial > 1.0e-9 and axial_span < min_axial:
        reasons.append("short_or_interrupted_axial_wall_coverage")
    if radial_rel_mad > 0.085 and (max_radial_mad <= 0.0 or radial_mad > max_radial_mad):
        reasons.append("loose_or_damaged_radius_layer")
    if normal_axis_q75 > 0.52 or radial_align_q25 < 0.34:
        reasons.append("unstable_wall_normal_support")
    if not reasons:
        reasons.append("bore_like_evidence_below_clean_ownership_gate")
    return tuple(reasons)


def damaged_bore_candidate_contract_fields_v174k(
    *,
    candidate_id: str,
    evidence_face_ids: tuple[int, ...],
    confidence: float,
    radius: float,
    axial_span: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return preview/review CandidateData for damaged BORE evidence.

    Damaged BORE is not a new final feature family.  It remains a BORE
    hypothesis with a damage condition, but because surface-role ownership is
    incomplete it stays review/preview-only and exposes no rebuild faces.
    """

    axis_raw = tuple(diagnostics.get("axis", (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
    center_raw = tuple(diagnostics.get("center", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    axis = tuple(float(v) for v in (axis_raw + (0.0, 0.0, 1.0))[:3])
    center = tuple(float(v) for v in (center_raw + (0.0, 0.0, 0.0))[:3])
    damage_reasons_override = tuple(str(v) for v in tuple(diagnostics.get("damage_reasons_override", ()) or ()) if str(v))
    damage_reasons = damage_reasons_override or damaged_bore_reason_flags_v174k(diagnostics)
    damage_state = str(diagnostics.get("damage_state", "") or "").strip() or ("damaged_bore_" + "__".join(damage_reasons[:3]))
    damage_candidate_source = str(diagnostics.get("damaged_candidate_source", "") or "selected_opening_review").strip()
    display_name = str(diagnostics.get("damaged_candidate_display_name", "") or "BORE — damaged bore wall rebuild trial")
    role = str(diagnostics.get("damaged_candidate_role", "") or "damaged_bore_wall_rebuild_trial")
    status = str(diagnostics.get("damaged_candidate_status", "") or "damaged_bore_wall_evidence_rebuild_trial_enabled")
    surface_condition = str(diagnostics.get("damaged_surface_condition", "") or "damaged_bore_wall_owned_faces_with_internal_defect_boundaries")
    repair_strategy = str(diagnostics.get("damaged_repair_strategy", "") or "delete_patch_from_damaged_bore_wall_owned_faces_with_defect_seal_trial_validation")
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "borehole",
        "feature_kind": "borehole",
        "candidate_scope": "damaged_bore_candidate_data",
        "candidate_variant": "damaged_bore",
        "damage_state": damage_state,
        "damage_reasons": damage_reasons,
        "candidate_authority": "surface_component_classifier_v70_review_only_quarantined_damaged_bore_evidence",
        "active_candidate_authority": "surface_component_classifier_v70_review_only_quarantined_damaged_bore_evidence",
        "feature_family": FeatureFamily.BORE.value,
        "recognition_stage": RecognitionStage.REVIEW.value,
        "display_name": display_name,
        "role": role,
        "status": status,
        "candidate_side_role": "raw_selected_edge_damaged_bore_evidence" if bool(diagnostics.get("selected_opening_frame_anchored", False)) else "independent_damaged_cylindrical_bore_evidence",
        "selection_seed_related": bool(diagnostics.get("seed_related", False)),
        "selected_opening_frame_anchored": bool(diagnostics.get("selected_opening_frame_anchored", False)),
        "selected_opening_radius_error": _safe_float(diagnostics.get("selected_opening_radius_error", 0.0), 0.0),
        "selected_opening_axial_gap": _safe_float(diagnostics.get("selected_opening_axial_gap", 0.0), 0.0),
        "promotion_state": "review",
        "candidate_action_enabled": False,
        "candidate_action": "preview_only",
        "rebuild_authorized": False,
        "rebuild_gate": "damaged_bore_wall_candidate",
        "face_ids": evidence_face_ids,
        "semantic_face_ids": evidence_face_ids,
        "display_face_ids": evidence_face_ids,
        "preview_face_ids": evidence_face_ids,
        "rebuild_face_ids": (),
        "owned_face_ids": evidence_face_ids,
        "evidence_face_ids": evidence_face_ids,
        "face_count": int(len(evidence_face_ids)),
        "confidence": float(confidence),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "depth": float(axial_span),
        "height": float(axial_span),
        "axial_span": float(axial_span),
        "primitive_axis": axis,
        "primitive_radius": float(radius),
        "primitive_depth": float(axial_span),
        "surface_condition": surface_condition,
        "repair_strategy": repair_strategy,
        "damage_profile": {
            "damage_state": damage_state,
            "damage_reasons": damage_reasons,
            "face_count": int(len(evidence_face_ids)),
            "axial_span": float(axial_span),
            "radius": float(radius),
            "radial_rel_mad": _safe_float(diagnostics.get("radial_rel_mad", 0.0), 0.0),
            "normal_axis_abs_median": _safe_float(diagnostics.get("normal_axis_abs_median", 0.0), 0.0),
            "radial_normal_alignment_median": _safe_float(diagnostics.get("radial_normal_alignment_median", 0.0), 0.0),
            "clean_ownership_gate_passed": True,
            "selected_opening_frame_anchored": bool(diagnostics.get("selected_opening_frame_anchored", False)),
            "candidate_source": damage_candidate_source,
            "connected_opening_component_preview": bool(diagnostics.get("damaged_anchor_connected_opening_component_preview", False)),
            "derived_from_accepted_bore_wall_candidate": bool(diagnostics.get("derived_from_accepted_bore_wall_candidate", False)),
            "accepted_bore_wall_face_count": int(diagnostics.get("accepted_bore_wall_face_count", 0) or 0),
            "boundary_loop_count": int(diagnostics.get("boundary_loop_count", 0) or 0),
            "remote_cylindrical_components_suppressed": int(diagnostics.get("remote_cylindrical_components_suppressed", 0) or 0),
            "opening_frame_radius": _safe_float(diagnostics.get("opening_frame_radius", radius), radius),
            "opening_hypothesis_depth": _safe_float(diagnostics.get("opening_hypothesis_depth", axial_span), axial_span),
            "evidence_axial_span": _safe_float(diagnostics.get("evidence_axial_span", axial_span), axial_span),
        },
        "evidence_kinds": (
            EvidenceKind.OPENING_RING.value,
            EvidenceKind.BORE_WALL_NORMALS.value,
            EvidenceKind.RADIUS_CONSISTENCY.value,
        ),
        "promotion_reasons": (
            "damaged_bore_review_candidate_created",
            "selected_opening_frame_anchored",
            "selected_opening_evidence_display_only",
            "bore_like_cylindrical_wall_evidence",
            "radius_layer_evidence_present",
            "damaged_bore_wall_rebuild_faces_exposed_for_trial_validation",
            "accepted_bore_wall_evidence_reused_for_damage_review" if bool(diagnostics.get("derived_from_accepted_bore_wall_candidate", False)) else "selected_opening_evidence_review",
        ),
        "rejection_reasons": (
            "damaged_bore_requires_defect_seal_trial_validation",
        ),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.CYLINDER_AXIS.value,
                "source": "recognition_component_engine.v91_heuristic_contract_registry_full_depth_wall_ownership",
                "role": "damaged_bore_rebuild_trial_descriptor_not_cad_body",
                "center": center,
                "axis": axis,
                "radius": float(radius),
                "diameter": float(2.0 * radius),
                "depth": float(axial_span),
                "confidence": float(confidence),
                "face_ids": evidence_face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native damaged-bore cylinder descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": False,
        "rebuild_target_policy_allowed": False,
        "damaged_bore_rebuild_trial_enabled": False,
        "requires_damaged_bore_target_seal": True,
        "rebuild_target_policy_reason": "quarantined damaged evidence is review-only; only v70 two-opening BoreWallOwnership may request DeletePatchProposal",
        "rebuild_block_reason": "damaged_evidence_review_only_until_two_opening_bore_wall_ownership",
        "recognition_rule": "v70_quarantined_legacy_damaged_bore_evidence_review_only",
        "feature_ownership_source": "quarantined_legacy_damaged_path_not_bore_wall_ownership",
        "feature_ownership_split": "no_ownership_review_only",
        "bore_rebuild_enable_scope": "disabled_legacy_path",
        "regiondata_rebuild_fallback_allowed": False,
        "diagnostics": dict(diagnostics),
    }


def bore_candidate_contract_fields_v174k(
    *,
    candidate_id: str,
    face_ids: tuple[int, ...],
    confidence: float,
    radius: float,
    axial_span: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return accepted CandidateData for one owned BORE wall surface.

    This function creates BORE feature meaning and surface-role ownership, then
    exposes only those owned wall faces as explicit rebuild_face_ids.  RegionData
    is still not rebuild input; rebuild_target.py must still construct a bounded
    DeletePatchProposal, and rebuild.py must still pass measured-loop watertight
    trial validation.
    """

    axis_raw = tuple(diagnostics.get("axis", (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
    center_raw = tuple(diagnostics.get("center", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    axis = tuple(float(v) for v in (axis_raw + (0.0, 0.0, 1.0))[:3])
    center = tuple(float(v) for v in (center_raw + (0.0, 0.0, 0.0))[:3])
    operational_depth = _safe_float(diagnostics.get("owned_bore_frame_depth", diagnostics.get("operational_bore_frame_depth", axial_span)), axial_span)
    if not math.isfinite(float(operational_depth)) or float(operational_depth) <= 1.0e-9:
        operational_depth = float(axial_span)
    measured_frame_depth = _safe_float(diagnostics.get("measured_two_opening_frame_depth", diagnostics.get("two_opening_bore_frame_depth", axial_span)), axial_span)
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "borehole",
        "feature_kind": "borehole",
        "candidate_scope": "candidate_data",
        "candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "family_local_ownership_source_guard_v173m": True,
        "family_local_ownership_source_rule_v173m": "BORE CandidateData exports only BORE_WALL ownership; selected-rail transition-band ownership is CHAMFER-only relationship/local-band evidence",
        "feature_family": FeatureFamily.BORE.value,
        "recognition_stage": RecognitionStage.ACCEPTED_CANDIDATE.value,
        "display_name": "BORE — cylindrical wall rebuild trial",
        "role": "rebuildable_bore_wall_operation",
        "status": "bore_wall_ownership_detected_rebuild_trial_enabled",
        "candidate_side_role": "selected_opening_bore_wall" if bool(diagnostics.get("seed_related", False)) else "independent_cylindrical_bore_wall",
        "selection_seed_related": bool(diagnostics.get("seed_related", False)),
        "promotion_state": "promoted",
        "candidate_action_enabled": True,
        "candidate_action": "rebuild",
        "rebuild_authorized": True,
        "rebuild_gate": "promoted_bore_wall_candidate",
        "face_ids": face_ids,
        "semantic_face_ids": face_ids,
        "display_face_ids": face_ids,
        "preview_face_ids": face_ids,
        "rebuild_face_ids": face_ids,
        "face_count": int(len(face_ids)),
        "measured_two_opening_frame_depth": float(measured_frame_depth),
        "owned_bore_frame_depth": float(operational_depth),
        "owned_bore_frame_depth_delta": float(float(operational_depth) - float(measured_frame_depth)),
        "bore_frame_depth_source": str(diagnostics.get("bore_frame_depth_source", "two_opening_measured_frame")),
        "confidence": float(confidence),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "depth": float(operational_depth),
        "height": float(operational_depth),
        "axial_span": float(operational_depth),
        "primitive_axis": axis,
        "primitive_radius": float(radius),
        "primitive_depth": float(operational_depth),
        "surface_condition": "cylindrical_bore_wall_owned_faces_only",
        "repair_strategy": "delete_patch_from_bore_wall_owned_faces_trial_validated",
        "evidence_kinds": (
            EvidenceKind.OPENING_RING.value,
            EvidenceKind.BORE_WALL_NORMALS.value,
            EvidenceKind.RADIUS_CONSISTENCY.value,
        ),
        "promotion_reasons": (
            "clean_bore_preview_recognition",
            "cylindrical_wall_surface_role",
            "radius_layer_compactness",
            "wall_normal_alignment",
            "axial_continuation_evidence",
            "explicit_bore_wall_rebuild_faces_exposed_for_trial_validation",
        ),
        "rejection_reasons": (),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.CYLINDER_AXIS.value,
                "source": "recognition_component_engine.v91_heuristic_contract_registry_full_depth_wall_ownership",
                "role": "accepted_physical_primitive_descriptor",
                "center": center,
                "axis": axis,
                "radius": float(radius),
                "diameter": float(2.0 * radius),
                "depth": float(operational_depth),
                "confidence": float(confidence),
                "face_ids": face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native cylinder descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": True,
        "rebuild_target_policy_allowed": True,
        "damaged_bore_rebuild_trial_enabled": True,
        "requires_damaged_bore_target_seal": True,
        "rebuild_target_policy_reason": "accepted BORE wall CandidateData may request DeletePatchProposal from explicit rebuild_face_ids; rebuild.py still performs measured-loop watertight trial validation",
        "rebuild_block_reason": "",
        "recognition_rule": "v91_heuristic_contract_registry_full_depth_wall_ownership",
        "feature_ownership_source": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "feature_ownership_source_family_guard_v173m": "bore_wall_ownership_only_no_selected_rail_transition_band_authority",
        "feature_ownership_split": "bore_wall_owned_face_ids_to_explicit_rebuild_face_ids_no_regiondata_fallback",
        "bore_rebuild_enable_scope": "candidate_owned_bore_wall_faces_only",
        "regiondata_rebuild_fallback_allowed": False,
        "diagnostics": dict(diagnostics),
    }


def bore_review_candidate_contract_fields_v174k(
    *,
    candidate_id: str,
    face_ids: tuple[int, ...],
    confidence: float,
    radius: float,
    axial_span: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return preview-only BORE evidence when measurement exists but ownership fails.

    This is deliberately not CandidateData authority for rebuild.  It exposes the
    measured two-opening frame / wall-search evidence so the operator can inspect
    the selected BORE hypothesis without allowing DeletePatchProposal creation.
    """

    axis_raw = tuple(diagnostics.get("axis", (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
    center_raw = tuple(diagnostics.get("center", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    axis = tuple(float(v) for v in (axis_raw + (0.0, 0.0, 1.0))[:3])
    center = tuple(float(v) for v in (center_raw + (0.0, 0.0, 0.0))[:3])
    reason = str(diagnostics.get("bore_wall_ownership_reject_reason", "bore_wall_ownership_unresolved") or "bore_wall_ownership_unresolved")
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "borehole",
        "feature_kind": "borehole",
        "candidate_scope": "candidate_data_review_only",
        "candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "family_local_ownership_source_guard_v173m": True,
        "family_local_ownership_source_rule_v173m": "BORE CandidateData exports only BORE_WALL ownership; selected-rail transition-band ownership is CHAMFER-only relationship/local-band evidence",
        "feature_family": FeatureFamily.BORE.value,
        "recognition_stage": RecognitionStage.REVIEW.value,
        "display_name": "BORE — measured frame, wall ownership unresolved",
        "role": "two_opening_bore_review_only",
        "status": "two_opening_bore_frame_valid_but_wall_ownership_failed",
        "candidate_side_role": "selected_opening_to_opposite_opening_bore_wall_review",
        "selection_seed_related": bool(diagnostics.get("seed_related", False)),
        "promotion_state": "evidence_only",
        "candidate_action_enabled": False,
        "candidate_action": "preview",
        "rebuild_authorized": False,
        "rebuild_gate": "review_only_two_opening_frame_no_wall_ownership",
        "face_ids": face_ids,
        "semantic_face_ids": face_ids,
        "display_face_ids": face_ids,
        "preview_face_ids": face_ids,
        "rebuild_face_ids": (),
        "face_count": int(len(face_ids)),
        "confidence": float(confidence),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "depth": float(axial_span),
        "height": float(axial_span),
        "axial_span": float(axial_span),
        "primitive_axis": axis,
        "primitive_radius": float(radius),
        "primitive_depth": float(axial_span),
        "surface_condition": "two_opening_bore_wall_not_owned_yet",
        "repair_strategy": "none_review_only_until_bore_wall_ownership_valid",
        "evidence_kinds": (
            EvidenceKind.OPENING_RING.value,
            EvidenceKind.OPPOSITE_OPENING.value,
            EvidenceKind.BORE_WALL_NORMALS.value,
            EvidenceKind.RADIUS_CONSISTENCY.value,
        ),
        "promotion_reasons": (
            "selected_opening_measured",
            "opposite_opening_measured",
            "two_opening_bore_frame_measured",
            "bore_wall_search_boundary_visible_for_review",
        ),
        "rejection_reasons": (reason,),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.CYLINDER_AXIS.value,
                "source": "recognition_component_engine.v91_heuristic_contract_registry_full_depth_wall_ownership",
                "role": "measured_two_opening_bore_frame_review_descriptor",
                "center": center,
                "axis": axis,
                "radius": float(radius),
                "diameter": float(2.0 * radius),
                "depth": float(axial_span),
                "confidence": float(confidence),
                "face_ids": face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native two-opening BORE review descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": False,
        "rebuild_target_policy_allowed": False,
        "damaged_bore_rebuild_trial_enabled": False,
        "requires_damaged_bore_target_seal": False,
        "rebuild_target_policy_reason": "review-only two-opening BORE evidence cannot request DeletePatchProposal until BoreWallOwnership is valid",
        "rebuild_block_reason": reason,
        "recognition_rule": "v91_heuristic_contract_registry_full_depth_wall_ownership",
        "feature_ownership_source": "two_opening_measurement_without_valid_wall_ownership",
        "feature_ownership_split": "no_rebuild_ownership_review_only",
        "bore_rebuild_enable_scope": "disabled_until_bore_wall_ownership_valid",
        "regiondata_rebuild_fallback_allowed": False,
        "diagnostics": dict(diagnostics),
    }


def bore_selected_opening_review_candidate_contract_fields_v174k(
    *,
    candidate_id: str,
    face_ids: tuple[int, ...],
    confidence: float,
    radius: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return preview-only BORE opening evidence.

    v1.5.2: a measured selected opening is visible as BORE-family evidence even
    when the opposite opening or wall ownership stage is not yet accepted.  This
    prevents CHAMFER or broad RegionData rows from hiding the user's selected
    BORE evidence, while still forbidding rebuild until BoreWallOwnership is
    valid.
    """

    axis_raw = tuple(diagnostics.get("axis", (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
    center_raw = tuple(diagnostics.get("center", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    axis = tuple(float(v) for v in (axis_raw + (0.0, 0.0, 1.0))[:3])
    center = tuple(float(v) for v in (center_raw + (0.0, 0.0, 0.0))[:3])
    reason = str(diagnostics.get("bore_wall_ownership_reject_reason", "selected_opening_measured_waiting_for_opposite_opening_or_wall_ownership") or "selected_opening_measured_waiting_for_opposite_opening_or_wall_ownership")
    radius_min = _safe_float(diagnostics.get("radius_min", radius), radius)
    radius_max = _safe_float(diagnostics.get("radius_max", radius), radius)
    diameter_min = _safe_float(diagnostics.get("diameter_min", 2.0 * radius_min), 2.0 * radius_min)
    diameter_max = _safe_float(diagnostics.get("diameter_max", 2.0 * radius_max), 2.0 * radius_max)
    radial_spread = _safe_float(diagnostics.get("radial_spread", max(0.0, radius_max - radius_min)), max(0.0, radius_max - radius_min))
    radial_spread_ratio = _safe_float(diagnostics.get("radial_spread_ratio", radial_spread / max(float(radius), 1.0e-12)), radial_spread / max(float(radius), 1.0e-12))
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "borehole",
        "feature_kind": "borehole",
        "candidate_scope": "candidate_data_review_only",
        "candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "family_local_ownership_source_guard_v173m": True,
        "family_local_ownership_source_rule_v173m": "BORE CandidateData exports only BORE_WALL ownership; selected-rail transition-band ownership is CHAMFER-only relationship/local-band evidence",
        "feature_family": FeatureFamily.BORE.value,
        "recognition_stage": RecognitionStage.REVIEW.value,
        "display_name": "BORE — selected opening measured",
        "role": "selected_opening_bore_review_only",
        "status": "selected_opening_measured_waiting_for_two_opening_frame_or_wall_ownership",
        "candidate_side_role": "selected_opening_bore_evidence",
        "selection_seed_related": True,
        "promotion_state": "evidence_only",
        "candidate_action_enabled": False,
        "candidate_action": "preview",
        "rebuild_authorized": False,
        "rebuild_gate": "review_only_selected_opening_no_wall_ownership",
        "face_ids": face_ids,
        "semantic_face_ids": face_ids,
        "display_face_ids": face_ids,
        "preview_face_ids": face_ids,
        "rebuild_face_ids": (),
        "face_count": int(len(face_ids)),
        "confidence": float(confidence),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "radius_min": float(radius_min),
        "radius_nominal": float(radius),
        "radius_max": float(radius_max),
        "diameter_min": float(diameter_min),
        "diameter_nominal": float(2.0 * radius),
        "diameter_max": float(diameter_max),
        "radial_spread": float(radial_spread),
        "radial_spread_ratio": float(radial_spread_ratio),
        "depth": 0.0,
        "height": 0.0,
        "axial_span": 0.0,
        "primitive_axis": axis,
        "primitive_radius": float(radius),
        "primitive_radius_min": float(radius_min),
        "primitive_radius_max": float(radius_max),
        "primitive_radial_spread_ratio": float(radial_spread_ratio),
        "primitive_depth": 0.0,
        "surface_condition": "selected_opening_evidence_only",
        "repair_strategy": "none_review_only_until_two_opening_wall_ownership_valid",
        "evidence_kinds": (EvidenceKind.OPENING_RING.value,),
        "promotion_reasons": (
            "selected_opening_measured",
            "bore_family_opening_evidence_visible",
            "review_only_until_opposite_opening_and_wall_ownership",
        ),
        "rejection_reasons": (reason,),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.CIRCULAR_OPENING.value,
                "source": "recognition_component_engine.v91_heuristic_contract_registry_full_depth_wall_ownership",
                "role": "measured_selected_opening_review_descriptor",
                "center": center,
                "axis": axis,
                "radius": float(radius),
                "diameter": float(2.0 * radius),
                "radius_min": float(radius_min),
                "radius_max": float(radius_max),
                "diameter_min": float(diameter_min),
                "diameter_max": float(diameter_max),
                "radial_spread": float(radial_spread),
                "radial_spread_ratio": float(radial_spread_ratio),
                "depth": 0.0,
                "confidence": float(confidence),
                "face_ids": face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native selected-opening BORE review descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": False,
        "rebuild_target_policy_allowed": False,
        "damaged_bore_rebuild_trial_enabled": False,
        "requires_damaged_bore_target_seal": False,
        "rebuild_target_policy_reason": "selected opening evidence cannot request DeletePatchProposal until a two-opening BoreWallOwnership is valid",
        "rebuild_block_reason": reason,
        "recognition_rule": "v79_selected_opening_or_no_sidewall_bore_review",
        "feature_ownership_source": "selected_opening_measurement_without_valid_bore_wall_ownership",
        "feature_ownership_split": "no_rebuild_ownership_review_only",
        "bore_rebuild_enable_scope": "disabled_until_bore_wall_ownership_valid",
        "regiondata_rebuild_fallback_allowed": False,
        "diagnostics": dict(diagnostics),
    }




BORE_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Z = (
    "v174z_split_bore_candidate_assembly_from_component_engine_no_behavior_change"
)



BORE_SECONDARY_OPENING_RIM_CONSUMPTION_CHECKPOINT_V175G = (
    "v175g_bore_consumes_secondary_opening_rim_heuristic_as_alternate_evidence_only"
)


def _best_secondary_opening_rim_proposal_v175g(
    secondary_opening_rim_heuristic_diag: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the best secondary rim proposal without creating BORE meaning.

    The proposal is still heuristic evidence.  It becomes useful to BORE only as
    alternate opening/anchor evidence and only if the normal BORE wall-ownership
    gates pass later.
    """

    diag = dict(secondary_opening_rim_heuristic_diag or {})
    rows = tuple(diag.get("secondary_opening_rim_candidates_v175f", ()) or ())
    best: dict[str, object] = {}
    best_score = -1.0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("is_secondary_to_selected_rail", False)):
            continue
        radius = _safe_float(row.get("radius", 0.0), 0.0)
        axis_dot = _safe_float(row.get("axis_abs_dot_to_selected", 0.0), 0.0)
        circularity = _safe_float(row.get("circularity", 0.0), 0.0)
        coverage = _safe_float(row.get("angular_coverage", 0.0), 0.0)
        score = _safe_float(row.get("score", 0.0), 0.0)
        if radius <= 1.0e-9 or axis_dot < 0.92 or circularity < 0.50 or coverage < 0.45:
            continue
        if score > best_score:
            best_score = float(score)
            best = dict(row)
    return best


def _secondary_opening_rim_anchor_face_ids_v175g(
    *,
    faces: np.ndarray,
    valid_face_ids: Iterable[int],
    proposal: Mapping[str, object],
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Return faces adjacent to a secondary rim proposal edge sample.

    This does not own the faces.  It only gives BORE wall ownership a local
    physical anchor other than the selected outer CHAMFER rail.  Ownership still
    requires the later BORE wall checks to pass.
    """

    support_edges_raw = tuple(proposal.get("support_edge_keys_sample", ()) or ())
    support_edges: set[tuple[int, int]] = set()
    for edge in support_edges_raw:
        try:
            a, b = tuple(edge)[:2]
            ia, ib = int(a), int(b)
            if ia == ib:
                continue
            support_edges.add((ia, ib) if ia < ib else (ib, ia))
        except Exception:
            continue
    if not support_edges:
        return (), {
            "secondary_opening_rim_anchor_used_v175g": False,
            "secondary_opening_rim_anchor_reason_v175g": "no_support_edges_in_proposal",
        }

    arr = np.asarray(faces, dtype=np.int64)
    valid = tuple_ints(valid_face_ids)
    out: list[int] = []
    for fid in valid:
        if int(fid) < 0 or int(fid) >= len(arr):
            continue
        try:
            f_edges = set(face_edges(arr[int(fid), :3]))
        except Exception:
            continue
        if bool(f_edges & support_edges):
            out.append(int(fid))
    ids = tuple(sorted(set(out)))
    return ids, {
        "secondary_opening_rim_anchor_used_v175g": bool(ids),
        "secondary_opening_rim_anchor_face_count_v175g": int(len(ids)),
        "secondary_opening_rim_anchor_edge_count_v175g": int(len(support_edges)),
        "secondary_opening_rim_anchor_proposal_relation_v175g": str(proposal.get("relation_to_selected_rail", "")),
        "secondary_opening_rim_anchor_proposal_radius_v175g": float(_safe_float(proposal.get("radius", 0.0), 0.0)),
        "secondary_opening_rim_anchor_proposal_score_v175g": float(_safe_float(proposal.get("score", 0.0), 0.0)),
        "secondary_opening_rim_anchor_semantic_rule_v175g": (
            "heuristic OpeningRimEvidenceProposal may seed/anchor BORE wall search but cannot own faces or emit CandidateData"
        ),
    }


def _secondary_rim_recontextualized_bore_frame_v175i(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_centroids: np.ndarray,
    valid_face_ids: Iterable[int],
    proposal: Mapping[str, object],
    selected_opening_support_weak: bool,
    secondary_radius_authority_used: bool,
    current_bore_depth: float,
    current_bore_radius: float,
    edge_scale: float,
) -> dict[str, object]:
    """Build a derived BORE opening context from a secondary rim proposal.

    This is not feature identity and not BORE_WALL ownership.  It is the
    measured-context consumption step after the heuristic proposal: a strong
    inner rim can replace the originally selected outer CHAMFER rail as the
    opening anchor used by the normal BORE wall-ownership gates.
    """

    diag: dict[str, object] = {
        "bore_secondary_opening_recontextualized_checkpoint_v175i": BORE_SECONDARY_OPENING_RECONTEXTUALIZED_MEASUREMENT_CHECKPOINT_V175I,
        "bore_secondary_opening_recontextualized_used_v175i": False,
        "bore_secondary_opening_recontextualized_authority_v175i": "measured_opening_context_only_not_feature_identity_not_surface_ownership",
        "bore_secondary_opening_recontextualized_reason_v175i": "not_evaluated",
    }
    if not proposal:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "no_secondary_opening_rim_proposal"
        return diag
    if not bool(selected_opening_support_weak):
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "primary_selected_opening_not_weak"
        return diag
    if not bool(secondary_radius_authority_used):
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "secondary_radius_authority_not_used"
        return diag

    radius = _safe_float(proposal.get("radius", 0.0), 0.0)
    edge_count = _safe_int(proposal.get("edge_count", 0), 0)
    axis_dot = _safe_float(proposal.get("axis_abs_dot_to_selected", 0.0), 0.0)
    circularity = _safe_float(proposal.get("circularity", 0.0), 0.0)
    angular_coverage = _safe_float(proposal.get("angular_coverage", 0.0), 0.0)
    if radius <= 1.0e-9 or edge_count < 8 or axis_dot < 0.92 or circularity < 0.50 or angular_coverage < 0.45:
        diag.update({
            "bore_secondary_opening_recontextualized_reason_v175i": "secondary_rim_proposal_failed_measurement_quality_gate",
            "bore_secondary_opening_recontextualized_radius_v175i": float(radius),
            "bore_secondary_opening_recontextualized_edge_count_v175i": int(edge_count),
            "bore_secondary_opening_recontextualized_axis_dot_v175i": float(axis_dot),
            "bore_secondary_opening_recontextualized_circularity_v175i": float(circularity),
            "bore_secondary_opening_recontextualized_angular_coverage_v175i": float(angular_coverage),
        })
        return diag

    try:
        center = np.asarray(proposal.get("center", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)[:3]
        axis = np.asarray(proposal.get("axis", (0.0, 0.0, 1.0)), dtype=float).reshape(-1)[:3]
    except Exception as exc:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = f"proposal_frame_parse_failed:{exc}"
        return diag
    if center.size < 3 or axis.size < 3 or not np.all(np.isfinite(center)) or not np.all(np.isfinite(axis)):
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "proposal_frame_invalid"
        return diag
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "proposal_axis_degenerate"
        return diag
    axis = axis / axis_norm

    ids = tuple_ints(valid_face_ids)
    if not ids:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "no_valid_region_faces"
        return diag
    arr_faces = np.asarray(faces, dtype=np.int64)
    arr_vertices = np.asarray(vertices, dtype=float)
    try:
        tri_vids = arr_faces[np.asarray(ids, dtype=np.int64), :3]
        tri_pts = arr_vertices[tri_vids, :3]
        tri_rel = tri_pts - center.reshape(1, 1, 3)
        tri_ax = tri_rel @ axis.reshape(3)
        finite_ax = tri_ax[np.isfinite(tri_ax)]
        centroid_ax = (np.asarray(face_centroids, dtype=float)[np.asarray(ids, dtype=np.int64), :3] - center.reshape(1, 3)) @ axis.reshape(3)
        centroid_ax = centroid_ax[np.isfinite(centroid_ax)]
    except Exception as exc:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = f"region_projection_failed:{exc}"
        return diag
    if finite_ax.size < 3:
        diag["bore_secondary_opening_recontextualized_reason_v175i"] = "insufficient_projected_region_support"
        return diag

    # Orient the derived axis so the discovered rim is the local opening and the
    # wall/feature volume extends mostly into positive axial coordinates.
    neg_span = abs(float(np.nanpercentile(finite_ax, 1.0)))
    pos_span = abs(float(np.nanpercentile(finite_ax, 99.0)))
    axis_flipped = False
    if neg_span > pos_span:
        axis = -axis
        finite_ax = -finite_ax
        centroid_ax = -centroid_ax
        axis_flipped = True

    positive = finite_ax[np.isfinite(finite_ax) & (finite_ax >= -max(3.0 * float(edge_scale), 0.75))]
    if positive.size < 3:
        positive = finite_ax[np.isfinite(finite_ax)]
    depth = float(max(0.0, np.nanpercentile(positive, 99.5))) if positive.size else 0.0
    max_depth = float(max(0.0, np.nanmax(positive))) if positive.size else 0.0
    # Keep the percentile depth stable but do not let it collapse below the
    # existing provisional frame when the AOI clearly extends farther.
    if max_depth > depth + max(4.0 * float(edge_scale), 1.0):
        depth = max(depth, min(max_depth, depth + max(16.0 * float(edge_scale), 0.25 * max(max_depth, 1.0))))
    if depth <= max(0.50, 2.5 * float(edge_scale), 0.35 * float(radius)):
        diag.update({
            "bore_secondary_opening_recontextualized_reason_v175i": "derived_depth_too_small",
            "bore_secondary_opening_recontextualized_depth_v175i": float(depth),
        })
        return diag

    current_depth = float(max(_safe_float(current_bore_depth, 0.0), 0.0))
    depth_gain = float(depth / max(current_depth, 1.0e-9)) if current_depth > 1.0e-9 else 999999.0
    diag.update({
        "bore_secondary_opening_recontextualized_used_v175i": True,
        "bore_secondary_opening_recontextualized_reason_v175i": "strong_secondary_rim_becomes_derived_opening_context_for_bore_wall_search",
        "bore_secondary_opening_recontextualized_semantic_stage_v175i": "OpeningRimEvidenceProposal -> DerivedMeasuredOpeningContext -> BORE wall ownership gates",
        "bore_secondary_opening_recontextualized_radius_v175i": float(radius),
        "bore_secondary_opening_recontextualized_previous_radius_v175i": float(current_bore_radius),
        "bore_secondary_opening_recontextualized_center_v175i": tuple(float(v) for v in center[:3]),
        "bore_secondary_opening_recontextualized_axis_v175i": tuple(float(v) for v in axis[:3]),
        "bore_secondary_opening_recontextualized_axis_flipped_v175i": bool(axis_flipped),
        "bore_secondary_opening_recontextualized_axial_min_v175i": 0.0,
        "bore_secondary_opening_recontextualized_axial_max_v175i": float(depth),
        "bore_secondary_opening_recontextualized_depth_v175i": float(depth),
        "bore_secondary_opening_recontextualized_previous_depth_v175i": float(current_depth),
        "bore_secondary_opening_recontextualized_depth_gain_v175i": float(depth_gain),
        "bore_secondary_opening_recontextualized_region_projected_min_v175i": float(np.nanmin(finite_ax)),
        "bore_secondary_opening_recontextualized_region_projected_max_v175i": float(np.nanmax(finite_ax)),
        "bore_secondary_opening_recontextualized_centroid_min_v175i": float(np.nanmin(centroid_ax)) if centroid_ax.size else 0.0,
        "bore_secondary_opening_recontextualized_centroid_max_v175i": float(np.nanmax(centroid_ax)) if centroid_ax.size else 0.0,
        "bore_secondary_opening_recontextualized_not_ownership_v175i": True,
        "bore_secondary_opening_recontextualized_not_candidate_data_v175i": True,
    })
    return diag


def assemble_bore_candidate_rows_v174z(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    valid_face_ids: Iterable[int],
    seed_face_set: set[int],
    selected_seed_face_set: set[int],
    adjacency: Mapping[int, Iterable[int]],
    kwargs: Mapping[str, object],
    region_diagnostics: Mapping[str, object],
    selected_opening_measurement_audit: Mapping[str, object],
    selected_opening_frame_resolver: Mapping[str, object],
    two_opening_bore_frame: Mapping[str, object],
    selected_edge_count: int,
    normalized_edge_count: int,
    region_face_count: int,
    broad_or_ambiguous_selection: bool,
    bore_center: np.ndarray,
    bore_axis: np.ndarray,
    bore_radius: float,
    bore_depth: float,
    bore_axial_min: float,
    bore_axial_max: float,
    bore_frame_reason: str,
    two_opening_valid: bool,
    two_opening_axis_direction_preserved: bool,
    two_opening_axis_dot_opening_to_opposite: float,
    region_frame: object,
    raw_region_edge_scale: float,
    edge_scale: float,
    secondary_opening_rim_heuristic_diag: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble BORE CandidateData rows and BORE-local diagnostics.

    v174z moves the BORE candidate assembly body out of
    ``recognition_component_engine.py`` without changing behavior.  The
    component engine remains the multi-family orchestrator and final candidate
    list merger; this helper owns only BORE-family row assembly state.
    """
    # ------------------------------------------------------------------
    # BORE wall ownership from two-opening frame.
    # ------------------------------------------------------------------
    bore_features: list[dict[str, object]] = []
    bore_rejected: list[dict[str, object]] = []
    bore_wall_candidate_components: tuple[tuple[int, ...], ...] = ()
    bore_owned_face_count = 0
    best_bore_wall_diag: dict[str, object] = {}
    bore_tube_face_ids: tuple[int, ...] = ()
    bore_wall_search_face_ids: tuple[int, ...] = ()
    # v1.5.8: keep mouth/chamfer evidence separate from BORE wall evidence.
    # If no sidewall ownership exists, BORE must not preview tube/mouth faces;
    # the mouth transition can still be shown as its own CHAMFER review row.
    bore_mouth_transition_face_ids: tuple[int, ...] = ()
    bore_mouth_transition_diag: dict[str, object] = {}
    selected_opening_primary_edge_count = int(selected_opening_frame_resolver.get("primary_edge_count", 0) or 0)
    selected_opening_source = str(selected_opening_frame_resolver.get("resolver_source", "") or "")
    selected_opening_support_weak = bool(
        selected_opening_source == "raw_component_candidates"
        and selected_opening_primary_edge_count > 0
        and selected_opening_primary_edge_count < 24
    )
    selected_opening_resolved = bool(selected_opening_frame_resolver.get("resolved", False))
    secondary_opening_rim_proposal_v175g = _best_secondary_opening_rim_proposal_v175g(secondary_opening_rim_heuristic_diag)
    secondary_opening_rim_consumption_diag_v175g: dict[str, object] = {
        "bore_secondary_opening_rim_consumption_checkpoint_v175g": BORE_SECONDARY_OPENING_RIM_CONSUMPTION_CHECKPOINT_V175G,
        "bore_secondary_opening_rim_consumption_used_v175g": bool(secondary_opening_rim_proposal_v175g),
        "bore_secondary_opening_rim_consumption_authority_v175g": "heuristic_evidence_proposal_only_not_feature_identity_not_ownership",
        "bore_secondary_opening_rim_proposal_radius_v175g": float(_safe_float(secondary_opening_rim_proposal_v175g.get("radius", 0.0), 0.0)) if secondary_opening_rim_proposal_v175g else 0.0,
        "bore_secondary_opening_rim_proposal_relation_v175g": str(secondary_opening_rim_proposal_v175g.get("relation_to_selected_rail", "")) if secondary_opening_rim_proposal_v175g else "",
    }

    # v150: when the selected opening itself is a clean measured loop, that
    # opening remains the radius authority for BORE wall ownership.  The
    # opposite-opening search may confirm direction/depth, but it must not
    # inflate the physical cylinder radius from ~selected-mouth radius to a
    # larger unrelated boundary loop.  This is the exact failure seen on the
    # coarse mesh after v149: raw/normalized selected opening agreed at R~19.7,
    # while the provisional opposite loop had radius_delta_rel~0.44 and expanded
    # the BORE frame to R~27.4.
    selected_opening_radius_authority = _safe_float(
        selected_opening_frame_resolver.get(
            "radius",
            selected_opening_measurement_audit.get("raw_component_best_radius", bore_radius),
        ),
        bore_radius,
    )
    selected_opening_spread_ratio = _safe_float(
        selected_opening_frame_resolver.get(
            "radial_spread_ratio",
            selected_opening_measurement_audit.get("raw_component_best_radial_spread_ratio", 999999.0),
        ),
        999999.0,
    )
    selected_opening_confidence = _safe_float(selected_opening_frame_resolver.get("confidence", 0.0), 0.0)
    secondary_radius_v175g = _safe_float(secondary_opening_rim_proposal_v175g.get("radius", 0.0), 0.0) if secondary_opening_rim_proposal_v175g else 0.0
    secondary_axis_dot_v175g = _safe_float(secondary_opening_rim_proposal_v175g.get("axis_abs_dot_to_selected", 0.0), 0.0) if secondary_opening_rim_proposal_v175g else 0.0
    secondary_radius_delta_rel_v175g = abs(float(secondary_radius_v175g) - float(selected_opening_radius_authority)) / max(abs(float(selected_opening_radius_authority)), 1.0e-9) if secondary_radius_v175g > 1.0e-9 and selected_opening_radius_authority > 1.0e-9 else 0.0
    secondary_radius_authority_used_v175g = bool(
        selected_opening_support_weak
        and secondary_radius_v175g > 1.0e-9
        and secondary_axis_dot_v175g >= 0.92
        and secondary_radius_delta_rel_v175g >= 0.04
        and secondary_radius_delta_rel_v175g <= 0.45
    )
    if secondary_radius_authority_used_v175g:
        selected_opening_radius_authority = float(secondary_radius_v175g)
        selected_opening_confidence = max(float(selected_opening_confidence), _safe_float(secondary_opening_rim_proposal_v175g.get("score", 0.0), 0.0))
        secondary_opening_rim_consumption_diag_v175g.update({
            "bore_secondary_opening_rim_radius_authority_used_v175g": True,
            "bore_secondary_opening_rim_radius_authority_reason_v175g": "selected_primary_rail_weak_and_secondary_rim_is_parallel_related_opening_evidence",
            "bore_secondary_opening_rim_radius_delta_rel_v175g": float(secondary_radius_delta_rel_v175g),
        })
    else:
        secondary_opening_rim_consumption_diag_v175g.update({
            "bore_secondary_opening_rim_radius_authority_used_v175g": False,
            "bore_secondary_opening_rim_radius_delta_rel_v175g": float(secondary_radius_delta_rel_v175g),
        })
    selected_raw_norm_agree = bool(selected_opening_measurement_audit.get("raw_vs_normalized_measurement_agree", False))
    selected_audit_status = str(selected_opening_measurement_audit.get("audit_status", "") or "").strip().lower()
    selected_opening_clean_radius_authority = bool(
        selected_opening_resolved
        and selected_opening_radius_authority > 1.0e-9
        and selected_opening_confidence >= 0.80
        and selected_opening_spread_ratio <= 0.035
        and (selected_raw_norm_agree or selected_audit_status == "raw_and_normalized_opening_measurements_agree")
        and selected_opening_primary_edge_count >= 12
    )
    # v156: a small clean bore can legitimately have fewer than 24 selected
    # rim edges.  Edge count alone is not a surface-ownership rejection.  Keep
    # the old signal as a risk flag, then let measured wall geometry decide.
    selected_opening_edge_count_is_small_v156 = bool(
        selected_opening_source == "raw_component_candidates"
        and selected_opening_primary_edge_count > 0
        and selected_opening_primary_edge_count < 24
    )
    small_clean_opening_is_trustworthy_v156 = bool(
        selected_opening_edge_count_is_small_v156
        and selected_opening_clean_radius_authority
        and selected_opening_confidence >= 0.80
        and selected_opening_spread_ratio <= 0.025
        and (selected_raw_norm_agree or selected_audit_status == "raw_and_normalized_opening_measurements_agree")
    )

    secondary_opening_recontext_diag_v175i = _secondary_rim_recontextualized_bore_frame_v175i(
        vertices=vertices,
        faces=faces,
        face_centroids=face_centroids,
        valid_face_ids=valid_face_ids,
        proposal=secondary_opening_rim_proposal_v175g,
        selected_opening_support_weak=bool(selected_opening_support_weak),
        secondary_radius_authority_used=bool(secondary_radius_authority_used_v175g),
        current_bore_depth=float(bore_depth),
        current_bore_radius=float(bore_radius),
        edge_scale=float(edge_scale),
    )
    secondary_opening_recontext_used_v175i = bool(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_used_v175i", False))
    if secondary_opening_recontext_used_v175i:
        bore_radius = float(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_radius_v175i", bore_radius))
        bore_center = np.asarray(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_center_v175i", to_vector3(bore_center)), dtype=float).reshape(3)
        bore_axis = np.asarray(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_axis_v175i", to_vector3(bore_axis)), dtype=float).reshape(3)
        bore_axis = bore_axis / max(float(np.linalg.norm(bore_axis)), 1.0e-12)
        bore_axial_min = float(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_axial_min_v175i", 0.0))
        bore_axial_max = float(secondary_opening_recontext_diag_v175i.get("bore_secondary_opening_recontextualized_axial_max_v175i", bore_depth))
        bore_depth = float(max(0.0, bore_axial_max - bore_axial_min))
        bore_frame_reason = "secondary_opening_rim_recontextualized_bore_frame_v175i"
        two_opening_valid = True
        two_opening_bore_frame = {
            **dict(two_opening_bore_frame or {}),
            "valid": True,
            "status": "secondary_opening_rim_recontextualized_bore_frame_v175i",
            "semantic_stage": "OpeningRimEvidenceProposal -> DerivedMeasuredOpeningContext -> BORE wall ownership gates",
            "radius": float(bore_radius),
            "diameter": float(2.0 * bore_radius),
            "center": to_vector3(bore_center),
            "opening_center": to_vector3(bore_center),
            "axis": to_vector3(bore_axis),
            "depth": float(bore_depth),
            "bore_secondary_opening_recontextualized_used_v175i": True,
        }
        secondary_opening_rim_consumption_diag_v175g.update(dict(secondary_opening_recontext_diag_v175i))
    else:
        secondary_opening_rim_consumption_diag_v175g.update(dict(secondary_opening_recontext_diag_v175i))

    best_opposite = two_opening_bore_frame.get("best_opposite_candidate", {}) if isinstance(two_opening_bore_frame, Mapping) else {}
    if not isinstance(best_opposite, Mapping):
        best_opposite = {}
    opposite_radius_delta_rel = _safe_float(best_opposite.get("radius_delta_rel", 0.0), 0.0)
    selected_radius_delta_rel_to_frame = abs(float(bore_radius) - float(selected_opening_radius_authority)) / max(float(selected_opening_radius_authority), 1.0e-9)
    selected_opening_radius_clamp_used = bool(
        two_opening_valid
        and selected_opening_clean_radius_authority
        and (opposite_radius_delta_rel > 0.18 or selected_radius_delta_rel_to_frame > 0.18)
    )
    unclamped_two_opening_radius = float(bore_radius)
    if selected_opening_radius_clamp_used:
        bore_radius = float(selected_opening_radius_authority)
        two_opening_bore_frame = {
            **dict(two_opening_bore_frame),
            "radius_before_selected_opening_clamp_v150": float(unclamped_two_opening_radius),
            "radius": float(bore_radius),
            "selected_opening_radius_clamp_used_v150": True,
            "selected_opening_radius_authority_v150": float(selected_opening_radius_authority),
            "selected_opening_radius_delta_rel_to_frame_v150": float(selected_radius_delta_rel_to_frame),
            "opposite_radius_delta_rel_v150": float(opposite_radius_delta_rel),
            "radius_authority_policy_v150": "clean_selected_opening_radius_owns_wall_scale_opposite_loop_depth_only",
        }

    # v1.7.4: derive BORE role scale from the opening evidence, not from the
    # neutral RegionData face pool.  The selected/opening ring is the correct
    # local physical sampling scale for wall ownership thresholds.
    selected_opening_radius_for_scale = _safe_float(
        selected_opening_frame_resolver.get("radius", selected_opening_measurement_audit.get("raw_component_best_radius", bore_radius)),
        bore_radius,
    )
    selected_opening_edge_count_for_scale = int(
        selected_opening_frame_resolver.get(
            "expanded_edge_count",
            selected_opening_frame_resolver.get(
                "edge_count",
                selected_opening_frame_resolver.get("primary_edge_count", normalized_edge_count or selected_edge_count),
            ),
        )
        or normalized_edge_count
        or selected_edge_count
        or 0
    )
    if secondary_radius_authority_used_v175g and secondary_radius_v175g > 1.0e-9:
        selected_opening_radius_for_scale = float(secondary_radius_v175g)
    if selected_opening_radius_for_scale > 1.0e-9 and selected_opening_edge_count_for_scale >= 3:
        selected_opening_edge_scale = float((2.0 * math.pi * selected_opening_radius_for_scale) / max(float(selected_opening_edge_count_for_scale), 3.0))
    else:
        selected_opening_edge_scale = float(raw_region_edge_scale)
    bore_role_scale_limit = max(4.0 * selected_opening_edge_scale, 0.060 * max(float(bore_radius), 1.0), 0.35)
    bore_role_scale_clamped = bool(raw_region_edge_scale > bore_role_scale_limit)
    edge_scale = float(min(raw_region_edge_scale, bore_role_scale_limit))
    role_scale_authority = (
        "selected_opening_ring_scale_clamped_from_regiondata_face_scale"
        if bore_role_scale_clamped
        else "regiondata_face_scale_within_selected_opening_limit"
    )

    # v1.5.2: BORE resolution starts as soon as selected-opening measurement
    # exists.  CHAMFER rows may remain visible as review evidence, but broad or
    # full-RegionData chamfer rows cannot be the only accepted result while the
    # user's selected BORE opening is under measurement.
    bore_resolution_active = bool(selected_opening_resolved or two_opening_valid)

    if two_opening_valid and bore_radius > 1.0e-9 and bore_depth > max(0.50, 0.35 * bore_radius):
        ids = np.asarray(valid_face_ids, dtype=np.int64)
        pts = face_centroids[ids, :3]
        normals = _unit_rows_v174z(face_normals)
        if len(normals) <= int(np.max(ids)) if len(ids) else False:
            normals = np.zeros_like(face_centroids[:, :3])
        n = normals[ids, :3]
        rel = pts - bore_center.reshape(1, 3)
        axial = rel @ bore_axis.reshape(3)
        radial_vec = rel - axial.reshape(-1, 1) * bore_axis.reshape(1, 3)
        radial = np.linalg.norm(radial_vec, axis=1)
        radial_dir = np.zeros_like(radial_vec)
        ok_radial = radial > 1.0e-12
        radial_dir[ok_radial] = radial_vec[ok_radial] / radial[ok_radial].reshape(-1, 1)
        normal_axis_abs = np.abs(n @ bore_axis.reshape(3))
        radial_normal_alignment = np.abs(np.sum(n * radial_dir, axis=1))
        finite = np.isfinite(axial) & np.isfinite(radial) & np.isfinite(normal_axis_abs) & np.isfinite(radial_normal_alignment)

        # v1.6.2 semantic/numeric correction:
        # Axial continuity for BoreWallOwnership must be measured from each
        # face's physical vertex span, not only from its centroid.  A clean
        # cylindrical wall may be represented by long triangles/quads that run
        # from opening A to opening B; all face centroids can then sit near the
        # middle of the bore and falsely report near-zero axial coverage.
        # Centroids remain useful for radial/normal role evidence; vertex spans
        # are the authority for axial depth, endpoint contact, and wall extent.
        face_axial_min = np.array(axial, dtype=float, copy=True)
        face_axial_max = np.array(axial, dtype=float, copy=True)
        face_radial_min = np.array(radial, dtype=float, copy=True)
        face_radial_max = np.array(radial, dtype=float, copy=True)
        face_axial_span_available = False
        try:
            tri_vids = np.asarray(faces, dtype=np.int64)[ids, :3]
            tri_pts = np.asarray(vertices, dtype=float)[tri_vids, :3]
            tri_rel = tri_pts - bore_center.reshape(1, 1, 3)
            tri_ax = tri_rel @ bore_axis.reshape(3)
            tri_min = np.nanmin(tri_ax, axis=1)
            tri_max = np.nanmax(tri_ax, axis=1)
            tri_ok = np.isfinite(tri_min) & np.isfinite(tri_max)
            face_axial_min[tri_ok] = tri_min[tri_ok]
            face_axial_max[tri_ok] = tri_max[tri_ok]
            # v1.6.8: endpoint-support faces can be rim/cut/cap support, not
            # pure wall faces.  Their centroids may not sit on the bore radius,
            # so keep vertex radial intervals as endpoint-support evidence.
            tri_radial_vec = tri_rel - tri_ax[:, :, None] * bore_axis.reshape(1, 1, 3)
            tri_radial = np.linalg.norm(tri_radial_vec, axis=2)
            tri_rmin = np.nanmin(tri_radial, axis=1)
            tri_rmax = np.nanmax(tri_radial, axis=1)
            tri_rok = np.isfinite(tri_rmin) & np.isfinite(tri_rmax)
            face_radial_min[tri_rok] = tri_rmin[tri_rok]
            face_radial_max[tri_rok] = tri_rmax[tri_rok]
            face_axial_span_available = bool(np.any(tri_ok))
        except Exception:
            face_axial_span_available = False

        axial_tol = max(0.04 * bore_depth, 0.08 * bore_radius, 3.0 * edge_scale, 0.25)
        radial_tol_strict = max(0.075 * bore_radius, 2.5 * edge_scale, 0.18)
        radial_tol_loose = max(0.20 * bore_radius, 5.0 * edge_scale, 0.45)
        between_openings = (face_axial_max >= (bore_axial_min - axial_tol)) & (face_axial_min <= (bore_axial_max + axial_tol))
        frame_strict_wall_band = np.abs(radial - bore_radius) <= radial_tol_strict
        frame_loose_wall_band = np.abs(radial - bore_radius) <= radial_tol_loose

        # v1.6.0 semantic correction: the first wall scan must not be trapped
        # by the opening/chamfer radius.  Opening A can be the chamfer mouth,
        # while the physical bore sidewall may sit at a different radius.
        # Therefore we build three meanings explicitly:
        #   1) two_opening_search_boundary: broad tube/context between A and B
        #   2) sidewall_normal_role_evidence: interior faces with sidewall normals
        #   3) BoreWallOwnership candidates: connected components near the learned
        #      sidewall radius, with normal direction retained as role evidence.
        strict_wall_normal = (normal_axis_abs <= 0.55) & (radial_normal_alignment >= 0.55)
        sidewall_role_normal = (normal_axis_abs <= 0.68) & (radial_normal_alignment >= 0.32)
        tube_mask = finite & between_openings & frame_loose_wall_band
        bore_tube_face_ids = tuple(int(fid) for fid, keep in zip(ids, tube_mask) if bool(keep))

        interior_margin = max(0.060 * bore_depth, 0.12 * bore_radius, 4.0 * edge_scale, 0.50)
        # Keep the interior scan away from the chamfer/mouth bands.  If the bore
        # is very short, fall back to the whole interval rather than returning no
        # evidence.
        interior_between_openings = between_openings & (face_axial_max >= (bore_axial_min + interior_margin)) & (face_axial_min <= (bore_axial_max - interior_margin))
        if int(np.count_nonzero(interior_between_openings)) < 8:
            interior_between_openings = between_openings

        broad_radial_min = max(0.08 * bore_radius, 1.0 * edge_scale, 0.05)
        broad_radial_max = max(2.75 * bore_radius, bore_radius + 18.0 * edge_scale, bore_radius + 4.0)
        broad_radial_scan = (radial >= broad_radial_min) & (radial <= broad_radial_max)
        if selected_opening_clean_radius_authority:
            # v150: a clean selected mouth loop is stronger radius evidence than
            # the broad neutral RegionData.  Do not let the broad sidewall role
            # scan collect side/bottom/exterior cylinders at other radii.  This
            # still permits damaged/disconnected wall strips, but only on the
            # selected-opening radius layer.
            selected_radius_band_tol = max(
                0.16 * max(float(selected_opening_radius_authority), 1.0),
                6.0 * edge_scale,
                0.55,
            )
            broad_radial_min = max(0.0, float(selected_opening_radius_authority) - selected_radius_band_tol)
            broad_radial_max = float(selected_opening_radius_authority) + selected_radius_band_tol
            broad_radial_scan = np.abs(radial - float(selected_opening_radius_authority)) <= float(selected_radius_band_tol)
        else:
            selected_radius_band_tol = 0.0

        frame_sidewall_evidence_mask = tube_mask & strict_wall_normal
        broad_sidewall_evidence_mask = finite & interior_between_openings & broad_radial_scan & sidewall_role_normal
        sidewall_normal_evidence_mask = frame_sidewall_evidence_mask | broad_sidewall_evidence_mask
        sidewall_normal_evidence_count = int(np.count_nonzero(sidewall_normal_evidence_mask))
        frame_sidewall_normal_evidence_count = int(np.count_nonzero(frame_sidewall_evidence_mask))
        broad_sidewall_normal_evidence_count = int(np.count_nonzero(broad_sidewall_evidence_mask))

        learned_wall_radius = float(bore_radius)
        learned_wall_radius_source = "two_opening_frame_radius"
        learned_wall_radius_tol = float(radial_tol_strict)
        if sidewall_normal_evidence_count > 0:
            sidewall_radii = radial[sidewall_normal_evidence_mask]
            sidewall_radii = sidewall_radii[np.isfinite(sidewall_radii)]
            if sidewall_radii.size > 0:
                learned_wall_radius = float(np.median(sidewall_radii))
                learned_wall_radius_source = "interior_sidewall_normal_role_median_radius" if broad_sidewall_normal_evidence_count > frame_sidewall_normal_evidence_count else "frame_sidewall_normal_role_median_radius"
                learned_wall_radius_tol = float(max(0.12 * max(learned_wall_radius, 1.0e-9), 5.0 * edge_scale, 0.45))

        if bool(locals().get("secondary_opening_recontext_used_v175i", False)) and selected_opening_radius_authority > 1.0e-9:
            # Once a strong secondary rim has become the derived BORE opening
            # context, keep the wall radius layer anchored to that measured rim.
            # The broad sidewall median may still include chamfer/pocket/exterior
            # context in compound AOIs; it is evidence, not radius authority.
            learned_wall_radius = float(selected_opening_radius_authority)
            learned_wall_radius_source = "secondary_opening_recontextualized_rim_radius_v175i"
            learned_wall_radius_tol = float(max(0.10 * max(learned_wall_radius, 1.0e-9), 5.0 * edge_scale, 0.40))

        strict_wall_band = np.abs(radial - learned_wall_radius) <= learned_wall_radius_tol
        # Normal direction stays a criterion/evidence of sidewall role, but the
        # radius is learned from the interior sidewall evidence rather than forced
        # from the mouth ring.  This prevents chamfer faces from owning the Bore
        # while allowing the true interior cylinder to be discovered.
        geometry_wall_mask = finite & between_openings & strict_wall_band & sidewall_role_normal

        wall_face_ids = tuple(int(fid) for fid, keep in zip(ids, geometry_wall_mask) if bool(keep))
        bore_wall_search_face_ids = wall_face_ids
        bore_wall_candidate_components = connected_face_components(faces, wall_face_ids)

        fid_to_local = {int(fid): int(i) for i, fid in enumerate(ids)}

        # v152: when the selected rim is a clean closed loop, later ownership
        # must remain bound to that rim's local face island.  A broad same-radius
        # exterior/side strip can pass radial/normal tests but is not the wall
        # indicated by the clicked opening.  Build a small topological anchor
        # around Region Select's seed faces and require clean-rim wall components
        # to touch that anchor before they can own CandidateData faces.
        selected_opening_anchor_face_set: set[int] = set(int(v) for v in selected_seed_face_set)
        try:
            frontier = set(selected_opening_anchor_face_set)
            for _depth in range(3):
                next_frontier: set[int] = set()
                for _fid in frontier:
                    for _nb in adjacency.get(int(_fid), ()):  # local RegionData patch adjacency
                        _nb_i = int(_nb)
                        if _nb_i not in selected_opening_anchor_face_set:
                            next_frontier.add(_nb_i)
                if not next_frontier:
                    break
                selected_opening_anchor_face_set.update(next_frontier)
                frontier = next_frontier
        except Exception:
            selected_opening_anchor_face_set = set(int(v) for v in selected_seed_face_set)

        secondary_anchor_faces_v175g, secondary_anchor_diag_v175g = _secondary_opening_rim_anchor_face_ids_v175g(
            faces=faces,
            valid_face_ids=valid_face_ids,
            proposal=secondary_opening_rim_proposal_v175g,
        ) if secondary_opening_rim_proposal_v175g else ((), {
            "secondary_opening_rim_anchor_used_v175g": False,
            "secondary_opening_rim_anchor_reason_v175g": "no_secondary_opening_rim_proposal",
        })
        if secondary_anchor_faces_v175g and selected_opening_support_weak:
            selected_opening_anchor_face_set.update(int(fid) for fid in secondary_anchor_faces_v175g)
            # Expand once from the secondary rim faces so a wall component adjacent
            # to the discovered rim can satisfy the local anchor relation.
            for _fid in tuple(secondary_anchor_faces_v175g):
                for _nb in adjacency.get(int(_fid), ()):
                    selected_opening_anchor_face_set.add(int(_nb))
        secondary_opening_rim_consumption_diag_v175g.update(dict(secondary_anchor_diag_v175g or {}))

        # v1.5.8: derive a separate mouth-transition/chamfer review candidate
        # from the measured BORE frame.  These faces are deliberately *not*
        # BORE wall ownership.  They are the annular transition near one of the
        # openings: short axial span, chamfer-like normals, located on the
        # two-opening tube.
        mouth_touch_tolerance = max(0.12 * bore_radius, 0.050 * bore_depth, 6.0 * edge_scale, 0.50)
        near_opening_a = np.abs(axial - bore_axial_min) <= mouth_touch_tolerance
        near_opening_b = np.abs(axial - bore_axial_max) <= mouth_touch_tolerance
        chamfer_like_normal = (normal_axis_abs >= 0.45) & (normal_axis_abs <= 0.92) & (radial_normal_alignment >= 0.38)
        mouth_transition_mask = tube_mask & (near_opening_a | near_opening_b) & chamfer_like_normal
        mouth_transition_face_ids_all = tuple(int(fid) for fid, keep in zip(ids, mouth_transition_mask) if bool(keep))
        mouth_transition_components = connected_face_components(faces, mouth_transition_face_ids_all)
        mouth_transition_rows: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
        for mt_comp in mouth_transition_components:
            mt_ids = tuple_ints(mt_comp)
            mt_local_idx = np.asarray([fid_to_local[int(fid)] for fid in mt_ids if int(fid) in fid_to_local], dtype=np.int64)
            if mt_local_idx.size == 0:
                continue
            mt_set = {int(fid) for fid in mt_ids}
            mt_direct_seed = mt_set & selected_seed_face_set
            mt_adjacent_seed = {
                int(fid)
                for fid in mt_set
                if any(int(nb) in selected_seed_face_set for nb in adjacency.get(int(fid), ()))
            }
            mt_seed_related = bool(mt_direct_seed or mt_adjacent_seed)
            mt_ax = axial[mt_local_idx]
            mt_rd = radial[mt_local_idx]
            mt_na = normal_axis_abs[mt_local_idx]
            mt_ra = radial_normal_alignment[mt_local_idx]
            mt_face_count = int(len(mt_ids))
            mt_axial_min = float(np.min(mt_ax)) if mt_ax.size else 0.0
            mt_axial_max = float(np.max(mt_ax)) if mt_ax.size else 0.0
            mt_axial_span = float(mt_axial_max - mt_axial_min)
            mt_radius_inner = float(np.min(mt_rd)) if mt_rd.size else float(bore_radius)
            mt_radius_outer = float(np.max(mt_rd)) if mt_rd.size else float(bore_radius)
            mt_normal_axis = float(np.median(mt_na)) if mt_na.size else 0.0
            mt_radial_align = float(np.median(mt_ra)) if mt_ra.size else 0.0
            mt_touches_a = bool(np.min(np.abs(mt_ax - bore_axial_min)) <= mouth_touch_tolerance) if mt_ax.size else False
            mt_touches_b = bool(np.min(np.abs(mt_ax - bore_axial_max)) <= mouth_touch_tolerance) if mt_ax.size else False
            mt_score = (
                1.7 * min(mt_face_count / 32.0, 3.0)
                + (1.2 if mt_seed_related else 0.0)
                + 0.8 * min(abs(mt_radius_outer - mt_radius_inner) / max(edge_scale, 1.0e-9), 2.0)
                + 0.4 * min(mt_axial_span / max(edge_scale, 1.0e-9), 2.0)
            )
            mouth_transition_rows.append((mt_score, mt_ids, {
                "face_count": mt_face_count,
                "seed_related": bool(mt_seed_related),
                "seed_direct_face_count": int(len(mt_direct_seed)),
                "seed_adjacent_face_count": int(len(mt_adjacent_seed)),
                "radius_inner": float(mt_radius_inner),
                "radius_outer": float(mt_radius_outer),
                "axial_span": float(mt_axial_span),
                "axial_min": float(mt_axial_min),
                "axial_max": float(mt_axial_max),
                "normal_axis_abs_median": float(mt_normal_axis),
                "radial_normal_alignment_median": float(mt_radial_align),
                "touches_selected_opening": bool(mt_touches_a),
                "touches_opposite_opening": bool(mt_touches_b),
                "mouth_touch_tolerance": float(mouth_touch_tolerance),
                "preview_source": "mouth_transition_chamfer_role_evidence",
                "semantic_transform": "TwoOpeningBoreFrame -> MouthTransitionEvidence -> CHAMFER review, not BORE wall",
                "candidate_isolation_policy": "mouth_transition_separate_from_bore_wall_ownership",
                "score": float(mt_score),
            }))
        mouth_transition_rows.sort(key=lambda item: (-item[0], -int(item[2].get("seed_related", False)), -len(item[1]), item[1][:1]))
        if mouth_transition_rows:
            _, bore_mouth_transition_face_ids, bore_mouth_transition_diag = mouth_transition_rows[0]

        selected_seed_related_components: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
        other_components: list[tuple[float, tuple[int, ...], dict[str, object]]] = []

        # v1.6.1 semantic correction: BoreWallOwnership is a surface role, not a
        # single topological connected component requirement.  Imported OBJ/STL
        # meshes can contain a cylindrical wall as many disconnected triangle
        # islands/strips even when they are one physical bore wall.  The previous
        # gate split the valid sidewall evidence into many 2-face components and
        # rejected every one for low per-component axial span.  Here we first
        # evaluate the *aggregate* sidewall evidence against the measured
        # two-opening frame.  Per-component reports remain diagnostics only.
        semantic_aggregate_sidewall_ownership_used = False
        aggregate_sidewall_report: dict[str, object] = {}
        aggregate_disallowed_by_clean_opening_component_policy_v151 = False
        aggregate_wall_ids = tuple_ints(wall_face_ids)
        if aggregate_wall_ids:
            agg_idx = np.asarray([fid_to_local[int(fid)] for fid in aggregate_wall_ids if int(fid) in fid_to_local], dtype=np.int64)
            if agg_idx.size > 0:
                agg_ax = axial[agg_idx]
                agg_face_ax_min = face_axial_min[agg_idx]
                agg_face_ax_max = face_axial_max[agg_idx]
                agg_rd = radial[agg_idx]
                agg_na = normal_axis_abs[agg_idx]
                agg_ra = radial_normal_alignment[agg_idx]
                agg_face_count = int(len(aggregate_wall_ids))
                # v1.6.2: wall extent comes from face vertex intervals, not
                # centroid positions.  This handles long sidewall triangles whose
                # centroids all lie near the bore midpoint.
                agg_axial_min = float(np.min(agg_face_ax_min)) if agg_face_ax_min.size else 0.0
                agg_axial_max = float(np.max(agg_face_ax_max)) if agg_face_ax_max.size else 0.0
                agg_centroid_axial_min = float(np.min(agg_ax)) if agg_ax.size else 0.0
                agg_centroid_axial_max = float(np.max(agg_ax)) if agg_ax.size else 0.0
                agg_centroid_axial_span = float(agg_centroid_axial_max - agg_centroid_axial_min)
                agg_axial_span = float(agg_axial_max - agg_axial_min)
                agg_interval_overlap = max(0.0, min(agg_axial_max, bore_axial_max) - max(agg_axial_min, bore_axial_min))
                agg_axial_coverage = float(agg_interval_overlap / max(bore_depth, 1.0e-9))
                agg_strict_ratio = float(np.mean(strict_wall_band[agg_idx])) if agg_idx.size else 0.0
                agg_radial_median = float(np.median(agg_rd)) if agg_rd.size else float(learned_wall_radius)
                # IMPORTANT: compare scatter to the learned sidewall radius, not
                # the averaged mouth/opposite BoreFrame radius.  The mouth can be
                # a chamfer radius while the actual sidewall lives at the smaller
                # inner radius.
                agg_radial_abs_mad = float(np.median(np.abs(agg_rd - learned_wall_radius))) if agg_rd.size else 999999.0
                agg_radial_rel_mad = float(agg_radial_abs_mad / max(abs(float(learned_wall_radius)), 1.0e-9))
                agg_normal_axis = float(np.median(agg_na)) if agg_na.size else 1.0
                agg_radial_align = float(np.median(agg_ra)) if agg_ra.size else 0.0
                aggregate_opening_touch_tolerance = max(0.18 * bore_radius, 0.060 * bore_depth, 6.0 * edge_scale, 0.50)
                if agg_face_ax_min.size:
                    agg_min_distance_to_a = float(np.min(np.where((agg_face_ax_min <= bore_axial_min) & (agg_face_ax_max >= bore_axial_min), 0.0, np.minimum(np.abs(agg_face_ax_min - bore_axial_min), np.abs(agg_face_ax_max - bore_axial_min)))))
                    agg_min_distance_to_b = float(np.min(np.where((agg_face_ax_min <= bore_axial_max) & (agg_face_ax_max >= bore_axial_max), 0.0, np.minimum(np.abs(agg_face_ax_min - bore_axial_max), np.abs(agg_face_ax_max - bore_axial_max)))))
                else:
                    agg_min_distance_to_a = 999999.0
                    agg_min_distance_to_b = 999999.0
                agg_touches_a = bool(agg_min_distance_to_a <= aggregate_opening_touch_tolerance)
                agg_touches_b = bool(agg_min_distance_to_b <= aggregate_opening_touch_tolerance)
                agg_touches_both = bool(agg_touches_a and agg_touches_b)

                # Angular support around the measured axis: disconnected strips
                # are acceptable only if they collectively describe a meaningful
                # cylindrical perimeter, not just a tiny random arc.
                agg_radial_dirs = radial_dir[agg_idx, :3] if agg_idx.size else np.zeros((0, 3), dtype=float)
                agg_angular_coverage = 0.0
                if agg_radial_dirs.shape[0] >= 3:
                    axis_u = bore_axis / max(float(np.linalg.norm(bore_axis)), 1.0e-12)
                    helper = np.array([1.0, 0.0, 0.0], dtype=float)
                    if abs(float(np.dot(axis_u, helper))) > 0.82:
                        helper = np.array([0.0, 1.0, 0.0], dtype=float)
                    basis0 = np.cross(axis_u, helper)
                    basis0 = basis0 / max(float(np.linalg.norm(basis0)), 1.0e-12)
                    basis1 = np.cross(axis_u, basis0)
                    basis1 = basis1 / max(float(np.linalg.norm(basis1)), 1.0e-12)
                    theta = np.arctan2(agg_radial_dirs @ basis1, agg_radial_dirs @ basis0)
                    theta = theta[np.isfinite(theta)]
                    if theta.size >= 3:
                        theta = np.sort(theta)
                        gaps = np.diff(np.concatenate([theta, theta[:1] + 2.0 * math.pi]))
                        max_gap = float(np.max(gaps)) if gaps.size else 2.0 * math.pi
                        agg_angular_coverage = float(max(0.0, min(1.0, (2.0 * math.pi - max_gap) / (2.0 * math.pi))))

                agg_rejection_reasons: list[str] = []
                # v151: when the selected opening is already a clean closed loop,
                # disconnected side-wall aggregation is no longer allowed to
                # override per-component ownership.  The previous aggregate path
                # could union the real bore wall with an exterior/side-wall slit:
                # the union passed broad radius/axial tests even though one
                # component would be rejected on its own.  For clean selected
                # openings, ownership must be component-first; aggregate evidence
                # remains diagnostic unless the wall is one physical component.
                aggregate_disallowed_by_clean_opening_component_policy_v151 = bool(
                    selected_opening_clean_radius_authority
                    and int(len(bore_wall_candidate_components)) > 1
                )
                if aggregate_disallowed_by_clean_opening_component_policy_v151:
                    agg_rejection_reasons.append("aggregate_wall_disallowed_for_clean_selected_opening_multi_component")
                if agg_face_count < max(24, int(0.12 * max(selected_opening_primary_edge_count, 1))):
                    agg_rejection_reasons.append("aggregate_wall_too_few_faces")
                if agg_axial_coverage < 0.55:
                    agg_rejection_reasons.append("aggregate_wall_axial_coverage_too_small")
                if agg_axial_span < max(0.45 * bore_depth, 0.85 * bore_radius, 8.0 * edge_scale, 1.0):
                    agg_rejection_reasons.append("aggregate_wall_axial_span_too_small")
                if agg_strict_ratio < 0.35:
                    agg_rejection_reasons.append("aggregate_wall_not_enough_radius_support")
                if agg_radial_rel_mad > 0.20:
                    agg_rejection_reasons.append("aggregate_wall_radius_scatter_too_high")
                if agg_angular_coverage < 0.20:
                    agg_rejection_reasons.append("aggregate_wall_angular_coverage_too_small")
                if not (agg_touches_both or agg_axial_coverage >= 0.78):
                    if not agg_touches_a:
                        agg_rejection_reasons.append("aggregate_wall_does_not_touch_selected_opening")
                    if not agg_touches_b:
                        agg_rejection_reasons.append("aggregate_wall_does_not_touch_opposite_opening")
                semantic_aggregate_sidewall_ownership_used = bool(not agg_rejection_reasons)
                agg_score = (
                    3.20 * agg_axial_coverage
                    + 1.60 * min(agg_face_count / 80.0, 2.5)
                    + 1.40 * agg_strict_ratio
                    + 1.25 * agg_angular_coverage
                    + 1.20 * (1.0 - min(agg_radial_rel_mad / 0.22, 1.0))
                    + (1.20 if agg_touches_both else 0.0)
                )
                aggregate_sidewall_report = {
                    "face_count": int(agg_face_count),
                    "axis": to_vector3(bore_axis),
                    "center": to_vector3(bore_center),
                    "radial_median": float(agg_radial_median),
                    "radial_rel_mad": float(agg_radial_rel_mad),
                    "radial_abs_mad": float(agg_radial_abs_mad),
                    "normal_axis_abs_median": float(agg_normal_axis),
                    "radial_normal_alignment_median": float(agg_radial_align),
                    "axial_min": float(agg_axial_min),
                    "axial_max": float(agg_axial_max),
                    "axial_span": float(agg_axial_span),
                    "centroid_axial_min": float(agg_centroid_axial_min),
                    "centroid_axial_max": float(agg_centroid_axial_max),
                    "centroid_axial_span": float(agg_centroid_axial_span),
                    "face_span_axial_ownership_used": bool(face_axial_span_available),
                    "two_opening_bore_frame_valid": True,
                    "two_opening_bore_frame_depth": float(bore_depth),
                    "two_opening_axial_min": float(bore_axial_min),
                    "two_opening_axial_max": float(bore_axial_max),
                    "wall_span_fraction_of_two_opening_depth": float(agg_axial_coverage),
                    "strict_wall_band_ratio": float(agg_strict_ratio),
                    "angular_coverage": float(agg_angular_coverage),
                    "seed_related": True,
                    "seed_direct_face_count": 0,
                    "seed_adjacent_face_count": 0,
                    "touches_selected_opening": bool(agg_touches_a),
                    "touches_opposite_opening": bool(agg_touches_b),
                    "touches_both_openings": bool(agg_touches_both),
                    "opening_touch_tolerance": float(aggregate_opening_touch_tolerance),
                    "min_distance_to_selected_opening": float(agg_min_distance_to_a),
                    "min_distance_to_opposite_opening": float(agg_min_distance_to_b),
                    "accepted_as_bore_wall_ownership": bool(semantic_aggregate_sidewall_ownership_used),
                    "wall_report_kind": "aggregate",
                    "selected_opening_support_weak": bool(selected_opening_support_weak),
                    "selected_opening_edge_count_is_small_v156": bool(selected_opening_edge_count_is_small_v156),
                    "small_clean_opening_is_trustworthy_v156": bool(small_clean_opening_is_trustworthy_v156),
                    "small_clean_component_geometry_strong_v156": False,
                    "selected_opening_primary_component_weak_overridden_v156": False,
                    "small_clean_bore_ownership_promoted_v156": False,
                    "selected_opening_primary_edge_count": int(selected_opening_primary_edge_count),
                    "normal_evidence_warning": "",
                    "sidewall_normal_role_ownership_used": True,
                    "semantic_aggregate_sidewall_ownership_used": bool(semantic_aggregate_sidewall_ownership_used),
                    "geometry_first_wall_ownership_used": False,
                    "sidewall_normal_evidence_count": int(sidewall_normal_evidence_count),
                    "frame_sidewall_normal_evidence_count": int(frame_sidewall_normal_evidence_count),
                    "broad_sidewall_normal_evidence_count": int(broad_sidewall_normal_evidence_count),
                    "bore_wall_candidate_component_count": int(len(bore_wall_candidate_components)),
                    "bore_wall_interior_margin": float(interior_margin),
                    "bore_wall_broad_radial_min": float(broad_radial_min),
                    "bore_wall_broad_radial_max": float(broad_radial_max),
                    "learned_wall_radius": float(learned_wall_radius),
                    "learned_wall_radius_source": str(learned_wall_radius_source),
                    "learned_wall_radius_tolerance": float(learned_wall_radius_tol),
                    "raw_region_edge_scale": float(raw_region_edge_scale),
                    "selected_opening_edge_scale": float(selected_opening_edge_scale),
                    "bore_role_edge_scale": float(edge_scale),
                    "bore_role_scale_limit": float(bore_role_scale_limit),
                    "bore_role_scale_clamped": bool(bore_role_scale_clamped),
                    "role_scale_authority": str(role_scale_authority),
                    "selected_opening_radius_authority_v150": float(selected_opening_radius_authority),
                    "selected_opening_clean_radius_authority_v150": bool(selected_opening_clean_radius_authority),
                    "selected_opening_radius_clamp_used_v150": bool(selected_opening_radius_clamp_used),
                    "aggregate_disallowed_by_clean_opening_component_policy_v151": bool(aggregate_disallowed_by_clean_opening_component_policy_v151),
                    "bore_wall_component_count_for_aggregate_policy_v151": int(len(bore_wall_candidate_components)),
                    "wall_ownership_policy_v151": (
                        "component_first_for_clean_selected_opening"
                        if aggregate_disallowed_by_clean_opening_component_policy_v151
                        else "aggregate_allowed_or_single_component"
                    ),
                    "two_opening_radius_before_selected_clamp_v150": float(unclamped_two_opening_radius),
                    "selected_opening_radius_delta_rel_to_frame_v150": float(selected_radius_delta_rel_to_frame),
                    "opposite_radius_delta_rel_v150": float(opposite_radius_delta_rel),
                    "selected_radius_band_tolerance_v150": float(selected_radius_band_tol),
                    "radius_authority_policy_v150": (
                        "clean_selected_opening_radius_owns_wall_scale_opposite_loop_depth_only"
                        if selected_opening_radius_clamp_used
                        else "two_opening_frame_radius_unmodified"
                    ),
                    "face_ids": tuple(aggregate_wall_ids),
                    "bore_wall_component_rejection_reasons": tuple(agg_rejection_reasons),
                    "bore_wall_ownership_reject_reason": "" if semantic_aggregate_sidewall_ownership_used else (str(agg_rejection_reasons[0]) if agg_rejection_reasons else "aggregate_wall_failed_role_ownership_gate"),
                    "semantic_transform": "TwoOpeningBoreFrame -> disconnected SidewallNormalRoleEvidence -> aggregate BoreWallOwnership -> CandidateData",
                    "score": float(agg_score),
                }
                if semantic_aggregate_sidewall_ownership_used:
                    selected_seed_related_components.append((float(agg_score), tuple(aggregate_wall_ids), aggregate_sidewall_report))
                else:
                    bore_rejected.append(dict(aggregate_sidewall_report))

        for comp in bore_wall_candidate_components:
            comp_ids = tuple_ints(comp)
            raw_comp_ids_v173u = tuple(comp_ids)
            local_idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
            if local_idx.size == 0:
                continue

            # v173u: before BORE_WALL ownership is evaluated, map the connected
            # wall evidence component to the physical BORE definition itself:
            # the cylindrical wall surface bounded by the selected opening rail
            # and the opposite / exit opening rail.  This is intentionally not
            # a cross-family subtraction rule.  It does not ask whether a face
            # has already been called CHAMFER, POCKET, or anything else.  It asks
            # only whether the face belongs to the measured opening-to-exit bore
            # wall interval: between the two openings, on the learned cylinder
            # radius band, with wall-like radial normals, and in a coherent wall
            # island.  Faces outside that physical BORE model remain evidence or
            # context, but must not become BORE_WALL ownership.
            bore_opening_to_exit_subset_diag_v173u: dict[str, object] = {
                "bore_opening_to_exit_wall_subset_mapping_v173u": False,
                "bore_opening_to_exit_wall_subset_used_v173u": False,
                "bore_opening_to_exit_wall_subset_input_face_count_v173u": int(len(raw_comp_ids_v173u)),
                "bore_opening_to_exit_wall_subset_face_count_v173u": int(len(raw_comp_ids_v173u)),
                "bore_opening_to_exit_wall_subset_context_face_count_v173u": 0,
                "bore_opening_to_exit_wall_subset_rule_v173u": (
                    "BORE_WALL is opening-to-exit cylindrical wall ownership; no other feature family is used as negative definition."
                ),
            }
            if two_opening_valid and selected_opening_resolved and local_idx.size > 0:
                try:
                    v173u_opening_touch_tolerance = max(0.18 * bore_radius, 0.060 * bore_depth, 6.0 * edge_scale, 0.50)
                    v173u_radial_span_limit = max(0.32 * max(float(learned_wall_radius), 1.0), 5.0 * edge_scale, 0.75)
                    v173u_core_mask = (
                        (face_axial_max[local_idx] >= (bore_axial_min - axial_tol))
                        & (face_axial_min[local_idx] <= (bore_axial_max + axial_tol))
                        & (np.abs(radial[local_idx] - float(learned_wall_radius)) <= float(learned_wall_radius_tol))
                        & ((face_radial_max[local_idx] - face_radial_min[local_idx]) <= float(v173u_radial_span_limit))
                        & (normal_axis_abs[local_idx] <= 0.42)
                        & (radial_normal_alignment[local_idx] >= 0.68)
                    )
                    v173u_core_ids_all = tuple(
                        int(fid)
                        for fid, keep in zip(comp_ids, v173u_core_mask)
                        if bool(keep)
                    )
                    v173u_best_ids: tuple[int, ...] = ()
                    v173u_best_score = -1.0e18
                    v173u_best_report: dict[str, object] = {}
                    if v173u_core_ids_all:
                        for _core_comp in connected_face_components(faces, v173u_core_ids_all):
                            _ids = tuple_ints(_core_comp)
                            if not _ids:
                                continue
                            _idx = np.asarray([fid_to_local[int(fid)] for fid in _ids if int(fid) in fid_to_local], dtype=np.int64)
                            if _idx.size == 0:
                                continue
                            _ax_min_arr = face_axial_min[_idx]
                            _ax_max_arr = face_axial_max[_idx]
                            _axial_min = float(np.min(_ax_min_arr))
                            _axial_max = float(np.max(_ax_max_arr))
                            _axial_span = float(_axial_max - _axial_min)
                            _overlap = max(0.0, min(_axial_max, bore_axial_max) - max(_axial_min, bore_axial_min))
                            _coverage = float(_overlap / max(bore_depth, 1.0e-9))
                            _d_a = float(np.min(np.where((_ax_min_arr <= bore_axial_min) & (_ax_max_arr >= bore_axial_min), 0.0, np.minimum(np.abs(_ax_min_arr - bore_axial_min), np.abs(_ax_max_arr - bore_axial_min)))))
                            _d_b = float(np.min(np.where((_ax_min_arr <= bore_axial_max) & (_ax_max_arr >= bore_axial_max), 0.0, np.minimum(np.abs(_ax_min_arr - bore_axial_max), np.abs(_ax_max_arr - bore_axial_max)))))
                            _touch_a = bool(_d_a <= v173u_opening_touch_tolerance)
                            _touch_b = bool(_d_b <= v173u_opening_touch_tolerance)
                            _rad_rel_mad = float(np.median(np.abs(radial[_idx] - float(learned_wall_radius))) / max(float(learned_wall_radius), 1.0e-9))
                            _normal_axis_med = float(np.median(normal_axis_abs[_idx]))
                            _radial_align_med = float(np.median(radial_normal_alignment[_idx]))
                            _score = (
                                3.0 * _coverage
                                + 1.6 * min(len(_ids) / max(float(len(raw_comp_ids_v173u)), 1.0), 1.0)
                                + (1.4 if (_touch_a and _touch_b) else 0.0)
                                + 1.0 * (1.0 - min(_rad_rel_mad / 0.22, 1.0))
                                + 0.8 * min(_radial_align_med, 1.0)
                                - 0.6 * min(_normal_axis_med, 1.0)
                            )
                            if _score > v173u_best_score:
                                v173u_best_score = float(_score)
                                v173u_best_ids = tuple_ints(_ids)
                                v173u_best_report = {
                                    "bore_opening_to_exit_candidate_face_count_v173u": int(len(_ids)),
                                    "bore_opening_to_exit_candidate_axial_coverage_v173u": float(_coverage),
                                    "bore_opening_to_exit_candidate_axial_span_v173u": float(_axial_span),
                                    "bore_opening_to_exit_candidate_touches_selected_opening_v173u": bool(_touch_a),
                                    "bore_opening_to_exit_candidate_touches_opposite_opening_v173u": bool(_touch_b),
                                    "bore_opening_to_exit_candidate_radial_rel_mad_v173u": float(_rad_rel_mad),
                                    "bore_opening_to_exit_candidate_normal_axis_median_v173u": float(_normal_axis_med),
                                    "bore_opening_to_exit_candidate_radial_alignment_median_v173u": float(_radial_align_med),
                                    "bore_opening_to_exit_candidate_score_v173u": float(_score),
                                }
                    # Use the contained BORE-wall subset when it is a coherent,
                    # measured wall island.  This may remove chamfer/mouth/support
                    # context faces, but it does so only by BORE physics.
                    v173u_min_subset_faces = max(24, int(0.55 * max(len(raw_comp_ids_v173u), 1)))
                    if v173u_best_ids and len(v173u_best_ids) >= v173u_min_subset_faces:
                        _best_set = set(int(v) for v in v173u_best_ids)
                        comp_ids = tuple_ints(v173u_best_ids)
                        local_idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
                        bore_opening_to_exit_subset_diag_v173u.update({
                            "bore_opening_to_exit_wall_subset_mapping_v173u": True,
                            "bore_opening_to_exit_wall_subset_used_v173u": bool(set(raw_comp_ids_v173u) != _best_set),
                            "bore_opening_to_exit_wall_subset_input_face_count_v173u": int(len(raw_comp_ids_v173u)),
                            "bore_opening_to_exit_wall_subset_face_count_v173u": int(len(comp_ids)),
                            "bore_opening_to_exit_wall_subset_context_face_count_v173u": int(len(set(raw_comp_ids_v173u) - _best_set)),
                            "bore_opening_to_exit_wall_subset_source_v173u": "opening_to_exit_cylindrical_wall_containment",
                            **v173u_best_report,
                        })
                except Exception as _v173u_exc:
                    bore_opening_to_exit_subset_diag_v173u.update({
                        "bore_opening_to_exit_wall_subset_mapping_v173u": True,
                        "bore_opening_to_exit_wall_subset_error_v173u": str(_v173u_exc),
                    })

            comp_set = {int(fid) for fid in comp_ids}
            direct_seed = comp_set & selected_seed_face_set
            adjacent_seed = set()
            for fid in comp_set:
                if any(int(nb) in selected_seed_face_set for nb in adjacency.get(int(fid), ())):
                    adjacent_seed.add(int(fid))
            anchor_seed = comp_set & selected_opening_anchor_face_set
            adjacent_anchor = set()
            for fid in comp_set:
                if any(int(nb) in selected_opening_anchor_face_set for nb in adjacency.get(int(fid), ())):
                    adjacent_anchor.add(int(fid))
            seed_related = bool(direct_seed or adjacent_seed or anchor_seed or adjacent_anchor)
            ax = axial[local_idx]
            face_ax_min = face_axial_min[local_idx]
            face_ax_max = face_axial_max[local_idx]
            rd = radial[local_idx]
            na = normal_axis_abs[local_idx]
            ra = radial_normal_alignment[local_idx]
            strict_ratio = float(np.mean(strict_wall_band[local_idx])) if local_idx.size else 0.0
            axial_min = float(np.min(face_ax_min))
            axial_max = float(np.max(face_ax_max))
            centroid_axial_min = float(np.min(ax))
            centroid_axial_max = float(np.max(ax))
            centroid_axial_span = float(centroid_axial_max - centroid_axial_min)
            axial_span = float(axial_max - axial_min)
            interval_overlap = max(0.0, min(axial_max, bore_axial_max) - max(axial_min, bore_axial_min))
            axial_coverage = float(interval_overlap / max(bore_depth, 1.0e-9))
            radial_rel_mad = float(np.median(np.abs(rd - bore_radius)) / max(bore_radius, 1.0e-9))
            radial_abs_mad = float(np.median(np.abs(rd - bore_radius))) if rd.size else 999999.0
            normal_axis_median = float(np.median(na)) if na.size else 1.0
            radial_alignment_median = float(np.median(ra)) if ra.size else 0.0
            face_count = int(len(comp_ids))
            min_faces = max(16, int(0.025 * max(region_face_count, 1))) if broad_or_ambiguous_selection else 8
            opening_touch_tolerance = max(0.18 * bore_radius, 0.060 * bore_depth, 6.0 * edge_scale, 0.50)
            if face_ax_min.size:
                min_distance_to_opening_a = float(np.min(np.where((face_ax_min <= bore_axial_min) & (face_ax_max >= bore_axial_min), 0.0, np.minimum(np.abs(face_ax_min - bore_axial_min), np.abs(face_ax_max - bore_axial_min)))))
                min_distance_to_opening_b = float(np.min(np.where((face_ax_min <= bore_axial_max) & (face_ax_max >= bore_axial_max), 0.0, np.minimum(np.abs(face_ax_min - bore_axial_max), np.abs(face_ax_max - bore_axial_max)))))
            else:
                min_distance_to_opening_a = 999999.0
                min_distance_to_opening_b = 999999.0
            touches_opening_a = bool(min_distance_to_opening_a <= opening_touch_tolerance)
            touches_opening_b = bool(min_distance_to_opening_b <= opening_touch_tolerance)
            touches_both_openings = bool(touches_opening_a and touches_opening_b)
            # v1.5.4 semantic correction: selected seed contact is evidence, but a
            # two-opening bore wall is owned by span between the two measured
            # openings.  Do not require the selected mouth seed faces to be part
            # of the wall component; require endpoint contact instead.
            spans_frame_well = bool(axial_coverage >= 0.78)
            endpoint_supported = bool(touches_both_openings or spans_frame_well)
            small_clean_component_geometry_strong_v156 = bool(
                small_clean_opening_is_trustworthy_v156
                and face_count >= max(32, int(0.010 * max(region_face_count, 1)))
                and axial_coverage >= 0.78
                and axial_span >= max(0.58 * bore_depth, 4.0 * edge_scale, 0.75)
                and strict_ratio >= 0.20
                and radial_rel_mad <= 0.08
                and radial_alignment_median >= 0.97
                and normal_axis_median <= 0.08
                and touches_opening_a
                and endpoint_supported
            )
            selected_opening_primary_component_weak_overridden_v156 = bool(
                selected_opening_support_weak and small_clean_component_geometry_strong_v156
            )
            small_clean_bore_ownership_promoted_v156 = bool(selected_opening_primary_component_weak_overridden_v156)

            # v173u: a small selected rail is not weak if the BORE wall itself
            # proves the physical opening-to-exit cylinder.  This is independent
            # BORE recognition: selected rail + opposite rail + measured axis +
            # contained cylindrical wall interval.  It does not subtract or depend
            # on CHAMFER recognition.
            bore_opening_to_exit_wall_ownership_valid_v173u = bool(
                two_opening_valid
                and selected_opening_resolved
                and face_count >= min_faces
                and axial_coverage >= 0.78
                and axial_span >= max(0.58 * bore_depth, 4.0 * edge_scale, 0.75)
                and strict_ratio >= 0.20
                and radial_rel_mad <= 0.18
                and radial_alignment_median >= 0.90
                and normal_axis_median <= 0.18
                and touches_opening_a
                and endpoint_supported
                and (seed_related or not selected_opening_clean_radius_authority)
            )
            selected_opening_primary_component_weak_overridden_v173u = bool(
                selected_opening_support_weak and bore_opening_to_exit_wall_ownership_valid_v173u
            )

            rejection_reasons: list[str] = []
            if (
                selected_opening_support_weak
                and not selected_opening_primary_component_weak_overridden_v156
                and not selected_opening_primary_component_weak_overridden_v173u
            ):
                rejection_reasons.append("selected_opening_primary_component_too_weak_for_wall_ownership")
            if selected_opening_clean_radius_authority and not seed_related:
                rejection_reasons.append("wall_component_not_bound_to_clean_selected_rim_anchor_v152")
            if face_count < min_faces:
                rejection_reasons.append("wall_component_too_few_faces")
            if not endpoint_supported:
                if not touches_opening_a:
                    rejection_reasons.append("wall_component_does_not_touch_selected_opening")
                if not touches_opening_b:
                    rejection_reasons.append("wall_component_does_not_touch_opposite_opening")
                if axial_coverage < 0.78:
                    rejection_reasons.append("wall_component_axial_coverage_too_small")
            if axial_coverage < 0.45:
                rejection_reasons.append("wall_component_axial_coverage_below_minimum")
            if axial_span < max(0.38 * bore_depth, 0.65 * bore_radius, 6.0 * edge_scale, 0.65):
                rejection_reasons.append("wall_component_axial_span_too_small")
            if strict_ratio < 0.20:
                rejection_reasons.append("wall_component_not_enough_strict_radius_support")
            if radial_rel_mad > 0.22:
                rejection_reasons.append("wall_component_radius_scatter_too_high")
            # Normals are no longer hard rejection authority.  They are retained
            # as evidence/warnings because imported triangle normals can be
            # unreliable around repaired or dense tessellated walls.
            normal_evidence_warning = ""
            if normal_axis_median > 0.62:
                normal_evidence_warning = "wall_component_normals_axial_or_chamfer_like_warning"
            elif radial_alignment_median < 0.32:
                normal_evidence_warning = "wall_component_normals_radial_alignment_weak_warning"
            valid_wall = bool(not rejection_reasons)
            score = (
                2.60 * axial_coverage
                + 1.80 * min(face_count / max(float(min_faces), 1.0), 2.5)
                + 1.30 * strict_ratio
                + 1.25 * (1.0 - min(radial_rel_mad / 0.24, 1.0))
                + (1.85 if touches_both_openings else -2.50)
                + (0.55 if seed_related else 0.0)
            )
            st = {
                "face_count": face_count,
                "axis": to_vector3(bore_axis),
                "center": to_vector3(bore_center),
                "radial_median": float(np.median(rd)),
                "radial_rel_mad": float(radial_rel_mad),
                "radial_abs_mad": float(radial_abs_mad),
                "normal_axis_abs_median": float(normal_axis_median),
                "radial_normal_alignment_median": float(radial_alignment_median),
                "axial_min": axial_min,
                "axial_max": axial_max,
                "axial_span": axial_span,
                "centroid_axial_min": float(centroid_axial_min),
                "centroid_axial_max": float(centroid_axial_max),
                "centroid_axial_span": float(centroid_axial_span),
                "face_span_axial_ownership_used": bool(face_axial_span_available),
                "two_opening_bore_frame_valid": True,
                "two_opening_bore_frame_depth": float(bore_depth),
                "two_opening_axial_min": float(bore_axial_min),
                "two_opening_axial_max": float(bore_axial_max),
                "wall_span_fraction_of_two_opening_depth": float(axial_coverage),
                "strict_wall_band_ratio": float(strict_ratio),
                "seed_related": bool(seed_related),
                "seed_direct_face_count": int(len(direct_seed)),
                "seed_adjacent_face_count": int(len(adjacent_seed)),
                "seed_anchor_face_count_v152": int(len(anchor_seed)),
                "seed_anchor_adjacent_face_count_v152": int(len(adjacent_anchor)),
                "selected_opening_anchor_face_count_v152": int(len(selected_opening_anchor_face_set)),
                **dict(secondary_opening_rim_consumption_diag_v175g),
                "selected_opening_anchor_policy_v152": "clean_selected_rim_component_must_touch_anchor" if selected_opening_clean_radius_authority else "anchor_diagnostic_only",
                "touches_selected_opening": bool(touches_opening_a),
                "touches_opposite_opening": bool(touches_opening_b),
                "touches_both_openings": bool(touches_both_openings),
                "opening_touch_tolerance": float(opening_touch_tolerance),
                "min_distance_to_selected_opening": float(min_distance_to_opening_a),
                "min_distance_to_opposite_opening": float(min_distance_to_opening_b),
                "accepted_as_bore_wall_ownership": bool(valid_wall),
                "wall_report_kind": "connected_component",
                "selected_opening_support_weak": bool(selected_opening_support_weak),
                "selected_opening_edge_count_is_small_v156": bool(selected_opening_edge_count_is_small_v156),
                "small_clean_opening_is_trustworthy_v156": bool(small_clean_opening_is_trustworthy_v156),
                "small_clean_component_geometry_strong_v156": bool(small_clean_component_geometry_strong_v156),
                "selected_opening_primary_component_weak_overridden_v156": bool(selected_opening_primary_component_weak_overridden_v156),
                "small_clean_bore_ownership_promoted_v156": bool(small_clean_bore_ownership_promoted_v156),
                "small_clean_bore_promotion_reason_v156": (
                    "small_clean_selected_rim_plus_strong_coaxial_full_span_wall_component"
                    if small_clean_bore_ownership_promoted_v156
                    else "small_clean_gate_not_applicable_or_component_not_strong_enough"
                ),
                "bore_opening_to_exit_wall_ownership_valid_v173u": bool(bore_opening_to_exit_wall_ownership_valid_v173u),
                "selected_opening_primary_component_weak_overridden_v173u": bool(selected_opening_primary_component_weak_overridden_v173u),
                "bore_opening_to_exit_wall_ownership_rule_v173u": (
                    "BORE_WALL ownership is proven by faces contained between the selected opening and exit rails on the measured cylindrical wall; no CHAMFER/other-family subtraction is used."
                ),
                **dict(bore_opening_to_exit_subset_diag_v173u),
                "selected_opening_primary_edge_count": int(selected_opening_primary_edge_count),
                "normal_evidence_warning": normal_evidence_warning,
                "sidewall_normal_role_ownership_used": True,
                "geometry_first_wall_ownership_used": False,
                "sidewall_normal_evidence_count": int(sidewall_normal_evidence_count),
                "frame_sidewall_normal_evidence_count": int(frame_sidewall_normal_evidence_count),
                "broad_sidewall_normal_evidence_count": int(broad_sidewall_normal_evidence_count),
                "bore_wall_interior_margin": float(interior_margin),
                "bore_wall_broad_radial_min": float(broad_radial_min),
                "bore_wall_broad_radial_max": float(broad_radial_max),
                "learned_wall_radius": float(learned_wall_radius),
                "learned_wall_radius_source": str(learned_wall_radius_source),
                "learned_wall_radius_tolerance": float(learned_wall_radius_tol),
                "selected_opening_radius_authority_v150": float(selected_opening_radius_authority),
                "selected_opening_clean_radius_authority_v150": bool(selected_opening_clean_radius_authority),
                "selected_opening_radius_clamp_used_v150": bool(selected_opening_radius_clamp_used),
                "two_opening_radius_before_selected_clamp_v150": float(unclamped_two_opening_radius),
                "selected_opening_radius_delta_rel_to_frame_v150": float(selected_radius_delta_rel_to_frame),
                "opposite_radius_delta_rel_v150": float(opposite_radius_delta_rel),
                "selected_radius_band_tolerance_v150": float(selected_radius_band_tol),
                "radius_authority_policy_v150": (
                    "clean_selected_opening_radius_owns_wall_scale_opposite_loop_depth_only"
                    if selected_opening_radius_clamp_used
                    else "two_opening_frame_radius_unmodified"
                ),
                "face_ids": tuple(comp_ids),
                "bore_wall_component_rejection_reasons": tuple(rejection_reasons),
                "bore_wall_ownership_reject_reason": "" if valid_wall else (str(rejection_reasons[0]) if rejection_reasons else "component_failed_two_opening_wall_ownership_gate"),
                "semantic_transform": "TwoOpeningBoreFrame -> BoreWallEvidence -> BoreSurfaceOwnership -> CandidateData",
                "score": float(score),
            }
            if valid_wall:
                selected_seed_related_components.append((score, comp_ids, st))
            else:
                bore_rejected.append(st)
                other_components.append((score, comp_ids, st))
        selected_seed_related_components.sort(key=lambda item: (-item[0], -len(item[1]), item[1][:1]))
        if selected_seed_related_components:
            score, comp_ids, st = selected_seed_related_components[0]
            bore_owned_face_count = int(len(comp_ids))
            best_bore_wall_diag = dict(st)
            confidence = max(0.25, min(0.94, 0.38 + 0.10 * float(score)))
            defect_boundary_count = int(region_diagnostics.get("boundary_loop_count", 0) or 0)

            # v1.6.7 semantic correction: damaged BORE endpoint authority is
            # a separate meaning stage after BoreWallOwnership.  The first
            # compatible opposite-opening ring can be an internal defect loop,
            # and owned wall span can stop short of the final endpoint.  Use
            # the local RegionData axial endpoint as a far-end candidate only
            # after wall ownership has proved a physical bore wall.
            try:
                _finite_face_min = face_axial_min[np.isfinite(face_axial_min)]
                _finite_face_max = face_axial_max[np.isfinite(face_axial_max)]
                _region_face_axial_min = float(np.nanmin(_finite_face_min)) if _finite_face_min.size else 0.0
                _region_face_axial_max = float(np.nanmax(_finite_face_max)) if _finite_face_max.size else float(bore_depth)
            except Exception:
                _region_face_axial_min = 0.0
                _region_face_axial_max = float(bore_depth)
            region_true_endpoint_depth = max(0.0, float(_region_face_axial_max))

            # v1.6.6 semantic correction: the initially measured opposite
            # opening can be an internal defect ring in damaged bores.  Once
            # BoreWallOwnership is valid, the operational frame must be
            # reconciled from the owned wall face-span rather than letting the
            # earlier provisional opposite-opening candidate cap the depth.
            measured_frame_depth = float(bore_depth)
            owned_axial_min = _safe_float(st.get("axial_min", 0.0), 0.0)
            owned_axial_max = _safe_float(st.get("axial_max", measured_frame_depth), measured_frame_depth)
            owned_axial_span = _safe_float(st.get("axial_span", measured_frame_depth), measured_frame_depth)
            owned_depth_candidate = max(float(measured_frame_depth), float(owned_axial_max), float(owned_axial_span))
            # v1.6.8 semantic correction: true endpoint authority is not a
            # sidewall-extension bonus and must not be defeated by an edge-scale
            # threshold.  v1.6.7 correctly measured the RegionData endpoint, but
            # rejected a real 4.166... endpoint gap because the gate used
            # ``2.0 * edge_scale`` as if the endpoint were only an optional face
            # extension.  Here the meaning is explicit:
            #
            #   BoreWallOwnership valid + damaged internal boundary evidence
            #   + a radius/axis-compatible RegionData far endpoint beyond the
            #   owned wall span => operational true endpoint authority.
            #
            # Owned wall span remains support evidence; it is not allowed to cap
            # the physical damaged-bore depth once the true endpoint is measured.
            true_endpoint_depth_candidate = float(max(float(owned_depth_candidate), float(region_true_endpoint_depth)))
            true_endpoint_gap = float(true_endpoint_depth_candidate - float(owned_depth_candidate))
            true_endpoint_reconcile_epsilon = max(0.004 * max(float(true_endpoint_depth_candidate), 1.0), 0.10)
            true_endpoint_max_extension = max(0.18 * max(float(true_endpoint_depth_candidate), 1.0), 14.0 * edge_scale, 7.50)

            # v153: full-depth continuation for clean selected rims.
            # v152 still allowed the mismatched opposite-opening frame to cap the
            # operational BORE depth at the first opposite candidate (~49.7) even
            # when RegionData and X1 probe evidence showed a same-radius wall
            # continuing toward the true endpoint (~100).  For a clean selected
            # opening, a large opposite radius mismatch means the opposite loop is
            # depth/direction evidence at best; it must not be the final endpoint
            # authority.  In that case, the RegionData axial endpoint becomes the
            # operational depth candidate, provided it is beyond the owned wall
            # span and still within a sane feature-local extension envelope.
            opposite_radius_mismatch_for_depth_v153 = bool(
                selected_opening_clean_radius_authority
                and (
                    opposite_radius_delta_rel > 0.18
                    or selected_opening_radius_clamp_used
                    or selected_radius_delta_rel_to_frame > 0.18
                )
            )
            clean_selected_endpoint_reconcile_allowed_v153 = bool(
                opposite_radius_mismatch_for_depth_v153
                and region_true_endpoint_depth > owned_depth_candidate + true_endpoint_reconcile_epsilon
                and region_true_endpoint_depth > measured_frame_depth + true_endpoint_reconcile_epsilon
            )
            # Keep the old diagnostic key, but make the v153 meaning explicit.
            clean_selected_endpoint_reconcile_allowed_v152 = bool(clean_selected_endpoint_reconcile_allowed_v153)
            true_endpoint_max_extension_v153 = max(
                float(true_endpoint_max_extension),
                0.65 * max(float(true_endpoint_depth_candidate), 1.0),
                24.0 * edge_scale,
                12.0,
            )
            true_endpoint_reconcile_gate_v152 = (
                "internal_defect_boundary_gate" if defect_boundary_count > 2
                else ("clean_selected_opening_opposite_radius_mismatch_full_depth_v153" if clean_selected_endpoint_reconcile_allowed_v153 else "not_allowed")
            )
            true_endpoint_reconciled = bool(
                (
                    defect_boundary_count > 2
                    or clean_selected_endpoint_reconcile_allowed_v153
                )
                and true_endpoint_gap > true_endpoint_reconcile_epsilon
                and (
                    true_endpoint_gap <= true_endpoint_max_extension
                    or (clean_selected_endpoint_reconcile_allowed_v153 and true_endpoint_gap <= true_endpoint_max_extension_v153)
                )
            )
            operational_depth_candidate = float(true_endpoint_depth_candidate if true_endpoint_reconciled else owned_depth_candidate)
            owned_frame_depth = float(measured_frame_depth)
            owned_frame_reconciled = False
            owned_opposite_center = None
            owned_frame = dict(two_opening_bore_frame)
            depth_reconcile_threshold = max(0.08 * max(float(measured_frame_depth), 1.0), 2.0 * edge_scale, 0.75)
            if true_endpoint_reconciled and operational_depth_candidate > float(measured_frame_depth) + depth_reconcile_threshold:
                owned_frame_depth = float(operational_depth_candidate)
                try:
                    opening_np = np.asarray(two_opening_bore_frame.get("opening_center", bore_center), dtype=float).reshape(3)
                    axis_np = np.asarray(bore_axis, dtype=float).reshape(3)
                    axis_len = float(np.linalg.norm(axis_np))
                    if np.isfinite(axis_len) and axis_len > 1.0e-12:
                        axis_np = axis_np / axis_len
                    owned_opposite_center = opening_np + axis_np * float(owned_frame_depth)
                    owned_frame.update({
                        "opposite_center": to_vector3(owned_opposite_center),
                        "depth": float(owned_frame_depth),
                        "diameter": float(2.0 * bore_radius),
                        "radius": float(bore_radius),
                        "axis": to_vector3(axis_np),
                        "status": "owned_bore_frame_reconciled_from_bore_wall_ownership",
                        "semantic_stage": "BoreWallOwnership_to_OperationalOwnedBoreFrame",
                        "measured_two_opening_frame_depth": float(measured_frame_depth),
                        "owned_wall_face_span_depth": float(owned_axial_span),
                        "owned_wall_axial_min": float(owned_axial_min),
                        "owned_wall_axial_max": float(owned_axial_max),
                        "owned_frame_depth_delta": float(float(owned_frame_depth) - float(measured_frame_depth)),
                        "region_true_endpoint_depth": float(region_true_endpoint_depth),
                        "true_endpoint_depth_candidate": float(true_endpoint_depth_candidate),
                        "true_endpoint_depth_gap": float(true_endpoint_gap),
                        "true_endpoint_reconcile_epsilon": float(true_endpoint_reconcile_epsilon),
                        "true_endpoint_max_extension": float(true_endpoint_max_extension),
                        "true_endpoint_reconciled_from_region_endpoint": bool(true_endpoint_reconciled),
                        "true_endpoint_reconcile_gate_v152": str(true_endpoint_reconcile_gate_v152),
                        "clean_selected_endpoint_reconcile_allowed_v152": bool(clean_selected_endpoint_reconcile_allowed_v152),
                        "clean_selected_endpoint_reconcile_allowed_v153": bool(clean_selected_endpoint_reconcile_allowed_v153),
                        "opposite_radius_mismatch_for_depth_v153": bool(opposite_radius_mismatch_for_depth_v153),
                        "true_endpoint_max_extension_v153": float(true_endpoint_max_extension_v153),
                        "opposite_radius_delta_rel_v152": float(opposite_radius_delta_rel),
                    })
                    owned_frame_reconciled = True
                except Exception:
                    owned_frame_depth = float(measured_frame_depth)
                    owned_frame_reconciled = False
                    owned_frame = dict(two_opening_bore_frame)

            # v1.7.0 semantic correction: endpoint authority is now valid,
            # but CandidateData must also own the terminal cylindrical wall band.
            # v1.6.9 kept endpoint-support faces diagnostic-only, restoring the
            # rebuild, but left BoreWallOwnership physically short: the primitive
            # said depth=100 while display/rebuild ownership stopped at ~95.83.
            #
            # Do NOT merge broad endpoint support/rim/cut faces.  Complete only
            # the missing terminal BORE WALL role by a strict role filter:
            #
            #   core comp_ids            -> already owned cylindrical wall
            #   endpoint_support_face_ids -> broad diagnostic endpoint evidence
            #   terminal_wall_face_ids    -> true terminal cylindrical sidewall
            #   full_depth_wall_face_ids  -> CandidateData ownership/rebuild patch
            #
            # Terminal-wall promotion is allowed only when it is radius-consistent,
            # has wall-like normals, lies in the true-endpoint axial gap, and is
            # connected to or immediately contiguous with the already-owned wall.
            core_bore_wall_owned_face_ids = tuple_ints(comp_ids)
            bore_wall_owned_face_ids = tuple_ints(core_bore_wall_owned_face_ids)
            endpoint_extension_added_face_count = 0
            endpoint_support_face_ids: tuple[int, ...] = ()
            terminal_wall_face_ids: tuple[int, ...] = ()
            terminal_wall_candidate_face_ids: tuple[int, ...] = ()
            terminal_wall_completion_used = False
            terminal_wall_rejection_reasons: list[str] = []
            if bool(true_endpoint_reconciled):
                try:
                    endpoint_band_start = max(float(owned_depth_candidate) - axial_tol, float(bore_axial_min) - axial_tol)
                    endpoint_band_stop = float(owned_frame_depth) + axial_tol
                    endpoint_between = finite & (face_axial_max >= endpoint_band_start) & (face_axial_min <= endpoint_band_stop)

                    # Broad endpoint evidence is kept for diagnostics only.  It
                    # intentionally includes rim/cut/support faces and therefore
                    # must never become BoreWallOwnership by itself.
                    endpoint_radius_tol = max(float(learned_wall_radius_tol), 0.18 * max(float(learned_wall_radius), 1.0), 4.0 * edge_scale, 0.45)
                    endpoint_radius_support = (
                        (np.abs(radial - float(learned_wall_radius)) <= endpoint_radius_tol)
                        | ((face_radial_min <= float(learned_wall_radius) + endpoint_radius_tol)
                           & (face_radial_max >= float(learned_wall_radius) - endpoint_radius_tol))
                    )
                    endpoint_support_mask = endpoint_between & endpoint_radius_support
                    endpoint_support_face_ids = tuple(int(fid) for fid, keep in zip(ids, endpoint_support_mask) if bool(keep))

                    # True terminal wall completion: stricter than endpoint support.
                    # Use centroid radius and wall-normal role to reject cap panels,
                    # broad exterior slabs, and diagonal closure triangles that only
                    # cross the cylinder radius at a vertex.
                    terminal_gap = max(float(owned_frame_depth) - float(owned_depth_candidate), 0.0)
                    terminal_band_start = max(float(owned_axial_max) - max(1.5 * edge_scale, 0.25), float(bore_axial_min) - axial_tol)
                    terminal_band_stop = float(owned_frame_depth) + max(1.5 * edge_scale, 0.25)
                    terminal_between = finite & (face_axial_max >= terminal_band_start) & (face_axial_min <= terminal_band_stop)
                    terminal_centroid_radius = np.abs(radial - float(learned_wall_radius)) <= float(learned_wall_radius_tol)
                    terminal_radial_span_limit = max(0.35 * max(float(learned_wall_radius), 1.0), 5.0 * edge_scale, 0.75)
                    terminal_radial_span_ok = (face_radial_max - face_radial_min) <= terminal_radial_span_limit
                    terminal_wall_mask = terminal_between & terminal_centroid_radius & sidewall_role_normal & terminal_radial_span_ok

                    core_set = {int(v) for v in core_bore_wall_owned_face_ids}
                    terminal_candidates = tuple(int(fid) for fid, keep in zip(ids, terminal_wall_mask) if bool(keep) and int(fid) not in core_set)
                    terminal_wall_candidate_face_ids = tuple_ints(terminal_candidates)

                    # Keep only terminal components that are continuous with the
                    # existing wall patch.  This prevents remote same-radius panels
                    # from being owned as BORE just because they lie near the endpoint.
                    connected_terminal: set[int] = set()
                    if terminal_wall_candidate_face_ids:
                        terminal_components = connected_face_components(faces, terminal_wall_candidate_face_ids)
                        continuity_axial_gap_limit = max(2.5 * edge_scale, 0.50, 0.08 * max(float(terminal_gap), 1.0))
                        for terminal_comp in terminal_components:
                            term_ids = tuple_ints(terminal_comp)
                            if not term_ids:
                                continue
                            term_set = {int(v) for v in term_ids}
                            touches_core_topology = any(
                                any(int(nb) in core_set for nb in adjacency.get(int(fid), ()))
                                for fid in term_set
                            )
                            term_local = np.asarray([fid_to_local[int(fid)] for fid in term_ids if int(fid) in fid_to_local], dtype=np.int64)
                            if term_local.size:
                                term_min = float(np.min(face_axial_min[term_local]))
                                term_max = float(np.max(face_axial_max[term_local]))
                                term_span = float(term_max - term_min)
                                term_reaches_endpoint = bool(term_max >= float(owned_frame_depth) - max(2.5 * edge_scale, 0.50))
                                term_near_owned_end = bool(term_min <= float(owned_axial_max) + continuity_axial_gap_limit)
                                term_strict_ratio = float(np.mean(strict_wall_band[term_local]))
                                term_normal_ratio = float(np.mean(sidewall_role_normal[term_local]))
                            else:
                                term_span = 0.0
                                term_reaches_endpoint = False
                                term_near_owned_end = False
                                term_strict_ratio = 0.0
                                term_normal_ratio = 0.0
                            # A component must either touch the core topology or
                            # bridge from the previous owned wall end toward the
                            # true endpoint with strong wall role support.
                            accept_component = bool(
                                (touches_core_topology or term_near_owned_end)
                                and term_reaches_endpoint
                                and term_span <= max(float(terminal_gap) + 3.0 * edge_scale, 1.0)
                                and term_strict_ratio >= 0.55
                                and term_normal_ratio >= 0.55
                            )
                            if accept_component:
                                connected_terminal.update(term_set)
                    terminal_wall_face_ids = tuple(sorted(connected_terminal))

                    # v153 fallback: when the selected rim is clean and the
                    # opposite loop was radius-mismatched, complete the missing
                    # same-radius terminal wall band even if the terminal pieces
                    # do not form one perfect topological component touching the
                    # core.  This is still not broad endpoint support: it requires
                    # selected-radius agreement, sidewall-normal role, and location
                    # in the axial continuation interval beyond the already-owned
                    # wall span.  It repairs the v152 failure where depth was
                    # measured but no terminal component passed the stricter
                    # continuity gate, leaving the candidate short.
                    if (
                        not terminal_wall_face_ids
                        and bool(clean_selected_endpoint_reconcile_allowed_v153)
                        and float(owned_frame_depth) > float(owned_depth_candidate) + max(2.0 * edge_scale, 0.50)
                    ):
                        v153_band_start = max(float(owned_axial_max) - max(2.0 * edge_scale, 0.50), float(bore_axial_min) - axial_tol)
                        v153_band_stop = float(owned_frame_depth) + max(2.0 * edge_scale, 0.50)
                        v153_between = finite & (face_axial_max >= v153_band_start) & (face_axial_min <= v153_band_stop)
                        v153_radius_tol = max(float(learned_wall_radius_tol), 0.14 * max(float(learned_wall_radius), 1.0), 4.0 * edge_scale, 0.50)
                        v153_same_radius = np.abs(radial - float(learned_wall_radius)) <= float(v153_radius_tol)
                        v153_wall_mask = v153_between & v153_same_radius & sidewall_role_normal
                        v153_candidates = tuple(
                            int(fid)
                            for fid, keep in zip(ids, v153_wall_mask)
                            if bool(keep) and int(fid) not in core_set
                        )
                        v153_terminal: set[int] = set()
                        if v153_candidates:
                            v153_components = connected_face_components(faces, v153_candidates)
                            v153_gap_limit = max(4.0 * edge_scale, 1.00, 0.12 * max(float(owned_frame_depth - owned_depth_candidate), 1.0))
                            for v153_comp in v153_components:
                                v153_ids = tuple_ints(v153_comp)
                                if not v153_ids:
                                    continue
                                v153_set = {int(v) for v in v153_ids}
                                v153_local = np.asarray([fid_to_local[int(fid)] for fid in v153_ids if int(fid) in fid_to_local], dtype=np.int64)
                                if not v153_local.size:
                                    continue
                                v153_min = float(np.min(face_axial_min[v153_local]))
                                v153_max = float(np.max(face_axial_max[v153_local]))
                                v153_span = float(v153_max - v153_min)
                                v153_touches_core = any(
                                    any(int(nb) in core_set for nb in adjacency.get(int(fid), ()))
                                    for fid in v153_set
                                )
                                v153_near_core_end = bool(v153_min <= float(owned_axial_max) + v153_gap_limit)
                                v153_reaches_deeper = bool(v153_max >= float(owned_axial_max) + max(1.5 * edge_scale, 0.50))
                                v153_role_ratio = float(np.mean(sidewall_role_normal[v153_local]))
                                v153_radius_ratio = float(np.mean(v153_same_radius[v153_local]))
                                if (
                                    (v153_touches_core or v153_near_core_end)
                                    and v153_reaches_deeper
                                    and v153_span <= max(float(owned_frame_depth - owned_axial_max) + 8.0 * edge_scale, 2.0)
                                    and v153_role_ratio >= 0.50
                                    and v153_radius_ratio >= 0.65
                                ):
                                    v153_terminal.update(v153_set)
                        if v153_terminal:
                            terminal_wall_face_ids = tuple(sorted(v153_terminal))
                            terminal_wall_rejection_reasons.append("v153_same_radius_terminal_continuation_used")
                        else:
                            terminal_wall_rejection_reasons.append("v153_same_radius_terminal_continuation_found_no_component")

                    if terminal_wall_face_ids:
                        merged_wall_ids = tuple(sorted(core_set | {int(v) for v in terminal_wall_face_ids}))
                        if len(merged_wall_ids) > len(core_bore_wall_owned_face_ids):
                            bore_wall_owned_face_ids = tuple_ints(merged_wall_ids)
                            endpoint_extension_added_face_count = int(len(bore_wall_owned_face_ids) - len(core_bore_wall_owned_face_ids))
                            terminal_wall_completion_used = True
                    else:
                        terminal_wall_rejection_reasons.append("no_terminal_wall_component_passed_strict_role_continuity_gate")
                except Exception as _terminal_exc:
                    endpoint_support_face_ids = ()
                    terminal_wall_face_ids = ()
                    terminal_wall_candidate_face_ids = ()
                    endpoint_extension_added_face_count = 0
                    terminal_wall_completion_used = False
                    terminal_wall_rejection_reasons.append(f"terminal_wall_completion_exception:{_terminal_exc}")
            # v154 diagnostic-only audit: prove whether the full-depth terminal
            # continuation faces are truly coaxial with the selected-rim cylinder.
            # This does not change ownership.  It records the geometric quality of
            # the terminal faces that v153 added so future patches can distinguish
            # "full depth and coaxial" from "full depth but broad support".
            terminal_wall_coaxial_audit: dict[str, object] = {
                "terminal_continuation_audit_version_v154": "terminal_continuation_coaxial_audit_v154",
                "terminal_continuation_face_count": int(len(tuple(terminal_wall_face_ids or ()))),
                "terminal_continuation_candidate_face_count": int(len(tuple(terminal_wall_candidate_face_ids or ()))),
                "terminal_continuation_connected_to_core_wall": False,
                "terminal_continuation_component_count": 0,
                "terminal_continuation_touch_core_face_count": 0,
                "terminal_continuation_axial_min": 0.0,
                "terminal_continuation_axial_max": 0.0,
                "terminal_continuation_axial_span": 0.0,
                "terminal_continuation_radius_min": 0.0,
                "terminal_continuation_radius_median": 0.0,
                "terminal_continuation_radius_max": 0.0,
                "terminal_continuation_radius_error_median": 0.0,
                "terminal_continuation_radius_error_max": 0.0,
                "terminal_continuation_radial_mad": 0.0,
                "terminal_continuation_radial_mad_rel": 0.0,
                "terminal_continuation_radial_align_median": 0.0,
                "terminal_continuation_normal_axis_abs_median": 1.0,
                "terminal_continuation_centerline_distance_median": 0.0,
                "terminal_continuation_centerline_distance_max": 0.0,
                "terminal_continuation_axis_dot_to_selected_opening": 0.0,
                "terminal_continuation_passed_coaxial_audit": False,
                "terminal_continuation_coaxial_audit_reason": "no_terminal_continuation_faces",
                # v155 diagnostic plumbing: make the suspected "outside rim"
                # condition measurable instead of relying on visual inspection.
                # These are diagnostics only; ownership is unchanged in v155.
                "terminal_continuation_angular_coverage": 0.0,
                "terminal_continuation_max_angular_gap_degrees": 360.0,
                "terminal_continuation_circumferential_sample_count": 0,
                "terminal_continuation_selected_radius_authority": float(bore_radius),
                "terminal_continuation_radius_gate_tolerance": 0.0,
                "terminal_continuation_outside_radius_gate_count": 0,
                "terminal_continuation_outside_radius_gate_ratio": 0.0,
                "terminal_continuation_outside_radius_gate_max_error": 0.0,
                "terminal_continuation_boundary_audit_reason_v155": "no_terminal_continuation_faces",
            }
            try:
                terminal_ids_for_audit = tuple_ints(terminal_wall_face_ids or ())
                if terminal_ids_for_audit:
                    term_local = np.asarray(
                        [fid_to_local[int(fid)] for fid in terminal_ids_for_audit if int(fid) in fid_to_local],
                        dtype=np.int64,
                    )
                    term_local = term_local[(term_local >= 0) & (term_local < len(radial))]
                    terminal_set_for_audit = {int(fid) for fid in terminal_ids_for_audit}
                    connected_to_core = any(
                        any(int(nb) in core_set for nb in adjacency.get(int(fid), ()))
                        for fid in terminal_set_for_audit
                    )
                    touch_core_face_count = sum(
                        1
                        for fid in terminal_set_for_audit
                        if any(int(nb) in core_set for nb in adjacency.get(int(fid), ()))
                    )
                    try:
                        terminal_component_count = len(connected_face_components(faces, terminal_ids_for_audit))
                    except Exception:
                        terminal_component_count = 0
                    selected_axis_for_audit = np.asarray(getattr(region_frame, "axis", bore_axis), dtype=float).reshape(3)
                    selected_axis_norm = float(np.linalg.norm(selected_axis_for_audit))
                    if not np.isfinite(selected_axis_norm) or selected_axis_norm <= 1.0e-12:
                        selected_axis_for_audit = np.asarray(bore_axis, dtype=float).reshape(3)
                        selected_axis_norm = float(np.linalg.norm(selected_axis_for_audit))
                    if np.isfinite(selected_axis_norm) and selected_axis_norm > 1.0e-12:
                        selected_axis_for_audit = selected_axis_for_audit / selected_axis_norm
                    else:
                        selected_axis_for_audit = np.asarray(bore_axis, dtype=float).reshape(3)
                    axis_dot_selected = float(abs(np.dot(np.asarray(bore_axis, dtype=float).reshape(3), selected_axis_for_audit)))
                    if term_local.size:
                        term_face_min = face_axial_min[term_local]
                        term_face_max = face_axial_max[term_local]
                        term_radial = radial[term_local]
                        term_radius_error = np.abs(term_radial - float(bore_radius))
                        term_radial_mad_abs = float(np.median(term_radius_error)) if term_radius_error.size else 0.0
                        term_normal_axis = normal_axis_abs[term_local]
                        term_radial_align = radial_normal_alignment[term_local]
                        term_axial_min = float(np.min(term_face_min))
                        term_axial_max = float(np.max(term_face_max))
                        term_axial_span = float(term_axial_max - term_axial_min)
                        term_radius_min = float(np.min(term_radial))
                        term_radius_median = float(np.median(term_radial))
                        term_radius_max = float(np.max(term_radial))
                        term_radius_error_max = float(np.max(term_radius_error)) if term_radius_error.size else 0.0
                        term_radial_mad_rel = float(term_radial_mad_abs / max(float(bore_radius), 1.0e-9))
                        term_radial_align_median = float(np.median(term_radial_align)) if term_radial_align.size else 0.0
                        term_normal_axis_median = float(np.median(term_normal_axis)) if term_normal_axis.size else 1.0

                        # v155: selected-rim boundary/coaxial envelope audit.
                        # The selected rim is the radius authority; terminal
                        # continuation faces may extend axially, but they should
                        # not drift outside that cylindrical envelope.
                        selected_radius_for_gate = float(selected_opening_radius_authority) if np.isfinite(float(selected_opening_radius_authority)) and float(selected_opening_radius_authority) > 1.0e-9 else float(bore_radius)
                        selected_radius_gate_tol = max(
                            0.16 * max(selected_radius_for_gate, 1.0),
                            6.0 * float(edge_scale),
                            0.55,
                        )
                        term_selected_radius_error = np.abs(term_radial - selected_radius_for_gate)
                        term_outside_radius_gate = term_selected_radius_error > selected_radius_gate_tol
                        term_outside_radius_gate_count = int(np.count_nonzero(term_outside_radius_gate))
                        term_outside_radius_gate_ratio = float(term_outside_radius_gate_count / max(int(term_selected_radius_error.size), 1))
                        term_outside_radius_gate_max_error = float(np.max(term_selected_radius_error)) if term_selected_radius_error.size else 0.0

                        term_angular_coverage = 0.0
                        term_max_gap_degrees = 360.0
                        term_circumferential_sample_count = int(term_local.size)
                        try:
                            audit_axis = np.asarray(selected_axis_for_audit, dtype=float).reshape(3)
                            ref = np.array([1.0, 0.0, 0.0], dtype=float)
                            if abs(float(np.dot(ref, audit_axis))) > 0.90:
                                ref = np.array([0.0, 1.0, 0.0], dtype=float)
                            basis0 = ref - audit_axis * float(np.dot(ref, audit_axis))
                            b0n = float(np.linalg.norm(basis0))
                            if np.isfinite(b0n) and b0n > 1.0e-12:
                                basis0 = basis0 / b0n
                                basis1 = np.cross(audit_axis, basis0)
                                b1n = float(np.linalg.norm(basis1))
                                if np.isfinite(b1n) and b1n > 1.0e-12:
                                    basis1 = basis1 / b1n
                                    term_dirs = radial_dir[term_local]
                                    term_theta = np.arctan2(term_dirs @ basis1, term_dirs @ basis0)
                                    term_theta = term_theta[np.isfinite(term_theta)]
                                    if term_theta.size >= 3:
                                        term_theta = np.sort(term_theta)
                                        gaps = np.diff(np.concatenate([term_theta, term_theta[:1] + 2.0 * math.pi]))
                                        if gaps.size:
                                            max_gap = float(np.max(gaps))
                                            term_angular_coverage = float(max(0.0, min(1.0, (2.0 * math.pi - max_gap) / (2.0 * math.pi))))
                                            term_max_gap_degrees = float(math.degrees(max_gap))
                        except Exception:
                            term_angular_coverage = 0.0
                            term_max_gap_degrees = 360.0

                        boundary_reasons = []
                        if term_outside_radius_gate_ratio > 0.05:
                            boundary_reasons.append("terminal_faces_outside_selected_rim_radius_gate")
                        if term_angular_coverage < 0.45:
                            boundary_reasons.append("terminal_angular_coverage_low_possible_local_exterior_strip")
                        if not connected_to_core:
                            boundary_reasons.append("terminal_faces_not_topologically_connected_to_core_wall")
                        boundary_reason_v155 = ";".join(boundary_reasons) or "terminal_faces_within_selected_rim_cylindrical_envelope"

                        # Face-center distance to the selected/opening axis.  For
                        # cylinder walls this should stay close to bore_radius;
                        # radius_error_* above is the actual coaxial-surface error.
                        term_centerline_median = float(np.median(term_radial)) if term_radial.size else 0.0
                        term_centerline_max = float(np.max(term_radial)) if term_radial.size else 0.0
                        term_passed = bool(
                            term_radial_mad_rel <= 0.12
                            and term_radius_error_max <= max(0.35 * float(bore_radius), 6.0 * edge_scale, 1.0)
                            and term_radial_align_median >= 0.55
                            and term_normal_axis_median <= 0.35
                            and axis_dot_selected >= 0.995
                            and term_axial_span > max(2.0 * edge_scale, 0.50)
                        )
                        if term_passed:
                            audit_reason = "terminal_faces_coaxial_with_selected_rim_cylinder"
                        else:
                            reasons = []
                            if term_radial_mad_rel > 0.12:
                                reasons.append("terminal_radial_mad_too_high")
                            if term_radius_error_max > max(0.35 * float(bore_radius), 6.0 * edge_scale, 1.0):
                                reasons.append("terminal_radius_error_max_too_high")
                            if term_radial_align_median < 0.55:
                                reasons.append("terminal_radial_alignment_weak")
                            if term_normal_axis_median > 0.35:
                                reasons.append("terminal_normals_not_sidewall_like")
                            if axis_dot_selected < 0.995:
                                reasons.append("terminal_axis_not_parallel_to_selected_opening")
                            if term_axial_span <= max(2.0 * edge_scale, 0.50):
                                reasons.append("terminal_axial_span_too_small")
                            audit_reason = ";".join(reasons) or "terminal_coaxial_audit_failed_unspecified"
                        terminal_wall_coaxial_audit.update({
                            "terminal_continuation_connected_to_core_wall": bool(connected_to_core),
                            "terminal_continuation_component_count": int(terminal_component_count),
                            "terminal_continuation_touch_core_face_count": int(touch_core_face_count),
                            "terminal_continuation_axial_min": float(term_axial_min),
                            "terminal_continuation_axial_max": float(term_axial_max),
                            "terminal_continuation_axial_span": float(term_axial_span),
                            "terminal_continuation_radius_min": float(term_radius_min),
                            "terminal_continuation_radius_median": float(term_radius_median),
                            "terminal_continuation_radius_max": float(term_radius_max),
                            "terminal_continuation_radius_error_median": float(term_radial_mad_abs),
                            "terminal_continuation_radius_error_max": float(term_radius_error_max),
                            "terminal_continuation_radial_mad": float(term_radial_mad_abs),
                            "terminal_continuation_radial_mad_rel": float(term_radial_mad_rel),
                            "terminal_continuation_radial_align_median": float(term_radial_align_median),
                            "terminal_continuation_normal_axis_abs_median": float(term_normal_axis_median),
                            "terminal_continuation_centerline_distance_median": float(term_centerline_median),
                            "terminal_continuation_centerline_distance_max": float(term_centerline_max),
                            "terminal_continuation_axis_dot_to_selected_opening": float(axis_dot_selected),
                            "terminal_continuation_passed_coaxial_audit": bool(term_passed),
                            "terminal_continuation_coaxial_audit_reason": str(audit_reason),
                            "terminal_continuation_angular_coverage": float(term_angular_coverage),
                            "terminal_continuation_max_angular_gap_degrees": float(term_max_gap_degrees),
                            "terminal_continuation_circumferential_sample_count": int(term_circumferential_sample_count),
                            "terminal_continuation_selected_radius_authority": float(selected_radius_for_gate),
                            "terminal_continuation_radius_gate_tolerance": float(selected_radius_gate_tol),
                            "terminal_continuation_outside_radius_gate_count": int(term_outside_radius_gate_count),
                            "terminal_continuation_outside_radius_gate_ratio": float(term_outside_radius_gate_ratio),
                            "terminal_continuation_outside_radius_gate_max_error": float(term_outside_radius_gate_max_error),
                            "terminal_continuation_boundary_audit_reason_v155": str(boundary_reason_v155),
                        })
                    else:
                        terminal_wall_coaxial_audit.update({
                            "terminal_continuation_coaxial_audit_reason": "terminal_faces_not_present_in_region_index",
                            "terminal_continuation_axis_dot_to_selected_opening": float(axis_dot_selected),
                            "terminal_continuation_component_count": int(terminal_component_count),
                            "terminal_continuation_connected_to_core_wall": bool(connected_to_core),
                            "terminal_continuation_touch_core_face_count": int(touch_core_face_count),
                        })
            except Exception as _terminal_audit_exc:
                terminal_wall_coaxial_audit.update({
                    "terminal_continuation_coaxial_audit_reason": f"terminal_coaxial_audit_exception:{_terminal_audit_exc}",
                    "terminal_continuation_passed_coaxial_audit": False,
                })

            # v173w/v174d: Terminal continuation is only evidence until its own
            # BORE_WALL role audit passes.  v174d extracts the existing v173w
            # filter into a BORE-local helper without changing ownership behavior.
            terminal_filter_v174d = filter_bore_terminal_continuation_ownership_v174h(
                core_bore_wall_owned_face_ids=core_bore_wall_owned_face_ids,
                bore_wall_owned_face_ids=bore_wall_owned_face_ids,
                terminal_wall_face_ids=terminal_wall_face_ids,
                terminal_wall_completion_used=bool(terminal_wall_completion_used),
                endpoint_extension_added_face_count=int(endpoint_extension_added_face_count),
                terminal_wall_rejection_reasons=terminal_wall_rejection_reasons,
                terminal_wall_coaxial_audit=terminal_wall_coaxial_audit,
            )
            bore_wall_owned_face_ids = tuple_ints(terminal_filter_v174d.get("bore_wall_owned_face_ids", bore_wall_owned_face_ids))
            terminal_wall_face_ids = tuple_ints(terminal_filter_v174d.get("terminal_wall_face_ids", terminal_wall_face_ids))
            terminal_wall_completion_used = bool(terminal_filter_v174d.get("terminal_wall_completion_used", terminal_wall_completion_used))
            endpoint_extension_added_face_count = int(terminal_filter_v174d.get("endpoint_extension_added_face_count", endpoint_extension_added_face_count) or 0)
            terminal_wall_rejection_reasons = list(terminal_filter_v174d.get("terminal_wall_rejection_reasons", terminal_wall_rejection_reasons) or ())
            terminal_wall_coaxial_audit = dict(terminal_filter_v174d.get("terminal_wall_coaxial_audit", terminal_wall_coaxial_audit) or {})
            terminal_continuation_removed_by_v173w = bool(terminal_filter_v174d.get("terminal_continuation_removed_by_v173w", False))
            terminal_continuation_removed_count_v173w = int(terminal_filter_v174d.get("terminal_continuation_removed_count_v173w", 0) or 0)
            terminal_continuation_removed_reason_v173w = str(terminal_filter_v174d.get("terminal_continuation_removed_reason_v173w", "") or "")

            comp_ids = tuple_ints(bore_wall_owned_face_ids)
            bore_owned_face_count = int(len(comp_ids))

            # v158: resolve over-selected BORE wall ownership before rebuild
            # authorization.  If full-depth/terminal evidence introduced an
            # unexplained extra boundary loop, split it back to the largest
            # semantic inside-wall subset that is actually bounded by the two
            # measured openings.  This repairs CandidateData face ownership; it
            # is not a radius tolerance or parameter-fit shortcut.
            damaged_internal_boundary_evidence_pre_v158 = bool(defect_boundary_count > 2)
            semantic_subset_v158 = select_bore_wall_semantic_owned_subset_v174i(
                faces=faces,
                owned_face_ids=comp_ids,
                core_face_ids=core_bore_wall_owned_face_ids,
                terminal_face_ids=terminal_wall_face_ids,
                damaged_internal_boundary_evidence=damaged_internal_boundary_evidence_pre_v158,
            )
            comp_ids = tuple_ints(semantic_subset_v158.get("face_ids", comp_ids))
            if bool(semantic_subset_v158.get("semantic_subset_used_v158", False)):
                bore_wall_owned_face_ids = comp_ids
                terminal_wall_face_ids = tuple_ints(fid for fid in tuple(terminal_wall_face_ids or ()) if int(fid) in set(comp_ids))
                endpoint_extension_added_face_count = max(0, int(len(comp_ids) - len(core_bore_wall_owned_face_ids)))
            bore_owned_face_count = int(len(comp_ids))

            # v157 semantic ownership proof: CandidateData may only expose a
            # rebuildable BORE target when the owned face set means "inside bore
            # wall between the measured openings".  Cylinder/radius evidence is
            # not enough.  The owned patch must also have the expected boundary
            # meaning: two opening loops, unless the RegionData already contains
            # explicit damaged/internal-boundary evidence.  This catches the v155
            # failure where full-depth terminal completion produced a plausible
            # cylinder descriptor but the delete patch had 3 boundary loops and
            # rebuild later reported missing boundary edges.
            core_boundary_report_v157 = _face_patch_boundary_semantic_report_v174z(faces, core_bore_wall_owned_face_ids)
            terminal_boundary_report_v157 = _face_patch_boundary_semantic_report_v174z(faces, terminal_wall_face_ids)
            owned_boundary_report_v157 = _face_patch_boundary_semantic_report_v174z(faces, comp_ids)
            owned_boundary_loop_count_v157 = int(owned_boundary_report_v157.get("boundary_loop_count", 0) or 0)
            core_boundary_loop_count_v157 = int(core_boundary_report_v157.get("boundary_loop_count", 0) or 0)
            terminal_boundary_loop_count_v157 = int(terminal_boundary_report_v157.get("boundary_loop_count", 0) or 0)
            terminal_face_count_after_v158 = int(len(tuple(terminal_wall_face_ids or ())))
            semantic_subset_used_for_repair_v159 = bool(semantic_subset_v158.get("semantic_subset_used_v158", False))
            terminal_continuation_removed_by_v158_v159 = bool(
                terminal_wall_completion_used
                and semantic_subset_used_for_repair_v159
                and terminal_face_count_after_v158 == 0
            )
            damaged_internal_boundary_evidence_v157 = bool(defect_boundary_count > 2)
            expected_two_opening_wall_patch_v157 = bool(not damaged_internal_boundary_evidence_v157)
            owned_patch_boundary_role_valid_v157 = bool(
                (owned_boundary_loop_count_v157 == 2)
                or damaged_internal_boundary_evidence_v157
            )
            terminal_completion_role_valid_v157 = bool(
                not terminal_wall_completion_used
                or (
                    owned_patch_boundary_role_valid_v157
                    and terminal_continuation_removed_by_v158_v159
                )
                or (
                    owned_patch_boundary_role_valid_v157
                    and bool(terminal_wall_coaxial_audit.get("terminal_continuation_connected_to_core_wall", False))
                    and bool(terminal_wall_coaxial_audit.get("terminal_continuation_passed_coaxial_audit", False))
                )
            )
            bore_rebuild_semantic_target_safe_v157 = bool(
                owned_patch_boundary_role_valid_v157
                and terminal_completion_role_valid_v157
            )
            bore_rebuild_semantic_rejection_reasons_v157: list[str] = []
            if not owned_patch_boundary_role_valid_v157:
                bore_rebuild_semantic_rejection_reasons_v157.append("owned_bore_wall_patch_boundary_not_two_opening_role_v157")
            if terminal_wall_completion_used and not terminal_completion_role_valid_v157:
                bore_rebuild_semantic_rejection_reasons_v157.append("terminal_continuation_not_proven_inside_bore_wall_role_v157")

            bore_heuristic_results = _heuristic_results_for_current_bore_state_v174z(
                selected_edge_count=int(selected_edge_count),
                normalized_edge_count=int(normalized_edge_count),
                region_face_count=int(region_face_count),
                two_opening_valid=bool(two_opening_valid),
                bore_depth=float(bore_depth),
                bore_radius=float(bore_radius),
                sidewall_normal_evidence_count=int(sidewall_normal_evidence_count),
                frame_sidewall_normal_evidence_count=int(frame_sidewall_normal_evidence_count),
                bore_wall_search_face_count=int(len(bore_wall_search_face_ids)),
                bore_wall_candidate_component_count=int(len(bore_wall_candidate_components)),
                bore_wall_owned_face_count=int(bore_owned_face_count),
                bore_wall_rejected_component_count=int(len(bore_rejected)),
                bore_wall_component_rejection_reasons=tuple(str(v) for row in tuple(bore_rejected or ()) for v in tuple(row.get("bore_wall_component_rejection_reasons", ()) or ())),
                true_endpoint_reconciled=bool(true_endpoint_reconciled),
                region_true_endpoint_depth=float(region_true_endpoint_depth),
                terminal_wall_candidate_face_count=int(len(tuple(terminal_wall_candidate_face_ids or ()))),
                terminal_wall_face_count=int(len(tuple(terminal_wall_face_ids or ()))),
                terminal_wall_completion_used=bool(terminal_wall_completion_used),
                endpoint_extension_added_face_count=int(endpoint_extension_added_face_count),
                display_face_count=int(len(comp_ids)),
                rebuild_face_count=int(len(comp_ids)),
                # v1.7.3 regression fix:
                # This call happens while emitting the accepted BORE candidate,
                # before the CHAMFER recipe has run.  A BORE heuristic summary
                # must not read CHAMFER-local variables here: that created an
                # UnboundLocalError and collapsed Recognition before CandidateData
                # could be emitted.  The CHAMFER recipe reports in the top-level
                # diagnostics after chamfer_rows exists.
                chamfer_candidate_count=0,
                chamfer_promoted_count=0,
            )
            compact_bore_heuristics = compact_heuristic_summary(bore_heuristic_results)

            candidate_diag = {
                **st,
                "axis": to_vector3(bore_axis),
                "center": to_vector3(bore_center),
                "radius": float(bore_radius),
                "two_opening_bore_frame": dict(owned_frame),
                "measured_two_opening_bore_frame": dict(two_opening_bore_frame),
                "owned_bore_frame": dict(owned_frame),
                "measured_two_opening_frame_depth": float(measured_frame_depth),
                "owned_bore_frame_depth": float(owned_frame_depth),
                "operational_bore_frame_depth": float(owned_frame_depth),
                "owned_bore_frame_depth_delta": float(float(owned_frame_depth) - float(measured_frame_depth)),
                "owned_frame_reconciled_from_wall_ownership": bool(owned_frame_reconciled),
                "owned_wall_axial_min": float(owned_axial_min),
                "owned_wall_axial_max": float(owned_axial_max),
                "owned_wall_face_span_depth": float(owned_axial_span),
                "bore_frame_depth_source": "region_true_endpoint_depth_reconciled" if bool(true_endpoint_reconciled) else ("owned_wall_face_span_reconciled" if bool(owned_frame_reconciled) else "measured_two_opening_frame"),
                "region_true_endpoint_depth": float(region_true_endpoint_depth),
                "true_endpoint_depth_candidate": float(true_endpoint_depth_candidate),
                "true_endpoint_depth_gap": float(true_endpoint_gap),
                "true_endpoint_reconcile_epsilon": float(true_endpoint_reconcile_epsilon),
                "true_endpoint_max_extension": float(true_endpoint_max_extension),
                "true_endpoint_reconciled_from_region_endpoint": bool(true_endpoint_reconciled),
                "true_endpoint_reconcile_gate_v152": str(true_endpoint_reconcile_gate_v152),
                "clean_selected_endpoint_reconcile_allowed_v152": bool(clean_selected_endpoint_reconcile_allowed_v152),
                "clean_selected_endpoint_reconcile_allowed_v153": bool(clean_selected_endpoint_reconcile_allowed_v153),
                "opposite_radius_mismatch_for_depth_v153": bool(opposite_radius_mismatch_for_depth_v153),
                "true_endpoint_max_extension_v153": float(true_endpoint_max_extension_v153),
                "opposite_radius_delta_rel_v152": float(opposite_radius_delta_rel),
                "selected_opening_anchor_face_count_v152": int(len(selected_opening_anchor_face_set)),
                "candidate_face_source_v152": "accepted_seed_bound_component_plus_terminal_completion" if selected_opening_clean_radius_authority else "legacy_component_or_aggregate_wall_ownership",
                "core_owned_component_face_count_v152": int(len(core_bore_wall_owned_face_ids)),
                "final_owned_wall_face_count_v152": int(len(bore_wall_owned_face_ids)),
                "candidate_equals_core_component_v152": bool(len(bore_wall_owned_face_ids) == len(core_bore_wall_owned_face_ids)),
                "endpoint_support_face_count": int(len(tuple(endpoint_support_face_ids or ()))),
                "endpoint_support_face_ids": tuple_ints(endpoint_support_face_ids),
                "endpoint_support_promoted_to_wall_ownership": False,
                "endpoint_support_delete_patch_allowed": False,
                "terminal_wall_candidate_face_count": int(len(tuple(terminal_wall_candidate_face_ids or ()))),
                "terminal_wall_face_count": int(len(tuple(terminal_wall_face_ids or ()))),
                "terminal_wall_face_ids": tuple_ints(terminal_wall_face_ids),
                "terminal_wall_completion_used": bool(terminal_wall_completion_used),
                "terminal_wall_completion_rejection_reasons": tuple(terminal_wall_rejection_reasons),
                "terminal_continuation_removed_from_bore_ownership_v173w": bool(terminal_continuation_removed_by_v173w),
                "terminal_continuation_removed_face_count_v173w": int(terminal_continuation_removed_count_v173w),
                "terminal_continuation_removed_reason_v173w": str(terminal_continuation_removed_reason_v173w),
                "bore_terminal_continuation_ownership_rule_v173w": (
                    "Terminal continuation may support endpoint diagnosis, but it may not enter BORE_WALL ownership unless its own bore-wall-role audit passes."
                ),
                "true_endpoint_extension_added_face_count": int(endpoint_extension_added_face_count),
                "v154_terminal_continuation_coaxial_audit_used": True,
                **terminal_wall_coaxial_audit,
                "bore_wall_owned_face_count_before_terminal_completion": int(len(core_bore_wall_owned_face_ids)),
                "bore_wall_owned_face_count_after_terminal_completion": int(len(bore_wall_owned_face_ids)),
                "bore_wall_owned_face_ids": tuple_ints(bore_wall_owned_face_ids),
                "full_depth_display_face_count": int(len(comp_ids)),
                "candidate_face_policy": "full_depth_bore_wall_faces_terminal_role_filtered_endpoint_support_diagnostic",
                "bore_rebuild_semantic_target_safe_v157": bool(bore_rebuild_semantic_target_safe_v157),
                "bore_rebuild_semantic_rejection_reasons_v157": tuple(bore_rebuild_semantic_rejection_reasons_v157),
                "bore_wall_owned_boundary_loop_count_v157": int(owned_boundary_loop_count_v157),
                "bore_wall_owned_boundary_edge_count_v157": int(owned_boundary_report_v157.get("boundary_edge_count", 0) or 0),
                "bore_wall_owned_boundary_loop_vertex_counts_v157": tuple(owned_boundary_report_v157.get("boundary_loop_vertex_counts", ()) or ()),
                "bore_wall_owned_boundary_loop_edge_counts_v157": tuple(owned_boundary_report_v157.get("boundary_loop_edge_counts", ()) or ()),
                "bore_wall_core_boundary_loop_count_v157": int(core_boundary_loop_count_v157),
                "bore_wall_terminal_boundary_loop_count_v157": int(terminal_boundary_loop_count_v157),
                "bore_wall_expected_two_opening_patch_v157": bool(expected_two_opening_wall_patch_v157),
                "bore_wall_damaged_internal_boundary_evidence_v157": bool(damaged_internal_boundary_evidence_v157),
                "owned_patch_boundary_role_valid_v157": bool(owned_patch_boundary_role_valid_v157),
                "terminal_completion_role_valid_v157": bool(terminal_completion_role_valid_v157),
                "semantic_ownership_proof_v157": "inside_bore_wall_between_measured_openings_boundary_proof",
                "v158_semantic_owned_face_mapping_used": bool(semantic_subset_v158.get("semantic_subset_used_v158", False)),
                "v158_semantic_owned_face_mapping_source": str(semantic_subset_v158.get("semantic_subset_source_v158", "")),
                "v158_semantic_owned_face_removed_count": int(len(tuple(semantic_subset_v158.get("removed_face_ids", ()) or ()))),
                "v158_semantic_owned_face_removed_ids": tuple_ints(semantic_subset_v158.get("removed_face_ids", ())),
                "v158_semantic_owned_subset_report": dict(semantic_subset_v158.get("semantic_subset_report_v158", {}) or {}),
                "v158_semantic_owned_subset_candidate_reports": tuple(semantic_subset_v158.get("semantic_subset_candidate_reports_v158", ()) or ()),
                "v158_semantic_mapping_policy": "owned_bore_wall_candidate_faces_must_be_inside_wall_subset_bounded_by_measured_openings",
                "v159_repaired_bore_subset_reauthorized": bool(owned_patch_boundary_role_valid_v157 and terminal_continuation_removed_by_v158_v159),
                "v159_repaired_bore_subset_reauthorization_reason": (
                    "invalid_terminal_continuation_removed_repaired_subset_has_two_opening_boundary"
                    if bool(owned_patch_boundary_role_valid_v157 and terminal_continuation_removed_by_v158_v159)
                    else "not_reauthorized_by_v159"
                ),
                "v159_terminal_face_count_after_repair": int(terminal_face_count_after_v158),
                "v159_terminal_continuation_removed_from_rebuild_target": bool(terminal_continuation_removed_by_v158_v159),
                "recognition_rule": "v91_heuristic_contract_registry_full_depth_wall_ownership",
                "heuristic_contract_version": HEURISTIC_CONTRACT_VERSION,
                "heuristic_recipe_name": "BORE_HEURISTIC_RECIPE",
                "heuristic_recipe": BORE_HEURISTIC_RECIPE,
                "heuristic_recipe_contracts": recipe_contracts("bore"),
                "heuristic_results": bore_heuristic_results,
                "heuristic_result_summaries": compact_bore_heuristics,
                "heuristic_authority_policy": "heuristics_propose_measurement_quantifies_recognition_interprets_ownership_assigns",
        "heuristic_scope_fix_v1_7_3": "BORE CandidateData construction no longer reads CHAMFER-local heuristic variables before the CHAMFER recipe runs",
                "candidate_side_role": "selected_opening_to_owned_bore_wall_operational_frame",
                "two_opening_axis_direction_preserved": bool(two_opening_axis_direction_preserved),
                "two_opening_axis_dot_opening_to_opposite": float(two_opening_axis_dot_opening_to_opposite),
                "bore_axis_source": "directed_opening_to_opposite_axis_no_canonical_flip",
            }
            if not bore_rebuild_semantic_target_safe_v157:
                candidate_diag.update({
                    "bore_wall_ownership_reject_reason": ";".join(bore_rebuild_semantic_rejection_reasons_v157) or "bore_wall_semantic_boundary_proof_failed_v157",
                    "feature_ownership_source": "bore_wall_semantic_ownership_proof_failed_v157",
                    "feature_ownership_split": "preview_only_owned_faces_not_rebuild_target",
                })
                candidate = bore_review_candidate_contract_fields_v174k(
                    candidate_id="component_engine.v157.bore.wall_ownership_semantic_review.1",
                    face_ids=comp_ids,
                    confidence=float(confidence),
                    radius=float(bore_radius),
                    axial_span=float(owned_frame_depth),
                    diagnostics=candidate_diag,
                )
                candidate.update({
                    "display_name": "BORE — wall ownership semantic review",
                    "status": "bore_wall_ownership_not_rebuild_safe_semantic_boundary_proof_failed_v157",
                    "surface_condition": "owned_bore_wall_faces_do_not_prove_two_opening_wall_patch",
                    "repair_strategy": "none_review_only_until_owned_wall_boundary_matches_measured_openings",
                    "promotion_state": "evidence_only",
                    "candidate_action_enabled": False,
                    "candidate_action": "preview",
                    "rebuild_authorized": False,
                    "rebuild_face_ids": (),
                    "candidate_rebuild_face_ids": (),
                    "delete_patch_request_allowed": False,
                    "rebuild_target_policy_allowed": False,
                    "rebuild_target_policy_reason": "BORE preview only: owned wall patch failed v157 semantic boundary proof",
                    "rebuild_block_reason": ";".join(bore_rebuild_semantic_rejection_reasons_v157) or "bore_wall_semantic_boundary_proof_failed_v157",
                    "bore_rebuild_enable_scope": "disabled_until_inside_wall_boundary_proof_valid",
                })
            else:
                candidate = bore_candidate_contract_fields_v174k(
                    candidate_id="component_engine.v91.bore.two_opening_wall_ownership.1",
                    face_ids=comp_ids,
                    confidence=float(confidence),
                    radius=float(bore_radius),
                    axial_span=float(owned_frame_depth),
                    diagnostics=candidate_diag,
                )
                if defect_boundary_count > 2:
                    candidate.update({
                        "candidate_variant": "damaged_bore",
                        "display_name": "BORE — damaged two-opening wall rebuild trial",
                        "surface_condition": "two_opening_bore_wall_owned_faces_with_internal_defect_boundaries",
                        "status": "accepted_two_opening_damaged_bore_wall_ownership",
                        "damage_state": "damaged_bore_internal_boundary_defects",
                        "damage_reasons": ("extra_boundary_loops_inside_owned_bore_wall",),
                        "requires_damaged_bore_target_seal": True,
                        "bore_frame_depth_source": str(candidate_diag.get("bore_frame_depth_source", "measured_two_opening_frame")),
                        "owned_frame_reconciled_from_wall_ownership": bool(candidate_diag.get("owned_frame_reconciled_from_wall_ownership", False)),
                    })
                else:
                    candidate.update({
                        "status": "accepted_two_opening_bore_wall_ownership",
                        "requires_damaged_bore_target_seal": False,
                    })
            bore_features.append(candidate)
        else:
            # v1.5.2: a valid two-opening frame is itself visible BORE evidence.
            # If wall ownership is not accepted, emit a review-only BORE row
            # instead of letting CHAMFER be the only visible result.
            review_diag: dict[str, object] = {
                "axis": to_vector3(bore_axis),
                "center": to_vector3(bore_center),
                "radius": float(bore_radius),
                "two_opening_bore_frame": dict(two_opening_bore_frame),
                "two_opening_bore_frame_valid": True,
                "two_opening_bore_frame_depth": float(bore_depth),
                "two_opening_search_boundary_face_count": int(len(bore_tube_face_ids)),
                "bore_wall_search_face_count": int(len(bore_wall_search_face_ids)),
                "bore_wall_candidate_component_count": int(len(bore_wall_candidate_components)),
                "selected_opening_support_weak": bool(selected_opening_support_weak),
                "selected_opening_primary_edge_count": int(selected_opening_primary_edge_count),
                "bore_wall_ownership_reject_reason": "selected_opening_primary_component_too_weak_for_accepted_ownership" if selected_opening_support_weak else "no_connected_component_passed_two_opening_wall_ownership_gate",
                "sidewall_normal_role_ownership_used": True,
                "geometry_first_wall_ownership_used": False,
                "sidewall_normal_evidence_count": int(locals().get("sidewall_normal_evidence_count", 0)),
                "frame_sidewall_normal_evidence_count": int(locals().get("frame_sidewall_normal_evidence_count", 0)),
                "broad_sidewall_normal_evidence_count": int(locals().get("broad_sidewall_normal_evidence_count", 0)),
                "bore_wall_interior_margin": float(locals().get("interior_margin", 0.0)),
                "bore_wall_broad_radial_min": float(locals().get("broad_radial_min", 0.0)),
                "bore_wall_broad_radial_max": float(locals().get("broad_radial_max", 0.0)),
                "learned_wall_radius": float(locals().get("learned_wall_radius", bore_radius)),
                "learned_wall_radius_source": str(locals().get("learned_wall_radius_source", "two_opening_frame_radius")),
                "learned_wall_radius_tolerance": float(locals().get("learned_wall_radius_tol", 0.0)),
                "semantic_transform": "TwoOpeningBoreFrame -> SidewallNormalRoleEvidence(review) -> BoreSurfaceOwnership/CandidateData",
                "two_opening_axis_direction_preserved": bool(two_opening_axis_direction_preserved),
                "two_opening_axis_dot_opening_to_opposite": float(two_opening_axis_dot_opening_to_opposite),
                "bore_axis_source": "secondary_opening_recontextualized_axis_v175i" if bool(locals().get("secondary_opening_recontext_used_v175i", False)) else "directed_opening_to_opposite_axis_no_canonical_flip",
                **dict(locals().get("secondary_opening_recontext_diag_v175i", {}) or {}),
            }
            if other_components:
                # A rejected sidewall component may be displayed for review only
                # if it is large enough to be meaningful sidewall evidence.  The
                # v1.5.9 two-face case was not a Bore candidate; it was a tiny
                # fragment and must degrade to frame-only evidence.
                other_components.sort(key=lambda item: (-item[0], -len(item[1]), item[1][:1]))
                _, preview_ids, preview_st = other_components[0]
                preview_ids_tuple = tuple_ints(preview_ids)
                preview_face_count = int(preview_st.get("face_count", len(preview_ids_tuple)) or len(preview_ids_tuple))
                preview_axial_span = _safe_float(preview_st.get("axial_span", 0.0), 0.0)
                preview_coverage = _safe_float(preview_st.get("wall_span_fraction_of_two_opening_depth", 0.0), 0.0)
                meaningful_rejected_sidewall = bool(
                    preview_face_count >= max(12, int(0.015 * max(region_face_count, 1)))
                    and (preview_axial_span >= max(0.12 * bore_depth, 0.35 * bore_radius, 4.0 * edge_scale, 0.50) or preview_coverage >= 0.18)
                )
                review_diag.update(dict(preview_st))
                review_diag.setdefault("bore_wall_ownership_reject_reason", str(preview_st.get("bore_wall_ownership_reject_reason", "component_failed_two_opening_wall_ownership_gate")))
                if meaningful_rejected_sidewall:
                    review_face_ids = preview_ids_tuple
                    review_diag["preview_source"] = "best_rejected_meaningful_bore_wall_component"
                else:
                    review_face_ids = ()
                    review_diag["preview_source"] = "frame_only_rejected_sidewall_fragment_too_small"
                    review_diag["frame_only_candidate"] = True
                    review_diag["no_sidewall_no_bore_face_preview_used"] = True
                    review_diag["bore_wall_display_policy"] = "tiny_rejected_sidewall_fragment_not_displayed_as_bore_candidate"
            else:
                # v1.5.8 hard semantic rule:
                # No SidewallNormalRoleEvidence means there is no BORE wall face
                # candidate to display.  Never fall back to bore_tube_face_ids,
                # because that shows mouth/chamfer/tube context as a false BORE.
                review_face_ids = ()
                review_diag["preview_source"] = "frame_only_no_sidewall_faces"
                review_diag["no_sidewall_no_bore_face_preview_used"] = True
                review_diag["frame_only_candidate"] = True
                review_diag["bore_wall_ownership_reject_reason"] = "no_sidewall_normal_role_evidence_found"
                review_diag["bore_wall_display_policy"] = "frame_only_no_tube_chamfer_or_selected_opening_face_fallback"
            review_heuristic_results = _heuristic_results_for_current_bore_state_v174z(
                selected_edge_count=int(selected_edge_count),
                normalized_edge_count=int(normalized_edge_count),
                region_face_count=int(region_face_count),
                two_opening_valid=bool(two_opening_valid),
                bore_depth=float(bore_depth),
                bore_radius=float(bore_radius),
                sidewall_normal_evidence_count=int(locals().get("sidewall_normal_evidence_count", 0)),
                frame_sidewall_normal_evidence_count=int(locals().get("frame_sidewall_normal_evidence_count", 0)),
                bore_wall_search_face_count=int(len(bore_wall_search_face_ids)),
                bore_wall_candidate_component_count=int(len(bore_wall_candidate_components)),
                bore_wall_owned_face_count=0,
                bore_wall_rejected_component_count=int(len(bore_rejected)),
                bore_wall_component_rejection_reasons=tuple(str(v) for row in tuple(bore_rejected or ()) for v in tuple(row.get("bore_wall_component_rejection_reasons", ()) or ())),
                true_endpoint_reconciled=False,
                region_true_endpoint_depth=0.0,
                terminal_wall_candidate_face_count=0,
                terminal_wall_face_count=0,
                terminal_wall_completion_used=False,
                endpoint_extension_added_face_count=0,
                display_face_count=int(len(review_face_ids)),
                rebuild_face_count=0,
                chamfer_candidate_count=0,
                chamfer_promoted_count=0,
            )
            review_diag.update({
                "heuristic_results": review_heuristic_results,
                "heuristic_result_summaries": compact_heuristic_summary(review_heuristic_results),
                "heuristic_contract_version": HEURISTIC_CONTRACT_VERSION,
                "heuristic_recipe_name": "BORE_HEURISTIC_RECIPE",
                "heuristic_authority_policy": "heuristics_propose_measurement_quantifies_recognition_interprets_ownership_assigns",
                "v156_review_heuristic_results_preserved": True,
            })
            bore_heuristic_results = review_heuristic_results

            # v1.5.9: emit the measured-frame review row even when it has
            # no face-owned preview.  Face IDs are not required for frame evidence.
            if review_face_ids or bool(review_diag.get("frame_only_candidate", False)):
                bore_features.append(
                    bore_review_candidate_contract_fields_v174k(
                        candidate_id="component_engine.v82.bore.two_opening_wall_review.1",
                        face_ids=review_face_ids,
                        confidence=float(max(0.05, min(0.74, _safe_float(two_opening_bore_frame.get("confidence", 0.35), 0.35)))),
                        radius=float(bore_radius),
                        axial_span=float(bore_depth),
                        diagnostics=review_diag,
                    )
                )
    else:
        bore_rejected.append({
            "accepted_as_bore_wall_ownership": False,
            "bore_wall_ownership_reject_reason": "missing_or_invalid_two_opening_bore_frame",
            "two_opening_bore_frame_valid": bool(two_opening_valid),
            "two_opening_bore_frame_status": bore_frame_reason,
            "two_opening_bore_frame_depth": float(bore_depth),
        })

    # v1.5.2: Do not hide BORE just because the stricter two-opening/wall
    # ownership chain has not accepted yet.  A measured selected opening is a
    # visible BORE-family review object, not a CHAMFER fallback.
    if not bore_features and selected_opening_resolved:
        opening_face_set = set(selected_seed_face_set) or set(seed_face_set)
        opening_face_ids = tuple(int(fid) for fid in valid_face_ids if int(fid) in opening_face_set)
        if opening_face_ids:
            opening_diag = {
                "axis": tuple(selected_opening_frame_resolver.get("axis", kwargs.get("region_axis", (0.0, 0.0, 1.0))) or (0.0, 0.0, 1.0)),
                "center": tuple(selected_opening_frame_resolver.get("center", kwargs.get("region_center", (0.0, 0.0, 0.0))) or (0.0, 0.0, 0.0)),
                "selected_opening_frame_resolver": dict(selected_opening_frame_resolver),
                "two_opening_bore_frame": dict(two_opening_bore_frame),
                "bore_wall_ownership_reject_reason": "selected_opening_measured_waiting_for_opposite_opening_or_wall_ownership",
                "selected_opening_review_face_count": int(len(opening_face_ids)),
                "semantic_transform": "SelectedOpeningEvidence -> BORE review evidence; no CandidateData rebuild authority",
            }
            bore_features.append(
                bore_selected_opening_review_candidate_contract_fields_v174k(
                    candidate_id="component_engine.v82.bore.selected_opening_review.1",
                    face_ids=opening_face_ids,
                    confidence=float(max(0.05, min(0.74, _safe_float(selected_opening_frame_resolver.get("confidence", 0.35), 0.35)))),
                    radius=float(_safe_float(selected_opening_frame_resolver.get("radius", kwargs.get("region_radius", 0.0)), 0.0)),
                    diagnostics=opening_diag,
                )
            )

    state = dict(locals())
    state.update({
        "bore_candidate_assembly_split_checkpoint_v174z": BORE_RECOGNITION_CANDIDATE_ASSEMBLY_SPLIT_CHECKPOINT_V174Z,
        "bore_candidate_assembly_moved_to_recognition_bore_v174z": True,
        "bore_candidate_assembly_behavior_change_v174z": False,
    })
    return state
