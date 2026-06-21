"""Clean semantic CHAMFER + BORE-preview recognition engine for FAR MESH BoreTool.

This file remains small and explicit. It keeps the validated CHAMFER_FORM path
and adds the first clean BORE detection path as preview-only CandidateData.
BORE preview is recognition/perception only here; rebuild authorization remains
disabled until the BORE rebuild path is validated end-to-end.

Semantic meaning paths
----------------------
RegionData faces
    -> measured annular-transition evidence
    -> independent CHAMFER feature hypotheses
    -> chamfer_owned_face_ids per hypothesis
    -> CandidateData per owned chamfer surface

RegionData faces
    -> measured cylindrical/radius-layer evidence
    -> BORE wall hypothesis
    -> bore_wall_owned_face_ids per hypothesis
    -> accepted trial CandidateData per owned bore-wall surface
    -> damaged BORE evidence review candidates when ownership is incomplete, anchored to the selected opening frame

Important semantic rule
-----------------------
Selected opening seed relation is context, not exclusive rebuild authority.
CHAMFER candidates keep the validated action path. BORE candidates are visible
CandidateData only; they must not be treated as DeletePatchProposal authority
or as proof that rebuild is safe.

This preserves the BoreTool meaning boundary:
    evidence != ownership != CandidateData != DeletePatchProposal
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math
import numpy as np

from .geometry import canonical_axis, to_vector3
from .topology import (
    boundary_edges_for_face_patch,
    connected_face_components,
    edge_loop_components,
    face_adjacency_for_patch,
)
from .heuristics import (
    HEURISTIC_CONTRACT_VERSION,
    BORE_HEURISTIC_RECIPE,
    CHAMFER_HEURISTIC_RECIPE,
    POCKET_HEURISTIC_RECIPE,
    compact_heuristic_summary,
    heuristic_registry_dict,
    make_heuristic_result,
    recipe_contracts,
)
from .types import (
    EvidenceKind,
    FeatureFamily,
    FeaturePrimitiveKind,
    FeatureRelationshipKind,
    RecognitionStage,
    tuple_ints,
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)




def _heuristic_results_for_current_bore_state(
    *,
    selected_edge_count: int,
    normalized_edge_count: int,
    region_face_count: int,
    two_opening_valid: bool,
    bore_depth: float,
    bore_radius: float,
    sidewall_normal_evidence_count: int,
    frame_sidewall_normal_evidence_count: int,
    bore_wall_search_face_count: int,
    bore_wall_candidate_component_count: int,
    bore_wall_owned_face_count: int,
    bore_wall_rejected_component_count: int,
    bore_wall_component_rejection_reasons: tuple[str, ...],
    true_endpoint_reconciled: bool,
    region_true_endpoint_depth: float,
    terminal_wall_candidate_face_count: int,
    terminal_wall_face_count: int,
    terminal_wall_completion_used: bool,
    endpoint_extension_added_face_count: int,
    display_face_count: int,
    rebuild_face_count: int,
    chamfer_candidate_count: int,
    chamfer_promoted_count: int,
) -> tuple[dict[str, object], ...]:
    """Map active v1.7.x recognition stages into formal heuristic rows.

    These rows do not change recognition behavior. They expose the real staged
    heuristic contract so the UI/logs can show *what was searched/proposed*
    separately from measurement, ownership, CandidateData, and rebuild.
    """

    normalized = int(max(normalized_edge_count, selected_edge_count, 0))
    bore_reject_reasons = tuple(str(v) for v in tuple(bore_wall_component_rejection_reasons or ()) if str(v))
    return (
        make_heuristic_result(
            "SelectedEdgesToOpeningRingHeuristic",
            input_count=int(selected_edge_count),
            proposal_count=1 if normalized > 0 else 0,
            accepted_count=1 if normalized > 0 else 0,
            proposal_edge_ids=range(normalized),
            diagnostics={
                "selected_edge_count": int(selected_edge_count),
                "normalized_edge_count": int(normalized_edge_count),
                "note": "edge IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "MeasureSelectedOpening",
            input_count=max(int(normalized_edge_count), int(selected_edge_count)),
            proposal_count=1 if normalized > 0 else 0,
            accepted_count=1 if normalized > 0 else 0,
            confidence=1.0 if normalized > 0 else 0.0,
            diagnostics={
                "opening_radius_hint": float(bore_radius),
                "semantic_output": "MeasuredOpeningFrame",
            },
        ),
        make_heuristic_result(
            "OppositeOpeningSearchHeuristic",
            input_count=int(region_face_count),
            proposal_count=1 if bool(two_opening_valid) else 0,
            accepted_count=1 if bool(two_opening_valid) else 0,
            diagnostics={
                "two_opening_bore_frame_valid": bool(two_opening_valid),
                "provisional_depth": float(bore_depth),
            },
        ),
        make_heuristic_result(
            "MeasureTwoOpeningFrame",
            input_count=2 if bool(two_opening_valid) else 0,
            proposal_count=1 if bool(two_opening_valid) else 0,
            accepted_count=1 if bool(two_opening_valid) else 0,
            confidence=1.0 if bool(two_opening_valid) else 0.0,
            diagnostics={
                "depth": float(bore_depth),
                "radius": float(bore_radius),
                "semantic_output": "MeasuredTwoOpeningBoreFrame",
            },
        ),
        make_heuristic_result(
            "BoreWallSearchHeuristic",
            input_count=int(region_face_count),
            proposal_count=int(bore_wall_candidate_component_count),
            accepted_count=int(1 if bore_wall_search_face_count else 0),
            proposal_face_ids=range(max(int(bore_wall_search_face_count), 0)),
            diagnostics={
                "sidewall_normal_evidence_count": int(sidewall_normal_evidence_count),
                "frame_sidewall_normal_evidence_count": int(frame_sidewall_normal_evidence_count),
                "bore_wall_search_face_count": int(bore_wall_search_face_count),
                "note": "face IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "BoreWallRoleMeasurement",
            input_count=int(bore_wall_search_face_count),
            proposal_count=int(bore_wall_candidate_component_count),
            accepted_count=1 if int(bore_wall_owned_face_count) > 0 else 0,
            rejected_count=int(bore_wall_rejected_component_count),
            rejection_reasons=bore_reject_reasons,
            diagnostics={
                "semantic_output": "MeasuredBoreWallEvidence",
                "bore_wall_candidate_component_count": int(bore_wall_candidate_component_count),
            },
        ),
        make_heuristic_result(
            "BoreWallOwnership",
            input_count=int(bore_wall_search_face_count),
            proposal_count=int(bore_wall_candidate_component_count),
            accepted_count=int(bore_wall_owned_face_count),
            rejected_count=int(bore_wall_rejected_component_count),
            proposal_face_ids=range(max(int(bore_wall_owned_face_count), 0)),
            rejection_reasons=bore_reject_reasons,
            diagnostics={
                "semantic_output": "BoreWallOwnership",
                "ownership_face_count": int(bore_wall_owned_face_count),
                "note": "face IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "TrueEndpointResolver",
            input_count=int(bore_wall_owned_face_count),
            proposal_count=1 if float(region_true_endpoint_depth) > 0.0 else 0,
            accepted_count=1 if bool(true_endpoint_reconciled) else 0,
            rejected_count=0 if bool(true_endpoint_reconciled) else 1,
            rejection_reasons=() if bool(true_endpoint_reconciled) else ("true_endpoint_not_reconciled_or_not_needed",),
            diagnostics={
                "region_true_endpoint_depth": float(region_true_endpoint_depth),
                "true_endpoint_reconciled_from_region_endpoint": bool(true_endpoint_reconciled),
            },
        ),
        make_heuristic_result(
            "TerminalWallCompletionHeuristic",
            input_count=int(region_face_count),
            proposal_count=int(terminal_wall_candidate_face_count),
            accepted_count=int(terminal_wall_face_count),
            rejected_count=max(int(terminal_wall_candidate_face_count) - int(terminal_wall_face_count), 0),
            proposal_face_ids=range(max(int(terminal_wall_face_count), 0)),
            diagnostics={
                "terminal_wall_completion_used": bool(terminal_wall_completion_used),
                "true_endpoint_extension_added_face_count": int(endpoint_extension_added_face_count),
                "note": "face IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "TerminalWallOwnership",
            input_count=int(terminal_wall_candidate_face_count),
            proposal_count=int(terminal_wall_candidate_face_count),
            accepted_count=int(terminal_wall_face_count),
            rejected_count=max(int(terminal_wall_candidate_face_count) - int(terminal_wall_face_count), 0),
            proposal_face_ids=range(max(int(terminal_wall_face_count), 0)),
            diagnostics={
                "semantic_output": "FullDepthBoreWallOwnership",
                "terminal_wall_completion_used": bool(terminal_wall_completion_used),
                "note": "face IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "EmitBoreCandidateData",
            input_count=int(bore_wall_owned_face_count),
            proposal_count=1 if int(bore_wall_owned_face_count) > 0 else 0,
            accepted_count=1 if int(bore_wall_owned_face_count) > 0 else 0,
            proposal_face_ids=range(max(int(display_face_count), 0)),
            diagnostics={
                "display_face_count": int(display_face_count),
                "rebuild_face_count": int(rebuild_face_count),
                "semantic_output": "BORE CandidateData",
                "note": "face IDs are count placeholders in this compact diagnostic row",
            },
        ),
        make_heuristic_result(
            "OpeningContextToChamferSearch",
            input_count=int(region_face_count),
            proposal_count=int(chamfer_candidate_count),
            accepted_count=int(chamfer_promoted_count),
            rejected_count=max(int(chamfer_candidate_count) - int(chamfer_promoted_count), 0),
            diagnostics={
                "semantic_output": "ChamferBandProposal",
                "chamfer_candidate_count": int(chamfer_candidate_count),
            },
        ),
    )

def _unit_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=float)
    arr = arr[:, :3]
    length = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    ok = np.isfinite(length) & (length > 1.0e-12)
    out[ok] = arr[ok] / length[ok].reshape(-1, 1)
    return out


def _edge_median_length(vertices: np.ndarray, faces: np.ndarray, face_ids: Iterable[int]) -> float:
    lengths: list[float] = []
    selected = tuple_ints(face_ids)
    for fid in selected[: min(len(selected), 2000)]:
        if fid < 0 or fid >= len(faces):
            continue
        tri = vertices[faces[int(fid), :3], :3]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            length = float(np.linalg.norm(tri[a] - tri[b]))
            if math.isfinite(length) and length > 1.0e-9:
                lengths.append(length)
    if not lengths:
        return 1.0
    return float(np.median(np.asarray(lengths, dtype=float)))


def _percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


class _Frame:
    def __init__(self, *, center: object, axis: object, radius: object) -> None:
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.axis = canonical_axis(axis)
        self.radius = _safe_float(radius, 0.0)


def _local_arrays(
    *,
    frame: _Frame,
    face_ids: tuple[int, ...],
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = np.asarray(face_ids, dtype=np.int64)
    pts = np.asarray(face_centroids, dtype=float)[ids, :3]
    normals = _unit_rows(np.asarray(face_normals, dtype=float))
    if len(normals) <= int(np.max(ids)) if len(ids) else False:
        normals = np.zeros_like(np.asarray(face_centroids, dtype=float)[:, :3])
    n = normals[ids, :3]
    rel = pts - frame.center.reshape(1, 3)
    axial = rel @ frame.axis.reshape(3)
    radial_vec = rel - axial.reshape(-1, 1) * frame.axis.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    radial_dir = np.zeros_like(radial_vec)
    ok_radial = radial > 1.0e-12
    radial_dir[ok_radial] = radial_vec[ok_radial] / radial[ok_radial].reshape(-1, 1)
    normal_axis_abs = np.abs(n @ frame.axis.reshape(3))
    radial_normal_alignment = np.abs(np.sum(n * radial_dir, axis=1))
    finite = (
        np.isfinite(axial)
        & np.isfinite(radial)
        & np.isfinite(normal_axis_abs)
        & np.isfinite(radial_normal_alignment)
    )
    return {
        "ids": ids,
        "points": pts,
        "axial": axial,
        "radial": radial,
        "normal_axis_abs": normal_axis_abs,
        "radial_normal_alignment": radial_normal_alignment,
        "finite": finite,
    }


def _component_stats(
    *,
    comp: tuple[int, ...],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
) -> dict[str, object]:
    idx = np.asarray([fid_to_local[int(fid)] for fid in comp if int(fid) in fid_to_local], dtype=np.int64)
    if idx.size == 0:
        return {"face_count": 0}
    comp_set = {int(fid) for fid in comp}
    direct_seed = comp_set & seed_face_set
    adjacent_seed: set[int] = set()
    if seed_face_set:
        for fid in comp_set:
            for nb in adjacency.get(int(fid), ()):  # selected-face adjacency only
                if int(nb) in seed_face_set:
                    adjacent_seed.add(int(fid))
                    break
    ax = axial[idx]
    rd = radial[idx]
    na = normal_axis_abs[idx]
    al = radial_normal_alignment[idx]
    radial_span = float(np.max(rd) - np.min(rd)) if rd.size else 0.0
    axial_span = float(np.max(ax) - np.min(ax)) if ax.size else 0.0
    return {
        "face_count": int(len(comp)),
        "axial_min": float(np.min(ax)),
        "axial_max": float(np.max(ax)),
        "axial_center": float(np.median(ax)),
        "axial_span": axial_span,
        "radial_min": float(np.min(rd)),
        "radial_max": float(np.max(rd)),
        "radial_median": float(np.median(rd)),
        "radial_span": radial_span,
        "radial_mad": float(np.median(np.abs(rd - np.median(rd)))) if rd.size else 0.0,
        "radial_rel_mad": float((np.median(np.abs(rd - np.median(rd))) / max(float(np.median(rd)), 1.0e-12))) if rd.size else 999999.0,
        "normal_axis_abs_median": float(np.median(na)),
        "normal_axis_abs_q75": float(np.percentile(na, 75.0)) if na.size else 1.0,
        "radial_normal_alignment_median": float(np.median(al)),
        "radial_normal_alignment_q25": float(np.percentile(al, 25.0)) if al.size else 0.0,
        "seed_direct_face_count": int(len(direct_seed)),
        "seed_adjacent_face_count": int(len(adjacent_seed)),
        "seed_related": bool(direct_seed or adjacent_seed),
    }


def _chamfer_score(st: Mapping[str, object], *, radius_scale: float, axial_span_all: float) -> float:
    face_count = int(st.get("face_count", 0) or 0)
    if face_count <= 0:
        return -1.0e9
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    radial_span = _safe_float(st.get("radial_span", 0.0), 0.0)
    normal_axis = _safe_float(st.get("normal_axis_abs_median", 0.0), 0.0)
    radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)

    # A chamfer is an annular transition: it has both axial and radial change.
    # The normal contains an axial component and a radial component, unlike a
    # pure cylindrical wall (mostly radial normal) or a flat cap (mostly axial).
    radial_transition = min(radial_span / max(0.020 * radius_scale, 0.20), 1.0)
    axial_transition = min(axial_span / max(0.020 * radius_scale, 0.20), 1.0)
    normal_mix = max(0.0, 1.0 - abs(normal_axis - radial_align))
    slope_present = min(normal_axis / 0.35, 1.0) * min(radial_align / 0.35, 1.0)
    seed_bonus = 2.2 if bool(st.get("seed_related", False)) else 0.0
    count_score = min(face_count / 80.0, 1.0)
    too_deep_penalty = 0.0
    if axial_span_all > 1.0e-9 and axial_span > 0.55 * axial_span_all:
        too_deep_penalty = 1.5
    return float(
        1.15 * radial_transition
        + 1.00 * axial_transition
        + 0.85 * normal_mix
        + 1.25 * slope_present
        + 0.25 * count_score
        + seed_bonus
        - too_deep_penalty
    )



def _bore_score(st: Mapping[str, object], *, radius_scale: float, axial_span_all: float, edge_scale: float) -> float:
    """Score measured cylindrical wall evidence under a BORE hypothesis.

    This is evidence/ownership scoring, not rebuild validation.  The score is
    derived from role predicates: constant radial layer, wall-like normals,
    axial continuation, and enough connected surface support.
    """

    face_count = int(st.get("face_count", 0) or 0)
    if face_count <= 0:
        return -1.0e9
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    radial_median = max(_safe_float(st.get("radial_median", 0.0), 0.0), 1.0e-12)
    radial_rel_mad = _safe_float(st.get("radial_rel_mad", 999999.0), 999999.0)
    radial_mad = _safe_float(st.get("radial_mad", 999999.0), 999999.0)
    normal_axis = _safe_float(st.get("normal_axis_abs_median", 1.0), 1.0)
    normal_axis_q75 = _safe_float(st.get("normal_axis_abs_q75", 1.0), 1.0)
    radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)
    radial_align_q25 = _safe_float(st.get("radial_normal_alignment_q25", 0.0), 0.0)

    radial_compact = max(
        0.0,
        1.0 - min(radial_rel_mad / 0.085, radial_mad / max(0.70 * edge_scale, 1.0e-9), 1.0),
    )
    wall_normal = max(0.0, 1.0 - normal_axis / 0.42) * min(radial_align / 0.72, 1.0)
    wall_normal_stability = max(0.0, 1.0 - normal_axis_q75 / 0.58) * min(radial_align_q25 / 0.42, 1.0)
    axial_continuation = min(axial_span / max(0.45 * radial_median, 1.75 * edge_scale, 1.0e-9), 1.0)
    context_span = min(axial_span / max(0.18 * max(axial_span_all, radius_scale, 1.0), 1.0e-9), 1.0)
    count_score = min(face_count / 120.0, 1.0)
    seed_bonus = 0.35 if bool(st.get("seed_related", False)) else 0.0
    return float(
        1.60 * radial_compact
        + 1.35 * wall_normal
        + 0.75 * wall_normal_stability
        + 1.05 * axial_continuation
        + 0.50 * context_span
        + 0.35 * count_score
        + seed_bonus
    )



def _damaged_bore_reason_flags(st: Mapping[str, object]) -> tuple[str, ...]:
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


def _damaged_bore_preview_allowed(st: Mapping[str, object]) -> bool:
    """Gate preview-only damaged BORE candidates.

    This is deliberately weaker than clean ownership but still requires bore-like
    physical evidence.  It must not expose rebuild_face_ids and must not repair
    recognition by converting broad RegionData into a candidate.
    """

    face_count = int(st.get("face_count", 0) or 0)
    if face_count <= 0:
        return False
    score = _safe_float(st.get("score", 0.0), 0.0)
    seed_related = bool(st.get("seed_related", False))
    radial_median = _safe_float(st.get("radial_median", 0.0), 0.0)
    radial_rel_mad = _safe_float(st.get("radial_rel_mad", 999999.0), 999999.0)
    radial_mad = _safe_float(st.get("radial_mad", 999999.0), 999999.0)
    max_radial_mad = _safe_float(st.get("max_bore_radial_mad", 0.0), 0.0)
    axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
    min_axial = _safe_float(st.get("min_bore_axial_span", 0.0), 0.0)
    normal_axis = _safe_float(st.get("normal_axis_abs_median", 1.0), 1.0)
    radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)

    enough_faces = face_count >= (4 if seed_related else 8)
    wall_like_normals = normal_axis <= 0.64 and radial_align >= 0.30
    radius_layer = radial_median > 1.0e-9 and (radial_rel_mad <= 0.20 or (max_radial_mad > 0.0 and radial_mad <= 1.75 * max_radial_mad))
    axial_hint = axial_span >= (0.35 * min_axial if min_axial > 1.0e-9 else 0.0) or seed_related
    score_gate = score >= (1.30 if seed_related else 1.85)
    return bool(enough_faces and wall_like_normals and radius_layer and axial_hint and score_gate)




def _safe_int(value: object, default: int = 0) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)
    return int(out)


def _seed_neighborhood_face_ids(
    *,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    max_depth: int = 4,
    max_faces: int = 900,
) -> tuple[int, ...]:
    """Return a bounded face neighborhood around selected-opening seed faces.

    This is Recognition-local evidence scoping for damaged BORE review.  It is
    not Region Select classification and it is not rebuild authorization.  The
    purpose is to prevent broad damaged RegionData from promoting remote
    chamfer-like fragments while still giving damaged bore evidence a selected
    opening context.
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


def _bounded_component_from_seed_faces(
    *,
    component_face_ids: Iterable[int],
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    max_depth: int = 7,
    max_faces: int = 900,
) -> tuple[int, ...]:
    """Return the selected-opening-local island inside a broader component.

    v1.3.8 damaged-bore locality guard.  The raw component measurement can now
    identify the selected opening, but broad RegionData may still contain a large
    connected cylindrical component that reaches unrelated mesh areas.  Candidate
    ownership may therefore only use the bounded neighborhood grown from the
    measured opening seed faces, inside the candidate component.  This keeps the
    semantic transfer clean: MeasuredBoreFrame support constrains wall ownership;
    neutral RegionData does not become display/rebuild authority.
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
            for nb in adjacency.get(int(fid), ()):
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


def _cylindrical_seed_island_from_seed_faces(
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

    v1.3.8: plain adjacency BFS can walk from the selected bore mouth into
    neighboring or unrelated mesh areas.  This variant only crosses faces that
    stay on the measured bore cylinder band and remain a plausible wall/defect
    continuation.  The measured primary seed island is the authority; broad
    RegionData and expanded measurement fragments are not.
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
            for nb in adjacency.get(int(fid), ()):
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



def _post_isolation_bore_wall_validation(
    st: Mapping[str, object],
    *,
    radial_scale: float,
    edge_scale: float,
) -> dict[str, object]:
    """Validate that an isolated seed island is still a bore wall.

    v1.3.8: seed-island clipping can correctly remove remote over-selection, but
    it can also leave only a shallow rim/transition strip. Such a strip may be
    close to the measured cylinder and seed-related, yet it is not BoreWall
    ownership. Re-run the bore-wall physical predicates after clipping before a
    CandidateData object is allowed to expose rebuild_face_ids.
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
    }


def _damaged_anchor_mode_from_region(
    *,
    region_diagnostics: Mapping[str, object],
    region_face_count: int,
    seed_face_count: int,
) -> dict[str, object]:
    """Detect the damaged/overbroad selection mode from neutral diagnostics.

    This does not say "this is a bore". It tells Recognition that the RegionData
    cutout is broad/fragmented enough that candidate promotion must be anchored
    to selected-opening evidence instead of letting every clean-looking annular
    fragment inside the volume become action-enabled.
    """

    selected_edge_count = _safe_int(
        region_diagnostics.get("selected_edge_count", region_diagnostics.get("raw_selected_edge_count", 0)),
        0,
    )
    normalized_edge_count = _safe_int(
        region_diagnostics.get(
            "normalized_edge_count",
            region_diagnostics.get("primary_anchor_edge_count", region_diagnostics.get("normalized_rim_edge_count", 0)),
        ),
        0,
    )
    boundary_loop_count = _safe_int(region_diagnostics.get("boundary_loop_count", 0), 0)
    normalized_ratio = float(normalized_edge_count) / max(float(selected_edge_count), 1.0)
    broad_region = bool(region_face_count >= 6000)
    fragmented_selection = bool(selected_edge_count >= 300 and (normalized_edge_count <= 0 or normalized_ratio <= 0.35))
    many_edges_small_anchor = bool(selected_edge_count >= 900 and normalized_edge_count <= 96)
    damaged_anchor_mode = bool(seed_face_count > 0 and (broad_region or fragmented_selection or many_edges_small_anchor))
    return {
        "damaged_anchor_mode": bool(damaged_anchor_mode),
        "damaged_anchor_reason": (
            "broad_region_or_fragmented_selected_opening" if damaged_anchor_mode else "clean_or_compact_region"
        ),
        "selected_edge_count": int(selected_edge_count),
        "normalized_edge_count": int(normalized_edge_count),
        "normalized_to_selected_edge_ratio": float(normalized_ratio),
        "boundary_loop_count": int(boundary_loop_count),
        "broad_region": bool(broad_region),
        "fragmented_selection": bool(fragmented_selection),
        "many_edges_small_anchor": bool(many_edges_small_anchor),
    }

def _interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Return zero for overlapping intervals, otherwise the positive gap."""

    lo0, hi0 = (float(a_min), float(a_max)) if float(a_min) <= float(a_max) else (float(a_max), float(a_min))
    lo1, hi1 = (float(b_min), float(b_max)) if float(b_min) <= float(b_max) else (float(b_max), float(b_min))
    if hi0 < lo1:
        return float(lo1 - hi0)
    if hi1 < lo0:
        return float(lo0 - hi1)
    return 0.0


def _opening_anchor_reference(
    *,
    frame: _Frame,
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
    hypothesis construction.  RegionData may be broad, but a damaged BORE review
    candidate must remain tied to this opening frame by radius and axial contact;
    otherwise a random cylinder-like fragment in the cutout can masquerade as
    the selected bore.
    """

    seed_idx = np.asarray([fid_to_local[int(fid)] for fid in seed_face_set if int(fid) in fid_to_local], dtype=np.int64)
    if seed_idx.size:
        seed_axial = axial[seed_idx]
        seed_radial = radial[seed_idx]
        seed_axial = seed_axial[np.isfinite(seed_axial)]
        seed_radial = seed_radial[np.isfinite(seed_radial)]
    else:
        seed_axial = np.asarray((0.0,), dtype=float)
        seed_radial = np.asarray((float(frame.radius),), dtype=float)

    opening_radius = float(frame.radius) if np.isfinite(float(frame.radius)) and float(frame.radius) > 1.0e-9 else _percentile(radial, 50.0, 1.0)
    median_seed_radial = float(np.median(seed_radial)) if seed_radial.size else float(opening_radius)
    seed_min = float(np.min(seed_axial)) if seed_axial.size else 0.0
    seed_max = float(np.max(seed_axial)) if seed_axial.size else 0.0
    seed_center = float(np.median(seed_axial)) if seed_axial.size else 0.0

    projection_half_depth = _safe_float(region_diagnostics.get("projection_axial_half_depth", 0.0), 0.0)
    projected_depth = float(2.0 * projection_half_depth) if projection_half_depth > 1.0e-9 else 0.0
    # v49: for damaged-bore review the selected opening frame is a hypothesis
    # descriptor, not a rebuild instruction.  Prefer the measured RegionData
    # axial span over the projection depth, because projection depth can be a
    # deliberately broad search volume and produced misleading 400+ depth
    # labels on damaged meshes.
    hypothesis_depth = float(axial_span_all) if float(axial_span_all) > 1.0e-9 else float(projected_depth)

    radius_tolerance = max(4.0 * max(float(edge_scale), 1.0e-9), 0.32 * max(float(opening_radius), 1.0), 0.35)
    axial_tolerance = max(6.0 * max(float(edge_scale), 1.0e-9), 0.45 * max(float(opening_radius), 1.0), 0.75)
    return {
        "selected_opening_anchor_available": bool(seed_idx.size > 0),
        "selected_opening_seed_face_count": int(seed_idx.size),
        "opening_frame_center": to_vector3(frame.center),
        "opening_frame_axis": to_vector3(frame.axis),
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
    }


def _opening_frame_anchor_metrics(
    st: Mapping[str, object],
    *,
    opening_anchor: Mapping[str, object],
) -> dict[str, object]:
    """Measure whether a component belongs to the selected opening frame.

    This is a Recognition-stage guard for damaged BORE review.  It does not
    create ownership and it does not authorize rebuild.  It only prevents the
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
    axial_gap = _interval_gap(axial_min, axial_max, seed_axial_min, seed_axial_max)
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
    }



def _selected_opening_evidence_face_ids(
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

    v49 damaged-bore review rule: the preview object must be physically located
    at the selected opening.  Broad RegionData, raw selected-edge pollution, and
    remote cylindrical fragments may remain diagnostics, but they must not become
    the visible damaged BORE candidate.  This returns normalized opening seed
    faces first, then only the seed-connected neighborhood that matches the
    opening radius band.
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


def _damaged_bore_candidate_contract_fields(
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
    damage_reasons = damage_reasons_override or _damaged_bore_reason_flags(diagnostics)
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

def _bore_candidate_contract_fields(
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
        "feature_ownership_split": "bore_wall_owned_face_ids_to_explicit_rebuild_face_ids_no_regiondata_fallback",
        "bore_rebuild_enable_scope": "candidate_owned_bore_wall_faces_only",
        "regiondata_rebuild_fallback_allowed": False,
        "diagnostics": dict(diagnostics),
    }


def _bore_review_candidate_contract_fields(
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


def _bore_selected_opening_review_candidate_contract_fields(
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
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "borehole",
        "feature_kind": "borehole",
        "candidate_scope": "candidate_data_review_only",
        "candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
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
        "depth": 0.0,
        "height": 0.0,
        "axial_span": 0.0,
        "primitive_axis": axis,
        "primitive_radius": float(radius),
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

def _candidate_contract_fields(
    *,
    candidate_id: str,
    face_ids: tuple[int, ...],
    accepted: bool,
    confidence: float,
    radius_inner: float,
    radius_outer: float,
    axial_span: float,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    stage = RecognitionStage.ACCEPTED_CANDIDATE.value if accepted else RecognitionStage.REVIEW.value
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "chamfer",
        "feature_kind": "chamfer",
        "candidate_scope": "candidate_data",
        "candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "feature_family": FeatureFamily.CHAMFER_FORM.value,
        "recognition_stage": stage,
        "display_name": "CHAMFER — clean annular transition surface" if accepted else "CHAMFER — annular transition review",
        "role": "rebuildable_chamfer_operation" if accepted else "chamfer_review_only",
        "status": "promoted_chamfer_from_clean_annular_transition_ownership" if accepted else "review_chamfer_annular_transition_evidence",
        "candidate_side_role": "selected_side_chamfer" if bool(diagnostics.get("seed_related", False)) else "independent_opposite_or_secondary_chamfer",
        "selection_seed_related": bool(diagnostics.get("seed_related", False)),
        "promotion_state": "promoted" if accepted else "review",
        "candidate_action_enabled": bool(accepted),
        "candidate_action": "rebuild" if accepted else "preview_only",
        "rebuild_authorized": bool(accepted),
        "rebuild_gate": "promoted_chamfer_candidate" if accepted else "chamfer_review_candidate",
        "face_ids": face_ids,
        "semantic_face_ids": face_ids,
        "display_face_ids": face_ids,
        "preview_face_ids": face_ids,
        "rebuild_face_ids": face_ids if accepted else (),
        "face_count": int(len(face_ids)),
        "confidence": float(confidence),
        "radius": float((radius_inner + radius_outer) * 0.5) if radius_outer >= radius_inner else float(radius_inner),
        "diameter": float((radius_inner + radius_outer)),
        "inner_radius": float(radius_inner),
        "outer_radius": float(radius_outer),
        "chamfer_width": float(abs(radius_outer - radius_inner)),
        "axial_span": float(axial_span),
        "depth": float(axial_span),
        "height": float(axial_span),
        "surface_condition": "annular_transition_owned_faces_only",
        "repair_strategy": "delete_patch_from_chamfer_owned_faces",
        "evidence_kinds": (
            EvidenceKind.OPENING_RING.value,
            EvidenceKind.CHAMFER_BAND.value,
            EvidenceKind.RADIUS_CONSISTENCY.value,
        ),
        "promotion_reasons": (
            "clean_chamfer_only_recognition",
            "annular_transition_surface_role",
            "selected_opening_seed_context" if bool(diagnostics.get("seed_related", False)) else "independent_annular_transition_geometry",
            "independent_clean_chamfer_rebuild_path_supported" if accepted else "review_only_chamfer_evidence_not_promoted",
        ),
        "rejection_reasons": () if accepted else ("chamfer_evidence_not_promoted",),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.ANNULAR_CHAMFER_BAND.value,
                "source": "recognition_component_engine.v80_mouth_transition_chamfer_rebuild",
                "role": "accepted_physical_primitive" if accepted else "diagnostic_primitive_descriptor",
                "radius": float((radius_inner + radius_outer) * 0.5) if radius_outer >= radius_inner else float(radius_inner),
                "inner_radius": float(radius_inner),
                "outer_radius": float(radius_outer),
                "depth": float(axial_span),
                "confidence": float(confidence),
                "face_ids": face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 1,
        "feature_relationships": (),
        "feature_relationship_count": 0,
        "x1_primitive_bridge_contract": "mesh-native primitive descriptor, not CAD body and not rebuild target",
        "delete_patch_request_allowed": bool(accepted),
        "rebuild_target_policy_allowed": bool(accepted),
        "rebuild_target_policy_reason": "accepted chamfer-form candidate may request DeletePatchProposal; rebuild.py still validates topology" if accepted else "review candidate is preview-only",
        "recognition_rule": "v40_clean_multi_chamfer_annular_transition_ownership",
        "feature_ownership_source": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "feature_ownership_split": "chamfer_owned_face_ids_only_no_bore_or_regiondata_ownership",
        "diagnostics": dict(diagnostics),
    }



def _normalize_edge_key(edge: object) -> tuple[int, int]:
    try:
        a, b = tuple(edge)[:2]  # type: ignore[arg-type]
        ia = int(a)
        ib = int(b)
    except Exception:
        return (-1, -1)
    if ia == ib:
        return (-1, -1)
    return (ia, ib) if ia < ib else (ib, ia)


def _pocket_floor_boundary_loop_records(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    floor_face_ids: tuple[int, ...],
    center: object,
    axis: object,
) -> tuple[dict[str, object], ...]:
    """Return neutral floor boundary-loop evidence for POCKET relationships.

    This does not create a new feature family.  It only describes the boundary
    loops of the owned pocket-floor role.  Recognition may use smaller interior
    loops as protected child-BORE opening evidence while keeping POCKET and BORE
    as separate feature hypotheses.
    """

    face_arr = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(vertices, dtype=float)
    if face_arr.ndim != 2 or face_arr.shape[1] < 3 or verts.ndim != 2 or verts.shape[1] < 3:
        return ()
    ids = tuple_ints(floor_face_ids)
    if not ids:
        return ()
    axis_vec = canonical_axis(axis)
    center_vec = np.asarray(center, dtype=float).reshape(3)
    loops: list[dict[str, object]] = []
    try:
        boundary = boundary_edges_for_face_patch(face_arr[:, :3], ids)
        components = edge_loop_components(boundary)
    except Exception:
        return ()
    for index, comp in enumerate(components):
        edges = tuple(sorted({_normalize_edge_key(edge) for edge in tuple(comp or ()) if _normalize_edge_key(edge) != (-1, -1)}))
        loop_vertices = tuple(sorted({int(v) for edge in edges for v in edge if 0 <= int(v) < len(verts)}))
        if len(loop_vertices) < 3:
            continue
        pts = verts[np.asarray(loop_vertices, dtype=np.int64), :3]
        rel = pts - center_vec.reshape(1, 3)
        axial = rel @ axis_vec
        radial_vec = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radii = np.linalg.norm(radial_vec, axis=1)
        loops.append({
            "index": int(index),
            "vertices": loop_vertices,
            "edges": edges,
            "vertex_count": int(len(loop_vertices)),
            "edge_count": int(len(edges)),
            "center": to_vector3(np.mean(pts, axis=0)),
            "radial_median": float(np.median(radii)) if radii.size else 0.0,
            "radial_max": float(np.max(radii)) if radii.size else 0.0,
            "axial_median": float(np.median(axial)) if axial.size else 0.0,
        })
    loops.sort(key=lambda row: (-_safe_float(row.get("radial_median", 0.0), 0.0), -int(row.get("edge_count", 0) or 0)))
    return tuple(loops)


def _pocket_floor_compound_boundary_metadata(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    floor_face_ids: tuple[int, ...],
    center: object,
    axis: object,
) -> dict[str, object]:
    """Return relationship metadata for a POCKET floor that contains BORE openings.

    The returned rows are metadata only.  They do not create a
    `pocket_with_bore` family.  The POCKET candidate may own the floor around a
    child bore, while the child BORE remains a separate rebuildable feature
    object.  Rebuild Target later protects these loops when rebuilding the
    pocket floor.
    """

    loops = _pocket_floor_boundary_loop_records(
        faces=faces,
        vertices=vertices,
        floor_face_ids=floor_face_ids,
        center=center,
        axis=axis,
    )
    if not loops:
        return {
            "pocket_floor_boundary_loop_count": 0,
            "pocket_protected_floor_bore_opening_count": 0,
            "pocket_protected_floor_bore_opening_loops": (),
            "pocket_floor_outer_boundary_loop": {},
        }
    outer = loops[0]
    outer_radius = max(_safe_float(outer.get("radial_median", 0.0), 0.0), 1.0e-9)
    protected: list[dict[str, object]] = []
    for row in loops[1:]:
        radius = _safe_float(row.get("radial_median", 0.0), 0.0)
        edge_count = int(row.get("edge_count", 0) or 0)
        # An inner boundary of the owned pocket floor is protected opening
        # evidence for a separate child BORE.  It is not pocket ownership and
        # not a new feature family.
        if edge_count >= 6 and radius <= 0.90 * outer_radius:
            protected.append({
                **dict(row),
                "relationship_meaning": "protected_child_bore_opening_boundary_in_owned_pocket_floor",
                "semantic_status": "relationship_metadata_only_not_feature_family",
            })
    return {
        "pocket_floor_boundary_loop_count": int(len(loops)),
        "pocket_floor_boundary_loops": loops,
        "pocket_floor_outer_boundary_loop": dict(outer),
        "pocket_protected_floor_bore_opening_count": int(len(protected)),
        "pocket_protected_floor_bore_opening_loops": tuple(protected),
        "pocket_compound_relationship_model": "POCKET candidate plus separate BORE candidate; protected floor opening metadata only",
    }


def _pocket_candidate_contract_fields(
    *,
    candidate_id: str,
    side_wall_face_ids: tuple[int, ...],
    floor_face_ids: tuple[int, ...],
    transition_face_ids: tuple[int, ...],
    protected_bore_opening_loops: tuple[Mapping[str, object], ...],
    embedded_bore_wall_evidence_face_ids: tuple[int, ...],
    confidence: float,
    depth: float,
    axis: object,
    rim_center: object,
    floor_center: object,
    footprint_radius: float,
    pocket_kind: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return accepted CandidateData for a recognized POCKET.

    This is a POCKET semantic object, not a disguised BORE.  Recognition has
    transformed measured rim/floor/side-wall/depth evidence into a POCKET
    hypothesis and then into owned pocket floor and side-wall surface roles.
    CandidateData may expose those owned roles to Rebuild Target, but it does
    not itself mutate topology.  The v99 rebuild path consumes the owned floor
    + side-wall roles to rebuild a recessed pocket cup: side wall plus floor,
    with the top opening left open and transition/chamfer evidence excluded.
    """

    side_wall_face_ids = tuple_ints(side_wall_face_ids)
    floor_face_ids = tuple_ints(floor_face_ids)
    transition_face_ids = tuple_ints(transition_face_ids)
    embedded_bore_wall_evidence_face_ids = tuple_ints(embedded_bore_wall_evidence_face_ids)
    protected_bore_opening_loops = tuple(dict(row) for row in tuple(protected_bore_opening_loops or ()))
    protected_loop_vertices = tuple(
        tuple_ints(row.get("vertices", ()))
        for row in protected_bore_opening_loops
        if tuple_ints(row.get("vertices", ()))
    )
    protected_loop_edges = tuple(
        tuple(tuple(_normalize_edge_key(edge) for edge in tuple(row.get("edges", ()) or ()) if _normalize_edge_key(edge) != (-1, -1)))
        for row in protected_bore_opening_loops
    )
    has_child_bore = bool(protected_loop_vertices or embedded_bore_wall_evidence_face_ids)
    semantic_face_ids = tuple_ints(side_wall_face_ids + floor_face_ids)
    display_face_ids = tuple_ints(semantic_face_ids + transition_face_ids)
    axis_tuple = to_vector3(axis)
    rim_center_tuple = to_vector3(rim_center)
    floor_center_tuple = to_vector3(floor_center)
    family = FeatureFamily.CIRCULAR_POCKET.value if str(pocket_kind).lower() == "circular" else FeatureFamily.POCKET.value
    display_kind = "CIRCULAR POCKET" if family == FeatureFamily.CIRCULAR_POCKET.value else "POCKET"
    return {
        "candidate_id": candidate_id,
        "feature_id": candidate_id,
        "entity_type": "pocket",
        "feature_kind": "pocket",
        "candidate_scope": "candidate_data",
        "candidate_variant": str(pocket_kind or "freeform"),
        "candidate_authority": "surface_component_classifier_v102_pocket_recess_hypothesis_role_ownership_compound_relationships",
        "active_candidate_authority": "surface_component_classifier_v102_pocket_recess_hypothesis_role_ownership_compound_relationships",
        "feature_family": family,
        "recognition_stage": RecognitionStage.ACCEPTED_CANDIDATE.value,
        "display_name": f"{display_kind} — pocket recess rebuild",
        "role": "pocket_recess_rebuild_floor_and_sidewall_owned_roles",
        "status": "pocket_recognition_accepted_recess_cup_rebuild_target",
        "promotion_state": "promoted",
        "candidate_action_enabled": True,
        "candidate_action": "delete_and_rebuild_owned_pocket_floor_and_side_wall_as_recess_cup",
        "rebuild_authorized": True,
        "rebuild_gate": "pocket_recess_cup_rebuild_target_contract_v102_protect_child_bore_boundaries",
        "rebuild_block_reason": "",
        "face_ids": semantic_face_ids,
        "semantic_face_ids": semantic_face_ids,
        "display_face_ids": display_face_ids,
        "preview_face_ids": display_face_ids,
        "rebuild_face_ids": semantic_face_ids,
        "candidate_rebuild_face_ids": semantic_face_ids,
        "owned_face_ids": semantic_face_ids,
        "pocket_side_wall_face_ids": side_wall_face_ids,
        "pocket_side_wall_face_count": int(len(side_wall_face_ids)),
        "pocket_floor_face_ids": floor_face_ids,
        "pocket_floor_face_count": int(len(floor_face_ids)),
        "pocket_transition_face_ids": transition_face_ids,
        "pocket_transition_face_count": int(len(transition_face_ids)),
        "pocket_has_child_bore_opening": bool(has_child_bore),
        "pocket_protected_floor_bore_opening_count": int(len(protected_loop_vertices)),
        "pocket_protected_floor_bore_opening_loop_vertex_ids": protected_loop_vertices,
        "pocket_protected_floor_bore_opening_loop_edges": protected_loop_edges,
        "pocket_embedded_bore_wall_evidence_face_ids": embedded_bore_wall_evidence_face_ids,
        "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)),
        "compound_feature_model": "separate POCKET candidate plus separate BORE candidate; protected child-bore opening relationship metadata only",
        "face_count": int(len(semantic_face_ids)),
        "display_face_count": int(len(display_face_ids)),
        "confidence": float(max(0.0, min(0.95, confidence))),
        "depth": float(depth),
        "height": float(depth),
        "axial_span": float(depth),
        "axis": axis_tuple,
        "primitive_axis": axis_tuple,
        "rim_center": rim_center_tuple,
        "floor_center": floor_center_tuple,
        "primitive_depth": float(depth),
        "primitive_radius": float(footprint_radius),
        "radius": float(footprint_radius),
        "diameter": float(2.0 * footprint_radius),
        "surface_condition": "owned_pocket_floor_and_side_wall_faces_for_recess_cup_rebuild",
        "repair_strategy": "pocket_native_recess_cup_rebuild_delete_owned_floor_plus_sidewall_generate_sidewall_plus_floor",
        "evidence_kinds": tuple(v for v in (
            EvidenceKind.POCKET_RIM.value,
            EvidenceKind.POCKET_FLOOR.value,
            EvidenceKind.POCKET_SIDE_WALL.value,
            EvidenceKind.POCKET_DEPTH.value,
            EvidenceKind.POCKET_TRANSITION.value,
            EvidenceKind.POCKET_PROTECTED_BORE_OPENING.value if has_child_bore else "",
        ) if v),
        "promotion_reasons": tuple(v for v in (
            "pocket_floor_surface_role_ownership",
            "pocket_side_wall_surface_role_ownership",
            "positive_recess_depth_resolved",
            "transition_faces_separated_from_floor_wall_ownership",
            "rebuild_faces_are_owned_pocket_side_wall_plus_floor_roles",
            "floor_faces_are_owned_pocket_floor_rebuild_role_not_parent_cap",
            "child_bore_opening_is_relationship_metadata_not_pocket_family" if has_child_bore else "",
        ) if v),
        "rejection_reasons": (),
        "feature_primitives": (
            {
                "primitive_kind": FeaturePrimitiveKind.POCKET_RECESS.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "preview_descriptor_not_cad_body_not_rebuild_target",
                "center": rim_center_tuple,
                "axis": axis_tuple,
                "radius": float(footprint_radius),
                "diameter": float(2.0 * footprint_radius),
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": semantic_face_ids,
                "diagnostics": dict(diagnostics),
            },
            {
                "primitive_kind": FeaturePrimitiveKind.PLANAR_FLOOR.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "owned_pocket_floor_surface_role",
                "center": floor_center_tuple,
                "axis": axis_tuple,
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": floor_face_ids,
                "diagnostics": dict(diagnostics),
            },
            {
                "primitive_kind": FeaturePrimitiveKind.POCKET_SIDE_WALL_SET.value,
                "source": "recognition_component_engine.v102_pocket_recess_cup_rebuild",
                "role": "owned_pocket_side_wall_surface_role",
                "center": rim_center_tuple,
                "axis": axis_tuple,
                "radius": float(footprint_radius),
                "depth": float(depth),
                "confidence": float(max(0.0, min(0.95, confidence))),
                "face_ids": side_wall_face_ids,
                "diagnostics": dict(diagnostics),
            },
        ),
        "feature_primitive_count": 3,
        "feature_relationships": tuple(
            {
                "relationship_kind": FeatureRelationshipKind.POCKET_CONTAINS_BORE_OPENING.value,
                "source": "recognition_component_engine.v102_compound_pocket_child_bore_relationship",
                "parent_feature_kind": "pocket",
                "child_feature_kind": "borehole",
                "role": "protected_floor_opening_relationship_metadata_only",
                "protected_loop_vertex_ids": tuple(loop),
                "relationship_semantics": "compound object: POCKET remains POCKET; BORE remains BORE; this metadata protects the floor hole during POCKET rebuild",
            }
            for loop in protected_loop_vertices
        ),
        "feature_relationship_count": int(len(protected_loop_vertices)),
        "x1_primitive_bridge_contract": "mesh-native pocket recess descriptor; rebuild target is owned pocket floor plus side-wall, not a bore wall and not a parent-surface cap",
        "delete_patch_request_allowed": True,
        "rebuild_target_policy_allowed": True,
        "rebuild_target_policy_reason": "POCKET recess-cup rebuild is allowed; delete patch is owned side-wall plus owned floor, transition faces excluded",
        "recognition_rule": "v102_pocket_hypothesis_floor_sidewall_ownership_with_child_bore_relationship_metadata",
        "feature_ownership_source": "recognition_component_engine_pocket_hypothesis_floor_sidewall_role_ownership_child_bore_relationships",
        "feature_ownership_split": "rebuild_face_ids_are_owned_pocket_side_wall_plus_floor_faces_transition_excluded",
        "pocket_rebuild_enable_scope": "owned_floor_plus_side_wall_recess_cup_protect_child_bore_openings",
        "pocket_rebuild_floor_policy": "floor_faces_are_owned_floor_role_and_rebuilt_as_bottom_of_recess",
        "diagnostics": dict(diagnostics),
    }


def _detect_pocket_preview_candidates(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    valid_face_ids: tuple[int, ...],
    region_frame: _Frame,
    seed_face_set: set[int],
    adjacency: Mapping[int, tuple[int, ...]],
    selected_opening_frame_resolver: Mapping[str, object],
    region_diagnostics: Mapping[str, object],
    edge_scale: float,
    region_face_count: int,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Recognize rebuildable POCKET hypothesis candidates from RegionData evidence.

    This detector intentionally stays in Recognition.  Region Select still owns
    only the neutral AOI.  The detector requires a recessed floor, side-wall
    evidence, and positive depth before emitting preview CandidateData.
    """

    base_diag: dict[str, object] = {
        "pocket_recognition_version": "v102_pocket_hypothesis_floor_wall_depth_with_child_bore_relationships",
        "pocket_semantic_order": "RegionData -> pocket floor/wall/depth evidence -> optional protected child-bore opening relationship -> POCKET hypothesis -> floor/side-wall ownership -> accepted POCKET CandidateData -> pocket recess-cup target",
        "pocket_rebuild_authorized": True,
        "pocket_rebuild_block_reason": "",
        "pocket_rebuild_policy": "delete/rebuild owned pocket side-wall plus floor as recessed cup; top opening remains open; child bore floor boundaries are protected relationship metadata; transitions remain excluded",
    }
    if len(valid_face_ids) < 8 or faces.ndim != 2 or face_centroids.ndim != 2:
        return (), {**base_diag, "pocket_status": "invalid_or_too_small_region"}

    resolver_resolved = bool(selected_opening_frame_resolver.get("resolved", False))
    rim_center = np.asarray(
        selected_opening_frame_resolver.get("center", region_frame.center),
        dtype=float,
    ).reshape(3)
    axis_value = selected_opening_frame_resolver.get("axis", region_frame.axis) if resolver_resolved else region_frame.axis
    pocket_axis = canonical_axis(axis_value)
    footprint_radius = _safe_float(
        selected_opening_frame_resolver.get("radius", region_frame.radius),
        _safe_float(region_frame.radius, 0.0),
    )
    if footprint_radius <= 1.0e-9:
        footprint_radius = max(_safe_float(region_frame.radius, 0.0), 1.0)

    frame = _Frame(center=rim_center, axis=pocket_axis, radius=footprint_radius)
    local = _local_arrays(frame=frame, face_ids=valid_face_ids, face_centroids=face_centroids, face_normals=face_normals)
    axial = local["axial"]
    radial = local["radial"]
    normal_axis_abs = local["normal_axis_abs"]
    finite = local["finite"]
    fid_to_local = {int(fid): int(i) for i, fid in enumerate(valid_face_ids)}
    raw_edge_scale = max(float(edge_scale), 1.0e-9)
    radial_p80 = _percentile(radial[finite], 80.0, footprint_radius) if len(radial) else footprint_radius
    footprint_limit = max(1.40 * float(footprint_radius), float(radial_p80) + 2.0 * raw_edge_scale, 4.0 * raw_edge_scale)
    # Pocket depth can be smaller than the local triangle size on coarse test
    # meshes.  Use edge scale as evidence tolerance, not as a hard multi-edge
    # minimum, otherwise a legitimate shallow/coarse pocket floor is rejected.
    min_depth = max(0.25 * raw_edge_scale, 0.04 * max(float(footprint_radius), 1.0), 0.05)

    floor_like_ids = tuple(
        int(fid)
        for fid, idx in fid_to_local.items()
        if bool(finite[idx])
        and float(normal_axis_abs[idx]) >= 0.82
        and abs(float(axial[idx])) >= float(min_depth)
        and float(radial[idx]) <= float(footprint_limit)
    )
    if not floor_like_ids:
        return (), {
            **base_diag,
            "pocket_status": "no_recessed_floor_evidence",
            "pocket_floor_candidate_count": 0,
            "pocket_min_depth": float(min_depth),
            "pocket_footprint_limit": float(footprint_limit),
        }

    floor_components = connected_face_components(faces, floor_like_ids)
    floor_rows: list[dict[str, object]] = []
    min_floor_faces = max(2, int(0.0015 * max(region_face_count, 1)))
    for comp in floor_components:
        comp_ids = tuple_ints(comp)
        idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
        if idx.size == 0:
            continue
        ax = axial[idx]
        rd = radial[idx]
        na = normal_axis_abs[idx]
        face_count = int(len(comp_ids))
        floor_level = float(np.median(ax)) if ax.size else 0.0
        depth_abs = abs(float(floor_level))
        radial_max = float(np.max(rd)) if rd.size else 0.0
        radial_median = float(np.median(rd)) if rd.size else 0.0
        normal_median = float(np.median(na)) if na.size else 0.0
        if face_count < min_floor_faces or depth_abs < min_depth:
            continue
        footprint_ratio = float(radial_max / max(footprint_limit, 1.0e-9))
        score = (
            1.25 * min(depth_abs / max(2.0 * min_depth, 1.0e-9), 1.0)
            + 0.90 * min(face_count / max(16.0, float(min_floor_faces)), 1.0)
            + 0.75 * max(0.0, min((normal_median - 0.80) / 0.20, 1.0))
            + 0.45 * max(0.0, 1.0 - min(footprint_ratio, 1.0))
        )
        floor_rows.append({
            "face_ids": comp_ids,
            "face_count": face_count,
            "floor_axial_level": float(floor_level),
            "depth_abs": float(depth_abs),
            "depth_sign": 1.0 if floor_level >= 0.0 else -1.0,
            "radial_max": float(radial_max),
            "radial_median": float(radial_median),
            "normal_axis_abs_median": float(normal_median),
            "score": float(score),
        })
    floor_rows.sort(key=lambda row: (-_safe_float(row.get("score", 0.0), 0.0), -int(row.get("face_count", 0) or 0)))
    if not floor_rows:
        return (), {
            **base_diag,
            "pocket_status": "floor_evidence_rejected",
            "pocket_floor_candidate_count": int(len(floor_components)),
            "pocket_min_floor_faces": int(min_floor_faces),
            "pocket_min_depth": float(min_depth),
        }

    floor = floor_rows[0]
    primary_floor_level = _safe_float(floor.get("floor_axial_level", 0.0), 0.0)
    primary_depth = abs(float(primary_floor_level))
    floor_level_tolerance = max(2.0 * raw_edge_scale, 0.08 * max(primary_depth, min_depth), 0.05)
    merged_floor_ids: set[int] = set()
    merged_floor_rows: list[dict[str, object]] = []
    for row in floor_rows:
        level = _safe_float(row.get("floor_axial_level", 0.0), 0.0)
        if (level >= 0.0) != (primary_floor_level >= 0.0):
            continue
        if abs(float(level - primary_floor_level)) > floor_level_tolerance:
            continue
        merged_floor_ids.update(int(fid) for fid in tuple_ints(row.get("face_ids", ())))
        merged_floor_rows.append(dict(row))
    floor_face_ids = tuple(sorted(merged_floor_ids)) or tuple_ints(floor.get("face_ids", ()))
    floor_face_set = {int(fid) for fid in floor_face_ids}
    floor_level = float(primary_floor_level)
    depth_sign = 1.0 if floor_level >= 0.0 else -1.0
    depth = abs(float(floor_level))
    if len(merged_floor_rows) > 1:
        floor = {
            **dict(floor),
            "merged_floor_component_count": int(len(merged_floor_rows)),
            "merged_floor_face_count": int(len(floor_face_ids)),
            "floor_merge_level_tolerance": float(floor_level_tolerance),
        }
    depth_lo = -0.08 * depth
    depth_hi = 1.10 * depth
    floor_union_idx = np.asarray([fid_to_local[int(fid)] for fid in floor_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_radial_max = float(np.max(radial[floor_union_idx])) if floor_union_idx.size else _safe_float(floor.get("radial_max", footprint_radius), footprint_radius)
    wall_radial_limit = max(footprint_limit, floor_radial_max + 2.0 * raw_edge_scale, float(footprint_radius) + 4.0 * raw_edge_scale)

    wall_proposal_ids = tuple(
        int(fid)
        for fid, idx in fid_to_local.items()
        if bool(finite[idx])
        and int(fid) not in floor_face_set
        and float(normal_axis_abs[idx]) <= 0.68
        and depth_lo <= float(axial[idx]) * depth_sign <= depth_hi
        and float(radial[idx]) <= float(wall_radial_limit)
    )
    wall_components = connected_face_components(faces, wall_proposal_ids) if wall_proposal_ids else ()
    accepted_wall_ids: set[int] = set()
    embedded_bore_wall_ids: set[int] = set()
    wall_rows: list[dict[str, object]] = []
    min_wall_faces = max(2, int(0.001 * max(region_face_count, 1)))
    for comp in wall_components:
        comp_ids = tuple_ints(comp)
        idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
        if idx.size == 0:
            continue
        comp_set = {int(fid) for fid in comp_ids}
        ax_depth = axial[idx] * depth_sign
        na = normal_axis_abs[idx]
        face_count = int(len(comp_ids))
        ax_min = float(np.min(ax_depth)) if ax_depth.size else 0.0
        ax_max = float(np.max(ax_depth)) if ax_depth.size else 0.0
        coverage = max(0.0, min(ax_max, depth) - max(ax_min, 0.0)) / max(depth, 1.0e-9)
        touches_floor = any(int(nb) in floor_face_set for fid in comp_set for nb in adjacency.get(int(fid), ()))
        touches_seed = any(int(nb) in seed_face_set for fid in comp_set for nb in adjacency.get(int(fid), ())) or bool(comp_set & seed_face_set)
        comp_radial_median = float(np.median(radial[idx])) if idx.size else 0.0
        comp_radial_max = float(np.max(radial[idx])) if idx.size else 0.0
        # Compound POCKET+BORE rule: a wall-like component that touches the
        # pocket floor but does not touch the selected pocket rim/opening is
        # child-BORE wall evidence, not pocket side-wall ownership.  Keep this
        # as relationship/protected-boundary metadata; do not promote a fake
        # pocket_with_bore family and do not swallow the BORE wall into POCKET.
        embedded_bore_like = bool(
            touches_floor
            and not touches_seed
            and face_count >= min_wall_faces
            and comp_radial_median <= 0.92 * max(float(footprint_radius), 1.0e-9)
        )
        accepted = bool(
            face_count >= min_wall_faces
            and coverage >= 0.18
            and not embedded_bore_like
            and (touches_seed or coverage >= 0.38 or comp_radial_median >= 0.72 * max(float(footprint_radius), 1.0e-9))
        )
        score = (
            1.20 * min(coverage / 0.70, 1.0)
            + 0.65 * min(face_count / max(12.0, float(min_wall_faces)), 1.0)
            + (0.55 if touches_floor else 0.0)
            + (0.35 if touches_seed else 0.0)
            + 0.35 * max(0.0, 1.0 - min(float(np.median(na)) / 0.68 if na.size else 1.0, 1.0))
        )
        wall_rows.append({
            "face_ids": comp_ids,
            "face_count": face_count,
            "accepted": bool(accepted),
            "axial_coverage": float(coverage),
            "touches_floor": bool(touches_floor),
            "touches_seed": bool(touches_seed),
            "embedded_bore_like_relationship_evidence": bool(embedded_bore_like),
            "radial_median": float(comp_radial_median),
            "radial_max": float(comp_radial_max),
            "normal_axis_abs_median": float(np.median(na)) if na.size else 1.0,
            "score": float(score),
        })
        if embedded_bore_like:
            embedded_bore_wall_ids.update(int(fid) for fid in comp_ids)
        if accepted:
            accepted_wall_ids.update(int(fid) for fid in comp_ids)

    if not accepted_wall_ids:
        return (), {
            **base_diag,
            "pocket_status": "no_side_wall_evidence",
            "pocket_floor_face_count": int(len(floor_face_ids)),
            "pocket_side_wall_candidate_component_count": int(len(wall_components)),
            "pocket_side_wall_component_reports": tuple(wall_rows[:12]),
            "pocket_depth": float(depth),
        }

    side_wall_face_ids = tuple(sorted(accepted_wall_ids))
    owned_set = set(side_wall_face_ids) | floor_face_set
    neighbor_ids = {
        int(nb)
        for fid in owned_set
        for nb in adjacency.get(int(fid), ())
        if int(nb) in fid_to_local and int(nb) not in owned_set
    }
    transition_ids: list[int] = []
    for fid in sorted(neighbor_ids):
        idx = fid_to_local[int(fid)]
        local_depth = float(axial[idx]) * depth_sign
        if -0.12 * depth <= local_depth <= 1.12 * depth and float(radial[idx]) <= wall_radial_limit + 2.0 * raw_edge_scale:
            na = float(normal_axis_abs[idx])
            if 0.30 < na < 0.84:
                transition_ids.append(int(fid))
    transition_face_ids = tuple_ints(transition_ids)
    floor_boundary_metadata = _pocket_floor_compound_boundary_metadata(
        faces=faces,
        vertices=vertices,
        floor_face_ids=floor_face_ids,
        center=rim_center,
        axis=pocket_axis,
    )
    protected_bore_opening_loops = tuple(dict(row) for row in tuple(floor_boundary_metadata.get("pocket_protected_floor_bore_opening_loops", ()) or ()))
    embedded_bore_wall_evidence_face_ids = tuple(sorted(embedded_bore_wall_ids))

    side_idx = np.asarray([fid_to_local[int(fid)] for fid in side_wall_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_idx = np.asarray([fid_to_local[int(fid)] for fid in floor_face_ids if int(fid) in fid_to_local], dtype=np.int64)
    floor_center = np.mean(face_centroids[list(floor_face_ids), :3], axis=0) if floor_face_ids else (rim_center + pocket_axis * floor_level)
    side_wall_coverage = 0.0
    if side_idx.size:
        side_depth = axial[side_idx] * depth_sign
        side_wall_coverage = max(0.0, min(float(np.max(side_depth)), depth) - max(float(np.min(side_depth)), 0.0)) / max(depth, 1.0e-9)
    floor_conf = min(1.0, len(floor_face_ids) / max(10.0, float(min_floor_faces)))
    wall_conf = min(1.0, len(side_wall_face_ids) / max(10.0, float(min_wall_faces)))
    depth_conf = min(1.0, depth / max(2.0 * min_depth, 1.0e-9))
    confidence = float(max(0.05, min(0.92, 0.20 + 0.22 * floor_conf + 0.24 * wall_conf + 0.20 * depth_conf + 0.16 * side_wall_coverage)))
    pocket_kind = "circular" if resolver_resolved and footprint_radius > 1.0e-9 and int(selected_opening_frame_resolver.get("edge_count", selected_opening_frame_resolver.get("expanded_edge_count", 0)) or 0) >= 8 else "freeform"

    heuristic_results = (
        make_heuristic_result("SelectedBoundaryToPocketRimHeuristic", input_count=int(region_diagnostics.get("selected_edge_count", 0) or 0), proposal_count=1, accepted_count=1 if resolver_resolved else 0, diagnostics={"resolved": bool(resolver_resolved), "footprint_radius": float(footprint_radius)}),
        make_heuristic_result("MeasurePocketOpening", input_count=int(region_face_count), proposal_count=1, accepted_count=1, confidence=float(confidence), diagnostics={"pocket_axis": to_vector3(pocket_axis), "pocket_footprint_radius": float(footprint_radius)}),
        make_heuristic_result("PocketFloorSurfaceSearchHeuristic", input_count=int(region_face_count), proposal_count=int(len(floor_components)), accepted_count=1, proposal_face_ids=floor_face_ids, diagnostics={"pocket_floor_face_count": int(len(floor_face_ids)), "pocket_floor_axial_level": float(floor_level)}),
        make_heuristic_result("PocketSideWallSearchHeuristic", input_count=int(region_face_count), proposal_count=int(len(wall_components)), accepted_count=int(len(side_wall_face_ids)), proposal_face_ids=side_wall_face_ids, diagnostics={"pocket_side_wall_face_count": int(len(side_wall_face_ids)), "pocket_side_wall_coverage": float(side_wall_coverage)}),
        make_heuristic_result("PocketDepthResolver", input_count=1, proposal_count=1, accepted_count=1, confidence=float(depth_conf), diagnostics={"pocket_depth": float(depth), "pocket_depth_sign": float(depth_sign), "pocket_depth_valid": True}),
        make_heuristic_result("PocketTransitionSeparation", input_count=int(len(owned_set)), proposal_count=int(len(transition_face_ids)), accepted_count=int(len(transition_face_ids)), proposal_face_ids=transition_face_ids, diagnostics={"pocket_transition_face_count": int(len(transition_face_ids))}),
        make_heuristic_result("PocketRoleOwnership", input_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_count=2, accepted_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_face_ids=tuple_ints(tuple(floor_face_ids) + tuple(side_wall_face_ids)), diagnostics={"pocket_floor_face_count": int(len(floor_face_ids)), "pocket_side_wall_face_count": int(len(side_wall_face_ids))}),
        make_heuristic_result("PocketChildBoreBoundaryRelationship", input_count=int(len(floor_face_ids)), proposal_count=int(len(protected_bore_opening_loops)), accepted_count=int(len(protected_bore_opening_loops)), proposal_face_ids=embedded_bore_wall_evidence_face_ids, diagnostics={"pocket_protected_floor_bore_opening_count": int(len(protected_bore_opening_loops)), "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)), "semantic_output": "relationship metadata only; no pocket_with_bore family"}),
        make_heuristic_result("EmitPocketCandidateData", input_count=int(len(floor_face_ids) + len(side_wall_face_ids)), proposal_count=1, accepted_count=1, proposal_face_ids=tuple_ints(tuple(floor_face_ids) + tuple(side_wall_face_ids)), diagnostics={"semantic_output": "POCKET CandidateData recess-cup rebuild", "compound_relationship_count": int(len(protected_bore_opening_loops))}),
    )

    candidate_diag = {
        **base_diag,
        "pocket_status": "candidate_emitted_rebuildable_pocket_recess_cup_trial",
        "pocket_kind": str(pocket_kind),
        "pocket_axis": to_vector3(pocket_axis),
        "pocket_rim_center": to_vector3(rim_center),
        "pocket_floor_center": to_vector3(floor_center),
        "pocket_footprint_radius": float(footprint_radius),
        "pocket_footprint_limit": float(footprint_limit),
        "pocket_min_depth": float(min_depth),
        "pocket_depth": float(depth),
        "pocket_depth_sign": float(depth_sign),
        "pocket_floor_candidate_count": int(len(floor_components)),
        "pocket_floor_face_count": int(len(floor_face_ids)),
        "pocket_floor_best_report": dict(floor),
        "pocket_side_wall_candidate_component_count": int(len(wall_components)),
        "pocket_side_wall_face_count": int(len(side_wall_face_ids)),
        "pocket_side_wall_coverage": float(side_wall_coverage),
        "pocket_side_wall_component_reports": tuple(wall_rows[:12]),
        "pocket_transition_face_count": int(len(transition_face_ids)),
        **dict(floor_boundary_metadata),
        "pocket_embedded_bore_wall_evidence_face_count": int(len(embedded_bore_wall_evidence_face_ids)),
        "pocket_embedded_bore_wall_evidence_face_ids": embedded_bore_wall_evidence_face_ids,
        "compound_pocket_bore_semantics": "POCKET remains parent candidate; BORE remains separate child candidate; floor bore loop is protected relationship metadata",
        "pocket_heuristic_recipe": POCKET_HEURISTIC_RECIPE,
        "pocket_heuristic_results": heuristic_results,
        "pocket_heuristic_result_summaries": compact_heuristic_summary(heuristic_results),
        "transition_policy": "exclude_chamfer_or_fillet_like_faces_from_floor_wall_ownership",
    }
    candidate = _pocket_candidate_contract_fields(
        candidate_id="component_engine.v102.pocket.recess_cup.1",
        side_wall_face_ids=side_wall_face_ids,
        floor_face_ids=floor_face_ids,
        transition_face_ids=transition_face_ids,
        protected_bore_opening_loops=protected_bore_opening_loops,
        embedded_bore_wall_evidence_face_ids=embedded_bore_wall_evidence_face_ids,
        confidence=float(confidence),
        depth=float(depth),
        axis=pocket_axis,
        rim_center=rim_center,
        floor_center=floor_center,
        footprint_radius=float(footprint_radius),
        pocket_kind=str(pocket_kind),
        diagnostics=candidate_diag,
    )
    return (candidate,), candidate_diag

def component_engine_feature_candidates(**kwargs: object) -> dict[str, object]:
    """Recognize physical CandidateData from explicit semantic stages.

    v1.5.2 cleanup: BORE CandidateData is no longer created from one local
    cylinder-like strip, selected-edge fragment, or RegionData extent; and
    a valid two-opening frame now creates a BORE review object even when wall
    ownership is not yet accepted.  A BORE
    candidate may be promoted only after this chain exists:

        measured selected opening A
        -> measured opposite opening B
        -> valid two-opening bore frame
        -> connected bore-wall ownership between A and B
        -> CandidateData

    CHAMFER recognition remains available, but broad/ambiguous BORE selections
    demote chamfer rows to review-only so random annular fragments do not become
    rebuildable while the selected BORE is unresolved.
    """

    faces = np.asarray(kwargs.get("faces"), dtype=np.int64)
    face_centroids = np.asarray(kwargs.get("face_centroids"), dtype=float)
    face_normals = np.asarray(kwargs.get("face_normals"), dtype=float)
    vertices = np.asarray(kwargs.get("vertices"), dtype=float)
    all_face_ids = tuple_ints(kwargs.get("face_ids", ()))
    valid_face_ids = tuple(
        int(fid) for fid in all_face_ids
        if 0 <= int(fid) < len(faces) and 0 <= int(fid) < len(face_centroids)
    )
    valid_face_set = {int(fid) for fid in valid_face_ids}
    seed_face_set = {int(fid) for fid in tuple_ints(kwargs.get("seed_face_ids", ())) if int(fid) in valid_face_set}
    resolved_seed_face_set = {int(fid) for fid in tuple_ints(kwargs.get("resolved_opening_seed_face_ids", ())) if int(fid) in valid_face_set}
    selected_seed_face_set = resolved_seed_face_set or seed_face_set

    region_diagnostics = dict(kwargs.get("region_diagnostics", {}) or {})
    selected_opening_measurement_audit = dict(kwargs.get("selected_opening_measurement_audit", {}) or {})
    selected_opening_frame_resolver = dict(kwargs.get("selected_opening_frame_resolver", {}) or {})
    two_opening_bore_frame = dict(kwargs.get("two_opening_bore_frame", {}) or {})

    selected_edge_count = int(region_diagnostics.get("selected_edge_count", len(tuple_ints(kwargs.get("selected_edge_ids", ()))) or 0) or 0)
    normalized_edge_count = int(region_diagnostics.get("normalized_edge_count", region_diagnostics.get("primary_anchor_edge_count", region_diagnostics.get("normalized_rim_edge_count", selected_edge_count))) or 0)
    region_face_count = int(len(valid_face_ids))
    broad_or_ambiguous_selection = bool(
        selected_edge_count >= 500
        or region_face_count >= 5000
        or bool(selected_opening_measurement_audit.get("raw_selection_severe_fragmentation_suspected", False))
        or bool(selected_opening_measurement_audit.get("normalized_rim_collapse_suspected", False))
    )

    if not valid_face_ids or faces.ndim != 2 or face_centroids.ndim != 2:
        return {
            "candidate_data": (),
            "features": (),
            "diagnostics": {
                "component_engine_version": 93,
                "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
                "recognition_cleanup": "invalid_or_empty_region_input",
                "region_face_count": int(region_face_count),
            },
            "promoted_candidate_count": 0,
        }

    # ------------------------------------------------------------------
    # Frame authority: BORE may use only a valid two-opening frame.
    # RegionData frame remains diagnostic context for CHAMFER only.
    # ------------------------------------------------------------------
    two_opening_valid = bool(two_opening_bore_frame.get("valid", False))
    bore_frame_reason = str(two_opening_bore_frame.get("status", "") or "")
    if two_opening_valid:
        # v1.5.9 semantic/numeric fix:
        # The two-opening frame axis is directed evidence from selected opening A
        # toward opposite opening B.  Do not canonicalize it independently,
        # because canonical_axis may flip -Z to +Z while bore_center remains at
        # opening A.  That puts the axial search interval outside the physical
        # bore and makes mouth/chamfer faces look like the only candidate.
        opening_center_arr = np.asarray(
            two_opening_bore_frame.get("opening_center", two_opening_bore_frame.get("center", (0.0, 0.0, 0.0))),
            dtype=float,
        ).reshape(3)
        opposite_center_arr = np.asarray(
            two_opening_bore_frame.get("opposite_center", opening_center_arr),
            dtype=float,
        ).reshape(3)
        center_delta = opposite_center_arr - opening_center_arr
        center_delta_len = float(np.linalg.norm(center_delta))
        raw_axis = np.asarray(two_opening_bore_frame.get("axis", center_delta if center_delta_len > 1.0e-12 else (0.0, 0.0, 1.0)), dtype=float).reshape(3)
        raw_axis_len = float(np.linalg.norm(raw_axis))
        if np.isfinite(center_delta_len) and center_delta_len > 1.0e-12:
            directed_axis = center_delta / center_delta_len
            if np.isfinite(raw_axis_len) and raw_axis_len > 1.0e-12:
                raw_axis_unit = raw_axis / raw_axis_len
                if float(np.dot(raw_axis_unit, directed_axis)) >= 0.0:
                    directed_axis = raw_axis_unit
                else:
                    directed_axis = -raw_axis_unit
            bore_depth_from_centers = center_delta_len
        elif np.isfinite(raw_axis_len) and raw_axis_len > 1.0e-12:
            directed_axis = raw_axis / raw_axis_len
            bore_depth_from_centers = _safe_float(two_opening_bore_frame.get("depth", 0.0), 0.0)
        else:
            directed_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            bore_depth_from_centers = _safe_float(two_opening_bore_frame.get("depth", 0.0), 0.0)
        bore_center = opening_center_arr
        bore_axis = directed_axis
        bore_radius = _safe_float(two_opening_bore_frame.get("radius", 0.0), 0.0)
        bore_depth = _safe_float(two_opening_bore_frame.get("depth", bore_depth_from_centers), bore_depth_from_centers)
        if not np.isfinite(bore_depth) or bore_depth <= 1.0e-12:
            bore_depth = float(bore_depth_from_centers)
        bore_axial_min = 0.0
        bore_axial_max = float(bore_depth)
        two_opening_axis_direction_preserved = True
        two_opening_axis_dot_opening_to_opposite = float(np.dot(bore_axis, center_delta / center_delta_len)) if np.isfinite(center_delta_len) and center_delta_len > 1.0e-12 else 0.0
    else:
        bore_center = np.asarray(kwargs.get("region_center", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        bore_axis = canonical_axis(kwargs.get("region_axis", (0.0, 0.0, 1.0)))
        bore_radius = _safe_float(kwargs.get("region_radius", 0.0), 0.0)
        bore_depth = 0.0
        bore_axial_min = 0.0
        bore_axial_max = 0.0
        two_opening_axis_direction_preserved = False
        two_opening_axis_dot_opening_to_opposite = 0.0

    region_frame = _Frame(
        center=kwargs.get("region_center", bore_center),
        axis=kwargs.get("region_axis", bore_axis),
        radius=kwargs.get("region_radius", bore_radius),
    )
    # v1.7.4 semantic scale correction:
    # The RegionData cutout is neutral AOI evidence and may include large cap,
    # panel, or previously rebuilt faces.  Its median triangle edge length must
    # not become BORE-wall ownership scale authority.  Use it only as a raw
    # diagnostic, then clamp the role scale to the selected/opening ring
    # measurement.  Otherwise a clean full-depth bore wall can be rejected as
    # "axial_span_too_small" because a broad AOI face made edge_scale larger
    # than the physical bore-wall sampling scale.
    raw_region_edge_scale = _edge_median_length(vertices, faces, valid_face_ids)
    edge_scale = float(raw_region_edge_scale)
    adjacency = face_adjacency_for_patch(faces, valid_face_ids)

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
        normals = _unit_rows(face_normals)
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
                    "selected_opening_support_weak": bool(selected_opening_support_weak),
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
            local_idx = np.asarray([fid_to_local[int(fid)] for fid in comp_ids if int(fid) in fid_to_local], dtype=np.int64)
            if local_idx.size == 0:
                continue
            comp_set = {int(fid) for fid in comp_ids}
            direct_seed = comp_set & selected_seed_face_set
            adjacent_seed = set()
            for fid in comp_set:
                if any(int(nb) in selected_seed_face_set for nb in adjacency.get(int(fid), ())):
                    adjacent_seed.add(int(fid))
            seed_related = bool(direct_seed or adjacent_seed)
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
            rejection_reasons: list[str] = []
            if selected_opening_support_weak:
                rejection_reasons.append("selected_opening_primary_component_too_weak_for_wall_ownership")
            if face_count < min_faces:
                rejection_reasons.append("wall_component_too_few_faces")
            spans_frame_well = bool(axial_coverage >= 0.78)
            endpoint_supported = bool(touches_both_openings or spans_frame_well)
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
                "touches_selected_opening": bool(touches_opening_a),
                "touches_opposite_opening": bool(touches_opening_b),
                "touches_both_openings": bool(touches_both_openings),
                "opening_touch_tolerance": float(opening_touch_tolerance),
                "min_distance_to_selected_opening": float(min_distance_to_opening_a),
                "min_distance_to_opposite_opening": float(min_distance_to_opening_b),
                "accepted_as_bore_wall_ownership": bool(valid_wall),
                "selected_opening_support_weak": bool(selected_opening_support_weak),
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
            true_endpoint_reconciled = bool(
                defect_boundary_count > 2
                and true_endpoint_gap > true_endpoint_reconcile_epsilon
                and true_endpoint_gap <= true_endpoint_max_extension
            )
            operational_depth_candidate = float(true_endpoint_depth_candidate if true_endpoint_reconciled else owned_depth_candidate)
            owned_frame_depth = float(measured_frame_depth)
            owned_frame_reconciled = False
            owned_opposite_center = None
            owned_frame = dict(two_opening_bore_frame)
            depth_reconcile_threshold = max(0.08 * max(float(measured_frame_depth), 1.0), 2.0 * edge_scale, 0.75)
            if defect_boundary_count > 2 and operational_depth_candidate > float(measured_frame_depth) + depth_reconcile_threshold:
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
            comp_ids = tuple_ints(bore_wall_owned_face_ids)
            bore_owned_face_count = int(len(comp_ids))
            bore_heuristic_results = _heuristic_results_for_current_bore_state(
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
                "endpoint_support_face_count": int(len(tuple(endpoint_support_face_ids or ()))),
                "endpoint_support_face_ids": tuple_ints(endpoint_support_face_ids),
                "endpoint_support_promoted_to_wall_ownership": False,
                "endpoint_support_delete_patch_allowed": False,
                "terminal_wall_candidate_face_count": int(len(tuple(terminal_wall_candidate_face_ids or ()))),
                "terminal_wall_face_count": int(len(tuple(terminal_wall_face_ids or ()))),
                "terminal_wall_face_ids": tuple_ints(terminal_wall_face_ids),
                "terminal_wall_completion_used": bool(terminal_wall_completion_used),
                "terminal_wall_completion_rejection_reasons": tuple(terminal_wall_rejection_reasons),
                "true_endpoint_extension_added_face_count": int(endpoint_extension_added_face_count),
                "bore_wall_owned_face_count_before_terminal_completion": int(len(core_bore_wall_owned_face_ids)),
                "bore_wall_owned_face_count_after_terminal_completion": int(len(bore_wall_owned_face_ids)),
                "bore_wall_owned_face_ids": tuple_ints(bore_wall_owned_face_ids),
                "full_depth_display_face_count": int(len(comp_ids)),
                "candidate_face_policy": "full_depth_bore_wall_faces_terminal_role_filtered_endpoint_support_diagnostic",
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
            candidate = _bore_candidate_contract_fields(
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
                "bore_axis_source": "directed_opening_to_opposite_axis_no_canonical_flip",
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
            # v1.5.9: emit the measured-frame review row even when it has
            # no face-owned preview.  Face IDs are not required for frame evidence.
            if review_face_ids or bool(review_diag.get("frame_only_candidate", False)):
                bore_features.append(
                    _bore_review_candidate_contract_fields(
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
                _bore_selected_opening_review_candidate_contract_fields(
                    candidate_id="component_engine.v82.bore.selected_opening_review.1",
                    face_ids=opening_face_ids,
                    confidence=float(max(0.05, min(0.74, _safe_float(selected_opening_frame_resolver.get("confidence", 0.35), 0.35)))),
                    radius=float(_safe_float(selected_opening_frame_resolver.get("radius", kwargs.get("region_radius", 0.0)), 0.0)),
                    diagnostics=opening_diag,
                )
            )

    # ------------------------------------------------------------------
    # CHAMFER recognition.  Keep the existing clean chamfer evidence transform,
    # but demote broad/ambiguous BORE selections to review-only.
    # ------------------------------------------------------------------
    local = _local_arrays(frame=region_frame, face_ids=valid_face_ids, face_centroids=face_centroids, face_normals=face_normals)
    axial_all = local["axial"]
    radial_all = local["radial"]
    normal_axis_abs_all = local["normal_axis_abs"]
    radial_normal_alignment_all = local["radial_normal_alignment"]
    axial_span_all = float(np.max(axial_all) - np.min(axial_all)) if len(axial_all) else 0.0
    radial_scale = max(region_frame.radius, _percentile(radial_all, 75.0, 1.0), 1.0)
    min_faces_general = max(18, int(0.0025 * max(region_face_count, 1)))
    min_radial_span = max(0.015 * radial_scale, 2.0 * edge_scale, 0.18)
    min_axial_span = max(0.015 * radial_scale, 2.0 * edge_scale, 0.18)

    components = connected_face_components(faces, valid_face_ids)
    fid_to_local = {int(fid): int(i) for i, fid in enumerate(valid_face_ids)}

    pocket_features, pocket_diag = _detect_pocket_preview_candidates(
        faces=faces,
        vertices=vertices,
        face_centroids=face_centroids,
        face_normals=face_normals,
        valid_face_ids=valid_face_ids,
        region_frame=region_frame,
        seed_face_set=selected_seed_face_set or seed_face_set,
        adjacency=adjacency,
        selected_opening_frame_resolver=selected_opening_frame_resolver,
        region_diagnostics=region_diagnostics,
        edge_scale=edge_scale,
        region_face_count=region_face_count,
    )

    chamfer_rows: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
    chamfer_rejected: list[dict[str, object]] = []
    for comp in components:
        st = _component_stats(
            comp=tuple_ints(comp),
            fid_to_local=fid_to_local,
            axial=axial_all,
            radial=radial_all,
            normal_axis_abs=normal_axis_abs_all,
            radial_normal_alignment=radial_normal_alignment_all,
            seed_face_set=seed_face_set,
            adjacency=adjacency,
        )
        face_count = int(st.get("face_count", 0) or 0)
        radial_span = _safe_float(st.get("radial_span", 0.0), 0.0)
        axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
        normal_axis = _safe_float(st.get("normal_axis_abs_median", 0.0), 0.0)
        radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)
        accept_evidence = bool(
            face_count >= (4 if bool(st.get("seed_related", False)) else min_faces_general)
            and radial_span >= (0.55 * min_radial_span if bool(st.get("seed_related", False)) else min_radial_span)
            and axial_span >= (0.55 * min_axial_span if bool(st.get("seed_related", False)) else min_axial_span)
            and normal_axis >= 0.15
            and radial_align >= 0.15
        )
        score = _chamfer_score(st, radius_scale=max(radial_scale, 1.0), axial_span_all=max(axial_span_all, 1.0e-9))
        chamfer_region_fraction = float(face_count) / max(float(region_face_count), 1.0)
        chamfer_too_large_for_annular_transition = bool(chamfer_region_fraction > 0.45)
        st = {
            **st,
            "score": float(score),
            "accepted_as_annular_transition_evidence": bool(accept_evidence and not chamfer_too_large_for_annular_transition),
            "chamfer_region_fraction": float(chamfer_region_fraction),
            "chamfer_too_large_for_annular_transition": bool(chamfer_too_large_for_annular_transition),
            "chamfer_action_demoted_by_bore_resolution": bool(broad_or_ambiguous_selection or two_opening_valid or chamfer_too_large_for_annular_transition),
        }
        if accept_evidence:
            chamfer_rows.append((score, tuple_ints(comp), st))
        else:
            chamfer_rejected.append(st)
    chamfer_rows.sort(key=lambda item: (-item[0], -int(item[2].get("seed_related", False)), -len(item[1]), item[1][:1]))

    features: list[dict[str, object]] = []
    features.extend(bore_features)
    features.extend(pocket_features)

    # v1.5.8: show the mouth/chamfer transition as CHAMFER review evidence,
    # not as BORE.  This gives the operator two clean, semantically distinct
    # rows: BORE opening/frame evidence and CHAMFER mouth-transition evidence.
    if bore_resolution_active and bore_mouth_transition_face_ids:
        mt = dict(bore_mouth_transition_diag or {})
        mt_accepted = bool(
            int(mt.get("face_count", 0) or 0) >= 12
            and bool(mt.get("touches_selected_opening", False))
            and not bool(mt.get("touches_opposite_opening", False))
            and _safe_float(mt.get("axial_span", 0.0), 0.0) <= max(0.18 * float(bore_depth), 2.5 * float(bore_radius), 1.0)
        )
        features.append(
            _candidate_contract_fields(
                candidate_id="component_engine.v82.chamfer.mouth_transition_rebuild.1" if mt_accepted else "component_engine.v82.chamfer.mouth_transition_review.1",
                face_ids=tuple_ints(bore_mouth_transition_face_ids),
                accepted=bool(mt_accepted),
                confidence=float(max(0.05, min(0.88, 0.32 + 0.08 * _safe_float(mt.get("score", 0.0), 0.0)))),
                radius_inner=_safe_float(mt.get("radius_inner", bore_radius), bore_radius),
                radius_outer=_safe_float(mt.get("radius_outer", bore_radius), bore_radius),
                axial_span=_safe_float(mt.get("axial_span", 0.0), 0.0),
                diagnostics={
                    **mt,
                    "recognition_rule": "v79_mouth_transition_is_chamfer_not_bore_wall_rebuildable",
                    "bore_resolution_active": bool(bore_resolution_active),
                    "rebuild_authority": "accepted_clean_mouth_transition_chamfer_owned_faces",
                },
            )
        )

    for index, (score, comp, st) in enumerate(chamfer_rows[:8], start=1):
        # v1.5.4: a full-RegionData chamfer row is not a surface-ownership
        # preview.  It is broad context evidence, so do not emit it as a
        # CandidateView while BORE resolution is active.
        if bool(st.get("chamfer_too_large_for_annular_transition", False)) and bool(bore_resolution_active):
            chamfer_rejected.append({
                **dict(st),
                "chamfer_display_suppressed_by_bore_resolution": True,
                "chamfer_display_suppress_reason": "chamfer_candidate_matches_full_regiondata_not_annular_surface_ownership",
            })
            continue
        comp_ids = tuple_ints(comp)
        radius_inner = _safe_float(st.get("radial_min", 0.0), 0.0)
        radius_outer = _safe_float(st.get("radial_max", radius_inner), radius_inner)
        axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
        confidence = max(0.05, min(0.92, 0.22 + 0.11 * float(score) + (0.07 if bool(st.get("seed_related", False)) else 0.0)))
        accepted = bool(st.get("accepted_as_annular_transition_evidence", False)) and not bool(st.get("chamfer_action_demoted_by_bore_resolution", False))
        cid = f"component_engine.v82.chamfer.{index}" if accepted else f"component_engine.v82.chamfer.review.{index}"
        features.append(
            _candidate_contract_fields(
                candidate_id=cid,
                face_ids=comp_ids,
                accepted=bool(accepted),
                confidence=float(confidence),
                radius_inner=float(radius_inner),
                radius_outer=float(radius_outer),
                axial_span=float(axial_span),
                diagnostics={
                    **st,
                    "rank": int(index),
                    "recognition_rule": "v71_chamfer_demoted_during_bore_frame_resolution",
                },
            )
        )

    promoted = tuple(item for item in features if bool(item.get("candidate_action_enabled", False)))

    # v1.5.5: produce compact top-level wall ownership reports.
    # Candidate rows may contain full face_ids for preview, but routed diagnostics
    # need small summaries that explain why ownership failed without flooding the UI.
    def _compact_wall_report(row: Mapping[str, object]) -> dict[str, object]:
        reasons = row.get("bore_wall_component_rejection_reasons", ())
        if not isinstance(reasons, (tuple, list)):
            reasons = (str(reasons),) if reasons else ()
        return {
            "face_count": int(row.get("face_count", 0) or 0),
            "accepted_as_bore_wall_ownership": bool(row.get("accepted_as_bore_wall_ownership", False)),
            "rejection_reasons": tuple(str(v) for v in tuple(reasons)),
            "reject_reason": str(row.get("bore_wall_ownership_reject_reason", "") or ""),
            "score": _safe_float(row.get("score", 0.0), 0.0),
            "axial_span": _safe_float(row.get("axial_span", 0.0), 0.0),
            "centroid_axial_span": _safe_float(row.get("centroid_axial_span", 0.0), 0.0),
            "face_span_axial_ownership_used": bool(row.get("face_span_axial_ownership_used", False)),
            "wall_span_fraction_of_two_opening_depth": _safe_float(row.get("wall_span_fraction_of_two_opening_depth", 0.0), 0.0),
            "strict_wall_band_ratio": _safe_float(row.get("strict_wall_band_ratio", 0.0), 0.0),
            "radial_rel_mad": _safe_float(row.get("radial_rel_mad", 999999.0), 999999.0),
            "radial_abs_mad": _safe_float(row.get("radial_abs_mad", 999999.0), 999999.0),
            "normal_axis_abs_median": _safe_float(row.get("normal_axis_abs_median", 1.0), 1.0),
            "radial_normal_alignment_median": _safe_float(row.get("radial_normal_alignment_median", 0.0), 0.0),
            "touches_selected_opening": bool(row.get("touches_selected_opening", False)),
            "touches_opposite_opening": bool(row.get("touches_opposite_opening", False)),
            "touches_both_openings": bool(row.get("touches_both_openings", False)),
            "min_distance_to_selected_opening": _safe_float(row.get("min_distance_to_selected_opening", 999999.0), 999999.0),
            "min_distance_to_opposite_opening": _safe_float(row.get("min_distance_to_opposite_opening", 999999.0), 999999.0),
            "opening_touch_tolerance": _safe_float(row.get("opening_touch_tolerance", 0.0), 0.0),
            "raw_region_edge_scale": _safe_float(row.get("raw_region_edge_scale", 0.0), 0.0),
            "selected_opening_edge_scale": _safe_float(row.get("selected_opening_edge_scale", 0.0), 0.0),
            "bore_role_edge_scale": _safe_float(row.get("bore_role_edge_scale", 0.0), 0.0),
            "bore_role_scale_clamped": bool(row.get("bore_role_scale_clamped", False)),
            "role_scale_authority": str(row.get("role_scale_authority", "")),
        }

    compact_bore_wall_reports = tuple(_compact_wall_report(row) for row in tuple(bore_rejected or ())[:12])
    flattened_bore_wall_reasons = tuple(sorted({
        str(reason)
        for row in compact_bore_wall_reports
        for reason in tuple(row.get("rejection_reasons", ()) or ())
        if str(reason)
    }))
    best_wall_candidate_report = compact_bore_wall_reports[0] if compact_bore_wall_reports else {}

    diag = {
        "component_engine_version": 93,
        "active_candidate_authority": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "heuristic_contract_version": HEURISTIC_CONTRACT_VERSION,
        "heuristic_registry": heuristic_registry_dict(),
        "bore_heuristic_recipe": BORE_HEURISTIC_RECIPE,
        "chamfer_heuristic_recipe": CHAMFER_HEURISTIC_RECIPE,
        "pocket_heuristic_recipe": POCKET_HEURISTIC_RECIPE,
        "pocket_detection": dict(locals().get("pocket_diag", {}) or {}),
        "pocket_candidate_count": int(len(tuple(locals().get("pocket_features", ()) or ()))),
        "pocket_floor_face_count": int((locals().get("pocket_diag", {}) or {}).get("pocket_floor_face_count", 0)) if isinstance(locals().get("pocket_diag", {}), Mapping) else 0,
        "pocket_side_wall_face_count": int((locals().get("pocket_diag", {}) or {}).get("pocket_side_wall_face_count", 0)) if isinstance(locals().get("pocket_diag", {}), Mapping) else 0,
        "pocket_transition_face_count": int((locals().get("pocket_diag", {}) or {}).get("pocket_transition_face_count", 0)) if isinstance(locals().get("pocket_diag", {}), Mapping) else 0,
        "pocket_depth": _safe_float((locals().get("pocket_diag", {}) or {}).get("pocket_depth", 0.0), 0.0) if isinstance(locals().get("pocket_diag", {}), Mapping) else 0.0,
        "heuristic_results": tuple(locals().get("bore_heuristic_results", ())),
        "heuristic_result_summaries": compact_heuristic_summary(tuple(locals().get("bore_heuristic_results", ()))),
        "heuristic_authority_policy": "heuristics_propose_measurement_quantifies_recognition_interprets_ownership_assigns",
        "heuristic_scope_fix_v1_7_3": "BORE CandidateData construction no longer reads CHAMFER-local heuristic variables before the CHAMFER recipe runs",
        "bore_role_scale_clamp_v1_7_4": "BORE wall ownership thresholds use selected-opening ring scale, not raw RegionData face scale",
        "recognition_cleanup": "patchy_bore_promotion_quarantined",
        "semantic_order": "selected_opening -> opposite_opening -> two_opening_bore_frame -> bore_wall_ownership -> CandidateData",
        "region_face_count": int(region_face_count),
        "selected_edge_count": int(selected_edge_count),
        "broad_or_ambiguous_selection": bool(broad_or_ambiguous_selection),
        "selected_opening_frame_resolved": bool(selected_opening_frame_resolver.get("resolved", False)),
        "selected_opening_frame_source": str(selected_opening_frame_resolver.get("resolver_source", "")),
        "selected_opening_primary_edge_count": int(selected_opening_frame_resolver.get("primary_edge_count", 0) or 0),
        "selected_opening_expanded_edge_count": int(selected_opening_frame_resolver.get("expanded_edge_count", 0) or 0),
        "raw_region_edge_scale": float(locals().get("raw_region_edge_scale", 0.0)),
        "selected_opening_edge_scale": float(locals().get("selected_opening_edge_scale", 0.0)),
        "bore_role_edge_scale": float(locals().get("edge_scale", 0.0)),
        "bore_role_scale_limit": float(locals().get("bore_role_scale_limit", 0.0)),
        "bore_role_scale_clamped": bool(locals().get("bore_role_scale_clamped", False)),
        "role_scale_authority": str(locals().get("role_scale_authority", "unknown")),
        "two_opening_bore_frame_valid": bool(two_opening_valid),
        "two_opening_bore_frame_status": str(two_opening_bore_frame.get("status", "")),
        "two_opening_bore_frame_depth": float(bore_depth),
        "two_opening_bore_frame_radius": float(bore_radius),
        "two_opening_axis_direction_preserved": bool(two_opening_axis_direction_preserved),
        "two_opening_axis_dot_opening_to_opposite": float(two_opening_axis_dot_opening_to_opposite),
        "bore_axis_source": "directed_opening_to_opposite_axis_no_canonical_flip" if bool(two_opening_valid) else "region_frame_context",
        "sidewall_normal_evidence_count": int(locals().get("sidewall_normal_evidence_count", 0)),
        "frame_sidewall_normal_evidence_count": int(locals().get("frame_sidewall_normal_evidence_count", 0)),
        "broad_sidewall_normal_evidence_count": int(locals().get("broad_sidewall_normal_evidence_count", 0)),
        "bore_wall_interior_margin": float(locals().get("interior_margin", 0.0)),
        "bore_wall_broad_radial_min": float(locals().get("broad_radial_min", 0.0)),
        "bore_wall_broad_radial_max": float(locals().get("broad_radial_max", 0.0)),
        "learned_wall_radius": float(locals().get("learned_wall_radius", bore_radius)),
        "learned_wall_radius_source": str(locals().get("learned_wall_radius_source", "two_opening_frame_radius")),
        "learned_wall_radius_tolerance": float(locals().get("learned_wall_radius_tol", 0.0)),
        "semantic_aggregate_sidewall_ownership_used": bool(locals().get("semantic_aggregate_sidewall_ownership_used", False)),
        "aggregate_wall_face_count": int((locals().get("aggregate_sidewall_report", {}) or {}).get("face_count", 0)) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else 0,
        "aggregate_wall_axial_coverage": _safe_float((locals().get("aggregate_sidewall_report", {}) or {}).get("wall_span_fraction_of_two_opening_depth", 0.0), 0.0) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_axial_span": _safe_float((locals().get("aggregate_sidewall_report", {}) or {}).get("axial_span", 0.0), 0.0) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_centroid_axial_span": _safe_float((locals().get("aggregate_sidewall_report", {}) or {}).get("centroid_axial_span", 0.0), 0.0) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "face_span_axial_ownership_used": bool((locals().get("aggregate_sidewall_report", {}) or {}).get("face_span_axial_ownership_used", False)) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else bool(locals().get("face_axial_span_available", False)),
        "aggregate_wall_angular_coverage": _safe_float((locals().get("aggregate_sidewall_report", {}) or {}).get("angular_coverage", 0.0), 0.0) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else 0.0,
        "aggregate_wall_rejection_reasons": tuple((locals().get("aggregate_sidewall_report", {}) or {}).get("bore_wall_component_rejection_reasons", ()) or ()) if isinstance(locals().get("aggregate_sidewall_report", {}), Mapping) else (),
        "measured_two_opening_frame_depth": _safe_float(locals().get("measured_frame_depth", locals().get("bore_depth", 0.0)), 0.0),
        "owned_bore_frame_depth": _safe_float(locals().get("owned_frame_depth", locals().get("bore_depth", 0.0)), 0.0),
        "owned_bore_frame_depth_delta": float(_safe_float(locals().get("owned_frame_depth", locals().get("bore_depth", 0.0)), 0.0) - _safe_float(locals().get("measured_frame_depth", locals().get("bore_depth", 0.0)), 0.0)),
        "owned_frame_reconciled_from_wall_ownership": bool(locals().get("owned_frame_reconciled", False)),
        "owned_wall_face_span_depth": _safe_float(locals().get("owned_axial_span", 0.0), 0.0),
        "owned_wall_axial_min": _safe_float(locals().get("owned_axial_min", 0.0), 0.0),
        "owned_wall_axial_max": _safe_float(locals().get("owned_axial_max", 0.0), 0.0),
        "region_true_endpoint_depth": _safe_float(locals().get("region_true_endpoint_depth", 0.0), 0.0),
        "true_endpoint_depth_candidate": _safe_float(locals().get("true_endpoint_depth_candidate", 0.0), 0.0),
        "true_endpoint_depth_gap": _safe_float(locals().get("true_endpoint_gap", 0.0), 0.0),
        "true_endpoint_reconcile_epsilon": _safe_float(locals().get("true_endpoint_reconcile_epsilon", 0.0), 0.0),
        "true_endpoint_max_extension": _safe_float(locals().get("true_endpoint_max_extension", 0.0), 0.0),
        "true_endpoint_reconciled_from_region_endpoint": bool(locals().get("true_endpoint_reconciled", False)),
        "endpoint_support_face_count": int(len(tuple(locals().get("endpoint_support_face_ids", ()) or ()))),
        "true_endpoint_extension_added_face_count": int(locals().get("endpoint_extension_added_face_count", 0) or 0),
        "bore_frame_depth_source": str(locals().get("candidate_diag", {}).get("bore_frame_depth_source", "measured_two_opening_frame")) if isinstance(locals().get("candidate_diag", {}), Mapping) else "measured_two_opening_frame",
        "two_opening_bore_wall_candidate_component_count": int(len(bore_wall_candidate_components)),
        "two_opening_search_boundary_face_count": int(len(bore_tube_face_ids)),
        "bore_wall_search_face_count": int(len(bore_wall_search_face_ids)),
        "selected_opening_support_weak": bool(selected_opening_support_weak),
        "bore_review_candidate_emitted": bool(any(str(item.get("feature_family", "")) == FeatureFamily.BORE.value and str(item.get("recognition_stage", "")) == RecognitionStage.REVIEW.value for item in features)),
        "bore_wall_owned_face_count": int(bore_owned_face_count),
        "bore_wall_ownership_valid": bool(bore_owned_face_count > 0),
        "bore_wall_best_diagnostics": dict(best_bore_wall_diag),
        "bore_wall_rejected_component_count": int(len(bore_rejected)),
        "bore_wall_rejected_component_summaries": tuple(bore_rejected[:12]),
        "bore_wall_component_reports": compact_bore_wall_reports,
        "bore_wall_component_rejection_reasons": flattened_bore_wall_reasons,
        "bore_wall_best_component_report": best_wall_candidate_report,
        "best_wall_candidate_face_count": int(best_wall_candidate_report.get("face_count", 0) or 0) if isinstance(best_wall_candidate_report, Mapping) else 0,
        "no_sidewall_no_bore_face_preview_used": bool(any(bool(item.get("diagnostics", {}).get("no_sidewall_no_bore_face_preview_used", False)) for item in bore_features)),
        "bore_mouth_transition_face_count": int(len(bore_mouth_transition_face_ids)),
        "bore_mouth_transition_preview_source": str((bore_mouth_transition_diag or {}).get("preview_source", "")),
        "clean_candidate_separation_used": True,
        "chamfer_candidate_count": int(len(chamfer_rows)),
        "chamfer_action_demoted_by_bore_resolution": bool(any(bool(row[2].get("chamfer_action_demoted_by_bore_resolution", False)) for row in chamfer_rows)),
        "bore_resolution_active": bool(bore_resolution_active),
        "selected_opening_review_candidate_emitted": bool(any(str(item.get("candidate_id", "")).startswith("component_engine.v82.bore.selected_opening_review") for item in features)),
        "promoted_candidate_count": int(len(promoted)),
        "candidate_summaries": tuple(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "feature_family": str(item.get("feature_family", "unknown") or "unknown"),
                "recognition_stage": str(item.get("recognition_stage", "diagnostic_only") or "diagnostic_only"),
                "face_count": int(item.get("face_count", 0) or 0),
                "accepted": bool(item.get("candidate_action_enabled", False)),
                "axial_span": _safe_float(item.get("axial_span", 0.0), 0.0),
                "confidence": _safe_float(item.get("confidence", 0.0), 0.0),
            }
            for item in features
        ),
    }
    return {
        "candidate_data": tuple(features),
        "features": tuple(features),
        "diagnostics": diag,
        "promoted_candidate_count": int(len(promoted)),
    }


def recognition_result_dict_from_component_features(
    *,
    features: tuple[dict[str, object], ...],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return a CandidateResult-like dictionary for GUI diagnostics."""

    promoted = tuple(
        item for item in tuple(features or ())
        if bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
        and str(item.get("recognition_stage", "") or "") == RecognitionStage.ACCEPTED_CANDIDATE.value
        and str(item.get("feature_family", "") or "") in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value, FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value}
    )
    return {
        "contract_type": "candidate_result",
        "engine": "surface_component_classifier_v93_heuristic_role_scale_clamp_full_depth_wall_ownership",
        "mode": "two_opening_bore_frame_to_wall_ownership_or_review_candidate",
        "candidate_count": int(len(tuple(features or ()))),
        "candidate_data": tuple(features or ()),
        "features": tuple(features or ()),
        "diagnostics": dict(diagnostics or {}),
        "promoted_candidate_count": int(len(promoted)),
    }


__all__ = [
    "component_engine_feature_candidates",
    "recognition_result_dict_from_component_features",
]
