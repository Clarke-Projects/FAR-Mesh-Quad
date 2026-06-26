"""Shared recognition helper utilities for FAR MESH BoreTool.

This module contains non-mutating helper code split out of
``recognition_component_engine.py`` during v174x.  The helpers here perform
local numeric/topology/evidence calculations used by recognition orchestration.
They do not classify features by themselves, do not assign final surface roles,
do not emit CandidateData, do not authorize DeletePatchProposal, and do not
mutate mesh topology. v175f adds a diagnostic-only secondary opening/rim
search heuristic: it proposes additional rim/opening evidence inside the
neutral AOI, but it still does not create feature identity or ownership. v175h extends that heuristic from RegionData boundary loops to internal feature-rail edge components so compound chamfer-mouth selections can propose the inner bore-mouth rim as evidence.

The split is behavior-preserving: the component engine imports these helpers
under the old private names while its public entry points and CandidateData
shape remain unchanged.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import math
import numpy as np

from ..geometry import canonical_axis
from .recognition_chamfer import (
    chamfer_band_role_proof_v174f as _chamfer_band_role_proof_v174f,
    chamfer_selected_rail_local_band_ownership_proof_v174f as _chamfer_selected_rail_local_band_ownership_proof_v174f,
)
from ..topology import (
    boundary_edges_for_face_patch,
    connected_face_components,
    edge_loop_components,
)
from ..heuristics import make_heuristic_result
from ..types import tuple_ints

RECOGNITION_COMPONENT_ENGINE_SHARED_HELPER_SPLIT_CHECKPOINT_V174X = (
    "v174x_split_component_engine_shared_measurement_topology_helpers_no_behavior_change"
)

SELECTED_ANNULAR_RAIL_CONTRACT_V173D = (
    "selected_annular_rail_role_resolver_v173d_selected_edges_are_neutral_rail_evidence_before_family_ownership"
)

SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_CHECKPOINT_V175F = (
    "v175f_secondary_opening_rim_search_heuristic_diagnostic_evidence_only"
)

SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_NAME_V175F = "SecondaryOpeningRimSearchHeuristic"

SECONDARY_INTERNAL_FEATURE_RAIL_SEARCH_CHECKPOINT_V175H = (
    "v175h_secondary_opening_rim_search_includes_internal_feature_rail_edges_diagnostic_evidence_only"
)

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)

def _face_patch_boundary_semantic_report(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Return a small semantic topology report for a candidate owned face patch.

    This is recognition / ownership evidence only.  It does not mutate topology and
    it does not claim a rebuild will succeed.  The purpose is to keep CandidateData
    honest: an owned BORE wall that is supposed to mean "wall surface bounded by
    the selected opening and the opposite/end opening" must not silently contain
    extra unclassified boundary loops.
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

def _selected_annular_rail_role_resolver_v173d(
    *,
    faces: np.ndarray,
    valid_face_ids: tuple[int, ...],
    selected_seed_face_ids: Iterable[int],
    seed_face_set: set[int],
    fid_to_local: Mapping[int, int],
    axial: np.ndarray,
    radial: np.ndarray,
    normal_axis_abs: np.ndarray,
    radial_normal_alignment: np.ndarray,
    adjacency: Mapping[int, tuple[int, ...]],
    selected_opening_frame_resolver: Mapping[str, object],
    edge_scale: float,
    radial_scale: float,
    axial_span_all: float,
) -> dict[str, object]:
    """Resolve selected circular/rim edges as neutral annular rail evidence.

    v173d semantic correction: the user-selected edge loop is not born as a
    BORE opening.  It is first a selected annular rail.  That rail may support a
    BORE opening, a CHAMFER outer/inner rail, a POCKET rim, or a relationship
    boundary.  This helper performs the first Recognition-stage meaning
    transform before family-local CandidateData builders consume the evidence.

    The helper does not authorize rebuild and does not mutate topology.  It only
    publishes role hypotheses and, when the selected seed face patch itself
    proves a CHAMFER_BAND role, returns a family-local CHAMFER evidence row for
    the independent chamfer builder.  BORE may still use the selected rail as
    evidence only when the rail is not resolved as a chamfer rail.
    """

    seed_ids = tuple_ints(selected_seed_face_ids) or tuple_ints(seed_face_set)
    valid_set = set(int(fid) for fid in valid_face_ids)
    seed_ids = tuple(int(fid) for fid in seed_ids if int(fid) in valid_set and int(fid) in fid_to_local)
    selected_radius = _safe_float(selected_opening_frame_resolver.get("radius", 0.0), 0.0)
    selected_confidence = _safe_float(selected_opening_frame_resolver.get("confidence", 0.0), 0.0)
    out: dict[str, object] = {
        "selected_annular_rail_role_resolver_v173d": True,
        "selected_annular_rail_contract_v173d": SELECTED_ANNULAR_RAIL_CONTRACT_V173D,
        "selected_annular_rail_face_count_v173d": int(len(seed_ids)),
        "selected_annular_rail_selected_radius_v173d": float(selected_radius),
        "selected_annular_rail_selected_confidence_v173d": float(selected_confidence),
        "selected_annular_rail_primary_role_v173d": "unresolved_annular_rail",
        "selected_annular_rail_role_hypotheses_v173d": (),
        "selected_annular_rail_chamfer_candidate_rows_v173d": (),
        "selected_annular_rail_semantic_rule_v173d": (
            "selected edge IDs become SelectedAnnularRailEvidence first; BORE, CHAMFER, and POCKET builders consume rail "
            "role hypotheses independently; relationship rails do not create cross-family CandidateData"
        ),
    }
    if not seed_ids:
        return out

    # v173e: incident face-side analysis before any BORE interpretation.
    #
    # v173d only evaluated the raw selected seed patch.  On a real selected rail,
    # the seed patch can contain both sides of the rail (parent/chamfer,
    # chamfer/bore, or pocket/chamfer).  Proving CHAMFER ownership from that
    # mixed incident set is impossible and the engine fell back to BORE review.
    #
    # The selected rail must therefore be split into local face-side role
    # evidence first.  We grow a small incident neighborhood from the selected
    # faces and extract chamfer-like transition faces: neither flat cap/parent
    # faces nor pure cylindrical wall faces, but faces whose normals carry both
    # axial and radial components.  These local components are the input to the
    # independent CHAMFER_BAND builder.
    seed_components = tuple(connected_face_components(np.asarray(faces, dtype=np.int64), seed_ids))
    candidate_sets: list[tuple[str, tuple[int, ...]]] = []
    for idx, comp in enumerate(seed_components):
        comp_ids = tuple_ints(comp)
        if comp_ids:
            candidate_sets.append((f"selected_seed_component_{idx}", comp_ids))
    if len(seed_components) != 1:
        candidate_sets.append(("selected_seed_union", seed_ids))

    seed_idx = np.asarray([fid_to_local[int(fid)] for fid in seed_ids if int(fid) in fid_to_local], dtype=np.int64)
    seed_axial_center = float(np.median(axial[seed_idx])) if seed_idx.size else 0.0
    seed_radial_center = float(np.median(radial[seed_idx])) if seed_idx.size else float(selected_radius)
    if not np.isfinite(seed_radial_center) or seed_radial_center <= 1.0e-9:
        seed_radial_center = float(selected_radius) if selected_radius > 1.0e-9 else max(float(radial_scale), 1.0)

    local_seen: set[int] = set(int(fid) for fid in seed_ids)
    frontier: set[int] = set(int(fid) for fid in seed_ids)
    # Four adjacency steps are enough to cross from a selected rail into the
    # annular band on coarse triangulations without walking the whole RegionData
    # volume.  The radial/axial/normal filters below are the real semantic fence.
    for _depth in range(4):
        nxt: set[int] = set()
        for fid in frontier:
            for nb in adjacency.get(int(fid), ()):  # selected-region adjacency only
                nb_i = int(nb)
                if nb_i in valid_set and nb_i not in local_seen:
                    nxt.add(nb_i)
        if not nxt:
            break
        local_seen.update(nxt)
        frontier = nxt
        if len(local_seen) >= 900:
            break

    radial_window = max(5.0 * max(float(edge_scale), 1.0e-9), 0.42 * max(seed_radial_center, selected_radius, 1.0), 0.75)
    axial_window = max(5.0 * max(float(edge_scale), 1.0e-9), 0.35 * max(seed_radial_center, selected_radius, 1.0), 0.75)
    chamfer_like_local_ids: list[int] = []
    for fid in sorted(local_seen):
        local = fid_to_local.get(int(fid))
        if local is None or local < 0 or local >= len(radial):
            continue
        na = float(normal_axis_abs[local])
        ra = float(radial_normal_alignment[local])
        rd = float(radial[local])
        ax = float(axial[local])
        if not (math.isfinite(na) and math.isfinite(ra) and math.isfinite(rd) and math.isfinite(ax)):
            continue
        # Role meaning, not permission: remove the two neighboring surface roles.
        # Flat parent/floor/cap: high axis normal, almost no radial component.
        if na >= 0.88 and ra <= 0.22:
            continue
        # Cylindrical BORE/POCKET wall: almost no axial normal, strong radial wall normal.
        if na <= 0.085 and ra >= 0.78:
            continue
        # CHAMFER_BAND candidate: transition normal has both axial and radial content.
        if na < 0.10 or ra < 0.10:
            continue
        if abs(rd - seed_radial_center) > radial_window:
            continue
        if abs(ax - seed_axial_center) > axial_window:
            continue
        chamfer_like_local_ids.append(int(fid))

    # v173f: strict selected-rail CHAMFER band extraction.
    #
    # v173e still failed because the incident set remained too broad: it found a
    # BORE review component but did not isolate the actual shallow annular
    # transition band.  The selected rail must first be interpreted as a *local
    # rail-side surface problem*: find the small transition strip next to the
    # rail, not every sidewall-like face in the RegionData volume.
    #
    # This stricter pass uses the selected rail radius/axial location and the
    # CHAMFER normal role directly.  It also enforces a local face-count window
    # based on the selected rail segment count, so a long BORE wall or broad
    # parent patch cannot become the selected-rail chamfer island.
    selected_segment_count_v173f = _safe_int(
        selected_opening_frame_resolver.get(
            "expanded_edge_count",
            selected_opening_frame_resolver.get(
                "edge_count",
                selected_opening_frame_resolver.get("primary_edge_count", len(seed_ids)),
            ),
        ),
        len(seed_ids),
    )
    selected_segment_count_v173f = max(int(selected_segment_count_v173f), 3)
    local_chamfer_face_max_v173f = max(96, int(8 * selected_segment_count_v173f))
    local_chamfer_face_min_v173f = max(4, int(0.25 * selected_segment_count_v173f))
    strict_radial_inward_window_v173f = max(0.62 * max(seed_radial_center, selected_radius, 1.0), 2.2 * max(float(edge_scale), 1.0e-9), 1.25)
    strict_radial_outward_window_v173f = max(0.20 * max(seed_radial_center, selected_radius, 1.0), 1.2 * max(float(edge_scale), 1.0e-9), 0.65)
    strict_axial_window_v173f = max(0.55 * max(seed_radial_center, selected_radius, 1.0), 1.8 * max(float(edge_scale), 1.0e-9), 1.25)
    strict_chamfer_band_ids_v173f: list[int] = []
    for fid in sorted(local_seen):
        local = fid_to_local.get(int(fid))
        if local is None or local < 0 or local >= len(radial):
            continue
        na = float(normal_axis_abs[local])
        ra = float(radial_normal_alignment[local])
        rd = float(radial[local])
        ax = float(axial[local])
        if not (math.isfinite(na) and math.isfinite(ra) and math.isfinite(rd) and math.isfinite(ax)):
            continue
        # Independent CHAMFER_BAND role: transition normals contain both axial
        # and radial components.  These limits deliberately exclude the parent
        # cap/floor role and the pure cylindrical BORE/POCKET wall role before
        # topology/loop proof is evaluated.
        # v173g: do not require a steep chamfer angle at this rail stage.
        # Coarse/remeshed chamfers can be shallow; the old v173f 0.30/0.30
        # normal floor missed the working CHAMFER cases and left only BORE
        # review evidence.  The rail-anchored component search only needs
        # transition-role evidence here; rebuild.py remains the topology
        # validator.
        if not (0.12 <= na <= 0.96 and 0.12 <= ra <= 0.99):
            continue
        if rd < seed_radial_center - strict_radial_inward_window_v173f:
            continue
        if rd > seed_radial_center + strict_radial_outward_window_v173f:
            continue
        if abs(ax - seed_axial_center) > strict_axial_window_v173f:
            continue
        strict_chamfer_band_ids_v173f.append(int(fid))

    strict_chamfer_components_v173f = (
        tuple(connected_face_components(np.asarray(faces, dtype=np.int64), tuple_ints(strict_chamfer_band_ids_v173f)))
        if strict_chamfer_band_ids_v173f else ()
    )
    strict_component_count_v173f = 0
    strict_component_reject_reports_v173f: list[dict[str, object]] = []
    seed_id_set_v173f = set(seed_ids)
    for idx, comp in enumerate(strict_chamfer_components_v173f):
        comp_ids = tuple_ints(comp)
        if not comp_ids:
            continue
        comp_set = set(int(v) for v in comp_ids)
        touches_seed = bool(comp_set & seed_id_set_v173f) or any(
            int(nb) in seed_id_set_v173f
            for fid in comp_set
            for nb in adjacency.get(int(fid), ())
        )
        local_count_ok = bool(local_chamfer_face_min_v173f <= len(comp_ids) <= local_chamfer_face_max_v173f)
        if touches_seed and local_count_ok:
            candidate_sets.insert(0, (f"v173f_selected_rail_strict_chamfer_band_component_{idx}", comp_ids))
            strict_component_count_v173f += 1
        else:
            strict_component_reject_reports_v173f.append({
                "source": f"v173f_selected_rail_strict_chamfer_band_component_{idx}",
                "face_count": int(len(comp_ids)),
                "touches_selected_rail_seed": bool(touches_seed),
                "local_count_ok": bool(local_count_ok),
                "min_faces": int(local_chamfer_face_min_v173f),
                "max_faces": int(local_chamfer_face_max_v173f),
            })

    incident_chamfer_components = tuple(connected_face_components(np.asarray(faces, dtype=np.int64), tuple_ints(chamfer_like_local_ids))) if chamfer_like_local_ids else ()
    incident_component_count = 0
    incident_component_reject_reports_v173f: list[dict[str, object]] = []
    for idx, comp in enumerate(incident_chamfer_components):
        comp_ids = tuple_ints(comp)
        if not comp_ids:
            continue
        comp_set = set(int(v) for v in comp_ids)
        touches_seed = bool(comp_set & seed_id_set_v173f) or any(
            int(nb) in seed_id_set_v173f
            for fid in comp_set
            for nb in adjacency.get(int(fid), ())
        )
        local_count_ok = bool(local_chamfer_face_min_v173f <= len(comp_ids) <= local_chamfer_face_max_v173f)
        if not touches_seed or not local_count_ok:
            incident_component_reject_reports_v173f.append({
                "source": f"v173e_incident_chamfer_like_component_{idx}",
                "face_count": int(len(comp_ids)),
                "touches_selected_rail_seed": bool(touches_seed),
                "local_count_ok": bool(local_count_ok),
                "min_faces": int(local_chamfer_face_min_v173f),
                "max_faces": int(local_chamfer_face_max_v173f),
            })
            continue
        # The incident component is a role-island candidate.  Put it before the
        # mixed seed patch so the CHAMFER builder sees it before broad BORE
        # review evidence.
        candidate_sets.insert(0, (f"v173e_incident_chamfer_like_component_{idx}", comp_ids))
        incident_component_count += 1

    out.update({
        "selected_annular_rail_role_resolver_v173e": True,
        "selected_annular_rail_role_resolver_v173f": True,
        "selected_annular_rail_semantic_stage_v173e": "SelectedAnnularRailEvidence -> incident_face_side_role_islands -> family_local_ownership",
        "selected_annular_rail_semantic_stage_v173f": "SelectedAnnularRailEvidence -> selected_rail_local_chamfer_band_role_island -> family_local_CHAMFER_ownership",
        "selected_annular_rail_local_neighborhood_face_count_v173e": int(len(local_seen)),
        "selected_annular_rail_chamfer_like_local_face_count_v173e": int(len(chamfer_like_local_ids)),
        "selected_annular_rail_incident_chamfer_component_count_v173e": int(incident_component_count),
        "selected_rail_strict_chamfer_band_face_count_v173f": int(len(strict_chamfer_band_ids_v173f)),
        "selected_rail_strict_chamfer_component_count_v173f": int(strict_component_count_v173f),
        "selected_rail_strict_chamfer_component_reject_reports_v173f": tuple(strict_component_reject_reports_v173f[:8]),
        "selected_rail_incident_component_reject_reports_v173f": tuple(incident_component_reject_reports_v173f[:8]),
        "selected_rail_local_chamfer_face_count_window_v173f": (int(local_chamfer_face_min_v173f), int(local_chamfer_face_max_v173f)),
        "selected_rail_anchored_annular_transition_scan_v173g": True,
        "selected_rail_anchored_annular_transition_normal_floor_v173g": (0.12, 0.12),
        "selected_rail_anchored_annular_transition_rule_v173g": "selected rail anchors the adjacent annular transition band; seed faces need not themselves be the full chamfer band",
        "selected_rail_strict_radial_window_v173f": (float(strict_radial_inward_window_v173f), float(strict_radial_outward_window_v173f)),
        "selected_rail_strict_axial_window_v173f": float(strict_axial_window_v173f),
        "selected_annular_rail_seed_radial_center_v173e": float(seed_radial_center),
        "selected_annular_rail_seed_axial_center_v173e": float(seed_axial_center),
        "selected_annular_rail_radial_window_v173e": float(radial_window),
        "selected_annular_rail_axial_window_v173e": float(axial_window),
    })

    min_faces = 4
    min_radial_span = max(0.45 * max(float(edge_scale), 1.0e-9), 0.010 * max(float(radial_scale), 1.0), 0.08)
    min_axial_span = max(0.45 * max(float(edge_scale), 1.0e-9), 0.010 * max(float(radial_scale), 1.0), 0.08)
    hypotheses: list[dict[str, object]] = []
    chamfer_rows: list[dict[str, object]] = []

    for source, ids in candidate_sets:
        st = _component_stats(
            comp=ids,
            fid_to_local=fid_to_local,
            axial=axial,
            radial=radial,
            normal_axis_abs=normal_axis_abs,
            radial_normal_alignment=radial_normal_alignment,
            seed_face_set=set(seed_ids),
            adjacency=adjacency,
        )
        face_count = int(st.get("face_count", 0) or 0)
        radial_span = _safe_float(st.get("radial_span", 0.0), 0.0)
        axial_span = _safe_float(st.get("axial_span", 0.0), 0.0)
        radial_min = _safe_float(st.get("radial_min", 0.0), 0.0)
        radial_max = _safe_float(st.get("radial_max", radial_min), radial_min)
        radial_median = _safe_float(st.get("radial_median", 0.0), 0.0)
        normal_axis = _safe_float(st.get("normal_axis_abs_median", 0.0), 0.0)
        radial_align = _safe_float(st.get("radial_normal_alignment_median", 0.0), 0.0)
        role_proof = _chamfer_band_role_proof_v174f(
            faces=faces,
            face_ids=ids,
            component_stats=st,
        )
        score = _chamfer_score(st, radius_scale=max(radial_scale, 1.0), axial_span_all=max(axial_span_all, 1.0e-9))
        rail_side = "unknown_chamfer_rail"
        if selected_radius > 1.0e-9:
            tol = max(2.0 * max(float(edge_scale), 1.0e-9), 0.10 * max(selected_radius, 1.0), 0.12)
            if selected_radius >= radial_max - tol:
                rail_side = "chamfer_outer_rail"
            elif selected_radius <= radial_min + tol:
                rail_side = "chamfer_inner_rail"
            else:
                rail_side = "chamfer_mid_band_or_relationship_rail"
        transition_like = bool(
            face_count >= min_faces
            and radial_span >= min_radial_span
            and axial_span >= min_axial_span
            and normal_axis >= 0.12
            and radial_align >= 0.12
        )
        chamfer_band_valid = bool(role_proof.get("chamfer_band_role_valid_v173c", False))
        # v173o: the selected rail may create CHAMFER ownership only after a
        # local transition-band proof.  This keeps the v173h/v173g successful
        # selected-rail extractor, but records/measures the missing semantic
        # mapping: selected rail anchor -> one local band component -> two rail
        # relationship -> normal transition -> axial/radial thickness -> angular
        # support -> contained CHAMFER_BAND ownership.
        local_count_ok_v173g = bool(local_chamfer_face_min_v173f <= face_count <= local_chamfer_face_max_v173f)
        seed_related_v173g = bool(st.get("seed_related", False))
        ownership_proof_v173o = _chamfer_selected_rail_local_band_ownership_proof_v174f(
            faces=faces,
            face_ids=ids,
            component_stats=st,
            selected_radius=float(selected_radius),
            selected_confidence=float(selected_confidence),
            selected_segment_count=int(max(1, round(selected_segment_count_v173f))),
            edge_scale=float(edge_scale),
            radial_scale=float(radial_scale),
            axial_span_all=float(axial_span_all),
            role_proof=role_proof,
        )
        selected_rail_local_chamfer_owned_v173g = bool(
            transition_like
            and local_count_ok_v173g
            and seed_related_v173g
            and bool(ownership_proof_v173o.get("chamfer_band_ownership_proof_valid_v173o", False))
        )
        role = rail_side if selected_rail_local_chamfer_owned_v173g else "possible_bore_opening_rail_or_relationship_boundary"
        row = {
            "source": source,
            "face_count": int(face_count),
            "face_ids": ids,
            "role_hypothesis": role,
            "transition_like": bool(transition_like),
            "chamfer_band_role_valid": bool(chamfer_band_valid),
            "selected_rail_local_chamfer_band_owned_v173g": bool(selected_rail_local_chamfer_owned_v173g),
            "selected_rail_local_count_ok_v173g": bool(local_count_ok_v173g),
            "selected_rail_seed_related_v173g": bool(seed_related_v173g),
            **ownership_proof_v173o,
            "score": float(score),
            "radial_min": float(radial_min),
            "radial_max": float(radial_max),
            "radial_median": float(radial_median),
            "radial_span": float(radial_span),
            "axial_span": float(axial_span),
            "normal_axis_abs_median": float(normal_axis),
            "radial_normal_alignment_median": float(radial_align),
            **role_proof,
        }
        hypotheses.append(row)
        if selected_rail_local_chamfer_owned_v173g:
            st_for_builder = {
                **dict(st),
                **dict(role_proof),
                "score": float(score + 4.0),
                "accepted_as_annular_transition_evidence": True,
                "seed_related": True,
                # v173g promotes the rail-anchored annular transition role even
                # when strict rebuild-loop proof is not yet available.  This is
                # CandidateData ownership; rebuild.py still validates the exact
                # delete patch.
                "selected_rail_local_chamfer_band_owned_v173g": True,
                **ownership_proof_v173o,
                "selected_rail_chamfer_role_source_v173g": "selected_rail_anchored_annular_transition_component_scan_restored_from_working_chamfer_path",
                "selected_rail_chamfer_role_source_v173o": "selected_rail_local_transition_band_ownership_proof",
                "chamfer_band_role_valid_v173c": bool(chamfer_band_valid),
                "chamfer_band_role_valid_for_candidate_v173g": True,
                "selected_annular_rail_source_v173d": source,
                "selected_annular_rail_role_resolver_v173d": True,
                "selected_rail_primary_role_v173d": role,
                "selected_annular_rail_contract_v173d": SELECTED_ANNULAR_RAIL_CONTRACT_V173D,
                "selected_annular_rail_selected_radius_v173d": float(selected_radius),
                "selected_annular_rail_semantic_transform_v173d": "SelectedAnnularRailEvidence -> selected-rail-anchored CHAMFER_BAND ownership evidence",
                "chamfer_action_independent_of_bore_resolution_v173d": True,
                "cross_family_candidate_creation_allowed_v173d": False,
            }
            chamfer_rows.append({
                "score": float(score + 4.0),
                "face_ids": ids,
                "stats": st_for_builder,
            })

    if chamfer_rows:
        chamfer_rows.sort(key=lambda row: (-_safe_float(row.get("score", 0.0), 0.0), -len(tuple_ints(row.get("face_ids", ())))))
        primary = str((chamfer_rows[0].get("stats", {}) or {}).get("selected_rail_primary_role_v173d", "chamfer_rail"))
        out.update({
            "selected_annular_rail_primary_role_v173d": primary,
            "selected_annular_rail_consumed_by_family_v173d": "chamfer",
            "selected_annular_rail_chamfer_candidate_count_v173d": int(len(chamfer_rows)),
            "selected_annular_rail_chamfer_candidate_rows_v173d": tuple(chamfer_rows[:4]),
        })
    else:
        out.update({
            "selected_annular_rail_consumed_by_family_v173d": "bore_or_relationship_review",
            "selected_annular_rail_chamfer_candidate_count_v173d": 0,
        })
    # Keep face_id payloads out of the public diagnostics row; CandidateData rows
    # carry actual face IDs when emitted.
    public_hypotheses = []
    for row in hypotheses[:8]:
        public_hypotheses.append({k: v for k, v in row.items() if k != "face_ids" and not str(k).endswith("boundary_report_v173c")})
    out["selected_annular_rail_role_hypotheses_v173d"] = tuple(public_hypotheses)
    return out

def _face_vertices_for_ids(faces: np.ndarray, face_ids: Iterable[int]) -> set[int]:
    """Return source vertex ids touched by the given face ids."""

    out: set[int] = set()
    arr = np.asarray(faces, dtype=np.int64)
    for fid in tuple_ints(face_ids):
        if 0 <= int(fid) < len(arr):
            for v in arr[int(fid), :3]:
                out.add(int(v))
    return out

def _face_edges_for_ids(faces: np.ndarray, face_ids: Iterable[int]) -> set[tuple[int, int]]:
    """Return undirected source edges touched by the given face ids."""

    out: set[tuple[int, int]] = set()
    arr = np.asarray(faces, dtype=np.int64)
    for fid in tuple_ints(face_ids):
        if 0 <= int(fid) < len(arr):
            tri = [int(v) for v in arr[int(fid), :3]]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if a == b:
                    continue
                out.add((a, b) if a < b else (b, a))
    return out

def _loop_records_vertices_and_edges(records: Iterable[Mapping[str, object]]) -> tuple[set[int], set[tuple[int, int]]]:
    """Collect protected-loop vertices/edges from diagnostic loop records."""

    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for row in tuple(records or ()):  # tolerate generators and None-like inputs
        for v in tuple(row.get("vertices", ()) or ()):  # type: ignore[union-attr]
            try:
                vertices.add(int(v))
            except Exception:
                continue
        for edge in tuple(row.get("edges", ()) or ()):  # type: ignore[union-attr]
            key = _normalize_edge_key(edge)
            if key != (-1, -1):
                edges.add(key)
    return vertices, edges

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



def _coerce_vector3_v175f(value: object, default: tuple[float, float, float]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        arr = np.asarray(default, dtype=float).reshape(3)
    out = np.asarray(default, dtype=float).reshape(3).copy()
    if arr.size >= 3:
        out[:] = arr[:3]
    length = float(np.linalg.norm(out))
    if not np.isfinite(length) or length <= 1.0e-12:
        out = np.asarray(default, dtype=float).reshape(3).copy()
    return out.astype(float, copy=False)


def _unit_vector_v175f(value: object, default: tuple[float, float, float]) -> np.ndarray:
    arr = _coerce_vector3_v175f(value, default)
    length = float(np.linalg.norm(arr))
    if not np.isfinite(length) or length <= 1.0e-12:
        arr = np.asarray(default, dtype=float).reshape(3)
        length = float(np.linalg.norm(arr))
    if not np.isfinite(length) or length <= 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / length


def _ring_fit_from_loop_vertices_v175f(
    *,
    vertices: np.ndarray,
    vertex_ids: Iterable[int],
    axis_hint: np.ndarray,
) -> dict[str, object] | None:
    ids = tuple(sorted({int(v) for v in tuple(vertex_ids or ()) if 0 <= int(v) < len(vertices)}))
    if len(ids) < 6:
        return None
    pts = np.asarray(vertices, dtype=float)[list(ids), :3]
    if pts.ndim != 2 or pts.shape[0] < 6 or pts.shape[1] < 3:
        return None
    center = np.mean(pts, axis=0)
    centered = pts - center.reshape(1, 3)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    if vh.shape[0] < 3:
        return None
    axis = vh[-1].astype(float, copy=False)
    axis_len = float(np.linalg.norm(axis))
    if not np.isfinite(axis_len) or axis_len <= 1.0e-12:
        axis = _unit_vector_v175f(axis_hint, (0.0, 0.0, 1.0))
    else:
        axis = axis / axis_len
    hint = _unit_vector_v175f(axis_hint, (0.0, 0.0, 1.0))
    if float(np.dot(axis, hint)) < 0.0:
        axis = -axis
    basis_u = vh[0].astype(float, copy=False)
    basis_u -= axis * float(np.dot(basis_u, axis))
    basis_u_len = float(np.linalg.norm(basis_u))
    if not np.isfinite(basis_u_len) or basis_u_len <= 1.0e-12:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, axis))) > 0.90:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = fallback - axis * float(np.dot(fallback, axis))
        basis_u_len = float(np.linalg.norm(basis_u))
    basis_u = basis_u / max(basis_u_len, 1.0e-12)
    basis_v = np.cross(axis, basis_u)
    basis_v_len = float(np.linalg.norm(basis_v))
    if not np.isfinite(basis_v_len) or basis_v_len <= 1.0e-12:
        return None
    basis_v = basis_v / basis_v_len
    xy = np.column_stack((centered @ basis_u, centered @ basis_v))
    radii0 = np.linalg.norm(xy, axis=1)
    radius0 = float(np.median(radii0)) if radii0.size else 0.0
    circle_center_2d = np.zeros(2, dtype=float)
    radius = radius0
    try:
        a = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
        b = xy[:, 0] ** 2 + xy[:, 1] ** 2
        sol, *_rest = np.linalg.lstsq(a, b, rcond=None)
        cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
        rr = float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))
        if np.isfinite(rr) and rr > 1.0e-12:
            circle_center_2d = np.array([cx, cy], dtype=float)
            radius = rr
    except Exception:
        pass
    fit_center = center + basis_u * float(circle_center_2d[0]) + basis_v * float(circle_center_2d[1])
    deltas = pts - fit_center.reshape(1, 3)
    plane_dist = deltas @ axis
    radial_vec = deltas - plane_dist.reshape(-1, 1) * axis.reshape(1, 3)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    residual = radial_dist - float(radius)
    radius_rms = float(math.sqrt(float(np.mean(residual * residual)))) if residual.size else 0.0
    plane_rms = float(math.sqrt(float(np.mean(plane_dist * plane_dist)))) if plane_dist.size else 0.0
    radius_rel_rms = float(radius_rms / max(float(radius), 1.0e-12))
    plane_rel_rms = float(plane_rms / max(float(radius), 1.0e-12))
    angles = np.arctan2(radial_vec @ basis_v, radial_vec @ basis_u)
    angular_coverage = 0.0
    max_gap_degrees = 360.0
    if angles.size >= 3:
        wrapped = np.sort((angles + 2.0 * math.pi) % (2.0 * math.pi))
        gaps = np.diff(np.concatenate([wrapped, wrapped[:1] + 2.0 * math.pi]))
        max_gap = float(np.max(gaps)) if gaps.size else 2.0 * math.pi
        angular_coverage = float(max(0.0, min(1.0, 1.0 - max_gap / (2.0 * math.pi))))
        max_gap_degrees = float(math.degrees(max_gap))
    circularity = float(max(0.0, min(1.0, 1.0 - radius_rel_rms)))
    return {
        "vertex_ids": ids,
        "center": (float(fit_center[0]), float(fit_center[1]), float(fit_center[2])),
        "axis": (float(axis[0]), float(axis[1]), float(axis[2])),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "radius_rms": float(radius_rms),
        "radius_rel_rms": float(radius_rel_rms),
        "plane_rms": float(plane_rms),
        "plane_rel_rms": float(plane_rel_rms),
        "circularity": float(circularity),
        "angular_coverage": float(angular_coverage),
        "max_angular_gap_degrees": float(max_gap_degrees),
        "point_count": int(len(ids)),
    }



def _face_normals_for_region_v175h(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return unit normals for triangular runtime faces; diagnostic helper only."""

    verts = np.asarray(vertices, dtype=float)
    arr = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] < 3 or arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=float)
    tri = verts[arr[:, :3], :3]
    raw = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    lengths = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw, dtype=float)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    out[valid] = raw[valid] / lengths[valid].reshape(-1, 1)
    return out


def _internal_feature_rail_edge_components_v175h(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    valid_face_ids: Iterable[int],
    selected_center: np.ndarray,
    selected_axis: np.ndarray,
    selected_radius: float,
    edge_scale: float,
    max_edges: int = 4096,
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    """Find internal AOI rail-like edge components as heuristic rim evidence.

    This is not feature identity and not ownership.  It only proposes additional
    annular rail/opening evidence inside RegionData, especially the inner rail
    between an accepted CHAMFER band and a BORE wall when the user clicked the
    outer chamfer rail.  Boundary-loop search alone cannot see that internal
    rail because it is not necessarily a RegionData boundary.
    """

    arr = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(vertices, dtype=float)
    ids = tuple_ints(valid_face_ids)
    diag: dict[str, object] = {
        "secondary_opening_rim_internal_rail_checkpoint_v175h": SECONDARY_INTERNAL_FEATURE_RAIL_SEARCH_CHECKPOINT_V175H,
        "secondary_opening_rim_internal_rail_search_used_v175h": False,
        "secondary_opening_rim_internal_rail_candidate_edge_count_v175h": 0,
        "secondary_opening_rim_internal_rail_component_count_v175h": 0,
        "secondary_opening_rim_internal_rail_rejected_reason_v175h": "not_evaluated",
        "secondary_opening_rim_internal_rail_semantic_rule_v175h": (
            "Internal rail edges are heuristic OpeningRimEvidenceProposal support only; they are not BORE/CHAMFER/POCKET identity, ownership, CandidateData, or DeletePatchProposal."
        ),
    }
    if arr.ndim != 2 or arr.shape[1] < 3 or verts.ndim != 2 or verts.shape[1] < 3 or not ids:
        diag["secondary_opening_rim_internal_rail_rejected_reason_v175h"] = "invalid_or_empty_input"
        return (), diag
    valid_set = {int(fid) for fid in ids if 0 <= int(fid) < len(arr)}
    if not valid_set:
        diag["secondary_opening_rim_internal_rail_rejected_reason_v175h"] = "no_valid_face_ids"
        return (), diag
    normals = _face_normals_for_region_v175h(verts, arr)
    if normals.shape[0] != len(arr):
        diag["secondary_opening_rim_internal_rail_rejected_reason_v175h"] = "normal_build_failed"
        return (), diag

    selected_axis_u = _unit_vector_v175f(selected_axis, (0.0, 0.0, 1.0))
    selected_center_v = np.asarray(selected_center, dtype=float).reshape(-1)[:3]
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fid in valid_set:
        face = arr[int(fid), :3]
        for edge in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = _normalize_edge_key(edge)
            if key[0] < 0 or key[1] < 0:
                continue
            edge_to_faces.setdefault(key, []).append(int(fid))

    candidate_edges: list[tuple[int, int]] = []
    rejected_not_internal = 0
    rejected_geometry = 0
    # Internal feature rails should live near the selected rail plane in the
    # compound chamfer-mouth case.  Keep this broad enough for coarse meshes.
    axial_window = max(0.28 * max(float(selected_radius), 1.0), 12.0 * float(edge_scale), 1.25)
    for edge, adj in edge_to_faces.items():
        if len(adj) != 2:
            rejected_not_internal += 1
            continue
        a, b = int(edge[0]), int(edge[1])
        if a < 0 or b < 0 or a >= len(verts) or b >= len(verts):
            rejected_geometry += 1
            continue
        p0 = verts[a, :3]
        p1 = verts[b, :3]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            rejected_geometry += 1
            continue
        mid = 0.5 * (p0 + p1)
        rel = mid - selected_center_v
        axial = float(np.dot(rel, selected_axis_u))
        radial_vec = rel - selected_axis_u * axial
        radial_len = float(np.linalg.norm(radial_vec))
        if abs(axial) > axial_window:
            rejected_geometry += 1
            continue
        if selected_radius > 1.0e-9 and radial_len > selected_radius * 1.18:
            # For the inner bore-mouth rail we expect same or smaller radius than
            # the clicked outer chamfer rail.  Larger rings are still available
            # from RegionData boundary loops; avoid broad exterior seams here.
            rejected_geometry += 1
            continue
        edir = p1 - p0
        elen = float(np.linalg.norm(edir))
        if elen <= 1.0e-12:
            rejected_geometry += 1
            continue
        edir = edir / elen
        radial_u = radial_vec / max(radial_len, 1.0e-12)
        tangent_like = bool(abs(float(np.dot(edir, selected_axis_u))) <= 0.42 and abs(float(np.dot(edir, radial_u))) <= 0.62)
        n0 = normals[int(adj[0]), :3]
        n1 = normals[int(adj[1]), :3]
        normal_dot = float(abs(np.dot(n0, n1))) if np.all(np.isfinite(n0)) and np.all(np.isfinite(n1)) else 1.0
        crease_like = bool(normal_dot <= 0.988)
        radial_relation_ok = bool(selected_radius <= 1.0e-9 or radial_len <= selected_radius * 1.05)
        if tangent_like and crease_like and radial_relation_ok:
            candidate_edges.append((a, b) if a < b else (b, a))
            if len(candidate_edges) >= int(max_edges):
                break
        else:
            rejected_geometry += 1

    if not candidate_edges:
        diag.update({
            "secondary_opening_rim_internal_rail_search_used_v175h": True,
            "secondary_opening_rim_internal_rail_rejected_reason_v175h": "no_internal_feature_rail_edges_passed_filters",
            "secondary_opening_rim_internal_rail_boundary_or_noninternal_rejected_v175h": int(rejected_not_internal),
            "secondary_opening_rim_internal_rail_geometry_rejected_v175h": int(rejected_geometry),
        })
        return (), diag
    try:
        comps = edge_loop_components(tuple(sorted(set(candidate_edges))))
    except Exception as exc:
        diag.update({
            "secondary_opening_rim_internal_rail_search_used_v175h": True,
            "secondary_opening_rim_internal_rail_rejected_reason_v175h": f"edge_component_build_failed:{exc}",
            "secondary_opening_rim_internal_rail_candidate_edge_count_v175h": int(len(set(candidate_edges))),
        })
        return (), diag
    # Keep compact annular components; reject tiny triangulation fragments.
    out_components: list[tuple[tuple[int, int], ...]] = []
    for comp in tuple(comps or ()):  # edge_loop_components returns edge components, not guaranteed clean loops
        comp_edges = tuple(_normalize_edge_key(edge) for edge in tuple(comp or ()))
        comp_edges = tuple(edge for edge in comp_edges if edge[0] >= 0 and edge[1] >= 0)
        if len(comp_edges) >= 6:
            out_components.append(tuple(sorted(set(comp_edges))))
    diag.update({
        "secondary_opening_rim_internal_rail_search_used_v175h": True,
        "secondary_opening_rim_internal_rail_rejected_reason_v175h": "" if out_components else "internal_feature_rail_components_too_small",
        "secondary_opening_rim_internal_rail_candidate_edge_count_v175h": int(len(set(candidate_edges))),
        "secondary_opening_rim_internal_rail_component_count_v175h": int(len(out_components)),
        "secondary_opening_rim_internal_rail_component_edge_counts_v175h": tuple(int(len(c)) for c in out_components[:8]),
        "secondary_opening_rim_internal_rail_boundary_or_noninternal_rejected_v175h": int(rejected_not_internal),
        "secondary_opening_rim_internal_rail_geometry_rejected_v175h": int(rejected_geometry),
    })
    return tuple(out_components[:8]), diag


def secondary_opening_rim_search_heuristic_v175f(
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    valid_face_ids: Iterable[int],
    selected_opening_frame_resolver: Mapping[str, object],
    region_frame: object,
    edge_scale: float,
    accepted_chamfer_present: bool = False,
    max_proposals: int = 8,
) -> dict[str, object]:
    """Search the neutral AOI for additional rim/opening evidence proposals.

    This is a Recognition-local heuristic stage, not feature identity and not
    surface ownership.  It consumes RegionData/AOI topology plus the selected
    rail context and proposes additional annular rim evidence that later family
    recognizers may measure.  It does not emit CandidateData, authorize rebuild,
    or mutate topology.
    """

    ids = tuple_ints(valid_face_ids)
    selected_center = _coerce_vector3_v175f(
        selected_opening_frame_resolver.get("center", getattr(region_frame, "center", (0.0, 0.0, 0.0))),
        (0.0, 0.0, 0.0),
    )
    selected_axis = _unit_vector_v175f(
        selected_opening_frame_resolver.get("axis", getattr(region_frame, "axis", (0.0, 0.0, 1.0))),
        (0.0, 0.0, 1.0),
    )
    selected_radius = _safe_float(
        selected_opening_frame_resolver.get("radius", selected_opening_frame_resolver.get("radius_nominal", getattr(region_frame, "radius", 0.0))),
        0.0,
    )
    selected_confidence = _safe_float(selected_opening_frame_resolver.get("confidence", 0.0), 0.0)
    out: dict[str, object] = {
        "secondary_opening_rim_search_checkpoint_v175f": SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_CHECKPOINT_V175F,
        "secondary_opening_rim_search_used_v175f": False,
        "secondary_opening_rim_search_semantic_stage_v175f": "RegionData/AOI -> SecondaryOpeningRimSearchHeuristic -> OpeningRimEvidenceProposal",
        "secondary_opening_rim_search_authority_v175f": "heuristic_evidence_proposal_only",
        "secondary_opening_rim_search_not_feature_identity_v175f": True,
        "secondary_opening_rim_search_not_surface_ownership_v175f": True,
        "secondary_opening_rim_search_not_candidate_data_v175f": True,
        "secondary_opening_rim_search_not_delete_patch_v175f": True,
        "secondary_opening_rim_search_not_rebuild_v175f": True,
        "secondary_opening_rim_candidate_count_v175f": 0,
        "secondary_opening_rim_candidates_v175f": (),
        "secondary_opening_rim_rejected_reasons_v175f": (),
        "secondary_opening_rim_heuristic_result_v175f": make_heuristic_result(
            SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_NAME_V175F,
            semantic_context="neutral RegionData/AOI with selected annular rail evidence",
            output_proposal_type="OpeningRimEvidenceProposal",
            input_count=int(len(ids)),
            proposal_count=0,
            accepted_count=0,
            rejected_count=0,
            confidence=0.0,
            diagnostics={"reason": "not_evaluated"},
        ),
    }
    if not ids or np.asarray(faces).ndim != 2 or np.asarray(vertices).ndim != 2:
        out["secondary_opening_rim_rejected_reasons_v175f"] = ("invalid_or_empty_region_input",)
        return out

    rejected: list[str] = []
    try:
        boundary_edges = boundary_edges_for_face_patch(np.asarray(faces, dtype=np.int64), ids)
        loops = edge_loop_components(boundary_edges)
    except Exception as exc:
        out["secondary_opening_rim_rejected_reasons_v175f"] = (f"boundary_loop_extraction_failed:{exc}",)
        return out

    internal_rail_loops_v175h, internal_rail_diag_v175h = _internal_feature_rail_edge_components_v175h(
        faces=np.asarray(faces, dtype=np.int64),
        vertices=np.asarray(vertices, dtype=float),
        valid_face_ids=ids,
        selected_center=selected_center,
        selected_axis=selected_axis,
        selected_radius=float(selected_radius),
        edge_scale=float(edge_scale),
    )

    loop_items: list[tuple[str, int, tuple[tuple[int, int], ...]]] = []
    for _idx, _loop in enumerate(tuple(loops or ())):
        loop_items.append(("region_boundary_loop", int(_idx), tuple(_loop or ())))
    for _idx, _loop in enumerate(tuple(internal_rail_loops_v175h or ())):
        loop_items.append(("internal_feature_rail_edge_component_v175h", int(_idx), tuple(_loop or ())))

    proposals: list[dict[str, object]] = []
    rejected_loop_count = 0
    for loop_source, loop_index, loop in loop_items:
        loop_edges = tuple(_normalize_edge_key(edge) for edge in tuple(loop or ()))
        loop_edges = tuple(edge for edge in loop_edges if edge[0] >= 0 and edge[1] >= 0)
        if len(loop_edges) < 6:
            rejected_loop_count += 1
            rejected.append("loop_too_few_edges")
            continue
        verts: set[int] = set()
        for a, b in loop_edges:
            verts.add(int(a))
            verts.add(int(b))
        fit = _ring_fit_from_loop_vertices_v175f(vertices=np.asarray(vertices, dtype=float), vertex_ids=verts, axis_hint=selected_axis)
        if not fit:
            rejected_loop_count += 1
            rejected.append("ring_fit_failed")
            continue
        radius = _safe_float(fit.get("radius", 0.0), 0.0)
        if radius <= 1.0e-9:
            rejected_loop_count += 1
            rejected.append("non_positive_radius")
            continue
        axis = _unit_vector_v175f(fit.get("axis", selected_axis), (0.0, 0.0, 1.0))
        center = _coerce_vector3_v175f(fit.get("center", selected_center), (0.0, 0.0, 0.0))
        center_delta = center - selected_center
        axial_delta = float(np.dot(center_delta, selected_axis))
        centerline_vec = center_delta - selected_axis * axial_delta
        centerline_distance = float(np.linalg.norm(centerline_vec))
        radius_delta = float(radius - selected_radius)
        radius_delta_rel = float(abs(radius_delta) / max(abs(selected_radius), 1.0e-12)) if selected_radius > 1.0e-12 else 0.0
        axis_abs_dot = float(abs(np.dot(axis, selected_axis)))
        angular_coverage = _safe_float(fit.get("angular_coverage", 0.0), 0.0)
        circularity = _safe_float(fit.get("circularity", 0.0), 0.0)
        same_selected_radius = bool(
            selected_radius > 1.0e-9
            and radius_delta_rel <= 0.04
            and centerline_distance <= max(3.0 * float(edge_scale), 0.75)
            and abs(axial_delta) <= max(3.0 * float(edge_scale), 0.75)
        )
        if same_selected_radius:
            relation = "primary_selected_rail_or_same_radius_boundary"
            is_secondary = False
        elif selected_radius > 1.0e-9 and radius < selected_radius:
            relation = "inner_related_opening_rim_proposal"
            is_secondary = True
        elif selected_radius > 1.0e-9 and radius > selected_radius:
            relation = "outer_related_opening_rim_proposal"
            is_secondary = True
        else:
            relation = "unclassified_related_annular_boundary_proposal"
            is_secondary = True
        quality = float(max(0.0, min(1.0, 0.45 * circularity + 0.35 * angular_coverage + 0.20 * max(0.0, min(1.0, axis_abs_dot)))))
        if angular_coverage < 0.45 or circularity < 0.50:
            rejected_loop_count += 1
            rejected.append("poor_annular_quality")
            continue
        proposals.append({
            "proposal_index": int(len(proposals)),
            "source_loop_index": int(loop_index),
            "source": str(loop_source),
            "semantic_object": "OpeningRimEvidenceProposal",
            "authority": "heuristic_evidence_proposal_only",
            "not_feature_identity": True,
            "not_surface_ownership": True,
            "not_candidate_data": True,
            "edge_count": int(len(loop_edges)),
            "vertex_count": int(len(verts)),
            "radius": float(radius),
            "diameter": float(2.0 * radius),
            "center": tuple(float(v) for v in center[:3]),
            "axis": tuple(float(v) for v in axis[:3]),
            "axis_abs_dot_to_selected": float(axis_abs_dot),
            "centerline_distance_to_selected": float(centerline_distance),
            "axial_delta_to_selected": float(axial_delta),
            "radius_delta_to_selected": float(radius_delta),
            "radius_delta_rel_to_selected": float(radius_delta_rel),
            "angular_coverage": float(angular_coverage),
            "max_angular_gap_degrees": _safe_float(fit.get("max_angular_gap_degrees", 360.0), 360.0),
            "circularity": float(circularity),
            "radius_rel_rms": _safe_float(fit.get("radius_rel_rms", 0.0), 0.0),
            "plane_rel_rms": _safe_float(fit.get("plane_rel_rms", 0.0), 0.0),
            "relation_to_selected_rail": relation,
            "is_secondary_to_selected_rail": bool(is_secondary),
            "accepted_chamfer_context_present": bool(accepted_chamfer_present),
            "score": float(quality + (0.20 if bool(is_secondary) else 0.0) + (0.10 if bool(accepted_chamfer_present) and bool(is_secondary) else 0.0) + (0.18 if str(loop_source) == "internal_feature_rail_edge_component_v175h" and bool(is_secondary) else 0.0)),
            "support_edge_source_v175h": str(loop_source),
            "support_edge_keys_sample": tuple(loop_edges[:24]),
            "support_vertex_ids_sample": tuple(sorted(verts)[:32]),
        })

    proposals = sorted(proposals, key=lambda row: (bool(row.get("is_secondary_to_selected_rail", False)), float(row.get("score", 0.0))), reverse=True)
    proposals = proposals[: max(1, int(max_proposals))]
    secondary_count = int(sum(1 for row in proposals if bool(row.get("is_secondary_to_selected_rail", False))))
    best_conf = float(max([_safe_float(row.get("score", 0.0), 0.0) for row in proposals], default=0.0))
    heuristic_result = make_heuristic_result(
        SECONDARY_OPENING_RIM_SEARCH_HEURISTIC_NAME_V175F,
        semantic_context="neutral RegionData/AOI with selected annular rail evidence",
        output_proposal_type="OpeningRimEvidenceProposal",
        input_count=int(len(tuple(loops or ()))),
        proposal_count=int(len(proposals)),
        accepted_count=int(secondary_count),
        rejected_count=int(rejected_loop_count),
        confidence=best_conf,
        proposal_face_ids=(),
        proposal_edge_ids=(),
        rejection_reasons=tuple(sorted(set(rejected)))[:12],
        diagnostics={
            "semantic_stage": "RegionData/AOI -> SecondaryOpeningRimSearchHeuristic -> OpeningRimEvidenceProposal",
            "accepted_chamfer_context_present": bool(accepted_chamfer_present),
            "selected_radius": float(selected_radius),
            "selected_confidence": float(selected_confidence),
            "boundary_loop_count": int(len(tuple(loops or ()))),
            "internal_feature_rail_loop_count_v175h": int(len(tuple(internal_rail_loops_v175h or ()))),
            "secondary_proposal_count": int(secondary_count),
            "not_feature_identity": True,
            "not_surface_ownership": True,
            "not_candidate_data": True,
        },
    )
    out.update({
        "secondary_opening_rim_search_used_v175f": True,
        **dict(internal_rail_diag_v175h or {}),
        "secondary_opening_rim_internal_feature_rail_candidate_count_v175h": int(len(tuple(internal_rail_loops_v175h or ()))),
        "secondary_opening_rim_boundary_loop_count_v175f": int(len(tuple(loops or ()))),
        "secondary_opening_rim_total_loop_source_count_v175h": int(len(loop_items)),
        "secondary_opening_rim_boundary_edge_count_v175f": int(len(tuple(boundary_edges or ()))),
        "secondary_opening_rim_candidate_count_v175f": int(len(proposals)),
        "secondary_opening_rim_secondary_candidate_count_v175f": int(secondary_count),
        "secondary_opening_rim_candidate_edge_counts_v175f": tuple(int(row.get("edge_count", 0) or 0) for row in proposals),
        "secondary_opening_rim_candidate_radii_v175f": tuple(float(row.get("radius", 0.0) or 0.0) for row in proposals),
        "secondary_opening_rim_candidate_axis_dot_to_selected_v175f": tuple(float(row.get("axis_abs_dot_to_selected", 0.0) or 0.0) for row in proposals),
        "secondary_opening_rim_candidate_centerline_distances_v175f": tuple(float(row.get("centerline_distance_to_selected", 0.0) or 0.0) for row in proposals),
        "secondary_opening_rim_candidate_relations_v175f": tuple(str(row.get("relation_to_selected_rail", "")) for row in proposals),
        "secondary_opening_rim_candidates_v175f": tuple(proposals),
        "secondary_opening_rim_rejected_reasons_v175f": tuple(sorted(set(rejected)))[:12],
        "secondary_opening_rim_heuristic_result_v175f": heuristic_result,
    })
    return out

# v174x public import aliases used by recognition_component_engine.py.
safe_float_v174x = _safe_float
face_patch_boundary_semantic_report_v174x = _face_patch_boundary_semantic_report
selected_annular_rail_role_resolver_v174x = _selected_annular_rail_role_resolver_v173d
face_vertices_for_ids_v174x = _face_vertices_for_ids
face_edges_for_ids_v174x = _face_edges_for_ids
loop_records_vertices_and_edges_v174x = _loop_records_vertices_and_edges
heuristic_results_for_current_bore_state_v174x = _heuristic_results_for_current_bore_state
unit_rows_v174x = _unit_rows
edge_median_length_v174x = _edge_median_length
percentile_v174x = _percentile
local_arrays_v174x = _local_arrays
component_stats_v174x = _component_stats
chamfer_score_v174x = _chamfer_score
bore_score_v174x = _bore_score
damaged_bore_preview_allowed_v174x = _damaged_bore_preview_allowed
safe_int_v174x = _safe_int
damaged_anchor_mode_from_region_v174x = _damaged_anchor_mode_from_region
interval_gap_v174x = _interval_gap
normalize_edge_key_v174x = _normalize_edge_key
