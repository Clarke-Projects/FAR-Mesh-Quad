"""Prepare delete-patch proposals from CandidateData.

The target contract is separate from Recognition and from mesh mutation:

    CandidateData in -> DeletePatchProposal out

Later phases will move the existing topology-seal/protected-fragment logic out
of rebuild.py into this module.  Phase 83 provides the typed adapter and
diagnostic shape without changing rebuild behavior yet.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .types import CandidateData, DeletePatchProposal, EdgeKey, FeatureFamily, RecognitionStage, tuple_edges, tuple_ints


_REBUILD_ALLOWED_FAMILIES = {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value, FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value}


def _candidate_stage_and_family(candidate: CandidateData | Mapping[str, object]) -> tuple[str, str]:
    """Return canonical recognition stage/family strings for target policy."""

    if isinstance(candidate, CandidateData):
        stage = candidate.recognition_stage
        family = candidate.feature_family
        stage_value = str(stage.value if hasattr(stage, "value") else stage)
        family_value = str(family.value if hasattr(family, "value") else family)
        return stage_value, family_value
    stage_value = str(candidate.get("recognition_stage", "") or "").strip().lower()
    family_value = str(candidate.get("feature_family", "") or "").strip().lower()
    # Compatibility for older current candidates that have not yet been tagged.
    kind = str(candidate.get("entity_type", candidate.get("feature_kind", "")) or "").strip().lower()
    if not stage_value and bool(candidate.get("candidate_action_enabled", candidate.get("rebuild_authorized", False))):
        stage_value = RecognitionStage.ACCEPTED_CANDIDATE.value
    if not family_value:
        if kind == "borehole":
            family_value = FeatureFamily.BORE.value
        elif kind == "chamfer":
            family_value = FeatureFamily.CHAMFER_FORM.value
        elif kind in {"pocket", "circular_pocket"}:
            family_value = FeatureFamily.CIRCULAR_POCKET.value
        else:
            family_value = FeatureFamily.UNKNOWN.value
    return stage_value, family_value


def candidate_can_request_delete_patch(candidate: CandidateData | Mapping[str, object]) -> bool:
    """Target-policy gate: only accepted supported families may request deletion."""

    stage, family = _candidate_stage_and_family(candidate)
    return bool(stage == RecognitionStage.ACCEPTED_CANDIDATE.value and family in _REBUILD_ALLOWED_FAMILIES)


def prepare_rebuild_target(
    candidate: CandidateData | Mapping[str, object],
    *,
    delete_patch_face_ids: Iterable[int] | None = None,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    protected_rim_edges: Iterable[object] = (),
    allowed_bridge_face_ids: Iterable[int] = (),
    forbidden_face_ids: Iterable[int] = (),
    diagnostics: Mapping[str, object] | None = None,
) -> DeletePatchProposal:
    """Build a DeletePatchProposal without repairing or mutating geometry."""

    if not candidate_can_request_delete_patch(candidate):
        stage, family = _candidate_stage_and_family(candidate)
        raise ValueError(
            "CandidateData is not allowed to request a DeletePatchProposal. "
            f"recognition_stage={stage!r}; feature_family={family!r}. "
            "Only accepted bore/chamfer/pocket CandidateData may request DeletePatchProposal; rebuild.py still validates topology."
        )

    if isinstance(candidate, CandidateData):
        candidate_id = candidate.feature_id
        kind = candidate.feature_kind
        semantic_face_ids = candidate.semantic_face_ids
        base_diag = candidate.to_dict()
    else:
        candidate_id = str(candidate.get("candidate_id", candidate.get("feature_id", "legacy_candidate")) or "legacy_candidate")
        kind = str(candidate.get("entity_type", candidate.get("feature_kind", "unknown")) or "unknown")
        if kind not in {"borehole", "chamfer", "mouth", "counterbore", "pocket"}:
            kind = "unknown"
        semantic_face_ids = tuple_ints(candidate.get("face_ids", candidate.get("semantic_face_ids", ())))
        base_diag = dict(candidate)

    delete_ids = tuple_ints(delete_patch_face_ids if delete_patch_face_ids is not None else semantic_face_ids)
    return DeletePatchProposal(
        target_id=candidate_id,
        feature_kind=kind,  # type: ignore[arg-type]
        semantic_face_ids=tuple_ints(semantic_face_ids),
        delete_patch_face_ids=delete_ids,
        protected_loop_pair=protected_loop_pair,
        protected_rim_edges=tuple_edges(protected_rim_edges),
        allowed_bridge_face_ids=tuple_ints(allowed_bridge_face_ids),
        forbidden_face_ids=tuple_ints(forbidden_face_ids),
        allow_unequal_loop_transition=True,
        diagnostics={
            "source": "candidate_to_delete_patch_proposal_adapter_phase83",
            "candidate_data": base_diag,
            "recognition_stage_target_gate": _candidate_stage_and_family(candidate)[0],
            "feature_family_target_gate": _candidate_stage_and_family(candidate)[1],
            "x1_family_target_policy": "accepted_candidate bore/chamfer_form/pocket/circular_pocket can build DeletePatchProposal; pocket uses owned pocket floor plus side-wall faces for recess-cup target in v99",
            **dict(diagnostics or {}),
        },
    )


def target_from_candidate_dict(candidate: Mapping[str, object]) -> DeletePatchProposal:
    """Compatibility helper: CandidateData dictionary -> DeletePatchProposal."""

    return prepare_rebuild_target(candidate)


# -----------------------------------------------------------------------------
# Bounded rebuild-target face-set construction
# -----------------------------------------------------------------------------

import numpy as np

from .topology import connected_face_components, face_edges


def loop_vertices_to_edge_keys(loop_vertices: Iterable[object]) -> tuple[EdgeKey, ...]:
    """Return normalized edges around an ordered loop-vertex sequence."""

    verts = tuple(int(v) for v in tuple(loop_vertices or ()) if int(v) >= 0)
    if len(verts) < 2:
        return ()
    edges: set[EdgeKey] = set()
    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        if int(a) == int(b):
            continue
        edges.add((int(a), int(b)) if int(a) < int(b) else (int(b), int(a)))
    return tuple(sorted(edges))


def build_bounded_rebuild_target_face_sets(
    *,
    source_faces: np.ndarray,
    initial_face_ids: Iterable[int],
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    extra_candidate_face_sets: Iterable[tuple[str, Iterable[int]]] = (),
    preview_candidate_patch_owns_delete: bool = False,
    topology_seal_callback: object | None = None,
    protected_fragment_bridge_callback: object | None = None,
) -> dict[str, object]:
    """Construct bounded delete-patch candidates for measured Bore rebuild.

    This module owns rebuild-target *policy*: candidate ownership, tiny island
    cleanup, topology-sealed two-rim targets, and protected-rim fragment bridges.
    It does not mutate the mesh, classify features, fit cylinders, or accept a
    final rebuild.  ``rebuild.py`` still performs loop-plan construction and
    watertight trial validation.

    The returned ``face_sets`` are ordered by semantic rebuild-target quality:

    1. topology-sealed two-rim targets of the owned candidate;
    2. explicit protected-fragment bridge targets;
    3. exact candidate patch;
    4. compatible component/fallback targets.

    Broad feature absorption is rejected here, before rebuild attempts are made.
    """

    faces = np.asarray(source_faces, dtype=np.int64)
    initial = tuple(sorted({int(fid) for fid in tuple(initial_face_ids or ()) if 0 <= int(fid) < len(faces)}))
    if not initial:
        return {
            "valid": False,
            "face_sets": (),
            "rejected_candidate_owned_face_sets": (),
            "diagnostics": {"reason": "empty_initial_face_set", "source": "rebuild_target"},
        }

    face_sets: list[tuple[str, tuple[int, ...]]] = [("initial_final_delete_faces", initial)]
    rejected_face_sets: list[dict[str, object]] = []

    protected_loop_edges: set[EdgeKey] = set()
    if protected_loop_pair is not None:
        for protected_loop in tuple(protected_loop_pair or ()):  # vertex loops
            protected_loop_edges.update(loop_vertices_to_edge_keys(tuple(int(v) for v in tuple(protected_loop or ()))))

    initial_set = set(initial)

    def _face_ids_touch_protected_rim(face_ids_to_check: set[int]) -> bool:
        if not protected_loop_edges or not face_ids_to_check:
            return False
        for fid in tuple(face_ids_to_check):
            if int(fid) < 0 or int(fid) >= len(faces):
                continue
            try:
                tri_edges = face_edges(tuple(int(v) for v in np.asarray(faces[int(fid)]).reshape(-1).tolist()))
            except Exception:
                continue
            if any(edge in protected_loop_edges for edge in tri_edges):
                return True
        return False

    def _is_candidate_owned_compatible(source: str, ids: tuple[int, ...]) -> tuple[bool, str]:
        if not bool(preview_candidate_patch_owns_delete):
            return True, "non_preview_wall_rebuild"
        ids_set = {int(v) for v in tuple(ids or ())}
        if ids_set == initial_set:
            return True, "initial_candidate_patch"
        if not initial_set:
            return False, "empty_initial_candidate_patch"

        added_set = ids_set - initial_set
        removed_set = initial_set - ids_set
        added = len(added_set)
        removed = len(removed_set)

        # Cap-free rebuild target policy:
        # Face-count budgets are not a rebuild criterion.  Candidate-owned
        # variants are allowed or rejected by topology/ownership constraints and
        # by the final watertight trial in rebuild.py.  The only hard rejection
        # kept here is dropping protected rim faces, because that destroys the
        # measured boundary contract before rebuild can test anything.
        if removed > 0 and _face_ids_touch_protected_rim(removed_set):
            return False, f"candidate_owned_patch_would_drop_{removed}_protected_rim_faces"

        if added > 0 and removed > 0:
            return True, f"candidate_owned_topology_variant_added_{added}_removed_{removed}_trial_validated_no_face_count_cap"
        if added > 0:
            return True, f"candidate_owned_topology_seal_added_{added}_trial_validated_no_face_count_cap"
        if removed > 0:
            return True, f"candidate_owned_topology_variant_removed_{removed}_trial_validated_no_face_count_cap"
        return True, "candidate_owned_equivalent_patch"

    normalized_extras: list[tuple[str, tuple[int, ...]]] = []
    candidate_plus_region_data_context_proposed = False
    candidate_plus_region_data_context_face_count = 0
    for extra_source, extra_ids_raw in tuple(extra_candidate_face_sets or ()):  # caller-provided local pools
        extra_source_text = str(extra_source)
        extra_ids = tuple(sorted({int(fid) for fid in tuple(extra_ids_raw or ()) if 0 <= int(fid) < len(faces)}))
        if not extra_ids:
            continue
        normalized_extras.append((extra_source_text, extra_ids))

        # v33 semantic boundary: RegionData is neutral AOI/context evidence and
        # cannot be unioned into CandidateData to create a delete patch.
        if extra_source_text == "region_data_face_pool":
            rejected_face_sets.append({
                "source": "candidate_core_plus_region_data_context",
                "face_count": int(len(initial_set | set(extra_ids))),
                "reason": "disabled_by_v33_regiondata_is_not_candidate_ownership",
            })
            continue

        if set(extra_ids) == initial_set:
            continue
        ok, reason = _is_candidate_owned_compatible(extra_source_text, extra_ids)
        if ok:
            face_sets.append((extra_source_text, extra_ids))
        else:
            rejected_face_sets.append({"source": extra_source_text, "face_count": int(len(extra_ids)), "reason": reason})

    components = connected_face_components(faces, initial)
    for comp_index, component in enumerate(components):
        if tuple(component) == initial:
            continue
        ok, reason = _is_candidate_owned_compatible(f"measured_connected_component_{comp_index}", tuple(component))
        if ok:
            face_sets.append((f"measured_connected_component_{comp_index}", tuple(component)))
        else:
            rejected_face_sets.append({"source": f"measured_connected_component_{comp_index}", "face_count": int(len(component)), "reason": reason})

    if callable(protected_fragment_bridge_callback):
        protected_bridge = protected_fragment_bridge_callback(
            source_faces=faces,
            initial_face_ids=initial,
            components=components,
            protected_loop_edges=protected_loop_edges,
            extra_candidate_face_sets=tuple(normalized_extras),
        )
        if protected_bridge is not None:
            bridge_ids = tuple(int(fid) for fid in tuple(protected_bridge.get("face_ids", ()) or ()))
            if not bool(protected_bridge.get("valid", False)):
                rejected_face_sets.append({
                    "source": "candidate_owned_protected_fragment_bridge",
                    "face_count": int(len(bridge_ids)),
                    "reason": str(protected_bridge.get("reason", "protected_fragment_bridge_unavailable")),
                    "bridge_diagnostics": dict(protected_bridge),
                })
            else:
                ok, reason = _is_candidate_owned_compatible("candidate_owned_protected_fragment_bridge", bridge_ids)
                if ok:
                    face_sets.append(("candidate_owned_protected_fragment_bridge", bridge_ids))
                else:
                    rejected_face_sets.append({
                        "source": "candidate_owned_protected_fragment_bridge",
                        "face_count": int(len(bridge_ids)),
                        "reason": reason,
                        "bridge_diagnostics": dict(protected_bridge),
                    })

    sealed_sets: list[tuple[str, tuple[int, ...]]] = []
    if callable(topology_seal_callback):
        for base_source, base_ids in tuple(face_sets):
            # No arbitrary seal budget.  The callback performs deterministic
            # leak-edge expansion until the patch boundary is exactly the
            # protected two-rim boundary, or until no topology progress exists.
            # rebuild.py still accepts only a fully watertight trial mesh.
            sealed = topology_seal_callback(
                source_faces=faces,
                initial_face_ids=base_ids,
                protected_loop_pair=protected_loop_pair,
                max_added_faces=0,
            )
            if sealed and set(sealed.get("face_ids", ())) != set(base_ids):
                sealed_ids = tuple(int(v) for v in tuple(sealed.get("face_ids", ()) or ()))
                ok, reason = _is_candidate_owned_compatible(f"{base_source}_topology_sealed_two_rim_patch", sealed_ids)
                if ok:
                    sealed_sets.append((f"{base_source}_topology_sealed_two_rim_patch", sealed_ids))
                else:
                    rejected_face_sets.append({
                        "source": f"{base_source}_topology_sealed_two_rim_patch",
                        "face_count": int(len(sealed_ids)),
                        "reason": reason,
                    })

    def _target_priority(source: str) -> int:
        source_text = str(source or "")
        if source_text.endswith("_topology_sealed_two_rim_patch"):
            return 0
        if source_text == "candidate_owned_protected_fragment_bridge":
            return 1
        if source_text == "initial_final_delete_faces":
            return 2
        if source_text.startswith("measured_connected_component_"):
            return 3
        return 4

    ordered_sets: list[tuple[str, tuple[int, ...]]] = []
    ordered_sets.extend(sealed_sets)
    ordered_sets.append(("initial_final_delete_faces", initial))
    for source, ids in face_sets:
        if source == "initial_final_delete_faces":
            continue
        ordered_sets.append((source, ids))

    deduped: list[tuple[str, tuple[int, ...]]] = []
    seen_sets: set[frozenset[int]] = set()
    for source, ids in sorted(
        ordered_sets,
        key=lambda item: (_target_priority(str(item[0])), int(len(tuple(item[1] or ()))), str(item[0])),
    ):
        key = frozenset(int(v) for v in tuple(ids or ()))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        deduped.append((str(source), tuple(int(v) for v in tuple(ids or ()))))

    return {
        "valid": True,
        "face_sets": tuple(deduped),
        "rejected_candidate_owned_face_sets": tuple(rejected_face_sets),
        "diagnostics": {
            "v33_semantic_boundary_hardening_used": True,
            "region_data_union_enabled": False,
            "source": "rebuild_target.build_bounded_rebuild_target_face_sets",
            "initial_face_count": int(len(initial)),
            "candidate_owned": bool(preview_candidate_patch_owns_delete),
            "protected_rim_edge_count": int(len(protected_loop_edges)),
            "component_count": int(len(components)),
            "component_face_counts": tuple(int(len(item)) for item in components),
            "face_set_count": int(len(deduped)),
            "face_set_sources": tuple(str(source) for source, _ in deduped),
            "candidate_plus_region_data_context_proposed": bool(candidate_plus_region_data_context_proposed),
            "candidate_plus_region_data_context_face_count": int(candidate_plus_region_data_context_face_count),
            "parameter_fit_used_for_target": False,
            "radius_used_for_delete_expansion": False,
            "axis_used_for_delete_expansion": False,
        },
    }



def build_rebuild_target_contract_for_feature(
    candidate: CandidateData | Mapping[str, object],
    *,
    source_faces: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    extra_candidate_face_sets: Iterable[tuple[str, Iterable[int]]] = (),
    preview_candidate_patch_owns_delete: bool = True,
    topology_seal_callback: object | None = None,
    protected_fragment_bridge_callback: object | None = None,
) -> dict[str, object]:
    """Return a serializable delete-patch proposal contract for CandidateData.

    Recognition is allowed to emit CandidateData; it is not allowed to hand
    raw, overselected RegionData faces directly to rebuild.  This helper turns the
    candidate data into the exact delete-patch proposal consumed by UI and
    rebuild paths:

    * semantic faces stay visible as candidate evidence;
    * delete-patch faces are chosen by bounded target policy;
    * unresolved overselected RegionData evidence stays inspection-only until a bounded
      delete target exists.

    No fitted axis/radius is used here.  Optional callbacks may provide topology
    sealing/fragment bridging, but acceptance is still only a target proposal;
    rebuild.py remains responsible for measured-loop quad planning and watertight
    trial validation.
    """

    candidate_dict = candidate.to_dict() if isinstance(candidate, CandidateData) else dict(candidate or {})
    semantic = tuple_ints(candidate_dict.get("semantic_face_ids", candidate_dict.get("face_ids", ())))
    target_sets = build_bounded_rebuild_target_face_sets(
        source_faces=np.asarray(source_faces, dtype=np.int64),
        initial_face_ids=semantic,
        protected_loop_pair=protected_loop_pair,
        extra_candidate_face_sets=tuple(extra_candidate_face_sets or ()),
        preview_candidate_patch_owns_delete=bool(preview_candidate_patch_owns_delete),
        topology_seal_callback=topology_seal_callback,
        protected_fragment_bridge_callback=protected_fragment_bridge_callback,
    )

    face_sets = tuple(target_sets.get("face_sets", ()) or ()) if bool(target_sets.get("valid", False)) else ()
    selected_source = ""
    delete_faces: tuple[int, ...] = ()
    if face_sets:
        selected_source = str(face_sets[0][0])
        delete_faces = tuple_ints(face_sets[0][1])
    if not delete_faces:
        delete_faces = semantic
        selected_source = "semantic_faces_no_bounded_target"

    diagnostics = dict(candidate_dict.get("diagnostics", {}) or {})
    status = str(candidate_dict.get("status", "") or "")
    overselected = bool(
        diagnostics.get("overselected_envelope", False)
        or "rebuild_target_unresolved_overselected_region_data_envelope" in status
    )
    bounded_target_resolved = bool(delete_faces) and not (
        overselected and set(delete_faces) == set(semantic) and selected_source in {"initial_final_delete_faces", "semantic_faces_no_bounded_target"}
    )

    target = prepare_rebuild_target(
        candidate_dict,
        delete_patch_face_ids=delete_faces,
        protected_loop_pair=protected_loop_pair,
        protected_rim_edges=candidate_dict.get("protected_rim_edges", ()),
        diagnostics={
            "source": "delete_patch_proposal_contract_for_candidate_data",
            "selected_face_set_source": selected_source,
            "bounded_target_resolved": bool(bounded_target_resolved),
            "overselected_envelope": bool(overselected),
            "target_sets": dict(target_sets),
        },
    ).to_dict()
    target["target_ready"] = bool(bounded_target_resolved)
    target["selected_face_set_source"] = selected_source
    target["rejected_candidate_owned_face_sets"] = tuple(target_sets.get("rejected_candidate_owned_face_sets", ()) or ())
    target["rebuild_target_diagnostics"] = dict(target_sets.get("diagnostics", {}) or {})
    return target


__all__ = ["candidate_can_request_delete_patch", "prepare_rebuild_target", "target_from_candidate_dict", "build_bounded_rebuild_target_face_sets"]
