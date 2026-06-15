"""Surface-component Bore feature recognizer.

Active feature ownership engine for FAR MESH Bore cleanup.

Phase 8.3a cleanup: shared adjacency/component/topology helpers come from topology.py.

X1 family update v2 adds mesh-native diagnostic loop-shape descriptors and
conservative tessellated-family preview rows.  These rows are never rebuild
authority; they only make X1-like evidence families visible to the display and
to later recognition development.

This module consumes RegionData and emits CandidateData. It deliberately
does not classify assemblies such as "single-chamfered bore" or
"double-chamfered bore". It emits independent candidate objects and relationship
metadata:

    BOREHOLE = connected cylindrical core surface component
    CHAMFER  = connected annular transition surface component
    UNCLASSIFIED = diagnostic-only RegionData remainder
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .geometry import BoundaryLoopGeometry, canonical_axis, to_vector3
from .topology import (
    boundary_edges_for_face_patch,
    connected_face_components,
    edge_loop_components,
    face_adjacency_for_patch,
    summarize_patch_topology,
)
from .types import (
    EvidenceKind,
    FeatureEvidenceItem,
    FeatureEvidenceLedger,
    FeatureFamily,
    FeaturePrimitiveData,
    FeaturePrimitiveKind,
    FeatureRelationshipData,
    FeatureRelationshipKind,
    RecognitionStage,
    X1_FREECAD_TO_FAR_MESH_DICTIONARY,
    tuple_ints,
)

EdgeKey = tuple[int, int]


def _x1_stage_value(stage: RecognitionStage | str) -> str:
    """Return the public recognition-stage string used in CandidateData rows."""

    return str(stage.value if isinstance(stage, RecognitionStage) else stage)


def _x1_family_value(family: FeatureFamily | str) -> str:
    """Return the public feature-family string used in CandidateData rows."""

    return str(family.value if isinstance(family, FeatureFamily) else family)


def _x1_evidence_values(*items: EvidenceKind | str) -> tuple[str, ...]:
    """Return stable evidence-kind strings."""

    return tuple(str(item.value if isinstance(item, EvidenceKind) else item) for item in items)


def _x1_primitive_kind_for_family(family: FeatureFamily | str) -> FeaturePrimitiveKind:
    """Map X1 feature families to non-mutating FAR MESH primitive descriptors."""

    family_value = _x1_family_value(family)
    if family_value == FeatureFamily.BORE.value:
        return FeaturePrimitiveKind.CYLINDER_AXIS
    if family_value == FeatureFamily.CHAMFER_FORM.value:
        return FeaturePrimitiveKind.ANNULAR_CHAMFER_BAND
    if family_value in {FeatureFamily.COUNTERBORE.value, FeatureFamily.STEPPED_BORE_STACK.value}:
        return FeaturePrimitiveKind.RADIUS_STACK
    if family_value == FeatureFamily.CIRCULAR_POCKET.value:
        return FeaturePrimitiveKind.CIRCULAR_OPENING
    if family_value in {
        FeatureFamily.HEX_NUT_POCKET.value,
        FeatureFamily.SLOT_OR_ADJUSTABLE_BORE.value,
        FeatureFamily.ELLIPTIC_OR_OVAL_BORE.value,
    }:
        return FeaturePrimitiveKind.NONROUND_LOOP_PROFILE
    if family_value in {FeatureFamily.TESSELLATED_BORE_CANDIDATE.value, FeatureFamily.TESSELLATED_CHAMFER_BODY.value}:
        return FeaturePrimitiveKind.TESSELLATED_SIDE_PAIR
    return FeaturePrimitiveKind.UNKNOWN


def _x1_feature_primitives_for_candidate(
    *,
    family: FeatureFamily | str,
    stage: RecognitionStage | str,
    candidate_id: str,
    face_ids: Iterable[int],
    primitive_axis: object | None,
    primitive_radius: object | None,
    primitive_depth: object | None,
    confidence: float,
    recognition_rule: str,
    diagnostics: Mapping[str, object] | None = None,
) -> tuple[FeaturePrimitiveData, ...]:
    """Build the mesh-native primitive layer for a CandidateData row.

    This is the v4 FreeCAD-to-FAR-MESH bridge: X1 could emit exact FreeCAD
    cylinders, wires, profiles and cutter solids; BoreTool only emits typed,
    non-mutating primitive descriptors that later algorithms may consume.
    """

    face_id_tuple = tuple_ints(face_ids)
    family_value = _x1_family_value(family)
    stage_value = _x1_stage_value(stage)
    axis_value = to_vector3(primitive_axis) if primitive_axis is not None else None
    radius_value: float | None = None
    depth_value: float | None = None
    if primitive_radius is not None:
        try:
            radius_value = float(primitive_radius)
        except Exception:
            radius_value = None
    if primitive_depth is not None:
        try:
            depth_value = float(primitive_depth)
        except Exception:
            depth_value = None
    primitive_kind = _x1_primitive_kind_for_family(family)
    return (
        FeaturePrimitiveData(
            primitive_kind=primitive_kind,
            source="recognition_component_engine.x1_family_v21",
            role="accepted_physical_primitive" if stage_value == RecognitionStage.ACCEPTED_CANDIDATE.value else "diagnostic_primitive_descriptor",
            axis=axis_value,
            radius=radius_value,
            diameter=(2.0 * radius_value) if radius_value is not None else None,
            depth=depth_value,
            confidence=float(confidence),
            face_ids=face_id_tuple,
            diagnostics={
                "candidate_id": str(candidate_id or "candidate"),
                "feature_family": family_value,
                "recognition_stage": stage_value,
                "recognition_rule": str(recognition_rule),
                "x1_freecad_bridge": "mesh_native_primitive_descriptor_not_cad_body",
                "freecad_to_far_mesh_dictionary": X1_FREECAD_TO_FAR_MESH_DICTIONARY,
                **dict(diagnostics or {}),
            },
        ),
    )


def _x1_candidate_contract_fields(
    *,
    family: FeatureFamily | str,
    stage: RecognitionStage | str,
    evidence_kinds: tuple[EvidenceKind | str, ...] = (),
    accepted: bool = False,
    promotion_reasons: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
    primitive_axis: object | None = None,
    primitive_radius: object | None = None,
    primitive_depth: object | None = None,
    candidate_id: str = "",
    face_ids: Iterable[int] = (),
    recognition_rule: str = "",
    status: str = "",
    confidence: float = 0.0,
    diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Common X1-style CandidateData fields.

    These fields are metadata only.  Rebuild permission remains a separate
    target-policy decision and requires ``accepted=True`` plus explicit candidate
    action flags on the physical feature candidate.  v3 adds a per-candidate
    FeatureEvidenceLedger so GUI/rebuild/debug code can inspect *why* a family
    was emitted without re-parsing Recognition internals.
    """

    family_value = _x1_family_value(family)
    stage_value = _x1_stage_value(stage)
    evidence_values = _x1_evidence_values(*evidence_kinds)
    target_policy_allowed = bool(
        accepted
        and stage_value == RecognitionStage.ACCEPTED_CANDIDATE.value
        and family_value in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value}
    )
    if target_policy_allowed:
        target_policy_reason = "accepted supported family may request DeletePatchProposal; rebuild.py still validates topology"
    elif stage_value != RecognitionStage.ACCEPTED_CANDIDATE.value:
        target_policy_reason = "recognition stage is not accepted_candidate"
    elif family_value not in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value}:
        target_policy_reason = "feature family has no dedicated mesh-native rebuild support yet"
    else:
        target_policy_reason = "candidate is not marked accepted by recognition contract"

    primitive_axis_value = to_vector3(primitive_axis) if primitive_axis is not None else None
    primitive_radius_value: float | None = None
    primitive_depth_value: float | None = None
    if primitive_radius is not None:
        try:
            primitive_radius_value = float(primitive_radius)
        except Exception:
            primitive_radius_value = None
    if primitive_depth is not None:
        try:
            primitive_depth_value = float(primitive_depth)
        except Exception:
            primitive_depth_value = None

    face_id_tuple = tuple_ints(face_ids)
    evidence_items = tuple(
        FeatureEvidenceItem(
            evidence_kind=kind,
            role="supporting" if stage_value != RecognitionStage.DIAGNOSTIC_ONLY.value else "diagnostic",
            source="recognition_component_engine.x1_family_v21",
            confidence=float(confidence),
            description=f"{_x1_family_value(family)} candidate uses {str(kind.value if isinstance(kind, EvidenceKind) else kind)} evidence",
            face_ids=face_id_tuple,
            diagnostics={
                "recognition_rule": str(recognition_rule),
                "recognition_stage": stage_value,
                "status": str(status),
            },
        )
        for kind in tuple(evidence_kinds or ())
    )

    feature_primitives = _x1_feature_primitives_for_candidate(
        family=family,
        stage=stage,
        candidate_id=str(candidate_id or "candidate"),
        face_ids=face_id_tuple,
        primitive_axis=primitive_axis,
        primitive_radius=primitive_radius,
        primitive_depth=primitive_depth,
        confidence=float(confidence),
        recognition_rule=str(recognition_rule),
        diagnostics=diagnostics,
    )

    ledger = FeatureEvidenceLedger(
        candidate_id=str(candidate_id or "candidate"),
        feature_family=family_value,
        recognition_stage=stage_value,
        evidence_items=evidence_items,
        evidence_kinds=evidence_kinds,
        feature_primitives=feature_primitives,
        promotion_reasons=tuple(str(v) for v in promotion_reasons),
        rejection_reasons=tuple(str(v) for v in rejection_reasons),
        target_policy_allowed=bool(target_policy_allowed),
        target_policy_reason=str(target_policy_reason),
        primitive_axis=primitive_axis_value,
        primitive_radius=primitive_radius_value,
        primitive_depth=primitive_depth_value,
        diagnostics={
            "recognition_rule": str(recognition_rule),
            "status": str(status),
            "confidence": float(confidence),
            "face_count": int(len(face_id_tuple)),
            "x1_family_update_version": "v21_explicit_chamfer_slope_profile",
            **dict(diagnostics or {}),
        },
    ).to_dict()

    out: dict[str, object] = {
        "feature_family": family_value,
        "recognition_stage": stage_value,
        "evidence_kinds": evidence_values,
        "promotion_reasons": tuple(str(v) for v in promotion_reasons),
        "rejection_reasons": tuple(str(v) for v in rejection_reasons),
        "x1_feature_family_contract": "mesh_native_feature_family_v21",
        "x1_promotion_policy": (
            "diagnostic evidence != accepted feature; review/promotion preview != rebuild permission; "
            "only accepted_candidate may enter rebuild_target policy"
        ),
        "x1_evidence_ledger": ledger,
        "feature_primitives": tuple(item.to_dict() for item in feature_primitives),
        "feature_primitive_count": int(len(feature_primitives)),
        "x1_freecad_to_far_mesh_dictionary": X1_FREECAD_TO_FAR_MESH_DICTIONARY,
        "x1_primitive_bridge_contract": "FreeCAD/OCCT feature representatives are translated into non-mutating mesh-native primitive descriptors",
        "delete_patch_request_allowed": bool(target_policy_allowed),
        "rebuild_target_policy_allowed": bool(target_policy_allowed),
        "rebuild_target_policy_reason": str(target_policy_reason),
    }
    if primitive_axis_value is not None:
        out["primitive_axis"] = primitive_axis_value
    if primitive_radius_value is not None:
        out["primitive_radius"] = float(primitive_radius_value)
    if primitive_depth_value is not None:
        out["primitive_depth"] = float(primitive_depth_value)
    out["candidate_action_enabled_by_stage"] = bool(target_policy_allowed)
    return out


def _x1_rebuild_allowed(candidate: Mapping[str, object]) -> bool:
    """Return whether CandidateData may ask rebuild_target for a delete patch."""

    stage = str(candidate.get("recognition_stage", "") or "").strip().lower()
    family = str(candidate.get("feature_family", "") or "").strip().lower()
    return bool(stage == RecognitionStage.ACCEPTED_CANDIDATE.value and family in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value})


def _diagnostic_candidate(
    *,
    candidate_id: str,
    family: FeatureFamily,
    display_name: str,
    face_ids: Iterable[int],
    evidence_kinds: tuple[EvidenceKind | str, ...],
    status: str,
    confidence: float,
    recognition_rule: str,
    primitive_axis: object | None = None,
    primitive_radius: object | None = None,
    primitive_depth: object | None = None,
    diagnostics: Mapping[str, object] | None = None,
    stage: RecognitionStage = RecognitionStage.REVIEW,
) -> dict[str, object]:
    """Create a non-rebuildable X1-family diagnostic CandidateData row."""

    ids = tuple_ints(face_ids)
    return {
        "candidate_id": str(candidate_id),
        "feature_id": str(candidate_id),
        "entity_type": str(family.value),
        "feature_kind": "unknown",
        "candidate_scope": "recognition_feature_family_diagnostic",
        "display_name": str(display_name),
        "role": "diagnostic_review_only",
        "status": str(status),
        "promotion_state": "diagnostic_only",
        "candidate_action_enabled": False,
        "candidate_action": "preview_only",
        "rebuild_authorized": False,
        "rebuild_gate": "x1_family_diagnostic_not_rebuild_candidate",
        "face_ids": ids,
        "semantic_face_ids": ids,
        "preview_face_ids": ids,
        "display_face_ids": ids,
        "rebuild_face_ids": (),
        "face_count": int(len(ids)),
        "confidence": float(confidence),
        "recognition_rule": str(recognition_rule),
        "feature_ownership_source": "surface_component_classifier_v7.x1_family_diagnostics",
        "feature_ownership_split": "diagnostic_family_only_no_delete_patch_ownership",
        "rebuild_disabled_reason": "feature family is diagnostic/review only until a dedicated rebuild path exists",
        **_x1_candidate_contract_fields(
            family=family,
            stage=stage,
            evidence_kinds=evidence_kinds,
            accepted=False,
            rejection_reasons=("new_family_rebuild_support_not_implemented",),
            primitive_axis=primitive_axis,
            primitive_radius=primitive_radius,
            primitive_depth=primitive_depth,
            candidate_id=str(candidate_id),
            face_ids=ids,
            recognition_rule=str(recognition_rule),
            status=str(status),
            confidence=float(confidence),
            diagnostics=diagnostics,
        ),
        "diagnostics": dict(diagnostics or {}),
    }



def _orthonormal_basis_from_axis(axis: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic in-plane basis vectors for a loop axis."""

    a = canonical_axis(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(a, ref))) > 0.88:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(a, ref)
    u_len = float(np.linalg.norm(u))
    if not np.isfinite(u_len) or u_len <= 1.0e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u = u / u_len
    v = np.cross(a, u)
    v_len = float(np.linalg.norm(v))
    if not np.isfinite(v_len) or v_len <= 1.0e-12:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v = v / v_len
    return a, u, v


def _boundary_loop_shape_descriptors(
    *,
    boundary_loops: tuple[tuple[EdgeKey, ...], ...],
    boundary_loop_geometry: tuple[BoundaryLoopGeometry, ...],
    vertices: object | None,
    axis: object,
) -> tuple[dict[str, object], ...]:
    """Return X1-style mesh-native loop-shape descriptors.

    This deliberately stays diagnostic.  It translates FreeCAD/OCCT notions like
    circular wires, non-round wires, polygonal mouths and slot-like openings into
    measurable mesh-loop descriptors without using CAD kernel entities.
    """

    if vertices is None:
        return ()
    try:
        verts = np.asarray(vertices, dtype=float)[:, :3]
    except Exception:
        return ()
    if verts.ndim != 2 or verts.shape[1] < 3 or len(verts) == 0:
        return ()

    _axis, u, v = _orthonormal_basis_from_axis(axis)
    out: list[dict[str, object]] = []
    loops = tuple(boundary_loops or ())
    geo_by_index = {int(getattr(item, "index", idx)): item for idx, item in enumerate(boundary_loop_geometry or ())}

    for idx, loop in enumerate(loops):
        edge_keys = tuple(loop or ())
        vertex_ids = tuple(sorted({int(vtx) for edge in edge_keys for vtx in tuple(edge)[:2] if 0 <= int(vtx) < len(verts)}))
        if len(vertex_ids) < 3:
            continue
        pts = verts[np.asarray(vertex_ids, dtype=np.int64), :3]
        center = pts.mean(axis=0)
        rel = pts - center.reshape(1, 3)
        xy = np.column_stack((rel @ u.reshape(3), rel @ v.reshape(3)))
        radii = np.linalg.norm(xy, axis=1)
        finite = np.isfinite(radii)
        if not np.any(finite):
            continue
        xy = xy[finite]
        radii = radii[finite]
        if len(radii) < 3:
            continue
        extent = xy.max(axis=0) - xy.min(axis=0)
        # PCA extent gives a rotation-invariant slot/ellipse hint.
        try:
            cov = np.cov(xy.T)
            eig = np.linalg.eigvalsh(cov)
            major = float(np.sqrt(max(float(eig[-1]), 0.0)))
            minor = float(np.sqrt(max(float(eig[0]), 0.0)))
        except Exception:
            major = float(max(extent))
            minor = float(min(extent))
        aspect_ratio = float(major / max(minor, 1.0e-9)) if major > 0.0 else 1.0
        radius_mean = float(np.mean(radii))
        radius_mad = float(np.median(np.abs(radii - float(np.median(radii))))) if len(radii) else 0.0
        radius_rel_mad = float(radius_mad / max(radius_mean, 1.0e-9))
        # Approximate polygon/corner character by sorting projected points around
        # the center and measuring angular irregularity.  This is diagnostic only;
        # it must not be used as final hex acceptance.
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        order = np.argsort(angles)
        sorted_angles = angles[order]
        angle_steps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2.0 * np.pi]]))
        angle_cv = float(np.std(angle_steps) / max(float(np.mean(angle_steps)), 1.0e-9)) if len(angle_steps) else 0.0
        geo = geo_by_index.get(idx)
        edge_count = int(len(edge_keys) or (getattr(geo, "edge_count", 0) if geo is not None else 0))
        roundness_rel_mad = float(getattr(geo, "radius_rel_mad", radius_rel_mad) if geo is not None else radius_rel_mad)
        descriptor = {
            "index": int(idx),
            "edge_count": int(edge_count),
            "vertex_count": int(len(vertex_ids)),
            "center": to_vector3(center),
            "radius_mean": float(radius_mean),
            "radius_rel_mad": float(radius_rel_mad),
            "geometry_radius_rel_mad": float(roundness_rel_mad),
            "aspect_ratio": float(aspect_ratio),
            "extent_u": float(extent[0]),
            "extent_v": float(extent[1]),
            "angle_step_cv": float(angle_cv),
            "shape_class": "round",
        }
        if aspect_ratio >= 2.20 and edge_count >= 8:
            descriptor["shape_class"] = "slot_like"
        elif 1.35 <= aspect_ratio < 2.20 and max(radius_rel_mad, roundness_rel_mad) >= 0.035:
            descriptor["shape_class"] = "elliptic_like"
        elif 6 <= edge_count <= 12 and aspect_ratio < 1.55 and max(radius_rel_mad, roundness_rel_mad) >= 0.025:
            descriptor["shape_class"] = "polygonal_low_edge"
        elif max(radius_rel_mad, roundness_rel_mad) >= 0.10:
            descriptor["shape_class"] = "nonround"
        out.append(descriptor)
    return tuple(out)


def _strong_tessellated_side_pair_descriptors(
    *,
    loop_descriptors: tuple[Mapping[str, object], ...],
    boundary_loop_geometry: tuple[BoundaryLoopGeometry, ...],
    selected_frame_radius: float,
) -> tuple[dict[str, object], ...]:
    """Find diagnostic two-side opening pairs without promoting a bore.

    This mirrors X1's guarded tessellated side-pair idea: two plausible opposite
    loops are evidence, not acceptance.  Rebuild remains forbidden until a true
    mesh-native algorithm promotes and validates a candidate.
    """

    if len(boundary_loop_geometry) < 2:
        return ()
    desc_by_index = {int(item.get("index", -1)): item for item in loop_descriptors}
    radius_ref = float(selected_frame_radius or 0.0)
    pairs: list[dict[str, object]] = []
    for i, a in enumerate(boundary_loop_geometry):
        ra = float(getattr(a, "radius", 0.0) or 0.0)
        if ra <= 1.0e-9 or int(getattr(a, "edge_count", 0) or 0) < 6:
            continue
        for b in boundary_loop_geometry[i + 1 :]:
            rb = float(getattr(b, "radius", 0.0) or 0.0)
            if rb <= 1.0e-9 or int(getattr(b, "edge_count", 0) or 0) < 6:
                continue
            radius_delta_rel = abs(ra - rb) / max(max(ra, rb), 1.0e-9)
            if radius_delta_rel > 0.28:
                continue
            axial_distance = abs(float(getattr(a, "axial_position", 0.0) or 0.0) - float(getattr(b, "axial_position", 0.0) or 0.0))
            if axial_distance < max(0.10 * max(ra, rb), 1.0e-6):
                continue
            selected_delta_rel = abs(0.5 * (ra + rb) - radius_ref) / max(max(radius_ref, ra, rb), 1.0e-9) if radius_ref > 0.0 else 0.0
            score = float(1.0 - min(0.55 * radius_delta_rel + 0.45 * selected_delta_rel, 1.0))
            pairs.append(
                {
                    "loop_indices": (int(getattr(a, "index", i)), int(getattr(b, "index", i + 1))),
                    "radius_a": float(ra),
                    "radius_b": float(rb),
                    "radius_delta_rel": float(radius_delta_rel),
                    "selected_radius_delta_rel": float(selected_delta_rel),
                    "axial_distance": float(axial_distance),
                    "shape_a": str(desc_by_index.get(int(getattr(a, "index", i)), {}).get("shape_class", "unknown")),
                    "shape_b": str(desc_by_index.get(int(getattr(b, "index", i + 1)), {}).get("shape_class", "unknown")),
                    "score": float(score),
                }
            )
    pairs.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return tuple(pairs)

def _x1_family_diagnostic_candidates(
    *,
    valid_face_ids: tuple[int, ...],
    boundary_loop_geometry: tuple[BoundaryLoopGeometry, ...],
    boundary_loops: tuple[tuple[EdgeKey, ...], ...] = (),
    vertices: object | None = None,
    region_axis: object,
    selected_frame_radius: float,
    axial_span_all: float,
    borehole_candidate: Mapping[str, object] | None,
    chamfer_candidates: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Emit conservative X1-inspired diagnostic feature-family rows.

    This is vocabulary and evidence-ledger support, not algorithm parity with the
    FreeCAD/OCCT X1 macro.  All rows from this helper are non-rebuildable until
    a mesh-native rebuild implementation exists for that family.
    """

    loops = tuple(boundary_loop_geometry or ())
    out: list[dict[str, object]] = []
    if not valid_face_ids:
        return ()
    axis = region_axis
    radii = [float(getattr(loop, "radius", 0.0) or 0.0) for loop in loops if float(getattr(loop, "radius", 0.0) or 0.0) > 1.0e-9]
    loop_count = int(len(loops))
    radius_families: list[list[BoundaryLoopGeometry]] = []
    for loop in sorted(loops, key=lambda item: float(getattr(item, "radius", 0.0) or 0.0)):
        r = float(getattr(loop, "radius", 0.0) or 0.0)
        if r <= 1.0e-9:
            continue
        placed = False
        for fam in radius_families:
            ref = float(np.median([float(getattr(member, "radius", 0.0) or 0.0) for member in fam])) if fam else r
            tol = max(0.20, 0.12 * max(abs(ref), abs(r), 1.0))
            if abs(r - ref) <= tol:
                fam.append(loop)
                placed = True
                break
        if not placed:
            radius_families.append([loop])
    family_count = int(len(radius_families))
    radius_span = float(max(radii) - min(radii)) if radii else 0.0
    median_radius = float(np.median(radii)) if radii else float(selected_frame_radius or 0.0)
    diag_base = {
        "boundary_loop_count": int(loop_count),
        "radius_family_count": int(family_count),
        "radius_span": float(radius_span),
        "median_radius": float(median_radius),
        "boundary_loop_sample": tuple(
            {
                "index": int(getattr(loop, "index", i)),
                "edge_count": int(getattr(loop, "edge_count", 0)),
                "radius": float(getattr(loop, "radius", 0.0) or 0.0),
                "radius_rel_mad": float(getattr(loop, "radius_rel_mad", 0.0) or 0.0),
                "axial_position": float(getattr(loop, "axial_position", 0.0) or 0.0),
            }
            for i, loop in enumerate(loops[:8])
        ),
    }

    loop_shape_descriptors = _boundary_loop_shape_descriptors(
        boundary_loops=tuple(boundary_loops or ()),
        boundary_loop_geometry=tuple(boundary_loop_geometry or ()),
        vertices=vertices,
        axis=axis,
    )
    tessellated_side_pairs = _strong_tessellated_side_pair_descriptors(
        loop_descriptors=tuple(loop_shape_descriptors),
        boundary_loop_geometry=tuple(boundary_loop_geometry or ()),
        selected_frame_radius=float(selected_frame_radius),
    )
    if loop_shape_descriptors:
        diag_base = {
            **diag_base,
            "loop_shape_descriptor_count": int(len(loop_shape_descriptors)),
            "loop_shape_descriptors": tuple(loop_shape_descriptors[:12]),
            "loop_shape_classes": tuple(str(item.get("shape_class", "unknown")) for item in loop_shape_descriptors),
        }
    if tessellated_side_pairs:
        diag_base = {
            **diag_base,
            "tessellated_side_pair_count": int(len(tessellated_side_pairs)),
            "tessellated_side_pairs": tuple(tessellated_side_pairs[:8]),
        }

    # Authority-boundary rule: a bore next to a chamfer is not a feature family.
    # It is composition/adjacency metadata only.  The physical feature objects
    # remain separate BORE and CHAMFER_FORM candidates, and rebuild_target policy
    # must never receive a synthetic "chamfered bore" candidate.

    if loop_count >= 3 and family_count >= 2 and radius_span > max(0.25, 0.08 * max(median_radius, 1.0)):
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.counterbore_or_step_stack.1",
                family=FeatureFamily.STEPPED_BORE_STACK,
                display_name="STEPPED BORE STACK — radius-family review",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.OPENING_RING, EvidenceKind.RADIUS_STACK, EvidenceKind.OPPOSITE_OPENING),
                status="review_radius_stack_detected_no_rebuild_support_yet",
                confidence=0.46,
                recognition_rule="multiple_boundary_loop_radius_families",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics=diag_base,
                stage=RecognitionStage.REVIEW,
            )
        )
        if family_count == 2:
            out.append(
                _diagnostic_candidate(
                    candidate_id="component_engine.family.counterbore.1",
                    family=FeatureFamily.COUNTERBORE,
                    display_name="COUNTERBORE — two-radius family review",
                    face_ids=valid_face_ids,
                    evidence_kinds=(EvidenceKind.OPENING_RING, EvidenceKind.RADIUS_STACK, EvidenceKind.PROJECTED_RADIUS_ANCHOR),
                    status="review_two_radius_families_no_rebuild_support_yet",
                    confidence=0.42,
                    recognition_rule="two_radius_boundary_loop_family_pattern",
                    primitive_axis=axis,
                    primitive_radius=max(radii) if radii else median_radius,
                    primitive_depth=axial_span_all,
                    diagnostics=diag_base,
                    stage=RecognitionStage.REVIEW,
                )
            )

    if loop_count == 1 and borehole_candidate is None:
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.circular_pocket.1",
                family=FeatureFamily.CIRCULAR_POCKET,
                display_name="CIRCULAR POCKET — single-opening review",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.OPENING_RING, EvidenceKind.PROJECTED_RADIUS_ANCHOR),
                status="review_single_opening_circular_pocket_candidate_no_rebuild_support_yet",
                confidence=0.34,
                recognition_rule="single_boundary_loop_without_promoted_borehole",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics=diag_base,
                stage=RecognitionStage.REVIEW,
            )
        )

    slot_like_loops = tuple(item for item in loop_shape_descriptors if str(item.get("shape_class", "")) == "slot_like")
    if slot_like_loops:
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.slot_or_adjustable_bore.1",
                family=FeatureFamily.SLOT_OR_ADJUSTABLE_BORE,
                display_name="SLOT / ADJUSTABLE BORE — elongated loop review",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.NONROUND_LOOP, EvidenceKind.SLOT_ASPECT),
                status="diagnostic_slot_like_opening_detected_no_rebuild_support_yet",
                confidence=0.34,
                recognition_rule="projected_loop_aspect_ratio_slot_like",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics={**diag_base, "slot_like_loop_count": int(len(slot_like_loops)), "slot_like_loop_sample": tuple(slot_like_loops[:6])},
                stage=RecognitionStage.DIAGNOSTIC_ONLY,
            )
        )

    elliptic_like_loops = tuple(
        item for item in loop_shape_descriptors
        if str(item.get("shape_class", "")) in {"elliptic_like", "nonround"}
    )
    if elliptic_like_loops:
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.elliptic_or_oval_bore.1",
                family=FeatureFamily.ELLIPTIC_OR_OVAL_BORE,
                display_name="ELLIPTIC / OVAL BORE — projected-loop review",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.NONROUND_LOOP, EvidenceKind.ELLIPSE_FIT),
                status="diagnostic_elliptic_or_nonround_loop_detected_no_rebuild_support_yet",
                confidence=0.32,
                recognition_rule="projected_loop_aspect_or_radius_deviation",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics={**diag_base, "elliptic_like_loop_count": int(len(elliptic_like_loops)), "elliptic_like_loop_sample": tuple(elliptic_like_loops[:6])},
                stage=RecognitionStage.DIAGNOSTIC_ONLY,
            )
        )

    polygonal_loops = tuple(item for item in loop_shape_descriptors if str(item.get("shape_class", "")) == "polygonal_low_edge")
    if polygonal_loops:
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.hex_nut_pocket.1",
                family=FeatureFamily.HEX_NUT_POCKET,
                display_name="HEX / NUT POCKET — polygonal loop review",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.NONROUND_LOOP, EvidenceKind.HEX_CORNER_PATTERN),
                status="diagnostic_polygonal_low_edge_loop_possible_hex_nut_pocket_no_rebuild_support_yet",
                confidence=0.30,
                recognition_rule="projected_low_edge_polygonal_loop",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics={**diag_base, "polygonal_loop_count": int(len(polygonal_loops)), "polygonal_loop_sample": tuple(polygonal_loops[:6])},
                stage=RecognitionStage.DIAGNOSTIC_ONLY,
            )
        )

    if tessellated_side_pairs and borehole_candidate is None:
        best_pair = dict(tessellated_side_pairs[0])
        out.append(
            _diagnostic_candidate(
                candidate_id="component_engine.family.tessellated_bore_candidate.1",
                family=FeatureFamily.TESSELLATED_BORE_CANDIDATE,
                display_name="TESSELLATED BORE CANDIDATE — guarded side-pair preview",
                face_ids=valid_face_ids,
                evidence_kinds=(EvidenceKind.SIDE_PAIR, EvidenceKind.OPENING_RING, EvidenceKind.PROJECTED_RADIUS_ANCHOR),
                status="promotion_preview_tessellated_side_pair_no_rebuild_permission",
                confidence=float(max(0.36, min(0.56, float(best_pair.get("score", 0.0) or 0.0)))),
                recognition_rule="guarded_two_loop_side_pair_without_promoted_borehole",
                primitive_axis=axis,
                primitive_radius=median_radius,
                primitive_depth=axial_span_all,
                diagnostics={**diag_base, "best_tessellated_side_pair": best_pair},
                stage=RecognitionStage.PROMOTION_PREVIEW,
            )
        )

    if chamfer_candidates:
        chamfer_face_ids: set[int] = set()
        for candidate in chamfer_candidates:
            chamfer_face_ids.update(tuple_ints(candidate.get("face_ids", ())))
        if chamfer_face_ids:
            out.append(
                _diagnostic_candidate(
                    candidate_id="component_engine.family.tessellated_chamfer_body.1",
                    family=FeatureFamily.TESSELLATED_CHAMFER_BODY,
                    display_name="TESSELLATED CHAMFER BODY — family evidence review",
                    face_ids=tuple(sorted(chamfer_face_ids)),
                    evidence_kinds=(EvidenceKind.CHAMFER_BAND, EvidenceKind.SELECTED_EDGE_LOOP),
                    status="review_tessellated_chamfer_body_evidence_no_independent_rebuild_permission",
                    confidence=0.38,
                    recognition_rule="accepted_chamfer_surface_components_grouped_as_tessellated_chamfer_body_family",
                    primitive_axis=axis,
                    primitive_radius=median_radius,
                    primitive_depth=axial_span_all,
                    diagnostics={**diag_base, "source_chamfer_candidate_count": int(len(chamfer_candidates))},
                    stage=RecognitionStage.REVIEW,
                )
            )

    return tuple(out)

def _component_engine_patch_boundary_diagnostics(faces: np.ndarray, face_ids: Iterable[int]) -> dict[str, object]:
    """Topology diagnostics for an owned candidate face patch.

    Phase 8.2b: this helper is component-engine-specific reporting, but the
    topology facts come from topology.py.  Recognition may promote only physical
    surface objects.  For rebuildable annular/cylindrical patches, the owned
    patch should normally be one connected face component with two boundary
    vertex loops.
    """

    selected = tuple(sorted({int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(faces)}))
    if not selected:
        return {
            "face_count": 0,
            "face_component_count": 0,
            "boundary_edge_count": 0,
            "boundary_loop_count": 0,
            "boundary_loop_lengths": (),
            "boundary_loop_edge_counts": (),
            "patch_topology_rebuildable": False,
            "reason": "empty_face_patch",
        }

    summary = summarize_patch_topology(faces, selected)
    boundary_edges = boundary_edges_for_face_patch(faces, selected)
    boundary_loop_edge_components = edge_loop_components(boundary_edges)
    loop_lengths: list[int] = []
    for component in boundary_loop_edge_components:
        vertices_in_component = {int(v) for edge in component for v in edge}
        loop_lengths.append(int(len(vertices_in_component)))
    loop_lengths.sort(reverse=True)

    face_components = int(summary.get("component_count", 0) or 0)
    boundary_loop_count = int(len(boundary_loop_edge_components))
    patch_topology_rebuildable = bool(face_components == 1 and boundary_loop_count == 2 and all(v >= 3 for v in loop_lengths[:2]))
    reason = "one_component_two_boundary_loops" if patch_topology_rebuildable else "not_one_component_two_boundary_loops"
    return {
        "face_count": int(len(selected)),
        "face_component_count": int(face_components),
        "component_face_counts": tuple(int(v) for v in tuple(summary.get("component_face_counts", ()) or ())),
        "boundary_edge_count": int(len(boundary_edges)),
        "boundary_loop_count": int(boundary_loop_count),
        "boundary_loop_lengths": tuple(int(v) for v in loop_lengths),
        "boundary_loop_edge_counts": tuple(int(len(item)) for item in boundary_loop_edge_components),
        "patch_topology_rebuildable": bool(patch_topology_rebuildable),
        "reason": reason,
        "topology_source": "topology.py",
    }



def _x1_safe_float(value: object, default: float = 0.0) -> float:
    """Convert numeric-ish diagnostics to finite floats."""

    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _x1_candidate_radius(item: Mapping[str, object], *, fallback: float = 0.0) -> float:
    """Return the best radius estimate available on a candidate dictionary."""

    for key in ("radius", "borehole_core_radius", "inner_radius", "outer_radius", "mouth_radius", "selected_frame_radius"):
        if key in item:
            value = _x1_safe_float(item.get(key), fallback)
            if value > 0.0:
                return value
    return float(fallback)


def _x1_interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    """Distance between two 1D intervals; zero means overlapping/touching."""

    lo0, hi0 = sorted((float(a_min), float(a_max)))
    lo1, hi1 = sorted((float(b_min), float(b_max)))
    if hi0 < lo1:
        return float(lo1 - hi0)
    if hi1 < lo0:
        return float(lo0 - hi1)
    return 0.0


def _x1_family_from_entity_type(entity_type: str) -> str:
    """Compatibility map from old entity_type strings to feature families."""

    kind = str(entity_type or "").strip().lower()
    if kind == "borehole":
        return FeatureFamily.BORE.value
    if kind == "chamfer":
        return FeatureFamily.CHAMFER_FORM.value
    if kind in {item.value for item in FeatureFamily}:
        return kind
    return FeatureFamily.UNKNOWN.value


def _x1_feature_relationship_graph(
    *,
    features: list[dict[str, object]],
    adjacency: Mapping[int, tuple[int, ...]],
) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, tuple[dict[str, object], ...]], tuple[dict[str, object], ...]]:
    """Build typed feature-object relationships without classifying assemblies.

    This is the v6 correction pass after the chamfered-bore family mistake: a
    bore plus chamfer is represented as two independent candidates linked by a
    relationship row.  Relationship rows are display/evidence metadata only.
    They cannot request a DeletePatchProposal.
    """

    relation_map: dict[str, list[dict[str, object]]] = {
        str(item.get("candidate_id", f"candidate_{idx}")): []
        for idx, item in enumerate(features)
    }
    typed_relation_map: dict[str, list[dict[str, object]]] = {
        str(item.get("candidate_id", f"candidate_{idx}")): []
        for idx, item in enumerate(features)
    }
    bore_chamfer_relation_pairs: list[dict[str, object]] = []

    for i, a in enumerate(features):
        a_id = str(a.get("candidate_id", f"candidate_{i}"))
        a_kind = str(a.get("entity_type", ""))
        a_family = str(a.get("feature_family", _x1_family_from_entity_type(a_kind)) or _x1_family_from_entity_type(a_kind))
        a_ids = set(tuple_ints(a.get("face_ids", ())))
        for j, b in enumerate(features):
            if j <= i:
                continue
            b_id = str(b.get("candidate_id", f"candidate_{j}"))
            b_kind = str(b.get("entity_type", ""))
            b_family = str(b.get("feature_family", _x1_family_from_entity_type(b_kind)) or _x1_family_from_entity_type(b_kind))
            b_ids = set(tuple_ints(b.get("face_ids", ())))

            adjacency_pairs = 0
            touching_a: set[int] = set()
            touching_b: set[int] = set()
            for fid in a_ids:
                for nb in adjacency.get(int(fid), ()):  # type: ignore[arg-type]
                    if int(nb) in b_ids:
                        adjacency_pairs += 1
                        touching_a.add(int(fid))
                        touching_b.add(int(nb))
            if adjacency_pairs <= 0:
                continue

            relationship_kind: FeatureRelationshipKind | str = FeatureRelationshipKind.ADJACENT_SURFACE_COMPONENT
            relation_name = FeatureRelationshipKind.ADJACENT_SURFACE_COMPONENT.value
            classification_policy = "relationship_only_not_feature_family"
            role = "adjacent_physical_feature_object"
            confidence = min(0.92, 0.35 + 0.04 * min(int(adjacency_pairs), 8))
            extra_diag: dict[str, object] = {}

            if {a_kind, b_kind} == {"borehole", "chamfer"}:
                relationship_kind = FeatureRelationshipKind.BORE_CHAMFER_ADJACENCY
                relation_name = FeatureRelationshipKind.BORE_CHAMFER_ADJACENCY.value
                role = "bore_chamfer_composition_metadata_only"
                bore = a if a_kind == "borehole" else b
                chamfer = a if a_kind == "chamfer" else b
                bore_radius = _x1_candidate_radius(bore)
                chamfer_inner = _x1_safe_float(chamfer.get("inner_radius", chamfer.get("borehole_core_radius", chamfer.get("radius", 0.0))), 0.0)
                if chamfer_inner <= 0.0:
                    chamfer_inner = _x1_candidate_radius(chamfer)
                radius_tolerance = max(0.75, 0.22 * max(abs(bore_radius), 1.0))
                radius_delta = abs(float(chamfer_inner) - float(bore_radius))
                radius_compatible = bool(radius_delta <= radius_tolerance)
                axial_gap = _x1_interval_gap(
                    _x1_safe_float(bore.get("axial_min", 0.0)),
                    _x1_safe_float(bore.get("axial_max", 0.0)),
                    _x1_safe_float(chamfer.get("axial_min", 0.0)),
                    _x1_safe_float(chamfer.get("axial_max", 0.0)),
                )
                axial_tolerance = max(0.75, 0.12 * max(abs(bore_radius), 1.0))
                axial_near = bool(axial_gap <= axial_tolerance)
                confidence = 0.30
                confidence += 0.24 * min(float(adjacency_pairs) / 8.0, 1.0)
                confidence += 0.22 if radius_compatible else 0.0
                confidence += 0.14 if axial_near else 0.0
                confidence += 0.10
                confidence = float(min(confidence, 0.97))
                extra_diag = {
                    "bore_radius": float(bore_radius),
                    "chamfer_inner_radius": float(chamfer_inner),
                    "radius_delta": float(radius_delta),
                    "radius_tolerance": float(radius_tolerance),
                    "radius_compatible": bool(radius_compatible),
                    "axial_gap": float(axial_gap),
                    "axial_tolerance": float(axial_tolerance),
                    "axial_near_or_touching": bool(axial_near),
                    "assembly_name_if_user_facing": "bore_with_chamfer",
                    "assembly_name_policy": "display_relationship_only_do_not_create_feature_family",
                }

            base_diag = {
                "relation": relation_name,
                "classification_policy": classification_policy,
                "adjacent_face_pair_count": int(adjacency_pairs),
                "source_entity_type": a_kind,
                "target_entity_type": b_kind,
                **extra_diag,
            }
            typed_ab = FeatureRelationshipData(
                relationship_kind=relationship_kind,
                source_candidate_id=a_id,
                target_candidate_id=b_id,
                source_feature_family=a_family,
                target_feature_family=b_family,
                role=role,
                confidence=confidence,
                source_face_ids=tuple(sorted(touching_a)),
                target_face_ids=tuple(sorted(touching_b)),
                relation_face_pairs=int(adjacency_pairs),
                diagnostics=base_diag,
            ).to_dict()
            typed_ba = FeatureRelationshipData(
                relationship_kind=relationship_kind,
                source_candidate_id=b_id,
                target_candidate_id=a_id,
                source_feature_family=b_family,
                target_feature_family=a_family,
                role=role,
                confidence=confidence,
                source_face_ids=tuple(sorted(touching_b)),
                target_face_ids=tuple(sorted(touching_a)),
                relation_face_pairs=int(adjacency_pairs),
                diagnostics={
                    **base_diag,
                    "source_entity_type": b_kind,
                    "target_entity_type": a_kind,
                },
            ).to_dict()
            relation_ab = {
                "candidate_id": b_id,
                "entity_type": b_kind,
                "relation": relation_name,
                "relationship_kind": relation_name,
                "adjacent_face_pair_count": int(adjacency_pairs),
                "local_face_count": int(len(touching_a)),
                "remote_face_count": int(len(touching_b)),
                "confidence": float(confidence),
                "classification_policy": classification_policy,
                "typed_relationship": typed_ab,
            }
            relation_ba = {
                "candidate_id": a_id,
                "entity_type": a_kind,
                "relation": relation_name,
                "relationship_kind": relation_name,
                "adjacent_face_pair_count": int(adjacency_pairs),
                "local_face_count": int(len(touching_b)),
                "remote_face_count": int(len(touching_a)),
                "confidence": float(confidence),
                "classification_policy": classification_policy,
                "typed_relationship": typed_ba,
            }
            relation_map.setdefault(a_id, []).append(relation_ab)
            relation_map.setdefault(b_id, []).append(relation_ba)
            typed_relation_map.setdefault(a_id, []).append(typed_ab)
            typed_relation_map.setdefault(b_id, []).append(typed_ba)

            if relation_name == FeatureRelationshipKind.BORE_CHAMFER_ADJACENCY.value:
                bore_chamfer_relation_pairs.append(
                    {
                        "a_candidate_id": a_id,
                        "a_entity_type": a_kind,
                        "b_candidate_id": b_id,
                        "b_entity_type": b_kind,
                        "relation": relation_name,
                        "classification_policy": classification_policy,
                        "adjacent_face_pair_count": int(adjacency_pairs),
                        "confidence": float(confidence),
                        **extra_diag,
                    }
                )

    return (
        {key: tuple(value) for key, value in relation_map.items()},
        {key: tuple(value) for key, value in typed_relation_map.items()},
        tuple(bore_chamfer_relation_pairs),
    )


def _component_engine_feature_candidates(
    *,
    faces: np.ndarray,
    face_ids: tuple[int, ...],
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    region_center: object,
    region_axis: object,
    region_radius: float,
    boundary_loop_geometry: tuple[BoundaryLoopGeometry, ...] = (),
    boundary_loops: tuple[tuple[EdgeKey, ...], ...] = (),
    vertices: object | None = None,
    seed_face_ids: Iterable[int] = (),
) -> dict[str, object]:
    """Recognize physical CandidateData objects from RegionData components.

    The selected RegionData frame and seed faces are rim/opening anchors. They are
    strong candidate-selection evidence, but not final rebuild targets.
    """

    all_face_ids = tuple_ints(face_ids)
    valid_face_ids = tuple(fid for fid in all_face_ids if 0 <= int(fid) < len(face_centroids))
    if not valid_face_ids:
        return {
            "candidate_data": (),
            "features": (),
            "diagnostics": {
                "active_candidate_authority": "surface_component_classifier_v7",
                "topology_helper_source": "topology.py",
                "failed": True,
                "reason": "no_valid_region_faces",
            },
        }

    axis_vec = canonical_axis(region_axis if region_axis is not None else (0.0, 0.0, 1.0))
    center_vec = np.asarray(region_center, dtype=float).reshape(3)
    selected_frame_radius = float(region_radius or 0.0)
    core_radius = float(selected_frame_radius)

    pts = np.asarray(face_centroids, dtype=float)
    nrm = np.asarray(face_normals, dtype=float)
    ids_arr = np.asarray(valid_face_ids, dtype=np.int64)

    local_pts = pts[ids_arr, :3]
    local_normals = nrm[ids_arr, :3] if len(nrm) >= len(pts) else np.zeros_like(local_pts)

    normal_len = np.linalg.norm(local_normals, axis=1)
    unit_normals = np.zeros_like(local_normals)
    ok_normal = normal_len > 1.0e-12
    unit_normals[ok_normal] = local_normals[ok_normal] / normal_len[ok_normal].reshape(-1, 1)

    # v16 BoreLocalFrame unification:
    # RegionData gives the area of interest, but physical feature ownership must
    # be classified in one shared bore-local frame.  Earlier v15 code could infer
    # a better opening-field frame and then continue using axial/radial arrays
    # computed from the selected RegionData frame.  That made wall/chamfer
    # ownership stale and fragmented.  All downstream masks/stats read these
    # arrays by name, so refreshing them here switches Recognition to the active
    # frame without giving Region Select any classification authority.
    rel = np.empty_like(local_pts)
    axial = np.zeros((len(local_pts),), dtype=float)
    radial_vec = np.empty_like(local_pts)
    radial = np.zeros((len(local_pts),), dtype=float)
    radial_dir = np.empty_like(local_pts)
    normal_axis_abs = np.zeros((len(local_pts),), dtype=float)
    radial_normal_alignment = np.zeros((len(local_pts),), dtype=float)
    finite = np.zeros((len(local_pts),), dtype=bool)
    radial_error = np.zeros((len(local_pts),), dtype=float)
    axial_span_all = 0.0
    radial_span_all = 0.0
    active_recognition_frame_reason = "uninitialized"
    recognition_frame_history: list[dict[str, object]] = []

    def _apply_recognition_frame(frame_center: object, frame_axis: object, frame_radius: float, *, reason: str) -> None:
        """Recompute all face-local measurements in one recognition frame.

        This is the boundary between RegionData frame and feature frame:
        RegionData locates the AOI; this active frame classifies wall/transition
        roles.  The selected ring remains evidence and is not automatically the
        bore-wall radius.
        """

        nonlocal axis_vec, center_vec, core_radius
        nonlocal rel, axial, radial_vec, radial, radial_dir
        nonlocal normal_axis_abs, radial_normal_alignment, finite
        nonlocal radial_error, axial_span_all, radial_span_all
        nonlocal active_recognition_frame_reason

        axis_vec = canonical_axis(frame_axis if frame_axis is not None else axis_vec)
        try:
            center_vec = np.asarray(frame_center, dtype=float).reshape(3)
        except Exception:
            center_vec = np.asarray(center_vec, dtype=float).reshape(3)
        try:
            candidate_radius = float(frame_radius or 0.0)
        except Exception:
            candidate_radius = 0.0

        rel = local_pts - center_vec.reshape(1, 3)
        axial = rel @ axis_vec.reshape(3)
        radial_vec = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radial = np.linalg.norm(radial_vec, axis=1)

        radial_dir = np.zeros_like(radial_vec)
        ok_radial = radial > 1.0e-12
        radial_dir[ok_radial] = radial_vec[ok_radial] / radial[ok_radial].reshape(-1, 1)

        normal_axis_abs = np.abs(unit_normals @ axis_vec.reshape(3))
        radial_normal_alignment = np.abs(np.sum(unit_normals * radial_dir, axis=1))
        finite = (
            np.isfinite(axial)
            & np.isfinite(radial)
            & np.isfinite(normal_axis_abs)
            & np.isfinite(radial_normal_alignment)
        )

        if candidate_radius <= 1.0e-9:
            cyl_like = finite & (normal_axis_abs <= 0.55) & (radial_normal_alignment >= 0.20)
            if np.any(cyl_like):
                candidate_radius = float(np.median(radial[cyl_like]))
            elif np.any(finite):
                candidate_radius = float(np.median(radial[finite]))
            else:
                candidate_radius = 0.0
        core_radius = float(candidate_radius)
        radial_error = np.abs(radial - float(core_radius))
        axial_span_all = float(np.max(axial[finite]) - np.min(axial[finite])) if np.any(finite) else 0.0
        radial_span_all = float(np.max(radial[finite]) - np.min(radial[finite])) if np.any(finite) else 0.0
        active_recognition_frame_reason = str(reason)
        recognition_frame_history.append(
            {
                "reason": str(reason),
                "center": to_vector3(center_vec),
                "axis": to_vector3(axis_vec),
                "radius": float(core_radius),
                "finite_face_count": int(np.count_nonzero(finite)),
                "axial_span": float(axial_span_all),
                "radial_span": float(radial_span_all),
            }
        )

    _apply_recognition_frame(center_vec, axis_vec, core_radius, reason="region_data_initial_frame")

    fid_to_local = {int(fid): int(i) for i, fid in enumerate(valid_face_ids)}
    local_to_fid = {int(i): int(fid) for i, fid in enumerate(valid_face_ids)}
    adjacency = face_adjacency_for_patch(faces, valid_face_ids)

    # Recognition anchor policy:
    # The RegionData cutout can be intentionally broad. A cylindrical-looking surface inside
    # that RegionData cutout is not automatically the selected Borehole; it must be anchored
    # to the operator's selected rim evidence. Seed faces come from
    # region_select's exact selected-edge neighborhood, and selected_frame_radius
    # comes from the selected ring fit. These are scoring/selection anchors
    # only; final deletion still belongs to rebuild_target/rebuild.
    seed_face_set = {int(fid) for fid in tuple_ints(seed_face_ids) if int(fid) in fid_to_local}
    selected_radius_valid = bool(np.isfinite(float(selected_frame_radius)) and float(selected_frame_radius) > 1.0e-6)
    selected_radius_anchor_tolerance = float(max(0.75, 0.32 * max(abs(float(selected_frame_radius)), 1.0)))

    def seed_affinity(ids: Iterable[int]) -> tuple[int, int, bool]:
        ids_set = {int(fid) for fid in tuple_ints(ids) if int(fid) in fid_to_local}
        if not ids_set or not seed_face_set:
            return (0, 0, False)
        direct = ids_set & seed_face_set
        adjacent: set[int] = set()
        for fid in ids_set:
            for nb in adjacency.get(int(fid), ()):
                if int(nb) in seed_face_set:
                    adjacent.add(int(fid))
        return (int(len(direct)), int(len(adjacent)), bool(direct or adjacent))

    def selected_radius_anchor(st: Mapping[str, object]) -> tuple[bool, float, float]:
        if not selected_radius_valid:
            return (False, 0.0, 0.0)
        try:
            radial_med = float(st.get("radial_median", selected_frame_radius) or selected_frame_radius)
        except Exception:
            radial_med = float(selected_frame_radius)
        delta = abs(float(radial_med) - float(selected_frame_radius))
        ratio = float(delta / max(selected_radius_anchor_tolerance, 1.0e-9))
        return (bool(delta <= selected_radius_anchor_tolerance), float(delta), float(ratio))

    def density_normalized_face_fraction(face_count: int, *, reference_count: int | None = None) -> float:
        """Bound face-count influence so dense rebuilds cannot dominate identity.

        Recognition owns physical feature identity.  Raw triangle count is a mesh
        density artifact after rebuilds, not a physical feature-size signal.
        This helper maps face count into a small saturated diagnostic score.
        """

        ref = int(reference_count if reference_count is not None else len(valid_face_ids))
        if ref <= 0:
            ref = 1
        value = float(max(int(face_count), 0)) / float(ref)
        # Saturate early; anything above ~20% of RegionData cutout should not keep gaining
        # recognition authority simply because the mesh is denser.
        return float(min(value / 0.20, 1.0))

    density_bias_guard_used = True
    raw_face_count_score_weight = 0.0
    density_normalized_score_weight = 0.08
    geometric_band_score_weight = 1.0
    selected_rim_anchor_score_weight = 3.2

    def adjacent_count_to_set(ids: Iterable[int], target_ids: Iterable[int]) -> int:
        ids_set = {int(fid) for fid in tuple_ints(ids) if int(fid) in fid_to_local}
        target_set = {int(fid) for fid in tuple_ints(target_ids) if int(fid) in fid_to_local}
        if not ids_set or not target_set:
            return 0
        count = 0
        for fid in ids_set:
            for nb in adjacency.get(int(fid), ()):
                if int(nb) in target_set:
                    count += 1
        return int(count)

    def selected_seed_required_for_remote_component() -> bool:
        """Return True when seed evidence exists and should localize recognition.

        A broad RegionData cutout can contain multiple same-radius or same-axis cylindrical
        surfaces. Radius/axis are geometric similarity evidence, not identity.
        When selected-edge seed faces exist, disconnected or remote components
        must prove they touch that selected-rim neighborhood before they may be
        merged into the BOREHOLE display candidate.
        """

        return bool(seed_face_set)

    def ids_from_mask(mask: np.ndarray) -> tuple[int, ...]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        return tuple(sorted(int(valid_face_ids[i]) for i, flag in enumerate(mask) if bool(flag)))

    def stats(ids: Iterable[int]) -> dict[str, object]:
        ids_t = tuple(fid for fid in tuple_ints(ids) if fid in fid_to_local)
        if not ids_t:
            return {"face_count": 0}
        idx = np.asarray([fid_to_local[fid] for fid in ids_t], dtype=np.int64)
        ax = axial[idx]
        rd = radial[idx]
        re = radial_error[idx]
        na = normal_axis_abs[idx]
        al = radial_normal_alignment[idx]
        return {
            "face_count": int(len(ids_t)),
            "axial_min": float(np.min(ax)),
            "axial_max": float(np.max(ax)),
            "axial_center": float(np.median(ax)),
            "axial_span": float(np.max(ax) - np.min(ax)),
            "radial_min": float(np.min(rd)),
            "radial_max": float(np.max(rd)),
            "radial_median": float(np.median(rd)),
            "radial_span": float(np.max(rd) - np.min(rd)),
            "radial_error_median": float(np.median(re)),
            "normal_axis_abs_median": float(np.median(na)),
            "radial_normal_alignment_median": float(np.median(al)),
        }

    def component_quality_core(ids: tuple[int, ...]) -> tuple[float, dict[str, object]]:
        st = stats(ids)
        if int(st.get("face_count", 0)) <= 0:
            return (-999999.0, st)
        axial_span = float(st.get("axial_span", 0.0) or 0.0)
        radial_span = float(st.get("radial_span", 999.0) or 999.0)
        radial_med = float(st.get("radial_median", core_radius) or core_radius)
        align = float(st.get("radial_normal_alignment_median", 0.0) or 0.0)
        na = float(st.get("normal_axis_abs_median", 1.0) or 1.0)

        seed_direct_count, seed_adjacent_count, seed_anchor = seed_affinity(ids)
        radius_anchor, radius_delta, radius_ratio = selected_radius_anchor(st)
        radius_anchor_score = 0.0
        if selected_radius_valid:
            radius_anchor_score = max(0.0, 1.0 - min(radius_ratio, 1.0))

        # Cylindrical components are long in axial direction, compact in radial
        # thickness, and have normals mostly perpendicular to the axis. But the
        # RegionData cutout can contain unrelated cylindrical/rounded surfaces. Therefore the
        # selected rim radius and selected-edge seed neighborhood are strong
        # recognition anchors for *which* cylinder is the Borehole.
        #
        # Do not score by raw face count. After a rebuild, the same physical bore
        # may contain thousands of triangles while the chamfer/pocket around it is
        # sparse. Raw face count would make Recognition density-dependent.
        radial_compactness = 1.0 - min(radial_span / max(abs(radial_med), 1.0e-9), 1.0)
        axial_coverage_score = min(axial_span / max(axial_span_all, 1.0e-9), 1.0)
        normal_family_score = max(0.0, min(align, 1.0)) + max(0.0, 1.0 - min(na / 0.82, 1.0))
        density_face_score = density_normalized_face_fraction(len(ids))
        geometric_band_score = (
            1.35 * axial_coverage_score
            + 1.10 * radial_compactness
            + 0.70 * normal_family_score
        )
        score = (
            geometric_band_score_weight * geometric_band_score
            + density_normalized_score_weight * density_face_score
            + selected_rim_anchor_score_weight * radius_anchor_score
            + (2.4 if seed_anchor else 0.0)
        )
        if selected_radius_valid and not seed_anchor:
            # Large unrelated cylindrical RegionData cutout surfaces must not win simply by
            # face count/depth when their radius is far from the selected rim.
            score -= 2.2 * min(radius_ratio, 3.0)
        st = {
            **st,
            "radial_compactness": float(radial_compactness),
            "axial_coverage_score": float(axial_coverage_score),
            "normal_family_score": float(normal_family_score),
            "density_normalized_face_score": float(density_face_score),
            "raw_face_count_score_weight": float(raw_face_count_score_weight),
            "density_normalized_score_weight": float(density_normalized_score_weight),
            "geometric_band_score": float(geometric_band_score),
            "geometric_band_score_weight": float(geometric_band_score_weight),
            "selected_rim_anchor_score_weight": float(selected_rim_anchor_score_weight),
            "density_bias_guard_used": bool(density_bias_guard_used),
            "selected_radius_anchor": bool(radius_anchor),
            "selected_radius_delta": float(radius_delta),
            "selected_radius_delta_ratio": float(radius_ratio),
            "selected_radius_anchor_tolerance": float(selected_radius_anchor_tolerance),
            "seed_direct_face_count": int(seed_direct_count),
            "seed_adjacent_face_count": int(seed_adjacent_count),
            "seed_anchor": bool(seed_anchor),
            "recognition_anchor_ok": bool(radius_anchor or seed_anchor or not selected_radius_valid),
        }
        return (float(score), st)

    # BOREHOLE core: component-first cylindrical surface detection.
    #
    # Do NOT require the selected RegionData cutout frame radius to equal the borehole radius.
    # On chamfered mouths the selected frame can describe a mouth/transition
    # radius, while the real borehole object is a different cylindrical surface.
    # The radius is measured from the chosen cylindrical component after
    # classification.
    min_core_faces = max(18, min(96, int(0.018 * min(len(valid_face_ids), 2400))))

    strict_cylinder_mask = (
        finite
        & (normal_axis_abs <= 0.42)
        & (radial_normal_alignment >= 0.24)
    )
    normal_cylinder_mask = (
        finite
        & (normal_axis_abs <= 0.56)
        & (radial_normal_alignment >= 0.12)
    )
    loose_cylinder_mask = (
        finite
        & (normal_axis_abs <= 0.66)
        & (radial_normal_alignment >= 0.06)
    )

    # Use the normal/loose cylindrical mask as the component source.  The
    # stricter mask is good as a confidence hint, but on double-ended chamfered
    # bores it can fragment the cylindrical wall into partial strips, which then
    # produces a non-watertight rebuild candidate.  Component scoring below
    # still rejects caps/transitions by normal quality and radial compactness.
    strict_core_ids_source = ids_from_mask(strict_cylinder_mask)
    core_rule = "normal_component_first_cylindrical_normals_complete_component_source"
    core_ids_source = ids_from_mask(normal_cylinder_mask)
    if len(core_ids_source) < min_core_faces:
        core_rule = "loose_component_first_cylindrical_normals_complete_component_source"
        core_ids_source = ids_from_mask(loose_cylinder_mask)

    # Remove obvious long/flat cap noise by splitting into connected components
    # and scoring intrinsic cylindrical quality.  Radius is only a measurement
    # result, not a gate.
    core_components = connected_face_components(faces, core_ids_source)
    scored_core_components: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
    for comp in core_components:
        if len(comp) < min_core_faces:
            continue
        score, st = component_quality_core(comp)
        scored_core_components.append((score, comp, st))
    scored_core_components.sort(key=lambda item: (-item[0], -len(item[1])))

    core_selection_anchor_policy = "unanchored_intrinsic_cylindrical_quality"
    anchored_core_components = [
        item for item in scored_core_components
        if bool(item[2].get("recognition_anchor_ok", False))
    ]
    if anchored_core_components:
        scored_core_components = anchored_core_components
        core_selection_anchor_policy = "selected_rim_radius_or_seed_face_anchored"

    chosen_core_ids: tuple[int, ...] = ()
    chosen_core_stats: dict[str, object] = {}
    merged_core_component_count = 0
    primary_core_component_face_count = 0
    if scored_core_components:
        primary_score, primary_ids, primary_stats = scored_core_components[0]
        primary_core_component_face_count = int(len(primary_ids))
        primary_radius = float(primary_stats.get("radial_median", core_radius) or core_radius)
        primary_axial_span = float(primary_stats.get("axial_span", 0.0) or 0.0)
        merge_ids: set[int] = set(primary_ids)
        merge_stats: list[dict[str, object]] = [dict(primary_stats)]
        radius_merge_tol = max(0.65, 0.10 * max(abs(primary_radius), 1.0))
        min_merge_faces = max(6, int(0.25 * float(min_core_faces)))
        for _, comp_ids, comp_stats in scored_core_components[1:]:
            if len(comp_ids) < min_merge_faces:
                continue
            comp_radius = float(comp_stats.get("radial_median", primary_radius) or primary_radius)
            comp_radial_span = float(comp_stats.get("radial_span", 999.0) or 999.0)
            comp_na = float(comp_stats.get("normal_axis_abs_median", 1.0) or 1.0)
            comp_align = float(comp_stats.get("radial_normal_alignment_median", 0.0) or 0.0)
            comp_axial_span = float(comp_stats.get("axial_span", 0.0) or 0.0)
            comp_radius_anchor = bool(comp_stats.get("selected_radius_anchor", False))
            comp_seed_anchor = bool(comp_stats.get("seed_anchor", False))
            comp_adjacent_to_merge = int(adjacent_count_to_set(comp_ids, merge_ids))
            # Display-candidate isolation rule:
            # A selected Borehole preview must remain one physical surface object.
            # Seed affinity proves a component is close to *some* selected-edge
            # neighborhood, but it does not prove that a disconnected component is
            # the same bore surface.  Nearby/stacked bores can have the same radius
            # and can both be seed-adjacent inside a broad RegionData cutout.  Therefore preview
            # ownership may merge only components that are face-adjacent to the
            # current core object.  Disconnected same-cylinder material is left as
            # unclassified/context for rebuild_target, not displayed as the active
            # BOREHOLE candidate.
            connected_to_current_core = bool(comp_adjacent_to_merge > 0)
            same_cylinder = bool(
                abs(comp_radius - primary_radius) <= radius_merge_tol
                and comp_radial_span <= max(2.0, 0.22 * max(abs(primary_radius), 1.0))
                and comp_na <= 0.68
                and comp_align >= 0.04
                and comp_axial_span >= max(0.35, 0.02 * max(axial_span_all, primary_axial_span, 1.0))
                and (comp_radius_anchor or comp_seed_anchor or not selected_radius_valid)
                and connected_to_current_core
            )
            if same_cylinder:
                merge_ids.update(int(fid) for fid in comp_ids)
                merged = dict(comp_stats)
                merged["component_merge_adjacent_to_current_count"] = int(comp_adjacent_to_merge)
                merged["component_merge_connected_to_current_core"] = bool(connected_to_current_core)
                merge_stats.append(merged)
        chosen_core_ids = tuple(sorted(merge_ids))
        chosen_core_stats = stats(chosen_core_ids)
        chosen_core_stats = {
            **dict(chosen_core_stats),
            "primary_component_score": float(primary_score),
            "primary_component_face_count": int(primary_core_component_face_count),
            "merged_component_count": int(len(merge_stats)),
            "component_merge_radius_tolerance": float(radius_merge_tol),
            "core_selection_anchor_policy": str(core_selection_anchor_policy),
            "selected_radius_anchor_tolerance": float(selected_radius_anchor_tolerance),
            "selected_seed_face_count": int(len(seed_face_set)),
            "density_bias_guard_used": bool(density_bias_guard_used),
            "raw_face_count_score_weight": float(raw_face_count_score_weight),
            "density_normalized_score_weight": float(density_normalized_score_weight),
            "candidate_isolation_policy": "display_candidate_requires_connected_same_cylinder_surface",
        }
        merged_core_component_count = int(len(merge_stats))

    # Measurement fix:
    # The RegionData cutout/selected frame radius is a search/reference frame.  On a chamfered
    # mouth it can be the outer mouth radius, while the BOREHOLE object is the
    # smaller cylindrical core component.  Candidate dimensions must therefore
    # come from the chosen component itself, not blindly from the selected frame.
    measured_core_radius = float(core_radius)
    if chosen_core_ids:
        try:
            measured_core_radius = float(dict(chosen_core_stats or {}).get("radial_median", measured_core_radius) or measured_core_radius)
        except Exception:
            measured_core_radius = float(core_radius)

    closure_added_face_count = 0
    same_cylinder_completion_added_face_count = 0
    same_cylinder_completion_source_face_count = 0
    same_cylinder_completion_component_count = 0
    same_cylinder_completion_used_component_count = 0
    same_cylinder_completion_diagnostics: tuple[dict[str, object], ...] = ()
    neutral_volume_cutout_completion_added_face_count = 0
    neutral_volume_cutout_completion_source_face_count = 0
    neutral_volume_cutout_completion_component_count = 0
    neutral_volume_cutout_completion_used_component_count = 0
    neutral_volume_cutout_completion_diagnostics: tuple[dict[str, object], ...] = ()
    core_patch_topology_before_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)
    core_patch_topology_after_closure = dict(core_patch_topology_before_closure)

    # Topological closure pass:
    # If the cylindrical candidate is a valid semantic cylinder but not a clean
    # annular patch, include directly connected same-cylinder faces that were
    # missed by strict normal filters.  This is still recognition-owned feature
    # completion, not rebuild expansion and not full RegionData cutout fallback.
    if chosen_core_ids:
        try:
            base_stats = dict(chosen_core_stats or {})
            ax_min = float(base_stats.get("axial_min", np.min(axial[finite]) if np.any(finite) else 0.0) or 0.0)
            ax_max = float(base_stats.get("axial_max", np.max(axial[finite]) if np.any(finite) else 0.0) or 0.0)
            axial_pad = max(0.35, 0.012 * max(axial_span_all, 1.0))
            closure_radial_tol = max(0.85, 0.16 * max(abs(measured_core_radius), 1.0))
            closure_mask = (
                finite
                & (np.abs(radial - measured_core_radius) <= closure_radial_tol)
                & (normal_axis_abs <= 0.76)
                & (axial >= ax_min - axial_pad)
                & (axial <= ax_max + axial_pad)
            )
            closure_allowed = set(ids_from_mask(closure_mask))
            start_ids = set(int(fid) for fid in chosen_core_ids)
            expanded = set(start_ids)
            stack = list(start_ids)
            while stack:
                fid = int(stack.pop())
                for nb in adjacency.get(fid, ()):
                    nb = int(nb)
                    if nb in closure_allowed and nb not in expanded:
                        expanded.add(nb)
                        stack.append(nb)
            if len(expanded) > len(start_ids):
                chosen_core_ids = tuple(sorted(expanded))
                chosen_core_stats = {
                    **dict(stats(chosen_core_ids)),
                    **{k: v for k, v in dict(chosen_core_stats or {}).items() if k.startswith("primary_") or k.startswith("merged_") or k.startswith("component_merge_")},
                    "closure_radial_tolerance": float(closure_radial_tol),
                    "closure_axial_padding": float(axial_pad),
                    "closure_allowed_face_count": int(len(closure_allowed)),
                }
                measured_core_radius = float(chosen_core_stats.get("radial_median", measured_core_radius) or measured_core_radius)
                closure_added_face_count = int(len(expanded) - len(start_ids))
                core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)
        except Exception as _closure_exc:
            chosen_core_stats = {**dict(chosen_core_stats or {}), "closure_failed": str(_closure_exc)}
            core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)

    if chosen_core_ids:
        try:
            # Same-cylinder completion pass:
            # Connected-component picking can select only one circumferential half of
            # a clean bore when imported topology is split or face normals are noisy.
            # Complete the BOREHOLE identity from geometry only: same measured
            # radius, same axial slab, cylinder-like normals. This is not a full-RegionData cutout
            # fallback and it deliberately runs before chamfer ownership.
            base_ids = set(int(fid) for fid in chosen_core_ids)
            base_stats = dict(stats(chosen_core_ids))
            core_ax_min = float(base_stats.get("axial_min", 0.0) or 0.0)
            core_ax_max = float(base_stats.get("axial_max", 0.0) or 0.0)
            core_ax_span = max(float(core_ax_max - core_ax_min), 1.0e-9)
            measured_core_radius = float(base_stats.get("radial_median", measured_core_radius) or measured_core_radius)

            completion_axial_pad = max(0.55, 0.035 * max(axial_span_all, core_ax_span, 1.0))
            completion_radial_tol = max(0.55, 0.075 * max(abs(measured_core_radius), 1.0))
            completion_mask = (
                finite
                & (np.abs(radial - measured_core_radius) <= completion_radial_tol)
                & (normal_axis_abs <= 0.62)
                & (radial_normal_alignment >= 0.08)
                & (axial >= core_ax_min - completion_axial_pad)
                & (axial <= core_ax_max + completion_axial_pad)
            )
            completion_source_ids = tuple(sorted(set(ids_from_mask(completion_mask)) | base_ids))
            same_cylinder_completion_source_face_count = int(len(completion_source_ids))
            completion_components = connected_face_components(faces, completion_source_ids)
            same_cylinder_completion_component_count = int(len(completion_components))
            min_completion_faces = max(8, min(48, int(0.18 * float(min_core_faces))))
            expanded = set(base_ids)
            completion_component_diags: list[dict[str, object]] = []

            for comp in completion_components:
                comp_ids = tuple(sorted(int(fid) for fid in comp))
                if len(comp_ids) < min_completion_faces:
                    continue
                comp_set = set(comp_ids)
                comp_stats = stats(comp_ids)
                comp_rad = float(comp_stats.get("radial_median", measured_core_radius) or measured_core_radius)
                comp_rad_span = float(comp_stats.get("radial_span", 999.0) or 999.0)
                comp_ax_min = float(comp_stats.get("axial_min", 0.0) or 0.0)
                comp_ax_max = float(comp_stats.get("axial_max", 0.0) or 0.0)
                comp_ax_span = max(float(comp_ax_max - comp_ax_min), 1.0e-9)
                comp_na = float(comp_stats.get("normal_axis_abs_median", 1.0) or 1.0)
                comp_align = float(comp_stats.get("radial_normal_alignment_median", 0.0) or 0.0)

                overlap = max(0.0, min(comp_ax_max, core_ax_max) - max(comp_ax_min, core_ax_min))
                overlap_ratio = float(overlap / max(min(comp_ax_span, core_ax_span), 1.0e-9))
                overlaps_base = bool(comp_set & base_ids)
                adjacent_to_base_count = int(adjacent_count_to_set(comp_ids, base_ids))
                comp_radius_anchor, comp_selected_radius_delta, comp_selected_radius_ratio = selected_radius_anchor(comp_stats)
                seed_direct, seed_adjacent, comp_seed_anchor = seed_affinity(comp_ids)
                # Completion may fill holes/fragments inside the same preview
                # object, but it must not pull in a separate neighboring bore just
                # because that component has the same radius or touches selected
                # seed faces.  Disconnected material remains RegionData cutout context for the
                # later target-repair stage.
                connected_to_base = bool(overlaps_base or adjacent_to_base_count > 0)
                locality_ok = bool(connected_to_base)
                same_cylinder = bool(
                    abs(comp_rad - measured_core_radius) <= completion_radial_tol
                    and comp_rad_span <= max(1.35, 0.16 * max(abs(measured_core_radius), 1.0))
                    and comp_na <= 0.64
                    and comp_align >= 0.07
                    and locality_ok
                    and (comp_radius_anchor or comp_seed_anchor or overlaps_base or adjacent_to_base_count > 0 or not selected_radius_valid)
                )
                completion_component_diags.append(
                    {
                        "face_count": int(len(comp_ids)),
                        "overlaps_base_candidate": bool(overlaps_base),
                        "adjacent_to_base_face_pair_count": int(adjacent_to_base_count),
                        "same_cylinder": bool(same_cylinder),
                        "locality_ok": bool(locality_ok),
                        "connected_to_base_candidate": bool(connected_to_base),
                        "axial_overlap_ratio": float(overlap_ratio),
                        "radial_median": float(comp_rad),
                        "radial_span": float(comp_rad_span),
                        "normal_axis_abs_median": float(comp_na),
                        "radial_normal_alignment_median": float(comp_align),
                        "selected_radius_anchor": bool(comp_radius_anchor),
                        "selected_radius_delta": float(comp_selected_radius_delta),
                        "selected_radius_delta_ratio": float(comp_selected_radius_ratio),
                        "seed_anchor": bool(comp_seed_anchor),
                        "seed_direct_face_count": int(seed_direct),
                        "seed_adjacent_face_count": int(seed_adjacent),
                    }
                )
                if same_cylinder:
                    expanded.update(comp_set)

            if len(expanded) > len(base_ids):
                chosen_core_ids = tuple(sorted(expanded))
                chosen_core_stats = {
                    **dict(stats(chosen_core_ids)),
                    **{
                        k: v
                        for k, v in dict(chosen_core_stats or {}).items()
                        if k.startswith("primary_") or k.startswith("merged_") or k.startswith("component_merge_")
                    },
                    "same_cylinder_completion_radial_tolerance": float(completion_radial_tol),
                    "same_cylinder_completion_axial_padding": float(completion_axial_pad),
                    "same_cylinder_completion_source_face_count": int(same_cylinder_completion_source_face_count),
                }
                measured_core_radius = float(chosen_core_stats.get("radial_median", measured_core_radius) or measured_core_radius)
                same_cylinder_completion_added_face_count = int(len(expanded) - len(base_ids))
                same_cylinder_completion_used_component_count = int(
                    sum(1 for item in completion_component_diags if bool(item.get("same_cylinder", False)))
                )
                core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)
            same_cylinder_completion_diagnostics = tuple(completion_component_diags[:16])
        except Exception as _completion_exc:
            chosen_core_stats = {**dict(chosen_core_stats or {}), "same_cylinder_completion_failed": str(_completion_exc)}
            core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)


    # Neutral volume-cutout recognition completion pass:
    # The v50/v51 Region Select contract is now a closed neutral selection
    # entity: it sends a measured cylindrical RegionData cutout, not a pre-classified selected
    # bore surface.  Therefore recognition must not expect Region Select to have
    # already isolated the exact BOREHOLE wall.  After the strict/normal component
    # core is found, complete only compact same-radius wall components that are
    # still inside this neutral volume cutout and are either connected to the base core or
    # directly tied to the exact selected rim seed faces.
    #
    # This is still Recognition ownership.  It is not a Region Select side
    # decision, not broad RegionData cutout promotion, and not Target Policy delete expansion.
    if chosen_core_ids:
        try:
            base_ids = set(int(fid) for fid in chosen_core_ids)
            base_stats = dict(stats(chosen_core_ids))
            core_ax_min = float(base_stats.get("axial_min", 0.0) or 0.0)
            core_ax_max = float(base_stats.get("axial_max", 0.0) or 0.0)
            core_ax_span = max(float(core_ax_max - core_ax_min), 1.0e-9)
            measured_core_radius = float(base_stats.get("radial_median", measured_core_radius) or measured_core_radius)

            neutral_radial_tol = max(0.95, 0.095 * max(abs(measured_core_radius), 1.0))
            neutral_axial_pad = max(0.85, 0.08 * max(core_ax_span, 1.0), 0.04 * max(axial_span_all, 1.0))
            neutral_mask = (
                finite
                & (np.abs(radial - measured_core_radius) <= neutral_radial_tol)
                & (normal_axis_abs <= 0.80)
                & (radial_normal_alignment >= 0.025)
                & (axial >= core_ax_min - neutral_axial_pad)
                & (axial <= core_ax_max + neutral_axial_pad)
            )

            neutral_source_ids = tuple(sorted(set(ids_from_mask(neutral_mask)) - base_ids))
            neutral_volume_cutout_completion_source_face_count = int(len(neutral_source_ids))
            neutral_components = connected_face_components(faces, neutral_source_ids)
            neutral_volume_cutout_completion_component_count = int(len(neutral_components))

            added_neutral: set[int] = set()
            neutral_diags: list[dict[str, object]] = []
            min_neutral_faces = max(8, min(48, int(0.12 * float(min_core_faces))))
            max_added_total = max(0, int(0.85 * max(len(base_ids), 1)))
            max_radial_span = max(1.55, 0.16 * max(abs(measured_core_radius), 1.0))
            max_axial_span = max(2.75, 1.15 * max(core_ax_span, 1.0))
            max_axial_gap = max(1.65, 0.20 * max(core_ax_span, 1.0), 0.055 * max(axial_span_all, 1.0))

            for comp in sorted(neutral_components, key=lambda c: len(c), reverse=True):
                comp_ids = tuple(sorted(int(fid) for fid in comp))
                if len(comp_ids) < min_neutral_faces:
                    neutral_diags.append({"face_count": int(len(comp_ids)), "accepted": False, "reason": "too_few_faces"})
                    continue

                comp_set = set(comp_ids)
                comp_stats = stats(comp_ids)
                comp_rad = float(comp_stats.get("radial_median", measured_core_radius) or measured_core_radius)
                comp_rad_span = float(comp_stats.get("radial_span", 999.0) or 999.0)
                comp_ax_min = float(comp_stats.get("axial_min", 0.0) or 0.0)
                comp_ax_max = float(comp_stats.get("axial_max", 0.0) or 0.0)
                comp_ax_span = max(float(comp_ax_max - comp_ax_min), 1.0e-9)
                comp_na = float(comp_stats.get("normal_axis_abs_median", 1.0) or 1.0)
                comp_align = float(comp_stats.get("radial_normal_alignment_median", 0.0) or 0.0)

                if comp_ax_max < core_ax_min:
                    axial_gap = float(core_ax_min - comp_ax_max)
                elif core_ax_max < comp_ax_min:
                    axial_gap = float(comp_ax_min - core_ax_max)
                else:
                    axial_gap = 0.0
                overlap = max(0.0, min(comp_ax_max, core_ax_max) - max(comp_ax_min, core_ax_min))
                overlap_ratio = float(overlap / max(min(comp_ax_span, core_ax_span), 1.0e-9))
                adjacent_to_core = int(adjacent_count_to_set(comp_ids, base_ids | added_neutral))
                seed_direct, seed_adjacent, comp_seed_anchor = seed_affinity(comp_ids)
                comp_radius_anchor, comp_selected_radius_delta, comp_selected_radius_ratio = selected_radius_anchor(comp_stats)

                connected_fragment = bool(adjacent_to_core > 0)
                seeded_fragment = bool(seed_direct > 0)
                same_local_cylinder = bool(
                    abs(comp_rad - measured_core_radius) <= neutral_radial_tol
                    and comp_rad_span <= max_radial_span
                    and comp_ax_span <= max_axial_span
                    and axial_gap <= max_axial_gap
                    and comp_na <= 0.74
                    and comp_align >= 0.035
                )
                # Seeded fragments get a small tolerance relaxation because direct
                # selected-rim evidence is stronger than pure RegionData cutout geometry, but
                # they still must be in the same measured cylinder slab.
                seeded_local_cylinder = bool(
                    seeded_fragment
                    and abs(comp_rad - measured_core_radius) <= max(neutral_radial_tol * 1.20, 1.15)
                    and comp_rad_span <= max(max_radial_span * 1.20, 1.85)
                    and comp_ax_span <= max(max_axial_span * 1.20, 3.25)
                    and axial_gap <= max(max_axial_gap * 1.20, 2.0)
                    and comp_na <= 0.82
                    and comp_align >= 0.020
                )
                accepted = bool((connected_fragment and same_local_cylinder) or seeded_local_cylinder)
                reason = "accepted_neutral_volume_cutout_connected_wall_fragment" if (connected_fragment and same_local_cylinder) else (
                    "accepted_neutral_volume_cutout_seeded_wall_fragment" if seeded_local_cylinder else "rejected_not_same_local_wall_fragment"
                )
                if accepted and max_added_total > 0 and len(added_neutral) + len(comp_ids) > max_added_total:
                    accepted = False
                    reason = "rejected_neutral_completion_cap_reached"

                if accepted:
                    added_neutral.update(comp_set)

                neutral_diags.append(
                    {
                        "face_count": int(len(comp_ids)),
                        "accepted": bool(accepted),
                        "reason": str(reason),
                        "radial_median": float(comp_rad),
                        "radial_span": float(comp_rad_span),
                        "axial_span": float(comp_ax_span),
                        "axial_gap": float(axial_gap),
                        "axial_overlap_ratio": float(overlap_ratio),
                        "normal_axis_abs_median": float(comp_na),
                        "radial_normal_alignment_median": float(comp_align),
                        "adjacent_to_core_face_pair_count": int(adjacent_to_core),
                        "seed_direct_face_count": int(seed_direct),
                        "seed_adjacent_face_count": int(seed_adjacent),
                        "seed_anchor": bool(comp_seed_anchor),
                        "selected_radius_anchor": bool(comp_radius_anchor),
                        "selected_radius_delta": float(comp_selected_radius_delta),
                        "selected_radius_delta_ratio": float(comp_selected_radius_ratio),
                    }
                )

            if added_neutral:
                base_before = set(int(fid) for fid in chosen_core_ids)
                completed = set(base_before) | added_neutral
                chosen_core_ids = tuple(sorted(completed))
                chosen_core_stats = {
                    **dict(stats(chosen_core_ids)),
                    **{
                        k: v
                        for k, v in dict(chosen_core_stats or {}).items()
                        if k.startswith("primary_") or k.startswith("merged_") or k.startswith("component_merge_")
                    },
                    "neutral_volume_cutout_completion_radial_tolerance": float(neutral_radial_tol),
                    "neutral_volume_cutout_completion_axial_padding": float(neutral_axial_pad),
                    "neutral_volume_cutout_completion_source_face_count": int(neutral_volume_cutout_completion_source_face_count),
                }
                measured_core_radius = float(chosen_core_stats.get("radial_median", measured_core_radius) or measured_core_radius)
                neutral_volume_cutout_completion_added_face_count = int(len(completed) - len(base_before))
                neutral_volume_cutout_completion_used_component_count = int(
                    sum(1 for item in neutral_diags if bool(item.get("accepted", False)))
                )
                core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)
            neutral_volume_cutout_completion_diagnostics = tuple(neutral_diags[:16])
        except Exception as _neutral_completion_exc:
            chosen_core_stats = {**dict(chosen_core_stats or {}), "neutral_volume_cutout_completion_failed": str(_neutral_completion_exc)}
            core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)

    # v10 Bore identity rewrite:
    # A BORE is not defined by a connected triangle island or by a minimum face
    # count.  Those are mesh-density/topology symptoms.  The feature identity is
    # a cylindrical negative field anchored by circular/coaxial opening evidence
    # and supported by curvature-compatible wall/perimeter evidence.  Wall
    # continuity and watertightness belong to rebuild quality and validation,
    # not to BORE identity.
    bore_identity_hypothesis_used = False
    bore_identity_hypothesis_ok = False
    bore_identity_hypothesis_diagnostics: dict[str, object] = {}
    bore_identity_hypothesis_face_ids: tuple[int, ...] = ()
    bore_identity_hypothesis_radius = float(measured_core_radius)
    bore_identity_hypothesis_axis = axis_vec
    bore_identity_hypothesis_center = center_vec
    bore_identity_hypothesis_axial_span = 0.0
    bore_identity_hypothesis_rule = "not_run"
    bore_identity_wall_role_face_ids: tuple[int, ...] = ()
    bore_identity_transition_role_face_ids: tuple[int, ...] = ()
    bore_identity_unowned_support_face_ids: tuple[int, ...] = ()
    bore_identity_role_diagnostics: dict[str, object] = {}

    def _angular_coverage_for_ids(ids: Iterable[int], *, center: object, axis: object) -> dict[str, object]:
        """Return density-insensitive angular/perimeter support diagnostics."""

        ids_t = tuple(fid for fid in tuple_ints(ids) if fid in fid_to_local)
        if not ids_t:
            return {"support_face_count": 0, "occupied_bin_count": 0, "angular_coverage": 0.0, "bin_count": 32}
        try:
            hyp_center = np.asarray(center, dtype=float).reshape(3)
            _a, u, v = _orthonormal_basis_from_axis(axis)
            idx = np.asarray([fid_to_local[fid] for fid in ids_t], dtype=np.int64)
            rel_cov = local_pts[idx, :3] - hyp_center.reshape(1, 3)
            x = rel_cov @ u.reshape(3)
            y = rel_cov @ v.reshape(3)
            angles = np.arctan2(y, x)
            finite_angles = angles[np.isfinite(angles)]
            if finite_angles.size == 0:
                return {"support_face_count": int(len(ids_t)), "occupied_bin_count": 0, "angular_coverage": 0.0, "bin_count": 32}
            bins = np.floor(((finite_angles + np.pi) / (2.0 * np.pi)) * 32.0).astype(int)
            bins = np.clip(bins, 0, 31)
            occupied = int(len(set(int(v) for v in bins.tolist())))
            return {
                "support_face_count": int(len(ids_t)),
                "occupied_bin_count": int(occupied),
                "angular_coverage": float(occupied / 32.0),
                "bin_count": 32,
            }
        except Exception as exc:
            return {"support_face_count": int(len(ids_t)), "occupied_bin_count": 0, "angular_coverage": 0.0, "bin_count": 32, "error": str(exc)}

    def _bore_identity_from_opening_field() -> tuple[bool, tuple[int, ...], dict[str, object]]:
        """Build a geometry-first BORE identity hypothesis.

        This replaces the old implicit definition "connected cylindrical wall
        component with enough faces".  Face IDs returned here are display/support
        evidence inside the cylindrical field; they are not the identity gate.
        """

        loops_raw: list[dict[str, object]] = []
        for i, loop in enumerate(tuple(boundary_loop_geometry or ())):
            try:
                r = float(getattr(loop, "radius", 0.0) or 0.0)
                edge_count = int(getattr(loop, "edge_count", 0) or 0)
                vertex_count = int(getattr(loop, "vertex_count", 0) or 0)
                if r <= 1.0e-9:
                    continue
                if max(edge_count, vertex_count) < 5:
                    continue
                loops_raw.append(
                    {
                        "index": int(getattr(loop, "index", i)),
                        "edge_count": int(edge_count),
                        "vertex_count": int(vertex_count),
                        "center": np.asarray(getattr(loop, "center", center_vec), dtype=float).reshape(3),
                        "axis": canonical_axis(getattr(loop, "axis", axis_vec)),
                        "axial_position": float(getattr(loop, "axial_position", 0.0) or 0.0),
                        "radius": float(r),
                        "radius_rel_mad": float(getattr(loop, "radius_rel_mad", 0.0) or 0.0),
                        "plane_rms": float(getattr(loop, "plane_rms", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue

        # Synthetic selected opening evidence.  This is a fallback for cases
        # where RegionData has a measured frame but the derived boundary-loop
        # ledger is incomplete.
        if selected_radius_valid and not loops_raw:
            loops_raw.append(
                {
                    "index": -1,
                    "edge_count": int(len(tuple(boundary_loops[0])) if boundary_loops else 0),
                    "vertex_count": 0,
                    "center": np.asarray(center_vec, dtype=float).reshape(3),
                    "axis": np.asarray(axis_vec, dtype=float).reshape(3),
                    "axial_position": 0.0,
                    "radius": float(selected_frame_radius),
                    "radius_rel_mad": 0.0,
                    "plane_rms": 0.0,
                    "synthetic_selected_frame": True,
                }
            )

        if not loops_raw:
            return False, (), {"used": False, "reason": "no_opening_loop_evidence"}

        pair_candidates: list[tuple[float, dict[str, object], dict[str, object], dict[str, object]]] = []
        for ia, a_loop in enumerate(loops_raw):
            ra = float(a_loop["radius"])
            ca = np.asarray(a_loop["center"], dtype=float).reshape(3)
            for b_loop in loops_raw[ia + 1 :]:
                rb = float(b_loop["radius"])
                cb = np.asarray(b_loop["center"], dtype=float).reshape(3)
                r_big = max(ra, rb, 1.0e-9)
                r_small = min(ra, rb)
                radius_ratio = float(r_small / r_big)
                radius_delta_rel = float(abs(ra - rb) / r_big)
                delta = cb - ca
                axial_sep_region = abs(float(delta @ axis_vec.reshape(3)))
                cross_sep_region = float(np.linalg.norm(delta - axis_vec.reshape(3) * float(delta @ axis_vec.reshape(3))))
                avg_radius = 0.5 * (ra + rb)
                if radius_ratio < 0.52:
                    continue
                if axial_sep_region < max(0.12 * avg_radius, 0.04 * max(axial_span_all, 1.0), 0.05):
                    continue
                if cross_sep_region > max(1.10, 0.62 * max(avg_radius, 1.0)):
                    continue
                selected_delta_rel = abs(avg_radius - selected_frame_radius) / max(avg_radius, selected_frame_radius, 1.0e-9) if selected_radius_valid else 0.0
                loop_size = min(int(a_loop.get("edge_count", 0) or a_loop.get("vertex_count", 0) or 0), int(b_loop.get("edge_count", 0) or b_loop.get("vertex_count", 0) or 0))
                loop_size_score = min(float(loop_size) / 32.0, 1.0)
                score = (
                    3.0 * radius_ratio
                    + 2.2 * min(axial_sep_region / max(avg_radius, 1.0e-9), 12.0) / 12.0
                    + 0.8 * loop_size_score
                    - 1.8 * min(cross_sep_region / max(1.10, 0.62 * max(avg_radius, 1.0)), 1.0)
                    - 0.9 * min(selected_delta_rel, 1.0)
                )
                pair_candidates.append(
                    (
                        float(score),
                        a_loop,
                        b_loop,
                        {
                            "radius_ratio": float(radius_ratio),
                            "radius_delta_rel": float(radius_delta_rel),
                            "axial_separation": float(axial_sep_region),
                            "center_cross_distance": float(cross_sep_region),
                            "selected_radius_delta_rel": float(selected_delta_rel),
                            "loop_size_score": float(loop_size_score),
                            "avg_radius": float(avg_radius),
                        },
                    )
                )

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        through_pair = pair_candidates[0] if pair_candidates else None

        if through_pair is not None:
            score, loop_a, loop_b, pair_diag = through_pair
            ca = np.asarray(loop_a["center"], dtype=float).reshape(3)
            cb = np.asarray(loop_b["center"], dtype=float).reshape(3)
            delta = cb - ca
            delta_len = float(np.linalg.norm(delta))
            hyp_axis = canonical_axis(delta if delta_len > 1.0e-9 else axis_vec)
            hyp_center = (ca + cb) * 0.5
            radius_h = float(pair_diag["avg_radius"])
            t_a = float((ca - hyp_center) @ hyp_axis.reshape(3))
            t_b = float((cb - hyp_center) @ hyp_axis.reshape(3))
            axial_min_h = min(t_a, t_b)
            axial_max_h = max(t_a, t_b)
            axial_span_h = max(abs(axial_max_h - axial_min_h), 1.0e-9)
            classification = "through_bore"
            identity_reason = "two_coaxial_compatible_opening_loops_define_bore_identity"
        else:
            # Blind/single-opening fallback: identity is weaker and requires
            # curvature support, but still uses perimeter-field evidence rather
            # than a connected face-component count.
            loop_a = max(loops_raw, key=lambda item: (float(item.get("radius", 0.0)), int(item.get("edge_count", 0) or 0)))
            hyp_axis = axis_vec
            hyp_center = np.asarray(loop_a["center"], dtype=float).reshape(3)
            radius_h = float(loop_a["radius"])
            ax_finite = axial[finite]
            axial_min_h = float(np.min(ax_finite)) if ax_finite.size else -max(radius_h, 1.0)
            axial_max_h = float(np.max(ax_finite)) if ax_finite.size else max(radius_h, 1.0)
            axial_span_h = max(abs(axial_max_h - axial_min_h), 1.0e-9)
            classification = "single_opening_bore_or_circular_pocket_review"
            identity_reason = "single_opening_plus_curvature_field_support"
            score = 0.0
            pair_diag = {"pair_candidate_count": 0}

        axial_pad_h = max(0.35, 0.08 * max(axial_span_h, 1.0), 0.05 * max(radius_h, 1.0))
        radial_tol_h = max(0.42, 0.16 * max(radius_h, 1.0))
        rel_h = local_pts - hyp_center.reshape(1, 3)
        axial_h = rel_h @ hyp_axis.reshape(3)
        radial_vec_h = rel_h - axial_h.reshape(-1, 1) * hyp_axis.reshape(1, 3)
        radial_h = np.linalg.norm(radial_vec_h, axis=1)
        field_mask = (
            finite
            & (axial_h >= axial_min_h - axial_pad_h)
            & (axial_h <= axial_max_h + axial_pad_h)
            & (np.abs(radial_h - radius_h) <= radial_tol_h)
        )
        field_ids = tuple(sorted(set(ids_from_mask(field_mask)) | seed_face_set))
        coverage = _angular_coverage_for_ids(field_ids, center=hyp_center, axis=hyp_axis)
        # Normals are quality evidence only.  They are intentionally not an
        # identity gate because imported meshes may contain fragmented, sparse,
        # or inconsistent surface evidence.
        if field_ids:
            idx = np.asarray([fid_to_local[fid] for fid in field_ids if fid in fid_to_local], dtype=np.int64)
            normal_axis_median = float(np.median(normal_axis_abs[idx])) if idx.size else 1.0
            radial_align_median = float(np.median(radial_normal_alignment[idx])) if idx.size else 0.0
        else:
            normal_axis_median = 1.0
            radial_align_median = 0.0

        through_identity = bool(through_pair is not None)
        single_opening_identity = bool(
            through_pair is None
            and selected_radius_valid
            and bool(field_ids)
            and float(coverage.get("angular_coverage", 0.0) or 0.0) >= 0.18
        )
        identity_ok = bool(through_identity or single_opening_identity)
        if not field_ids:
            # CandidateData still needs something displayable.  Do not use this
            # to accept/reject identity; it only provides a visible anchor row.
            field_ids = tuple(sorted(seed_face_set))
        if not field_ids and identity_ok:
            field_ids = tuple(valid_face_ids[: min(len(valid_face_ids), 64)])

        diagnostics = {
            "used": bool(identity_ok),
            "identity_ok": bool(identity_ok),
            "identity_reason": str(identity_reason if identity_ok else "opening_field_hypothesis_not_strong_enough"),
            "classification": str(classification),
            "pair_candidate_count": int(len(pair_candidates)),
            "selected_pair_score": float(score),
            "selected_pair": (
                int(loop_a.get("index", -1)),
                int(loop_b.get("index", -1)) if through_pair is not None else None,
            ),
            "selected_pair_metrics": dict(pair_diag),
            "loop_count_considered": int(len(loops_raw)),
            "loops_considered_sample": tuple(
                {
                    "index": int(item.get("index", -1)),
                    "edge_count": int(item.get("edge_count", 0) or 0),
                    "vertex_count": int(item.get("vertex_count", 0) or 0),
                    "radius": float(item.get("radius", 0.0) or 0.0),
                    "radius_rel_mad": float(item.get("radius_rel_mad", 0.0) or 0.0),
                }
                for item in loops_raw[:12]
            ),
            "center": to_vector3(hyp_center),
            "axis": to_vector3(hyp_axis),
            "radius": float(radius_h),
            "diameter": float(2.0 * radius_h),
            "axial_min": float(axial_min_h),
            "axial_max": float(axial_max_h),
            "axial_span": float(axial_span_h),
            "radial_tolerance": float(radial_tol_h),
            "axial_padding": float(axial_pad_h),
            "field_support_face_count": int(len(field_ids)),
            "field_support_face_ids_sample": tuple(int(fid) for fid in field_ids[:64]),
            "angular_coverage": coverage,
            "normal_axis_abs_median_quality_only": float(normal_axis_median),
            "radial_normal_alignment_median_quality_only": float(radial_align_median),
            "identity_policy": "opening_and_curvature_field_define_bore_face_count_and_wall_continuity_are_quality_not_identity",
            "wall_continuity_policy": "fragmented_or_continuous_wall_is_rebuild_condition_not_recognition_condition",
        }
        return bool(identity_ok), tuple(sorted(field_ids)), diagnostics

    def _classify_bore_identity_surface_roles(
        identity_face_ids: Iterable[int],
        identity_diagnostics: Mapping[str, object],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, object]]:
        """Split BORE identity support from physical surface ownership.

        v10 correctly allowed opening/radius/axis/curvature-field evidence to
        define BORE identity, but it then reused the whole support field as the
        BORE-owned patch.  That collapsed annular transition surfaces into the
        bore candidate and prevented CHAMFER_FORM from being emitted.

        v11 keeps the broad support field as evidence only and assigns actual
        candidate ownership by surface role:

            wall-role faces       -> BORE candidate ownership
            transition-role faces -> CHAMFER_FORM may own them later
            unowned support       -> diagnostics only

        This helper does not classify damage.  It only separates geometric face
        roles around the same opening-field hypothesis.
        """

        support_ids = tuple(fid for fid in tuple_ints(identity_face_ids) if fid in fid_to_local)
        support_set = set(int(fid) for fid in support_ids)
        if not support_set:
            return (), (), (), {
                "used": False,
                "reason": "no_identity_support_faces",
                "policy": "identity_support_is_evidence_not_bore_ownership",
            }

        try:
            hyp_center = np.asarray(identity_diagnostics.get("center", center_vec), dtype=float).reshape(3)
        except Exception:
            hyp_center = np.asarray(center_vec, dtype=float).reshape(3)
        try:
            hyp_axis = canonical_axis(identity_diagnostics.get("axis", axis_vec))
        except Exception:
            hyp_axis = np.asarray(axis_vec, dtype=float).reshape(3)
        try:
            hyp_radius = float(identity_diagnostics.get("radius", measured_core_radius) or measured_core_radius)
        except Exception:
            hyp_radius = float(measured_core_radius)
        try:
            hyp_radial_tol = float(identity_diagnostics.get("radial_tolerance", max(0.42, 0.16 * max(abs(hyp_radius), 1.0))) or 0.0)
        except Exception:
            hyp_radial_tol = max(0.42, 0.16 * max(abs(hyp_radius), 1.0))
        hyp_radial_tol = max(float(hyp_radial_tol), 1.0e-9)

        rel_role = local_pts - hyp_center.reshape(1, 3)
        axial_role = rel_role @ hyp_axis.reshape(3)
        radial_vec_role = rel_role - axial_role.reshape(-1, 1) * hyp_axis.reshape(1, 3)
        radial_role = np.linalg.norm(radial_vec_role, axis=1)
        radial_dir_role = np.zeros_like(radial_vec_role)
        ok_role_radial = radial_role > 1.0e-12
        radial_dir_role[ok_role_radial] = radial_vec_role[ok_role_radial] / radial_role[ok_role_radial].reshape(-1, 1)
        normal_axis_role = np.abs(unit_normals @ hyp_axis.reshape(3))
        radial_align_role = np.abs(np.sum(unit_normals * radial_dir_role, axis=1))
        finite_role = (
            np.isfinite(axial_role)
            & np.isfinite(radial_role)
            & np.isfinite(normal_axis_role)
            & np.isfinite(radial_align_role)
        )
        support_mask = np.asarray([int(fid) in support_set for fid in valid_face_ids], dtype=bool)

        # Wall role: faces near the cylindrical radius whose normals behave like
        # a cylindrical side wall.  This is ownership, not identity.  It may be
        # sparse on poor meshes, but it must not absorb annular transition faces.
        #
        # v13 selected-ring role disambiguation:
        # The user may select the inner bore rim, the outer chamfer mouth, a step
        # ring, or an intermediate transition ring.  Therefore the selected
        # radius is only an entry point into a radius-layer stack.  It is not
        # automatically the bore-wall radius.
        #
        # Recognition first builds a local radius-layer graph from measured
        # loops compatible with the current centerline.  Then it chooses a
        # wall-radius layer from cylindrical side-wall evidence.  Larger coaxial
        # layers remain transition/mouth/radius-stack evidence and must stay
        # available to CHAMFER_FORM.  Broad identity support is never allowed to
        # become BORE-owned wall by itself.
        radius_layer_items: list[dict[str, object]] = []
        for li, loop in enumerate(tuple(boundary_loop_geometry or ())):
            try:
                lr = float(getattr(loop, "radius", 0.0) or 0.0)
                if lr <= 1.0e-9:
                    continue
                le = int(getattr(loop, "edge_count", 0) or 0)
                lv = int(getattr(loop, "vertex_count", 0) or le)
                if max(le, lv) < 5:
                    continue
                lc = np.asarray(getattr(loop, "center", center_vec), dtype=float).reshape(3)
                delta_c = lc - hyp_center.reshape(3)
                la = float(delta_c @ hyp_axis.reshape(3))
                cross_c = float(np.linalg.norm(delta_c - hyp_axis.reshape(3) * la))
                # Keep only loops that belong to the same local centerline stack.
                # This prevents nearby screw holes or unrelated cutout boundary
                # loops from becoming the "smallest radius" layer of the bore.
                centerline_tol = max(1.25, 0.22 * max(abs(lr), abs(hyp_radius), abs(selected_frame_radius or 0.0), 1.0))
                if cross_c > centerline_tol:
                    continue
                radius_layer_items.append(
                    {
                        "index": int(getattr(loop, "index", li)),
                        "radius": float(lr),
                        "axial_position": float(la),
                        "center_cross_distance": float(cross_c),
                        "centerline_tolerance": float(centerline_tol),
                        "edge_count": int(le),
                        "vertex_count": int(lv),
                        "radius_rel_mad": float(getattr(loop, "radius_rel_mad", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue

        radius_layers: list[dict[str, object]] = []
        for item in sorted(radius_layer_items, key=lambda v: float(v["radius"])):
            r = float(item["radius"])
            placed = False
            for layer in radius_layers:
                ref = float(layer["radius_median"])
                tol = max(0.20, 0.060 * max(abs(ref), abs(r), 1.0))
                if abs(r - ref) <= tol:
                    members = list(layer["members"])
                    members.append(item)
                    rs = [float(m["radius"]) for m in members]
                    ax = [float(m["axial_position"]) for m in members]
                    cross_values = [float(m.get("center_cross_distance", 0.0) or 0.0) for m in members]
                    layer["members"] = tuple(members)
                    layer["radius_median"] = float(np.median(rs))
                    layer["radius_min"] = float(min(rs))
                    layer["radius_max"] = float(max(rs))
                    layer["axial_positions"] = tuple(float(v) for v in ax)
                    layer["center_cross_median"] = float(np.median(cross_values)) if cross_values else 0.0
                    layer["loop_count"] = int(len(members))
                    placed = True
                    break
            if not placed:
                radius_layers.append(
                    {
                        "radius_median": float(r),
                        "radius_min": float(r),
                        "radius_max": float(r),
                        "axial_positions": (float(item["axial_position"]),),
                        "center_cross_median": float(item.get("center_cross_distance", 0.0) or 0.0),
                        "loop_count": 1,
                        "members": (item,),
                    }
                )

        hyp_radius_from_identity = float(hyp_radius)
        try:
            axial_min_role = float(identity_diagnostics.get("axial_min", np.min(axial_role[finite_role])) or 0.0)
            axial_max_role = float(identity_diagnostics.get("axial_max", np.max(axial_role[finite_role])) or 0.0)
        except Exception:
            axial_min_role = float(np.min(axial_role[finite_role])) if np.any(finite_role) else -max(abs(hyp_radius_from_identity), 1.0)
            axial_max_role = float(np.max(axial_role[finite_role])) if np.any(finite_role) else max(abs(hyp_radius_from_identity), 1.0)
        axial_span_role = max(abs(float(axial_max_role - axial_min_role)), 1.0e-9)
        axial_pad_role = max(0.35, 0.10 * max(axial_span_role, 1.0), 0.05 * max(abs(hyp_radius_from_identity), 1.0))
        axial_stack_mask = finite_role & (axial_role >= axial_min_role - axial_pad_role) & (axial_role <= axial_max_role + axial_pad_role)

        def _role_ids_from_local_mask(mask: np.ndarray) -> tuple[int, ...]:
            return ids_from_mask(mask)

        def _role_angular_coverage(mask_ids: Iterable[int]) -> float:
            return float(_angular_coverage_for_ids(mask_ids, center=hyp_center, axis=hyp_axis).get("angular_coverage", 0.0) or 0.0)

        # Score every radius layer as a possible cylindrical wall layer.  This
        # makes the actual wall radius a derived layer in the stack, not a direct
        # copy of the selected ring radius.
        layer_role_stats: list[dict[str, object]] = []
        for layer_index, layer in enumerate(sorted(radius_layers, key=lambda v: float(v["radius_median"]))):
            lr = float(layer["radius_median"])
            layer_tol = max(0.18, 0.055 * max(abs(lr), 1.0))
            layer_band = axial_stack_mask & (np.abs(radial_role - lr) <= layer_tol)
            side_mask = layer_band & (normal_axis_role <= 0.58) & (radial_align_role >= 0.10)
            strict_side_mask = layer_band & (normal_axis_role <= 0.46) & (radial_align_role >= 0.16)
            transitionish_mask = layer_band & (normal_axis_role >= 0.06) & (normal_axis_role <= 0.995) & (radial_align_role >= 0.006)
            layer_ids = _role_ids_from_local_mask(layer_band)
            side_ids = _role_ids_from_local_mask(side_mask)
            strict_side_ids = _role_ids_from_local_mask(strict_side_mask)
            transitionish_ids = _role_ids_from_local_mask(transitionish_mask)
            if layer_ids:
                idx = np.asarray([fid_to_local[fid] for fid in layer_ids if fid in fid_to_local], dtype=np.int64)
                na_med = float(np.median(normal_axis_role[idx])) if idx.size else 1.0
                ra_med = float(np.median(radial_align_role[idx])) if idx.size else 0.0
                ax_values = axial_role[idx] if idx.size else np.asarray((), dtype=float)
                ax_span = float(np.max(ax_values) - np.min(ax_values)) if ax_values.size else 0.0
            else:
                na_med = 1.0
                ra_med = 0.0
                ax_span = 0.0
            layer_coverage = _role_angular_coverage(layer_ids)
            side_coverage = _role_angular_coverage(side_ids)
            strict_side_coverage = _role_angular_coverage(strict_side_ids)
            selected_delta_rel = abs(lr - float(selected_frame_radius or hyp_radius_from_identity)) / max(abs(lr), abs(float(selected_frame_radius or 0.0)), 1.0e-9) if selected_radius_valid else 0.0
            radius_delta_from_identity = abs(lr - hyp_radius_from_identity) / max(abs(lr), abs(hyp_radius_from_identity), 1.0e-9)
            side_fraction = float(len(side_ids) / max(len(layer_ids), 1)) if layer_ids else 0.0
            strict_fraction = float(len(strict_side_ids) / max(len(layer_ids), 1)) if layer_ids else 0.0
            # Prefer layer radii with cylindrical-normal evidence and axial
            # continuation.  Avoid treating the largest selected mouth ring as
            # the wall simply because it is the selected ring.
            wall_score = (
                2.20 * side_coverage
                + 1.20 * strict_side_coverage
                + 0.90 * min(ax_span / max(axial_span_role, 1.0e-9), 1.0)
                + 0.65 * min(float(layer.get("loop_count", 1)) / 2.0, 1.0)
                + 0.55 * side_fraction
                + 0.35 * strict_fraction
                - 0.35 * min(radius_delta_from_identity, 1.0)
            )
            layer_role_stats.append(
                {
                    "layer_index": int(layer_index),
                    "radius": float(lr),
                    "tolerance": float(layer_tol),
                    "loop_count": int(layer.get("loop_count", 1)),
                    "axial_positions": tuple(float(v) for v in tuple(layer.get("axial_positions", ()) or ())),
                    "center_cross_median": float(layer.get("center_cross_median", 0.0) or 0.0),
                    "layer_face_count": int(len(layer_ids)),
                    "side_face_count": int(len(side_ids)),
                    "strict_side_face_count": int(len(strict_side_ids)),
                    "transitionish_face_count": int(len(transitionish_ids)),
                    "normal_axis_median": float(na_med),
                    "radial_alignment_median": float(ra_med),
                    "axial_support_span": float(ax_span),
                    "angular_coverage": float(layer_coverage),
                    "side_angular_coverage": float(side_coverage),
                    "strict_side_angular_coverage": float(strict_side_coverage),
                    "side_fraction": float(side_fraction),
                    "strict_side_fraction": float(strict_fraction),
                    "selected_radius_delta_rel": float(selected_delta_rel),
                    "identity_radius_delta_rel": float(radius_delta_from_identity),
                    "wall_score": float(wall_score),
                }
            )

        # v15 two-sided internal radius scan:
        # Boundary-loop geometry tells us about the visible cutout boundary, but
        # it does not necessarily contain the internal feature rings that matter
        # for a selected outer chamfer rim.  A clean single-sided chamfer+bore can
        # therefore have a correct RegionData volume while the bore wall is only
        # found as a few fragments.  Recognition must scan both axial sides of
        # the selected/opening plane for cylindrical radius layers and treat the
        # selected ring as a neutral stack anchor, not as the bore radius.
        internal_two_sided_layer_stats: list[dict[str, object]] = []

        def _append_internal_two_sided_radius_layers() -> None:
            """Add face-derived radius layers from both axial sides.

            These rows are still evidence.  They do not create candidates by
            themselves.  They give the wall-radius decision a chance to choose
            the actual inner cylindrical bore radius even when no internal ring
            appears in ``boundary_loop_geometry``.
            """

            if not np.any(finite_role):
                return
            radial_reference = max(abs(float(hyp_radius_from_identity)), abs(float(selected_frame_radius or 0.0)), 1.0)
            side_deadband = max(0.06, 0.018 * radial_reference)
            # Very loose wall-ish mask.  This is not ownership.  It is only a
            # source for radial-layer evidence.  The final wall-role mask below
            # remains stricter and role-based.
            face_layer_base = (
                axial_stack_mask
                & (radial_role > max(0.02, 0.0025 * radial_reference))
                & (normal_axis_role <= 0.78)
                & (radial_align_role >= 0.018)
            )
            if selected_radius_valid:
                # For selected outer chamfer mouths, the actual bore radius is
                # usually smaller than the selected radius.  Keep all smaller
                # coaxial layers plus a small margin above the selected layer,
                # but do not let the whole RegionData cutout dominate the scan.
                face_layer_base &= radial_role <= (float(selected_frame_radius) + max(0.75, 0.08 * radial_reference))

            for side_sign, side_name in ((1.0, "positive_axis_side"), (-1.0, "negative_axis_side")):
                if side_sign > 0.0:
                    side_mask = face_layer_base & (axial_role >= side_deadband)
                else:
                    side_mask = face_layer_base & (axial_role <= -side_deadband)
                side_indices = np.nonzero(side_mask)[0]
                if side_indices.size < 2:
                    continue
                side_radii = radial_role[side_indices]
                finite_side = np.isfinite(side_radii)
                side_indices = side_indices[finite_side]
                side_radii = side_radii[finite_side]
                if side_indices.size < 2:
                    continue
                order = np.argsort(side_radii)
                sorted_indices = side_indices[order]
                sorted_radii = side_radii[order]

                clusters: list[list[int]] = []
                current: list[int] = [int(sorted_indices[0])]
                current_values: list[float] = [float(sorted_radii[0])]
                for idx_value, radius_value in zip(sorted_indices[1:], sorted_radii[1:]):
                    ref = float(np.median(current_values)) if current_values else float(radius_value)
                    gap_tol = max(0.14, 0.030 * max(abs(ref), abs(float(radius_value)), 1.0))
                    if abs(float(radius_value) - ref) <= gap_tol:
                        current.append(int(idx_value))
                        current_values.append(float(radius_value))
                    else:
                        clusters.append(current)
                        current = [int(idx_value)]
                        current_values = [float(radius_value)]
                if current:
                    clusters.append(current)

                for cluster_index, cluster_local_indices in enumerate(clusters):
                    if not cluster_local_indices:
                        continue
                    cluster_idx = np.asarray(cluster_local_indices, dtype=np.int64)
                    cluster_ids = tuple(sorted(int(local_to_fid[int(i)]) for i in cluster_idx if int(i) in local_to_fid))
                    if not cluster_ids:
                        continue
                    cluster_radii = radial_role[cluster_idx]
                    cluster_axial = axial_role[cluster_idx]
                    cluster_normal_axis = normal_axis_role[cluster_idx]
                    cluster_align = radial_align_role[cluster_idx]
                    radius_med = float(np.median(cluster_radii))
                    radius_min = float(np.min(cluster_radii))
                    radius_max = float(np.max(cluster_radii))
                    radius_span = float(radius_max - radius_min)
                    layer_tol = max(0.16, 0.050 * max(abs(radius_med), 1.0))
                    layer_band = axial_stack_mask & (np.abs(radial_role - radius_med) <= layer_tol)
                    side_band = layer_band & (axial_role >= side_deadband if side_sign > 0.0 else axial_role <= -side_deadband)
                    side_band &= (normal_axis_role <= 0.72) & (radial_align_role >= 0.025)
                    strict_side_band = side_band & (normal_axis_role <= 0.56) & (radial_align_role >= 0.070)
                    side_ids = _role_ids_from_local_mask(side_band)
                    strict_side_ids = _role_ids_from_local_mask(strict_side_band)
                    if not side_ids:
                        continue
                    side_coverage = _role_angular_coverage(side_ids)
                    strict_side_coverage = _role_angular_coverage(strict_side_ids)
                    ax_span = float(np.max(cluster_axial) - np.min(cluster_axial)) if cluster_axial.size else 0.0
                    na_med = float(np.median(cluster_normal_axis)) if cluster_normal_axis.size else 1.0
                    ra_med = float(np.median(cluster_align)) if cluster_align.size else 0.0
                    radial_compactness = 1.0 - min(radius_span / max(layer_tol * 2.5, 1.0e-9), 1.0)
                    selected_delta_rel = abs(radius_med - float(selected_frame_radius or radius_med)) / max(abs(radius_med), abs(float(selected_frame_radius or radius_med)), 1.0e-9) if selected_radius_valid else 0.0
                    smaller_than_selected_bonus = 0.0
                    if selected_radius_valid and radius_med < float(selected_frame_radius) - max(0.20, 0.025 * max(abs(float(selected_frame_radius)), 1.0)):
                        smaller_than_selected_bonus = 0.75
                    wall_score = (
                        2.45 * side_coverage
                        + 1.30 * strict_side_coverage
                        + 0.95 * min(ax_span / max(axial_span_role, 1.0e-9), 1.0)
                        + 0.80 * max(0.0, 1.0 - min(na_med / 0.78, 1.0))
                        + 0.55 * max(0.0, min(ra_med, 1.0))
                        + 0.45 * radial_compactness
                        + smaller_than_selected_bonus
                        - 0.20 * min(selected_delta_rel, 1.0)
                    )
                    row = {
                        "layer_index": int(len(layer_role_stats) + len(internal_two_sided_layer_stats)),
                        "radius": float(radius_med),
                        "tolerance": float(layer_tol),
                        "loop_count": 0,
                        "axial_positions": (float(np.median(cluster_axial)) if cluster_axial.size else 0.0,),
                        "center_cross_median": 0.0,
                        "layer_face_count": int(len(cluster_ids)),
                        "side_face_count": int(len(side_ids)),
                        "strict_side_face_count": int(len(strict_side_ids)),
                        "transitionish_face_count": 0,
                        "normal_axis_median": float(na_med),
                        "radial_alignment_median": float(ra_med),
                        "axial_support_span": float(ax_span),
                        "angular_coverage": float(side_coverage),
                        "side_angular_coverage": float(side_coverage),
                        "strict_side_angular_coverage": float(strict_side_coverage),
                        "side_fraction": 1.0,
                        "strict_side_fraction": float(len(strict_side_ids) / max(len(side_ids), 1)),
                        "selected_radius_delta_rel": float(selected_delta_rel),
                        "identity_radius_delta_rel": float(abs(radius_med - hyp_radius_from_identity) / max(abs(radius_med), abs(hyp_radius_from_identity), 1.0e-9)),
                        "wall_score": float(wall_score),
                        "source": "internal_two_sided_face_radius_scan",
                        "side": str(side_name),
                        "side_sign": float(side_sign),
                        "cluster_index": int(cluster_index),
                        "radius_min": float(radius_min),
                        "radius_max": float(radius_max),
                        "radius_span": float(radius_span),
                        "radial_compactness": float(radial_compactness),
                        "smaller_than_selected_bonus": float(smaller_than_selected_bonus),
                    }
                    # Avoid adding pure noise rows.  This is an evidence quality
                    # filter, not a feature-identity face-count gate.
                    if float(row["side_angular_coverage"]) >= 0.030 or int(row["strict_side_face_count"]) >= 2:
                        internal_two_sided_layer_stats.append(row)

        _append_internal_two_sided_radius_layers()
        if internal_two_sided_layer_stats:
            layer_role_stats.extend(internal_two_sided_layer_stats)

        wall_radius = hyp_radius_from_identity
        wall_layer_decision = "identity_radius_no_radius_layer_graph"
        if layer_role_stats:
            sorted_stats = sorted(layer_role_stats, key=lambda v: float(v["radius"]))
            plausible = [
                st_layer for st_layer in sorted_stats
                if (
                    float(st_layer.get("side_angular_coverage", 0.0) or 0.0) >= 0.020
                    or int(st_layer.get("strict_side_face_count", 0) or 0) > 0
                    or (float(st_layer.get("normal_axis_median", 1.0) or 1.0) <= 0.62 and float(st_layer.get("radial_alignment_median", 0.0) or 0.0) >= 0.08)
                )
            ]
            selected_radius_value = float(selected_frame_radius or hyp_radius_from_identity)
            # If the selected ring is a larger layer and smaller plausible layers
            # exist, prefer those smaller layers as the bore wall.  This is the
            # chamfer-mouth case shown in runtime testing.
            lower_than_selected = [
                st_layer for st_layer in plausible
                if float(st_layer["radius"]) < selected_radius_value - max(0.22, 0.040 * max(abs(selected_radius_value), 1.0))
            ] if selected_radius_valid else []
            if lower_than_selected:
                chosen_layer = max(lower_than_selected, key=lambda v: float(v.get("wall_score", 0.0)))
                wall_layer_decision = "selected_ring_is_outer_or_transition_layer_used_smaller_cylindrical_wall_layer"
            elif plausible:
                # Prefer strongest cylindrical layer, but bias away from the
                # largest layer when there is a near-tie with a smaller radius.
                best_score = max(float(v.get("wall_score", 0.0)) for v in plausible)
                near_best = [v for v in plausible if float(v.get("wall_score", 0.0)) >= best_score * 0.82]
                chosen_layer = min(near_best, key=lambda v: float(v["radius"])) if near_best else max(plausible, key=lambda v: float(v.get("wall_score", 0.0)))
                wall_layer_decision = "best_cylindrical_radius_layer"
            else:
                chosen_layer = min(sorted_stats, key=lambda v: abs(float(v["radius"]) - hyp_radius_from_identity))
                wall_layer_decision = "fallback_closest_radius_layer_no_strong_cylindrical_stats"
            wall_radius = float(chosen_layer["radius"])

        all_layer_radii = [float(layer["radius_median"]) for layer in radius_layers]
        outer_radius = max(all_layer_radii + [float(hyp_radius_from_identity), float(selected_frame_radius or 0.0), float(wall_radius)])
        inner_radius = min(all_layer_radii + [float(wall_radius)]) if all_layer_radii else float(wall_radius)
        ring_layer_separation = float(max(0.0, outer_radius - wall_radius))

        wall_radial_tol = min(
            max(0.18, 0.055 * max(abs(wall_radius), 1.0)),
            max(0.30, 0.42 * hyp_radial_tol),
        )
        stack_low = min(float(inner_radius), float(wall_radius)) - max(0.18, 0.65 * wall_radial_tol)
        stack_high = max(float(outer_radius), float(hyp_radius_from_identity), float(selected_frame_radius or 0.0)) + max(0.22, 0.050 * max(abs(outer_radius), 1.0))
        role_stack_mask = axial_stack_mask & (radial_role >= stack_low) & (radial_role <= stack_high)

        # Make ownership stricter than identity.  Identity may survive sparse or
        # noisy normals; ownership should not swallow annular/frustum faces.
        wall_role_mask = (
            role_stack_mask
            & (np.abs(radial_role - wall_radius) <= wall_radial_tol)
            & (normal_axis_role <= 0.52)
            & (radial_align_role >= 0.10)
        )
        strict_wall_role_mask = wall_role_mask & (normal_axis_role <= 0.46) & (radial_align_role >= 0.16)
        wall_role_count = int(np.count_nonzero(wall_role_mask))
        strict_wall_role_count = int(np.count_nonzero(strict_wall_role_mask))
        if strict_wall_role_count > 0:
            wall_ids_loose_preview = ids_from_mask(wall_role_mask)
            wall_ids_strict_preview = ids_from_mask(strict_wall_role_mask)
            loose_coverage = _role_angular_coverage(wall_ids_loose_preview)
            strict_coverage = _role_angular_coverage(wall_ids_strict_preview)
            strict_preserves_role = bool(
                strict_wall_role_count >= max(8, int(0.35 * max(wall_role_count, 1)))
                or strict_coverage >= max(0.030, 0.50 * loose_coverage)
            )
            if strict_preserves_role:
                wall_role_mask = strict_wall_role_mask

        transition_inner = wall_radius + max(0.08, 0.35 * wall_radial_tol)
        transition_outer = stack_high
        # Transition role is deliberately ring/layer based, not "whatever is
        # left over".  It describes annular/frustum/radius-stack support between
        # the bore wall radius and larger mouth/radius layers.  It uses the
        # radius-layer stack, not the selected-radius identity mask, so selecting
        # an outer chamfer mouth can still emit a separate CHAMFER_FORM.
        transition_role_mask = (
            role_stack_mask
            & (~wall_role_mask)
            & (radial_role >= transition_inner)
            & (radial_role <= transition_outer)
            & (normal_axis_role >= 0.040)
            & (normal_axis_role <= 0.995)
            & (radial_align_role >= 0.004)
        )

        wall_ids = ids_from_mask(wall_role_mask)
        transition_ids = ids_from_mask(transition_role_mask)
        role_support_ids = ids_from_mask(role_stack_mask)
        # Keep identity_support separately: it proves the bore.  role_support is
        # the local stack area considered for ownership partition.
        unowned_ids = tuple(sorted(set(role_support_ids) - set(wall_ids) - set(transition_ids)))
        diagnostics = {
            "used": True,
            "policy": "bore_identity_support_is_evidence_wall_role_faces_are_bore_ownership",
            "identity_support_face_count": int(len(support_set)),
            "role_support_face_count": int(len(role_support_ids)),
            "wall_role_face_count": int(len(wall_ids)),
            "transition_role_face_count": int(len(transition_ids)),
            "unowned_support_face_count": int(len(unowned_ids)),
            "identity_support_face_ids_sample": tuple(int(fid) for fid in sorted(support_set)[:64]),
            "role_support_face_ids_sample": tuple(int(fid) for fid in role_support_ids[:64]),
            "wall_role_face_ids_sample": tuple(int(fid) for fid in wall_ids[:64]),
            "transition_role_face_ids_sample": tuple(int(fid) for fid in transition_ids[:64]),
            "wall_layer_decision": str(wall_layer_decision),
            "wall_radial_tolerance": float(wall_radial_tol),
            "identity_radial_tolerance": float(hyp_radial_tol),
            "hypothesis_radius": float(hyp_radius),
            "selected_frame_radius": float(selected_frame_radius or 0.0),
            "wall_role_radius": float(wall_radius),
            "inner_radius_layer": float(inner_radius),
            "outer_radius_layer": float(outer_radius),
            "ring_layer_separation": float(ring_layer_separation),
            "role_stack_radial_low": float(stack_low),
            "role_stack_radial_high": float(stack_high),
            "role_stack_axial_min": float(axial_min_role - axial_pad_role),
            "role_stack_axial_max": float(axial_max_role + axial_pad_role),
            "radius_layer_count": int(len(radius_layers)),
            "radius_layer_role_stats": tuple(layer_role_stats),
            "internal_two_sided_radius_scan_count": int(len(internal_two_sided_layer_stats)),
            "internal_two_sided_radius_scan": tuple(internal_two_sided_layer_stats[:16]),
            "recognition_side_scan_policy": "selected_ring_is_neutral_anchor_scan_positive_and_negative_axis_sides_for_inner_wall_radius",
            "radius_layers": tuple(
                {
                    "radius_median": float(layer["radius_median"]),
                    "radius_min": float(layer["radius_min"]),
                    "radius_max": float(layer["radius_max"]),
                    "loop_count": int(layer["loop_count"]),
                    "axial_positions": tuple(float(v) for v in tuple(layer["axial_positions"])),
                }
                for layer in radius_layers
            ),
            "role_center": to_vector3(hyp_center),
            "role_axis": to_vector3(hyp_axis),
            "wall_role_definition": "radius_layer_wall_radius_plus_cylindrical_side_wall_normals",
            "transition_role_definition": "faces_between_wall_radius_and_outer_radius_layers_left_for_chamfer_or_stack_ownership",
            "damage_policy": "not_classified_in_recognition_rebuild_quality_only",
        }
        return tuple(wall_ids), tuple(transition_ids), tuple(unowned_ids), diagnostics

    bore_hyp_ok, bore_hyp_ids, bore_hyp_diag = _bore_identity_from_opening_field()
    bore_identity_hypothesis_diagnostics = dict(bore_hyp_diag)
    if bore_hyp_ok:
        bore_identity_hypothesis_used = True
        bore_identity_hypothesis_ok = True
        bore_identity_hypothesis_face_ids = tuple(sorted(bore_hyp_ids))
        bore_identity_hypothesis_radius = float(bore_hyp_diag.get("radius", measured_core_radius) or measured_core_radius)
        bore_identity_hypothesis_axis = canonical_axis(bore_hyp_diag.get("axis", axis_vec))
        bore_identity_hypothesis_center = np.asarray(bore_hyp_diag.get("center", center_vec), dtype=float).reshape(3)
        bore_identity_hypothesis_axial_span = float(bore_hyp_diag.get("axial_span", 0.0) or 0.0)
        bore_identity_hypothesis_rule = str(bore_hyp_diag.get("identity_reason", "opening_field_bore_identity"))
        (
            bore_identity_wall_role_face_ids,
            bore_identity_transition_role_face_ids,
            bore_identity_unowned_support_face_ids,
            bore_identity_role_diagnostics,
        ) = _classify_bore_identity_surface_roles(
            bore_identity_hypothesis_face_ids,
            bore_identity_hypothesis_diagnostics,
        )
        if bore_identity_wall_role_face_ids:
            previous_core_ids = tuple(sorted(chosen_core_ids))
            role_wall_ids = tuple(sorted(bore_identity_wall_role_face_ids))
            transition_role_set = {int(fid) for fid in tuple_ints(bore_identity_transition_role_face_ids)}

            # v14 correction:
            # v13 correctly stopped treating the selected ring as the automatic
            # bore radius, but it over-constrained wall ownership.  On real
            # meshes the strict radius-layer wall mask can collapse to only a
            # handful of triangles, producing a fake 2-face/4-face BORE.  That
            # is not usable physical ownership.  When an older component-core
            # candidate already exists, keep it as wall ownership support but
            # subtract transition-role faces so CHAMFER_FORM remains separate.
            # This preserves the semantic separation:
            #     identity support != ownership
            #     transition role != bore wall
            # while avoiding a brittle strict-mask-only bore candidate.
            min_semantic_wall_faces = int(max(8, min(24, int(min_core_faces))))
            component_wall_fallback_ids = tuple(
                sorted({int(fid) for fid in previous_core_ids if int(fid) not in transition_role_set})
            )
            identity_wall_too_sparse = bool(len(role_wall_ids) < min_semantic_wall_faces)
            fallback_has_semantic_size = bool(len(component_wall_fallback_ids) >= min_semantic_wall_faces)
            v14_component_wall_fallback_used = bool(identity_wall_too_sparse and fallback_has_semantic_size)

            if v14_component_wall_fallback_used:
                chosen_core_ids = component_wall_fallback_ids
                ownership_source = "component_core_minus_transition_role_fallback"
            else:
                chosen_core_ids = role_wall_ids
                ownership_source = "strict_radius_layer_wall_role"

            # v16 frame unification:
            # The role splitter may have selected a wall radius/axis different
            # from the selected outer rim.  Before any downstream stats, masks,
            # chamfer completion, or final candidate metrics run, switch the
            # entire recognition pipeline to that bore-local frame.
            measured_core_radius = float(
                dict(bore_identity_role_diagnostics).get("wall_role_radius", bore_identity_hypothesis_radius)
                or bore_identity_hypothesis_radius
            )
            _apply_recognition_frame(
                bore_identity_hypothesis_center,
                bore_identity_hypothesis_axis,
                measured_core_radius,
                reason="bore_local_frame_after_opening_field_identity",
            )

            chosen_core_stats = {
                **dict(stats(chosen_core_ids)),
                "bore_identity_hypothesis_used": True,
                "bore_identity_hypothesis": dict(bore_hyp_diag),
                "bore_identity_role_split": dict(bore_identity_role_diagnostics),
                "previous_component_core_face_count": int(len(previous_core_ids)),
                "strict_wall_role_face_count": int(len(role_wall_ids)),
                "transition_role_face_count": int(len(transition_role_set)),
                "component_wall_fallback_face_count": int(len(component_wall_fallback_ids)),
                "min_semantic_wall_faces": int(min_semantic_wall_faces),
                "identity_wall_too_sparse": bool(identity_wall_too_sparse),
                "v14_component_wall_fallback_used": bool(v14_component_wall_fallback_used),
                "v14_ownership_source": str(ownership_source),
                "v16_active_recognition_frame_reason": str(active_recognition_frame_reason),
                "v16_recognition_frame_history": tuple(recognition_frame_history),
                "v12_role_partition_replaced_component_core": bool(previous_core_ids != chosen_core_ids),
                "core_selection_anchor_policy": "opening_field_identity_wall_role_ownership_not_support_field",
                "candidate_isolation_policy": "identity_support_faces_are_diagnostics_wall_role_faces_are_bore_ownership_transition_faces_are_subtracted",
            }
            core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)

    # v9 fragmented-surface recognition fallback:
    # Imported or incomplete meshes can lose the normal cylindrical wall component
    # when the observable wall surface is split into many pieces. Rebuild may still
    # have two strong rim loops, but Recognition must
    # first keep the BOREHOLE candidate alive. This fallback is deliberately
    # still a Recognition decision: it uses selected RegionData / measured loop
    # evidence to recover a physical BORE candidate. It does not construct a
    # DeletePatchProposal and it does not mutate mesh geometry.
    boundary_loop_bore_fallback_used = False
    boundary_loop_bore_fallback_diagnostics: dict[str, object] = {}

    def _boundary_loop_bore_fallback_core_ids() -> tuple[tuple[int, ...], dict[str, object]]:
        loops_raw = tuple(boundary_loop_geometry or ())
        usable: list[dict[str, object]] = []
        for item in loops_raw:
            try:
                radius_value = float(getattr(item, "radius", 0.0) or 0.0)
                vertex_count = int(getattr(item, "vertex_count", getattr(item, "edge_count", 0)) or 0)
                edge_count = int(getattr(item, "edge_count", vertex_count) or 0)
                center_value = np.asarray(getattr(item, "center", center_vec), dtype=float).reshape(3)
                index_value = int(getattr(item, "index", len(usable)) or 0)
            except Exception:
                continue
            if not np.isfinite(radius_value) or radius_value <= 1.0e-9:
                continue
            if vertex_count < 8 and edge_count < 8:
                continue
            usable.append(
                {
                    "index": int(index_value),
                    "radius": float(radius_value),
                    "vertex_count": int(vertex_count),
                    "edge_count": int(edge_count),
                    "center": center_value,
                    "axial": float(np.dot(center_value - center_vec.reshape(3), axis_vec.reshape(3))),
                }
            )
        if len(usable) < 2:
            return (), {"used": False, "reason": "not_enough_measured_boundary_loops", "measured_loop_count": int(len(usable))}

        best_pair: tuple[dict[str, object], dict[str, object]] | None = None
        best_score = -1.0e30
        for i, a_loop in enumerate(usable):
            ca = np.asarray(a_loop["center"], dtype=float).reshape(3)
            ra = float(a_loop["radius"])
            for b_loop in usable[i + 1:]:
                cb = np.asarray(b_loop["center"], dtype=float).reshape(3)
                rb = float(b_loop["radius"])
                radius_ratio = min(ra, rb) / max(max(ra, rb), 1.0e-9)
                if radius_ratio < 0.58:
                    continue
                delta = cb - ca
                axial_sep = abs(float(np.dot(delta, axis_vec.reshape(3))))
                cross_sep = float(np.linalg.norm(delta - axis_vec.reshape(3) * float(np.dot(delta, axis_vec.reshape(3)))))
                max_radius = max(ra, rb, 1.0)
                if axial_sep <= max(0.35, 0.03 * max(axial_span_all, 1.0)):
                    continue
                if cross_sep > max(2.5, 0.40 * max_radius):
                    continue
                size_score = min(float(a_loop["vertex_count"]), float(b_loop["vertex_count"]))
                score = 8.0 * axial_sep + 0.15 * size_score + 3.0 * radius_ratio - 1.5 * cross_sep
                if score > best_score:
                    best_score = float(score)
                    best_pair = (a_loop, b_loop)
        if best_pair is None:
            return (), {"used": False, "reason": "no_coaxial_two_loop_pair", "measured_loop_count": int(len(usable))}

        loop0, loop1 = best_pair
        c0 = np.asarray(loop0["center"], dtype=float).reshape(3)
        c1 = np.asarray(loop1["center"], dtype=float).reshape(3)
        fallback_center = (c0 + c1) * 0.5
        t0 = float(np.dot(c0 - fallback_center, axis_vec.reshape(3)))
        t1 = float(np.dot(c1 - fallback_center, axis_vec.reshape(3)))
        axial_min_fb = min(t0, t1)
        axial_max_fb = max(t0, t1)
        axial_span_fb = max(abs(axial_max_fb - axial_min_fb), 1.0e-9)
        radius_fb = float((float(loop0["radius"]) + float(loop1["radius"])) * 0.5)
        radial_tol_fb = max(0.95, 0.22 * max(abs(radius_fb), 1.0))
        axial_pad_fb = max(0.75, 0.10 * max(axial_span_fb, 1.0), 0.035 * max(axial_span_all, 1.0))

        rel_fb = local_pts - fallback_center.reshape(1, 3)
        axial_fb = rel_fb @ axis_vec.reshape(3)
        radial_vec_fb = rel_fb - axial_fb.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radial_fb = np.linalg.norm(radial_vec_fb, axis=1)
        # The fallback is looser than the normal cylindrical detector but still
        # bounded by a two-rim cylindrical slab and wall-like normals. Cap-like
        # faces are excluded by the normal-axis limit when normals are usable.
        fallback_mask = (
            finite
            & (axial_fb >= axial_min_fb - axial_pad_fb)
            & (axial_fb <= axial_max_fb + axial_pad_fb)
            & (np.abs(radial_fb - radius_fb) <= radial_tol_fb)
            & (normal_axis_abs <= 0.84)
            & (radial_normal_alignment >= 0.0)
        )
        source_ids = ids_from_mask(fallback_mask)
        if not source_ids:
            return (), {
                "used": False,
                "reason": "two_loop_slab_found_but_no_faces_in_cylindrical_band",
                "loop_pair": (int(loop0["index"]), int(loop1["index"])),
                "radius": float(radius_fb),
            }

        components = connected_face_components(faces, source_ids)
        accepted: set[int] = set()
        component_diags: list[dict[str, object]] = []
        min_fallback_component_faces = max(4, min(32, int(0.08 * float(min_core_faces))))
        for comp in sorted(components, key=lambda c: len(c), reverse=True):
            comp_ids = tuple(sorted(int(fid) for fid in comp))
            if len(comp_ids) < min_fallback_component_faces:
                component_diags.append({"face_count": int(len(comp_ids)), "accepted": False, "reason": "too_few_faces"})
                continue
            st = stats(comp_ids)
            seed_direct, seed_adjacent, seed_anchor = seed_affinity(comp_ids)
            comp_na = float(st.get("normal_axis_abs_median", 1.0) or 1.0)
            comp_align = float(st.get("radial_normal_alignment_median", 0.0) or 0.0)
            comp_radial_span = float(st.get("radial_span", 999.0) or 999.0)
            comp_axial_span = float(st.get("axial_span", 0.0) or 0.0)
            # Accept seed-tied fragments, and also larger same-slab fragments when
            # imported topology breaks direct seed adjacency.
            accepted_fragment = bool(
                (seed_anchor and comp_na <= 0.90)
                or (
                    len(comp_ids) >= max(min_core_faces, 18)
                    and comp_na <= 0.84
                    and comp_radial_span <= max(2.25, 0.34 * max(abs(radius_fb), 1.0))
                    and comp_axial_span >= max(0.25, 0.018 * max(axial_span_fb, 1.0))
                    and comp_align >= 0.0
                )
            )
            if accepted_fragment:
                accepted.update(comp_ids)
            component_diags.append(
                {
                    "face_count": int(len(comp_ids)),
                    "accepted": bool(accepted_fragment),
                    "seed_direct_face_count": int(seed_direct),
                    "seed_adjacent_face_count": int(seed_adjacent),
                    "seed_anchor": bool(seed_anchor),
                    "normal_axis_abs_median": float(comp_na),
                    "radial_normal_alignment_median": float(comp_align),
                    "radial_span": float(comp_radial_span),
                    "axial_span": float(comp_axial_span),
                }
            )

        if len(accepted) < min_core_faces:
            return (), {
                "used": False,
                "reason": "two_loop_fallback_insufficient_accepted_faces",
                "accepted_face_count": int(len(accepted)),
                "min_core_faces": int(min_core_faces),
                "loop_pair": (int(loop0["index"]), int(loop1["index"])),
                "component_sample": tuple(component_diags[:12]),
            }
        return tuple(sorted(accepted)), {
            "used": True,
            "reason": "accepted_boundary_loop_pair_cylindrical_slab_fallback",
            "loop_pair": (int(loop0["index"]), int(loop1["index"])),
            "loop_vertex_counts": (int(loop0["vertex_count"]), int(loop1["vertex_count"])),
            "loop_edge_counts": (int(loop0["edge_count"]), int(loop1["edge_count"])),
            "radius": float(radius_fb),
            "radial_tolerance": float(radial_tol_fb),
            "axial_span": float(axial_span_fb),
            "axial_padding": float(axial_pad_fb),
            "source_face_count": int(len(source_ids)),
            "accepted_face_count": int(len(accepted)),
            "component_count": int(len(components)),
            "component_sample": tuple(component_diags[:12]),
        }

    if len(chosen_core_ids) < min_core_faces:
        fallback_ids, fallback_diag = _boundary_loop_bore_fallback_core_ids()
        boundary_loop_bore_fallback_diagnostics = dict(fallback_diag)
        if len(fallback_ids) >= min_core_faces:
            chosen_core_ids = tuple(sorted(fallback_ids))
            chosen_core_stats = {
                **dict(stats(chosen_core_ids)),
                "boundary_loop_bore_fallback_used": True,
                "boundary_loop_bore_fallback": dict(fallback_diag),
                "core_selection_anchor_policy": "boundary_loop_pair_cylindrical_slab_fallback_for_fragmented_surface_evidence",
                "candidate_isolation_policy": "fallback_still_regiondata_recognition_not_rebuild_target",
            }
            measured_core_radius = float(chosen_core_stats.get("radial_median", fallback_diag.get("radius", measured_core_radius)) or measured_core_radius)
            boundary_loop_bore_fallback_used = True
            core_patch_topology_after_closure = _component_engine_patch_boundary_diagnostics(faces, chosen_core_ids)

    # CHAMFER: annular transition components. This is component ownership, not
    # "single/double chamfered bore" branching. Any number of connected annular
    # transition components may be emitted.
    core_set = set(chosen_core_ids)
    core_mask_for_local = np.asarray([int(fid) in core_set for fid in valid_face_ids], dtype=bool)

    transition_radial_high = max(
        float(np.max(radial[finite])) if np.any(finite) else measured_core_radius,
        selected_frame_radius,
        measured_core_radius,
    ) + max(0.25 * max(abs(measured_core_radius), 1.0), 1.0)
    transition_radial_low = max(0.0, min(measured_core_radius, selected_frame_radius) * 0.35)
    transition_mask = (
        finite
        & (~core_mask_for_local)
        & (radial >= transition_radial_low)
        & (radial <= transition_radial_high)
        & (normal_axis_abs >= 0.12)
        & (normal_axis_abs <= 0.985)
        & (radial_normal_alignment >= 0.04)
    )

    # v12: include ring-layer transition-role faces explicitly.  These faces
    # were separated from the bore identity field before BORE ownership was
    # finalized, so they are valid physical-feature seeds for CHAMFER_FORM /
    # radius-stack ownership instead of broad diagnostic rows.
    transition_ids_source = tuple(sorted(set(ids_from_mask(transition_mask)) | set(bore_identity_transition_role_face_ids)))
    transition_components = connected_face_components(faces, transition_ids_source)

    chamfer_components: list[tuple[int, ...]] = []
    rejected_transition_components: list[dict[str, object]] = []
    min_chamfer_faces = max(10, min(80, int(0.010 * min(len(valid_face_ids), 2400))))
    max_chamfer_axial_span = max(2.5, 0.22 * max(axial_span_all, 1.0), 0.75 * max(measured_core_radius, 1.0))
    min_chamfer_radial_span = max(0.25, 0.025 * max(measured_core_radius, 1.0))

    def _adjacent_to_core(ids: tuple[int, ...]) -> int:
        if not core_set:
            return 0
        count = 0
        ids_set = set(ids)
        for fid in ids_set:
            for nb in adjacency.get(fid, ()):
                if int(nb) in core_set:
                    count += 1
        return int(count)

    def _near_core_axial_or_radial(st: Mapping[str, object]) -> bool:
        if not chosen_core_stats:
            return False
        try:
            comp_ax_min = float(st.get("axial_min", 0.0) or 0.0)
            comp_ax_max = float(st.get("axial_max", 0.0) or 0.0)
            core_ax_min = float(chosen_core_stats.get("axial_min", 0.0) or 0.0)
            core_ax_max = float(chosen_core_stats.get("axial_max", 0.0) or 0.0)
            if comp_ax_max < core_ax_min:
                axial_gap = core_ax_min - comp_ax_max
            elif core_ax_max < comp_ax_min:
                axial_gap = comp_ax_min - core_ax_max
            else:
                axial_gap = 0.0
            radial_min_v = float(st.get("radial_min", measured_core_radius) or measured_core_radius)
            radial_max_v = float(st.get("radial_max", measured_core_radius) or measured_core_radius)
            radial_touches_core = radial_min_v <= measured_core_radius + max(1.5, 0.18 * max(measured_core_radius, 1.0))
            return bool(axial_gap <= max(2.0, 0.10 * max(axial_span_all, 1.0)) and radial_touches_core)
        except Exception:
            return False

    for comp in transition_components:
        st = stats(comp)
        face_count = int(st.get("face_count", 0) or 0)
        axial_span = float(st.get("axial_span", 0.0) or 0.0)
        radial_span = float(st.get("radial_span", 0.0) or 0.0)
        na = float(st.get("normal_axis_abs_median", 0.0) or 0.0)
        align = float(st.get("radial_normal_alignment_median", 0.0) or 0.0)
        core_adjacency_pairs = _adjacent_to_core(tuple(sorted(comp)))
        near_core = _near_core_axial_or_radial(st)
        ring_transition_seed_overlap = int(len(set(tuple_ints(comp)) & set(bore_identity_transition_role_face_ids)))
        ring_layer_seeded = bool(ring_transition_seed_overlap > 0)
        accepted = bool(
            face_count >= (max(6, int(0.45 * min_chamfer_faces)) if ring_layer_seeded else min_chamfer_faces)
            and axial_span <= max_chamfer_axial_span
            and radial_span >= (0.55 * min_chamfer_radial_span if ring_layer_seeded else min_chamfer_radial_span)
            and na >= (0.06 if ring_layer_seeded else 0.12)
            and align >= (0.010 if ring_layer_seeded else 0.04)
            and (core_adjacency_pairs > 0 or near_core or ring_layer_seeded)
        )
        if accepted:
            chamfer_components.append(tuple(sorted(comp)))
        else:
            rejected_transition_components.append(
                {
                    "face_count": face_count,
                    "axial_span": axial_span,
                    "radial_span": radial_span,
                    "normal_axis_abs_median": na,
                    "radial_normal_alignment_median": align,
                    "core_adjacency_pair_count": int(core_adjacency_pairs),
                    "near_core": bool(near_core),
                    "ring_layer_seeded": bool(ring_layer_seeded),
                    "ring_transition_seed_overlap": int(ring_transition_seed_overlap),
                    "reason": "not_core_adjacent_or_ring_layer_seeded_annular_transition_component",
                }
            )

    # v7 correction: do NOT group chamfer components only because they lie on
    # the same axial side.  A broad RegionData cutout can contain many unrelated
    # mouths/chamfers on the same mounting plane.  The previous axial-side
    # aggregation collapsed those independent physical features into one
    # CHAMFER row.  Recognition must emit one CHAMFER_FORM CandidateData object
    # per connected annular transition component unless a later, explicit
    # centerline/loop-anchored merge proves that fragments belong to the same
    # physical chamfer.
    chamfer_groups: list[list[tuple[int, ...]]] = [[comp] for comp in sorted(
        chamfer_components,
        key=lambda ids: (
            float(stats(ids).get("axial_center", 0.0) or 0.0),
            -int(stats(ids).get("face_count", 0) or 0),
        ),
    )]
    chamfer_grouping_policy = "one_chamfer_candidate_per_connected_transition_component_no_axial_side_merging"

    # Chamfer completion from the neutral volume cutout:
    #
    # Region Select no longer pre-classifies a "bore surface". It hands Recognition
    # a neutral volume cutout. On rebuilt or fragmented meshes, the annular
    # transition can be split into several disconnected/local components. If we
    # emit only one such partial component as CHAMFER, the preview misses part of
    # the physical chamfer and rebuild_target later tries to repair from the
    # whole RegionData cutout context, which can explode into a non-watertight target.
    #
    # Complete each recognized chamfer side from geometry inside the same neutral
    # volume cutout: same small axial band, same annular radial band, transition-
    # like normals, excluding the already-owned borehole core. This is Recognition
    # ownership of a physical feature object, not Region Select classification and
    # not a rebuild-target fallback.
    chamfer_completion_diagnostics: list[dict[str, object]] = []

    def _complete_chamfer_group_from_neutral_volume_cutout(
        *,
        group_index: int,
        parts: list[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], dict[str, object]]:
        base_ids = tuple(sorted({int(fid) for part in parts for fid in part} - core_set))
        if not base_ids:
            return (), {
                "group_index": int(group_index),
                "used": False,
                "reason": "empty_chamfer_group",
                "base_face_count": 0,
                "completed_face_count": 0,
            }

        base_set = set(base_ids)
        base_stats = stats(base_ids)
        base_ax_min = float(base_stats.get("axial_min", 0.0) or 0.0)
        base_ax_max = float(base_stats.get("axial_max", 0.0) or 0.0)
        base_ax_center = float(base_stats.get("axial_center", 0.0) or 0.0)
        base_ax_span = max(float(base_stats.get("axial_span", 0.0) or 0.0), 1.0e-9)
        base_rad_min = float(base_stats.get("radial_min", measured_core_radius) or measured_core_radius)
        base_rad_max = float(base_stats.get("radial_max", selected_frame_radius or measured_core_radius) or (selected_frame_radius or measured_core_radius))
        base_rad_span = max(float(base_stats.get("radial_span", 0.0) or 0.0), 1.0e-9)

        axial_pad = max(0.42, 0.70 * base_ax_span, 0.025 * max(axial_span_all, 1.0))
        radial_pad = max(0.42, 0.28 * base_rad_span, 0.030 * max(abs(selected_frame_radius), abs(measured_core_radius), 1.0))
        axial_min = base_ax_min - axial_pad
        axial_max = base_ax_max + axial_pad
        radial_min = max(0.0, base_rad_min - radial_pad)
        radial_max = base_rad_max + radial_pad

        core_mask = np.asarray([int(fid) in core_set for fid in valid_face_ids], dtype=bool)
        source_mask = (
            finite
            & (~core_mask)
            & (axial >= axial_min)
            & (axial <= axial_max)
            & (radial >= radial_min)
            & (radial <= radial_max)
            & (normal_axis_abs >= 0.075)
            & (normal_axis_abs <= 0.992)
            & (radial_normal_alignment >= 0.014)
        )
        source_ids = tuple(sorted(set(ids_from_mask(source_mask)) - core_set))
        source_components = connected_face_components(faces, source_ids)

        completed: set[int] = set(base_set)
        component_diags: list[dict[str, object]] = []
        max_completed_total = max(len(base_set), int(max(len(base_set) * 2.75, len(base_set) + 256)))
        for comp in sorted(source_components, key=lambda item: len(item), reverse=True):
            comp_ids = tuple(sorted(int(fid) for fid in comp if int(fid) not in core_set))
            if not comp_ids:
                continue
            comp_set = set(comp_ids)
            comp_stats = stats(comp_ids)
            comp_ax_center = float(comp_stats.get("axial_center", base_ax_center) or base_ax_center)
            comp_ax_span = float(comp_stats.get("axial_span", 0.0) or 0.0)
            comp_rad_min = float(comp_stats.get("radial_min", base_rad_min) or base_rad_min)
            comp_rad_max = float(comp_stats.get("radial_max", base_rad_max) or base_rad_max)
            comp_rad_span = float(comp_stats.get("radial_span", 0.0) or 0.0)
            comp_na = float(comp_stats.get("normal_axis_abs_median", 0.0) or 0.0)
            comp_align = float(comp_stats.get("radial_normal_alignment_median", 0.0) or 0.0)

            overlaps_base = bool(comp_set & base_set)
            adjacent_to_completed = int(adjacent_count_to_set(comp_ids, completed))
            axial_close = bool(abs(comp_ax_center - base_ax_center) <= max(axial_pad, 0.55, 0.50 * base_ax_span))
            radial_overlap = bool(comp_rad_max >= radial_min and comp_rad_min <= radial_max)
            compact = bool(
                comp_ax_span <= max(base_ax_span + 2.0 * axial_pad, 2.8)
                and comp_rad_span <= max(base_rad_span + 2.0 * radial_pad, 4.2)
            )
            transition_like = bool(comp_na >= 0.075 and comp_na <= 0.992 and comp_align >= 0.014)
            enough_support = bool(len(comp_ids) >= max(6, int(0.18 * min_chamfer_faces)))
            # v7 correction: completion may close an already-owned chamfer
            # from adjacent fragments only.  It must not absorb disconnected
            # same-band islands simply because they are large; those islands are
            # separate physical features and deserve separate CHAMFER candidates.
            accepted = bool(
                overlaps_base
                or (
                    enough_support
                    and axial_close
                    and radial_overlap
                    and compact
                    and transition_like
                    and adjacent_to_completed > 0
                )
            )
            reason = "accepted" if accepted else "rejected_not_same_annular_transition_band"
            if accepted and len(completed) + len(comp_set - completed) > max_completed_total:
                accepted = False
                reason = "rejected_chamfer_completion_cap_reached"

            if accepted:
                completed.update(comp_set)

            component_diags.append(
                {
                    "face_count": int(len(comp_ids)),
                    "accepted": bool(accepted),
                    "reason": str(reason),
                    "overlaps_base": bool(overlaps_base),
                    "adjacent_to_completed_face_pair_count": int(adjacent_to_completed),
                    "axial_center": float(comp_ax_center),
                    "axial_span": float(comp_ax_span),
                    "radial_min": float(comp_rad_min),
                    "radial_max": float(comp_rad_max),
                    "radial_span": float(comp_rad_span),
                    "normal_axis_abs_median": float(comp_na),
                    "radial_normal_alignment_median": float(comp_align),
                }
            )

        completed_ids = tuple(sorted(completed))
        topology = _component_engine_patch_boundary_diagnostics(faces, completed_ids)
        return completed_ids, {
            "group_index": int(group_index),
            "used": bool(len(completed_ids) > len(base_set)),
            "policy": "complete_annular_chamfer_from_neutral_volume_cutout_adjacent_fragments_only",
            "chamfer_grouping_policy": chamfer_grouping_policy,
            "base_face_count": int(len(base_set)),
            "source_face_count": int(len(source_ids)),
            "source_component_count": int(len(source_components)),
            "completed_face_count": int(len(completed_ids)),
            "added_face_count": int(max(0, len(completed_ids) - len(base_set))),
            "axial_min": float(axial_min),
            "axial_max": float(axial_max),
            "radial_min": float(radial_min),
            "radial_max": float(radial_max),
            "patch_topology": topology,
            "patch_topology_rebuildable": bool(topology.get("patch_topology_rebuildable", False)),
            "component_sample": tuple(component_diags[:8]),
        }

    chamfer_candidates: list[dict[str, object]] = []
    chamfer_owned: set[int] = set()
    for group_index, parts in enumerate(chamfer_groups):
        group_ids, chamfer_completion_diag = _complete_chamfer_group_from_neutral_volume_cutout(
            group_index=int(group_index),
            parts=parts,
        )
        chamfer_completion_diagnostics.append(dict(chamfer_completion_diag))
        if len(group_ids) < min_chamfer_faces:
            continue
        st = stats(group_ids)
        radial_min = float(st.get("radial_min", measured_core_radius) or measured_core_radius)
        radial_max = float(st.get("radial_max", selected_frame_radius or measured_core_radius) or (selected_frame_radius or measured_core_radius))
        radial_med = float(st.get("radial_median", (radial_min + radial_max) * 0.5) or (radial_min + radial_max) * 0.5)
        axial_min = float(st.get("axial_min", 0.0) or 0.0)
        axial_max = float(st.get("axial_max", 0.0) or 0.0)
        chamfer_patch_topology = _component_engine_patch_boundary_diagnostics(faces, group_ids)
        # Diagnostic only. Recognition owns physical feature identity; rebuild.py
        # owns final topology acceptance/rejection. Do not gate rebuild here.
        chamfer_owned.update(group_ids)
        chamfer_candidates.append(
            {
                "candidate_id": f"component_engine.chamfer.{group_index + 1}",
                "feature_id": f"component_engine.chamfer.{group_index + 1}",
                "entity_type": "chamfer",
                "feature_kind": "chamfer",
                "candidate_scope": "recognition_feature",
                "display_name": f"CHAMFER — annular transition {group_index + 1}",
                "role": "rebuildable_chamfer_operation",
                "status": "promoted_component_chamfer_rebuild_candidate",
                "promotion_state": "promoted",
                "candidate_action_enabled": True,
                "candidate_action": "rebuild",
                "rebuild_authorized": True,
                "rebuild_gate": "promoted_chamfer_candidate",
                "face_ids": group_ids,
                "semantic_face_ids": group_ids,
                "preview_face_ids": group_ids,
                "rebuild_face_ids": group_ids,
                "face_count": int(len(group_ids)),
                "inner_radius": float(min(radial_min, measured_core_radius)),
                "mouth_radius": float(max(radial_max, selected_frame_radius, measured_core_radius)),
                "outer_radius": float(radial_max),
                "borehole_core_radius": float(measured_core_radius),
                "selected_frame_radius": float(selected_frame_radius),
                "radius": float(radial_med),
                "diameter": float(2.0 * radial_med),
                "height": float(max(0.0, axial_max - axial_min)),
                "axial_span": float(max(0.0, axial_max - axial_min)),
                "axial_min": float(axial_min),
                "axial_max": float(axial_max),
                "confidence": 0.82,
                "recognition_rule": "completed_annular_transition_from_neutral_volume_cutout" if bool(chamfer_completion_diag.get("used", False)) else "connected_annular_transition_component",
                "feature_ownership_source": "surface_component_classifier_v7",
                "feature_ownership_split": "one_chamfer_object_per_connected_annular_transition_component",
                "chamfer_grouping_policy": chamfer_grouping_policy,
                "ownership_metrics": st,
                "component_count": int(len(parts)),
                "chamfer_completion_policy": "complete_annular_chamfer_from_neutral_volume_cutout",
                "ring_layer_role_partition_policy": "ring_evidence_transition_role_faces_seed_physical_chamfer_candidates",
                "bore_identity_transition_role_face_count": int(len(bore_identity_transition_role_face_ids)),
                "chamfer_completion": dict(chamfer_completion_diag),
                "chamfer_completion_added_face_count": int(chamfer_completion_diag.get("added_face_count", 0) or 0),
                "density_bias_guard_used": bool(density_bias_guard_used),
                "raw_face_count_score_weight": float(raw_face_count_score_weight),
                "patch_topology": chamfer_patch_topology,
                "patch_topology_rebuildable": bool(chamfer_patch_topology.get("patch_topology_rebuildable", False)),
                "patch_topology_role": "diagnostic_only_rebuild_owns_acceptance",
                "display_face_ids": group_ids,
                **_x1_candidate_contract_fields(
                    family=FeatureFamily.CHAMFER_FORM,
                    stage=RecognitionStage.ACCEPTED_CANDIDATE,
                    evidence_kinds=(EvidenceKind.SELECTED_EDGE_LOOP, EvidenceKind.OPENING_RING, EvidenceKind.CHAMFER_BAND),
                    accepted=True,
                    promotion_reasons=("connected_annular_transition_surface", "current_chamfer_rebuild_path_supported"),
                    primitive_axis=axis_vec,
                    primitive_radius=radial_med,
                    primitive_depth=max(0.0, axial_max - axial_min),
                    candidate_id=f"component_engine.chamfer.{len(chamfer_candidates) + 1}",
                    face_ids=group_ids,
                    recognition_rule="completed_annular_transition_from_neutral_volume_cutout" if bool(chamfer_completion_diag.get("used", False)) else "connected_annular_transition_component",
                    status="promoted_component_chamfer_rebuild_candidate",
                    confidence=0.82,
                    diagnostics={
                        "patch_topology": chamfer_patch_topology,
                        "chamfer_completion": dict(chamfer_completion_diag),
                        "chamfer_grouping_policy": chamfer_grouping_policy,
                    },
                ),
            }
        )

    # Final borehole ownership is the physical cylindrical wall role.  Chamfer
    # candidates are independent feature objects and relationship metadata.  The
    # broad opening-field support may prove BORE identity, but it must not be
    # promoted into BORE-owned faces because it can include annular transition
    # surfaces that belong to CHAMFER_FORM.
    core_before_chamfer_subtract = tuple(sorted(chosen_core_ids))
    core_final_candidate = tuple(sorted(set(chosen_core_ids) - chamfer_owned))
    bore_core_preservation_used = False
    if len(core_final_candidate) < min_core_faces and len(core_before_chamfer_subtract) >= min_core_faces:
        # Preserve the already wall-role-owned core if an overlapping chamfer mask
        # would otherwise erase it.  Do not restore the whole identity support field.
        core_final = core_before_chamfer_subtract
        bore_core_preservation_used = True
    elif bore_identity_hypothesis_ok and bore_identity_wall_role_face_ids and not core_final_candidate:
        core_final = tuple(sorted(bore_identity_wall_role_face_ids))
        bore_core_preservation_used = True
    elif bore_identity_hypothesis_ok and bore_identity_wall_role_face_ids and len(core_final_candidate) < len(bore_identity_wall_role_face_ids):
        core_final = tuple(sorted(set(core_final_candidate) | set(bore_identity_wall_role_face_ids)))
        bore_core_preservation_used = True
    else:
        core_final = core_final_candidate
    borehole_candidate: dict[str, object] | None = None
    bore_identity_allows_candidate = bool(bore_identity_hypothesis_ok and bool(core_final))
    if len(core_final) >= min_core_faces or bore_identity_allows_candidate:
        st = stats(core_final)
        if bore_identity_hypothesis_ok:
            axial_min = float(bore_identity_hypothesis_diagnostics.get("axial_min", st.get("axial_min", 0.0)) or 0.0)
            axial_max = float(bore_identity_hypothesis_diagnostics.get("axial_max", st.get("axial_max", 0.0)) or 0.0)
            # v16: final BORE dimensions must use the active wall-role radius,
            # not the broad opening-field / selected mouth radius.
            measured_core_radius = float(
                dict(bore_identity_role_diagnostics).get("wall_role_radius", measured_core_radius)
                or measured_core_radius
            )
        else:
            axial_min = float(st.get("axial_min", 0.0) or 0.0)
            axial_max = float(st.get("axial_max", 0.0) or 0.0)
        core_final_topology = _component_engine_patch_boundary_diagnostics(faces, core_final)
        core_final_patch_topology_rebuildable = bool(core_final_topology.get("patch_topology_rebuildable", False))
        # Diagnostic only. Recognition owns physical feature identity; rebuild.py
        # owns final topology acceptance/rejection.  A physically recognized
        # BOREHOLE candidate must not be demoted or blocked here because its
        # owned patch is not already a clean two-loop delete patch.
        borehole_candidate = {
            "candidate_id": "component_engine.borehole.1",
            "feature_id": "component_engine.borehole.1",
            "entity_type": "borehole",
            "feature_kind": "borehole",
            "candidate_scope": "recognition_feature",
            "display_name": "BOREHOLE — cylindrical opening-field feature",
            "role": "rebuildable_bore_operation",
            "status": "promoted_bore_opening_field_rebuild_candidate" if bore_identity_hypothesis_ok else "promoted_component_borehole_rebuild_candidate",
            "promotion_state": "promoted",
            "candidate_action_enabled": True,
            "candidate_action": "rebuild",
            "rebuild_authorized": True,
            "rebuild_gate": "promoted_borehole_candidate",
            "face_ids": core_final,
            "semantic_face_ids": core_final,
            "preview_face_ids": core_final,
            "rebuild_face_ids": core_final,
            "face_count": int(len(core_final)),
            "radius": float(measured_core_radius),
            "diameter": float(2.0 * measured_core_radius),
            "selected_frame_radius": float(selected_frame_radius),
            "search_reference_radius": float(core_radius),
            "depth": float(max(0.0, axial_max - axial_min)),
            "height": float(max(0.0, axial_max - axial_min)),
            "axial_span": float(max(0.0, axial_max - axial_min)),
            "axial_min": float(axial_min),
            "axial_max": float(axial_max),
            "confidence": 0.88 if bore_identity_hypothesis_ok else 0.84,
            "recognition_rule": "opening_loop_cylindrical_field_bore_identity" if bore_identity_hypothesis_ok else "connected_cylindrical_component_at_selected_core_radius",
            "feature_ownership_source": "surface_component_classifier_v7",
            "feature_ownership_split": "bore_identity_support_separated_from_wall_role_face_ownership",
            "core_face_rule": str(core_rule),
            "core_selection_anchor_policy": str(core_selection_anchor_policy),
            "selected_seed_face_count": int(len(seed_face_set)),
            "selected_radius_anchor_tolerance": float(selected_radius_anchor_tolerance),
            "density_bias_guard_used": bool(density_bias_guard_used),
            "raw_face_count_score_weight": float(raw_face_count_score_weight),
            "density_normalized_score_weight": float(density_normalized_score_weight),
            "geometric_band_score_weight": float(geometric_band_score_weight),
            "selected_rim_anchor_score_weight": float(selected_rim_anchor_score_weight),
            "excluded_chamfer_face_count": int(len(chamfer_owned)),
            "excluded_chamfer_face_ids": tuple(sorted(chamfer_owned)),
            "core_face_count_before_chamfer_subtract": int(len(core_before_chamfer_subtract)),
            "core_face_count_after_chamfer_subtract": int(len(core_final_candidate)),
            "bore_core_preservation_used": bool(bore_core_preservation_used),
            "bore_identity_hypothesis_used": bool(bore_identity_hypothesis_used),
            "bore_identity_hypothesis_ok": bool(bore_identity_hypothesis_ok),
            "bore_identity_hypothesis": dict(bore_identity_hypothesis_diagnostics),
            "bore_identity_support_face_count": int(len(bore_identity_hypothesis_face_ids)),
            "bore_identity_support_face_ids_sample": tuple(int(fid) for fid in bore_identity_hypothesis_face_ids[:64]),
            "bore_identity_wall_role_face_count": int(len(bore_identity_wall_role_face_ids)),
            "bore_identity_wall_role_face_ids_sample": tuple(int(fid) for fid in bore_identity_wall_role_face_ids[:64]),
            "bore_identity_transition_role_face_count": int(len(bore_identity_transition_role_face_ids)),
            "bore_identity_transition_role_face_ids_sample": tuple(int(fid) for fid in bore_identity_transition_role_face_ids[:64]),
            "bore_identity_unowned_support_face_count": int(len(bore_identity_unowned_support_face_ids)),
            "bore_identity_role_split": dict(bore_identity_role_diagnostics),
            "bore_identity_policy": "bore_identity_from_opening_radius_axis_curvature_field_not_face_count",
            "bore_ownership_policy": "candidate_face_ids_are_wall_role_faces_identity_support_is_diagnostics_only",
            "v16_active_recognition_frame_reason": str(active_recognition_frame_reason),
            "v16_recognition_frame_history": tuple(recognition_frame_history),
            "v16_frame_policy": "RegionData frame locates AOI; BoreLocalFrame classifies physical ownership",
            "boundary_loop_bore_fallback_used": bool(boundary_loop_bore_fallback_used),
            "boundary_loop_bore_fallback": dict(boundary_loop_bore_fallback_diagnostics),
            "unclassified_faces_policy": "diagnostic_only_not_rebuild_candidate",
            "ownership_metrics": st,
            "radius_source": "opening_field_hypothesis_radius" if bore_identity_hypothesis_ok else "measured_cylindrical_component_radial_median",
            "selected_frame_radius_source": "region_select_region_radius",
            "patch_topology": core_final_topology,
            "patch_topology_rebuildable": bool(core_final_patch_topology_rebuildable),
            "patch_topology_role": "diagnostic_only_rebuild_owns_acceptance",
            "recognition_rebuild_entry_policy": "physical_feature_identity_can_enter_rebuild_rebuild_py_validates_topology",
            "closure_added_face_count": int(closure_added_face_count),
            "same_cylinder_completion_added_face_count": int(same_cylinder_completion_added_face_count),
            "same_cylinder_completion_source_face_count": int(same_cylinder_completion_source_face_count),
            "same_cylinder_completion_component_count": int(same_cylinder_completion_component_count),
            "same_cylinder_completion_used_component_count": int(same_cylinder_completion_used_component_count),
            "neutral_volume_cutout_completion_added_face_count": int(neutral_volume_cutout_completion_added_face_count),
            "neutral_volume_cutout_completion_source_face_count": int(neutral_volume_cutout_completion_source_face_count),
            "neutral_volume_cutout_completion_component_count": int(neutral_volume_cutout_completion_component_count),
            "neutral_volume_cutout_completion_used_component_count": int(neutral_volume_cutout_completion_used_component_count),
            "neutral_volume_cutout_completion_policy": "recognition_completes_physical_feature_fragments_inside_neutral_volume_cutout",
            "neutral_volume_cutout_completion_components_sample": tuple(neutral_volume_cutout_completion_diagnostics[:8]),
            "same_cylinder_completion_policy": "complete_only_connected_same_surface_fragments_not_remote_seed_components",
            "candidate_isolation_policy": "display_candidate_owns_wall_role_faces_identity_support_stays_diagnostics",
            "core_patch_topology_before_closure": core_patch_topology_before_closure,
            "core_patch_topology_after_closure": core_patch_topology_after_closure,
            "display_face_ids": core_final,
            **_x1_candidate_contract_fields(
                family=FeatureFamily.BORE,
                stage=RecognitionStage.ACCEPTED_CANDIDATE,
                evidence_kinds=(EvidenceKind.SELECTED_EDGE_LOOP, EvidenceKind.OPENING_RING, EvidenceKind.BORE_WALL_NORMALS, EvidenceKind.RADIUS_CONSISTENCY, EvidenceKind.OPPOSITE_OPENING) if bore_identity_hypothesis_ok else (EvidenceKind.SELECTED_EDGE_LOOP, EvidenceKind.OPENING_RING, EvidenceKind.BORE_WALL_NORMALS, EvidenceKind.RADIUS_CONSISTENCY),
                accepted=True,
                promotion_reasons=("opening_field_identity", "cylindrical_curvature_support", "current_bore_rebuild_path_supported") if bore_identity_hypothesis_ok else ("connected_cylindrical_surface_component", "selected_rim_anchor", "current_bore_rebuild_path_supported"),
                primitive_axis=axis_vec,
                primitive_radius=measured_core_radius,
                primitive_depth=max(0.0, axial_max - axial_min),
                candidate_id="component_engine.borehole.1",
                face_ids=core_final,
                recognition_rule="opening_loop_cylindrical_field_bore_identity" if bore_identity_hypothesis_ok else "connected_cylindrical_component_at_selected_core_radius",
                status="promoted_bore_opening_field_rebuild_candidate" if bore_identity_hypothesis_ok else "promoted_component_borehole_rebuild_candidate",
                confidence=0.88 if bore_identity_hypothesis_ok else 0.84,
                diagnostics={
                    "patch_topology": core_final_topology,
                    "core_face_rule": str(core_rule),
                    "bore_identity_hypothesis": dict(bore_identity_hypothesis_diagnostics),
                    "bore_identity_role_split": dict(bore_identity_role_diagnostics),
                    "bore_ownership_policy": "candidate_face_ids_are_wall_role_faces_identity_support_is_diagnostics_only",
                    "v16_active_recognition_frame_reason": str(active_recognition_frame_reason),
                    "v16_recognition_frame_history": tuple(recognition_frame_history),
                },
            ),
        }

    region_context_topology: dict[str, object] = {}
    region_context_candidate_data_used = False
    region_context_target_policy_gate_used = False

    # Borehole RegionData cutout context diagnostics:
    #
    # Keep the visible BOREHOLE identity owned by the cylindrical core component.
    # Do not replace ``face_ids`` / ``preview_face_ids`` with the full RegionData cutout here.
    # Recognition may report that the neutral volume cutout contains additional
    # context useful to rebuild_target.py, but it must not turn topology facts
    # into a recognition-side rebuild gate.
    #
    #   recognition candidate = measured cylindrical BOREHOLE core
    #   preview faces         = core_final
    #   RegionData cutout context           = diagnostics only / downstream target-policy input
    #
    # Final geometry is accepted or rejected only by rebuild.py's measured-loop
    # and watertight trial validation.
    if (
        borehole_candidate is not None
        and not chamfer_candidates
        and len(valid_face_ids) > len(tuple_ints(borehole_candidate.get("face_ids", ())))
    ):
        original_core_ids = tuple_ints(borehole_candidate.get("face_ids", ()))
        region_context_ids = tuple(sorted(valid_face_ids))
        region_context_topology = _component_engine_patch_boundary_diagnostics(faces, region_context_ids)
        region_context_patch_shape_is_two_loop = bool(region_context_topology.get("patch_topology_rebuildable", False))
        measured_two_loop_context = bool(len(tuple(boundary_loop_geometry or ())) >= 2)

        borehole_candidate.update(
            {
                "surface_condition": "core_patch_with_neutral_volume_context",
                "recognition_rule": "connected_cylindrical_identity_with_neutral_volume_context_diagnostics",
                "feature_ownership_split": "borehole_identity_and_preview_from_core_delete_patch_from_rebuild_target_policy",
                "unclassified_faces_policy": "available_to_rebuild_target_as_region_context_not_absorbed_by_recognition",
                "target_context_source": "region_data_face_pool",
                "target_context_face_count": int(len(region_context_ids)),
                "target_context_added_face_count": int(len(region_context_ids) - len(original_core_ids)),
                "target_context_face_ids_sample": tuple(int(fid) for fid in region_context_ids[:64]),
                "target_context_patch_shape_is_two_loop": bool(region_context_patch_shape_is_two_loop),
                "target_context_measured_two_loop_context": bool(measured_two_loop_context),
                "region_context_topology": region_context_topology,
                "original_core_face_count": int(len(original_core_ids)),
                # Keep these explicitly core-owned.  Do not replace preview with RegionData cutout.
                "face_ids": original_core_ids,
                "semantic_face_ids": original_core_ids,
                "preview_face_ids": original_core_ids,
                "rebuild_face_ids": original_core_ids,
                "face_count": int(len(original_core_ids)),
                "delete_patch_face_ids": (),
                "region_context_policy": "diagnostic_only_rebuild_target_constructs_delete_patch",
            }
        )
        # These diagnostics remain false intentionally: RegionData context is
        # reported only; it is not a Recognition-side CandidateData or target gate.
        region_context_candidate_data_used = False
        region_context_target_policy_gate_used = False

    features: list[dict[str, object]] = []
    if borehole_candidate is not None:
        features.append(borehole_candidate)
    features.extend(chamfer_candidates)

    x1_diagnostic_family_candidates = _x1_family_diagnostic_candidates(
        valid_face_ids=valid_face_ids,
        boundary_loop_geometry=tuple(boundary_loop_geometry or ()),
        boundary_loops=tuple(boundary_loops or ()),
        vertices=vertices,
        region_axis=axis_vec,
        selected_frame_radius=float(selected_frame_radius),
        axial_span_all=float(axial_span_all),
        borehole_candidate=borehole_candidate,
        chamfer_candidates=tuple(chamfer_candidates),
    )
    features.extend(x1_diagnostic_family_candidates)

    owned = set()
    for item in features:
        owned.update(tuple_ints(item.get("face_ids", ())))
    unclassified = tuple(sorted(set(valid_face_ids) - owned))

    # Emit UNCLASSIFIED as a diagnostic-only feature when the classifier could
    # not promote any physical object. This keeps edge/RegionData cutout selection visible to
    # the GUI without authorizing rebuild or asking BoreActions to recover
    # candidates from old diagnostic ledgers.
    if not features and unclassified:
        st = stats(unclassified)
        unclassified_candidate = {
            "candidate_id": "component_engine.unclassified.1",
            "feature_id": "component_engine.unclassified.1",
            "entity_type": "unclassified",
            "feature_kind": "unclassified",
            "candidate_scope": "recognition_feature",
            "display_name": "UNCLASSIFIED — RegionData cutout surface evidence",
            "role": "diagnostic_preview_only",
            "status": "diagnostic_unclassified_region_data_faces_not_candidate_action_enabled",
            "promotion_state": "diagnostic_only",
            "candidate_action_enabled": False,
            "candidate_action": "preview_only",
            "rebuild_authorized": False,
            "rebuild_gate": "diagnostic_unclassified_not_rebuild_candidate",
            "face_ids": unclassified,
            "semantic_face_ids": unclassified,
            "preview_face_ids": unclassified,
            "rebuild_face_ids": (),
            "face_count": int(len(unclassified)),
            "radius": float(st.get("radial_median", measured_core_radius) or measured_core_radius),
            "diameter": float(2.0 * float(st.get("radial_median", measured_core_radius) or measured_core_radius)),
            "axial_span": float(st.get("axial_span", 0.0) or 0.0),
            "confidence": 0.0,
            "recognition_rule": "diagnostic_unclassified_region_remainder_when_no_physical_object_promoted",
            "feature_ownership_source": "surface_component_classifier_v7",
            "unclassified_faces_policy": "diagnostic_only_not_rebuild_candidate",
            "ownership_metrics": st,
            "display_face_ids": unclassified,
            **_x1_candidate_contract_fields(
                family=FeatureFamily.UNKNOWN,
                stage=RecognitionStage.DIAGNOSTIC_ONLY,
                evidence_kinds=(EvidenceKind.SELECTED_EDGE_LOOP,),
                accepted=False,
                rejection_reasons=("no_physical_feature_family_promoted",),
                primitive_axis=axis_vec,
                primitive_radius=float(st.get("radial_median", measured_core_radius) or measured_core_radius),
                primitive_depth=float(st.get("axial_span", 0.0) or 0.0),
                candidate_id="component_engine.unclassified.1",
                face_ids=unclassified,
                recognition_rule="diagnostic_unclassified_region_remainder_when_no_physical_object_promoted",
                status="diagnostic_unclassified_region_data_faces_not_candidate_action_enabled",
                confidence=0.0,
                diagnostics={"ownership_metrics": st},
            ),
        }
        features.append(unclassified_candidate)

    overlap_pairs: list[dict[str, object]] = []
    for i, a in enumerate(features):
        a_ids = set(tuple_ints(a.get("face_ids", ())))
        for j, b in enumerate(features):
            if j <= i:
                continue
            b_ids = set(tuple_ints(b.get("face_ids", ())))
            overlap = sorted(a_ids & b_ids)
            if overlap:
                overlap_pairs.append(
                    {
                        "a": str(a.get("candidate_id", "")),
                        "b": str(b.get("candidate_id", "")),
                        "overlap_face_count": int(len(overlap)),
                    }
                )

    # Feature graph: composition metadata, not type classification.
    # A chamfer next to a borehole is represented as an adjacency relation; the
    # borehole remains a BOREHOLE and the chamfer remains a CHAMFER.
    # v6 makes that relation typed and serializable so the GUI/debug ledger can
    # show the assembly relation without creating a synthetic feature family.
    relation_map, typed_relation_map, bore_chamfer_relation_pairs = _x1_feature_relationship_graph(
        features=features,
        adjacency=adjacency,
    )

    for idx, item in enumerate(features):
        cid = str(item.get("candidate_id", f"candidate_{idx}"))
        relations = tuple(relation_map.get(cid, ()))
        typed_relations = tuple(typed_relation_map.get(cid, ()))
        item["relationships"] = {
            "composition_role": "physical_feature_object",
            "assembly_classification": "none",
            "assembly_classification_policy": "relationships_only_do_not_create_feature_families",
            "adjacent_features": relations,
            "adjacent_feature_count": int(len(relations)),
            "feature_relationships": typed_relations,
            "feature_relationship_count": int(len(typed_relations)),
        }
        item["feature_relationships"] = typed_relations
        item["feature_relationship_count"] = int(len(typed_relations))
        ledger = dict(item.get("x1_evidence_ledger", {}) or {})
        ledger["feature_relationships"] = typed_relations
        ledger["feature_relationship_count"] = int(len(typed_relations))
        ledger["relationship_graph_update"] = "v6_typed_relationship_metadata_added_after_candidate_identity"
        item["x1_evidence_ledger"] = ledger

    diagnostics = {
        "active_candidate_authority": "surface_component_classifier_v7",
        "component_engine_version": 10,
        "assembly_classification_policy": "do_not_classify_assemblies_classify_physical_surface_objects",
        "chamfered_bore_policy": "not_a_feature_family_bore_and_chamfer_are_separate_physical_feature_objects",
        "bore_chamfer_relation_count": int(len(bore_chamfer_relation_pairs)),
        "bore_chamfer_relations": tuple(bore_chamfer_relation_pairs[:16]),
        "feature_relationship_policy": "typed_relationships_only_no_assembly_feature_families",
        "feature_relationship_kind_vocabulary": tuple(item.value for item in FeatureRelationshipKind),
        "supported_feature_objects": ("borehole", "chamfer"),
        "supported_feature_families": tuple(item.value for item in FeatureFamily),
        "recognition_stage_policy": tuple(item.value for item in RecognitionStage),
        "evidence_kind_vocabulary": tuple(item.value for item in EvidenceKind),
        "x1_family_diagnostic_candidate_count": int(len(x1_diagnostic_family_candidates)),
        "x1_family_diagnostic_candidate_ids": tuple(str(item.get("candidate_id", "")) for item in x1_diagnostic_family_candidates),
        "x1_family_diagnostic_version": "v14_sparse_wall_role_fallback_no_assembly_families",
        "selected_frame_radius": float(selected_frame_radius),
        "search_reference_radius": float(core_radius),
        "borehole_measured_radius": float(measured_core_radius),
        "selected_core_center": to_vector3(center_vec),
        "selected_core_axis": to_vector3(axis_vec),
        "region_face_count": int(len(valid_face_ids)),
        "region_axial_span": float(axial_span_all),
        "region_radial_span": float(radial_span_all),
        "face_adjacency_count": int(len(adjacency)),
        "core_rule": str(core_rule),
        "core_selection_anchor_policy": str(core_selection_anchor_policy),
        "selected_seed_face_count": int(len(seed_face_set)),
        "selected_radius_anchor_tolerance": float(selected_radius_anchor_tolerance),
        "core_detection_policy": "v10_opening_field_bore_identity_component_faces_are_support_quality_not_identity_gate",
        "bore_identity_hypothesis_used": bool(bore_identity_hypothesis_used),
        "bore_identity_hypothesis_ok": bool(bore_identity_hypothesis_ok),
        "bore_identity_hypothesis": dict(bore_identity_hypothesis_diagnostics),
        "bore_identity_policy": "opening_radius_axis_and_curvature_field_define_bore_wall_continuity_is_rebuild_quality",
        "density_bias_guard_used": bool(density_bias_guard_used),
        "raw_face_count_score_weight": float(raw_face_count_score_weight),
        "density_normalized_score_weight": float(density_normalized_score_weight),
        "geometric_band_score_weight": float(geometric_band_score_weight),
        "selected_rim_anchor_score_weight": float(selected_rim_anchor_score_weight),
        "min_core_faces_density_guarded": int(min_core_faces),
        "min_chamfer_faces_density_guarded": int(min_chamfer_faces),
        "neutral_volume_cutout_completion_added_face_count": int(neutral_volume_cutout_completion_added_face_count),
        "neutral_volume_cutout_completion_source_face_count": int(neutral_volume_cutout_completion_source_face_count),
        "neutral_volume_cutout_completion_component_count": int(neutral_volume_cutout_completion_component_count),
        "neutral_volume_cutout_completion_used_component_count": int(neutral_volume_cutout_completion_used_component_count),
        "neutral_volume_cutout_completion_components_sample": tuple(neutral_volume_cutout_completion_diagnostics[:8]),
        "candidate_isolation_policy": "display_candidate_owns_wall_role_faces_identity_support_stays_diagnostics",
        "selected_seed_required_for_remote_component": bool(selected_seed_required_for_remote_component()),
        "topology_helper_source": "topology.py",
        "strict_core_source_face_count": int(len(strict_core_ids_source)),
        "core_source_face_count": int(len(core_ids_source)),
        "core_component_count": int(len(core_components)),
        "chosen_core_face_count": int(len(chosen_core_ids)),
        "closure_added_face_count": int(closure_added_face_count),
        "same_cylinder_completion_added_face_count": int(same_cylinder_completion_added_face_count),
        "same_cylinder_completion_source_face_count": int(same_cylinder_completion_source_face_count),
        "same_cylinder_completion_component_count": int(same_cylinder_completion_component_count),
        "same_cylinder_completion_used_component_count": int(same_cylinder_completion_used_component_count),
        "same_cylinder_completion_diagnostics": same_cylinder_completion_diagnostics,
        "core_patch_topology_before_closure": core_patch_topology_before_closure,
        "core_patch_topology_after_closure": core_patch_topology_after_closure,
        "primary_core_component_face_count": int(primary_core_component_face_count),
        "merged_core_component_count": int(merged_core_component_count),
        "borehole_face_count": int(len(tuple_ints(borehole_candidate.get("face_ids", ()))) if borehole_candidate is not None else len(core_final)),
        "raw_core_borehole_face_count": int(len(core_final)),
        "region_context_candidate_data_used": bool(region_context_candidate_data_used),
        "region_context_target_policy_gate_used": bool(region_context_target_policy_gate_used),
        "region_context_topology": dict(region_context_topology),
        "borehole_topology_role": "diagnostic_only_rebuild_owns_acceptance",
        "borehole_recognition_rebuild_entry_policy": "physical_feature_identity_can_enter_rebuild_rebuild_py_validates_topology",
        "transition_source_face_count": int(len(transition_ids_source)),
        "transition_component_count": int(len(transition_components)),
        "accepted_chamfer_component_count": int(len(chamfer_components)),
        "chamfer_candidate_count": int(len(chamfer_candidates)),
        "chamfer_completion_policy": "complete_annular_chamfer_from_neutral_volume_cutout_adjacent_fragments_only",
        "chamfer_grouping_policy": chamfer_grouping_policy if 'chamfer_grouping_policy' in locals() else "not_run",
        "chamfer_group_count": int(len(chamfer_groups)) if 'chamfer_groups' in locals() else 0,
        "chamfer_completion_diagnostics": tuple(chamfer_completion_diagnostics[:12]),
        "chamfer_completion_added_face_count": int(sum(int(item.get("added_face_count", 0) or 0) for item in chamfer_completion_diagnostics)),
        "chamfer_completion_used_group_count": int(sum(1 for item in chamfer_completion_diagnostics if bool(item.get("used", False)))),
        "rejected_transition_components": tuple(rejected_transition_components[:12]),
        "unclassified_face_count": int(len(unclassified)),
        "unclassified_face_ids_sample": tuple(unclassified[:64]),
        "candidate_overlap_pairs": tuple(overlap_pairs),
        "candidate_overlap_pair_count": int(len(overlap_pairs)),
        "candidate_face_counts": tuple(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "entity_type": str(item.get("entity_type", "")),
                "face_count": int(len(tuple_ints(item.get("face_ids", ())))),
            }
            for item in features
        ),
        "boundary_loop_count": int(len(boundary_loop_geometry)),
        "boundary_loop_edge_counts": tuple(int(getattr(loop, "edge_count", 0)) for loop in tuple(boundary_loop_geometry or ())),
        "unclassified_faces_policy": "diagnostic_only_not_rebuild_candidate",
        "candidate_stage_counts": tuple(
            {
                "recognition_stage": str(stage),
                "count": int(sum(1 for item in features if str(item.get("recognition_stage", "")) == str(stage))),
            }
            for stage in tuple(item.value for item in RecognitionStage)
        ),
        "candidate_family_counts": tuple(
            {
                "feature_family": str(family),
                "count": int(sum(1 for item in features if str(item.get("feature_family", "")) == str(family))),
            }
            for family in tuple(item.value for item in FeatureFamily)
        ),
    }

    return {
        "candidate_data": tuple(features),
        "features": tuple(features),
        "diagnostics": diagnostics,
    }


def _recognition_result_dict_from_component_features(
    *,
    features: tuple[dict[str, object], ...],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Return a CandidateResult-like dictionary for GUI diagnostics."""

    return {
        "contract_type": "candidate_result",
        "engine": "surface_component_classifier_v7",
        "mode": "region_data_to_candidate_data",
        "candidate_count": int(len(features)),
        "candidate_data": features,
        "features": features,
        "diagnostics": dict(diagnostics or {}),
        "promoted_candidate_count": int(sum(1 for item in features if _x1_rebuild_allowed(item) and bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False))))),
    }


# -----------------------------------------------------------------------------
# v18 semantic-layer recognition rewrite scaffold
# -----------------------------------------------------------------------------


def _finite_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    """Return a finite percentile with deterministic empty fallback."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, float(q)))


def _v17_scalar_layers(
    values: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    face_ids: np.ndarray | None = None,
    tolerance: float,
    min_count: int = 3,
) -> tuple[dict[str, object], ...]:
    """Cluster scalar radius evidence into explicit RadiusLayer-like rows.

    This is intentionally an evidence builder, not a feature classifier.  It
    groups nearby radial observations so recognition can reason with named
    layers instead of allowing one selected radius, one component, or one mask to
    become semantic authority.
    """

    vals = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(vals)
    if scores is None:
        score_arr = np.ones_like(vals, dtype=float)
    else:
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        if score_arr.shape != vals.shape:
            score_arr = np.ones_like(vals, dtype=float)
    if face_ids is None:
        fid_arr = np.arange(len(vals), dtype=np.int64)
    else:
        fid_arr = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        if fid_arr.shape != vals.shape:
            fid_arr = np.arange(len(vals), dtype=np.int64)
    vals = vals[finite_mask]
    score_arr = score_arr[finite_mask]
    fid_arr = fid_arr[finite_mask]
    if vals.size == 0:
        return ()
    order = np.argsort(vals)
    vals = vals[order]
    score_arr = score_arr[order]
    fid_arr = fid_arr[order]
    tol = max(float(tolerance), 1.0e-9)
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    for idx in range(1, len(vals)):
        prev = float(vals[idx - 1])
        cur = float(vals[idx])
        # The layer window grows slightly with radius but remains local.
        local_tol = max(tol, 0.025 * max(abs(prev), abs(cur), 1.0))
        if abs(cur - prev) > local_tol:
            groups.append((vals[start:idx], score_arr[start:idx], fid_arr[start:idx]))
            start = idx
    groups.append((vals[start:], score_arr[start:], fid_arr[start:]))

    layers: list[dict[str, object]] = []
    for layer_index, (g_vals, g_scores, g_fids) in enumerate(groups):
        if len(g_vals) < int(min_count):
            continue
        radius = float(np.median(g_vals))
        mad = float(np.median(np.abs(g_vals - radius))) if len(g_vals) else 0.0
        support = float(np.sum(np.clip(g_scores, 0.0, 1.0)))
        layers.append(
            {
                "layer_id": f"radius_layer_{len(layers)}",
                "layer_index": int(layer_index),
                "radius": float(radius),
                "diameter": float(2.0 * radius),
                "radius_mad": float(mad),
                "radius_rel_mad": float(mad / max(abs(radius), 1.0e-9)),
                "observation_count": int(len(g_vals)),
                "support_score": float(support),
                "face_ids": tuple(int(v) for v in sorted({int(fid) for fid in g_fids})),
                "role": "evidence_only_radius_layer",
            }
        )
    layers.sort(key=lambda item: (float(item.get("radius", 0.0)), -float(item.get("support_score", 0.0))))
    return tuple(layers)


def _v18_semantic_layered_feature_candidates(**kwargs: object) -> dict[str, object]:
    """First-pass semantic rewrite of Bore/Chamfer recognition.

    This helper implements the constitution agreed in the debugging discussion:

    * RegionData locates the area only.
    * Selected ring is entry evidence, not automatically the bore radius.
    * Radius layers are explicit evidence objects.
    * Feature identity, surface role ownership, relationships, and rebuild
      authority are separated.

    It is deliberately conservative: it only emits accepted BORE/CHAMFER_FORM
    physical candidates when it can assign non-empty owned face roles.  It never
    treats damage/topology quality as a recognition category.
    """

    faces = np.asarray(kwargs.get("faces"), dtype=np.int64)
    face_ids = tuple_ints(kwargs.get("face_ids", ()))
    face_centroids = np.asarray(kwargs.get("face_centroids"), dtype=float)
    face_normals = np.asarray(kwargs.get("face_normals"), dtype=float)
    region_center = kwargs.get("region_center", (0.0, 0.0, 0.0))
    region_axis = kwargs.get("region_axis", (0.0, 0.0, 1.0))
    region_radius = float(kwargs.get("region_radius", 0.0) or 0.0)
    boundary_loop_geometry = tuple(kwargs.get("boundary_loop_geometry", ()) or ())
    seed_face_ids = tuple_ints(kwargs.get("seed_face_ids", ()))

    valid_face_ids = tuple(fid for fid in face_ids if 0 <= int(fid) < len(face_centroids))
    if not valid_face_ids:
        return {"candidate_data": (), "features": (), "diagnostics": {"v21_semantic_rewrite_used": False, "reason": "no_valid_region_faces"}}

    ids_arr = np.asarray(valid_face_ids, dtype=np.int64)
    pts = np.asarray(face_centroids, dtype=float)[:, :3]
    local_pts = pts[ids_arr, :3]
    if face_normals.shape == pts.shape:
        local_normals = np.asarray(face_normals, dtype=float)[ids_arr, :3]
    else:
        local_normals = np.zeros_like(local_pts)
    normal_len = np.linalg.norm(local_normals, axis=1)
    unit_normals = np.zeros_like(local_normals)
    ok_normal = normal_len > 1.0e-12
    unit_normals[ok_normal] = local_normals[ok_normal] / normal_len[ok_normal].reshape(-1, 1)

    axis_vec = canonical_axis(region_axis if region_axis is not None else (0.0, 0.0, 1.0))
    center_vec = np.asarray(region_center, dtype=float).reshape(3)
    rel = local_pts - center_vec.reshape(1, 3)
    axial = rel @ axis_vec.reshape(3)
    radial_vec = rel - axial.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    radial_dir = np.zeros_like(radial_vec)
    ok_radial = radial > 1.0e-12
    radial_dir[ok_radial] = radial_vec[ok_radial] / radial[ok_radial].reshape(-1, 1)
    normal_axis_abs = np.abs(unit_normals @ axis_vec.reshape(3))
    radial_alignment = np.abs(np.sum(unit_normals * radial_dir, axis=1))
    finite = np.isfinite(axial) & np.isfinite(radial) & np.isfinite(normal_axis_abs) & np.isfinite(radial_alignment)

    # v21 per-face profile evidence.  Centroid radius alone cannot distinguish a
    # cylindrical wall from a chamfer/frustum: both can occupy the same annular
    # interval.  A chamfer is a sloped radius transition, so its triangles show
    # radial change together with axial change in the bore-local frame.  This
    # evidence remains a surface-role heuristic, not feature identity authority.
    vertices_arr = np.asarray(kwargs.get("vertices", ()), dtype=float)
    face_axial_span = np.zeros(len(ids_arr), dtype=float)
    face_radial_span = np.zeros(len(ids_arr), dtype=float)
    face_profile_slope = np.zeros(len(ids_arr), dtype=float)
    face_profile_valid = np.zeros(len(ids_arr), dtype=bool)
    try:
        if (
            vertices_arr.ndim == 2
            and vertices_arr.shape[1] >= 3
            and faces.ndim == 2
            and faces.shape[1] >= 3
            and len(faces) > 0
        ):
            tri_vids = faces[ids_arr, :3].astype(np.int64, copy=False)
            valid_tri = np.all((tri_vids >= 0) & (tri_vids < len(vertices_arr)), axis=1)
            tri_pts = np.zeros((len(ids_arr), 3, 3), dtype=float)
            if np.any(valid_tri):
                tri_pts[valid_tri] = vertices_arr[tri_vids[valid_tri], :3]
                tri_rel = tri_pts - center_vec.reshape(1, 1, 3)
                tri_axial = tri_rel @ axis_vec.reshape(3)
                tri_radial_vec = tri_rel - tri_axial[:, :, None] * axis_vec.reshape(1, 1, 3)
                tri_radial = np.linalg.norm(tri_radial_vec, axis=2)
                face_axial_span[valid_tri] = np.max(tri_axial[valid_tri], axis=1) - np.min(tri_axial[valid_tri], axis=1)
                face_radial_span[valid_tri] = np.max(tri_radial[valid_tri], axis=1) - np.min(tri_radial[valid_tri], axis=1)
                face_profile_slope[valid_tri] = face_radial_span[valid_tri] / np.maximum(face_axial_span[valid_tri], 1.0e-9)
                face_profile_valid[valid_tri] = np.isfinite(face_profile_slope[valid_tri])
    except Exception:
        # Optional profile evidence must never kill recognition.
        face_axial_span = np.zeros(len(ids_arr), dtype=float)
        face_radial_span = np.zeros(len(ids_arr), dtype=float)
        face_profile_slope = np.zeros(len(ids_arr), dtype=float)
        face_profile_valid = np.zeros(len(ids_arr), dtype=bool)

    if not np.any(finite):
        return {"candidate_data": (), "features": (), "diagnostics": {"v21_semantic_rewrite_used": False, "reason": "no_finite_face_measurements"}}

    selected_radius = float(region_radius if region_radius > 1.0e-9 else _finite_percentile(radial[finite], 50.0, 0.0))
    radial_span = float(_finite_percentile(radial[finite], 95.0, selected_radius) - _finite_percentile(radial[finite], 5.0, selected_radius))
    base_tol = max(0.025 * max(selected_radius, 1.0), 0.015 * max(radial_span, 1.0), 1.0e-6)

    # Evidence layer: collect explicit radius layers from cylinder-like face
    # evidence and from measured boundary loops.  These are evidence rows only.
    cylinder_score = np.clip((1.0 - normal_axis_abs) * np.maximum(radial_alignment, 0.0), 0.0, 1.0)
    cylinder_evidence_mask = finite & (normal_axis_abs <= 0.72) & (radial_alignment >= 0.10)
    cylinder_layers = _v17_scalar_layers(
        radial[cylinder_evidence_mask],
        scores=cylinder_score[cylinder_evidence_mask],
        face_ids=ids_arr[cylinder_evidence_mask],
        tolerance=base_tol,
        min_count=3,
    )

    ring_rows: list[dict[str, object]] = [
        {
            "ring_id": "selected_region_frame",
            "source": "region_select.selected_opening_frame",
            "center": to_vector3(center_vec),
            "axis": to_vector3(axis_vec),
            "radius": float(selected_radius),
            "diameter": float(2.0 * selected_radius),
            "axial_position": 0.0,
            "confidence": 0.70 if selected_radius > 0.0 else 0.20,
            "role": "selected_ring_evidence_not_feature_identity",
        }
    ]
    for loop in boundary_loop_geometry:
        try:
            r = float(getattr(loop, "radius", 0.0) or 0.0)
            if r <= 1.0e-9:
                continue
            c = getattr(loop, "center", center_vec)
            ring_rows.append(
                {
                    "ring_id": f"boundary_loop_{int(getattr(loop, 'index', len(ring_rows)))}",
                    "source": "region_data_patch_boundary_loop",
                    "center": to_vector3(c),
                    "axis": to_vector3(getattr(loop, "axis", axis_vec)),
                    "radius": float(r),
                    "diameter": float(2.0 * r),
                    "axial_position": float(getattr(loop, "axial_position", 0.0) or 0.0),
                    "edge_count": int(getattr(loop, "edge_count", 0) or 0),
                    "radius_rel_mad": float(getattr(loop, "radius_rel_mad", 0.0) or 0.0),
                    "confidence": max(0.05, min(0.95, 1.0 - float(getattr(loop, "radius_rel_mad", 0.25) or 0.25) * 5.0)),
                    "role": "boundary_loop_evidence_not_ownership",
                }
            )
        except Exception:
            continue

    # Identity layer: infer the actual wall radius from a true RadiusLayerGraph.
    # v17 accidentally made cylindrical face-normal evidence the only entrance
    # into the graph.  That violated the semantic model: a selected opening ring
    # and measured boundary/internal radius layers are evidence for identity even
    # when wall normals are sparse or fragmented.  v18 therefore builds the graph
    # from three evidence channels:
    #   1. cylindrical face bands     -> strong ownership support
    #   2. all radial face bands      -> density/geometry support, not identity by itself
    #   3. ring evidence              -> opening/radius identity support
    broad_scores = np.clip(0.20 + 0.80 * cylinder_score, 0.0, 1.0)
    broad_layers = _v17_scalar_layers(
        radial[finite],
        scores=broad_scores[finite],
        face_ids=ids_arr[finite],
        tolerance=max(base_tol, 0.018 * max(selected_radius, 1.0)),
        min_count=3,
    )

    layer_candidates: list[dict[str, object]] = []
    for idx, layer in enumerate(cylinder_layers):
        item = dict(layer)
        item["layer_source"] = "cylindrical_face_band"
        item["identity_evidence_role"] = "wall_ownership_support"
        item["source_priority"] = 1.00
        item["source_index"] = int(idx)
        layer_candidates.append(item)
    for idx, layer in enumerate(broad_layers):
        item = dict(layer)
        item["layer_source"] = "radial_face_band"
        item["identity_evidence_role"] = "radius_layer_support_not_wall_authority"
        item["source_priority"] = 0.42
        item["source_index"] = int(idx)
        r = float(item.get("radius", 0.0) or 0.0)
        if any(abs(float(c.get("radius", 0.0) or 0.0) - r) <= max(base_tol, 0.025 * max(abs(r), 1.0)) for c in layer_candidates):
            item["source_priority"] = 0.22
            item["duplicate_of_stronger_cylindrical_layer"] = True
        layer_candidates.append(item)
    for idx, row in enumerate(ring_rows):
        r = float(row.get("radius", 0.0) or 0.0)
        if r <= 1.0e-9:
            continue
        item = {
            "layer_id": f"ring_layer_{idx}",
            "radius": float(r),
            "diameter": float(2.0 * r),
            "radius_mad": float(row.get("radius_rel_mad", 0.0) or 0.0) * max(float(r), 1.0),
            "support_score": float(row.get("confidence", 0.35) or 0.35),
            "observation_count": int(max(1, int(row.get("edge_count", 1) or 1))),
            "face_ids": (),
            "layer_source": str(row.get("source", "ring_evidence")),
            "identity_evidence_role": "opening_ring_identity_support_not_face_ownership",
            "source_priority": 0.55,
            "source_index": int(idx),
            "ring_evidence": dict(row),
        }
        layer_candidates.append(item)

    if not layer_candidates:
        return {
            "candidate_data": (),
            "features": (),
            "diagnostics": {
                "v21_semantic_rewrite_used": False,
                "reason": "no_radius_layer_evidence_from_faces_or_rings",
                "selected_radius": float(selected_radius),
                "ring_evidence": tuple(ring_rows),
                "cylinder_layer_count": int(len(cylinder_layers)),
                "broad_radial_layer_count": int(len(broad_layers)),
            },
        }

    # ------------------------------------------------------------------
    # v21 role eligibility: wall and chamfer are explicit surface profiles.
    #
    # Earlier semantic passes still allowed a broad radial face band to win as
    # ``wall_layer`` and then defined chamfer as the remaining outside band.
    # That is still a layer leak: the chamfer is not a leftover of the bore wall.
    # A BORE wall must prove a vertical cylindrical side profile relative to the
    # opening axis.  A CHAMFER_FORM must independently prove a sloped annular
    # radius-transition profile between two coaxial radius layers.  Only after
    # both roles have been classified do we remove overlap to make ownership
    # disjoint.
    # ------------------------------------------------------------------

    strict_vertical_normal_mask = finite & (normal_axis_abs <= 0.34) & (radial_alignment >= 0.22)
    medium_vertical_normal_mask = finite & (normal_axis_abs <= 0.46) & (radial_alignment >= 0.16)
    sloped_frustum_normal_mask = finite & (normal_axis_abs >= 0.18) & (normal_axis_abs <= 0.88) & (radial_alignment >= 0.10)

    has_inner_layer = bool(
        selected_radius > 0.0
        and any(float(other.get("radius", 0.0) or 0.0) < selected_radius * 0.94 for other in layer_candidates)
    )

    wall_layer_candidates: list[dict[str, object]] = []
    rejected_wall_layer_candidates: list[dict[str, object]] = []

    for layer in layer_candidates:
        r = float(layer.get("radius", 0.0) or 0.0)
        if r <= 1.0e-9:
            continue
        source = str(layer.get("layer_source", "unknown"))
        source_priority = float(layer.get("source_priority", 0.30) or 0.30)
        support = float(layer.get("support_score", 0.0) or 0.0)
        count = int(layer.get("observation_count", 0) or 0)
        selected_delta_rel = abs(r - selected_radius) / max(abs(selected_radius), abs(r), 1.0e-9)
        smaller_than_selected_bonus = 0.35 if (selected_radius > 0.0 and r < selected_radius * 0.94) else 0.0
        selected_radius_penalty = 0.26 if (has_inner_layer and selected_radius > 0.0 and r > selected_radius * 0.97) else 0.0
        very_large_radius_penalty = 0.18 if (selected_radius > 0.0 and r > selected_radius * 1.08) else 0.0

        layer_tol = max(
            base_tol,
            float(layer.get("radius_mad", 0.0) or 0.0) * 3.0,
            0.040 * max(r, 1.0),
            0.010 * max(radial_span, 1.0),
        )
        near_mask = finite & (np.abs(radial - r) <= layer_tol)
        strict_near = near_mask & strict_vertical_normal_mask
        medium_near = near_mask & medium_vertical_normal_mask
        sloped_near = near_mask & sloped_frustum_normal_mask

        strict_count = int(np.count_nonzero(strict_near))
        medium_count = int(np.count_nonzero(medium_near))
        sloped_count = int(np.count_nonzero(sloped_near))
        near_count = int(np.count_nonzero(near_mask))
        if np.any(medium_near):
            axial_span_for_layer = float(_finite_percentile(axial[medium_near], 95.0, 0.0) - _finite_percentile(axial[medium_near], 5.0, 0.0))
        elif np.any(near_mask):
            axial_span_for_layer = float(_finite_percentile(axial[near_mask], 95.0, 0.0) - _finite_percentile(axial[near_mask], 5.0, 0.0))
        else:
            axial_span_for_layer = 0.0
        sloped_ratio = float(sloped_count / max(near_count, 1))
        vertical_ratio = float(medium_count / max(near_count, 1))
        radial_stability = float(1.0 - min(float(layer.get("radius_rel_mad", 0.0) or 0.0) * 8.0, 1.0))

        # A broad radial band is allowed to support identity, but cannot be the
        # wall authority unless the faces near that radius independently prove
        # vertical-cylinder behavior.  This prevents sampled chamfer radii from
        # becoming BORE wall ownership.
        broad_radial_only = bool(source == "radial_face_band" and strict_count < 3)
        ring_only_without_wall_support = bool(source not in {"cylindrical_face_band", "radial_face_band"} and medium_count < 3)
        eligible = bool(
            medium_count >= 3
            and strict_count >= 1
            and vertical_ratio >= 0.08
            and not broad_radial_only
            and not ring_only_without_wall_support
        )
        score = float(
            source_priority
            + 0.52 * np.log1p(max(strict_count, 0))
            + 0.32 * np.log1p(max(medium_count, 0))
            + 0.18 * np.log1p(max(support, 0.0))
            + 0.18 * radial_stability
            + 0.08 * np.log1p(max(axial_span_for_layer, 0.0))
            + smaller_than_selected_bonus
            - 0.32 * sloped_ratio
            - 0.18 * selected_delta_rel
            - selected_radius_penalty
            - very_large_radius_penalty
        )
        layer["wall_role_score"] = score
        layer["wall_role_eligible"] = bool(eligible)
        layer["wall_role_rejection_reason"] = "" if eligible else (
            "broad_radial_band_without_vertical_wall_support" if broad_radial_only else
            "ring_layer_without_wall_support" if ring_only_without_wall_support else
            "insufficient_vertical_cylinder_profile"
        )
        layer["selected_delta_rel"] = float(selected_delta_rel)
        layer["smaller_than_selected_bonus"] = float(smaller_than_selected_bonus)
        layer["selected_radius_penalty"] = float(selected_radius_penalty)
        layer["very_large_radius_penalty"] = float(very_large_radius_penalty)
        layer["wall_profile_metrics"] = {
            "near_face_count": int(near_count),
            "strict_vertical_count": int(strict_count),
            "medium_vertical_count": int(medium_count),
            "sloped_frustum_count": int(sloped_count),
            "vertical_ratio": float(vertical_ratio),
            "sloped_ratio": float(sloped_ratio),
            "axial_span": float(axial_span_for_layer),
            "radial_stability": float(radial_stability),
            "layer_tolerance": float(layer_tol),
        }
        if eligible:
            wall_layer_candidates.append(layer)
        else:
            rejected_wall_layer_candidates.append(layer)

    if wall_layer_candidates:
        wall_layer = max(wall_layer_candidates, key=lambda item: float(item.get("wall_role_score", 0.0)))
    else:
        # Conservative fallback: keep the strongest layer as identity diagnostics
        # only.  Ownership below will likely be sparse, which is preferable to
        # promoting a chamfer sample as a bore wall.
        wall_layer = max(layer_candidates, key=lambda item: float(item.get("wall_role_score", -999.0)))
        wall_layer["wall_role_forced_from_ineligible_layer"] = True

    wall_radius = float(wall_layer.get("radius", selected_radius) or selected_radius)
    wall_tol = max(
        base_tol,
        float(wall_layer.get("radius_mad", 0.0) or 0.0) * 2.5,
        0.040 * max(wall_radius, 1.0),
        0.010 * max(radial_span, 1.0),
    )

    # The outer layer for chamfer recognition may come from the selected ring,
    # boundary/ring evidence, or sloped-face evidence, but it does not define
    # the bore wall.  It only defines the outer side of an annular transition.
    ring_outer = max((float(row.get("radius", 0.0) or 0.0) for row in ring_rows), default=selected_radius)
    sloped_between_seed = sloped_frustum_normal_mask & (radial > wall_radius + 0.20 * wall_tol)
    outer_radius = max(
        selected_radius,
        ring_outer,
        _finite_percentile(radial[sloped_between_seed], 92.0, selected_radius) if np.any(sloped_between_seed) else selected_radius,
    )
    outer_tol = max(base_tol, 0.035 * max(outer_radius, 1.0))
    radius_transition_delta = float(max(0.0, outer_radius - wall_radius))

    # Surface-role layer.  BORE wall ownership is now strict vertical-cylinder
    # role evidence.  No geometry-only fallback is allowed to swallow sloped
    # chamfer faces.
    wall_band_mask = finite & (np.abs(radial - wall_radius) <= wall_tol)
    strict_wall_mask = wall_band_mask & strict_vertical_normal_mask
    medium_wall_mask = wall_band_mask & medium_vertical_normal_mask
    if np.count_nonzero(strict_wall_mask) >= 3:
        wall_mask = strict_wall_mask | (medium_wall_mask & (normal_axis_abs <= 0.42))
    else:
        wall_mask = medium_wall_mask & (normal_axis_abs <= 0.38)
    geometry_only_wall_mask_used = False

    # CHAMFER_FORM is not leftover.  It is a sloped annular transition between
    # two coaxial radius layers: inner cylindrical wall radius -> outer mouth
    # radius.  Therefore the chamfer classifier is built from a radius-changing
    # surface profile, not from "outside wall radius" alone.
    transition_enough_radius = bool(radius_transition_delta > max(0.75 * wall_tol, 0.020 * max(outer_radius, 1.0), 1.0e-6))
    transition_radius_band_mask = (
        finite
        & transition_enough_radius
        & (radial >= wall_radius + max(0.04 * wall_tol, 1.0e-6))
        & (radial <= outer_radius + outer_tol)
    )

    # Independent sloped-profile evidence.  Normal evidence catches clean
    # frustum/chamfer faces.  Per-face vertex profile evidence catches cases
    # where face normals are noisy or radial alignment is weak but the triangle
    # itself clearly changes radius while advancing axially.  Neither evidence
    # path is allowed to define the bore wall.
    min_profile_radial_span = max(0.035 * max(radius_transition_delta, 0.0), 0.10 * base_tol, 1.0e-6)
    min_profile_axial_span = max(0.06 * base_tol, 1.0e-6)
    vertex_slope_profile_mask = (
        face_profile_valid
        & (face_radial_span >= min_profile_radial_span)
        & (face_axial_span >= min_profile_axial_span)
        & (face_profile_slope >= 0.04)
    )
    normal_slope_profile_mask = (
        (normal_axis_abs >= 0.075)
        & (normal_axis_abs <= 0.965)
        & ((radial_alignment >= 0.020) | vertex_slope_profile_mask)
    )
    transition_profile_mask = transition_radius_band_mask & (normal_slope_profile_mask | vertex_slope_profile_mask)

    if np.any(transition_profile_mask):
        transition_radial_span = float(_finite_percentile(radial[transition_profile_mask], 90.0, 0.0) - _finite_percentile(radial[transition_profile_mask], 10.0, 0.0))
        transition_axial_span = float(_finite_percentile(axial[transition_profile_mask], 90.0, 0.0) - _finite_percentile(axial[transition_profile_mask], 10.0, 0.0))
        transition_vertex_profile_count = int(np.count_nonzero(transition_profile_mask & vertex_slope_profile_mask))
        transition_normal_profile_count = int(np.count_nonzero(transition_profile_mask & normal_slope_profile_mask))
        transition_profile_face_count = int(np.count_nonzero(transition_profile_mask))
        radial_fraction = min(1.0, transition_radial_span / max(radius_transition_delta, 1.0e-9))
        axial_presence = min(1.0, max(transition_axial_span, 1.0e-9) / max(base_tol, 1.0e-9))
        evidence_density = min(1.0, transition_profile_face_count / max(8.0, 0.020 * max(len(valid_face_ids), 1)))
        transition_profile_score = float(0.45 * radial_fraction + 0.25 * axial_presence + 0.30 * evidence_density)
    else:
        transition_radial_span = 0.0
        transition_axial_span = 0.0
        transition_vertex_profile_count = 0
        transition_normal_profile_count = 0
        transition_profile_face_count = 0
        transition_profile_score = 0.0

    min_transition_faces = max(6, int(0.006 * max(len(valid_face_ids), 1)))
    transition_band_mask = transition_profile_mask & (
        (transition_profile_score >= 0.10)
        | (np.count_nonzero(transition_profile_mask) >= min_transition_faces)
    )
    # Ownership de-duplication only.  Chamfer identity was already classified by
    # slope/profile evidence above; it is not defined by being non-wall.
    transition_band_mask = transition_band_mask & (~wall_mask)

    wall_ids = tuple(int(fid) for fid in sorted(set(ids_arr[wall_mask].tolist())))
    transition_ids = tuple(int(fid) for fid in sorted(set(ids_arr[transition_band_mask].tolist())))
    support_mask = finite & (radial <= max(outer_radius + outer_tol, wall_radius + wall_tol))
    support_ids = tuple(int(fid) for fid in sorted(set(ids_arr[support_mask].tolist())))
    unowned_ids = tuple(int(fid) for fid in sorted(set(support_ids) - set(wall_ids) - set(transition_ids)))

    if len(wall_ids) < 3 and len(transition_ids) < 6:
        return {
            "candidate_data": (),
            "features": (),
            "diagnostics": {
                "v21_semantic_rewrite_used": False,
                "reason": "semantic_role_assignment_too_sparse",
                "selected_radius": float(selected_radius),
                "wall_radius": float(wall_radius),
                "outer_radius": float(outer_radius),
                "wall_face_count": int(len(wall_ids)),
                "transition_face_count": int(len(transition_ids)),
                "ring_evidence": tuple(ring_rows),
                "radius_layers": tuple(layer_candidates),
                "cylinder_layer_count": int(len(cylinder_layers)),
                "broad_radial_layer_count": int(len(broad_layers)),
                "geometry_only_wall_mask_used": bool(geometry_only_wall_mask_used),
            },
        }

    features: list[dict[str, object]] = []
    axial_min = float(np.min(axial[wall_mask])) if np.any(wall_mask) else float(_finite_percentile(axial[finite], 5.0, 0.0))
    axial_max = float(np.max(axial[wall_mask])) if np.any(wall_mask) else float(_finite_percentile(axial[finite], 95.0, 0.0))
    wall_depth = float(max(0.0, axial_max - axial_min))

    if wall_ids:
        bore_topology = _component_engine_patch_boundary_diagnostics(faces, wall_ids)
        bore_candidate = {
            "candidate_id": "component_engine.v21.borehole.1",
            "feature_id": "component_engine.v21.borehole.1",
            "entity_type": "borehole",
            "feature_kind": "borehole",
            "candidate_scope": "semantic_layered_recognition_feature",
            "display_name": "BOREHOLE — semantic cylindrical wall role",
            "role": "rebuildable_bore_operation",
            "status": "promoted_bore_from_radius_layer_wall_role",
            "promotion_state": "promoted",
            "candidate_action_enabled": True,
            "candidate_action": "rebuild",
            "rebuild_authorized": True,
            "rebuild_gate": "semantic_wall_role_candidate_rebuild_still_validated_later",
            "face_ids": wall_ids,
            "semantic_face_ids": wall_ids,
            "preview_face_ids": wall_ids,
            "display_face_ids": wall_ids,
            "rebuild_face_ids": wall_ids,
            "face_count": int(len(wall_ids)),
            "radius": float(wall_radius),
            "diameter": float(2.0 * wall_radius),
            "selected_frame_radius": float(selected_radius),
            "outer_radius": float(outer_radius),
            "depth": float(wall_depth),
            "height": float(wall_depth),
            "axial_min": float(axial_min),
            "axial_max": float(axial_max),
            "confidence": 0.82,
            "recognition_rule": "v21_vertical_wall_role_assignment",
            "feature_ownership_source": "surface_component_classifier_v21.semantic_layers",
            "feature_ownership_split": "identity_evidence_radius_layers_surface_roles_are_separate",
            "semantic_layer_policy": "feature_identity_is_geometric_surface_ownership_is_role_based_relationships_are_metadata_rebuildability_validates_later",
            "patch_topology": bore_topology,
            "diagnostics": {
                "ring_evidence": tuple(ring_rows),
                "radius_layers": tuple(layer_candidates),
                "cylinder_layer_count": int(len(cylinder_layers)),
                "broad_radial_layer_count": int(len(broad_layers)),
                "geometry_only_wall_mask_used": bool(geometry_only_wall_mask_used),
                "selected_ring_policy": "selected_ring_is_entry_evidence_not_automatic_bore_radius",
                "wall_layer": dict(wall_layer),
                "wall_radius": float(wall_radius),
                "wall_tolerance": float(wall_tol),
                "outer_radius": float(outer_radius),
                "surface_role_assignment": {
                    "bore_wall_face_count": int(len(wall_ids)),
                    "chamfer_transition_face_count": int(len(transition_ids)),
                    "unowned_context_face_count": int(len(unowned_ids)),
                    "identity_support_face_count": int(len(support_ids)),
                    "chamfer_role_definition": "explicit_sloped_radius_transition_between_inner_and_outer_coaxial_layers_not_leftover",
                    "chamfer_profile_score": float(transition_profile_score),
                    "chamfer_profile_face_count": int(transition_profile_face_count),
                    "chamfer_vertex_profile_face_count": int(transition_vertex_profile_count),
                    "chamfer_normal_profile_face_count": int(transition_normal_profile_count),
                },
            },
            **_x1_candidate_contract_fields(
                family=FeatureFamily.BORE,
                stage=RecognitionStage.ACCEPTED_CANDIDATE,
                evidence_kinds=(EvidenceKind.SELECTED_EDGE_LOOP, EvidenceKind.OPENING_RING, EvidenceKind.BORE_WALL_NORMALS, EvidenceKind.RADIUS_CONSISTENCY),
                accepted=True,
                promotion_reasons=("radius_layer_graph", "cylindrical_wall_role", "current_bore_rebuild_path_supported"),
                primitive_axis=axis_vec,
                primitive_radius=wall_radius,
                primitive_depth=wall_depth,
                candidate_id="component_engine.v21.borehole.1",
                face_ids=wall_ids,
                recognition_rule="v21_vertical_wall_role_assignment",
                status="promoted_bore_from_radius_layer_wall_role",
                confidence=0.82,
                diagnostics={
                    "wall_layer": dict(wall_layer),
                    "selected_ring_policy": "selected_ring_is_entry_evidence_not_automatic_bore_radius",
                    "patch_topology": bore_topology,
                },
            ),
        }
        features.append(bore_candidate)

    if transition_ids:
        trans_axial_min = float(np.min(axial[transition_band_mask])) if np.any(transition_band_mask) else 0.0
        trans_axial_max = float(np.max(axial[transition_band_mask])) if np.any(transition_band_mask) else 0.0
        chamfer_topology = _component_engine_patch_boundary_diagnostics(faces, transition_ids)
        chamfer_candidate = {
            "candidate_id": "component_engine.v21.chamfer.1",
            "feature_id": "component_engine.v21.chamfer.1",
            "entity_type": "chamfer",
            "feature_kind": "chamfer",
            "candidate_scope": "semantic_layered_recognition_feature",
            "display_name": "CHAMFER — semantic annular transition role",
            "role": "rebuildable_chamfer_operation",
            "status": "promoted_chamfer_from_radius_transition_role",
            "promotion_state": "promoted",
            "candidate_action_enabled": True,
            "candidate_action": "rebuild",
            "rebuild_authorized": True,
            "rebuild_gate": "semantic_transition_role_candidate_rebuild_still_validated_later",
            "face_ids": transition_ids,
            "semantic_face_ids": transition_ids,
            "preview_face_ids": transition_ids,
            "display_face_ids": transition_ids,
            "rebuild_face_ids": transition_ids,
            "face_count": int(len(transition_ids)),
            "inner_radius": float(wall_radius),
            "outer_radius": float(outer_radius),
            "radius": float(0.5 * (wall_radius + outer_radius)),
            "diameter": float(wall_radius + outer_radius),
            "selected_frame_radius": float(selected_radius),
            "axial_min": float(trans_axial_min),
            "axial_max": float(trans_axial_max),
            "depth": float(max(0.0, trans_axial_max - trans_axial_min)),
            "height": float(max(0.0, trans_axial_max - trans_axial_min)),
            "confidence": 0.76,
            "recognition_rule": "v21_explicit_annular_slope_transition_role_assignment",
            "feature_ownership_source": "surface_component_classifier_v21.semantic_layers",
            "feature_ownership_split": "chamfer_transition_role_is_independent_from_bore_wall_role",
            "semantic_layer_policy": "chamfer_may_support_bore_identity_but_chamfer_faces_belong_to_chamfer_form",
            "patch_topology": chamfer_topology,
            "diagnostics": {
                "ring_evidence": tuple(ring_rows),
                "radius_layers": tuple(layer_candidates),
                "transition_band": {
                    "definition": "explicit_sloped_radius_transition_between_inner_and_outer_coaxial_layers_not_leftover",
                    "inner_radius": float(wall_radius),
                    "outer_radius": float(outer_radius),
                    "radius_transition_delta": float(radius_transition_delta),
                    "transition_face_count": int(len(transition_ids)),
                    "transition_profile_score": float(transition_profile_score),
                    "transition_profile_face_count": int(transition_profile_face_count),
                    "transition_vertex_profile_face_count": int(transition_vertex_profile_count),
                    "transition_normal_profile_face_count": int(transition_normal_profile_count),
                    "transition_radial_span": float(transition_radial_span),
                    "transition_axial_span": float(transition_axial_span),
                    "normal_axis_abs_range": (
                        float(np.min(normal_axis_abs[transition_band_mask])) if np.any(transition_band_mask) else 0.0,
                        float(np.max(normal_axis_abs[transition_band_mask])) if np.any(transition_band_mask) else 0.0,
                    ),
                },
            },
            **_x1_candidate_contract_fields(
                family=FeatureFamily.CHAMFER_FORM,
                stage=RecognitionStage.ACCEPTED_CANDIDATE,
                evidence_kinds=(EvidenceKind.OPENING_RING, EvidenceKind.CHAMFER_BAND, EvidenceKind.RADIUS_STACK),
                accepted=True,
                promotion_reasons=("radius_transition_band", "annular_transition_role", "current_chamfer_rebuild_path_supported"),
                primitive_axis=axis_vec,
                primitive_radius=0.5 * (wall_radius + outer_radius),
                primitive_depth=max(0.0, trans_axial_max - trans_axial_min),
                candidate_id="component_engine.v21.chamfer.1",
                face_ids=transition_ids,
                recognition_rule="v21_explicit_annular_slope_transition_role_assignment",
                status="promoted_chamfer_from_radius_transition_role",
                confidence=0.76,
                diagnostics={
                    "inner_radius": float(wall_radius),
                    "outer_radius": float(outer_radius),
                    "radius_transition_delta": float(radius_transition_delta),
                    "transition_profile_score": float(transition_profile_score),
                    "transition_profile_face_count": int(transition_profile_face_count),
                    "transition_vertex_profile_face_count": int(transition_vertex_profile_count),
                    "transition_normal_profile_face_count": int(transition_normal_profile_count),
                    "selected_ring_policy": "selected_ring_can_be_outer_chamfer_mouth_evidence",
                    "chamfer_role_definition": "sloped_annular_profile_between_radius_layers",
                    "patch_topology": chamfer_topology,
                },
            ),
        }
        features.append(chamfer_candidate)

    # Relationships layer: metadata only, never a feature family.
    adjacency = face_adjacency_for_patch(faces, valid_face_ids)
    relation_map, typed_relation_map, bore_chamfer_pairs = _x1_feature_relationship_graph(features=features, adjacency=adjacency)
    for item in features:
        cid = str(item.get("candidate_id", ""))
        relations = tuple(relation_map.get(cid, ()))
        typed_relations = tuple(typed_relation_map.get(cid, ()))
        item["feature_relationships"] = typed_relations
        item["feature_relationship_count"] = int(len(typed_relations))
        item["relations"] = relations

    diagnostics = {
        "v21_semantic_rewrite_used": True,
        "v21_semantic_pipeline": (
            "neutral_region_input",
            "ring_evidence",
            "radius_layer_graph",
            "surface_role_assignment",
            "candidate_data",
            "relationship_metadata",
        ),
        "semantic_invariants": (
            "RegionData is not a feature",
            "selected ring is not automatically bore radius",
            "radius layers are evidence",
            "feature identity is geometric",
            "surface ownership is role based",
            "relationships are metadata",
            "rebuildability is validated later",
        ),
        "selected_radius": float(selected_radius),
        "wall_radius": float(wall_radius),
        "outer_radius": float(outer_radius),
        "base_tolerance": float(base_tol),
        "wall_tolerance": float(wall_tol),
        "region_face_count": int(len(valid_face_ids)),
        "ring_evidence_count": int(len(ring_rows)),
        "ring_evidence": tuple(ring_rows),
        "radius_layer_count": int(len(layer_candidates)),
        "cylinder_layer_count": int(len(cylinder_layers)),
        "broad_radial_layer_count": int(len(broad_layers)),
        "geometry_only_wall_mask_used": bool(geometry_only_wall_mask_used),
        "radius_layers": tuple(layer_candidates),
        "surface_role_assignment": {
            "bore_wall_face_count": int(len(wall_ids)),
            "chamfer_transition_face_count": int(len(transition_ids)),
            "unowned_context_face_count": int(len(unowned_ids)),
            "identity_support_face_count": int(len(support_ids)),
            "seed_face_ids": tuple(seed_face_ids),
            "chamfer_role_definition": "explicit_sloped_radius_transition_between_inner_and_outer_coaxial_layers_not_leftover",
            "transition_profile_score": float(transition_profile_score),
            "transition_profile_face_count": int(transition_profile_face_count),
            "transition_vertex_profile_face_count": int(transition_vertex_profile_count),
            "transition_normal_profile_face_count": int(transition_normal_profile_count),
            "transition_radial_span": float(transition_radial_span),
            "transition_axial_span": float(transition_axial_span),
        },
        "bore_chamfer_relationship_pairs": tuple(bore_chamfer_pairs),
        "promoted_candidate_count": int(sum(1 for item in features if _x1_rebuild_allowed(item) and bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False))))),
    }
    return {"candidate_data": tuple(features), "features": tuple(features), "diagnostics": diagnostics}


def component_engine_feature_candidates(**kwargs: object) -> dict[str, object]:
    """Public wrapper for the active candidate engine.

    v21 continues the semantic-layer rewrite.  The path builds explicit
    RingEvidence/RadiusLayer/SurfaceRole diagnostics and emits physical
    BORE/CHAMFER_FORM candidates from role-owned faces.  v21 defines chamfer as
    an explicit sloped annular transition profile, not as leftover outside BORE.  The legacy component
    engine is still kept as a fallback and as a source of diagnostic/review
    family rows while the rewrite is validated.
    """

    legacy_result = _component_engine_feature_candidates(**kwargs)
    semantic_result = _v18_semantic_layered_feature_candidates(**kwargs)

    semantic_features = tuple(
        dict(item)
        for item in tuple(semantic_result.get("candidate_data", semantic_result.get("features", ())) or ())
        if isinstance(item, Mapping)
    )
    semantic_promoted = int(
        sum(
            1
            for item in semantic_features
            if _x1_rebuild_allowed(item) and bool(item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
        )
    )

    if semantic_promoted > 0:
        legacy_features = tuple(
            dict(item)
            for item in tuple(legacy_result.get("candidate_data", legacy_result.get("features", ())) or ())
            if isinstance(item, Mapping)
        )
        # Keep only review/diagnostic legacy family rows.  Physical BORE/CHAMFER
        # ownership now comes from the v18 semantic role assignment, not from the
        # old component masks.
        merged: list[dict[str, object]] = list(semantic_features)
        seen_ids = {str(item.get("candidate_id", "")) for item in merged}
        for item in legacy_features:
            is_promoted = bool(_x1_rebuild_allowed(item) and item.get("candidate_action_enabled", item.get("rebuild_authorized", False)))
            if is_promoted:
                continue
            cid = str(item.get("candidate_id", ""))
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            merged.append(item)
        diag = {
            "component_engine_module": "recognition_component_engine",
            "component_engine_version": 18,
            "component_engine_path": "v18_semantic_layers_with_legacy_diagnostics",
            "legacy_component_engine_version": 16,
            "legacy_diagnostics": dict(legacy_result.get("diagnostics", {}) or {}),
            **dict(semantic_result.get("diagnostics", {}) or {}),
            "merged_candidate_count": int(len(merged)),
            "semantic_promoted_candidate_count": int(semantic_promoted),
        }
        return {
            "candidate_data": tuple(merged),
            "features": tuple(merged),
            "diagnostics": diag,
            "promoted_candidate_count": int(semantic_promoted),
        }

    result = legacy_result
    diag = dict(result.get("diagnostics", {}) or {})
    diag["component_engine_module"] = "recognition_component_engine"
    diag["component_engine_version"] = 18
    diag["component_engine_path"] = "legacy_component_engine_fallback_after_v18_semantic_no_promotion"
    diag["v18_semantic_attempt"] = dict(semantic_result.get("diagnostics", {}) or {})
    result["diagnostics"] = diag
    return result


def recognition_result_dict_from_component_features(
    *,
    features: tuple[dict[str, object], ...],
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Public wrapper returning the CandidateResult dictionary shape."""

    result = _recognition_result_dict_from_component_features(features=features, diagnostics=diagnostics)
    result["engine"] = "surface_component_classifier_v21_wall_chamfer_role_profiles"
    return result


__all__ = [
    "component_engine_feature_candidates",
    "recognition_result_dict_from_component_features",
]
