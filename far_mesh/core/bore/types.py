"""Typed Bore data contracts.

These dataclasses are deliberately small and serializable.  They mark the
architecture boundary used by the Bore folder revisit:

    RegionData -> CandidateData -> DeletePatchProposal -> RebuildResult

No class in this module performs feature recognition, topology repair, mesh
mutation, or parameter fitting.  The classes only make ownership explicit so
region selection, recognition, UI rows and rebuild cannot silently reinterpret
one another's dictionaries.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Literal

EdgeKey = tuple[int, int]
Vector3 = tuple[float, float, float]
FeatureKind = Literal[
    "borehole",
    "chamfer",
    "mouth",
    "counterbore",
    "pocket",
    "slot_or_adjustable_bore",
    "elliptic_or_oval_bore",
    "hex_nut_pocket",
    "unclassified",
    "unknown",
]
PromotionState = Literal["promoted", "evidence_only", "rejected", "diagnostic_only"]


class FeatureFamily(str, Enum):
    """X1-inspired feature-family vocabulary for mesh-native Bore recognition.

    These values are recognition vocabulary, not rebuild authorization.  A family
    may be detected for display/review while remaining forbidden from target
    construction until a dedicated mesh-native rebuild path exists.
    """

    BORE = "bore"
    COUNTERBORE = "counterbore"
    STEPPED_BORE_STACK = "stepped_bore_stack"
    CIRCULAR_POCKET = "circular_pocket"
    HEX_NUT_POCKET = "hex_nut_pocket"
    SLOT_OR_ADJUSTABLE_BORE = "slot_or_adjustable_bore"
    ELLIPTIC_OR_OVAL_BORE = "elliptic_or_oval_bore"
    CHAMFER_FORM = "chamfer_form"
    TESSELLATED_BORE_CANDIDATE = "tessellated_bore_candidate"
    TESSELLATED_CHAMFER_BODY = "tessellated_chamfer_body"
    UNKNOWN = "unknown"


class RecognitionStage(str, Enum):
    """Promotion stage separating evidence, review, preview and actionability."""

    DIAGNOSTIC_ONLY = "diagnostic_only"
    REVIEW = "review"
    PROMOTION_PREVIEW = "promotion_preview"
    ACCEPTED_CANDIDATE = "accepted_candidate"


class EvidenceKind(str, Enum):
    """Mesh-native evidence kinds inspired by the X1 evidence-ledger model."""

    SELECTED_EDGE_LOOP = "selected_edge_loop"
    OPENING_RING = "opening_ring"
    OPPOSITE_OPENING = "opposite_opening"
    BORE_WALL_NORMALS = "bore_wall_normals"
    RADIUS_CONSISTENCY = "radius_consistency"
    CHAMFER_BAND = "chamfer_band"
    RADIUS_STACK = "radius_stack"
    NONROUND_LOOP = "nonround_loop"
    HEX_CORNER_PATTERN = "hex_corner_pattern"
    SLOT_ASPECT = "slot_aspect"
    ELLIPSE_FIT = "ellipse_fit"
    SIDE_PAIR = "side_pair"
    FAST_STACK = "fast_stack"
    PROJECTED_RADIUS_ANCHOR = "projected_radius_anchor"


class FeaturePrimitiveKind(str, Enum):
    """Mesh-native primitive descriptors translated from X1/FreeCAD concepts.

    These are non-mutating recognition descriptors.  They represent what shape
    Recognition believes it is seeing, but they are not CAD bodies, not boolean
    cutters, and not rebuild targets.
    """

    CYLINDER_AXIS = "cylinder_axis"
    CIRCULAR_OPENING = "circular_opening"
    ANNULAR_CHAMFER_BAND = "annular_chamfer_band"
    RADIUS_STACK = "radius_stack"
    NONROUND_LOOP_PROFILE = "nonround_loop_profile"
    TESSELLATED_SIDE_PAIR = "tessellated_side_pair"
    UNKNOWN = "unknown"


class FeatureRelationshipKind(str, Enum):
    """Relationship vocabulary for physical feature objects.

    Relationships are composition/evidence metadata only.  They are not feature
    families, not assembly classifications, not rebuild permissions, and not
    DeletePatchProposal authority.  Example: a BORE adjacent to a CHAMFER remains
    two separate feature objects linked by a relationship row.
    """

    ADJACENT_SURFACE_COMPONENT = "adjacent_surface_component"
    BORE_CHAMFER_ADJACENCY = "bore_chamfer_adjacency"
    SAME_AXIS_OR_CENTERLINE = "same_axis_or_centerline"
    RADIUS_STACK_MEMBER = "radius_stack_member"
    POSSIBLE_COUNTERBORE_STACK = "possible_counterbore_stack"
    REVIEW_ONLY_ASSEMBLY_RELATIONSHIP = "review_only_assembly_relationship"


def enum_value(value: object) -> str:
    """Return a stable public string for enum-like values."""

    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def tuple_enum_values(values: Iterable[object] | object) -> tuple[str, ...]:
    """Return stable string values for an iterable of enum-like objects."""

    try:
        raw = tuple(values or ())
    except Exception:
        return ()
    return tuple(str(enum_value(v)) for v in raw if str(enum_value(v)))


@dataclass(frozen=True, slots=True)
class FeatureEvidenceItem:
    """One mesh-native evidence item supporting or rejecting a feature family.

    This is the small typed equivalent of X1's evidence-ledger rows.  It is
    diagnostic metadata only: it never authorizes deletion, rebuild, viewport
    actions, or mesh mutation by itself.
    """

    evidence_kind: EvidenceKind | str
    role: str = "supporting"
    source: str = "recognition_component_engine"
    confidence: float = 0.0
    description: str = ""
    face_ids: tuple[int, ...] = ()
    edge_ids: tuple[int, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "feature_evidence_item",
            "evidence_kind": enum_value(self.evidence_kind),
            "role": str(self.role),
            "source": str(self.source),
            "confidence": float(self.confidence),
            "description": str(self.description),
            "face_ids": tuple(int(v) for v in self.face_ids),
            "face_count": int(len(self.face_ids)),
            "edge_ids": tuple(int(v) for v in self.edge_ids),
            "edge_count": int(len(self.edge_ids)),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class FeaturePrimitiveData:
    """One mesh-native primitive descriptor associated with a CandidateData row.

    This is the bridge between the X1 macro's CAD-native representatives
    (cylinders, circular wires, non-round profiles, side-pair evidence) and FAR
    MESH's mesh-native world.  It describes a primitive; it does not create one,
    execute booleans, authorize deletion, or mutate mesh geometry.
    """

    primitive_kind: FeaturePrimitiveKind | str
    source: str = "recognition_component_engine"
    role: str = "diagnostic_descriptor"
    center: Vector3 | None = None
    axis: Vector3 | None = None
    radius: float | None = None
    diameter: float | None = None
    depth: float | None = None
    confidence: float = 0.0
    face_ids: tuple[int, ...] = ()
    edge_ids: tuple[int, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "feature_primitive_data",
            "primitive_kind": enum_value(self.primitive_kind),
            "source": str(self.source),
            "role": str(self.role),
            "center": self.center,
            "axis": self.axis,
            "radius": self.radius,
            "diameter": self.diameter,
            "depth": self.depth,
            "confidence": float(self.confidence),
            "face_ids": tuple(int(v) for v in self.face_ids),
            "face_count": int(len(self.face_ids)),
            "edge_ids": tuple(int(v) for v in self.edge_ids),
            "edge_count": int(len(self.edge_ids)),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class FeatureRelationshipData:
    """Typed relation between two independent CandidateData feature objects.

    This is the corrected replacement for assembly-as-family mistakes.  A
    relationship may say that a bore touches a chamfer or that two radius levels
    appear to form a stack, but it never changes the identity of either feature
    and never authorizes rebuild by itself.
    """

    relationship_kind: FeatureRelationshipKind | str
    source_candidate_id: str
    target_candidate_id: str
    source_feature_family: FeatureFamily | str = FeatureFamily.UNKNOWN
    target_feature_family: FeatureFamily | str = FeatureFamily.UNKNOWN
    role: str = "composition_metadata_only"
    confidence: float = 0.0
    source_face_ids: tuple[int, ...] = ()
    target_face_ids: tuple[int, ...] = ()
    relation_face_pairs: int = 0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "feature_relationship_data",
            "relationship_kind": enum_value(self.relationship_kind),
            "source_candidate_id": str(self.source_candidate_id),
            "target_candidate_id": str(self.target_candidate_id),
            "source_feature_family": enum_value(self.source_feature_family),
            "target_feature_family": enum_value(self.target_feature_family),
            "role": str(self.role),
            "confidence": float(self.confidence),
            "source_face_ids": tuple(int(v) for v in self.source_face_ids),
            "source_face_count": int(len(self.source_face_ids)),
            "target_face_ids": tuple(int(v) for v in self.target_face_ids),
            "target_face_count": int(len(self.target_face_ids)),
            "relation_face_pairs": int(self.relation_face_pairs),
            "diagnostics": {
                "feature_relationship_policy": "relationship_metadata_only_not_feature_family_not_rebuild_authority",
                **dict(self.diagnostics or {}),
            },
        }


X1_FREECAD_TO_FAR_MESH_DICTIONARY: tuple[dict[str, str], ...] = (
    {
        "x1_freecad_concept": "Wire",
        "far_mesh_equivalent": "ordered mesh edge loop / boundary-loop evidence",
        "contract_layer": "RegionData.loop_edges or CandidateData.boundary_loops",
    },
    {
        "x1_freecad_concept": "InnerWire",
        "far_mesh_equivalent": "inner boundary loop / opening loop inside a RegionData cutout",
        "contract_layer": "RegionData seed/opening evidence",
    },
    {
        "x1_freecad_concept": "Part.Circle / analytic circle edge",
        "far_mesh_equivalent": "fitted circular loop evidence from selected mesh vertices",
        "contract_layer": "BoreOpeningMeasurement / FeaturePrimitiveData.circular_opening",
    },
    {
        "x1_freecad_concept": "Part.Cylinder / analytic cylindrical face",
        "far_mesh_equivalent": "cylindrical wall-band evidence from normals, radius consistency, and axial span",
        "contract_layer": "FeaturePrimitiveData.cylinder_axis",
    },
    {
        "x1_freecad_concept": "B-Rep face",
        "far_mesh_equivalent": "connected mesh surface patch with boundary and topology diagnostics",
        "contract_layer": "CandidateData.semantic_face_ids",
    },
    {
        "x1_freecad_concept": "Boolean cutter",
        "far_mesh_equivalent": "DeletePatchProposal plus rebuild primitive, validated later by rebuild.py",
        "contract_layer": "DeletePatchProposal",
    },
    {
        "x1_freecad_concept": "X1 feature tree / evidence ledger",
        "far_mesh_equivalent": "FeatureEvidenceLedger and FeaturePrimitiveData attached to CandidateData",
        "contract_layer": "CandidateData.x1_evidence_ledger / feature_primitives",
    },
)


@dataclass(frozen=True, slots=True)
class FeatureEvidenceLedger:
    """Per-candidate X1-style evidence ledger.

    The ledger explains *why* Recognition emitted a family/stage and *why* the
    target policy may or may not permit rebuild.  It is intentionally separate
    from RegionData and DeletePatchProposal: display may show it, Recognition may
    use it for diagnostics, and rebuild_target may read the stage/family gate,
    but the ledger itself is not an action object.
    """

    candidate_id: str
    feature_family: FeatureFamily | str
    recognition_stage: RecognitionStage | str
    evidence_items: tuple[FeatureEvidenceItem, ...] = ()
    evidence_kinds: tuple[EvidenceKind | str, ...] = ()
    feature_primitives: tuple[FeaturePrimitiveData, ...] = ()
    feature_relationships: tuple[FeatureRelationshipData, ...] = ()
    promotion_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    target_policy_allowed: bool = False
    target_policy_reason: str = ""
    primitive_axis: Vector3 | None = None
    primitive_radius: float | None = None
    primitive_depth: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "feature_evidence_ledger",
            "candidate_id": str(self.candidate_id),
            "feature_family": enum_value(self.feature_family),
            "recognition_stage": enum_value(self.recognition_stage),
            "evidence_kinds": tuple_enum_values(self.evidence_kinds),
            "evidence_items": tuple(item.to_dict() for item in self.evidence_items),
            "evidence_item_count": int(len(self.evidence_items)),
            "feature_primitives": tuple(item.to_dict() for item in self.feature_primitives),
            "feature_primitive_count": int(len(self.feature_primitives)),
            "feature_relationships": tuple(item.to_dict() for item in self.feature_relationships),
            "feature_relationship_count": int(len(self.feature_relationships)),
            "promotion_reasons": tuple(str(v) for v in self.promotion_reasons),
            "rejection_reasons": tuple(str(v) for v in self.rejection_reasons),
            "target_policy_allowed": bool(self.target_policy_allowed),
            "target_policy_reason": str(self.target_policy_reason),
            "primitive_axis": self.primitive_axis,
            "primitive_radius": self.primitive_radius,
            "primitive_depth": self.primitive_depth,
            "diagnostics": dict(self.diagnostics or {}),
        }


def tuple_ints(values: Iterable[object] | object) -> tuple[int, ...]:
    """Return stable sorted unique non-negative integer IDs."""

    try:
        return tuple(sorted({int(v) for v in tuple(values or ()) if int(v) >= 0}))
    except Exception:
        return ()


def tuple_edges(values: Iterable[object] | object) -> tuple[EdgeKey, ...]:
    """Return stable normalized edge keys."""

    out: set[EdgeKey] = set()
    try:
        raw = tuple(values or ())
    except Exception:
        return ()
    for item in raw:
        try:
            a, b = tuple(item)[:2]  # type: ignore[arg-type]
            ia = int(a)
            ib = int(b)
        except Exception:
            continue
        if ia == ib:
            continue
        out.add((ia, ib) if ia < ib else (ib, ia))
    return tuple(sorted(out))


def vector3(value: object, default: Vector3 = (0.0, 0.0, 0.0)) -> Vector3:
    """Convert a tuple/list/array-like value into a 3-tuple of floats."""

    try:
        seq = tuple(value)  # type: ignore[arg-type]
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return default


@dataclass(frozen=True, slots=True)
class RegionData:
    """Neutral mesh region selected from the mesh: WHERE Recognition should look.

    ``face_ids`` are RegionData faces for Recognition input only. They are not a
    recognized Bore region and not a rebuild/delete target.

    ``region_preview_face_ids`` is the display-only preview of the neutral
    RegionData cutout. Candidate previews must come from Recognition.
    """

    edge_ids: tuple[int, ...]
    face_ids: tuple[int, ...]
    loop_edges: tuple[EdgeKey, ...]
    loop_vertices: tuple[int, ...]
    center: Vector3
    axis: Vector3
    radius: float
    seed_face_ids: tuple[int, ...] = ()
    region_preview_face_ids: tuple[int, ...] = ()
    derived_boundary_loops: tuple[tuple[EdgeKey, ...], ...] = ()
    derived_opposite_rim_edge_ids: tuple[EdgeKey, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        face_ids = tuple(int(v) for v in self.face_ids)
        return {
            "contract_type": "region_data",
            "semantic_role": "region_data_only",
            "edge_ids": tuple(int(v) for v in self.edge_ids),
            "region_face_ids": face_ids,
            "face_ids": face_ids,
            "face_count": int(len(face_ids)),
            "loop_edges": self.loop_edges,
            "loop_vertices": tuple(int(v) for v in self.loop_vertices),
            "center": self.center,
            "axis": self.axis,
            "radius": float(self.radius),
            "diameter": float(2.0 * self.radius),
            "seed_face_ids": tuple(int(v) for v in self.seed_face_ids),
            "seed_face_count": int(len(self.seed_face_ids)),
            "region_preview_face_ids": tuple(int(v) for v in self.region_preview_face_ids),
            "region_preview_face_count": int(len(self.region_preview_face_ids)),
            "derived_boundary_loops": self.derived_boundary_loops,
            "derived_boundary_loop_count": int(len(self.derived_boundary_loops)),
            "derived_opposite_rim_edge_ids": self.derived_opposite_rim_edge_ids,
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class RegionEvidenceLedger:
    """Neutral RegionData ledger consumed by the recognition engine."""

    region_data: RegionData
    measured_face_ids: tuple[int, ...]
    measured_boundary_loops: tuple[tuple[EdgeKey, ...], ...] = ()
    recognition_context_face_ids: tuple[int, ...] = ()
    recognition_context_boundary_loops: tuple[tuple[EdgeKey, ...], ...] = ()
    feature_patch_measurement: Mapping[str, object] = field(default_factory=dict)
    feature_layer_analysis: Mapping[str, object] = field(default_factory=dict)
    macro_family_ledger: Mapping[str, object] = field(default_factory=dict)
    recognition_contract: Mapping[str, object] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "region_evidence_ledger",
            "pipeline_stage": "evidence_ledger",
            "region_data": self.region_data.to_dict(),
            "measured_face_ids": tuple(int(v) for v in self.measured_face_ids),
            "measured_face_count": int(len(self.measured_face_ids)),
            "measured_boundary_loop_count": int(len(self.measured_boundary_loops)),
            "measured_boundary_loop_edge_counts": tuple(int(len(loop)) for loop in self.measured_boundary_loops),
            "recognition_context_face_ids": tuple(int(v) for v in self.recognition_context_face_ids),
            "recognition_context_face_count": int(len(self.recognition_context_face_ids)),
            "recognition_context_boundary_loop_count": int(len(self.recognition_context_boundary_loops)),
            "recognition_context_boundary_loop_edge_counts": tuple(int(len(loop)) for loop in self.recognition_context_boundary_loops),
            "feature_patch_measurement": dict(self.feature_patch_measurement or {}),
            "feature_layer_analysis": dict(self.feature_layer_analysis or {}),
            "macro_family_ledger": dict(self.macro_family_ledger or {}),
            "recognition_contract": dict(self.recognition_contract or {}),
            "conflicts": tuple(str(v) for v in self.conflicts),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class CandidateData:
    """Candidate data produced by Recognition: WHAT was found in RegionData."""

    feature_id: str
    feature_kind: FeatureKind
    promotion_state: PromotionState
    candidate_action_enabled: bool
    semantic_face_ids: tuple[int, ...]
    boundary_loops: tuple[tuple[EdgeKey, ...], ...] = ()
    role: str = "inspection_only"
    status: str = ""
    confidence: float = 0.0
    source_evidence: tuple[str, ...] = ()
    feature_family: FeatureFamily | str = FeatureFamily.UNKNOWN
    recognition_stage: RecognitionStage | str = RecognitionStage.DIAGNOSTIC_ONLY
    evidence_kinds: tuple[EvidenceKind | str, ...] = ()
    promotion_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    primitive_axis: Vector3 | None = None
    primitive_radius: float | None = None
    primitive_depth: float | None = None
    feature_primitives: tuple[FeaturePrimitiveData, ...] = ()
    feature_relationships: tuple[FeatureRelationshipData, ...] = ()
    x1_evidence_ledger: Mapping[str, object] = field(default_factory=dict)
    display_face_ids: tuple[int, ...] = ()
    rebuild_face_ids: tuple[int, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "candidate_data",
            "pipeline_stage": "recognition",
            "feature_id": self.feature_id,
            "feature_kind": self.feature_kind,
            "entity_type": self.feature_kind,
            "promotion_state": self.promotion_state,
            "candidate_action_enabled": bool(self.candidate_action_enabled),
            "rebuild_authorized": bool(self.candidate_action_enabled),
            "semantic_face_ids": tuple(int(v) for v in self.semantic_face_ids),
            "face_ids": tuple(int(v) for v in self.semantic_face_ids),
            "face_count": int(len(self.semantic_face_ids)),
            "boundary_loop_count": int(len(self.boundary_loops)),
            "boundary_loop_edge_counts": tuple(int(len(loop)) for loop in self.boundary_loops),
            "role": self.role,
            "status": self.status,
            "confidence": float(self.confidence),
            "source_evidence": tuple(str(v) for v in self.source_evidence),
            "feature_family": enum_value(self.feature_family),
            "recognition_stage": enum_value(self.recognition_stage),
            "evidence_kinds": tuple_enum_values(self.evidence_kinds),
            "promotion_reasons": tuple(str(v) for v in self.promotion_reasons),
            "rejection_reasons": tuple(str(v) for v in self.rejection_reasons),
            "primitive_axis": self.primitive_axis,
            "primitive_radius": self.primitive_radius,
            "primitive_depth": self.primitive_depth,
            "feature_primitives": tuple(item.to_dict() for item in self.feature_primitives),
            "feature_primitive_count": int(len(self.feature_primitives)),
            "feature_relationships": tuple(item.to_dict() for item in self.feature_relationships),
            "feature_relationship_count": int(len(self.feature_relationships)),
            "x1_evidence_ledger": dict(self.x1_evidence_ledger or {}),
            "display_face_ids": tuple(int(v) for v in (self.display_face_ids or self.semantic_face_ids)),
            "rebuild_face_ids": tuple(int(v) for v in self.rebuild_face_ids),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Recognition result: CandidateData collection plus source RegionData ledger."""

    candidates: tuple[CandidateData, ...]
    ledger: RegionEvidenceLedger
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "candidate_result",
            "pipeline_stage": "recognition_result",
            "candidate_data": tuple(item.to_dict() for item in self.candidates),
            "candidate_count": int(len(self.candidates)),
            "promoted_candidate_count": int(sum(1 for item in self.candidates if item.promotion_state == "promoted")),
            "ledger": self.ledger.to_dict(),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True, slots=True)
class DeletePatchProposal:
    """Delete-patch proposal consumed by rebuild.py; not a Recognition object."""

    target_id: str
    feature_kind: FeatureKind
    semantic_face_ids: tuple[int, ...]
    delete_patch_face_ids: tuple[int, ...]
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    protected_rim_edges: tuple[EdgeKey, ...] = ()
    allowed_bridge_face_ids: tuple[int, ...] = ()
    forbidden_face_ids: tuple[int, ...] = ()
    allow_unequal_loop_transition: bool = True
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_type": "delete_patch_proposal",
            "pipeline_stage": "rebuild_target",
            "target_id": self.target_id,
            "feature_kind": self.feature_kind,
            "semantic_face_ids": tuple(int(v) for v in self.semantic_face_ids),
            "semantic_face_count": int(len(self.semantic_face_ids)),
            "delete_patch_face_ids": tuple(int(v) for v in self.delete_patch_face_ids),
            "delete_patch_face_count": int(len(self.delete_patch_face_ids)),
            "protected_loop_pair": self.protected_loop_pair,
            "protected_loop_pair_available": self.protected_loop_pair is not None,
            "protected_rim_edges": self.protected_rim_edges,
            "protected_rim_edge_count": int(len(self.protected_rim_edges)),
            "allowed_bridge_face_ids": tuple(int(v) for v in self.allowed_bridge_face_ids),
            "allowed_bridge_face_count": int(len(self.allowed_bridge_face_ids)),
            "forbidden_face_ids": tuple(int(v) for v in self.forbidden_face_ids),
            "forbidden_face_count": int(len(self.forbidden_face_ids)),
            "allow_unequal_loop_transition": bool(self.allow_unequal_loop_transition),
            "parameter_fit_used_for_target": False,
            "radius_used_for_delete_expansion": False,
            "axis_used_for_delete_expansion": False,
            "diagnostics": dict(self.diagnostics or {}),
        }
