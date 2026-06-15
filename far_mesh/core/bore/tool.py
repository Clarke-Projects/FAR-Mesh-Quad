"""High-level BoreTool core facade.

This module is the single BoreTool logic boundary used by the GUI. It keeps
``bore_actions.py`` display-only and prevents a second GUI controller file from
being created.

Clean flow
----------
    selected edge IDs -> RegionData -> CandidateData -> display DTOs -> bore_actions.py display

``bore_actions.py`` receives only ready-to-display DTOs and face IDs for
preview. All interpretation of raw recognition dictionaries and all Bore action
flow live here inside the core BoreTool boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any

import math
import os

from .region_select import faces_inside_boundary, select_region_data
from .recognition import recognize_bore_region_selection
from .rebuild import delete_and_rebuild_candidate_region
from .types import tuple_ints

RGBA = tuple[int, int, int, int]
DEFAULT_REBUILT_FACE_COLOR: RGBA = (0, 213, 255, 255)


@dataclass(frozen=True, slots=True)
class BoreCandidateView:
    """A UI-safe Bore candidate view.

    This is the only candidate shape the display layer should consume.  It is
    already resolved into explicit display, status, and rebuild fields.
    """

    candidate_id: str
    feature_id: str
    entity_type: str
    feature_kind: str
    feature_family: str
    recognition_stage: str
    label: str
    table_object: str
    table_faces: str
    table_geometry: str
    table_role: str
    description: str
    display_face_ids: tuple[int, ...]
    rebuild_face_ids: tuple[int, ...]
    can_preview: bool
    can_rebuild: bool
    rebuild_disabled_reason: str = ""
    rebuild_token: Mapping[str, object] = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return int(len(self.display_face_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "feature_id": self.feature_id,
            "entity_type": self.entity_type,
            "feature_kind": self.feature_kind,
            "feature_family": self.feature_family,
            "recognition_stage": self.recognition_stage,
            "label": self.label,
            "table_object": self.table_object,
            "table_faces": self.table_faces,
            "table_geometry": self.table_geometry,
            "table_role": self.table_role,
            "description": self.description,
            "display_face_ids": self.display_face_ids,
            "preview_face_ids": self.display_face_ids,
            "rebuild_face_ids": self.rebuild_face_ids,
            "can_preview": bool(self.can_preview),
            "can_rebuild": bool(self.can_rebuild),
            "rebuild_disabled_reason": self.rebuild_disabled_reason,
            "rebuild_token": dict(self.rebuild_token or {}),
        }


@dataclass(frozen=True, slots=True)
class BoreToolDisplayResult:
    """Complete read-only payload for the Bore display."""

    selected_edge_ids: tuple[int, ...]
    normalized_edge_ids: tuple[int, ...]
    region_face_ids: tuple[int, ...]
    seed_face_ids: tuple[int, ...]
    region_preview_face_ids: tuple[int, ...]
    candidates: tuple[BoreCandidateView, ...]
    diagnostics: Mapping[str, object]
    analysis_text: str
    preview_text: str
    status_text: str
    boundary_status_text: str
    selected_candidate_id: str = ""

    @property
    def candidate_count(self) -> int:
        return int(len(self.candidates))

    def candidate_by_id(self, candidate_id: str) -> BoreCandidateView | None:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class BoreInsideBoundaryPreview:
    """Display payload for the legacy cap/interior selection preview."""

    selected_edge_ids: tuple[int, ...]
    face_ids: tuple[int, ...]
    analysis_text: str
    status_text: str


# -----------------------------------------------------------------------------
# Public BoreTool facade
# -----------------------------------------------------------------------------


def analyze_bore_candidates(mesh: object, edge_ids: Iterable[int]) -> BoreToolDisplayResult:
    """Run the BoreTool analysis pipeline and return display-ready data.

    This is the only analysis function the GUI controller needs for candidate
    listing.  It owns the sequence:

    ``region_select -> recognition -> candidate-view normalization``.
    """

    selected_edge_ids = _tuple_ints_keep_order(edge_ids)
    if not selected_edge_ids:
        raise ValueError("No selected Bore rim/opening edges.")

    # Stage A — Selection/Region Select is a closed neutral volume-cutout
    # entity.  It must produce a displayable selected volume even if Recognition
    # later fails.  This is the first visual contract of the BoreTool:
    # "what local mesh volume did the picked rim/opening select?"
    region = select_region_data(mesh, selected_edge_ids)
    region_diag = dict(getattr(region, "diagnostics", {}) or {})

    region_face_ids = tuple_ints(getattr(region, "face_ids", ()))
    seed_face_ids = tuple_ints(getattr(region, "seed_face_ids", ()))
    # For the neutral selector, region_preview_face_ids is the selected volume
    # cutout.  Never replace it with candidate faces.
    preliminary_diagnostics: dict[str, object] = dict(region_diag)
    region_preview_face_ids = _region_preview_face_ids_from_region(region, preliminary_diagnostics)
    normalized_edge_ids = _normalized_edge_ids_from_diagnostics(preliminary_diagnostics, selected_edge_ids)

    # Stage B — Recognition consumes the neutral cutout and may produce feature
    # candidates.  Recognition is allowed to fail without breaking the selected
    # volume preview.  The GUI must still be able to visualize the selection
    # result independently from candidates.
    recognition_diag: dict[str, object] = {}
    recognition_failed = False
    try:
        recognition = recognize_bore_region_selection(mesh, region)
        recognition_diag = dict(recognition or {})
    except Exception as exc:
        recognition_failed = True
        recognition_diag = {
            "pipeline_stage": "recognition_failed_after_neutral_volume_selection",
            "recognition_failed": True,
            "recognition_error": str(exc),
            "recognition_features": (),
            "recognition_engine_features": (),
            "promoted_feature_candidates": (),
            "rebuild_ready": False,
            "rebuild_block_reason": "recognition_failed_but_selection_volume_is_displayable",
            "active_candidate_authority": "recognition_unavailable",
        }

    diagnostics: dict[str, object] = {
        **region_diag,
        **recognition_diag,
        "selection_preview_contract": "neutral_volume_cutout_preview_is_independent_of_recognition",
        "selection_preview_face_count": int(len(region_preview_face_ids)),
        "selection_preview_source": "region_select.region_preview_face_ids",
        "recognition_failed": bool(recognition_failed or bool(recognition_diag.get("recognition_failed", False))),
    }

    # Re-read normalized IDs after merged diagnostics, but keep region preview
    # strictly from region_select output.
    normalized_edge_ids = _normalized_edge_ids_from_diagnostics(diagnostics, selected_edge_ids)
    candidates = _candidate_views_from_recognition(diagnostics)
    selected_candidate_id = candidates[0].candidate_id if candidates else ""

    routed_text = _routed_outputs_text(
        selected_edge_ids=selected_edge_ids,
        normalized_edge_ids=normalized_edge_ids,
        region_face_ids=region_face_ids,
        seed_face_ids=seed_face_ids,
        region_preview_face_ids=region_preview_face_ids,
        candidates=candidates,
        diagnostics=diagnostics,
    )
    analysis_text = _analysis_text(
        selected_edge_ids=selected_edge_ids,
        normalized_edge_ids=normalized_edge_ids,
        region_face_ids=region_face_ids,
        seed_face_ids=seed_face_ids,
        axis=getattr(region, "axis", "-"),
        radius=getattr(region, "radius", "-"),
        diagnostics=diagnostics,
    ) + "\n\n" + routed_text
    recognition_status = "failed" if bool(diagnostics.get("recognition_failed", False)) else "complete"
    preview_text = (
        "Neutral volume cutout selected.\n"
        f"RegionData faces: {len(region_face_ids)}\n"
        f"Selection preview faces: {len(region_preview_face_ids)}\n"
        f"Recognition status: {recognition_status}\n"
        f"Candidates emitted by Recognition: {len(candidates)}\n"
        "Display receives already-normalized DTOs from BoreTool.\n"
        "Selection preview and candidate preview are separate visual states.\n\n"
        + routed_text
    )
    rebuildable = sum(1 for item in candidates if item.can_rebuild)
    if bool(diagnostics.get("recognition_failed", False)):
        status_text = f"Neutral volume selected: {len(region_preview_face_ids)} faces. Recognition failed; candidate list unavailable."
    else:
        status_text = f"Neutral volume selected: {len(region_preview_face_ids)} faces; {len(candidates)} candidate(s), {rebuildable} rebuild-authorized."
    boundary_status_text = "Selection volume preview is independent of Recognition; display layer only renders returned DTOs."

    return BoreToolDisplayResult(
        selected_edge_ids=selected_edge_ids,
        normalized_edge_ids=normalized_edge_ids,
        region_face_ids=region_face_ids,
        seed_face_ids=seed_face_ids,
        region_preview_face_ids=region_preview_face_ids,
        candidates=candidates,
        diagnostics=diagnostics,
        analysis_text=analysis_text,
        preview_text=preview_text,
        status_text=status_text,
        boundary_status_text=boundary_status_text,
        selected_candidate_id=selected_candidate_id,
    )


def preview_faces_inside_boundary(mesh: object, edge_ids: Iterable[int]) -> BoreInsideBoundaryPreview:
    """Return display-ready face IDs for the legacy closed-loop interior preview."""

    selected_edge_ids = _tuple_ints_keep_order(edge_ids)
    if not selected_edge_ids:
        raise ValueError("No selected Bore boundary edges.")
    face_ids = tuple_ints(faces_inside_boundary(mesh, selected_edge_ids))
    if not face_ids:
        raise ValueError("No faces were found inside the selected Bore boundary.")
    return BoreInsideBoundaryPreview(
        selected_edge_ids=selected_edge_ids,
        face_ids=face_ids,
        analysis_text=(
            f"Boundary edges: {len(selected_edge_ids)}\n"
            f"Selected faces: {len(face_ids)}\n"
            "Status: Success"
        ),
        status_text=f"Selected {len(face_ids)} interior faces.",
    )


def rebuild_bore_candidate(
    mesh: object,
    *,
    edge_ids: Iterable[int],
    candidate: BoreCandidateView | Mapping[str, object],
    quad_density_mode: str = "lean_pi_opening",
    color_rebuilt_faces: bool = True,
    rebuilt_face_color: RGBA = DEFAULT_REBUILT_FACE_COLOR,
) -> object:
    """Dispatch the selected candidate to the Bore rebuild pipeline.

    Target expansion/repair remains inside ``rebuild_target.py`` and final
    topology validation remains inside ``rebuild.py``.
    """

    selected_edge_ids = _tuple_ints_keep_order(edge_ids)
    if not selected_edge_ids:
        raise ValueError("No cached selected Bore rim/opening edges for rebuild.")

    view = _candidate_view_from_any(candidate)
    if view is None:
        raise ValueError("No Bore candidate view was provided for rebuild.")
    if not view.can_rebuild:
        reason = view.rebuild_disabled_reason or "BoreTool marked this candidate as preview-only."
        raise ValueError(reason)
    if not view.rebuild_face_ids:
        raise ValueError("Bore candidate has no rebuild input face IDs.")

    metadata = dict(view.rebuild_token or {})
    metadata.setdefault("candidate_id", view.candidate_id)
    metadata.setdefault("feature_id", view.feature_id)
    metadata.setdefault("entity_type", view.entity_type)
    metadata.setdefault("feature_kind", view.feature_kind)
    metadata.setdefault("feature_family", view.feature_family)
    metadata.setdefault("recognition_stage", view.recognition_stage)
    metadata.setdefault("display_face_ids", view.display_face_ids)
    metadata.setdefault("preview_face_ids", view.display_face_ids)
    metadata.setdefault("rebuild_face_ids", view.rebuild_face_ids)
    metadata["quad_density_mode"] = quad_density_mode
    metadata["rebuild_density_mode"] = quad_density_mode

    # Copy only when the mesh object supports it; callers may already provide a
    # defensive copy.  This helper remains non-mutating with respect to the
    # caller's active mesh object.
    try:
        mesh_for_rebuild = mesh.copy()
    except Exception:
        mesh_for_rebuild = mesh

    return delete_and_rebuild_candidate_region(
        mesh_for_rebuild,
        selected_edge_ids,
        region_face_ids=view.rebuild_face_ids,
        feature_candidate_metadata=metadata,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        allow_diagnostic_preview_rebuild=False,
        quad_density_mode=str(quad_density_mode or "lean_pi_opening"),
        rebuilt_face_color=rebuilt_face_color,
        isolate_rebuilt_vertices_for_color=False,
    )


# -----------------------------------------------------------------------------
# Candidate normalization owned by BoreTool, not by display
# -----------------------------------------------------------------------------


def _candidate_views_from_recognition(diagnostics: Mapping[str, object]) -> tuple[BoreCandidateView, ...]:
    raw_features = diagnostics.get("candidate_data")
    if not isinstance(raw_features, (list, tuple)):
        raw_features = diagnostics.get("recognition_features")
    if not isinstance(raw_features, (list, tuple)):
        raw_features = diagnostics.get("recognition_engine_features")
    if not isinstance(raw_features, (list, tuple)):
        raw_features = diagnostics.get("promoted_candidate_data")
    if not isinstance(raw_features, (list, tuple)):
        raw_features = diagnostics.get("promoted_feature_candidates")
    if not isinstance(raw_features, (list, tuple)):
        return ()

    out: list[BoreCandidateView] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_features, start=1):
        if not isinstance(raw, Mapping):
            continue
        candidate = _candidate_view_from_raw_feature(raw, index=index)
        if candidate is None:
            continue
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        out.append(candidate)
    return tuple(out)


def _candidate_view_from_raw_feature(raw: Mapping[str, object], *, index: int) -> BoreCandidateView | None:
    display_face_ids = _first_face_ids(raw, ("display_face_ids", "preview_face_ids", "face_ids", "semantic_face_ids"))
    if not display_face_ids:
        return None

    rebuild_face_ids = _first_face_ids(raw, ("rebuild_face_ids", "candidate_rebuild_face_ids")) or display_face_ids
    entity_type = str(raw.get("entity_type", raw.get("feature_kind", "feature")) or "feature").strip().lower() or "feature"
    feature_kind = str(raw.get("feature_kind", entity_type) or entity_type).strip().lower() or entity_type
    candidate_id = str(raw.get("candidate_id", raw.get("feature_id", f"boretool.{entity_type}.{index}")) or f"boretool.{entity_type}.{index}")
    feature_id = str(raw.get("feature_id", candidate_id) or candidate_id)
    display_name = str(raw.get("display_name", entity_type.upper()) or entity_type.upper())
    feature_family = str(raw.get("feature_family", "unknown") or "unknown").strip().lower()
    recognition_stage = str(raw.get("recognition_stage", raw.get("promotion_state", "diagnostic_only")) or "diagnostic_only").strip().lower()

    raw_can_rebuild = bool(raw.get("candidate_action_enabled", raw.get("can_rebuild", False) or raw.get("rebuild_authorized", False) or raw.get("rebuild_target_ready", False)))
    can_rebuild = bool(raw_can_rebuild and recognition_stage == "accepted_candidate" and feature_family in {"bore", "chamfer_form"})
    can_preview = bool(display_face_ids)
    disabled_reason = "" if can_rebuild else str(raw.get("rebuild_block_reason", raw.get("status", "preview-only candidate")) or "preview-only candidate")

    geometry = _geometry_text(raw, entity_type=entity_type)
    primitive_rows = _feature_primitive_rows(raw)
    primitive_kinds = tuple(str(item.get("primitive_kind", "unknown")) for item in primitive_rows)
    primitive_label = f" | primitives={','.join(primitive_kinds[:2])}" if primitive_kinds else ""
    relationship_rows = _feature_relationship_rows(raw)
    relationship_label = f" | relations={len(relationship_rows)}" if relationship_rows else ""
    role = "rebuildable" if can_rebuild else f"{recognition_stage or 'preview-only'}"
    label = f"{display_name} | family={feature_family} | stage={recognition_stage} | {len(display_face_ids)} faces | {geometry}{primitive_label}{relationship_label} | {role}"
    description = _candidate_description(raw, display_face_ids=display_face_ids, rebuild_face_ids=rebuild_face_ids, can_rebuild=can_rebuild)
    token = dict(raw)
    token.setdefault("display_face_ids", display_face_ids)
    token.setdefault("preview_face_ids", display_face_ids)
    token.setdefault("rebuild_face_ids", rebuild_face_ids)
    token.setdefault("can_rebuild", bool(can_rebuild))
    token.setdefault("can_preview", bool(can_preview))

    return BoreCandidateView(
        candidate_id=candidate_id,
        feature_id=feature_id,
        entity_type=entity_type,
        feature_kind=feature_kind,
        feature_family=feature_family,
        recognition_stage=recognition_stage,
        label=label,
        table_object=display_name,
        table_faces=str(len(display_face_ids)),
        table_geometry=geometry,
        table_role=role,
        description=description,
        display_face_ids=display_face_ids,
        rebuild_face_ids=rebuild_face_ids,
        can_preview=can_preview,
        can_rebuild=can_rebuild,
        rebuild_disabled_reason=disabled_reason,
        rebuild_token=token,
    )


def _candidate_view_from_any(value: BoreCandidateView | Mapping[str, object]) -> BoreCandidateView | None:
    if isinstance(value, BoreCandidateView):
        return value
    if isinstance(value, Mapping):
        if "rebuild_token" in value and "display_face_ids" in value:
            return BoreCandidateView(
                candidate_id=str(value.get("candidate_id", value.get("feature_id", "candidate")) or "candidate"),
                feature_id=str(value.get("feature_id", value.get("candidate_id", "candidate")) or "candidate"),
                entity_type=str(value.get("entity_type", value.get("feature_kind", "feature")) or "feature"),
                feature_kind=str(value.get("feature_kind", value.get("entity_type", "feature")) or "feature"),
                feature_family=str(value.get("feature_family", "unknown") or "unknown"),
                recognition_stage=str(value.get("recognition_stage", "diagnostic_only") or "diagnostic_only"),
                label=str(value.get("label", value.get("table_object", "candidate")) or "candidate"),
                table_object=str(value.get("table_object", value.get("label", "candidate")) or "candidate"),
                table_faces=str(value.get("table_faces", len(tuple_ints(value.get("display_face_ids", ()))) or "")),
                table_geometry=str(value.get("table_geometry", "") or ""),
                table_role=str(value.get("table_role", "rebuildable" if value.get("can_rebuild") else "preview-only") or ""),
                description=str(value.get("description", "") or ""),
                display_face_ids=tuple_ints(value.get("display_face_ids", ())),
                rebuild_face_ids=tuple_ints(value.get("rebuild_face_ids", value.get("display_face_ids", ()))),
                can_preview=bool(value.get("can_preview", bool(tuple_ints(value.get("display_face_ids", ())))),),
                can_rebuild=bool(value.get("can_rebuild", False)),
                rebuild_disabled_reason=str(value.get("rebuild_disabled_reason", "") or ""),
                rebuild_token=dict(value.get("rebuild_token", {}) or {}),
            )
        return _candidate_view_from_raw_feature(value, index=1)
    return None


def _first_face_ids(raw: Mapping[str, object], keys: tuple[str, ...]) -> tuple[int, ...]:
    for key in keys:
        ids = tuple_ints(raw.get(key, ()))
        if ids:
            return ids
    return ()


def _geometry_text(raw: Mapping[str, object], *, entity_type: str) -> str:
    if entity_type == "chamfer":
        return "R %s → %s  H %s" % (
            _short_number(raw.get("inner_radius", 0.0)),
            _short_number(raw.get("outer_radius", raw.get("mouth_radius", 0.0))),
            _short_number(raw.get("height", raw.get("axial_span", 0.0))),
        )
    if entity_type == "borehole":
        return "Ø %s  R %s  D %s" % (
            _short_number(raw.get("diameter", 0.0)),
            _short_number(raw.get("radius", 0.0)),
            _short_number(raw.get("depth", raw.get("height", raw.get("axial_span", 0.0)))),
        )
    return "R %s" % _short_number(raw.get("radius", raw.get("mouth_radius", 0.0)))


def _feature_primitive_rows(raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return non-mutating primitive descriptors from a CandidateData-like row."""

    direct = raw.get("feature_primitives")
    if isinstance(direct, tuple) or isinstance(direct, list):
        rows = tuple(item for item in direct if isinstance(item, Mapping))
        if rows:
            return rows
    ledger = raw.get("x1_evidence_ledger")
    if isinstance(ledger, Mapping):
        value = ledger.get("feature_primitives")
        if isinstance(value, tuple) or isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _feature_relationship_rows(raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Return typed feature-object relationship descriptors from a candidate row."""

    direct = raw.get("feature_relationships")
    if isinstance(direct, tuple) or isinstance(direct, list):
        rows = tuple(item for item in direct if isinstance(item, Mapping))
        if rows:
            return rows
    relationships = raw.get("relationships")
    if isinstance(relationships, Mapping):
        value = relationships.get("feature_relationships")
        if isinstance(value, tuple) or isinstance(value, list):
            rows = tuple(item for item in value if isinstance(item, Mapping))
            if rows:
                return rows
    ledger = raw.get("x1_evidence_ledger")
    if isinstance(ledger, Mapping):
        value = ledger.get("feature_relationships")
        if isinstance(value, tuple) or isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _candidate_description(raw: Mapping[str, object], *, display_face_ids: tuple[int, ...], rebuild_face_ids: tuple[int, ...], can_rebuild: bool) -> str:
    lines = [
        f"Candidate: {raw.get('display_name', raw.get('entity_type', 'feature'))}",
        f"ID: {raw.get('candidate_id', raw.get('feature_id', '-'))}",
        f"Display faces: {len(display_face_ids)}",
        f"Rebuild input faces: {len(rebuild_face_ids)}",
        f"Can preview: {bool(display_face_ids)}",
        f"Can rebuild: {bool(can_rebuild)}",
    ]
    primitive_rows = _feature_primitive_rows(raw)
    if primitive_rows:
        lines.append(f"feature_primitives: {len(primitive_rows)}")
        for idx, primitive in enumerate(primitive_rows[:3], start=1):
            lines.append(
                "feature_primitive_%d: kind=%s radius=%s depth=%s role=%s"
                % (
                    idx,
                    primitive.get("primitive_kind", "unknown"),
                    _short_number(primitive.get("radius", 0.0)),
                    _short_number(primitive.get("depth", 0.0)),
                    primitive.get("role", "-"),
                )
            )
    relationship_rows = _feature_relationship_rows(raw)
    if relationship_rows:
        lines.append(f"feature_relationships: {len(relationship_rows)}")
        for idx, relation in enumerate(relationship_rows[:4], start=1):
            lines.append(
                "feature_relationship_%d: kind=%s target=%s confidence=%s role=%s"
                % (
                    idx,
                    relation.get("relationship_kind", relation.get("relation", "unknown")),
                    relation.get("target_candidate_id", relation.get("candidate_id", "-")),
                    _short_number(relation.get("confidence", 0.0)),
                    relation.get("role", "-"),
                )
            )
    for key in ("feature_family", "recognition_stage", "evidence_kinds", "promotion_reasons", "rejection_reasons", "feature_primitive_count", "feature_relationship_count", "x1_primitive_bridge_contract", "rebuild_target_policy_allowed", "rebuild_target_policy_reason", "delete_patch_request_allowed", "status", "promotion_state", "role", "candidate_action_enabled", "candidate_action", "rebuild_disabled_reason", "surface_condition", "repair_strategy", "confidence", "radius", "diameter", "depth", "height", "axial_span"):
        if key in raw:
            lines.append(f"{key}: {raw[key]}")
    ledger = raw.get("x1_evidence_ledger")
    if isinstance(ledger, Mapping):
        lines.append("x1_evidence_ledger: present")
        lines.append(f"x1_evidence_items: {ledger.get('evidence_item_count', 0)}")
        lines.append(f"x1_feature_primitives: {ledger.get('feature_primitive_count', 0)}")
        lines.append(f"x1_feature_relationships: {ledger.get('feature_relationship_count', 0)}")
        lines.append(f"x1_target_policy: {ledger.get('target_policy_reason', '-')}")
    return "\n".join(lines)


def _routed_outputs_text(
    *,
    selected_edge_ids: tuple[int, ...],
    normalized_edge_ids: tuple[int, ...],
    region_face_ids: tuple[int, ...],
    seed_face_ids: tuple[int, ...],
    region_preview_face_ids: tuple[int, ...],
    candidates: tuple[BoreCandidateView, ...],
    diagnostics: Mapping[str, object],
) -> str:
    """Return block-diagram outputs routed explicitly for display.

    The display layer should not inspect raw recognition dictionaries. This text
    is generated inside BoreTool so every output in the block diagram is visible
    without adding GUI logic.
    """

    engine = diagnostics.get("component_engine_diagnostics")
    if not isinstance(engine, Mapping):
        engine = {}
    feature_patch = diagnostics.get("feature_patch_measurement")
    if not isinstance(feature_patch, Mapping):
        feature_patch = {}

    lines: list[str] = [
        "Routed BoreTool outputs:",
        "1. Region Select output:",
        f"  selected_edge_ids: {len(selected_edge_ids)}",
        f"  normalized_edge_ids: {len(normalized_edge_ids)}",
        f"  RegionData faces: {len(region_face_ids)}",
        f"  seed faces: {len(seed_face_ids)}",
        f"  selection preview faces: {len(region_preview_face_ids)}",
        f"  selection_preview_contract: {diagnostics.get('selection_preview_contract', '-')}",
        f"  semantic_role: {diagnostics.get('semantic_role', diagnostics.get('selection_contract', '-'))}",
        f"  boundary_loop_count: {diagnostics.get('boundary_loop_count', '-')}",
        f"  volumetric_anchor_policy: {diagnostics.get('volumetric_anchor_policy', '-')}",
        f"  primary_anchor_edge_count: {diagnostics.get('primary_anchor_edge_count', '-')}",
        "2. Recognition / CandidateData output:",
        f"  measured patch faces: {feature_patch.get('face_count', '-')}",
        f"  measured radius: {feature_patch.get('radius', '-')}",
        f"  axial/depth span: {feature_patch.get('axial_span', feature_patch.get('depth_estimate', '-'))}",
        f"  rebuild_ready: {diagnostics.get('rebuild_ready', '-')}",
        f"  recognition_failed: {diagnostics.get('recognition_failed', False)}",
        f"  recognition_error: {diagnostics.get('recognition_error', '')}",
        "3. Component Engine output:",
        f"  candidate_count: {len(candidates)}",
        f"  candidate families: {tuple(candidate.feature_family for candidate in candidates)}",
        f"  recognition stages: {tuple(candidate.recognition_stage for candidate in candidates)}",
        f"  borehole_face_count: {engine.get('borehole_face_count', '-')}",
        f"  unclassified_face_count: {engine.get('unclassified_face_count', '-')}",
        f"  candidate_isolation_policy: {engine.get('candidate_isolation_policy', '-')}",
        f"  density_bias_guard_used: {engine.get('density_bias_guard_used', '-')}",
        f"  raw_face_count_score_weight: {engine.get('raw_face_count_score_weight', '-')}",
        f"  geometric_band_score_weight: {engine.get('geometric_band_score_weight', '-')}",
        f"  selected_seed_required_for_remote_component: {engine.get('selected_seed_required_for_remote_component', '-')}",
        f"  same_cylinder_completion_added_faces: {engine.get('same_cylinder_completion_added_face_count', '-')}",
        f"  neutral_volume_cutout_completion_added_faces: {engine.get('neutral_volume_cutout_completion_added_face_count', '-')}",
        f"  neutral_volume_cutout_completion_used_components: {engine.get('neutral_volume_cutout_completion_used_component_count', '-')}",
        f"  chamfer_completion_added_faces: {engine.get('chamfer_completion_added_face_count', '-')}",
        f"  chamfer_completion_used_groups: {engine.get('chamfer_completion_used_group_count', '-')}",
        "4. CandidateView output:",
    ]
    if candidates:
        for idx, candidate in enumerate(candidates, start=1):
            token = dict(candidate.rebuild_token or {})
            lines.append(
                "  [%d] %s display_faces=%d rebuild_faces=%d can_preview=%s can_rebuild=%s"
                % (
                    idx,
                    candidate.candidate_id,
                    len(candidate.display_face_ids),
                    len(candidate.rebuild_face_ids),
                    bool(candidate.can_preview),
                    bool(candidate.can_rebuild),
                )
            )
            for key in (
                "candidate_isolation_policy",
                "core_selection_anchor_policy",
                "same_cylinder_completion_policy",
                "neutral_volume_cutout_completion_policy",
                "neutral_volume_cutout_completion_policy",
                "selected_seed_face_count",
                "same_cylinder_completion_used_component_count",
                "neutral_volume_cutout_completion_added_face_count",
                "neutral_volume_cutout_completion_used_component_count",
                "neutral_volume_cutout_completion_added_face_count",
                "neutral_volume_cutout_completion_used_component_count",
                "chamfer_completion_policy",
                "chamfer_completion_added_face_count",
                "patch_topology_rebuildable",
                "patch_topology_role",
                "density_bias_guard_used",
                "raw_face_count_score_weight",
                "geometric_band_score",
                "density_normalized_face_score",
                "excluded_chamfer_face_count",
            ):
                if key in token:
                    lines.append(f"      {key}: {token[key]}")
    else:
        lines.append("  none")

    completion_diags = engine.get("same_cylinder_completion_diagnostics", ())
    if isinstance(completion_diags, (list, tuple)) and completion_diags:
        lines.append("5. Same-cylinder completion components sample:")
        for idx, item in enumerate(tuple(completion_diags)[:6], start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  [%d] faces=%s same_cylinder=%s locality_ok=%s seed_anchor=%s adjacent_to_base=%s radial=%s span=%s"
                % (
                    idx,
                    item.get("face_count", "-"),
                    item.get("same_cylinder", "-"),
                    item.get("locality_ok", "-"),
                    item.get("seed_anchor", "-"),
                    item.get("adjacent_to_base_face_pair_count", "-"),
                    item.get("radial_median", "-"),
                    item.get("radial_span", "-"),
                )
            )

    neutral_diags = engine.get("neutral_volume_cutout_completion_components_sample", ())
    if isinstance(neutral_diags, (list, tuple)) and neutral_diags:
        lines.append("5b. Neutral volume-cutout completion components sample:")
        for idx, item in enumerate(tuple(neutral_diags)[:6], start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  [%d] faces=%s accepted=%s reason=%s seed_direct=%s adjacent_to_core=%s radial=%s span=%s"
                % (
                    idx,
                    item.get("face_count", "-"),
                    item.get("accepted", "-"),
                    item.get("reason", "-"),
                    item.get("seed_direct_face_count", "-"),
                    item.get("adjacent_to_core_face_pair_count", "-"),
                    item.get("radial_median", "-"),
                    item.get("radial_span", "-"),
                )
            )

    chamfer_diags = engine.get("chamfer_completion_diagnostics", ())
    if isinstance(chamfer_diags, (list, tuple)) and chamfer_diags:
        lines.append("5c. Chamfer completion components sample:")
        for idx, item in enumerate(tuple(chamfer_diags)[:6], start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "  [%d] base=%s completed=%s added=%s rebuildable=%s source=%s comps=%s"
                % (
                    idx,
                    item.get("base_face_count", "-"),
                    item.get("completed_face_count", "-"),
                    item.get("added_face_count", "-"),
                    item.get("patch_topology_rebuildable", "-"),
                    item.get("source_face_count", "-"),
                    item.get("source_component_count", "-"),
                )
            )

    lines.extend([
        "6. Target Policy output:",
        "  deferred until rebuild is requested",
        "7. Rebuild output:",
        "  not run yet",
    ])
    return "\n".join(lines)


def _region_preview_face_ids_from_region(region: object, diagnostics: Mapping[str, object]) -> tuple[int, ...]:
    """Return the display-only volumetric region preview produced by Region Select.

    This is separate from the recognized candidate preview.  It answers the
    operator's immediate visual question: "what neutral RegionData cutout did my edge
    evidence expand into?"  It is not a feature decision and not a rebuild target.
    """

    ids = tuple_ints(getattr(region, "region_preview_face_ids", ()))
    if ids:
        return ids
    cutout = diagnostics.get("cutout", {}) if isinstance(diagnostics, Mapping) else {}
    if isinstance(cutout, Mapping):
        for key in ("region_preview_face_ids", "region_face_ids"):
            ids = tuple_ints(cutout.get(key, ()))
            if ids:
                return ids
    ids = tuple_ints(diagnostics.get("region_preview_face_ids", ())) if isinstance(diagnostics, Mapping) else ()
    if ids:
        return ids
    # Fallback: show seeds rather than the broad RegionData cutout so display never suggests
    # that the full RegionData cutout is a recognized feature.
    return tuple_ints(getattr(region, "seed_face_ids", ()))


def _normalized_edge_ids_from_diagnostics(diagnostics: Mapping[str, object], selected_edge_ids: tuple[int, ...]) -> tuple[int, ...]:
    cutout = diagnostics.get("cutout", {})
    if isinstance(cutout, Mapping):
        ids = tuple_ints(cutout.get("selected_edge_ids", ()))
        if ids:
            return ids
    ids = tuple_ints(diagnostics.get("selected_edge_ids", ()))
    if ids:
        return ids
    return selected_edge_ids


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------


def _analysis_text(
    *,
    selected_edge_ids: tuple[int, ...],
    normalized_edge_ids: tuple[int, ...],
    region_face_ids: tuple[int, ...],
    seed_face_ids: tuple[int, ...],
    axis: object,
    radius: object,
    diagnostics: Mapping[str, object],
) -> str:
    return (
        f"Raw viewport edge selection: {len(selected_edge_ids)}\n"
        f"Normalized Bore ring evidence: {len(normalized_edge_ids)}\n"
        f"Selected RegionData faces: {len(region_face_ids)}\n"
        f"Seed faces: {len(seed_face_ids)}\n"
        f"Axis hint: {axis}\n"
        f"Opening radius hint: {radius}\n"
        "Region selection: Success\n"
        "Recognition: complete\n\n"
        f"{format_bore_diagnostics(diagnostics)}"
    )


def format_bore_diagnostics(diagnostics: Mapping[str, object] | object) -> str:
    if not isinstance(diagnostics, Mapping):
        return ""
    lines: list[str] = []
    preferred = (
        "mode",
        "pipeline_stage",
        "active_candidate_authority",
        "region_select_feature_authority",
        "selected_edge_count",
        "loop_count",
        "closed_loop_count",
        "seed_face_count",
        "selected_face_count",
        "boundary_loop_count",
        "median_selected_edge_length",
        "rebuild_ready",
        "rebuild_block_reason",
    )
    for key in preferred:
        if key in diagnostics:
            lines.append(f"{key}: {diagnostics[key]}")

    feature_patch = diagnostics.get("feature_patch_measurement")
    if isinstance(feature_patch, Mapping):
        lines.append("feature patch measurement:")
        for key in ("face_count", "radius", "diameter", "axial_span", "depth_estimate", "normal_axis_abs_median", "boundary_loop_count", "mouth_loop_radius", "measurement_frame_source"):
            if key in feature_patch:
                lines.append(f"  {key}: {feature_patch[key]}")

    engine = diagnostics.get("component_engine_diagnostics")
    if isinstance(engine, Mapping):
        lines.append("component engine:")
        for key in (
            "component_engine_version",
            "region_face_count",
            "borehole_face_count",
            "chamfer_candidate_count",
            "unclassified_face_count",
            "candidate_isolation_policy",
            "selected_seed_required_for_remote_component",
            "selected_seed_face_count",
            "selected_radius_anchor_tolerance",
            "core_selection_anchor_policy",
            "chosen_core_face_count",
            "same_cylinder_completion_added_face_count",
            "same_cylinder_completion_used_component_count",
            "selected_rim_anchor_used",
            "selected_seed_anchor_used",
        ):
            if key in engine:
                lines.append(f"  {key}: {engine[key]}")

    features = diagnostics.get("recognition_features")
    if isinstance(features, (list, tuple)):
        lines.append(f"recognition features: {len(features)}")
        for idx, raw in enumerate(features, start=1):
            if not isinstance(raw, Mapping):
                continue
            preview_faces = _first_face_ids(raw, ("display_face_ids", "preview_face_ids", "face_ids", "semantic_face_ids"))
            lines.append(
                f"  [{idx}] {raw.get('entity_type', raw.get('feature_kind', 'feature'))} "
                f"id={raw.get('candidate_id', raw.get('feature_id', '-'))} "
                f"preview_faces={len(preview_faces)} "
                f"candidate_action_enabled={bool(raw.get('candidate_action_enabled', raw.get('can_rebuild', raw.get('rebuild_authorized', False))))}"
            )
    return "\n".join(lines)


def _short_number(value: object, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return "-"
    if not math.isfinite(val):
        return "-"
    return f"{val:.{digits}g}"


def _tuple_ints_keep_order(values: Iterable[object] | object) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    try:
        raw = tuple(values or ())  # type: ignore[arg-type]
    except Exception:
        raw = ()
    for item in raw:
        try:
            value = int(item)
        except Exception:
            continue
        if value < 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


# -----------------------------------------------------------------------------
# Core-owned BoreTool runtime used by the single GUI display file
# -----------------------------------------------------------------------------

class BoreToolRuntime:
    """Core-owned BoreTool runtime for GUI button events.

    Flow:
        selected edge snapshot -> BoreTool -> display DTOs -> display

    The runtime is intentionally located in ``far_mesh.core.bore.tool`` so the
    GUI has no second Bore controller module. It talks to the host through a
    tiny duck-typed port: processor, selection_controller, viewport, task runner
    and display callbacks supplied by ``bore_actions.py``.
    """

    def __init__(self, owner: object) -> None:
        self.owner = owner
        self.analysis_result: BoreToolDisplayResult | None = None
        self.selected_candidate_id: str = ""
        self.previewed_candidate_id: str = ""
        self.cached_edge_ids: tuple[int, ...] = ()
        self._source_mesh_signature: tuple[int, int, int] | None = None

    # ------------------------------------------------------------------
    # Small owner adapters
    # ------------------------------------------------------------------

    def _processor(self) -> object | None:
        return getattr(self.owner, "processor", None)

    def _mesh(self) -> object | None:
        return getattr(self._processor(), "mesh", None)

    def _has_mesh(self) -> bool:
        return self._mesh() is not None

    def _mesh_signature(self) -> tuple[int, int, int]:
        mesh = self._mesh()
        if mesh is None:
            return (0, 0, 0)
        try:
            vertices = len(getattr(mesh, "vertices", ()))
        except Exception:
            vertices = 0
        try:
            faces = len(getattr(mesh, "faces", ()))
        except Exception:
            faces = 0
        return (int(id(mesh)), int(vertices), int(faces))

    def _candidates_match_active_mesh(self) -> bool:
        if self._source_mesh_signature is None:
            return True
        return tuple(self._source_mesh_signature) == self._mesh_signature()

    def _use_execution_layer_for_bore(self) -> bool:
        """Return whether Bore operations should route through Phase 6 tasks."""

        raw = getattr(self.owner, "use_execution_layer_for_bore", None)
        if raw is None:
            raw = os.environ.get("FAR_MESH_BORE_PROCESS_TASKS", "1")
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "no", "off", "direct"}
        return bool(raw)

    def _candidate_index_for(self, candidate: BoreCandidateView) -> int | None:
        if self.analysis_result is None:
            return None
        for index, item in enumerate(self.analysis_result.candidates):
            if item.candidate_id == candidate.candidate_id:
                return int(index)
        return None

    def _log(self, message: str) -> None:
        fn = getattr(self.owner, "_bore_display_log", None)
        if callable(fn):
            fn(message)
        elif hasattr(self.owner, "log"):
            self.owner.log(message)
        else:
            print(message)

    def _status(self, message: str, timeout: int = 2000) -> None:
        fn = getattr(self.owner, "_bore_display_status", None)
        if callable(fn):
            fn(message, timeout)
            return
        if hasattr(self.owner, "statusBar"):
            bar = self.owner.statusBar()
            if bar is not None:
                bar.showMessage(message, timeout)

    def _info(self, title: str, message: str) -> None:
        fn = getattr(self.owner, "_bore_display_info", None)
        if callable(fn):
            fn(title, message)
            return
        self._log(f"{title}: {message}")
        self._status(message, 3000)

    def _critical(self, title: str, message: str) -> None:
        fn = getattr(self.owner, "_bore_display_critical", None)
        if callable(fn):
            fn(title, message)
            return
        self._log(f"{title}: {message}")
        self._status(message, 5000)

    def _confirm(self, title: str, message: str) -> bool:
        fn = getattr(self.owner, "_bore_display_confirm", None)
        if callable(fn):
            return bool(fn(title, message))
        self._log(f"{title}: {message}")
        return True

    def _sync_selection_controller(self) -> None:
        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None and hasattr(controller, "sync_from_viewport"):
            try:
                controller.sync_from_viewport(reason="boretool_request")
            except Exception:
                pass

    def _selected_edge_ids(self) -> tuple[int, ...]:
        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None:
            snapshot = getattr(controller, "selected_edge_ids_snapshot", None)
            if callable(snapshot):
                try:
                    return tuple(int(v) for v in snapshot(reason="boretool_selected_edge_ids"))
                except Exception:
                    return ()

            getter = getattr(controller, "selected_edge_ids", None)
            if callable(getter):
                try:
                    return tuple(int(v) for v in getter())
                except Exception:
                    return ()
        return ()

    def _set_edge_region_strategy(self, strategy: str) -> None:
        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None and hasattr(controller, "set_edge_region_strategy"):
            try:
                controller.set_edge_region_strategy(strategy, reason="boretool_edge_region_strategy")
                return
            except Exception:
                pass
        viewport = getattr(self.owner, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_edge_region_strategy"):
            try:
                viewport.set_edge_region_strategy(strategy)
            except Exception:
                pass

    def _clear_semantic_selection_after_rebuild(self) -> None:
        """Clear stale edge/face selection after mesh replacement.

        A successful rebuild replaces the active mesh object and invalidates all
        previous edge/face IDs.  Leaving old SelectionController or viewport pick
        caches alive can make the next Ctrl+click return zero edges, or worse,
        route stale IDs into Region Select.  This helper is workflow state reset,
        not geometry/recognition logic.
        """

        controller = getattr(self.owner, "selection_controller", None)
        clear_after_mesh_replacement = getattr(controller, "clear_after_mesh_replacement", None) if controller is not None else None
        if callable(clear_after_mesh_replacement):
            try:
                clear_after_mesh_replacement(
                    reason="boretool_rebuild_committed_clear_stale_selection",
                    edge_region_strategy="bore_rim",
                )
                return
            except Exception:
                # Fall through to compatibility cleanup below.
                pass

        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None:
            clear = getattr(controller, "clear_selection", None)
            if callable(clear):
                try:
                    clear(keep_mode=True, reason="boretool_rebuild_committed_clear_stale_selection")
                except TypeError:
                    try:
                        clear()
                    except Exception:
                        pass
                except Exception:
                    pass
            for name in (
                "clear_edge_selection",
                "clear_face_selection",
                "clear_cached_selection",
                "discard_cached_edge_ids",
                "reset_cached_selection",
                "invalidate_selection_cache",
            ):
                fn = getattr(controller, name, None)
                if callable(fn):
                    try:
                        fn(reason="boretool_rebuild_committed_clear_stale_selection")
                    except TypeError:
                        try:
                            fn()
                        except Exception:
                            pass
                    except Exception:
                        pass

        viewport = getattr(self.owner, "viewport", None)
        if viewport is not None:
            for name, args in (
                ("set_edge_selection", ((),)),
                ("set_selected_edge_ids", ((),)),
                ("set_face_selection", ((),)),
                ("highlight_cells", ((),)),
                ("clear_selection", ()),
                ("clear_edge_selection", ()),
                ("clear_face_selection", ()),
            ):
                fn = getattr(viewport, name, None)
                if callable(fn):
                    try:
                        fn(*args)
                    except Exception:
                        pass

    def _rearm_bore_edge_pick_after_rebuild(self) -> None:
        """Put the post-rebuild viewport back into Bore rim edge-pick mode."""

        controller = getattr(self.owner, "selection_controller", None)
        rearm = getattr(controller, "rearm_tool_edge_selection", None) if controller is not None else None
        if callable(rearm):
            try:
                rearm(
                    tool="bore",
                    region_strategy="bore_rim",
                    reason="boretool_rebuild_committed_rearm_edge_pick",
                )
            except Exception:
                pass
        else:
            if controller is not None:
                apply_mode = getattr(controller, "apply_viewer_mode", None)
                if callable(apply_mode):
                    try:
                        apply_mode("edge", reason="boretool_rebuild_committed_rearm_edge_pick")
                    except Exception:
                        pass
                set_mode = getattr(controller, "set_mode", None)
                if callable(set_mode):
                    try:
                        set_mode("edge")
                    except Exception:
                        pass

            viewport = getattr(self.owner, "viewport", None)
            if viewport is not None:
                for name in ("apply_viewer_mode", "set_viewer_mode", "set_selection_mode"):
                    fn = getattr(viewport, name, None)
                    if callable(fn):
                        try:
                            fn("edge", reason="boretool_rebuild_committed_rearm_edge_pick")
                        except TypeError:
                            try:
                                fn("edge")
                            except Exception:
                                pass
                        except Exception:
                            pass

            self._set_edge_region_strategy("bore_rim")

        self._set_combo_current_data(getattr(self.owner, "viewer_selection_combo", None), "edge")
        self._set_combo_current_data(getattr(self.owner, "brush_selection_mode_combo", None), "edge")
        brush_check = getattr(self.owner, "brush_enable_check", None)
        if brush_check is not None:
            try:
                brush_check.setChecked(False)
            except Exception:
                pass
        sync = getattr(self.owner, "_sync_selection_ui_from_backend", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass
        display = getattr(self.owner, "_bore_display_edge_pick_active", None)
        if callable(display):
            try:
                display()
            except Exception:
                pass

    @staticmethod
    def _set_combo_current_data(combo: object | None, value: object) -> None:
        if combo is None:
            return
        try:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
                return
        except Exception:
            pass
        try:
            for idx in range(combo.count()):
                if combo.itemText(idx) == str(value):
                    combo.setCurrentIndex(idx)
                    return
        except Exception:
            pass

    def _current_quad_density_mode(self) -> str:
        fn = getattr(self.owner, "_bore_display_current_quad_density_mode", None)
        if callable(fn):
            return str(fn())
        return "lean_pi_opening"

    def _quad_density_label(self, mode: str) -> str:
        labels = {
            "full_equal_edge": "Full / equal-edge",
            "pi_opening": "Balanced / pi-opening",
            "lean_pi_opening": "Lean / Low — 882 quads test style",
        }
        return labels.get(str(mode), str(mode))

    def _candidate_by_id(self, candidate_id: str) -> BoreCandidateView | None:
        if self.analysis_result is None:
            return None
        return self.analysis_result.candidate_by_id(candidate_id)

    def _current_candidate(self) -> BoreCandidateView | None:
        if not self.selected_candidate_id and self.analysis_result and self.analysis_result.candidates:
            self.selected_candidate_id = self.analysis_result.candidates[0].candidate_id
        return self._candidate_by_id(self.selected_candidate_id)

    # ------------------------------------------------------------------
    # Page / selection actions
    # ------------------------------------------------------------------

    def on_page_requested(self) -> None:
        if hasattr(self.owner, "_show_page"):
            self.owner._show_page(getattr(self.owner, "PAGE_BORE", "bore"))
        else:
            self._status("Bore page requested.")

    def on_page_shown(self, key: str) -> None:
        if key == getattr(self.owner, "PAGE_BORE", "bore"):
            self._set_edge_region_strategy("bore_rim")
            self._push_action_state()
        else:
            self._set_edge_region_strategy("safe")

    def on_select_opening_clicked(self) -> None:
        """Entry point from the Display button.

        Architecture rule: Edge Selection is upstream of BoreTool.  If edge IDs
        already exist, this button routes that raw evidence directly into the
        BoreTool pipeline.  It must not switch viewer mode first in a way that
        clears those IDs.  The SelectionController owns the edge-pick session and
        edge-region strategy lifecycle.
        """

        controller = getattr(self.owner, "selection_controller", None)
        existing_edge_ids: tuple[int, ...] = ()

        prepare = getattr(controller, "prepare_tool_edge_selection", None) if controller is not None else None
        if callable(prepare):
            try:
                existing_edge_ids = tuple(
                    int(v)
                    for v in prepare(
                        tool="bore",
                        region_strategy="bore_rim",
                        preserve_existing=True,
                        reason="boretool_select_opening",
                    )
                )
            except Exception:
                existing_edge_ids = ()
        else:
            self._sync_selection_controller()
            existing_edge_ids = self._selected_edge_ids()
            if controller is not None and hasattr(controller, "apply_viewer_mode"):
                try:
                    controller.apply_viewer_mode("edge", reason="boretool_select_opening")
                except Exception:
                    pass
            self._set_edge_region_strategy("bore_rim")

        if existing_edge_ids:
            self.cached_edge_ids = tuple(existing_edge_ids)
            self._log(
                f"BoreTool received {len(existing_edge_ids)} selected edge ID(s) from Edge Selection; "
                "routing into Region Select."
            )
            self.on_list_candidates_clicked()
            return

        # Only clear display state when preparing a new empty edge-pick session.
        # Never clear semantic selection here; edge IDs are the input channel
        # into BoreTool, not display state.
        self.clear_display_state(clear_semantic_selection=False)

        self._set_combo_current_data(getattr(self.owner, "viewer_selection_combo", None), "edge")
        self._set_combo_current_data(getattr(self.owner, "brush_selection_mode_combo", None), "edge")
        brush_check = getattr(self.owner, "brush_enable_check", None)
        if brush_check is not None:
            try:
                brush_check.setChecked(False)
            except Exception:
                pass
        sync = getattr(self.owner, "_sync_selection_ui_from_backend", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass
        display = getattr(self.owner, "_bore_display_edge_pick_active", None)
        if callable(display):
            display()
        self._status("Bore edge-pick active. Select rim edges, then click Select/List Bore again.", 3000)
        self._push_action_state()

    def on_boundary_highlight_toggled(self, enabled: bool) -> None:
        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None and hasattr(controller, "set_boundary_highlight"):
            try:
                controller.set_boundary_highlight(bool(enabled), reason="boretool_boundary_highlight_toggled")
            except Exception:
                pass
        for name in ("brush_boundary_check", "viewer_boundary_check"):
            widget = getattr(self.owner, name, None)
            if widget is not None:
                try:
                    widget.setChecked(bool(enabled))
                except Exception:
                    pass
        self._push_action_state()

    def on_focus_selection_clicked(self) -> None:
        viewport = getattr(self.owner, "viewport", None)
        if viewport is not None and hasattr(viewport, "focus_on_selection"):
            try:
                viewport.focus_on_selection()
            except Exception:
                pass

    def on_clear_selection_clicked(self) -> None:
        controller = getattr(self.owner, "selection_controller", None)
        if controller is not None and hasattr(controller, "clear_selection"):
            try:
                controller.clear_selection(keep_mode=True, reason="boretool_clear_selection")
            except TypeError:
                controller.clear_selection()
            except Exception:
                pass
        elif hasattr(self.owner, "_clear_viewport_selection"):
            try:
                self.owner._clear_viewport_selection()
            except Exception:
                pass
        self.clear_display_state(clear_semantic_selection=False)
        self._status("Bore selection cleared.")
        self._push_action_state()

    def clear_display_state(self, *, clear_semantic_selection: bool = False) -> None:
        self.analysis_result = None
        self.selected_candidate_id = ""
        self.previewed_candidate_id = ""
        self.cached_edge_ids = ()
        self._source_mesh_signature = None
        fn = getattr(self.owner, "_bore_display_clear_all", None)
        if callable(fn):
            fn(clear_semantic_selection=bool(clear_semantic_selection))

    def clear_after_rebuild(self) -> None:
        self.clear_display_state(clear_semantic_selection=False)
        self._clear_semantic_selection_after_rebuild()

    # ------------------------------------------------------------------
    # BoreTool analysis / candidate display
    # ------------------------------------------------------------------

    def on_preview_inside_boundary_clicked(self) -> None:
        if not self._has_mesh():
            self._info("No mesh", "Load a mesh first.")
            return
        self._sync_selection_controller()
        edge_ids = self._selected_edge_ids() or self.cached_edge_ids
        if not edge_ids:
            self._info("No selection", "Select Bore rim/opening edges first.")
            return
        try:
            result = preview_faces_inside_boundary(self._mesh(), edge_ids)
            self.cached_edge_ids = result.selected_edge_ids
            fn = getattr(self.owner, "_bore_display_inside_boundary_preview", None)
            if callable(fn):
                fn(result)
            self._log(result.status_text)
            self._status(result.status_text)
            self._push_action_state()
        except Exception as exc:
            self._log(f"Bore interior selection failed: {exc}")
            self._critical("Bore interior selection failed", str(exc))

    def on_list_candidates_clicked(self) -> None:
        if not self._has_mesh():
            self._info("No mesh", "Load a mesh first.")
            return
        self._sync_selection_controller()
        edge_ids = self._selected_edge_ids() or self.cached_edge_ids
        if not edge_ids:
            self._info("No selection", "Select Bore rim/opening edges first.")
            return

        processor = self._processor()
        mesh = self._mesh()
        use_planned = self._use_execution_layer_for_bore()

        def task() -> BoreToolDisplayResult:
            planned = getattr(processor, "analyze_bore_candidates_planned", None) if processor is not None else None
            if use_planned and callable(planned):
                return planned(edge_ids)
            return analyze_bore_candidates(mesh, edge_ids)

        def on_success(result: object) -> None:
            if not isinstance(result, BoreToolDisplayResult):
                raise TypeError("Bore candidate analysis returned an unexpected result type.")
            self._handle_analysis_success(result)

        def on_failure(error: object) -> str:
            message = str(error or "BoreTool analysis failed.")
            self._log(f"BoreTool analysis failed: {message}")
            fn = getattr(self.owner, "_bore_display_error", None)
            if callable(fn):
                fn(message)
            return message

        runner = getattr(self.owner, "_run_task", None)
        if callable(runner):
            runner("Analyzing Bore candidates...", task, on_success, on_failure)
        else:
            try:
                on_success(task())
            except Exception as exc:
                on_failure(exc)
                self._critical("Bore recognition failed", str(exc))

    def _handle_analysis_success(self, result: BoreToolDisplayResult) -> None:
        self.analysis_result = result
        self.selected_candidate_id = result.selected_candidate_id
        self.previewed_candidate_id = ""
        self.cached_edge_ids = result.normalized_edge_ids or result.selected_edge_ids
        self._source_mesh_signature = self._mesh_signature()
        fn = getattr(self.owner, "_bore_display_analysis_result", None)
        if callable(fn):
            fn(result)
        execution_note = " via Phase 6 task" if self._use_execution_layer_for_bore() else ""
        self._log(
            f"BoreTool listed {len(result.candidates)} candidate(s) from {len(result.region_face_ids)} RegionData faces; "
            f"normalized ring edges {len(result.normalized_edge_ids)}{execution_note}."
        )
        self._status(
            f"Selected neutral volume: {len(result.region_preview_face_ids)} faces; "
            f"{len(result.candidates)} candidate(s) listed.",
            3000,
        )
        # Do not auto-preview a candidate here.  The first visual state after
        # selecting/listing must remain the neutral volume cutout returned by
        # region_select.  Candidate preview is a separate explicit user action.
        self._push_action_state()

    def on_candidate_changed(self, candidate_id_or_index: object) -> None:
        candidate_id = ""
        if isinstance(candidate_id_or_index, str):
            candidate_id = candidate_id_or_index
        elif self.analysis_result is not None:
            try:
                idx = int(candidate_id_or_index)
            except Exception:
                idx = -1
            if 0 <= idx < len(self.analysis_result.candidates):
                candidate_id = self.analysis_result.candidates[idx].candidate_id
        if candidate_id:
            self.selected_candidate_id = candidate_id
            fn = getattr(self.owner, "_bore_display_select_candidate", None)
            if callable(fn):
                fn(candidate_id)
        self._push_action_state()

    def preview_current_candidate(self) -> None:
        candidate = self._current_candidate()
        if candidate is None:
            self._info("No Bore candidate", "List Bore candidates first, then select one row.")
            return
        self.preview_candidate(candidate.candidate_id, auto=False)

    def preview_candidate(self, candidate_id: str, *, auto: bool = False) -> bool:
        if not self._candidates_match_active_mesh():
            self.clear_display_state(clear_semantic_selection=False)
            if not auto:
                self._info("Stale Bore candidates", "The mesh changed. List Bore candidates again.")
            return False
        candidate = self._candidate_by_id(candidate_id)
        if candidate is None or not candidate.can_preview:
            if not auto:
                self._info("No preview faces", "The selected BoreTool candidate has no display face IDs.")
            return False
        self.selected_candidate_id = candidate.candidate_id
        self.previewed_candidate_id = candidate.candidate_id
        fn = getattr(self.owner, "_bore_display_preview_candidate", None)
        if callable(fn):
            fn(candidate, auto=auto)
        self._status(f"Previewing Bore candidate: {candidate.face_count} faces.")
        self._push_action_state()
        return True

    def reset_preview(self) -> None:
        self.previewed_candidate_id = ""
        fn = getattr(self.owner, "_bore_display_reset_candidate_preview", None)
        if callable(fn):
            fn()
        self._status("Bore candidate preview cleared.")
        self._push_action_state()

    # ------------------------------------------------------------------
    # Rebuild dispatch
    # ------------------------------------------------------------------

    def rebuild_previewed_candidate(self) -> None:
        if not self._has_mesh():
            self._info("No mesh", "Load a mesh first.")
            return
        if not self._candidates_match_active_mesh():
            self.clear_display_state(clear_semantic_selection=False)
            self._info("Stale Bore candidates", "The mesh changed. List Bore candidates again.")
            return
        candidate = self._candidate_by_id(self.previewed_candidate_id or self.selected_candidate_id)
        if candidate is None:
            self._info("No previewed Bore candidate", "Preview one BoreTool candidate before rebuilding.")
            return
        if not candidate.can_rebuild:
            self._info("Candidate is not rebuild-authorized", candidate.rebuild_disabled_reason or "BoreTool marked this candidate as preview-only.")
            return
        edge_ids = self.cached_edge_ids
        if not edge_ids:
            self._info("No Bore rim edges", "No cached Bore rim edges are available. List candidates again.")
            return

        quad_density_mode = self._current_quad_density_mode()
        quad_density_label = self._quad_density_label(quad_density_mode)
        if not self._confirm(
            "Delete and rebuild previewed Bore candidate",
            (
                "This dispatches the previewed BoreTool candidate to core.bore.rebuild.\n\n"
                f"Candidate: {candidate.label}\n"
                f"Boundary edges: {len(edge_ids)}\n"
                f"Display faces: {candidate.face_count}\n"
                f"Rebuild input faces: {len(candidate.rebuild_face_ids)}\n"
                f"Quad density: {quad_density_label}\n\n"
                "Continue?"
            ),
        ):
            return

        processor = self._processor()
        if processor is None or getattr(processor, "mesh", None) is None:
            self._info("No mesh", "No active processor mesh is available.")
            return

        def task() -> object:
            planned = getattr(processor, "rebuild_bore_candidate_planned", None)
            if self._use_execution_layer_for_bore() and callable(planned):
                return planned(
                    edge_ids=edge_ids,
                    candidate=candidate,
                    candidate_index=self._candidate_index_for(candidate),
                    quad_density_mode=quad_density_mode,
                    color_rebuilt_faces=True,
                )
            return rebuild_bore_candidate(
                processor.mesh,
                edge_ids=edge_ids,
                candidate=candidate,
                quad_density_mode=quad_density_mode,
                color_rebuilt_faces=True,
            )

        def on_success(result: object) -> None:
            self._handle_rebuild_success(
                result,
                candidate=candidate,
                quad_density_label=quad_density_label,
                quad_density_mode=quad_density_mode,
            )

        def on_failure(error: object) -> str:
            message = str(error or "Bore rebuild failed.")
            self._log(f"Bore rebuild failed: {message}")
            fn = getattr(self.owner, "_bore_display_error", None)
            if callable(fn):
                fn(message)
            return message

        runner = getattr(self.owner, "_run_task", None)
        if callable(runner):
            runner("Deleting and rebuilding previewed Bore candidate...", task, on_success, on_failure)
        else:
            try:
                on_success(task())
            except Exception as exc:
                self._critical("Bore rebuild failed", str(exc))
                on_failure(exc)

    def _handle_rebuild_success(
        self,
        result: object,
        *,
        candidate: BoreCandidateView,
        quad_density_label: str,
        quad_density_mode: str | None = None,
    ) -> None:
        processor = self._processor()
        if processor is None:
            raise ValueError("No processor available for Bore rebuild commit.")
        mesh_after = getattr(result, "mesh", None)
        if mesh_after is None:
            raise ValueError("Bore rebuild returned no mesh.")
        diagnostics = getattr(result, "diagnostics", {}) or {}
        before_faces = len(getattr(getattr(processor, "mesh", None), "faces", ()))
        after_faces = len(getattr(mesh_after, "faces", ()))
        if isinstance(diagnostics, Mapping):
            before_faces = int(diagnostics.get("before_face_count", before_faces) or before_faces)
            after_faces = int(diagnostics.get("after_face_count", after_faces) or after_faces)
        removed_count = int(getattr(result, "removed_face_count", len(candidate.rebuild_face_ids)))
        try:
            added_face_ids = tuple(sorted({int(v) for v in tuple(getattr(result, "added_face_ids", ()) or ()) if int(v) >= 0}))
        except Exception:
            added_face_ids = ()
        added_count = int(getattr(result, "added_face_count", len(added_face_ids)))
        if isinstance(diagnostics, Mapping):
            added_count = int(diagnostics.get("added_logical_quad_count", diagnostics.get("actual_added_face_count", added_count)) or added_count)

        used_phase6_rebuild = False
        if isinstance(diagnostics, Mapping):
            execution_layer = str(diagnostics.get("execution_layer") or "").strip().lower()
            task_kind = str(diagnostics.get("task_kind") or "").strip().lower()
            used_phase6_rebuild = (
                execution_layer == "process_task"
                or task_kind == "bore_rebuild_candidate"
            )
        if used_phase6_rebuild:
            self._log("Bore rebuild candidate computed via Phase 6 task.")

        commit_result = None
        commit = getattr(processor, "commit_bore_rebuild_result", None)
        if callable(commit):
            commit_result = commit(
                result,
                selected_edge_ids=self.cached_edge_ids,
                candidate_metadata=candidate.to_dict(),
                quad_density_mode=quad_density_mode,
            )
            before_faces = int(getattr(commit_result, "before_faces", before_faces))
            after_faces = int(getattr(commit_result, "after_faces", after_faces))
        else:
            # Compatibility fallback for older MeshProcessor test doubles.
            # Production FAR MESH Quad should use commit_bore_rebuild_result()
            # so snapshot-backed history and undo/redo are written before the
            # active mesh is replaced.
            processor.set_mesh(mesh_after)
            try:
                self.owner.current_output_path = None
            except Exception:
                pass
        refresh = getattr(self.owner, "_refresh_viewport_from_processor", None)
        if callable(refresh):
            refresh()
        elif hasattr(self.owner, "_set_mesh_info_from_trimesh"):
            self.owner._set_mesh_info_from_trimesh(processor.mesh)

        # Mesh replacement invalidates all edge/face IDs.  Clear stale semantic
        # selection immediately, then re-arm Bore rim edge picking on the new mesh.
        self._clear_semantic_selection_after_rebuild()
        self._rearm_bore_edge_pick_after_rebuild()

        fn = getattr(self.owner, "_bore_display_rebuild_success", None)
        if callable(fn):
            fn(
                result=result,
                candidate=candidate,
                added_face_ids=added_face_ids,
                removed_count=removed_count,
                added_count=added_count,
                before_faces=before_faces,
                after_faces=after_faces,
                quad_density_label=quad_density_label,
            )

        update_project_status = getattr(self.owner, "_update_project_status_ui_if_available", None)
        if callable(update_project_status):
            try:
                update_project_status()
            except Exception:
                pass
        if hasattr(self.owner, "_update_undo_redo_action_state"):
            try:
                self.owner._update_undo_redo_action_state()
            except Exception:
                pass
        self.analysis_result = None
        self.selected_candidate_id = ""
        self.previewed_candidate_id = ""
        self.cached_edge_ids = ()
        self._source_mesh_signature = None
        execution_note = " via Phase 6 task" if used_phase6_rebuild else ""
        self._log(f"Bore rebuild committed{execution_note}: removed {removed_count}, added {added_count}, faces {before_faces} -> {after_faces}.")
        self._status("Bore rebuild committed; stale selection cleared and Bore edge-pick re-armed.", 3000)
        self._push_action_state()

    def _push_action_state(self) -> None:
        edge_ids = self._selected_edge_ids() or self.cached_edge_ids
        selected = self._current_candidate()
        previewed = self._candidate_by_id(self.previewed_candidate_id)
        fn = getattr(self.owner, "_bore_display_set_action_state", None)
        if callable(fn):
            fn(
                has_mesh=self._has_mesh(),
                selected_edge_count=len(edge_ids),
                has_candidates=bool(self.analysis_result and self.analysis_result.candidates),
                has_selected_candidate=selected is not None,
                has_preview=previewed is not None,
                can_rebuild=bool(previewed and previewed.can_rebuild),
            )


__all__ = [
    "BoreCandidateView",
    "BoreToolDisplayResult",
    "BoreInsideBoundaryPreview",
    "analyze_bore_candidates",
    "preview_faces_inside_boundary",
    "rebuild_bore_candidate",
    "format_bore_diagnostics",
    "BoreToolRuntime",
]
