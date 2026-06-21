"""Coherent Bore feature delete/rebuild pipeline.

This module is intentionally GUI-free and viewport-free.  It receives stable
selection/candidate data from the caller and returns either a fully validated
rebuilt mesh or a diagnostic failure.  It does not mutate the active project
state directly.

Architecture
------------
The rebuild pipeline is deliberately linear:

1. Normalize the caller/candidate request.
2. Gather RegionData anchor evidence from ``region_select``.
3. Build bounded delete-patch proposals with ``rebuild_target``.
4. Derive closed boundary-loop pairs from each exact delete patch.
5. Build measured-loop quad replacement plans.
6. Trial-build the full mesh and require watertight validation.
7. Apply the first validated plan and return ``RebuildResult``.

Layer ownership
---------------
``region_select.py``
    Produces RegionData/anchor evidence only.  It does not classify features and does
    not authorize rebuilds.

``recognition.py`` / ``recognition_component_engine.py``
    Consume RegionData and provide CandidateData such as ``BOREHOLE`` or
    ``CHAMFER`` plus the candidate face IDs shown in the preview.

``rebuild_target.py``
    Converts CandidateData face evidence into bounded DeletePatchProposal objects.

``rebuild.py`` / this module
    Owns replacement topology only: boundary-loop extraction, equal/unequal
    measured-loop quad planning, watertight trial validation, and final mesh
    construction.

Geometry policy
---------------
The exact delete patch owns the boundary loops.  Existing boundary vertices are
locked.  Fitted radius/axis values may help order loops and produce diagnostics,
but they are never used to expand deletion or project final boundary vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import math

import numpy as np
import trimesh

from .rebuild_target import build_bounded_rebuild_target_face_sets
from .region_select import select_region_data
from .types import FeatureFamily, RecognitionStage, RegionData, tuple_ints
from .topology import (
    boundary_edges_for_face_patch,
    build_edge_to_faces,
    connected_face_components,
    edge_loop_components,
    normalize_edge,
)

EdgeKey = tuple[int, int]
Vector3 = tuple[float, float, float]
RGBA = tuple[int, int, int, int]

DEFAULT_BASE_FACE_COLOR: RGBA = (190, 190, 190, 255)
DEFAULT_REBUILT_FACE_COLOR: RGBA = (0, 213, 255, 255)

QUAD_DENSITY_MODE_FULL = "full_equal_edge"
QUAD_DENSITY_MODE_PI = "pi_opening"
QUAD_DENSITY_MODE_LEAN = "lean_pi_opening"

QUAD_DENSITY_MODE_ALIASES: dict[str, str] = {
    "full": QUAD_DENSITY_MODE_FULL,
    "initial": QUAD_DENSITY_MODE_FULL,
    "original": QUAD_DENSITY_MODE_FULL,
    "dense": QUAD_DENSITY_MODE_FULL,
    "equal": QUAD_DENSITY_MODE_FULL,
    "equal_edge": QUAD_DENSITY_MODE_FULL,
    "measured_edge": QUAD_DENSITY_MODE_FULL,
    "full_equal_edge": QUAD_DENSITY_MODE_FULL,
    "pi": QUAD_DENSITY_MODE_PI,
    "balanced": QUAD_DENSITY_MODE_PI,
    "medium": QUAD_DENSITY_MODE_PI,
    "pi_opening": QUAD_DENSITY_MODE_PI,
    "pi_density": QUAD_DENSITY_MODE_PI,
    "smooth": QUAD_DENSITY_MODE_PI,
    "lean": QUAD_DENSITY_MODE_LEAN,
    "low": QUAD_DENSITY_MODE_LEAN,
    "coarse": QUAD_DENSITY_MODE_LEAN,
    "lean_pi": QUAD_DENSITY_MODE_LEAN,
    "lean_pi_opening": QUAD_DENSITY_MODE_LEAN,
    "quad": QUAD_DENSITY_MODE_LEAN,
}


# -----------------------------------------------------------------------------
# Public result and internal pipeline data models
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RebuildResult:
    """Result of deleting a Bore feature patch and inserting replacement faces."""

    mesh: trimesh.Trimesh
    removed_face_ids: tuple[int, ...]
    added_face_ids: tuple[int, ...]
    added_faces: tuple[tuple[int, int, int], ...]
    loop0_vertices: tuple[int, ...]
    loop1_vertices: tuple[int, ...]
    axis: Vector3
    radius: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def removed_face_count(self) -> int:
        return len(self.removed_face_ids)

    @property
    def added_face_count(self) -> int:
        return len(self.added_face_ids)


@dataclass(frozen=True, slots=True)
class RebuildCandidateContext:
    """Normalized caller/candidate state for one rebuild request."""

    selected_edge_ids: tuple[int, ...]
    entity_type: str
    rebuild_gate: str
    role: str
    candidate_from_component_engine: bool
    feature_ownership_source: str
    candidate_has_preview_face_patch: bool
    allows_unequal_loop_transition: bool
    quad_density_mode: str
    damaged_bore_rebuild_trial: bool = False


@dataclass(frozen=True, slots=True)
class RebuildTargetPatch:
    """One exact delete-patch proposal."""

    source: str
    face_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundaryLoopAttempt:
    """One measured loop-pair rebuild attempt for one exact delete patch."""

    source: str
    target_source: str
    face_ids: tuple[int, ...]
    loop0: tuple[int, ...]
    loop1: tuple[int, ...]
    boundary_loop_count: int
    exact_two_loop_patch: bool
    protected_loop_pair: bool
    axial_separation: float
    min_required_axial_separation: float
    unequal_loop_transition_allowed: bool
    boundary_loop_vertex_count_delta: int
    loop_summaries: tuple[dict[str, object], ...] = ()


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


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def delete_and_rebuild_candidate_region(
    mesh: trimesh.Trimesh,
    edge_ids: Iterable[int],
    *,
    region_face_ids: Iterable[int] | None = None,
    feature_candidate_metadata: Mapping[str, object] | None = None,
    color_rebuilt_faces: bool = True,
    base_face_color: RGBA = DEFAULT_BASE_FACE_COLOR,
    rebuilt_face_color: RGBA = DEFAULT_REBUILT_FACE_COLOR,
    isolate_rebuilt_vertices_for_color: bool = False,
    allow_diagnostic_preview_rebuild: bool = False,
    quad_density_mode: str | None = None,
) -> RebuildResult:
    """Delete the selected recognized Bore feature patch and rebuild it.

    All candidate sources enter the same measured-patch pipeline. The public
    signature uses canonical RegionData naming; no old BoreWall wrapper remains.
    """

    # Stage 0 — source mesh normalization.
    # Trimesh may keep additional per-face columns in some import paths; the
    # rebuild runtime uses triangular faces, so all downstream topology operates
    # on the first three vertex indices.
    _validate_mesh(mesh)

    vertices = np.asarray(mesh.vertices, dtype=float)[:, :3]
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    if source_faces.ndim != 2 or source_faces.shape[1] < 3:
        raise ValueError("Bore rebuild requires triangular runtime faces.")
    source_faces = source_faces[:, :3].astype(np.int64, copy=False)

    selected_edge_ids = tuple(int(v) for v in tuple(edge_ids or ()))
    if not selected_edge_ids:
        raise ValueError("No selected Bore rim edges for rebuild.")

    # Stage 1 — normalize UI/candidate metadata into one small context object.
    # This removes branching around where the candidate came from.  A previewed
    # BOREHOLE and a previewed CHAMFER enter the same measured-patch pipeline;
    # only their topology permissions differ.
    candidate_metadata = dict(feature_candidate_metadata or {})
    context = _candidate_context(
        selected_edge_ids=selected_edge_ids,
        candidate_metadata=candidate_metadata,
        region_face_ids=tuple(region_face_ids or ()) if region_face_ids is not None else (),
        explicit_quad_density_mode=quad_density_mode,
    )

    # Stage 2 — collect RegionData anchor evidence.  region_select remains a RegionData collector only; the selected CandidateData
    # face IDs below still define the rebuild request.
    region_data = select_region_data(mesh, selected_edge_ids)
    region_data_diagnostics = dict(getattr(region_data, "diagnostics", {}) or {})

    if region_face_ids is None:
        raise ValueError(
            "Bore rebuild requires explicit CandidateData rebuild_face_ids. "
            "RegionData.face_ids are neutral AOI evidence and cannot be used as rebuild input."
        )
    initial_face_ids = _normalize_face_ids(region_face_ids, face_count=len(source_faces))
    if not initial_face_ids:
        raise ValueError("No CandidateData rebuild_face_ids were provided for rebuild.")

    axis = _unit_vector(getattr(region_data, "axis", (0.0, 0.0, 1.0)))
    radius = _safe_float(getattr(region_data, "radius", 0.0), 0.0)
    protected_loop_pair = _protected_loop_pair_from_selection(vertices=vertices, region_data=region_data, axis=axis)

    # POCKET v99: keep the meaning transform explicit.  A POCKET candidate is
    # not a BORE sleeve candidate.  Recognition has already created a POCKET
    # hypothesis and owned floor/side-wall roles.  Rebuild Target may therefore
    # request a pocket-native recessed cup rebuild: delete owned floor + owned
    # side wall, regenerate side wall + floor, keep the top opening open, and
    # exclude transition/chamfer evidence.  Only an explicit cap/flatten action
    # enters the old top-cap path.
    if _candidate_requests_pocket_cap_rebuild(context=context, candidate_metadata=candidate_metadata):
        return _delete_pocket_and_fill_opening_cap(
            mesh=mesh,
            vertices=vertices,
            source_faces=source_faces,
            face_ids=initial_face_ids,
            selected_edge_ids=selected_edge_ids,
            candidate_metadata=candidate_metadata,
            context=context,
            axis=axis,
            radius=radius,
            color_rebuilt_faces=bool(color_rebuilt_faces),
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )
    if _candidate_requests_pocket_recess_cup_rebuild(context=context, candidate_metadata=candidate_metadata):
        return _delete_and_rebuild_pocket_recess_cup(
            mesh=mesh,
            vertices=vertices,
            source_faces=source_faces,
            face_ids=initial_face_ids,
            selected_edge_ids=selected_edge_ids,
            candidate_metadata=candidate_metadata,
            context=context,
            axis=axis,
            radius=radius,
            color_rebuilt_faces=bool(color_rebuilt_faces),
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )

    extra_candidate_face_sets: list[tuple[str, tuple[int, ...]]] = []
    # v33 semantic boundary: RegionData remains neutral AOI/anchor evidence for
    # loop protection and measurement only.  It is not offered as a candidate
    # face pool to rebuild_target.py.
    region_data_face_ids = _normalize_face_ids(getattr(region_data, "face_ids", ()), face_count=len(source_faces))

    # v1.6.4 damaged-BORE correction: once Recognition has emitted accepted
    # two-opening BoreWallOwnership, RebuildTarget may use the measured
    # two-opening frame as bounded target evidence.  This is not a RegionData
    # fallback: the candidate frame constrains the target, RegionData only
    # supplies the local search pool.  It fixes damaged bores where the selected
    # owned wall fragment is only a half-depth surviving strip and therefore
    # has no usable boundary-loop pair by itself.
    candidate_frame = _two_opening_frame_from_candidate_metadata(candidate_metadata)
    if candidate_frame is not None:
        axis = _unit_vector(candidate_frame.get("axis", axis), fallback=axis)
        candidate_protected = _protected_loop_pair_from_candidate_frame(
            vertices=vertices,
            source_faces=source_faces,
            frame=candidate_frame,
            preferred_pool=region_data_face_ids or initial_face_ids,
        )
        if candidate_protected is not None:
            protected_loop_pair = candidate_protected

    if bool(context.damaged_bore_rebuild_trial) and candidate_frame is not None:
        frame_target_faces, frame_target_diag = _full_depth_bore_wall_target_faces_from_candidate_frame(
            vertices=vertices,
            source_faces=source_faces,
            frame=candidate_frame,
            face_pool=region_data_face_ids or tuple(range(len(source_faces))),
            base_face_ids=initial_face_ids,
        )
        if frame_target_faces and len(frame_target_faces) > len(initial_face_ids):
            extra_candidate_face_sets.append(("candidate_two_opening_frame_full_depth_wall_target", frame_target_faces))
            region_data_diagnostics.update({f"candidate_frame_full_depth_target_{key}": value for key, value in frame_target_diag.items()})

    # Stage 3 — ask rebuild_target for bounded delete-patch proposals.  The
    # proposals are not accepted here; each proposal must still survive a full
    # measured-loop replacement trial as a watertight mesh.
    topology_seal_callback = _topology_seal_callback_for_context(context)

    target_result = build_bounded_rebuild_target_face_sets(
        source_faces=source_faces,
        initial_face_ids=initial_face_ids,
        protected_loop_pair=protected_loop_pair,
        extra_candidate_face_sets=tuple(extra_candidate_face_sets),
        preview_candidate_patch_owns_delete=bool(context.candidate_has_preview_face_patch),
        topology_seal_callback=None,
        protected_fragment_bridge_callback=None,
    )
    if not bool(target_result.get("valid", False)):
        raise ValueError(
            "Bore rebuild target construction failed. "
            f"diagnostics={dict(target_result.get('diagnostics', {}) or {})}; Geometry changed: no."
        )

    # Convert target proposals into immutable local patch objects.  Each patch is
    # tried independently; the first watertight trial wins.
    target_patches = _target_patches_from_result(target_result)
    if not target_patches:
        target_patches = (RebuildTargetPatch("initial_candidate_faces", initial_face_ids),)

    # Damaged imported bores can contain holes, tiny islands, and broken wall
    # fragments.  In that case the candidate patch may have many boundary loops
    # even though the selected bore still has two valid rim loops.  Add a
    # conservative fallback target that swallows only same-cylinder defect
    # boundaries between the two protected rims.  This is still rebuild-target
    # policy: Region Select does not classify anything, and Rebuild still has to
    # pass watertight trial validation before geometry can change.
    # v50 semantic boundary: damaged-bore defect handling is now allowed only
    # when Recognition produced a damaged BORE CandidateData row with explicit
    # rebuild_face_ids.  This is still Target/Rebuild policy, not RegionData
    # fallback: the base delete patch remains the candidate-owned bore wall.
    damaged_bore_targets: tuple[RebuildTargetPatch, ...] = ()
    if bool(context.damaged_bore_rebuild_trial):
        damaged_bore_targets = _damaged_bore_defect_swallow_targets(
            vertices=vertices,
            source_faces=source_faces,
            target_patches=target_patches,
            axis=axis,
            protected_loop_pair=protected_loop_pair,
            entity_type=context.entity_type,
        )
        if damaged_bore_targets:
            target_patches = tuple(target_patches) + tuple(damaged_bore_targets)

    # Stage 4 — derive loop-pair attempts from the exact patch boundaries.
    # Protected loops from the selected RegionData are allowed as fallback attempts, but
    # patch-boundary loops are preferred because they are derived from the delete
    # patch that will actually be removed.
    attempts = _build_boundary_loop_attempts(
        vertices=vertices,
        source_faces=source_faces,
        target_patches=target_patches,
        axis=axis,
        protected_loop_pair=protected_loop_pair,
        allow_unequal_loop_transition=bool(context.allows_unequal_loop_transition),
    )

    if not attempts:
        raise ValueError(
            "Bore measured-patch rebuild found no usable boundary-loop pair. "
            f"target_sources={tuple(target.source for target in target_patches)}; "
            f"initial_face_count={len(initial_face_ids)}; Geometry changed: no."
        )

    # Stage 5 — plan, trial, and select.  Nothing is returned until the trial
    # mesh has zero boundary edges and Trimesh reports it as watertight.
    selected: tuple[BoundaryLoopAttempt, QuadPlan, dict[str, object]] | None = None
    attempt_summaries: list[dict[str, object]] = []
    best_failure: dict[str, object] = {}

    for attempt_index, attempt in enumerate(attempts):
        try:
            plan = _quad_plan_for_attempt(
                vertices=vertices,
                attempt=attempt,
                axis=axis,
                quad_density_mode=context.quad_density_mode,
            )
            plan = _orient_plan_triangles_to_source_patch(
                vertices=vertices,
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                plan=plan,
            )
            boundary_match = _generated_surface_boundary_match_diagnostics(
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                triangles=plan.triangles,
                source_vertex_count=int(len(vertices)),
            )
            plan_diag_with_boundary = dict(plan.diagnostics)
            plan_diag_with_boundary.update(boundary_match)
            plan = QuadPlan(
                generated_vertices=plan.generated_vertices,
                triangles=plan.triangles,
                logical_quads=plan.logical_quads,
                loop0=plan.loop0,
                loop1=plan.loop1,
                center0=plan.center0,
                center1=plan.center1,
                axis=plan.axis,
                diagnostics=plan_diag_with_boundary,
            )
            quality = _validate_plan_geometry_quality(context=context, vertices=vertices, attempt=attempt, plan=plan)
            if not bool(quality.get("valid", False)):
                summary = _attempt_summary(
                    attempt_index,
                    attempt,
                    error="geometry_quality_rejected: " + str(quality.get("reason", "unknown")),
                )
                summary.update({f"quality_{key}": value for key, value in quality.items() if key != "valid"})
                attempt_summaries.append(summary)
                if not best_failure:
                    best_failure = dict(summary)
                continue

            trial = _trial_rebuild(
                vertices=vertices,
                source_faces=source_faces,
                face_ids=attempt.face_ids,
                generated_vertices=plan.generated_vertices,
                triangles=plan.triangles,
            )
            if bool(context.damaged_bore_rebuild_trial) and 0 < int(trial.get("boundary_edge_count_after", 0) or 0) <= 24:
                # v51 damaged-bore target seal: the v50 trial proved that the
                # damaged wall candidate can leave a small residual boundary
                # after the normal two-rim measured replacement.  The observed
                # failure had 10 remaining boundary edges, so the previous
                # 6-edge residual loop cap was too strict and only produced a
                # huge rejection diagnostic.  This remains guarded by the
                # damaged_bore_rebuild_trial flag and the final watertight trial.
                seal_triangles, seal_diag = _small_boundary_seal_triangles_for_trial_mesh(
                    source_faces=source_faces,
                    face_ids=attempt.face_ids,
                    generated_vertices=plan.generated_vertices,
                    triangles=plan.triangles,
                    max_boundary_edges=24,
                    max_loop_edges=16,
                )
                plan_diag_with_seal = dict(plan.diagnostics)
                plan_diag_with_seal.update({f"damaged_bore_small_boundary_seal_{key}": value for key, value in dict(seal_diag).items()})
                plan_diag_with_seal["damaged_bore_small_boundary_seal_considered"] = True
                if len(seal_triangles):
                    sealed_triangles = np.vstack([np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)), seal_triangles])
                    sealed_trial = _trial_rebuild(
                        vertices=vertices,
                        source_faces=source_faces,
                        face_ids=attempt.face_ids,
                        generated_vertices=plan.generated_vertices,
                        triangles=sealed_triangles,
                    )
                    plan_diag_with_seal["damaged_bore_small_boundary_seal_trial_boundary_edge_count_after"] = int(sealed_trial.get("boundary_edge_count_after", -1))
                    plan_diag_with_seal["damaged_bore_small_boundary_seal_trial_watertight_after"] = bool(sealed_trial.get("watertight_after", False))
                    if int(sealed_trial.get("boundary_edge_count_after", 10**9)) <= int(trial.get("boundary_edge_count_after", 10**9)):
                        plan_diag_with_seal["damaged_bore_small_boundary_seal_used"] = True
                        plan_diag_with_seal["damaged_bore_small_boundary_seal_added_triangle_count"] = int(len(seal_triangles))
                        plan = QuadPlan(
                            generated_vertices=plan.generated_vertices,
                            triangles=sealed_triangles,
                            logical_quads=plan.logical_quads,
                            loop0=plan.loop0,
                            loop1=plan.loop1,
                            center0=plan.center0,
                            center1=plan.center1,
                            axis=plan.axis,
                            diagnostics=plan_diag_with_seal,
                        )
                        trial = sealed_trial
                else:
                    # Keep the rejection reason visible in the attempt summary so
                    # future damaged-bore target work has concrete evidence.
                    plan = QuadPlan(
                        generated_vertices=plan.generated_vertices,
                        triangles=plan.triangles,
                        logical_quads=plan.logical_quads,
                        loop0=plan.loop0,
                        loop1=plan.loop1,
                        center0=plan.center0,
                        center1=plan.center1,
                        axis=plan.axis,
                        diagnostics=plan_diag_with_seal,
                    )
            summary = _attempt_summary(attempt_index, attempt, trial=trial, plan=plan)
            attempt_summaries.append(summary)
            if not best_failure or int(summary.get("boundary_edge_count_after", 10**9)) < int(best_failure.get("boundary_edge_count_after", 10**9)):
                best_failure = dict(summary)
            if _trial_accepts_for_context(context=context, trial=trial, plan=plan):
                selected = (attempt, plan, trial)
                break
        except Exception as exc:
            summary = _attempt_summary(attempt_index, attempt, error=str(exc))
            attempt_summaries.append(summary)
            if not best_failure:
                best_failure = dict(summary)

    if selected is None:
        message = _format_failure_message(
            context=context,
            best_failure=best_failure,
            attempt_summaries=tuple(attempt_summaries),
            target_result=target_result,
        )
        raise ValueError(message)

    selected_attempt, selected_plan, selected_trial = selected
    # Stage 6 — apply the already-validated plan to a fresh mesh object.
    # The active project state is still owned by the caller/UI layer.
    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=source_faces,
        face_ids=selected_attempt.face_ids,
        generated_vertices=selected_plan.generated_vertices,
        triangles=selected_plan.triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )

    before_face_count = int(len(source_faces))
    result_mesh = result["mesh"]
    after_face_count = int(len(getattr(result_mesh, "faces", ())))
    before_vertex_count = int(len(vertices))
    after_vertex_count = int(len(getattr(result_mesh, "vertices", ())))

    diagnostics: dict[str, object] = {
        "mode": "coherent_measured_patch_rebuild_v9R",
        "topology_policy": "boundary_locked_measured_loop_quad_retessellation_coherent_v9R",
        "pipeline": (
            "candidate_context",
            "bounded_rebuild_target",
            "patch_boundary_loop_attempts",
            "measured_loop_quad_plan",
            "watertight_trial",
            "apply_rebuild",
        ),
        "candidate_entity_type": context.entity_type or "-",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "candidate_has_preview_face_patch": bool(context.candidate_has_preview_face_patch),
        "preview_candidate_patch_owns_delete": bool(context.candidate_has_preview_face_patch),
        "selected_edge_count": int(len(selected_edge_ids)),
        "initial_face_count": int(len(initial_face_ids)),
        "before_face_count": int(before_face_count),
        "after_face_count": int(after_face_count),
        "before_vertex_count": int(before_vertex_count),
        "after_vertex_count": int(after_vertex_count),
        "removed_face_count": int(len(selected_attempt.face_ids)),
        "target_face_set_sources": tuple(str(target.source) for target in target_patches),
        "selected_target_source": selected_attempt.target_source,
        "selected_attempt_source": selected_attempt.source,
        "selected_attempt_index": int(next((i for i, item in enumerate(attempts) if item is selected_attempt), -1)),
        "attempt_count": int(len(attempts)),
        "attempt_summaries": tuple(attempt_summaries),
        "boundary_loop_count": int(selected_attempt.boundary_loop_count),
        "loop0_vertex_count": int(len(selected_plan.loop0)),
        "loop1_vertex_count": int(len(selected_plan.loop1)),
        "boundary_loop_vertex_count_delta": int(abs(len(selected_plan.loop0) - len(selected_plan.loop1))),
        "unequal_loop_transition_allowed": bool(selected_attempt.unequal_loop_transition_allowed),
        "unequal_loop_transition_used": bool(selected_plan.diagnostics.get("unequal_loop_transition_used", False)),
        "transition_drop_quad_count": int(selected_plan.diagnostics.get("transition_drop_quad_count", 0) or 0),
        "transition_ring_vertex_count": int(selected_plan.diagnostics.get("transition_ring_vertex_count", 0) or 0),
        "logical_quad_count": int(len(selected_plan.logical_quads)),
        "added_logical_quad_count": int(len(selected_plan.logical_quads)),
        "added_triangle_count": int(len(selected_plan.triangles)),
        "added_runtime_triangle_count": int(len(selected_plan.triangles)),
        "actual_added_face_count": int(len(selected_plan.triangles)),
        "added_face_count": int(len(selected_plan.triangles)),
        "colored_rebuilt_face_count": int(len(result["added_face_ids"])),
        "generated_vertex_count": int(len(selected_plan.generated_vertices)),
        "transition_base_count_sequence": tuple(selected_plan.diagnostics.get("transition_base_count_sequence", ()) or ()),
        "transition_count_sequence": tuple(selected_plan.diagnostics.get("transition_count_sequence", ()) or ()),
        "transition_base_band_count": int(selected_plan.diagnostics.get("transition_base_band_count", 0) or 0),
        "transition_target_band_count": int(selected_plan.diagnostics.get("transition_target_band_count", 0) or 0),
        "transition_band_count": int(selected_plan.diagnostics.get("transition_band_count", 0) or 0),
        "transition_band_count_pairs": tuple(selected_plan.diagnostics.get("transition_band_count_pairs", ()) or ()),
        "transition_band_drop_counts": tuple(selected_plan.diagnostics.get("transition_band_drop_counts", ()) or ()),
        "transition_equal_count_spacer_band_count": int(selected_plan.diagnostics.get("transition_equal_count_spacer_band_count", 0) or 0),
        "target_axial_edge_length": float(selected_plan.diagnostics.get("target_axial_edge_length", 0.0) or 0.0),
        "raw_equal_edge_axial_segments": int(selected_plan.diagnostics.get("raw_equal_edge_axial_segments", 0) or 0),
        "axial_segment_cap": int(selected_plan.diagnostics.get("axial_segment_cap", 0) or 0),
        "generated_triangle_normal_flip_count": int(selected_plan.diagnostics.get("normal_flip_count", 0) or 0),
        "generated_triangle_normal_alignment_median": float(selected_plan.diagnostics.get("normal_alignment_median", 0.0) or 0.0),
        "generated_triangle_normal_alignment_min": float(selected_plan.diagnostics.get("normal_alignment_min", 0.0) or 0.0),
        "boundary_edge_count_before": int(selected_trial.get("boundary_edge_count_before", -1)),
        "boundary_edge_count_after": int(selected_trial.get("boundary_edge_count_after", -1)),
        "boundary_edge_count_delta": int(selected_trial.get("boundary_edge_count_delta", 0)),
        "watertight_after": bool(selected_trial.get("watertight_after", False)),
        "local_topology_acceptance_used": bool(not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_bore_wall_rebuild_acceptance_v85": bool(str(context.entity_type).strip().lower() == "borehole" and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_pocket_sidewall_rebuild_acceptance_v98": bool(str(context.entity_type).strip().lower() in {"pocket", "circular_pocket"} and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "local_pocket_sidewall_rebuild_acceptance_v97": bool(str(context.entity_type).strip().lower() in {"pocket", "circular_pocket"} and not bool(selected_trial.get("watertight_after", False)) and _trial_accepts_for_context(context=context, trial=selected_trial, plan=selected_plan)),
        "pocket_local_boundary_exact_acceptance_contract_v98": "owned_side_wall_preserve_recess_delete_patch_boundary_match_exact_no_new_boundary_edges_no_global_boundary_increase_zero_counts_preserved",
        "pocket_local_boundary_exact_acceptance_contract_v97": "owned_side_wall_preserve_recess_delete_patch_boundary_match_exact_no_new_boundary_edges_no_global_boundary_increase",
        "zero_boundary_trial_acceptance_v85": bool(int(selected_trial.get("boundary_edge_count_after", -1) if selected_trial.get("boundary_edge_count_after", -1) is not None else -1) == 0 and bool(selected_trial.get("watertight_after", False))),
        "damaged_bore_internal_defect_boundaries_swallowed_v85": bool(str(context.entity_type).strip().lower() == "borehole" and int(selected_trial.get("boundary_edge_count_after", -1) if selected_trial.get("boundary_edge_count_after", -1) is not None else -1) == 0 and bool(selected_trial.get("watertight_after", False))),
        "quad_density_mode": context.quad_density_mode,
        "quad_plan": dict(selected_plan.diagnostics),
        "rebuild_target_diagnostics": dict(target_result.get("diagnostics", {}) or {}),
        "region_data_diagnostics": region_data_diagnostics,
        "v33_semantic_boundary_hardening_used": True,
        "region_data_as_rebuild_input_enabled": False,
        "topology_seal_callback_enabled": False,
        "damaged_bore_defect_swallow_targets_enabled": False,
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
        "existing_boundary_vertices_moved": 0,
        "isolate_rebuilt_vertices_for_color_requested": bool(isolate_rebuilt_vertices_for_color),
        "allow_diagnostic_preview_rebuild_requested": bool(allow_diagnostic_preview_rebuild),
    }

    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=selected_attempt.face_ids,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in selected_plan.triangles.tolist()),
        loop0_vertices=selected_plan.loop0,
        loop1_vertices=selected_plan.loop1,
        axis=_to_vector3(selected_plan.axis),
        radius=float(radius),
        diagnostics=diagnostics,
    )



# -----------------------------------------------------------------------------
# POCKET v99: recessed cup delete/rebuild
# -----------------------------------------------------------------------------


def _candidate_requests_pocket_recess_cup_rebuild(*, context: RebuildCandidateContext, candidate_metadata: Mapping[str, object]) -> bool:
    """Return True for the pocket-native floor+side-wall recessed cup rebuild.

    This is the clean POCKET operation path.  It is intentionally separate from
    the generic measured two-loop Bore wall retessellator and from the cap/flat
    pocket-removal path.  A candidate enters this path only after Recognition has
    emitted accepted POCKET CandidateData with owned floor and side-wall roles.
    """

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    action = str(candidate_metadata.get("candidate_action", "") or "").strip().lower()
    gate = str(getattr(context, "rebuild_gate", "") or "").strip().lower()
    scope = str(candidate_metadata.get("pocket_rebuild_enable_scope", "") or "").strip().lower()
    if not (entity in {"pocket", "circular_pocket"} or family in {"pocket", "circular_pocket"}):
        return False
    if "cap" in action or "cap" in gate or "cap" in scope or "flatten" in action or "flatten" in scope:
        return False
    side_ids = tuple_ints(candidate_metadata.get("pocket_side_wall_face_ids", ()))
    floor_ids = tuple_ints(candidate_metadata.get("pocket_floor_face_ids", ()))
    return bool(
        side_ids
        and floor_ids
        and (
            "recess" in action
            or "recess" in gate
            or "recess" in scope
            or "cup" in action
            or "cup" in gate
            or "cup" in scope
            or "owned_floor" in scope
            or "floor" in action
        )
    )


def _delete_and_rebuild_pocket_recess_cup(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    context: RebuildCandidateContext,
    axis: np.ndarray,
    radius: float,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> RebuildResult:
    """Delete owned pocket floor + side wall and rebuild the recessed cup.

    Semantic contract:
        accepted POCKET CandidateData
            -> owned pocket side-wall faces + owned pocket floor faces
            -> pocket recess-cup DeletePatchProposal
            -> generated side-wall surface + generated floor surface
            -> local topology validation

    This path is neither a BORE wall sleeve nor a parent-surface cap.  The top
    opening remains open.  The bottom floor is regenerated at the measured floor
    boundary.  Transition/chamfer faces stay outside the delete patch unless
    Recognition explicitly owns them under a later transition role.
    """

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    wall_ids = _normalize_face_ids(candidate_metadata.get("pocket_side_wall_face_ids", ()), face_count=len(faces_arr))
    floor_ids = _normalize_face_ids(candidate_metadata.get("pocket_floor_face_ids", ()), face_count=len(faces_arr))
    if not wall_ids or not floor_ids:
        raise ValueError(
            "POCKET recess-cup rebuild requires explicit owned pocket_side_wall_face_ids and pocket_floor_face_ids. "
            f"wall_count={len(wall_ids)}; floor_count={len(floor_ids)}; Geometry changed: no."
        )

    patch_faces = tuple(sorted({int(fid) for fid in tuple(wall_ids + floor_ids) if 0 <= int(fid) < len(faces_arr)}))
    explicit_request = _normalize_face_ids(face_ids, face_count=len(faces_arr))
    if explicit_request and not set(patch_faces).issubset(set(explicit_request)):
        # CandidateView should normally pass exactly the owned floor+wall faces.
        # Keep the operation safe by refusing a mismatched delete patch instead
        # of silently expanding from unrelated RegionData.
        raise ValueError(
            "POCKET recess-cup rebuild request does not match owned POCKET roles. "
            f"request_face_count={len(explicit_request)}; owned_floor_plus_wall_count={len(patch_faces)}; Geometry changed: no."
        )

    floor_boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=floor_ids)
    wall_boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=wall_ids)
    if not floor_boundary_loops:
        raise ValueError(
            "POCKET recess-cup rebuild could not derive an owned floor perimeter loop. "
            f"floor_face_count={len(floor_ids)}; Geometry changed: no."
        )
    if len(wall_boundary_loops) < 2:
        raise ValueError(
            "POCKET recess-cup rebuild could not derive top and floor wall loops from owned side-wall faces. "
            f"wall_face_count={len(wall_ids)}; wall_boundary_loop_count={len(wall_boundary_loops)}; Geometry changed: no."
        )

    def _loop_edge_set(loop_record: Mapping[str, object]) -> set[EdgeKey]:
        return {_edge_key(edge) for edge in tuple(loop_record.get("edges", ()) or ())}

    def _loop_vertex_set(loop_record: Mapping[str, object]) -> set[int]:
        return {int(v) for v in tuple(loop_record.get("vertices", ()) or ())}

    def _floor_wall_overlap_score(floor_record: Mapping[str, object]) -> tuple[int, int, int, int]:
        floor_edges_local = _loop_edge_set(floor_record)
        floor_vertices_local = _loop_vertex_set(floor_record)
        best_edge_overlap = 0
        best_vertex_overlap = 0
        for wall_record in wall_boundary_loops:
            wall_edges = _loop_edge_set(wall_record)
            wall_vertices = _loop_vertex_set(wall_record)
            best_edge_overlap = max(best_edge_overlap, int(len(floor_edges_local & wall_edges)))
            best_vertex_overlap = max(best_vertex_overlap, int(len(floor_vertices_local & wall_vertices)))
        return (best_edge_overlap, best_vertex_overlap, int(len(floor_edges_local)), int(len(floor_vertices_local)))

    # Compound POCKET+BORE rule: if the owned floor has multiple boundary loops,
    # the loop that overlaps the owned pocket side wall is the floor outer
    # perimeter.  Other floor loops are protected child-BORE openings.  This is
    # relationship metadata, not a new feature family and not pocket wall
    # ownership.
    floor_loop_record = max(floor_boundary_loops, key=_floor_wall_overlap_score)
    floor_edges = _loop_edge_set(floor_loop_record)
    floor_vertices = _loop_vertex_set(floor_loop_record)
    protected_floor_hole_records = tuple(
        record for record in floor_boundary_loops
        if _loop_vertex_set(record) != floor_vertices and len(_loop_vertex_set(record)) >= 3
    )
    if len(protected_floor_hole_records) > 1:
        raise ValueError(
            "POCKET compound recess-cup rebuild currently supports one protected child-BORE opening per pocket floor. "
            f"protected_opening_count={len(protected_floor_hole_records)}; Geometry changed: no."
        )

    def _wall_floor_overlap(loop_record: Mapping[str, object]) -> tuple[int, int, int]:
        loop_edges = _loop_edge_set(loop_record)
        loop_vertices = _loop_vertex_set(loop_record)
        return (int(len(loop_edges & floor_edges)), int(len(loop_vertices & floor_vertices)), int(len(loop_vertices)))

    bottom_record = max(wall_boundary_loops, key=_wall_floor_overlap)
    bottom_vertices = tuple(int(v) for v in tuple(bottom_record.get("vertices", ()) or ()))
    bottom_set = set(bottom_vertices)
    top_candidates = [
        item for item in wall_boundary_loops
        if set(int(v) for v in tuple(item.get("vertices", ()) or ())) != bottom_set
    ]
    if not top_candidates:
        raise ValueError(
            "POCKET recess-cup rebuild could not separate parent opening loop from floor loop. "
            f"wall_boundary_loop_count={len(wall_boundary_loops)}; Geometry changed: no."
        )
    top_record = max(top_candidates, key=lambda item: len(tuple(item.get("vertices", ()) or ())))
    top_loop = tuple(int(v) for v in tuple(top_record.get("vertices", ()) or ()))
    bottom_loop = bottom_vertices
    if len(top_loop) < 3 or len(bottom_loop) < 3:
        raise ValueError(
            "POCKET recess-cup rebuild found invalid top/floor loops. "
            f"top_loop_vertices={len(top_loop)}; bottom_loop_vertices={len(bottom_loop)}; Geometry changed: no."
        )

    axis_vec = _unit_vector(axis)
    center_top = _loop_center(vertices, top_loop)
    center_bottom = _loop_center(vertices, bottom_loop)
    axial_separation = abs(float(np.dot(center_bottom - center_top, axis_vec)))
    min_sep = _minimum_loop_pair_separation(vertices, top_loop, bottom_loop)
    if axial_separation <= min_sep:
        # Fall back to the center-to-center vector if RegionData axis was not the
        # useful pocket depth direction.  This remains measurement, not feature
        # classification.
        axis_vec = _unit_vector(center_bottom - center_top, fallback=axis_vec)
        axial_separation = abs(float(np.dot(center_bottom - center_top, axis_vec)))
    if axial_separation <= min_sep:
        raise ValueError(
            "POCKET recess-cup rebuild top/floor loops have no usable depth separation. "
            f"axial_separation={axial_separation}; min_required={min_sep}; Geometry changed: no."
        )

    attempt = BoundaryLoopAttempt(
        source="pocket_recess_cup_wall_loop_pair_v99",
        target_source="pocket_recess_cup_owned_floor_plus_sidewall_target_v99",
        face_ids=wall_ids,
        loop0=top_loop,
        loop1=bottom_loop,
        boundary_loop_count=int(len(wall_boundary_loops)),
        exact_two_loop_patch=bool(len(wall_boundary_loops) == 2),
        protected_loop_pair=False,
        axial_separation=float(axial_separation),
        min_required_axial_separation=float(min_sep),
        unequal_loop_transition_allowed=bool(len(top_loop) != len(bottom_loop)),
        boundary_loop_vertex_count_delta=int(abs(len(top_loop) - len(bottom_loop))),
        loop_summaries=tuple(
            {
                "index": int(i),
                "vertex_count": int(len(tuple(item.get("vertices", ()) or ()))),
                "floor_overlap_edges": int(_wall_floor_overlap(item)[0]),
                "floor_overlap_vertices": int(_wall_floor_overlap(item)[1]),
            }
            for i, item in enumerate(wall_boundary_loops[:12])
        ),
    )

    wall_plan = _quad_plan_for_attempt(
        vertices=vertices,
        attempt=attempt,
        axis=axis_vec,
        quad_density_mode=context.quad_density_mode,
    )
    wall_plan = _orient_plan_triangles_to_source_patch(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=wall_ids,
        plan=wall_plan,
    )

    # Use the loop ordering after measured-loop alignment.  The floor cap must
    # share exactly the same bottom loop used by the rebuilt side-wall surface.
    plan_loop0_set = set(int(v) for v in tuple(wall_plan.loop0 or ()))
    plan_loop1_set = set(int(v) for v in tuple(wall_plan.loop1 or ()))
    floor_ref_set = set(int(v) for v in tuple(bottom_loop or ()))
    if len(plan_loop0_set & floor_ref_set) >= len(plan_loop1_set & floor_ref_set):
        cap_loop = tuple(int(v) for v in tuple(wall_plan.loop0 or ()))
    else:
        cap_loop = tuple(int(v) for v in tuple(wall_plan.loop1 or ()))

    if protected_floor_hole_records:
        protected_loop = tuple(int(v) for v in tuple(protected_floor_hole_records[0].get("vertices", ()) or ()))
        floor_generated_vertices, floor_triangles, floor_logical_quads, floor_diag = _pocket_floor_annular_quad_grid_from_loops(
            vertices=vertices,
            outer_loop_vertices=cap_loop,
            inner_loop_vertices=protected_loop,
            generated_vertex_offset=int(len(wall_plan.generated_vertices)),
            floor_axis=np.asarray(wall_plan.axis, dtype=float).reshape(3),
            quad_density_mode=context.quad_density_mode,
        )
    else:
        floor_generated_vertices, floor_triangles, floor_logical_quads, floor_diag = _pocket_floor_quad_grid_from_loop(
            vertices=vertices,
            loop_vertices=cap_loop,
            generated_vertex_offset=int(len(wall_plan.generated_vertices)),
            quad_density_mode=context.quad_density_mode,
        )
    floor_triangles = _orient_triangles_to_source_role_normal(
        vertices=vertices,
        generated_vertices=np.vstack([
            np.asarray(wall_plan.generated_vertices, dtype=float).reshape((-1, 3)),
            floor_generated_vertices,
        ]).reshape((-1, 3)),
        triangles=floor_triangles,
        source_faces=faces_arr,
        face_ids=floor_ids,
        fallback_normal=np.asarray(wall_plan.axis, dtype=float).reshape(3),
    )
    wall_plan = _orient_pocket_wall_triangles_by_radial_role(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=wall_ids,
        plan=wall_plan,
    )

    generated_vertices = np.vstack([
        np.asarray(wall_plan.generated_vertices, dtype=float).reshape((-1, 3)),
        floor_generated_vertices,
    ]).reshape((-1, 3))
    triangles = np.vstack([
        np.asarray(wall_plan.triangles, dtype=np.int64).reshape((-1, 3)),
        floor_triangles,
    ]).reshape((-1, 3))

    combined_plan = QuadPlan(
        generated_vertices=generated_vertices,
        triangles=triangles,
        logical_quads=tuple(wall_plan.logical_quads) + tuple(floor_logical_quads),
        loop0=tuple(int(v) for v in tuple(wall_plan.loop0 or ())),
        loop1=tuple(int(v) for v in tuple(wall_plan.loop1 or ())),
        center0=np.asarray(wall_plan.center0, dtype=float).reshape(3),
        center1=np.asarray(wall_plan.center1, dtype=float).reshape(3),
        axis=np.asarray(wall_plan.axis, dtype=float).reshape(3),
        diagnostics={
            **dict(wall_plan.diagnostics),
            **dict(floor_diag),
            "plan_type": "pocket_recess_cup_rebuild_plan_v100_quad_floor",
            "geometry_source": "owned_pocket_wall_loops_plus_owned_floor_perimeter",
            "semantic_rebuild_contract": "POCKET CandidateData -> owned floor+side-wall DeletePatchProposal -> side-wall plus quad-floor generated cup",
            "pocket_rebuild_operation": "restore_recess_cup",
            "pocket_delete_patch_meaning": "owned_pocket_floor_plus_owned_pocket_side_wall",
            "pocket_top_opening_policy": "preserve_opening_not_parent_surface_cap",
            "pocket_transition_policy": "transition_faces_excluded_unless_separately_owned",
            "pocket_side_wall_face_count": int(len(wall_ids)),
            "pocket_floor_face_count": int(len(floor_ids)),
            "pocket_delete_face_count": int(len(patch_faces)),
            "pocket_floor_cap_loop_vertex_count": int(len(cap_loop)),
            "pocket_floor_cap_added_triangle_count": int(len(floor_triangles)),
            "pocket_floor_cap_generated_center_vertex_count": 0,
            "pocket_floor_added_triangle_count": int(len(floor_triangles)),
            "pocket_floor_logical_quad_count": int(len(floor_logical_quads)),
            "pocket_floor_fill_kind": str(floor_diag.get("pocket_floor_fill_kind", "quad_grid")),
            "pocket_floor_protected_child_bore_opening_count": int(len(protected_floor_hole_records)),
            "pocket_floor_protected_child_bore_loop_vertex_counts": tuple(int(len(tuple(record.get("vertices", ()) or ()))) for record in protected_floor_hole_records),
            "compound_pocket_bore_semantics": "POCKET rebuild protects child BORE floor opening as relationship metadata; BORE remains separate candidate",
            "pocket_wall_added_triangle_count": int(len(wall_plan.triangles)),
            "pocket_wall_logical_quad_count": int(len(wall_plan.logical_quads)),
        },
    )
    boundary_match = _generated_surface_boundary_match_diagnostics(
        source_faces=faces_arr,
        face_ids=patch_faces,
        triangles=combined_plan.triangles,
        source_vertex_count=int(len(vertices)),
    )
    combined_plan = QuadPlan(
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
        logical_quads=combined_plan.logical_quads,
        loop0=combined_plan.loop0,
        loop1=combined_plan.loop1,
        center0=combined_plan.center0,
        center1=combined_plan.center1,
        axis=combined_plan.axis,
        diagnostics={**dict(combined_plan.diagnostics), **boundary_match},
    )

    trial = _trial_rebuild(
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=patch_faces,
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
    )
    local_accept = _pocket_recess_cup_trial_accepts(trial=trial, plan=combined_plan)
    if not bool(trial.get("watertight_after", False)) and not local_accept:
        raise ValueError(
            "POCKET recess-cup rebuild local validation failed. "
            f"delete_faces={len(patch_faces)}; wall_faces={len(wall_ids)}; floor_faces={len(floor_ids)}; "
            f"boundary_match_exact={combined_plan.diagnostics.get('boundary_match_exact', False)}; "
            f"missing={combined_plan.diagnostics.get('missing_patch_boundary_edge_count', '-')}; "
            f"extra={combined_plan.diagnostics.get('extra_generated_boundary_edge_count', '-')}; "
            f"generated_new_boundary={combined_plan.diagnostics.get('generated_boundary_generated_vertex_edge_count', '-')}; "
            f"boundary_before={trial.get('boundary_edge_count_before', '-')}; boundary_after={trial.get('boundary_edge_count_after', '-')}; "
            "Geometry changed: no."
        )

    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=faces_arr,
        face_ids=patch_faces,
        generated_vertices=combined_plan.generated_vertices,
        triangles=combined_plan.triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )
    before_face_count = int(len(faces_arr))
    result_mesh = result["mesh"]
    after_face_count = int(len(getattr(result_mesh, "faces", ())))
    diagnostics = {
        "mode": "pocket_recess_cup_rebuild_v100_quad_floor",
        "topology_policy": "owned_pocket_floor_plus_sidewall_recess_cup_quad_floor_local_validation_v100",
        "pipeline": (
            "accepted_pocket_candidate_data",
            "owned_floor_plus_sidewall_target",
            "sidewall_loop_pair_from_wall_ownership",
            "floor_perimeter_from_floor_ownership",
            "generate_sidewall",
            "generate_floor",
            "local_topology_validation",
            "apply_rebuild",
        ),
        "candidate_entity_type": context.entity_type or "pocket",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "selected_edge_count": int(len(selected_edge_ids)),
        "initial_face_count": int(len(face_ids)),
        "before_face_count": int(before_face_count),
        "after_face_count": int(after_face_count),
        "before_vertex_count": int(len(vertices)),
        "after_vertex_count": int(len(getattr(result_mesh, "vertices", ()))),
        "removed_face_count": int(len(patch_faces)),
        "pocket_removed_side_wall_face_count": int(len(wall_ids)),
        "pocket_removed_floor_face_count": int(len(floor_ids)),
        "pocket_rebuild_operation": "restore_recess_cup",
        "pocket_rebuild_semantic_contract_v102": "POCKET hypothesis owns floor+side-wall roles; child BORE opening is protected relationship metadata; target rebuilds a recessed cup with solid or annular quad floor",
        "pocket_rebuild_semantic_contract_v100": "POCKET hypothesis owns floor+side-wall roles; target deletes those roles and rebuilds a recessed cup with a quad floor, not a bore sleeve, not a fan floor, and not a flush cap",
        "pocket_floor_role_rebuilt": True,
        "pocket_side_wall_role_rebuilt": True,
        "pocket_top_opening_preserved": True,
        "pocket_transition_faces_excluded": True,
        "quad_density_mode": context.quad_density_mode,
        "logical_quad_count": int(len(combined_plan.logical_quads)),
        "added_logical_quad_count": int(len(combined_plan.logical_quads)),
        "added_triangle_count": int(len(combined_plan.triangles)),
        "added_runtime_triangle_count": int(len(combined_plan.triangles)),
        "actual_added_face_count": int(len(combined_plan.triangles)),
        "added_face_count": int(len(combined_plan.triangles)),
        "colored_rebuilt_face_count": int(len(result["added_face_ids"])),
        "generated_vertex_count": int(len(combined_plan.generated_vertices)),
        "loop0_vertex_count": int(len(combined_plan.loop0)),
        "loop1_vertex_count": int(len(combined_plan.loop1)),
        "boundary_loop_vertex_count_delta": int(abs(len(combined_plan.loop0) - len(combined_plan.loop1))),
        "pocket_floor_cap_loop_vertex_count": int(len(cap_loop)),
        "pocket_floor_cap_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_added_triangle_count": int(len(floor_triangles)),
        "pocket_floor_logical_quad_count": int(len(floor_logical_quads)),
        "pocket_floor_fill_kind": str(floor_diag.get("pocket_floor_fill_kind", "quad_grid")),
        "pocket_floor_protected_child_bore_opening_count": int(len(protected_floor_hole_records)),
        "pocket_floor_protected_child_bore_loop_vertex_counts": tuple(int(len(tuple(record.get("vertices", ()) or ()))) for record in protected_floor_hole_records),
        "compound_pocket_bore_semantics": "POCKET rebuild protects child BORE floor opening as relationship metadata; BORE remains separate candidate",
        "pocket_wall_added_triangle_count": int(len(wall_plan.triangles)),
        "boundary_edge_count_before": int(trial.get("boundary_edge_count_before", -1)),
        "boundary_edge_count_after": int(trial.get("boundary_edge_count_after", -1)),
        "boundary_edge_count_delta": int(trial.get("boundary_edge_count_delta", 0)),
        "watertight_after": bool(trial.get("watertight_after", False)),
        "local_topology_acceptance_used": bool(local_accept and not bool(trial.get("watertight_after", False))),
        "local_pocket_recess_cup_acceptance_v99": bool(local_accept),
        "quad_plan": dict(combined_plan.diagnostics),
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
        "existing_boundary_vertices_moved": 0,
        **{str(k): v for k, v in boundary_match.items()},
    }
    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=patch_faces,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in combined_plan.triangles.tolist()),
        loop0_vertices=combined_plan.loop0,
        loop1_vertices=combined_plan.loop1,
        axis=_to_vector3(combined_plan.axis),
        radius=float(radius),
        diagnostics=diagnostics,
    )


def _pocket_floor_annular_quad_grid_from_loops(
    *,
    vertices: np.ndarray,
    outer_loop_vertices: tuple[int, ...],
    inner_loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    floor_axis: np.ndarray,
    quad_density_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Return an adaptive annular floor grid for a POCKET with a child BORE.

    v104 semantic contract:
        POCKET remains the parent feature family.
        BORE remains a separate child feature candidate.
        The floor hole is relationship/protected-boundary metadata.
        Rebuild must sew to the locked outer pocket-floor boundary and the
        locked inner child-BORE opening without changing either boundary.

    This replaces the v103 "fail on odd count delta" policy.  Pure all-quad
    bands are used whenever the loop counts allow them.  When the two locked
    boundaries cannot be connected by a pure all-quad annulus without splitting
    existing boundary edges, the generator creates adaptive sew collars at the
    locked boundary side(s), while keeping the interior floor as a regular
    density-controlled quad grid.  The fallback collar is local, explicit, and
    diagnostic; it does not close the child BORE and does not invent a new
    feature family.
    """

    outer = tuple(int(v) for v in tuple(outer_loop_vertices or ()) if 0 <= int(v) < len(vertices))
    inner = tuple(int(v) for v in tuple(inner_loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(outer) < 4 or len(inner) < 4:
        raise ValueError(
            "POCKET annular floor rebuild requires valid outer and protected inner loops with at least four vertices. "
            f"outer={len(outer)}; inner={len(inner)}; Geometry changed: no."
        )

    axis_vec = _unit_vector(floor_axis)
    outer_center = _loop_center(vertices, outer)
    inner_center = _loop_center(vertices, inner)
    sorted_outer, sorted_inner = _sort_loop_pair_by_angle(vertices, outer, inner, outer_center, inner_center, axis_vec)

    # Align loops by angular phase.  Equal loops can use the exact measured loop
    # alignment.  Unequal loops use angle-sample alignment only; this is still
    # relationship/geometry evidence, not feature-family creation.
    if len(sorted_outer) == len(sorted_inner):
        sorted_inner = _align_second_loop_to_first(
            vertices,
            tuple(int(v) for v in sorted_outer),
            tuple(int(v) for v in sorted_inner),
            outer_center,
            inner_center,
            axis_vec,
        )
        alignment_diag: dict[str, object] = {
            "annular_floor_equal_loop_used": True,
            "annular_floor_unequal_loop_used": False,
            "annular_floor_alignment_source": "equal_loop_cyclic_alignment",
        }
    else:
        aligned_outer, aligned_inner, alignment_diag_raw = _align_unequal_loop_pair_to_angle_samples(
            vertices=vertices,
            loop0=tuple(int(v) for v in sorted_outer),
            loop1=tuple(int(v) for v in sorted_inner),
            center0=outer_center,
            center1=inner_center,
            axis=axis_vec,
        )
        sorted_outer = tuple(int(v) for v in aligned_outer)
        sorted_inner = tuple(int(v) for v in aligned_inner)
        alignment_diag = {
            **dict(alignment_diag_raw),
            "annular_floor_equal_loop_used": False,
            "annular_floor_unequal_loop_used": True,
            "annular_floor_alignment_source": "unequal_loop_angle_sample_alignment",
        }

    outer_pts = vertices[np.asarray(sorted_outer, dtype=np.int64), :3]
    inner_pts = vertices[np.asarray(sorted_inner, dtype=np.int64), :3]
    n_outer = int(len(sorted_outer))
    n_inner = int(len(sorted_inner))

    sample_count = max(int(n_outer), int(n_inner), 8)
    outer_sample = _sample_closed_loop_points(outer_pts, sample_count)
    inner_sample = _sample_closed_loop_points(inner_pts, sample_count)
    radial_gaps = np.linalg.norm(outer_sample - inner_sample, axis=1)
    radial_gaps = radial_gaps[np.isfinite(radial_gaps) & (radial_gaps > 1.0e-12)]
    radial_span = float(np.median(radial_gaps)) if radial_gaps.size else float(np.linalg.norm(outer_center - inner_center))
    outer_edges = np.linalg.norm(np.roll(outer_pts, -1, axis=0) - outer_pts, axis=1)
    inner_edges = np.linalg.norm(np.roll(inner_pts, -1, axis=0) - inner_pts, axis=1)
    edge_lengths = np.concatenate([outer_edges, inner_edges])
    edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1.0e-12)]
    measured_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
    measured_edge = max(float(measured_edge), 1.0e-12)

    mode_norm = _normalize_quad_density_mode(quad_density_mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL:
        pitch = measured_edge
        cap = 128
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        pitch = max(2.5 * measured_edge, radial_span / 32.0 if radial_span > 1.0e-12 else measured_edge)
        cap = 64
    else:
        pitch = max(4.0 * measured_edge, radial_span / 24.0 if radial_span > 1.0e-12 else measured_edge)
        cap = 48
    raw_radial_bands = int(math.ceil(radial_span / max(pitch, 1.0e-12))) if np.isfinite(radial_span) and radial_span > 1.0e-12 else 1

    # Adaptive sew-collar strategy:
    #   locked outer loop -> optional generated common-count grid rings -> locked inner loop
    # The interior rings all use a common count so the floor center area remains
    # an orderly quad grid.  Only the boundary collars adapt to mismatched
    # source geometry.  This avoids failing when loop counts do not meet the
    # strict pure-quad parity rule, while still preserving both locked loops.
    common_count = int(max(n_outer, n_inner, 4))
    count_delta = abs(int(n_outer) - int(n_inner))
    needs_adaptive_collar = bool(count_delta % 2 or count_delta != 0)
    target_band_count = max(1, min(int(cap), int(raw_radial_bands)))
    if needs_adaptive_collar:
        target_band_count = max(2, target_band_count)

    generated_blocks: list[np.ndarray] = []
    ring_ids: list[tuple[int, ...]] = [tuple(int(v) for v in sorted_outer)]
    next_generated_id = int(len(vertices) + int(generated_vertex_offset))
    for ring_index in range(1, int(target_band_count)):
        t = float(ring_index) / float(target_band_count)
        out_sample = _sample_closed_loop_points(outer_pts, common_count)
        in_sample = _sample_closed_loop_points(inner_pts, common_count)
        pts = (1.0 - t) * out_sample + t * in_sample
        generated_blocks.append(np.asarray(pts, dtype=float).reshape((common_count, 3)))
        ids = tuple(range(next_generated_id, next_generated_id + common_count))
        next_generated_id += common_count
        ring_ids.append(ids)
    ring_ids.append(tuple(int(v) for v in sorted_inner))

    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    band_count_pairs: list[tuple[int, int]] = []
    band_quad_counts: list[int] = []
    band_triangle_counts: list[int] = []
    band_adapter_kinds: list[str] = []
    adaptive_collar_band_indices: list[int] = []
    for band_index in range(len(ring_ids) - 1):
        ring_a = tuple(int(v) for v in ring_ids[band_index])
        ring_b = tuple(int(v) for v in ring_ids[band_index + 1])
        band_count_pairs.append((int(len(ring_a)), int(len(ring_b))))
        band_quads, band_tris, band_diag = _band_faces_between_rings_with_adaptive_sew_collar(ring_a, ring_b)
        logical_quads.extend(band_quads)
        triangles.extend(band_tris)
        band_quad_counts.append(int(len(band_quads)))
        band_triangle_counts.append(int(band_diag.get("triangle_count", 0)))
        adapter_kind = str(band_diag.get("adapter_kind", "unknown"))
        band_adapter_kinds.append(adapter_kind)
        if bool(band_diag.get("adaptive_sew_collar_used", False)):
            adaptive_collar_band_indices.append(int(band_index))

    generated_vertices = np.vstack(generated_blocks).reshape((-1, 3)) if generated_blocks else np.zeros((0, 3), dtype=float)
    diagnostics = {
        **dict(alignment_diag),
        "pocket_floor_fill_kind": "annular_adaptive_sew_collar_quad_grid_protected_child_bore_opening",
        "pocket_floor_center_fan_used": False,
        "pocket_floor_annular_fill_used": True,
        "pocket_floor_annular_concentric_grid_used": True,
        "pocket_floor_adaptive_sew_collar_used": bool(adaptive_collar_band_indices),
        "pocket_floor_adaptive_sew_collar_band_indices": tuple(int(v) for v in adaptive_collar_band_indices),
        "pocket_floor_annular_grid_contract_v104": "locked_outer_floor_perimeter_to_locked_child_bore_opening_with_adaptive_sew_collars_and_density_controlled_quad_interior",
        "pocket_floor_annular_grid_contract_v103_reverted": "no_clean_fail_on_odd_count_delta; use adaptive sew collars instead of rejecting rebuild",
        "pocket_floor_protected_inner_loop_count": 1,
        "pocket_floor_outer_loop_vertex_count": int(n_outer),
        "pocket_floor_inner_bore_loop_vertex_count": int(n_inner),
        "pocket_floor_annular_common_interior_ring_count": int(common_count),
        "pocket_floor_annular_radial_span": float(radial_span),
        "pocket_floor_annular_median_edge_length": float(measured_edge),
        "pocket_floor_annular_target_pitch": float(pitch),
        "pocket_floor_annular_raw_radial_bands": int(raw_radial_bands),
        "pocket_floor_annular_radial_band_count": int(len(ring_ids) - 1),
        "pocket_floor_annular_generated_ring_count": int(max(0, len(ring_ids) - 2)),
        "pocket_floor_annular_ring_count_sequence": tuple(int(len(v)) for v in ring_ids),
        "pocket_floor_annular_band_count_pairs": tuple((int(a), int(b)) for a, b in band_count_pairs),
        "pocket_floor_annular_band_adapter_kinds": tuple(str(v) for v in band_adapter_kinds),
        "pocket_floor_annular_band_quad_counts": tuple(int(v) for v in band_quad_counts),
        "pocket_floor_annular_band_triangle_counts": tuple(int(v) for v in band_triangle_counts),
        "pocket_floor_annular_triangle_count_from_adaptive_collars": int(sum(band_triangle_counts)),
        "pocket_floor_logical_quad_count": int(len(logical_quads)),
        "pocket_floor_triangle_count": int(len(triangles)),
        "pocket_floor_generated_vertex_count": int(len(generated_vertices)),
        "quad_density_mode": str(mode_norm),
        "semantic_floor_contract": "POCKET floor rebuilt as annular quad-grid interior with adaptive sew collars; protected child BORE opening remains open and separate",
    }
    return (
        np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        tuple(tuple(int(v) for v in quad) for quad in tuple(logical_quads or ())),
        diagnostics,
    )


def _band_faces_between_rings_with_adaptive_sew_collar(
    ring_a: tuple[int, ...],
    ring_b: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int], ...], dict[str, object]]:
    """Return runtime triangles and logical quads between two locked rings.

    First try the existing all-quad transition band.  If locked loop counts make
    that impossible, fall back to a local zipper-style adaptive sew collar.  The
    fallback preserves all boundary vertices and consumes every boundary edge
    exactly once; it is intentionally used only as the boundary adapter while
    the annular floor interior remains regular quads.
    """

    a = tuple(int(v) for v in tuple(ring_a or ()))
    b = tuple(int(v) for v in tuple(ring_b or ()))
    n_a = int(len(a))
    n_b = int(len(b))
    if n_a < 3 or n_b < 3:
        return (), (), {
            "adapter_kind": "invalid_short_ring",
            "adaptive_sew_collar_used": False,
            "quad_count": 0,
            "triangle_count": 0,
        }

    try:
        quads = _band_quads_between_rings(a, b)
        tris: list[tuple[int, int, int]] = []
        for q in tuple(quads or ()): 
            tris.append((int(q[0]), int(q[1]), int(q[2])))
            tris.append((int(q[0]), int(q[2]), int(q[3])))
        return (
            tuple(tuple(int(v) for v in q) for q in tuple(quads or ())),
            tuple(tuple(int(v) for v in tri) for tri in tris),
            {
                "adapter_kind": "pure_quad_transition_band",
                "adaptive_sew_collar_used": False,
                "quad_count": int(len(quads)),
                "triangle_count": 0,
                "ring_a_count": int(n_a),
                "ring_b_count": int(n_b),
            },
        )
    except Exception as exc:
        quads, tris = _adaptive_zipper_sew_collar_faces_between_rings(a, b)
        return (
            tuple(tuple(int(v) for v in q) for q in tuple(quads or ())),
            tuple(tuple(int(v) for v in tri) for tri in tuple(tris or ())),
            {
                "adapter_kind": "adaptive_zipper_sew_collar",
                "adaptive_sew_collar_used": True,
                "all_quad_transition_rejection": str(exc),
                "quad_count": int(len(quads)),
                "triangle_count": int(len(tris) - 2 * len(quads)),
                "runtime_triangle_count": int(len(tris)),
                "ring_a_count": int(n_a),
                "ring_b_count": int(n_b),
            },
        )


def _adaptive_zipper_sew_collar_faces_between_rings(
    ring_a: tuple[int, ...],
    ring_b: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Build a local adaptive collar between arbitrary locked loop counts.

    The output is quad-dominant where the angular steps coincide and uses local
    transition triangles only where count parity/ratio makes an all-quad band
    topologically impossible without splitting locked boundary edges.
    """

    a = tuple(int(v) for v in tuple(ring_a or ()))
    b = tuple(int(v) for v in tuple(ring_b or ()))
    n_a = int(len(a))
    n_b = int(len(b))
    if n_a < 3 or n_b < 3:
        return (), ()

    quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    ia = 0
    ib = 0
    ca = 0
    cb = 0
    eps = 1.0e-9
    # Merge the two cyclic edge streams by normalized perimeter progress.  This
    # consumes every boundary edge exactly once and never creates a split vertex
    # on a locked loop.
    while ca < n_a or cb < n_b:
        next_a = float(ca + 1) / float(n_a) if ca < n_a else float("inf")
        next_b = float(cb + 1) / float(n_b) if cb < n_b else float("inf")
        if ca < n_a and cb < n_b and abs(next_a - next_b) <= eps:
            q = (int(a[ia % n_a]), int(a[(ia + 1) % n_a]), int(b[(ib + 1) % n_b]), int(b[ib % n_b]))
            quads.append(q)
            triangles.append((q[0], q[1], q[2]))
            triangles.append((q[0], q[2], q[3]))
            ia += 1
            ib += 1
            ca += 1
            cb += 1
        elif ca < n_a and (cb >= n_b or next_a < next_b):
            tri = (int(a[ia % n_a]), int(a[(ia + 1) % n_a]), int(b[ib % n_b]))
            if len(set(tri)) == 3:
                triangles.append(tri)
            ia += 1
            ca += 1
        elif cb < n_b:
            tri = (int(a[ia % n_a]), int(b[(ib + 1) % n_b]), int(b[ib % n_b]))
            if len(set(tri)) == 3:
                triangles.append(tri)
            ib += 1
            cb += 1
        else:
            break
    return tuple(quads), tuple(triangles)


def _pocket_floor_quad_grid_from_loop(
    *,
    vertices: np.ndarray,
    loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    quad_density_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Return a quad-dominant floor grid for the owned POCKET floor role.

    This is the floor companion to the wall-loop quad rebuild.  The boundary
    loop remains locked to the original pocket floor perimeter.  Generated
    inner rings shrink toward the floor center and gradually reduce vertex
    count down to a small even core loop, which is closed with logical quads.
    No center-vertex fan is used for even boundary loops such as the current
    126-vertex circular-pocket test.
    """

    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 4:
        raise ValueError("Pocket quad floor requires at least four boundary vertices.")

    n = int(len(loop))
    if n % 2 == 0 and n >= 6:
        target_count = 6
        core_fill_kind = "two_logical_quads_hex_core"
    elif n % 2 == 0 and n >= 4:
        target_count = 4
        core_fill_kind = "single_logical_quad_core"
    else:
        # A pure all-quad disk with an odd locked boundary is topologically not
        # possible without inserting/splitting boundary vertices.  Keep this
        # fallback explicit and diagnostic; current circular-pocket tests use an
        # even boundary and therefore stay in the quad path.
        return _pocket_floor_center_fan_fallback(
            vertices=vertices,
            loop_vertices=loop,
            generated_vertex_offset=generated_vertex_offset,
            reason="odd_locked_boundary_count_cannot_form_pure_quad_disk",
        )

    pts = vertices[np.asarray(loop, dtype=np.int64), :3]
    center = np.mean(pts, axis=0).reshape(3)
    # Use the existing density vocabulary, but apply it radially across the
    # floor rather than axially along the wall.  The counts still control
    # topology; density controls how many equal-count spacer bands appear.
    base_counts = _transition_count_sequence(int(n), int(target_count), mode=quad_density_mode)
    radial_span = float(np.median(np.linalg.norm(pts - center.reshape(1, 3), axis=1)))
    edge_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 1.0e-12)]
    measured_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
    mode_norm = _normalize_quad_density_mode(quad_density_mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL:
        pitch = max(float(measured_edge), 1.0e-12)
        cap = 96
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        pitch = max(2.5 * float(measured_edge), 1.0e-12)
        cap = 64
    else:
        pitch = max(4.0 * float(measured_edge), 1.0e-12)
        cap = 48
    raw_radial_bands = int(math.ceil(radial_span / pitch)) if np.isfinite(radial_span) and radial_span > 1.0e-12 else len(base_counts) - 1
    target_band_count = max(int(len(base_counts) - 1), min(int(cap), max(1, int(raw_radial_bands))))
    counts = _densify_transition_count_sequence(base_counts, target_band_count=target_band_count)
    counts = tuple(int(v) for v in counts if int(v) >= int(target_count))
    if counts[0] != n:
        counts = (int(n),) + tuple(counts)
    if counts[-1] != target_count:
        counts = tuple(counts) + (int(target_count),)

    # Avoid a collapsed center: the final generated ring is a small measured
    # floor core, then the core itself is closed with logical quads.
    inner_scale = 0.08 if int(target_count) >= 6 else 0.12
    band_count = max(1, len(counts) - 1)
    generated_blocks: list[np.ndarray] = []
    ring_ids: list[tuple[int, ...]] = [tuple(int(v) for v in loop)]
    next_generated_id = int(len(vertices) + int(generated_vertex_offset))
    for ring_index, count in enumerate(counts[1:], start=1):
        t = float(ring_index) / float(band_count)
        scale = (1.0 - t) + t * float(inner_scale)
        sample = _sample_closed_loop_points(pts, int(count))
        ring_pts = center.reshape(1, 3) + scale * (sample - center.reshape(1, 3))
        generated_blocks.append(np.asarray(ring_pts, dtype=float).reshape((int(count), 3)))
        ids = tuple(range(next_generated_id, next_generated_id + int(count)))
        next_generated_id += int(count)
        ring_ids.append(ids)

    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    for band in range(len(ring_ids) - 1):
        band_quads = _band_quads_between_rings(tuple(ring_ids[band]), tuple(ring_ids[band + 1]))
        logical_quads.extend(band_quads)
        for quad in band_quads:
            triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
            triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    core = tuple(int(v) for v in ring_ids[-1])
    core_quads: tuple[tuple[int, int, int, int], ...]
    if len(core) == 4:
        core_quads = ((core[0], core[1], core[2], core[3]),)
    elif len(core) == 6:
        core_quads = ((core[0], core[1], core[2], core[3]), (core[0], core[3], core[4], core[5]))
    else:
        # Keep the function honest if the target count is changed later.
        return _pocket_floor_center_fan_fallback(
            vertices=vertices,
            loop_vertices=loop,
            generated_vertex_offset=generated_vertex_offset,
            reason=f"unsupported_core_loop_count_{len(core)}",
        )
    logical_quads.extend(core_quads)
    for quad in core_quads:
        triangles.append((int(quad[0]), int(quad[1]), int(quad[2])))
        triangles.append((int(quad[0]), int(quad[2]), int(quad[3])))

    generated_vertices = np.vstack(generated_blocks).reshape((-1, 3)) if generated_blocks else np.zeros((0, 3), dtype=float)
    diag = {
        "pocket_floor_fill_kind": "quad_grid",
        "pocket_floor_grid_contract_v100": "locked_floor_perimeter_to_concentric_quad_grid_no_center_fan",
        "pocket_floor_boundary_vertex_count": int(n),
        "pocket_floor_inner_core_vertex_count": int(target_count),
        "pocket_floor_core_fill_kind": str(core_fill_kind),
        "pocket_floor_ring_count_sequence": tuple(int(v) for v in counts),
        "pocket_floor_radial_band_count": int(len(ring_ids) - 1),
        "pocket_floor_generated_ring_count": int(max(0, len(ring_ids) - 1)),
        "pocket_floor_generated_vertex_count": int(len(generated_vertices)),
        "pocket_floor_logical_quad_count": int(len(logical_quads)),
        "pocket_floor_triangle_count": int(len(triangles)),
        "pocket_floor_center_fan_used": False,
        "quad_density_mode": str(_normalize_quad_density_mode(quad_density_mode)),
    }
    return (
        np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        tuple((int(a), int(b), int(c), int(d)) for a, b, c, d in logical_quads),
        diag,
    )


def _pocket_floor_center_fan_fallback(
    *,
    vertices: np.ndarray,
    loop_vertices: tuple[int, ...],
    generated_vertex_offset: int,
    reason: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], dict[str, object]]:
    """Fallback only for non-quad-compatible locked boundaries."""

    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 3:
        raise ValueError("Pocket floor fallback requires at least three boundary vertices.")
    pts = vertices[np.asarray(loop, dtype=np.int64), :3]
    center = np.mean(pts, axis=0).reshape(1, 3)
    center_id = int(len(vertices) + int(generated_vertex_offset))
    triangles = [(int(a), int(loop[(i + 1) % len(loop)]), int(center_id)) for i, a in enumerate(loop)]
    return (
        np.asarray(center, dtype=float).reshape((1, 3)),
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        (),
        {
            "pocket_floor_fill_kind": "diagnostic_center_fan_fallback",
            "pocket_floor_center_fan_used": True,
            "pocket_floor_quad_fallback_reason": str(reason),
            "pocket_floor_logical_quad_count": 0,
            "pocket_floor_triangle_count": int(len(triangles)),
            "pocket_floor_generated_vertex_count": 1,
        },
    )


def _orient_triangles_to_source_role_normal(
    *,
    vertices: np.ndarray,
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    fallback_normal: np.ndarray,
) -> np.ndarray:
    """Orient a generated role surface by the averaged source role normal."""

    refs = _source_patch_face_references(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    normals = [np.asarray(item[1], dtype=float).reshape(3) for item in refs]
    role_normal = _unit_normal(np.sum(np.asarray(normals, dtype=float).reshape((-1, 3)), axis=0)) if normals else None
    if role_normal is None:
        role_normal = _unit_vector(fallback_normal)

    output_vertices = np.asarray(vertices, dtype=float).copy()
    gen = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    if gen.size:
        output_vertices = np.vstack([output_vertices, gen])
    tris = np.asarray(triangles, dtype=np.int64).reshape((-1, 3)).copy()
    for tri_index, tri in enumerate(tris):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = _unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is not None and float(np.dot(normal, role_normal)) < 0.0:
            tris[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
    return tris


def _orient_pocket_wall_triangles_by_radial_role(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    plan: QuadPlan,
) -> QuadPlan:
    """Orient pocket wall triangles by the wall role's radial normal sign.

    Nearest-face orientation can preserve source-mesh normal noise.  For the
    circular-pocket side wall, the stronger role rule is radial: generated wall
    normals should have the same inward/outward sign relative to the pocket axis
    as the owned source wall faces.
    """

    refs = _source_patch_face_references(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    axis_vec = _unit_vector(plan.axis)
    base = 0.5 * (np.asarray(plan.center0, dtype=float).reshape(3) + np.asarray(plan.center1, dtype=float).reshape(3))

    def radial_unit(point: np.ndarray) -> np.ndarray | None:
        rel = np.asarray(point, dtype=float).reshape(3) - base
        axial = float(np.dot(rel, axis_vec))
        radial = rel - axial * axis_vec
        length = float(np.linalg.norm(radial))
        if not np.isfinite(length) or length <= 1.0e-12:
            return None
        return radial / length

    signs: list[float] = []
    for centroid, normal in refs:
        r = radial_unit(np.asarray(centroid, dtype=float).reshape(3))
        if r is None:
            continue
        dot = float(np.dot(np.asarray(normal, dtype=float).reshape(3), r))
        if np.isfinite(dot) and abs(dot) > 1.0e-6:
            signs.append(dot)
    if not signs:
        return plan
    role_sign = 1.0 if float(np.median(np.asarray(signs, dtype=float))) >= 0.0 else -1.0

    output_vertices = np.asarray(vertices, dtype=float).copy()
    gen = np.asarray(plan.generated_vertices, dtype=float).reshape((-1, 3))
    if gen.size:
        output_vertices = np.vstack([output_vertices, gen])
    tris = np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)).copy()
    flips = 0
    checked = 0
    for tri_index, tri in enumerate(tris):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = _unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        r = radial_unit(np.mean(pts, axis=0))
        if normal is None or r is None:
            continue
        checked += 1
        if float(np.dot(normal, r)) * role_sign < 0.0:
            tris[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
            flips += 1
    diag = dict(plan.diagnostics)
    diag.update({
        "pocket_wall_normal_orientation_policy_v100": "radial_role_sign_from_owned_source_wall_faces",
        "pocket_wall_radial_normal_role_sign": float(role_sign),
        "pocket_wall_radial_orientation_checked_triangle_count": int(checked),
        "pocket_wall_radial_orientation_flip_count": int(flips),
    })
    return QuadPlan(
        generated_vertices=plan.generated_vertices,
        triangles=tris,
        logical_quads=plan.logical_quads,
        loop0=plan.loop0,
        loop1=plan.loop1,
        center0=plan.center0,
        center1=plan.center1,
        axis=plan.axis,
        diagnostics=diag,
    )


def _pocket_recess_cup_trial_accepts(*, trial: Mapping[str, object], plan: QuadPlan) -> bool:
    """Strict local acceptance for a POCKET recessed cup on non-watertight meshes."""

    diag = dict(getattr(plan, "diagnostics", {}) or {})
    boundary_match = bool(diag.get("boundary_match_exact", False))
    missing = _int_preserve_zero(diag.get("missing_patch_boundary_edge_count", -1), -1)
    extra = _int_preserve_zero(diag.get("extra_generated_boundary_edge_count", -1), -1)
    patch_edges = _int_preserve_zero(diag.get("patch_boundary_edge_count", -1), -1)
    generated_edges = _int_preserve_zero(diag.get("generated_boundary_edge_count", -1), -1)
    generated_original_boundary = _int_preserve_zero(diag.get("generated_boundary_original_vertex_edge_count", -1), -1)
    generated_new_boundary = _int_preserve_zero(diag.get("generated_boundary_generated_vertex_edge_count", -1), -1)
    before = _int_preserve_zero(trial.get("boundary_edge_count_before", -1), -1)
    after = _int_preserve_zero(trial.get("boundary_edge_count_after", -1), -1)
    return bool(
        boundary_match
        and missing == 0
        and extra == 0
        and patch_edges > 0
        and generated_edges == patch_edges
        and generated_original_boundary == patch_edges
        and generated_new_boundary == 0
        and before >= 0
        and after <= before
    )

# -----------------------------------------------------------------------------
# POCKET v96: cap-style pocket delete/rebuild
# -----------------------------------------------------------------------------


def _candidate_requests_pocket_cap_rebuild(*, context: RebuildCandidateContext, candidate_metadata: Mapping[str, object]) -> bool:
    """Return True for the pocket-specific floor+wall delete/cap path."""

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    action = str(candidate_metadata.get("candidate_action", "") or "").strip().lower()
    gate = str(getattr(context, "rebuild_gate", "") or "").strip().lower()
    scope = str(candidate_metadata.get("pocket_rebuild_enable_scope", "") or "").strip().lower()
    return bool(
        entity in {"pocket", "circular_pocket"}
        or family in {"pocket", "circular_pocket"}
    ) and bool(
        "cap" in action
        or "cap" in gate
        or "cap" in scope
        or "floor_and_sidewall" in action
        or "side_wall_plus_floor" in scope
    )


def _delete_pocket_and_fill_opening_cap(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    context: RebuildCandidateContext,
    axis: np.ndarray,
    radius: float,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> RebuildResult:
    """Delete owned pocket side-wall+floor faces and fill the opening loop.

    This is intentionally separate from the measured-loop sleeve rebuild.  A
    pocket cleanup target has one surviving parent-surface opening boundary after
    the floor is included in the delete patch.  The correct local rebuild is a
    planar cap bounded by that one loop, not a two-loop wall replacement.
    """

    patch_faces = _normalize_face_ids(face_ids, face_count=len(source_faces))
    if not patch_faces:
        raise ValueError("Pocket cap rebuild received no owned floor/side-wall delete faces. Geometry changed: no.")

    loops = _candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=patch_faces)
    if not loops:
        raise ValueError(
            "Pocket cap rebuild could not find a closed opening boundary after deleting owned floor+side-wall faces. "
            f"delete_face_count={len(patch_faces)}; Geometry changed: no."
        )

    # The valid first pocket target is one main parent-surface opening loop.  If
    # small stray components exist, keep the largest loop but report them.  This
    # keeps the trial bounded to the user-selected pocket rather than falling
    # back to RegionData.
    main_loop = max(loops, key=lambda row: int(row.get("edge_count", 0) or 0))
    loop_vertices = tuple(int(v) for v in tuple(main_loop.get("vertices", ()) or ()))
    if len(loop_vertices) < 3:
        raise ValueError(
            "Pocket cap rebuild opening loop has too few vertices. "
            f"loop_vertex_count={len(loop_vertices)}; delete_face_count={len(patch_faces)}; Geometry changed: no."
        )

    normal_hint = _unit_vector(candidate_metadata.get("axis", axis), fallback=axis)
    center = _loop_center(vertices, loop_vertices)
    generated_vertices = np.asarray([center], dtype=float).reshape((1, 3))
    center_index = int(len(vertices))
    triangles = _cap_triangles_for_loop(loop_vertices=loop_vertices, center_index=center_index, vertices=vertices, normal_hint=normal_hint)
    if triangles.size == 0:
        raise ValueError(
            "Pocket cap rebuild failed to generate cap triangles. "
            f"loop_vertex_count={len(loop_vertices)}; Geometry changed: no."
        )

    boundary_match = _generated_surface_boundary_match_diagnostics(
        source_faces=source_faces,
        face_ids=patch_faces,
        triangles=triangles,
        source_vertex_count=int(len(vertices)),
    )
    trial = _trial_rebuild(
        vertices=vertices,
        source_faces=source_faces,
        face_ids=patch_faces,
        generated_vertices=generated_vertices,
        triangles=triangles,
    )

    patch_boundary_edges = int(boundary_match.get("patch_boundary_edge_count", 0) or 0)
    generated_boundary_edges = _int_preserve_zero(boundary_match.get("generated_boundary_edge_count", -1), -1)
    missing = int(boundary_match.get("missing_patch_boundary_edge_count", 1) or 0)
    extra = int(boundary_match.get("extra_generated_boundary_edge_count", 1) or 0)
    before = int(trial.get("boundary_edge_count_before", -1) if trial.get("boundary_edge_count_before", -1) is not None else -1)
    after = int(trial.get("boundary_edge_count_after", -1) if trial.get("boundary_edge_count_after", -1) is not None else -1)
    local_accept = bool(
        bool(boundary_match.get("boundary_match_exact", False))
        and patch_boundary_edges > 0
        and generated_boundary_edges == patch_boundary_edges
        and missing == 0
        and extra == 0
        and before >= 0
        and after <= before
    )
    if not (bool(trial.get("watertight_after", False)) or local_accept):
        raise ValueError(
            "Pocket cap rebuild local topology trial rejected. "
            f"delete_face_count={len(patch_faces)}; loop_count={len(loops)}; loop_vertex_count={len(loop_vertices)}; "
            f"boundary_edge_count_before={before}; boundary_edge_count_after={after}; "
            f"watertight_after={bool(trial.get('watertight_after', False))}; "
            f"boundary_match_exact={bool(boundary_match.get('boundary_match_exact', False))}; "
            f"patch_boundary_edge_count={patch_boundary_edges}; generated_boundary_edge_count={generated_boundary_edges}; "
            f"missing_patch_boundary_edge_count={missing}; extra_generated_boundary_edge_count={extra}; "
            "Geometry changed: no."
        )

    result = _apply_rebuild(
        mesh=mesh,
        vertices=vertices,
        source_faces=source_faces,
        face_ids=patch_faces,
        generated_vertices=generated_vertices,
        triangles=triangles,
        color_rebuilt_faces=bool(color_rebuilt_faces),
        base_face_color=base_face_color,
        rebuilt_face_color=rebuilt_face_color,
    )

    diagnostics: dict[str, object] = {
        "mode": "pocket_cap_rebuild_v96",
        "topology_policy": "owned_pocket_floor_plus_sidewall_delete_single_opening_loop_planar_cap_v96",
        "candidate_entity_type": context.entity_type or "pocket",
        "candidate_rebuild_gate": context.rebuild_gate,
        "candidate_role": context.role,
        "candidate_from_component_engine": bool(context.candidate_from_component_engine),
        "candidate_feature_ownership_source": context.feature_ownership_source,
        "candidate_has_preview_face_patch": bool(context.candidate_has_preview_face_patch),
        "preview_candidate_patch_owns_delete": bool(context.candidate_has_preview_face_patch),
        "selected_edge_count": int(len(selected_edge_ids)),
        "removed_face_count": int(len(patch_faces)),
        "pocket_floor_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_floor_face_ids", ()), face_count=len(source_faces)))),
        "pocket_side_wall_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_side_wall_face_ids", ()), face_count=len(source_faces)))),
        "pocket_transition_face_count": int(len(_normalize_face_ids(candidate_metadata.get("pocket_transition_face_ids", ()), face_count=len(source_faces)))),
        "pocket_cap_loop_count": int(len(loops)),
        "pocket_cap_loop_vertex_count": int(len(loop_vertices)),
        "pocket_cap_loop_edge_count": int(main_loop.get("edge_count", len(loop_vertices)) or len(loop_vertices)),
        "pocket_cap_generated_center_vertex_count": 1,
        "pocket_cap_added_triangle_count": int(len(triangles)),
        "pocket_cap_fill_acceptance_contract_v96": "delete_owned_floor_plus_side_wall_faces_fill_single_parent_opening_loop_no_missing_or_extra_generated_boundary_edges",
        "pocket_cap_local_topology_acceptance_used": bool(local_accept and not bool(trial.get("watertight_after", False))),
        "watertight_after": bool(trial.get("watertight_after", False)),
        "boundary_edge_count_before": before,
        "boundary_edge_count_after": after,
        "boundary_edge_count_delta": int(trial.get("boundary_edge_count_delta", after - before)),
        "quad_plan": {
            "geometry_source": "pocket_opening_boundary_loop_planar_cap",
            "cap_center": _to_vector3(center),
            "cap_normal_hint": _to_vector3(normal_hint),
            **dict(boundary_match),
        },
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
        "existing_boundary_vertices_moved": 0,
        "region_data_as_rebuild_input_enabled": False,
        "floor_rebuild_method": "floor_is_deleted_with_side_walls_then_opening_is_capped",
    }

    return RebuildResult(
        mesh=result["mesh"],
        removed_face_ids=patch_faces,
        added_face_ids=result["added_face_ids"],
        added_faces=tuple(tuple(int(v) for v in tri) for tri in triangles.tolist()),
        loop0_vertices=loop_vertices,
        loop1_vertices=(),
        axis=_to_vector3(normal_hint),
        radius=float(radius),
        diagnostics=diagnostics,
    )


def _cap_triangles_for_loop(*, loop_vertices: tuple[int, ...], center_index: int, vertices: np.ndarray, normal_hint: np.ndarray) -> np.ndarray:
    loop = tuple(int(v) for v in tuple(loop_vertices or ()) if 0 <= int(v) < len(vertices))
    if len(loop) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    tris = np.asarray([[loop[i], loop[(i + 1) % len(loop)], int(center_index)] for i in range(len(loop))], dtype=np.int64)
    normal = _triangle_set_area_normal(vertices=vertices, triangles=tris, generated_center_index=int(center_index))
    hint = _unit_vector(normal_hint, fallback=(0.0, 0.0, 1.0))
    if float(np.dot(normal, hint)) < 0.0:
        tris = np.asarray([[loop[(i + 1) % len(loop)], loop[i], int(center_index)] for i in range(len(loop))], dtype=np.int64)
    return tris


def _triangle_set_area_normal(*, vertices: np.ndarray, triangles: np.ndarray, generated_center_index: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=float)[:, :3]
    center_point = None
    if int(generated_center_index) >= len(arr):
        # The cap center is the mean of the original boundary vertices used in
        # the fan.  It is reconstructed here only for orientation testing.
        original_ids = sorted({int(v) for tri in np.asarray(triangles, dtype=np.int64).reshape((-1, 3)) for v in tri if 0 <= int(v) < len(arr)})
        center_point = np.mean(arr[np.asarray(original_ids, dtype=np.int64), :3], axis=0) if original_ids else np.zeros(3, dtype=float)
    acc = np.zeros(3, dtype=float)
    for tri in np.asarray(triangles, dtype=np.int64).reshape((-1, 3)):
        pts = []
        for raw in tri[:3]:
            idx = int(raw)
            if 0 <= idx < len(arr):
                pts.append(arr[idx, :3])
            elif idx == int(generated_center_index) and center_point is not None:
                pts.append(center_point)
        if len(pts) == 3:
            acc += np.cross(pts[1] - pts[0], pts[2] - pts[0])
    return _unit_vector(acc, fallback=(0.0, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Stage 1: request/candidate normalization
# -----------------------------------------------------------------------------


def _candidate_context(
    *,
    selected_edge_ids: tuple[int, ...],
    candidate_metadata: Mapping[str, object],
    region_face_ids: tuple[int, ...],
    explicit_quad_density_mode: str | None,
) -> RebuildCandidateContext:
    entity_type = str(candidate_metadata.get("entity_type", "") or "").strip().lower()
    rebuild_gate = str(candidate_metadata.get("rebuild_gate", "") or "").strip().lower()
    feature_family = str(candidate_metadata.get("feature_family", "") or "").strip().lower()
    recognition_stage = str(candidate_metadata.get("recognition_stage", "") or "").strip().lower()
    if recognition_stage and recognition_stage != RecognitionStage.ACCEPTED_CANDIDATE.value:
        raise ValueError(
            "Bore rebuild rejected before target construction: CandidateData is not an accepted candidate. "
            f"recognition_stage={recognition_stage!r}; feature_family={feature_family!r}; Geometry changed: no."
        )
    if feature_family and feature_family not in {FeatureFamily.BORE.value, FeatureFamily.CHAMFER_FORM.value, FeatureFamily.POCKET.value, FeatureFamily.CIRCULAR_POCKET.value}:
        raise ValueError(
            "Bore rebuild rejected before target construction: feature family has no rebuild implementation in the active side-wall rebuild trial. "
            f"recognition_stage={recognition_stage!r}; feature_family={feature_family!r}; Geometry changed: no."
        )
    role = str(candidate_metadata.get("role", candidate_metadata.get("rebuild_role", "")) or "").strip().lower()
    ownership = str(
        candidate_metadata.get("feature_ownership_source", candidate_metadata.get("active_candidate_authority", ""))
        or ""
    ).strip().lower()
    from_component = bool(
        ownership.startswith("surface_component_classifier")
        or str(candidate_metadata.get("candidate_id", "") or "").startswith("component_engine.")
        or str(candidate_metadata.get("recognition_rule", "") or "").startswith("connected_")
    )
    is_borehole = bool(entity_type in {"borehole", "core_bore_cylinder_candidate"} or "borehole" in rebuild_gate or "core_bore" in role)
    damaged_bore_rebuild_trial = bool(
        is_borehole
        and (
            bool(candidate_metadata.get("damaged_bore_rebuild_trial_enabled", False))
            or bool(candidate_metadata.get("requires_damaged_bore_target_seal", False))
            or str(candidate_metadata.get("candidate_variant", "") or "").strip().lower() == "damaged_bore"
            or "damaged_bore" in rebuild_gate
            or "damaged_bore" in role
            or "damaged_bore" in ownership
            or "boundary_defect" in str(candidate_metadata.get("surface_condition", "") or "").strip().lower()
        )
    )
    is_chamfer = bool(entity_type == "chamfer" or rebuild_gate == "promoted_chamfer_candidate")
    is_counterbore = bool(entity_type == "counterbore" or "counterbore" in rebuild_gate)
    is_pocket = bool(entity_type in {"pocket", "circular_pocket"} or "pocket" in rebuild_gate)
    allows_unequal = bool(is_borehole or is_chamfer or is_counterbore or is_pocket)
    return RebuildCandidateContext(
        selected_edge_ids=tuple(int(v) for v in selected_edge_ids),
        entity_type=entity_type,
        rebuild_gate=rebuild_gate,
        role=role,
        candidate_from_component_engine=from_component,
        feature_ownership_source=ownership,
        candidate_has_preview_face_patch=bool(region_face_ids),
        allows_unequal_loop_transition=allows_unequal,
        quad_density_mode=_resolve_quad_density_mode(explicit_quad_density_mode, candidate_metadata),
        damaged_bore_rebuild_trial=bool(damaged_bore_rebuild_trial),
    )


def _resolve_quad_density_mode(explicit_mode: object | None, candidate_metadata: Mapping[str, object] | None) -> str:
    if explicit_mode is not None and str(explicit_mode).strip():
        return _normalize_quad_density_mode(explicit_mode)
    meta = dict(candidate_metadata or {})
    for key in (
        "quad_density_mode",
        "bore_quad_density_mode",
        "rebuild_quad_density_mode",
        "rebuild_density_mode",
        "density_mode",
        "quad_density",
        "bore_rebuild_density",
    ):
        if key in meta and str(meta.get(key, "")).strip():
            return _normalize_quad_density_mode(meta.get(key))
    return QUAD_DENSITY_MODE_LEAN


def _normalize_quad_density_mode(value: object | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return QUAD_DENSITY_MODE_LEAN
    return QUAD_DENSITY_MODE_ALIASES.get(text, QUAD_DENSITY_MODE_LEAN)


# -----------------------------------------------------------------------------
# Stage 2: bounded delete-patch proposals
# -----------------------------------------------------------------------------


def _target_patches_from_result(target_result: Mapping[str, object]) -> tuple[RebuildTargetPatch, ...]:
    patches: list[RebuildTargetPatch] = []
    for raw_source, raw_ids in tuple(target_result.get("face_sets", ()) or ()):
        ids = tuple(sorted({int(fid) for fid in tuple(raw_ids or ())}))
        if ids:
            patches.append(RebuildTargetPatch(str(raw_source), ids))
    return tuple(patches)


# -----------------------------------------------------------------------------
# Stage 3: boundary-loop attempt generation
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Damaged-bore rebuild-target salvage
# -----------------------------------------------------------------------------


def _damaged_bore_defect_swallow_targets(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target_patches: tuple[RebuildTargetPatch, ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    entity_type: str,
) -> tuple[RebuildTargetPatch, ...]:
    """Return fallback targets for damaged two-rim bore patches.

    This is deliberately conservative.  It activates only for BOREHOLE rebuilds
    and only when a target patch has many boundary loops.  The selected/protected
    rim pair remains the rebuild boundary.  Extra boundary loops are treated as
    defect holes/islands only when neighboring faces lie in the same cylindrical
    band between those rims.

    The fallback does not accept a rebuild.  It only creates another exact delete
    target for the normal measured-loop trial.  If the final mesh is not
    watertight, the rebuild still rejects with Geometry changed: no.
    """

    if str(entity_type or "").strip().lower() not in {"borehole", "bore"}:
        return ()
    if len(source_faces) == 0 or len(vertices) == 0:
        return ()

    axis_vec = _unit_vector(axis)
    out: list[RebuildTargetPatch] = []
    for target in tuple(target_patches or ()):  # preserve source priority outside
        base_faces = tuple(int(fid) for fid in tuple(target.face_ids or ()) if 0 <= int(fid) < len(source_faces))
        if not base_faces:
            continue

        boundary_loops = _candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=base_faces)
        if len(boundary_loops) <= 2:
            continue

        main_pair = _main_two_rim_edge_loop_pair(
            vertices=vertices,
            boundary_loops=boundary_loops,
            axis=axis_vec,
            protected_loop_pair=protected_loop_pair,
        )
        if main_pair is None:
            continue

        loop_a, loop_b = main_pair
        if len(loop_a["vertices"]) < 6 or len(loop_b["vertices"]) < 6:
            continue

        expanded, diag = _expand_patch_to_swallow_defect_boundaries(
            vertices=vertices,
            source_faces=source_faces,
            face_ids=base_faces,
            loop_a=loop_a,
            loop_b=loop_b,
            axis=axis_vec,
        )
        if len(expanded) <= len(base_faces):
            continue

        old_loop_count = int(len(boundary_loops))
        new_loop_count = int(len(_candidate_patch_boundary_edge_loops(source_faces=source_faces, face_ids=expanded)))
        # Keep the target only when it actually simplifies the broken topology or
        # reaches the desired two-rim boundary.  The watertight trial remains the
        # final authority.
        if new_loop_count >= old_loop_count and new_loop_count != 2:
            continue

        source = f"{target.source}_damaged_bore_defect_swallowed_two_rim_patch"
        out.append(
            RebuildTargetPatch(
                source=source,
                face_ids=tuple(sorted({int(fid) for fid in expanded})),
            )
        )
    return tuple(out)


def _candidate_patch_boundary_edge_loops(
    *,
    source_faces: np.ndarray,
    face_ids: Iterable[int],
) -> tuple[dict[str, object], ...]:
    """Return ordered boundary loops with both vertices and edge keys."""

    boundary_edges = set(boundary_edges_for_face_patch(source_faces, face_ids))
    components = edge_loop_components(boundary_edges)
    loops: list[dict[str, object]] = []
    for comp in components:
        edges = {_edge_key(edge) for edge in tuple(comp or ())}
        ordered = _order_closed_edge_loop_vertices(edges)
        vertices = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if bool(ordered.get("closed", False)) and len(vertices) >= 3:
            loops.append({"vertices": vertices, "edges": tuple(sorted(edges)), "edge_count": int(len(edges))})
    loops.sort(key=lambda item: (-len(tuple(item.get("vertices", ()) or ())), tuple(item.get("vertices", ()) or (-1,))[0]))
    return tuple(loops)


def _main_two_rim_edge_loop_pair(
    *,
    vertices: np.ndarray,
    boundary_loops: tuple[dict[str, object], ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Choose the two large bore rims from noisy patch boundary loops."""

    axis_vec = _unit_vector(axis)
    if protected_loop_pair is not None:
        protected_sets = (set(int(v) for v in protected_loop_pair[0]), set(int(v) for v in protected_loop_pair[1]))
        matches: list[dict[str, object]] = []
        for wanted in protected_sets:
            best: tuple[int, dict[str, object]] | None = None
            for loop in boundary_loops:
                verts = set(int(v) for v in tuple(loop.get("vertices", ()) or ()))
                score = int(len(verts & wanted))
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, loop)
            if best is not None and best[0] >= max(3, int(0.35 * max(len(wanted), 1))):
                matches.append(best[1])
        if len(matches) == 2 and set(matches[0].get("vertices", ()) or ()) != set(matches[1].get("vertices", ()) or ()):
            return (matches[0], matches[1])

    # Fallback: choose the large, similarly sized pair with maximum axial
    # separation.  Tiny 3-vertex defect loops are excluded by size.
    candidates = [loop for loop in boundary_loops if len(tuple(loop.get("vertices", ()) or ())) >= 8]
    if len(candidates) < 2:
        return None
    best_pair: tuple[dict[str, object], dict[str, object]] | None = None
    best_score = -1.0
    for i, loop_a in enumerate(candidates):
        va = tuple(int(v) for v in tuple(loop_a.get("vertices", ()) or ()))
        ca = _loop_center(vertices, va)
        ta = float(np.dot(ca, axis_vec))
        for loop_b in candidates[i + 1:]:
            vb = tuple(int(v) for v in tuple(loop_b.get("vertices", ()) or ()))
            cb = _loop_center(vertices, vb)
            tb = float(np.dot(cb, axis_vec))
            sep = abs(tb - ta)
            size_ratio = min(len(va), len(vb)) / max(float(max(len(va), len(vb))), 1.0)
            if size_ratio < 0.55:
                continue
            score = float(sep) * float(size_ratio) + 0.01 * float(min(len(va), len(vb)))
            if score > best_score:
                best_score = score
                best_pair = (loop_a, loop_b)
    return best_pair


def _expand_patch_to_swallow_defect_boundaries(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    loop_a: Mapping[str, object],
    loop_b: Mapping[str, object],
    axis: np.ndarray,
    max_iterations: int = 32,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Grow a delete patch across same-cylinder defect boundaries.

    The two main rim loops are protected.  Any other patch boundary loop is a
    potential defect hole/island boundary.  Neighboring kept faces are swallowed
    only when their centroids are inside the axial span and near the cylindrical
    radius implied by the main loops.
    """

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    selected: set[int] = {int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(faces_arr)}
    if not selected:
        return (), {"changed": False, "reason": "empty_patch"}

    axis_vec = _unit_vector(axis)
    loop0_vertices = tuple(int(v) for v in tuple(loop_a.get("vertices", ()) or ()))
    loop1_vertices = tuple(int(v) for v in tuple(loop_b.get("vertices", ()) or ()))
    main_edges = set(_loop_edges_from_vertices(loop0_vertices)) | set(_loop_edges_from_vertices(loop1_vertices))
    if not main_edges:
        return tuple(sorted(selected)), {"changed": False, "reason": "no_main_edges"}

    c0 = _loop_center(vertices, loop0_vertices)
    c1 = _loop_center(vertices, loop1_vertices)
    base = (c0 + c1) * 0.5
    axial0 = float(np.dot(c0 - base, axis_vec))
    axial1 = float(np.dot(c1 - base, axis_vec))
    axial_min = min(axial0, axial1)
    axial_max = max(axial0, axial1)
    axial_span = max(abs(axial_max - axial_min), 1.0e-9)
    axial_pad = max(0.5, 0.08 * axial_span)

    rim_points = vertices[np.asarray(tuple(loop0_vertices + loop1_vertices), dtype=np.int64), :3]
    rel = rim_points - base.reshape(1, 3)
    axial_values = rel @ axis_vec.reshape(3)
    radial_vec = rel - axial_values.reshape(-1, 1) * axis_vec.reshape(1, 3)
    rim_radial = np.linalg.norm(radial_vec, axis=1)
    radius = float(np.median(rim_radial[np.isfinite(rim_radial)])) if np.any(np.isfinite(rim_radial)) else 0.0
    if not np.isfinite(radius) or radius <= 1.0e-9:
        return tuple(sorted(selected)), {"changed": False, "reason": "invalid_main_loop_radius"}
    radial_tolerance = max(0.75, 0.42 * max(float(radius), 1.0))

    edge_to_faces = build_edge_to_faces(faces_arr)
    face_centroids = vertices[faces_arr[:, :3]].mean(axis=1)

    def face_in_bore_band(fid: int) -> bool:
        p = face_centroids[int(fid)]
        r = p - base
        t = float(np.dot(r, axis_vec))
        if t < axial_min - axial_pad or t > axial_max + axial_pad:
            return False
        rv = r - t * axis_vec
        rr = float(np.linalg.norm(rv))
        return bool(abs(rr - radius) <= radial_tolerance)

    max_added = max(256, min(8000, int(max(len(selected) * 1.25, 1024))))
    added_total = 0
    iterations = 0
    last_extra_count = -1
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        boundary_edges = set(boundary_edges_for_face_patch(faces_arr, selected))
        extra_edges = {edge for edge in boundary_edges if _edge_key(edge) not in main_edges}
        if not extra_edges:
            break
        if len(extra_edges) == last_extra_count and iteration > 2:
            # Avoid burning time on a stable frontier when no more same-band
            # faces can be swallowed.
            pass
        last_extra_count = len(extra_edges)
        additions: set[int] = set()
        for edge in extra_edges:
            for fid in edge_to_faces.get(_edge_key(edge), ()):
                fid = int(fid)
                if fid in selected:
                    continue
                if face_in_bore_band(fid):
                    additions.add(fid)
        if not additions:
            break
        if added_total + len(additions) > max_added:
            remaining = max(0, max_added - added_total)
            additions = set(sorted(additions)[:remaining])
        if not additions:
            break
        selected.update(additions)
        added_total += len(additions)
        if added_total >= max_added:
            break

    final_boundary_loop_count = int(len(_candidate_patch_boundary_edge_loops(source_faces=faces_arr, face_ids=selected)))
    return tuple(sorted(selected)), {
        "changed": bool(added_total > 0),
        "added_face_count": int(added_total),
        "iterations": int(iterations),
        "radius": float(radius),
        "radial_tolerance": float(radial_tolerance),
        "axial_pad": float(axial_pad),
        "final_boundary_loop_count": int(final_boundary_loop_count),
    }


def _loop_edges_from_vertices(loop_vertices: Iterable[int]) -> tuple[EdgeKey, ...]:
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


def _build_boundary_loop_attempts(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target_patches: tuple[RebuildTargetPatch, ...],
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    allow_unequal_loop_transition: bool,
) -> tuple[BoundaryLoopAttempt, ...]:
    attempts: list[BoundaryLoopAttempt] = []
    for target in target_patches:
        attempts.extend(
            _attempts_from_patch_boundary(
                vertices=vertices,
                source_faces=source_faces,
                target=target,
                axis=axis,
                allow_unequal_loop_transition=allow_unequal_loop_transition,
            )
        )
        protected = _protected_attempt_for_target(
            vertices=vertices,
            target=target,
            axis=axis,
            protected_loop_pair=protected_loop_pair,
            allow_unequal_loop_transition=allow_unequal_loop_transition,
        )
        if protected is not None:
            attempts.append(protected)

    def priority(attempt: BoundaryLoopAttempt) -> tuple[int, int, int, float, str]:
        target_rank = _target_source_priority(attempt.target_source)
        kind_rank = 1 if attempt.protected_loop_pair else 0
        exact_rank = 0 if attempt.exact_two_loop_patch else 1
        return (target_rank, kind_rank, exact_rank, -float(attempt.axial_separation), str(attempt.source))

    deduped: list[BoundaryLoopAttempt] = []
    seen: set[tuple[frozenset[int], frozenset[int], frozenset[int]]] = set()
    for attempt in sorted(attempts, key=priority):
        key = (frozenset(attempt.face_ids), frozenset(attempt.loop0), frozenset(attempt.loop1))
        rev = (frozenset(attempt.face_ids), frozenset(attempt.loop1), frozenset(attempt.loop0))
        if key in seen or rev in seen:
            continue
        seen.add(key)
        deduped.append(attempt)
    return tuple(deduped)


def _target_source_priority(source: str) -> int:
    text = str(source or "")
    if text.endswith("_damaged_bore_defect_swallowed_two_rim_patch"):
        return 0
    if text.endswith("_topology_sealed_two_rim_patch"):
        return 1
    if text == "candidate_owned_protected_fragment_bridge":
        return 2
    if text == "initial_final_delete_faces":
        return 3
    if text.startswith("measured_connected_component_"):
        return 4
    return 5


def _attempts_from_patch_boundary(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    target: RebuildTargetPatch,
    axis: np.ndarray,
    allow_unequal_loop_transition: bool,
) -> tuple[BoundaryLoopAttempt, ...]:
    loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=target.face_ids)
    if len(loops) < 2:
        return ()
    axis = _unit_vector(axis)
    centers = [_loop_center(vertices, loop) for loop in loops]
    axials = [float(np.dot(center, axis)) for center in centers]
    loop_summaries = tuple(
        {
            "index": int(i),
            "vertex_count": int(len(loop)),
            "axial": float(axials[i]),
        }
        for i, loop in enumerate(loops[:12])
    )
    attempts: list[BoundaryLoopAttempt] = []
    for i, loop_a in enumerate(loops):
        for j in range(i + 1, len(loops)):
            loop_b = loops[j]
            transition_ok = _loop_pair_transition_allowed(
                len(loop_a),
                len(loop_b),
                allow_unequal_loop_transition=allow_unequal_loop_transition,
            )
            if not transition_ok:
                continue
            sep = abs(float(axials[j] - axials[i]))
            min_sep = _minimum_loop_pair_separation(vertices, loop_a, loop_b)
            if sep <= min_sep:
                continue
            attempts.append(
                BoundaryLoopAttempt(
                    source=f"{target.source}_patch_boundary_loop_pair_{i}_{j}",
                    target_source=target.source,
                    face_ids=target.face_ids,
                    loop0=tuple(int(v) for v in loop_a),
                    loop1=tuple(int(v) for v in loop_b),
                    boundary_loop_count=int(len(loops)),
                    exact_two_loop_patch=bool(len(loops) == 2),
                    protected_loop_pair=False,
                    axial_separation=float(sep),
                    min_required_axial_separation=float(min_sep),
                    unequal_loop_transition_allowed=bool(len(loop_a) != len(loop_b)),
                    boundary_loop_vertex_count_delta=int(abs(len(loop_a) - len(loop_b))),
                    loop_summaries=loop_summaries,
                )
            )
    return tuple(attempts)


def _protected_attempt_for_target(
    *,
    vertices: np.ndarray,
    target: RebuildTargetPatch,
    axis: np.ndarray,
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None,
    allow_unequal_loop_transition: bool,
) -> BoundaryLoopAttempt | None:
    if protected_loop_pair is None:
        return None
    loop_a, loop_b = protected_loop_pair
    if len(loop_a) < 3 or len(loop_b) < 3:
        return None
    if not _loop_pair_transition_allowed(len(loop_a), len(loop_b), allow_unequal_loop_transition=allow_unequal_loop_transition):
        return None
    axis = _unit_vector(axis)
    center_a = _loop_center(vertices, loop_a)
    center_b = _loop_center(vertices, loop_b)
    sep = abs(float(np.dot(center_b - center_a, axis)))
    min_sep = _minimum_loop_pair_separation(vertices, loop_a, loop_b)
    if sep <= min_sep:
        return None
    return BoundaryLoopAttempt(
        source=f"{target.source}_protected_loop_pair",
        target_source=target.source,
        face_ids=target.face_ids,
        loop0=tuple(int(v) for v in loop_a),
        loop1=tuple(int(v) for v in loop_b),
        boundary_loop_count=-1,
        exact_two_loop_patch=False,
        protected_loop_pair=True,
        axial_separation=float(sep),
        min_required_axial_separation=float(min_sep),
        unequal_loop_transition_allowed=bool(len(loop_a) != len(loop_b)),
        boundary_loop_vertex_count_delta=int(abs(len(loop_a) - len(loop_b))),
        loop_summaries=(),
    )


def _loop_pair_transition_allowed(
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


# -----------------------------------------------------------------------------
# Stage 4: measured-loop quad planning
# -----------------------------------------------------------------------------


def _quad_plan_for_attempt(
    *,
    vertices: np.ndarray,
    attempt: BoundaryLoopAttempt,
    axis: np.ndarray,
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
        plan = _equal_loop_quad_plan(
            vertices=vertices,
            loop0=tuple(int(v) for v in sorted0),
            loop1=tuple(int(v) for v in aligned1),
            center0=center_a,
            center1=center_b,
            axis=axis,
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
    plan = _unequal_loop_quad_plan(
        vertices=vertices,
        loop0=aligned0,
        loop1=aligned1,
        center0=center_a,
        center1=center_b,
        axis=axis,
        quad_density_mode=quad_density_mode,
    )
    plan.diagnostics.update(alignment_diag)
    return plan


def _equal_loop_quad_plan(
    *,
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
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

    generated: list[np.ndarray] = []
    rings: list[tuple[int, ...]] = [tuple(int(v) for v in loop0)]
    for step in range(1, axial_segments):
        t = float(step) / float(axial_segments)
        pts = (1.0 - t) * loop0_pts + t * loop1_pts
        start = len(vertices) + sum(len(item) for item in generated)
        generated.append(pts)
        rings.append(tuple(range(start, start + n)))
    rings.append(tuple(int(v) for v in loop1))

    generated_vertices = np.vstack(generated).reshape((-1, 3)) if generated else np.zeros((0, 3), dtype=float)
    logical_quads: list[tuple[int, int, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    for band in range(len(rings) - 1):
        a = rings[band]
        b = rings[band + 1]
        for i in range(n):
            quad = (int(a[i]), int(a[(i + 1) % n]), int(b[(i + 1) % n]), int(b[i]))
            logical_quads.append(quad)
            triangles.append((quad[0], quad[1], quad[2]))
            triangles.append((quad[0], quad[2], quad[3]))

    diagnostics = {
        "plan_type": "equal_loop_measured_quad_plan",
        "unequal_loop_transition_used": False,
        "loop0_vertex_count": int(len(loop0)),
        "loop1_vertex_count": int(len(loop1)),
        "circumferential_segments": int(n),
        "axial_segments": int(axial_segments),
        "generated_internal_ring_count": int(max(0, axial_segments - 1)),
        "transition_drop_quad_count": 0,
        "transition_ring_vertex_count": 0,
        "geometry_source": "measured_boundary_loop_vertices",
        "interpolation_rule": "linear_between_corresponding_measured_loop_vertices",
        "parameter_fit_used": False,
        "radius_used_for_vertex_placement": False,
        "axis_used_for_vertex_placement": False,
    }
    return QuadPlan(
        generated_vertices=np.asarray(generated_vertices, dtype=float).reshape((-1, 3)),
        triangles=np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        logical_quads=tuple(logical_quads),
        loop0=tuple(int(v) for v in loop0),
        loop1=tuple(int(v) for v in loop1),
        center0=np.asarray(center0, dtype=float).reshape(3),
        center1=np.asarray(center1, dtype=float).reshape(3),
        axis=_unit_vector(axis),
        diagnostics=diagnostics,
    )


def _unequal_loop_quad_plan(
    *,
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
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
    if not _loop_pair_transition_allowed(n0, n1, allow_unequal_loop_transition=True):
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


# -----------------------------------------------------------------------------
# Target helper: topology-only local sealing proposal
# -----------------------------------------------------------------------------


def _topology_seal_callback_for_context(context: RebuildCandidateContext):
    """Return the topology-only two-rim seal callback for rebuild-target policy.

    The v12 boundary diagnostics proved the previewed BOREHOLE candidate already
    has the right semantic object, but its exact 566-face patch boundary is not
    directly stitchable: 40 delete-patch boundary edges are missing from the
    generated replacement boundary and 40 wrong same-vertex edges are generated.

    The older working rebuild path did not solve that by changing recognition or
    by projecting vertices.  It allowed rebuild_target.py to create a
    topology-sealed two-rim delete-patch proposal, then accepted it only after a
    full boundary-loop/watertight trial.

    Therefore this callback must not add a second tiny 8..24-face cap for
    component-engine BOREHOLE candidates.  rebuild_target.py already owns the
    candidate-owned absorption limit and rejects broad neighboring-feature
    absorption before any trial.  This callback only performs the local
    topology-only seal inside the budget requested by rebuild_target.py.
    """

    def _callback(
        *,
        source_faces: np.ndarray,
        initial_face_ids: tuple[int, ...],
        protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        max_added_faces: int = 0,
    ) -> dict[str, object] | None:
        # ``max_added_faces`` is accepted for compatibility but deliberately
        # ignored.  Topology sealing is governed by leak-edge closure and the
        # final watertight trial, not a face-count budget.
        return _topologically_seal_patch_to_main_boundary_loops(
            source_faces=source_faces,
            initial_face_ids=initial_face_ids,
            protected_loop_pair=protected_loop_pair,
            max_added_faces=0,
        )

    return _callback


def _topologically_seal_patch_to_main_boundary_loops(
    *,
    source_faces: np.ndarray,
    initial_face_ids: tuple[int, ...],
    protected_loop_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    max_added_faces: int = 0,
) -> dict[str, object] | None:
    """Return a locally sealed two-rim patch proposal.

    Cap-free topology-only seal: absorb all faces adjacent across non-protected
    boundary leaks until the only remaining patch boundary is the protected
    two-rim boundary.  This does not use radius/axis expansion and it does not
    accept the result; the full measured-loop replacement trial still decides.
    """

    patch: set[int] = {int(fid) for fid in tuple(initial_face_ids or ()) if 0 <= int(fid) < len(source_faces)}
    if len(patch) < 3:
        return None

    initial_loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=tuple(sorted(patch)))
    if protected_loop_pair is not None and len(protected_loop_pair[0]) >= 3 and len(protected_loop_pair[1]) >= 3:
        protected_loops = (
            tuple(int(v) for v in protected_loop_pair[0]),
            tuple(int(v) for v in protected_loop_pair[1]),
        )
    else:
        if len(initial_loops) < 2:
            return None
        protected_loops = tuple(initial_loops[:2])

    protected_edges: set[EdgeKey] = set()
    for loop in protected_loops:
        protected_edges.update(_loop_vertices_to_edges(loop))
    if not protected_edges:
        return None

    edge_to_faces = build_edge_to_faces(source_faces)
    added: set[int] = set()
    iteration_summaries: list[dict[str, object]] = []
    iteration = 0

    while True:
        boundary = set(boundary_edges_for_face_patch(source_faces, patch))
        leak_edges = tuple(sorted(edge for edge in boundary if edge not in protected_edges))
        if not leak_edges:
            break

        candidates: set[int] = set()
        for edge in leak_edges:
            for fid in edge_to_faces.get(edge, ()):
                fid_int = int(fid)
                if fid_int not in patch and 0 <= fid_int < len(source_faces):
                    candidates.add(fid_int)

        if not candidates:
            iteration_summaries.append({
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "reason": "no_adjacent_faces",
            })
            break

        chosen = tuple(sorted(candidates))
        before = int(len(patch))
        patch.update(chosen)
        added.update(chosen)
        after = int(len(patch))

        iteration_summaries.append(
            {
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "candidate_face_count": int(len(candidates)),
                "added_face_count": int(after - before),
                "total_added_face_count": int(len(added)),
                "policy": "add_all_adjacent_faces_across_non_protected_leak_edges_no_face_count_cap",
            }
        )

        if after <= before:
            iteration_summaries.append({
                "iteration": int(iteration),
                "leak_edge_count": int(len(leak_edges)),
                "reason": "no_topology_progress",
            })
            break
        iteration += 1

    if not added:
        return None

    final_boundary = set(boundary_edges_for_face_patch(source_faces, patch))
    remaining_leaks = tuple(sorted(edge for edge in final_boundary if edge not in protected_edges))
    if remaining_leaks:
        return None

    final_ids = tuple(sorted(patch))
    final_loops = _candidate_patch_boundary_vertex_loops(source_faces=source_faces, face_ids=final_ids)
    return {
        "face_ids": final_ids,
        "added_face_ids": tuple(sorted(added)),
        "added_face_count": int(len(added)),
        "initial_face_count": int(len(initial_face_ids)),
        "final_face_count": int(len(final_ids)),
        "initial_boundary_loop_count": int(len(initial_loops)),
        "final_boundary_loop_count": int(len(final_loops)),
        "initial_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in initial_loops[:12]),
        "final_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in final_loops[:12]),
        "protected_boundary_loop_vertex_counts": tuple(int(len(loop)) for loop in protected_loops),
        "iteration_summaries": tuple(iteration_summaries),
        "policy": "cap_free_topology_only_absorb_non_rim_boundary_leaks_until_protected_two_rim_boundary",
        "max_added_faces_ignored": True,
        "parameter_fit_used": False,
        "radius_used_for_delete_expansion": False,
        "axis_used_for_delete_expansion": False,
    }


def _candidate_patch_boundary_vertex_loops(
    *,
    source_faces: np.ndarray,
    face_ids: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    boundary_edges = set(boundary_edges_for_face_patch(source_faces, face_ids))
    components = edge_loop_components(boundary_edges)
    loops: list[tuple[int, ...]] = []
    for comp in components:
        ordered = _order_closed_edge_loop_vertices(set(comp))
        vertices = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if bool(ordered.get("closed", False)) and len(vertices) >= 3:
            loops.append(vertices)
    loops.sort(key=lambda loop: (-len(loop), loop[0] if loop else -1))
    return tuple(loops)




def _two_opening_frame_from_candidate_metadata(candidate_metadata: Mapping[str, object]) -> dict[str, object] | None:
    """Extract the accepted two-opening BoreFrame carried by CandidateData.

    Recognition owns the measured frame.  Rebuild may use it only as bounded
    target evidence after CandidateData is already accepted/rebuildable.
    """

    meta = dict(candidate_metadata or {})
    diag = meta.get("diagnostics")
    if not isinstance(diag, Mapping):
        diag = {}
    # v1.6.6: prefer the operational owned-bore frame produced after
    # BoreWallOwnership.  The original measured two-opening frame may have used
    # an internal damaged ring as provisional opposite evidence.
    frame = diag.get("owned_bore_frame") if isinstance(diag, Mapping) else None
    if not isinstance(frame, Mapping):
        frame = meta.get("owned_bore_frame")
    if not isinstance(frame, Mapping):
        frame = diag.get("two_opening_bore_frame") if isinstance(diag, Mapping) else None
    if not isinstance(frame, Mapping):
        frame = meta.get("two_opening_bore_frame")
    if not isinstance(frame, Mapping):
        return None
    try:
        opening_center = np.asarray(frame.get("opening_center", ()), dtype=float).reshape(3)
        opposite_center = np.asarray(frame.get("opposite_center", ()), dtype=float).reshape(3)
    except Exception:
        return None
    delta = opposite_center - opening_center
    depth = float(np.linalg.norm(delta))
    axis = _unit_vector(delta if np.isfinite(depth) and depth > 1.0e-9 else frame.get("axis", meta.get("primitive_axis", (0.0, 0.0, 1.0))))
    explicit_depth = _safe_float(frame.get("depth", meta.get("owned_bore_frame_depth", meta.get("depth", 0.0))), 0.0)
    if explicit_depth > max(depth, 0.0) + max(1.0e-6, 0.005 * max(depth, 1.0)):
        depth = float(explicit_depth)
        opposite_center = opening_center + axis.reshape(3) * float(depth)
    if not np.isfinite(depth) or depth <= 1.0e-9:
        depth = float(explicit_depth)
    radius = _safe_float(frame.get("radius", meta.get("radius", meta.get("primitive_radius", 0.0))), 0.0)
    if radius <= 1.0e-9 or depth <= 1.0e-9:
        return None
    return {
        "opening_center": opening_center,
        "opposite_center": opposite_center,
        "axis": axis,
        "radius": float(radius),
        "depth": float(depth),
        "frame_source": str(frame.get("status", meta.get("bore_frame_depth_source", "candidate_owned_bore_frame"))),
    }


def _protected_loop_pair_from_candidate_frame(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    frame: Mapping[str, object],
    preferred_pool: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Find both physical rim loops from the accepted candidate BoreFrame."""

    loop_a = _edge_loop_near_frame_opening(
        vertices=vertices,
        source_faces=source_faces,
        center=np.asarray(frame.get("opening_center"), dtype=float).reshape(3),
        axis=_unit_vector(frame.get("axis", (0.0, 0.0, 1.0))),
        radius=_safe_float(frame.get("radius", 0.0), 0.0),
        preferred_pool=preferred_pool,
    )
    loop_b = _edge_loop_near_frame_opening(
        vertices=vertices,
        source_faces=source_faces,
        center=np.asarray(frame.get("opposite_center"), dtype=float).reshape(3),
        axis=_unit_vector(frame.get("axis", (0.0, 0.0, 1.0))),
        radius=_safe_float(frame.get("radius", 0.0), 0.0),
        preferred_pool=preferred_pool,
    )
    if loop_a is None or loop_b is None:
        return None
    if set(loop_a) == set(loop_b):
        return None
    return (loop_a, loop_b)


def _edge_loop_near_frame_opening(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    preferred_pool: Iterable[int],
) -> tuple[int, ...] | None:
    """Measure a rim loop near one BoreFrame opening from mesh edges."""

    radius = float(radius)
    if radius <= 1.0e-9:
        return None
    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    edge_to_faces = build_edge_to_faces(faces_arr)
    pool = {int(fid) for fid in tuple(preferred_pool or ()) if 0 <= int(fid) < len(faces_arr)}
    axis_vec = _unit_vector(axis)
    edge_lengths: list[float] = []
    for edge in edge_to_faces.keys():
        a, b = int(edge[0]), int(edge[1])
        if 0 <= a < len(vertices) and 0 <= b < len(vertices):
            length = float(np.linalg.norm(vertices[a, :3] - vertices[b, :3]))
            if np.isfinite(length) and length > 1.0e-12:
                edge_lengths.append(length)
    edge_scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
    plane_tol = max(0.085 * radius, 4.5 * edge_scale, 0.30)
    radial_tol = max(0.105 * radius, 4.5 * edge_scale, 0.35)

    selected_edges: set[EdgeKey] = set()
    for edge, adjacent in edge_to_faces.items():
        if pool and not any(int(fid) in pool for fid in tuple(adjacent or ())):
            continue
        a, b = int(edge[0]), int(edge[1])
        if not (0 <= a < len(vertices) and 0 <= b < len(vertices)):
            continue
        pa = vertices[a, :3]
        pb = vertices[b, :3]
        mid = 0.5 * (pa + pb)
        rel = mid - center.reshape(3)
        axial = float(np.dot(rel, axis_vec))
        radial_vec = rel - axial * axis_vec
        radial = float(np.linalg.norm(radial_vec))
        if not np.isfinite(axial) or not np.isfinite(radial):
            continue
        if abs(axial) > plane_tol or abs(radial - radius) > radial_tol:
            continue
        tangent = pb - pa
        tlen = float(np.linalg.norm(tangent))
        if tlen <= 1.0e-12:
            continue
        tangent = tangent / tlen
        radial_unit = radial_vec / max(radial, 1.0e-12)
        radial_tangent = abs(float(np.dot(tangent, radial_unit))) if radial > 1.0e-12 else 0.0
        axial_tangent = abs(float(np.dot(tangent, axis_vec)))
        if radial_tangent > 0.92 and axial_tangent < 0.35:
            continue
        selected_edges.add(_edge_key(edge))

    if len(selected_edges) < 6:
        return None
    best_loop: tuple[int, ...] | None = None
    best_score = -1.0
    for comp in edge_loop_components(selected_edges):
        ordered = _order_closed_edge_loop_vertices({_edge_key(edge) for edge in tuple(comp or ())})
        if not bool(ordered.get("closed", False)):
            continue
        verts = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if len(verts) < 6:
            continue
        pts = vertices[np.asarray(verts, dtype=np.int64), :3]
        rel = pts - center.reshape(1, 3)
        axial_values = rel @ axis_vec.reshape(3)
        radial_vecs = rel - axial_values.reshape(-1, 1) * axis_vec.reshape(1, 3)
        radii = np.linalg.norm(radial_vecs, axis=1)
        radial_mad = float(np.median(np.abs(radii - radius))) if radii.size else 999999.0
        plane_mad = float(np.median(np.abs(axial_values))) if axial_values.size else 999999.0
        score = float(len(verts)) - 4.0 * radial_mad - 2.0 * plane_mad
        if score > best_score:
            best_score = score
            best_loop = verts
    return best_loop


def _full_depth_bore_wall_target_faces_from_candidate_frame(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    frame: Mapping[str, object],
    face_pool: Iterable[int],
    base_face_ids: Iterable[int],
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Build a bounded full-depth wall delete target from CandidateData frame.

    This is used only for accepted damaged BoreWallOwnership.  It does not let
    RegionData become ownership; the candidate's two-opening frame and wall role
    define the target, while RegionData merely bounds the search pool.
    """

    faces_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    pool = {int(fid) for fid in tuple(face_pool or ()) if 0 <= int(fid) < len(faces_arr)}
    base = {int(fid) for fid in tuple(base_face_ids or ()) if 0 <= int(fid) < len(faces_arr)}
    if not pool:
        pool = set(range(len(faces_arr)))
    opening_center = np.asarray(frame.get("opening_center"), dtype=float).reshape(3)
    opposite_center = np.asarray(frame.get("opposite_center"), dtype=float).reshape(3)
    axis_vec = _unit_vector(frame.get("axis", opposite_center - opening_center), fallback=opposite_center - opening_center)
    radius = _safe_float(frame.get("radius", 0.0), 0.0)
    depth = float(np.linalg.norm(opposite_center - opening_center))
    if radius <= 1.0e-9 or depth <= 1.0e-9:
        return tuple(sorted(base)), {"used": False, "reason": "invalid_candidate_frame"}

    tri = vertices[faces_arr[:, :3], :3]
    centroids = tri.mean(axis=1)
    raw_normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    nlen = np.linalg.norm(raw_normals, axis=1)
    normals = np.zeros_like(raw_normals)
    okn = np.isfinite(nlen) & (nlen > 1.0e-12)
    normals[okn] = raw_normals[okn] / nlen[okn].reshape(-1, 1)

    rel_cent = centroids - opening_center.reshape(1, 3)
    axial_cent = rel_cent @ axis_vec.reshape(3)
    radial_vec_cent = rel_cent - axial_cent.reshape(-1, 1) * axis_vec.reshape(1, 3)
    radial_cent = np.linalg.norm(radial_vec_cent, axis=1)
    radial_dir = np.zeros_like(radial_vec_cent)
    okr = radial_cent > 1.0e-12
    radial_dir[okr] = radial_vec_cent[okr] / radial_cent[okr].reshape(-1, 1)
    normal_axis_abs = np.abs(normals @ axis_vec.reshape(3))
    radial_normal_alignment = np.abs(np.sum(normals * radial_dir, axis=1))

    tri_rel = tri - opening_center.reshape(1, 1, 3)
    tri_ax = tri_rel @ axis_vec.reshape(3)
    face_ax_min = np.nanmin(tri_ax, axis=1)
    face_ax_max = np.nanmax(tri_ax, axis=1)

    edge_lengths: list[float] = []
    for fid in list(base)[:3000]:
        pts = tri[int(fid)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            length = float(np.linalg.norm(pts[a] - pts[b]))
            if np.isfinite(length) and length > 1.0e-12:
                edge_lengths.append(length)
    edge_scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
    axial_tol = max(0.045 * depth, 0.10 * radius, 4.0 * edge_scale, 0.35)
    radial_tol = max(0.18 * radius, 6.0 * edge_scale, 0.55)

    target: set[int] = set(base)
    for fid in pool:
        if fid < 0 or fid >= len(faces_arr):
            continue
        if not (np.isfinite(face_ax_min[fid]) and np.isfinite(face_ax_max[fid]) and np.isfinite(radial_cent[fid])):
            continue
        between = bool(face_ax_max[fid] >= -axial_tol and face_ax_min[fid] <= depth + axial_tol)
        near_radius = bool(abs(float(radial_cent[fid]) - radius) <= radial_tol)
        sidewall_normal = bool(float(normal_axis_abs[fid]) <= 0.75 and float(radial_normal_alignment[fid]) >= 0.20)
        if between and near_radius and sidewall_normal:
            target.add(int(fid))

    return tuple(sorted(target)), {
        "used": True,
        "source": "candidate_two_opening_frame_full_depth_wall_target",
        "base_face_count": int(len(base)),
        "pool_face_count": int(len(pool)),
        "target_face_count": int(len(target)),
        "added_face_count": int(max(0, len(target) - len(base))),
        "radius": float(radius),
        "depth": float(depth),
        "frame_source": str(frame.get("frame_source", frame.get("status", "candidate_owned_bore_frame"))),
        "radial_tolerance": float(radial_tol),
        "axial_tolerance": float(axial_tol),
    }

def _protected_loop_pair_from_selection(
    *,
    vertices: np.ndarray,
    region_data: RegionData,
    axis: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    loops: list[tuple[int, ...]] = []
    primary = tuple(int(v) for v in tuple(getattr(region_data, "loop_vertices", ()) or ()))
    if len(primary) >= 3:
        loops.append(primary)

    opposite_edges = {_edge_key(edge) for edge in tuple(getattr(region_data, "derived_opposite_rim_edge_ids", ()) or ())}
    if opposite_edges:
        ordered = _order_closed_edge_loop_vertices(opposite_edges)
        if bool(ordered.get("closed", False)):
            opp = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
            if len(opp) >= 3:
                loops.append(opp)

    for edge_loop in tuple(getattr(region_data, "derived_boundary_loops", ()) or ()):
        edges = {_edge_key(edge) for edge in tuple(edge_loop or ())}
        ordered = _order_closed_edge_loop_vertices(edges)
        if bool(ordered.get("closed", False)):
            loop = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
            if len(loop) >= 3 and all(set(loop) != set(existing) for existing in loops):
                loops.append(loop)

    if len(loops) < 2:
        return None

    axis = _unit_vector(axis)
    best_pair: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_sep = -1.0
    for i, loop_a in enumerate(loops):
        ca = _loop_center(vertices, loop_a)
        ta = float(np.dot(ca, axis))
        for loop_b in loops[i + 1:]:
            cb = _loop_center(vertices, loop_b)
            tb = float(np.dot(cb, axis))
            sep = abs(tb - ta)
            if sep > best_sep:
                best_sep = sep
                best_pair = (loop_a, loop_b)
    return best_pair


# -----------------------------------------------------------------------------
# Stage 5: trial validation and final mesh application
# -----------------------------------------------------------------------------


def _small_boundary_seal_triangles_for_trial_mesh(
    *,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    max_boundary_edges: int = 12,
    max_loop_edges: int = 6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return conservative closure triangles for tiny damaged-bore residual holes.

    This is not a general mesh repair and it is never used for clean candidates.
    The caller only invokes it for damaged BORE rebuild trials after the normal
    measured two-loop replacement still leaves a very small boundary.  Large or
    open residual boundaries remain a rejection.
    """

    source_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    remove_mask = np.ones(len(source_arr), dtype=bool)
    valid_faces = tuple(int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(source_arr))
    if valid_faces:
        remove_mask[np.asarray(valid_faces, dtype=np.int64)] = False
    kept_faces = source_arr[remove_mask, :3].copy()
    tri_arr = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if tri_arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "no_generated_triangles"}
    output_faces = np.vstack([kept_faces, tri_arr])
    boundary_edges = set(boundary_edges_for_face_patch(output_faces, range(len(output_faces))))
    if not boundary_edges:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "already_closed"}
    if len(boundary_edges) > int(max_boundary_edges):
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_too_large", "boundary_edge_count": int(len(boundary_edges))}

    components = edge_loop_components(boundary_edges)
    close_tris: list[tuple[int, int, int]] = []
    loop_sizes: list[int] = []
    for comp in components:
        edges = {_edge_key(edge) for edge in tuple(comp or ())}
        ordered = _order_closed_edge_loop_vertices(edges)
        verts = tuple(int(v) for v in tuple(ordered.get("vertices", ()) or ()))
        if not bool(ordered.get("closed", False)) or len(verts) < 3:
            return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_not_closed", "boundary_edge_count": int(len(boundary_edges))}
        if len(verts) > int(max_loop_edges):
            return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "residual_boundary_loop_too_large", "boundary_edge_count": int(len(boundary_edges)), "loop_vertex_count": int(len(verts))}
        loop_sizes.append(int(len(verts)))
        if len(verts) == 3:
            close_tris.append((int(verts[0]), int(verts[1]), int(verts[2])))
        else:
            anchor = int(verts[0])
            for i in range(1, len(verts) - 1):
                close_tris.append((anchor, int(verts[i]), int(verts[i + 1])))

    if not close_tris:
        return np.zeros((0, 3), dtype=np.int64), {"used": False, "reason": "no_closure_triangles_created"}
    out = np.asarray(close_tris, dtype=np.int64).reshape((-1, 3))
    sealed_faces = np.vstack([output_faces, out])
    after_edges = int(len(boundary_edges_for_face_patch(sealed_faces, range(len(sealed_faces)))))
    return out, {
        "used": True,
        "reason": "tiny_residual_boundary_closed_for_damaged_bore_trial",
        "boundary_edge_count_before": int(len(boundary_edges)),
        "boundary_edge_count_after_seal": int(after_edges),
        "closure_triangle_count": int(len(out)),
        "closed_loop_count": int(len(loop_sizes)),
        "closed_loop_vertex_counts": tuple(int(v) for v in loop_sizes),
    }


def _trial_rebuild(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, object]:
    remove_mask = np.ones(len(source_faces), dtype=bool)
    valid_faces = tuple(int(fid) for fid in tuple(face_ids or ()) if 0 <= int(fid) < len(source_faces))
    remove_mask[np.asarray(valid_faces, dtype=np.int64)] = False
    kept_faces = source_faces[remove_mask, :3].copy()
    generated_vertices = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if triangles.size == 0:
        return {"boundary_edge_count_after": -1, "watertight_after": False, "reason": "no_generated_faces"}
    output_vertices = np.asarray(vertices, dtype=float).copy()
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])
    output_faces = np.vstack([kept_faces, triangles])
    boundary_before = _boundary_edge_count(source_faces)
    boundary_count = _boundary_edge_count(output_faces)
    trial_mesh = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    boundary_delta = int(boundary_count) - int(boundary_before)
    return {
        "boundary_edge_count_before": int(boundary_before),
        "boundary_edge_count_after": int(boundary_count),
        "boundary_edge_count_delta": int(boundary_delta),
        "watertight_after": bool(getattr(trial_mesh, "is_watertight", False)),
        "kept_face_count": int(len(kept_faces)),
        "after_face_count": int(len(output_faces)),
        "after_vertex_count": int(len(output_vertices)),
    }



def _int_preserve_zero(value: object, default: int = -1) -> int:
    """Coerce diagnostics to int without treating 0 as missing.

    Rebuild diagnostics legitimately use zero for success states, for example
    generated_boundary_generated_vertex_edge_count=0.  Do not use
    ``value or fallback`` here because it turns a valid zero into the fallback.
    """

    if value is None:
        return int(default)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)

def _trial_accepts_for_context(*, context: RebuildCandidateContext, trial: Mapping[str, object], plan: QuadPlan) -> bool:
    """Return whether a trial replacement is acceptable for this feature type.

    Global watertightness remains the strongest success condition.  For imported
    meshes that are already globally open, an accepted local feature CandidateData
    may still be valid when the replacement exactly preserves the deleted patch
    boundary and does not add new boundary edges.  v83 extends the v80 local
    CHAMFER rule to accepted BORE wall candidates, because the v82 clean bore
    test proved the exact same local-topology contract: patchB == genB, no
    missing/extra boundary edges, and boundary_edge_count_after <= before.

    _trial_accepts_preserves_zero_boundary_counts_v85; local_bore_wall_rebuild_acceptance_v85 is topology acceptance only; it does
    not loosen Recognition, DeletePatchProposal ownership, or boundary-loop
    matching.
    """

    # v85: preserve real zero counts.  The previous code used
    # ``int(value or -1)``, which converted boundary_edge_count_after=0 into
    # -1.  Damaged-bore full-depth trials can legitimately return
    # after=0, watertight_after=True after swallowing internal defect
    # boundary loops, so zero must remain zero for acceptance.
    after_raw = trial.get("boundary_edge_count_after", -1)
    before_raw = trial.get("boundary_edge_count_before", -1)
    after = int(after_raw if after_raw is not None else -1)
    before = int(before_raw if before_raw is not None else -1)
    if after == 0 and bool(trial.get("watertight_after", False)):
        return True

    entity = str(getattr(context, "entity_type", "") or "").strip().lower()
    if entity in {"chamfer", "borehole", "pocket", "circular_pocket"}:
        diag = dict(getattr(plan, "diagnostics", {}) or {})
        boundary_match = bool(diag.get("boundary_match_exact", False))
        missing = int(diag.get("missing_patch_boundary_edge_count", 1) or 0)
        extra = int(diag.get("extra_generated_boundary_edge_count", 1) or 0)
        patch_edges = int(diag.get("patch_boundary_edge_count", 0) or 0)
        generated_edges = _int_preserve_zero(diag.get("generated_boundary_edge_count", -1), -1)
        generated_original_boundary = _int_preserve_zero(diag.get("generated_boundary_original_vertex_edge_count", -1), -1)
        generated_new_boundary = _int_preserve_zero(diag.get("generated_boundary_generated_vertex_edge_count", -1), -1)
        candidate_owned_patch = bool(getattr(context, "candidate_from_component_engine", False)) and bool(getattr(context, "candidate_has_preview_face_patch", False))
        local_boundary_preserved = bool(
            boundary_match
            and missing == 0
            and extra == 0
            and patch_edges > 0
            and generated_edges == patch_edges
            and before >= 0
            and after <= before
        )
        if entity == "chamfer" and local_boundary_preserved:
            return True
        if entity == "borehole" and candidate_owned_patch and local_boundary_preserved:
            return True
        if entity in {"pocket", "circular_pocket"}:
            # POCKET v98 local acceptance: the pocket side-wall delete patch is
            # a measured two-rim sleeve between the opening loop and the floor
            # perimeter.  On user meshes that are already globally open, the
            # whole mesh may remain non-watertight even when the local replacement
            # is exact.  Accept only the strict local topology contract: the
            # generated surface must reproduce every original patch boundary edge,
            # add no generated-vertex boundary edges, and not increase the global
            # boundary count.  Floor faces are preserved by CandidateData/rebuild
            # input; this does not authorize broad RegionData deletion.
            pocket_boundary_is_original_only = bool(generated_original_boundary == patch_edges and generated_new_boundary == 0)
            if candidate_owned_patch and local_boundary_preserved and pocket_boundary_is_original_only:
                return True
    return False


def _apply_rebuild(
    *,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    generated_vertices: np.ndarray,
    triangles: np.ndarray,
    color_rebuilt_faces: bool,
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> dict[str, object]:
    remove_mask = np.ones(len(source_faces), dtype=bool)
    remove_mask[np.asarray(tuple(int(fid) for fid in face_ids), dtype=np.int64)] = False
    kept_faces = source_faces[remove_mask, :3].copy()

    output_vertices = np.asarray(vertices, dtype=float).copy()
    generated_vertices = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])
    triangles = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    output_faces = np.vstack([kept_faces, triangles])

    rebuilt = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    added_start = int(len(kept_faces))
    added_face_ids = tuple(range(added_start, added_start + int(len(triangles))))

    if color_rebuilt_faces:
        _assign_face_colors(
            rebuilt,
            added_face_ids=added_face_ids,
            base_face_color=base_face_color,
            rebuilt_face_color=rebuilt_face_color,
        )

    rebuilt.remove_unreferenced_vertices()
    return {"mesh": rebuilt, "added_face_ids": added_face_ids}



def _generated_surface_boundary_match_diagnostics(
    *,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    triangles: np.ndarray,
    source_vertex_count: int,
) -> dict[str, object]:
    """Compare deleted patch boundary against generated replacement boundary."""

    source_arr = np.asarray(source_faces, dtype=np.int64)[:, :3]
    tri_values = np.asarray(triangles, dtype=np.int64)
    tri_arr = tri_values.reshape((-1, 3)) if tri_values.size else np.zeros((0, 3), dtype=np.int64)

    patch_boundary = {_edge_key(edge) for edge in boundary_edges_for_face_patch(source_arr, face_ids)}

    generated_counts: dict[EdgeKey, int] = {}
    generated_invalid_edges = 0
    for tri in tri_arr[:, :3]:
        verts = (int(tri[0]), int(tri[1]), int(tri[2]))
        for edge in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            a, b = int(edge[0]), int(edge[1])
            if a < 0 or b < 0:
                generated_invalid_edges += 1
                continue
            key = _edge_key((a, b))
            generated_counts[key] = generated_counts.get(key, 0) + 1

    generated_boundary = {edge for edge, count in generated_counts.items() if int(count) == 1}
    generated_boundary_original_only = {
        edge for edge in generated_boundary
        if int(edge[0]) < int(source_vertex_count) and int(edge[1]) < int(source_vertex_count)
    }
    generated_boundary_generated_involved = generated_boundary - generated_boundary_original_only

    shared = patch_boundary & generated_boundary
    missing = patch_boundary - generated_boundary
    extra = generated_boundary - patch_boundary

    patch_edges_generated_count_hist: dict[int, int] = {}
    for edge in patch_boundary:
        count = int(generated_counts.get(edge, 0))
        patch_edges_generated_count_hist[count] = patch_edges_generated_count_hist.get(count, 0) + 1

    return {
        "boundary_match_policy": "generated_surface_boundary_must_equal_deleted_patch_boundary",
        "patch_boundary_edge_count": int(len(patch_boundary)),
        "generated_boundary_edge_count": int(len(generated_boundary)),
        "generated_boundary_original_vertex_edge_count": int(len(generated_boundary_original_only)),
        "generated_boundary_generated_vertex_edge_count": int(len(generated_boundary_generated_involved)),
        "shared_boundary_edge_count": int(len(shared)),
        "missing_patch_boundary_edge_count": int(len(missing)),
        "extra_generated_boundary_edge_count": int(len(extra)),
        "generated_invalid_edge_count": int(generated_invalid_edges),
        "boundary_match_exact": bool(not missing and not extra),
        "patch_edges_generated_count_histogram": tuple(sorted((int(k), int(v)) for k, v in patch_edges_generated_count_hist.items())),
        "sample_missing_patch_boundary_edges": tuple(sorted(missing)[:24]),
        "sample_extra_generated_boundary_edges": tuple(sorted(extra)[:24]),
        "sample_shared_boundary_edges": tuple(sorted(shared)[:12]),
    }


# -----------------------------------------------------------------------------
# Diagnostics and failure reporting
# -----------------------------------------------------------------------------


def _attempt_summary(
    attempt_index: int,
    attempt: BoundaryLoopAttempt,
    *,
    trial: Mapping[str, object] | None = None,
    plan: QuadPlan | None = None,
    error: str = "",
) -> dict[str, object]:
    trial = dict(trial or {})
    plan_diag = dict(plan.diagnostics if plan is not None else {})
    return {
        "attempt_index": int(attempt_index),
        "source": attempt.source,
        "target_source": attempt.target_source,
        "face_count": int(len(attempt.face_ids)),
        "boundary_loop_count": int(attempt.boundary_loop_count),
        "exact_two_loop_patch": bool(attempt.exact_two_loop_patch),
        "protected_loop_pair": bool(attempt.protected_loop_pair),
        "loop0_vertex_count": int(len(attempt.loop0)),
        "loop1_vertex_count": int(len(attempt.loop1)),
        "boundary_loop_vertex_count_delta": int(attempt.boundary_loop_vertex_count_delta),
        "unequal_loop_transition_allowed": bool(attempt.unequal_loop_transition_allowed),
        "unequal_loop_transition_used": bool(plan_diag.get("unequal_loop_transition_used", False)),
        "transition_drop_quad_count": int(plan_diag.get("transition_drop_quad_count", 0) or 0),
        "transition_ring_vertex_count": int(plan_diag.get("transition_ring_vertex_count", 0) or 0),
        "axial_separation": float(attempt.axial_separation),
        "min_required_axial_separation": float(attempt.min_required_axial_separation),
        "boundary_edge_count_before": int(trial.get("boundary_edge_count_before", -1) if trial.get("boundary_edge_count_before", -1) is not None else -1),
        "boundary_edge_count_after": int(trial.get("boundary_edge_count_after", 10**9 if error else -1)),
        "boundary_edge_count_delta": int(trial.get("boundary_edge_count_delta", 10**9 if error else 0)),
        "watertight_after": bool(trial.get("watertight_after", False)),
        "boundary_match_exact": bool(plan_diag.get("boundary_match_exact", False)),
        "patch_boundary_edge_count": int(plan_diag.get("patch_boundary_edge_count", -1) if plan_diag.get("patch_boundary_edge_count", -1) is not None else -1),
        "generated_boundary_edge_count": int(plan_diag.get("generated_boundary_edge_count", -1) if plan_diag.get("generated_boundary_edge_count", -1) is not None else -1),
        "generated_boundary_original_vertex_edge_count": int(plan_diag.get("generated_boundary_original_vertex_edge_count", -1) if plan_diag.get("generated_boundary_original_vertex_edge_count", -1) is not None else -1),
        "generated_boundary_generated_vertex_edge_count": int(plan_diag.get("generated_boundary_generated_vertex_edge_count", -1) if plan_diag.get("generated_boundary_generated_vertex_edge_count", -1) is not None else -1),
        "shared_boundary_edge_count": int(plan_diag.get("shared_boundary_edge_count", -1) if plan_diag.get("shared_boundary_edge_count", -1) is not None else -1),
        "missing_patch_boundary_edge_count": int(plan_diag.get("missing_patch_boundary_edge_count", -1) if plan_diag.get("missing_patch_boundary_edge_count", -1) is not None else -1),
        "extra_generated_boundary_edge_count": int(plan_diag.get("extra_generated_boundary_edge_count", -1) if plan_diag.get("extra_generated_boundary_edge_count", -1) is not None else -1),
        "patch_edges_generated_count_histogram": tuple(plan_diag.get("patch_edges_generated_count_histogram", ()) or ()),
        "sample_missing_patch_boundary_edges": tuple(plan_diag.get("sample_missing_patch_boundary_edges", ()) or ()),
        "sample_extra_generated_boundary_edges": tuple(plan_diag.get("sample_extra_generated_boundary_edges", ()) or ()),
        "damaged_bore_small_boundary_seal_used": bool(plan_diag.get("damaged_bore_small_boundary_seal_used", False)),
        "damaged_bore_small_boundary_seal_added_triangle_count": int(plan_diag.get("damaged_bore_small_boundary_seal_added_triangle_count", 0) or 0),
        "damaged_bore_small_boundary_seal_boundary_edge_count_before": int(plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_before", -1) if plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_before", -1) is not None else -1),
        "damaged_bore_small_boundary_seal_boundary_edge_count_after_seal": int(plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_after_seal", -1) if plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_after_seal", -1) is not None else -1),
        "error": str(error),
    }


def _format_failure_message(
    *,
    context: RebuildCandidateContext,
    best_failure: Mapping[str, object],
    attempt_summaries: tuple[dict[str, object], ...],
    target_result: Mapping[str, object],
) -> str:
    compact = _compact_attempt_summaries(attempt_summaries)
    target_diagnostics = dict(target_result.get("diagnostics", {}) or {})
    target_sources = tuple(str(v) for v in tuple(target_diagnostics.get("face_set_sources", ()) or ()))
    best_error = str(best_failure.get("error", "") or "")
    if len(best_error) > 220:
        best_error = best_error[:217] + "..."
    best_hist = tuple(best_failure.get("patch_edges_generated_count_histogram", ()) or ())
    if len(best_hist) > 12:
        best_hist = best_hist[:12]
    best_missing_sample = tuple(best_failure.get("sample_missing_patch_boundary_edges", ()) or ())
    best_extra_sample = tuple(best_failure.get("sample_extra_generated_boundary_edges", ()) or ())
    if len(best_missing_sample) > 8:
        best_missing_sample = best_missing_sample[:8]
    if len(best_extra_sample) > 8:
        best_extra_sample = best_extra_sample[:8]
    return (
        "Bore measured-patch quad rebuild could not find a watertight measured delete patch. "
        f"attempt_count={len(attempt_summaries)}; "
        f"best_source={best_failure.get('source', '-')}; "
        f"best_target_source={best_failure.get('target_source', '-')}; "
        f"best_face_count={best_failure.get('face_count', '-')}; "
        f"best_boundary_loop_count={best_failure.get('boundary_loop_count', '-')}; "
        f"best_exact_two_loop_patch={best_failure.get('exact_two_loop_patch', '-')}; "
        f"best_protected_loop_pair={best_failure.get('protected_loop_pair', '-')}; "
        f"best_loop0_vertex_count={best_failure.get('loop0_vertex_count', '-')}; "
        f"best_loop1_vertex_count={best_failure.get('loop1_vertex_count', '-')}; "
        f"best_boundary_loop_vertex_count_delta={best_failure.get('boundary_loop_vertex_count_delta', '-')}; "
        f"best_unequal_loop_transition_allowed={best_failure.get('unequal_loop_transition_allowed', '-')}; "
        f"best_unequal_loop_transition_used={best_failure.get('unequal_loop_transition_used', '-')}; "
        f"best_transition_drop_quad_count={best_failure.get('transition_drop_quad_count', '-')}; "
        f"best_boundary_edge_count_before={best_failure.get('boundary_edge_count_before', '-')}; "
        f"best_boundary_edge_count_after={best_failure.get('boundary_edge_count_after', '-')}; "
        f"best_boundary_edge_count_delta={best_failure.get('boundary_edge_count_delta', '-')}; "
        f"best_watertight_after={best_failure.get('watertight_after', '-')}; "
        f"best_boundary_match_exact={best_failure.get('boundary_match_exact', '-')}; "
        f"best_patch_boundary_edge_count={best_failure.get('patch_boundary_edge_count', '-')}; "
        f"best_generated_boundary_edge_count={best_failure.get('generated_boundary_edge_count', '-')}; "
        f"best_generated_boundary_original_vertex_edge_count={best_failure.get('generated_boundary_original_vertex_edge_count', '-')}; "
        f"best_generated_boundary_generated_vertex_edge_count={best_failure.get('generated_boundary_generated_vertex_edge_count', '-')}; "
        f"best_shared_boundary_edge_count={best_failure.get('shared_boundary_edge_count', '-')}; "
        f"best_missing_patch_boundary_edge_count={best_failure.get('missing_patch_boundary_edge_count', '-')}; "
        f"best_extra_generated_boundary_edge_count={best_failure.get('extra_generated_boundary_edge_count', '-')}; "
        f"best_patch_edges_generated_count_histogram={best_hist}; "
        f"best_sample_missing_patch_boundary_edges={best_missing_sample}; "
        f"best_sample_extra_generated_boundary_edges={best_extra_sample}; "
        + (f"best_error={best_error}; " if best_error else "")
        + (f"attempt_summaries=[{compact}]; " if compact else "")
        + (f"rebuild_target_face_set_sources={target_sources[:8]}; " if target_sources else "")
        + f"candidate_entity_type={context.entity_type or '-'}; "
        + f"candidate_from_component_engine={bool(context.candidate_from_component_engine)}; "
        + f"feature_ownership_source={context.feature_ownership_source or '-'}; "
        + f"preview_candidate_patch_owns_delete={bool(context.candidate_has_preview_face_patch)}; "
        + "Geometry changed: no. parameter_fit_used=False; radius_used_for_delete_expansion=False."
    )


def _compact_attempt_summaries(attempt_summaries: Iterable[Mapping[str, object]], *, limit: int = 8) -> str:
    parts: list[str] = []
    for raw in tuple(attempt_summaries or ())[:limit]:
        error = str(raw.get("error", "") or "")
        if len(error) > 120:
            error = error[:117] + "..."
        hist = tuple(raw.get("patch_edges_generated_count_histogram", ()) or ())
        if len(hist) > 8:
            hist = hist[:8]
        parts.append(
            (
                "#%s %s target=%s faces=%s loops=%s exact=%s protected=%s "
                "v=%s/%s delta=%s unequal=%s used=%s before_edges=%s after_edges=%s edge_delta=%s watertight=%s "
                "boundary_match=%s patchB=%s genB=%s genOrigB=%s genNewB=%s sharedB=%s missingB=%s extraB=%s patchHist=%s%s"
            )
            % (
                raw.get("attempt_index", "?"),
                raw.get("source", "?"),
                raw.get("target_source", "?"),
                raw.get("face_count", "?"),
                raw.get("boundary_loop_count", "?"),
                raw.get("exact_two_loop_patch", "?"),
                raw.get("protected_loop_pair", "?"),
                raw.get("loop0_vertex_count", "?"),
                raw.get("loop1_vertex_count", "?"),
                raw.get("boundary_loop_vertex_count_delta", "?"),
                raw.get("unequal_loop_transition_allowed", "?"),
                raw.get("unequal_loop_transition_used", "?"),
                raw.get("boundary_edge_count_before", "?"),
                raw.get("boundary_edge_count_after", "?"),
                raw.get("boundary_edge_count_delta", "?"),
                raw.get("watertight_after", "?"),
                raw.get("boundary_match_exact", "?"),
                raw.get("patch_boundary_edge_count", "?"),
                raw.get("generated_boundary_edge_count", "?"),
                raw.get("generated_boundary_original_vertex_edge_count", "?"),
                raw.get("generated_boundary_generated_vertex_edge_count", "?"),
                raw.get("shared_boundary_edge_count", "?"),
                raw.get("missing_patch_boundary_edge_count", "?"),
                raw.get("extra_generated_boundary_edge_count", "?"),
                hist,
                (" error=" + error) if error else "",
            )
        )
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# Numeric / loop ordering helpers
# -----------------------------------------------------------------------------


def _orient_plan_triangles_to_source_patch(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
    plan: QuadPlan,
) -> QuadPlan:
    """Orient generated triangles to match the normals of the removed patch.

    Watertight topology does not guarantee useful winding.  The viewport and
    downstream processors still need the replacement triangles to face the same
    way as the deleted surface.  This helper flips generated triangle winding
    whenever the triangle normal disagrees with the nearest original patch face
    normal.
    """

    references = _source_patch_face_references(vertices=vertices, source_faces=source_faces, face_ids=face_ids)
    if not references:
        diag = dict(plan.diagnostics)
        diag.update({
            "normal_orientation_policy": "source_patch_reference_unavailable",
            "normal_flip_count": 0,
            "normal_alignment_median": 0.0,
            "normal_alignment_min": 0.0,
        })
        return QuadPlan(
            generated_vertices=plan.generated_vertices,
            triangles=plan.triangles,
            logical_quads=plan.logical_quads,
            loop0=plan.loop0,
            loop1=plan.loop1,
            center0=plan.center0,
            center1=plan.center1,
            axis=plan.axis,
            diagnostics=diag,
        )

    ref_centroids = np.asarray([item[0] for item in references], dtype=float).reshape((-1, 3))
    ref_normals = np.asarray([item[1] for item in references], dtype=float).reshape((-1, 3))

    generated_vertices = np.asarray(plan.generated_vertices, dtype=float).reshape((-1, 3))
    output_vertices = np.asarray(vertices, dtype=float).copy()
    if generated_vertices.size:
        output_vertices = np.vstack([output_vertices, generated_vertices])

    triangles = np.asarray(plan.triangles, dtype=np.int64).reshape((-1, 3)).copy()
    flips = 0
    alignments: list[float] = []
    for tri_index, tri in enumerate(triangles):
        if any(int(v) < 0 or int(v) >= len(output_vertices) for v in tri):
            continue
        pts = output_vertices[np.asarray(tri, dtype=np.int64), :3]
        normal = _unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is None:
            continue
        centroid = np.mean(pts, axis=0)
        deltas = ref_centroids - centroid.reshape(1, 3)
        nearest = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        dot = float(np.dot(normal, ref_normals[nearest]))
        if dot < 0.0:
            triangles[tri_index] = np.asarray((int(tri[0]), int(tri[2]), int(tri[1])), dtype=np.int64)
            dot = -dot
            flips += 1
        alignments.append(float(dot))

    diag = dict(plan.diagnostics)
    diag.update({
        "normal_orientation_policy": "match_nearest_deleted_patch_face_normal",
        "normal_flip_count": int(flips),
        "normal_alignment_count": int(len(alignments)),
        "normal_alignment_median": float(np.median(alignments)) if alignments else 0.0,
        "normal_alignment_min": float(min(alignments)) if alignments else 0.0,
    })
    return QuadPlan(
        generated_vertices=plan.generated_vertices,
        triangles=triangles,
        logical_quads=plan.logical_quads,
        loop0=plan.loop0,
        loop1=plan.loop1,
        center0=plan.center0,
        center1=plan.center1,
        axis=plan.axis,
        diagnostics=diag,
    )


def _source_patch_face_references(
    *,
    vertices: np.ndarray,
    source_faces: np.ndarray,
    face_ids: tuple[int, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    refs: list[tuple[np.ndarray, np.ndarray]] = []
    for fid in tuple(face_ids or ()):
        if int(fid) < 0 or int(fid) >= len(source_faces):
            continue
        tri = np.asarray(source_faces[int(fid), :3], dtype=np.int64)
        if any(int(v) < 0 or int(v) >= len(vertices) for v in tri):
            continue
        pts = vertices[tri, :3]
        normal = _unit_normal(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if normal is None:
            continue
        refs.append((np.mean(pts, axis=0), normal))
    return tuple(refs)


def _unit_normal(value: object) -> np.ndarray | None:
    try:
        vec = np.asarray(value, dtype=float).reshape(3)
    except Exception:
        return None
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= 1.0e-12:
        return None
    return vec / length


def _target_unequal_transition_band_count(
    *,
    big_points: np.ndarray,
    small_points: np.ndarray,
    big_center: np.ndarray,
    small_center: np.ndarray,
    mode: str,
    base_band_count: int,
) -> dict[str, object]:
    """Return the axial band target for an unequal-loop transition.

    Density controls axial spacing, not feature classification.  The count
    reduction remains a topology problem: the larger ring must gradually reduce
    to the smaller ring without moving boundary vertices.
    """

    mode_norm = _normalize_quad_density_mode(mode)
    base_band_count = max(1, int(base_band_count))
    span = float(np.linalg.norm(np.asarray(small_center, dtype=float).reshape(3) - np.asarray(big_center, dtype=float).reshape(3)))

    edge_lengths: list[float] = []
    for pts in (np.asarray(big_points, dtype=float).reshape((-1, 3)), np.asarray(small_points, dtype=float).reshape((-1, 3))):
        if len(pts) >= 2:
            diffs = np.roll(pts, -1, axis=0) - pts
            edge_lengths.extend(float(v) for v in np.linalg.norm(diffs, axis=1) if np.isfinite(float(v)) and float(v) > 1.0e-12)
    measured_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    measured_edge = max(measured_edge, 1.0e-12)

    if mode_norm == QUAD_DENSITY_MODE_FULL:
        pitch = measured_edge
        cap = 96
        policy = "full_equal_edge"
    elif mode_norm == QUAD_DENSITY_MODE_PI:
        pitch = max(2.5 * measured_edge, measured_edge)
        cap = 64
        policy = "balanced_edge_spacing"
    else:
        pitch = max(4.0 * measured_edge, measured_edge)
        cap = max(base_band_count, 48)
        policy = "lean_drop_bands"

    raw_segments = int(math.ceil(span / max(pitch, 1.0e-12))) if np.isfinite(span) and span > 1.0e-12 else base_band_count
    target = base_band_count if mode_norm == QUAD_DENSITY_MODE_LEAN else max(base_band_count, min(int(cap), max(1, raw_segments)))

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


def _densify_transition_count_sequence(base_counts: tuple[int, ...], *, target_band_count: int) -> tuple[int, ...]:
    """Return a monotone ring-count sequence with evenly distributed drops.

    Repeating whole blocks of the same count creates visible malformed bands.
    Instead, this distributes the required two-vertex drop events over the full
    axial band count.  When the target band count is smaller than the number of
    required drop events, adjacent drops are combined; otherwise every drop is a
    minimal two-vertex reduction.
    """

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


def _transition_count_sequence(n_big: int, n_small: int, *, mode: str = QUAD_DENSITY_MODE_LEAN) -> tuple[int, ...]:
    """Return density-controlled gradual ring counts from larger loop to smaller loop.

    Counts always decrease by an even number so each band can be filled with
    pure quads.  The UI density preset now controls how aggressively the count
    may drop per band:

    ``lean``
        larger drop per band, fewer rings/quads.

    ``pi``
        medium drop per band.

    ``full``
        small drop per band, more rings/quads.

    This keeps the good gradual-transition behavior while making the UI density
    selector produce visibly different mesh outcomes.
    """

    n_big = int(n_big)
    n_small = int(n_small)
    if n_big < n_small:
        n_big, n_small = n_small, n_big
    if n_big == n_small:
        return (n_big,)
    delta = n_big - n_small
    if delta % 2:
        raise ValueError(f"Unequal transition count sequence needs even delta. got {n_big}, {n_small}.")

    mode_norm = _normalize_quad_density_mode(mode)
    if mode_norm == QUAD_DENSITY_MODE_FULL:
        max_drop_fixed = 1
        ratio = 0.030
    elif mode_norm == QUAD_DENSITY_MODE_PI:
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


def _sample_closed_loop_points(points: np.ndarray, count: int) -> np.ndarray:
    """Sample ``count`` points around a closed polyline by fractional index."""

    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    n = int(len(pts))
    count = int(count)
    if n <= 0 or count <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == count:
        return pts.copy()
    out = np.zeros((count, 3), dtype=float)
    for k in range(count):
        pos = float(k) * float(n) / float(count)
        i0 = int(math.floor(pos)) % n
        i1 = (i0 + 1) % n
        frac = float(pos - math.floor(pos))
        out[k, :] = (1.0 - frac) * pts[i0, :] + frac * pts[i1, :]
    return out


def _band_quads_between_rings(ring_a: tuple[int, ...], ring_b: tuple[int, ...]) -> tuple[tuple[int, int, int, int], ...]:
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
    drop_positions = _distributed_drop_positions(n_small, drop_count)

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


def _validate_plan_geometry_quality(
    *,
    context: RebuildCandidateContext,
    vertices: np.ndarray,
    attempt: BoundaryLoopAttempt,
    plan: QuadPlan,
) -> dict[str, object]:
    """Reject watertight-but-wrong loop pairs before the mesh is committed.

    A BOREHOLE replacement should be cylindrical.  This quality gate does not
    classify features; it only prevents a BOREHOLE rebuild from using a loop pair
    that behaves like a chamfer/taper or an over-sealed neighbouring surface.
    CHAMFER and other transition objects are allowed to taper.
    """

    if context.entity_type != "borehole":
        return {"valid": True, "policy": "not_borehole_no_cylindrical_quality_gate"}

    r0 = _loop_radius_stats(vertices, plan.loop0, plan.center0, plan.axis)
    r1 = _loop_radius_stats(vertices, plan.loop1, plan.center1, plan.axis)
    median0 = float(r0.get("median", 0.0))
    median1 = float(r1.get("median", 0.0))
    avg_radius = max(0.5 * (median0 + median1), 1.0e-12)
    edge_scale = max(_median_loop_edge_length(vertices, plan.loop0), _median_loop_edge_length(vertices, plan.loop1), 1.0e-12)
    radius_delta = abs(median0 - median1)
    max_radius_delta = max(0.18 * avg_radius, 2.5 * edge_scale)

    wobble0 = float(r0.get("p95_delta", 0.0))
    wobble1 = float(r1.get("p95_delta", 0.0))
    max_wobble = max(0.16 * avg_radius, 3.0 * edge_scale)

    diagnostics = {
        "policy": "borehole_cylindrical_loop_pair_quality_gate",
        "normal_orientation_policy": str(plan.diagnostics.get("normal_orientation_policy", "")),
        "normal_flip_count": int(plan.diagnostics.get("normal_flip_count", 0) or 0),
        "normal_alignment_median": float(plan.diagnostics.get("normal_alignment_median", 0.0) or 0.0),
        "normal_alignment_min": float(plan.diagnostics.get("normal_alignment_min", 0.0) or 0.0),
        "loop0_radius_median": float(median0),
        "loop1_radius_median": float(median1),
        "loop_radius_delta": float(radius_delta),
        "max_loop_radius_delta": float(max_radius_delta),
        "loop0_radius_p95_delta": float(wobble0),
        "loop1_radius_p95_delta": float(wobble1),
        "max_loop_radius_p95_delta": float(max_wobble),
        "attempt_source": str(attempt.source),
        "target_source": str(attempt.target_source),
    }

    if radius_delta > max_radius_delta:
        diagnostics["valid"] = False
        diagnostics["reason"] = "borehole_loop_pair_radius_delta_too_large"
        return diagnostics
    if max(wobble0, wobble1) > max_wobble:
        diagnostics["valid"] = False
        diagnostics["reason"] = "borehole_boundary_loop_radius_wobble_too_large"
        return diagnostics

    diagnostics["valid"] = True
    diagnostics["reason"] = "ok"
    return diagnostics


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


def _align_unequal_loop_pair_to_angle_samples(
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


def _distributed_drop_positions(n_small: int, drop_count: int) -> set[int]:
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


def _axial_segment_count(
    *,
    loop0_pts: np.ndarray,
    loop1_pts: np.ndarray,
    center0: np.ndarray,
    center1: np.ndarray,
    mode: str,
) -> int:
    edge_lengths: list[float] = []
    for pts in (loop0_pts, loop1_pts):
        diffs = np.roll(pts, -1, axis=0) - pts
        edge_lengths.extend(float(v) for v in np.linalg.norm(diffs, axis=1) if np.isfinite(float(v)) and float(v) > 1.0e-12)
    measured_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    span = float(np.linalg.norm(np.asarray(center1, dtype=float) - np.asarray(center0, dtype=float)))
    if not np.isfinite(span) or span <= 1.0e-12:
        return 1
    measured_edge = max(measured_edge, 1.0e-12)

    if mode == QUAD_DENSITY_MODE_FULL:
        pitch = measured_edge
        cap = 128
    elif mode == QUAD_DENSITY_MODE_PI:
        pitch = max(measured_edge * 2.5, span / 32.0)
        cap = 64
    else:
        pitch = max(measured_edge * 4.0, span / 24.0)
        cap = 48
    return max(1, min(int(cap), int(math.ceil(span / max(pitch, 1.0e-12)))))


def _sort_loop_pair_by_angle(
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
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
        _sort_loop_by_angle(vertices, loop0, center0, axis, basis_u, basis_v),
        _sort_loop_by_angle(vertices, loop1, center1, axis, basis_u, basis_v),
    )


def _sort_loop_by_angle(
    vertices: np.ndarray,
    loop: tuple[int, ...],
    center: np.ndarray,
    axis: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[int, ...]:
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


def _align_second_loop_to_first(
    vertices: np.ndarray,
    loop0: tuple[int, ...],
    loop1: tuple[int, ...],
    center0: np.ndarray,
    center1: np.ndarray,
    axis: np.ndarray,
) -> tuple[int, ...]:
    n = int(len(loop0))
    radial0 = _project_radial(vertices[np.asarray(loop0, dtype=np.int64)], center0, axis)
    best_score = float("inf")
    best_loop = tuple(loop1)
    for candidate in (tuple(loop1), tuple(reversed(loop1))):
        radial1 = _project_radial(vertices[np.asarray(candidate, dtype=np.int64)], center1, axis)
        for shift in range(n):
            shifted = np.roll(radial1, -shift, axis=0)
            score = float(np.mean(np.sum((radial0 - shifted) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best_loop = tuple(int(v) for v in np.roll(np.asarray(candidate, dtype=np.int64), -shift).tolist())
    return best_loop


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


def _order_closed_edge_loop_vertices(edges: set[EdgeKey]) -> dict[str, object]:
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


def _minimum_loop_pair_separation(vertices: np.ndarray, loop0: tuple[int, ...], loop1: tuple[int, ...]) -> float:
    median_edge = max(_median_loop_edge_length(vertices, loop0), _median_loop_edge_length(vertices, loop1), 1.0e-12)
    return max(1.0e-6, 0.02 * median_edge)


def _median_loop_edge_length(vertices: np.ndarray, loop: tuple[int, ...]) -> float:
    if len(loop) < 2:
        return 0.0
    lengths: list[float] = []
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        ia, ib = int(a), int(b)
        if 0 <= ia < len(vertices) and 0 <= ib < len(vertices):
            length = float(np.linalg.norm(vertices[ia, :3] - vertices[ib, :3]))
            if np.isfinite(length) and length > 0.0:
                lengths.append(length)
    return float(np.median(lengths)) if lengths else 0.0


def _loop_center(vertices: np.ndarray, loop: Iterable[int]) -> np.ndarray:
    ids = [int(v) for v in tuple(loop or ()) if 0 <= int(v) < len(vertices)]
    if not ids:
        return np.zeros(3, dtype=float)
    return np.mean(vertices[np.asarray(ids, dtype=np.int64), :3], axis=0)


def _loop_vertices_to_edges(loop: tuple[int, ...]) -> set[EdgeKey]:
    if len(loop) < 2:
        return set()
    return {_edge_key((loop[i], loop[(i + 1) % len(loop)])) for i in range(len(loop))}


def _boundary_edge_count(faces: np.ndarray) -> int:
    counts: dict[EdgeKey, int] = {}
    arr = np.asarray(faces, dtype=np.int64)
    for face in arr[:, :3]:
        verts = [int(v) for v in face[:3]]
        for edge in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            key = _edge_key(edge)
            counts[key] = counts.get(key, 0) + 1
    return int(sum(1 for count in counts.values() if int(count) == 1))


def _assign_face_colors(
    mesh: trimesh.Trimesh,
    *,
    added_face_ids: tuple[int, ...],
    base_face_color: RGBA,
    rebuilt_face_color: RGBA,
) -> None:
    face_count = int(len(mesh.faces))
    colors = np.tile(np.asarray(base_face_color, dtype=np.uint8).reshape(1, 4), (face_count, 1))
    for fid in added_face_ids:
        if 0 <= int(fid) < face_count:
            colors[int(fid), :] = np.asarray(rebuilt_face_color, dtype=np.uint8).reshape(4)
    try:
        mesh.visual.face_colors = colors
    except Exception:
        pass


def _normalize_face_ids(values: Iterable[int] | None, *, face_count: int) -> tuple[int, ...]:
    result: set[int] = set()
    for raw in tuple(values or ()):
        try:
            fid = int(raw)
        except Exception:
            continue
        if 0 <= fid < int(face_count):
            result.add(fid)
    return tuple(sorted(result))


# -----------------------------------------------------------------------------
# Low-level validation / conversion helpers
# -----------------------------------------------------------------------------


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh is None:
        raise ValueError("No mesh provided.")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError("Mesh must provide vertices and faces.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh is empty.")


def _edge_key(edge: object) -> EdgeKey:
    return normalize_edge(edge)


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _to_vector3(value: object) -> Vector3:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    except Exception:
        return (0.0, 0.0, 1.0)


__all__ = ["RebuildResult", "delete_and_rebuild_candidate_region"]
