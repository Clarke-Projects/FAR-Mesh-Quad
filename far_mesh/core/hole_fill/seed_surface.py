from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import trimesh

from far_mesh.core.hole_context import candidate_boundary_vertex_ids
from far_mesh.core.hole_curvature import (
    build_curvature_sphere_uvdelaunay_preview_mesh,
    fit_sphere_for_hole_context,
    project_points_to_sphere,
)
from far_mesh.core.hole_patch_quality import (
    analyze_patch_quality,
    compute_boundary_target_normals,
    mesh_edge_length_median,
)
from far_mesh.core.hole_surface_context import (
    build_mls_surface_projector,
    collect_normal_compatible_support_context,
    validate_patch_topology,
)

from far_mesh.core.hole_fill.surface_target import (
    build_confidence_weighted_target_pull_probe,
    build_support_target_report,
    build_target_confidence_profile,
)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: object) -> float | None:
    try:
        if value in (None, "", "-", {}, [], ()):
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _fit_value(fit: object, key: str) -> float | None:
    if fit is None:
        return None
    if isinstance(fit, Mapping):
        return _float_or_none(fit.get(key))
    return _float_or_none(getattr(fit, key, None))


def _normalize_or_default(value: object, default: np.ndarray) -> np.ndarray:
    try:
        vec = np.asarray(value, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm > 1.0e-12 and np.isfinite(norm):
            return vec / norm
    except Exception:
        pass
    return np.asarray(default, dtype=float).reshape(3)


def _boundary_edge_median(mesh: trimesh.Trimesh, boundary_ids: tuple[int, ...]) -> float:
    vertices = np.asarray(mesh.vertices, dtype=float)
    lengths: list[float] = []
    for index, a in enumerate(boundary_ids):
        b = boundary_ids[(index + 1) % len(boundary_ids)]
        if int(a) < 0 or int(b) < 0 or int(a) >= len(vertices) or int(b) >= len(vertices):
            continue
        length = float(np.linalg.norm(vertices[int(b)] - vertices[int(a)]))
        if np.isfinite(length) and length > 1.0e-12:
            lengths.append(length)
    if not lengths:
        return 1.0
    return float(np.median(np.asarray(lengths, dtype=float)))


def analyze_seed_surface_alignment(
    *,
    base_mesh: trimesh.Trimesh,
    candidate: Any,
    seed_metadata: Mapping[str, object] | None = None,
    support_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
    nearest_count: int = 16,
) -> dict[str, object]:
    """Measure how the raw curvature-sphere UV-Delaunay seed aligns to support.

    Diagnostic-only:
    - rebuilds the raw seed in preview mode
    - projects generated seed vertices to the local MLS support surface
    - reports unsigned and signed seed/support offset
    - does not mutate source mesh and does not change selected preview geometry
    """

    metadata = _mapping(seed_metadata)

    try:
        boundary_ids = tuple(int(v) for v in candidate_boundary_vertex_ids(candidate))
        seed = build_curvature_sphere_uvdelaunay_preview_mesh(
            base_mesh,
            candidate,
            rings=int(support_rings),
        )
        seed_mesh = seed.get("preview_mesh")
        if not isinstance(seed_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_uvdelaunay seed did not return a Trimesh")

        new_vertex_ids = tuple(
            int(v)
            for v in seed.get("new_vertex_ids", ())
            if int(v) >= 0 and int(v) < int(len(seed_mesh.vertices))
        )

        support_context = collect_normal_compatible_support_context(
            base_mesh,
            boundary_ids,
            max_rings=int(support_rings),
            smooth_dot_threshold=float(smooth_dot_threshold),
        )
        projector = build_mls_surface_projector(
            base_mesh,
            support_face_ids=support_context.support_face_ids,
            nearest_count=int(nearest_count),
        )

        try:
            context_edge = float(
                mesh_edge_length_median(
                    base_mesh,
                    face_ids=support_context.support_face_ids,
                )
            )
        except Exception:
            context_edge = _boundary_edge_median(base_mesh, boundary_ids)

        context_edge = max(float(context_edge), 1.0e-12)

        mean_normal = _normalize_or_default(
            support_context.mean_normal,
            np.asarray([0.0, 0.0, 1.0], dtype=float),
        )

        seed_vertices = np.asarray(seed_mesh.vertices, dtype=float)
        distances: list[float] = []
        signed_offsets: list[float] = []

        for vertex_id in new_vertex_ids:
            point = seed_vertices[int(vertex_id)]
            target = projector.project(point)
            delta = np.asarray(target, dtype=float) - point
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance):
                continue
            distances.append(distance)
            signed_offsets.append(float(np.dot(delta, mean_normal)))

        if distances:
            dist = np.asarray(distances, dtype=float)
            signed = np.asarray(signed_offsets, dtype=float)
            projection_mean = float(np.mean(dist))
            projection_median = float(np.median(dist))
            projection_max = float(np.max(dist))
            signed_mean = float(np.mean(signed))
            signed_min = float(np.min(signed))
            signed_max = float(np.max(signed))
        else:
            projection_mean = 0.0
            projection_median = 0.0
            projection_max = 0.0
            signed_mean = 0.0
            signed_min = 0.0
            signed_max = 0.0

        mean_ratio = float(projection_mean / context_edge)
        max_ratio = float(projection_max / context_edge)

        requested_surface_weight = _float_or_none(
            metadata.get("requested_relaxation_surface_weight")
        )
        effective_surface_weight = _float_or_none(
            metadata.get("relaxation_surface_weight")
        )
        if requested_surface_weight is None:
            requested_surface_weight = _float_or_none(
                metadata.get("adaptive_selected_surface_weight")
            )
        if effective_surface_weight is None:
            effective_surface_weight = requested_surface_weight

        fit = seed.get("fit") or _mapping(seed.get("fit_report")).get("fit")
        sphere_rms = _fit_value(fit, "rms_error")
        sphere_max = _fit_value(fit, "max_abs_error")
        sphere_point_count = _fit_value(fit, "point_count")

        reasons: list[str] = []

        if len(new_vertex_ids) <= 0:
            status = "not_applicable"
            action = "no_generated_seed_vertices"
            reasons.append("seed generated no interior vertices")
        else:
            status = "ok"
            action = "seed_alignment_ok"

            if max_ratio > 0.18 or mean_ratio > 0.075:
                status = "bad"
                action = "inspect_or_replace_seed_surface_target"
                reasons.append(
                    f"seed-to-support projection error is high: mean_ratio={mean_ratio:.6g}, max_ratio={max_ratio:.6g}"
                )
            elif max_ratio > 0.08 or mean_ratio > 0.030:
                status = "warning"
                action = "inspect_seed_surface_target"
                reasons.append(
                    f"seed-to-support projection error is elevated: mean_ratio={mean_ratio:.6g}, max_ratio={max_ratio:.6g}"
                )

            if abs(signed_mean) / context_edge > 0.025:
                if status == "ok":
                    status = "warning"
                    action = "inspect_seed_surface_signed_bias"
                reasons.append(
                    f"seed has signed support-surface bias: signed_mean_ratio={signed_mean / context_edge:.6g}"
                )

            if (effective_surface_weight or 0.0) <= 1.0e-9 and max_ratio > 0.030:
                if status == "ok":
                    status = "warning"
                action = "surface_pull_disabled_despite_seed_offset"
                reasons.append(
                    "effective surface weight is zero while seed/support offset is nontrivial"
                )

            if support_context.normal_spread_degrees >= 45.0 and max_ratio > 0.030:
                reasons.append(
                    f"support normal spread is elevated ({support_context.normal_spread_degrees:.6g}°), so global MLS may need per-vertex confidence"
                )

            if not reasons:
                reasons.append("raw seed aligns with MLS support surface within diagnostic tolerance")

        return {
            "status": status,
            "action": action,
            "seed_backend": str(metadata.get("seed_backend") or "curvature_sphere_uvdelaunay"),
            "seed_surface_kind": str(seed.get("seed_surface_kind") or metadata.get("seed_surface_kind") or "unknown"),
            "seed_surface_error": seed.get("seed_surface_error", metadata.get("seed_surface_error", "-")),
            "sphere_fit_rms_error": sphere_rms if sphere_rms is not None else "-",
            "sphere_fit_max_abs_error": sphere_max if sphere_max is not None else "-",
            "sphere_fit_point_count": int(sphere_point_count) if sphere_point_count is not None else "-",
            "surface_guidance": str(metadata.get("surface_guidance") or "normal_compatible_mls_quadratic"),
            "requested_surface_weight": requested_surface_weight if requested_surface_weight is not None else "-",
            "effective_surface_weight": effective_surface_weight if effective_surface_weight is not None else "-",
            "support_normal_spread_degrees": float(support_context.normal_spread_degrees),
            "support_face_count": int(len(support_context.support_face_ids)),
            "support_vertex_count": int(len(support_context.support_vertex_ids)),
            "seed_generated_vertex_count": int(len(new_vertex_ids)),
            "seed_projection_distance_mean": float(projection_mean),
            "seed_projection_distance_median": float(projection_median),
            "seed_projection_distance_max": float(projection_max),
            "seed_projection_distance_mean_to_context_edge_ratio": float(mean_ratio),
            "seed_projection_distance_max_to_context_edge_ratio": float(max_ratio),
            "seed_signed_offset_mean": float(signed_mean),
            "seed_signed_offset_min": float(signed_min),
            "seed_signed_offset_max": float(signed_max),
            "context_edge_length_median": float(context_edge),
            "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        }
    except Exception as exc:
        return {
            "status": "error",
            "action": "seed_alignment_diagnostic_failed",
            "seed_backend": str(metadata.get("seed_backend") or "curvature_sphere_uvdelaunay"),
            "seed_surface_kind": str(metadata.get("seed_surface_kind") or "unknown"),
            "seed_surface_error": metadata.get("seed_surface_error", "-"),
            "sphere_fit_rms_error": "-",
            "sphere_fit_max_abs_error": "-",
            "sphere_fit_point_count": "-",
            "surface_guidance": str(metadata.get("surface_guidance") or "-"),
            "requested_surface_weight": metadata.get("requested_relaxation_surface_weight", "-"),
            "effective_surface_weight": metadata.get("relaxation_surface_weight", "-"),
            "support_normal_spread_degrees": "-",
            "support_face_count": "-",
            "support_vertex_count": "-",
            "seed_generated_vertex_count": "-",
            "seed_projection_distance_mean": "-",
            "seed_projection_distance_median": "-",
            "seed_projection_distance_max": "-",
            "seed_projection_distance_mean_to_context_edge_ratio": "-",
            "seed_projection_distance_max_to_context_edge_ratio": "-",
            "seed_signed_offset_mean": "-",
            "seed_signed_offset_min": "-",
            "seed_signed_offset_max": "-",
            "context_edge_length_median": "-",
            "reasons": (f"seed surface alignment diagnostic failed: {exc}",),
        }


def _mean_max_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.size == 0:
        return 0.0, 0.0

    distances = np.linalg.norm(left - right, axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return 0.0, 0.0

    return float(np.mean(distances)), float(np.max(distances))


def _project_points_to_pca_plane(points: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    support = np.asarray(support_points, dtype=float).reshape((-1, 3))

    if len(support) < 3:
        return pts.copy()

    centroid = np.mean(support, axis=0)
    centered = support - centroid.reshape(1, 3)

    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
    except Exception:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=float)

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12 or not np.isfinite(normal_norm):
        return pts.copy()

    normal = normal / normal_norm
    offsets = (pts - centroid.reshape(1, 3)) @ normal
    return pts - offsets.reshape((-1, 1)) * normal.reshape((1, 3))


def _mls_target_points_for_seed(
    *,
    base_mesh: trimesh.Trimesh,
    boundary_ids: tuple[int, ...],
    seed_points: np.ndarray,
    rings: int,
    smooth_dot_threshold: float,
    nearest_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    support_context = collect_normal_compatible_support_context(
        base_mesh,
        boundary_ids,
        max_rings=int(rings),
        smooth_dot_threshold=float(smooth_dot_threshold),
    )
    projector = build_mls_surface_projector(
        base_mesh,
        support_face_ids=support_context.support_face_ids,
        nearest_count=int(nearest_count),
    )
    targets = np.asarray(
        [projector.project(point) for point in np.asarray(seed_points, dtype=float)],
        dtype=float,
    )
    return targets, {
        "rings": int(rings),
        "support_face_count": int(len(support_context.support_face_ids)),
        "support_vertex_count": int(len(support_context.support_vertex_ids)),
        "normal_spread_degrees": float(support_context.normal_spread_degrees),
        "contamination_score": float(support_context.contamination_score),
    }


def analyze_support_target_disagreement(
    *,
    base_mesh: trimesh.Trimesh,
    candidate: Any,
    seed_metadata: Mapping[str, object] | None = None,
    support_rings: int = 2,
    smooth_dot_threshold: float = 0.50,
    nearest_count: int = 16,
) -> dict[str, object]:
    """Compare candidate support-surface targets for the same raw seed points.

    Diagnostic-only:
    - rebuilds the raw curvature-sphere UV-Delaunay seed
    - samples several possible support targets at the generated seed vertices
    - reports target disagreement
    - does not mutate source mesh and does not change selected preview geometry
    """

    metadata = _mapping(seed_metadata)

    try:
        boundary_ids = tuple(int(v) for v in candidate_boundary_vertex_ids(candidate))
        seed = build_curvature_sphere_uvdelaunay_preview_mesh(
            base_mesh,
            candidate,
            rings=int(support_rings),
        )
        seed_mesh = seed.get("preview_mesh")
        if not isinstance(seed_mesh, trimesh.Trimesh):
            raise ValueError("curvature_sphere_uvdelaunay seed did not return a Trimesh")

        new_vertex_ids = tuple(
            int(v)
            for v in seed.get("new_vertex_ids", ())
            if int(v) >= 0 and int(v) < int(len(seed_mesh.vertices))
        )
        if not new_vertex_ids:
            return {
                "status": "not_applicable",
                "action": "no_generated_seed_vertices",
                "reasons": ("seed generated no interior vertices",),
            }

        seed_points = np.asarray(seed_mesh.vertices, dtype=float)[list(new_vertex_ids)]

        try:
            context_edge = float(mesh_edge_length_median(base_mesh))
        except Exception:
            context_edge = _boundary_edge_median(base_mesh, boundary_ids)
        context_edge = max(float(context_edge), 1.0e-12)

        mls2, mls2_meta = _mls_target_points_for_seed(
            base_mesh=base_mesh,
            boundary_ids=boundary_ids,
            seed_points=seed_points,
            rings=int(support_rings),
            smooth_dot_threshold=float(smooth_dot_threshold),
            nearest_count=int(nearest_count),
        )

        mls1, mls1_meta = _mls_target_points_for_seed(
            base_mesh=base_mesh,
            boundary_ids=boundary_ids,
            seed_points=seed_points,
            rings=1,
            smooth_dot_threshold=float(smooth_dot_threshold),
            nearest_count=int(nearest_count),
        )

        mls3, mls3_meta = _mls_target_points_for_seed(
            base_mesh=base_mesh,
            boundary_ids=boundary_ids,
            seed_points=seed_points,
            rings=3,
            smooth_dot_threshold=float(smooth_dot_threshold),
            nearest_count=int(nearest_count),
        )

        sphere_fit_report = fit_sphere_for_hole_context(
            base_mesh,
            candidate,
            rings=int(support_rings),
        )
        sphere_fit = sphere_fit_report["fit"]
        sphere_targets = project_points_to_sphere(seed_points, sphere_fit)

        support_context = collect_normal_compatible_support_context(
            base_mesh,
            boundary_ids,
            max_rings=int(support_rings),
            smooth_dot_threshold=float(smooth_dot_threshold),
        )
        support_points = np.asarray(base_mesh.vertices, dtype=float)[
            list(support_context.support_vertex_ids)
        ]
        plane_targets = _project_points_to_pca_plane(seed_points, support_points)

        mls2_vs_sphere_mean, mls2_vs_sphere_max = _mean_max_distance(mls2, sphere_targets)
        mls2_vs_plane_mean, mls2_vs_plane_max = _mean_max_distance(mls2, plane_targets)
        mls1_vs_mls2_mean, mls1_vs_mls2_max = _mean_max_distance(mls1, mls2)
        mls2_vs_mls3_mean, mls2_vs_mls3_max = _mean_max_distance(mls2, mls3)

        mls2_vs_sphere_mean_ratio = float(mls2_vs_sphere_mean / context_edge)
        mls2_vs_sphere_max_ratio = float(mls2_vs_sphere_max / context_edge)
        mls2_vs_plane_mean_ratio = float(mls2_vs_plane_mean / context_edge)
        mls2_vs_plane_max_ratio = float(mls2_vs_plane_max / context_edge)
        mls1_vs_mls2_mean_ratio = float(mls1_vs_mls2_mean / context_edge)
        mls1_vs_mls2_max_ratio = float(mls1_vs_mls2_max / context_edge)
        mls2_vs_mls3_mean_ratio = float(mls2_vs_mls3_mean / context_edge)
        mls2_vs_mls3_max_ratio = float(mls2_vs_mls3_max / context_edge)

        comparison_ratios = {
            "mls2_vs_sphere": max(mls2_vs_sphere_mean_ratio, mls2_vs_sphere_max_ratio),
            "mls2_vs_plane": max(mls2_vs_plane_mean_ratio, mls2_vs_plane_max_ratio),
            "mls1_vs_mls2": max(mls1_vs_mls2_mean_ratio, mls1_vs_mls2_max_ratio),
            "mls2_vs_mls3": max(mls2_vs_mls3_mean_ratio, mls2_vs_mls3_max_ratio),
        }
        target_disagreement_source = max(
            comparison_ratios,
            key=lambda key: float(comparison_ratios[key]),
        )

        mean_values = [
            mls2_vs_sphere_mean,
            mls2_vs_plane_mean,
            mls1_vs_mls2_mean,
            mls2_vs_mls3_mean,
        ]
        max_values = [
            mls2_vs_sphere_max,
            mls2_vs_plane_max,
            mls1_vs_mls2_max,
            mls2_vs_mls3_max,
        ]

        target_disagreement_mean = float(max(mean_values))
        target_disagreement_max = float(max(max_values))
        target_disagreement_mean_ratio = float(target_disagreement_mean / context_edge)
        target_disagreement_max_ratio = float(target_disagreement_max / context_edge)

        mean_normal = _normalize_or_default(
            support_context.mean_normal,
            np.asarray([0.0, 0.0, 1.0], dtype=float),
        )
        signed_mls2_sphere = (sphere_targets - mls2) @ mean_normal
        signed_mls1_mls2 = (mls1 - mls2) @ mean_normal
        signed_mls2_plane = (plane_targets - mls2) @ mean_normal

        signed_values = np.concatenate(
            [
                np.asarray(signed_mls2_sphere, dtype=float).reshape(-1),
                np.asarray(signed_mls1_mls2, dtype=float).reshape(-1),
                np.asarray(signed_mls2_plane, dtype=float).reshape(-1),
            ]
        )
        signed_values = signed_values[np.isfinite(signed_values)]

        if signed_values.size:
            signed_mean = float(np.mean(signed_values))
            signed_min = float(np.min(signed_values))
            signed_max = float(np.max(signed_values))
        else:
            signed_mean = 0.0
            signed_min = 0.0
            signed_max = 0.0

        signed_plane_values = np.asarray(signed_mls2_plane, dtype=float).reshape(-1)
        signed_plane_values = signed_plane_values[np.isfinite(signed_plane_values)]

        if signed_plane_values.size:
            signed_plane_mean = float(np.mean(signed_plane_values))
            signed_plane_min = float(np.min(signed_plane_values))
            signed_plane_max = float(np.max(signed_plane_values))
        else:
            signed_plane_mean = 0.0
            signed_plane_min = 0.0
            signed_plane_max = 0.0

        reasons: list[str] = []

        status = "ok"
        action = "current_target_consistent"

        if target_disagreement_max_ratio > 0.18 or target_disagreement_mean_ratio > 0.075:
            status = "bad"
            action = "replace_or_confidence_blend_surface_target"
            reasons.append(
                "support target models strongly disagree: "
                f"mean_ratio={target_disagreement_mean_ratio:.6g}, "
                f"max_ratio={target_disagreement_max_ratio:.6g}"
            )
        elif target_disagreement_max_ratio > 0.08 or target_disagreement_mean_ratio > 0.030:
            status = "warning"
            action = "confidence_blend_surface_target"
            reasons.append(
                "support target models disagree: "
                f"mean_ratio={target_disagreement_mean_ratio:.6g}, "
                f"max_ratio={target_disagreement_max_ratio:.6g}"
            )
        else:
            reasons.append("support target models agree within diagnostic tolerance")

        if mls2_meta["normal_spread_degrees"] >= 45.0:
            if status == "ok":
                status = "warning"
                action = "use_per_vertex_target_confidence"
            reasons.append(
                f"ring-2 support normal spread is elevated ({mls2_meta['normal_spread_degrees']:.6g}°)"
            )

        if mls1_vs_mls2_max / context_edge > 0.05:
            reasons.append(
                "ring-1 and ring-2 MLS targets differ; local ring choice affects seed target"
            )

        if mls2_vs_sphere_max / context_edge > 0.05:
            reasons.append(
                "MLS and sphere targets differ; curvature-aware target blend should be evaluated"
            )

        if abs(signed_mean) / context_edge > 0.02:
            reasons.append(
                f"target disagreement has signed bias: signed_mean_ratio={signed_mean / context_edge:.6g}"
            )

        target_report = build_support_target_report(
            mls2_vs_sphere_mean=mls2_vs_sphere_mean,
            mls2_vs_sphere_max=mls2_vs_sphere_max,
            mls2_vs_plane_mean=mls2_vs_plane_mean,
            mls2_vs_plane_max=mls2_vs_plane_max,
            mls1_vs_mls2_mean=mls1_vs_mls2_mean,
            mls1_vs_mls2_max=mls1_vs_mls2_max,
            mls2_vs_mls3_mean=mls2_vs_mls3_mean,
            mls2_vs_mls3_max=mls2_vs_mls3_max,
            context_edge_length_median=context_edge,
            normal_spread_degrees=mls2_meta["normal_spread_degrees"],
        )
        target_model_report = target_report.to_dict()
        target_model_recommendation_report = target_report.recommendation

        target_disagreement_source = target_model_recommendation_report.disagreement_source
        target_model_recommendation = target_model_recommendation_report.recommendation
        target_model_confidence = target_model_recommendation_report.confidence

        for reason in target_model_recommendation_report.reasons:
            reasons.append(str(reason))

        target_confidence_profile = build_target_confidence_profile(
            mls2_targets=mls2,
            sphere_targets=sphere_targets,
            plane_targets=plane_targets,
            mls1_targets=mls1,
            mls3_targets=mls3,
            context_edge_length_median=context_edge,
            normal_spread_degrees=mls2_meta["normal_spread_degrees"],
            max_surface_weight=0.25,
        )
        target_confidence_profile_dict = target_confidence_profile.to_dict()

        for reason in target_confidence_profile.reasons:
            reasons.append(str(reason))

        confidence_weighted_target_probe = build_confidence_weighted_target_pull_probe(
            seed_points=seed_points,
            mls2_targets=mls2,
            sphere_targets=sphere_targets,
            confidence_profile=target_confidence_profile,
            context_edge_length_median=context_edge,
        )
        confidence_weighted_target_probe_dict = confidence_weighted_target_probe.to_dict()

        for reason in confidence_weighted_target_probe.reasons:
            reasons.append(str(reason))

        confidence_candidate_status = "not_evaluated"
        confidence_candidate_action = "not_evaluated"
        confidence_candidate_selected = False
        confidence_candidate_applied = False
        confidence_candidate_available = False
        confidence_candidate_topology_status = "not_evaluated"
        confidence_candidate_quality_status = "not_evaluated"
        confidence_candidate_g2_status = "not_evaluated"
        confidence_candidate_accepted_by_gates = False
        confidence_candidate_reasons: list[str] = []
        confidence_candidate_topology_report: dict[str, object] = {}
        confidence_candidate_quality_report: dict[str, object] = {}
        confidence_candidate_delta_vectors: tuple[tuple[float, float, float], ...] = ()
        confidence_candidate_target_points: tuple[tuple[float, float, float], ...] = ()
        confidence_candidate_g2_report: dict[str, object] = {}
        confidence_candidate_curvature_delta_mean: object = "-"
        confidence_candidate_curvature_delta_max: object = "-"
        confidence_candidate_curvature_relative_delta_mean: object = "-"
        confidence_candidate_movement_mean = float(confidence_weighted_target_probe.movement_mean)
        confidence_candidate_movement_max = float(confidence_weighted_target_probe.movement_max)
        confidence_candidate_movement_mean_ratio = float(confidence_weighted_target_probe.movement_mean_ratio)
        confidence_candidate_movement_max_ratio = float(confidence_weighted_target_probe.movement_max_ratio)

        try:
            new_face_ids = tuple(
                int(v)
                for v in seed.get("new_face_ids", ())
                if int(v) >= 0 and int(v) < int(len(seed_mesh.faces))
            )
            movable_vertex_ids = tuple(int(v) for v in new_vertex_ids)

            count = min(
                int(len(seed_points)),
                int(len(mls2)),
                int(len(sphere_targets)),
                int(len(target_confidence_profile.recommended_surface_weights)),
                int(len(movable_vertex_ids)),
            )

            if count <= 0 or not new_face_ids:
                confidence_candidate_status = "not_applicable"
                confidence_candidate_action = "no_candidate_mesh_evaluation"
                confidence_candidate_reasons.append("candidate mesh evaluation had no movable vertices or patch faces")
            else:
                seed_vertices = np.asarray(seed_mesh.vertices, dtype=float)
                candidate_vertices = np.array(seed_vertices, dtype=float, copy=True)
                weights = np.asarray(
                    target_confidence_profile.recommended_surface_weights[:count],
                    dtype=float,
                )
                blended_targets = 0.5 * np.asarray(mls2[:count], dtype=float) + 0.5 * np.asarray(
                    sphere_targets[:count],
                    dtype=float,
                )

                confidence_candidate_target_points = tuple(
                    tuple(float(component) for component in row)
                    for row in np.asarray(blended_targets, dtype=float)
                )
                candidate_points = np.asarray(seed_points[:count], dtype=float) + weights.reshape((-1, 1)) * (
                    blended_targets - np.asarray(seed_points[:count], dtype=float)
                )

                candidate_delta_vectors_np = candidate_points - np.asarray(seed_points[:count], dtype=float)
                confidence_candidate_delta_vectors = tuple(
                    tuple(float(component) for component in row)
                    for row in np.asarray(candidate_delta_vectors_np, dtype=float)
                )

                movable = np.asarray(movable_vertex_ids[:count], dtype=np.int64)
                candidate_vertices[movable] = candidate_points

                confidence_candidate_mesh = trimesh.Trimesh(
                    vertices=candidate_vertices,
                    faces=np.asarray(seed_mesh.faces, dtype=np.int64),
                    process=False,
                )
                confidence_candidate_available = True

                topology_report = validate_patch_topology(
                    confidence_candidate_mesh,
                    patch_face_ids=new_face_ids,
                    boundary_vertex_ids=boundary_ids,
                )
                confidence_candidate_topology_report = topology_report.to_dict()

                seam_coverage = float(
                    confidence_candidate_topology_report.get("seam_coverage_ratio") or 0.0
                )
                missing_seam_edges = int(
                    confidence_candidate_topology_report.get("missing_seam_edge_count") or 0
                )
                extra_open_edges = int(
                    confidence_candidate_topology_report.get("extra_open_boundary_edge_count") or 0
                )
                nonmanifold_edges = int(
                    confidence_candidate_topology_report.get("nonmanifold_patch_edge_count") or 0
                )

                if (
                    seam_coverage >= 0.999999
                    and missing_seam_edges == 0
                    and extra_open_edges == 0
                    and nonmanifold_edges == 0
                ):
                    confidence_candidate_topology_status = "passed"
                else:
                    confidence_candidate_topology_status = "blocked"
                    confidence_candidate_reasons.append(
                        "candidate topology gate blocked: "
                        f"coverage={seam_coverage:.6g}, missing={missing_seam_edges}, "
                        f"extra_open={extra_open_edges}, nonmanifold={nonmanifold_edges}"
                    )

                target_normals = compute_boundary_target_normals(base_mesh, boundary_ids)
                quality_report = analyze_patch_quality(
                    confidence_candidate_mesh,
                    patch_face_ids=new_face_ids,
                    boundary_vertex_ids=boundary_ids,
                    movable_vertex_ids=movable_vertex_ids[:count],
                    context_edge_length_median=context_edge,
                    target_boundary_normals=target_normals,
                )
                confidence_candidate_quality_report = quality_report.to_dict()

                degenerate = int(
                    confidence_candidate_quality_report.get("degenerate_face_count") or 0
                )
                min_angle = float(
                    confidence_candidate_quality_report.get("min_triangle_angle_degrees") or 0.0
                )
                median_aspect = float(
                    confidence_candidate_quality_report.get("median_triangle_aspect_ratio") or 0.0
                )
                max_aspect = float(
                    confidence_candidate_quality_report.get("max_triangle_aspect_ratio") or 0.0
                )

                if degenerate > 0:
                    confidence_candidate_quality_status = "blocked"
                    confidence_candidate_reasons.append(
                        f"candidate quality gate blocked: degenerate faces={degenerate}"
                    )
                elif min_angle < 1.0:
                    confidence_candidate_quality_status = "warning"
                    confidence_candidate_reasons.append(
                        f"candidate quality warning: min angle {min_angle:.6g}° < 1°"
                    )
                elif median_aspect > 25.0 or max_aspect > 100.0:
                    confidence_candidate_quality_status = "warning"
                    confidence_candidate_reasons.append(
                        "candidate quality warning: triangle aspect ratio is elevated"
                    )
                else:
                    confidence_candidate_quality_status = "passed"

                # H-CORE-R3C-B: lightweight curvature/G2 comparison.
                #
                # This intentionally stays local and comparison-only:
                # - compare the alternate candidate patch against the current seed patch
                # - do not select the candidate
                # - do not mutate committed geometry
                #
                # The estimator mirrors the existing normal-angle-over-edge-length
                # idea used by the adaptive smoke, but keeps this refined module
                # independent from hole_fill_preview.py to avoid circular imports.
                def _patch_curvature_summary(
                    mesh_obj: trimesh.Trimesh,
                    face_ids: tuple[int, ...],
                ) -> dict[str, object]:
                    vertices_local = np.asarray(mesh_obj.vertices, dtype=float)
                    faces_local = np.asarray(mesh_obj.faces, dtype=np.int64)
                    normals_local = np.asarray(mesh_obj.face_normals, dtype=float)

                    valid_faces = tuple(
                        int(face_id)
                        for face_id in face_ids
                        if int(face_id) >= 0 and int(face_id) < int(len(faces_local))
                    )

                    edge_to_faces: dict[tuple[int, int], list[int]] = {}
                    for face_id in valid_faces:
                        tri = [int(v) for v in faces_local[int(face_id), :3]]
                        for edge_index, a in enumerate(tri):
                            b = tri[(edge_index + 1) % 3]
                            if a == b:
                                continue
                            edge = (a, b) if a < b else (b, a)
                            edge_to_faces.setdefault(edge, []).append(int(face_id))

                    values: list[float] = []
                    for edge, owner_faces in edge_to_faces.items():
                        if len(owner_faces) < 2:
                            continue
                        a, b = edge
                        if (
                            a < 0
                            or b < 0
                            or a >= int(len(vertices_local))
                            or b >= int(len(vertices_local))
                        ):
                            continue

                        edge_length = float(np.linalg.norm(vertices_local[b] - vertices_local[a]))
                        if not np.isfinite(edge_length) or edge_length <= 1.0e-12:
                            continue

                        f0 = int(owner_faces[0])
                        f1 = int(owner_faces[1])
                        if f0 >= int(len(normals_local)) or f1 >= int(len(normals_local)):
                            continue

                        n0 = np.asarray(normals_local[f0], dtype=float)
                        n1 = np.asarray(normals_local[f1], dtype=float)
                        n0_norm = float(np.linalg.norm(n0))
                        n1_norm = float(np.linalg.norm(n1))
                        if n0_norm <= 1.0e-12 or n1_norm <= 1.0e-12:
                            continue

                        n0 = n0 / n0_norm
                        n1 = n1 / n1_norm
                        dot = float(np.clip(float(np.dot(n0, n1)), -1.0, 1.0))
                        angle = float(np.arccos(dot))
                        value = angle / edge_length
                        if np.isfinite(value):
                            values.append(float(value))

                    if values:
                        arr = np.asarray(values, dtype=float)
                        return {
                            "status": "ok",
                            "estimator": "normal_angle_over_edge_length_patch_only_v1",
                            "mean": float(np.mean(arr)),
                            "max": float(np.max(arr)),
                            "std": float(np.std(arr)),
                            "sample_count": int(arr.size),
                        }

                    return {
                        "status": "not_reported",
                        "estimator": "normal_angle_over_edge_length_patch_only_v1",
                        "mean": "-",
                        "max": "-",
                        "std": "-",
                        "sample_count": 0,
                    }

                baseline_curvature = _patch_curvature_summary(
                    seed_mesh,
                    new_face_ids,
                )
                candidate_curvature = _patch_curvature_summary(
                    confidence_candidate_mesh,
                    new_face_ids,
                )

                confidence_candidate_g2_report = {
                    "baseline": baseline_curvature,
                    "candidate": candidate_curvature,
                    "estimator": "normal_angle_over_edge_length_patch_only_v1",
                }

                baseline_mean = baseline_curvature.get("mean")
                baseline_max = baseline_curvature.get("max")
                candidate_mean = candidate_curvature.get("mean")
                candidate_max = candidate_curvature.get("max")

                if (
                    isinstance(baseline_mean, (int, float))
                    and isinstance(baseline_max, (int, float))
                    and isinstance(candidate_mean, (int, float))
                    and isinstance(candidate_max, (int, float))
                ):
                    confidence_candidate_curvature_delta_mean = float(
                        abs(float(candidate_mean) - float(baseline_mean))
                    )
                    confidence_candidate_curvature_delta_max = float(
                        abs(float(candidate_max) - float(baseline_max))
                    )
                    confidence_candidate_curvature_relative_delta_mean = float(
                        confidence_candidate_curvature_delta_mean
                        / max(abs(float(baseline_mean)), 1.0e-6)
                    )

                    # Because the candidate movement is intentionally tiny,
                    # R3C-B blocks only clear curvature worsening. This is a
                    # comparison gate for the future R3D selector, not a commit gate.
                    if (
                        confidence_candidate_curvature_relative_delta_mean <= 0.05
                        and confidence_candidate_curvature_delta_max <= 0.05
                    ):
                        confidence_candidate_g2_status = "passed"
                        confidence_candidate_reasons.append(
                            "candidate G2 comparison passed; patch-only curvature was preserved"
                        )
                    elif confidence_candidate_curvature_relative_delta_mean <= 0.15:
                        confidence_candidate_g2_status = "warning"
                        confidence_candidate_reasons.append(
                            "candidate G2 comparison warning: patch-only curvature changed slightly"
                        )
                    else:
                        confidence_candidate_g2_status = "blocked"
                        confidence_candidate_reasons.append(
                            "candidate G2 comparison blocked: patch-only curvature worsened"
                        )
                else:
                    confidence_candidate_curvature_delta_mean = "-"
                    confidence_candidate_curvature_delta_max = "-"
                    confidence_candidate_curvature_relative_delta_mean = "-"
                    confidence_candidate_g2_status = "not_reported"
                    confidence_candidate_reasons.append(
                        "candidate G2 comparison could not compute patch-only curvature samples"
                    )

                confidence_candidate_accepted_by_gates = bool(
                    confidence_weighted_target_probe.accepted_by_basic_gate
                    and confidence_candidate_topology_status == "passed"
                    and confidence_candidate_quality_status == "passed"
                    and confidence_candidate_g2_status in {"passed", "warning"}
                )

                confidence_candidate_status = "evaluated"
                confidence_candidate_action = (
                    "candidate_ready_for_selection_policy"
                    if confidence_candidate_accepted_by_gates
                    else "keep_probe_diagnostic_only"
                )

                confidence_candidate_reasons.append(
                    "H-CORE-R3C-A built an alternate confidence-weighted candidate mesh"
                )
                confidence_candidate_reasons.append(
                    "candidate moved generated seed vertices only; seam/boundary vertices remain fixed"
                )
                confidence_candidate_reasons.append(
                    "candidate is not selected; R3D may select only if policy allows it"
                )

        except Exception as candidate_exc:
            confidence_candidate_status = "error"
            confidence_candidate_action = "candidate_mesh_evaluation_failed"
            confidence_candidate_selected = False
            confidence_candidate_applied = False
            confidence_candidate_available = False
            confidence_candidate_accepted_by_gates = False
            confidence_candidate_topology_status = "error"
            confidence_candidate_quality_status = "error"
            confidence_candidate_g2_status = "not_evaluated"
            confidence_candidate_g2_report = {}
            confidence_candidate_curvature_delta_mean = "-"
            confidence_candidate_curvature_delta_max = "-"
            confidence_candidate_curvature_relative_delta_mean = "-"
            confidence_candidate_reasons.append(
                f"confidence-weighted candidate evaluation failed: {candidate_exc}"
            )
            confidence_candidate_delta_vectors = ()
            confidence_candidate_target_points = ()

        for reason in confidence_candidate_reasons:
            reasons.append(str(reason))

        return {
            "status": status,
            "action": action,
            "seed_backend": str(metadata.get("seed_backend") or "curvature_sphere_uvdelaunay"),
            "generated_vertex_count": int(len(new_vertex_ids)),
            "context_edge_length_median": float(context_edge),

            "target_disagreement_mean": float(target_disagreement_mean),
            "target_disagreement_max": float(target_disagreement_max),
            "target_disagreement_mean_ratio": float(target_disagreement_mean_ratio),
            "target_disagreement_max_ratio": float(target_disagreement_max_ratio),

            "target_disagreement_source": str(target_disagreement_source),
            "target_model_recommendation": str(target_model_recommendation),
            "target_model_confidence": float(target_model_confidence),
            "support_target_report": target_model_report,

            "target_confidence_profile": target_confidence_profile_dict,
            "target_confidence_model": str(target_confidence_profile.model),
            "target_confidence_min": float(target_confidence_profile.confidence_min),
            "target_confidence_mean": float(target_confidence_profile.confidence_mean),
            "target_confidence_median": float(target_confidence_profile.confidence_median),
            "target_confidence_max": float(target_confidence_profile.confidence_max),
            "target_confidence_low_count": int(target_confidence_profile.low_confidence_count),
            "target_confidence_low_threshold": float(target_confidence_profile.low_confidence_threshold),
            "target_confidence_vertex_count": int(target_confidence_profile.vertex_count),
            "target_recommended_surface_weight_min": float(target_confidence_profile.recommended_surface_weight_min),
            "target_recommended_surface_weight_mean": float(target_confidence_profile.recommended_surface_weight_mean),
            "target_recommended_surface_weight_max": float(target_confidence_profile.recommended_surface_weight_max),
            "target_recommended_surface_weight_limit": float(target_confidence_profile.max_surface_weight),
            "target_confidence_profile_reasons": tuple(target_confidence_profile.reasons),

            "confidence_weighted_target_probe": confidence_weighted_target_probe_dict,
            "confidence_target_probe_status": str(confidence_weighted_target_probe.status),
            "confidence_target_probe_action": str(confidence_weighted_target_probe.action),
            "confidence_target_probe_attempted": bool(confidence_weighted_target_probe.attempted),
            "confidence_target_probe_applied": bool(confidence_weighted_target_probe.applied),
            "confidence_target_probe_selected": bool(confidence_weighted_target_probe.selected),
            "confidence_target_probe_accepted_by_basic_gate": bool(confidence_weighted_target_probe.accepted_by_basic_gate),
            "confidence_target_probe_target_kind": str(confidence_weighted_target_probe.target_kind),
            "confidence_target_probe_vertex_count": int(confidence_weighted_target_probe.vertex_count),
            "confidence_target_probe_movement_mean": float(confidence_weighted_target_probe.movement_mean),
            "confidence_target_probe_movement_max": float(confidence_weighted_target_probe.movement_max),
            "confidence_target_probe_movement_mean_ratio": float(confidence_weighted_target_probe.movement_mean_ratio),
            "confidence_target_probe_movement_max_ratio": float(confidence_weighted_target_probe.movement_max_ratio),
            "confidence_target_probe_confidence_mean": float(confidence_weighted_target_probe.confidence_mean),
            "confidence_target_probe_recommended_surface_weight_mean": float(confidence_weighted_target_probe.recommended_surface_weight_mean),
            "confidence_target_probe_basic_gate_status": str(confidence_weighted_target_probe.basic_gate_status),
            "confidence_target_probe_basic_gate_reasons": tuple(confidence_weighted_target_probe.basic_gate_reasons),
            "confidence_target_probe_quality_gate_status": str(confidence_weighted_target_probe.quality_gate_status),
            "confidence_target_probe_g2_gate_status": str(confidence_weighted_target_probe.g2_gate_status),
            "confidence_target_probe_reasons": tuple(confidence_weighted_target_probe.reasons),

            "confidence_target_candidate_status": str(confidence_candidate_status),
            "confidence_target_candidate_action": str(confidence_candidate_action),
            "confidence_target_candidate_available": bool(confidence_candidate_available),
            "confidence_target_candidate_applied": bool(confidence_candidate_applied),
            "confidence_target_candidate_selected": bool(confidence_candidate_selected),
            "confidence_target_candidate_accepted_by_gates": bool(confidence_candidate_accepted_by_gates),
            "confidence_target_candidate_topology_status": str(confidence_candidate_topology_status),
            "confidence_target_candidate_quality_status": str(confidence_candidate_quality_status),
            "confidence_target_candidate_g2_status": str(confidence_candidate_g2_status),
            "confidence_target_candidate_movement_mean": float(confidence_candidate_movement_mean),
            "confidence_target_candidate_movement_max": float(confidence_candidate_movement_max),
            "confidence_target_candidate_movement_mean_ratio": float(confidence_candidate_movement_mean_ratio),
            "confidence_target_candidate_movement_max_ratio": float(confidence_candidate_movement_max_ratio),
            "confidence_target_candidate_delta_vectors": confidence_candidate_delta_vectors,
            "confidence_target_candidate_target_points": confidence_candidate_target_points,
            "confidence_target_candidate_topology_report": confidence_candidate_topology_report,
            "confidence_target_candidate_quality_report": confidence_candidate_quality_report,
            "confidence_target_candidate_g2_report": confidence_candidate_g2_report,
            "confidence_target_candidate_curvature_delta_mean": confidence_candidate_curvature_delta_mean,
            "confidence_target_candidate_curvature_delta_max": confidence_candidate_curvature_delta_max,
            "confidence_target_candidate_curvature_relative_delta_mean": confidence_candidate_curvature_relative_delta_mean,
            "confidence_target_candidate_reasons": tuple(dict.fromkeys(str(reason) for reason in confidence_candidate_reasons if str(reason))),

            "mls2_vs_sphere_mean": float(mls2_vs_sphere_mean),
            "mls2_vs_sphere_max": float(mls2_vs_sphere_max),
            "mls2_vs_plane_mean": float(mls2_vs_plane_mean),
            "mls2_vs_plane_max": float(mls2_vs_plane_max),
            "mls1_vs_mls2_mean": float(mls1_vs_mls2_mean),
            "mls1_vs_mls2_max": float(mls1_vs_mls2_max),
            "mls2_vs_mls3_mean": float(mls2_vs_mls3_mean),
            "mls2_vs_mls3_max": float(mls2_vs_mls3_max),

            "mls2_vs_sphere_mean_ratio": float(mls2_vs_sphere_mean_ratio),
            "mls2_vs_sphere_max_ratio": float(mls2_vs_sphere_max_ratio),
            "mls2_vs_plane_mean_ratio": float(mls2_vs_plane_mean_ratio),
            "mls2_vs_plane_max_ratio": float(mls2_vs_plane_max_ratio),
            "mls1_vs_mls2_mean_ratio": float(mls1_vs_mls2_mean_ratio),
            "mls1_vs_mls2_max_ratio": float(mls1_vs_mls2_max_ratio),
            "mls2_vs_mls3_mean_ratio": float(mls2_vs_mls3_mean_ratio),
            "mls2_vs_mls3_max_ratio": float(mls2_vs_mls3_max_ratio),

            "target_signed_disagreement_mean": float(signed_mean),
            "target_signed_disagreement_min": float(signed_min),
            "target_signed_disagreement_max": float(signed_max),

            "target_signed_plane_disagreement_mean": float(signed_plane_mean),
            "target_signed_plane_disagreement_min": float(signed_plane_min),
            "target_signed_plane_disagreement_max": float(signed_plane_max),

            "mls_ring1_support_vertex_count": int(mls1_meta["support_vertex_count"]),
            "mls_ring2_support_vertex_count": int(mls2_meta["support_vertex_count"]),
            "mls_ring3_support_vertex_count": int(mls3_meta["support_vertex_count"]),
            "mls_ring1_normal_spread_degrees": float(mls1_meta["normal_spread_degrees"]),
            "mls_ring2_normal_spread_degrees": float(mls2_meta["normal_spread_degrees"]),
            "mls_ring3_normal_spread_degrees": float(mls3_meta["normal_spread_degrees"]),

            "reasons": tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        }
    except Exception as exc:
        return {
            "status": "error",
            "action": "support_target_disagreement_diagnostic_failed",
            "seed_backend": str(metadata.get("seed_backend") or "curvature_sphere_uvdelaunay"),
            "generated_vertex_count": "-",
            "context_edge_length_median": "-",
            "target_disagreement_mean": "-",
            "target_disagreement_max": "-",
            "target_disagreement_mean_ratio": "-",
            "target_disagreement_max_ratio": "-",
            "target_disagreement_source": "-",
            "target_model_recommendation": "-",
            "target_model_confidence": "-",
            "target_confidence_profile": {},
            "target_confidence_model": "-",
            "target_confidence_min": "-",
            "target_confidence_mean": "-",
            "target_confidence_median": "-",
            "target_confidence_max": "-",
            "target_confidence_low_count": "-",
            "target_confidence_low_threshold": "-",
            "target_confidence_vertex_count": "-",
            "target_recommended_surface_weight_min": "-",
            "target_recommended_surface_weight_mean": "-",
            "target_recommended_surface_weight_max": "-",
            "target_recommended_surface_weight_limit": "-",
            "target_confidence_profile_reasons": (),
            "confidence_weighted_target_probe": {},
            "confidence_target_probe_status": "-",
            "confidence_target_probe_action": "-",
            "confidence_target_probe_attempted": "-",
            "confidence_target_probe_applied": "-",
            "confidence_target_probe_selected": "-",
            "confidence_target_probe_accepted_by_basic_gate": "-",
            "confidence_target_probe_target_kind": "-",
            "confidence_target_probe_vertex_count": "-",
            "confidence_target_probe_movement_mean": "-",
            "confidence_target_probe_movement_max": "-",
            "confidence_target_probe_movement_mean_ratio": "-",
            "confidence_target_probe_movement_max_ratio": "-",
            "confidence_target_probe_confidence_mean": "-",
            "confidence_target_probe_recommended_surface_weight_mean": "-",
            "confidence_target_probe_basic_gate_status": "-",
            "confidence_target_probe_basic_gate_reasons": (),
            "confidence_target_probe_quality_gate_status": "-",
            "confidence_target_probe_g2_gate_status": "-",
            "confidence_target_probe_reasons": (),
            "confidence_target_candidate_status": "-",
            "confidence_target_candidate_action": "-",
            "confidence_target_candidate_available": "-",
            "confidence_target_candidate_applied": "-",
            "confidence_target_candidate_selected": "-",
            "confidence_target_candidate_accepted_by_gates": "-",
            "confidence_target_candidate_topology_status": "-",
            "confidence_target_candidate_quality_status": "-",
            "confidence_target_candidate_g2_status": "-",
            "confidence_target_candidate_movement_mean": "-",
            "confidence_target_candidate_movement_max": "-",
            "confidence_target_candidate_movement_mean_ratio": "-",
            "confidence_target_candidate_movement_max_ratio": "-",
            "confidence_target_candidate_delta_vectors": (),
            "confidence_target_candidate_target_points": (),
            "confidence_target_candidate_topology_report": {},
            "confidence_target_candidate_quality_report": {},
            "confidence_target_candidate_g2_report": {},
            "confidence_target_candidate_curvature_delta_mean": "-",
            "confidence_target_candidate_curvature_delta_max": "-",
            "confidence_target_candidate_curvature_relative_delta_mean": "-",
            "confidence_target_candidate_reasons": (),
            "mls2_vs_sphere_mean": "-",
            "mls2_vs_sphere_max": "-",
            "mls2_vs_plane_mean": "-",
            "mls2_vs_plane_max": "-",
            "mls1_vs_mls2_mean": "-",
            "mls1_vs_mls2_max": "-",
            "mls2_vs_mls3_mean": "-",
            "mls2_vs_mls3_max": "-",
            "mls2_vs_sphere_mean_ratio": "-",
            "mls2_vs_sphere_max_ratio": "-",
            "mls2_vs_plane_mean_ratio": "-",
            "mls2_vs_plane_max_ratio": "-",
            "mls1_vs_mls2_mean_ratio": "-",
            "mls1_vs_mls2_max_ratio": "-",
            "mls2_vs_mls3_mean_ratio": "-",
            "mls2_vs_mls3_max_ratio": "-",
            "target_signed_disagreement_mean": "-",
            "target_signed_disagreement_min": "-",
            "target_signed_disagreement_max": "-",
            "target_signed_plane_disagreement_mean": "-",
            "target_signed_plane_disagreement_min": "-",
            "target_signed_plane_disagreement_max": "-",
            "mls_ring1_support_vertex_count": "-",
            "mls_ring2_support_vertex_count": "-",
            "mls_ring3_support_vertex_count": "-",
            "mls_ring1_normal_spread_degrees": "-",
            "mls_ring2_normal_spread_degrees": "-",
            "mls_ring3_normal_spread_degrees": "-",
            "reasons": (f"support target disagreement diagnostic failed: {exc}",),
        }

