from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TargetModelComparison:
    """Distance comparison between two candidate support target models.

    Diagnostic-only value object:
    - no mesh mutation
    - no preview selection
    - safe to serialize through metadata
    """

    name: str
    mean_distance: float
    max_distance: float
    mean_ratio: float
    max_ratio: float

    @property
    def dominant_ratio(self) -> float:
        return float(max(self.mean_ratio, self.max_ratio))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TargetModelRecommendation:
    """Interpretation of support target model disagreement.

    The recommendation tells later solver stages how to treat the target
    family. It does not apply geometry and must not be used as a commit gate.
    """

    disagreement_source: str
    recommendation: str
    confidence: float
    ring_stability_ratio: float
    mls_sphere_ratio: float
    plane_ratio: float
    normal_spread_degrees: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportTargetReport:
    """Portable report for the refined support-target layer.

    This is the R2 diagnostic/report shape. R3 can consume this without having
    to know where the raw MLS/sphere/plane samples came from.
    """

    comparisons: tuple[TargetModelComparison, ...]
    recommendation: TargetModelRecommendation

    def to_dict(self) -> dict[str, object]:
        return {
            "comparisons": tuple(item.to_dict() for item in self.comparisons),
            "recommendation": self.recommendation.to_dict(),
        }


@dataclass(frozen=True)
class TargetConfidenceProfile:
    """Per-sample confidence profile for the support target family.

    R3A is diagnostic-only:
    - reports local confidence and suggested local surface weights
    - does not move patch vertices
    - does not select a different preview
    """

    model: str
    vertex_count: int
    confidence_min: float
    confidence_mean: float
    confidence_median: float
    confidence_max: float
    low_confidence_count: int
    low_confidence_threshold: float
    recommended_surface_weight_min: float
    recommended_surface_weight_mean: float
    recommended_surface_weight_max: float
    max_surface_weight: float
    confidence_values: tuple[float, ...]
    recommended_surface_weights: tuple[float, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not isfinite(number):
        return float(default)
    return float(number)


def _as_point_array(value: object) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.zeros((0, 3), dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)

    return arr[np.isfinite(arr).all(axis=1)]


def _distance_ratio_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    context_edge_length_median: float,
) -> np.ndarray:
    count = min(int(len(left)), int(len(right)))
    if count <= 0:
        return np.zeros(0, dtype=float)

    distances = np.linalg.norm(left[:count] - right[:count], axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return np.zeros(0, dtype=float)

    context_edge = max(float(context_edge_length_median), 1.0e-12)
    return distances / context_edge


def build_target_model_comparison(
    *,
    name: str,
    mean_distance: object,
    max_distance: object,
    context_edge_length_median: object,
) -> TargetModelComparison:
    """Build one normalized target-model comparison."""

    context_edge = max(_safe_float(context_edge_length_median, 1.0), 1.0e-12)
    mean = _safe_float(mean_distance, 0.0)
    maximum = _safe_float(max_distance, 0.0)

    return TargetModelComparison(
        name=str(name),
        mean_distance=float(mean),
        max_distance=float(maximum),
        mean_ratio=float(mean / context_edge),
        max_ratio=float(maximum / context_edge),
    )


def dominant_target_disagreement_source(
    comparisons: Mapping[str, TargetModelComparison],
) -> str:
    """Return the comparison name with the largest normalized disagreement."""

    if not comparisons:
        return "unknown"
    return str(
        max(
            comparisons,
            key=lambda key: float(comparisons[key].dominant_ratio),
        )
    )


def recommend_target_model_from_comparisons(
    *,
    comparisons: Mapping[str, TargetModelComparison],
    normal_spread_degrees: object,
    ring_stability_threshold: float = 0.05,
    mls_sphere_threshold: float = 0.05,
    plane_outlier_threshold: float = 0.18,
) -> TargetModelRecommendation:
    """Classify target disagreement and produce a confidence recommendation.

    Important policy:
    - PCA plane disagreement is evidence of non-planar/high-spread context.
    - Plane disagreement should reduce confidence.
    - Plane disagreement must not automatically make the plane the target.
    """

    source = dominant_target_disagreement_source(comparisons)
    normal_spread = _safe_float(normal_spread_degrees, 0.0)

    mls1_vs_mls2 = comparisons.get("mls1_vs_mls2")
    mls2_vs_mls3 = comparisons.get("mls2_vs_mls3")
    mls2_vs_sphere = comparisons.get("mls2_vs_sphere")
    mls2_vs_plane = comparisons.get("mls2_vs_plane")

    ring_stability_ratio = max(
        float(mls1_vs_mls2.dominant_ratio) if mls1_vs_mls2 is not None else 0.0,
        float(mls2_vs_mls3.dominant_ratio) if mls2_vs_mls3 is not None else 0.0,
    )
    mls_sphere_ratio = (
        float(mls2_vs_sphere.dominant_ratio)
        if mls2_vs_sphere is not None
        else 0.0
    )
    plane_ratio = (
        float(mls2_vs_plane.dominant_ratio)
        if mls2_vs_plane is not None
        else 0.0
    )

    reasons: list[str] = []

    if (
        source == "mls2_vs_plane"
        and ring_stability_ratio <= float(ring_stability_threshold)
        and mls_sphere_ratio <= float(mls_sphere_threshold)
    ):
        recommendation = "mls_sphere_confidence_blend"
        reasons.append(
            "PCA plane is the dominant target outlier while MLS rings and sphere target agree; "
            "plane disagreement should lower target confidence, not flatten the patch"
        )
    elif source in {"mls1_vs_mls2", "mls2_vs_mls3"}:
        recommendation = "ring_adaptive_mls_target"
        reasons.append("MLS ring target disagreement dominates; support ring selection should become adaptive")
    elif source == "mls2_vs_sphere":
        recommendation = "curvature_aware_mls_sphere_blend"
        reasons.append("MLS and sphere target disagreement dominates; curvature-aware blending should be evaluated")
    elif source == "mls2_vs_plane":
        recommendation = "mls_sphere_confidence_blend"
        reasons.append("PCA plane disagreement dominates; use it as a confidence reducer, not as a flattening target")
    else:
        recommendation = "current_target_with_confidence_guard"
        reasons.append("target model disagreement source is weak or unknown; retain guarded current target behavior")

    confidence_penalty = max(
        0.0,
        min(
            1.0,
            0.45 * min(plane_ratio / max(float(plane_outlier_threshold), 1.0e-12), 1.0)
            + 0.35 * min(max(normal_spread - 45.0, 0.0) / 45.0, 1.0)
            + 0.20 * min(ring_stability_ratio / max(float(ring_stability_threshold), 1.0e-12), 1.0),
        ),
    )
    confidence = float(max(0.0, min(1.0, 1.0 - confidence_penalty)))

    return TargetModelRecommendation(
        disagreement_source=str(source),
        recommendation=str(recommendation),
        confidence=float(confidence),
        ring_stability_ratio=float(ring_stability_ratio),
        mls_sphere_ratio=float(mls_sphere_ratio),
        plane_ratio=float(plane_ratio),
        normal_spread_degrees=float(normal_spread),
        reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
    )


def build_support_target_report(
    *,
    mls2_vs_sphere_mean: object,
    mls2_vs_sphere_max: object,
    mls2_vs_plane_mean: object,
    mls2_vs_plane_max: object,
    mls1_vs_mls2_mean: object,
    mls1_vs_mls2_max: object,
    mls2_vs_mls3_mean: object,
    mls2_vs_mls3_max: object,
    context_edge_length_median: object,
    normal_spread_degrees: object,
) -> SupportTargetReport:
    """Build the R2 target-family diagnostic report from raw distances."""

    comparisons = {
        "mls2_vs_sphere": build_target_model_comparison(
            name="mls2_vs_sphere",
            mean_distance=mls2_vs_sphere_mean,
            max_distance=mls2_vs_sphere_max,
            context_edge_length_median=context_edge_length_median,
        ),
        "mls2_vs_plane": build_target_model_comparison(
            name="mls2_vs_plane",
            mean_distance=mls2_vs_plane_mean,
            max_distance=mls2_vs_plane_max,
            context_edge_length_median=context_edge_length_median,
        ),
        "mls1_vs_mls2": build_target_model_comparison(
            name="mls1_vs_mls2",
            mean_distance=mls1_vs_mls2_mean,
            max_distance=mls1_vs_mls2_max,
            context_edge_length_median=context_edge_length_median,
        ),
        "mls2_vs_mls3": build_target_model_comparison(
            name="mls2_vs_mls3",
            mean_distance=mls2_vs_mls3_mean,
            max_distance=mls2_vs_mls3_max,
            context_edge_length_median=context_edge_length_median,
        ),
    }

    recommendation = recommend_target_model_from_comparisons(
        comparisons=comparisons,
        normal_spread_degrees=normal_spread_degrees,
    )

    return SupportTargetReport(
        comparisons=tuple(
            comparisons[key]
            for key in (
                "mls2_vs_sphere",
                "mls2_vs_plane",
                "mls1_vs_mls2",
                "mls2_vs_mls3",
            )
        ),
        recommendation=recommendation,
    )


def build_target_confidence_profile(
    *,
    mls2_targets: object,
    sphere_targets: object,
    plane_targets: object,
    mls1_targets: object,
    mls3_targets: object,
    context_edge_length_median: object,
    normal_spread_degrees: object,
    max_surface_weight: float = 0.25,
    low_confidence_threshold: float = 0.35,
    plane_outlier_threshold: float = 0.18,
    ring_stability_threshold: float = 0.05,
) -> TargetConfidenceProfile:
    """Build per-generated-vertex target confidence diagnostics.

    Confidence policy:
    - high MLS2-vs-plane disagreement lowers confidence
    - high normal spread lowers confidence globally
    - MLS ring instability lowers confidence locally
    - this is only a recommendation profile; it does not move geometry
    """

    mls2 = _as_point_array(mls2_targets)
    sphere = _as_point_array(sphere_targets)
    plane = _as_point_array(plane_targets)
    mls1 = _as_point_array(mls1_targets)
    mls3 = _as_point_array(mls3_targets)

    count = min(int(len(mls2)), int(len(sphere)), int(len(plane)), int(len(mls1)), int(len(mls3)))
    context_edge = max(_safe_float(context_edge_length_median, 1.0), 1.0e-12)
    max_weight = max(0.0, _safe_float(max_surface_weight, 0.25))

    if count <= 0:
        return TargetConfidenceProfile(
            model="mls_sphere_plane_ring_confidence_v1",
            vertex_count=0,
            confidence_min=0.0,
            confidence_mean=0.0,
            confidence_median=0.0,
            confidence_max=0.0,
            low_confidence_count=0,
            low_confidence_threshold=float(low_confidence_threshold),
            recommended_surface_weight_min=0.0,
            recommended_surface_weight_mean=0.0,
            recommended_surface_weight_max=0.0,
            max_surface_weight=float(max_weight),
            confidence_values=(),
            recommended_surface_weights=(),
            reasons=("no target samples available for confidence profile",),
        )

    mls2 = mls2[:count]
    sphere = sphere[:count]
    plane = plane[:count]
    mls1 = mls1[:count]
    mls3 = mls3[:count]

    plane_ratio = _distance_ratio_array(
        plane,
        mls2,
        context_edge_length_median=context_edge,
    )
    sphere_ratio = _distance_ratio_array(
        sphere,
        mls2,
        context_edge_length_median=context_edge,
    )
    ring1_ratio = _distance_ratio_array(
        mls1,
        mls2,
        context_edge_length_median=context_edge,
    )
    ring3_ratio = _distance_ratio_array(
        mls3,
        mls2,
        context_edge_length_median=context_edge,
    )

    local_count = min(
        int(len(plane_ratio)),
        int(len(sphere_ratio)),
        int(len(ring1_ratio)),
        int(len(ring3_ratio)),
    )

    if local_count <= 0:
        return TargetConfidenceProfile(
            model="mls_sphere_plane_ring_confidence_v1",
            vertex_count=0,
            confidence_min=0.0,
            confidence_mean=0.0,
            confidence_median=0.0,
            confidence_max=0.0,
            low_confidence_count=0,
            low_confidence_threshold=float(low_confidence_threshold),
            recommended_surface_weight_min=0.0,
            recommended_surface_weight_mean=0.0,
            recommended_surface_weight_max=0.0,
            max_surface_weight=float(max_weight),
            confidence_values=(),
            recommended_surface_weights=(),
            reasons=("target sample ratios could not be computed",),
        )

    plane_ratio = plane_ratio[:local_count]
    sphere_ratio = sphere_ratio[:local_count]
    ring_ratio = np.maximum(ring1_ratio[:local_count], ring3_ratio[:local_count])

    normal_spread = _safe_float(normal_spread_degrees, 0.0)
    normal_spread_penalty = max(0.0, min(1.0, max(normal_spread - 45.0, 0.0) / 45.0))

    penalty = (
        0.45 * np.clip(plane_ratio / max(float(plane_outlier_threshold), 1.0e-12), 0.0, 1.0)
        + 0.35 * normal_spread_penalty
        + 0.20 * np.clip(ring_ratio / max(float(ring_stability_threshold), 1.0e-12), 0.0, 1.0)
    )

    confidence = np.clip(1.0 - penalty, 0.0, 1.0)
    surface_weights = np.clip(confidence * max_weight, 0.0, max_weight)

    low_threshold = float(low_confidence_threshold)
    low_count = int(np.count_nonzero(confidence < low_threshold))

    reasons: list[str] = []
    if float(np.max(plane_ratio)) > float(plane_outlier_threshold):
        reasons.append("PCA plane disagreement lowers local target confidence")
    if normal_spread >= 45.0:
        reasons.append(f"support normal spread lowers global target confidence ({normal_spread:.6g}°)")
    if float(np.max(ring_ratio)) > float(ring_stability_threshold):
        reasons.append("MLS ring instability lowers local target confidence")
    if not reasons:
        reasons.append("target confidence profile is stable within diagnostic thresholds")

    return TargetConfidenceProfile(
        model="mls_sphere_plane_ring_confidence_v1",
        vertex_count=int(local_count),
        confidence_min=float(np.min(confidence)),
        confidence_mean=float(np.mean(confidence)),
        confidence_median=float(np.median(confidence)),
        confidence_max=float(np.max(confidence)),
        low_confidence_count=int(low_count),
        low_confidence_threshold=float(low_threshold),
        recommended_surface_weight_min=float(np.min(surface_weights)),
        recommended_surface_weight_mean=float(np.mean(surface_weights)),
        recommended_surface_weight_max=float(np.max(surface_weights)),
        max_surface_weight=float(max_weight),
        confidence_values=tuple(float(v) for v in confidence),
        recommended_surface_weights=tuple(float(v) for v in surface_weights),
        reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
    )


@dataclass(frozen=True)
class ConfidenceWeightedTargetPullProbe:
    """Diagnostic-only candidate for confidence-weighted target pull.

    H-CORE-R3B policy:
    - compute candidate displacement
    - report movement and basic safety gate
    - do not apply candidate geometry
    - do not select candidate as preview output
    """

    status: str
    action: str
    attempted: bool
    applied: bool
    selected: bool
    accepted_by_basic_gate: bool
    target_kind: str
    vertex_count: int
    movement_mean: float
    movement_max: float
    movement_mean_ratio: float
    movement_max_ratio: float
    confidence_mean: float
    recommended_surface_weight_mean: float
    basic_gate_status: str
    basic_gate_reasons: tuple[str, ...]
    quality_gate_status: str
    g2_gate_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_confidence_weighted_target_pull_probe(
    *,
    seed_points: object,
    mls2_targets: object,
    sphere_targets: object,
    confidence_profile: TargetConfidenceProfile,
    context_edge_length_median: object,
    sphere_blend_weight: float = 0.50,
    max_movement_ratio: float = 0.06,
    min_confidence_mean: float = 0.20,
) -> ConfidenceWeightedTargetPullProbe:
    """Build a diagnostic confidence-weighted MLS/sphere pull candidate.

    This function intentionally does not return moved mesh geometry as a
    selected result. It only describes the candidate movement that R3C may
    later evaluate with full seam/quality/G2 gates.
    """

    seed = _as_point_array(seed_points)
    mls2 = _as_point_array(mls2_targets)
    sphere = _as_point_array(sphere_targets)

    count = min(
        int(len(seed)),
        int(len(mls2)),
        int(len(sphere)),
        int(len(confidence_profile.recommended_surface_weights)),
    )
    context_edge = max(_safe_float(context_edge_length_median, 1.0), 1.0e-12)

    if count <= 0:
        return ConfidenceWeightedTargetPullProbe(
            status="not_applicable",
            action="no_target_pull_probe",
            attempted=False,
            applied=False,
            selected=False,
            accepted_by_basic_gate=False,
            target_kind="mls_sphere_confidence_blend",
            vertex_count=0,
            movement_mean=0.0,
            movement_max=0.0,
            movement_mean_ratio=0.0,
            movement_max_ratio=0.0,
            confidence_mean=0.0,
            recommended_surface_weight_mean=0.0,
            basic_gate_status="not_applicable",
            basic_gate_reasons=("no target samples available for probe",),
            quality_gate_status="not_evaluated",
            g2_gate_status="not_evaluated",
            reasons=("confidence-weighted target pull probe had no samples",),
        )

    seed = seed[:count]
    mls2 = mls2[:count]
    sphere = sphere[:count]
    weights = np.asarray(confidence_profile.recommended_surface_weights[:count], dtype=float)

    blend_weight = max(0.0, min(1.0, _safe_float(sphere_blend_weight, 0.50)))
    blended_target = (1.0 - blend_weight) * mls2 + blend_weight * sphere

    displacement = weights.reshape((-1, 1)) * (blended_target - seed)
    movement = np.linalg.norm(displacement, axis=1)
    movement = movement[np.isfinite(movement)]

    if movement.size:
        movement_mean = float(np.mean(movement))
        movement_max = float(np.max(movement))
    else:
        movement_mean = 0.0
        movement_max = 0.0

    movement_mean_ratio = float(movement_mean / context_edge)
    movement_max_ratio = float(movement_max / context_edge)
    confidence_mean = float(confidence_profile.confidence_mean)
    recommended_weight_mean = float(confidence_profile.recommended_surface_weight_mean)

    gate_reasons: list[str] = []
    accepted = True

    if movement_max_ratio > float(max_movement_ratio):
        accepted = False
        gate_reasons.append(
            f"probe movement max ratio {movement_max_ratio:.6g} exceeds limit {float(max_movement_ratio):.6g}"
        )

    if confidence_mean < float(min_confidence_mean):
        accepted = False
        gate_reasons.append(
            f"probe confidence mean {confidence_mean:.6g} below minimum {float(min_confidence_mean):.6g}"
        )

    if accepted:
        basic_gate_status = "passed"
        gate_reasons.append("diagnostic target-pull probe satisfies basic movement/confidence gate")
    else:
        basic_gate_status = "blocked"

    reasons = [
        "H-CORE-R3B is diagnostic-only; confidence-weighted target pull was not selected",
        "probe uses MLS/sphere blend and per-vertex recommended surface weights",
        "full seam/quality/G2 candidate evaluation is deferred to R3C",
    ]

    return ConfidenceWeightedTargetPullProbe(
        status="probe_ready",
        action="evaluate_confidence_weighted_surface_pull_candidate",
        attempted=True,
        applied=False,
        selected=False,
        accepted_by_basic_gate=bool(accepted),
        target_kind="mls_sphere_confidence_blend",
        vertex_count=int(count),
        movement_mean=float(movement_mean),
        movement_max=float(movement_max),
        movement_mean_ratio=float(movement_mean_ratio),
        movement_max_ratio=float(movement_max_ratio),
        confidence_mean=float(confidence_mean),
        recommended_surface_weight_mean=float(recommended_weight_mean),
        basic_gate_status=str(basic_gate_status),
        basic_gate_reasons=tuple(dict.fromkeys(reason for reason in gate_reasons if reason)),
        quality_gate_status="not_evaluated",
        g2_gate_status="not_evaluated",
        reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
    )


__all__ = (
    "SupportTargetReport",
    "TargetConfidenceProfile",    "ConfidenceWeightedTargetPullProbe",

    "TargetModelComparison",
    "TargetModelRecommendation",
    "build_support_target_report",
    "build_target_confidence_profile",
    "build_confidence_weighted_target_pull_probe",
    "build_target_model_comparison",
    "dominant_target_disagreement_source",
    "recommend_target_model_from_comparisons",
)
