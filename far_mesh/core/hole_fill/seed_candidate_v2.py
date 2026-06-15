from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping
import re


@dataclass(frozen=True)
class AdaptiveSurfaceFillV2SeedCandidateBlueprint:
    """Adaptive Surface Fill v2 seed-candidate blueprint.

    This is the first v2 candidate-construction layer.

    It still does not create or select mesh geometry.  It defines what the
    geometry builder must construct next:

    surrounding mesh curvature
    -> normal-continuity field
    -> confidence-filtered support context
    -> seed candidate family
    -> later topology/quality/G1/G2 gates
    """

    status: str
    action: str
    candidate_family: str
    geometry_status: str
    selectable: bool

    curvature_normal_field_status: str
    support_filter: str
    frame_policy: str
    target_policy: str
    density_policy: str
    acceptance_policy: str

    boundary_vertices: int
    legacy_seed_vertices: int
    planned_seed_vertices: int
    planned_support_rings: int
    planned_interior_rings: int

    normal_continuity_mismatch_score: float
    target_confidence: float
    target_low_confidence_fraction: float
    seed_projection_max_ratio: float
    curvature_relative_delta_mean: float
    curvature_sign_consistency: bool

    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "-"}:
        return True
    return False


def _tokens(key: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.sub(r"[^a-zA-Z0-9]+", "_", str(key).lower()).split("_")
        if token and token not in {"adaptive", "degrees", "degree", "deviation"}
    )


def _first_value(
    metadata: Mapping[str, object],
    *keys: str,
    semantic: set[str] | None = None,
    default: object = None,
) -> object:
    for key in keys:
        if key in metadata and not _missing(metadata.get(key)):
            return metadata.get(key)

    items = tuple(metadata.items())

    for key in keys:
        requested = set(_tokens(key))
        if not requested:
            continue
        for actual_key, actual_value in items:
            if _missing(actual_value):
                continue
            actual = set(_tokens(str(actual_key)))
            if requested.issubset(actual):
                return actual_value

    if semantic:
        for actual_key, actual_value in items:
            if _missing(actual_value):
                continue
            actual = set(_tokens(str(actual_key)))
            if semantic.issubset(actual):
                return actual_value

    return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not isfinite(number):
        return float(default)
    return float(number)


def _int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "ok", "passed", "ready"}:
            return True
        if normalized in {"false", "no", "0", "blocked", "warning", "not_selected"}:
            return False
    return bool(default)


def _str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def build_adaptive_surface_fill_v2_seed_candidate_blueprint(
    metadata: Mapping[str, object],
) -> AdaptiveSurfaceFillV2SeedCandidateBlueprint:
    reasons: list[str] = []

    v2_case = _str(
        _first_value(
            metadata,
            "adaptive_surface_v2_case",
            semantic={"surface", "v2", "case"},
            default="",
        )
    ).strip()

    require_new_seed = _bool(
        _first_value(
            metadata,
            "adaptive_surface_v2_require_new_seed",
            semantic={"surface", "v2", "require", "new", "seed"},
            default=False,
        ),
        False,
    )


    direct_requested = bool(
        v2_case == "direct_v2_candidate_requested"
        or _bool(
            _first_value(
                metadata,
                "adaptive_surface_v2_direct_requested",
                "adaptive_surface_v2_force_candidate",
                "force_adaptive_surface_v2_candidate",
                semantic={"surface", "v2", "direct"},
                default=False,
            ),
            False,
        )
    )

    prototype_family = _str(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_family",
            "adaptive_surface_v2_seed_family",
            semantic={"surface", "v2", "seed", "family"},
            default="unknown",
        ),
        "unknown",
    )

    prototype_orientation = _str(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_orientation_status",
            semantic={"surface", "v2", "seed", "prototype", "orientation"},
            default="unknown",
        ),
        "unknown",
    )

    mismatch_score = _float(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_side_score_max",
            "adaptive_surface_v2_seed_prototype_normal_continuity_mismatch_score_max",
            semantic={"surface", "v2", "seed", "prototype", "score", "max"},
            default=0.0,
        )
    )

    boundary_vertices = _int(
        _first_value(
            metadata,
            "adaptive_feature_boundary_vertex_count",
            "adaptive_feature_boundary_vertices",
            "feature_boundary_vertices",
            semantic={"feature", "boundary", "vertices"},
            default=0,
        )
    )

    legacy_seed_vertices = _int(
        _first_value(
            metadata,
            "adaptive_seed_generated_vertex_count",
            "seed_generated_vertex_count",
            semantic={"seed", "generated", "vertex", "count"},
            default=0,
        )
    )

    target_confidence = _float(
        _first_value(
            metadata,
            "adaptive_target_model_confidence",
            "adaptive_surface_v2_seed_prototype_target_confidence",
            semantic={"target", "model", "confidence"},
            default=1.0,
        ),
        1.0,
    )

    low_fraction = _float(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_target_low_confidence_fraction",
            "adaptive_surface_v2_target_low_confidence_fraction",
            semantic={"target", "low", "confidence", "fraction"},
            default=0.0,
        )
    )

    seed_projection_max_ratio = _float(
        _first_value(
            metadata,
            "adaptive_seed_projection_max_ratio",
            "adaptive_surface_v2_seed_projection_max_ratio",
            semantic={"seed", "projection", "max", "ratio"},
            default=0.0,
        )
    )

    curvature_relative = _float(
        _first_value(
            metadata,
            "adaptive_g2_relative_delta_mean",
            "adaptive_g2_curvature_relative_delta_mean",
            "adaptive_surface_v2_curvature_relative_delta_mean",
            semantic={"g2", "relative", "delta", "mean"},
            default=0.0,
        )
    )

    curvature_sign_consistency = _bool(
        _first_value(
            metadata,
            "adaptive_g2_sign_consistency",
            "adaptive_surface_v2_curvature_sign_consistency",
            semantic={"sign", "consistency"},
            default=True,
        ),
        True,
    )

    normal_mismatch = bool(
        direct_requested
        or v2_case == "curvature_normal_continuity_failure"
        or prototype_orientation == "normal_continuity_mismatch"
        or mismatch_score >= 0.50
        or not curvature_sign_consistency
    )

    low_confidence = bool(target_confidence < 0.35 or low_fraction >= 0.50)
    bad_seed_projection = bool(seed_projection_max_ratio >= 0.18)

    if normal_mismatch:
        candidate_family = "curvature_normal_aligned_directional_seed_candidate"
        curvature_normal_field_status = "required"
        support_filter = "curvature_normal_filtered_ring_support"
        frame_policy = "local_curvature_normal_frames"
        target_policy = "curvature_normal_checked_mls_sphere_target"
        acceptance_policy = "must_improve_g1_g2_before_selection"
        reasons.append(
            "candidate must be built from surrounding curvature-normal continuation before any mesh seed is accepted"
        )
    elif low_confidence:
        candidate_family = "confidence_weighted_directional_seed_candidate"
        curvature_normal_field_status = "guarded"
        support_filter = "confidence_filtered_ring_support"
        frame_policy = "ring_adaptive_local_frames"
        target_policy = "confidence_weighted_mls_sphere_target"
        acceptance_policy = "must_pass_quality_g2_before_selection"
        reasons.append(
            "candidate must use confidence-filtered support because target confidence is low"
        )
    elif bad_seed_projection:
        candidate_family = "support_projected_directional_seed_candidate"
        curvature_normal_field_status = "guarded"
        support_filter = "normal_compatible_support"
        frame_policy = "support_projected_local_frames"
        target_policy = "support_projected_confidence_target"
        acceptance_policy = "must_pass_topology_quality_before_selection"
        reasons.append(
            "candidate must replace the legacy seed because seed projection error is high"
        )
    else:
        candidate_family = "legacy_seed_candidate_not_required"
        curvature_normal_field_status = "not_required"
        support_filter = "current_support_context"
        frame_policy = "current_seed_frames"
        target_policy = "current_target"
        acceptance_policy = "not_requested"
        reasons.append("v2 seed candidate is not required for this case")

    if normal_mismatch:
        planned_support_rings = 3
        planned_interior_rings = 2
    elif low_confidence:
        planned_support_rings = 3
        planned_interior_rings = 1
    else:
        planned_support_rings = 2
        planned_interior_rings = 1

    base_count = max(boundary_vertices, legacy_seed_vertices, 0)
    if base_count <= 0:
        planned_seed_vertices = 0
    elif normal_mismatch:
        # Do not explode density yet.  Preserve the existing budget as the
        # first candidate envelope, but require the construction method to
        # change.
        planned_seed_vertices = max(legacy_seed_vertices, boundary_vertices * 3)
    elif low_confidence:
        planned_seed_vertices = max(legacy_seed_vertices, boundary_vertices * 2)
    else:
        planned_seed_vertices = legacy_seed_vertices

    if normal_mismatch and boundary_vertices > 0:
        density_policy = "preserve_density_change_seed_field"
    elif low_confidence:
        density_policy = "preserve_or_slightly_reduce_density"
    else:
        density_policy = "preserve_current_density"

    build_ready = bool((require_new_seed or direct_requested) and candidate_family != "legacy_seed_candidate_not_required")

    status = "blueprint_ready" if build_ready else "not_required"
    action = "build_nonselecting_v2_seed_candidate_geometry_next" if build_ready else "do_not_build_v2_seed_candidate"
    geometry_status = "not_built_yet" if build_ready else "not_requested"

    if direct_requested:
        reasons.append(
            "public Adaptive Surface Fill v2 route requested a curvature-normal seed candidate"
        )
    if v2_case:
        reasons.append(f"adaptive surface v2 classifier case: {v2_case}")
    if low_confidence:
        reasons.append("target confidence is low; seed construction must be confidence-filtered")
    if bad_seed_projection:
        reasons.append("legacy seed projection error is high; seed geometry must be rebuilt, not patched")
    if curvature_relative >= 0.50:
        reasons.append("curvature deviation is high; candidate must be checked against G2 before selection")
    if not curvature_sign_consistency:
        reasons.append("curvature sign consistency is false; candidate must re-establish curvature-normal continuity")

    return AdaptiveSurfaceFillV2SeedCandidateBlueprint(
        status=str(status),
        action=str(action),
        candidate_family=str(candidate_family),
        geometry_status=str(geometry_status),
        selectable=False,
        curvature_normal_field_status=str(curvature_normal_field_status),
        support_filter=str(support_filter),
        frame_policy=str(frame_policy),
        target_policy=str(target_policy),
        density_policy=str(density_policy),
        acceptance_policy=str(acceptance_policy),
        boundary_vertices=int(boundary_vertices),
        legacy_seed_vertices=int(legacy_seed_vertices),
        planned_seed_vertices=int(planned_seed_vertices),
        planned_support_rings=int(planned_support_rings),
        planned_interior_rings=int(planned_interior_rings),
        normal_continuity_mismatch_score=float(mismatch_score),
        target_confidence=float(target_confidence),
        target_low_confidence_fraction=float(low_fraction),
        seed_projection_max_ratio=float(seed_projection_max_ratio),
        curvature_relative_delta_mean=float(curvature_relative),
        curvature_sign_consistency=bool(curvature_sign_consistency),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    )


__all__ = (
    "AdaptiveSurfaceFillV2SeedCandidateBlueprint",
    "build_adaptive_surface_fill_v2_seed_candidate_blueprint",
)
