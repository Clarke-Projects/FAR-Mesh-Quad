from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping
import re


@dataclass(frozen=True)
class AdaptiveSurfaceFillV2SeedPrototype:
    """First executable Adaptive Surface Fill v2 seed prototype decision.

    This is not the final geometry builder yet. It is the v2 pre-seed layer:
    - read support/seed/curvature/normal-side diagnostics
    - choose the seed family before geometry is generated
    - reject v1-style fix stacking when the seed is fundamentally wrong
    """

    status: str
    action: str
    build_required: bool
    seed_family: str
    geometry_status: str

    orientation_status: str
    orientation_action: str
    orientation_confidence: float
    boundary_normal_mean_deviation: float
    boundary_normal_max_deviation: float
    boundary_normal_side_score_mean: float
    boundary_normal_side_score_max: float
    normal_sign_status: str

    support_context_policy: str
    target_policy: str
    curvature_policy: str
    confidence_policy: str

    seed_projection_mean_ratio: float
    seed_projection_max_ratio: float
    target_confidence: float
    target_low_confidence_fraction: float
    support_normal_spread: float
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


def _first_exact(metadata: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in metadata and not _missing(metadata.get(key)):
            return metadata.get(key)
    return default


def _first_semantic(
    metadata: Mapping[str, object],
    required: set[str],
    *,
    default: object = None,
) -> object:
    for key, value in metadata.items():
        if _missing(value):
            continue
        actual = set(_tokens(str(key)))
        if required.issubset(actual):
            return value
    return default


def _first_value(
    metadata: Mapping[str, object],
    *keys: str,
    semantic: set[str] | None = None,
    default: object = None,
) -> object:
    exact = _first_exact(metadata, *keys, default=None)
    if not _missing(exact):
        return exact

    for key in keys:
        requested = set(_tokens(key))
        if requested:
            value = _first_semantic(metadata, requested, default=None)
            if not _missing(value):
                return value

    if semantic:
        value = _first_semantic(metadata, semantic, default=None)
        if not _missing(value):
            return value

    return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not isfinite(number):
        return float(default)
    return float(number)


def _bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "ok", "passed"}:
            return True
        if normalized in {"false", "no", "0", "blocked", "warning"}:
            return False
    return bool(default)


def _status(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def build_adaptive_surface_fill_v2_seed_prototype(
    metadata: Mapping[str, object],
) -> AdaptiveSurfaceFillV2SeedPrototype:
    reasons: list[str] = []

    v2_requires_new_seed = _bool(
        _first_value(
            metadata,
            "adaptive_surface_v2_require_new_seed",
            "require_new_seed",
            semantic={"surface", "v2", "require", "new", "seed"},
            default=False,
        ),
        False,
    )

    v2_case = _status(
        _first_value(
            metadata,
            "adaptive_surface_v2_case",
            semantic={"surface", "v2", "case"},
            default="",
        )
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

    g1_gate = _status(
        _first_value(
            metadata,
            "adaptive_g1_gate_status",
            "g1_gate_status",
            "g1_gate",
            semantic={"g1", "gate", "status"},
            default="",
        )
    )

    boundary_mean = _float(
        _first_value(
            metadata,
            "adaptive_g1_boundary_normal_mean_deviation",
            "adaptive_g1_boundary_normal_mean_deviation_degrees",
            "adaptive_g1_boundary_normal_mean",
            "g1_boundary_normal_mean_deviation",
            semantic={"g1", "boundary", "normal", "mean"},
            default=0.0,
        )
    )

    boundary_max = _float(
        _first_value(
            metadata,
            "adaptive_g1_boundary_normal_max_deviation",
            "adaptive_g1_boundary_normal_max_deviation_degrees",
            "adaptive_g1_boundary_normal_max",
            "g1_boundary_normal_max_deviation",
            semantic={"g1", "boundary", "normal", "max"},
            default=0.0,
        )
    )

    support_spread = _float(
        _first_value(
            metadata,
            "adaptive_g1_support_normal_spread",
            "adaptive_seed_support_normal_spread",
            "seed_support_normal_spread",
            semantic={"support", "normal", "spread"},
            default=0.0,
        )
    )

    seed_mean_ratio = _float(
        _first_value(
            metadata,
            "adaptive_seed_projection_mean_ratio",
            "seed_projection_mean_ratio",
            semantic={"seed", "projection", "mean", "ratio"},
            default=0.0,
        )
    )

    seed_max_ratio = _float(
        _first_value(
            metadata,
            "adaptive_seed_projection_max_ratio",
            "seed_projection_max_ratio",
            semantic={"seed", "projection", "max", "ratio"},
            default=0.0,
        )
    )

    target_confidence = _float(
        _first_value(
            metadata,
            "adaptive_target_model_confidence",
            "target_model_confidence",
            semantic={"target", "model", "confidence"},
            default=0.5,
        ),
        0.5,
    )

    low_count = _float(
        _first_value(
            metadata,
            "adaptive_target_confidence_low_count",
            "target_confidence_low_count",
            semantic={"target", "confidence", "low"},
            default=0.0,
        )
    )
    vertex_count = _float(
        _first_value(
            metadata,
            "adaptive_target_confidence_vertex_count",
            "target_confidence_vertex_count",
            semantic={"target", "confidence", "vertex"},
            default=0.0,
        )
    )

    low_fraction = float(low_count / max(vertex_count, 1.0)) if vertex_count > 0.0 else 0.0

    curvature_relative = _float(
        _first_value(
            metadata,
            "adaptive_g2_relative_delta_mean",
            "adaptive_g2_curvature_relative_delta_mean",
            "g2_relative_delta_mean",
            "curvature_relative_delta_mean",
            semantic={"g2", "relative", "delta", "mean"},
            default=0.0,
        )
    )

    sign_consistent = _bool(
        _first_value(
            metadata,
            "adaptive_g2_sign_consistency",
            "g2_sign_consistency",
            "curvature_sign_consistency",
            semantic={"sign", "consistency"},
            default=True,
        ),
        True,
    )

    # Side score is 0 at aligned, 1 at fully opposite.
    side_score_mean = float(max(0.0, min(1.0, boundary_mean / 180.0)))
    side_score_max = float(max(0.0, min(1.0, boundary_max / 180.0)))

    high_spread = bool(support_spread >= 60.0)
    bad_seed = bool(seed_mean_ratio >= 0.08 or seed_max_ratio >= 0.18)
    low_confidence = bool(target_confidence < 0.25 or low_fraction >= 0.50)
    curvature_unstable = bool(curvature_relative >= 0.50 or not sign_consistent)

    # V2-C-D:
    # Do not wait for late numeric G1 boundary-normal fields.  A seed with
    # sign-inconsistent curvature plus bad projection / low confidence is
    # already a curvature-normal-continuity risk and must be checked before geometry.
    wrong_side = bool(
        g1_gate == "blocked"
        or boundary_mean >= 120.0
        or boundary_max >= 140.0
        or (not sign_consistent and curvature_relative >= 0.75)
        or (
            not sign_consistent
            and (
                curvature_relative >= 0.25
                or bad_seed
                or low_confidence
            )
        )
    )

    if wrong_side and side_score_mean == 0.0 and side_score_max == 0.0:
        inferred_side_score = float(
            max(
                min(1.0, max(curvature_relative, 0.0) / 2.0),
                min(1.0, seed_max_ratio / 0.25) if seed_max_ratio > 0.0 else 0.0,
                min(1.0, max(0.0, 1.0 - target_confidence)),
                0.75 if not sign_consistent else 0.0,
            )
        )
        side_score_mean = inferred_side_score
        side_score_max = inferred_side_score

    if wrong_side:
        orientation_status = "normal_continuity_mismatch"
        orientation_action = "build_curvature_normal_aligned_seed_before_fill"
        orientation_confidence = float(
            max(
                side_score_mean,
                side_score_max,
                min(0.98, max(curvature_relative, 0.0) / 2.0),
                min(0.98, seed_max_ratio / 0.25) if seed_max_ratio > 0.0 else 0.0,
                min(0.98, max(0.0, 1.0 - target_confidence)),
            )
        )
        normal_sign_status = "normal_continuity_inconsistent"
        reasons.append(
            "curvature and normal continuity indicate the v1 seed does not follow the surrounding mesh continuation"
        )
    elif high_spread:
        orientation_status = "high_spread_directional"
        orientation_action = "build_seed_in_local_curvature_frames"
        orientation_confidence = float(max(0.35, min(0.85, support_spread / 90.0)))
        normal_sign_status = "mixed_support_normals"
        reasons.append(
            f"support normal spread is high ({support_spread:.6g}°); v2 must use local curvature frames"
        )
    else:
        orientation_status = "nominal"
        orientation_action = "standard_local_frame_seed_allowed"
        orientation_confidence = float(max(0.0, 1.0 - max(side_score_mean, side_score_max)))
        normal_sign_status = "consistent"

    if direct_requested:
        seed_family = "curvature_normal_aligned_directional_seed"
        support_context_policy = "curvature_normal_filtered_ring_adaptive_support"
        target_policy = "curvature_normal_checked_ring_adaptive_mls_sphere_target"
        reasons.append(
            "public Adaptive Surface Fill v2 route requested curvature-normal seed construction"
        )
    elif wrong_side:
        seed_family = "curvature_normal_aligned_directional_seed"
        support_context_policy = "curvature_normal_filtered_ring_adaptive_support"
        target_policy = "curvature_normal_checked_ring_adaptive_mls_sphere_target"
        reasons.append(
            "v2 seed must derive curvature-normal continuation from the surrounding mesh before constructing patch vertices"
        )
    elif bad_seed and low_confidence:
        seed_family = "curvature_directional_confidence_seed"
        support_context_policy = "ring_adaptive_support_context"
        target_policy = "ring_adaptive_mls_sphere_target"
        reasons.append(
            "seed projection is bad and target confidence is low; v2 must construct a new curvature-aware seed"
        )
    elif bad_seed:
        seed_family = "support_projected_curvature_seed"
        support_context_policy = "normal_compatible_support_context"
        target_policy = "confidence_weighted_surface_target"
        reasons.append(
            "seed projection is too high for v1 fix-stacking"
        )
    elif low_confidence:
        seed_family = "confidence_weighted_directional_seed"
        support_context_policy = "ring_adaptive_support_context"
        target_policy = "ring_adaptive_mls_sphere_target"
        reasons.append(
            "target confidence is low; v2 must use per-vertex confidence before seed acceptance"
        )
    else:
        seed_family = "current_seed_allowed"
        support_context_policy = "normal_compatible_support_context"
        target_policy = "current_confidence_target"

    if curvature_unstable:
        curvature_policy = "pre_seed_directional_curvature_constraint"
        reasons.append(
            "curvature is unstable or sign-inconsistent; v2 must evaluate curvature before accepting generated vertices"
        )
    else:
        curvature_policy = "curvature_preserving_seed_constraint"

    if direct_requested or low_confidence or bad_seed:
        confidence_policy = "per_vertex_confidence_required"
    else:
        confidence_policy = "confidence_guard_only"

    build_required = bool(
        v2_requires_new_seed
        or direct_requested
        or wrong_side
        or bad_seed
        or low_confidence
        or curvature_unstable
    )

    if build_required:
        status = "prototype_ready"
        action = "build_v2_seed_candidate_next"
        geometry_status = "not_built_yet"
    else:
        status = "not_required"
        action = "keep_current_seed"
        geometry_status = "not_requested"

    if v2_case:
        reasons.append(f"adaptive surface v2 classifier case: {v2_case}")

    return AdaptiveSurfaceFillV2SeedPrototype(
        status=status,
        action=action,
        build_required=bool(build_required),
        seed_family=str(seed_family),
        geometry_status=str(geometry_status),
        orientation_status=str(orientation_status),
        orientation_action=str(orientation_action),
        orientation_confidence=float(orientation_confidence),
        boundary_normal_mean_deviation=float(boundary_mean),
        boundary_normal_max_deviation=float(boundary_max),
        boundary_normal_side_score_mean=float(side_score_mean),
        boundary_normal_side_score_max=float(side_score_max),
        normal_sign_status=str(normal_sign_status),
        support_context_policy=str(support_context_policy),
        target_policy=str(target_policy),
        curvature_policy=str(curvature_policy),
        confidence_policy=str(confidence_policy),
        seed_projection_mean_ratio=float(seed_mean_ratio),
        seed_projection_max_ratio=float(seed_max_ratio),
        target_confidence=float(target_confidence),
        target_low_confidence_fraction=float(low_fraction),
        support_normal_spread=float(support_spread),
        curvature_relative_delta_mean=float(curvature_relative),
        curvature_sign_consistency=bool(sign_consistent),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    )


__all__ = (
    "AdaptiveSurfaceFillV2SeedPrototype",
    "build_adaptive_surface_fill_v2_seed_prototype",
)
