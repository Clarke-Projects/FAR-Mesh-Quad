from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Mapping


@dataclass(frozen=True)
class CurvatureDirectionalSeedPlan:
    """Adaptive Surface Fill v2 seed-construction plan.

    This is the first real v2 seed layer.

    V1 behavior:
    - build a natural seed
    - diagnose curvature afterward
    - apply fix-over-fix if needed

    V2 behavior:
    - classify normal side, support confidence, curvature family, and target
      reliability before seed construction
    - choose a seed family that already knows which curvature/side policy it
      must satisfy
    """

    status: str
    action: str
    build_required: bool
    seed_family: str
    orientation_case: str
    orientation_action: str
    target_policy: str
    curvature_policy: str
    confidence_policy: str
    support_context_policy: str
    boundary_normal_mean_deviation: float
    boundary_normal_max_deviation: float
    support_normal_spread: float
    seed_projection_mean_ratio: float
    seed_projection_max_ratio: float
    target_confidence: float
    target_low_confidence_fraction: float
    curvature_relative_delta_mean: float
    curvature_sign_consistency: bool
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


def _safe_bool(value: object, default: bool = False) -> bool:
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


def _is_missing_metadata_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "-"}:
        return True
    return False


def _metadata_key_tokens(key: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.sub(r"[^a-zA-Z0-9]+", "_", str(key).lower()).split("_")
        if token and token not in {"adaptive"}
    )


def _first_value(metadata: Mapping[str, object], *keys: str, default: object = None) -> object:
    # Exact lookup first.
    for key in keys:
        if key in metadata:
            value = metadata.get(key)
            if not _is_missing_metadata_value(value):
                return value

    # Robust metadata alias fallback.
    #
    # The legacy adaptive controller has accumulated several naming styles:
    # - adaptive_g1_boundary_normal_mean_deviation
    # - adaptive_g1_boundary_normal_mean_deviation_degrees
    # - adaptive_g2_relative_delta_mean
    # - adaptive_g2_curvature_relative_delta_mean
    #
    # V2 must not miss critical gates because of suffix/alias drift.
    items = tuple(metadata.items())

    for requested_key in keys:
        requested_tokens = _metadata_key_tokens(requested_key)
        if not requested_tokens:
            continue

        for actual_key, actual_value in items:
            if _is_missing_metadata_value(actual_value):
                continue

            actual_tokens = _metadata_key_tokens(str(actual_key))
            if all(token in actual_tokens for token in requested_tokens):
                return actual_value

    # Extra semantic fallbacks for common curvature aliases.
    for requested_key in keys:
        requested_tokens = set(_metadata_key_tokens(requested_key))

        if {"g2", "relative", "delta", "mean"}.issubset(requested_tokens):
            for actual_key, actual_value in items:
                if _is_missing_metadata_value(actual_value):
                    continue
                actual_tokens = set(_metadata_key_tokens(str(actual_key)))
                if {"g2", "curvature", "relative", "delta", "mean"}.issubset(actual_tokens):
                    return actual_value

        if {"g2", "sign", "consistency"}.issubset(requested_tokens):
            for actual_key, actual_value in items:
                if _is_missing_metadata_value(actual_value):
                    continue
                actual_tokens = set(_metadata_key_tokens(str(actual_key)))
                if {"g2", "sign", "consistency"}.issubset(actual_tokens):
                    return actual_value

    return default

def _status(metadata: Mapping[str, object], *keys: str) -> str:
    value = _first_value(metadata, *keys, default="")
    if value is None:
        return ""
    return str(value).strip().lower()


def _number(metadata: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    return _safe_float(_first_value(metadata, *keys, default=default), default)


def build_curvature_directional_seed_plan(
    metadata: Mapping[str, object],
) -> CurvatureDirectionalSeedPlan:
    """Build the Adaptive Surface Fill v2 seed plan.

    This planner intentionally does not construct geometry yet.
    It defines what v2 must build next.
    """

    reasons: list[str] = []

    v2_case = _status(metadata, "adaptive_surface_v2_case", "case")
    v2_requires_new_seed = _safe_bool(
        _first_value(metadata, "adaptive_surface_v2_require_new_seed", "require_new_seed", default=False),
        False,
    )
    direct_requested = bool(
        v2_case == "direct_v2_candidate_requested"
        or _safe_bool(
            _first_value(
                metadata,
                "adaptive_surface_v2_direct_requested",
                "adaptive_surface_v2_force_candidate",
                "force_adaptive_surface_v2_candidate",
                default=False,
            ),
            False,
        )
    )

    g1_gate = _status(metadata, "adaptive_g1_gate_status", "g1_gate_status", "g1_gate")
    g2_gate = _status(metadata, "adaptive_g2_gate_status", "g2_gate_status", "g2_gate")
    seed_alignment = _status(metadata, "adaptive_seed_alignment_status", "seed_alignment_status")

    boundary_mean = _number(
        metadata,
        "adaptive_g1_boundary_normal_mean_deviation",
        "g1_boundary_normal_mean_deviation",
        default=0.0,
    )
    boundary_max = _number(
        metadata,
        "adaptive_g1_boundary_normal_max_deviation",
        "g1_boundary_normal_max_deviation",
        default=0.0,
    )
    support_spread = _number(
        metadata,
        "adaptive_g1_support_normal_spread",
        "adaptive_seed_support_normal_spread",
        "seed_support_normal_spread",
        default=0.0,
    )

    seed_mean_ratio = _number(
        metadata,
        "adaptive_seed_projection_mean_ratio",
        "seed_projection_mean_ratio",
        default=0.0,
    )
    seed_max_ratio = _number(
        metadata,
        "adaptive_seed_projection_max_ratio",
        "seed_projection_max_ratio",
        default=0.0,
    )

    target_confidence = _number(
        metadata,
        "adaptive_target_model_confidence",
        "target_model_confidence",
        default=0.5,
    )
    target_low_count = _number(
        metadata,
        "adaptive_target_confidence_low_count",
        "target_confidence_low_count",
        default=0.0,
    )
    target_vertex_count = _number(
        metadata,
        "adaptive_target_confidence_vertex_count",
        "target_confidence_vertex_count",
        default=0.0,
    )
    target_low_fraction = (
        float(target_low_count / max(target_vertex_count, 1.0))
        if target_vertex_count > 0.0
        else 0.0
    )

    curvature_relative = _number(
        metadata,
        "adaptive_g2_relative_delta_mean",
        "g2_relative_delta_mean",
        "curvature_relative_delta_mean",
        default=0.0,
    )
    sign_consistency = _safe_bool(
        _first_value(
            metadata,
            "adaptive_g2_sign_consistency",
            "g2_sign_consistency",
            "curvature_sign_consistency",
            default=True,
        ),
        True,
    )

    curvature_normal_continuity_failure = bool(
        g1_gate == "blocked"
        or boundary_mean >= 120.0
        or boundary_max >= 140.0
        or (not sign_consistency and curvature_relative >= 0.75)
        or (
            not sign_consistency
            and (
                curvature_relative >= 0.25
                or seed_max_ratio >= 0.18
                or target_confidence < 0.35
            )
        )
    )

    bad_seed_alignment = bool(
        seed_alignment == "bad"
        or seed_mean_ratio >= 0.08
        or seed_max_ratio >= 0.18
    )

    low_target_confidence = bool(
        target_confidence < 0.25
        or target_low_fraction >= 0.50
    )

    curvature_unstable = bool(
        g2_gate == "warning"
        or curvature_relative >= 0.50
        or not sign_consistency
    )

    if direct_requested:
        orientation_case = "direct_v2_curvature_normal_seed"
        orientation_action = "build_requested_v2_seed_in_local_curvature_normal_frames"
        reasons.append(
            "public Adaptive Surface Fill v2 route requested a v2 curvature-normal seed candidate"
        )
    elif curvature_normal_continuity_failure:
        orientation_case = "curvature_normal_continuity_mismatch"
        orientation_action = "build_seed_after_curvature_normal_continuity_check"
        reasons.append(
            "curvature and normal continuity are inconsistent with the surrounding mesh; v2 seed must be curvature-normal aligned before geometry is generated"
        )
    elif support_spread >= 60.0:
        orientation_case = "high_spread_mixed_curved_support"
        orientation_action = "build_seed_in_local_directional_frames"
        reasons.append(
            f"support normal spread is high ({support_spread:.6g}°); v2 seed must use local frames instead of one global target"
        )
    else:
        orientation_case = "orientation_nominal"
        orientation_action = "use_standard_local_frame_seed"

    if direct_requested:
        seed_family = "curvature_normal_aligned_directional_seed"
        target_policy = "curvature_normal_checked_ring_adaptive_target"
        confidence_policy = "per_vertex_confidence_required"
        reasons.append(
            "direct v2 route builds the curvature-normal-aligned directional seed as the test candidate"
        )
    elif curvature_normal_continuity_failure:
        seed_family = "curvature_normal_aligned_directional_seed"
        target_policy = "curvature_normal_checked_ring_adaptive_target"
        confidence_policy = "per_vertex_confidence_required"
        reasons.append(
            "curvature/normal-continuity failure requires a curvature-normal-aligned directional seed"
        )
        if bad_seed_alignment and low_target_confidence:
            reasons.append(
                "seed alignment is bad and target confidence is low; this strengthens the curvature-normal continuity failure diagnosis"
            )
    elif bad_seed_alignment and low_target_confidence:
        seed_family = "curvature_directional_confidence_seed"
        target_policy = "ring_adaptive_mls_sphere_target"
        confidence_policy = "per_vertex_confidence_required"
        reasons.append(
            "seed alignment is bad and target confidence is low; v2 must build a new confidence-weighted directional seed"
        )
    elif bad_seed_alignment:
        seed_family = "support_projected_directional_seed"
        target_policy = "confidence_weighted_surface_target"
        confidence_policy = "per_vertex_confidence_required"
        reasons.append(
            "seed projection error is too high for v1 fix-stacking"
        )
    elif low_target_confidence:
        seed_family = "confidence_weighted_directional_seed"
        target_policy = "ring_adaptive_mls_sphere_target"
        confidence_policy = "per_vertex_confidence_required"
        reasons.append(
            "target confidence is low across the patch; v2 must avoid global target pulling"
        )
    else:
        seed_family = "current_seed_family_allowed"
        target_policy = "current_confidence_target"
        confidence_policy = "confidence_guard_only"
        reasons.append(
            "current seed is acceptable; full v2 seed replacement is not required"
        )

    if curvature_unstable:
        curvature_policy = "curvature_directional_pre_seed_constraint"
        reasons.append(
            "curvature diagnostics are unstable; v2 must evaluate curvature direction before accepting a seed"
        )
    else:
        curvature_policy = "curvature_preserving_seed_constraint"

    if support_spread >= 60.0 or target_low_fraction >= 0.50:
        support_context_policy = "ring_adaptive_support_context"
    else:
        support_context_policy = "normal_compatible_support_context"

    build_required = bool(
        v2_requires_new_seed
        or direct_requested
        or curvature_normal_continuity_failure
        or bad_seed_alignment
        or low_target_confidence
        or curvature_unstable
    )

    status = "ready"
    action = (
        "build_adaptive_surface_fill_v2_seed"
        if build_required
        else "keep_current_seed_family"
    )

    if v2_case:
        reasons.append(f"adaptive surface v2 classifier case: {v2_case}")

    return CurvatureDirectionalSeedPlan(
        status=status,
        action=action,
        build_required=bool(build_required),
        seed_family=str(seed_family),
        orientation_case=str(orientation_case),
        orientation_action=str(orientation_action),
        target_policy=str(target_policy),
        curvature_policy=str(curvature_policy),
        confidence_policy=str(confidence_policy),
        support_context_policy=str(support_context_policy),
        boundary_normal_mean_deviation=float(boundary_mean),
        boundary_normal_max_deviation=float(boundary_max),
        support_normal_spread=float(support_spread),
        seed_projection_mean_ratio=float(seed_mean_ratio),
        seed_projection_max_ratio=float(seed_max_ratio),
        target_confidence=float(target_confidence),
        target_low_confidence_fraction=float(target_low_fraction),
        curvature_relative_delta_mean=float(curvature_relative),
        curvature_sign_consistency=bool(sign_consistency),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    )


__all__ = (
    "CurvatureDirectionalSeedPlan",
    "build_curvature_directional_seed_plan",
)
