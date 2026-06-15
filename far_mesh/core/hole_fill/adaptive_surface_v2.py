from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Mapping


@dataclass(frozen=True)
class AdaptiveSurfaceFillV2Decision:
    """Routing decision for the refined adaptive surface fill v2 path.

    V2-A is a classifier/guard only:
    - no mesh mutation
    - no preview construction
    - no replacement of adaptive_surface yet
    - prevents legacy fix-stacking when the case requires a new seed
    """

    status: str
    case: str
    action: str
    block_legacy_selection: bool
    allow_confidence_target_delta: bool
    allow_local_anisotropic_correction: bool
    require_new_seed: bool
    recommended_seed_family: str
    recommended_target_policy: str
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


def _safe_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _safe_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "passed", "allowed", "selected"}:
            return True
        if normalized in {"false", "no", "0", "blocked", "not_selected"}:
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
    return _safe_str(_first_value(metadata, *keys, default="")).strip().lower()


def _number(metadata: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    return _safe_float(_first_value(metadata, *keys, default=default), default)


def build_adaptive_surface_v2_decision(
    metadata: Mapping[str, object],
) -> AdaptiveSurfaceFillV2Decision:
    """Classify whether legacy adaptive fix-stacking is still appropriate.

    Learned cases:
    - localized anisotropic residual: old seed can be locally corrected.
    - global normal-side failure: old seed/fix stack must stop.
    - bad seed/target alignment: build a new seed/target model.
    - low confidence global target: v2 target policy required.
    """

    reasons: list[str] = []

    direct_requested = _safe_bool(
        _first_value(
            metadata,
            "adaptive_surface_v2_direct_requested",
            "adaptive_surface_v2_force_candidate",
            "force_adaptive_surface_v2_candidate",
            default=False,
        ),
        False,
    )

    commit_policy = _status(metadata, "adaptive_commit_policy", "commit_policy")
    g1_gate = _status(metadata, "adaptive_g1_gate_status", "g1_gate_status", "g1_gate")
    g2_gate = _status(metadata, "adaptive_g2_gate_status", "g2_gate_status", "g2_gate")
    seed_alignment = _status(metadata, "adaptive_seed_alignment_status", "seed_alignment_status")
    directional_status = _status(metadata, "adaptive_directional_target_status", "directional_target_status")

    g1_mean = _number(
        metadata,
        "adaptive_g1_boundary_normal_mean_deviation",
        "g1_boundary_normal_mean_deviation",
        default=0.0,
    )
    g1_max = _number(
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

    curvature_relative = _number(
        metadata,
        "adaptive_g2_relative_delta_mean",
        "g2_relative_delta_mean",
        "curvature_relative_delta_mean",
        default=0.0,
    )
    curvature_sign_consistency = _first_value(
        metadata,
        "adaptive_g2_sign_consistency",
        "g2_sign_consistency",
        "curvature_sign_consistency",
        default=True,
    )
    sign_consistent = _safe_bool(curvature_sign_consistency, True)

    anisotropic_selected = _safe_bool(
        _first_value(
            metadata,
            "adaptive_anisotropic_candidate_selected_by_policy",
            "adaptive_anisotropic_candidate_selected",
            default=False,
        ),
        False,
    )

    confidence_selected = _safe_bool(
        _first_value(
            metadata,
            "adaptive_confidence_target_candidate_selected_by_policy",
            "adaptive_confidence_target_candidate_selected",
            default=False,
        ),
        False,
    )

    curvature_normal_continuity_failure = bool(
        g1_gate == "blocked"
        or g1_mean >= 120.0
        or g1_max >= 140.0
        or (not sign_consistent and curvature_relative >= 0.75)
        or (
            not sign_consistent
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
        or (
            target_vertex_count > 0
            and target_low_count / max(target_vertex_count, 1.0) >= 0.50
        )
    )

    localized_anisotropic = bool(
        directional_status == "directional_warning"
        and not curvature_normal_continuity_failure
        and not bad_seed_alignment
    )

    commit_blocked = bool(commit_policy == "blocked")

    if direct_requested:
        case = "direct_v2_candidate_requested"
        action = "build_curvature_normal_aligned_seed_candidate"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "curvature_normal_aligned_directional_seed"
        recommended_target_policy = "curvature_normal_checked_confidence_target"
        reasons.append(
            "Adaptive Surface Fill v2 was selected explicitly; build the v2 curvature-normal seed candidate even when the legacy adaptive candidate is acceptable"
        )
    elif curvature_normal_continuity_failure:
        case = "curvature_normal_continuity_failure"
        action = "stop_legacy_fix_stack_and_build_curvature_normal_aligned_seed"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "curvature_normal_aligned_directional_seed"
        recommended_target_policy = "curvature_normal_checked_confidence_target"
        reasons.append(
            "Curvature/normal-continuity diagnostics indicate the v1 seed does not follow the surrounding mesh continuation; legacy confidence/aniso fix stacking should stop"
        )
    elif bad_seed_alignment and low_target_confidence:
        case = "bad_seed_low_confidence_target"
        action = "build_adaptive_surface_fill_v2_seed"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "curvature_directional_confidence_seed"
        recommended_target_policy = "ring_adaptive_mls_sphere_target"
        reasons.append(
            "seed alignment is bad and target confidence is low; a new v2 seed/target model is required"
        )
    elif bad_seed_alignment:
        case = "bad_seed_alignment"
        action = "replace_seed_before_local_corrections"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "support_projected_directional_seed"
        recommended_target_policy = "confidence_weighted_surface_target"
        reasons.append(
            "seed-to-support projection error is too high for fix-stacking; replace the seed first"
        )
    elif localized_anisotropic:
        case = "localized_anisotropic_residual"
        action = "allow_local_anisotropic_correction"
        block_legacy_selection = False
        require_new_seed = False
        recommended_seed_family = "current_seed_with_local_directional_correction"
        recommended_target_policy = "localized_anisotropic_target"
        reasons.append(
            "directional diagnostics indicate a localized anisotropic residual; local correction may be appropriate"
        )
    elif low_target_confidence:
        case = "low_confidence_target"
        action = "prefer_v2_target_model_before_selection"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "confidence_weighted_directional_seed"
        recommended_target_policy = "ring_adaptive_mls_sphere_target"
        reasons.append(
            "target confidence is low across many vertices; v2 target policy should replace global fix-stacking"
        )
    elif commit_blocked:
        case = "commit_blocked"
        action = "do_not_select_legacy_candidates"
        block_legacy_selection = True
        require_new_seed = True
        recommended_seed_family = "blocked_commit_diagnostic_seed"
        recommended_target_policy = "strict_gate_first"
        reasons.append(
            "commit policy is blocked; legacy candidate selection must not run"
        )
    else:
        case = "legacy_candidate_allowed"
        action = "allow_gated_legacy_candidate_selection"
        block_legacy_selection = False
        require_new_seed = False
        recommended_seed_family = "current_seed"
        recommended_target_policy = "current_confidence_target"
        reasons.append(
            "no v2 blocking condition detected; gated legacy candidate selection may continue"
        )

    if support_spread >= 60.0:
        reasons.append(
            f"support normal spread is high ({support_spread:.6g}°); v2 should use local directional frames"
        )
    if g2_gate == "warning":
        reasons.append("G2 gate is warning; v2 should preserve curvature comparison as a selector gate")
    if confidence_selected:
        reasons.append("confidence target delta has been selected by legacy policy")
    if anisotropic_selected:
        reasons.append("localized anisotropic correction has been selected by legacy policy")

    allow_confidence = bool(not block_legacy_selection and not require_new_seed)
    allow_anisotropic = bool(localized_anisotropic and not block_legacy_selection)

    return AdaptiveSurfaceFillV2Decision(
        status="ready",
        case=str(case),
        action=str(action),
        block_legacy_selection=bool(block_legacy_selection),
        allow_confidence_target_delta=bool(allow_confidence),
        allow_local_anisotropic_correction=bool(allow_anisotropic),
        require_new_seed=bool(require_new_seed),
        recommended_seed_family=str(recommended_seed_family),
        recommended_target_policy=str(recommended_target_policy),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
    )


__all__ = (
    "AdaptiveSurfaceFillV2Decision",
    "build_adaptive_surface_v2_decision",
)
