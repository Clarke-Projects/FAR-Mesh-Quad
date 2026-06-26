"""Emit/diagnostic formatting helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It may compact diagnostics, compose
log-safe policy fragments, and prepare result metadata strings, but it must not
own feature meaning, delete authorization, geometry generation, trial topology,
or mesh replacement.  ``rebuild.py`` remains the single mutation authority.
"""

from __future__ import annotations

from typing import Iterable, Mapping

REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B = (
    "v176b_rebuild_emit_extracted_diagnostics_result_packing_no_behavior_change"
)

REBUILD_EMIT_NON_MUTATION_CONTRACT_V176B = (
    "rebuild_emit_may_format_diagnostics_only_rebuild_py_remains_mutation_authority"
)

REBUILD_FAILURE_REPORTING_EXTRACTION_CHECKPOINT_V176O = (
    "v176o_rebuild_failure_reporting_helpers_extracted_no_behavior_change"
)

REBUILD_FAILURE_REPORTING_NON_MUTATION_CONTRACT_V176O = (
    "rebuild_emit_may_format_failure_diagnostics_only_no_geometry_no_validation_no_mutation"
)


def attempt_summary_v176o(
    attempt_index: int,
    attempt: object,
    *,
    trial: Mapping[str, object] | None = None,
    plan: object | None = None,
    error: str = "",
) -> dict[str, object]:
    """Return the diagnostic summary for one rebuild attempt.

    v176o extraction note:
        This was moved from ``rebuild.py`` without behavior change.  The helper
        only reads the attempt object, optional trial diagnostics, and optional
        plan diagnostics.  It must not authorize deletion, generate geometry,
        validate topology, or mutate a mesh.
    """

    trial = dict(trial or {})
    plan_diag = dict(getattr(plan, "diagnostics", {}) if plan is not None else {})
    return {
        "attempt_index": int(attempt_index),
        "source": getattr(attempt, "source"),
        "target_source": getattr(attempt, "target_source"),
        "face_count": int(len(getattr(attempt, "face_ids"))),
        "boundary_loop_count": int(getattr(attempt, "boundary_loop_count")),
        "exact_two_loop_patch": bool(getattr(attempt, "exact_two_loop_patch")),
        "protected_loop_pair": bool(getattr(attempt, "protected_loop_pair")),
        "loop0_vertex_count": int(len(getattr(attempt, "loop0"))),
        "loop1_vertex_count": int(len(getattr(attempt, "loop1"))),
        "boundary_loop_vertex_count_delta": int(getattr(attempt, "boundary_loop_vertex_count_delta")),
        "unequal_loop_transition_allowed": bool(getattr(attempt, "unequal_loop_transition_allowed")),
        "unequal_loop_transition_used": bool(plan_diag.get("unequal_loop_transition_used", False)),
        "transition_drop_quad_count": int(plan_diag.get("transition_drop_quad_count", 0) or 0),
        "transition_ring_vertex_count": int(plan_diag.get("transition_ring_vertex_count", 0) or 0),
        "axial_separation": float(getattr(attempt, "axial_separation")),
        "min_required_axial_separation": float(getattr(attempt, "min_required_axial_separation")),
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
        "damaged_bore_internal_boundary_swallow_v173n": bool(plan_diag.get("damaged_bore_internal_boundary_swallow_v173n", False)),
        "damaged_bore_internal_boundary_swallow_acceptance_v173n": bool(plan_diag.get("damaged_bore_internal_boundary_swallow_acceptance_v173n", False)),
        "damaged_bore_boundary_match_scope_v173n": str(plan_diag.get("damaged_bore_boundary_match_scope_v173n", "-")),
        "damaged_bore_protected_boundary_edge_count_v173n": int(plan_diag.get("damaged_bore_protected_boundary_edge_count_v173n", -1) if plan_diag.get("damaged_bore_protected_boundary_edge_count_v173n", -1) is not None else -1),
        "damaged_bore_swallowed_defect_boundary_edge_count_v173n": int(plan_diag.get("damaged_bore_swallowed_defect_boundary_edge_count_v173n", -1) if plan_diag.get("damaged_bore_swallowed_defect_boundary_edge_count_v173n", -1) is not None else -1),
        "damaged_bore_small_boundary_seal_used": bool(plan_diag.get("damaged_bore_small_boundary_seal_used", False)),
        "damaged_bore_small_boundary_seal_added_triangle_count": int(plan_diag.get("damaged_bore_small_boundary_seal_added_triangle_count", 0) or 0),
        "damaged_bore_small_boundary_seal_boundary_edge_count_before": int(plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_before", -1) if plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_before", -1) is not None else -1),
        "damaged_bore_small_boundary_seal_boundary_edge_count_after_seal": int(plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_after_seal", -1) if plan_diag.get("damaged_bore_small_boundary_seal_boundary_edge_count_after_seal", -1) is not None else -1),
        "error": str(error),
    }


def format_failure_message_v176o(
    *,
    context: object,
    best_failure: Mapping[str, object],
    attempt_summaries: tuple[dict[str, object], ...],
    target_result: Mapping[str, object],
) -> str:
    """Return the measured-patch rebuild failure message.

    This helper is formatting-only.  It reads context fields and diagnostic
    mappings exactly as the previous ``rebuild.py`` helper did.
    """

    compact = compact_attempt_summaries_v176o(attempt_summaries)
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
        + f"candidate_entity_type={getattr(context, 'entity_type', '') or '-'}; "
        + f"candidate_from_component_engine={bool(getattr(context, 'candidate_from_component_engine', False))}; "
        + f"feature_ownership_source={getattr(context, 'feature_ownership_source', '') or '-'}; "
        + f"preview_candidate_patch_owns_delete={bool(getattr(context, 'candidate_has_preview_face_patch', False))}; "
        + "Geometry changed: no. parameter_fit_used=False; radius_used_for_delete_expansion=False."
    )


def compact_attempt_summaries_v176o(attempt_summaries: Iterable[Mapping[str, object]], *, limit: int = 8) -> str:
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


def pocket_rebuild_trace_summary_v175k(diagnostics: Mapping[str, object]) -> str:
    """Return a compact log-safe POCKET rebuild trace summary.

    v176b extraction note:
        This helper was moved out of ``rebuild.py`` without behavior change as
        the first low-risk rebuild roadmap split.  It only reads diagnostic
        mappings and returns a string.  It must not change generated vertices,
        face IDs, validation, rebuild acceptance, or mesh mutation.

    v175k contract:
        v175j records detailed loop-phase/radius-authority diagnostics, but the
        UI commit log only surfaces a small fixed subset of rebuild diagnostics.
        This helper condenses the v175j trace into one compact scalar string so
        it can be copied into fields that are already visible in the existing
        result/log path.
    """

    diag = dict(diagnostics or {})
    planned = dict(diag.get("pocket_planned_loop_phase_trace_v175j", {}) or {})
    input_trace = dict(diag.get("pocket_input_loop_phase_trace_v175j", {}) or {})
    source = str(
        diag.get(
            "semantic_radius_authority_source",
            diag.get("v163_constraint_radius_authority_source", "-"),
        )
        or "-"
    )
    mismatch = bool("bore" in source.lower())
    if not planned and not input_trace:
        return "pocket_trace_v175k=no_v175j_loop_phase_trace_available"

    def _ival(mapping: Mapping[str, object], key: str) -> int:
        try:
            return int(mapping.get(key, 0) or 0)
        except Exception:
            return 0

    def _fval(mapping: Mapping[str, object], key: str) -> float:
        try:
            return float(mapping.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _bval(mapping: Mapping[str, object], key: str) -> bool:
        try:
            return bool(mapping.get(key, False))
        except Exception:
            return False

    in0 = _ival(input_trace, "loop0_count")
    in1 = _ival(input_trace, "loop1_count")
    pl0 = _ival(planned, "loop0_count")
    pl1 = _ival(planned, "loop1_count")
    shift = _ival(planned, "best_cyclic_shift")
    shift_deg = _fval(planned, "best_cyclic_shift_degrees")
    reversed_loop = _bval(planned, "best_alignment_reversed_loop1")
    mean_err = _fval(planned, "best_alignment_mean_unit_chord_error")
    max_err = _fval(planned, "best_alignment_max_unit_chord_error")
    wave = bool(_bval(input_trace, "phase_wave_risk") or _bval(planned, "phase_wave_risk"))
    return (
        "pocket_trace_v175k="
        f"in{in0}x{in1}_plan{pl0}x{pl1}_"
        f"shift{shift}_{shift_deg:.1f}deg_"
        f"rev{int(reversed_loop)}_mean{mean_err:.4f}_max{max_err:.4f}_"
        f"wave{int(wave)}_radiusBoreLeak{int(mismatch)}_radiusSrc={source}"
    )


__all__ = [
    "REBUILD_EMIT_EXTRACTION_CHECKPOINT_V176B",
    "REBUILD_EMIT_NON_MUTATION_CONTRACT_V176B",
    "REBUILD_FAILURE_REPORTING_EXTRACTION_CHECKPOINT_V176O",
    "REBUILD_FAILURE_REPORTING_NON_MUTATION_CONTRACT_V176O",
    "attempt_summary_v176o",
    "compact_attempt_summaries_v176o",
    "format_failure_message_v176o",
    "pocket_rebuild_trace_summary_v175k",
]
