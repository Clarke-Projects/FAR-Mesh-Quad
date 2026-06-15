from __future__ import annotations

import numpy as np
import trimesh
from PySide6.QtWidgets import QMessageBox

from far_mesh.core.hole_fill_preview import (
    available_hole_fill_preview_methods,
    hole_fill_method_capability,
)
from far_mesh.core.manual_edit_pipeline import ManualEditPreview, ManualEditResult
from far_mesh.core.selection_topology import diagnose_hole_candidates
from far_mesh.core.tool_preview_state import REGION_KIND_HOLE_PATCH

from .project_actions import _update_project_status_ui_if_available
from .status_formatters import (
    format_optional_centroid,
    format_optional_float,
    topology_scope_label,
)


class TopologyToolsMixin:
    """GUI-controller helpers for Phase 2 topology and hole-fill workflows.

    This mixin contains UI orchestration only. Mesh topology analysis, hole
    candidate detection, preview construction, and commit authority remain in
    MeshProcessor and the core Phase 2 modules.
    """

    def _current_topology_face_ids(self) -> tuple[int, ...] | None:
        """
        Return selected face IDs for topology tools, or None for whole-mesh analysis.

        SelectionController remains the selection authority. MeshProcessor owns
        the mesh and receives these IDs only as an optional analysis scope.
        """
        try:
            self.selection_controller.sync_from_viewport(reason="topology_request")
        except Exception:
            pass

        try:
            face_ids = tuple(int(v) for v in self.selection_controller.selected_face_ids())
        except Exception:
            face_ids = ()

        return face_ids if face_ids else None

    @staticmethod
    def _topology_scope_label(face_ids: tuple[int, ...] | None) -> str:
        return topology_scope_label(face_ids)

    @staticmethod
    def _format_optional_float(value: object, *, digits: int = 6) -> str:
        return format_optional_float(value, digits=digits)

    @staticmethod
    def _format_optional_centroid(value: object) -> str:
        return format_optional_centroid(value)

    def _set_topology_result_text(self, text: str) -> None:
        if hasattr(self, "topology_result_text"):
            self.topology_result_text.setPlainText(str(text))

    def _set_hole_fill_status(self, text: str) -> None:
        if hasattr(self, "hole_fill_status_label"):
            self.hole_fill_status_label.setText(str(text))

    def _current_hole_fill_method_key(self) -> str:
        method = "fan"
        if hasattr(self, "hole_fill_method_combo"):
            try:
                method = str(self.hole_fill_method_combo.currentData() or method)
            except Exception:
                method = "fan"
        return method

    @staticmethod
    def _hole_fill_preview_is_batch(preview: object | None) -> bool:
        if preview is None:
            return False

        summary = getattr(preview, "selection_summary", None)
        if isinstance(summary, dict) and bool(summary.get("batch_mode", False)):
            return True

        preview_mesh = getattr(preview, "preview_mesh", None)
        metadata = getattr(preview_mesh, "metadata", None)
        return isinstance(metadata, dict) and bool(metadata.get("batch_mode", False))

    def _set_hole_fill_batch_buttons_enabled(
        self,
        *,
        preview_enabled: bool | None = None,
        commit_enabled: bool | None = None,
    ) -> None:
        busy = bool(getattr(self, "_worker", None) is not None)
        has_candidates = bool(getattr(self, "_last_hole_candidates", []))

        if preview_enabled is None:
            preview_enabled = has_candidates and not busy
        if commit_enabled is None:
            commit_enabled = (
                self._hole_fill_preview is not None
                and self._hole_fill_preview_is_batch(self._hole_fill_preview)
                and not busy
            )

        if hasattr(self, "hole_fill_batch_preview_btn"):
            self.hole_fill_batch_preview_btn.setEnabled(bool(preview_enabled))
        if hasattr(self, "hole_fill_batch_commit_btn"):
            self.hole_fill_batch_commit_btn.setEnabled(bool(commit_enabled))


    @staticmethod
    def _hole_fill_gui_bool(value: object, *, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "allowed", "ok"}:
            return True
        if text in {"0", "false", "no", "n", "blocked", "not_allowed"}:
            return False

        return bool(default)

    @staticmethod
    def _hole_fill_gui_text_tuple(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,) if value else ()

        try:
            return tuple(str(item) for item in value if str(item))
        except Exception:
            text = str(value)
            return (text,) if text else ()

    def _hole_fill_preview_commit_policy_for_gui(
        self,
        preview: ManualEditPreview,
    ) -> dict[str, object]:
        """Read routed preview commit policy for GUI controls.

        This method intentionally keeps GUI-side policy calculation defensive:
        it accepts metadata from ManualEditPreview.selection_summary,
        preview_mesh.metadata, nested preview metadata, adaptive_diagnostics,
        and ToolPreviewState metadata.
        """

        summary = getattr(preview, "selection_summary", None)
        if not isinstance(summary, dict):
            summary = {}

        metadata: dict[str, object] = {}

        preview_mesh = getattr(preview, "preview_mesh", None)
        raw_preview_metadata = getattr(preview_mesh, "metadata", None)
        if isinstance(raw_preview_metadata, dict):
            metadata.update(raw_preview_metadata)
            for nested_key in (
                "hole_fill_preview",
                "preview_builder_metadata",
                "adaptive_diagnostics",
            ):
                nested = raw_preview_metadata.get(nested_key)
                if isinstance(nested, dict):
                    metadata.update(nested)

        processor = getattr(self, "processor", None)
        getter = getattr(processor, "last_tool_preview_state", None)
        if callable(getter):
            try:
                tool_state = getter()
                raw_metadata = getattr(tool_state, "metadata", None)
                if isinstance(raw_metadata, dict):
                    metadata.update(raw_metadata)
            except Exception:
                metadata = {}

        def is_empty(value: object) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return not value.strip() or value.strip() == "-"
            if isinstance(value, (tuple, list, dict, set)):
                return len(value) == 0
            return False

        def first_value(key: str, default: object = None) -> object:
            if key in summary and not is_empty(summary.get(key)):
                return summary.get(key)
            if key in metadata and not is_empty(metadata.get(key)):
                return metadata.get(key)
            return default

        raw_diagnostics = first_value("adaptive_diagnostics", {})
        if not isinstance(raw_diagnostics, dict):
            raw_diagnostics = {}

        def diagnostic_value(key: str, default: object = "-") -> object:
            if key in summary and not is_empty(summary.get(key)):
                return summary.get(key)
            if key in metadata and not is_empty(metadata.get(key)):
                return metadata.get(key)
            if key in raw_diagnostics and not is_empty(raw_diagnostics.get(key)):
                return raw_diagnostics.get(key)
            return default

        def nested_diagnostic_value(
            nested_key: str,
            value_key: str,
            default: object = "-",
        ) -> object:
            for source in (summary, metadata, raw_diagnostics):
                if not isinstance(source, dict):
                    continue
                nested = source.get(nested_key)
                if isinstance(nested, dict) and not is_empty(nested.get(value_key)):
                    return nested.get(value_key)
            return default

        def mapping_from_sources(key: str) -> dict[str, object]:
            for source in (summary, metadata, raw_diagnostics):
                if not isinstance(source, dict):
                    continue
                value = source.get(key)
                if isinstance(value, dict):
                    return value
            return {}

        def nested_mapping_from_sources(
            parent_key: str,
            child_key: str,
        ) -> dict[str, object]:
            for source in (summary, metadata, raw_diagnostics):
                if not isinstance(source, dict):
                    continue
                parent = source.get(parent_key)
                if isinstance(parent, dict):
                    child = parent.get(child_key)
                    if isinstance(child, dict):
                        return child
            return {}

        def to_int(value: object, default: int = 0) -> int:
            try:
                if is_empty(value):
                    return int(default)
                return int(value)
            except Exception:
                return int(default)

        def to_float(value: object, default: float | None = None) -> float | None:
            try:
                if is_empty(value):
                    return default
                return float(value)
            except Exception:
                return default

        def to_bool(value: object, default: bool | None = None) -> bool | None:
            if isinstance(value, bool):
                return value
            if is_empty(value):
                return default
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "allowed", "ok"}:
                return True
            if text in {"0", "false", "no", "n", "blocked", "not_allowed"}:
                return False
            return default

        def text_tuple(value: object) -> tuple[str, ...]:
            if is_empty(value):
                return ()
            if isinstance(value, str):
                return (value,) if value else ()
            try:
                return tuple(str(item) for item in value if str(item))
            except Exception:
                text = str(value)
                return (text,) if text else ()

        def clean_display_value(value: object, default: object = "-") -> object:
            return default if is_empty(value) else value

        def float_display_value(value: object) -> object:
            number = to_float(value, None)
            if number is None:
                return "-"
            return round(float(number), 3)

        commit_allowed = self._hole_fill_gui_bool(
            first_value("commit_allowed", True),
            default=True,
        )
        blocking_reasons = self._hole_fill_gui_text_tuple(
            first_value("commit_blocking_reasons", ())
        )
        warnings = self._hole_fill_gui_text_tuple(
            first_value("commit_warnings", ())
        )

        if not commit_allowed and not blocking_reasons:
            blocking_reasons = ("Preview policy marked this patch as not commit-eligible.",)

        def derived_quality_status() -> object:
            raw = diagnostic_value("quality_status", None)
            if isinstance(raw, str) and raw.strip().lower() == "unknown":
                raw = None
            if not is_empty(raw):
                return raw

            quality_after = mapping_from_sources("quality_after")
            if not quality_after:
                quality_after = nested_mapping_from_sources("relaxation", "quality_after")
            if not quality_after:
                quality_after = nested_mapping_from_sources("biharmonic_fairing", "quality_after")

            if not quality_after:
                return "not_reported"

            degenerate = to_int(quality_after.get("degenerate_face_count", 0), 0)
            min_angle = to_float(quality_after.get("min_triangle_angle_degrees", 999.0), 999.0)
            aspect = to_float(quality_after.get("max_triangle_aspect_ratio", 0.0), 0.0)

            if degenerate > 0:
                return "blocked"
            if min_angle is not None and min_angle < 1.0:
                return "warning"
            if aspect is not None and aspect > 100.0:
                return "warning"
            return "ok"

        method_for_gate = str(
            first_value("public_method", first_value("method", ""))
        ).strip().lower().replace("-", "_")

        strict_gate_reasons: list[str] = []
        strict_gate_status = "not_applicable"

        if method_for_gate in {
            "adaptive_surface",
            "adaptive",
            "adaptive_surface_fill",
            "adaptive_surface_v2",
            "adaptive_surface_fill_v2",
        }:
            strict_gate_status = "passed"

            seam_missing = to_int(
                diagnostic_value(
                    "seam_missing_edge_count",
                    nested_diagnostic_value("seam_constraint_report", "missing_seam_edge_count", 0),
                )
            )
            seam_overused = to_int(
                diagnostic_value(
                    "seam_overused_edge_count",
                    nested_diagnostic_value("seam_constraint_report", "overused_seam_edge_count", 0),
                )
            )
            seam_coverage = to_float(
                diagnostic_value(
                    "seam_coverage_ratio",
                    nested_diagnostic_value("seam_constraint_report", "seam_coverage_ratio", None),
                )
            )
            nonmanifold_edges = to_int(
                diagnostic_value(
                    "nonmanifold_patch_edge_count",
                    nested_diagnostic_value("topology_after", "nonmanifold_patch_edge_count", 0),
                )
            )
            extra_open_edges = to_int(
                diagnostic_value(
                    "extra_open_boundary_edge_count",
                    nested_diagnostic_value("topology_after", "extra_open_boundary_edge_count", 0),
                )
            )
            degenerate_faces = to_int(
                diagnostic_value(
                    "degenerate_face_count",
                    nested_diagnostic_value("quality_after", "degenerate_face_count", 0),
                )
            )

            if seam_missing > 0:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: missing seam edges = {seam_missing}"
                )
            if seam_overused > 0:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: overused seam edges = {seam_overused}"
                )
            if seam_coverage is not None and seam_coverage < 0.999999:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: seam coverage {seam_coverage:.6g} < 1.0"
                )
            if nonmanifold_edges > 0:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: nonmanifold patch edges = {nonmanifold_edges}"
                )
            if extra_open_edges > 0:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: extra open patch boundary edges = {extra_open_edges}"
                )
            if degenerate_faces > 0:
                strict_gate_reasons.append(
                    f"adaptive_surface strict gate: degenerate patch faces = {degenerate_faces}"
                )

            if strict_gate_reasons:
                strict_gate_status = "blocked"
                commit_allowed = False
                blocking_reasons = tuple([*blocking_reasons, *strict_gate_reasons])

        def g1_quality_after() -> dict[str, object]:
            quality = mapping_from_sources("quality_after")
            if quality:
                return quality
            quality = nested_mapping_from_sources("relaxation", "quality_after")
            if quality:
                return quality
            quality = nested_mapping_from_sources("biharmonic_fairing", "quality_after")
            if quality:
                return quality
            return {}

        def g1_support_context() -> dict[str, object]:
            support = mapping_from_sources("support_context")
            if support:
                return support
            surface_context = mapping_from_sources("surface_context_decision")
            support = surface_context.get("support_context")
            if isinstance(support, dict):
                return support
            return {}

        def derived_g1_diagnostics() -> dict[str, object]:
            quality = g1_quality_after()
            support = g1_support_context()

            mean_value = to_float(quality.get("boundary_normal_mean_deviation_degrees"), None)
            max_value = to_float(quality.get("boundary_normal_max_deviation_degrees"), None)
            spread_value = to_float(support.get("normal_spread_degrees"), None)

            reasons: list[str] = []

            if mean_value is None and max_value is None and spread_value is None:
                status = "not_reported"
                reasons.append("no boundary normal continuity diagnostics were reported")
            else:
                feature_like = bool(spread_value is not None and spread_value > 75.0)

                if feature_like:
                    status = "feature_like"
                    reasons.append(
                        f"support normal spread is high ({spread_value:.3g}°); smooth G1 should not be forced"
                    )
                elif (
                    (mean_value is not None and mean_value > 25.0)
                    or (max_value is not None and max_value > 55.0)
                ):
                    status = "warning"
                    if mean_value is not None and mean_value > 25.0:
                        reasons.append(
                            f"boundary normal mean deviation is high ({mean_value:.3g}°)"
                        )
                    if max_value is not None and max_value > 55.0:
                        reasons.append(
                            f"boundary normal max deviation is high ({max_value:.3g}°)"
                        )
                elif (
                    (mean_value is not None and mean_value > 12.0)
                    or (max_value is not None and max_value > 30.0)
                    or (spread_value is not None and spread_value > 45.0)
                ):
                    status = "warning"
                    if mean_value is not None and mean_value > 12.0:
                        reasons.append(
                            f"boundary normal mean deviation is elevated ({mean_value:.3g}°)"
                        )
                    if max_value is not None and max_value > 30.0:
                        reasons.append(
                            f"boundary normal max deviation is elevated ({max_value:.3g}°)"
                        )
                    if spread_value is not None and spread_value > 45.0:
                        reasons.append(
                            f"support normal spread is elevated ({spread_value:.3g}°)"
                        )
                else:
                    status = "ok"

            return {
                "g1_status": status,
                "g1_boundary_normal_mean_deviation_degrees": float_display_value(mean_value),
                "g1_boundary_normal_max_deviation_degrees": float_display_value(max_value),
                "g1_support_normal_spread_degrees": float_display_value(spread_value),
                "g1_reasons": tuple(reasons),
            }

        g1_diagnostics = derived_g1_diagnostics()

        def adaptive_attempted_conservative_g1() -> object:
            attempted = diagnostic_value("adaptive_conservative_g1_attempted", None)
            if not is_empty(attempted):
                return attempted

            strategies = diagnostic_value("adaptive_attempted_strategies", ())
            try:
                return any("conservative_g1" in str(item) for item in strategies)
            except Exception:
                return "-"

        def adaptive_g1_policy_value() -> object:
            raw = diagnostic_value("adaptive_g1_relaxation_policy", None)
            if not is_empty(raw):
                return raw

            support = to_float(g1_diagnostics.get("g1_support_normal_spread_degrees"), None)
            if support is not None and support >= 75.0:
                return "feature_preserve_no_surface_pull"
            if support is not None and support >= 45.0:
                return "conservative_no_surface_pull"
            return "smooth_surface_guided_relaxation"

        def adaptive_g1_policy_reasons_value() -> object:
            raw = diagnostic_value("adaptive_g1_policy_reasons", None)
            if not is_empty(raw):
                return raw
            return diagnostic_value("g1_reasons", ())

        def adaptive_feature_context_value() -> object:
            raw = diagnostic_value("adaptive_feature_context_kind", None)
            if not is_empty(raw):
                return raw

            support = to_float(g1_diagnostics.get("g1_support_normal_spread_degrees"), None)
            boundary_count = to_int(diagnostic_value("adaptive_feature_boundary_vertex_count", 0), 0)

            if support is not None and support >= 75.0:
                return "tiny_feature_like" if 0 < boundary_count <= 6 else "feature_like"
            if support is not None and support >= 45.0:
                return "mixed_curved"
            return "smooth"

        def adaptive_feature_preservation_mode_value() -> object:
            raw = diagnostic_value("adaptive_feature_preservation_mode", None)
            if not is_empty(raw):
                return raw

            context = str(adaptive_feature_context_value())
            if context == "tiny_feature_like":
                return "preserve_minimal_topology"
            if context == "feature_like":
                return "preserve_feature_no_surface_pull"
            if context == "mixed_curved":
                return "conservative_no_surface_pull"
            return "allow_surface_guidance"

        def adaptive_feature_policy_reasons_value() -> object:
            raw = diagnostic_value("adaptive_feature_policy_reasons", None)
            if not is_empty(raw):
                return raw
            return diagnostic_value("adaptive_g1_policy_reasons", diagnostic_value("g1_reasons", ()))

        g1_gate_status = "not_applicable"
        g1_gate_reasons: list[str] = []

        if method_for_gate in {
            "adaptive_surface",
            "adaptive",
            "adaptive_surface_fill",
            "adaptive_surface_v2",
            "adaptive_surface_fill_v2",
        }:
            g1_gate_status = "not_reported"

            g1_mean = to_float(g1_diagnostics.get("g1_boundary_normal_mean_deviation_degrees"), None)
            g1_max = to_float(g1_diagnostics.get("g1_boundary_normal_max_deviation_degrees"), None)
            g1_support = to_float(g1_diagnostics.get("g1_support_normal_spread_degrees"), None)

            if g1_mean is not None or g1_max is not None or g1_support is not None:
                g1_gate_status = "passed"

                if g1_mean is not None and g1_mean > 35.0:
                    g1_gate_reasons.append(
                        f"adaptive_surface G1 gate: boundary normal mean deviation {g1_mean:.6g}° > 35°"
                    )
                if g1_max is not None and g1_max > 70.0:
                    g1_gate_reasons.append(
                        f"adaptive_surface G1 gate: boundary normal max deviation {g1_max:.6g}° > 70°"
                    )

                if g1_gate_reasons:
                    g1_gate_status = "blocked"
                    commit_allowed = False
                    blocking_reasons = tuple([*blocking_reasons, *g1_gate_reasons])
                elif g1_support is not None and g1_support >= 75.0:
                    g1_gate_status = "feature_like"
                    g1_gate_reasons.append(
                        f"adaptive_surface G1 gate: feature-like support normal spread {g1_support:.6g}°; smooth G1 was not forced"
                    )
                    warnings = tuple([*warnings, *g1_gate_reasons])
                elif g1_support is not None and g1_support >= 45.0:
                    g1_gate_status = "warning"
                    g1_gate_reasons.append(
                        f"adaptive_surface G1 gate: elevated support normal spread {g1_support:.6g}°; conservative relaxation is acceptable"
                    )
                    warnings = tuple([*warnings, *g1_gate_reasons])
                elif (
                    (g1_mean is not None and g1_mean > 20.0)
                    or (g1_max is not None and g1_max > 45.0)
                ):
                    g1_gate_status = "warning"
                    if g1_mean is not None and g1_mean > 20.0:
                        g1_gate_reasons.append(
                            f"adaptive_surface G1 gate: boundary normal mean deviation is elevated ({g1_mean:.6g}°)"
                        )
                    if g1_max is not None and g1_max > 45.0:
                        g1_gate_reasons.append(
                            f"adaptive_surface G1 gate: boundary normal max deviation is elevated ({g1_max:.6g}°)"
                        )
                    warnings = tuple([*warnings, *g1_gate_reasons])

        def derived_g2_diagnostics() -> dict[str, object]:
            raw_status = str(diagnostic_value("adaptive_curvature_status", "-")).strip().lower()
            raw_context = str(diagnostic_value("adaptive_curvature_context_kind", "-")).strip().lower()

            support_mean = to_float(diagnostic_value("adaptive_support_curvature_mean", None), None)
            patch_mean = to_float(diagnostic_value("adaptive_patch_curvature_mean", None), None)
            delta_mean = to_float(diagnostic_value("adaptive_curvature_delta_mean", None), None)
            delta_max = to_float(diagnostic_value("adaptive_curvature_delta_max", None), None)
            relative_delta = to_float(diagnostic_value("adaptive_curvature_relative_delta_mean", None), None)
            sign_consistency = to_bool(diagnostic_value("adaptive_curvature_sign_consistency", None), None)

            reasons: list[str] = []
            reasons.extend(text_tuple(diagnostic_value("adaptive_curvature_reasons", ())))

            if raw_status in {"-", "", "not_reported"}:
                status = "not_reported"
                if not reasons:
                    reasons.append("curvature diagnostics were not reported")
            elif raw_status == "feature_like" or raw_context == "mixed_curvature":
                status = "feature_like"
                if raw_context == "mixed_curvature":
                    reasons.append("support curvature is mixed; smooth G2 should not be forced")
            else:
                status = "ok"

                if sign_consistency is False:
                    status = "warning"
                    reasons.append("patch/support mean normal sign consistency is false")

                if (
                    relative_delta is not None
                    and relative_delta > 1.5
                    and delta_mean is not None
                    and delta_mean > 0.20
                ):
                    status = "warning"
                    reasons.append(f"relative curvature mean delta is high ({relative_delta:.6g})")

                if delta_max is not None and delta_max > 0.75:
                    status = "warning"
                    reasons.append(f"curvature max delta is elevated ({delta_max:.6g})")

                if (
                    support_mean is not None
                    and patch_mean is not None
                    and support_mean > 1.0e-6
                    and patch_mean > support_mean * 3.0
                    and delta_mean is not None
                    and delta_mean > 0.25
                ):
                    status = "warning"
                    reasons.append("patch curvature is much higher than support curvature")

            return {
                "g2_status": status,
                "g2_context": raw_context if raw_context not in {"", "-"} else "unknown",
                "g2_reasons": tuple(dict.fromkeys(reason for reason in reasons if reason)),
                "g2_support_curvature_mean": support_mean if support_mean is not None else "-",
                "g2_patch_curvature_mean": patch_mean if patch_mean is not None else "-",
                "g2_curvature_delta_mean": delta_mean if delta_mean is not None else "-",
                "g2_curvature_delta_max": delta_max if delta_max is not None else "-",
                "g2_curvature_relative_delta_mean": relative_delta if relative_delta is not None else "-",
                "g2_curvature_sign_consistency": sign_consistency if sign_consistency is not None else "-",
            }

        g2_diagnostics = derived_g2_diagnostics()


        def derived_adaptive_fairing_gate() -> dict[str, object]:
            """Adaptive-level acceptance diagnostics for backend fairing.

            H-ADAPT-5C3 is diagnostic-only. It does not replace the mesh yet.
            It reports whether the backend-accepted fairing also satisfies the
            adaptive displacement tolerance requested by the curvature proposal.
            """

            trial_status = str(
                diagnostic_value("adaptive_curvature_fairing_trial_status", "-")
            ).strip().lower()
            trial_action = str(
                diagnostic_value("adaptive_curvature_fairing_trial_action", "-")
            )
            trial_attempted = to_bool(
                diagnostic_value("adaptive_curvature_fairing_trial_attempted", None),
                None,
            )
            trial_accepted = to_bool(
                diagnostic_value("adaptive_curvature_fairing_trial_accepted", None),
                None,
            )
            trial_applied = to_bool(
                diagnostic_value("adaptive_curvature_fairing_trial_applied", None),
                None,
            )
            proposal_status = str(
                diagnostic_value("adaptive_curvature_fairing_status", "-")
            ).strip().lower()
            proposal_action = str(
                diagnostic_value("adaptive_curvature_fairing_action", "-")
            )
            proposal_limit = to_float(
                diagnostic_value("adaptive_curvature_fairing_max_displacement_factor", None),
                None,
            )
            movement_ratio = to_float(
                diagnostic_value(
                    "adaptive_curvature_fairing_trial_movement_to_context_edge_ratio",
                    None,
                ),
                None,
            )
            max_displacement = to_float(
                diagnostic_value("adaptive_curvature_fairing_trial_max_displacement", None),
                None,
            )
            mean_displacement = to_float(
                diagnostic_value("adaptive_curvature_fairing_trial_mean_displacement", None),
                None,
            )

            reasons: list[str] = []

            if proposal_status != "proposal_ready":
                status = "not_requested"
                action = "keep_current_patch"
                accepted_by_gate = False
                reasons.append(f"fairing proposal status is {proposal_status or 'unknown'}")
            elif trial_status in {"-", "", "not_requested"}:
                status = "not_attempted"
                action = "keep_current_patch"
                accepted_by_gate = False
                reasons.append("backend fairing trial was not requested or not reported")
            elif trial_attempted is False:
                status = "not_attempted"
                action = "keep_current_patch"
                accepted_by_gate = False
                reasons.append("backend fairing trial was not attempted")
            elif trial_accepted is False or trial_status in {"rejected", "not_eligible"}:
                status = "passed"
                action = "keep_current_patch"
                accepted_by_gate = False
                reasons.append("backend fairing was rejected or not eligible; current patch is retained")
            else:
                accepted_by_gate = True
                status = "passed"
                action = "accept_backend_fairing"

                if proposal_limit is not None and movement_ratio is not None:
                    # H-ADAPT-5C5B:
                    # The low-displacement fairing scales to the requested limit.
                    # Allow a tiny numerical tolerance so 0.03000000000000006
                    # is treated as equal to 0.03, not as a policy failure.
                    movement_tolerance = max(1.0e-9, abs(float(proposal_limit)) * 1.0e-6)
                    movement_excess = float(movement_ratio) - float(proposal_limit)

                    if movement_excess > movement_tolerance:
                        accepted_by_gate = False
                        status = "warning"
                        action = "require_stricter_lower_displacement_trial"
                        reasons.append(
                            "backend fairing movement ratio "
                            f"{movement_ratio:.6g} exceeds adaptive proposal limit {proposal_limit:.6g}"
                        )

                # A secondary soft guard: even if no explicit ratio is present,
                # do not silently bless very large absolute movement.
                if (
                    max_displacement is not None
                    and mean_displacement is not None
                    and max_displacement > max(mean_displacement * 3.0, 1.0e-9)
                ):
                    if status == "passed":
                        status = "warning"
                        action = "inspect_backend_fairing_displacement"
                        accepted_by_gate = False
                    reasons.append(
                        "backend fairing max displacement is much larger than mean displacement"
                    )

                if status == "passed":
                    reasons.append("backend fairing satisfies adaptive displacement gate")

            return {
                "status": status,
                "action": action,
                "accepted_by_gate": bool(accepted_by_gate),
                "trial_status": trial_status or "-",
                "trial_action": trial_action,
                "proposal_status": proposal_status or "-",
                "proposal_action": proposal_action,
                "proposal_limit": proposal_limit if proposal_limit is not None else "-",
                "movement_ratio": movement_ratio if movement_ratio is not None else "-",
                "max_displacement": max_displacement if max_displacement is not None else "-",
                "mean_displacement": mean_displacement if mean_displacement is not None else "-",
                "trial_applied": trial_applied if trial_applied is not None else "-",
                "reasons": tuple(dict.fromkeys(reason for reason in reasons if reason)),
            }

        adaptive_fairing_gate = derived_adaptive_fairing_gate()


        def derived_g2_commit_gate() -> dict[str, object]:
            """Convert G2/C2 curvature diagnostics into commit policy.

            H-ADAPT-5D is policy-only. It does not change preview geometry.
            """

            g2_status_raw = str(g2_diagnostics.get("g2_status") or "-").strip().lower()
            g2_context = str(g2_diagnostics.get("g2_context") or "-").strip().lower()
            feature_mode = str(adaptive_feature_preservation_mode_value() or "-").strip()
            feature_context = str(adaptive_feature_context_value() or "-").strip()

            reasons: list[str] = []
            reasons.extend(text_tuple(g2_diagnostics.get("g2_reasons", ())))

            commit_allowed_by_gate = True
            action = "allow_commit"

            if g2_status_raw in {"ok", "passed"}:
                gate_status = "passed"
                reasons.append("G2 curvature diagnostics passed")
            elif g2_status_raw == "warning":
                gate_status = "warning"
                action = "allow_commit_with_g2_warning"
                if not reasons:
                    reasons.append("G2 curvature diagnostics reported a warning")
            elif g2_status_raw == "feature_like":
                feature_preservation_active = feature_mode in {
                    "preserve_feature_no_surface_pull",
                    "preserve_minimal_topology",
                    "feature_preserve_no_surface_pull",
                    "feature_preserve_minimal_topology",
                    "conservative_no_surface_pull",
                } or feature_context in {
                    "feature_like",
                    "tiny_feature_like",
                    "mixed_curved",
                }

                if feature_preservation_active:
                    gate_status = "feature_like_allowed"
                    action = "allow_commit_preserve_feature_context"
                    reasons.append(
                        "G2 feature-like context is allowed because feature/conservative preservation is active"
                    )
                else:
                    gate_status = "warning"
                    action = "allow_commit_with_g2_feature_warning"
                    reasons.append(
                        "G2 feature-like context reported without an active feature-preservation mode"
                    )
            elif g2_status_raw == "blocked":
                gate_status = "blocked"
                action = "block_commit"
                commit_allowed_by_gate = False
                if not reasons:
                    reasons.append("G2 curvature diagnostics blocked commit")
            elif g2_status_raw in {"not_reported", "-", ""}:
                gate_status = "warning"
                action = "allow_commit_with_missing_g2_warning"
                reasons.append("G2 curvature diagnostics were not reported")
            else:
                gate_status = "warning"
                action = "allow_commit_with_unknown_g2_warning"
                reasons.append(f"G2 curvature diagnostics reported unknown status {g2_status_raw!r}")

            return {
                "status": gate_status,
                "action": action,
                "commit_allowed": bool(commit_allowed_by_gate),
                "source_status": g2_status_raw or "-",
                "context": g2_context or "-",
                "feature_context": feature_context or "-",
                "feature_preservation_mode": feature_mode or "-",
                "reasons": tuple(dict.fromkeys(reason for reason in reasons if reason)),
            }

        g2_commit_gate = derived_g2_commit_gate()

        if method_for_gate in {
            "adaptive_surface",
            "adaptive",
            "adaptive_surface_fill",
            "adaptive_surface_v2",
            "adaptive_surface_fill_v2",
        }:
            g2_gate_reasons = tuple(
                str(reason)
                for reason in g2_commit_gate.get("reasons", ())
                if str(reason)
            )

            if not bool(g2_commit_gate.get("commit_allowed", True)):
                commit_allowed = False
                blocking_reasons = tuple([*blocking_reasons, *g2_gate_reasons])
            elif str(g2_commit_gate.get("status", "")).lower() in {
                "warning",
                "feature_like_allowed",
            }:
                warnings = tuple([*warnings, *g2_gate_reasons])


        def derived_end_layer_rerun_gate() -> dict[str, object]:
            """Policy gate for allowing one bounded end-layer rerun evaluation.

            H-ADAPT-5E3 is policy-only. It does not execute the rerun and does
            not replace selected geometry.
            """

            status_raw = str(
                diagnostic_value("adaptive_end_layer_status", "-")
            ).strip().lower()
            action_raw = str(
                diagnostic_value("adaptive_end_layer_action", "-")
            ).strip().lower()

            local_region_available = to_bool(
                diagnostic_value("adaptive_end_layer_local_region_available", None),
                None,
            )
            patch_available = to_bool(
                diagnostic_value("adaptive_end_layer_patch_available", None),
                None,
            )
            reference_patch_available = to_bool(
                diagnostic_value("adaptive_end_layer_reference_patch_available", None),
                None,
            )
            refinement_recommended = to_bool(
                diagnostic_value("adaptive_end_layer_refinement_recommended", None),
                None,
            )

            geometry_deviation_max = to_float(
                diagnostic_value("adaptive_end_layer_geometry_deviation_max", None),
                None,
            )
            geometry_deviation_mean = to_float(
                diagnostic_value("adaptive_end_layer_geometry_deviation_mean", None),
                None,
            )
            curvature_deviation_mean = to_float(
                diagnostic_value("adaptive_end_layer_curvature_deviation_mean", None),
                None,
            )
            curvature_relative_deviation_mean = to_float(
                diagnostic_value(
                    "adaptive_end_layer_curvature_relative_deviation_mean",
                    None,
                ),
                None,
            )

            reasons: list[str] = []
            reasons.extend(text_tuple(diagnostic_value("adaptive_end_layer_reasons", ())))

            strict_status = str(
                diagnostic_value("strict_seam_topology_gate", strict_gate_status)
            ).strip().lower()
            # H-ADAPT-5E3B:
            # Some backend metadata still reports raw quality_status="unknown"
            # even when derived_quality_status() correctly resolves quality_after
            # to "ok". The end-layer rerun gate must use the same derived
            # policy status that the GUI already displays.
            quality = str(
                clean_display_value(derived_quality_status(), "not_reported")
            ).strip().lower()
            g2_gate_allowed = to_bool(
                diagnostic_value("g2_gate_commit_allowed", g2_commit_gate.get("commit_allowed", True)),
                True,
            )
            fairing_gate_status = str(
                diagnostic_value("adaptive_curvature_fairing_gate_status", "-")
            ).strip().lower()

            blockers: list[str] = []

            if local_region_available is not True:
                blockers.append("end-layer local region is not available")
            if patch_available is not True:
                blockers.append("end-layer selected patch region is not available")
            if reference_patch_available is not True:
                blockers.append("end-layer reference overlay patch is not available")
            if refinement_recommended is not True:
                blockers.append("end-layer refinement is not recommended")
            if strict_status not in {"passed", "not_applicable"}:
                blockers.append(f"strict seam/topology gate is {strict_status}")
            if quality not in {"ok", "warning"}:
                blockers.append(f"quality status is {quality}")
            if g2_gate_allowed is False:
                blockers.append("G2 gate does not allow commit")
            if fairing_gate_status not in {"passed", "not_requested", "-", ""}:
                blockers.append(f"adaptive fairing gate is {fairing_gate_status}")

            if blockers:
                gate_status = "blocked"
                gate_action = "do_not_run_end_layer_rerun"
                rerun_allowed = False
                reasons.extend(blockers)
            else:
                # The reference overlay exists and all safety gates are clean.
                # We still require a measurable signal before allowing a rerun
                # evaluation, otherwise 5E4 would waste work.
                measurable_geometry_signal = (
                    geometry_deviation_max is not None
                    and geometry_deviation_max > 1.0e-6
                )
                measurable_curvature_signal = (
                    curvature_relative_deviation_mean is not None
                    and curvature_relative_deviation_mean > 1.0e-4
                ) or (
                    curvature_deviation_mean is not None
                    and curvature_deviation_mean > 1.0e-6
                )

                if measurable_geometry_signal or measurable_curvature_signal:
                    gate_status = "ready"
                    gate_action = "allow_bounded_rerun_evaluation"
                    rerun_allowed = True
                    reasons.append(
                        "reference overlay is available and safety gates allow one bounded rerun evaluation"
                    )
                else:
                    gate_status = "warning"
                    gate_action = "skip_rerun_no_measurable_deviation"
                    rerun_allowed = False
                    reasons.append(
                        "reference overlay is available but deviation signal is too small for rerun evaluation"
                    )

            return {
                "status": gate_status,
                "action": gate_action,
                "rerun_allowed": bool(rerun_allowed),
                "source_status": status_raw or "-",
                "source_action": action_raw or "-",
                "geometry_deviation_mean": geometry_deviation_mean if geometry_deviation_mean is not None else "-",
                "geometry_deviation_max": geometry_deviation_max if geometry_deviation_max is not None else "-",
                "curvature_deviation_mean": curvature_deviation_mean if curvature_deviation_mean is not None else "-",
                "curvature_relative_deviation_mean": (
                    curvature_relative_deviation_mean
                    if curvature_relative_deviation_mean is not None
                    else "-"
                ),
                "reasons": tuple(dict.fromkeys(reason for reason in reasons if reason)),
            }

        end_layer_rerun_gate = derived_end_layer_rerun_gate()

        return {
            "commit_allowed": commit_allowed,
            "commit_blocking_reasons": blocking_reasons,
            "commit_warnings": warnings,
            "public_method": str(first_value("public_method", summary.get("method", "-"))),
            "method": str(first_value("method", summary.get("method", "-"))),
            "backend": str(first_value("backend", summary.get("backend", summary.get("method", "-")))),
            "adaptive_stage": str(first_value("adaptive_stage", summary.get("backend", "-"))),
            "adaptive_diagnostics": raw_diagnostics,
            "adaptive_context_kind": clean_display_value(diagnostic_value("adaptive_context_kind", diagnostic_value("context_kind", "-")), "unknown"),
            "adaptive_context_confidence": clean_display_value(diagnostic_value("adaptive_context_confidence", diagnostic_value("context_confidence", "-"))),
            "selected_seed_strategy": diagnostic_value("selected_seed_strategy", diagnostic_value("adaptive_selected_strategy", "-")),
            "adaptive_controller": diagnostic_value("adaptive_controller", "-"),
            "adaptive_surface_v2_status": diagnostic_value("adaptive_surface_v2_status", "-"),
            "adaptive_surface_v2_case": diagnostic_value("adaptive_surface_v2_case", "-"),
            "adaptive_surface_v2_action": diagnostic_value("adaptive_surface_v2_action", "-"),
            "adaptive_surface_v2_block_legacy_selection": diagnostic_value("adaptive_surface_v2_block_legacy_selection", "-"),
            "adaptive_surface_v2_allow_confidence_target_delta": diagnostic_value("adaptive_surface_v2_allow_confidence_target_delta", "-"),
            "adaptive_surface_v2_allow_local_anisotropic_correction": diagnostic_value("adaptive_surface_v2_allow_local_anisotropic_correction", "-"),
            "adaptive_surface_v2_require_new_seed": diagnostic_value("adaptive_surface_v2_require_new_seed", "-"),
            "adaptive_surface_v2_recommended_seed_family": diagnostic_value("adaptive_surface_v2_recommended_seed_family", "-"),
            "adaptive_surface_v2_recommended_target_policy": diagnostic_value("adaptive_surface_v2_recommended_target_policy", "-"),
            "adaptive_surface_v2_reasons": diagnostic_value("adaptive_surface_v2_reasons", ()),
            "adaptive_surface_v2_seed_plan_status": diagnostic_value("adaptive_surface_v2_seed_plan_status", "-"),
            "adaptive_surface_v2_seed_plan_action": diagnostic_value("adaptive_surface_v2_seed_plan_action", "-"),
            "adaptive_surface_v2_seed_plan_build_required": diagnostic_value("adaptive_surface_v2_seed_plan_build_required", "-"),
            "adaptive_surface_v2_seed_family": diagnostic_value("adaptive_surface_v2_seed_family", "-"),
            "adaptive_surface_v2_orientation_case": diagnostic_value("adaptive_surface_v2_orientation_case", "-"),
            "adaptive_surface_v2_orientation_action": diagnostic_value("adaptive_surface_v2_orientation_action", "-"),
            "adaptive_surface_v2_target_policy": diagnostic_value("adaptive_surface_v2_target_policy", "-"),
            "adaptive_surface_v2_curvature_policy": diagnostic_value("adaptive_surface_v2_curvature_policy", "-"),
            "adaptive_surface_v2_confidence_policy": diagnostic_value("adaptive_surface_v2_confidence_policy", "-"),
            "adaptive_surface_v2_support_context_policy": diagnostic_value("adaptive_surface_v2_support_context_policy", "-"),
            "adaptive_surface_v2_boundary_normal_mean_deviation": diagnostic_value("adaptive_surface_v2_boundary_normal_mean_deviation", "-"),
            "adaptive_surface_v2_boundary_normal_max_deviation": diagnostic_value("adaptive_surface_v2_boundary_normal_max_deviation", "-"),
            "adaptive_surface_v2_support_normal_spread": diagnostic_value("adaptive_surface_v2_support_normal_spread", "-"),
            "adaptive_surface_v2_seed_projection_mean_ratio": diagnostic_value("adaptive_surface_v2_seed_projection_mean_ratio", "-"),
            "adaptive_surface_v2_seed_projection_max_ratio": diagnostic_value("adaptive_surface_v2_seed_projection_max_ratio", "-"),
            "adaptive_surface_v2_target_confidence": diagnostic_value("adaptive_surface_v2_target_confidence", "-"),
            "adaptive_surface_v2_target_low_confidence_fraction": diagnostic_value("adaptive_surface_v2_target_low_confidence_fraction", "-"),
            "adaptive_surface_v2_curvature_relative_delta_mean": diagnostic_value("adaptive_surface_v2_curvature_relative_delta_mean", "-"),
            "adaptive_surface_v2_curvature_sign_consistency": diagnostic_value("adaptive_surface_v2_curvature_sign_consistency", "-"),
            "adaptive_surface_v2_seed_plan_reasons": diagnostic_value("adaptive_surface_v2_seed_plan_reasons", ()),
            "adaptive_surface_v2_seed_prototype_status": diagnostic_value("adaptive_surface_v2_seed_prototype_status", "-"),
            "adaptive_surface_v2_seed_prototype_action": diagnostic_value("adaptive_surface_v2_seed_prototype_action", "-"),
            "adaptive_surface_v2_seed_prototype_build_required": diagnostic_value("adaptive_surface_v2_seed_prototype_build_required", "-"),
            "adaptive_surface_v2_seed_prototype_family": diagnostic_value("adaptive_surface_v2_seed_prototype_family", "-"),
            "adaptive_surface_v2_seed_prototype_geometry_status": diagnostic_value("adaptive_surface_v2_seed_prototype_geometry_status", "-"),
            "adaptive_surface_v2_seed_prototype_orientation_status": diagnostic_value("adaptive_surface_v2_seed_prototype_orientation_status", "-"),
            "adaptive_surface_v2_seed_prototype_orientation_action": diagnostic_value("adaptive_surface_v2_seed_prototype_orientation_action", "-"),
            "adaptive_surface_v2_seed_prototype_orientation_confidence": diagnostic_value("adaptive_surface_v2_seed_prototype_orientation_confidence", "-"),
            "adaptive_surface_v2_seed_prototype_side_score_mean": diagnostic_value("adaptive_surface_v2_seed_prototype_side_score_mean", "-"),
            "adaptive_surface_v2_seed_prototype_side_score_max": diagnostic_value("adaptive_surface_v2_seed_prototype_side_score_max", "-"),
            "adaptive_surface_v2_seed_prototype_normal_sign_status": diagnostic_value("adaptive_surface_v2_seed_prototype_normal_sign_status", "-"),
            "adaptive_surface_v2_seed_prototype_support_context_policy": diagnostic_value("adaptive_surface_v2_seed_prototype_support_context_policy", "-"),
            "adaptive_surface_v2_seed_prototype_target_policy": diagnostic_value("adaptive_surface_v2_seed_prototype_target_policy", "-"),
            "adaptive_surface_v2_seed_prototype_curvature_policy": diagnostic_value("adaptive_surface_v2_seed_prototype_curvature_policy", "-"),
            "adaptive_surface_v2_seed_prototype_confidence_policy": diagnostic_value("adaptive_surface_v2_seed_prototype_confidence_policy", "-"),
            "adaptive_surface_v2_seed_prototype_seed_projection_mean_ratio": diagnostic_value("adaptive_surface_v2_seed_prototype_seed_projection_mean_ratio", "-"),
            "adaptive_surface_v2_seed_prototype_seed_projection_max_ratio": diagnostic_value("adaptive_surface_v2_seed_prototype_seed_projection_max_ratio", "-"),
            "adaptive_surface_v2_seed_prototype_target_confidence": diagnostic_value("adaptive_surface_v2_seed_prototype_target_confidence", "-"),
            "adaptive_surface_v2_seed_prototype_target_low_confidence_fraction": diagnostic_value("adaptive_surface_v2_seed_prototype_target_low_confidence_fraction", "-"),
            "adaptive_surface_v2_seed_prototype_support_normal_spread": diagnostic_value("adaptive_surface_v2_seed_prototype_support_normal_spread", "-"),
            "adaptive_surface_v2_seed_prototype_curvature_relative_delta_mean": diagnostic_value("adaptive_surface_v2_seed_prototype_curvature_relative_delta_mean", "-"),
            "adaptive_surface_v2_seed_prototype_curvature_sign_consistency": diagnostic_value("adaptive_surface_v2_seed_prototype_curvature_sign_consistency", "-"),
            "adaptive_surface_v2_seed_prototype_reasons": diagnostic_value("adaptive_surface_v2_seed_prototype_reasons", ()),
            "adaptive_surface_v2_seed_candidate_status": diagnostic_value("adaptive_surface_v2_seed_candidate_status", "-"),
            "adaptive_surface_v2_seed_candidate_action": diagnostic_value("adaptive_surface_v2_seed_candidate_action", "-"),
            "adaptive_surface_v2_seed_candidate_family": diagnostic_value("adaptive_surface_v2_seed_candidate_family", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_status", "-"),
            "adaptive_surface_v2_seed_candidate_selectable": diagnostic_value("adaptive_surface_v2_seed_candidate_selectable", "-"),
            "adaptive_surface_v2_seed_candidate_curvature_normal_field_status": diagnostic_value("adaptive_surface_v2_seed_candidate_curvature_normal_field_status", "-"),
            "adaptive_surface_v2_seed_candidate_support_filter": diagnostic_value("adaptive_surface_v2_seed_candidate_support_filter", "-"),
            "adaptive_surface_v2_seed_candidate_frame_policy": diagnostic_value("adaptive_surface_v2_seed_candidate_frame_policy", "-"),
            "adaptive_surface_v2_seed_candidate_target_policy": diagnostic_value("adaptive_surface_v2_seed_candidate_target_policy", "-"),
            "adaptive_surface_v2_seed_candidate_density_policy": diagnostic_value("adaptive_surface_v2_seed_candidate_density_policy", "-"),
            "adaptive_surface_v2_seed_candidate_acceptance_policy": diagnostic_value("adaptive_surface_v2_seed_candidate_acceptance_policy", "-"),
            "adaptive_surface_v2_seed_candidate_boundary_vertices": diagnostic_value("adaptive_surface_v2_seed_candidate_boundary_vertices", "-"),
            "adaptive_surface_v2_seed_candidate_legacy_seed_vertices": diagnostic_value("adaptive_surface_v2_seed_candidate_legacy_seed_vertices", "-"),
            "adaptive_surface_v2_seed_candidate_planned_seed_vertices": diagnostic_value("adaptive_surface_v2_seed_candidate_planned_seed_vertices", "-"),
            "adaptive_surface_v2_seed_candidate_planned_support_rings": diagnostic_value("adaptive_surface_v2_seed_candidate_planned_support_rings", "-"),
            "adaptive_surface_v2_seed_candidate_planned_interior_rings": diagnostic_value("adaptive_surface_v2_seed_candidate_planned_interior_rings", "-"),
            "adaptive_surface_v2_seed_candidate_normal_continuity_mismatch_score": diagnostic_value("adaptive_surface_v2_seed_candidate_normal_continuity_mismatch_score", "-"),
            "adaptive_surface_v2_seed_candidate_target_confidence": diagnostic_value("adaptive_surface_v2_seed_candidate_target_confidence", "-"),
            "adaptive_surface_v2_seed_candidate_target_low_confidence_fraction": diagnostic_value("adaptive_surface_v2_seed_candidate_target_low_confidence_fraction", "-"),
            "adaptive_surface_v2_seed_candidate_seed_projection_max_ratio": diagnostic_value("adaptive_surface_v2_seed_candidate_seed_projection_max_ratio", "-"),
            "adaptive_surface_v2_seed_candidate_curvature_relative_delta_mean": diagnostic_value("adaptive_surface_v2_seed_candidate_curvature_relative_delta_mean", "-"),
            "adaptive_surface_v2_seed_candidate_curvature_sign_consistency": diagnostic_value("adaptive_surface_v2_seed_candidate_curvature_sign_consistency", "-"),
            "adaptive_surface_v2_seed_candidate_reasons": diagnostic_value("adaptive_surface_v2_seed_candidate_reasons", ()),
            "adaptive_surface_v2_seed_candidate_geometry_action": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_action", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_available": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_available", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_applied": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_applied", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_selected": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_selected", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_family": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_family", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_mode": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_mode", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_face_count": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_face_count", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_vertex_count": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_vertex_count", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_moved_vertex_count": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_moved_vertex_count", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_movement_mean": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_movement_mean", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_movement_max": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_movement_max", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_topology_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_topology_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_topology_reasons": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_topology_reasons", ()),
            "adaptive_surface_v2_seed_candidate_geometry_reasons": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_reasons", ()),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_action": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_action", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_limit": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_limit", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_limit": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_limit", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_policy": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_policy", "-"),
            "adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons": diagnostic_value("adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons", ()),
            "adaptive_controller_schema_version": diagnostic_value("adaptive_controller_schema_version", "-"),
            "adaptive_fallback_used": diagnostic_value("adaptive_fallback_used", "-"),
            "adaptive_fallback_reason": diagnostic_value("adaptive_fallback_reason", "-"),
            "adaptive_fallback_error": diagnostic_value("adaptive_fallback_error", ""),
            "adaptive_attempted_strategies": diagnostic_value("adaptive_attempted_strategies", ()),
            "adaptive_primary_seam_decision": diagnostic_value("adaptive_primary_seam_decision", {}),
            "adaptive_fallback_seam_decision": diagnostic_value("adaptive_fallback_seam_decision", {}),
            "seam_status": diagnostic_value("seam_status", "-"),
            "seam_coverage_ratio": diagnostic_value("seam_coverage_ratio", nested_diagnostic_value("seam_constraint_report", "seam_coverage_ratio", "-")),
            "seam_missing_edge_count": diagnostic_value("seam_missing_edge_count", nested_diagnostic_value("seam_constraint_report", "missing_seam_edge_count", "-")),
            "seam_overused_edge_count": diagnostic_value("seam_overused_edge_count", nested_diagnostic_value("seam_constraint_report", "overused_seam_edge_count", "-")),
            "seam_weak_edge_count": diagnostic_value("seam_weak_edge_count", nested_diagnostic_value("seam_constraint_report", "weak_seam_edge_count", "-")),
            "seam_problem_edge_count": diagnostic_value("seam_problem_edge_count", nested_diagnostic_value("seam_recovery_decision", "problem_edge_count", "-")),
            "seam_recovery_required": diagnostic_value("seam_recovery_required", nested_diagnostic_value("seam_recovery_decision", "recovery_required", "-")),
            "seam_recovery_strategy": diagnostic_value("seam_recovery_strategy", nested_diagnostic_value("seam_recovery_decision", "strategy", "-")),
            "seam_problem_edges": diagnostic_value("seam_problem_edges", nested_diagnostic_value("seam_recovery_decision", "problem_edges", "-")),
            "seam_problem_edge_runs": diagnostic_value("seam_problem_edge_runs", nested_diagnostic_value("seam_recovery_decision", "problem_edge_runs", "-")),
            "quality_status": clean_display_value(derived_quality_status(), "not_reported"),
            "relaxation_status": clean_display_value(diagnostic_value("relaxation_status", "-")),
            "density_status": clean_display_value(diagnostic_value("density_status", "-")),
            "strict_seam_topology_gate": strict_gate_status,
            "strict_gate_reasons": tuple(strict_gate_reasons),
            "g1_status": g1_diagnostics.get("g1_status"),
            "g1_boundary_normal_mean_deviation_degrees": g1_diagnostics.get("g1_boundary_normal_mean_deviation_degrees"),
            "g1_boundary_normal_max_deviation_degrees": g1_diagnostics.get("g1_boundary_normal_max_deviation_degrees"),
            "g1_support_normal_spread_degrees": g1_diagnostics.get("g1_support_normal_spread_degrees"),
            "g1_reasons": g1_diagnostics.get("g1_reasons", ()),
            "g1_gate_status": g1_gate_status,
            "g1_gate_reasons": tuple(g1_gate_reasons),
            "adaptive_g1_relaxation_policy": adaptive_g1_policy_value(),
            "adaptive_g1_policy_reasons": adaptive_g1_policy_reasons_value(),
            "adaptive_feature_context_kind": adaptive_feature_context_value(),
            "adaptive_feature_preservation_mode": adaptive_feature_preservation_mode_value(),
            "adaptive_feature_policy_reasons": adaptive_feature_policy_reasons_value(),
            "adaptive_feature_recommended_action": diagnostic_value("adaptive_feature_recommended_action", "-"),
            "adaptive_feature_density_mode": diagnostic_value("adaptive_feature_density_mode", "-"),
            "adaptive_feature_smooth_context_valid": diagnostic_value("adaptive_feature_smooth_context_valid", "-"),
            "adaptive_feature_boundary_vertex_count": diagnostic_value("adaptive_feature_boundary_vertex_count", "-"),
            "adaptive_curvature_status": diagnostic_value("adaptive_curvature_status", "-"),
            "adaptive_curvature_context_kind": diagnostic_value("adaptive_curvature_context_kind", "-"),
            "adaptive_curvature_estimator": diagnostic_value("adaptive_curvature_estimator", "-"),
            "adaptive_support_curvature_mean": diagnostic_value("adaptive_support_curvature_mean", "-"),
            "adaptive_support_curvature_max": diagnostic_value("adaptive_support_curvature_max", "-"),
            "adaptive_support_curvature_std": diagnostic_value("adaptive_support_curvature_std", "-"),
            "adaptive_patch_curvature_mean": diagnostic_value("adaptive_patch_curvature_mean", "-"),
            "adaptive_patch_curvature_max": diagnostic_value("adaptive_patch_curvature_max", "-"),
            "adaptive_patch_curvature_std": diagnostic_value("adaptive_patch_curvature_std", "-"),
            "adaptive_curvature_delta_mean": diagnostic_value("adaptive_curvature_delta_mean", "-"),
            "adaptive_curvature_delta_max": diagnostic_value("adaptive_curvature_delta_max", "-"),
            "adaptive_curvature_relative_delta_mean": diagnostic_value("adaptive_curvature_relative_delta_mean", "-"),
            "adaptive_curvature_sign_consistency": diagnostic_value("adaptive_curvature_sign_consistency", "-"),
            "adaptive_curvature_support_sample_count": diagnostic_value("adaptive_curvature_support_sample_count", "-"),
            "adaptive_curvature_patch_sample_count": diagnostic_value("adaptive_curvature_patch_sample_count", "-"),
            "adaptive_curvature_reasons": diagnostic_value("adaptive_curvature_reasons", ()),
            "g2_status": g2_diagnostics.get("g2_status"),
            "g2_context": g2_diagnostics.get("g2_context"),
            "g2_reasons": g2_diagnostics.get("g2_reasons", ()),
            "g2_support_curvature_mean": g2_diagnostics.get("g2_support_curvature_mean"),
            "g2_patch_curvature_mean": g2_diagnostics.get("g2_patch_curvature_mean"),
            "g2_curvature_delta_mean": g2_diagnostics.get("g2_curvature_delta_mean"),
            "g2_curvature_delta_max": g2_diagnostics.get("g2_curvature_delta_max"),
            "g2_curvature_relative_delta_mean": g2_diagnostics.get("g2_curvature_relative_delta_mean"),
            "g2_curvature_sign_consistency": g2_diagnostics.get("g2_curvature_sign_consistency"),
            "g2_gate_status": g2_commit_gate.get("status"),
            "g2_gate_action": g2_commit_gate.get("action"),
            "g2_gate_commit_allowed": g2_commit_gate.get("commit_allowed"),
            "g2_gate_reasons": g2_commit_gate.get("reasons", ()),
            "g2_gate_source_status": g2_commit_gate.get("source_status"),
            "g2_gate_context": g2_commit_gate.get("context"),
            "g2_gate_feature_context": g2_commit_gate.get("feature_context"),
            "g2_gate_feature_preservation_mode": g2_commit_gate.get("feature_preservation_mode"),
            "adaptive_curvature_fairing_status": diagnostic_value("adaptive_curvature_fairing_status", "-"),
            "adaptive_curvature_fairing_action": diagnostic_value("adaptive_curvature_fairing_action", "-"),
            "adaptive_curvature_fairing_eligible": diagnostic_value("adaptive_curvature_fairing_eligible", "-"),
            "adaptive_curvature_fairing_needed": diagnostic_value("adaptive_curvature_fairing_needed", "-"),
            "adaptive_curvature_fairing_strength": diagnostic_value("adaptive_curvature_fairing_strength", "-"),
            "adaptive_curvature_fairing_iterations": diagnostic_value("adaptive_curvature_fairing_iterations", "-"),
            "adaptive_curvature_fairing_max_displacement_factor": diagnostic_value("adaptive_curvature_fairing_max_displacement_factor", "-"),
            "adaptive_curvature_fairing_reasons": diagnostic_value("adaptive_curvature_fairing_reasons", ()),
            "adaptive_curvature_fairing_trial_status": diagnostic_value("adaptive_curvature_fairing_trial_status", "-"),
            "adaptive_curvature_fairing_trial_action": diagnostic_value("adaptive_curvature_fairing_trial_action", "-"),
            "adaptive_curvature_fairing_trial_attempted": diagnostic_value("adaptive_curvature_fairing_trial_attempted", "-"),
            "adaptive_curvature_fairing_trial_applied": diagnostic_value("adaptive_curvature_fairing_trial_applied", "-"),
            "adaptive_curvature_fairing_trial_accepted": diagnostic_value("adaptive_curvature_fairing_trial_accepted", "-"),
            "adaptive_curvature_fairing_trial_mode": diagnostic_value("adaptive_curvature_fairing_trial_mode", "-"),
            "adaptive_curvature_fairing_trial_reasons": diagnostic_value("adaptive_curvature_fairing_trial_reasons", ()),
            "adaptive_curvature_fairing_trial_notes": diagnostic_value("adaptive_curvature_fairing_trial_notes", ()),
            "adaptive_curvature_fairing_trial_error": diagnostic_value("adaptive_curvature_fairing_trial_error", ""),
            "adaptive_curvature_fairing_trial_max_displacement": diagnostic_value("adaptive_curvature_fairing_trial_max_displacement", "-"),
            "adaptive_curvature_fairing_trial_mean_displacement": diagnostic_value("adaptive_curvature_fairing_trial_mean_displacement", "-"),
            "adaptive_curvature_fairing_trial_movement_to_context_edge_ratio": diagnostic_value("adaptive_curvature_fairing_trial_movement_to_context_edge_ratio", "-"),
            "adaptive_curvature_fairing_gate_status": adaptive_fairing_gate.get("status"),
            "adaptive_curvature_fairing_gate_action": adaptive_fairing_gate.get("action"),
            "adaptive_curvature_fairing_accepted_by_adaptive_gate": adaptive_fairing_gate.get("accepted_by_gate"),
            "adaptive_curvature_fairing_gate_reasons": adaptive_fairing_gate.get("reasons", ()),
            "adaptive_curvature_fairing_gate_proposal_limit": adaptive_fairing_gate.get("proposal_limit"),
            "adaptive_curvature_fairing_gate_movement_ratio": adaptive_fairing_gate.get("movement_ratio"),
            "adaptive_curvature_fairing_gate_max_displacement": adaptive_fairing_gate.get("max_displacement"),
            "adaptive_curvature_fairing_gate_mean_displacement": adaptive_fairing_gate.get("mean_displacement"),
            "adaptive_end_layer_status": diagnostic_value("adaptive_end_layer_status", "-"),
            "adaptive_end_layer_action": diagnostic_value("adaptive_end_layer_action", "-"),
            "adaptive_end_layer_local_region_available": diagnostic_value("adaptive_end_layer_local_region_available", "-"),
            "adaptive_end_layer_patch_available": diagnostic_value("adaptive_end_layer_patch_available", "-"),
            "adaptive_end_layer_reference_patch_available": diagnostic_value("adaptive_end_layer_reference_patch_available", "-"),
            "adaptive_end_layer_support_ring_count": diagnostic_value("adaptive_end_layer_support_ring_count", "-"),
            "adaptive_end_layer_support_vertex_count": diagnostic_value("adaptive_end_layer_support_vertex_count", "-"),
            "adaptive_end_layer_patch_vertex_count": diagnostic_value("adaptive_end_layer_patch_vertex_count", "-"),
            "adaptive_end_layer_curvature_deviation_mean": diagnostic_value("adaptive_end_layer_curvature_deviation_mean", "-"),
            "adaptive_end_layer_curvature_deviation_max": diagnostic_value("adaptive_end_layer_curvature_deviation_max", "-"),
            "adaptive_end_layer_curvature_relative_deviation_mean": diagnostic_value("adaptive_end_layer_curvature_relative_deviation_mean", "-"),
            "adaptive_end_layer_geometry_deviation_mean": diagnostic_value("adaptive_end_layer_geometry_deviation_mean", "-"),
            "adaptive_end_layer_geometry_deviation_max": diagnostic_value("adaptive_end_layer_geometry_deviation_max", "-"),
            "adaptive_end_layer_problem_region_count": diagnostic_value("adaptive_end_layer_problem_region_count", "-"),
            "adaptive_end_layer_refinement_recommended": diagnostic_value("adaptive_end_layer_refinement_recommended", "-"),
            "adaptive_end_layer_rerun_allowed": diagnostic_value("adaptive_end_layer_rerun_allowed", "-"),
            "adaptive_end_layer_rerun_reason": diagnostic_value("adaptive_end_layer_rerun_reason", "-"),
            "adaptive_end_layer_selected_patch_source": diagnostic_value("adaptive_end_layer_selected_patch_source", "-"),
            "adaptive_end_layer_reasons": diagnostic_value("adaptive_end_layer_reasons", ()),
            "adaptive_seed_alignment_status": diagnostic_value("adaptive_seed_alignment_status", "-"),
            "adaptive_seed_alignment_action": diagnostic_value("adaptive_seed_alignment_action", "-"),
            "adaptive_seed_alignment_reasons": diagnostic_value("adaptive_seed_alignment_reasons", ()),
            "adaptive_target_disagreement_status": diagnostic_value("adaptive_target_disagreement_status", "-"),
            "adaptive_target_disagreement_action": diagnostic_value("adaptive_target_disagreement_action", "-"),
            "adaptive_target_disagreement_reasons": diagnostic_value("adaptive_target_disagreement_reasons", ()),
            "adaptive_target_disagreement_mean": diagnostic_value("adaptive_target_disagreement_mean", "-"),
            "adaptive_target_disagreement_max": diagnostic_value("adaptive_target_disagreement_max", "-"),
            "adaptive_target_disagreement_mean_ratio": diagnostic_value("adaptive_target_disagreement_mean_ratio", "-"),
            "adaptive_target_disagreement_max_ratio": diagnostic_value("adaptive_target_disagreement_max_ratio", "-"),
            "adaptive_target_disagreement_source": diagnostic_value("adaptive_target_disagreement_source", "-"),
            "adaptive_target_model_recommendation": diagnostic_value("adaptive_target_model_recommendation", "-"),
            "adaptive_target_model_confidence": diagnostic_value("adaptive_target_model_confidence", "-"),
            "adaptive_target_confidence_model": diagnostic_value("adaptive_target_confidence_model", "-"),
            "adaptive_target_confidence_min": diagnostic_value("adaptive_target_confidence_min", "-"),
            "adaptive_target_confidence_mean": diagnostic_value("adaptive_target_confidence_mean", "-"),
            "adaptive_target_confidence_median": diagnostic_value("adaptive_target_confidence_median", "-"),
            "adaptive_target_confidence_max": diagnostic_value("adaptive_target_confidence_max", "-"),
            "adaptive_target_confidence_low_count": diagnostic_value("adaptive_target_confidence_low_count", "-"),
            "adaptive_target_confidence_low_threshold": diagnostic_value("adaptive_target_confidence_low_threshold", "-"),
            "adaptive_target_confidence_vertex_count": diagnostic_value("adaptive_target_confidence_vertex_count", "-"),
            "adaptive_target_recommended_surface_weight_min": diagnostic_value("adaptive_target_recommended_surface_weight_min", "-"),
            "adaptive_target_recommended_surface_weight_mean": diagnostic_value("adaptive_target_recommended_surface_weight_mean", "-"),
            "adaptive_target_recommended_surface_weight_max": diagnostic_value("adaptive_target_recommended_surface_weight_max", "-"),
            "adaptive_target_recommended_surface_weight_limit": diagnostic_value("adaptive_target_recommended_surface_weight_limit", "-"),
            "adaptive_target_confidence_profile_reasons": diagnostic_value("adaptive_target_confidence_profile_reasons", ()),
            "adaptive_confidence_target_probe_status": diagnostic_value("adaptive_confidence_target_probe_status", "-"),
            "adaptive_confidence_target_probe_action": diagnostic_value("adaptive_confidence_target_probe_action", "-"),
            "adaptive_confidence_target_probe_attempted": diagnostic_value("adaptive_confidence_target_probe_attempted", "-"),
            "adaptive_confidence_target_probe_applied": diagnostic_value("adaptive_confidence_target_probe_applied", "-"),
            "adaptive_confidence_target_probe_selected": diagnostic_value("adaptive_confidence_target_probe_selected", "-"),
            "adaptive_confidence_target_probe_accepted_by_basic_gate": diagnostic_value("adaptive_confidence_target_probe_accepted_by_basic_gate", "-"),
            "adaptive_confidence_target_probe_target_kind": diagnostic_value("adaptive_confidence_target_probe_target_kind", "-"),
            "adaptive_confidence_target_probe_vertex_count": diagnostic_value("adaptive_confidence_target_probe_vertex_count", "-"),
            "adaptive_confidence_target_probe_movement_mean": diagnostic_value("adaptive_confidence_target_probe_movement_mean", "-"),
            "adaptive_confidence_target_probe_movement_max": diagnostic_value("adaptive_confidence_target_probe_movement_max", "-"),
            "adaptive_confidence_target_probe_movement_mean_ratio": diagnostic_value("adaptive_confidence_target_probe_movement_mean_ratio", "-"),
            "adaptive_confidence_target_probe_movement_max_ratio": diagnostic_value("adaptive_confidence_target_probe_movement_max_ratio", "-"),
            "adaptive_confidence_target_probe_confidence_mean": diagnostic_value("adaptive_confidence_target_probe_confidence_mean", "-"),
            "adaptive_confidence_target_probe_recommended_surface_weight_mean": diagnostic_value("adaptive_confidence_target_probe_recommended_surface_weight_mean", "-"),
            "adaptive_confidence_target_probe_basic_gate_status": diagnostic_value("adaptive_confidence_target_probe_basic_gate_status", "-"),
            "adaptive_confidence_target_probe_basic_gate_reasons": diagnostic_value("adaptive_confidence_target_probe_basic_gate_reasons", ()),
            "adaptive_confidence_target_probe_quality_gate_status": diagnostic_value("adaptive_confidence_target_probe_quality_gate_status", "-"),
            "adaptive_confidence_target_probe_g2_gate_status": diagnostic_value("adaptive_confidence_target_probe_g2_gate_status", "-"),
            "adaptive_confidence_target_probe_reasons": diagnostic_value("adaptive_confidence_target_probe_reasons", ()),
            "adaptive_confidence_target_candidate_status": diagnostic_value("adaptive_confidence_target_candidate_status", "-"),
            "adaptive_confidence_target_candidate_action": diagnostic_value("adaptive_confidence_target_candidate_action", "-"),
            "adaptive_confidence_target_candidate_available": diagnostic_value("adaptive_confidence_target_candidate_available", "-"),
            "adaptive_confidence_target_candidate_applied": diagnostic_value("adaptive_confidence_target_candidate_applied", "-"),
            "adaptive_confidence_target_candidate_selected": diagnostic_value("adaptive_confidence_target_candidate_selected", "-"),
            "adaptive_confidence_target_candidate_accepted_by_gates": diagnostic_value("adaptive_confidence_target_candidate_accepted_by_gates", "-"),
            "adaptive_confidence_target_candidate_topology_status": diagnostic_value("adaptive_confidence_target_candidate_topology_status", "-"),
            "adaptive_confidence_target_candidate_quality_status": diagnostic_value("adaptive_confidence_target_candidate_quality_status", "-"),
            "adaptive_confidence_target_candidate_g2_status": diagnostic_value("adaptive_confidence_target_candidate_g2_status", "-"),
            "adaptive_confidence_target_candidate_movement_mean": diagnostic_value("adaptive_confidence_target_candidate_movement_mean", "-"),
            "adaptive_confidence_target_candidate_movement_max": diagnostic_value("adaptive_confidence_target_candidate_movement_max", "-"),
            "adaptive_confidence_target_candidate_movement_mean_ratio": diagnostic_value("adaptive_confidence_target_candidate_movement_mean_ratio", "-"),
            "adaptive_confidence_target_candidate_movement_max_ratio": diagnostic_value("adaptive_confidence_target_candidate_movement_max_ratio", "-"),
            "adaptive_confidence_target_candidate_curvature_delta_mean": diagnostic_value("adaptive_confidence_target_candidate_curvature_delta_mean", "-"),
            "adaptive_confidence_target_candidate_curvature_delta_max": diagnostic_value("adaptive_confidence_target_candidate_curvature_delta_max", "-"),
            "adaptive_confidence_target_candidate_curvature_relative_delta_mean": diagnostic_value("adaptive_confidence_target_candidate_curvature_relative_delta_mean", "-"),
            "adaptive_confidence_target_candidate_reasons": diagnostic_value("adaptive_confidence_target_candidate_reasons", ()),
            "adaptive_confidence_target_candidate_selected_by_policy": diagnostic_value("adaptive_confidence_target_candidate_selected_by_policy", "-"),
            "adaptive_confidence_target_candidate_selection_status": diagnostic_value("adaptive_confidence_target_candidate_selection_status", "-"),
            "adaptive_confidence_target_candidate_selection_reason": diagnostic_value("adaptive_confidence_target_candidate_selection_reason", "-"),
            "adaptive_confidence_target_candidate_selection_error": diagnostic_value("adaptive_confidence_target_candidate_selection_error", "-"),
            "adaptive_confidence_target_candidate_selected_vertex_count": diagnostic_value("adaptive_confidence_target_candidate_selected_vertex_count", "-"),
            "adaptive_confidence_target_candidate_applied_delta_mean": diagnostic_value("adaptive_confidence_target_candidate_applied_delta_mean", "-"),
            "adaptive_confidence_target_candidate_applied_delta_max": diagnostic_value("adaptive_confidence_target_candidate_applied_delta_max", "-"),
            "adaptive_directional_target_status": diagnostic_value("adaptive_directional_target_status", "-"),
            "adaptive_directional_target_action": diagnostic_value("adaptive_directional_target_action", "-"),
            "adaptive_directional_target_model": diagnostic_value("adaptive_directional_target_model", "-"),
            "adaptive_directional_target_vertex_count": diagnostic_value("adaptive_directional_target_vertex_count", "-"),
            "adaptive_directional_target_residual_mean": diagnostic_value("adaptive_directional_target_residual_mean", "-"),
            "adaptive_directional_target_residual_max": diagnostic_value("adaptive_directional_target_residual_max", "-"),
            "adaptive_directional_target_residual_mean_ratio": diagnostic_value("adaptive_directional_target_residual_mean_ratio", "-"),
            "adaptive_directional_target_residual_max_ratio": diagnostic_value("adaptive_directional_target_residual_max_ratio", "-"),
            "adaptive_directional_target_normal_residual_mean": diagnostic_value("adaptive_directional_target_normal_residual_mean", "-"),
            "adaptive_directional_target_normal_residual_min": diagnostic_value("adaptive_directional_target_normal_residual_min", "-"),
            "adaptive_directional_target_normal_residual_max": diagnostic_value("adaptive_directional_target_normal_residual_max", "-"),
            "adaptive_directional_target_axis_u_curvature": diagnostic_value("adaptive_directional_target_axis_u_curvature", "-"),
            "adaptive_directional_target_axis_v_curvature": diagnostic_value("adaptive_directional_target_axis_v_curvature", "-"),
            "adaptive_directional_target_cross_curvature": diagnostic_value("adaptive_directional_target_cross_curvature", "-"),
            "adaptive_directional_target_anisotropy_ratio": diagnostic_value("adaptive_directional_target_anisotropy_ratio", "-"),
            "adaptive_directional_target_dominant_axis": diagnostic_value("adaptive_directional_target_dominant_axis", "-"),
            "adaptive_directional_target_cluster_count": diagnostic_value("adaptive_directional_target_cluster_count", "-"),
            "adaptive_directional_target_cluster_threshold": diagnostic_value("adaptive_directional_target_cluster_threshold", "-"),
            "adaptive_directional_target_reasons": diagnostic_value("adaptive_directional_target_reasons", ()),
            "adaptive_anisotropic_candidate_status": diagnostic_value("adaptive_anisotropic_candidate_status", "-"),
            "adaptive_anisotropic_candidate_action": diagnostic_value("adaptive_anisotropic_candidate_action", "-"),
            "adaptive_anisotropic_candidate_available": diagnostic_value("adaptive_anisotropic_candidate_available", "-"),
            "adaptive_anisotropic_candidate_applied": diagnostic_value("adaptive_anisotropic_candidate_applied", "-"),
            "adaptive_anisotropic_candidate_selected": diagnostic_value("adaptive_anisotropic_candidate_selected", "-"),
            "adaptive_anisotropic_candidate_accepted_by_gates": diagnostic_value("adaptive_anisotropic_candidate_accepted_by_gates", "-"),
            "adaptive_anisotropic_candidate_target_axis": diagnostic_value("adaptive_anisotropic_candidate_target_axis", "-"),
            "adaptive_anisotropic_candidate_cluster_count": diagnostic_value("adaptive_anisotropic_candidate_cluster_count", "-"),
            "adaptive_anisotropic_candidate_influenced_vertices": diagnostic_value("adaptive_anisotropic_candidate_influenced_vertices", "-"),
            "adaptive_anisotropic_candidate_movement_mean": diagnostic_value("adaptive_anisotropic_candidate_movement_mean", "-"),
            "adaptive_anisotropic_candidate_movement_max": diagnostic_value("adaptive_anisotropic_candidate_movement_max", "-"),
            "adaptive_anisotropic_candidate_movement_mean_ratio": diagnostic_value("adaptive_anisotropic_candidate_movement_mean_ratio", "-"),
            "adaptive_anisotropic_candidate_movement_max_ratio": diagnostic_value("adaptive_anisotropic_candidate_movement_max_ratio", "-"),
            "adaptive_anisotropic_candidate_quality_status": diagnostic_value("adaptive_anisotropic_candidate_quality_status", "-"),
            "adaptive_anisotropic_candidate_quality_min_angle": diagnostic_value("adaptive_anisotropic_candidate_quality_min_angle", "-"),
            "adaptive_anisotropic_candidate_quality_degenerate_count": diagnostic_value("adaptive_anisotropic_candidate_quality_degenerate_count", "-"),
            "adaptive_anisotropic_candidate_g2_status": diagnostic_value("adaptive_anisotropic_candidate_g2_status", "-"),
            "adaptive_anisotropic_candidate_curvature_delta_mean": diagnostic_value("adaptive_anisotropic_candidate_curvature_delta_mean", "-"),
            "adaptive_anisotropic_candidate_curvature_delta_max": diagnostic_value("adaptive_anisotropic_candidate_curvature_delta_max", "-"),
            "adaptive_anisotropic_candidate_curvature_relative_delta_mean": diagnostic_value("adaptive_anisotropic_candidate_curvature_relative_delta_mean", "-"),
            "adaptive_anisotropic_candidate_reasons": diagnostic_value("adaptive_anisotropic_candidate_reasons", ()),
            "adaptive_anisotropic_candidate_selected_by_policy": diagnostic_value("adaptive_anisotropic_candidate_selected_by_policy", "-"),
            "adaptive_anisotropic_candidate_selection_status": diagnostic_value("adaptive_anisotropic_candidate_selection_status", "-"),
            "adaptive_anisotropic_candidate_selection_reason": diagnostic_value("adaptive_anisotropic_candidate_selection_reason", "-"),
            "adaptive_anisotropic_candidate_selection_error": diagnostic_value("adaptive_anisotropic_candidate_selection_error", "-"),
            "adaptive_anisotropic_candidate_selected_vertex_count": diagnostic_value("adaptive_anisotropic_candidate_selected_vertex_count", "-"),
            "adaptive_anisotropic_candidate_selection_policy": diagnostic_value("adaptive_anisotropic_candidate_selection_policy", "-"),
            "adaptive_target_mls2_vs_sphere_mean": diagnostic_value("adaptive_target_mls2_vs_sphere_mean", "-"),
            "adaptive_target_mls2_vs_sphere_max": diagnostic_value("adaptive_target_mls2_vs_sphere_max", "-"),
            "adaptive_target_mls2_vs_sphere_mean_ratio": diagnostic_value("adaptive_target_mls2_vs_sphere_mean_ratio", "-"),
            "adaptive_target_mls2_vs_sphere_max_ratio": diagnostic_value("adaptive_target_mls2_vs_sphere_max_ratio", "-"),
            "adaptive_target_mls2_vs_plane_mean": diagnostic_value("adaptive_target_mls2_vs_plane_mean", "-"),
            "adaptive_target_mls2_vs_plane_max": diagnostic_value("adaptive_target_mls2_vs_plane_max", "-"),
            "adaptive_target_mls2_vs_plane_mean_ratio": diagnostic_value("adaptive_target_mls2_vs_plane_mean_ratio", "-"),
            "adaptive_target_mls2_vs_plane_max_ratio": diagnostic_value("adaptive_target_mls2_vs_plane_max_ratio", "-"),
            "adaptive_target_mls1_vs_mls2_mean": diagnostic_value("adaptive_target_mls1_vs_mls2_mean", "-"),
            "adaptive_target_mls1_vs_mls2_max": diagnostic_value("adaptive_target_mls1_vs_mls2_max", "-"),
            "adaptive_target_mls2_vs_mls3_mean": diagnostic_value("adaptive_target_mls2_vs_mls3_mean", "-"),
            "adaptive_target_mls2_vs_mls3_max": diagnostic_value("adaptive_target_mls2_vs_mls3_max", "-"),
            "adaptive_target_mls1_vs_mls2_mean_ratio": diagnostic_value("adaptive_target_mls1_vs_mls2_mean_ratio", "-"),
            "adaptive_target_mls1_vs_mls2_max_ratio": diagnostic_value("adaptive_target_mls1_vs_mls2_max_ratio", "-"),
            "adaptive_target_mls2_vs_mls3_mean_ratio": diagnostic_value("adaptive_target_mls2_vs_mls3_mean_ratio", "-"),
            "adaptive_target_mls2_vs_mls3_max_ratio": diagnostic_value("adaptive_target_mls2_vs_mls3_max_ratio", "-"),
            "adaptive_target_signed_disagreement_mean": diagnostic_value("adaptive_target_signed_disagreement_mean", "-"),
            "adaptive_target_signed_disagreement_min": diagnostic_value("adaptive_target_signed_disagreement_min", "-"),
            "adaptive_target_signed_disagreement_max": diagnostic_value("adaptive_target_signed_disagreement_max", "-"),
            "adaptive_target_signed_plane_disagreement_mean": diagnostic_value("adaptive_target_signed_plane_disagreement_mean", "-"),
            "adaptive_target_signed_plane_disagreement_min": diagnostic_value("adaptive_target_signed_plane_disagreement_min", "-"),
            "adaptive_target_signed_plane_disagreement_max": diagnostic_value("adaptive_target_signed_plane_disagreement_max", "-"),
            "adaptive_target_mls_ring1_normal_spread_degrees": diagnostic_value("adaptive_target_mls_ring1_normal_spread_degrees", "-"),
            "adaptive_target_mls_ring2_normal_spread_degrees": diagnostic_value("adaptive_target_mls_ring2_normal_spread_degrees", "-"),
            "adaptive_target_mls_ring3_normal_spread_degrees": diagnostic_value("adaptive_target_mls_ring3_normal_spread_degrees", "-"),
            "adaptive_seed_projection_distance_mean": diagnostic_value("adaptive_seed_projection_distance_mean", "-"),
            "adaptive_seed_projection_distance_max": diagnostic_value("adaptive_seed_projection_distance_max", "-"),
            "adaptive_seed_projection_mean_ratio": diagnostic_value("adaptive_seed_projection_mean_ratio", "-"),
            "adaptive_seed_projection_max_ratio": diagnostic_value("adaptive_seed_projection_max_ratio", "-"),
            "adaptive_seed_signed_offset_mean": diagnostic_value("adaptive_seed_signed_offset_mean", "-"),
            "adaptive_seed_signed_offset_min": diagnostic_value("adaptive_seed_signed_offset_min", "-"),
            "adaptive_seed_signed_offset_max": diagnostic_value("adaptive_seed_signed_offset_max", "-"),
            "adaptive_seed_effective_surface_weight": diagnostic_value("adaptive_seed_effective_surface_weight", "-"),
            "adaptive_seed_requested_surface_weight": diagnostic_value("adaptive_seed_requested_surface_weight", "-"),
            "adaptive_seed_support_normal_spread_degrees": diagnostic_value("adaptive_seed_support_normal_spread_degrees", "-"),
            "adaptive_seed_generated_vertex_count": diagnostic_value("adaptive_seed_generated_vertex_count", "-"),
            "adaptive_seed_context_edge_length_median": diagnostic_value("adaptive_seed_context_edge_length_median", "-"),
            "adaptive_end_layer_rerun_gate_status": end_layer_rerun_gate.get("status"),
            "adaptive_end_layer_rerun_gate_action": end_layer_rerun_gate.get("action"),
            "adaptive_end_layer_rerun_gate_allowed": end_layer_rerun_gate.get("rerun_allowed"),
            "adaptive_end_layer_rerun_gate_reasons": end_layer_rerun_gate.get("reasons", ()),
            "adaptive_end_layer_rerun_gate_geometry_deviation_mean": end_layer_rerun_gate.get("geometry_deviation_mean"),
            "adaptive_end_layer_rerun_gate_geometry_deviation_max": end_layer_rerun_gate.get("geometry_deviation_max"),
            "adaptive_end_layer_rerun_gate_curvature_deviation_mean": end_layer_rerun_gate.get("curvature_deviation_mean"),
            "adaptive_end_layer_rerun_gate_curvature_relative_deviation_mean": end_layer_rerun_gate.get("curvature_relative_deviation_mean"),
            "adaptive_conservative_g1_attempted": adaptive_attempted_conservative_g1(),
            "adaptive_conservative_g1_used": diagnostic_value("adaptive_conservative_g1_used", "-"),
            "adaptive_conservative_g1_reason": diagnostic_value("adaptive_conservative_g1_reason", "-"),
            "adaptive_selected_relaxation_iterations": diagnostic_value("adaptive_selected_relaxation_iterations", "-"),
            "adaptive_selected_relaxation_strength": diagnostic_value("adaptive_selected_relaxation_strength", "-"),
            "adaptive_selected_surface_weight": diagnostic_value("adaptive_selected_surface_weight", "-"),
            "adaptive_primary_score": diagnostic_value("adaptive_primary_score", "-"),
            "adaptive_conservative_g1_score": diagnostic_value("adaptive_conservative_g1_score", "-"),
            "adaptive_selected_score": diagnostic_value("adaptive_selected_score", "-"),
            "adaptive_score_decision": diagnostic_value("adaptive_score_decision", "-"),
            "adaptive_score_delta": diagnostic_value("adaptive_score_delta", "-"),
        }

    def _set_hole_fill_commit_button_from_policy(
        self,
        *,
        commit_allowed: bool,
    ) -> None:
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(bool(commit_allowed))

    def _set_hole_fill_preview_buttons_from_policy(
        self,
        *,
        commit_allowed: bool,
        cancel_allowed: bool = True,
    ) -> None:
        """Apply the normalized hole-fill preview policy to GUI buttons."""

        self._set_hole_fill_commit_button_from_policy(
            commit_allowed=bool(commit_allowed),
        )
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(bool(cancel_allowed))

    def _clear_hole_fill_tool_preview_state_reference(
        self,
        *,
        silent: bool = False,
    ) -> None:
        """Clear the processor-owned disk-backed hole-fill preview reference.

        P4 keeps GUI preview lifecycle and ToolPreviewState lifecycle aligned.
        The GUI does not delete preview files from disk here; it only tells
        MeshProcessor to drop the active in-memory preview-state reference so
        canceled, replaced, failed, or invalidated previews cannot keep a stale
        patch snapshot active.
        """

        processor = getattr(self, "processor", None)
        clearer = getattr(processor, "clear_tool_preview_state", None)
        if not callable(clearer):
            return

        try:
            clearer()
        except Exception:
            if not silent:
                raise

    def _hole_fill_preview_gui_policy_state_for_gui(
        self,
        preview: ManualEditPreview,
    ) -> dict[str, object]:
        """Return the normalized GUI policy state for one hole-fill preview.

        The visible GUI state reads from one routed preview policy. The
        status label, commit button, cancel button,
        result panel, and status-bar text should all read from this normalized
        state instead of re-deriving commit/cancel behavior independently.
        """

        summary = getattr(preview, "selection_summary", None)
        if not isinstance(summary, dict):
            summary = {}

        policy = self._hole_fill_preview_commit_policy_for_gui(preview)

        commit_allowed = self._hole_fill_gui_bool(
            policy.get("commit_allowed", True),
            default=True,
        )
        blocking_reasons = self._hole_fill_gui_text_tuple(
            policy.get("commit_blocking_reasons", ()),
        )
        warnings = self._hole_fill_gui_text_tuple(
            policy.get("commit_warnings", ()),
        )

        method_key = str(
            policy.get("public_method")
            or summary.get("public_method")
            or summary.get("method")
            or ""
        ).strip().lower().replace("-", "_")

        method_display = policy.get("method", summary.get("method"))
        backend_display = policy.get(
            "backend",
            summary.get("backend", summary.get("method")),
        )

        try:
            candidate_index = int(summary.get("candidate_index", 0)) + 1
        except Exception:
            candidate_index = 1

        policy_status = (
            "Commit is available for this validated preview."
            if commit_allowed
            else "Commit is blocked for this preview."
        )
        status_text = (
            "Preview ready: "
            f"candidate {candidate_index} | "
            f"boundary vertices={summary.get('boundary_vertices')} | "
            f"boundary edges={summary.get('boundary_edges')} | "
            f"method={method_display} | "
            f"backend={backend_display}. "
            f"{policy_status}"
        )

        return {
            "summary": summary,
            "policy": policy,
            "commit_allowed": commit_allowed,
            "cancel_allowed": True,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
            "method_key": method_key,
            "method_display": method_display,
            "backend_display": backend_display,
            "status_text": status_text,
            "status_bar_text": (
                "Hole fill preview ready"
                if commit_allowed
                else "Hole fill preview blocked from commit"
            ),
        }


    @staticmethod
    def _hole_fill_method_label(method: str) -> str:
        labels = {
            "fan": "Triangulate boundary fan (preview)",
            "fan_triangulate": "Fan triangulate alias",
            "center_fan": "Center fan",
            "curvature_sphere": "Curvature sphere cap (experimental)",
            "curvature_sphere_refined": "Curvature sphere refined (experimental)",
            "curvature_sphere_grid8": "Curvature sphere grid8 (experimental)",
            "curvature_sphere_uvgrid": "Curvature sphere UV grid (experimental)",
            "curvature_sphere_uvdelaunay": "Curvature sphere UV Delaunay (experimental)",
            "curvature_sphere_uvdelaunay_relaxed": "Curvature sphere UV Delaunay relaxed (experimental)",
            "adaptive_surface": "Adaptive surface fill",
            "adaptive_surface_v2": "Adaptive surface fill v2",
            "surface_uvdelaunay_relaxed": "Local surface UV Delaunay relaxed (experimental)",
            "surface_uvdelaunay_sealed_relaxed": "Local surface UV Delaunay sealed relaxed (experimental)",
            "surface_uvdelaunay_sealed_dense_relaxed": "Local surface UV Delaunay sealed dense relaxed (experimental)",
            "open3d": "Open3D tensor fill_holes (single-hole)",
        }
        return labels.get(str(method), str(method))

    def _sync_hole_fill_method_options(self) -> None:
        if not hasattr(self, "hole_fill_method_combo"):
            return

        combo = self.hole_fill_method_combo

        try:
            current = str(combo.currentData() or "fan")
        except Exception:
            current = "fan"

        self._syncing_hole_fill_methods = True
        try:
            combo.clear()

            methods = available_hole_fill_preview_methods(include_unavailable=False)
            if not methods:
                methods = ("fan",)

            for method in methods:
                combo.addItem(self._hole_fill_method_label(method), method)

            try:
                target_index = int(combo.findData(current))
            except Exception:
                target_index = -1

            if target_index < 0:
                try:
                    target_index = int(combo.findData("fan"))
                except Exception:
                    target_index = -1

            if target_index >= 0:
                combo.setCurrentIndex(target_index)

            try:
                combo.setEnabled(bool(methods))
            except Exception:
                pass

            try:
                combo.setToolTip(
                    "Adaptive surface fill is the stable adaptive route. "
                    "Adaptive surface fill v2 is the direct v2 test route. "
                    "Open3D fill_holes is available only for single-hole previews. "
                    "Fan methods remain a simple candidate-scoped fallback."
                )
            except Exception:
                pass
        finally:
            self._syncing_hole_fill_methods = False

    def _hole_candidate_diagnostics_for_gui(self, candidates: list[object] | tuple[object, ...]) -> tuple[object | None, ...]:
        candidate_list = list(candidates)
        if not candidate_list:
            return ()

        processor = getattr(self, "processor", None)
        mesh = getattr(processor, "mesh", None)
        if mesh is None:
            return tuple(None for _ in candidate_list)

        try:
            diagnostics = tuple(diagnose_hole_candidates(mesh, candidate_list))  # type: ignore[arg-type]
        except Exception as exc:
            if hasattr(self, "log"):
                try:
                    self.log(f"Hole candidate diagnostics unavailable: {exc}")
                except Exception:
                    pass
            return tuple(None for _ in candidate_list)

        if len(diagnostics) != len(candidate_list):
            padded: list[object | None] = list(diagnostics[: len(candidate_list)])
            while len(padded) < len(candidate_list):
                padded.append(None)
            return tuple(padded)

        return diagnostics

    @staticmethod
    def _hole_candidate_diagnostic_kind_text_for_gui(diagnostic: object | None) -> str:
        if diagnostic is None:
            return "unknown"

        kind = getattr(diagnostic, "kind", None)
        return str(getattr(kind, "value", kind or "unknown"))

    @staticmethod
    def _hole_candidate_diagnostic_confidence_text_for_gui(diagnostic: object | None) -> str:
        if diagnostic is None:
            return "-"

        confidence = getattr(diagnostic, "confidence", None)
        try:
            return f"{float(confidence):.2f}"
        except Exception:
            return "-"

    @staticmethod
    def _hole_candidate_diagnostic_notes_text_for_gui(diagnostic: object | None) -> str:
        if diagnostic is None:
            return "-"

        notes = getattr(diagnostic, "notes", ()) or ()
        try:
            note_list = [str(note) for note in notes if str(note)]
        except Exception:
            note_list = [str(notes)]

        return "; ".join(note_list) if note_list else "-"

    def _hole_filter_values(self) -> tuple[float | None, float | None]:
        max_area_hint: float | None = None
        max_perimeter: float | None = None

        if hasattr(self, "hole_fill_max_area_spin"):
            try:
                value = float(self.hole_fill_max_area_spin.value())
                if value > 0.0:
                    max_area_hint = value
            except Exception:
                max_area_hint = None

        if hasattr(self, "hole_fill_max_perimeter_spin"):
            try:
                value = float(self.hole_fill_max_perimeter_spin.value())
                if value > 0.0:
                    max_perimeter = value
            except Exception:
                max_perimeter = None

        return max_area_hint, max_perimeter

    def _reset_hole_fill_ui(self, *, status: str | None = None) -> None:
        self._last_hole_candidates = []
        self._last_hole_candidate_scope = None
        self._hole_fill_preview = None
        self._sync_hole_fill_method_options()
        if hasattr(self.viewport, "clear_preview_mesh"):
            try:
                self.viewport.clear_preview_mesh()
            except Exception:
                pass

        if hasattr(self, "hole_fill_candidate_combo"):
            self.hole_fill_candidate_combo.clear()
            self.hole_fill_candidate_combo.addItem("No hole candidates", None)
            self.hole_fill_candidate_combo.setEnabled(False)

        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(False)
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(False)
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(False)
        TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
            preview_enabled=False,
            commit_enabled=False,
        )

        self._set_hole_fill_status(status or "No hole fill preview active.")

    def _populate_hole_fill_candidate_combo(
        self,
        candidates: list[object],
        face_ids: tuple[int, ...] | None,
    ) -> None:
        self._last_hole_candidates = list(candidates)
        self._last_hole_candidate_scope = face_ids
        self._hole_fill_preview = None
        self._sync_hole_fill_method_options()

        if not hasattr(self, "hole_fill_candidate_combo"):
            return

        combo = self.hole_fill_candidate_combo
        combo.clear()

        if not candidates:
            combo.addItem("No hole candidates", None)
            combo.setEnabled(False)
            if hasattr(self, "hole_fill_preview_btn"):
                self.hole_fill_preview_btn.setEnabled(False)
            if hasattr(self, "hole_fill_commit_btn"):
                self.hole_fill_commit_btn.setEnabled(False)
            if hasattr(self, "hole_fill_cancel_btn"):
                self.hole_fill_cancel_btn.setEnabled(False)
            TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
                preview_enabled=False,
                commit_enabled=False,
            )
            self._set_hole_fill_status("No hole candidates available for fill preview.")
            return

        diagnostics = self._hole_candidate_diagnostics_for_gui(candidates)

        for index, candidate in enumerate(candidates, start=1):
            classified = getattr(candidate, "classified_loop", None)
            kind = getattr(classified, "kind", None)
            kind_text = str(getattr(kind, "value", kind or "unknown"))
            diagnostic = diagnostics[index - 1] if index - 1 < len(diagnostics) else None
            diagnostic_kind = self._hole_candidate_diagnostic_kind_text_for_gui(diagnostic)
            perimeter = self._format_optional_float(getattr(candidate, "perimeter", None), digits=5)
            area = self._format_optional_float(getattr(candidate, "area_hint", None), digits=5)
            combo.addItem(
                f"Candidate {index}: {diagnostic_kind} | {kind_text} | area {area} | perimeter {perimeter}",
                index - 1,
            )

        combo.setCurrentIndex(0)
        combo.setEnabled(True)

        busy = self._worker is not None
        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(not busy)
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(False)
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(False)
        TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
            preview_enabled=not busy,
            commit_enabled=False,
        )

        self._set_hole_fill_status(
            f"{len(candidates)} candidate(s) ready for fill preview."
        )

    def _selected_hole_candidate(self) -> object | None:
        if not hasattr(self, "hole_fill_candidate_combo"):
            return None
        try:
            index = self.hole_fill_candidate_combo.currentData()
            if index is None:
                return None
            index = int(index)
        except Exception:
            return None

        if index < 0 or index >= len(self._last_hole_candidates):
            return None
        return self._last_hole_candidates[index]

    def _on_hole_fill_candidate_changed(self) -> None:
        candidate = self._selected_hole_candidate()
        busy = self._worker is not None
        can_preview = candidate is not None and not busy

        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(can_preview)
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(False)
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(False)
        TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
            preview_enabled=bool(getattr(self, "_last_hole_candidates", [])) and not busy,
            commit_enabled=False,
        )

        if candidate is None:
            self._set_hole_fill_status("No hole candidate selected.")
            return

        area = self._format_optional_float(getattr(candidate, "area_hint", None), digits=5)
        perimeter = self._format_optional_float(getattr(candidate, "perimeter", None), digits=5)
        diagnostics = self._hole_candidate_diagnostics_for_gui([candidate])
        diagnostic = diagnostics[0] if diagnostics else None
        diagnostic_kind = self._hole_candidate_diagnostic_kind_text_for_gui(diagnostic)
        self._set_hole_fill_status(
            f"Candidate selected: {diagnostic_kind} | area {area}, perimeter {perimeter}. "
            "Ready to build preview."
        )

    def _hole_fill_patch_mesh_from_tool_state(self) -> trimesh.Trimesh | None:
        """
        Load the named hole-fill patch mesh from the active ToolPreviewState.

        New Phase 2 preview state writes an isolated patch_mesh.ply and records it
        as a ToolRegion. Prefer that named patch over deriving faces from the
        full preview mesh. This removes the old appended-face assumption while
        still allowing legacy ManualEditPreview fallback below.
        """
        processor = getattr(self, "processor", None)
        getter = getattr(processor, "last_tool_preview_state", None)
        if not callable(getter):
            return None

        tool_state = getter()
        if tool_state is None:
            return None

        for region in getattr(tool_state, "output_regions", ()) or ():
            if getattr(region, "kind", None) != REGION_KIND_HOLE_PATCH:
                continue

            mesh_snapshot = getattr(region, "mesh_snapshot", None)
            if mesh_snapshot is None:
                continue

            return mesh_snapshot.load()

        return None

    def _double_sided_hole_fill_overlay_mesh(self, patch_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Return a double-sided overlay mesh for reliable patch visibility.

        Some viewport backends use one-sided face rendering. Duplicating the
        patch faces with reversed winding keeps the red fill patch visible from
        both sides.
        """
        vertices = np.asarray(getattr(patch_mesh, "vertices", ()), dtype=float)
        faces = np.asarray(getattr(patch_mesh, "faces", ()), dtype=np.int64)

        if vertices.ndim != 2 or vertices.shape[1] < 3:
            raise ValueError("Hole fill patch mesh does not contain valid vertices.")
        if faces.ndim != 2 or faces.shape[1] < 3:
            raise ValueError("Hole fill patch mesh does not contain polygonal faces.")
        if faces.size == 0:
            raise ValueError("Hole fill patch mesh does not contain visible fill faces.")

        double_sided_faces = np.vstack([faces, faces[:, ::-1]])

        return trimesh.Trimesh(
            vertices=vertices.copy(),
            faces=double_sided_faces.copy(),
            process=False,
        )

    def _build_hole_fill_overlay_mesh(self, preview: ManualEditPreview) -> trimesh.Trimesh:
        """
        Build a patch-only overlay mesh for the hole-fill preview.

        Prefer the named disk-backed patch mesh from ToolPreviewState. Fall back
        to the legacy ManualEditPreview appended-face derivation only when no
        ToolPreviewState patch is available. Low-memory previews already carry
        a patch-only preview mesh, so they can be rendered directly.
        """
        summary = getattr(preview, "selection_summary", None)
        if isinstance(summary, dict) and bool(summary.get("low_memory_patch_only", False)):
            patch_preview_mesh = getattr(preview, "preview_mesh", None)
            if not isinstance(patch_preview_mesh, trimesh.Trimesh):
                raise ValueError("Low-memory hole fill preview is missing its patch mesh.")
            return self._double_sided_hole_fill_overlay_mesh(patch_preview_mesh)

        patch_mesh = self._hole_fill_patch_mesh_from_tool_state()

        if patch_mesh is not None:
            return self._double_sided_hole_fill_overlay_mesh(patch_mesh)

        base_mesh = getattr(preview, "base_mesh", None)
        preview_mesh = getattr(preview, "preview_mesh", None)

        if base_mesh is None or preview_mesh is None:
            raise ValueError("Hole fill preview is missing base or preview mesh data.")

        vertices = np.asarray(preview_mesh.vertices, dtype=float)
        preview_faces = np.asarray(preview_mesh.faces, dtype=np.int64)
        base_face_count = int(len(getattr(base_mesh, "faces", ())))

        if preview_faces.ndim != 2 or preview_faces.shape[1] < 3:
            raise ValueError("Hole fill preview mesh does not contain polygonal faces.")

        new_faces = preview_faces[base_face_count:]
        if new_faces.size == 0:
            raise ValueError("Hole fill preview did not add any visible fill faces.")

        legacy_patch_mesh = trimesh.Trimesh(
            vertices=vertices.copy(),
            faces=new_faces.copy(),
            process=False,
        )
        return self._double_sided_hole_fill_overlay_mesh(legacy_patch_mesh)

    def _show_hole_fill_preview_mesh(self, preview: ManualEditPreview) -> None:
        if not hasattr(self.viewport, "show_preview_mesh"):
            raise RuntimeError("The active viewport backend does not support preview meshes.")

        preview_mesh = self._build_hole_fill_overlay_mesh(preview)

        self.viewport.show_preview_mesh(
            preview_mesh,
            color="#ff3b30",
            opacity=0.95,
            show_edges=True,
        )

    def _clear_hole_fill_preview(self, *, silent: bool = False) -> None:
        self._hole_fill_preview = None
        TopologyToolsMixin._clear_hole_fill_tool_preview_state_reference(
            self,
            silent=silent,
        )
        if hasattr(self.viewport, "clear_preview_mesh"):
            try:
                self.viewport.clear_preview_mesh()
            except Exception:
                if not silent:
                    raise
        if hasattr(self, "hole_fill_commit_btn"):
            self.hole_fill_commit_btn.setEnabled(False)
        if hasattr(self, "hole_fill_cancel_btn"):
            self.hole_fill_cancel_btn.setEnabled(False)
        TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
            preview_enabled=bool(getattr(self, "_last_hole_candidates", [])),
            commit_enabled=False,
        )
        self._set_hole_fill_status("No hole fill preview active.")
        self._update_brush_action_state()

    def _on_hole_fill_method_changed(self, *_args: object) -> None:
        if bool(getattr(self, "_syncing_hole_fill_methods", False)):
            return

        method = "fan"
        if hasattr(self, "hole_fill_method_combo"):
            try:
                method = str(self.hole_fill_method_combo.currentData() or method)
            except Exception:
                method = "fan"

        label = self._hole_fill_method_label(method)

        had_preview = getattr(self, "_hole_fill_preview", None) is not None
        if had_preview:
            self._clear_hole_fill_preview(silent=True)
            if hasattr(self, "log"):
                try:
                    self.log("Hole fill preview invalidated: fill method changed.")
                except Exception:
                    pass
        else:
            if hasattr(self, "hole_fill_commit_btn"):
                self.hole_fill_commit_btn.setEnabled(False)
            if hasattr(self, "hole_fill_cancel_btn"):
                self.hole_fill_cancel_btn.setEnabled(False)

        candidate = self._selected_hole_candidate()
        selected_edge_ids = tuple()
        try:
            selected_edge_ids = self._selected_edge_ids_for_gui()
        except Exception:
            selected_edge_ids = tuple()
        has_selected_edge_boundary = (
            bool(selected_edge_ids)
            and self._current_semantic_mode_value() == "edge"
        )
        busy = self._worker is not None
        can_preview = (candidate is not None or has_selected_edge_boundary) and not busy

        if hasattr(self, "hole_fill_preview_btn"):
            self.hole_fill_preview_btn.setEnabled(can_preview)
        TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
            preview_enabled=bool(getattr(self, "_last_hole_candidates", [])) and not busy,
            commit_enabled=False,
        )

        if busy:
            self._set_hole_fill_status(
                f"Method selected: {label}. Wait for the current task to finish before building a preview."
            )
        elif candidate is not None:
            self._set_hole_fill_status(
                f"Method selected: {label}. Build Fill Preview to validate this candidate."
            )
        elif has_selected_edge_boundary:
            self._set_hole_fill_status(
                f"Method selected: {label}. Selected edge boundary ready for Fill Preview."
            )
        else:
            self._set_hole_fill_status(
                f"Method selected: {label}. Select a hole candidate or one closed edge boundary before building a preview."
            )

        updater = getattr(self, "_update_brush_action_state", None)
        if callable(updater):
            updater()



    def _on_hole_fill_batch_preview_clicked(self) -> None:
        candidates = list(getattr(self, "_last_hole_candidates", []) or [])
        if not candidates:
            QMessageBox.information(
                self,
                "No hole candidates",
                "Run Find Hole Candidates before building a batch fill preview.",
            )
            return

        method = self._current_hole_fill_method_key()

        try:
            capability = hole_fill_method_capability(method)
        except ValueError as exc:
            QMessageBox.information(
                self,
                "Fill method not available",
                str(exc),
            )
            return

        method = str(capability.get("method") or method)
        if not bool(capability.get("available")):
            QMessageBox.information(
                self,
                "Fill method not available",
                str(capability.get("reason") or "Selected fill method is not available."),
            )
            return

        if method == "open3d":
            QMessageBox.information(
                self,
                "Batch Open3D fill is not enabled",
                (
                    "Open3D tensor fill_holes currently operates on the whole mesh. "
                    "Use a single-candidate preview for Open3D, or choose a fan-family method for batch preview."
                ),
            )
            return

        face_ids = self._last_hole_candidate_scope
        max_area_hint, max_perimeter = self._hole_filter_values()

        if getattr(self, "_hole_fill_preview", None) is not None:
            TopologyToolsMixin._clear_hole_fill_preview(self, silent=True)

        def task() -> object:
            builder = getattr(self.processor, "build_batch_hole_fill_preview", None)
            if not callable(builder):
                raise RuntimeError(
                    "Batch hole-fill preview requires the core MeshProcessor.build_batch_hole_fill_preview route."
                )
            return builder(
                face_ids=face_ids,
                max_area_hint=max_area_hint,
                max_perimeter=max_perimeter,
                method=method,
            )

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditPreview)
            try:
                self._show_hole_fill_preview_mesh(result)
            except Exception:
                self._hole_fill_preview = None
                TopologyToolsMixin._clear_hole_fill_tool_preview_state_reference(
                    self,
                    silent=True,
                )
                if hasattr(self.viewport, "clear_preview_mesh"):
                    try:
                        self.viewport.clear_preview_mesh()
                    except Exception:
                        pass
                if hasattr(self, "hole_fill_commit_btn"):
                    self.hole_fill_commit_btn.setEnabled(False)
                if hasattr(self, "hole_fill_cancel_btn"):
                    self.hole_fill_cancel_btn.setEnabled(False)
                TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self, commit_enabled=False)
                raise

            self._hole_fill_preview = result
            gui_policy_state = TopologyToolsMixin._hole_fill_preview_gui_policy_state_for_gui(
                self,
                result,
            )
            commit_allowed = bool(gui_policy_state["commit_allowed"])
            cancel_allowed = bool(gui_policy_state["cancel_allowed"])

            if hasattr(self, "hole_fill_commit_btn"):
                self.hole_fill_commit_btn.setEnabled(False)
            if hasattr(self, "hole_fill_cancel_btn"):
                self.hole_fill_cancel_btn.setEnabled(cancel_allowed)
            TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self,
                preview_enabled=True,
                commit_enabled=commit_allowed,
            )

            summary = getattr(result, "selection_summary", None)
            if not isinstance(summary, dict):
                summary = {}
            candidate_count = summary.get("candidate_count", len(candidates))
            patch_faces = summary.get("batch_patch_face_count", "-")

            self._set_hole_fill_status(
                f"Batch preview ready: {candidate_count} candidate(s), method {method}, patch faces {patch_faces}. "
                "Inspect the preview, then Commit Batch Fill."
            )
            self.statusBar().showMessage("Batch hole fill preview ready", 3000)
            self._update_brush_action_state()

        self._run_task("Building batch hole fill preview...", task, on_success)

    def _on_hole_fill_batch_commit_clicked(self) -> None:
        preview = getattr(self, "_hole_fill_preview", None)
        if preview is None or not self._hole_fill_preview_is_batch(preview):
            QMessageBox.information(
                self,
                "No batch preview",
                "Build and inspect Batch Preview All before committing a batch fill.",
            )
            return

        self._on_hole_fill_commit_clicked()


    def _on_hole_fill_preview_clicked(self) -> None:
        try:
            self.selection_controller.sync_from_viewport(reason="hole_fill_preview")
        except Exception:
            pass

        selected_edge_ids = tuple()
        try:
            selected_edge_ids = self._selected_edge_ids_for_gui()
        except Exception:
            selected_edge_ids = tuple()

        prefer_selected_edges = (
            bool(selected_edge_ids)
            and self._current_semantic_mode_value() == "edge"
        )

        candidate = None if prefer_selected_edges else self._selected_hole_candidate()
        if candidate is None and not selected_edge_ids:
            QMessageBox.information(
                self,
                "No hole boundary",
                (
                    "Run Find Hole Candidates and select a candidate, or switch to "
                    "Edge selection and select one closed hole-boundary loop before "
                    "building a fill preview."
                ),
            )
            return

        method = "fan"
        if hasattr(self, "hole_fill_method_combo"):
            try:
                method = str(self.hole_fill_method_combo.currentData() or method)
            except Exception:
                method = "fan"

        try:
            capability = hole_fill_method_capability(method)
        except ValueError as exc:
            QMessageBox.information(
                self,
                "Fill method not available",
                str(exc),
            )
            return

        method = str(capability.get("method") or method)

        if not bool(capability.get("available")):
            QMessageBox.information(
                self,
                "Fill method not available",
                str(capability.get("reason") or "Selected fill method is not available."),
            )
            return

        if method == "open3d" and (candidate is None or len(getattr(self, "_last_hole_candidates", [])) != 1):
            QMessageBox.information(
                self,
                "Open3D fill requires one detected hole",
                (
                    "Open3D tensor fill_holes currently requires exactly one hole candidate. "
                    "Run Find Hole Candidates so exactly one hole candidate is available, "
                    "or use fan/adaptive_surface_v2 for selected-edge low-memory boundaries."
                ),
            )
            return

        candidate_index = 0
        if candidate is not None and hasattr(self, "hole_fill_candidate_combo"):
            try:
                candidate_index = int(self.hole_fill_candidate_combo.currentData())
            except Exception:
                candidate_index = 0

        face_ids = self._last_hole_candidate_scope if candidate is not None else None
        max_area_hint, max_perimeter = self._hole_filter_values()

        if getattr(self, "_hole_fill_preview", None) is not None:
            TopologyToolsMixin._clear_hole_fill_preview(self, silent=True)

        def task() -> object:
            return self.processor.build_hole_fill_preview(
                candidate_index=candidate_index,
                candidate=candidate,
                selected_edge_ids=selected_edge_ids if candidate is None else None,
                face_ids=face_ids,
                max_area_hint=max_area_hint,
                max_perimeter=max_perimeter,
                method=method,
            )

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditPreview)
            try:
                self._show_hole_fill_preview_mesh(result)
            except Exception:
                self._hole_fill_preview = None
                TopologyToolsMixin._clear_hole_fill_tool_preview_state_reference(
                    self,
                    silent=True,
                )
                if hasattr(self.viewport, "clear_preview_mesh"):
                    try:
                        self.viewport.clear_preview_mesh()
                    except Exception:
                        pass
                if hasattr(self, "hole_fill_commit_btn"):
                    self.hole_fill_commit_btn.setEnabled(False)
                if hasattr(self, "hole_fill_cancel_btn"):
                    self.hole_fill_cancel_btn.setEnabled(False)
                raise

            self._hole_fill_preview = result

            gui_policy_state = TopologyToolsMixin._hole_fill_preview_gui_policy_state_for_gui(
                self,
                result,
            )
            summary = gui_policy_state["summary"]
            policy = gui_policy_state["policy"]
            assert isinstance(summary, dict)
            assert isinstance(policy, dict)

            commit_allowed = bool(gui_policy_state["commit_allowed"])
            cancel_allowed = bool(gui_policy_state["cancel_allowed"])
            blocking_reasons = tuple(gui_policy_state["blocking_reasons"] or ())
            warnings = tuple(gui_policy_state["warnings"] or ())

            self._set_hole_fill_status(str(gui_policy_state["status_text"]))
            TopologyToolsMixin._set_hole_fill_batch_buttons_enabled(self, commit_enabled=False)
            TopologyToolsMixin._set_hole_fill_preview_buttons_from_policy(
                self,
                commit_allowed=commit_allowed,
                cancel_allowed=cancel_allowed,
            )

            policy_lines = [
                "Commit policy:",
                f"  allowed: {commit_allowed}",
                f"  public method: {policy.get('public_method')}",
                f"  backend: {policy.get('backend')}",
                f"  adaptive stage: {policy.get('adaptive_stage')}",
                f"  context kind: {policy.get('adaptive_context_kind', summary.get('adaptive_context_kind', '-'))}",
                f"  context confidence: {policy.get('adaptive_context_confidence', summary.get('adaptive_context_confidence', '-'))}",
                f"  seed strategy: {policy.get('selected_seed_strategy', summary.get('selected_seed_strategy', '-'))}",
                f"  adaptive controller: {policy.get('adaptive_controller', '-')}",

                f"  adaptive surface v2 status/case/action: {policy.get('adaptive_surface_v2_status', '-')} / {policy.get('adaptive_surface_v2_case', '-')} / {policy.get('adaptive_surface_v2_action', '-')}",
                f"  adaptive surface v2 block legacy selection: {policy.get('adaptive_surface_v2_block_legacy_selection', '-')}",
                f"  adaptive surface v2 require new seed: {policy.get('adaptive_surface_v2_require_new_seed', '-')}",
                f"  adaptive surface v2 seed/target policy: {policy.get('adaptive_surface_v2_recommended_seed_family', '-')} / {policy.get('adaptive_surface_v2_recommended_target_policy', '-')}",
                f"  adaptive surface v2 reasons: {policy.get('adaptive_surface_v2_reasons', ())}",

                f"  adaptive surface v2 seed plan status/action/build: {policy.get('adaptive_surface_v2_seed_plan_status', '-')} / {policy.get('adaptive_surface_v2_seed_plan_action', '-')} / {policy.get('adaptive_surface_v2_seed_plan_build_required', '-')}",
                f"  adaptive surface v2 seed family: {policy.get('adaptive_surface_v2_seed_family', '-')}",
                f"  adaptive surface v2 orientation case/action: {policy.get('adaptive_surface_v2_orientation_case', '-')} / {policy.get('adaptive_surface_v2_orientation_action', '-')}",
                f"  adaptive surface v2 target/curvature/confidence policies: {policy.get('adaptive_surface_v2_target_policy', '-')} / {policy.get('adaptive_surface_v2_curvature_policy', '-')} / {policy.get('adaptive_surface_v2_confidence_policy', '-')}",
                f"  adaptive surface v2 support context policy: {policy.get('adaptive_surface_v2_support_context_policy', '-')}",
                f"  adaptive surface v2 seed plan reasons: {policy.get('adaptive_surface_v2_seed_plan_reasons', ())}",

                f"  adaptive surface v2 seed prototype status/action/build: {policy.get('adaptive_surface_v2_seed_prototype_status', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_action', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_build_required', '-')}",
                f"  adaptive surface v2 seed prototype family/geometry: {policy.get('adaptive_surface_v2_seed_prototype_family', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_geometry_status', '-')}",
                f"  adaptive surface v2 seed prototype orientation: {policy.get('adaptive_surface_v2_seed_prototype_orientation_status', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_orientation_action', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_orientation_confidence', '-')}",
                f"  adaptive surface v2 seed prototype normal-continuity mismatch score mean/max: {policy.get('adaptive_surface_v2_seed_prototype_side_score_mean', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_side_score_max', '-')}",
                f"  adaptive surface v2 seed prototype policies: {policy.get('adaptive_surface_v2_seed_prototype_support_context_policy', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_target_policy', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_curvature_policy', '-')} / {policy.get('adaptive_surface_v2_seed_prototype_confidence_policy', '-')}",
                f"  adaptive surface v2 seed prototype reasons: {policy.get('adaptive_surface_v2_seed_prototype_reasons', ())}",

                f"  adaptive surface v2 seed candidate status/action/selectable: {policy.get('adaptive_surface_v2_seed_candidate_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_action', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_selectable', '-')}",
                f"  adaptive surface v2 seed candidate family/geometry: {policy.get('adaptive_surface_v2_seed_candidate_family', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_status', '-')}",
                f"  adaptive surface v2 seed candidate field/support/frame: {policy.get('adaptive_surface_v2_seed_candidate_curvature_normal_field_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_support_filter', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_frame_policy', '-')}",
                f"  adaptive surface v2 seed candidate target/density/acceptance: {policy.get('adaptive_surface_v2_seed_candidate_target_policy', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_density_policy', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_acceptance_policy', '-')}",
                f"  adaptive surface v2 seed candidate vertices legacy/planned: {policy.get('adaptive_surface_v2_seed_candidate_legacy_seed_vertices', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_planned_seed_vertices', '-')}",
                f"  adaptive surface v2 seed candidate rings support/interior: {policy.get('adaptive_surface_v2_seed_candidate_planned_support_rings', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_planned_interior_rings', '-')}",
                f"  adaptive surface v2 seed candidate reasons: {policy.get('adaptive_surface_v2_seed_candidate_reasons', ())}",

                f"  adaptive surface v2 geometry probe status/action: {policy.get('adaptive_surface_v2_seed_candidate_geometry_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_action', '-')}",
                f"  adaptive surface v2 geometry probe available/applied/selected: {policy.get('adaptive_surface_v2_seed_candidate_geometry_available', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_applied', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_selected', '-')}",
                f"  adaptive surface v2 geometry probe family/mode: {policy.get('adaptive_surface_v2_seed_candidate_geometry_family', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_mode', '-')}",
                f"  adaptive surface v2 geometry probe faces/vertices/reoriented: {policy.get('adaptive_surface_v2_seed_candidate_geometry_face_count', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_vertex_count', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count', '-')}",
                f"  adaptive surface v2 geometry probe movement mean/max/ratio: {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_mean', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_max', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max', '-')}",
                f"  adaptive surface v2 geometry probe predicted G1 mean/max/status: {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status', '-')}",
                f"  adaptive surface v2 geometry probe topology: {policy.get('adaptive_surface_v2_seed_candidate_geometry_topology_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_topology_reasons', ())}",
                f"  adaptive surface v2 geometry probe reasons: {policy.get('adaptive_surface_v2_seed_candidate_geometry_reasons', ())}",
                f"  adaptive surface v2 geometry probe gate evaluation: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_action', '-')} / selectable={policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable', '-')}",
                f"  adaptive surface v2 geometry probe evaluated G1 mean/max/status: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status', '-')}",
                f"  adaptive surface v2 geometry probe evaluation quality/G2/policy: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status', '-')} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_policy', '-')}",
                f"  adaptive surface v2 geometry probe evaluation reasons: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons', ())}",
                f"  attempted strategies: {policy.get('adaptive_attempted_strategies', '-')}",
                f"  fallback used: {policy.get('adaptive_fallback_used', '-')}",
                f"  fallback reason: {policy.get('adaptive_fallback_reason', '-')}",
                f"  fallback error: {policy.get('adaptive_fallback_error', '')}",
                f"  seam status: {policy.get('seam_status', summary.get('seam_status', '-'))}",
                f"  quality status: {policy.get('quality_status', summary.get('quality_status', '-'))}",
                f"  relaxation status: {policy.get('relaxation_status', summary.get('relaxation_status', '-'))}",
                f"  density status: {policy.get('density_status', summary.get('density_status', '-'))}",
                f"  strict gate: {policy.get('strict_seam_topology_gate', '-')}",
                f"  strict gate reasons: {policy.get('strict_gate_reasons', ())}",
                f"  g1 status: {policy.get('g1_status', '-')}",
                f"  g1 boundary normal mean deviation: {policy.get('g1_boundary_normal_mean_deviation_degrees', '-')}°",
                f"  g1 boundary normal max deviation: {policy.get('g1_boundary_normal_max_deviation_degrees', '-')}°",
                f"  g1 support normal spread: {policy.get('g1_support_normal_spread_degrees', '-')}°",
                f"  g1 reasons: {policy.get('g1_reasons', ())}",
                f"  g1 gate: {policy.get('g1_gate_status', '-')}",
                f"  g1 gate reasons: {policy.get('g1_gate_reasons', ())}",
                f"  g1 relaxation policy: {policy.get('adaptive_g1_relaxation_policy', '-')}",
                f"  g1 policy reasons: {policy.get('adaptive_g1_policy_reasons', ())}",
                f"  feature context: {policy.get('adaptive_feature_context_kind', '-')}",
                f"  feature preservation mode: {policy.get('adaptive_feature_preservation_mode', '-')}",
                f"  feature policy reasons: {policy.get('adaptive_feature_policy_reasons', ())}",
                f"  feature recommended action: {policy.get('adaptive_feature_recommended_action', '-')}",
                f"  feature density mode: {policy.get('adaptive_feature_density_mode', '-')}",
                f"  feature boundary vertices: {policy.get('adaptive_feature_boundary_vertex_count', '-')}",
                f"  curvature status: {policy.get('adaptive_curvature_status', '-')}",
                f"  curvature context: {policy.get('adaptive_curvature_context_kind', '-')}",
                f"  curvature estimator: {policy.get('adaptive_curvature_estimator', '-')}",
                f"  support curvature mean: {policy.get('adaptive_support_curvature_mean', '-')}",
                f"  patch curvature mean: {policy.get('adaptive_patch_curvature_mean', '-')}",
                f"  curvature delta mean: {policy.get('adaptive_curvature_delta_mean', '-')}",
                f"  curvature delta max: {policy.get('adaptive_curvature_delta_max', '-')}",
                f"  curvature relative delta mean: {policy.get('adaptive_curvature_relative_delta_mean', '-')}",
                f"  curvature sign consistency: {policy.get('adaptive_curvature_sign_consistency', '-')}",
                f"  curvature samples: support={policy.get('adaptive_curvature_support_sample_count', '-')}, patch={policy.get('adaptive_curvature_patch_sample_count', '-')}",
                f"  curvature reasons: {policy.get('adaptive_curvature_reasons', ())}",
                f"  g2 status: {policy.get('g2_status', '-')}",
                f"  g2 context: {policy.get('g2_context', '-')}",
                f"  g2 reasons: {policy.get('g2_reasons', ())}",
                f"  g2 support curvature mean: {policy.get('g2_support_curvature_mean', '-')}",
                f"  g2 patch curvature mean: {policy.get('g2_patch_curvature_mean', '-')}",
                f"  g2 curvature delta mean: {policy.get('g2_curvature_delta_mean', '-')}",
                f"  g2 curvature delta max: {policy.get('g2_curvature_delta_max', '-')}",
                f"  g2 relative delta mean: {policy.get('g2_curvature_relative_delta_mean', '-')}",
                f"  g2 sign consistency: {policy.get('g2_curvature_sign_consistency', '-')}",
                f"  g2 gate status: {policy.get('g2_gate_status', '-')}",
                f"  g2 gate action: {policy.get('g2_gate_action', '-')}",
                f"  g2 gate commit allowed: {policy.get('g2_gate_commit_allowed', '-')}",
                f"  g2 gate reasons: {policy.get('g2_gate_reasons', ())}",
                f"  curvature fairing status: {policy.get('adaptive_curvature_fairing_status', '-')}",
                f"  curvature fairing action: {policy.get('adaptive_curvature_fairing_action', '-')}",
                f"  curvature fairing eligible: {policy.get('adaptive_curvature_fairing_eligible', '-')}",
                f"  curvature fairing needed: {policy.get('adaptive_curvature_fairing_needed', '-')}",
                f"  curvature fairing strength: {policy.get('adaptive_curvature_fairing_strength', '-')}",
                f"  curvature fairing iterations: {policy.get('adaptive_curvature_fairing_iterations', '-')}",
                f"  curvature fairing max displacement factor: {policy.get('adaptive_curvature_fairing_max_displacement_factor', '-')}",
                f"  curvature fairing reasons: {policy.get('adaptive_curvature_fairing_reasons', ())}",
                f"  curvature fairing trial status: {policy.get('adaptive_curvature_fairing_trial_status', '-')}",
                f"  curvature fairing trial action: {policy.get('adaptive_curvature_fairing_trial_action', '-')}",
                f"  curvature fairing trial attempted: {policy.get('adaptive_curvature_fairing_trial_attempted', '-')}",
                f"  curvature fairing trial applied: {policy.get('adaptive_curvature_fairing_trial_applied', '-')}",
                f"  curvature fairing trial accepted: {policy.get('adaptive_curvature_fairing_trial_accepted', '-')}",
                f"  curvature fairing trial mode: {policy.get('adaptive_curvature_fairing_trial_mode', '-')}",
                f"  curvature fairing trial max displacement: {policy.get('adaptive_curvature_fairing_trial_max_displacement', '-')}",
                f"  curvature fairing trial mean displacement: {policy.get('adaptive_curvature_fairing_trial_mean_displacement', '-')}",
                f"  curvature fairing trial movement ratio: {policy.get('adaptive_curvature_fairing_trial_movement_to_context_edge_ratio', '-')}",
                f"  curvature fairing trial reasons: {policy.get('adaptive_curvature_fairing_trial_reasons', ())}",
                f"  adaptive fairing gate status: {policy.get('adaptive_curvature_fairing_gate_status', '-')}",
                f"  adaptive fairing gate action: {policy.get('adaptive_curvature_fairing_gate_action', '-')}",
                f"  adaptive fairing accepted by gate: {policy.get('adaptive_curvature_fairing_accepted_by_adaptive_gate', '-')}",
                f"  adaptive fairing gate proposal limit: {policy.get('adaptive_curvature_fairing_gate_proposal_limit', '-')}",
                f"  adaptive fairing gate movement ratio: {policy.get('adaptive_curvature_fairing_gate_movement_ratio', '-')}",
                f"  adaptive fairing gate max displacement: {policy.get('adaptive_curvature_fairing_gate_max_displacement', '-')}",
                f"  adaptive fairing gate mean displacement: {policy.get('adaptive_curvature_fairing_gate_mean_displacement', '-')}",
                f"  adaptive fairing gate reasons: {policy.get('adaptive_curvature_fairing_gate_reasons', ())}",
                f"  end-layer status: {policy.get('adaptive_end_layer_status', '-')}",
                f"  end-layer action: {policy.get('adaptive_end_layer_action', '-')}",
                f"  end-layer local region available: {policy.get('adaptive_end_layer_local_region_available', '-')}",
                f"  end-layer patch available: {policy.get('adaptive_end_layer_patch_available', '-')}",
                f"  end-layer reference patch available: {policy.get('adaptive_end_layer_reference_patch_available', '-')}",
                f"  end-layer support rings: {policy.get('adaptive_end_layer_support_ring_count', '-')}",
                f"  end-layer support vertices: {policy.get('adaptive_end_layer_support_vertex_count', '-')}",
                f"  end-layer patch vertices: {policy.get('adaptive_end_layer_patch_vertex_count', '-')}",
                f"  end-layer curvature deviation mean: {policy.get('adaptive_end_layer_curvature_deviation_mean', '-')}",
                f"  end-layer curvature deviation max: {policy.get('adaptive_end_layer_curvature_deviation_max', '-')}",
                f"  end-layer relative curvature deviation mean: {policy.get('adaptive_end_layer_curvature_relative_deviation_mean', '-')}",
                f"  end-layer geometry deviation mean: {policy.get('adaptive_end_layer_geometry_deviation_mean', '-')}",
                f"  end-layer geometry deviation max: {policy.get('adaptive_end_layer_geometry_deviation_max', '-')}",
                f"  end-layer problem regions: {policy.get('adaptive_end_layer_problem_region_count', '-')}",
                f"  end-layer refinement recommended: {policy.get('adaptive_end_layer_refinement_recommended', '-')}",
                f"  end-layer rerun allowed: {policy.get('adaptive_end_layer_rerun_allowed', '-')}",
                f"  end-layer rerun reason: {policy.get('adaptive_end_layer_rerun_reason', '-')}",
                f"  end-layer selected patch source: {policy.get('adaptive_end_layer_selected_patch_source', '-')}",
                f"  end-layer reasons: {policy.get('adaptive_end_layer_reasons', ())}",
                f"  seed alignment status: {policy.get('adaptive_seed_alignment_status', '-')}",
                f"  seed alignment action: {policy.get('adaptive_seed_alignment_action', '-')}",
                f"  seed projection mean: {policy.get('adaptive_seed_projection_distance_mean', '-')}",
                f"  seed projection max: {policy.get('adaptive_seed_projection_distance_max', '-')}",
                f"  seed projection mean ratio: {policy.get('adaptive_seed_projection_mean_ratio', '-')}",
                f"  seed projection max ratio: {policy.get('adaptive_seed_projection_max_ratio', '-')}",
                f"  seed signed offset mean: {policy.get('adaptive_seed_signed_offset_mean', '-')}",
                f"  seed signed offset min: {policy.get('adaptive_seed_signed_offset_min', '-')}",
                f"  seed signed offset max: {policy.get('adaptive_seed_signed_offset_max', '-')}",
                f"  seed effective surface weight: {policy.get('adaptive_seed_effective_surface_weight', '-')}",
                f"  seed requested surface weight: {policy.get('adaptive_seed_requested_surface_weight', '-')}",
                f"  seed support normal spread: {policy.get('adaptive_seed_support_normal_spread_degrees', '-')}",
                f"  seed generated vertices: {policy.get('adaptive_seed_generated_vertex_count', '-')}",
                f"  seed context edge median: {policy.get('adaptive_seed_context_edge_length_median', '-')}",
                f"  seed alignment reasons: {policy.get('adaptive_seed_alignment_reasons', ())}",
                f"  target disagreement status: {policy.get('adaptive_target_disagreement_status', '-')}",
                f"  target disagreement action: {policy.get('adaptive_target_disagreement_action', '-')}",
                f"  target disagreement mean: {policy.get('adaptive_target_disagreement_mean', '-')}",
                f"  target disagreement max: {policy.get('adaptive_target_disagreement_max', '-')}",
                f"  target disagreement mean ratio: {policy.get('adaptive_target_disagreement_mean_ratio', '-')}",
                f"  target disagreement max ratio: {policy.get('adaptive_target_disagreement_max_ratio', '-')}",
                f"  target disagreement source: {policy.get('adaptive_target_disagreement_source', '-')}",
                f"  target model recommendation: {policy.get('adaptive_target_model_recommendation', '-')}",
                f"  target model confidence: {policy.get('adaptive_target_model_confidence', '-')}",
                f"  target confidence model: {policy.get('adaptive_target_confidence_model', '-')}",
                f"  target confidence min/mean/median/max: {policy.get('adaptive_target_confidence_min', '-')} / {policy.get('adaptive_target_confidence_mean', '-')} / {policy.get('adaptive_target_confidence_median', '-')} / {policy.get('adaptive_target_confidence_max', '-')}",
                f"  target confidence low vertices: {policy.get('adaptive_target_confidence_low_count', '-')} / {policy.get('adaptive_target_confidence_vertex_count', '-')} below {policy.get('adaptive_target_confidence_low_threshold', '-')}",
                f"  target recommended surface weight min/mean/max: {policy.get('adaptive_target_recommended_surface_weight_min', '-')} / {policy.get('adaptive_target_recommended_surface_weight_mean', '-')} / {policy.get('adaptive_target_recommended_surface_weight_max', '-')}",
                f"  target recommended surface weight limit: {policy.get('adaptive_target_recommended_surface_weight_limit', '-')}",
                f"  target confidence reasons: {policy.get('adaptive_target_confidence_profile_reasons', ())}",
                f"  confidence target probe status: {policy.get('adaptive_confidence_target_probe_status', '-')}",
                f"  confidence target probe action: {policy.get('adaptive_confidence_target_probe_action', '-')}",
                f"  confidence target probe attempted/applied/selected: {policy.get('adaptive_confidence_target_probe_attempted', '-')} / {policy.get('adaptive_confidence_target_probe_applied', '-')} / {policy.get('adaptive_confidence_target_probe_selected', '-')}",
                f"  confidence target probe basic gate: {policy.get('adaptive_confidence_target_probe_basic_gate_status', '-')}",
                f"  confidence target probe accepted by basic gate: {policy.get('adaptive_confidence_target_probe_accepted_by_basic_gate', '-')}",
                f"  confidence target probe movement mean/max: {policy.get('adaptive_confidence_target_probe_movement_mean', '-')} / {policy.get('adaptive_confidence_target_probe_movement_max', '-')}",
                f"  confidence target probe movement mean/max ratio: {policy.get('adaptive_confidence_target_probe_movement_mean_ratio', '-')} / {policy.get('adaptive_confidence_target_probe_movement_max_ratio', '-')}",
                f"  confidence target probe quality/g2 gates: {policy.get('adaptive_confidence_target_probe_quality_gate_status', '-')} / {policy.get('adaptive_confidence_target_probe_g2_gate_status', '-')}",
                f"  confidence target probe reasons: {policy.get('adaptive_confidence_target_probe_reasons', ())}",
                f"  confidence target candidate status: {policy.get('adaptive_confidence_target_candidate_status', '-')}",
                f"  confidence target candidate action: {policy.get('adaptive_confidence_target_candidate_action', '-')}",
                f"  confidence target candidate available/applied/selected: {policy.get('adaptive_confidence_target_candidate_available', '-')} / {policy.get('adaptive_confidence_target_candidate_applied', '-')} / {policy.get('adaptive_confidence_target_candidate_selected', '-')}",
                f"  confidence target candidate accepted by gates: {policy.get('adaptive_confidence_target_candidate_accepted_by_gates', '-')}",
                f"  confidence target candidate topology/quality/g2: {policy.get('adaptive_confidence_target_candidate_topology_status', '-')} / {policy.get('adaptive_confidence_target_candidate_quality_status', '-')} / {policy.get('adaptive_confidence_target_candidate_g2_status', '-')}",
                f"  confidence target candidate curvature delta mean/max/relative: {policy.get('adaptive_confidence_target_candidate_curvature_delta_mean', '-')} / {policy.get('adaptive_confidence_target_candidate_curvature_delta_max', '-')} / {policy.get('adaptive_confidence_target_candidate_curvature_relative_delta_mean', '-')}",
                f"  confidence target candidate movement mean/max: {policy.get('adaptive_confidence_target_candidate_movement_mean', '-')} / {policy.get('adaptive_confidence_target_candidate_movement_max', '-')}",
                f"  confidence target candidate movement mean/max ratio: {policy.get('adaptive_confidence_target_candidate_movement_mean_ratio', '-')} / {policy.get('adaptive_confidence_target_candidate_movement_max_ratio', '-')}",
                f"  confidence target candidate reasons: {policy.get('adaptive_confidence_target_candidate_reasons', ())}",
                f"  confidence target candidate selection status: {policy.get('adaptive_confidence_target_candidate_selection_status', '-')}",
                f"  confidence target candidate selected by policy: {policy.get('adaptive_confidence_target_candidate_selected_by_policy', '-')}",
                f"  confidence target candidate selected vertices: {policy.get('adaptive_confidence_target_candidate_selected_vertex_count', '-')}",
                f"  confidence target candidate applied delta mean/max: {policy.get('adaptive_confidence_target_candidate_applied_delta_mean', '-')} / {policy.get('adaptive_confidence_target_candidate_applied_delta_max', '-')}",
                f"  confidence target candidate selection reason: {policy.get('adaptive_confidence_target_candidate_selection_reason', '-')}",

                f"  directional target status/action: {policy.get('adaptive_directional_target_status', '-')} / {policy.get('adaptive_directional_target_action', '-')}",
                f"  directional target residual mean/max: {policy.get('adaptive_directional_target_residual_mean', '-')} / {policy.get('adaptive_directional_target_residual_max', '-')}",
                f"  directional target residual mean/max ratio: {policy.get('adaptive_directional_target_residual_mean_ratio', '-')} / {policy.get('adaptive_directional_target_residual_max_ratio', '-')}",
                f"  directional target normal residual mean/min/max: {policy.get('adaptive_directional_target_normal_residual_mean', '-')} / {policy.get('adaptive_directional_target_normal_residual_min', '-')} / {policy.get('adaptive_directional_target_normal_residual_max', '-')}",
                f"  directional target U/V/cross curvature: {policy.get('adaptive_directional_target_axis_u_curvature', '-')} / {policy.get('adaptive_directional_target_axis_v_curvature', '-')} / {policy.get('adaptive_directional_target_cross_curvature', '-')}",
                f"  directional target anisotropy/dominant axis: {policy.get('adaptive_directional_target_anisotropy_ratio', '-')} / {policy.get('adaptive_directional_target_dominant_axis', '-')}",
                f"  directional target cluster count/threshold: {policy.get('adaptive_directional_target_cluster_count', '-')} / {policy.get('adaptive_directional_target_cluster_threshold', '-')}",
                f"  directional target reasons: {policy.get('adaptive_directional_target_reasons', ())}",

                f"  anisotropic candidate status/action: {policy.get('adaptive_anisotropic_candidate_status', '-')} / {policy.get('adaptive_anisotropic_candidate_action', '-')}",
                f"  anisotropic candidate available/applied/selected: {policy.get('adaptive_anisotropic_candidate_available', '-')} / {policy.get('adaptive_anisotropic_candidate_applied', '-')} / {policy.get('adaptive_anisotropic_candidate_selected', '-')}",
                f"  anisotropic candidate accepted by gates: {policy.get('adaptive_anisotropic_candidate_accepted_by_gates', '-')}",
                f"  anisotropic candidate axis/cluster/influenced: {policy.get('adaptive_anisotropic_candidate_target_axis', '-')} / {policy.get('adaptive_anisotropic_candidate_cluster_count', '-')} / {policy.get('adaptive_anisotropic_candidate_influenced_vertices', '-')}",
                f"  anisotropic candidate movement mean/max: {policy.get('adaptive_anisotropic_candidate_movement_mean', '-')} / {policy.get('adaptive_anisotropic_candidate_movement_max', '-')}",
                f"  anisotropic candidate movement mean/max ratio: {policy.get('adaptive_anisotropic_candidate_movement_mean_ratio', '-')} / {policy.get('adaptive_anisotropic_candidate_movement_max_ratio', '-')}",
                f"  anisotropic candidate quality/g2: {policy.get('adaptive_anisotropic_candidate_quality_status', '-')} / {policy.get('adaptive_anisotropic_candidate_g2_status', '-')}",
                f"  anisotropic candidate curvature delta mean/max/relative: {policy.get('adaptive_anisotropic_candidate_curvature_delta_mean', '-')} / {policy.get('adaptive_anisotropic_candidate_curvature_delta_max', '-')} / {policy.get('adaptive_anisotropic_candidate_curvature_relative_delta_mean', '-')}",
                f"  anisotropic candidate reasons: {policy.get('adaptive_anisotropic_candidate_reasons', ())}",

                f"  anisotropic candidate selection status/policy: {policy.get('adaptive_anisotropic_candidate_selection_status', '-')} / {policy.get('adaptive_anisotropic_candidate_selection_policy', '-')}",
                f"  anisotropic candidate selected by policy: {policy.get('adaptive_anisotropic_candidate_selected_by_policy', '-')}",
                f"  anisotropic candidate selected vertices: {policy.get('adaptive_anisotropic_candidate_selected_vertex_count', '-')}",
                f"  anisotropic candidate selection reason: {policy.get('adaptive_anisotropic_candidate_selection_reason', '-')}",
                f"  target MLS2-vs-sphere mean: {policy.get('adaptive_target_mls2_vs_sphere_mean', '-')}",
                f"  target MLS2-vs-sphere max: {policy.get('adaptive_target_mls2_vs_sphere_max', '-')}",
                f"  target MLS2-vs-sphere mean ratio: {policy.get('adaptive_target_mls2_vs_sphere_mean_ratio', '-')}",
                f"  target MLS2-vs-sphere max ratio: {policy.get('adaptive_target_mls2_vs_sphere_max_ratio', '-')}",
                f"  target MLS2-vs-plane mean: {policy.get('adaptive_target_mls2_vs_plane_mean', '-')}",
                f"  target MLS2-vs-plane max: {policy.get('adaptive_target_mls2_vs_plane_max', '-')}",
                f"  target MLS2-vs-plane mean ratio: {policy.get('adaptive_target_mls2_vs_plane_mean_ratio', '-')}",
                f"  target MLS2-vs-plane max ratio: {policy.get('adaptive_target_mls2_vs_plane_max_ratio', '-')}",
                f"  target MLS1-vs-MLS2 mean: {policy.get('adaptive_target_mls1_vs_mls2_mean', '-')}",
                f"  target MLS1-vs-MLS2 max: {policy.get('adaptive_target_mls1_vs_mls2_max', '-')}",
                f"  target MLS2-vs-MLS3 mean: {policy.get('adaptive_target_mls2_vs_mls3_mean', '-')}",
                f"  target MLS2-vs-MLS3 max: {policy.get('adaptive_target_mls2_vs_mls3_max', '-')}",
                f"  target MLS1-vs-MLS2 mean ratio: {policy.get('adaptive_target_mls1_vs_mls2_mean_ratio', '-')}",
                f"  target MLS1-vs-MLS2 max ratio: {policy.get('adaptive_target_mls1_vs_mls2_max_ratio', '-')}",
                f"  target MLS2-vs-MLS3 mean ratio: {policy.get('adaptive_target_mls2_vs_mls3_mean_ratio', '-')}",
                f"  target MLS2-vs-MLS3 max ratio: {policy.get('adaptive_target_mls2_vs_mls3_max_ratio', '-')}",
                f"  target signed disagreement mean: {policy.get('adaptive_target_signed_disagreement_mean', '-')}",
                f"  target signed disagreement min: {policy.get('adaptive_target_signed_disagreement_min', '-')}",
                f"  target signed disagreement max: {policy.get('adaptive_target_signed_disagreement_max', '-')}",
                f"  target signed plane disagreement mean: {policy.get('adaptive_target_signed_plane_disagreement_mean', '-')}",
                f"  target signed plane disagreement min: {policy.get('adaptive_target_signed_plane_disagreement_min', '-')}",
                f"  target signed plane disagreement max: {policy.get('adaptive_target_signed_plane_disagreement_max', '-')}",
                f"  target MLS ring1 normal spread: {policy.get('adaptive_target_mls_ring1_normal_spread_degrees', '-')}",
                f"  target MLS ring2 normal spread: {policy.get('adaptive_target_mls_ring2_normal_spread_degrees', '-')}",
                f"  target MLS ring3 normal spread: {policy.get('adaptive_target_mls_ring3_normal_spread_degrees', '-')}",
                f"  target disagreement reasons: {policy.get('adaptive_target_disagreement_reasons', ())}",
                f"  end-layer rerun gate status: {policy.get('adaptive_end_layer_rerun_gate_status', '-')}",
                f"  end-layer rerun gate action: {policy.get('adaptive_end_layer_rerun_gate_action', '-')}",
                f"  end-layer rerun gate allowed: {policy.get('adaptive_end_layer_rerun_gate_allowed', '-')}",
                f"  end-layer rerun gate geometry deviation mean: {policy.get('adaptive_end_layer_rerun_gate_geometry_deviation_mean', '-')}",
                f"  end-layer rerun gate geometry deviation max: {policy.get('adaptive_end_layer_rerun_gate_geometry_deviation_max', '-')}",
                f"  end-layer rerun gate curvature deviation mean: {policy.get('adaptive_end_layer_rerun_gate_curvature_deviation_mean', '-')}",
                f"  end-layer rerun gate relative curvature deviation mean: {policy.get('adaptive_end_layer_rerun_gate_curvature_relative_deviation_mean', '-')}",
                f"  end-layer rerun gate reasons: {policy.get('adaptive_end_layer_rerun_gate_reasons', ())}",
                f"  conservative g1 attempted: {policy.get('adaptive_conservative_g1_attempted', '-')}",
                f"  conservative g1 used: {policy.get('adaptive_conservative_g1_used', '-')}",
                f"  conservative g1 reason: {policy.get('adaptive_conservative_g1_reason', '-')}",
                f"  selected relaxation iterations: {policy.get('adaptive_selected_relaxation_iterations', '-')}",
                f"  selected relaxation strength: {policy.get('adaptive_selected_relaxation_strength', '-')}",
                f"  selected surface weight: {policy.get('adaptive_selected_surface_weight', '-')}",
                f"  adaptive score decision: {policy.get('adaptive_score_decision', '-')}",
                f"  primary score: {policy.get('adaptive_primary_score', '-')}",
                f"  conservative g1 score: {policy.get('adaptive_conservative_g1_score', '-')}",
                f"  selected score: {policy.get('adaptive_selected_score', '-')}",
                f"  score delta: {policy.get('adaptive_score_delta', '-')}",
            ]

            if blocking_reasons:
                policy_lines.append("  blocking reasons:")
                policy_lines.extend(f"    - {reason}" for reason in blocking_reasons)

            if warnings:
                policy_lines.append("  warnings:")
                policy_lines.extend(f"    - {warning}" for warning in warnings)


            self._set_topology_result_text(
                "\n".join(
                    [
                        "Hole fill preview",
                        f"Scope: {self._topology_scope_label(face_ids)}",
                        f"Candidate: {int(summary.get('candidate_index', 0)) + 1}",
                        f"Method: {policy.get('method', summary.get('method'))}",
                        f"Backend: {policy.get('backend', summary.get('backend', summary.get('method')))}",
                        f"Adaptive stage: {policy.get('adaptive_stage', '-')}",
                        f"Boundary vertices: {summary.get('boundary_vertices')}",
                        f"Boundary edges: {summary.get('boundary_edges')}",
                        f"Area hint: {self._format_optional_float(summary.get('area_hint'))}",
                        f"Perimeter: {self._format_optional_float(summary.get('perimeter'))}",
                        "",
                        *policy_lines,
                        "",
                        "Preview-only: mesh data was not modified.",
                        "Red double-sided overlay shows only the newly generated fill patch before commit.",
                    ]
                )
            )

            self.log(
                "Hole fill preview ready: "
                f"candidate {int(summary.get('candidate_index', 0)) + 1}, "
                f"method={policy.get('method', summary.get('method'))}, "
                f"backend={policy.get('backend', summary.get('backend', summary.get('method')))}, "
                f"commit_allowed={commit_allowed}"
            )

            if not commit_allowed:
                self.log(
                    "Hole fill commit blocked: "
                    + "; ".join(str(reason) for reason in blocking_reasons)
                )

            self.statusBar().showMessage(
                str(gui_policy_state["status_bar_text"]),
                3000,
            )
            self._update_brush_action_state()

        self._run_task("Building hole fill preview...", task, on_success)

    def _on_hole_fill_commit_clicked(self) -> None:
        if self._hole_fill_preview is None:
            QMessageBox.information(self, "No preview", "Build a hole fill preview first.")
            return

        preview = self._hole_fill_preview
        preview_summary = getattr(preview, "selection_summary", None)
        if not isinstance(preview_summary, dict):
            preview_summary = {}

        policy = self._hole_fill_preview_commit_policy_for_gui(preview)

        method_for_preflight = (
            policy.get("public_method")
            or preview_summary.get("public_method")
            or policy.get("method")
            or preview_summary.get("method")
        )

        def task() -> object:
            return self.processor.commit_hole_fill_preview(preview)

        def on_success(result: object) -> None:
            assert isinstance(result, ManualEditResult)

            # Clear the overlay before replacing the viewport mesh.
            self._clear_hole_fill_preview(silent=True)
            self.current_output_path = None
            self._refresh_viewport_from_processor()
            self._set_mesh_info_from_trimesh(self.processor.mesh)

            # Mesh topology changed; old selections and candidate lists are stale.
            try:
                self.selection_controller.clear_selection(
                    keep_mode=True,
                    push=True,
                    reason="hole_fill_committed",
                )
            except Exception:
                pass

            self._reset_hole_fill_ui(
                status=(
                    f"Hole fill committed: faces {result.before_faces} -> {result.after_faces} | "
                    f"vertices {result.before_vertices} -> {result.after_vertices}. "
                    "Run Find Hole Candidates again if needed."
                )
            )

            self._set_topology_result_text(
                "\n".join(
                    [
                        "Hole fill committed",
                        f"Operation: {result.operation}",
                        f"Faces: {result.before_faces} -> {result.after_faces}",
                        f"Vertices: {result.before_vertices} -> {result.after_vertices}",
                        "",
                        "Committed preview:",
                        f"  method: {preview_summary.get('method', policy.get('method', '-'))}",
                        f"  public method: {policy.get('public_method', preview_summary.get('public_method', '-'))}",
                        f"  backend: {policy.get('backend', preview_summary.get('backend', '-'))}",
                        f"  adaptive stage: {policy.get('adaptive_stage', preview_summary.get('adaptive_stage', '-'))}",
                        f"  commit policy: {'allowed' if bool(policy.get('commit_allowed', True)) else 'blocked'}",
                        f"  context kind: {policy.get('adaptive_context_kind', preview_summary.get('adaptive_context_kind', '-'))}",
                        f"  context confidence: {policy.get('adaptive_context_confidence', preview_summary.get('adaptive_context_confidence', '-'))}",
                        f"  seed strategy: {policy.get('selected_seed_strategy', preview_summary.get('selected_seed_strategy', '-'))}",
                        f"  adaptive controller: {policy.get('adaptive_controller', preview_summary.get('adaptive_controller', '-'))}",

                        f"  adaptive surface v2 status/case/action: {policy.get('adaptive_surface_v2_status', preview_summary.get('adaptive_surface_v2_status', '-'))} / {policy.get('adaptive_surface_v2_case', preview_summary.get('adaptive_surface_v2_case', '-'))} / {policy.get('adaptive_surface_v2_action', preview_summary.get('adaptive_surface_v2_action', '-'))}",
                        f"  adaptive surface v2 block legacy selection: {policy.get('adaptive_surface_v2_block_legacy_selection', preview_summary.get('adaptive_surface_v2_block_legacy_selection', '-'))}",
                        f"  adaptive surface v2 require new seed: {policy.get('adaptive_surface_v2_require_new_seed', preview_summary.get('adaptive_surface_v2_require_new_seed', '-'))}",
                        f"  adaptive surface v2 seed/target policy: {policy.get('adaptive_surface_v2_recommended_seed_family', preview_summary.get('adaptive_surface_v2_recommended_seed_family', '-'))} / {policy.get('adaptive_surface_v2_recommended_target_policy', preview_summary.get('adaptive_surface_v2_recommended_target_policy', '-'))}",
                        f"  adaptive surface v2 reasons: {policy.get('adaptive_surface_v2_reasons', preview_summary.get('adaptive_surface_v2_reasons', ())) }",

                        f"  adaptive surface v2 seed plan status/action/build: {policy.get('adaptive_surface_v2_seed_plan_status', preview_summary.get('adaptive_surface_v2_seed_plan_status', '-'))} / {policy.get('adaptive_surface_v2_seed_plan_action', preview_summary.get('adaptive_surface_v2_seed_plan_action', '-'))} / {policy.get('adaptive_surface_v2_seed_plan_build_required', preview_summary.get('adaptive_surface_v2_seed_plan_build_required', '-'))}",
                        f"  adaptive surface v2 seed family: {policy.get('adaptive_surface_v2_seed_family', preview_summary.get('adaptive_surface_v2_seed_family', '-'))}",
                        f"  adaptive surface v2 orientation case/action: {policy.get('adaptive_surface_v2_orientation_case', preview_summary.get('adaptive_surface_v2_orientation_case', '-'))} / {policy.get('adaptive_surface_v2_orientation_action', preview_summary.get('adaptive_surface_v2_orientation_action', '-'))}",
                        f"  adaptive surface v2 target/curvature/confidence policies: {policy.get('adaptive_surface_v2_target_policy', preview_summary.get('adaptive_surface_v2_target_policy', '-'))} / {policy.get('adaptive_surface_v2_curvature_policy', preview_summary.get('adaptive_surface_v2_curvature_policy', '-'))} / {policy.get('adaptive_surface_v2_confidence_policy', preview_summary.get('adaptive_surface_v2_confidence_policy', '-'))}",
                        f"  adaptive surface v2 support context policy: {policy.get('adaptive_surface_v2_support_context_policy', preview_summary.get('adaptive_surface_v2_support_context_policy', '-'))}",
                        f"  adaptive surface v2 seed plan reasons: {policy.get('adaptive_surface_v2_seed_plan_reasons', preview_summary.get('adaptive_surface_v2_seed_plan_reasons', ())) }",

                        f"  adaptive surface v2 seed prototype status/action/build: {policy.get('adaptive_surface_v2_seed_prototype_status', preview_summary.get('adaptive_surface_v2_seed_prototype_status', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_action', preview_summary.get('adaptive_surface_v2_seed_prototype_action', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_build_required', preview_summary.get('adaptive_surface_v2_seed_prototype_build_required', '-'))}",
                        f"  adaptive surface v2 seed prototype family/geometry: {policy.get('adaptive_surface_v2_seed_prototype_family', preview_summary.get('adaptive_surface_v2_seed_prototype_family', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_geometry_status', preview_summary.get('adaptive_surface_v2_seed_prototype_geometry_status', '-'))}",
                        f"  adaptive surface v2 seed prototype orientation: {policy.get('adaptive_surface_v2_seed_prototype_orientation_status', preview_summary.get('adaptive_surface_v2_seed_prototype_orientation_status', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_orientation_action', preview_summary.get('adaptive_surface_v2_seed_prototype_orientation_action', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_orientation_confidence', preview_summary.get('adaptive_surface_v2_seed_prototype_orientation_confidence', '-'))}",
                        f"  adaptive surface v2 seed prototype normal-continuity mismatch score mean/max: {policy.get('adaptive_surface_v2_seed_prototype_side_score_mean', preview_summary.get('adaptive_surface_v2_seed_prototype_side_score_mean', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_side_score_max', preview_summary.get('adaptive_surface_v2_seed_prototype_side_score_max', '-'))}",
                        f"  adaptive surface v2 seed prototype policies: {policy.get('adaptive_surface_v2_seed_prototype_support_context_policy', preview_summary.get('adaptive_surface_v2_seed_prototype_support_context_policy', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_target_policy', preview_summary.get('adaptive_surface_v2_seed_prototype_target_policy', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_curvature_policy', preview_summary.get('adaptive_surface_v2_seed_prototype_curvature_policy', '-'))} / {policy.get('adaptive_surface_v2_seed_prototype_confidence_policy', preview_summary.get('adaptive_surface_v2_seed_prototype_confidence_policy', '-'))}",
                        f"  adaptive surface v2 seed prototype reasons: {policy.get('adaptive_surface_v2_seed_prototype_reasons', preview_summary.get('adaptive_surface_v2_seed_prototype_reasons', ())) }",

                        f"  adaptive surface v2 seed candidate status/action/selectable: {policy.get('adaptive_surface_v2_seed_candidate_status', preview_summary.get('adaptive_surface_v2_seed_candidate_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_action', preview_summary.get('adaptive_surface_v2_seed_candidate_action', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_selectable', preview_summary.get('adaptive_surface_v2_seed_candidate_selectable', '-'))}",
                        f"  adaptive surface v2 seed candidate family/geometry: {policy.get('adaptive_surface_v2_seed_candidate_family', preview_summary.get('adaptive_surface_v2_seed_candidate_family', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_status', '-'))}",
                        f"  adaptive surface v2 seed candidate field/support/frame: {policy.get('adaptive_surface_v2_seed_candidate_curvature_normal_field_status', preview_summary.get('adaptive_surface_v2_seed_candidate_curvature_normal_field_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_support_filter', preview_summary.get('adaptive_surface_v2_seed_candidate_support_filter', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_frame_policy', preview_summary.get('adaptive_surface_v2_seed_candidate_frame_policy', '-'))}",
                        f"  adaptive surface v2 seed candidate target/density/acceptance: {policy.get('adaptive_surface_v2_seed_candidate_target_policy', preview_summary.get('adaptive_surface_v2_seed_candidate_target_policy', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_density_policy', preview_summary.get('adaptive_surface_v2_seed_candidate_density_policy', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_acceptance_policy', preview_summary.get('adaptive_surface_v2_seed_candidate_acceptance_policy', '-'))}",
                        f"  adaptive surface v2 seed candidate vertices legacy/planned: {policy.get('adaptive_surface_v2_seed_candidate_legacy_seed_vertices', preview_summary.get('adaptive_surface_v2_seed_candidate_legacy_seed_vertices', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_planned_seed_vertices', preview_summary.get('adaptive_surface_v2_seed_candidate_planned_seed_vertices', '-'))}",
                        f"  adaptive surface v2 seed candidate rings support/interior: {policy.get('adaptive_surface_v2_seed_candidate_planned_support_rings', preview_summary.get('adaptive_surface_v2_seed_candidate_planned_support_rings', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_planned_interior_rings', preview_summary.get('adaptive_surface_v2_seed_candidate_planned_interior_rings', '-'))}",
                        f"  adaptive surface v2 seed candidate reasons: {policy.get('adaptive_surface_v2_seed_candidate_reasons', preview_summary.get('adaptive_surface_v2_seed_candidate_reasons', ())) }",

                        f"  adaptive surface v2 geometry probe status/action: {policy.get('adaptive_surface_v2_seed_candidate_geometry_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_action', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_action', '-'))}",
                        f"  adaptive surface v2 geometry probe available/applied/selected: {policy.get('adaptive_surface_v2_seed_candidate_geometry_available', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_available', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_applied', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_applied', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_selected', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_selected', '-'))}",
                        f"  adaptive surface v2 geometry probe family/mode: {policy.get('adaptive_surface_v2_seed_candidate_geometry_family', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_family', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_mode', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_mode', '-'))}",
                        f"  adaptive surface v2 geometry probe faces/vertices/reoriented: {policy.get('adaptive_surface_v2_seed_candidate_geometry_face_count', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_face_count', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_vertex_count', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_vertex_count', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_reoriented_face_count', '-'))}",
                        f"  adaptive surface v2 geometry probe movement mean/max/ratio: {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_mean', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_movement_mean', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_max', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_movement_max', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_movement_ratio_max', '-'))}",
                        f"  adaptive surface v2 geometry probe predicted G1 mean/max/status: {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_mean_deviation', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_max_deviation', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_predicted_g1_status', '-'))}",
                        f"  adaptive surface v2 geometry probe topology: {policy.get('adaptive_surface_v2_seed_candidate_geometry_topology_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_topology_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_topology_reasons', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_topology_reasons', ())) }",
                        f"  adaptive surface v2 geometry probe reasons: {policy.get('adaptive_surface_v2_seed_candidate_geometry_reasons', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_reasons', ())) }",
                        f"  adaptive surface v2 geometry probe gate evaluation: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_action', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_action', '-'))} / selectable={policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_selectable', '-'))}",
                        f"  adaptive surface v2 geometry probe evaluated G1 mean/max/status: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_mean_deviation', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_max_deviation', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g1_status', '-'))}",
                        f"  adaptive surface v2 geometry probe evaluation quality/G2/policy: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_quality_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_g2_status', '-'))} / {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_policy', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_policy', '-'))}",
                        f"  adaptive surface v2 geometry probe evaluation reasons: {policy.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons', preview_summary.get('adaptive_surface_v2_seed_candidate_geometry_evaluation_reasons', ())) }",
                        f"  attempted strategies: {policy.get('adaptive_attempted_strategies', preview_summary.get('adaptive_attempted_strategies', '-'))}",
                        f"  fallback used: {policy.get('adaptive_fallback_used', preview_summary.get('adaptive_fallback_used', '-'))}",
                        f"  fallback reason: {policy.get('adaptive_fallback_reason', preview_summary.get('adaptive_fallback_reason', '-'))}",
                        f"  fallback error: {policy.get('adaptive_fallback_error', preview_summary.get('adaptive_fallback_error', ''))}",
                        f"  seam status: {policy.get('seam_status', preview_summary.get('seam_status', '-'))}",
                        f"  seam coverage: {policy.get('seam_coverage_ratio', preview_summary.get('seam_coverage_ratio', '-'))}",
                        f"  seam missing edges: {policy.get('seam_missing_edge_count', preview_summary.get('seam_missing_edge_count', '-'))}",
                        f"  seam overused edges: {policy.get('seam_overused_edge_count', preview_summary.get('seam_overused_edge_count', '-'))}",
                        f"  seam weak edges: {policy.get('seam_weak_edge_count', preview_summary.get('seam_weak_edge_count', '-'))}",
                        f"  seam problem edges: {policy.get('seam_problem_edge_count', preview_summary.get('seam_problem_edge_count', '-'))}",
                        f"  seam recovery required: {policy.get('seam_recovery_required', preview_summary.get('seam_recovery_required', '-'))}",
                        f"  seam recovery strategy: {policy.get('seam_recovery_strategy', preview_summary.get('seam_recovery_strategy', '-'))}",
                        f"  quality status: {policy.get('quality_status', preview_summary.get('quality_status', '-'))}",
                        f"  relaxation status: {policy.get('relaxation_status', preview_summary.get('relaxation_status', '-'))}",
                        f"  density status: {policy.get('density_status', preview_summary.get('density_status', '-'))}",
                        f"  strict gate: {policy.get('strict_seam_topology_gate', preview_summary.get('strict_seam_topology_gate', '-'))}",
                        f"  strict gate reasons: {policy.get('strict_gate_reasons', preview_summary.get('strict_gate_reasons', ())) }",
                        f"  g1 status: {policy.get('g1_status', preview_summary.get('g1_status', '-'))}",
                        f"  g1 boundary normal mean deviation: {policy.get('g1_boundary_normal_mean_deviation_degrees', preview_summary.get('g1_boundary_normal_mean_deviation_degrees', '-'))}°",
                        f"  g1 boundary normal max deviation: {policy.get('g1_boundary_normal_max_deviation_degrees', preview_summary.get('g1_boundary_normal_max_deviation_degrees', '-'))}°",
                        f"  g1 support normal spread: {policy.get('g1_support_normal_spread_degrees', preview_summary.get('g1_support_normal_spread_degrees', '-'))}°",
                        f"  g1 reasons: {policy.get('g1_reasons', preview_summary.get('g1_reasons', ())) }",
                        f"  g1 gate: {policy.get('g1_gate_status', preview_summary.get('g1_gate_status', '-'))}",
                        f"  g1 gate reasons: {policy.get('g1_gate_reasons', preview_summary.get('g1_gate_reasons', ())) }",
                        f"  g1 relaxation policy: {policy.get('adaptive_g1_relaxation_policy', preview_summary.get('adaptive_g1_relaxation_policy', '-'))}",
                        f"  g1 policy reasons: {policy.get('adaptive_g1_policy_reasons', preview_summary.get('adaptive_g1_policy_reasons', ())) }",
                        f"  feature context: {policy.get('adaptive_feature_context_kind', preview_summary.get('adaptive_feature_context_kind', '-'))}",
                        f"  feature preservation mode: {policy.get('adaptive_feature_preservation_mode', preview_summary.get('adaptive_feature_preservation_mode', '-'))}",
                        f"  feature policy reasons: {policy.get('adaptive_feature_policy_reasons', preview_summary.get('adaptive_feature_policy_reasons', ())) }",
                        f"  feature recommended action: {policy.get('adaptive_feature_recommended_action', preview_summary.get('adaptive_feature_recommended_action', '-'))}",
                        f"  feature density mode: {policy.get('adaptive_feature_density_mode', preview_summary.get('adaptive_feature_density_mode', '-'))}",
                        f"  feature boundary vertices: {policy.get('adaptive_feature_boundary_vertex_count', preview_summary.get('adaptive_feature_boundary_vertex_count', '-'))}",
                        f"  curvature status: {policy.get('adaptive_curvature_status', preview_summary.get('adaptive_curvature_status', '-'))}",
                        f"  curvature context: {policy.get('adaptive_curvature_context_kind', preview_summary.get('adaptive_curvature_context_kind', '-'))}",
                        f"  curvature estimator: {policy.get('adaptive_curvature_estimator', preview_summary.get('adaptive_curvature_estimator', '-'))}",
                        f"  support curvature mean: {policy.get('adaptive_support_curvature_mean', preview_summary.get('adaptive_support_curvature_mean', '-'))}",
                        f"  support curvature max: {policy.get('adaptive_support_curvature_max', preview_summary.get('adaptive_support_curvature_max', '-'))}",
                        f"  support curvature std: {policy.get('adaptive_support_curvature_std', preview_summary.get('adaptive_support_curvature_std', '-'))}",
                        f"  patch curvature mean: {policy.get('adaptive_patch_curvature_mean', preview_summary.get('adaptive_patch_curvature_mean', '-'))}",
                        f"  patch curvature max: {policy.get('adaptive_patch_curvature_max', preview_summary.get('adaptive_patch_curvature_max', '-'))}",
                        f"  patch curvature std: {policy.get('adaptive_patch_curvature_std', preview_summary.get('adaptive_patch_curvature_std', '-'))}",
                        f"  curvature delta mean: {policy.get('adaptive_curvature_delta_mean', preview_summary.get('adaptive_curvature_delta_mean', '-'))}",
                        f"  curvature delta max: {policy.get('adaptive_curvature_delta_max', preview_summary.get('adaptive_curvature_delta_max', '-'))}",
                        f"  curvature relative delta mean: {policy.get('adaptive_curvature_relative_delta_mean', preview_summary.get('adaptive_curvature_relative_delta_mean', '-'))}",
                        f"  curvature sign consistency: {policy.get('adaptive_curvature_sign_consistency', preview_summary.get('adaptive_curvature_sign_consistency', '-'))}",
                        f"  curvature samples: support={policy.get('adaptive_curvature_support_sample_count', preview_summary.get('adaptive_curvature_support_sample_count', '-'))}, patch={policy.get('adaptive_curvature_patch_sample_count', preview_summary.get('adaptive_curvature_patch_sample_count', '-'))}",
                        f"  curvature reasons: {policy.get('adaptive_curvature_reasons', preview_summary.get('adaptive_curvature_reasons', ())) }",
                        f"  g2 status: {policy.get('g2_status', preview_summary.get('g2_status', '-'))}",
                        f"  g2 context: {policy.get('g2_context', preview_summary.get('g2_context', '-'))}",
                        f"  g2 reasons: {policy.get('g2_reasons', preview_summary.get('g2_reasons', ())) }",
                        f"  g2 support curvature mean: {policy.get('g2_support_curvature_mean', preview_summary.get('g2_support_curvature_mean', '-'))}",
                        f"  g2 patch curvature mean: {policy.get('g2_patch_curvature_mean', preview_summary.get('g2_patch_curvature_mean', '-'))}",
                        f"  g2 curvature delta mean: {policy.get('g2_curvature_delta_mean', preview_summary.get('g2_curvature_delta_mean', '-'))}",
                        f"  g2 curvature delta max: {policy.get('g2_curvature_delta_max', preview_summary.get('g2_curvature_delta_max', '-'))}",
                        f"  g2 relative delta mean: {policy.get('g2_curvature_relative_delta_mean', preview_summary.get('g2_curvature_relative_delta_mean', '-'))}",
                        f"  g2 sign consistency: {policy.get('g2_curvature_sign_consistency', preview_summary.get('g2_curvature_sign_consistency', '-'))}",
                        f"  g2 gate status: {policy.get('g2_gate_status', preview_summary.get('g2_gate_status', '-'))}",
                        f"  g2 gate action: {policy.get('g2_gate_action', preview_summary.get('g2_gate_action', '-'))}",
                        f"  g2 gate commit allowed: {policy.get('g2_gate_commit_allowed', preview_summary.get('g2_gate_commit_allowed', '-'))}",
                        f"  g2 gate reasons: {policy.get('g2_gate_reasons', preview_summary.get('g2_gate_reasons', ())) }",
                        f"  curvature fairing status: {policy.get('adaptive_curvature_fairing_status', preview_summary.get('adaptive_curvature_fairing_status', '-'))}",
                        f"  curvature fairing action: {policy.get('adaptive_curvature_fairing_action', preview_summary.get('adaptive_curvature_fairing_action', '-'))}",
                        f"  curvature fairing eligible: {policy.get('adaptive_curvature_fairing_eligible', preview_summary.get('adaptive_curvature_fairing_eligible', '-'))}",
                        f"  curvature fairing needed: {policy.get('adaptive_curvature_fairing_needed', preview_summary.get('adaptive_curvature_fairing_needed', '-'))}",
                        f"  curvature fairing strength: {policy.get('adaptive_curvature_fairing_strength', preview_summary.get('adaptive_curvature_fairing_strength', '-'))}",
                        f"  curvature fairing iterations: {policy.get('adaptive_curvature_fairing_iterations', preview_summary.get('adaptive_curvature_fairing_iterations', '-'))}",
                        f"  curvature fairing max displacement factor: {policy.get('adaptive_curvature_fairing_max_displacement_factor', preview_summary.get('adaptive_curvature_fairing_max_displacement_factor', '-'))}",
                        f"  curvature fairing reasons: {policy.get('adaptive_curvature_fairing_reasons', preview_summary.get('adaptive_curvature_fairing_reasons', ())) }",
                        f"  curvature fairing trial status: {policy.get('adaptive_curvature_fairing_trial_status', preview_summary.get('adaptive_curvature_fairing_trial_status', '-'))}",
                        f"  curvature fairing trial action: {policy.get('adaptive_curvature_fairing_trial_action', preview_summary.get('adaptive_curvature_fairing_trial_action', '-'))}",
                        f"  curvature fairing trial attempted: {policy.get('adaptive_curvature_fairing_trial_attempted', preview_summary.get('adaptive_curvature_fairing_trial_attempted', '-'))}",
                        f"  curvature fairing trial applied: {policy.get('adaptive_curvature_fairing_trial_applied', preview_summary.get('adaptive_curvature_fairing_trial_applied', '-'))}",
                        f"  curvature fairing trial accepted: {policy.get('adaptive_curvature_fairing_trial_accepted', preview_summary.get('adaptive_curvature_fairing_trial_accepted', '-'))}",
                        f"  curvature fairing trial mode: {policy.get('adaptive_curvature_fairing_trial_mode', preview_summary.get('adaptive_curvature_fairing_trial_mode', '-'))}",
                        f"  curvature fairing trial max displacement: {policy.get('adaptive_curvature_fairing_trial_max_displacement', preview_summary.get('adaptive_curvature_fairing_trial_max_displacement', '-'))}",
                        f"  curvature fairing trial mean displacement: {policy.get('adaptive_curvature_fairing_trial_mean_displacement', preview_summary.get('adaptive_curvature_fairing_trial_mean_displacement', '-'))}",
                        f"  curvature fairing trial movement ratio: {policy.get('adaptive_curvature_fairing_trial_movement_to_context_edge_ratio', preview_summary.get('adaptive_curvature_fairing_trial_movement_to_context_edge_ratio', '-'))}",
                        f"  curvature fairing trial reasons: {policy.get('adaptive_curvature_fairing_trial_reasons', preview_summary.get('adaptive_curvature_fairing_trial_reasons', ())) }",
                        f"  adaptive fairing gate status: {policy.get('adaptive_curvature_fairing_gate_status', preview_summary.get('adaptive_curvature_fairing_gate_status', '-'))}",
                        f"  adaptive fairing gate action: {policy.get('adaptive_curvature_fairing_gate_action', preview_summary.get('adaptive_curvature_fairing_gate_action', '-'))}",
                        f"  adaptive fairing accepted by gate: {policy.get('adaptive_curvature_fairing_accepted_by_adaptive_gate', preview_summary.get('adaptive_curvature_fairing_accepted_by_adaptive_gate', '-'))}",
                        f"  adaptive fairing gate proposal limit: {policy.get('adaptive_curvature_fairing_gate_proposal_limit', preview_summary.get('adaptive_curvature_fairing_gate_proposal_limit', '-'))}",
                        f"  adaptive fairing gate movement ratio: {policy.get('adaptive_curvature_fairing_gate_movement_ratio', preview_summary.get('adaptive_curvature_fairing_gate_movement_ratio', '-'))}",
                        f"  adaptive fairing gate max displacement: {policy.get('adaptive_curvature_fairing_gate_max_displacement', preview_summary.get('adaptive_curvature_fairing_gate_max_displacement', '-'))}",
                        f"  adaptive fairing gate mean displacement: {policy.get('adaptive_curvature_fairing_gate_mean_displacement', preview_summary.get('adaptive_curvature_fairing_gate_mean_displacement', '-'))}",
                        f"  adaptive fairing gate reasons: {policy.get('adaptive_curvature_fairing_gate_reasons', preview_summary.get('adaptive_curvature_fairing_gate_reasons', ())) }",
                        f"  end-layer status: {policy.get('adaptive_end_layer_status', preview_summary.get('adaptive_end_layer_status', '-'))}",
                        f"  end-layer action: {policy.get('adaptive_end_layer_action', preview_summary.get('adaptive_end_layer_action', '-'))}",
                        f"  end-layer local region available: {policy.get('adaptive_end_layer_local_region_available', preview_summary.get('adaptive_end_layer_local_region_available', '-'))}",
                        f"  end-layer patch available: {policy.get('adaptive_end_layer_patch_available', preview_summary.get('adaptive_end_layer_patch_available', '-'))}",
                        f"  end-layer reference patch available: {policy.get('adaptive_end_layer_reference_patch_available', preview_summary.get('adaptive_end_layer_reference_patch_available', '-'))}",
                        f"  end-layer support rings: {policy.get('adaptive_end_layer_support_ring_count', preview_summary.get('adaptive_end_layer_support_ring_count', '-'))}",
                        f"  end-layer support vertices: {policy.get('adaptive_end_layer_support_vertex_count', preview_summary.get('adaptive_end_layer_support_vertex_count', '-'))}",
                        f"  end-layer patch vertices: {policy.get('adaptive_end_layer_patch_vertex_count', preview_summary.get('adaptive_end_layer_patch_vertex_count', '-'))}",
                        f"  end-layer curvature deviation mean: {policy.get('adaptive_end_layer_curvature_deviation_mean', preview_summary.get('adaptive_end_layer_curvature_deviation_mean', '-'))}",
                        f"  end-layer curvature deviation max: {policy.get('adaptive_end_layer_curvature_deviation_max', preview_summary.get('adaptive_end_layer_curvature_deviation_max', '-'))}",
                        f"  end-layer relative curvature deviation mean: {policy.get('adaptive_end_layer_curvature_relative_deviation_mean', preview_summary.get('adaptive_end_layer_curvature_relative_deviation_mean', '-'))}",
                        f"  end-layer geometry deviation mean: {policy.get('adaptive_end_layer_geometry_deviation_mean', preview_summary.get('adaptive_end_layer_geometry_deviation_mean', '-'))}",
                        f"  end-layer geometry deviation max: {policy.get('adaptive_end_layer_geometry_deviation_max', preview_summary.get('adaptive_end_layer_geometry_deviation_max', '-'))}",
                        f"  end-layer problem regions: {policy.get('adaptive_end_layer_problem_region_count', preview_summary.get('adaptive_end_layer_problem_region_count', '-'))}",
                        f"  end-layer refinement recommended: {policy.get('adaptive_end_layer_refinement_recommended', preview_summary.get('adaptive_end_layer_refinement_recommended', '-'))}",
                        f"  end-layer rerun allowed: {policy.get('adaptive_end_layer_rerun_allowed', preview_summary.get('adaptive_end_layer_rerun_allowed', '-'))}",
                        f"  end-layer rerun reason: {policy.get('adaptive_end_layer_rerun_reason', preview_summary.get('adaptive_end_layer_rerun_reason', '-'))}",
                        f"  end-layer selected patch source: {policy.get('adaptive_end_layer_selected_patch_source', preview_summary.get('adaptive_end_layer_selected_patch_source', '-'))}",
                        f"  end-layer reasons: {policy.get('adaptive_end_layer_reasons', preview_summary.get('adaptive_end_layer_reasons', ())) }",
                        f"  seed alignment status: {policy.get('adaptive_seed_alignment_status', preview_summary.get('adaptive_seed_alignment_status', '-'))}",
                        f"  seed alignment action: {policy.get('adaptive_seed_alignment_action', preview_summary.get('adaptive_seed_alignment_action', '-'))}",
                        f"  seed projection mean: {policy.get('adaptive_seed_projection_distance_mean', preview_summary.get('adaptive_seed_projection_distance_mean', '-'))}",
                        f"  seed projection max: {policy.get('adaptive_seed_projection_distance_max', preview_summary.get('adaptive_seed_projection_distance_max', '-'))}",
                        f"  seed projection mean ratio: {policy.get('adaptive_seed_projection_mean_ratio', preview_summary.get('adaptive_seed_projection_mean_ratio', '-'))}",
                        f"  seed projection max ratio: {policy.get('adaptive_seed_projection_max_ratio', preview_summary.get('adaptive_seed_projection_max_ratio', '-'))}",
                        f"  seed signed offset mean: {policy.get('adaptive_seed_signed_offset_mean', preview_summary.get('adaptive_seed_signed_offset_mean', '-'))}",
                        f"  seed signed offset min: {policy.get('adaptive_seed_signed_offset_min', preview_summary.get('adaptive_seed_signed_offset_min', '-'))}",
                        f"  seed signed offset max: {policy.get('adaptive_seed_signed_offset_max', preview_summary.get('adaptive_seed_signed_offset_max', '-'))}",
                        f"  seed effective surface weight: {policy.get('adaptive_seed_effective_surface_weight', preview_summary.get('adaptive_seed_effective_surface_weight', '-'))}",
                        f"  seed requested surface weight: {policy.get('adaptive_seed_requested_surface_weight', preview_summary.get('adaptive_seed_requested_surface_weight', '-'))}",
                        f"  seed support normal spread: {policy.get('adaptive_seed_support_normal_spread_degrees', preview_summary.get('adaptive_seed_support_normal_spread_degrees', '-'))}",
                        f"  seed generated vertices: {policy.get('adaptive_seed_generated_vertex_count', preview_summary.get('adaptive_seed_generated_vertex_count', '-'))}",
                        f"  seed context edge median: {policy.get('adaptive_seed_context_edge_length_median', preview_summary.get('adaptive_seed_context_edge_length_median', '-'))}",
                        f"  seed alignment reasons: {policy.get('adaptive_seed_alignment_reasons', preview_summary.get('adaptive_seed_alignment_reasons', ())) }",
                        f"  target disagreement status: {policy.get('adaptive_target_disagreement_status', preview_summary.get('adaptive_target_disagreement_status', '-'))}",
                        f"  target disagreement action: {policy.get('adaptive_target_disagreement_action', preview_summary.get('adaptive_target_disagreement_action', '-'))}",
                        f"  target disagreement mean: {policy.get('adaptive_target_disagreement_mean', preview_summary.get('adaptive_target_disagreement_mean', '-'))}",
                        f"  target disagreement max: {policy.get('adaptive_target_disagreement_max', preview_summary.get('adaptive_target_disagreement_max', '-'))}",
                        f"  target disagreement mean ratio: {policy.get('adaptive_target_disagreement_mean_ratio', preview_summary.get('adaptive_target_disagreement_mean_ratio', '-'))}",
                        f"  target disagreement max ratio: {policy.get('adaptive_target_disagreement_max_ratio', preview_summary.get('adaptive_target_disagreement_max_ratio', '-'))}",
                        f"  target disagreement source: {policy.get('adaptive_target_disagreement_source', preview_summary.get('adaptive_target_disagreement_source', '-'))}",
                        f"  target model recommendation: {policy.get('adaptive_target_model_recommendation', preview_summary.get('adaptive_target_model_recommendation', '-'))}",
                        f"  target model confidence: {policy.get('adaptive_target_model_confidence', preview_summary.get('adaptive_target_model_confidence', '-'))}",
                        f"  target confidence model: {policy.get('adaptive_target_confidence_model', preview_summary.get('adaptive_target_confidence_model', '-'))}",
                        f"  target confidence min/mean/median/max: {policy.get('adaptive_target_confidence_min', preview_summary.get('adaptive_target_confidence_min', '-'))} / {policy.get('adaptive_target_confidence_mean', preview_summary.get('adaptive_target_confidence_mean', '-'))} / {policy.get('adaptive_target_confidence_median', preview_summary.get('adaptive_target_confidence_median', '-'))} / {policy.get('adaptive_target_confidence_max', preview_summary.get('adaptive_target_confidence_max', '-'))}",
                        f"  target confidence low vertices: {policy.get('adaptive_target_confidence_low_count', preview_summary.get('adaptive_target_confidence_low_count', '-'))} / {policy.get('adaptive_target_confidence_vertex_count', preview_summary.get('adaptive_target_confidence_vertex_count', '-'))} below {policy.get('adaptive_target_confidence_low_threshold', preview_summary.get('adaptive_target_confidence_low_threshold', '-'))}",
                        f"  target recommended surface weight min/mean/max: {policy.get('adaptive_target_recommended_surface_weight_min', preview_summary.get('adaptive_target_recommended_surface_weight_min', '-'))} / {policy.get('adaptive_target_recommended_surface_weight_mean', preview_summary.get('adaptive_target_recommended_surface_weight_mean', '-'))} / {policy.get('adaptive_target_recommended_surface_weight_max', preview_summary.get('adaptive_target_recommended_surface_weight_max', '-'))}",
                        f"  target recommended surface weight limit: {policy.get('adaptive_target_recommended_surface_weight_limit', preview_summary.get('adaptive_target_recommended_surface_weight_limit', '-'))}",
                        f"  target confidence reasons: {policy.get('adaptive_target_confidence_profile_reasons', preview_summary.get('adaptive_target_confidence_profile_reasons', ())) }",
                        f"  confidence target probe status: {policy.get('adaptive_confidence_target_probe_status', preview_summary.get('adaptive_confidence_target_probe_status', '-'))}",
                        f"  confidence target probe action: {policy.get('adaptive_confidence_target_probe_action', preview_summary.get('adaptive_confidence_target_probe_action', '-'))}",
                        f"  confidence target probe attempted/applied/selected: {policy.get('adaptive_confidence_target_probe_attempted', preview_summary.get('adaptive_confidence_target_probe_attempted', '-'))} / {policy.get('adaptive_confidence_target_probe_applied', preview_summary.get('adaptive_confidence_target_probe_applied', '-'))} / {policy.get('adaptive_confidence_target_probe_selected', preview_summary.get('adaptive_confidence_target_probe_selected', '-'))}",
                        f"  confidence target probe basic gate: {policy.get('adaptive_confidence_target_probe_basic_gate_status', preview_summary.get('adaptive_confidence_target_probe_basic_gate_status', '-'))}",
                        f"  confidence target probe accepted by basic gate: {policy.get('adaptive_confidence_target_probe_accepted_by_basic_gate', preview_summary.get('adaptive_confidence_target_probe_accepted_by_basic_gate', '-'))}",
                        f"  confidence target probe movement mean/max: {policy.get('adaptive_confidence_target_probe_movement_mean', preview_summary.get('adaptive_confidence_target_probe_movement_mean', '-'))} / {policy.get('adaptive_confidence_target_probe_movement_max', preview_summary.get('adaptive_confidence_target_probe_movement_max', '-'))}",
                        f"  confidence target probe movement mean/max ratio: {policy.get('adaptive_confidence_target_probe_movement_mean_ratio', preview_summary.get('adaptive_confidence_target_probe_movement_mean_ratio', '-'))} / {policy.get('adaptive_confidence_target_probe_movement_max_ratio', preview_summary.get('adaptive_confidence_target_probe_movement_max_ratio', '-'))}",
                        f"  confidence target probe quality/g2 gates: {policy.get('adaptive_confidence_target_probe_quality_gate_status', preview_summary.get('adaptive_confidence_target_probe_quality_gate_status', '-'))} / {policy.get('adaptive_confidence_target_probe_g2_gate_status', preview_summary.get('adaptive_confidence_target_probe_g2_gate_status', '-'))}",
                        f"  confidence target probe reasons: {policy.get('adaptive_confidence_target_probe_reasons', preview_summary.get('adaptive_confidence_target_probe_reasons', ())) }",
                        f"  confidence target candidate status: {policy.get('adaptive_confidence_target_candidate_status', preview_summary.get('adaptive_confidence_target_candidate_status', '-'))}",
                        f"  confidence target candidate action: {policy.get('adaptive_confidence_target_candidate_action', preview_summary.get('adaptive_confidence_target_candidate_action', '-'))}",
                        f"  confidence target candidate available/applied/selected: {policy.get('adaptive_confidence_target_candidate_available', preview_summary.get('adaptive_confidence_target_candidate_available', '-'))} / {policy.get('adaptive_confidence_target_candidate_applied', preview_summary.get('adaptive_confidence_target_candidate_applied', '-'))} / {policy.get('adaptive_confidence_target_candidate_selected', preview_summary.get('adaptive_confidence_target_candidate_selected', '-'))}",
                        f"  confidence target candidate accepted by gates: {policy.get('adaptive_confidence_target_candidate_accepted_by_gates', preview_summary.get('adaptive_confidence_target_candidate_accepted_by_gates', '-'))}",
                        f"  confidence target candidate topology/quality/g2: {policy.get('adaptive_confidence_target_candidate_topology_status', preview_summary.get('adaptive_confidence_target_candidate_topology_status', '-'))} / {policy.get('adaptive_confidence_target_candidate_quality_status', preview_summary.get('adaptive_confidence_target_candidate_quality_status', '-'))} / {policy.get('adaptive_confidence_target_candidate_g2_status', preview_summary.get('adaptive_confidence_target_candidate_g2_status', '-'))}",
                        f"  confidence target candidate curvature delta mean/max/relative: {policy.get('adaptive_confidence_target_candidate_curvature_delta_mean', preview_summary.get('adaptive_confidence_target_candidate_curvature_delta_mean', '-'))} / {policy.get('adaptive_confidence_target_candidate_curvature_delta_max', preview_summary.get('adaptive_confidence_target_candidate_curvature_delta_max', '-'))} / {policy.get('adaptive_confidence_target_candidate_curvature_relative_delta_mean', preview_summary.get('adaptive_confidence_target_candidate_curvature_relative_delta_mean', '-'))}",
                        f"  confidence target candidate movement mean/max: {policy.get('adaptive_confidence_target_candidate_movement_mean', preview_summary.get('adaptive_confidence_target_candidate_movement_mean', '-'))} / {policy.get('adaptive_confidence_target_candidate_movement_max', preview_summary.get('adaptive_confidence_target_candidate_movement_max', '-'))}",
                        f"  confidence target candidate movement mean/max ratio: {policy.get('adaptive_confidence_target_candidate_movement_mean_ratio', preview_summary.get('adaptive_confidence_target_candidate_movement_mean_ratio', '-'))} / {policy.get('adaptive_confidence_target_candidate_movement_max_ratio', preview_summary.get('adaptive_confidence_target_candidate_movement_max_ratio', '-'))}",
                        f"  confidence target candidate reasons: {policy.get('adaptive_confidence_target_candidate_reasons', preview_summary.get('adaptive_confidence_target_candidate_reasons', ())) }",
                        f"  confidence target candidate selection status: {policy.get('adaptive_confidence_target_candidate_selection_status', preview_summary.get('adaptive_confidence_target_candidate_selection_status', '-'))}",
                        f"  confidence target candidate selected by policy: {policy.get('adaptive_confidence_target_candidate_selected_by_policy', preview_summary.get('adaptive_confidence_target_candidate_selected_by_policy', '-'))}",
                        f"  confidence target candidate selected vertices: {policy.get('adaptive_confidence_target_candidate_selected_vertex_count', preview_summary.get('adaptive_confidence_target_candidate_selected_vertex_count', '-'))}",
                        f"  confidence target candidate applied delta mean/max: {policy.get('adaptive_confidence_target_candidate_applied_delta_mean', preview_summary.get('adaptive_confidence_target_candidate_applied_delta_mean', '-'))} / {policy.get('adaptive_confidence_target_candidate_applied_delta_max', preview_summary.get('adaptive_confidence_target_candidate_applied_delta_max', '-'))}",
                        f"  confidence target candidate selection reason: {policy.get('adaptive_confidence_target_candidate_selection_reason', preview_summary.get('adaptive_confidence_target_candidate_selection_reason', '-'))}",

                        f"  directional target status/action: {policy.get('adaptive_directional_target_status', preview_summary.get('adaptive_directional_target_status', '-'))} / {policy.get('adaptive_directional_target_action', preview_summary.get('adaptive_directional_target_action', '-'))}",
                        f"  directional target residual mean/max: {policy.get('adaptive_directional_target_residual_mean', preview_summary.get('adaptive_directional_target_residual_mean', '-'))} / {policy.get('adaptive_directional_target_residual_max', preview_summary.get('adaptive_directional_target_residual_max', '-'))}",
                        f"  directional target residual mean/max ratio: {policy.get('adaptive_directional_target_residual_mean_ratio', preview_summary.get('adaptive_directional_target_residual_mean_ratio', '-'))} / {policy.get('adaptive_directional_target_residual_max_ratio', preview_summary.get('adaptive_directional_target_residual_max_ratio', '-'))}",
                        f"  directional target normal residual mean/min/max: {policy.get('adaptive_directional_target_normal_residual_mean', preview_summary.get('adaptive_directional_target_normal_residual_mean', '-'))} / {policy.get('adaptive_directional_target_normal_residual_min', preview_summary.get('adaptive_directional_target_normal_residual_min', '-'))} / {policy.get('adaptive_directional_target_normal_residual_max', preview_summary.get('adaptive_directional_target_normal_residual_max', '-'))}",
                        f"  directional target U/V/cross curvature: {policy.get('adaptive_directional_target_axis_u_curvature', preview_summary.get('adaptive_directional_target_axis_u_curvature', '-'))} / {policy.get('adaptive_directional_target_axis_v_curvature', preview_summary.get('adaptive_directional_target_axis_v_curvature', '-'))} / {policy.get('adaptive_directional_target_cross_curvature', preview_summary.get('adaptive_directional_target_cross_curvature', '-'))}",
                        f"  directional target anisotropy/dominant axis: {policy.get('adaptive_directional_target_anisotropy_ratio', preview_summary.get('adaptive_directional_target_anisotropy_ratio', '-'))} / {policy.get('adaptive_directional_target_dominant_axis', preview_summary.get('adaptive_directional_target_dominant_axis', '-'))}",
                        f"  directional target cluster count/threshold: {policy.get('adaptive_directional_target_cluster_count', preview_summary.get('adaptive_directional_target_cluster_count', '-'))} / {policy.get('adaptive_directional_target_cluster_threshold', preview_summary.get('adaptive_directional_target_cluster_threshold', '-'))}",
                        f"  directional target reasons: {policy.get('adaptive_directional_target_reasons', preview_summary.get('adaptive_directional_target_reasons', ())) }",

                        f"  anisotropic candidate status/action: {policy.get('adaptive_anisotropic_candidate_status', preview_summary.get('adaptive_anisotropic_candidate_status', '-'))} / {policy.get('adaptive_anisotropic_candidate_action', preview_summary.get('adaptive_anisotropic_candidate_action', '-'))}",
                        f"  anisotropic candidate available/applied/selected: {policy.get('adaptive_anisotropic_candidate_available', preview_summary.get('adaptive_anisotropic_candidate_available', '-'))} / {policy.get('adaptive_anisotropic_candidate_applied', preview_summary.get('adaptive_anisotropic_candidate_applied', '-'))} / {policy.get('adaptive_anisotropic_candidate_selected', preview_summary.get('adaptive_anisotropic_candidate_selected', '-'))}",
                        f"  anisotropic candidate accepted by gates: {policy.get('adaptive_anisotropic_candidate_accepted_by_gates', preview_summary.get('adaptive_anisotropic_candidate_accepted_by_gates', '-'))}",
                        f"  anisotropic candidate axis/cluster/influenced: {policy.get('adaptive_anisotropic_candidate_target_axis', preview_summary.get('adaptive_anisotropic_candidate_target_axis', '-'))} / {policy.get('adaptive_anisotropic_candidate_cluster_count', preview_summary.get('adaptive_anisotropic_candidate_cluster_count', '-'))} / {policy.get('adaptive_anisotropic_candidate_influenced_vertices', preview_summary.get('adaptive_anisotropic_candidate_influenced_vertices', '-'))}",
                        f"  anisotropic candidate movement mean/max: {policy.get('adaptive_anisotropic_candidate_movement_mean', preview_summary.get('adaptive_anisotropic_candidate_movement_mean', '-'))} / {policy.get('adaptive_anisotropic_candidate_movement_max', preview_summary.get('adaptive_anisotropic_candidate_movement_max', '-'))}",
                        f"  anisotropic candidate movement mean/max ratio: {policy.get('adaptive_anisotropic_candidate_movement_mean_ratio', preview_summary.get('adaptive_anisotropic_candidate_movement_mean_ratio', '-'))} / {policy.get('adaptive_anisotropic_candidate_movement_max_ratio', preview_summary.get('adaptive_anisotropic_candidate_movement_max_ratio', '-'))}",
                        f"  anisotropic candidate quality/g2: {policy.get('adaptive_anisotropic_candidate_quality_status', preview_summary.get('adaptive_anisotropic_candidate_quality_status', '-'))} / {policy.get('adaptive_anisotropic_candidate_g2_status', preview_summary.get('adaptive_anisotropic_candidate_g2_status', '-'))}",
                        f"  anisotropic candidate curvature delta mean/max/relative: {policy.get('adaptive_anisotropic_candidate_curvature_delta_mean', preview_summary.get('adaptive_anisotropic_candidate_curvature_delta_mean', '-'))} / {policy.get('adaptive_anisotropic_candidate_curvature_delta_max', preview_summary.get('adaptive_anisotropic_candidate_curvature_delta_max', '-'))} / {policy.get('adaptive_anisotropic_candidate_curvature_relative_delta_mean', preview_summary.get('adaptive_anisotropic_candidate_curvature_relative_delta_mean', '-'))}",
                        f"  anisotropic candidate reasons: {policy.get('adaptive_anisotropic_candidate_reasons', preview_summary.get('adaptive_anisotropic_candidate_reasons', ())) }",

                        f"  anisotropic candidate selection status/policy: {policy.get('adaptive_anisotropic_candidate_selection_status', preview_summary.get('adaptive_anisotropic_candidate_selection_status', '-'))} / {policy.get('adaptive_anisotropic_candidate_selection_policy', preview_summary.get('adaptive_anisotropic_candidate_selection_policy', '-'))}",
                        f"  anisotropic candidate selected by policy: {policy.get('adaptive_anisotropic_candidate_selected_by_policy', preview_summary.get('adaptive_anisotropic_candidate_selected_by_policy', '-'))}",
                        f"  anisotropic candidate selected vertices: {policy.get('adaptive_anisotropic_candidate_selected_vertex_count', preview_summary.get('adaptive_anisotropic_candidate_selected_vertex_count', '-'))}",
                        f"  anisotropic candidate selection reason: {policy.get('adaptive_anisotropic_candidate_selection_reason', preview_summary.get('adaptive_anisotropic_candidate_selection_reason', '-'))}",
                        f"  target MLS2-vs-sphere mean: {policy.get('adaptive_target_mls2_vs_sphere_mean', preview_summary.get('adaptive_target_mls2_vs_sphere_mean', '-'))}",
                        f"  target MLS2-vs-sphere max: {policy.get('adaptive_target_mls2_vs_sphere_max', preview_summary.get('adaptive_target_mls2_vs_sphere_max', '-'))}",
                        f"  target MLS2-vs-sphere mean ratio: {policy.get('adaptive_target_mls2_vs_sphere_mean_ratio', preview_summary.get('adaptive_target_mls2_vs_sphere_mean_ratio', '-'))}",
                        f"  target MLS2-vs-sphere max ratio: {policy.get('adaptive_target_mls2_vs_sphere_max_ratio', preview_summary.get('adaptive_target_mls2_vs_sphere_max_ratio', '-'))}",
                        f"  target MLS2-vs-plane mean: {policy.get('adaptive_target_mls2_vs_plane_mean', preview_summary.get('adaptive_target_mls2_vs_plane_mean', '-'))}",
                        f"  target MLS2-vs-plane max: {policy.get('adaptive_target_mls2_vs_plane_max', preview_summary.get('adaptive_target_mls2_vs_plane_max', '-'))}",
                        f"  target MLS2-vs-plane mean ratio: {policy.get('adaptive_target_mls2_vs_plane_mean_ratio', preview_summary.get('adaptive_target_mls2_vs_plane_mean_ratio', '-'))}",
                        f"  target MLS2-vs-plane max ratio: {policy.get('adaptive_target_mls2_vs_plane_max_ratio', preview_summary.get('adaptive_target_mls2_vs_plane_max_ratio', '-'))}",
                        f"  target MLS1-vs-MLS2 mean: {policy.get('adaptive_target_mls1_vs_mls2_mean', preview_summary.get('adaptive_target_mls1_vs_mls2_mean', '-'))}",
                        f"  target MLS1-vs-MLS2 max: {policy.get('adaptive_target_mls1_vs_mls2_max', preview_summary.get('adaptive_target_mls1_vs_mls2_max', '-'))}",
                        f"  target MLS2-vs-MLS3 mean: {policy.get('adaptive_target_mls2_vs_mls3_mean', preview_summary.get('adaptive_target_mls2_vs_mls3_mean', '-'))}",
                        f"  target MLS2-vs-MLS3 max: {policy.get('adaptive_target_mls2_vs_mls3_max', preview_summary.get('adaptive_target_mls2_vs_mls3_max', '-'))}",
                        f"  target MLS1-vs-MLS2 mean ratio: {policy.get('adaptive_target_mls1_vs_mls2_mean_ratio', preview_summary.get('adaptive_target_mls1_vs_mls2_mean_ratio', '-'))}",
                        f"  target MLS1-vs-MLS2 max ratio: {policy.get('adaptive_target_mls1_vs_mls2_max_ratio', preview_summary.get('adaptive_target_mls1_vs_mls2_max_ratio', '-'))}",
                        f"  target MLS2-vs-MLS3 mean ratio: {policy.get('adaptive_target_mls2_vs_mls3_mean_ratio', preview_summary.get('adaptive_target_mls2_vs_mls3_mean_ratio', '-'))}",
                        f"  target MLS2-vs-MLS3 max ratio: {policy.get('adaptive_target_mls2_vs_mls3_max_ratio', preview_summary.get('adaptive_target_mls2_vs_mls3_max_ratio', '-'))}",
                        f"  target signed disagreement mean: {policy.get('adaptive_target_signed_disagreement_mean', preview_summary.get('adaptive_target_signed_disagreement_mean', '-'))}",
                        f"  target signed disagreement min: {policy.get('adaptive_target_signed_disagreement_min', preview_summary.get('adaptive_target_signed_disagreement_min', '-'))}",
                        f"  target signed disagreement max: {policy.get('adaptive_target_signed_disagreement_max', preview_summary.get('adaptive_target_signed_disagreement_max', '-'))}",
                        f"  target signed plane disagreement mean: {policy.get('adaptive_target_signed_plane_disagreement_mean', preview_summary.get('adaptive_target_signed_plane_disagreement_mean', '-'))}",
                        f"  target signed plane disagreement min: {policy.get('adaptive_target_signed_plane_disagreement_min', preview_summary.get('adaptive_target_signed_plane_disagreement_min', '-'))}",
                        f"  target signed plane disagreement max: {policy.get('adaptive_target_signed_plane_disagreement_max', preview_summary.get('adaptive_target_signed_plane_disagreement_max', '-'))}",
                        f"  target MLS ring1 normal spread: {policy.get('adaptive_target_mls_ring1_normal_spread_degrees', preview_summary.get('adaptive_target_mls_ring1_normal_spread_degrees', '-'))}",
                        f"  target MLS ring2 normal spread: {policy.get('adaptive_target_mls_ring2_normal_spread_degrees', preview_summary.get('adaptive_target_mls_ring2_normal_spread_degrees', '-'))}",
                        f"  target MLS ring3 normal spread: {policy.get('adaptive_target_mls_ring3_normal_spread_degrees', preview_summary.get('adaptive_target_mls_ring3_normal_spread_degrees', '-'))}",
                        f"  target disagreement reasons: {policy.get('adaptive_target_disagreement_reasons', preview_summary.get('adaptive_target_disagreement_reasons', ())) }",
                        f"  end-layer rerun gate status: {policy.get('adaptive_end_layer_rerun_gate_status', preview_summary.get('adaptive_end_layer_rerun_gate_status', '-'))}",
                        f"  end-layer rerun gate action: {policy.get('adaptive_end_layer_rerun_gate_action', preview_summary.get('adaptive_end_layer_rerun_gate_action', '-'))}",
                        f"  end-layer rerun gate allowed: {policy.get('adaptive_end_layer_rerun_gate_allowed', preview_summary.get('adaptive_end_layer_rerun_gate_allowed', '-'))}",
                        f"  end-layer rerun gate geometry deviation mean: {policy.get('adaptive_end_layer_rerun_gate_geometry_deviation_mean', preview_summary.get('adaptive_end_layer_rerun_gate_geometry_deviation_mean', '-'))}",
                        f"  end-layer rerun gate geometry deviation max: {policy.get('adaptive_end_layer_rerun_gate_geometry_deviation_max', preview_summary.get('adaptive_end_layer_rerun_gate_geometry_deviation_max', '-'))}",
                        f"  end-layer rerun gate curvature deviation mean: {policy.get('adaptive_end_layer_rerun_gate_curvature_deviation_mean', preview_summary.get('adaptive_end_layer_rerun_gate_curvature_deviation_mean', '-'))}",
                        f"  end-layer rerun gate relative curvature deviation mean: {policy.get('adaptive_end_layer_rerun_gate_curvature_relative_deviation_mean', preview_summary.get('adaptive_end_layer_rerun_gate_curvature_relative_deviation_mean', '-'))}",
                        f"  end-layer rerun gate reasons: {policy.get('adaptive_end_layer_rerun_gate_reasons', preview_summary.get('adaptive_end_layer_rerun_gate_reasons', ())) }",
                        f"  conservative g1 attempted: {policy.get('adaptive_conservative_g1_attempted', preview_summary.get('adaptive_conservative_g1_attempted', '-'))}",
                        f"  conservative g1 used: {policy.get('adaptive_conservative_g1_used', preview_summary.get('adaptive_conservative_g1_used', '-'))}",
                        f"  conservative g1 reason: {policy.get('adaptive_conservative_g1_reason', preview_summary.get('adaptive_conservative_g1_reason', '-'))}",
                        f"  selected relaxation iterations: {policy.get('adaptive_selected_relaxation_iterations', preview_summary.get('adaptive_selected_relaxation_iterations', '-'))}",
                        f"  selected relaxation strength: {policy.get('adaptive_selected_relaxation_strength', preview_summary.get('adaptive_selected_relaxation_strength', '-'))}",
                        f"  selected surface weight: {policy.get('adaptive_selected_surface_weight', preview_summary.get('adaptive_selected_surface_weight', '-'))}",
                        "",
                        "Preview overlay cleared.",
                        "Selection and stale hole candidates cleared.",
                    ]
                )
            )

            for note in result.notes:
                self.log(f"Hole fill note: {note}")
            self.log(
                f"Hole fill committed: faces {result.before_faces} -> {result.after_faces} | "
                f"vertices {result.before_vertices} -> {result.after_vertices}"
            )
            self.statusBar().showMessage("Hole fill committed", 3000)
            self._update_undo_redo_action_state()
            _update_project_status_ui_if_available(self)
            self._sync_viewport_ui_from_backend()

        self._run_task("Committing hole fill preview...", task, on_success)

    def _on_hole_fill_cancel_clicked(self) -> None:
        self._clear_hole_fill_preview(silent=True)
        self._set_hole_fill_status("Hole fill preview cleared.")
        self.log("Hole fill preview cleared.")

    def _format_topology_report(self, report: object, face_ids: tuple[int, ...] | None) -> str:
        scope = self._topology_scope_label(face_ids)
        component_sizes = getattr(report, "component_sizes", ())
        try:
            component_sizes_text = ", ".join(str(int(v)) for v in component_sizes) or "-"
        except Exception:
            component_sizes_text = str(component_sizes)

        return "\n".join(
            [
                "Topology analysis",
                f"Scope: {scope}",
                "",
                f"Mesh faces: {getattr(report, 'face_count', '-')}",
                f"Selected faces: {getattr(report, 'selected_face_count', '-')}",
                f"Connected components: {getattr(report, 'component_count', '-')}",
                f"Component sizes: {component_sizes_text}",
                f"Boundary edges: {getattr(report, 'boundary_edge_count', '-')}",
                f"Boundary loops/chains: {getattr(report, 'boundary_loop_count', '-')}",
                f"Closed boundary loops: {getattr(report, 'closed_boundary_loop_count', '-')}",
                f"Open boundary chains: {getattr(report, 'open_boundary_chain_count', '-')}",
                "",
                "Read-only: mesh data was not modified.",
            ]
        )

    def _format_hole_candidates(self, candidates: list[object], face_ids: tuple[int, ...] | None) -> str:
        scope = self._topology_scope_label(face_ids)
        lines = [
            "Hole candidate analysis",
            f"Scope: {scope}",
            f"Candidates: {len(candidates)}",
            "",
        ]

        if not candidates:
            lines.append("No candidate hole boundaries matched the current filters.")
            lines.append("Read-only: mesh data was not modified.")
            return "\n".join(lines)

        diagnostics = self._hole_candidate_diagnostics_for_gui(candidates)

        for index, candidate in enumerate(candidates, start=1):
            classified = getattr(candidate, "classified_loop", None)
            kind = getattr(classified, "kind", None)
            kind_text = str(getattr(kind, "value", kind or "unknown"))
            boundary_vertices = getattr(candidate, "boundary_vertices", ()) or ()
            boundary_edges = getattr(candidate, "boundary_edges", ()) or ()

            diagnostic = diagnostics[index - 1] if index - 1 < len(diagnostics) else None
            diagnostic_kind = self._hole_candidate_diagnostic_kind_text_for_gui(diagnostic)
            diagnostic_confidence = self._hole_candidate_diagnostic_confidence_text_for_gui(diagnostic)
            diagnostic_notes = self._hole_candidate_diagnostic_notes_text_for_gui(diagnostic)

            lines.extend(
                [
                    f"Candidate {index}",
                    f"  kind: {kind_text}",
                    f"  diagnostic kind: {diagnostic_kind}",
                    f"  diagnostic confidence: {diagnostic_confidence}",
                    f"  diagnostic notes: {diagnostic_notes}",
                    f"  boundary vertices: {len(boundary_vertices)}",
                    f"  boundary edges: {len(boundary_edges)}",
                    f"  perimeter: {self._format_optional_float(getattr(candidate, 'perimeter', None))}",
                    f"  area hint: {self._format_optional_float(getattr(candidate, 'area_hint', None))}",
                    f"  centroid: {self._format_optional_centroid(getattr(candidate, 'centroid', None))}",
                    f"  fill priority: {self._format_optional_float(getattr(candidate, 'fill_priority', None))}",
                    "",
                ]
            )

        lines.append("Read-only: mesh data was not modified.")
        return "\n".join(lines)

    def _on_analyze_topology_clicked(self) -> None:
        if getattr(self.processor, "mesh", None) is None:
            QMessageBox.information(self, "No mesh loaded", "Load a mesh before analyzing topology.")
            return

        try:
            face_ids = self._current_topology_face_ids()
            report = self.processor.analyze_topology(face_ids=face_ids)
            text = self._format_topology_report(report, face_ids)
        except Exception as exc:
            QMessageBox.critical(self, "Topology analysis failed", str(exc))
            self.log(f"Topology analysis failed: {exc}")
            return

        self._set_topology_result_text(text)
        self.log(f"Topology analysis complete for {self._topology_scope_label(face_ids)}.")
        self.statusBar().showMessage("Topology analysis complete", 3000)

    def _on_find_hole_candidates_clicked(self) -> None:
        if getattr(self.processor, "mesh", None) is None:
            QMessageBox.information(self, "No mesh loaded", "Load a mesh before finding hole candidates.")
            return

        try:
            # Hole filling defaults to whole-mesh hole detection.
            # Selection-aware topology analysis remains available through
            # Analyze Topology, but Find Hole Candidates should locate real
            # open mesh holes by default rather than treating a selected face
            # boundary as a fill target.
            face_ids = None
            max_area_hint, max_perimeter = self._hole_filter_values()
            candidates = list(
                self.processor.find_hole_candidates(
                    face_ids=face_ids,
                    max_area_hint=max_area_hint,
                    max_perimeter=max_perimeter,
                )
            )
            text = self._format_hole_candidates(candidates, face_ids)
        except Exception as exc:
            QMessageBox.critical(self, "Hole candidate analysis failed", str(exc))
            self.log(f"Hole candidate analysis failed: {exc}")
            return

        self._set_topology_result_text(text)
        self._populate_hole_fill_candidate_combo(candidates, face_ids)
        self.log(
            f"Hole candidate analysis complete for {self._topology_scope_label(face_ids)}: "
            f"{len(candidates)} candidate(s)."
        )
        self.statusBar().showMessage("Hole candidate analysis complete", 3000)


