"""Density-mode helpers for BoreTool rebuild.

This module is intentionally non-mutating.  It owns only quad-density labels,
normalization, candidate metadata resolution, and measured axial segment counts.
It must not authorize deletion, interpret feature identity, apply generated
geometry to a mesh, validate final topology, or emit RebuildResult.
"""

from __future__ import annotations

from typing import Mapping

import math

import numpy as np


REBUILD_DENSITY_HELPER_EXTRACTION_CHECKPOINT_V176Q = (
    "v176q_rebuild_density_helper_extraction_no_behavior_change"
)

REBUILD_DENSITY_HELPER_NON_MUTATION_CONTRACT_V176Q = (
    "density_helpers_may_normalize_modes_and_count_segments_but_must_not_mutate_or_authorize_rebuild"
)

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


def normalize_quad_density_mode_v176q(value: object | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return QUAD_DENSITY_MODE_LEAN
    return QUAD_DENSITY_MODE_ALIASES.get(text, QUAD_DENSITY_MODE_LEAN)


def resolve_quad_density_mode_v176q(explicit_mode: object | None, candidate_metadata: Mapping[str, object] | None) -> str:
    if explicit_mode is not None and str(explicit_mode).strip():
        return normalize_quad_density_mode_v176q(explicit_mode)
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
            return normalize_quad_density_mode_v176q(meta.get(key))
    return QUAD_DENSITY_MODE_LEAN


def axial_segment_count_v176q(
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
