"""Lightweight Bore rim resolver for live edge selection.

This module is intentionally smaller than Region Select.  It is safe to call
from the live Ctrl/Cmd-click edge-selection path because it only analyzes the
already-selected local edge cloud and does not build RegionData, search feature
faces, run recognition, or mutate topology.

Semantic boundary
-----------------
Input meaning:
    clicked edge + conservative edge cloud = raw rim/navigation evidence

Output meaning:
    compact existing mesh edge IDs that best represent the user-indicated rim

It does not classify a BORE/POCKET/CHAMFER and it does not authorize rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any

import math
import numpy as np

EdgeKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ResolvedBoreRimEdges:
    edge_ids: tuple[int, ...]
    confidence: float
    source: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_bore_rim_edges_from_click_arrays(
    *,
    vertices: np.ndarray,
    faces: np.ndarray | None,
    edge_index_to_vertices: np.ndarray,
    edge_to_faces: Mapping[EdgeKey, Iterable[int]] | None,
    selected_edge_ids: Iterable[int],
    start_edge_index: int,
    min_loop_edges: int = 8,
    max_output_edges: int = 96,
) -> ResolvedBoreRimEdges | None:
    """Resolve a compact rim from the conservative live edge cloud.

    The important coarse-mesh case is a fragmented edge cloud where the raw
    conservative selection contains many unrelated components, but the clicked
    component itself is the local rim arc/loop.  This function prefers that
    clicked component when it has plausible circular geometry.  If needed, it
    expands only to selected fragments that agree with the clicked component's
    fitted plane/radius.  It never returns the whole raw cloud.
    """

    verts = _as_vertices(vertices)
    edges = _as_edges(edge_index_to_vertices)
    if verts.size == 0 or edges.size == 0:
        return None
    if int(start_edge_index) < 0 or int(start_edge_index) >= len(edges):
        return None

    raw_ids = tuple(sorted({int(v) for v in tuple(selected_edge_ids or ()) if 0 <= int(v) < len(edges)}))
    if not raw_ids:
        return None

    key_to_id = {_edge_key(edge): int(i) for i, edge in enumerate(edges)}
    selected_keys = {_edge_key(edges[int(eid)]) for eid in raw_ids if 0 <= int(eid) < len(edges)}
    if not selected_keys:
        return None
    start_key = _edge_key(edges[int(start_edge_index)])

    components = _connected_edge_components(selected_keys)
    if not components:
        return None

    start_mid = _edge_midpoint(verts, start_key)
    all_lengths = [_edge_length(verts, key) for key in selected_keys]
    all_lengths = [v for v in all_lengths if math.isfinite(v) and v > 1.0e-12]
    median_len = float(np.median(np.asarray(all_lengths, dtype=float))) if all_lengths else 1.0
    min_edges = max(3, int(min_loop_edges))
    max_edges = max(int(min_edges), int(max_output_edges))

    component_reports: list[dict[str, Any]] = []
    scored: list[tuple[float, set[EdgeKey], _RingFitLite, dict[str, Any]]] = []
    for idx, comp in enumerate(components):
        comp = {_edge_key(edge) for edge in comp}
        comp_ids = tuple(sorted(key_to_id[key] for key in comp if key in key_to_id))
        contains_start = start_key in comp
        dist = _component_distance_to_point(verts, comp, start_mid)
        ring = _fit_ring_to_edge_vertices(verts, comp, min_points=max(6, min_edges))
        report: dict[str, Any] = {
            "component_index": int(idx),
            "edge_count": int(len(comp)),
            "contains_start_edge": bool(contains_start),
            "distance_to_clicked_edge": float(dist),
            "edge_ids_sample": comp_ids[:12],
        }
        if ring is None:
            report["accepted"] = False
            report["reason"] = "ring_fit_failed"
            component_reports.append(report)
            continue
        report.update(ring.to_diagnostics())
        plausible, reason = _component_ring_is_plausible(ring=ring, edge_count=len(comp), median_edge_length=median_len)
        report["accepted"] = bool(plausible)
        report["reason"] = str(reason)
        component_reports.append(report)
        if not plausible:
            continue

        coverage = min(float(len(comp)) / 48.0, 1.0)
        proximity = 1.0 / (1.0 + float(dist) / max(float(ring.radius), median_len, 1.0e-9))
        score = (
            (4.0 if contains_start else 0.0)
            + 1.20 * float(ring.circularity)
            + 0.80 * coverage
            + 0.70 * proximity
            - 1.40 * min(float(ring.radius_rel_rms), 1.0)
            - 0.70 * min(float(ring.plane_rel_rms), 1.0)
        )
        scored.append((float(score), comp, ring, report))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    _score, seed_component, seed_ring, seed_report = scored[0]

    # Prefer the clicked component if it is plausible even if a remote component
    # has slightly higher circularity.  The click is the user's navigation anchor.
    clicked_scored = [item for item in scored if start_key in item[1]]
    if clicked_scored:
        clicked_scored.sort(key=lambda item: item[0], reverse=True)
        clicked = clicked_scored[0]
        if clicked[0] >= scored[0][0] - 1.25:
            _score, seed_component, seed_ring, seed_report = clicked

    # The seed component itself is the safest visual answer.  On the coarse test
    # this is expected to be the compact primary raw rim component (~28 edges),
    # instead of the full conservative cloud (~274 edges).
    output_keys = set(seed_component)

    # If the clicked component is only a short arc, add nearby selected fragments
    # that agree with the seed ring.  This is bounded and O(raw selected edges).
    if len(output_keys) < min_edges or _angular_coverage(verts, output_keys, seed_ring) < 0.45:
        expanded = _expand_selected_fragments_in_ring_band(
            vertices=verts,
            selected_keys=selected_keys,
            seed_ring=seed_ring,
            start_key=start_key,
            median_edge_length=median_len,
            max_edges=max_edges,
        )
        if len(expanded) >= len(output_keys):
            output_keys = expanded

    if len(output_keys) > max_edges:
        output_keys = _rank_edges_for_ring_output(
            vertices=verts,
            keys=output_keys,
            ring=seed_ring,
            start_key=start_key,
            max_edges=max_edges,
        )

    output_ids = tuple(sorted(int(key_to_id[key]) for key in output_keys if key in key_to_id))
    if len(output_ids) < min_edges:
        return None
    if len(output_ids) >= max(len(raw_ids) * 0.70, max_edges + 1):
        # Safety guard: never turn the broad cloud back into the visual rim.
        return None

    return ResolvedBoreRimEdges(
        edge_ids=output_ids,
        confidence=float(max(0.50, min(0.92, 0.52 + 0.02 * min(len(output_ids), 20) - 0.20 * seed_ring.radius_rel_rms))),
        source="local_clicked_component_circle_navigation",
        diagnostics={
            "semantic_stage": "clicked_edge_plus_conservative_cloud_to_compact_rim_edges",
            "resolver": "bore_local_rim_component_resolver_v1",
            "selected_edge_count_in": int(len(raw_ids)),
            "selected_edge_count_out": int(len(output_ids)),
            "component_count": int(len(components)),
            "clicked_component_used": bool(start_key in seed_component),
            "seed_component_edge_count": int(len(seed_component)),
            "expanded_output_edge_count": int(len(output_keys)),
            "median_selected_edge_length": float(median_len),
            "ring_radius": float(seed_ring.radius),
            "ring_circularity": float(seed_ring.circularity),
            "ring_radius_rel_rms": float(seed_ring.radius_rel_rms),
            "ring_plane_rel_rms": float(seed_ring.plane_rel_rms),
            "ring_angular_coverage": float(_angular_coverage(verts, output_keys, seed_ring)),
            "seed_component_report": dict(seed_report),
            "component_reports": tuple(component_reports[:16]),
            "pcu_metric_followup": "future_optional_chamfer_or_hausdorff_score_can_compare_candidate_ring_samples_to_edge_midpoint_cloud",
        },
    )


@dataclass(frozen=True, slots=True)
class _RingFitLite:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    plane_rms: float
    plane_rel_rms: float
    radius_rms: float
    radius_rel_rms: float
    circularity: float
    point_count: int

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "ring_radius": float(self.radius),
            "ring_plane_rel_rms": float(self.plane_rel_rms),
            "ring_radius_rel_rms": float(self.radius_rel_rms),
            "ring_circularity": float(self.circularity),
            "ring_point_count": int(self.point_count),
        }


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3].astype(float, copy=False)


def _as_edges(edge_index_to_vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(edge_index_to_vertices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    return arr[:, :2].astype(np.int64, copy=False)


def _edge_key(edge: object) -> EdgeKey:
    a, b = tuple(edge)[:2]  # type: ignore[arg-type]
    ia, ib = int(a), int(b)
    return (ia, ib) if ia <= ib else (ib, ia)


def _edge_midpoint(vertices: np.ndarray, key: EdgeKey) -> np.ndarray:
    a, b = key
    return 0.5 * (vertices[int(a), :3] + vertices[int(b), :3])


def _edge_length(vertices: np.ndarray, key: EdgeKey) -> float:
    a, b = key
    if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
        return 0.0
    return float(np.linalg.norm(vertices[int(a), :3] - vertices[int(b), :3]))


def _connected_edge_components(edges: set[EdgeKey]) -> list[set[EdgeKey]]:
    normalized = {_edge_key(edge) for edge in edges}
    vertex_to_edges: dict[int, set[EdgeKey]] = {}
    for edge in normalized:
        a, b = edge
        vertex_to_edges.setdefault(int(a), set()).add(edge)
        vertex_to_edges.setdefault(int(b), set()).add(edge)
    remaining = set(normalized)
    comps: list[set[EdgeKey]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            a, b = stack.pop()
            for v in (a, b):
                for nxt in tuple(vertex_to_edges.get(int(v), ())):
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.add(nxt)
                        stack.append(nxt)
        comps.append(comp)
    comps.sort(key=lambda comp: (-len(comp), tuple(sorted(comp))[:1]))
    return comps


def _component_distance_to_point(vertices: np.ndarray, edges: set[EdgeKey], point: np.ndarray) -> float:
    vals: list[float] = []
    for key in edges:
        vals.append(float(np.linalg.norm(_edge_midpoint(vertices, key) - point)))
    return float(min(vals)) if vals else float("inf")


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if math.isfinite(n) and n > 1.0e-12:
        return arr / n
    fb = np.asarray(fallback, dtype=float).reshape(3)
    fn = float(np.linalg.norm(fb))
    return fb / fn if math.isfinite(fn) and fn > 1.0e-12 else np.array([0.0, 0.0, 1.0], dtype=float)


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, axis))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _unit(np.cross(axis, ref), np.array([0.0, 1.0, 0.0], dtype=float))
    v = _unit(np.cross(axis, u), np.array([1.0, 0.0, 0.0], dtype=float))
    return u, v


def _fit_ring_to_edge_vertices(vertices: np.ndarray, edges: set[EdgeKey], *, min_points: int) -> _RingFitLite | None:
    vids = sorted({int(v) for edge in edges for v in edge if 0 <= int(v) < len(vertices)})
    if len(vids) < int(min_points):
        return None
    pts = vertices[np.asarray(vids, dtype=np.int64), :3].astype(float, copy=False)
    if len(pts) < int(min_points):
        return None
    center0 = pts.mean(axis=0)
    centered = pts - center0
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = _unit(np.asarray(vh[-1], dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
    except Exception:
        return None
    u, v = _orthonormal_basis(axis)
    xy = np.column_stack((centered @ u, centered @ v))
    try:
        mat = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
        rhs = -(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1])
        sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
        cx = -0.5 * float(sol[0])
        cy = -0.5 * float(sol[1])
        radius_sq = max(0.0, cx * cx + cy * cy - float(sol[2]))
        radius = float(np.sqrt(radius_sq))
        center = center0 + cx * u + cy * v
    except Exception:
        rel = centered - np.outer(centered @ axis, axis)
        radii0 = np.linalg.norm(rel, axis=1)
        radius = float(np.median(radii0[radii0 > 1.0e-12])) if np.any(radii0 > 1.0e-12) else 0.0
        center = center0
    if not math.isfinite(radius) or radius <= 1.0e-12:
        return None
    rel2 = pts - center
    axial = rel2 @ axis
    radial_vec = rel2 - np.outer(axial, axis)
    radii = np.linalg.norm(radial_vec, axis=1)
    plane_rms = float(np.sqrt(np.mean(axial * axial))) if len(axial) else 999999.0
    radius_rms = float(np.sqrt(np.mean((radii - radius) ** 2))) if len(radii) else 999999.0
    plane_rel = plane_rms / max(radius, 1.0e-9)
    radius_rel = radius_rms / max(radius, 1.0e-9)
    circularity = float(max(0.0, 1.0 - min(radius_rel * 3.0 + plane_rel * 1.4, 1.0)))
    return _RingFitLite(
        center=np.asarray(center, dtype=float).reshape(3),
        axis=np.asarray(axis, dtype=float).reshape(3),
        radius=float(radius),
        plane_rms=plane_rms,
        plane_rel_rms=float(plane_rel),
        radius_rms=radius_rms,
        radius_rel_rms=float(radius_rel),
        circularity=float(circularity),
        point_count=int(len(pts)),
    )


def _component_ring_is_plausible(*, ring: _RingFitLite, edge_count: int, median_edge_length: float) -> tuple[bool, str]:
    if int(edge_count) < 3:
        return False, "too_few_edges"
    if ring.radius <= max(float(median_edge_length) * 1.25, 1.0e-9):
        return False, "radius_too_small_for_edge_scale"
    if ring.radius_rel_rms > 0.30:
        return False, "radius_scatter_too_high"
    if ring.plane_rel_rms > 0.30:
        return False, "plane_scatter_too_high"
    if ring.circularity < 0.18:
        return False, "circularity_too_low"
    return True, "plausible_clicked_rim_component"


def _angular_coverage(vertices: np.ndarray, keys: set[EdgeKey], ring: _RingFitLite) -> float:
    if not keys:
        return 0.0
    u, v = _orthonormal_basis(ring.axis)
    angles: list[float] = []
    for key in keys:
        mid = _edge_midpoint(vertices, key)
        rel = mid - ring.center
        rel = rel - ring.axis * float(np.dot(rel, ring.axis))
        x, y = float(np.dot(rel, u)), float(np.dot(rel, v))
        if math.isfinite(x) and math.isfinite(y):
            angles.append(float(math.atan2(y, x)))
    if len(angles) < 2:
        return 0.0
    vals = np.sort(np.mod(np.asarray(angles, dtype=float), 2.0 * math.pi))
    gaps = np.diff(np.concatenate([vals, vals[:1] + 2.0 * math.pi]))
    largest_gap = float(np.max(gaps)) if len(gaps) else 2.0 * math.pi
    return float(max(0.0, min(1.0, (2.0 * math.pi - largest_gap) / (2.0 * math.pi))))


def _expand_selected_fragments_in_ring_band(
    *,
    vertices: np.ndarray,
    selected_keys: set[EdgeKey],
    seed_ring: _RingFitLite,
    start_key: EdgeKey,
    median_edge_length: float,
    max_edges: int,
) -> set[EdgeKey]:
    plane_tol = max(seed_ring.plane_rms * 2.75, seed_ring.radius * 0.10, median_edge_length * 2.5, 1.0e-6)
    radial_tol = max(seed_ring.radius_rms * 2.75, seed_ring.radius * 0.075, median_edge_length * 2.0, 1.0e-6)
    local_limit = max(seed_ring.radius * 2.10, median_edge_length * 12.0, 1.0e-6)
    start_mid = _edge_midpoint(vertices, start_key)
    out: set[EdgeKey] = set()
    for key in selected_keys:
        mid = _edge_midpoint(vertices, key)
        if float(np.linalg.norm(mid - start_mid)) > local_limit + seed_ring.radius:
            continue
        rel = mid - seed_ring.center
        axial = abs(float(np.dot(rel, seed_ring.axis)))
        radial_vec = rel - seed_ring.axis * float(np.dot(rel, seed_ring.axis))
        radial = float(np.linalg.norm(radial_vec))
        if axial <= plane_tol and abs(radial - seed_ring.radius) <= radial_tol:
            # Reject obvious spokes when there is enough information.
            a, b = key
            vec = vertices[b, :3] - vertices[a, :3]
            length = float(np.linalg.norm(vec))
            if length > 1.0e-12 and radial > 1.0e-12:
                tangent = vec / length
                radial_unit = radial_vec / radial
                radial_alignment = abs(float(np.dot(tangent, radial_unit)))
                axial_alignment = abs(float(np.dot(tangent, seed_ring.axis)))
                if radial_alignment > 0.94 and axial_alignment < 0.35:
                    continue
            out.add(key)
    if len(out) > int(max_edges):
        out = _rank_edges_for_ring_output(vertices=vertices, keys=out, ring=seed_ring, start_key=start_key, max_edges=int(max_edges))
    out.add(start_key)
    return out


def _rank_edges_for_ring_output(
    *,
    vertices: np.ndarray,
    keys: set[EdgeKey],
    ring: _RingFitLite,
    start_key: EdgeKey,
    max_edges: int,
) -> set[EdgeKey]:
    start_mid = _edge_midpoint(vertices, start_key)
    scored: list[tuple[float, EdgeKey]] = []
    for key in keys:
        mid = _edge_midpoint(vertices, key)
        rel = mid - ring.center
        axial = abs(float(np.dot(rel, ring.axis)))
        radial_vec = rel - ring.axis * float(np.dot(rel, ring.axis))
        radial = float(np.linalg.norm(radial_vec))
        radial_err = abs(radial - ring.radius) / max(ring.radius, 1.0e-9)
        plane_err = axial / max(ring.radius, 1.0e-9)
        prox = 1.0 / (1.0 + float(np.linalg.norm(mid - start_mid)) / max(ring.radius, 1.0e-9))
        score = 1.0 * prox - 1.15 * min(radial_err, 1.0) - 0.70 * min(plane_err, 1.0) + (1.0 if key == start_key else 0.0)
        scored.append((float(score), key))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept = {key for _score, key in scored[: max(1, int(max_edges))]}
    kept.add(start_key)
    return kept


__all__ = ["ResolvedBoreRimEdges", "resolve_bore_rim_edges_from_click_arrays"]
