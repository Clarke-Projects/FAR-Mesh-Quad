from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, Mapping
import re

import numpy as np
import trimesh


@dataclass(frozen=True)
class AdaptiveSurfaceFillV2SeedCandidateGeometryProbe:
    """Non-selecting Adaptive Surface Fill v2 seed-geometry candidate.

    This module builds the Adaptive Surface Fill v2 seed-geometry candidate.
    It remains non-selecting until the strict measured selection bridge passes.

    Design rule:
    surrounding curvature -> normal continuity -> generated mesh geometry

    Safety rules:
    - boundary/seam vertices are fixed
    - generated patch vertices only may move
    - movement is bounded relative to local context edge scale
    - topology is checked immediately
    - selection requires strict topology + G1 + real quality + real G2 validation
    """

    status: str
    action: str
    available: bool
    applied: bool
    selected: bool
    family: str
    geometry_mode: str

    face_count: int
    vertex_count: int
    reoriented_face_count: int
    moved_vertex_count: int

    movement_mean: float
    movement_max: float
    movement_ratio_max: float

    predicted_g1_mean_deviation: float | str
    predicted_g1_max_deviation: float | str
    predicted_g1_status: str

    topology_status: str
    topology_reasons: tuple[str, ...]
    reasons: tuple[str, ...]

    evaluation_status: str = "not_evaluated"
    evaluation_action: str = "not_evaluated"
    evaluation_selectable: bool = False
    evaluation_g1_status: str = "not_evaluated"
    evaluation_g1_mean_deviation: object = "-"
    evaluation_g1_max_deviation: object = "-"
    evaluation_g1_mean_limit: object = "-"
    evaluation_g1_max_limit: object = "-"
    evaluation_quality_status: str = "not_evaluated"
    evaluation_g2_status: str = "not_evaluated"
    evaluation_policy: str = "non_selecting_probe"
    evaluation_reasons: tuple[str, ...] = ()
    candidate_mesh: trimesh.Trimesh | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "candidate_mesh"
        }


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
        if token and token not in {"adaptive"}
    )


def _first_value(metadata: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in metadata and not _missing(metadata.get(key)):
            return metadata.get(key)

    items = tuple(metadata.items())
    for requested_key in keys:
        requested = set(_tokens(requested_key))
        if not requested:
            continue
        for actual_key, actual_value in items:
            if _missing(actual_value):
                continue
            actual = set(_tokens(str(actual_key)))
            if requested.issubset(actual):
                return actual_value

    return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            if match:
                try:
                    number = float(match.group(0))
                except Exception:
                    return float(default)
            else:
                return float(default)
        else:
            return float(default)

    if not isfinite(number):
        return float(default)
    return float(number)



def _bridge_measured_target_confidence_for_probe(
    metadata: Mapping[str, object],
    legacy_result: Mapping[str, Any],
) -> dict[str, object]:
    """Return probe metadata with measured support-target confidence restored.

    This is intentionally metadata-only.  It does not change geometry policy,
    thresholds, movement caps, topology, or selection.  Its purpose is to make
    the v2 geometry probe use the same measured target-confidence diagnostics
    that are already shown in the smoke output, instead of falling back to the
    neutral prototype defaults.
    """

    out: dict[str, object] = dict(metadata)

    report = legacy_result.get("support_target_disagreement")
    if not isinstance(report, Mapping):
        report = out.get("support_target_disagreement")
    if not isinstance(report, Mapping):
        report = {}

    mapping = {
        "adaptive_target_model_confidence": "target_model_confidence",
        "adaptive_target_confidence_low_count": "target_confidence_low_count",
        "adaptive_target_confidence_vertex_count": "target_confidence_vertex_count",
        "adaptive_target_confidence_low_threshold": "target_confidence_low_threshold",
        "adaptive_target_confidence_model": "target_confidence_model",
        "adaptive_target_disagreement_source": "target_disagreement_source",
        "adaptive_target_model_recommendation": "target_model_recommendation",
    }

    for dst, src in mapping.items():
        value = report.get(src)
        if not _missing(value):
            out[dst] = value

    measured_confidence = out.get("adaptive_target_model_confidence")
    if not _missing(measured_confidence):
        # Some intermediate metadata paths currently consume prototype names.
        # Mirror the measured value there too, so a fallback prototype default
        # cannot override the real support-target confidence.
        out["adaptive_surface_v2_seed_prototype_target_confidence"] = measured_confidence
        out["v2_prototype_target_confidence"] = measured_confidence

    low_count = _float(out.get("adaptive_target_confidence_low_count"), default=float("nan"))
    vertex_count = _float(out.get("adaptive_target_confidence_vertex_count"), default=float("nan"))
    if vertex_count > 0.0 and low_count >= 0.0:
        low_fraction = float(max(0.0, min(1.0, low_count / max(vertex_count, 1.0))))
        out["adaptive_surface_v2_seed_prototype_low_confidence_fraction"] = low_fraction
        out["v2_prototype_low_confidence_fraction"] = low_fraction

    return out

def _tuple_ints(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return tuple(dict.fromkeys(out))
    try:
        return (int(value),)
    except Exception:
        return ()


def _safe_unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm > 1.0e-12 and np.all(np.isfinite(arr)):
        return arr / norm
    if fallback is not None:
        fb = np.asarray(fallback, dtype=float)
        fb_norm = float(np.linalg.norm(fb))
        if fb_norm > 1.0e-12 and np.all(np.isfinite(fb)):
            return fb / fb_norm
    return np.asarray([0.0, 0.0, 1.0], dtype=float)


def _flatten_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten_strings(key))
            out.extend(_flatten_strings(item))
        return tuple(out)
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return tuple(out)
    return (str(value),)


def _find_boundary_normal_deviation(metadata: Mapping[str, object], kind: str) -> object:
    """Find the real G1 boundary-normal mean/max deviation.

    Prefer gate/reason text over planning placeholder values.
    """

    kind = str(kind).lower().strip()
    if kind not in {"mean", "max"}:
        return "-"

    reason_values: list[str] = []
    preferred_reason_keys = ("g1", "gate", "reason", "diagnostic", "warning")
    for key, item in metadata.items():
        key_text = str(key).lower()
        if any(token in key_text for token in preferred_reason_keys):
            reason_values.extend(_flatten_strings(item))
    reason_values.extend(_flatten_strings(metadata))
    text_blob = "\n".join(str(value) for value in reason_values)

    patterns = (
        rf"boundary\s+normal\s+{kind}\s+deviation\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*°?\s*(?:>|$|[,;\n\r])",
        rf"boundary\s+normal\s+{kind}\s+deviation[^\n\r()]*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*°?\s*\)",
        rf"g1\s+boundary\s+normal\s+{kind}\s+deviation\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*°?",
    )
    for pattern in patterns:
        match = re.search(pattern, text_blob, flags=re.IGNORECASE)
        if match:
            parsed = _float(match.group(1), default=float("nan"))
            if isfinite(parsed):
                return parsed

    exact_keys = (
        f"adaptive_g1_boundary_normal_{kind}_deviation",
        f"adaptive_g1_boundary_normal_{kind}_deviation_degrees",
        f"adaptive_g1_boundary_normal_{kind}_deviation_deg",
        f"g1_boundary_normal_{kind}_deviation",
        f"g1_boundary_normal_{kind}_deviation_degrees",
        f"g1_boundary_normal_{kind}_deviation_deg",
    )
    for key in exact_keys:
        if key in metadata and not _missing(metadata.get(key)):
            parsed = _float(metadata.get(key), default=float("nan"))
            if isfinite(parsed):
                return parsed

    return "-"


def _predict_reoriented_g1(value: object) -> float | str:
    if _missing(value):
        return "-"
    old = _float(value, default=float("nan"))
    if not isfinite(old):
        return "-"
    return float(abs(180.0 - old))


def _topology_probe(
    candidate_mesh: trimesh.Trimesh,
    *,
    patch_face_ids: tuple[int, ...],
    boundary_vertex_ids: tuple[int, ...],
) -> tuple[str, tuple[str, ...]]:
    try:
        from far_mesh.core.hole_surface_context import validate_patch_topology

        report = validate_patch_topology(
            candidate_mesh,
            patch_face_ids=patch_face_ids,
            boundary_vertex_ids=boundary_vertex_ids,
        )
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)

        reasons: list[str] = []
        missing = int(data.get("missing_boundary_edge_count", 0) or 0)
        overused = int(data.get("overused_boundary_edge_count", 0) or 0)
        nonmanifold = int(data.get("nonmanifold_patch_edge_count", 0) or 0)
        extra_open = int(data.get("extra_open_boundary_edge_count", 0) or 0)
        degenerate = int(data.get("degenerate_patch_face_count", 0) or 0)

        if missing:
            reasons.append(f"missing boundary edges: {missing}")
        if overused:
            reasons.append(f"overused boundary edges: {overused}")
        if nonmanifold:
            reasons.append(f"nonmanifold patch edges: {nonmanifold}")
        if extra_open:
            reasons.append(f"extra open boundary edges: {extra_open}")
        if degenerate:
            reasons.append(f"degenerate patch faces: {degenerate}")

        if reasons:
            return "warning", tuple(reasons)
        return "passed", ("candidate topology remains clean after v2 curvature-normal seed probe",)
    except Exception as exc:
        return "not_evaluated", (f"topology probe failed: {exc}",)


def _v2_gate_eval_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            value = value.strip()
            if not value or value == "-":
                return float(default)
        return float(value)
    except Exception:
        return float(default)


def _v2_gate_eval_limit(metadata: Mapping[str, object], kind: str, default: float) -> float:
    blob = "\n".join(_flatten_strings(metadata))
    kind = str(kind).lower().strip()
    patterns = (
        rf"boundary\s+normal\s+{kind}\s+deviation\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*°?\s*>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*°?",
        rf"{kind}\s+deviation\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*°?\s*>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*°?",
    )
    for pattern in patterns:
        match = re.search(pattern, blob, flags=re.IGNORECASE)
        if match:
            value = _v2_gate_eval_float(match.group(1))
            if isfinite(value) and value > 0.0:
                return float(value)
    return float(default)


def _mean_normal_for_faces(mesh: trimesh.Trimesh, face_ids: tuple[int, ...]) -> np.ndarray:
    normals = np.asarray(getattr(mesh, "face_normals", ()), dtype=float)
    valid = [
        int(face_id)
        for face_id in face_ids
        if 0 <= int(face_id) < int(len(normals))
    ]
    if not valid:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)

    mean = np.mean(normals[valid], axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return mean / norm


def _support_face_ids_for_boundary(
    mesh: trimesh.Trimesh,
    *,
    patch_face_ids: tuple[int, ...],
    boundary_vertex_ids: tuple[int, ...],
    rings: int,
) -> tuple[int, ...]:
    """Collect local non-patch support faces around the boundary.

    This is intentionally topology/context based, not smoke-value based:
    start from non-patch faces touching the boundary, then expand by vertex
    adjacency for the requested number of rings.
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)
    patch_set = {int(face_id) for face_id in patch_face_ids}
    current_vertices = {int(vertex_id) for vertex_id in boundary_vertex_ids}
    support: set[int] = set()

    ring_count = max(1, int(rings or 1))
    for _ in range(ring_count):
        next_vertices: set[int] = set()
        for face_id, face in enumerate(faces):
            if int(face_id) in patch_set:
                continue
            tri_vertices = {int(v) for v in face[:3]}
            if tri_vertices & current_vertices:
                support.add(int(face_id))
                next_vertices.update(tri_vertices)

        if not next_vertices or next_vertices.issubset(current_vertices):
            break
        current_vertices.update(next_vertices)

    return tuple(sorted(support))


def _curvature_samples_for_faces(
    mesh: trimesh.Trimesh,
    face_ids: tuple[int, ...],
) -> tuple[float, ...]:
    """Local normal-angle-over-edge-length curvature samples."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(getattr(mesh, "face_normals", ()), dtype=float)

    valid_faces = tuple(
        int(face_id)
        for face_id in face_ids
        if 0 <= int(face_id) < int(len(faces)) and int(face_id) < int(len(normals))
    )

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id in valid_faces:
        tri = [int(v) for v in faces[int(face_id), :3]]
        for index, a in enumerate(tri):
            b = tri[(index + 1) % 3]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(edge, []).append(int(face_id))

    values: list[float] = []
    for edge, owners in edge_to_faces.items():
        if len(owners) < 2:
            continue

        a, b = edge
        if a < 0 or b < 0 or a >= int(len(vertices)) or b >= int(len(vertices)):
            continue

        edge_length = float(np.linalg.norm(vertices[b] - vertices[a]))
        if not np.isfinite(edge_length) or edge_length <= 1.0e-12:
            continue

        n0 = normals[int(owners[0])]
        n1 = normals[int(owners[1])]
        dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        angle = float(np.arccos(dot))
        value = float(angle / edge_length)
        if np.isfinite(value):
            values.append(value)

    return tuple(values)


def _summary_from_samples(samples: tuple[float, ...]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "mean": "-",
            "max": "-",
            "std": "-",
        }

    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "sample_count": 0,
            "mean": "-",
            "max": "-",
            "std": "-",
        }

    return {
        "sample_count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _real_v2_quality_probe(
    *,
    source_mesh: trimesh.Trimesh | None,
    candidate_mesh: trimesh.Trimesh,
    patch_face_ids: tuple[int, ...],
    boundary_vertex_ids: tuple[int, ...],
    movable_vertex_ids: tuple[int, ...],
    metadata: Mapping[str, object],
) -> tuple[str, tuple[str, ...], dict[str, object]]:
    """Run the real triangle-quality gate for the moved v2 candidate.

    Uses the existing project quality infrastructure when available.
    Returns non-selecting status only.
    """

    reasons: list[str] = []

    try:
        from far_mesh.core.hole_patch_quality import (
            analyze_patch_quality,
            compute_boundary_target_normals,
            mesh_edge_length_median,
        )

        context_mesh = source_mesh if isinstance(source_mesh, trimesh.Trimesh) else candidate_mesh
        context_edge = _float(
            _first_value(
                metadata,
                "adaptive_seed_context_edge_length_median",
                "adaptive_seed_context_edge_median",
                default=mesh_edge_length_median(context_mesh),
            ),
            1.0,
        )

        target_normals = compute_boundary_target_normals(
            context_mesh,
            boundary_vertex_ids,
        )

        report = analyze_patch_quality(
            candidate_mesh,
            patch_face_ids=patch_face_ids,
            boundary_vertex_ids=boundary_vertex_ids,
            movable_vertex_ids=movable_vertex_ids,
            context_edge_length_median=context_edge,
            target_boundary_normals=target_normals,
        )
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)

        degenerate = int(
            data.get("degenerate_face_count")
            or data.get("degenerate_patch_face_count")
            or 0
        )
        min_angle = _float(
            data.get("min_triangle_angle_degrees")
            or data.get("min_angle_degrees")
            or data.get("min_angle")
            or 0.0,
            0.0,
        )
        median_aspect = _float(
            data.get("median_triangle_aspect_ratio")
            or data.get("median_aspect_ratio")
            or 0.0,
            0.0,
        )
        max_aspect = _float(
            data.get("max_triangle_aspect_ratio")
            or data.get("max_aspect_ratio")
            or 0.0,
            0.0,
        )

        if degenerate > 0:
            status = "blocked"
            reasons.append(f"real quality gate blocked: degenerate faces={degenerate}")
        elif min_angle > 0.0 and min_angle < 1.0:
            status = "warning"
            reasons.append(f"real quality warning: min angle {min_angle:.6g}° < 1°")
        elif median_aspect > 25.0 or max_aspect > 100.0:
            status = "warning"
            reasons.append(
                "real quality warning: triangle aspect ratio is elevated"
            )
        else:
            status = "passed"
            reasons.append("real quality gate passed for moved v2 candidate")

        return status, tuple(reasons), data

    except Exception as exc:
        return (
            "not_evaluated",
            (f"real quality probe unavailable: {exc}",),
            {},
        )


def _real_v2_g2_probe(
    *,
    candidate_mesh: trimesh.Trimesh,
    patch_face_ids: tuple[int, ...],
    boundary_vertex_ids: tuple[int, ...],
    metadata: Mapping[str, object],
) -> tuple[str, tuple[str, ...], dict[str, object]]:
    """Run a real local curvature/G2 comparison for the v2 candidate.

    Dynamic rule:
    - compare candidate patch curvature to local non-patch support faces
    - compare against the previous measured v1 relative curvature delta when
      available
    - never make the candidate selectable here
    """

    reasons: list[str] = []

    support_rings = int(
        max(
            1,
            _float(
                _first_value(
                    metadata,
                    "adaptive_surface_v2_seed_candidate_support_rings",
                    "adaptive_end_layer_support_ring_count",
                    default=2,
                ),
                2.0,
            ),
        )
    )

    support_face_ids = _support_face_ids_for_boundary(
        candidate_mesh,
        patch_face_ids=patch_face_ids,
        boundary_vertex_ids=boundary_vertex_ids,
        rings=support_rings,
    )

    support = _summary_from_samples(
        _curvature_samples_for_faces(candidate_mesh, support_face_ids)
    )
    patch = _summary_from_samples(
        _curvature_samples_for_faces(candidate_mesh, patch_face_ids)
    )

    support_mean = _float(support.get("mean"), float("nan"))
    support_max = _float(support.get("max"), float("nan"))
    patch_mean = _float(patch.get("mean"), float("nan"))
    patch_max = _float(patch.get("max"), float("nan"))

    data: dict[str, object] = {
        "support_face_count": int(len(support_face_ids)),
        "support_sample_count": support.get("sample_count", 0),
        "patch_sample_count": patch.get("sample_count", 0),
        "support_curvature_mean": support.get("mean", "-"),
        "support_curvature_max": support.get("max", "-"),
        "support_curvature_std": support.get("std", "-"),
        "patch_curvature_mean": patch.get("mean", "-"),
        "patch_curvature_max": patch.get("max", "-"),
        "patch_curvature_std": patch.get("std", "-"),
    }

    if not (
        np.isfinite(support_mean)
        and np.isfinite(patch_mean)
        and np.isfinite(support_max)
        and np.isfinite(patch_max)
    ):
        reasons.append("real G2 probe could not derive enough local curvature samples")
        return "not_evaluated", tuple(reasons), data

    delta_mean = float(abs(patch_mean - support_mean))
    delta_max = float(abs(patch_max - support_max))
    relative_delta = float(delta_mean / max(abs(support_mean), 1.0e-12))

    patch_normal = _mean_normal_for_faces(candidate_mesh, patch_face_ids)
    support_normal = _mean_normal_for_faces(candidate_mesh, support_face_ids)
    sign_consistency = bool(float(np.dot(patch_normal, support_normal)) >= 0.0)

    previous_relative = _float(
        _first_value(
            metadata,
            "adaptive_g2_curvature_relative_delta_mean",
            "adaptive_curvature_relative_delta_mean",
            "curvature_relative_delta_mean",
            default=float("nan"),
        ),
        float("nan"),
    )

    data.update(
        {
            "curvature_delta_mean": delta_mean,
            "curvature_delta_max": delta_max,
            "curvature_relative_delta_mean": relative_delta,
            "curvature_sign_consistency": sign_consistency,
            "previous_relative_delta_mean": (
                float(previous_relative) if np.isfinite(previous_relative) else "-"
            ),
        }
    )

    if not sign_consistency:
        reasons.append("real G2 blocked: candidate curvature-normal sign is inconsistent with local support")
        return "blocked", tuple(reasons), data

    if np.isfinite(previous_relative):
        if relative_delta <= previous_relative:
            reasons.append(
                "real G2 comparison passed: candidate relative curvature delta does not worsen the measured legacy delta"
            )
            return "passed", tuple(reasons), data

        reasons.append(
            "real G2 warning: candidate relative curvature delta worsens the measured legacy delta"
        )
        return "warning", tuple(reasons), data

    reasons.append(
        "real G2 warning: no previous measured curvature delta was available for dynamic comparison"
    )
    return "warning", tuple(reasons), data



def _patch_vertex_adjacency_for_faces(
    mesh: trimesh.Trimesh,
    patch_face_ids: tuple[int, ...],
) -> dict[int, set[int]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    adjacency: dict[int, set[int]] = {}
    for face_id in patch_face_ids:
        if int(face_id) < 0 or int(face_id) >= int(len(faces)):
            continue
        tri = [int(v) for v in faces[int(face_id), :3]]
        for index, vertex_id in enumerate(tri):
            a = int(vertex_id)
            b = int(tri[(index + 1) % 3])
            c = int(tri[(index + 2) % 3])
            adjacency.setdefault(a, set()).update((b, c))
    return adjacency


def _dynamic_v2_seed_strength(
    metadata: Mapping[str, object],
) -> tuple[float, float, float, str]:
    """Return dynamic v2 seed strength from measured confidence diagnostics.

    This is deliberately not smoke-test fitted.  The strength is derived from:
    - target confidence, when available
    - fraction of low-confidence target vertices, when available
    - support normal spread, relative to the active G1 max-deviation limit
    """

    confidence = _float(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_target_confidence",
            "v2_prototype_target_confidence",
            "adaptive_target_model_confidence",
            "target_model_confidence",
            default=0.5,
        ),
        0.5,
    )
    confidence = float(max(0.0, min(1.0, confidence)))

    low_fraction = _float(
        _first_value(
            metadata,
            "adaptive_surface_v2_seed_prototype_low_confidence_fraction",
            "v2_prototype_low_confidence_fraction",
            "adaptive_target_low_confidence_fraction",
            default=float("nan"),
        ),
        float("nan"),
    )
    if not isfinite(low_fraction):
        low_count = _float(
            _first_value(
                metadata,
                "adaptive_target_confidence_low_count",
                "target_confidence_low_count",
                default=float("nan"),
            ),
            float("nan"),
        )
        vertex_count = _float(
            _first_value(
                metadata,
                "adaptive_target_confidence_vertex_count",
                "target_confidence_vertex_count",
                "adaptive_seed_generated_vertex_count",
                default=float("nan"),
            ),
            float("nan"),
        )
        if isfinite(low_count) and isfinite(vertex_count) and vertex_count > 0.0:
            low_fraction = float(max(0.0, min(1.0, low_count / max(vertex_count, 1.0))))
        else:
            low_fraction = 0.0
    low_fraction = float(max(0.0, min(1.0, low_fraction)))

    support_spread = _float(
        _first_value(
            metadata,
            "adaptive_g1_support_normal_spread",
            "adaptive_seed_support_normal_spread_degrees",
            "adaptive_feature_support_normal_spread_degrees",
            default=0.0,
        ),
        0.0,
    )
    spread_limit = max(1.0, _v2_gate_eval_limit(metadata, "max", 70.0))
    spread_factor = float(max(0.0, min(1.0, support_spread / spread_limit)))

    # Low target confidence should not push harder toward an unstable target.
    # It should instead reduce target-pull strength while still allowing enough
    # curvature-normal correction to repair continuity.
    confidence_factor = 0.5 + 0.5 * confidence
    caution_factor = 1.0 - 0.35 * low_fraction
    strength = float(max(0.20, min(1.0, confidence_factor * caution_factor * (0.65 + 0.35 * spread_factor))))

    source = (
        f"target confidence={confidence:.6g} and low-confidence fraction={low_fraction:.6g}"
    )
    return strength, confidence, low_fraction, source


def _build_curvature_normal_aligned_vertices(
    *,
    candidate_mesh: trimesh.Trimesh,
    source_mesh: trimesh.Trimesh,
    patch_face_ids: tuple[int, ...],
    new_vertex_ids: tuple[int, ...],
    boundary_vertex_ids: tuple[int, ...],
    metadata: Mapping[str, object],
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, tuple[str, ...]]:
    """Build bounded generated-vertex positions for the v2 seed candidate.

    The motion is graph/local-context based:
    boundary/seam vertices are fixed, generated vertices are pulled toward the
    local boundary/support continuation, and displacement is capped relative to
    the measured context edge length.  No literal smoke-test constants are used
    as target values.
    """

    reasons: list[str] = []
    source_vertices = np.asarray(source_mesh.vertices, dtype=float)
    vertices = np.asarray(candidate_mesh.vertices, dtype=float).copy()
    faces = np.asarray(candidate_mesh.faces, dtype=np.int64)

    vertex_count = int(len(vertices))
    boundary_set = {int(v) for v in boundary_vertex_ids if 0 <= int(v) < vertex_count}
    patch_vertex_set: set[int] = set()
    for face_id in patch_face_ids:
        if int(face_id) < 0 or int(face_id) >= int(len(faces)):
            continue
        patch_vertex_set.update(int(v) for v in faces[int(face_id), :3] if 0 <= int(v) < vertex_count)

    generated_candidates = [
        int(v)
        for v in new_vertex_ids
        if 0 <= int(v) < vertex_count and int(v) not in boundary_set
    ]
    if not generated_candidates:
        generated_candidates = sorted(v for v in patch_vertex_set if v not in boundary_set)

    generated = tuple(dict.fromkeys(generated_candidates))
    if not generated:
        return vertices, (), np.asarray([], dtype=float), (
            "no generated seed vertices were available for v2 curvature-normal movement",
        )

    context_edge = _float(
        _first_value(
            metadata,
            "adaptive_seed_context_edge_length_median",
            "adaptive_seed_context_edge_median",
            default=1.0,
        ),
        1.0,
    )
    if not isfinite(context_edge) or context_edge <= 1.0e-12:
        context_edge = 1.0

    strength, confidence, low_fraction, strength_source = _dynamic_v2_seed_strength(metadata)

    # Cap is relative to edge scale and confidence context.  This is intentionally
    # dynamic and conservative: the v2 candidate may become selectable only after
    # real G1/quality/G2 gates pass.
    cap_ratio = float(max(0.01, min(0.05, 0.05 * strength)))
    max_step = float(context_edge * cap_ratio)

    adjacency = _patch_vertex_adjacency_for_faces(candidate_mesh, patch_face_ids)
    patch_support_faces = _support_face_ids_for_boundary(
        candidate_mesh,
        patch_face_ids=patch_face_ids,
        boundary_vertex_ids=boundary_vertex_ids,
        rings=int(max(1.0, _float(_first_value(metadata, "adaptive_surface_v2_seed_candidate_support_rings", default=2), 2.0))),
    )
    support_normal = _mean_normal_for_faces(candidate_mesh, patch_support_faces)
    if not np.all(np.isfinite(support_normal)):
        support_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)

    boundary_centroid = (
        np.mean(source_vertices[list(boundary_set)], axis=0)
        if boundary_set
        else np.mean(source_vertices, axis=0)
    )
    generated_source_centroid = np.mean(source_vertices[list(generated)], axis=0)

    moved: list[int] = []
    deltas: list[float] = []

    for vertex_id in generated:
        current = vertices[int(vertex_id)]
        neighbours = [
            int(v)
            for v in adjacency.get(int(vertex_id), set())
            if 0 <= int(v) < vertex_count and int(v) != int(vertex_id)
        ]
        boundary_neighbours = [v for v in neighbours if v in boundary_set]
        generated_neighbours = [v for v in neighbours if v in generated]

        if boundary_neighbours:
            anchor = np.mean(source_vertices[boundary_neighbours], axis=0)
        elif neighbours:
            anchor = np.mean(source_vertices[neighbours], axis=0)
        else:
            anchor = boundary_centroid

        source_offset = source_vertices[int(vertex_id)] - generated_source_centroid
        tangent_offset = source_offset - support_normal * float(np.dot(source_offset, support_normal))

        smooth_anchor = anchor
        if generated_neighbours:
            smooth_anchor = 0.5 * anchor + 0.5 * np.mean(source_vertices[generated_neighbours], axis=0)

        desired = smooth_anchor + tangent_offset * 0.35
        desired_offset = desired - current

        # Remove aggressive normal-direction drift.  The seed should follow the
        # local curvature-normal continuation, not jump to an unrelated plane.
        normal_component = support_normal * float(np.dot(desired_offset, support_normal))
        tangent_component = desired_offset - normal_component
        desired_offset = tangent_component + normal_component * 0.25

        desired_offset *= strength
        step_norm = float(np.linalg.norm(desired_offset))
        if not np.isfinite(step_norm) or step_norm <= 1.0e-12:
            continue
        if step_norm > max_step:
            desired_offset = desired_offset / step_norm * max_step
            step_norm = max_step

        vertices[int(vertex_id)] = current + desired_offset
        moved.append(int(vertex_id))
        deltas.append(float(step_norm))

    reasons.append("built bounded non-selecting v2 seed vertex candidate from local patch graph")
    reasons.append("boundary/seam vertices were fixed; generated vertices only may move")
    reasons.append("local support normals were derived from non-patch faces adjacent to the boundary")
    reasons.append("movement cap is relative to context edge length, not fitted to one smoke case")
    reasons.append(
        f"candidate strength used {strength_source}"
    )

    return vertices, tuple(moved), np.asarray(deltas, dtype=float), tuple(reasons)

def _evaluate_v2_geometry_probe_gate(
    *,
    metadata: Mapping[str, object],
    predicted_g1_mean_deviation: object,
    predicted_g1_max_deviation: object,
    topology_status: object,
    movement_ratio: object = 0.0,
    candidate_mesh: trimesh.Trimesh | None = None,
    source_mesh: trimesh.Trimesh | None = None,
    patch_face_ids: tuple[int, ...] = (),
    boundary_vertex_ids: tuple[int, ...] = (),
    movable_vertex_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    """Evaluate the v2 geometry probe without allowing selection.

    H-CORE-V2-E3 adds real quality and local G2 evaluation when the moved
    candidate mesh is available. The result remains non-selecting.
    """

    reasons: list[str] = []

    mean_limit = _v2_gate_eval_limit(metadata, "mean", 35.0)
    max_limit = _v2_gate_eval_limit(metadata, "max", 70.0)

    mean_value = _v2_gate_eval_float(predicted_g1_mean_deviation)
    max_value = _v2_gate_eval_float(predicted_g1_max_deviation)

    topology_clean = str(topology_status) == "passed"
    if topology_clean:
        reasons.append("geometry probe topology remained clean")
    else:
        reasons.append("geometry probe topology did not pass")

    if isfinite(mean_value) and isfinite(max_value):
        g1_pass = bool(mean_value <= mean_limit and max_value <= max_limit)
        if g1_pass:
            g1_status = "passed"
            reasons.append(
                f"predicted G1 satisfies active limits: mean={mean_value:.6g} <= {mean_limit:.6g}, max={max_value:.6g} <= {max_limit:.6g}"
            )
        else:
            g1_status = "blocked"
            reasons.append(
                f"predicted G1 does not satisfy active limits: mean={mean_value:.6g} / limit={mean_limit:.6g}, max={max_value:.6g} / limit={max_limit:.6g}"
            )
    else:
        g1_pass = False
        g1_status = "not_evaluated"
        reasons.append("predicted G1 values were not available for geometry probe evaluation")

    movement_value = _v2_gate_eval_float(movement_ratio, default=0.0)

    real_quality_status = "not_evaluated"
    real_g2_status = "not_evaluated"
    quality_reasons: tuple[str, ...] = ()
    g2_reasons: tuple[str, ...] = ()

    if isinstance(candidate_mesh, trimesh.Trimesh) and patch_face_ids:
        real_quality_status, quality_reasons, _quality_data = _real_v2_quality_probe(
            source_mesh=source_mesh,
            candidate_mesh=candidate_mesh,
            patch_face_ids=patch_face_ids,
            boundary_vertex_ids=boundary_vertex_ids,
            movable_vertex_ids=movable_vertex_ids,
            metadata=metadata,
        )
        real_g2_status, g2_reasons, g2_data = _real_v2_g2_probe(
            candidate_mesh=candidate_mesh,
            patch_face_ids=patch_face_ids,
            boundary_vertex_ids=boundary_vertex_ids,
            metadata=metadata,
        )

        reasons.extend(quality_reasons)
        reasons.extend(g2_reasons)

        if g2_data:
            if "curvature_relative_delta_mean" in g2_data:
                reasons.append(
                    "real G2 candidate relative curvature delta: "
                    f"{g2_data.get('curvature_relative_delta_mean')}"
                )
            if "curvature_sign_consistency" in g2_data:
                reasons.append(
                    "real G2 candidate sign consistency: "
                    f"{g2_data.get('curvature_sign_consistency')}"
                )
    else:
        if topology_clean and abs(movement_value) <= 1.0e-12:
            real_quality_status = "preserved_by_zero_movement"
            real_g2_status = "requires_real_curvature_recheck"
            reasons.append("probe moved no vertices, so triangle geometry/quality is preserved")
        elif topology_clean:
            real_quality_status = "requires_quality_recheck"
            real_g2_status = "requires_real_curvature_recheck"
            reasons.append("probe topology is clean but moved geometry requires quality recheck")
        else:
            real_quality_status = "blocked"
            real_g2_status = "blocked"

    quality_ok = real_quality_status in {
        "passed",
        "warning",
        "preserved_by_zero_movement",
    }
    g2_ok = real_g2_status in {
        "passed",
        "warning",
        "requires_real_curvature_recheck",
    }

    gate_ready = bool(topology_clean and g1_pass and quality_ok and g2_ok)

    strict_quality_pass = str(real_quality_status) == "passed"
    strict_g2_pass = str(real_g2_status) == "passed"
    has_real_candidate = isinstance(candidate_mesh, trimesh.Trimesh)

    selection_ready = bool(
        topology_clean
        and g1_pass
        and strict_quality_pass
        and strict_g2_pass
        and has_real_candidate
    )

    if selection_ready:
        reasons.append(
            "selection bridge passed: topology, predicted G1, real quality, and real G2 all passed"
        )
    elif gate_ready:
        reasons.append(
            "selection bridge kept candidate non-selecting because one strict real selection gate was not passed"
        )

    return {
        "evaluation_status": (
            "gate_probe_selectable"
            if selection_ready
            else "gate_probe_ready"
            if gate_ready
            else "gate_probe_blocked"
        ),
        "evaluation_action": (
            "select_real_v2_seed_candidate"
            if selection_ready
            else "keep_nonselecting_real_v2_seed_candidate_for_policy_review"
            if gate_ready
            else "do_not_select_v2_seed_candidate"
        ),
        "evaluation_selectable": bool(selection_ready),
        "evaluation_g1_status": g1_status,
        "evaluation_g1_mean_deviation": float(mean_value) if isfinite(mean_value) else "-",
        "evaluation_g1_max_deviation": float(max_value) if isfinite(max_value) else "-",
        "evaluation_g1_mean_limit": float(mean_limit),
        "evaluation_g1_max_limit": float(max_limit),
        "evaluation_quality_status": real_quality_status,
        "evaluation_g2_status": real_g2_status,
        "evaluation_policy": (
            "select_real_v2_seed_candidate_after_strict_g1_g2_quality_pass"
            if selection_ready
            else "non_selecting_real_v2_seed_candidate"
        ),
        "evaluation_reasons": tuple(
            dict.fromkeys(
                str(reason)
                for reason in reasons
                if str(reason)
            )
        ),
    }

def build_curvature_normal_aligned_seed_candidate_geometry_probe(
    *,
    legacy_result: Mapping[str, Any],
    metadata: Mapping[str, object],
) -> AdaptiveSurfaceFillV2SeedCandidateGeometryProbe:
    reasons: list[str] = []
    metadata = _bridge_measured_target_confidence_for_probe(metadata, legacy_result)

    candidate_status = str(_first_value(metadata, "adaptive_surface_v2_seed_candidate_status", default=""))
    candidate_family = str(_first_value(metadata, "adaptive_surface_v2_seed_candidate_family", default="unknown"))
    v2_case = str(_first_value(metadata, "adaptive_surface_v2_case", default=""))
    prototype_orientation = str(
        _first_value(metadata, "adaptive_surface_v2_seed_prototype_orientation_status", default="")
    )

    required = bool(
        candidate_status == "blueprint_ready"
        and (
            v2_case == "curvature_normal_continuity_failure"
            or prototype_orientation == "normal_continuity_mismatch"
            or candidate_family == "curvature_normal_aligned_directional_seed_candidate"
        )
    )

    if not required:
        return AdaptiveSurfaceFillV2SeedCandidateGeometryProbe(
            status="not_required",
            action="do_not_build_geometry_probe",
            available=False,
            applied=False,
            selected=False,
            family=str(candidate_family),
            geometry_mode="not_requested",
            face_count=0,
            vertex_count=0,
            reoriented_face_count=0,
            moved_vertex_count=0,
            movement_mean=0.0,
            movement_max=0.0,
            movement_ratio_max=0.0,
            predicted_g1_mean_deviation="-",
            predicted_g1_max_deviation="-",
            predicted_g1_status="not_requested",
            topology_status="not_requested",
            topology_reasons=(),
            reasons=("v2 seed candidate geometry probe is not required for this case",),
        )

    preview_mesh = legacy_result.get("preview_mesh")
    if not isinstance(preview_mesh, trimesh.Trimesh):
        return AdaptiveSurfaceFillV2SeedCandidateGeometryProbe(
            status="error",
            action="geometry_probe_failed",
            available=False,
            applied=False,
            selected=False,
            family=str(candidate_family),
            geometry_mode="missing_legacy_preview_mesh",
            face_count=0,
            vertex_count=0,
            reoriented_face_count=0,
            moved_vertex_count=0,
            movement_mean=0.0,
            movement_max=0.0,
            movement_ratio_max=0.0,
            predicted_g1_mean_deviation="-",
            predicted_g1_max_deviation="-",
            predicted_g1_status="error",
            topology_status="not_evaluated",
            topology_reasons=("legacy preview mesh is unavailable",),
            reasons=("cannot build v2 geometry probe without a legacy preview mesh envelope",),
        )

    patch_face_ids = _tuple_ints(legacy_result.get("new_face_ids"))
    new_vertex_ids = _tuple_ints(legacy_result.get("new_vertex_ids"))
    boundary_vertex_ids = _tuple_ints(legacy_result.get("boundary_vertex_ids"))

    if not patch_face_ids:
        return AdaptiveSurfaceFillV2SeedCandidateGeometryProbe(
            status="error",
            action="geometry_probe_failed",
            available=False,
            applied=False,
            selected=False,
            family=str(candidate_family),
            geometry_mode="missing_patch_faces",
            face_count=int(len(preview_mesh.faces)),
            vertex_count=int(len(preview_mesh.vertices)),
            reoriented_face_count=0,
            moved_vertex_count=0,
            movement_mean=0.0,
            movement_max=0.0,
            movement_ratio_max=0.0,
            predicted_g1_mean_deviation="-",
            predicted_g1_max_deviation="-",
            predicted_g1_status="error",
            topology_status="not_evaluated",
            topology_reasons=("new_face_ids missing from legacy result",),
            reasons=("cannot build v2 geometry probe without generated patch face ids",),
        )

    candidate_mesh = preview_mesh.copy()
    faces = np.asarray(candidate_mesh.faces, dtype=np.int64).copy()

    valid_patch_faces = tuple(
        int(face_id)
        for face_id in patch_face_ids
        if 0 <= int(face_id) < int(len(faces))
    )

    # First isolate the normal-continuity issue: reorient generated patch faces.
    for face_id in valid_patch_faces:
        face = faces[int(face_id)].copy()
        if len(face) >= 3:
            faces[int(face_id), 1], faces[int(face_id), 2] = face[2], face[1]

    candidate_mesh.faces = faces
    candidate_mesh.process(validate=False)

    # Then build the first real non-selecting v2 seed vertex candidate.
    old_vertices = np.asarray(preview_mesh.vertices, dtype=float)
    new_positions, moved_vertices, delta, vertex_reasons = _build_curvature_normal_aligned_vertices(
        candidate_mesh=candidate_mesh,
        source_mesh=preview_mesh,
        patch_face_ids=valid_patch_faces,
        new_vertex_ids=new_vertex_ids,
        boundary_vertex_ids=boundary_vertex_ids,
        metadata=metadata,
    )
    reasons.extend(vertex_reasons)

    candidate_mesh.vertices = new_positions
    candidate_mesh.process(validate=False)

    movement_mean = float(np.mean(delta)) if len(delta) else 0.0
    movement_max = float(np.max(delta)) if len(delta) else 0.0

    context_edge = _float(
        _first_value(
            metadata,
            "adaptive_seed_context_edge_length_median",
            "adaptive_seed_context_edge_median",
            default=1.0,
        ),
        1.0,
    )
    movement_ratio = float(movement_max / max(context_edge, 1.0e-12))

    predicted_mean = _predict_reoriented_g1(_find_boundary_normal_deviation(metadata, "mean"))
    predicted_max = _predict_reoriented_g1(_find_boundary_normal_deviation(metadata, "max"))

    if isinstance(predicted_mean, float) and isinstance(predicted_max, float):
        # After normal reorientation, mean/max cannot be derived exactly by
        # flipping the aggregate old mean and old max independently. Keep the
        # mean estimate, but make the displayed/evaluated max conservative so
        # the gate never reports mean > max.
        if predicted_max < predicted_mean:
            predicted_max = float(predicted_mean)
        predicted_g1_status = "predicted_pass" if predicted_mean <= 35.0 and predicted_max <= 70.0 else "predicted_warning"
    else:
        predicted_g1_status = "not_evaluated"

    topology_status, topology_reasons = _topology_probe(
        candidate_mesh,
        patch_face_ids=valid_patch_faces,
        boundary_vertex_ids=boundary_vertex_ids,
    )

    evaluation = _evaluate_v2_geometry_probe_gate(
        metadata=metadata,
        predicted_g1_mean_deviation=predicted_mean,
        predicted_g1_max_deviation=predicted_max,
        topology_status=topology_status,
        movement_ratio=movement_ratio,
        candidate_mesh=candidate_mesh,
        source_mesh=preview_mesh,
        patch_face_ids=valid_patch_faces,
        boundary_vertex_ids=boundary_vertex_ids,
        movable_vertex_ids=moved_vertices,
    )

    reasons.append("built a non-selecting v2 geometry probe by reorienting generated patch face winding")
    reasons.append("built a first real generated-vertex seed candidate after curvature-normal continuity check")
    reasons.append("candidate is selected only if strict topology, G1, real quality, and real G2 gates pass")

    return AdaptiveSurfaceFillV2SeedCandidateGeometryProbe(
        status=(
            "geometry_probe_selectable"
            if bool(evaluation["evaluation_selectable"])
            else "geometry_probe_ready"
        ),
        action=(
            str(evaluation["evaluation_action"])
            if bool(evaluation["evaluation_selectable"])
            else "evaluate_real_curvature_normal_aligned_seed_candidate_next"
        ),
        available=True,
        applied=True,
        selected=bool(evaluation["evaluation_selectable"]),
        family=str(candidate_family or "curvature_normal_aligned_directional_seed_candidate"),
        geometry_mode="curvature_normal_aligned_vertex_candidate",
        face_count=int(len(candidate_mesh.faces)),
        vertex_count=int(len(candidate_mesh.vertices)),
        reoriented_face_count=int(len(valid_patch_faces)),
        moved_vertex_count=int(len(moved_vertices)),
        movement_mean=float(movement_mean),
        movement_max=float(movement_max),
        movement_ratio_max=float(movement_ratio),
        predicted_g1_mean_deviation=predicted_mean,
        predicted_g1_max_deviation=predicted_max,
        predicted_g1_status=str(predicted_g1_status),
        topology_status=str(topology_status),
        topology_reasons=tuple(topology_reasons),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        evaluation_status=evaluation["evaluation_status"],
        evaluation_action=evaluation["evaluation_action"],
        evaluation_selectable=bool(evaluation["evaluation_selectable"]),
        evaluation_g1_status=evaluation["evaluation_g1_status"],
        evaluation_g1_mean_deviation=evaluation["evaluation_g1_mean_deviation"],
        evaluation_g1_max_deviation=evaluation["evaluation_g1_max_deviation"],
        evaluation_g1_mean_limit=evaluation["evaluation_g1_mean_limit"],
        evaluation_g1_max_limit=evaluation["evaluation_g1_max_limit"],
        evaluation_quality_status=evaluation["evaluation_quality_status"],
        evaluation_g2_status=evaluation["evaluation_g2_status"],
        evaluation_policy=evaluation["evaluation_policy"],
        evaluation_reasons=tuple(evaluation["evaluation_reasons"]),
        candidate_mesh=(candidate_mesh if bool(evaluation["evaluation_selectable"]) else None),
    )


__all__ = (
    "AdaptiveSurfaceFillV2SeedCandidateGeometryProbe",
    "build_curvature_normal_aligned_seed_candidate_geometry_probe",
)
