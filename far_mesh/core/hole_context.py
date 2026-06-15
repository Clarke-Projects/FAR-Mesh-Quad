from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from far_mesh.core.open3d_tensor_bridge import fill_holes_with_open3d_tensor
from far_mesh.core.selection_topology import find_hole_candidates


@dataclass(frozen=True)
class LocalHoleContext:
    mesh: trimesh.Trimesh
    source_face_ids: tuple[int, ...]
    source_vertex_ids: tuple[int, ...]
    local_to_source_vertex_ids: tuple[int, ...]
    source_to_local_vertex_ids: dict[int, int]
    target_boundary_source_vertex_ids: tuple[int, ...]
    target_boundary_local_vertex_ids: tuple[int, ...]
    rings: int


def candidate_boundary_vertex_ids(candidate: Any) -> tuple[int, ...]:
    values = getattr(candidate, "boundary_vertices", None)
    if values is None:
        loop = getattr(candidate, "loop", None)
        values = getattr(loop, "vertices", ()) if loop is not None else ()

    ids: list[int] = []
    for value in values or ():
        vid = int(value)
        if not ids or ids[-1] != vid:
            ids.append(vid)

    if len(ids) >= 2 and ids[0] == ids[-1]:
        ids.pop()

    if not ids:
        raise ValueError("candidate has no boundary vertices")

    return tuple(ids)


def _face_adjacency_map(mesh: trimesh.Trimesh) -> dict[int, set[int]]:
    face_count = int(len(mesh.faces))
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(face_count)}

    pairs = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=np.int64)))
    if pairs.size == 0:
        return adjacency

    for a, b in pairs:
        ia = int(a)
        ib = int(b)
        if 0 <= ia < face_count and 0 <= ib < face_count:
            adjacency[ia].add(ib)
            adjacency[ib].add(ia)

    return adjacency


def collect_boundary_context_face_ids(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
) -> tuple[int, ...]:
    if rings < 0:
        raise ValueError("rings must be >= 0")

    boundary_ids = set(candidate_boundary_vertex_ids(candidate))
    faces = np.asarray(mesh.faces)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh must have triangular faces")

    seed_faces = {
        int(face_id)
        for face_id, face in enumerate(faces)
        if any(int(vertex_id) in boundary_ids for vertex_id in face)
    }

    if not seed_faces:
        raise ValueError("could not find faces touching candidate boundary")

    selected = set(seed_faces)
    frontier = set(seed_faces)
    adjacency = _face_adjacency_map(mesh)

    for _ in range(rings):
        next_frontier: set[int] = set()
        for face_id in frontier:
            next_frontier.update(adjacency.get(face_id, set()))
        next_frontier.difference_update(selected)
        selected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return tuple(sorted(selected))




def collect_boundary_context_face_ids_low_memory(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
) -> tuple[int, ...]:
    """Return local support faces around a hole boundary without global face adjacency.

    This helper is meant for very large meshes.  It avoids ``mesh.face_adjacency``
    because trimesh may build a full global adjacency table for millions of
    faces.  Instead it scans the face array a small bounded number of times:

    - pass 0 finds faces touching the target boundary vertices;
    - each additional ring expands through vertices touched by the current
      frontier faces.

    The algorithm is O(rings * face_count) but its memory use stays close to
    the face array plus small boolean masks.  That is preferable for large-mesh
    hole-fill work units where preserving RAM is more important than building a
    global adjacency cache.
    """

    if rings < 0:
        raise ValueError("rings must be >= 0")

    boundary_ids = np.asarray(candidate_boundary_vertex_ids(candidate), dtype=np.int64)
    if boundary_ids.size == 0:
        raise ValueError("candidate has no boundary vertices")

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("mesh must have triangular faces")

    tri_faces = faces[:, :3]
    face_count = int(tri_faces.shape[0])
    selected_mask = np.zeros(face_count, dtype=bool)
    frontier_vertices = np.unique(boundary_ids)

    for _ring in range(int(rings) + 1):
        if frontier_vertices.size == 0:
            break

        touching_mask = np.isin(tri_faces, frontier_vertices).any(axis=1)
        new_mask = touching_mask & ~selected_mask
        if not bool(np.any(new_mask)):
            break

        selected_mask |= new_mask
        frontier_vertices = np.setdiff1d(
            np.unique(tri_faces[new_mask].reshape(-1)),
            frontier_vertices,
            assume_unique=False,
        )

    selected = np.flatnonzero(selected_mask).astype(np.int64)
    if selected.size == 0:
        raise ValueError("could not find faces touching candidate boundary")

    return tuple(int(v) for v in selected.tolist())

def build_local_hole_context_mesh(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
    low_memory: bool = True,
) -> LocalHoleContext:
    # Do not copy the full source mesh here.  Large-mesh preview work units must
    # avoid duplicating millions of vertices/faces before the local ROI is known.
    source_vertex_shape_before = np.asarray(mesh.vertices).shape
    source_face_shape_before = np.asarray(mesh.faces).shape

    if low_memory:
        source_face_ids = collect_boundary_context_face_ids_low_memory(mesh, candidate, rings=rings)
    else:
        source_face_ids = collect_boundary_context_face_ids(mesh, candidate, rings=rings)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)[list(source_face_ids)]

    source_vertex_ids = tuple(sorted(int(v) for v in np.unique(source_faces.reshape(-1))))
    source_to_local = {source_id: local_id for local_id, source_id in enumerate(source_vertex_ids)}

    local_vertices = np.array(
        np.asarray(mesh.vertices, dtype=float)[list(source_vertex_ids)],
        dtype=float,
        copy=True,
    )
    local_faces = np.array(
        [[source_to_local[int(vertex_id)] for vertex_id in face] for face in source_faces],
        dtype=np.int64,
        copy=True,
    )

    target_source_ids = candidate_boundary_vertex_ids(candidate)
    missing = [vid for vid in target_source_ids if vid not in source_to_local]
    if missing:
        raise ValueError(f"local hole context is missing target boundary vertices: {missing}")

    target_local_ids = tuple(source_to_local[vid] for vid in target_source_ids)

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    if np.asarray(mesh.vertices).shape != source_vertex_shape_before:
        raise RuntimeError("local hole context extraction unexpectedly changed source vertex shape")
    if np.asarray(mesh.faces).shape != source_face_shape_before:
        raise RuntimeError("local hole context extraction unexpectedly changed source face shape")

    return LocalHoleContext(
        mesh=local_mesh,
        source_face_ids=source_face_ids,
        source_vertex_ids=source_vertex_ids,
        local_to_source_vertex_ids=source_vertex_ids,
        source_to_local_vertex_ids=source_to_local,
        target_boundary_source_vertex_ids=target_source_ids,
        target_boundary_local_vertex_ids=target_local_ids,
        rings=int(rings),
    )


def _candidate_matches_boundary(candidate: Any, boundary_ids: tuple[int, ...]) -> bool:
    candidate_ids = set(candidate_boundary_vertex_ids(candidate))
    return candidate_ids == set(int(v) for v in boundary_ids)


def inspect_open3d_local_hole_context_fill(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: int = 1,
    hole_size: float = 1_000_000.0,
) -> dict[str, Any]:
    """Dry-run Open3D fill_holes on a local N-ring context mesh.

    This is diagnostic only. It does not mutate the source mesh and does not
    return a commit-ready mesh. The main purpose is to learn whether a local
    context extraction creates artificial boundaries and whether Open3D fills
    the target boundary in the local copy.
    """

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    context = build_local_hole_context_mesh(mesh, candidate, rings=rings)

    before_candidates = tuple(find_hole_candidates(context.mesh))
    target_candidate_count_before = sum(
        1
        for item in before_candidates
        if _candidate_matches_boundary(item, context.target_boundary_local_vertex_ids)
    )
    target_found_before = target_candidate_count_before > 0

    filled = fill_holes_with_open3d_tensor(context.mesh, hole_size=hole_size)
    after_candidates = tuple(find_hole_candidates(filled))
    target_candidate_count_after = sum(
        1
        for item in after_candidates
        if _candidate_matches_boundary(item, context.target_boundary_local_vertex_ids)
    )
    target_found_after = target_candidate_count_after > 0

    non_target_candidate_count_before = max(
        0,
        int(len(before_candidates)) - int(target_candidate_count_before),
    )
    non_target_candidate_count_after = max(
        0,
        int(len(after_candidates)) - int(target_candidate_count_after),
    )

    added_faces = int(len(filled.faces) - len(context.mesh.faces))
    added_vertices = int(len(filled.vertices) - len(context.mesh.vertices))

    target_boundary_filled = bool(target_found_before and not target_found_after)
    local_result_clean = bool(target_boundary_filled and len(after_candidates) == 0)
    residual_local_candidates_after = int(len(after_candidates))

    notes: list[str] = [
        "Local context dry-run only; source mesh was not modified.",
        "Open3D tensor fill_holes ran on an extracted N-ring context copy.",
        "This report is not a repair commit and does not return a replacement mesh.",
    ]

    if len(before_candidates) > 1:
        notes.append(
            "Local context has more than one candidate; artificial outer boundaries may exist."
        )

    if non_target_candidate_count_before > 0:
        notes.append(
            "Local context has non-target candidates before fill; extracted context may include artificial boundaries."
        )

    if non_target_candidate_count_after > 0:
        notes.append(
            "Local context has residual non-target candidates after fill; result should remain diagnostic-only."
        )

    if not target_found_before:
        notes.append(
            "Target boundary was not matched in local candidates before fill; mapping should be inspected."
        )

    if added_faces > 0 and target_found_after:
        notes.append(
            "Faces were added but target boundary still appears present after fill."
        )

    if not local_result_clean:
        notes.append(
            "Local context result is not clean enough to be considered commit-ready."
        )

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("local Open3D dry-run unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("local Open3D dry-run unexpectedly mutated source faces")

    return {
        "operation": "open3d_local_hole_context_fill_dry_run",
        "dry_run": True,
        "rings": int(rings),
        "hole_size": float(hole_size),
        "source_face_count": int(len(mesh.faces)),
        "source_vertex_count": int(len(mesh.vertices)),
        "context_face_count": int(len(context.mesh.faces)),
        "context_vertex_count": int(len(context.mesh.vertices)),
        "source_face_ids": context.source_face_ids,
        "source_vertex_ids": context.source_vertex_ids,
        "target_boundary_source_vertex_ids": context.target_boundary_source_vertex_ids,
        "target_boundary_local_vertex_ids": context.target_boundary_local_vertex_ids,
        "local_candidate_count_before": int(len(before_candidates)),
        "local_candidate_count_after": int(len(after_candidates)),
        "target_candidate_count_before": int(target_candidate_count_before),
        "target_candidate_count_after": int(target_candidate_count_after),
        "non_target_candidate_count_before": int(non_target_candidate_count_before),
        "non_target_candidate_count_after": int(non_target_candidate_count_after),
        "residual_local_candidates_after": int(residual_local_candidates_after),
        "target_found_before": bool(target_found_before),
        "target_found_after": bool(target_found_after),
        "target_boundary_filled": bool(target_boundary_filled),
        "local_result_clean": bool(local_result_clean),
        "source_patch_ready": False,
        "commit_recommendation": "diagnostic_only",
        "added_faces": added_faces,
        "added_vertices": added_vertices,
        "notes": notes,
    }


def inspect_open3d_local_context_fill_across_rings(
    mesh: trimesh.Trimesh,
    candidate: Any,
    *,
    rings: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4),
    hole_size: float = 1_000_000.0,
) -> dict[str, Any]:
    """Compare whole-mesh Open3D fill_holes with local N-ring dry-runs.

    This is diagnostic only. It helps identify whether local context extraction
    changes Open3D's fill behavior compared with whole-mesh fill.
    """

    source_vertices_before = np.asarray(mesh.vertices).copy()
    source_faces_before = np.asarray(mesh.faces).copy()

    whole_before_candidates = tuple(find_hole_candidates(mesh))
    whole_filled = fill_holes_with_open3d_tensor(mesh, hole_size=hole_size)
    whole_after_candidates = tuple(find_hole_candidates(whole_filled))

    whole_added_faces = int(len(whole_filled.faces) - len(mesh.faces))
    whole_added_vertices = int(len(whole_filled.vertices) - len(mesh.vertices))
    whole_filled_candidate_delta = int(
        len(whole_before_candidates) - len(whole_after_candidates)
    )

    local_reports: list[dict[str, Any]] = []
    for ring_count in rings:
        report = dict(
            inspect_open3d_local_hole_context_fill(
                mesh,
                candidate,
                rings=int(ring_count),
                hole_size=hole_size,
            )
        )

        report["whole_mesh_added_faces"] = whole_added_faces
        report["whole_mesh_added_vertices"] = whole_added_vertices
        report["added_faces_minus_whole"] = int(report["added_faces"]) - whole_added_faces
        report["added_vertices_minus_whole"] = int(report["added_vertices"]) - whole_added_vertices
        report["local_added_faces_exceeds_whole"] = int(report["added_faces"]) > whole_added_faces
        report["local_added_vertices_exceeds_whole"] = int(report["added_vertices"]) > whole_added_vertices
        report["added_faces_ratio_to_whole"] = (
            None if whole_added_faces == 0 else float(report["added_faces"]) / float(whole_added_faces)
        )

        notes = list(report.get("notes") or [])
        if report["local_added_faces_exceeds_whole"]:
            notes.append(
                "Local context added more faces than whole-mesh fill; local result should remain diagnostic-only."
            )
        if report["local_added_vertices_exceeds_whole"]:
            notes.append(
                "Local context added more vertices than whole-mesh fill; local result should remain diagnostic-only."
            )
        report["notes"] = notes

        local_reports.append(report)

    if not np.allclose(np.asarray(mesh.vertices), source_vertices_before):
        raise RuntimeError("local/whole Open3D comparison unexpectedly mutated source vertices")
    if not np.array_equal(np.asarray(mesh.faces), source_faces_before):
        raise RuntimeError("local/whole Open3D comparison unexpectedly mutated source faces")

    return {
        "operation": "open3d_local_context_vs_whole_fill_dry_run",
        "dry_run": True,
        "hole_size": float(hole_size),
        "whole_candidate_count_before": int(len(whole_before_candidates)),
        "whole_candidate_count_after": int(len(whole_after_candidates)),
        "whole_filled_candidate_delta": int(whole_filled_candidate_delta),
        "whole_added_faces": int(whole_added_faces),
        "whole_added_vertices": int(whole_added_vertices),
        "local_reports": tuple(local_reports),
        "notes": (
            "Comparison is diagnostic only; source mesh was not modified.",
            "Local context reports are not commit-ready patch data.",
        ),
    }
