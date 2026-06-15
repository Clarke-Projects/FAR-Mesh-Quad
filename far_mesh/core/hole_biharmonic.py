from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class BiharmonicFairingResult:
    preview_mesh: trimesh.Trimesh
    moved_vertex_ids: tuple[int, ...]
    fixed_vertex_ids: tuple[int, ...]
    patch_face_ids: tuple[int, ...]
    max_displacement: float
    mean_displacement: float
    solver: str
    notes: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "moved_vertex_ids": self.moved_vertex_ids,
            "fixed_vertex_ids": self.fixed_vertex_ids,
            "patch_face_ids": self.patch_face_ids,
            "max_displacement": float(self.max_displacement),
            "mean_displacement": float(self.mean_displacement),
            "notes": self.notes,
        }


def _triangle_area(points: np.ndarray) -> float:
    a, b, c = np.asarray(points, dtype=float).reshape(3, 3)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def _cotangent(a: np.ndarray, b: np.ndarray) -> float:
    cross_norm = float(np.linalg.norm(np.cross(a, b)))
    if cross_norm <= 1.0e-14:
        return 0.0
    return float(np.dot(a, b) / cross_norm)


def _assemble_uniform_laplacian(
    *,
    local_vertex_count: int,
    local_faces: np.ndarray,
) -> np.ndarray:
    """Build a simple graph Laplacian.

    First biharmonic checkpoint intentionally uses a uniform Laplacian because
    it is robust for tests and avoids cotangent instability on poor seeds.
    Cotangent/mass-weighted biharmonic can replace this after safety gates.
    """

    adjacency: list[set[int]] = [set() for _ in range(local_vertex_count)]

    for face in np.asarray(local_faces, dtype=np.int64).reshape(-1, 3):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        adjacency[a].add(b)
        adjacency[a].add(c)
        adjacency[b].add(a)
        adjacency[b].add(c)
        adjacency[c].add(a)
        adjacency[c].add(b)

    laplacian = np.zeros((local_vertex_count, local_vertex_count), dtype=float)
    for i, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree <= 0:
            laplacian[i, i] = 1.0
            continue

        laplacian[i, i] = 1.0
        weight = -1.0 / float(degree)
        for j in sorted(neighbors):
            laplacian[i, int(j)] = weight

    return laplacian


def _assemble_cotan_laplacian(
    *,
    local_vertices: np.ndarray,
    local_faces: np.ndarray,
) -> np.ndarray:
    """Build a symmetric cotangent Laplacian.

    This is available for later experiments, but the public function defaults
    to the uniform Laplacian for the first stable checkpoint.
    """

    verts = np.asarray(local_vertices, dtype=float)
    faces = np.asarray(local_faces, dtype=np.int64).reshape(-1, 3)
    n = int(len(verts))
    weights = np.zeros((n, n), dtype=float)

    for face in faces:
        i, j, k = (int(face[0]), int(face[1]), int(face[2]))
        pi, pj, pk = verts[i], verts[j], verts[k]

        cot_k = _cotangent(pi - pk, pj - pk)
        cot_i = _cotangent(pj - pi, pk - pi)
        cot_j = _cotangent(pk - pj, pi - pj)

        for a, b, w in (
            (i, j, cot_k),
            (j, k, cot_i),
            (k, i, cot_j),
        ):
            if not np.isfinite(w):
                continue
            w = max(-1.0e6, min(1.0e6, float(w)))
            weights[a, b] += 0.5 * w
            weights[b, a] += 0.5 * w

    laplacian = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_sum = float(np.sum(weights[i]))
        if row_sum <= 1.0e-14 or not np.isfinite(row_sum):
            laplacian[i, i] = 1.0
            continue
        laplacian[i, i] = 1.0
        for j in range(n):
            if i == j:
                continue
            if weights[i, j] != 0.0:
                laplacian[i, j] = -float(weights[i, j]) / row_sum

    return laplacian


def fair_patch_biharmonic(
    mesh: trimesh.Trimesh,
    *,
    patch_face_ids: tuple[int, ...] | list[int],
    fixed_vertex_ids: tuple[int, ...] | list[int],
    movable_vertex_ids: tuple[int, ...] | list[int],
    laplacian: str = "uniform",
    regularization: float = 1.0e-8,
) -> BiharmonicFairingResult:
    """Fair a patch by solving a fixed-boundary biharmonic system.

    This solves the C0 thin-plate problem:

        minimize || L² X ||²
        with fixed boundary vertices.

    Boundary vertices are not moved. Only movable patch vertices are solved.
    This function is preview-safe and does not mutate the input mesh.
    """

    if laplacian not in {"uniform", "cotan"}:
        raise ValueError("laplacian must be 'uniform' or 'cotan'")

    face_ids = tuple(int(v) for v in patch_face_ids)
    fixed_ids = tuple(int(v) for v in fixed_vertex_ids)
    movable_ids = tuple(int(v) for v in movable_vertex_ids)

    if not face_ids:
        raise ValueError("biharmonic fairing requires patch_face_ids")
    if not fixed_ids:
        raise ValueError("biharmonic fairing requires fixed_vertex_ids")
    if not movable_ids:
        raise ValueError("biharmonic fairing requires movable_vertex_ids")

    source_vertices = np.asarray(mesh.vertices, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)

    if max(face_ids) >= len(source_faces) or min(face_ids) < 0:
        raise ValueError("patch_face_ids contain out-of-range face ids")

    patch_faces_global = source_faces[list(face_ids), :3]
    patch_vertex_ids = tuple(
        int(v) for v in sorted(set(int(v) for face in patch_faces_global for v in face))
    )

    patch_vertex_set = set(patch_vertex_ids)
    fixed_set = set(fixed_ids)
    movable_set = set(movable_ids)

    if not movable_set.issubset(patch_vertex_set):
        raise ValueError("movable_vertex_ids must be part of the patch")
    if not fixed_set.issubset(patch_vertex_set):
        raise ValueError("fixed_vertex_ids must be part of the patch")

    local_index = {global_id: i for i, global_id in enumerate(patch_vertex_ids)}
    local_vertices = source_vertices[list(patch_vertex_ids)].copy()
    local_faces = np.asarray(
        [[local_index[int(v)] for v in face] for face in patch_faces_global],
        dtype=np.int64,
    )

    fixed_local = np.asarray([local_index[v] for v in fixed_ids], dtype=np.int64)
    movable_local = np.asarray([local_index[v] for v in movable_ids], dtype=np.int64)

    if laplacian == "cotan":
        l_matrix = _assemble_cotan_laplacian(
            local_vertices=local_vertices,
            local_faces=local_faces,
        )
    else:
        l_matrix = _assemble_uniform_laplacian(
            local_vertex_count=len(local_vertices),
            local_faces=local_faces,
        )

    # Thin-plate / biharmonic operator.
    biharmonic = l_matrix.T @ l_matrix

    ii = np.ix_(movable_local, movable_local)
    ib = np.ix_(movable_local, fixed_local)

    a_matrix = biharmonic[ii].copy()
    a_matrix += np.eye(len(movable_local), dtype=float) * float(regularization)

    rhs = -biharmonic[ib] @ local_vertices[fixed_local]

    try:
        solved = np.linalg.solve(a_matrix, rhs)
        solver_name = f"dense_{laplacian}_biharmonic"
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(a_matrix, rhs, rcond=None)[0]
        solver_name = f"dense_{laplacian}_biharmonic_lstsq"

    out_vertices = source_vertices.copy()
    before = source_vertices[list(movable_ids)].copy()

    for global_id, local_id, position in zip(movable_ids, movable_local, solved):
        del local_id
        out_vertices[int(global_id)] = np.asarray(position, dtype=float).reshape(3)

    after = out_vertices[list(movable_ids)]
    displacement = np.linalg.norm(after - before, axis=1)

    preview = trimesh.Trimesh(
        vertices=out_vertices,
        faces=np.array(source_faces, dtype=np.int64, copy=True),
        process=False,
    )

    if not np.array_equal(np.asarray(mesh.faces), source_faces):
        raise RuntimeError("biharmonic fairing unexpectedly mutated source faces")
    if not np.allclose(np.asarray(mesh.vertices), source_vertices):
        raise RuntimeError("biharmonic fairing unexpectedly mutated source vertices")

    return BiharmonicFairingResult(
        preview_mesh=preview,
        moved_vertex_ids=movable_ids,
        fixed_vertex_ids=fixed_ids,
        patch_face_ids=face_ids,
        max_displacement=float(np.max(displacement)) if len(displacement) else 0.0,
        mean_displacement=float(np.mean(displacement)) if len(displacement) else 0.0,
        solver=solver_name,
        notes=(
            "Standalone C0 biharmonic fairing kernel.",
            "Boundary/fixed vertices were held fixed.",
            "Only movable patch vertices were solved.",
            "No source mesh mutation occurred.",
            "G1 normal constraints are intentionally reserved for a later checkpoint.",
        ),
    )
