from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class Open3DTensorMeshSummary:
    vertices_shape: tuple[int, ...]
    faces_shape: tuple[int, ...]
    vertex_dtype: str
    face_dtype: str
    vertex_device: str
    face_device: str
    is_cpu: bool


def require_open3d() -> Any:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Open3D is unavailable: {exc}") from exc

    if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry"):
        raise RuntimeError("Open3D tensor geometry module is unavailable.")

    if not hasattr(o3d.t.geometry, "TriangleMesh"):
        raise RuntimeError("Open3D tensor TriangleMesh is unavailable.")

    return o3d


def validate_trimesh_for_open3d_tensor(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N, 3), got {vertices.shape!r}.")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape (M, 3), got {faces.shape!r}.")

    if len(vertices) == 0:
        raise ValueError("Open3D tensor mesh conversion requires at least one vertex.")

    if len(faces) == 0:
        raise ValueError("Open3D tensor mesh conversion requires at least one face.")

    if not np.isfinite(vertices).all():
        raise ValueError("Open3D tensor mesh conversion requires finite vertex positions.")

    if np.any(faces < 0):
        raise ValueError("Open3D tensor mesh conversion requires non-negative face indices.")

    if int(np.max(faces)) >= len(vertices):
        raise ValueError("Open3D tensor mesh conversion found face index outside vertex range.")


def trimesh_to_open3d_tensor_mesh(
    mesh: trimesh.Trimesh,
    *,
    vertex_dtype: str = "float32",
    face_dtype: str = "int64",
) -> Any:
    """Convert trimesh to Open3D tensor TriangleMesh with explicit copy ownership.

    Important:
    - Do not use Tensor.from_numpy() here because it shares memory.
    - Use np.array(..., copy=True) before constructing Open3D tensors.
    - The Open3D Tensor constructor also copies list/NumPy data, but the explicit
      copy makes ownership clear and protects against non-contiguous views.
    """

    validate_trimesh_for_open3d_tensor(mesh)
    o3d = require_open3d()

    if vertex_dtype == "float32":
        np_vertex_dtype = np.float32
        o3d_vertex_dtype = o3d.core.Dtype.Float32
    elif vertex_dtype == "float64":
        np_vertex_dtype = np.float64
        o3d_vertex_dtype = o3d.core.Dtype.Float64
    else:
        raise ValueError(f"Unsupported Open3D vertex dtype: {vertex_dtype!r}")

    if face_dtype == "int64":
        np_face_dtype = np.int64
        o3d_face_dtype = o3d.core.Dtype.Int64
    elif face_dtype == "int32":
        np_face_dtype = np.int32
        o3d_face_dtype = o3d.core.Dtype.Int32
    else:
        raise ValueError(f"Unsupported Open3D face dtype: {face_dtype!r}")

    vertices = np.array(mesh.vertices, dtype=np_vertex_dtype, copy=True)
    faces = np.array(mesh.faces, dtype=np_face_dtype, copy=True)

    vertex_tensor = o3d.core.Tensor(
        vertices,
        dtype=o3d_vertex_dtype,
        device=o3d.core.Device("CPU:0"),
    )
    face_tensor = o3d.core.Tensor(
        faces,
        dtype=o3d_face_dtype,
        device=o3d.core.Device("CPU:0"),
    )

    return o3d.t.geometry.TriangleMesh(
        vertex_positions=vertex_tensor,
        triangle_indices=face_tensor,
    )


def open3d_tensor_mesh_to_trimesh(tensor_mesh: Any) -> trimesh.Trimesh:
    """Convert Open3D tensor TriangleMesh to detached trimesh.Trimesh.

    Open3D tensor .numpy() can share memory with the tensor. We immediately copy
    into Python-owned NumPy arrays before constructing trimesh.
    """

    vertices = np.array(
        tensor_mesh.vertex.positions.cpu().numpy(),
        dtype=float,
        copy=True,
    )
    faces = np.array(
        tensor_mesh.triangle.indices.cpu().numpy(),
        dtype=np.int64,
        copy=True,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Open3D tensor output vertices have invalid shape {vertices.shape!r}.")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Open3D tensor output faces have invalid shape {faces.shape!r}.")

    if not np.isfinite(vertices).all():
        raise ValueError("Open3D tensor output contains non-finite vertex positions.")

    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out


def summarize_open3d_tensor_mesh(tensor_mesh: Any) -> Open3DTensorMeshSummary:
    vertices = tensor_mesh.vertex.positions
    faces = tensor_mesh.triangle.indices

    return Open3DTensorMeshSummary(
        vertices_shape=tuple(int(v) for v in vertices.shape),
        faces_shape=tuple(int(v) for v in faces.shape),
        vertex_dtype=str(vertices.dtype),
        face_dtype=str(faces.dtype),
        vertex_device=str(vertices.device),
        face_device=str(faces.device),
        is_cpu=bool(tensor_mesh.is_cpu),
    )


def fill_holes_with_open3d_tensor(
    mesh: trimesh.Trimesh,
    *,
    hole_size: float = 1_000_000.0,
    compute_normals: bool = False,
) -> trimesh.Trimesh:
    """Run Open3D tensor TriangleMesh.fill_holes safely on a mesh copy."""

    tensor_mesh = trimesh_to_open3d_tensor_mesh(mesh)

    if compute_normals:
        tensor_mesh = tensor_mesh.compute_triangle_normals()
        tensor_mesh = tensor_mesh.compute_vertex_normals()

    filled = tensor_mesh.fill_holes(float(hole_size))
    return open3d_tensor_mesh_to_trimesh(filled)
