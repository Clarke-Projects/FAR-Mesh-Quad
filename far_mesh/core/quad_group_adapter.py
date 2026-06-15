from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


Int1 = np.ndarray
Int2 = np.ndarray


@dataclass(slots=True)
class DecodedGroupData:
    group_id: np.ndarray
    patch_id: np.ndarray
    parent_quad_id: np.ndarray


@dataclass(slots=True)
class QuadGroupProcessOptions:
    decode_mode: str = "auto"  # auto | material_ids | face_colors | texture_lookup | synthetic_faces
    texture_path: str | None = None
    cleanup: bool = True
    reduce: bool = False
    target_ratio: float = 1.0
    boundary_weight: float = 5.0
    allow_non_manifold_edge_removal: bool = False


@dataclass(slots=True)
class QuadProxyMetadata:
    group_id: Int1
    patch_id: Int1
    parent_quad_id: Int1
    is_group_boundary: np.ndarray
    source_face_kind: str = "quad"
    parent_biquad_id: Int1 | None = None
    corner_ids: Int2 | None = None
    mid_edge_ids: Int2 | None = None
    interior_ids: Int1 | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuadProxyMesh:
    proxy_trimesh: trimesh.Trimesh
    metadata: QuadProxyMetadata


@dataclass(slots=True)
class ProcessedGroupResult:
    group_id: int
    mesh: Any
    triangle_count_before: int
    triangle_count_after: int


def _require_open3d() -> Any:
    """
    Import Open3D only inside operations that actually need it.

    This avoids loading Open3D native libraries merely by importing
    quad_group_adapter.py or task_registry.py. Group processing should be
    routed through PROCESS where possible, but lazy import also keeps the
    parent process cleaner during tests and normal app startup.
    """

    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required for this quad/group processing operation but could not be imported."
        ) from exc

    return o3d


def _normalize_ids(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids).reshape(-1)
    unique = np.unique(ids)
    remap = {int(v): i for i, v in enumerate(unique.tolist())}
    return np.asarray([remap[int(v)] for v in ids], dtype=np.int64)


def _sanitize_int_ids(ids: np.ndarray | None) -> np.ndarray:
    if ids is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    return np.unique(arr)


def _sanitize_face_ids_for_count(
    face_ids: np.ndarray | None,
    num_faces: int,
) -> np.ndarray:
    ids = _sanitize_int_ids(face_ids)
    if num_faces <= 0 or ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    return ids[(ids >= 0) & (ids < num_faces)]


def _concat_or_empty(
    arrays: list[np.ndarray],
    *,
    shape_tail: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not arrays:
        return np.empty((0, *shape_tail), dtype=dtype)
    return np.concatenate(arrays, axis=0)


def decode_groups_from_material_ids(tm: trimesh.Trimesh) -> DecodedGroupData:
    material_ids = getattr(tm.visual, "face_materials", None)
    if material_ids is None:
        raise ValueError("No face_materials found on trimesh visual.")
    material_ids = np.asarray(material_ids, dtype=np.int64).reshape(-1)
    group_id = _normalize_ids(material_ids)
    return DecodedGroupData(
        group_id=group_id,
        patch_id=group_id.copy(),
        parent_quad_id=np.arange(len(group_id), dtype=np.int64),
    )


def decode_groups_from_face_colors(tm: trimesh.Trimesh) -> DecodedGroupData:
    face_colors = getattr(tm.visual, "face_colors", None)
    if face_colors is None:
        raise ValueError("No face_colors found on trimesh visual.")
    rgba = np.asarray(face_colors, dtype=np.uint8)
    if rgba.ndim != 2 or rgba.shape[1] < 3:
        raise ValueError("face_colors must be shaped (N, >=3).")

    rgb = rgba[:, :3]
    _, inverse = np.unique(rgb, axis=0, return_inverse=True)
    group_id = inverse.astype(np.int64)

    return DecodedGroupData(
        group_id=group_id,
        patch_id=group_id.copy(),
        parent_quad_id=np.arange(len(group_id), dtype=np.int64),
    )


def decode_groups_from_texture_lookup(
    tm: trimesh.Trimesh,
    texture_path: str | Path,
) -> DecodedGroupData:
    image = np.asarray(Image.open(texture_path).convert("RGB"))
    h, w, _ = image.shape

    uv = getattr(tm.visual, "uv", None)
    faces = np.asarray(tm.faces, dtype=np.int64)

    if uv is None:
        raise ValueError("No vertex UVs found on trimesh visual.")
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] < 2:
        raise ValueError("UV array must be shaped (V, 2).")

    tri_uv = uv[faces]
    tri_uv_centroid = tri_uv.mean(axis=1)

    u = np.clip(tri_uv_centroid[:, 0], 0.0, 1.0)
    v = np.clip(1.0 - tri_uv_centroid[:, 1], 0.0, 1.0)

    px = np.minimum((u * (w - 1)).astype(np.int64), w - 1)
    py = np.minimum((v * (h - 1)).astype(np.int64), h - 1)

    rgb = image[py, px]
    _, inverse = np.unique(rgb, axis=0, return_inverse=True)
    group_id = inverse.astype(np.int64)

    return DecodedGroupData(
        group_id=group_id,
        patch_id=group_id.copy(),
        parent_quad_id=np.arange(len(group_id), dtype=np.int64),
    )


def decode_groups_synthetic_per_face(tm: trimesh.Trimesh) -> DecodedGroupData:
    n = int(len(tm.faces))
    ids = np.arange(n, dtype=np.int64)
    return DecodedGroupData(
        group_id=ids,
        patch_id=ids.copy(),
        parent_quad_id=ids.copy(),
    )


def compute_group_boundary_flags(
    tm: trimesh.Trimesh,
    group_id: np.ndarray,
) -> np.ndarray:
    faces = np.asarray(tm.faces, dtype=np.int64)
    num_faces = len(faces)
    if num_faces == 0:
        return np.zeros((0,), dtype=bool)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}

    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            e = (int(min(u, v)), int(max(u, v)))
            edge_to_faces.setdefault(e, []).append(fi)

    is_boundary = np.zeros(num_faces, dtype=bool)

    for face_indices in edge_to_faces.values():
        if len(face_indices) == 2:
            f0, f1 = face_indices
            if int(group_id[f0]) != int(group_id[f1]):
                is_boundary[f0] = True
                is_boundary[f1] = True

    return is_boundary


def build_quadwild_proxy_from_mesh(
    mesh: trimesh.Trimesh,
    opts: QuadGroupProcessOptions,
) -> QuadProxyMesh:
    tm = mesh.copy()
    mode = str(opts.decode_mode or "auto").strip().lower()

    decoded: DecodedGroupData | None = None

    if mode == "material_ids":
        decoded = decode_groups_from_material_ids(tm)

    elif mode == "face_colors":
        decoded = decode_groups_from_face_colors(tm)

    elif mode == "texture_lookup":
        if not opts.texture_path:
            raise ValueError("texture_path is required for texture_lookup decode mode.")
        decoded = decode_groups_from_texture_lookup(tm, opts.texture_path)

    elif mode == "synthetic_faces":
        decoded = decode_groups_synthetic_per_face(tm)

    else:  # auto
        try:
            decoded = decode_groups_from_material_ids(tm)
        except Exception:
            pass

        if decoded is None:
            try:
                decoded = decode_groups_from_face_colors(tm)
            except Exception:
                pass

        if decoded is None and opts.texture_path:
            try:
                decoded = decode_groups_from_texture_lookup(tm, opts.texture_path)
            except Exception:
                pass

        if decoded is None:
            decoded = decode_groups_synthetic_per_face(tm)

    is_group_boundary = compute_group_boundary_flags(tm, decoded.group_id)

    return QuadProxyMesh(
        proxy_trimesh=tm,
        metadata=QuadProxyMetadata(
            group_id=decoded.group_id,
            patch_id=decoded.patch_id,
            parent_quad_id=decoded.parent_quad_id,
            is_group_boundary=is_group_boundary,
            source_face_kind="quad",
        ),
    )


def trimesh_to_tensor_proxy(proxy: QuadProxyMesh) -> Any:
    o3d = _require_open3d()

    tm = proxy.proxy_trimesh
    meta = proxy.metadata

    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(
        np.asarray(tm.vertices, dtype=np.float32),
        dtype=o3d.core.float32,
    )
    tmesh.triangle.indices = o3d.core.Tensor(
        np.asarray(tm.faces, dtype=np.int64),
        dtype=o3d.core.int64,
    )

    tmesh.triangle["group_id"] = o3d.core.Tensor(
        np.asarray(meta.group_id, dtype=np.int32),
        dtype=o3d.core.int32,
    )
    tmesh.triangle["patch_id"] = o3d.core.Tensor(
        np.asarray(meta.patch_id, dtype=np.int32),
        dtype=o3d.core.int32,
    )
    tmesh.triangle["parent_quad_id"] = o3d.core.Tensor(
        np.asarray(meta.parent_quad_id, dtype=np.int32),
        dtype=o3d.core.int32,
    )
    tmesh.triangle["is_group_boundary"] = o3d.core.Tensor(
        np.asarray(meta.is_group_boundary, dtype=bool),
        dtype=o3d.core.bool,
    )

    if meta.parent_biquad_id is not None:
        tmesh.triangle["parent_biquad_id"] = o3d.core.Tensor(
            np.asarray(meta.parent_biquad_id, dtype=np.int32),
            dtype=o3d.core.int32,
        )

    return tmesh


def expand_face_selection_to_groups(
    selected_face_ids: np.ndarray,
    group_id: np.ndarray,
) -> np.ndarray:
    ids = _sanitize_face_ids_for_count(selected_face_ids, len(group_id))
    if ids.size == 0:
        return np.empty((0,), dtype=np.int64)

    selected_groups = np.unique(group_id[ids])
    mask = np.isin(group_id, selected_groups)
    return np.flatnonzero(mask).astype(np.int64)


def select_group_tensor_mesh(
    tmesh: Any,
    group_id: int,
) -> Any:
    mask = tmesh.triangle["group_id"] == int(group_id)
    return tmesh.select_faces_by_mask(mask)


def conservative_group_cleanup_legacy(
    group_tmesh: Any,
    *,
    allow_non_manifold_edge_removal: bool = False,
) -> Any:
    o3d = _require_open3d()

    legacy = group_tmesh.to_legacy()
    legacy = legacy.remove_duplicated_vertices()
    legacy = legacy.remove_duplicated_triangles()
    legacy = legacy.remove_degenerate_triangles()
    legacy = legacy.remove_unreferenced_vertices()

    if allow_non_manifold_edge_removal:
        legacy = legacy.remove_non_manifold_edges()
        legacy = legacy.remove_unreferenced_vertices()

    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def reduce_group_with_boundary_weight(
    group_tmesh: Any,
    *,
    target_triangles: int,
    boundary_weight: float = 5.0,
) -> Any:
    o3d = _require_open3d()

    if int(target_triangles) <= 0:
        raise ValueError("target_triangles must be > 0")

    legacy = group_tmesh.to_legacy()
    current_faces = int(len(legacy.triangles))
    if current_faces == 0:
        return group_tmesh.clone()
    if int(target_triangles) >= current_faces:
        return group_tmesh.clone()

    legacy = legacy.remove_duplicated_vertices()
    legacy = legacy.remove_duplicated_triangles()
    legacy = legacy.remove_degenerate_triangles()
    legacy = legacy.remove_unreferenced_vertices()

    simplified = legacy.simplify_quadric_decimation(
        target_number_of_triangles=int(target_triangles),
        boundary_weight=float(boundary_weight),
    )

    simplified = simplified.remove_duplicated_vertices()
    simplified = simplified.remove_duplicated_triangles()
    simplified = simplified.remove_degenerate_triangles()
    simplified = simplified.remove_unreferenced_vertices()

    return o3d.t.geometry.TriangleMesh.from_legacy(simplified)


def merge_processed_groups_to_proxy(
    processed: list[ProcessedGroupResult],
    *,
    source_face_kind: str = "quad",
) -> QuadProxyMesh:
    if not processed:
        raise ValueError("No processed groups were provided.")

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []

    group_id_parts: list[np.ndarray] = []
    patch_id_parts: list[np.ndarray] = []
    parent_quad_id_parts: list[np.ndarray] = []
    is_group_boundary_parts: list[np.ndarray] = []

    vertex_offset = 0
    next_parent_face = 0

    for item in processed:
        verts = np.asarray(item.mesh.vertex.positions.numpy(), dtype=np.float64)
        faces = np.asarray(item.mesh.triangle.indices.numpy(), dtype=np.int64)

        if verts.ndim != 2:
            verts = np.empty((0, 3), dtype=np.float64)
        if faces.ndim != 2:
            faces = np.empty((0, 3), dtype=np.int64)

        all_vertices.append(verts)
        all_faces.append(faces + vertex_offset)

        tri_count = len(faces)
        group_id_parts.append(np.full((tri_count,), item.group_id, dtype=np.int64))
        patch_id_parts.append(np.full((tri_count,), item.group_id, dtype=np.int64))
        parent_quad_id_parts.append(
            np.arange(next_parent_face, next_parent_face + tri_count, dtype=np.int64)
        )
        is_group_boundary_parts.append(np.zeros((tri_count,), dtype=bool))

        vertex_offset += len(verts)
        next_parent_face += tri_count

    merged_vertices = np.vstack(all_vertices) if all_vertices else np.empty((0, 3), dtype=np.float64)
    merged_faces = np.vstack(all_faces) if all_faces else np.empty((0, 3), dtype=np.int64)

    merged_tm = trimesh.Trimesh(
        vertices=merged_vertices,
        faces=merged_faces,
        process=False,
    )

    try:
        merge_vertices = getattr(merged_tm, "merge_vertices", None)
        if callable(merge_vertices):
            merge_vertices()
    except Exception:
        pass

    for attr in ("remove_duplicate_faces", "remove_degenerate_faces"):
        fn = getattr(merged_tm, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    try:
        merged_tm.remove_unreferenced_vertices()
    except Exception:
        pass

    meta = QuadProxyMetadata(
        group_id=_concat_or_empty(group_id_parts, shape_tail=(), dtype=np.int64),
        patch_id=_concat_or_empty(patch_id_parts, shape_tail=(), dtype=np.int64),
        parent_quad_id=_concat_or_empty(parent_quad_id_parts, shape_tail=(), dtype=np.int64),
        is_group_boundary=_concat_or_empty(is_group_boundary_parts, shape_tail=(), dtype=bool),
        source_face_kind=source_face_kind,
    )
    return QuadProxyMesh(proxy_trimesh=merged_tm, metadata=meta)


def process_selected_groups_locally(
    proxy: QuadProxyMesh,
    selected_face_ids: np.ndarray,
    opts: QuadGroupProcessOptions,
) -> QuadProxyMesh:
    selected_ids = _sanitize_face_ids_for_count(
        selected_face_ids,
        len(proxy.metadata.group_id),
    )
    if selected_ids.size == 0:
        raise ValueError("No valid selected face ids were provided for group processing.")

    selected_group_ids = np.unique(proxy.metadata.group_id[selected_ids]).tolist()

    tmesh = trimesh_to_tensor_proxy(proxy)
    all_group_ids = np.unique(proxy.metadata.group_id)
    results: list[ProcessedGroupResult] = []

    for gid in all_group_ids.tolist():
        part = select_group_tensor_mesh(tmesh, gid)
        before = int(part.triangle.indices.shape[0])

        if gid in selected_group_ids:
            if opts.cleanup:
                part = conservative_group_cleanup_legacy(
                    part,
                    allow_non_manifold_edge_removal=opts.allow_non_manifold_edge_removal,
                )

            if opts.reduce and before > 0:
                target = max(1, int(round(before * float(opts.target_ratio))))
                if target < before:
                    part = reduce_group_with_boundary_weight(
                        part,
                        target_triangles=target,
                        boundary_weight=opts.boundary_weight,
                    )

        after = int(part.triangle.indices.shape[0])
        results.append(
            ProcessedGroupResult(
                group_id=int(gid),
                mesh=part,
                triangle_count_before=before,
                triangle_count_after=after,
            )
        )

    return merge_processed_groups_to_proxy(
        results,
        source_face_kind=proxy.metadata.source_face_kind,
    )


__all__ = [
    "DecodedGroupData",
    "QuadGroupProcessOptions",
    "QuadProxyMetadata",
    "QuadProxyMesh",
    "ProcessedGroupResult",
    "build_quadwild_proxy_from_mesh",
    "trimesh_to_tensor_proxy",
    "expand_face_selection_to_groups",
    "select_group_tensor_mesh",
    "conservative_group_cleanup_legacy",
    "reduce_group_with_boundary_weight",
    "merge_processed_groups_to_proxy",
    "process_selected_groups_locally",
]
