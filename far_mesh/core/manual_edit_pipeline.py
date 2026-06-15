from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import trimesh


SelectionMode = Literal["faces", "vertices"]
EditOperation = Literal[
    "cleanup",
    "reduce",
    "delete_faces",
    "delete_vertices",
    "clip_plane",
    "smooth_laplacian",
    "group_cleanup",
    "group_reduce",
]


@dataclass(slots=True)
class ManualSelection:
    mode: SelectionMode
    face_ids: np.ndarray | None = None
    vertex_ids: np.ndarray | None = None
    source_path: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManualEditRequest:
    operation: EditOperation
    selection: ManualSelection
    preview_only: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ManualEditPreview:
    operation: EditOperation
    preview_mesh: trimesh.Trimesh
    base_mesh: trimesh.Trimesh
    selection_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManualEditResult:
    operation: EditOperation
    mesh: trimesh.Trimesh
    before_faces: int
    after_faces: int
    before_vertices: int
    after_vertices: int
    notes: list[str] = field(default_factory=list)


def _require_open3d() -> Any:
    """
    Import Open3D only inside operations that actually need it.

    This avoids loading Open3D native libraries merely by importing
    manual_edit_pipeline.py or MeshProcessor. Open3D-backed work should be
    routed through PROCESS where possible, but lazy import also keeps the
    parent process cleaner during tests and normal app startup.
    """

    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required for this manual edit operation but could not be imported."
        ) from exc

    return o3d


def _sanitize_int_ids(ids: np.ndarray | None) -> np.ndarray:
    if ids is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    return np.unique(arr)


def _sanitize_face_ids(
    tmesh: Any,
    face_ids: np.ndarray | None,
) -> np.ndarray:
    ids = _sanitize_int_ids(face_ids)
    num_faces = int(tmesh.triangle.indices.shape[0])
    if num_faces <= 0 or ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    return ids[(ids >= 0) & (ids < num_faces)]


def _sanitize_vertex_ids(
    tmesh: Any,
    vertex_ids: np.ndarray | None,
) -> np.ndarray:
    ids = _sanitize_int_ids(vertex_ids)
    num_vertices = int(tmesh.vertex.positions.shape[0])
    if num_vertices <= 0 or ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    return ids[(ids >= 0) & (ids < num_vertices)]


def _mark_roi_only(notes: list[str], message: str) -> None:
    notes.append("__ROI_ONLY_PREVIEW__")
    notes.append(message)


def trimesh_to_tmesh(
    mesh: trimesh.Trimesh,
    *,
    triangle_attrs: dict[str, np.ndarray] | None = None,
    vertex_attrs: dict[str, np.ndarray] | None = None,
) -> Any:
    o3d = _require_open3d()

    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(
        np.asarray(mesh.vertices, dtype=np.float32),
        dtype=o3d.core.float32,
    )
    tmesh.triangle.indices = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.int64),
        dtype=o3d.core.int64,
    )

    for name, arr in (vertex_attrs or {}).items():
        tmesh.vertex[name] = o3d.core.Tensor(np.asarray(arr))
    for name, arr in (triangle_attrs or {}).items():
        tmesh.triangle[name] = o3d.core.Tensor(np.asarray(arr))
    return tmesh


def tmesh_to_trimesh(tmesh: Any) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=tmesh.vertex.positions.numpy(),
        faces=tmesh.triangle.indices.numpy(),
        process=False,
    )


def select_face_roi(
    tmesh: Any,
    face_ids: np.ndarray,
) -> Any:
    o3d = _require_open3d()

    ids = _sanitize_face_ids(tmesh, face_ids)
    num_faces = int(tmesh.triangle.indices.shape[0])

    if ids.size == 0 or num_faces == 0:
        return o3d.t.geometry.TriangleMesh()

    mask = np.zeros(num_faces, dtype=bool)
    mask[ids] = True
    return tmesh.select_faces_by_mask(o3d.core.Tensor(mask, dtype=o3d.core.bool))


def select_vertex_roi(
    tmesh: Any,
    vertex_ids: np.ndarray,
) -> Any:
    o3d = _require_open3d()

    ids = _sanitize_vertex_ids(tmesh, vertex_ids)
    if ids.size == 0:
        return o3d.t.geometry.TriangleMesh()

    vid = o3d.core.Tensor(ids, dtype=o3d.core.int64)
    return tmesh.select_by_index(vid, copy_attributes=True)


def local_cleanup_faces(
    roi: Any,
    *,
    allow_non_manifold_edge_removal: bool = False,
) -> Any:
    o3d = _require_open3d()

    legacy = roi.to_legacy()
    legacy = legacy.remove_duplicated_vertices()
    legacy = legacy.remove_duplicated_triangles()
    legacy = legacy.remove_degenerate_triangles()
    legacy = legacy.remove_unreferenced_vertices()

    if allow_non_manifold_edge_removal:
        legacy = legacy.remove_non_manifold_edges()
        legacy = legacy.remove_unreferenced_vertices()

    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def local_reduce_faces(
    roi: Any,
    *,
    target_triangles: int,
    boundary_weight: float = 5.0,
) -> Any:
    o3d = _require_open3d()

    if int(target_triangles) <= 0:
        raise ValueError("target_triangles must be > 0")

    legacy = roi.to_legacy()
    current_faces = int(len(legacy.triangles))
    if current_faces == 0:
        return roi.clone()
    if int(target_triangles) >= current_faces:
        return roi.clone()

    legacy = legacy.remove_duplicated_vertices()
    legacy = legacy.remove_duplicated_triangles()
    legacy = legacy.remove_degenerate_triangles()
    legacy = legacy.remove_unreferenced_vertices()

    reduced = legacy.simplify_quadric_decimation(
        target_number_of_triangles=int(target_triangles),
        boundary_weight=float(boundary_weight),
    )

    reduced = reduced.remove_duplicated_vertices()
    reduced = reduced.remove_duplicated_triangles()
    reduced = reduced.remove_degenerate_triangles()
    reduced = reduced.remove_unreferenced_vertices()
    return o3d.t.geometry.TriangleMesh.from_legacy(reduced)


def local_smooth_laplacian(
    roi: Any,
    *,
    number_of_iterations: int = 1,
    lambda_filter: float = 0.5,
) -> Any:
    o3d = _require_open3d()

    if int(number_of_iterations) <= 0:
        return roi.clone()

    legacy = roi.to_legacy()
    legacy = legacy.filter_smooth_laplacian(
        number_of_iterations=int(number_of_iterations),
        lambda_filter=float(lambda_filter),
    )
    legacy = legacy.remove_degenerate_triangles()
    legacy = legacy.remove_unreferenced_vertices()
    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def delete_selected_faces(
    tmesh: Any,
    face_ids: np.ndarray,
) -> Any:
    o3d = _require_open3d()

    ids = _sanitize_face_ids(tmesh, face_ids)
    legacy = tmesh.to_legacy()

    if ids.size == 0:
        return o3d.t.geometry.TriangleMesh.from_legacy(legacy)

    legacy.remove_triangles_by_index(ids.tolist())
    legacy = legacy.remove_unreferenced_vertices()
    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def delete_selected_vertices(
    tmesh: Any,
    vertex_ids: np.ndarray,
) -> Any:
    o3d = _require_open3d()

    ids = _sanitize_vertex_ids(tmesh, vertex_ids)
    legacy = tmesh.to_legacy()

    if ids.size == 0:
        return o3d.t.geometry.TriangleMesh.from_legacy(legacy)

    legacy.remove_vertices_by_index(ids.tolist())
    legacy = legacy.remove_unreferenced_vertices()
    legacy = legacy.remove_degenerate_triangles()
    return o3d.t.geometry.TriangleMesh.from_legacy(legacy)


def clip_roi_before_edit(
    tmesh: Any,
    *,
    point: np.ndarray,
    normal: np.ndarray,
) -> Any:
    o3d = _require_open3d()

    return tmesh.clip_plane(
        point=o3d.core.Tensor(np.asarray(point, dtype=np.float32)),
        normal=o3d.core.Tensor(np.asarray(normal, dtype=np.float32)),
    )


def _selection_summary(req: ManualEditRequest) -> dict[str, int | str]:
    selected_faces = 0 if req.selection.face_ids is None else int(len(req.selection.face_ids))
    selected_vertices = 0 if req.selection.vertex_ids is None else int(len(req.selection.vertex_ids))
    return {
        "mode": req.selection.mode,
        "selected_faces": selected_faces,
        "selected_vertices": selected_vertices,
    }


def build_manual_edit_preview(
    mesh: trimesh.Trimesh,
    req: ManualEditRequest,
    *,
    triangle_attrs: dict[str, np.ndarray] | None = None,
    vertex_attrs: dict[str, np.ndarray] | None = None,
) -> ManualEditPreview:
    tmesh = trimesh_to_tmesh(
        mesh,
        triangle_attrs=triangle_attrs,
        vertex_attrs=vertex_attrs,
    )

    if req.selection.mode == "faces":
        if req.selection.face_ids is None or len(req.selection.face_ids) == 0:
            raise ValueError("No face selection provided.")
        roi = select_face_roi(tmesh, req.selection.face_ids)
    else:
        if req.selection.vertex_ids is None or len(req.selection.vertex_ids) == 0:
            raise ValueError("No vertex selection provided.")
        roi = select_vertex_roi(tmesh, req.selection.vertex_ids)

    op = req.operation
    params = req.parameters
    notes: list[str] = []

    if op == "cleanup":
        roi = local_cleanup_faces(
            roi,
            allow_non_manifold_edge_removal=bool(
                params.get("allow_non_manifold_edge_removal", False)
            ),
        )
        _mark_roi_only(
            notes,
            "Preview shows only the processed ROI. Safe whole-mesh commit requires a merge strategy; "
            "use group_cleanup/group_reduce for patch-aware commits.",
        )

    elif op == "reduce":
        roi = local_reduce_faces(
            roi,
            target_triangles=int(params.get("target_triangles", 1000)),
            boundary_weight=float(params.get("boundary_weight", 5.0)),
        )
        _mark_roi_only(
            notes,
            "Preview shows only the processed ROI. Safe whole-mesh commit requires a merge strategy; "
            "use group_cleanup/group_reduce for patch-aware commits.",
        )

    elif op == "smooth_laplacian":
        roi = local_smooth_laplacian(
            roi,
            number_of_iterations=int(params.get("number_of_iterations", 1)),
            lambda_filter=float(params.get("lambda_filter", 0.5)),
        )
        _mark_roi_only(
            notes,
            "Preview shows only the smoothed ROI. Safe whole-mesh commit requires a merge strategy.",
        )

    elif op == "delete_faces":
        if req.selection.face_ids is None:
            raise ValueError("delete_faces requires face selection")
        roi = delete_selected_faces(tmesh, req.selection.face_ids)
        notes.append("Preview shows full mesh after deleting selected faces.")
        return ManualEditPreview(
            operation=op,
            preview_mesh=tmesh_to_trimesh(roi),
            base_mesh=mesh.copy(),
            selection_summary=_selection_summary(req),
            notes=notes,
        )

    elif op == "delete_vertices":
        if req.selection.vertex_ids is None:
            raise ValueError("delete_vertices requires vertex selection")
        roi = delete_selected_vertices(tmesh, req.selection.vertex_ids)
        notes.append("Preview shows full mesh after deleting selected vertices.")
        return ManualEditPreview(
            operation=op,
            preview_mesh=tmesh_to_trimesh(roi),
            base_mesh=mesh.copy(),
            selection_summary=_selection_summary(req),
            notes=notes,
        )

    elif op == "clip_plane":
        if "point" not in params or "normal" not in params:
            raise ValueError("clip_plane requires 'point' and 'normal' parameters")
        roi = clip_roi_before_edit(
            roi,
            point=np.asarray(params["point"], dtype=np.float32),
            normal=np.asarray(params["normal"], dtype=np.float32),
        )
        _mark_roi_only(
            notes,
            "Preview shows only the clipped ROI. Safe whole-mesh commit requires a merge strategy.",
        )

    elif op in {"group_cleanup", "group_reduce"}:
        raise NotImplementedError(
            "group_cleanup/group_reduce are handled by the quad-group adapter path, "
            "not by build_manual_edit_preview(). Route these through the patch-aware "
            "group processing pipeline."
        )

    else:
        raise ValueError(f"Unsupported manual edit operation: {op}")

    return ManualEditPreview(
        operation=op,
        preview_mesh=tmesh_to_trimesh(roi),
        base_mesh=mesh.copy(),
        selection_summary=_selection_summary(req),
        notes=notes,
    )


def commit_manual_edit_preview(
    preview: ManualEditPreview,
) -> ManualEditResult:
    if any(note == "__ROI_ONLY_PREVIEW__" for note in preview.notes):
        raise NotImplementedError(
            "Committing partial topology-changing ROI edits is not enabled in the initial patch. "
            "Use group_cleanup/group_reduce for patch-aware whole-mesh commits, or delete_faces/"
            "delete_vertices for destructive local commits."
        )

    out = preview.preview_mesh.copy()
    out.remove_unreferenced_vertices()

    return ManualEditResult(
        operation=preview.operation,
        mesh=out,
        before_faces=int(len(preview.base_mesh.faces)),
        after_faces=int(len(out.faces)),
        before_vertices=int(len(preview.base_mesh.vertices)),
        after_vertices=int(len(out.vertices)),
        notes=list(preview.notes),
    )
