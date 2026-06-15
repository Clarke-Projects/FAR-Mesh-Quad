from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(slots=True)
class MeshReductionResult:
    backend: str
    before_vertices: int
    before_faces: int
    after_vertices: int
    after_faces: int
    target_faces: int
    reduction_ratio: float
    elapsed_seconds: float
    note: str | None = None


class MeshReducer:
    """
    Dedicated mesh reduction / simplification stage.

    Current production backend:
    - open3d

    Planned later:
    - pymeshlab
    """

    def available_backends(self) -> dict[str, str]:
        """
        Return only actually usable reduction backends.

        PyMeshLab is intentionally not exposed here yet because the reducer
        is not implemented, even if the Python package happens to be installed.
        """
        backends: dict[str, str] = {}
        try:
            import open3d  # noqa: F401
            backends["open3d"] = "Open3D"
        except Exception:
            pass
        return backends

    def reduce(
        self,
        mesh: trimesh.Trimesh,
        backend: str = "open3d",
        target_faces: int = 50000,
        boundary_weight: float = 5.0,
        cleanup: bool = True,
    ) -> tuple[trimesh.Trimesh, MeshReductionResult]:
        if target_faces <= 0:
            raise ValueError("target_faces must be > 0")

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("MeshReducer expects a trimesh.Trimesh")

        before_vertices = int(len(mesh.vertices))
        before_faces = int(len(mesh.faces))

        if before_faces == 0:
            raise ValueError("Mesh is empty")

        if target_faces >= before_faces:
            result = MeshReductionResult(
                backend=backend,
                before_vertices=before_vertices,
                before_faces=before_faces,
                after_vertices=before_vertices,
                after_faces=before_faces,
                target_faces=target_faces,
                reduction_ratio=1.0,
                elapsed_seconds=0.0,
                note="Target face count is >= current face count. No reduction was applied.",
            )
            return mesh.copy(), result

        if backend != "open3d":
            available = ", ".join(self.available_backends().keys()) or "(none)"
            raise ValueError(
                f"Unknown or unavailable reduction backend: {backend}. "
                f"Available backends: {available}"
            )

        t0 = time.perf_counter()

        reduced = self._reduce_open3d(
            mesh=mesh,
            target_faces=target_faces,
            boundary_weight=boundary_weight,
            cleanup=cleanup,
        )

        elapsed = time.perf_counter() - t0
        after_vertices = int(len(reduced.vertices))
        after_faces = int(len(reduced.faces))
        ratio = after_faces / max(before_faces, 1)

        result = MeshReductionResult(
            backend=backend,
            before_vertices=before_vertices,
            before_faces=before_faces,
            after_vertices=after_vertices,
            after_faces=after_faces,
            target_faces=target_faces,
            reduction_ratio=ratio,
            elapsed_seconds=elapsed,
            note=None,
        )
        return reduced, result

    def _reduce_open3d(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int,
        boundary_weight: float,
        cleanup: bool,
    ) -> trimesh.Trimesh:
        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError("Open3D is not available for mesh reduction.") from exc

        working = mesh.copy()
        working.remove_unreferenced_vertices()

        o3_mesh = o3d.geometry.TriangleMesh()
        o3_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(working.vertices))
        o3_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(working.faces))

        o3_mesh.remove_duplicated_vertices()
        o3_mesh.remove_duplicated_triangles()
        o3_mesh.remove_degenerate_triangles()
        o3_mesh.remove_unreferenced_vertices()

        simplified = o3_mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_faces),
            boundary_weight=float(boundary_weight),
        )

        simplified.remove_duplicated_vertices()
        simplified.remove_duplicated_triangles()
        simplified.remove_degenerate_triangles()
        simplified.remove_unreferenced_vertices()

        vertices = np.asarray(simplified.vertices)
        faces = np.asarray(simplified.triangles)

        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError("Open3D reduction produced an empty mesh.")

        reduced = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        if cleanup:
            reduced.remove_unreferenced_vertices()
            try:
                reduced.remove_degenerate_faces()
            except Exception:
                pass
            try:
                reduced.remove_duplicate_faces()
            except Exception:
                pass

        return reduced

    def _reduce_pymeshlab(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int,
        cleanup: bool,
    ) -> trimesh.Trimesh:
        raise NotImplementedError(
            "PyMeshLab reduction is not implemented yet. "
            "The reduction stage is currently using Open3D only."
        )
