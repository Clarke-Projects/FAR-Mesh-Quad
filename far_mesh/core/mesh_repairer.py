from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import trimesh

from far_mesh.core.open3d_tensor_bridge import fill_holes_with_open3d_tensor


@dataclass(slots=True)
class MeshRepairStep:
    method: str
    executed_method: str
    elapsed_seconds: float
    backend_chain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "executed_method": self.executed_method,
            "elapsed_seconds": self.elapsed_seconds,
            "backend_chain": list(self.backend_chain),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class MeshRepairResult:
    mesh: trimesh.Trimesh
    method: str
    executed_method: str
    join_comp: bool
    fill_holes: bool
    elapsed_seconds: float
    backend_chain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    inspection_before: dict[str, Any] | None = None
    inspection_after: dict[str, Any] | None = None
    stats_before: dict[str, Any] | None = None
    stats_after: dict[str, Any] | None = None
    steps: list[MeshRepairStep] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "mesh": self.mesh,
            "method": self.method,
            "executed_method": self.executed_method,
            "join_comp": self.join_comp,
            "fill_holes": self.fill_holes,
            "elapsed_seconds": self.elapsed_seconds,
            "backend_chain": list(self.backend_chain),
            "notes": list(self.notes),
            "inspection_before": dict(self.inspection_before or {}),
            "inspection_after": dict(self.inspection_after or {}),
            "stats_before": dict(self.stats_before or {}),
            "stats_after": dict(self.stats_after or {}),
            "steps": [step.to_dict() for step in self.steps],
        }


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> Any:
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    return m


def o3d_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )


class MeshRepairer:
    """
    Feature-preserving mesh repair wrapper.

    Public API:
    - available_methods()
    - available_options()
    - inspect(mesh)
    - clean(mesh, ...)
    - clean_with_report(mesh, ...)

    Compatibility notes:
    - accepts collect_inspection for older mesh_processor versions
    - accepts repair_options / workflow_options for newer GUI-driven tuning
    - swallows unknown legacy kwargs to ease transition
    """

    _KNOWN_METHODS: tuple[str, ...] = (
        "trimesh",
        "open3d",
        "pymeshfix",
        "pymeshlab",
        "cad_safe_pymeshlab",
        "light_normalize",
        "cad_safe",
        "topology_cleanup",
        "scan_closing",
        "hybrid",
        "cad_workflow",
    )

    def __init__(self) -> None:
        self._open3d_module: Any | None = None
        self._pymeshlab_module: Any | None = None
        self._pymeshfix_module: Any | None = None
        self._import_attempted: dict[str, bool] = {
            "open3d": False,
            "pymeshlab": False,
            "pymeshfix": False,
        }

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------
    def available_methods(self) -> list[str]:
        return list(self._KNOWN_METHODS)

    def available_options(self) -> list[str]:
        return list(self._KNOWN_METHODS)

    def has_open3d(self) -> bool:
        return self._import_open3d() is not None

    def has_pymeshlab(self) -> bool:
        return self._import_pymeshlab() is not None

    def has_pymeshfix(self) -> bool:
        return self._import_pymeshfix() is not None

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def clean(
        self,
        mesh: Any,
        *,
        method: str = "cad_safe",
        join_comp: bool = True,
        fill_holes: bool = True,
        collect_inspection: bool = True,
        repair_options: dict[str, Any] | None = None,
        workflow_options: dict[str, Any] | None = None,
        **_legacy_kwargs: Any,
    ) -> trimesh.Trimesh:
        result = self.clean_with_report(
            mesh,
            method=method,
            join_comp=join_comp,
            fill_holes=fill_holes,
            collect_inspection=collect_inspection,
            repair_options=repair_options,
            workflow_options=workflow_options,
        )
        return result.mesh

    def clean_with_report(
        self,
        mesh: Any,
        *,
        method: str = "cad_safe",
        join_comp: bool = True,
        fill_holes: bool = True,
        collect_inspection: bool = True,
        repair_options: dict[str, Any] | None = None,
        workflow_options: dict[str, Any] | None = None,
        **_legacy_kwargs: Any,
    ) -> MeshRepairResult:
        source_mesh = self._ensure_trimesh(mesh)
        method = self._normalize_method(method)
        resolved_options = self._resolve_repair_options(
            method=method,
            repair_options=repair_options,
            workflow_options=workflow_options,
        )

        inspection_before = self.inspect(source_mesh) if collect_inspection else None
        stats_before = self._mesh_stats(source_mesh)

        t0 = perf_counter()
        repaired_mesh, executed_method, backend_chain, notes, steps = self._run_method(
            source_mesh,
            method=method,
            join_comp=join_comp,
            fill_holes=fill_holes,
            repair_options=resolved_options,
        )
        elapsed = perf_counter() - t0

        inspection_after = self.inspect(repaired_mesh) if collect_inspection else None
        stats_after = self._mesh_stats(repaired_mesh)

        return MeshRepairResult(
            mesh=repaired_mesh,
            method=method,
            executed_method=executed_method,
            join_comp=join_comp,
            fill_holes=fill_holes,
            elapsed_seconds=elapsed,
            backend_chain=backend_chain,
            notes=notes,
            inspection_before=inspection_before,
            inspection_after=inspection_after,
            stats_before=stats_before,
            stats_after=stats_after,
            steps=steps,
        )

    def inspect(self, mesh: Any) -> dict[str, Any]:
        tm = self._ensure_trimesh(mesh)
        boundary_edges, boundary_loops = self._boundary_metrics(tm)
        defects = self._compute_pymeshlab_defects(tm)

        inspection: dict[str, Any] = {
            "vertices": int(len(tm.vertices)),
            "faces": int(len(tm.faces)),
            "is_watertight": bool(getattr(tm, "is_watertight", False)),
            "boundary_edge_count": int(boundary_edges),
            "boundary_loop_count": int(boundary_loops),
            "connected_components": int(self._component_count(tm)),
            "pymeshlab_defects": defects if defects is not None else None,
        }

        if defects is None:
            inspection["pymeshlab_defects_error"] = "pymeshlab unavailable"

        inspection["recommended_workflow"] = self._recommend_workflow(inspection)
        return inspection

    @staticmethod
    def evaluate_open3d_tensor_fill_holes_policy(
        report: dict[str, Any],
        *,
        max_faces_added: int | None = None,
        max_vertices_added: int | None = 0,
        max_candidate_delta: int | None = None,
        require_dry_run: bool = True,
        require_candidate_reduction: bool = True,
    ) -> dict[str, Any]:
        """Evaluate whether an Open3D tensor fill_holes dry-run is safe to apply.

        This is policy-only. It does not mutate meshes, does not run Open3D, and
        does not authorize a repair commit by itself. It exists so future repair
        UI/commit code can require a dry-run report before applying whole-mesh
        tensor fill_holes.
        """

        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

        reasons: list[str] = []
        warnings: list[str] = []

        dry_run = bool(report.get("dry_run"))
        added_faces = int(report.get("added_faces") or 0)
        added_vertices = int(report.get("added_vertices") or 0)
        candidate_count_before = int(report.get("candidate_count_before") or 0)
        candidate_count_after = int(report.get("candidate_count_after") or 0)
        filled_candidate_delta = int(report.get("filled_candidate_delta") or 0)

        if require_dry_run and not dry_run:
            reasons.append("report is not marked as a dry run")

        if added_faces < 0:
            reasons.append(f"added_faces is negative: {added_faces}")
        if added_vertices < 0:
            reasons.append(f"added_vertices is negative: {added_vertices}")

        if max_faces_added is not None and added_faces > int(max_faces_added):
            reasons.append(
                f"added_faces {added_faces} exceeds max_faces_added {int(max_faces_added)}"
            )

        if max_vertices_added is not None and added_vertices > int(max_vertices_added):
            reasons.append(
                f"added_vertices {added_vertices} exceeds max_vertices_added {int(max_vertices_added)}"
            )

        if max_candidate_delta is not None and filled_candidate_delta > int(max_candidate_delta):
            reasons.append(
                "filled_candidate_delta "
                f"{filled_candidate_delta} exceeds max_candidate_delta {int(max_candidate_delta)}"
            )

        if require_candidate_reduction and filled_candidate_delta <= 0:
            reasons.append("dry run did not reduce the hole candidate count")

        if candidate_count_before <= 0:
            warnings.append("dry run started with no detected hole candidates")

        if candidate_count_after > candidate_count_before:
            reasons.append(
                "candidate_count_after "
                f"{candidate_count_after} is greater than candidate_count_before {candidate_count_before}"
            )

        if added_faces == 0 and filled_candidate_delta > 0:
            warnings.append("candidate count decreased but no faces were added")

        if added_faces > 0 and filled_candidate_delta <= 0:
            warnings.append("faces were added but hole candidate count did not decrease")

        allowed = not reasons

        return {
            "operation": "open3d_tensor_fill_holes_policy_evaluation",
            "allowed": allowed,
            "reasons": reasons,
            "warnings": warnings,
            "limits": {
                "max_faces_added": max_faces_added,
                "max_vertices_added": max_vertices_added,
                "max_candidate_delta": max_candidate_delta,
                "require_dry_run": require_dry_run,
                "require_candidate_reduction": require_candidate_reduction,
            },
            "summary": {
                "dry_run": dry_run,
                "added_faces": added_faces,
                "added_vertices": added_vertices,
                "candidate_count_before": candidate_count_before,
                "candidate_count_after": candidate_count_after,
                "filled_candidate_delta": filled_candidate_delta,
            },
        }

    def build_open3d_tensor_fill_holes_repair_mesh(
        self,
        mesh: Any,
        *,
        hole_size: float = 1_000_000.0,
        max_faces_added: int | None = None,
        max_vertices_added: int | None = 0,
        max_candidate_delta: int | None = None,
        require_candidate_reduction: bool = True,
    ) -> dict[str, Any]:
        """Build an Open3D tensor fill_holes repair mesh if policy allows it.

        This is the guarded core apply primitive for future repair integration.
        It intentionally does not mutate the source mesh and does not write any
        project/history state. Callers must explicitly decide whether/how to
        commit the returned mesh.
        """

        source_mesh = self._ensure_trimesh(mesh)
        source_vertices = np.asarray(source_mesh.vertices).copy()
        source_faces = np.asarray(source_mesh.faces).copy()

        dry_run_report = self.inspect_open3d_tensor_fill_holes(
            source_mesh,
            hole_size=hole_size,
        )
        policy_evaluation = self.evaluate_open3d_tensor_fill_holes_policy(
            dry_run_report,
            max_faces_added=max_faces_added,
            max_vertices_added=max_vertices_added,
            max_candidate_delta=max_candidate_delta,
            require_candidate_reduction=require_candidate_reduction,
        )

        if not bool(policy_evaluation.get("allowed")):
            reasons = policy_evaluation.get("reasons") or []
            reason_text = "; ".join(str(reason) for reason in reasons) or "policy blocked repair"
            raise ValueError(f"Open3D tensor fill_holes repair blocked: {reason_text}")

        try:
            repaired_mesh = fill_holes_with_open3d_tensor(
                source_mesh,
                hole_size=hole_size,
            )
        except Exception as exc:
            raise RuntimeError(f"Open3D tensor fill_holes repair failed: {exc}") from exc


        if not np.array_equal(np.asarray(source_mesh.faces), source_faces):
            raise RuntimeError("Open3D tensor fill_holes repair unexpectedly mutated source faces.")
        if not np.allclose(np.asarray(source_mesh.vertices), source_vertices):
            raise RuntimeError("Open3D tensor fill_holes repair unexpectedly mutated source vertices.")

        return {
            "operation": "open3d_tensor_fill_holes_repair_mesh",
            "mesh": repaired_mesh,
            "dry_run_report": dry_run_report,
            "policy_evaluation": policy_evaluation,
            "hole_size": float(hole_size),
            "notes": [
                "Guarded Open3D tensor fill_holes repair mesh built on a copy.",
                "Source mesh was not modified.",
                "Caller must explicitly commit returned mesh through project/history path.",
            ],
        }

    def inspect_open3d_tensor_fill_holes(
        self,
        mesh: Any,
        *,
        hole_size: float = 1_000_000.0,
    ) -> dict[str, Any]:
        """Dry-run Open3D tensor TriangleMesh.fill_holes on a mesh copy.

        This method is intentionally observational:
        - it does not mutate the source mesh
        - it does not return a replacement mesh for commit
        - it does not write project files
        - it reports before/after loop counts and candidate diagnostics

        It exists as the first repair-side safety layer before exposing
        Open3D tensor hole filling as an actual repair operation.
        """

        from far_mesh.core.selection_topology import (
            diagnose_hole_candidates,
            find_hole_candidates,
        )

        source_mesh = self._ensure_trimesh(mesh)
        source_vertices = np.asarray(source_mesh.vertices).copy()
        source_faces = np.asarray(source_mesh.faces).copy()

        before_inspection = self.inspect(source_mesh)
        before_stats = self._mesh_stats(source_mesh)
        before_candidates = tuple(find_hole_candidates(source_mesh))
        before_diagnostics = tuple(diagnose_hole_candidates(source_mesh, before_candidates))

        t0 = perf_counter()
        try:
            filled_mesh = fill_holes_with_open3d_tensor(
                source_mesh,
                hole_size=hole_size,
            )
        except Exception as exc:
            raise RuntimeError(f"Open3D tensor fill_holes dry-run failed: {exc}") from exc
        elapsed = perf_counter() - t0


        after_inspection = self.inspect(filled_mesh)
        after_stats = self._mesh_stats(filled_mesh)
        after_candidates = tuple(find_hole_candidates(filled_mesh))
        after_diagnostics = tuple(diagnose_hole_candidates(filled_mesh, after_candidates))

        # Assert the source mesh was not mutated by this dry-run.
        if not np.array_equal(np.asarray(source_mesh.faces), source_faces):
            raise RuntimeError("Open3D tensor fill_holes dry-run unexpectedly mutated source faces.")
        if not np.allclose(np.asarray(source_mesh.vertices), source_vertices):
            raise RuntimeError("Open3D tensor fill_holes dry-run unexpectedly mutated source vertices.")

        def _diagnostic_payload(candidate: Any, diagnostic: Any) -> dict[str, Any]:
            kind = getattr(diagnostic, "kind", None)
            return {
                "kind": str(getattr(kind, "value", kind or "unknown")),
                "confidence": float(getattr(diagnostic, "confidence", 0.0) or 0.0),
                "notes": list(getattr(diagnostic, "notes", ()) or ()),
                "boundary_vertices": int(len(getattr(candidate, "boundary_vertices", ()) or ())),
                "boundary_edges": int(len(getattr(candidate, "boundary_edges", ()) or ())),
                "perimeter": float(getattr(candidate, "perimeter", 0.0) or 0.0),
                "area_hint": (
                    None
                    if getattr(candidate, "area_hint", None) is None
                    else float(getattr(candidate, "area_hint"))
                ),
            }

        before_candidate_payloads = [
            _diagnostic_payload(candidate, diagnostic)
            for candidate, diagnostic in zip(before_candidates, before_diagnostics, strict=False)
        ]
        after_candidate_payloads = [
            _diagnostic_payload(candidate, diagnostic)
            for candidate, diagnostic in zip(after_candidates, after_diagnostics, strict=False)
        ]

        added_vertices = int(after_stats["vertices"]) - int(before_stats["vertices"])
        added_faces = int(after_stats["faces"]) - int(before_stats["faces"])
        filled_candidate_delta = max(0, len(before_candidates) - len(after_candidates))

        return {
            "operation": "open3d_tensor_fill_holes_dry_run",
            "dry_run": True,
            "method": "open3d_tensor_fill_holes",
            "hole_size": float(hole_size),
            "elapsed_seconds": elapsed,
            "backend_chain": ["open3d_tensor_fill_holes"],
            "stats_before": before_stats,
            "stats_after": after_stats,
            "inspection_before": before_inspection,
            "inspection_after": after_inspection,
            "candidate_count_before": len(before_candidates),
            "candidate_count_after": len(after_candidates),
            "filled_candidate_delta": filled_candidate_delta,
            "added_vertices": added_vertices,
            "added_faces": added_faces,
            "candidate_diagnostics_before": before_candidate_payloads,
            "candidate_diagnostics_after": after_candidate_payloads,
            "notes": [
                "Dry run only; source mesh was not modified.",
                "Open3D tensor TriangleMesh.fill_holes ran on a copy.",
                "This report is not a repair commit and does not return a replacement mesh.",
            ],
        }

    # ------------------------------------------------------------------
    # workflow dispatch
    # ------------------------------------------------------------------
    def _run_method(
        self,
        mesh: trimesh.Trimesh,
        *,
        method: str,
        join_comp: bool,
        fill_holes: bool,
        repair_options: dict[str, Any],
    ) -> tuple[trimesh.Trimesh, str, list[str], list[str], list[MeshRepairStep]]:
        notes: list[str] = []
        steps: list[MeshRepairStep] = []

        if method == "light_normalize":
            repaired, chain = self._workflow_light_normalize(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            return repaired, "light_normalize", chain, notes, steps

        if method in {"cad_safe", "cad_safe_pymeshlab", "cad_workflow"}:
            if self.has_pymeshlab():
                repaired, chain = self._workflow_cad_safe_pymeshlab(
                    mesh,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                    repair_options=repair_options,
                )
                return repaired, "cad_safe", chain, notes, steps

            notes.append("pymeshlab unavailable; fell back to light_normalize")
            repaired, chain = self._workflow_light_normalize(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            return repaired, "cad_safe", chain, notes, steps

        if method == "topology_cleanup":
            if self.has_pymeshlab():
                repaired, chain = self._workflow_topology_cleanup_pymeshlab(
                    mesh,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                    repair_options=repair_options,
                )
                return repaired, "topology_cleanup", chain, notes, steps

            notes.append("pymeshlab unavailable; fell back to hybrid")
            repaired, chain, hybrid_notes = self._workflow_hybrid(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
                repair_options=repair_options,
            )
            notes.extend(hybrid_notes)
            return repaired, "hybrid", chain, notes, steps

        if method == "pymeshlab":
            if self.has_pymeshlab():
                repaired, chain = self._workflow_general_pymeshlab(
                    mesh,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                    repair_options=repair_options,
                )
                return repaired, "pymeshlab", chain, notes, steps

            notes.append("pymeshlab unavailable; fell back to light_normalize")
            repaired, chain = self._workflow_light_normalize(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            return repaired, "light_normalize", chain, notes, steps

        if method == "scan_closing":
            if self.has_pymeshfix():
                repaired, chain = self._workflow_scan_closing_pymeshfix(
                    mesh,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                )
                return repaired, "scan_closing", chain, notes, steps

            notes.append("pymeshfix unavailable; fell back to hybrid")
            repaired, chain, hybrid_notes = self._workflow_hybrid(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
                repair_options=repair_options,
            )
            notes.extend(hybrid_notes)
            return repaired, "hybrid", chain, notes, steps

        if method == "hybrid":
            repaired, chain, hybrid_notes = self._workflow_hybrid(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
                repair_options=repair_options,
            )
            notes.extend(hybrid_notes)
            return repaired, "hybrid", chain, notes, steps

        if method == "open3d":
            repaired, chain = self._workflow_open3d(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            return repaired, "open3d", chain, notes, steps

        if method == "pymeshfix":
            if self.has_pymeshfix():
                repaired, chain = self._workflow_scan_closing_pymeshfix(
                    mesh,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                )
                return repaired, "pymeshfix", chain, notes, steps

            notes.append("pymeshfix unavailable; fell back to light_normalize")
            repaired, chain = self._workflow_light_normalize(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            return repaired, "light_normalize", chain, notes, steps

        repaired, chain = self._workflow_trimesh(
            mesh,
            join_comp=join_comp,
            fill_holes=fill_holes,
        )
        return repaired, "trimesh", chain, notes, steps

    # ------------------------------------------------------------------
    # workflows
    # ------------------------------------------------------------------
    def _workflow_trimesh(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "trimesh_finalize"]

    def _workflow_light_normalize(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        if self.has_open3d():
            working = self._open3d_cleanup(working)
            chain = ["trimesh_pre", "open3d", "trimesh_finalize"]
        else:
            chain = ["trimesh_pre", "trimesh_finalize"]
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, chain

    def _workflow_open3d(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
    ) -> tuple[trimesh.Trimesh, list[str]]:
        if not self.has_open3d():
            return self._workflow_light_normalize(mesh, join_comp=join_comp, fill_holes=fill_holes)
        working = self._trimesh_pre_cleanup(mesh)
        working = self._open3d_cleanup(working)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "open3d", "trimesh_finalize"]

    def _workflow_general_pymeshlab(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
        repair_options: dict[str, Any],
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        working = self._pymeshlab_cleanup_general(working, repair_options=repair_options)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "pymeshlab_general", "trimesh_finalize"]

    def _workflow_cad_safe_pymeshlab(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
        repair_options: dict[str, Any],
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        working = self._pymeshlab_cleanup_cad_safe(working, repair_options=repair_options)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "pymeshlab_cad_safe", "trimesh_finalize"]

    def _workflow_topology_cleanup_pymeshlab(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
        repair_options: dict[str, Any],
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        working = self._pymeshlab_cleanup_topology(working, repair_options=repair_options)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "pymeshlab_topology", "trimesh_finalize"]

    def _workflow_scan_closing_pymeshfix(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
    ) -> tuple[trimesh.Trimesh, list[str]]:
        working = self._trimesh_pre_cleanup(mesh)
        working = self._pymeshfix_cleanup(working, join_comp=join_comp)
        working = self._trimesh_finalize(working, join_comp=join_comp, fill_holes=fill_holes)
        return working, ["trimesh_pre", "pymeshfix", "trimesh_finalize"]

    def _workflow_hybrid(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
        repair_options: dict[str, Any],
    ) -> tuple[trimesh.Trimesh, list[str], list[str]]:
        inspection = self.inspect(mesh)
        notes: list[str] = []

        if inspection.get("recommended_workflow") == "light_normalize":
            repaired, chain = self._workflow_light_normalize(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            notes.append("hybrid selected light_normalize based on inspection")
            return repaired, chain, notes

        if self.has_pymeshlab():
            repaired, chain = self._workflow_cad_safe_pymeshlab(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
                repair_options=repair_options,
            )
            notes.append("hybrid selected cad_safe PyMeshLab cleanup")
            return repaired, chain, notes

        if self.has_pymeshfix():
            repaired, chain = self._workflow_scan_closing_pymeshfix(
                mesh,
                join_comp=join_comp,
                fill_holes=fill_holes,
            )
            notes.append("hybrid selected pymeshfix closing cleanup")
            return repaired, chain, notes

        repaired, chain = self._workflow_light_normalize(
            mesh,
            join_comp=join_comp,
            fill_holes=fill_holes,
        )
        notes.append("hybrid fell back to light_normalize")
        return repaired, chain, notes

    # ------------------------------------------------------------------
    # repair option resolution
    # ------------------------------------------------------------------
    def _resolve_repair_options(
        self,
        *,
        method: str,
        repair_options: dict[str, Any] | None,
        workflow_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if isinstance(workflow_options, dict):
            merged.update(workflow_options)
        if isinstance(repair_options, dict):
            merged.update(repair_options)

        tuning_profile = str(merged.get("tuning_profile") or "workflow_default")
        raw_pymeshlab = merged.get("pymeshlab") if isinstance(merged.get("pymeshlab"), dict) else {}

        defaults = self._default_pymeshlab_options_for_method(method)

        if tuning_profile == "workflow_default":
            final_pymeshlab = defaults
        else:
            final_pymeshlab = {
                "non_manifold_edge_method": str(
                    raw_pymeshlab.get("non_manifold_edge_method", defaults["non_manifold_edge_method"])
                ),
                "non_manifold_vertex_displacement": float(
                    raw_pymeshlab.get(
                        "non_manifold_vertex_displacement",
                        defaults["non_manifold_vertex_displacement"],
                    )
                ),
                "t_vertices_enabled": bool(
                    raw_pymeshlab.get("t_vertices_enabled", defaults["t_vertices_enabled"])
                ),
                "t_vertex_method": str(
                    raw_pymeshlab.get("t_vertex_method", defaults["t_vertex_method"])
                ),
                "t_vertex_threshold": float(
                    raw_pymeshlab.get("t_vertex_threshold", defaults["t_vertex_threshold"])
                ),
                "t_vertex_repeat": bool(
                    raw_pymeshlab.get("t_vertex_repeat", defaults["t_vertex_repeat"])
                ),
            }

        return {
            "tuning_profile": tuning_profile,
            "pymeshlab": final_pymeshlab,
        }

    def _default_pymeshlab_options_for_method(self, method: str) -> dict[str, Any]:
        if method in {"cad_safe", "cad_safe_pymeshlab", "cad_workflow"}:
            return {
                "non_manifold_edge_method": "split_vertices",
                "non_manifold_vertex_displacement": 0.0,
                "t_vertices_enabled": False,
                "t_vertex_method": "edge_flip",
                "t_vertex_threshold": 5.0,
                "t_vertex_repeat": False,
            }

        if method == "topology_cleanup":
            return {
                "non_manifold_edge_method": "split_vertices",
                "non_manifold_vertex_displacement": 0.0,
                "t_vertices_enabled": True,
                "t_vertex_method": "edge_flip",
                "t_vertex_threshold": 10.0,
                "t_vertex_repeat": False,
            }

        if method == "pymeshlab":
            return {
                "non_manifold_edge_method": "split_vertices",
                "non_manifold_vertex_displacement": 0.0,
                "t_vertices_enabled": True,
                "t_vertex_method": "edge_flip",
                "t_vertex_threshold": 10.0,
                "t_vertex_repeat": False,
            }

        return {
            "non_manifold_edge_method": "split_vertices",
            "non_manifold_vertex_displacement": 0.0,
            "t_vertices_enabled": False,
            "t_vertex_method": "edge_flip",
            "t_vertex_threshold": 5.0,
            "t_vertex_repeat": False,
        }

    # ------------------------------------------------------------------
    # trimesh cleanup
    # ------------------------------------------------------------------
    def _trimesh_pre_cleanup(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        working = self._ensure_trimesh(mesh)

        fn = getattr(working, "remove_infinite_values", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

        fn = getattr(working, "remove_unreferenced_vertices", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

        for attr in ("remove_duplicate_faces", "remove_degenerate_faces"):
            fn = getattr(working, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        fix_normals = getattr(working, "fix_normals", None)
        if callable(fix_normals):
            try:
                fix_normals()
            except Exception:
                pass

        return self._ensure_trimesh(working)

    def _trimesh_finalize(
        self,
        mesh: trimesh.Trimesh,
        *,
        join_comp: bool,
        fill_holes: bool,
    ) -> trimesh.Trimesh:
        working = self._ensure_trimesh(mesh)

        if join_comp:
            try:
                parts = list(working.split(only_watertight=False))
                if len(parts) > 1:
                    working = trimesh.util.concatenate(parts)
            except Exception:
                pass

        if fill_holes:
            fill_fn = getattr(working, "fill_holes", None)
            if callable(fill_fn):
                try:
                    fill_fn()
                except Exception:
                    pass

        fn = getattr(working, "remove_unreferenced_vertices", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

        fix_normals = getattr(working, "fix_normals", None)
        if callable(fix_normals):
            try:
                fix_normals()
            except Exception:
                pass

        return self._ensure_trimesh(working)

    # ------------------------------------------------------------------
    # open3d cleanup
    # ------------------------------------------------------------------
    def _open3d_cleanup(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if not self.has_open3d():
            return self._ensure_trimesh(mesh)

        o3d_mesh = trimesh_to_o3d(mesh)
        for attr in (
            "remove_duplicated_vertices",
            "remove_duplicated_triangles",
            "remove_degenerate_triangles",
            "remove_unreferenced_vertices",
            "remove_non_manifold_edges",
        ):
            fn = getattr(o3d_mesh, attr, None)
            if callable(fn):
                try:
                    result = fn()
                    if result is not None:
                        o3d_mesh = result
                except Exception:
                    pass

        return self._ensure_trimesh(o3d_to_trimesh(o3d_mesh))

    # ------------------------------------------------------------------
    # pymeshfix cleanup
    # ------------------------------------------------------------------
    def _pymeshfix_cleanup(self, mesh: trimesh.Trimesh, *, join_comp: bool) -> trimesh.Trimesh:
        pymeshfix = self._import_pymeshfix()
        if pymeshfix is None:
            return self._ensure_trimesh(mesh)

        working = self._ensure_trimesh(mesh)

        meshfix_cls = getattr(pymeshfix, "MeshFix", None)
        if meshfix_cls is None:
            return working

        fixer = meshfix_cls(
            np.asarray(working.vertices, dtype=np.float64),
            np.asarray(working.faces, dtype=np.int32),
        )

        repair_fn = getattr(fixer, "repair", None)
        if callable(repair_fn):
            try:
                repair_fn(verbose=False, joincomp=join_comp)
            except TypeError:
                try:
                    repair_fn(verbose=False)
                except Exception:
                    pass
            except Exception:
                pass

        vertices = np.asarray(getattr(fixer, "v", working.vertices))
        faces = np.asarray(getattr(fixer, "f", working.faces))
        return self._build_trimesh(vertices, faces)

    # ------------------------------------------------------------------
    # pymeshlab cleanup
    # ------------------------------------------------------------------
    def _pymeshlab_cleanup_general(
        self,
        mesh: trimesh.Trimesh,
        *,
        repair_options: dict[str, Any],
    ) -> trimesh.Trimesh:
        ms = self._meshset_from_trimesh(mesh)
        if ms is None:
            return self._ensure_trimesh(mesh)

        opts = repair_options.get("pymeshlab", {})
        self._apply_filter_sequence_common(ms)
        self._apply_non_manifold_edge_repair(ms, opts)
        self._apply_non_manifold_vertex_repair(ms, opts)
        self._apply_t_vertex_cleanup(ms, opts)
        return self._meshset_to_trimesh(ms)

    def _pymeshlab_cleanup_cad_safe(
        self,
        mesh: trimesh.Trimesh,
        *,
        repair_options: dict[str, Any],
    ) -> trimesh.Trimesh:
        ms = self._meshset_from_trimesh(mesh)
        if ms is None:
            return self._ensure_trimesh(mesh)

        opts = repair_options.get("pymeshlab", {})
        self._apply_filter_sequence_common(ms)
        self._apply_non_manifold_edge_repair(ms, opts)
        self._apply_non_manifold_vertex_repair(ms, opts)
        self._apply_t_vertex_cleanup(ms, opts)
        return self._meshset_to_trimesh(ms)

    def _pymeshlab_cleanup_topology(
        self,
        mesh: trimesh.Trimesh,
        *,
        repair_options: dict[str, Any],
    ) -> trimesh.Trimesh:
        ms = self._meshset_from_trimesh(mesh)
        if ms is None:
            return self._ensure_trimesh(mesh)

        opts = dict(repair_options.get("pymeshlab", {}))
        self._apply_filter_sequence_common(ms)
        self._apply_non_manifold_edge_repair(ms, opts)
        self._apply_non_manifold_vertex_repair(ms, opts)

        if not bool(opts.get("t_vertices_enabled", False)):
            opts["t_vertices_enabled"] = True
            opts.setdefault("t_vertex_method", "edge_flip")
            opts.setdefault("t_vertex_threshold", 10.0)
            opts.setdefault("t_vertex_repeat", False)

        self._apply_t_vertex_cleanup(ms, opts)
        return self._meshset_to_trimesh(ms)

    def _apply_filter_sequence_common(self, ms: Any) -> None:
        self._safe_apply_filter(ms, "meshing_remove_duplicate_vertices")
        self._safe_apply_filter(ms, "meshing_remove_duplicate_faces")
        self._safe_apply_filter(ms, "meshing_remove_null_faces")
        self._safe_apply_filter(ms, "meshing_remove_unreferenced_vertices")

    def _apply_non_manifold_edge_repair(self, ms: Any, opts: dict[str, Any]) -> None:
        mode = str(opts.get("non_manifold_edge_method", "split_vertices")).strip().lower()
        if mode == "remove_faces":
            self._safe_apply_filter(
                ms,
                "meshing_repair_non_manifold_edges",
                method="Remove Faces",
            )
            return

        self._safe_apply_filter(
            ms,
            "meshing_repair_non_manifold_edges",
            method="Split Vertices",
        )

    def _apply_non_manifold_vertex_repair(self, ms: Any, opts: dict[str, Any]) -> None:
        displacement = float(opts.get("non_manifold_vertex_displacement", 0.0))
        displacement = max(0.0, min(0.1, displacement))
        self._safe_apply_filter(
            ms,
            "meshing_repair_non_manifold_vertices",
            vertdispratio=displacement,
        )

    def _apply_t_vertex_cleanup(self, ms: Any, opts: dict[str, Any]) -> None:
        if not bool(opts.get("t_vertices_enabled", False)):
            return

        method = str(opts.get("t_vertex_method", "edge_flip")).strip().lower()
        filter_method = "Edge Collapse" if method == "edge_collapse" else "Edge Flip"
        threshold = float(opts.get("t_vertex_threshold", 5.0))
        repeat = bool(opts.get("t_vertex_repeat", False))

        self._safe_apply_filter(
            ms,
            "meshing_remove_t_vertices",
            method=filter_method,
            threshold=threshold,
            repeat=repeat,
        )

    # ------------------------------------------------------------------
    # defect inspection / recommendation
    # ------------------------------------------------------------------
    def _compute_pymeshlab_defects(self, mesh: trimesh.Trimesh) -> dict[str, Any] | None:
        if not self.has_pymeshlab():
            return None

        defects: dict[str, Any] = {
            "self_intersecting_faces": None,
            "non_manifold_edge_faces": None,
            "non_manifold_vertices": None,
            "boundary_edges_estimate": None,
            "boundary_loops_estimate": None,
        }

        defects["boundary_edges_estimate"], defects["boundary_loops_estimate"] = self._boundary_metrics(mesh)

        defects["self_intersecting_faces"] = self._pymeshlab_selected_face_count(
            mesh,
            "compute_selection_by_self_intersections_per_face",
        )
        defects["non_manifold_edge_faces"] = self._pymeshlab_selected_face_count(
            mesh,
            "compute_selection_by_non_manifold_edges_per_face",
        )
        defects["non_manifold_vertices"] = self._pymeshlab_selected_vertex_count(
            mesh,
            "compute_selection_by_non_manifold_per_vertex",
        )

        return defects

    def _recommend_workflow(self, inspection: dict[str, Any]) -> str:
        watertight = bool(inspection.get("is_watertight", False))
        boundary_edges = int(inspection.get("boundary_edge_count") or 0)
        components = int(inspection.get("connected_components") or 1)

        defects = inspection.get("pymeshlab_defects") if isinstance(inspection.get("pymeshlab_defects"), dict) else {}
        self_intersections = int(defects.get("self_intersecting_faces") or 0)
        non_manifold_edge_faces = int(defects.get("non_manifold_edge_faces") or 0)
        non_manifold_vertices = int(defects.get("non_manifold_vertices") or 0)

        if watertight and boundary_edges == 0 and components <= 1:
            if self_intersections == 0 and non_manifold_edge_faces == 0 and non_manifold_vertices == 0:
                return "light_normalize"

        if self_intersections > 0 or non_manifold_edge_faces > 0 or non_manifold_vertices > 0:
            return "cad_safe"

        if not watertight or boundary_edges > 0 or components > 1:
            return "scan_closing" if self.has_pymeshfix() else "hybrid"

        return "light_normalize"

    # ------------------------------------------------------------------
    # pymeshlab meshset helpers
    # ------------------------------------------------------------------
    def _meshset_from_trimesh(self, mesh: trimesh.Trimesh) -> Any | None:
        pymeshlab = self._import_pymeshlab()
        if pymeshlab is None:
            return None

        ms = pymeshlab.MeshSet()
        pm_mesh = pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm_mesh, "mesh")
        return ms

    def _meshset_to_trimesh(self, ms: Any) -> trimesh.Trimesh:
        current = ms.current_mesh()
        vertices = np.asarray(current.vertex_matrix(), dtype=np.float64)
        faces = np.asarray(current.face_matrix(), dtype=np.int64)
        return self._build_trimesh(vertices, faces)

    def _safe_apply_filter(self, ms: Any, filter_name: str, **kwargs: Any) -> bool:
        fn = getattr(ms, "apply_filter", None)
        if not callable(fn):
            return False
        try:
            fn(filter_name, **kwargs)
            return True
        except Exception:
            return False

    def _pymeshlab_selected_face_count(self, mesh: trimesh.Trimesh, filter_name: str) -> int | None:
        ms = self._meshset_from_trimesh(mesh)
        if ms is None:
            return None
        if not self._safe_apply_filter(ms, filter_name):
            return None
        current = ms.current_mesh()
        selected_fn = getattr(current, "selected_face_number", None)
        if callable(selected_fn):
            try:
                return int(selected_fn())
            except Exception:
                return None
        return None

    def _pymeshlab_selected_vertex_count(self, mesh: trimesh.Trimesh, filter_name: str) -> int | None:
        ms = self._meshset_from_trimesh(mesh)
        if ms is None:
            return None
        if not self._safe_apply_filter(ms, filter_name):
            return None
        current = ms.current_mesh()
        selected_fn = getattr(current, "selected_vertex_number", None)
        if callable(selected_fn):
            try:
                return int(selected_fn())
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # mesh math helpers
    # ------------------------------------------------------------------
    def _ensure_trimesh(self, mesh: Any) -> trimesh.Trimesh:
        if isinstance(mesh, trimesh.Trimesh):
            return trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
                faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
                process=False,
            )

        vertices = np.asarray(getattr(mesh, "vertices", None))
        faces = np.asarray(getattr(mesh, "faces", None))
        if vertices.ndim == 2 and vertices.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] >= 3:
            if faces.shape[1] != 3:
                faces = faces[:, :3]
            return self._build_trimesh(vertices, faces)

        raise TypeError(f"Unsupported mesh type for MeshRepairer: {type(mesh)!r}")

    def _build_trimesh(self, vertices: Any, faces: Any) -> trimesh.Trimesh:
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(faces, dtype=np.int64)
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("Expected triangle faces shaped (M, 3).")
        return trimesh.Trimesh(vertices=verts, faces=tris, process=False)

    def _mesh_stats(self, mesh: trimesh.Trimesh) -> dict[str, Any]:
        return {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(getattr(mesh, "is_watertight", False)),
        }

    def _component_count(self, mesh: trimesh.Trimesh) -> int:
        try:
            parts = list(mesh.split(only_watertight=False))
            return max(1, len(parts))
        except Exception:
            return 1

    def _boundary_metrics(self, mesh: trimesh.Trimesh) -> tuple[int, int]:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            return 0, 0

        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack((e01, e12, e20))
        edges = np.sort(edges, axis=1)

        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]

        if len(boundary_edges) == 0:
            return 0, 0

        vertex_adj: dict[int, set[int]] = {}
        for a_raw, b_raw in boundary_edges:
            a = int(a_raw)
            b = int(b_raw)
            vertex_adj.setdefault(a, set()).add(b)
            vertex_adj.setdefault(b, set()).add(a)

        visited: set[int] = set()
        components = 0
        for start in list(vertex_adj.keys()):
            if start in visited:
                continue
            components += 1
            stack = [start]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                stack.extend(n for n in vertex_adj.get(v, ()) if n not in visited)

        return int(len(boundary_edges)), int(components)

    def _normalize_method(self, method: str | None) -> str:
        value = str(method or "cad_safe").strip().lower()
        aliases = {
            "cad-safe": "cad_safe",
            "cad safe": "cad_safe",
            "cad-safe-pymeshlab": "cad_safe_pymeshlab",
            "light normalize": "light_normalize",
            "topology cleanup": "topology_cleanup",
            "scan closing": "scan_closing",
            "cad workflow": "cad_workflow",
        }
        return aliases.get(value, value)

    # ------------------------------------------------------------------
    # lazy imports
    # ------------------------------------------------------------------
    def _import_open3d(self) -> Any | None:
        if self._import_attempted["open3d"]:
            return self._open3d_module
        self._import_attempted["open3d"] = True
        try:
            import open3d as o3d

            self._open3d_module = o3d
        except Exception:
            self._open3d_module = None
        return self._open3d_module

    def _import_pymeshlab(self) -> Any | None:
        if self._import_attempted["pymeshlab"]:
            return self._pymeshlab_module
        self._import_attempted["pymeshlab"] = True
        try:
            import pymeshlab

            self._pymeshlab_module = pymeshlab
        except Exception:
            self._pymeshlab_module = None
        return self._pymeshlab_module

    def _import_pymeshfix(self) -> Any | None:
        if self._import_attempted["pymeshfix"]:
            return self._pymeshfix_module
        self._import_attempted["pymeshfix"] = True
        try:
            import pymeshfix

            self._pymeshfix_module = pymeshfix
        except Exception:
            self._pymeshfix_module = None
        return self._pymeshfix_module


__all__ = [
    "MeshRepairStep",
    "MeshRepairResult",
    "MeshRepairer",
    "trimesh_to_o3d",
    "o3d_to_trimesh",
]
