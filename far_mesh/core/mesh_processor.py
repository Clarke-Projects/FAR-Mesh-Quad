from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .manual_edit_pipeline import (
    ManualEditPreview,
    ManualEditRequest,
    ManualEditResult,
    build_manual_edit_preview,
    commit_manual_edit_preview,
)
from .mesh_reducer import MeshReducer
from .mesh_repairer import MeshRepairer, MeshRepairResult
from .quad_group_adapter import (
    QuadGroupProcessOptions,
    QuadProxyMesh,
    build_quadwild_proxy_from_mesh,
    process_selected_groups_locally,
)
from .quadwild_bimdf_runner import QuadWildBiMDFRunner
from .task_registry import register_core_tasks
from .mesh_history import MeshHistoryEntry
from .project_storage import (
    SCHEMA_VERSION_PROJECT_STORAGE,
    ProjectStorage,
    ProjectStorageError,
)
from .tool_preview_state import (
    HOLE_FILL_PREVIEW_MARKER,
    LOCAL_REGION_OPERATION_MARKER,
    REGION_KIND_BORE_REGION,
    REGION_KIND_HOLE_BOUNDARY,
    REGION_KIND_HOLE_PATCH,
    REGION_KIND_ZIPPER_BRIDGE,
    REGION_KIND_ZIPPER_CHAIN,
    SNAPSHOT_ROLE_AFTER,
    SNAPSHOT_ROLE_BASE,
    SNAPSHOT_ROLE_BEFORE,
    SNAPSHOT_ROLE_PATCH,
    SNAPSHOT_ROLE_PREVIEW,
    SNAPSHOT_ROLE_REMOVED_REGION,
    WHOLE_MESH_OPERATION_MARKER,
    MeshSnapshot,
    ToolPreviewState,
    ToolRegion,
    has_commit_blocking_marker,
)

from far_mesh.system.execution_manager import execute_task, get_lifecycle_manager
from far_mesh.system.execution_plan import ExecutionPlan
from far_mesh.system.resource_probe import SystemResources, probe_system_resources
from far_mesh.system.task_protocol import TaskKind, TaskRequest, TaskResult
from far_mesh.system.task_router import plan_task

try:
    from .remesher_wrapper import InstantMeshesRunner
except Exception:  # pragma: no cover
    try:
        from .instant_meshes_runner import InstantMeshesRunner  # type: ignore
    except Exception:  # pragma: no cover
        InstantMeshesRunner = None  # type: ignore



def _hole_fill_backend_for_method(method: object) -> str:
    method_key = str(method or "fan").strip().lower().replace("-", "_")

    if method_key in {
        "adaptive",
        "adaptive_surface",
        "adaptive_surface_fill",
        "adaptive_uvdelaunay_relaxed",
    }:
        return "surface_uvdelaunay_relaxed"

    if method_key in {
        "open3d",
        "curvature_sphere",
        "curvature_sphere_refined",
        "curvature_sphere_grid8",
        "curvature_sphere_uvgrid",
        "curvature_sphere_uvdelaunay",
        "curvature_sphere_uvdelaunay_relaxed",
        "surface_uvdelaunay_relaxed",
        "surface_uvdelaunay_sealed_relaxed",
        "surface_uvdelaunay_sealed_dense_relaxed",
    }:
        return method_key

    return "fan"

def _normalize_hole_fill_preview_backend_metadata(
    summary: dict[str, Any],
    *,
    requested_method: object,
) -> dict[str, Any]:
    normalized = dict(summary)
    method_key = str(normalized.get("method") or requested_method or "fan").strip().lower().replace("-", "_")
    normalized["method"] = method_key
    normalized["backend"] = _hole_fill_backend_for_method(method_key)
    return normalized





class MeshProcessorError(RuntimeError):
    """High-level processing error for FAR Mesh core operations."""


class _UnavailableRunner:
    def __init__(
        self,
        *,
        name: str,
        reason: str,
        repo_root: Path | None = None,
        default_stage1_config: str | None = None,
        default_stage2_config: str | None = None,
    ) -> None:
        self._name = name
        self._reason = reason
        self._repo_root = repo_root
        self._default_stage1_config = default_stage1_config
        self._default_stage2_config = default_stage2_config

    def is_available(self) -> bool:
        return False

    def get_unavailable_reason(self) -> str:
        return self._reason

    def get_available_stage1_presets(self) -> dict[str, str]:
        return {}

    def get_available_stage2_configs(self) -> dict[str, str]:
        return {}

    def debug_paths(self) -> dict[str, str]:
        repo_root = self._repo_root.resolve() if self._repo_root is not None else Path(".").resolve()
        return {
            "repo_root": str(repo_root),
            "quadwild": str((repo_root / "build" / "quadwild").resolve()),
            "quad_from_patches": str((repo_root / "build" / "quad_from_patches").resolve()),
            "lib_dir": str((repo_root / "build").resolve()),
            "default_stage1_config": str(
                (repo_root / (self._default_stage1_config or "config/prep_config/basic_setup.txt")).resolve()
            ),
            "default_stage2_config": str(
                (repo_root / (self._default_stage2_config or "config/main_config/flow_noalign_lemon.txt")).resolve()
            ),
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise MeshProcessorError(f"{self._name} is unavailable: {self._reason}")


class MeshProcessor:
    def __init__(self) -> None:
        self.mesh: trimesh.Trimesh | None = None
        self.filepath: str | None = None
        self.current_mesh_path: str | None = None
        self.last_output_path: str | None = None
        self.original_mesh: trimesh.Trimesh | None = None

        self.project_root = self._resolve_project_root()
        self.project_storage = self._make_project_storage()

        self.repairer = MeshRepairer()
        self.reducer = MeshReducer()

        self.instant_remesher = self._make_instant_remesher()

        self.quadwild_bimdf_root = self._resolve_quadwild_repo_root()
        self.quadwild_bimdf_runner = self._make_quadwild_runner()

        self.last_remesh_result: dict[str, Any] | None = None
        self.last_reduce_result: dict[str, Any] | None = None
        self.last_repair_result: dict[str, Any] | None = None
        self.last_manual_edit_result: dict[str, Any] | None = None
        self._last_tool_preview_state: ToolPreviewState | None = None
        self._last_mesh_history_entry: MeshHistoryEntry | None = None
        self._undo_stack: list[MeshHistoryEntry] = []
        self._redo_stack: list[MeshHistoryEntry] = []
        self._max_undo = 20
        self.last_quad_group_proxy: QuadProxyMesh | None = None

        self.last_system_resources: SystemResources | None = None
        self.last_execution_plan: ExecutionPlan | None = None
        self.last_task_result: TaskResult | None = None
        self._core_tasks_registered = False

    @staticmethod
    def _resolve_project_root() -> Path:
        """
        Resolve the Farmesh project root from this file location.

        Expected layout:
            <project_root>/
                far_mesh/
                    core/
                        mesh_processor.py
        """
        return Path(__file__).resolve().parents[2]

    def _make_project_storage(self) -> ProjectStorage:
        """
        Create the default unsaved session storage for this processor.

        This is intentionally a lightweight core storage root. It owns folders
        such as previews/ and history/, but it does not delete files, trim
        undo history, or perform disk-usage policy decisions.

        Tests or embedded callers can override the base cache directory with
        FAR_MESH_SESSION_CACHE_DIR.
        """

        base_cache_dir = os.environ.get("FAR_MESH_SESSION_CACHE_DIR")
        return ProjectStorage.create_unsaved_session(
            base_cache_dir=base_cache_dir,
            session_id=uuid.uuid4().hex[:12],
        )

    def _new_operation_id(self, prefix: str) -> str:
        safe_prefix = str(prefix or "op").strip().replace(" ", "_") or "op"
        return f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _resolve_quadwild_repo_root(self) -> Path:
        """
        Resolve the QuadWild-BiMDF repository root.

        Priority:
        1) FAR_MESH_QUADWILD_ROOT environment variable
        2) <project_root>/quadwild-bimdf
        """
        env_override = os.environ.get("FAR_MESH_QUADWILD_ROOT")
        if env_override:
            return Path(env_override).expanduser().resolve()
        return (self.project_root / "quadwild-bimdf").resolve()

    def _make_instant_remesher(self) -> Any:
        if InstantMeshesRunner is None:
            return _UnavailableRunner(name="Instant Meshes", reason="InstantMeshesRunner could not be imported")
        try:
            return InstantMeshesRunner()
        except Exception as exc:
            return _UnavailableRunner(name="Instant Meshes", reason=f"initialization failed: {exc}")

    def _make_quadwild_runner(self) -> Any:
        stage1_rel = "config/prep_config/basic_setup.txt"
        stage2_rel = "config/main_config/flow_noalign_lemon.txt"
        try:
            return QuadWildBiMDFRunner(
                repo_root=self.quadwild_bimdf_root,
                stage1_config_rel=stage1_rel,
                stage2_config_rel=stage2_rel,
                stop_after_step=2,
                output_index=123,
            )
        except Exception as exc:
            return _UnavailableRunner(
                name="QuadWild-BiMDF",
                reason=f"initialization failed: {exc}",
                repo_root=self.quadwild_bimdf_root,
                default_stage1_config=stage1_rel,
                default_stage2_config=stage2_rel,
            )

    def _ensure_core_tasks_registered(self) -> None:
        if not self._core_tasks_registered:
            register_core_tasks()
            self._core_tasks_registered = True

    def execute_task_request(
        self,
        request: TaskRequest,
        *,
        hints: dict[str, Any] | None = None,
    ) -> TaskResult:
        """
        Execute an explicit Phase 1.5 task request.

        MeshProcessor remains the mesh authority. The execution layer only
        decides and runs the task; it does not own application mesh state.
        """

        self._ensure_core_tasks_registered()

        resources = probe_system_resources()
        plan = plan_task(request, resources, hints=hints)
        result = execute_task(request, plan)

        self.last_system_resources = resources
        self.last_execution_plan = plan
        self.last_task_result = result

        return result

    def load_mesh(self, filepath: str) -> trimesh.Trimesh:
        loaded = trimesh.load(filepath, force="mesh")
        mesh = self._coerce_to_trimesh(loaded)
        if mesh.is_empty or len(mesh.faces) == 0:
            raise ValueError("File does not contain a valid non-empty mesh.")
        mesh.remove_unreferenced_vertices()

        self.mesh = mesh
        self.filepath = filepath
        self.current_mesh_path = filepath
        self.last_output_path = None
        self.clear_tool_preview_state()
        if self.original_mesh is None:
            self.original_mesh = mesh.copy()

        self.sync_project_state_metadata(reason="load_mesh")
        return self.mesh

    def set_mesh(
        self,
        mesh: Any,
        *,
        set_as_original: bool = False,
        source_path: str | Path | None = None,
    ) -> trimesh.Trimesh:
        tm = self._coerce_to_trimesh(mesh)
        tm.remove_unreferenced_vertices()
        self.mesh = tm
        self.clear_tool_preview_state()
        if source_path is not None:
            resolved = str(Path(source_path).expanduser().resolve())
            self.filepath = resolved
            self.current_mesh_path = resolved
        if set_as_original or self.original_mesh is None:
            self.original_mesh = tm.copy()

        self.sync_project_state_metadata(reason="set_mesh")
        return self.mesh

    def save_mesh(self, filepath: str) -> None:
        self._require_mesh()
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix not in {".obj", ".ply", ".stl"}:
            path = path.with_suffix(".obj")
        self.mesh.export(str(path))
        self.last_output_path = str(path.resolve())

    def repair(
        self,
        method: str = "hybrid",
        join_comp: bool = True,
        fill_holes: bool = True,
        collect_inspection: bool = True,
        repair_options: dict[str, Any] | None = None,
        workflow_options: dict[str, Any] | None = None,
        use_execution_layer: bool = True,
    ) -> dict[str, Any]:
        """
        Repair the current working mesh.

        By default this routes standard single-pass repair methods through the
        Phase 1.5 execution layer using TaskKind.MESH_REPAIR. The multi-step
        cad_workflow path remains direct for now because it is a MeshProcessor
        orchestration workflow, not a single pure-core handler task yet.

        Pass use_execution_layer=False to force the direct in-process repair
        path for debugging or fallback.
        """

        if use_execution_layer and method != "cad_workflow":
            return self.repair_planned(
                method=method,
                join_comp=join_comp,
                fill_holes=fill_holes,
                collect_inspection=collect_inspection,
                repair_options=repair_options,
                workflow_options=workflow_options,
            )

        return self._repair_direct(
            method=method,
            join_comp=join_comp,
            fill_holes=fill_holes,
            collect_inspection=collect_inspection,
            repair_options=repair_options,
            workflow_options=workflow_options,
        )

    def repair_planned(
        self,
        method: str = "hybrid",
        join_comp: bool = True,
        fill_holes: bool = True,
        collect_inspection: bool = True,
        repair_options: dict[str, Any] | None = None,
        workflow_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Repair the current mesh through the Phase 1.5 execution layer.

        MeshProcessor remains the mesh authority:
        - it copies the current mesh into the task payload,
        - the execution layer runs the registered MeshRepairer handler,
        - this method validates the returned mesh,
        - then this method commits the result to self.mesh.

        cad_workflow is intentionally excluded from this path for now because
        it is a multi-step MeshProcessor workflow. Use repair(...,
        use_execution_layer=False) or method="cad_workflow" for the direct
        orchestration path.
        """

        self._require_mesh()

        if method == "cad_workflow":
            return self._repair_direct(
                method=method,
                join_comp=join_comp,
                fill_holes=fill_holes,
                collect_inspection=collect_inspection,
                repair_options=repair_options,
                workflow_options=workflow_options,
            )

        before = self.get_mesh_stats()
        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        request = TaskRequest(
            kind=TaskKind.MESH_REPAIR,
            payload={
                "mesh": self.mesh.copy(),
                "method": method,
                "join_comp": bool(join_comp),
                "fill_holes": bool(fill_holes),
                "collect_inspection": bool(collect_inspection),
                "repair_options": repair_options or {},
                "workflow_options": workflow_options or {},
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            source_mesh_ref=self.current_mesh_path,
            description=f"Full mesh repair: {method}",
        )

        result = self.execute_task_request(request)

        if not result.ok:
            raise MeshProcessorError(result.error or "Mesh repair task failed.")

        result_payload = result.payload
        repaired_mesh = result_payload.get("mesh")
        if not isinstance(repaired_mesh, trimesh.Trimesh):
            raise MeshProcessorError("Mesh repair task did not return a trimesh.Trimesh mesh.")

        repaired_mesh = self._coerce_to_trimesh(repaired_mesh)
        repaired_mesh.remove_unreferenced_vertices()
        self.mesh = repaired_mesh

        after = self.get_mesh_stats()

        notes = result_payload.get("notes")
        if not isinstance(notes, list):
            notes = []
        else:
            notes = list(notes)

        if self.last_execution_plan is not None:
            notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        backend_chain = result_payload.get("backend_chain")
        if not isinstance(backend_chain, list):
            backend_chain = []

        stats_before = result_payload.get("stats_before")
        if not isinstance(stats_before, dict):
            stats_before = before

        stats_after = result_payload.get("stats_after")
        if not isinstance(stats_after, dict):
            stats_after = after

        payload: dict[str, Any] = {
            "backend": "repair",
            "method": method,
            "requested_method": result_payload.get("requested_method", method),
            "executed_method": result_payload.get("executed_method", method),
            "join_comp": bool(result_payload.get("join_comp", join_comp)),
            "fill_holes": bool(result_payload.get("fill_holes", fill_holes)),
            "collect_inspection": bool(result_payload.get("collect_inspection", collect_inspection)),
            "elapsed_seconds": float(result_payload.get("elapsed_seconds", 0.0)),
            "steps": [],
            "backend_chain": list(backend_chain),
            "notes": [str(note) for note in notes],
            "note": str(notes[0]) if notes else None,
            "inspection_before": result_payload.get("inspection_before"),
            "inspection_after": result_payload.get("inspection_after"),
            "stats_before": stats_before,
            "stats_after": stats_after,
            "before_vertices": stats_before.get("vertices", before.get("vertices")),
            "before_faces": stats_before.get("faces", before.get("faces")),
            "after_vertices": stats_after.get("vertices", after.get("vertices")),
            "after_faces": stats_after.get("faces", after.get("faces")),
            "mesh": self.mesh,
        }

        self.last_repair_result = payload
        return payload

    def _repair_direct(
        self,
        method: str = "hybrid",
        join_comp: bool = True,
        fill_holes: bool = True,
        collect_inspection: bool = True,
        repair_options: dict[str, Any] | None = None,
        workflow_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Original direct repair path.

        This is kept as an explicit fallback and for cad_workflow orchestration
        while Phase 1.5 rollout continues.
        """

        self._require_mesh()

        before = self.get_mesh_stats()
        t0 = time.perf_counter()

        steps: list[dict[str, Any]] = []
        notes: list[str] = []
        inspection_before: dict[str, Any] | None = None
        inspection_after: dict[str, Any] | None = None
        executed_method = method
        backend_chain: list[str] = []

        if method == "cad_workflow":
            working = self.mesh
            executed_method = "cad_workflow"

            first_step_before: dict[str, Any] | None = None
            final_step_after: dict[str, Any] | None = None

            for step_method in ("pymeshfix", "open3d", "trimesh"):
                report = self.repairer.clean_with_report(
                    working,
                    method=step_method,
                    join_comp=join_comp,
                    fill_holes=fill_holes,
                    collect_inspection=collect_inspection,
                    repair_options=repair_options,
                    workflow_options=workflow_options,
                )
                if not isinstance(report, MeshRepairResult):
                    raise MeshProcessorError("Repairer returned an unexpected report object.")

                working = report.mesh

                if first_step_before is None:
                    first_step_before = report.inspection_before
                final_step_after = report.inspection_after

                backend_chain.extend(report.backend_chain)
                notes.extend(report.notes)

                step_before_stats = report.stats_before or {}
                step_after_stats = report.stats_after or {}

                steps.append(
                    {
                        "method": step_method,
                        "executed_method": report.executed_method,
                        "backend_chain": list(report.backend_chain),
                        "elapsed_seconds": report.elapsed_seconds,
                        "input_vertices": step_before_stats.get("vertices"),
                        "input_faces": step_before_stats.get("faces"),
                        "output_vertices": step_after_stats.get("vertices"),
                        "output_faces": step_after_stats.get("faces"),
                        "notes": list(report.notes),
                    }
                )

            self.mesh = self._coerce_to_trimesh(working)
            inspection_before = first_step_before
            inspection_after = final_step_after

        else:
            report = self.repairer.clean_with_report(
                self.mesh,
                method=method,
                join_comp=join_comp,
                fill_holes=fill_holes,
                collect_inspection=collect_inspection,
                repair_options=repair_options,
                workflow_options=workflow_options,
            )
            if not isinstance(report, MeshRepairResult):
                raise MeshProcessorError("Repairer returned an unexpected report object.")

            self.mesh = self._coerce_to_trimesh(report.mesh)

            executed_method = report.executed_method
            backend_chain = list(report.backend_chain)
            notes = list(report.notes)
            inspection_before = report.inspection_before
            inspection_after = report.inspection_after

        self.mesh.remove_unreferenced_vertices()

        elapsed = time.perf_counter() - t0
        after = self.get_mesh_stats()

        payload: dict[str, Any] = {
            "backend": "repair",
            "method": method,
            "requested_method": method,
            "executed_method": executed_method,
            "join_comp": join_comp,
            "fill_holes": fill_holes,
            "collect_inspection": collect_inspection,
            "elapsed_seconds": elapsed,
            "steps": steps,
            "backend_chain": backend_chain,
            "notes": notes,
            "note": notes[0] if notes else None,
            "inspection_before": inspection_before,
            "inspection_after": inspection_after,
            "stats_before": before,
            "stats_after": after,
            "before_vertices": before.get("vertices"),
            "before_faces": before.get("faces"),
            "after_vertices": after.get("vertices"),
            "after_faces": after.get("faces"),
            "mesh": self.mesh,
        }

        self.last_repair_result = payload
        return payload

    def repair_open3d_tensor_fill_holes_guarded(
        self,
        *,
        hole_size: float = 1_000_000.0,
        max_faces_added: int | None = None,
        max_vertices_added: int | None = 0,
        max_candidate_delta: int | None = None,
        require_candidate_reduction: bool = True,
    ) -> dict[str, Any]:
        """Apply Open3D tensor fill_holes repair through guarded history path.

        This is the MeshProcessor-level authority operation for Open3D tensor
        repair. It only mutates self.mesh after:
        1. the repairer builds a candidate repaired mesh on a copy,
        2. the dry-run policy allows the operation,
        3. before/after snapshots and history_entry.json are written.
        """

        self._require_mesh()
        assert self.mesh is not None

        before_stats = self.get_mesh_stats()

        repair_payload = self.repairer.build_open3d_tensor_fill_holes_repair_mesh(
            self.mesh.copy(),
            hole_size=float(hole_size),
            max_faces_added=max_faces_added,
            max_vertices_added=max_vertices_added,
            max_candidate_delta=max_candidate_delta,
            require_candidate_reduction=require_candidate_reduction,
        )

        repaired_mesh = repair_payload.get("mesh")
        if not isinstance(repaired_mesh, trimesh.Trimesh):
            raise MeshProcessorError(
                "Open3D tensor fill_holes repair did not return a trimesh.Trimesh mesh."
            )

        repaired_mesh = self._coerce_to_trimesh(repaired_mesh)
        repaired_mesh.remove_unreferenced_vertices()

        if repaired_mesh.is_empty or len(repaired_mesh.faces) == 0:
            raise MeshProcessorError("Open3D tensor fill_holes repair returned an empty mesh.")

        dry_run_report = repair_payload.get("dry_run_report") or {}
        policy_evaluation = repair_payload.get("policy_evaluation") or {}

        notes = [
            "Open3D tensor fill_holes repair committed through guarded MeshProcessor path.",
            "Repair mesh was built on a copy before MeshProcessor mutation.",
            "Policy evaluation allowed this repair before commit.",
        ]
        for note in repair_payload.get("notes") or []:
            notes.append(str(note))

        entry = self._write_open3d_tensor_fill_holes_repair_history_entry(
            committed_mesh=repaired_mesh,
            repair_payload=repair_payload,
            notes=notes,
        )

        # Mutation starts only after history snapshots and JSON were written.
        before_vertices = int(len(self.mesh.vertices))
        before_faces = int(len(self.mesh.faces))

        self._push_mesh_history_entry(entry)
        self.mesh = repaired_mesh.copy()
        self.last_output_path = None
        self.clear_tool_preview_state()

        after_stats = self.get_mesh_stats()
        after_vertices = int(len(self.mesh.vertices))
        after_faces = int(len(self.mesh.faces))

        payload: dict[str, Any] = {
            "backend": "repair",
            "method": "open3d_tensor_fill_holes",
            "requested_method": "open3d_tensor_fill_holes",
            "executed_method": "open3d_tensor_fill_holes",
            "operation": "open3d_tensor_fill_holes_repair",
            "hole_size": float(hole_size),
            "max_faces_added": max_faces_added,
            "max_vertices_added": max_vertices_added,
            "max_candidate_delta": max_candidate_delta,
            "require_candidate_reduction": bool(require_candidate_reduction),
            "dry_run_report": dry_run_report,
            "policy_evaluation": policy_evaluation,
            "backend_chain": ["open3d_tensor_fill_holes"],
            "notes": list(notes),
            "note": notes[0],
            "stats_before": before_stats,
            "stats_after": after_stats,
            "before_vertices": before_vertices,
            "before_faces": before_faces,
            "after_vertices": after_vertices,
            "after_faces": after_faces,
            "history_entry": entry,
            "history_dir": entry.history_dir,
            "mesh": self.mesh,
        }

        self.last_repair_result = payload
        self.last_manual_edit_result = {
            "operation": "open3d_tensor_fill_holes_repair",
            "before_faces": before_faces,
            "after_faces": after_faces,
            "before_vertices": before_vertices,
            "after_vertices": after_vertices,
            "notes": list(notes),
            "mesh": self.mesh,
            "history_entry": entry,
            "history_dir": entry.history_dir,
        }

        self.sync_project_state_metadata(
            reason="repair_open3d_tensor_fill_holes_guarded"
        )

        return payload

    def reduce(
        self,
        backend: str = "open3d",
        target_faces: int = 50000,
        boundary_weight: float = 5.0,
        cleanup: bool = True,
        use_execution_layer: bool = True,
    ) -> dict[str, Any]:
        """
        Reduce the current working mesh.

        By default this routes through the Phase 1.5 execution layer using
        TaskKind.MESH_REDUCE. Pass use_execution_layer=False to run the old
        direct in-process reduction path for debugging or fallback.
        """

        if use_execution_layer:
            return self.reduce_planned(
                backend=backend,
                target_faces=target_faces,
                boundary_weight=boundary_weight,
                cleanup=cleanup,
            )

        return self._reduce_direct(
            backend=backend,
            target_faces=target_faces,
            boundary_weight=boundary_weight,
            cleanup=cleanup,
        )

    def reduce_planned(
        self,
        backend: str = "open3d",
        target_faces: int = 50000,
        boundary_weight: float = 5.0,
        cleanup: bool = True,
    ) -> dict[str, Any]:
        """
        Reduce the current mesh through the Phase 1.5 execution layer.

        MeshProcessor remains the mesh authority:
        - it copies the current mesh into the task payload,
        - the execution layer runs the registered MeshReducer handler,
        - this method validates the returned mesh,
        - then this method commits the result to self.mesh.
        """

        self._require_mesh()

        before = self.get_mesh_stats()
        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        request = TaskRequest(
            kind=TaskKind.MESH_REDUCE,
            payload={
                "mesh": self.mesh.copy(),
                "backend": backend,
                "target_faces": int(target_faces),
                "boundary_weight": float(boundary_weight),
                "cleanup": bool(cleanup),
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            source_mesh_ref=self.current_mesh_path,
            description="Full mesh reduction",
        )

        result = self.execute_task_request(request)

        if not result.ok:
            raise MeshProcessorError(result.error or "Mesh reduction task failed.")

        result_payload = result.payload
        reduced_mesh = result_payload.get("mesh")
        if not isinstance(reduced_mesh, trimesh.Trimesh):
            raise MeshProcessorError("Mesh reduction task did not return a trimesh.Trimesh mesh.")

        reduced_mesh = self._coerce_to_trimesh(reduced_mesh)
        reduced_mesh.remove_unreferenced_vertices()
        self.mesh = reduced_mesh

        after = self.get_mesh_stats()

        reduction = result_payload.get("reduction")
        if not isinstance(reduction, dict):
            reduction = {}

        notes = result_payload.get("notes")
        if not isinstance(notes, list):
            notes = []
        else:
            notes = list(notes)

        if self.last_execution_plan is not None:
            notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        payload: dict[str, Any] = {
            "backend": result_payload.get("backend", backend),
            "target_faces": int(result_payload.get("target_faces", target_faces)),
            "boundary_weight": float(result_payload.get("boundary_weight", boundary_weight)),
            "cleanup": bool(result_payload.get("cleanup", cleanup)),
            "before_vertices": int(reduction.get("before_vertices", before.get("vertices", 0))),
            "before_faces": int(reduction.get("before_faces", before.get("faces", 0))),
            "after_vertices": int(reduction.get("after_vertices", after.get("vertices", 0))),
            "after_faces": int(reduction.get("after_faces", after.get("faces", 0))),
            "reduction_ratio": float(reduction.get("reduction_ratio", 0.0)),
            "elapsed_seconds": float(reduction.get("elapsed_seconds", 0.0)),
            "note": reduction.get("note"),
            "notes": [str(note) for note in notes],
            "stats_before": before,
            "stats_after": after,
            "mesh": self.mesh,
        }
        self.last_reduce_result = payload
        return payload

    def _reduce_direct(
        self,
        backend: str = "open3d",
        target_faces: int = 50000,
        boundary_weight: float = 5.0,
        cleanup: bool = True,
    ) -> dict[str, Any]:
        """
        Original direct reduction path.

        This is kept as an explicit fallback and as a debugging path while
        Phase 1.5 rollout continues.
        """

        self._require_mesh()

        before = self.get_mesh_stats()
        reduced_mesh, reduction = self.reducer.reduce(
            mesh=self.mesh,
            backend=backend,
            target_faces=target_faces,
            boundary_weight=boundary_weight,
            cleanup=cleanup,
        )

        reduced_mesh = self._coerce_to_trimesh(reduced_mesh)
        reduced_mesh.remove_unreferenced_vertices()
        self.mesh = reduced_mesh

        after = self.get_mesh_stats()

        payload: dict[str, Any] = {
            "backend": backend,
            "target_faces": target_faces,
            "boundary_weight": boundary_weight,
            "cleanup": cleanup,
            "before_vertices": reduction.before_vertices,
            "before_faces": reduction.before_faces,
            "after_vertices": reduction.after_vertices,
            "after_faces": reduction.after_faces,
            "reduction_ratio": reduction.reduction_ratio,
            "elapsed_seconds": reduction.elapsed_seconds,
            "note": reduction.note,
            "notes": ["Execution path: direct MeshProcessor reduction."],
            "stats_before": before,
            "stats_after": after,
            "mesh": self.mesh,
        }
        self.last_reduce_result = payload
        return payload

    def remesh(
        self,
        backend: str = "instant_meshes",
        target_faces: int = 5000,
        crease_angle: float = 30.0,
        smooth_iterations: int = 2,
        deterministic: bool = False,
        quadwild_stage1_config_rel: str = "config/prep_config/basic_setup.txt",
        quadwild_stage2_config_rel: str = "config/main_config/flow_noalign_lemon.txt",
        quadwild_do_remesh: bool = True,
        quadwild_sharp_feature_threshold: float = 35.0,
        quadwild_alpha: float = 0.02,
        quadwild_scale_factor: float = 1.0,
        quadwild_use_original_input_file: bool = True,
        quadwild_pre_repair_workflow: bool = False,
        quadwild_cleanup_method: str = "cad_safe_pymeshlab",
        quadwild_fill_holes: bool = True,
        auto_reduce_after_quadwild: bool = False,
        auto_reduce_backend: str = "open3d",
        auto_reduce_target_faces: int = 50000,
        auto_reduce_boundary_weight: float = 5.0,
        auto_reduce_cleanup: bool = True,
        post_decimate: bool = False,
        decimate_target_faces: int = 5000,
        use_execution_layer: bool = True,
    ) -> dict[str, Any]:
        """
        Remesh the current working mesh.

        Phase 1.5G closure:
        - Instant Meshes routes through TaskKind.EXTERNAL_INSTANT_MESHES by default.
        - QuadWild-BiMDF routes through TaskKind.EXTERNAL_QUADWILD by default.
        - MeshProcessor still owns mesh validation and commit.
        - Direct backend methods remain available with use_execution_layer=False.
        """

        self._require_mesh()
        normalized_backend = str(backend or "").strip().lower().replace("-", "_")

        if normalized_backend == "instant_meshes":
            if use_execution_layer:
                result = self._remesh_instant_meshes_planned(
                    target_faces=target_faces,
                    crease_angle=crease_angle,
                    smooth_iterations=smooth_iterations,
                    deterministic=deterministic,
                )
            else:
                result = self._remesh_instant_meshes(
                    target_faces=target_faces,
                    crease_angle=crease_angle,
                    smooth_iterations=smooth_iterations,
                    deterministic=deterministic,
                )
            self.last_remesh_result = result
            return result

        if normalized_backend in {"quadwild_bimdf", "quadwild"}:
            if use_execution_layer:
                result = self._remesh_quadwild_bimdf_planned(
                    quadwild_stage1_config_rel=quadwild_stage1_config_rel,
                    quadwild_stage2_config_rel=quadwild_stage2_config_rel,
                    quadwild_do_remesh=quadwild_do_remesh,
                    quadwild_sharp_feature_threshold=quadwild_sharp_feature_threshold,
                    quadwild_alpha=quadwild_alpha,
                    quadwild_scale_factor=quadwild_scale_factor,
                    quadwild_use_original_input_file=quadwild_use_original_input_file,
                    quadwild_pre_repair_workflow=quadwild_pre_repair_workflow,
                    quadwild_cleanup_method=quadwild_cleanup_method,
                    quadwild_fill_holes=quadwild_fill_holes,
                    auto_reduce_after_quadwild=auto_reduce_after_quadwild,
                    auto_reduce_backend=auto_reduce_backend,
                    auto_reduce_target_faces=auto_reduce_target_faces,
                    auto_reduce_boundary_weight=auto_reduce_boundary_weight,
                    auto_reduce_cleanup=auto_reduce_cleanup,
                    post_decimate=post_decimate,
                    decimate_target_faces=decimate_target_faces,
                )
            else:
                result = self._remesh_quadwild_bimdf(
                    quadwild_stage1_config_rel=quadwild_stage1_config_rel,
                    quadwild_stage2_config_rel=quadwild_stage2_config_rel,
                    quadwild_do_remesh=quadwild_do_remesh,
                    quadwild_sharp_feature_threshold=quadwild_sharp_feature_threshold,
                    quadwild_alpha=quadwild_alpha,
                    quadwild_scale_factor=quadwild_scale_factor,
                    quadwild_use_original_input_file=quadwild_use_original_input_file,
                    quadwild_pre_repair_workflow=quadwild_pre_repair_workflow,
                    quadwild_cleanup_method=quadwild_cleanup_method,
                    quadwild_fill_holes=quadwild_fill_holes,
                    auto_reduce_after_quadwild=auto_reduce_after_quadwild,
                    auto_reduce_backend=auto_reduce_backend,
                    auto_reduce_target_faces=auto_reduce_target_faces,
                    auto_reduce_boundary_weight=auto_reduce_boundary_weight,
                    auto_reduce_cleanup=auto_reduce_cleanup,
                    post_decimate=post_decimate,
                    decimate_target_faces=decimate_target_faces,
                )
            self.last_remesh_result = result
            return result

        raise ValueError(f"Unknown remesh backend: {backend}")

    def _remesh_instant_meshes_planned(
        self,
        target_faces: int,
        crease_angle: float,
        smooth_iterations: int,
        deterministic: bool,
    ) -> dict[str, Any]:
        """
        Run Instant Meshes through Phase 1.5 EXTERNAL_INSTANT_MESHES routing.

        The external task is path-based. MeshProcessor exports the current mesh,
        asks the execution layer to run the external backend, then loads and
        validates the returned output mesh before committing it to self.mesh.
        """

        self._require_mesh()

        if hasattr(self.instant_remesher, "is_available") and not self.instant_remesher.is_available():
            raise MeshProcessorError(self.instant_remesher.get_unavailable_reason())

        t0 = time.perf_counter()
        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        executable_path: str | None = None
        for attr_name in (
            "executable_path",
            "executable",
            "binary_path",
            "instant_meshes_path",
            "exe_path",
        ):
            value = getattr(self.instant_remesher, attr_name, None)
            if value:
                executable_path = str(value)
                break

        with tempfile.TemporaryDirectory(prefix="far_mesh_instant_external_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.obj"
            output_path = tmp_path / "output.obj"
            self.mesh.export(input_path)

            payload: dict[str, Any] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "target_faces": int(target_faces),
                "target_face_count": int(target_faces),
                "crease_angle": float(crease_angle),
                "smooth_iterations": int(smooth_iterations),
                "deterministic": bool(deterministic),
                "face_count": face_count,
                "vertex_count": vertex_count,
                "options": {
                    "target_faces": int(target_faces),
                    "target_face_count": int(target_faces),
                    "crease_angle": float(crease_angle),
                    "smooth_iterations": int(smooth_iterations),
                    "deterministic": bool(deterministic),
                },
            }
            if executable_path:
                payload["executable_path"] = executable_path

            request = TaskRequest(
                kind=TaskKind.EXTERNAL_INSTANT_MESHES,
                payload=payload,
                hints={
                    "face_count": face_count,
                    "vertex_count": vertex_count,
                },
                source_mesh_ref=self.current_mesh_path,
                description="External remesh: Instant Meshes",
            )

            result = self.execute_task_request(request)
            if not result.ok:
                raise MeshProcessorError(result.error or "Instant Meshes external task failed.")

            result_payload = result.payload
            task_output_path = Path(str(result_payload.get("output_path") or output_path)).expanduser().resolve()
            if not task_output_path.exists():
                raise FileNotFoundError(f"Instant Meshes output was not created: {task_output_path}")

            loaded = trimesh.load(task_output_path, force="mesh")
            new_mesh = self._coerce_to_trimesh(loaded)
            if new_mesh.is_empty or len(new_mesh.faces) == 0:
                raise ValueError("Instant Meshes produced an empty output mesh.")

            new_mesh.remove_unreferenced_vertices()

            stable_copy_dir = Path(tempfile.mkdtemp(prefix="far_mesh_instant_result_"))
            stable_output_path = stable_copy_dir / task_output_path.name
            shutil.copy2(task_output_path, stable_output_path)

        self.mesh = new_mesh
        self.last_output_path = str(stable_output_path)
        elapsed = time.perf_counter() - t0

        notes = [
            "Instant Meshes completed through Phase 1.5 EXTERNAL_INSTANT_MESHES routing.",
        ]
        if self.last_execution_plan is not None:
            notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        runner_result = result_payload.get("runner_result")
        return {
            "backend": "instant_meshes",
            "output_path": str(stable_output_path),
            "final_stage": "instant_meshes",
            "generated_files": {"instant_output": str(stable_output_path)},
            "elapsed_seconds": elapsed,
            "stats": self.get_mesh_stats(),
            "runner_result": runner_result,
            "task_payload": result_payload,
            "notes": notes,
            "mesh": self.mesh,
        }

    def _remesh_quadwild_bimdf_planned(
        self,
        quadwild_stage1_config_rel: str,
        quadwild_stage2_config_rel: str,
        quadwild_do_remesh: bool,
        quadwild_sharp_feature_threshold: float,
        quadwild_alpha: float,
        quadwild_scale_factor: float,
        quadwild_use_original_input_file: bool,
        quadwild_pre_repair_workflow: bool,
        quadwild_cleanup_method: str,
        quadwild_fill_holes: bool,
        auto_reduce_after_quadwild: bool,
        auto_reduce_backend: str,
        auto_reduce_target_faces: int,
        auto_reduce_boundary_weight: float,
        auto_reduce_cleanup: bool,
        post_decimate: bool,
        decimate_target_faces: int,
    ) -> dict[str, Any]:
        """
        Run QuadWild-BiMDF through Phase 1.5 EXTERNAL_QUADWILD routing.

        MeshProcessor still owns the preparation workflow, result validation,
        optional post-reduction, stable result copying, and self.mesh commit.
        The execution layer owns routing and task execution. The external
        runner owns only the external subprocess calls.
        """

        self._require_mesh()
        pipeline_t0 = time.perf_counter()

        if not self.quadwild_bimdf_runner.is_available():
            paths = self.quadwild_bimdf_runner.debug_paths()
            raise FileNotFoundError(
                "QuadWild-BiMDF backend is not available.\n"
                f"quadwild repo root: {paths['repo_root']}\n"
                f"quadwild binary: {paths['quadwild']}\n"
                f"quad_from_patches binary: {paths['quad_from_patches']}\n"
                f"lib dir: {paths['lib_dir']}\n"
                f"default stage1 config: {paths['default_stage1_config']}\n"
                f"default stage2 config: {paths['default_stage2_config']}"
            )

        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))
        used_original_file = False
        workflow_steps: list[str] = []
        pre_repair_reports: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="far_mesh_quadwild_prepare_") as prep_tmpdir:
            source_name = "input.obj"
            if self.filepath:
                original_name = Path(self.filepath).name
                if original_name:
                    source_name = original_name

            if (
                quadwild_use_original_input_file
                and not quadwild_pre_repair_workflow
                and self.filepath is not None
            ):
                original = Path(self.filepath).expanduser().resolve()
                if original.exists() and original.is_file() and original.suffix.lower() in {".obj", ".ply"}:
                    source_mesh_path = original
                    used_original_file = True
                else:
                    source_mesh_path = Path(prep_tmpdir) / (Path(source_name).stem + ".obj")
                    self.mesh.export(source_mesh_path)
                    workflow_steps.append(
                        "Original file could not be used directly; exported current mesh to OBJ for QuadWild-BiMDF."
                    )
            else:
                working_mesh = self.mesh.copy()

                if quadwild_pre_repair_workflow:
                    try:
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="pymeshfix",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "pymeshfix",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 1: repaired mesh with pymeshfix.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 1 note: {note}")
                    except Exception as exc:
                        workflow_steps.append(
                            f"Pre-workflow step 1: pymeshfix failed ({exc}); falling back to hybrid repair."
                        )
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="hybrid",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "hybrid",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 1 fallback: repaired mesh with hybrid.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 1 fallback note: {note}")

                    try:
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method=quadwild_cleanup_method,
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": quadwild_cleanup_method,
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append(
                            f"Pre-workflow step 2: cleanup pass with {quadwild_cleanup_method}."
                        )
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 2 note: {note}")
                    except Exception as exc:
                        workflow_steps.append(
                            f"Pre-workflow step 2: cleanup with {quadwild_cleanup_method} failed ({exc}); "
                            "falling back to trimesh cleanup."
                        )
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="trimesh",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "trimesh",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 2 fallback: cleanup pass with trimesh.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 2 fallback note: {note}")

                source_mesh_path = Path(prep_tmpdir) / (Path(source_name).stem + ".obj")
                self._coerce_to_trimesh(working_mesh).export(source_mesh_path)
                used_original_file = False

            with tempfile.TemporaryDirectory(prefix="far_mesh_quadwild_bimdf_external_") as run_tmpdir:
                output_dir = Path(run_tmpdir) / "quadwild_bimdf_run"

                request = TaskRequest(
                    kind=TaskKind.EXTERNAL_QUADWILD,
                    payload={
                        "input_path": str(source_mesh_path),
                        "output_dir": str(output_dir),
                        "root_dir": str(self.quadwild_bimdf_root),
                        "runner_kwargs": {
                            "repo_root": str(self.quadwild_bimdf_root),
                            "stage1_config_rel": quadwild_stage1_config_rel,
                            "stage2_config_rel": quadwild_stage2_config_rel,
                            "stop_after_step": 2,
                            "output_index": 123,
                        },
                        "options": {
                            "timeout_stage1": None,
                            "timeout_stage2": None,
                            "overwrite": True,
                            "stage1_config_rel": quadwild_stage1_config_rel,
                            "stage2_config_rel": quadwild_stage2_config_rel,
                            "stage1_overrides": {
                                "do_remesh": quadwild_do_remesh,
                                "sharp_feature_thr": quadwild_sharp_feature_threshold,
                                "alpha": quadwild_alpha,
                                "scaleFact": quadwild_scale_factor,
                            },
                        },
                        "face_count": face_count,
                        "vertex_count": vertex_count,
                    },
                    hints={
                        "face_count": face_count,
                        "vertex_count": vertex_count,
                    },
                    source_mesh_ref=str(source_mesh_path),
                    description="External remesh: QuadWild-BiMDF",
                )

                result = self.execute_task_request(request)
                if not result.ok:
                    raise MeshProcessorError(result.error or "QuadWild-BiMDF external task failed.")

                result_payload = result.payload
                runner_result = result_payload.get("runner_result")
                if not isinstance(runner_result, dict):
                    raise MeshProcessorError("QuadWild-BiMDF task did not return runner_result metadata.")

                final_mesh_path_raw = runner_result.get("final_mesh_path")
                if not final_mesh_path_raw:
                    generated = runner_result.get("generated_files")
                    if isinstance(generated, dict):
                        for key in (
                            "final_mesh",
                            "final_output",
                            "quad_mesh",
                            "quad_output",
                            "output",
                            "result",
                        ):
                            if generated.get(key):
                                final_mesh_path_raw = generated[key]
                                break

                if not final_mesh_path_raw:
                    raise RuntimeError("QuadWild-BiMDF completed without a final quadrangulation mesh.")

                final_mesh_path = Path(str(final_mesh_path_raw)).expanduser().resolve()
                if not final_mesh_path.exists():
                    raise FileNotFoundError(f"QuadWild-BiMDF final mesh does not exist: {final_mesh_path}")

                loaded = trimesh.load(final_mesh_path, force="mesh")
                new_mesh = self._coerce_to_trimesh(loaded)

                if new_mesh.is_empty or len(new_mesh.faces) == 0:
                    raise ValueError("QuadWild-BiMDF produced an empty output mesh.")

                quadwild_output_faces = int(len(new_mesh.faces))
                quadwild_output_vertices = int(len(new_mesh.vertices))

                if post_decimate and not auto_reduce_after_quadwild:
                    auto_reduce_after_quadwild = True
                    auto_reduce_backend = "open3d"
                    auto_reduce_target_faces = decimate_target_faces
                    auto_reduce_boundary_weight = 5.0
                    auto_reduce_cleanup = True

                auto_reduce_applied = False
                auto_reduce_payload: dict[str, Any] | None = None
                auto_reduce_note: str | None = None

                if auto_reduce_after_quadwild and auto_reduce_target_faces > 0:
                    reduced_mesh, reduction = self.reducer.reduce(
                        mesh=new_mesh,
                        backend=auto_reduce_backend,
                        target_faces=auto_reduce_target_faces,
                        boundary_weight=auto_reduce_boundary_weight,
                        cleanup=auto_reduce_cleanup,
                    )
                    new_mesh = self._coerce_to_trimesh(reduced_mesh)
                    auto_reduce_applied = True
                    auto_reduce_note = (
                        "Auto reduction was applied after QuadWild-BiMDF using the dedicated reduction stage."
                    )
                    workflow_steps.append(
                        f"Auto reduction: {reduction.before_faces} -> "
                        f"{reduction.after_faces} faces using {auto_reduce_backend} "
                        f"in {reduction.elapsed_seconds:.2f}s."
                    )
                    auto_reduce_payload = {
                        "backend": auto_reduce_backend,
                        "target_faces": auto_reduce_target_faces,
                        "boundary_weight": auto_reduce_boundary_weight,
                        "cleanup": auto_reduce_cleanup,
                        "before_vertices": reduction.before_vertices,
                        "before_faces": reduction.before_faces,
                        "after_vertices": reduction.after_vertices,
                        "after_faces": reduction.after_faces,
                        "reduction_ratio": reduction.reduction_ratio,
                        "elapsed_seconds": reduction.elapsed_seconds,
                        "note": reduction.note,
                    }

                new_mesh.remove_unreferenced_vertices()
                self.mesh = new_mesh

                stable_copy_dir = Path(tempfile.mkdtemp(prefix="far_mesh_quadwild_bimdf_result_"))
                copied_outputs: dict[str, str] = {}

                generated_files = runner_result.get("generated_files")
                if not isinstance(generated_files, dict):
                    generated_files = {}

                for key, value in generated_files.items():
                    src = Path(str(value)).expanduser()
                    if src.exists() and src.is_file():
                        dst = stable_copy_dir / src.name
                        shutil.copy2(src, dst)
                        copied_outputs[str(key)] = str(dst)

                if auto_reduce_applied:
                    final_copy = stable_copy_dir / (final_mesh_path.stem + "_reduced.obj")
                    new_mesh.export(final_copy)
                    copied_outputs["reduced_output"] = str(final_copy)
                    final_stage = "reduced_after_quadwild"
                else:
                    final_copy = stable_copy_dir / final_mesh_path.name
                    if final_mesh_path.exists():
                        shutil.copy2(final_mesh_path, final_copy)
                    final_stage = str(runner_result.get("final_stage") or "quadwild_bimdf")

                pipeline_elapsed = time.perf_counter() - pipeline_t0
                self.last_output_path = str(final_copy)

                notes = [
                    "QuadWild-BiMDF completed through Phase 1.5 EXTERNAL_QUADWILD routing.",
                ]
                if self.last_execution_plan is not None:
                    notes.append(
                        f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                        f"{self.last_execution_plan.reason}"
                    )

                return {
                    "backend": "quadwild_bimdf",
                    "output_path": str(final_copy),
                    "final_stage": final_stage,
                    "generated_files": copied_outputs,
                    "working_directory": runner_result.get("working_directory"),
                    "stage1_command": runner_result.get("stage1_command"),
                    "stage2_command": runner_result.get("stage2_command"),
                    "stage1_stdout": runner_result.get("stage1_stdout"),
                    "stage1_stderr": runner_result.get("stage1_stderr"),
                    "stage2_stdout": runner_result.get("stage2_stdout"),
                    "stage2_stderr": runner_result.get("stage2_stderr"),
                    "stage1_returncode": runner_result.get("stage1_returncode"),
                    "stage2_returncode": runner_result.get("stage2_returncode"),
                    "stage1_elapsed_seconds": runner_result.get("stage1_elapsed_seconds"),
                    "stage2_elapsed_seconds": runner_result.get("stage2_elapsed_seconds"),
                    "total_elapsed_seconds": runner_result.get("total_elapsed_seconds"),
                    "pipeline_total_elapsed_seconds": pipeline_elapsed,
                    "stage1_config_requested": runner_result.get("stage1_config_requested"),
                    "stage1_config_used": runner_result.get("stage1_config_used"),
                    "stage2_config_used": runner_result.get("stage2_config_used"),
                    "stage1_overrides_used": runner_result.get("stage1_overrides_used"),
                    "stage1_fallback_used": runner_result.get("stage1_fallback_used"),
                    "stage1_fallback_reason": runner_result.get("stage1_fallback_reason"),
                    "used_original_input_file": used_original_file,
                    "source_mesh_path": str(source_mesh_path),
                    "workflow_steps": workflow_steps,
                    "pre_repair_reports": pre_repair_reports,
                    "quadwild_output_vertices": quadwild_output_vertices,
                    "quadwild_output_faces": quadwild_output_faces,
                    "auto_reduce_applied": auto_reduce_applied,
                    "auto_reduce_payload": auto_reduce_payload,
                    "auto_reduce_note": auto_reduce_note,
                    "stats": self.get_mesh_stats(),
                    "runner_result": runner_result,
                    "task_payload": result_payload,
                    "notes": notes,
                    "mesh": self.mesh,
                }

    def current_project_storage(self) -> ProjectStorage:
        """Return the active project/session storage object."""

        return self.project_storage

    def set_project_storage(self, storage: ProjectStorage) -> None:
        """
        Replace the active project/session storage.

        This does not move files or mutate meshes. It only changes where future
        preview/history operation folders are created.
        """

        if not isinstance(storage, ProjectStorage):
            raise TypeError("storage must be a ProjectStorage instance")
        storage.ensure_layout()
        self.project_storage = storage

    def project_storage_root(self) -> Path:
        """Return the active project/session root path."""

        return self.project_storage.root

    def project_disk_usage(self):
        """Return read-only disk usage for the active project/session root."""

        from far_mesh.system.resource_probe import probe_project_disk_usage

        return probe_project_disk_usage(self.project_storage.root)

    def _project_relative_or_string(self, path: str | Path | None) -> str | None:
        if path is None:
            return None

        path_obj = Path(path).expanduser()
        try:
            return self.project_storage.relative_path(path_obj)
        except Exception:
            return str(path_obj)

    def _history_entry_json_path(self, entry: MeshHistoryEntry | None) -> Path | None:
        if entry is None:
            return None
        history_dir = Path(entry.history_dir).expanduser()
        return history_dir / "history_entry.json"

    def _history_stack_metadata(self, stack: list[MeshHistoryEntry]) -> list[str]:
        values: list[str] = []
        for entry in stack:
            path = self._history_entry_json_path(entry)
            rel = self._project_relative_or_string(path)
            if rel is not None:
                values.append(rel)
        return values

    def _build_project_state_extra(
        self,
        *,
        current_snapshot: MeshSnapshot | None,
        reason: str,
    ) -> dict[str, Any]:
        latest_entry = self._last_mesh_history_entry
        latest_history_json = self._history_entry_json_path(latest_entry)

        last_operation = None
        if isinstance(self.last_manual_edit_result, dict):
            last_operation = self.last_manual_edit_result.get("operation")

        if self.mesh is not None:
            current_vertices = int(len(self.mesh.vertices))
            current_faces = int(len(self.mesh.faces))
        else:
            current_vertices = 0
            current_faces = 0

        return {
            "mesh_loaded": self.mesh is not None,
            "current_mesh_snapshot": self._project_relative_or_string(
                None if current_snapshot is None else current_snapshot.path
            ),
            "current_mesh_vertices": current_vertices,
            "current_mesh_faces": current_faces,
            "latest_history_entry": self._project_relative_or_string(latest_history_json),
            "latest_history_dir": self._project_relative_or_string(
                None if latest_entry is None else latest_entry.history_dir
            ),
            "latest_history_operation_id": None if latest_entry is None else latest_entry.operation_id,
            "latest_history_operation": None if latest_entry is None else latest_entry.operation,
            "undo_stack": self._history_stack_metadata(self._undo_stack),
            "redo_stack": self._history_stack_metadata(self._redo_stack),
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "last_operation": last_operation,
            "sync_reason": str(reason),
        }

    def sync_project_state_metadata(
        self,
        *,
        reason: str = "sync",
        write_current_mesh_snapshot: bool = True,
    ) -> dict[str, Any]:
        current_snapshot: MeshSnapshot | None = None

        if write_current_mesh_snapshot and self.mesh is not None:
            current_snapshot = MeshSnapshot.capture(
                self.mesh.copy(),
                self.project_storage.snapshots_dir,
                role="current",
                name="current_mesh",
                metadata={
                    "reason": str(reason),
                    "latest_history_operation_id": (
                        None
                        if self._last_mesh_history_entry is None
                        else self._last_mesh_history_entry.operation_id
                    ),
                },
            )

        extra = self._build_project_state_extra(
            current_snapshot=current_snapshot,
            reason=reason,
        )
        self.project_storage.write_metadata(extra=extra)
        return self.project_storage.read_metadata()

    def _resolve_project_metadata_path(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None

        raw = Path(path).expanduser()
        return self.project_storage.resolve_path(raw)

    def load_history_entry_from_project_path(
        self,
        path: str | Path,
    ) -> MeshHistoryEntry:
        resolved = self._resolve_project_metadata_path(path)
        if resolved is None:
            raise ValueError("No history entry path was provided")
        if not resolved.exists():
            raise FileNotFoundError(f"Project history entry does not exist: {resolved}")

        return MeshHistoryEntry.read_json(
            resolved,
            base_dir=self.project_storage.root,
        )

    def _load_history_entries_from_metadata_paths(
        self,
        paths: Any,
        *,
        strict: bool,
        skipped: list[str],
    ) -> list[MeshHistoryEntry]:
        if paths is None:
            return []
        if not isinstance(paths, list):
            if strict:
                raise ValueError("History stack metadata must be a list")
            return []

        entries: list[MeshHistoryEntry] = []

        for raw_path in paths:
            path_text = str(raw_path)
            try:
                entries.append(self.load_history_entry_from_project_path(path_text))
            except Exception:
                if strict:
                    raise
                skipped.append(path_text)

        return entries


    def _current_history_restore_summary(self) -> dict[str, Any]:
        return {
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "latest_history_entry": None
            if self._last_mesh_history_entry is None
            else self._last_mesh_history_entry.operation_id,
            "skipped_history_entries": [],
        }

    def _restore_project_state_schema_error_summary(self, error: Exception) -> dict[str, Any]:
        history_summary = self._current_history_restore_summary()
        return {
            "restored_current_mesh": False,
            "current_mesh_snapshot": None,
            "skipped_current_mesh_snapshot": None,
            "current_mesh_restore_error": None,
            "undo_count": int(history_summary.get("undo_count", 0)),
            "redo_count": int(history_summary.get("redo_count", 0)),
            "latest_history_entry": history_summary.get("latest_history_entry"),
            "skipped_history_entries": list(history_summary.get("skipped_history_entries", [])),
            "metadata_schema_supported": False,
            "metadata_schema_error": str(error),
        }


    def _load_project_mesh_snapshot_for_restore(self, snapshot_path: str | Path) -> trimesh.Trimesh:
        """Load a project mesh snapshot with defensive fallbacks.

        Project open must be robust: saved snapshots are the authority for the
        current mesh, and restore should not silently fail because one trimesh
        loader variant is unhappy with a valid PLY/OBJ snapshot.  Keep this in
        MeshProcessor so ProjectActions and the viewport remain display-only.
        """

        path = Path(snapshot_path).expanduser()
        errors: list[str] = []

        attempts = (
            ("trimesh.load(force='mesh')", lambda: trimesh.load(path, force="mesh")),
            (
                "trimesh.load(force='mesh', process=False, maintain_order=True)",
                lambda: trimesh.load(path, force="mesh", process=False, maintain_order=True),
            ),
            (
                "trimesh.load_mesh(process=False, maintain_order=True)",
                lambda: trimesh.load_mesh(path, process=False, maintain_order=True),
            ),
            ("trimesh.load_mesh()", lambda: trimesh.load_mesh(path)),
        )

        for label, loader in attempts:
            try:
                loaded = loader()
                mesh = self._coerce_to_trimesh(loaded)
                mesh = mesh.copy()
                try:
                    mesh.remove_unreferenced_vertices()
                except Exception:
                    pass
                if getattr(mesh, "is_empty", False) or len(mesh.faces) <= 0 or len(mesh.vertices) <= 0:
                    raise ValueError(
                        f"snapshot loaded as empty mesh: faces={len(mesh.faces)} vertices={len(mesh.vertices)}"
                    )
                return mesh
            except Exception as exc:
                errors.append(f"{label}: {exc!r}")

        details = "; ".join(errors) if errors else "no loader attempts were executed"
        raise ValueError(f"Could not restore project mesh snapshot {path}: {details}")

    @staticmethod
    def _validate_project_metadata_schema_for_restore(
        metadata: object,
        *,
        source: str = "project metadata",
    ) -> int:
        """
        Validate project/session metadata before restore touches mesh/history state.

        ProjectStorage validates metadata read from disk, but restore also accepts
        caller-supplied metadata dictionaries for tests and GUI adapters. Validate
        those payloads here as well so unsupported or malformed schema versions
        cannot partially restore mesh/history state.
        """

        if not isinstance(metadata, dict):
            raise ProjectStorageError(
                f"Invalid ProjectStorage metadata in {source}: expected dict, "
                f"got {type(metadata).__name__}."
            )

        if "schema_version" not in metadata:
            return SCHEMA_VERSION_PROJECT_STORAGE

        raw_version = metadata.get("schema_version")
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise ProjectStorageError(
                f"Invalid ProjectStorage schema_version in {source}: expected integer, "
                f"got {type(raw_version).__name__}."
            )

        if raw_version < 1:
            raise ProjectStorageError(
                f"ProjectStorage schema version {raw_version} in {source} is older than the "
                "minimum supported version 1."
            )

        if raw_version > SCHEMA_VERSION_PROJECT_STORAGE:
            raise ProjectStorageError(
                f"ProjectStorage schema version {raw_version} in {source} is newer than supported "
                f"version {SCHEMA_VERSION_PROJECT_STORAGE}."
            )

        return int(raw_version)

    def restore_history_stacks_from_metadata(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        metadata_source = str(self.project_storage.metadata_path)
        if metadata is None:
            try:
                metadata = self.project_storage.read_metadata(required=False)
            except Exception as exc:
                if strict:
                    raise
                summary = self._current_history_restore_summary()
                summary["metadata_schema_supported"] = False
                summary["metadata_schema_error"] = str(exc)
                return summary
        else:
            metadata_source = "supplied restore metadata"

        try:
            self._validate_project_metadata_schema_for_restore(
                metadata,
                source=metadata_source,
            )
        except Exception as exc:
            if strict:
                raise
            summary = self._current_history_restore_summary()
            summary["metadata_schema_supported"] = False
            summary["metadata_schema_error"] = str(exc)
            return summary

        extra = metadata.get("extra") if isinstance(metadata, dict) else {}
        if not isinstance(extra, dict):
            extra = {}

        skipped: list[str] = []

        undo_entries = self._load_history_entries_from_metadata_paths(
            extra.get("undo_stack", []),
            strict=strict,
            skipped=skipped,
        )
        redo_entries = self._load_history_entries_from_metadata_paths(
            extra.get("redo_stack", []),
            strict=strict,
            skipped=skipped,
        )

        latest_entry: MeshHistoryEntry | None = None
        latest_path = extra.get("latest_history_entry")

        if latest_path:
            latest_text = str(latest_path)
            all_entries = undo_entries + redo_entries
            for entry in all_entries:
                entry_json = self._history_entry_json_path(entry)
                if self._project_relative_or_string(entry_json) == latest_text:
                    latest_entry = entry
                    break

            if latest_entry is None:
                try:
                    latest_entry = self.load_history_entry_from_project_path(latest_text)
                except Exception:
                    if strict:
                        raise
                    if latest_text not in skipped:
                        skipped.append(latest_text)

        self._undo_stack = undo_entries
        self._redo_stack = redo_entries
        self._last_mesh_history_entry = latest_entry or (undo_entries[-1] if undo_entries else None)
        self.clear_tool_preview_state()

        return {
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "latest_history_entry": None
            if self._last_mesh_history_entry is None
            else self._last_mesh_history_entry.operation_id,
            "skipped_history_entries": skipped,
        }

    def restore_project_state_from_metadata(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        restore_current_mesh: bool = True,
        restore_history: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        metadata_source = str(self.project_storage.metadata_path)
        if metadata is None:
            try:
                metadata = self.project_storage.read_metadata(required=False)
            except Exception as exc:
                if strict:
                    raise
                return self._restore_project_state_schema_error_summary(exc)
        else:
            metadata_source = "supplied restore metadata"

        try:
            self._validate_project_metadata_schema_for_restore(
                metadata,
                source=metadata_source,
            )
        except Exception as exc:
            if strict:
                raise
            return self._restore_project_state_schema_error_summary(exc)

        extra = metadata.get("extra") if isinstance(metadata, dict) else {}
        if not isinstance(extra, dict):
            extra = {}

        restored_mesh: trimesh.Trimesh | None = None
        restored_current_mesh = False
        skipped_current_mesh_snapshot: str | None = None
        current_mesh_restore_error: str | None = None

        current_snapshot_path = extra.get("current_mesh_snapshot")

        if restore_current_mesh and current_snapshot_path:
            current_snapshot_text = str(current_snapshot_path)
            resolved = self._resolve_project_metadata_path(current_snapshot_text)
            assert resolved is not None

            if not resolved.exists():
                if strict:
                    raise FileNotFoundError(f"Project current mesh snapshot does not exist: {resolved}")
                skipped_current_mesh_snapshot = current_snapshot_text
            else:
                try:
                    restored_mesh = self._load_project_mesh_snapshot_for_restore(resolved)
                except Exception as exc:
                    message = (
                        f"Failed to restore current mesh snapshot "
                        f"{current_snapshot_text!r}: {exc}"
                    )
                    current_mesh_restore_error = message
                    if strict:
                        raise ValueError(message) from exc
                    skipped_current_mesh_snapshot = current_snapshot_text
                    restored_mesh = None
                    restored_current_mesh = False
                else:
                    restored_current_mesh = True
                    current_mesh_restore_error = None

        history_summary = {
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "latest_history_entry": None
            if self._last_mesh_history_entry is None
            else self._last_mesh_history_entry.operation_id,
            "skipped_history_entries": [],
        }

        if restore_history:
            old_undo = list(self._undo_stack)
            old_redo = list(self._redo_stack)
            old_latest = self._last_mesh_history_entry

            try:
                history_summary = self.restore_history_stacks_from_metadata(
                    metadata=metadata,
                    strict=strict,
                )
            except Exception:
                self._undo_stack = old_undo
                self._redo_stack = old_redo
                self._last_mesh_history_entry = old_latest
                raise

        if restore_current_mesh and restored_current_mesh and restored_mesh is not None:
            self.mesh = restored_mesh.copy()
            if self.original_mesh is None:
                self.original_mesh = restored_mesh.copy()
            resolved_current = self._resolve_project_metadata_path(current_snapshot_path)
            self.current_mesh_path = None if resolved_current is None else str(resolved_current)
            self.filepath = self.current_mesh_path

        self.clear_tool_preview_state()

        return {
            "restored_current_mesh": restored_current_mesh,
            "current_mesh_snapshot": None if current_snapshot_path is None else str(current_snapshot_path),
            "skipped_current_mesh_snapshot": skipped_current_mesh_snapshot,
            "current_mesh_restore_error": current_mesh_restore_error,
            "undo_count": int(history_summary.get("undo_count", 0)),
            "redo_count": int(history_summary.get("redo_count", 0)),
            "latest_history_entry": history_summary.get("latest_history_entry"),
            "skipped_history_entries": list(history_summary.get("skipped_history_entries", [])),
            "metadata_schema_supported": bool(history_summary.get("metadata_schema_supported", True)),
            "metadata_schema_error": history_summary.get("metadata_schema_error"),
        }

    def last_tool_preview_state(self) -> ToolPreviewState | None:
        """Return the most recent disk-backed tool preview state, if one exists."""

        return self._last_tool_preview_state

    def clear_tool_preview_state(self) -> None:
        """Clear any active disk-backed tool preview state reference."""

        self._last_tool_preview_state = None

    def last_mesh_history_entry(self) -> MeshHistoryEntry | None:
        """Return the most recent committed mesh history entry, if one exists."""

        return self._last_mesh_history_entry

    def clear_mesh_history_state(self) -> None:
        """Clear in-memory mesh history references. Does not delete snapshot files."""

        self._last_mesh_history_entry = None
        self._undo_stack.clear()
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo_last_mesh_operation(self) -> ManualEditResult:
        """
        Undo the most recent disk-backed mesh operation.

        Undo is snapshot-backed:
        - load MeshHistoryEntry.before_snapshot first
        - mutate self.mesh only after the snapshot loads successfully
        - move the history entry from undo stack to redo stack
        """

        if not self._undo_stack:
            raise ValueError("No mesh operation available to undo")

        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot undo mesh operation")

        entry = self._undo_stack[-1]

        # Load first. If loading fails, self.mesh and history stacks remain unchanged.
        target_mesh = self._coerce_to_trimesh(entry.before_snapshot.load())
        current_mesh = self.mesh

        before_vertices = int(len(current_mesh.vertices))
        before_faces = int(len(current_mesh.faces))
        after_vertices = int(len(target_mesh.vertices))
        after_faces = int(len(target_mesh.faces))

        self.mesh = target_mesh.copy()
        self.last_output_path = None

        self._undo_stack.pop()
        self._redo_stack.append(entry)
        self._last_mesh_history_entry = entry
        self.clear_tool_preview_state()

        result = ManualEditResult(
            operation=f"undo_{entry.operation}",
            mesh=self.mesh.copy(),
            before_vertices=before_vertices,
            after_vertices=after_vertices,
            before_faces=before_faces,
            after_faces=after_faces,
            notes=[
                f"Undo restored before_snapshot for operation {entry.operation_id}.",
            ],
        )

        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": entry,
            "history_dir": entry.history_dir,
        }

        self.sync_project_state_metadata(reason="undo_last_mesh_operation")

        return result

    def redo_last_mesh_operation(self) -> ManualEditResult:
        """
        Redo the most recently undone disk-backed mesh operation.

        Redo is snapshot-backed:
        - load MeshHistoryEntry.after_snapshot first
        - mutate self.mesh only after the snapshot loads successfully
        - move the history entry from redo stack back to undo stack
        """

        if not self._redo_stack:
            raise ValueError("No mesh operation available to redo")

        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot redo mesh operation")

        entry = self._redo_stack[-1]

        # Load first. If loading fails, self.mesh and history stacks remain unchanged.
        target_mesh = self._coerce_to_trimesh(entry.after_snapshot.load())
        current_mesh = self.mesh

        before_vertices = int(len(current_mesh.vertices))
        before_faces = int(len(current_mesh.faces))
        after_vertices = int(len(target_mesh.vertices))
        after_faces = int(len(target_mesh.faces))

        self.mesh = target_mesh.copy()
        self.last_output_path = None

        self._redo_stack.pop()
        self._undo_stack.append(entry)
        self._last_mesh_history_entry = entry
        self.clear_tool_preview_state()

        result = ManualEditResult(
            operation=f"redo_{entry.operation}",
            mesh=self.mesh.copy(),
            before_vertices=before_vertices,
            after_vertices=after_vertices,
            before_faces=before_faces,
            after_faces=after_faces,
            notes=[
                f"Redo restored after_snapshot for operation {entry.operation_id}.",
            ],
        )

        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": entry,
            "history_dir": entry.history_dir,
        }

        self.sync_project_state_metadata(reason="redo_last_mesh_operation")

        return result

    def get_mesh_stats(self) -> dict[str, Any]:
        self._require_mesh()
        mesh = self.mesh

        try:
            bounds = mesh.bounds
        except Exception:
            bounds = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        return {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "faces": int(len(mesh.faces)),
            "bounds": bounds,
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "euler_number": int(mesh.euler_number),
        }

    def analyze_topology(self, face_ids=None):
        """
        Analyze topology for the current working mesh or a selected face subset.

        This is a read-only Phase 2A helper. It does not modify self.mesh.

        Parameters
        ----------
        face_ids:
            Optional iterable of face IDs. If provided, topology is analyzed for
            that selected face region. If None, the full current mesh is analyzed.

        Returns
        -------
        TopologyReport
            Compact topology summary from far_mesh.core.selection_topology.

        Raises
        ------
        ValueError
            If no mesh is currently loaded.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot analyze topology")

        from far_mesh.core.selection_topology import analyze_selection_topology

        return analyze_selection_topology(self.mesh, face_ids=face_ids)

    def find_hole_candidates(
        self,
        face_ids=None,
        max_area_hint=None,
        max_perimeter=None,
    ):
        """
        Find hole candidates on the current mesh or selected face region.

        This is a read-only Phase 2C helper. It does not modify self.mesh.

        Parameters
        ----------
        face_ids:
            Optional iterable of face IDs. If provided, candidate detection is
            limited to that selected face region. If None, the full current mesh
            is analyzed.
        max_area_hint:
            Optional projected-area threshold. Candidates larger than this hint
            are filtered out.
        max_perimeter:
            Optional perimeter threshold. Candidates larger than this value are
            filtered out.

        Returns
        -------
        list[HoleCandidate]
            Candidate hole boundaries from far_mesh.core.selection_topology.

        Raises
        ------
        ValueError
            If no mesh is currently loaded.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot find hole candidates")

        from far_mesh.core.selection_topology import find_hole_candidates

        return find_hole_candidates(
            self.mesh,
            face_ids=face_ids,
            max_area_hint=max_area_hint,
            max_perimeter=max_perimeter,
        )

    @staticmethod
    def _hole_fill_unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
        """Return the stable unique-edge table used by the WGPU viewport.

        WGPU edge selection IDs are indices into a sorted unique edge table
        derived from triangle faces.  Rebuilding the same table in the core
        lets MeshProcessor convert a selection snapshot into source-mesh
        boundary edges without asking the viewport to become processing
        authority.
        """

        faces_arr = np.asarray(faces, dtype=np.int64)
        if faces_arr.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if faces_arr.ndim != 2 or faces_arr.shape[1] < 3:
            raise ValueError("mesh faces must be triangular to resolve selected edges")

        tri = faces_arr[:, :3]
        e01 = tri[:, [0, 1]]
        e12 = tri[:, [1, 2]]
        e20 = tri[:, [2, 0]]
        edges = np.vstack([e01, e12, e20]).astype(np.int64, copy=False)
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    @staticmethod
    def _normalize_edge_pair(a: int, b: int) -> tuple[int, int]:
        ia = int(a)
        ib = int(b)
        return (ia, ib) if ia <= ib else (ib, ia)

    @classmethod
    def _ordered_closed_loop_from_edge_pairs(
        cls,
        edge_pairs: Any,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        """Order selected edge pairs as one closed boundary loop.

        The first edge-selected hole-fill workflow is intentionally strict:
        the selected edges must form exactly one closed 2-valent loop.  This
        avoids treating branch selections, open chains, or multiple separate
        loops as commit-ready hole boundaries.  Later checkpoints can add
        explicit bridge/zipper semantics for interrupted loops.
        """

        unique_edges = {
            cls._normalize_edge_pair(int(edge[0]), int(edge[1]))
            for edge in edge_pairs or ()
            if len(edge) == 2 and int(edge[0]) != int(edge[1])
        }
        if len(unique_edges) < 3:
            raise ValueError("Selected edge hole boundary requires at least three unique edges.")

        vertex_to_neighbors: dict[int, set[int]] = {}
        for a, b in unique_edges:
            vertex_to_neighbors.setdefault(int(a), set()).add(int(b))
            vertex_to_neighbors.setdefault(int(b), set()).add(int(a))

        bad_degrees = {
            int(vertex_id): len(neighbors)
            for vertex_id, neighbors in vertex_to_neighbors.items()
            if len(neighbors) != 2
        }
        if bad_degrees:
            raise ValueError(
                "Selected edges do not form one closed hole boundary loop; "
                f"expected degree 2 at every boundary vertex, got {bad_degrees}."
            )

        start = min(vertex_to_neighbors)
        first_next = min(vertex_to_neighbors[start])
        ordered: list[int] = [int(start)]
        visited_vertices = {int(start)}
        prev = int(start)
        current = int(first_next)

        while current != start:
            if current in visited_vertices:
                raise ValueError("Selected edges contain a repeated vertex before closing the loop.")
            ordered.append(int(current))
            visited_vertices.add(int(current))
            neighbors = sorted(vertex_to_neighbors.get(int(current), set()))
            candidates = [int(v) for v in neighbors if int(v) != prev]
            if not candidates:
                raise ValueError("Selected edge loop ended before returning to the start vertex.")
            prev, current = int(current), int(candidates[0])

        if len(visited_vertices) != len(vertex_to_neighbors):
            raise ValueError("Selected edges contain multiple disconnected loops; select one hole boundary at a time.")

        ordered_edges = tuple(
            cls._normalize_edge_pair(ordered[i], ordered[(i + 1) % len(ordered)])
            for i in range(len(ordered))
        )
        if set(ordered_edges) != unique_edges:
            raise ValueError("Selected edge loop ordering did not cover the selected edges exactly once.")

        return tuple(int(v) for v in ordered), ordered_edges

    @staticmethod
    def _edge_selected_hole_fill_priority(
        *,
        perimeter: float,
        area_hint: float | None,
        edge_count: int,
    ) -> float:
        safe_perimeter = max(float(perimeter), 0.0)
        safe_edge_count = max(int(edge_count), 1)
        perimeter_score = 1.0 / (1.0 + safe_perimeter)
        edge_score = 1.0 / (1.0 + float(safe_edge_count))
        area_score = 0.0 if area_hint is None else 1.0 / (1.0 + max(float(area_hint), 0.0))
        return float(area_score + perimeter_score + edge_score)

    def _candidate_from_selected_edge_ids(self, selected_edge_ids: Any) -> Any:
        """Build a HoleCandidate-like boundary source from selected viewport edges.

        The selected edge IDs come from SelectionController snapshots and are
        stable WGPU unique-edge indices for the currently displayed mesh.  The
        core reconstructs that same unique-edge table from ``self.mesh.faces``
        so the viewport remains a selection source, not a processing authority.
        """

        self._require_mesh()
        assert self.mesh is not None

        try:
            raw_ids = np.asarray(selected_edge_ids, dtype=np.int64).reshape(-1)
        except Exception as exc:
            raise ValueError("Selected edge IDs could not be interpreted as integers.") from exc

        if raw_ids.size == 0:
            raise ValueError("No selected edges are available for edge-boundary hole fill.")

        edge_ids = np.unique(raw_ids[raw_ids >= 0]).astype(np.int64, copy=False)
        if edge_ids.size == 0:
            raise ValueError("No valid selected edges are available for edge-boundary hole fill.")

        all_edges = self._hole_fill_unique_edges_from_faces(np.asarray(self.mesh.faces, dtype=np.int64))
        if all_edges.size == 0:
            raise ValueError("The current mesh has no resolvable edge table.")

        invalid = [int(v) for v in edge_ids.tolist() if int(v) >= len(all_edges)]
        if invalid:
            raise ValueError(
                "Selected edge IDs are outside the current mesh edge table: "
                f"{invalid[:12]}"
            )

        selected_pairs = tuple(
            (int(all_edges[int(edge_id), 0]), int(all_edges[int(edge_id), 1]))
            for edge_id in edge_ids.tolist()
        )
        ordered_vertices, ordered_edges = self._ordered_closed_loop_from_edge_pairs(selected_pairs)

        from far_mesh.core.selection_topology import (
            BoundaryLoop,
            BoundaryLoopKind,
            ClassifiedBoundaryLoop,
            HoleCandidate,
            measure_boundary_loop,
        )

        loop = BoundaryLoop(
            vertices=ordered_vertices,
            edges=ordered_edges,
            closed=True,
        )
        measurement = measure_boundary_loop(self.mesh, loop)
        classified = ClassifiedBoundaryLoop(
            loop=loop,
            kind=BoundaryLoopKind.HOLE_BOUNDARY,
            measurement=measurement,
        )
        fill_priority = self._edge_selected_hole_fill_priority(
            perimeter=float(measurement.perimeter),
            area_hint=measurement.area_hint,
            edge_count=int(loop.edge_count),
        )
        candidate = HoleCandidate(
            loop=loop,
            classified_loop=classified,
            boundary_vertices=ordered_vertices,
            boundary_edges=ordered_edges,
            perimeter=float(measurement.perimeter),
            area_hint=measurement.area_hint,
            centroid=measurement.centroid,
            fill_priority=fill_priority,
        )

        # Dataclasses here are intentionally not slotted.  Attach source
        # metadata for downstream preview/history logs without changing the
        # public HoleCandidate schema.
        setattr(candidate, "source_kind", "selected_edges")
        setattr(candidate, "source_edge_ids", tuple(int(v) for v in edge_ids.tolist()))
        return candidate

    def build_hole_fill_preview(
        self,
        *,
        candidate_index: int = 0,
        candidate: Any | None = None,
        selected_edge_ids: Any | None = None,
        face_ids=None,
        max_area_hint=None,
        max_perimeter=None,
        method: str = "fan",
        use_execution_layer: bool = True,
        preview_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        operation_id: str | None = None,
    ) -> ManualEditPreview:
        """
        Build a non-destructive hole-fill preview.

        If use_execution_layer=True, route through TaskKind.HOLE_FILL_PREVIEW.
        If use_execution_layer=False, use the legacy direct helper path for
        fallback/debugging.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot build hole fill preview")

        self.clear_tool_preview_state()

        if candidate is None and selected_edge_ids is not None:
            candidate = self._candidate_from_selected_edge_ids(selected_edge_ids)
            candidate_index = 0

        if use_execution_layer:
            return self.build_hole_fill_preview_planned(
                candidate_index=candidate_index,
                candidate=candidate,
                face_ids=face_ids,
                max_area_hint=max_area_hint,
                max_perimeter=max_perimeter,
                method=method,
                preview_dir=preview_dir,
                storage_dir=storage_dir,
                operation_id=operation_id,
            )

        return self._build_hole_fill_preview_direct(
            candidate_index=candidate_index,
            candidate=candidate,
            face_ids=face_ids,
            max_area_hint=max_area_hint,
            max_perimeter=max_perimeter,
            method=method,
        )

    @staticmethod
    def _low_memory_hole_fill_face_threshold() -> int:
        raw = os.environ.get("FAR_MESH_LOW_MEMORY_HOLE_FILL_FACE_THRESHOLD", "500000")
        try:
            return max(1, int(raw))
        except Exception:
            return 500_000

    @staticmethod
    def _low_memory_hole_fill_context_rings() -> int:
        raw = os.environ.get("FAR_MESH_LOW_MEMORY_HOLE_FILL_CONTEXT_RINGS", "2")
        try:
            return max(0, int(raw))
        except Exception:
            return 2

    def _should_use_low_memory_hole_fill_preview(
        self,
        *,
        face_count: int,
        vertex_count: int,
        method: object,
        candidate: Any | None,
    ) -> bool:
        """Return whether routed hole-fill preview should use a local work unit.

        This path is intentionally candidate-driven.  It avoids sending the full
        application mesh through multiprocessing.Pool and runs the heavy preview
        builder on a local N-ring context mesh instead.
        """

        del vertex_count

        if candidate is None:
            return False

        method_key = str(method or "fan").strip().lower().replace("-", "_")
        if method_key in {"open3d", "open3d_fill"}:
            # Open3D's direct fill_holes route is whole-mesh oriented and is not
            # promoted to the low-memory local work-unit path here.
            return False

        if str(getattr(candidate, "source_kind", "") or "") == "selected_edges":
            return True

        threshold = self._low_memory_hole_fill_face_threshold()
        if int(face_count) < threshold:
            return False

        return True

    def _build_low_memory_hole_fill_payload(
        self,
        *,
        candidate: Any | None,
        candidate_index: int,
        method: str,
        face_count: int,
        vertex_count: int,
    ) -> dict[str, Any]:
        """Build a small PROCESS payload for large-mesh hole-fill preview."""

        self._require_mesh()
        assert self.mesh is not None
        if candidate is None:
            raise ValueError("Low-memory hole fill preview requires an explicit candidate.")

        from far_mesh.core.hole_context import build_local_hole_context_mesh

        rings = self._low_memory_hole_fill_context_rings()
        context = build_local_hole_context_mesh(
            self.mesh,
            candidate,
            rings=rings,
            low_memory=True,
        )

        boundary_edges: list[list[int]] = []
        for edge in getattr(candidate, "boundary_edges", ()) or ():
            try:
                if len(edge) == 2:
                    boundary_edges.append([int(edge[0]), int(edge[1])])
            except Exception:
                continue

        centroid = getattr(candidate, "centroid", None)
        centroid_payload: list[float] | None = None
        if centroid is not None:
            try:
                centroid_values = np.asarray(centroid, dtype=float).reshape(-1)
                if centroid_values.size >= 3:
                    centroid_payload = [float(v) for v in centroid_values[:3].tolist()]
            except Exception:
                centroid_payload = None

        return {
            "mesh": context.mesh,
            "low_memory_hole_fill": True,
            "boundary_source": str(getattr(candidate, "source_kind", "candidate_boundary") or "candidate_boundary"),
            "selected_edge_ids": [int(v) for v in getattr(candidate, "source_edge_ids", ()) or ()],
            "candidate_index": 0,
            "source_candidate_index": int(candidate_index),
            "method": str(method),
            "face_count": int(len(context.mesh.faces)),
            "vertex_count": int(len(context.mesh.vertices)),
            "source_mesh_face_count": int(face_count),
            "source_mesh_vertex_count": int(vertex_count),
            "context_rings": int(rings),
            "local_to_source_vertex_ids": list(context.local_to_source_vertex_ids),
            "source_face_ids": list(context.source_face_ids),
            "target_boundary_local_vertex_ids": list(context.target_boundary_local_vertex_ids),
            "target_boundary_source_vertex_ids": list(context.target_boundary_source_vertex_ids),
            "target_boundary_source_edges": boundary_edges,
            "target_area_hint": getattr(candidate, "area_hint", None),
            "target_perimeter": getattr(candidate, "perimeter", None),
            "target_fill_priority": getattr(candidate, "fill_priority", None),
            "target_centroid": centroid_payload,
        }

    def build_hole_fill_preview_planned(
        self,
        *,
        candidate_index: int = 0,
        candidate: Any | None = None,
        face_ids=None,
        max_area_hint=None,
        max_perimeter=None,
        method: str = "fan",
        preview_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        operation_id: str | None = None,
    ) -> ManualEditPreview:
        """
        Build a hole-fill preview through the Phase 1.5 execution layer.

        MeshProcessor remains the mesh authority. The task handler receives a
        copy of the mesh and returns plain preview payload data. This method
        validates that payload, stores ToolPreviewState when present, and
        converts the result back to ManualEditPreview for existing GUI code.
        """

        self._require_mesh()
        assert self.mesh is not None

        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        low_memory_preview = self._should_use_low_memory_hole_fill_preview(
            face_count=face_count,
            vertex_count=vertex_count,
            method=method,
            candidate=candidate,
        )

        if low_memory_preview:
            payload = self._build_low_memory_hole_fill_payload(
                candidate=candidate,
                candidate_index=int(candidate_index),
                method=method,
                face_count=face_count,
                vertex_count=vertex_count,
            )
        else:
            payload: dict[str, Any] = {
                "mesh": self.mesh.copy(),
                "candidate_index": int(candidate_index),
                "method": str(method),
                "face_count": face_count,
                "vertex_count": vertex_count,
            }

            if face_ids is not None:
                payload["face_ids"] = np.asarray(face_ids, dtype=np.int64).reshape(-1)
            if max_area_hint is not None:
                payload["max_area_hint"] = float(max_area_hint)
            if max_perimeter is not None:
                payload["max_perimeter"] = float(max_perimeter)

        resolved_operation_id = str(operation_id or self._new_operation_id("hole_fill"))
        payload["operation_id"] = resolved_operation_id

        state_dir = preview_dir if preview_dir is not None else storage_dir
        if state_dir is None:
            state_dir = self.project_storage.create_preview_dir(resolved_operation_id)

        payload["preview_dir"] = str(Path(state_dir).expanduser().resolve())

        request = TaskRequest(
            kind=TaskKind.HOLE_FILL_PREVIEW,
            payload=payload,
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            source_mesh_ref=self.current_mesh_path,
            description="Hole fill preview",
        )

        result = self.execute_task_request(request)

        if not result.ok:
            error_text = str(result.error or "Hole fill preview task failed.")

            # Preserve the public exception contract from the direct Phase 2E
            # path while the implementation routes through Phase 1.5.
            # Existing tests and callers expect invalid candidate indexes to
            # surface as IndexError, and missing candidates/invalid inputs as
            # ValueError, not as the generic MeshProcessorError wrapper.
            lowered_error = error_text.lower()
            if "candidate_index" in lowered_error and "out of range" in lowered_error:
                raise IndexError(error_text)
            if "no hole candidates" in lowered_error or "candidate" in lowered_error:
                raise ValueError(error_text)

            raise MeshProcessorError(error_text)

        result_payload = result.payload
        if not bool(result_payload.get("implemented", True)):
            status = result_payload.get("status") or "Hole fill preview task is not implemented."
            raise MeshProcessorError(str(status))

        preview_mesh = result_payload.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise MeshProcessorError("Hole fill preview task did not return preview_mesh.")

        base_mesh = result_payload.get("base_mesh")
        if not isinstance(base_mesh, trimesh.Trimesh):
            base_mesh = self.mesh.copy()

        selection_summary = result_payload.get("selection_summary")
        if not isinstance(selection_summary, dict):
            selection_summary = {
                "mode": "hole_boundary",
                "candidate_index": int(candidate_index),
                "method": str(method),
                "selected_faces": int(len(tuple(face_ids))) if face_ids is not None else 0,
                "selected_vertices": 0,
            }
        selection_summary = _normalize_hole_fill_preview_backend_metadata(
            selection_summary,
            requested_method=method,
        )

        notes = result_payload.get("notes")
        if not isinstance(notes, list):
            notes = []
        else:
            notes = list(notes)

        if self.last_execution_plan is not None:
            notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        tool_preview_state = result_payload.get("tool_preview_state")
        if isinstance(tool_preview_state, ToolPreviewState):
            self._last_tool_preview_state = tool_preview_state
        else:
            self._last_tool_preview_state = None

        if bool(selection_summary.get("low_memory_patch_only", False)):
            # In low-memory mode preview_mesh is intentionally patch-only.
            # Do not copy the full application mesh into ManualEditPreview.base_mesh.
            return ManualEditPreview(
                operation=str(result_payload.get("operation") or "hole_fill_preview"),
                preview_mesh=preview_mesh.copy(),
                base_mesh=preview_mesh.copy(),
                selection_summary=dict(selection_summary),
                notes=[str(note) for note in notes],
            )

        return ManualEditPreview(
            operation=str(result_payload.get("operation") or "hole_fill_preview"),
            preview_mesh=preview_mesh.copy(),
            base_mesh=base_mesh.copy(),
            selection_summary=dict(selection_summary),
            notes=[str(note) for note in notes],
        )

    @staticmethod
    def _extract_appended_patch_mesh_for_preview(
        base_mesh: trimesh.Trimesh,
        preview_mesh: trimesh.Trimesh,
    ) -> tuple[trimesh.Trimesh, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Extract an appended preview patch as an isolated patch mesh.

        The current fan-family preview builders append generated patch faces to
        the end of the preview mesh and keep global vertex ids in those faces.
        This helper centralizes that contract for both single and batch preview
        routes without depending on the task-registry private helper.
        """

        base_face_count = int(len(base_mesh.faces))
        base_vertex_count = int(len(base_mesh.vertices))

        preview_vertices = np.asarray(preview_mesh.vertices, dtype=float)
        preview_faces = np.asarray(preview_mesh.faces, dtype=np.int64)

        if preview_faces.ndim != 2 or preview_faces.shape[1] < 3:
            raise ValueError("Hole fill preview mesh does not contain triangular faces.")
        if preview_faces.shape[0] < base_face_count:
            raise ValueError("Preview mesh has fewer faces than base mesh; cannot extract patch.")

        patch_faces_global = preview_faces[base_face_count:, :3]
        if patch_faces_global.size == 0:
            raise ValueError("Hole fill preview did not append any patch faces.")

        used_vertex_ids = np.unique(patch_faces_global.reshape(-1)).astype(np.int64)
        if used_vertex_ids.size == 0:
            raise ValueError("Hole fill preview patch does not reference any vertices.")
        if int(np.min(used_vertex_ids)) < 0 or int(np.max(used_vertex_ids)) >= len(preview_vertices):
            raise ValueError("Hole fill preview patch references vertices outside the preview mesh.")

        remap = {int(old_id): new_id for new_id, old_id in enumerate(used_vertex_ids.tolist())}
        patch_vertices = preview_vertices[used_vertex_ids]
        patch_faces = np.asarray(
            [[remap[int(vertex_id)] for vertex_id in face] for face in patch_faces_global],
            dtype=np.int64,
        )

        patch_mesh = trimesh.Trimesh(
            vertices=patch_vertices,
            faces=patch_faces,
            process=False,
        )
        patch_mesh.remove_unreferenced_vertices()

        patch_face_ids = tuple(range(base_face_count, int(len(preview_faces))))
        patch_vertex_ids = tuple(int(v) for v in used_vertex_ids.tolist())
        new_face_ids = patch_face_ids
        new_vertex_ids = tuple(
            int(v) for v in used_vertex_ids.tolist() if int(v) >= base_vertex_count
        )

        return patch_mesh, patch_face_ids, patch_vertex_ids, new_face_ids, new_vertex_ids

    @staticmethod
    def _candidate_boundary_vertices_for_region(candidate: Any) -> tuple[int, ...]:
        values = getattr(candidate, "boundary_vertices", None)
        if values is None:
            loop = getattr(candidate, "loop", None)
            values = getattr(loop, "vertices", ()) if loop is not None else ()
        return tuple(int(v) for v in values or ())

    @staticmethod
    def _candidate_boundary_edges_for_region(candidate: Any) -> tuple[tuple[int, int], ...]:
        edges: list[tuple[int, int]] = []
        for item in getattr(candidate, "boundary_edges", ()) or ():
            try:
                a, b = item
            except Exception:
                continue
            edges.append((int(a), int(b)))
        return tuple(edges)

    @staticmethod
    def _candidate_metadata_for_region(candidate: Any) -> dict[str, Any]:
        return {
            "perimeter": getattr(candidate, "perimeter", None),
            "area_hint": getattr(candidate, "area_hint", None),
            "fill_priority": getattr(candidate, "fill_priority", None),
            "boundary_vertices": len(
                MeshProcessor._candidate_boundary_vertices_for_region(candidate)
            ),
            "boundary_edges": len(
                MeshProcessor._candidate_boundary_edges_for_region(candidate)
            ),
        }

    def build_batch_hole_fill_preview(
        self,
        *,
        face_ids=None,
        max_area_hint=None,
        max_perimeter=None,
        method: str = "fan",
        preview_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        operation_id: str | None = None,
    ) -> ManualEditPreview:
        """Build one disk-backed preview that fills all current candidates.

        BATCH-HF-B is intentionally conservative: it delegates geometry to the
        core batch fan-family builder, writes a ToolPreviewState, and leaves
        commit authority in commit_hole_fill_preview().  Adaptive Surface Fill
        v2 and Open3D batch policies remain blocked by the lower-level batch
        builder until they have explicit per-hole gates.
        """

        self._require_mesh()
        assert self.mesh is not None

        from far_mesh.core.hole_fill_preview import build_batch_hole_fill_preview_mesh
        from far_mesh.core.selection_topology import find_hole_candidates

        self.clear_tool_preview_state()

        candidates = find_hole_candidates(
            self.mesh,
            face_ids=face_ids,
            max_area_hint=max_area_hint,
            max_perimeter=max_perimeter,
        )
        if not candidates:
            raise ValueError("No hole candidates available for batch hole fill preview")

        base_mesh = self.mesh.copy()
        preview_mesh = build_batch_hole_fill_preview_mesh(
            self.mesh,
            candidates,
            method=method,
        )
        preview_mesh = self._coerce_to_trimesh(preview_mesh.copy())

        preview_metadata = dict(getattr(preview_mesh, "metadata", {}) or {})
        patch_mesh, patch_face_ids, patch_vertex_ids, new_face_ids, new_vertex_ids = (
            self._extract_appended_patch_mesh_for_preview(base_mesh, preview_mesh)
        )

        method_key = str(preview_metadata.get("method") or method or "fan").strip().lower().replace("-", "_")
        backend = str(preview_metadata.get("backend") or "trimesh_fan_batch")

        # Batch previews are candidate-list operations.  STAB-LM2-A selected-edge
        # source metadata belongs to the single-boundary preview path; this
        # function has no singular ``candidate`` object and must not inspect one.
        scope = "selected_faces" if face_ids is not None else "whole_mesh"
        mode = "hole_candidate_batch"

        candidate_summaries = self._json_safe_metadata_payload(
            preview_metadata.get("candidate_summaries", ())
        )

        selection_summary: dict[str, Any] = {
            "mode": "hole_candidate_batch",
            "scope": scope,
            "batch_mode": True,
            "method": method_key,
            "public_method": method_key,
            "backend": backend,
            "candidate_count": int(len(candidates)),
            "successful_candidate_count": int(preview_metadata.get("successful_candidate_count", len(candidates)) or len(candidates)),
            "failed_candidate_count": int(preview_metadata.get("failed_candidate_count", 0) or 0),
            "selected_faces": int(len(tuple(face_ids))) if face_ids is not None else 0,
            "selected_vertices": int(sum(len(getattr(candidate, "boundary_vertices", ()) or ()) for candidate in candidates)),
            "patch_faces": int(len(patch_face_ids)),
            "patch_vertices": int(len(patch_vertex_ids)),
            "new_faces": int(len(new_face_ids)),
            "new_vertices": int(len(new_vertex_ids)),
            "new_face_ids": tuple(int(v) for v in new_face_ids),
            "new_vertex_ids": tuple(int(v) for v in new_vertex_ids),
            "batch_patch_face_count": int(preview_metadata.get("batch_patch_face_count", len(patch_face_ids)) or len(patch_face_ids)),
            "batch_original_face_count": int(preview_metadata.get("batch_original_face_count", len(base_mesh.faces)) or len(base_mesh.faces)),
            "candidate_summaries": candidate_summaries,
            "commit_allowed": True,
            "commit_blocking_reasons": (),
            "commit_warnings": (),
        }

        for key in (
            "hole_fill_batch_patch_orientation",
            "new_face_ids",
            "new_vertex_ids",
        ):
            if key in preview_metadata:
                selection_summary[key] = self._json_safe_metadata_payload(preview_metadata[key])

        resolved_operation_id = str(operation_id or self._new_operation_id("hole_fill_batch"))
        state_dir = preview_dir if preview_dir is not None else storage_dir
        if state_dir is None:
            state_dir = self.project_storage.create_preview_dir(resolved_operation_id)
        state_dir = Path(state_dir).expanduser().resolve()
        state_dir.mkdir(parents=True, exist_ok=True)

        base_snapshot = MeshSnapshot.capture(
            base_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_BASE,
            name="base_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": resolved_operation_id,
                "batch_mode": True,
            },
        )
        preview_snapshot = MeshSnapshot.capture(
            preview_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_PREVIEW,
            name="preview_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": resolved_operation_id,
                "batch_mode": True,
                "method": method_key,
                "backend": backend,
                "candidate_count": int(len(candidates)),
            },
        )
        patch_snapshot = MeshSnapshot.capture(
            patch_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_PATCH,
            name="patch_mesh",
            metadata={
                "operation": "hole_fill_preview",
                "operation_id": resolved_operation_id,
                "batch_mode": True,
                "method": method_key,
                "backend": backend,
                "candidate_count": int(len(candidates)),
            },
        )

        input_regions = tuple(
            ToolRegion(
                name=f"Hole boundary {index + 1}",
                kind=REGION_KIND_HOLE_BOUNDARY,
                mesh_snapshot=None,
                face_ids=(),
                vertex_ids=self._candidate_boundary_vertices_for_region(candidate),
                edge_ids=self._candidate_boundary_edges_for_region(candidate),
                source="find_hole_candidates",
                metadata={
                    "candidate_index": int(index),
                    **self._candidate_metadata_for_region(candidate),
                },
            )
            for index, candidate in enumerate(candidates)
        )

        patch_region = ToolRegion(
            name="Generated batch hole-fill patch",
            kind=REGION_KIND_HOLE_PATCH,
            mesh_snapshot=patch_snapshot,
            face_ids=tuple(int(v) for v in patch_face_ids),
            vertex_ids=tuple(int(v) for v in patch_vertex_ids),
            new_face_ids=tuple(int(v) for v in new_face_ids),
            new_vertex_ids=tuple(int(v) for v in new_vertex_ids),
            source="hole_fill_batch_preview",
            metadata={
                "batch_mode": True,
                "method": method_key,
                "public_method": method_key,
                "backend": backend,
                "candidate_count": int(len(candidates)),
                "candidate_summaries": candidate_summaries,
            },
        )

        notes = [
            "Batch hole fill preview created through MeshProcessor BATCH-HF-B route.",
            "Preview is non-destructive; commit uses the snapshot-backed hole-fill commit path.",
            f"Batch candidate count: {len(candidates)}.",
        ]

        tool_state_metadata = self._json_safe_metadata_payload(
            {
                **selection_summary,
                "requested_method": method,
                "source_preview_operation": "hole_fill_batch_preview",
            }
        )

        tool_preview_state = ToolPreviewState(
            operation_id=resolved_operation_id,
            operation="hole_fill_preview",
            base_snapshot=base_snapshot,
            preview_snapshot=preview_snapshot,
            input_regions=input_regions,
            output_regions=(patch_region,),
            committable=True,
            markers=(HOLE_FILL_PREVIEW_MARKER, LOCAL_REGION_OPERATION_MARKER),
            notes=tuple(notes),
            metadata=tool_state_metadata,
        )
        json_path = state_dir / "preview_state.json"
        tool_preview_state.write_json(json_path, base_dir=state_dir)
        notes.append(f"ToolPreviewState written: {json_path}")
        self._last_tool_preview_state = tool_preview_state

        return ManualEditPreview(
            operation="hole_fill_preview",
            preview_mesh=preview_mesh.copy(),
            base_mesh=base_mesh.copy(),
            selection_summary=dict(selection_summary),
            notes=[str(note) for note in notes],
        )

    def _build_hole_fill_preview_direct(
        self,
        *,
        candidate_index: int = 0,
        candidate: Any | None = None,
        face_ids=None,
        max_area_hint=None,
        max_perimeter=None,
        method: str = "fan",
    ) -> ManualEditPreview:
        """
        Legacy direct hole-fill preview path.

        Kept as an explicit fallback while the Phase 1.5 routed path rolls out.
        This method remains read-only and does not create ToolPreviewState.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot build hole fill preview")

        from far_mesh.core.hole_fill_preview import build_hole_fill_preview_mesh
        from far_mesh.core.selection_topology import find_hole_candidates

        if candidate is None:
            candidates = find_hole_candidates(
                self.mesh,
                face_ids=face_ids,
                max_area_hint=max_area_hint,
                max_perimeter=max_perimeter,
            )

            if not candidates:
                raise ValueError("No hole candidates available for hole fill preview")

            index = int(candidate_index)
            if index < 0 or index >= len(candidates):
                raise IndexError(
                    f"Hole candidate index out of range: {index}; "
                    f"available candidates: {len(candidates)}"
                )

            candidate = candidates[index]
        else:
            index = int(candidate_index)
            candidates = (candidate,)
        preview_mesh = build_hole_fill_preview_mesh(
            self.mesh,
            candidate,
            method=method,
            all_candidates=candidates,
        )

        boundary_vertices = tuple(int(v) for v in candidate.boundary_vertices)
        boundary_edges = tuple(tuple(int(x) for x in edge) for edge in candidate.boundary_edges)
        scope = "selected_faces" if face_ids is not None else "whole_mesh"

        self._last_tool_preview_state = None

        selection_summary = _normalize_hole_fill_preview_backend_metadata(
            {
                "mode": mode,
                "scope": scope,
                "candidate_index": index,
                "candidate_count": len(candidates),
                "selected_faces": int(len(tuple(face_ids))) if face_ids is not None else 0,
                "selected_edges": len(tuple(getattr(candidate, "source_edge_ids", ()) or ())),
                "boundary_source": source_kind or "hole_candidate",
                "boundary_vertices": len(boundary_vertices),
                "boundary_edges": len(boundary_edges),
                "area_hint": candidate.area_hint,
                "perimeter": candidate.perimeter,
                "fill_priority": candidate.fill_priority,
                "method": str(method),
            },
            requested_method=method,
        )

        notes = [
            "Hole fill preview built with direct Phase 2E fallback path.",
            "No ToolPreviewState was created on the direct fallback path.",
            "__HOLE_FILL_PREVIEW_ONLY__",
        ]
        backend = str(selection_summary.get("backend") or "fan")
        if backend == "open3d":
            notes.append(
                "Open3D direct fallback preserved the single-hole safety guard by passing all candidates."
            )
        elif backend == "curvature_sphere":
            notes.append("Experimental curvature sphere center-fan preview backend used.")

        return ManualEditPreview(
            operation="hole_fill_preview",
            preview_mesh=preview_mesh.copy(),
            base_mesh=self.mesh.copy(),
            selection_summary=selection_summary,
            notes=notes,
        )



    @staticmethod
    def _bore_patch_to_trimesh(patch: Any) -> trimesh.Trimesh:
        """Return a local trimesh for a bore rebuild patch."""

        vertices = np.asarray(getattr(patch, "vertices", np.empty((0, 3))), dtype=float)
        faces = np.asarray(getattr(patch, "faces", np.empty((0, 3))), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            vertices = np.empty((0, 3), dtype=float)
        else:
            vertices = vertices[:, :3].astype(float, copy=True)
        if faces.ndim != 2 or faces.shape[1] < 3:
            faces = np.empty((0, 3), dtype=np.int64)
        else:
            faces = faces[:, :3].astype(np.int64, copy=True)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @staticmethod
    def _bore_removed_region_mesh(
        mesh: trimesh.Trimesh,
        face_ids: tuple[int, ...] | list[int] | np.ndarray,
    ) -> trimesh.Trimesh:
        """Return a compact mesh containing only the bore faces marked for removal.

        This helper is intentionally defensive because it is used for disk-backed
        preview/history region snapshots. Invalid face IDs are ignored; if no
        valid removed faces remain, an empty trimesh is returned instead of
        raising from the snapshot path.
        """

        if not isinstance(mesh, trimesh.Trimesh):
            return trimesh.Trimesh(
                vertices=np.empty((0, 3), dtype=float),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )

        ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        ids = ids[(ids >= 0) & (ids < int(len(mesh.faces)))]
        if ids.size == 0:
            return trimesh.Trimesh(
                vertices=np.empty((0, 3), dtype=float),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )

        try:
            submesh = mesh.submesh([ids], append=True, repair=False)
            if isinstance(submesh, trimesh.Trimesh) and not submesh.is_empty:
                return submesh.copy()
        except Exception:
            pass

        faces = np.asarray(mesh.faces, dtype=np.int64)[ids]
        if faces.size == 0:
            return trimesh.Trimesh(
                vertices=np.empty((0, 3), dtype=float),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )

        used = np.unique(faces.reshape(-1))
        used = used[(used >= 0) & (used < int(len(mesh.vertices)))]
        if used.size == 0:
            return trimesh.Trimesh(
                vertices=np.empty((0, 3), dtype=float),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )

        remap = {int(old): int(new) for new, old in enumerate(used.tolist())}
        local_faces = np.empty_like(faces, dtype=np.int64)
        for row_index, face in enumerate(faces):
            local_faces[row_index] = [remap[int(v)] for v in face]

        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=float)[used].copy(),
            faces=local_faces,
            process=False,
        )

    @staticmethod
    def _bore_preview_mesh_from_patch(
        base_mesh: trimesh.Trimesh,
        *,
        removal_face_ids: tuple[int, ...],
        patch: Any,
    ) -> tuple[trimesh.Trimesh, tuple[int, ...], tuple[int, ...]]:
        """Build a full preview mesh by removing bore wall faces and appending patch faces.

        BORE-C6 hardens the preview assembly path before destructive commit is
        enabled. The method validates patch face indices after they are shifted
        into the appended-vertex range and refuses to create an invalid preview
        mesh if the patch references missing vertices.
        """

        if not isinstance(base_mesh, trimesh.Trimesh):
            raise ValueError("Bore preview requires a trimesh base mesh.")

        vertices = np.asarray(base_mesh.vertices, dtype=float)
        faces = np.asarray(base_mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            vertices = np.empty((0, 3), dtype=float)
        else:
            vertices = vertices[:, :3].astype(float, copy=True)
        if faces.ndim != 2 or faces.shape[1] < 3:
            faces = np.empty((0, 3), dtype=np.int64)
        else:
            faces = faces[:, :3].astype(np.int64, copy=True)

        removal = np.asarray(removal_face_ids, dtype=np.int64).reshape(-1)
        removal = removal[(removal >= 0) & (removal < int(len(faces)))]

        keep = np.ones((int(len(faces)),), dtype=bool)
        if removal.size:
            keep[removal] = False
        kept_faces = faces[keep]

        patch_mesh = MeshProcessor._bore_patch_to_trimesh(patch)
        patch_vertices = np.asarray(patch_mesh.vertices, dtype=float)
        patch_faces = np.asarray(patch_mesh.faces, dtype=np.int64)
        if patch_vertices.ndim != 2 or patch_vertices.shape[1] < 3:
            patch_vertices = np.empty((0, 3), dtype=float)
        else:
            patch_vertices = patch_vertices[:, :3].astype(float, copy=True)
        if patch_faces.ndim != 2 or patch_faces.shape[1] < 3:
            patch_faces = np.empty((0, 3), dtype=np.int64)
        else:
            patch_faces = patch_faces[:, :3].astype(np.int64, copy=True)

        if len(patch_faces):
            local_min = int(np.min(patch_faces))
            local_max = int(np.max(patch_faces))
            if local_min < 0 or local_max >= int(len(patch_vertices)):
                raise ValueError(
                    "Bore rebuild patch face vertices reference outside the patch vertex array: "
                    f"min={local_min}, max={local_max}, expected 0..{max(int(len(patch_vertices)) - 1, 0)}"
                )

        base_vertex_count = int(len(vertices))
        preview_vertices = (
            np.vstack([vertices, patch_vertices])
            if len(patch_vertices)
            else vertices.copy()
        )

        if len(patch_faces):
            shifted_patch_faces = patch_faces + base_vertex_count
            max_vertex = int(len(preview_vertices)) - 1
            shifted_min = int(np.min(shifted_patch_faces))
            shifted_max = int(np.max(shifted_patch_faces))
            if shifted_min < 0 or shifted_max > max_vertex:
                raise ValueError(
                    "Shifted bore rebuild patch face vertices reference outside the preview vertex array: "
                    f"min={shifted_min}, max={shifted_max}, expected 0..{max_vertex}"
                )
            preview_faces = np.vstack([kept_faces, shifted_patch_faces])
        else:
            shifted_patch_faces = np.empty((0, 3), dtype=np.int64)
            preview_faces = kept_faces.copy()

        preview_mesh = trimesh.Trimesh(
            vertices=preview_vertices,
            faces=preview_faces,
            process=False,
        )
        new_face_start = int(len(kept_faces))
        new_face_ids = tuple(range(new_face_start, new_face_start + int(len(shifted_patch_faces))))
        new_vertex_ids = tuple(range(base_vertex_count, base_vertex_count + int(len(patch_vertices))))
        return preview_mesh, new_face_ids, new_vertex_ids

    @staticmethod
    def _bore_boundary_edge_pairs(boundary: Any) -> tuple[tuple[int, int], ...]:
        """Return selected bore loop edges as ToolRegion edge pairs."""

        pairs: list[tuple[int, int]] = []
        for loop in getattr(boundary, "loops", ()) or ():
            for edge in getattr(loop, "edges", ()) or ():
                try:
                    a, b = edge
                except Exception:
                    continue
                pairs.append((int(a), int(b)))
        return tuple(pairs)

    def analyze_bore_candidates_planned(
        self,
        selected_edge_ids: tuple[int, ...] | list[int] | np.ndarray,
    ) -> Any:
        """Analyze Bore candidates through the Phase 6 execution layer.

        The PROCESS task receives a defensive mesh copy and returns a plain
        payload. MeshProcessor rehydrates the UI-safe BoreToolDisplayResult in
        the parent process so GUI/display code stays unchanged.
        """

        self._require_mesh()
        assert self.mesh is not None

        edge_ids = self._nonnegative_int_tuple(selected_edge_ids)
        if not edge_ids:
            raise ValueError("Bore candidate analysis requires selected edge IDs.")

        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))
        payload: dict[str, Any] = {
            "mesh": self.mesh.copy(),
            "selected_edge_ids": edge_ids,
            "face_count": face_count,
            "vertex_count": vertex_count,
        }

        request = TaskRequest(
            kind=TaskKind.BORE_REGION_EXTRACT,
            payload=payload,
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
                "bore": True,
            },
            source_mesh_ref=self.current_mesh_path,
            description="Bore candidate analysis",
        )

        result = self.execute_task_request(request)
        if not result.ok:
            raise MeshProcessorError(str(result.error or "Bore candidate analysis task failed."))

        result_payload = result.payload
        if not bool(result_payload.get("implemented", True)):
            raise MeshProcessorError(str(result_payload.get("status") or "Bore candidate analysis task is not implemented."))

        return self._bore_display_result_from_task_payload(result_payload)


    def rebuild_bore_candidate_planned(
        self,
        *,
        edge_ids: tuple[int, ...] | list[int] | np.ndarray,
        candidate: Any,
        candidate_index: int | None = None,
        quad_density_mode: str = "lean_pi_opening",
        color_rebuilt_faces: bool = True,
        rebuilt_face_color: tuple[int, int, int, int] | None = None,
    ) -> Any:
        """Compute a Bore candidate rebuild through PROCESS task routing.

        This method does not commit the result. The returned RebuildResult is
        still committed by commit_bore_rebuild_result(), preserving MeshProcessor
        authority over active mesh replacement, undo/redo, and project storage.
        """

        self._require_mesh()
        assert self.mesh is not None

        selected_edge_ids = self._nonnegative_int_tuple(edge_ids)
        if not selected_edge_ids:
            raise ValueError("Bore rebuild requires selected edge IDs.")

        candidate_payload = self._bore_candidate_payload(candidate)
        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        payload: dict[str, Any] = {
            "mesh": self.mesh.copy(),
            "selected_edge_ids": selected_edge_ids,
            "candidate": candidate_payload,
            "candidate_id": str(candidate_payload.get("candidate_id") or ""),
            "quad_density_mode": str(quad_density_mode or "lean_pi_opening"),
            "color_rebuilt_faces": bool(color_rebuilt_faces),
            "face_count": face_count,
            "vertex_count": vertex_count,
        }
        if candidate_index is not None:
            payload["candidate_index"] = int(candidate_index)
        if rebuilt_face_color is not None:
            payload["rebuilt_face_color"] = tuple(int(v) for v in rebuilt_face_color)

        request = TaskRequest(
            kind=TaskKind.BORE_REBUILD_CANDIDATE,
            payload=payload,
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
                "bore": True,
            },
            source_mesh_ref=self.current_mesh_path,
            description="Bore candidate rebuild",
        )

        result = self.execute_task_request(request)
        if not result.ok:
            raise MeshProcessorError(str(result.error or "Bore candidate rebuild task failed."))

        result_payload = result.payload
        if not bool(result_payload.get("implemented", True)):
            raise MeshProcessorError(str(result_payload.get("status") or "Bore candidate rebuild task is not implemented."))

        return self._bore_rebuild_result_from_task_payload(result_payload)


    def _bore_display_result_from_task_payload(self, payload: dict[str, Any]) -> Any:
        from .bore.tool import BoreToolDisplayResult

        candidates = tuple(
            self._bore_candidate_view_from_task_payload(item)
            for item in tuple(payload.get("candidates", ()) or ())
            if isinstance(item, dict)
        )

        return BoreToolDisplayResult(
            selected_edge_ids=self._nonnegative_int_tuple(payload.get("selected_edge_ids")),
            normalized_edge_ids=self._nonnegative_int_tuple(payload.get("normalized_edge_ids")),
            region_face_ids=self._nonnegative_int_tuple(payload.get("region_face_ids")),
            seed_face_ids=self._nonnegative_int_tuple(payload.get("seed_face_ids")),
            region_preview_face_ids=self._nonnegative_int_tuple(payload.get("region_preview_face_ids")),
            candidates=candidates,
            diagnostics=dict(payload.get("diagnostics") or {}),
            analysis_text=str(payload.get("analysis_text") or ""),
            preview_text=str(payload.get("preview_text") or ""),
            status_text=str(payload.get("status_text") or ""),
            boundary_status_text=str(payload.get("boundary_status_text") or ""),
            selected_candidate_id=str(payload.get("selected_candidate_id") or (candidates[0].candidate_id if candidates else "")),
        )


    def _bore_candidate_view_from_task_payload(self, payload: dict[str, Any]) -> Any:
        from .bore.tool import BoreCandidateView

        rebuild_token = payload.get("rebuild_token")
        if not isinstance(rebuild_token, dict):
            rebuild_token = {}

        return BoreCandidateView(
            candidate_id=str(payload.get("candidate_id") or ""),
            feature_id=str(payload.get("feature_id") or ""),
            entity_type=str(payload.get("entity_type") or ""),
            feature_kind=str(payload.get("feature_kind") or ""),
            feature_family=str(payload.get("feature_family") or ""),
            recognition_stage=str(payload.get("recognition_stage") or ""),
            label=str(payload.get("label") or payload.get("candidate_id") or "Bore candidate"),
            table_object=str(payload.get("table_object") or ""),
            table_faces=str(payload.get("table_faces") or ""),
            table_geometry=str(payload.get("table_geometry") or ""),
            table_role=str(payload.get("table_role") or ""),
            description=str(payload.get("description") or ""),
            display_face_ids=self._nonnegative_int_tuple(payload.get("display_face_ids") or payload.get("preview_face_ids")),
            rebuild_face_ids=self._nonnegative_int_tuple(payload.get("rebuild_face_ids")),
            can_preview=bool(payload.get("can_preview", True)),
            can_rebuild=bool(payload.get("can_rebuild", False)),
            rebuild_disabled_reason=str(payload.get("rebuild_disabled_reason") or ""),
            rebuild_token=rebuild_token,
        )


    def _bore_candidate_payload(self, candidate: Any) -> dict[str, Any]:
        if isinstance(candidate, dict):
            return dict(candidate)
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, dict):
                return dict(value)
        return {
            "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
            "display_face_ids": self._nonnegative_int_tuple(getattr(candidate, "display_face_ids", ())),
            "rebuild_face_ids": self._nonnegative_int_tuple(getattr(candidate, "rebuild_face_ids", ())),
            "can_rebuild": bool(getattr(candidate, "can_rebuild", False)),
        }


    def _bore_rebuild_result_from_task_payload(self, payload: dict[str, Any]) -> Any:
        from .bore.rebuild import RebuildResult

        mesh = payload.get("mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise MeshProcessorError("Bore rebuild task did not return a trimesh mesh.")

        axis_raw = payload.get("axis") or (0.0, 0.0, 1.0)
        try:
            axis_values = tuple(float(v) for v in tuple(axis_raw)[:3])
        except Exception:
            axis_values = (0.0, 0.0, 1.0)
        if len(axis_values) != 3:
            axis_values = (0.0, 0.0, 1.0)

        added_faces_raw = payload.get("added_faces") or ()
        added_faces: list[tuple[int, int, int]] = []
        for face in added_faces_raw:
            try:
                vals = tuple(int(v) for v in tuple(face)[:3])
            except Exception:
                continue
            if len(vals) == 3:
                added_faces.append(vals)

        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics = dict(diagnostics)
        diagnostics.setdefault("execution_layer", "process_task")
        diagnostics.setdefault("task_kind", "bore_rebuild_candidate")

        return RebuildResult(
            mesh=mesh.copy(),
            removed_face_ids=self._nonnegative_int_tuple(payload.get("removed_face_ids")),
            added_face_ids=self._nonnegative_int_tuple(payload.get("added_face_ids")),
            added_faces=tuple(added_faces),
            loop0_vertices=self._nonnegative_int_tuple(payload.get("loop0_vertices")),
            loop1_vertices=self._nonnegative_int_tuple(payload.get("loop1_vertices")),
            axis=axis_values,
            radius=float(payload.get("radius") or 0.0),
            diagnostics=diagnostics,
        )


    def build_bore_cleanup_preview_from_selected_edges(
        self,
        selected_edge_ids: tuple[int, ...] | list[int] | np.ndarray,
        *,
        manual_depth: float | None = None,
        depth_mode: str = "auto",
        require_wall_region: bool = True,
        preview_dir: str | Path | None = None,
        storage_dir: str | Path | None = None,
        operation_id: str | None = None,
    ) -> ManualEditPreview:
        """Build a disk-backed, non-mutating bore cleanup preview from selected edge IDs.

        BORE-C5 keeps commit authority out of this method.  It converts the
        existing edge-selection workflow into the validated core bore pipeline,
        writes a ToolPreviewState, and returns a ManualEditPreview for GUI/status
        use.  The active mesh is not mutated here.
        """

        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot build bore cleanup preview")

        self.clear_tool_preview_state()

        from .bore import (
            build_bore_boundary_resource,
            analyze_bore_boundary_resource,
            analysis_summary,
            extract_bore_wall_region,
            wall_region_summary,
            build_bore_cleanup_preview as build_core_bore_cleanup_preview,
            preview_is_committable,
            preview_summary,
            rebuild_patch_summary,
        )

        base_mesh = self._coerce_to_trimesh(self.mesh.copy())
        edge_ids = tuple(int(v) for v in np.asarray(selected_edge_ids, dtype=np.int64).reshape(-1).tolist())
        if not edge_ids:
            raise ValueError("Bore cleanup preview requires selected edge IDs.")

        try:
            boundary = build_bore_boundary_resource(base_mesh, edge_ids, force_recovery=True)
        except TypeError:
            boundary = build_bore_boundary_resource(base_mesh, edge_ids)
        if not getattr(boundary, "is_valid", False):
            reasons = ", ".join(str(v) for v in getattr(boundary, "blocking_reasons", ()) or ())
            diagnostics = dict(getattr(boundary, "diagnostics", {}) or {})
            component_summary = dict(diagnostics.get("component_summary", {}) or {})
            hint = ""
            if component_summary.get("fragmented_selection"):
                hint = " Selection is highly fragmented; use a more continuous rim/region selection."
            raise ValueError(f"Selected edges do not form a valid bore boundary after recovery: {reasons or 'invalid boundary'}.{hint}")

        axis_estimate = analyze_bore_boundary_resource(boundary)
        axis_diag = dict(getattr(axis_estimate, "diagnostics", {}) or {})
        axis_blocking = tuple(str(v) for v in axis_diag.get("blocking_reasons", ()) or ())
        if axis_blocking:
            raise ValueError(
                "Bore boundary analysis failed: " + ", ".join(axis_blocking)
            )

        wall_region = extract_bore_wall_region(base_mesh, boundary, axis_estimate)
        wall_diag = dict(getattr(wall_region, "diagnostics", {}) or {})
        wall_blocking = tuple(str(v) for v in wall_diag.get("blocking_reasons", ()) or ())
        if require_wall_region and wall_blocking:
            raise ValueError(
                "Bore wall extraction failed: " + ", ".join(wall_blocking)
            )

        core_preview = build_core_bore_cleanup_preview(
            boundary,
            axis_estimate,
            wall_region,
            manual_depth=manual_depth,
            depth_mode=depth_mode,
            require_wall_region=require_wall_region,
        )
        if not preview_is_committable(core_preview):
            reasons = ", ".join(str(v) for v in getattr(core_preview, "blocking_reasons", ()) or ())
            raise ValueError(f"Bore cleanup preview is not committable: {reasons or 'preview blocked'}")

        patch = getattr(core_preview, "patch", None)
        if patch is None:
            raise ValueError("Bore cleanup preview did not produce a rebuild patch.")

        removal_face_ids = tuple(int(v) for v in getattr(wall_region, "wall_face_ids", ()) or ())
        if require_wall_region and not removal_face_ids:
            raise ValueError("Bore cleanup preview did not identify bore wall faces to remove.")

        preview_mesh, new_face_ids, new_vertex_ids = self._bore_preview_mesh_from_patch(
            base_mesh,
            removal_face_ids=removal_face_ids,
            patch=patch,
        )
        if preview_mesh.is_empty or int(len(preview_mesh.faces)) == 0:
            raise ValueError("Bore cleanup preview produced an empty mesh.")

        removed_region_mesh = self._bore_removed_region_mesh(base_mesh, removal_face_ids)
        patch_mesh = self._bore_patch_to_trimesh(patch)

        resolved_operation_id = operation_id or self._new_operation_id("bore_cleanup_preview")
        if preview_dir is not None:
            state_dir = Path(preview_dir).expanduser().resolve()
        elif storage_dir is not None:
            state_dir = Path(storage_dir).expanduser().resolve()
        else:
            state_dir = self.project_storage.create_preview_dir(resolved_operation_id)
        state_dir.mkdir(parents=True, exist_ok=True)

        boundary_summary = {
            "mode": str(getattr(boundary, "mode", "invalid")),
            "selected_edge_count": int(len(getattr(boundary, "selected_edge_ids", ()) or ())),
            "valid_edge_count": int(len(getattr(boundary, "valid_edge_ids", ()) or ())),
            "invalid_edge_count": int(len(getattr(boundary, "invalid_edge_ids", ()) or ())),
            "loop_count": int(getattr(boundary, "loop_count", 0)),
            "warnings": tuple(getattr(boundary, "warnings", ()) or ()),
            "blocking_reasons": tuple(getattr(boundary, "blocking_reasons", ()) or ()),
        }
        axis_summary = analysis_summary(axis_estimate)
        wall_summary = wall_region_summary(wall_region)
        patch_summary = rebuild_patch_summary(patch)
        core_preview_summary = preview_summary(core_preview)

        selection_summary = self._json_safe_metadata_payload(
            {
                "operation": "bore_cleanup_preview",
                "operation_id": resolved_operation_id,
                "selected_edges": int(len(edge_ids)),
                "selected_edge_count": int(len(edge_ids)),
                "boundary": boundary_summary,
                "axis": axis_summary,
                "wall_region": wall_summary,
                "patch": patch_summary,
                "preview": core_preview_summary,
                "bore_mode": boundary_summary.get("mode"),
                "axis_hint": axis_summary.get("axis_hint"),
                "radius": axis_summary.get("radius"),
                "diameter": axis_summary.get("diameter"),
                "depth": patch_summary.get("depth"),
                "depth_source": patch_summary.get("depth_source"),
                "wall_face_count": int(len(removal_face_ids)),
                "patch_vertex_count": int(len(patch_mesh.vertices)),
                "patch_face_count": int(len(patch_mesh.faces)),
                "new_face_ids": list(new_face_ids),
                "new_vertex_ids": list(new_vertex_ids),
                "deletion_face_ids": list(removal_face_ids),
                "committable": True,
                "require_wall_region": bool(require_wall_region),
            }
        )

        base_snapshot = MeshSnapshot.capture(
            base_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_BASE,
            name="base_mesh",
            metadata={"operation": "bore_cleanup_preview", "operation_id": resolved_operation_id},
        )
        preview_snapshot = MeshSnapshot.capture(
            preview_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_PREVIEW,
            name="preview_mesh",
            metadata={"operation": "bore_cleanup_preview", "operation_id": resolved_operation_id},
        )
        patch_snapshot = MeshSnapshot.capture(
            patch_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_PATCH,
            name="bore_rebuild_patch",
            metadata={"operation": "bore_cleanup_preview", "operation_id": resolved_operation_id},
        )
        removed_snapshot = MeshSnapshot.capture(
            removed_region_mesh,
            state_dir,
            role=SNAPSHOT_ROLE_REMOVED_REGION,
            name="bore_removed_wall_region",
            metadata={"operation": "bore_cleanup_preview", "operation_id": resolved_operation_id},
        )

        input_regions = (
            ToolRegion(
                name="selected_bore_boundary",
                kind=REGION_KIND_ZIPPER_CHAIN,
                edge_ids=self._bore_boundary_edge_pairs(boundary),
                source="selected_edges",
                metadata=self._json_safe_metadata_payload(boundary_summary),
            ),
        )
        output_regions = (
            ToolRegion(
                name="bore_wall_faces_to_remove",
                kind=REGION_KIND_BORE_REGION,
                mesh_snapshot=removed_snapshot,
                face_ids=removal_face_ids,
                source="bore_wall_extract",
                metadata=self._json_safe_metadata_payload(wall_summary),
            ),
            ToolRegion(
                name="bore_rebuild_patch",
                kind=REGION_KIND_ZIPPER_BRIDGE,
                mesh_snapshot=patch_snapshot,
                new_face_ids=new_face_ids,
                new_vertex_ids=new_vertex_ids,
                source="bore_rebuild",
                metadata=self._json_safe_metadata_payload(patch_summary),
            ),
        )

        notes = [
            "Bore cleanup preview created from selected edge boundary resources.",
            "BORE-C5 preview integration: active mesh was not mutated.",
            f"Bore boundary mode: {boundary_summary.get('mode')}; loops={boundary_summary.get('loop_count')}",
            f"Bore rebuild patch: faces={len(patch_mesh.faces)}, vertices={len(patch_mesh.vertices)}.",
            f"ToolPreviewState written: {state_dir / 'preview_state.json'}",
        ]
        for warning in getattr(core_preview, "warnings", ()) or ():
            text = str(warning)
            if text and text not in notes:
                notes.append(text)

        tool_preview_state = ToolPreviewState(
            operation_id=resolved_operation_id,
            operation="bore_cleanup_preview",
            base_snapshot=base_snapshot,
            preview_snapshot=preview_snapshot,
            input_regions=input_regions,
            output_regions=output_regions,
            committable=True,
            markers=(LOCAL_REGION_OPERATION_MARKER,),
            notes=tuple(notes),
            metadata=selection_summary,
        )
        json_path = state_dir / "preview_state.json"
        tool_preview_state.write_json(json_path, base_dir=state_dir)
        self._last_tool_preview_state = tool_preview_state

        preview = ManualEditPreview(
            operation="bore_cleanup_preview",  # type: ignore[arg-type]
            preview_mesh=preview_mesh.copy(),
            base_mesh=base_mesh.copy(),
            selection_summary=selection_summary,
            notes=notes,
        )
        return preview

    def _history_output_regions_from_bore_tool_state(
        self,
        state: ToolPreviewState,
        history_dir: Path,
    ) -> tuple[Any, ...]:
        """Copy bore preview output region snapshots into history storage.

        Preview snapshots live in the preview directory. A committed history
        record should be self-contained, so patch and removed-region meshes are
        recaptured beside the before/after snapshots.
        """

        output_regions: list[Any] = []
        for region in state.output_regions:
            snapshot = getattr(region, "mesh_snapshot", None)
            if snapshot is None:
                output_regions.append(region)
                continue

            try:
                region_mesh = snapshot.load()
            except Exception:
                output_regions.append(region)
                continue

            role = str(getattr(snapshot, "role", "") or SNAPSHOT_ROLE_PATCH)
            if region.kind == REGION_KIND_BORE_REGION:
                role = SNAPSHOT_ROLE_REMOVED_REGION
                name = "bore_removed_wall_region"
            elif region.kind == REGION_KIND_ZIPPER_BRIDGE:
                role = SNAPSHOT_ROLE_PATCH
                name = "bore_rebuild_patch"
            else:
                name = str(getattr(snapshot, "role", "region_mesh") or "region_mesh")

            region_snapshot = MeshSnapshot.capture(
                region_mesh,
                history_dir,
                role=role,
                name=name,
                metadata={
                    "operation": "bore_cleanup",
                    "operation_id": state.operation_id,
                    "source_preview_snapshot": snapshot.path,
                },
            )
            output_regions.append(replace(region, mesh_snapshot=region_snapshot))

        return tuple(output_regions)

    def _write_bore_history_entry(
        self,
        *,
        state: ToolPreviewState,
        committed_mesh: trimesh.Trimesh,
        notes: list[str],
    ) -> MeshHistoryEntry:
        """Write disk-backed history for a bore cleanup commit.

        This method must be called before ``self.mesh`` is mutated. If any
        snapshot or history JSON write fails, the exception propagates and the
        current working mesh/undo stack are left unchanged.
        """

        self._require_mesh()
        assert self.mesh is not None

        history_dir = self._history_dir_for_tool_preview_state(state)

        before_snapshot = MeshSnapshot.capture(
            self.mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": "bore_cleanup",
                "operation_id": state.operation_id,
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": "bore_cleanup",
                "operation_id": state.operation_id,
                "source_preview_snapshot": state.preview_snapshot.path,
            },
        )

        history_output_regions = self._history_output_regions_from_bore_tool_state(
            state,
            history_dir,
        )

        metadata = self._json_safe_metadata_payload(
            {
                "source_preview_snapshot": state.preview_snapshot.path,
                "source_base_snapshot": state.base_snapshot.path,
                "source_preview_operation": state.operation,
                "bore_mode": state.metadata.get("bore_mode"),
                "selected_edge_count": state.metadata.get("selected_edge_count", 0),
                "wall_face_count": state.metadata.get("wall_face_count", 0),
                "patch_face_count": state.metadata.get("patch_face_count", 0),
                **dict(state.metadata),
            }
        )

        entry = MeshHistoryEntry(
            operation_id=state.operation_id,
            operation="bore_cleanup",
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            input_regions=state.input_regions,
            output_regions=history_output_regions,
            notes=tuple(notes),
            markers=state.markers,
            metadata=metadata,
        )

        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    @staticmethod
    def _nonnegative_int_tuple(values: Any) -> tuple[int, ...]:
        """Return stable non-negative integer IDs from arbitrary tool payloads."""

        if values is None:
            return ()
        try:
            arr = np.asarray(values, dtype=np.int64).reshape(-1)
        except Exception:
            return ()
        if arr.size == 0:
            return ()
        arr = arr[arr >= 0]
        if arr.size == 0:
            return ()
        return tuple(int(v) for v in np.unique(arr).tolist())

    def _write_bore_rebuild_history_entry(
        self,
        *,
        rebuild_result: object,
        committed_mesh: trimesh.Trimesh,
        selected_edge_ids: tuple[int, ...],
        candidate_metadata: dict[str, Any],
        quad_density_mode: str | None,
        notes: list[str],
    ) -> MeshHistoryEntry:
        """Write disk-backed history for the current BoreTool RebuildResult path.

        This is the v17-v21 BoreTool commit adapter. It consumes the current
        core.bore.rebuild ``RebuildResult`` shape directly instead of going
        through the older ``bore_cleanup_preview`` ToolPreviewState path.

        Critical safety rule: this method must run before ``self.mesh`` or the
        undo/redo stacks are mutated. If snapshot/history writing fails, the
        active mesh remains untouched.
        """

        self._require_mesh()
        assert self.mesh is not None

        operation = "bore_rebuild"
        operation_id = self._new_operation_id(operation)
        history_dir = self.project_storage.create_history_dir(operation_id)

        removed_face_ids = self._nonnegative_int_tuple(
            getattr(rebuild_result, "removed_face_ids", ())
        )
        added_face_ids = self._nonnegative_int_tuple(
            getattr(rebuild_result, "added_face_ids", ())
        )
        candidate_display_face_ids = self._nonnegative_int_tuple(
            candidate_metadata.get("display_face_ids", candidate_metadata.get("preview_face_ids", ()))
        )
        candidate_rebuild_face_ids = self._nonnegative_int_tuple(
            candidate_metadata.get("rebuild_face_ids", candidate_metadata.get("delete_patch_face_ids", ()))
        )

        diagnostics = getattr(rebuild_result, "diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {"value": str(diagnostics)}

        metadata = self._json_safe_metadata_payload(
            {
                "operation": operation,
                "operation_id": operation_id,
                "selected_edge_ids": selected_edge_ids,
                "selected_edge_count": len(selected_edge_ids),
                "candidate_id": candidate_metadata.get("candidate_id"),
                "feature_id": candidate_metadata.get("feature_id"),
                "entity_type": candidate_metadata.get("entity_type"),
                "feature_kind": candidate_metadata.get("feature_kind"),
                "feature_family": candidate_metadata.get("feature_family"),
                "recognition_stage": candidate_metadata.get("recognition_stage"),
                "display_face_ids": candidate_display_face_ids,
                "rebuild_face_ids": candidate_rebuild_face_ids,
                "removed_face_ids": removed_face_ids,
                "added_face_ids": added_face_ids,
                "removed_face_count": int(getattr(rebuild_result, "removed_face_count", len(removed_face_ids))),
                "added_face_count": int(getattr(rebuild_result, "added_face_count", len(added_face_ids))),
                "quad_density_mode": quad_density_mode,
                "diagnostics": diagnostics,
                "candidate_metadata": candidate_metadata,
            }
        )

        before_snapshot = MeshSnapshot.capture(
            self.mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": operation,
                "operation_id": operation_id,
                "selected_edge_count": len(selected_edge_ids),
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": operation,
                "operation_id": operation_id,
                "removed_face_count": metadata.get("removed_face_count", 0),
                "added_face_count": metadata.get("added_face_count", 0),
            },
        )

        input_regions: list[ToolRegion] = []
        if candidate_display_face_ids:
            input_regions.append(
                ToolRegion(
                    name="bore_candidate_display_faces",
                    kind=REGION_KIND_BORE_REGION,
                    face_ids=candidate_display_face_ids,
                    source="core.bore.recognition_candidate",
                    metadata={
                        "candidate_id": str(candidate_metadata.get("candidate_id") or ""),
                        "contract": "display_face_ids_preview_only",
                    },
                )
            )
        if candidate_rebuild_face_ids:
            input_regions.append(
                ToolRegion(
                    name="bore_candidate_rebuild_faces",
                    kind=REGION_KIND_BORE_REGION,
                    face_ids=candidate_rebuild_face_ids,
                    source="core.bore.candidate_rebuild_request",
                    metadata={
                        "candidate_id": str(candidate_metadata.get("candidate_id") or ""),
                        "contract": "candidate_rebuild_face_ids_before_mesh_replacement",
                    },
                )
            )
        if removed_face_ids:
            input_regions.append(
                ToolRegion(
                    name="bore_removed_faces",
                    kind=REGION_KIND_BORE_REGION,
                    face_ids=removed_face_ids,
                    source="core.bore.rebuild_result",
                    metadata={
                        "contract": "removed_face_ids_before_mesh_replacement",
                    },
                )
            )

        output_regions: list[ToolRegion] = []
        if added_face_ids:
            output_regions.append(
                ToolRegion(
                    name="bore_rebuilt_faces",
                    kind=REGION_KIND_BORE_REGION,
                    new_face_ids=added_face_ids,
                    source="core.bore.rebuild_result",
                    metadata={
                        "contract": "added_face_ids_after_mesh_replacement",
                    },
                )
            )

        entry = MeshHistoryEntry(
            operation_id=operation_id,
            operation=operation,
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            input_regions=tuple(input_regions),
            output_regions=tuple(output_regions),
            notes=tuple(notes),
            markers=(LOCAL_REGION_OPERATION_MARKER,),
            metadata=metadata,
        )

        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    def commit_bore_rebuild_result(
        self,
        rebuild_result: object,
        *,
        selected_edge_ids: tuple[int, ...] = (),
        candidate_metadata: dict[str, Any] | None = None,
        quad_density_mode: str | None = None,
    ) -> ManualEditResult:
        """Commit a current BoreTool ``RebuildResult`` through MeshProcessor authority.

        This is the host-owned commit handoff for the integrated BoreTool
        pipeline. Bore rebuild geometry is produced by ``far_mesh.core.bore``;
        durable application mutation happens here:

        - capture before/after snapshots
        - write ``MeshHistoryEntry``
        - replace ``self.mesh`` only after history writes succeed
        - push undo stack and clear redo stack
        - sync project/session metadata
        """

        self._require_mesh()
        assert self.mesh is not None

        mesh_after = getattr(rebuild_result, "mesh", None)
        if mesh_after is None:
            raise ValueError("Bore RebuildResult does not contain a mesh.")

        committed_mesh = self._coerce_to_trimesh(mesh_after)
        if committed_mesh.is_empty or int(len(committed_mesh.faces)) == 0:
            raise ValueError("Bore RebuildResult produced an empty mesh; commit refused.")
        committed_mesh.remove_unreferenced_vertices()

        before_faces = int(len(self.mesh.faces))
        before_vertices = int(len(self.mesh.vertices))
        after_faces = int(len(committed_mesh.faces))
        after_vertices = int(len(committed_mesh.vertices))

        selected_edge_ids = self._nonnegative_int_tuple(selected_edge_ids)
        candidate_metadata = dict(candidate_metadata or {})
        diagnostics = getattr(rebuild_result, "diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {"value": str(diagnostics)}

        removed_count = int(getattr(rebuild_result, "removed_face_count", 0) or 0)
        added_count = int(getattr(rebuild_result, "added_face_count", 0) or 0)
        if not removed_count:
            removed_count = len(self._nonnegative_int_tuple(getattr(rebuild_result, "removed_face_ids", ())))
        if not added_count:
            added_count = len(self._nonnegative_int_tuple(getattr(rebuild_result, "added_face_ids", ())))

        notes = [
            "Bore RebuildResult committed through MeshProcessor authority.",
            "Snapshot-backed bore_rebuild commit path: history written before active mesh replacement.",
            f"Selected edge IDs: {len(selected_edge_ids)}",
            f"Candidate: {candidate_metadata.get('candidate_id', candidate_metadata.get('feature_id', 'unknown'))}",
            f"Removed faces: {removed_count}",
            f"Added faces: {added_count}",
        ]

        # Critical safety rule: all history/snapshot writes happen before
        # self.mesh or undo/redo state is mutated.
        history_entry = self._write_bore_rebuild_history_entry(
            rebuild_result=rebuild_result,
            committed_mesh=committed_mesh,
            selected_edge_ids=selected_edge_ids,
            candidate_metadata=candidate_metadata,
            quad_density_mode=quad_density_mode,
            notes=notes,
        )
        notes.append(f"MeshHistoryEntry written: {history_entry.history_dir}")

        self.mesh = committed_mesh
        self.last_output_path = None
        self._push_mesh_history_entry(history_entry)
        self.clear_tool_preview_state()

        result = ManualEditResult(
            operation="bore_rebuild",  # type: ignore[arg-type]
            mesh=self.mesh.copy(),
            before_faces=before_faces,
            after_faces=after_faces,
            before_vertices=before_vertices,
            after_vertices=after_vertices,
            notes=notes,
        )

        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": history_entry,
            "history_dir": history_entry.history_dir,
            "selected_edge_ids": selected_edge_ids,
            "candidate_metadata": candidate_metadata,
            "quad_density_mode": quad_density_mode,
        }

        self.sync_project_state_metadata(reason="commit_bore_rebuild_result")
        return result

    def commit_bore_cleanup_preview(
        self,
        preview: ManualEditPreview | None = None,
    ) -> ManualEditResult:
        """Commit a validated bore cleanup preview into the working mesh.

        BORE-C6 consumes the ``ToolPreviewState`` created by
        ``build_bore_cleanup_preview_from_selected_edges``. The commit path is
        intentionally snapshot-backed: history is written first, then ``self.mesh``
        and the undo stack are mutated.
        """

        self._require_mesh()
        assert self.mesh is not None

        active_state = self._last_tool_preview_state
        if active_state is None:
            raise ValueError("No active bore cleanup preview to commit.")

        if active_state.operation != "bore_cleanup_preview":
            raise ValueError(
                f"Active preview state is not a bore cleanup preview: {active_state.operation!r}"
            )

        if has_commit_blocking_marker(active_state.markers):
            raise ValueError("Bore cleanup preview has a commit-blocking marker; commit refused.")

        if not bool(active_state.committable):
            raise ValueError("Bore cleanup preview is not marked as committable.")

        if preview is not None:
            operation = str(getattr(preview, "operation", "") or "")
            if operation != "bore_cleanup_preview":
                raise ValueError(f"Preview is not a bore cleanup preview: {operation!r}")

        before_faces = int(len(self.mesh.faces))
        before_vertices = int(len(self.mesh.vertices))
        if int(active_state.base_snapshot.face_count) != before_faces or int(active_state.base_snapshot.vertex_count) != before_vertices:
            raise ValueError(
                "Bore cleanup preview is stale; the current mesh changed after the preview was built. "
                "Rebuild the Bore preview before committing."
            )

        committed_mesh = self._coerce_to_trimesh(active_state.preview_snapshot.load())
        if committed_mesh.is_empty or int(len(committed_mesh.faces)) == 0:
            raise ValueError("Bore cleanup preview produced an empty mesh; commit refused.")
        committed_mesh.remove_unreferenced_vertices()

        after_faces = int(len(committed_mesh.faces))
        after_vertices = int(len(committed_mesh.vertices))

        notes = [
            "Bore cleanup preview committed from ToolPreviewState preview_snapshot.",
            "BORE-C6 snapshot-backed bore cleanup commit path.",
            f"Bore mode: {active_state.metadata.get('bore_mode', 'unknown')}",
            f"Wall faces removed: {active_state.metadata.get('wall_face_count', 0)}",
            f"Patch faces added: {active_state.metadata.get('patch_face_count', 0)}",
        ]
        if preview is not None:
            for note in getattr(preview, "notes", []) or []:
                text = str(note)
                if text and text not in notes:
                    notes.append(text)

        # Critical safety rule: all history/snapshot writes happen before
        # self.mesh or undo/redo state is mutated.
        history_entry = self._write_bore_history_entry(
            state=active_state,
            committed_mesh=committed_mesh,
            notes=notes,
        )
        notes.append(f"MeshHistoryEntry written: {history_entry.history_dir}")

        self.mesh = committed_mesh
        self.last_output_path = None
        self._push_mesh_history_entry(history_entry)
        self.clear_tool_preview_state()

        result = ManualEditResult(
            operation="bore_cleanup",  # type: ignore[arg-type]
            mesh=self.mesh.copy(),
            before_faces=before_faces,
            after_faces=after_faces,
            before_vertices=before_vertices,
            after_vertices=after_vertices,
            notes=notes,
        )

        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": history_entry,
            "history_dir": history_entry.history_dir,
        }

        self.sync_project_state_metadata(reason="commit_bore_cleanup_preview")
        return result

    def _history_dir_for_tool_preview_state(self, state: ToolPreviewState) -> Path:
        """
        Choose a history directory near the preview-state directory.

        If preview snapshots live in:
            <root>/previews/preview_<operation_id>/preview_mesh.ply

        history is written to:
            <root>/history/op_<operation_id>/

        If the preview directory is not under a folder literally named
        "previews", history is written beside the preview parent as a safe
        fallback.
        """

        preview_dir = Path(state.preview_snapshot.path).expanduser().resolve().parent
        operation_id = str(state.operation_id or "hole_fill")

        # Preferred path: if the preview lives inside the active ProjectStorage
        # root, create history through ProjectStorage so naming/layout stays
        # consistent.
        try:
            preview_dir.relative_to(self.project_storage.root.resolve(strict=False))
        except ValueError:
            pass
        else:
            return self.project_storage.create_history_dir(operation_id)

        # Compatibility fallback for explicit preview_dir/storage_dir values that
        # are outside the active ProjectStorage root.
        preview_parent = preview_dir.parent

        if preview_parent.name == "previews":
            history_root = preview_parent.parent / "history"
        else:
            history_root = preview_parent / "history"

        history_dir = history_root / f"op_{operation_id}"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir

    def _history_output_regions_from_tool_state(
        self,
        state: ToolPreviewState,
        history_dir: Path,
    ) -> tuple[Any, ...]:
        """
        Copy patch-region snapshots into the history directory.

        The preview state owns preview-time patch snapshots. A committed
        history entry should have its own history-local patch snapshot so the
        history folder is self-contained.
        """

        output_regions: list[Any] = []

        for region in state.output_regions:
            snapshot = getattr(region, "mesh_snapshot", None)
            if region.kind == REGION_KIND_HOLE_PATCH and snapshot is not None:
                patch_mesh = snapshot.load()
                patch_snapshot = MeshSnapshot.capture(
                    patch_mesh,
                    history_dir,
                    role=SNAPSHOT_ROLE_PATCH,
                    name="patch_mesh",
                    metadata={
                        "operation": "hole_fill",
                        "operation_id": state.operation_id,
                        "source_preview_snapshot": snapshot.path,
                    },
                )
                output_regions.append(replace(region, mesh_snapshot=patch_snapshot))
            else:
                output_regions.append(region)

        return tuple(output_regions)

    def _write_hole_fill_history_entry(
        self,
        *,
        state: ToolPreviewState,
        committed_mesh: trimesh.Trimesh,
        notes: list[str],
    ) -> MeshHistoryEntry:
        """
        Write disk-backed history for a hole-fill commit.

        This method must be called before self.mesh is mutated. If any disk
        write fails, the exception propagates and the current working mesh is
        left unchanged.
        """

        self._require_mesh()
        assert self.mesh is not None

        history_dir = self._history_dir_for_tool_preview_state(state)

        before_snapshot = MeshSnapshot.capture(
            self.mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": "hole_fill",
                "operation_id": state.operation_id,
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": "hole_fill",
                "operation_id": state.operation_id,
                "source_preview_snapshot": state.preview_snapshot.path,
            },
        )

        history_output_regions = self._history_output_regions_from_tool_state(
            state,
            history_dir,
        )

        entry = MeshHistoryEntry(
            operation_id=state.operation_id,
            operation="hole_fill",
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            input_regions=state.input_regions,
            output_regions=history_output_regions,
            notes=tuple(notes),
            markers=state.markers,
            metadata={
                "source_preview_snapshot": state.preview_snapshot.path,
                "source_base_snapshot": state.base_snapshot.path,
                "source_preview_operation": state.operation,
                **dict(state.metadata),
            },
        )

        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    def _write_manual_edit_history_entry(
        self,
        *,
        preview: ManualEditPreview,
        committed_mesh: trimesh.Trimesh,
        operation: str,
        notes: list[str],
    ) -> MeshHistoryEntry:
        """
        Write disk-backed history for a manual edit commit.

        Phase 2G-A safety rule: snapshot/history writes happen before self.mesh
        or undo/redo state is mutated. If any write fails, the working mesh and
        history stacks remain unchanged.

        Currently enabled for destructive whole-mesh manual edits that already
        produce a full preview mesh: delete_faces, delete_vertices, and group_cleanup.
        """

        self._require_mesh()
        assert self.mesh is not None

        normalized_operation = str(operation or "").strip().lower()
        if normalized_operation not in {"delete_faces", "delete_vertices", "group_cleanup", "group_reduce"}:
            raise ValueError(f"Unsupported manual edit history operation: {operation!r}")

        operation_id = self._new_operation_id(normalized_operation)
        history_dir = self.project_storage.create_history_dir(operation_id)

        before_snapshot = MeshSnapshot.capture(
            self.mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": normalized_operation,
                "operation_id": operation_id,
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": normalized_operation,
                "operation_id": operation_id,
                "source_preview_operation": preview.operation,
            },
        )

        selection_summary = dict(getattr(preview, "selection_summary", {}) or {})
        entry = MeshHistoryEntry(
            operation_id=operation_id,
            operation=normalized_operation,
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            notes=tuple(notes),
            markers=(WHOLE_MESH_OPERATION_MARKER,),
            metadata={
                "source_preview_operation": preview.operation,
                "selection_summary": selection_summary,
                "selected_faces": int(selection_summary.get("selected_faces", 0) or 0),
                "selected_vertices": int(selection_summary.get("selected_vertices", 0) or 0),
            },
        )

        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    @staticmethod
    def _json_safe_metadata_payload(value: Any) -> Any:
        """Return a JSON-safe copy for history metadata."""

        return json.loads(json.dumps(value, default=str))

    def _write_open3d_tensor_fill_holes_repair_history_entry(
        self,
        *,
        committed_mesh: trimesh.Trimesh,
        repair_payload: dict[str, Any],
        notes: list[str],
    ) -> MeshHistoryEntry:
        """Write disk-backed history for guarded Open3D tensor repair.

        This method must be called before self.mesh or undo/redo state is
        mutated. If snapshot/history writing fails, the caller must leave mesh
        and stacks unchanged.
        """

        self._require_mesh()
        assert self.mesh is not None

        operation = "open3d_tensor_fill_holes_repair"
        operation_id = self._new_operation_id(operation)
        history_dir = self.project_storage.create_history_dir(operation_id)

        before_snapshot = MeshSnapshot.capture(
            self.mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": operation,
                "operation_id": operation_id,
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh.copy(),
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": operation,
                "operation_id": operation_id,
                "method": "open3d_tensor_fill_holes",
            },
        )

        dry_run_report = self._json_safe_metadata_payload(
            repair_payload.get("dry_run_report") or {}
        )
        policy_evaluation = self._json_safe_metadata_payload(
            repair_payload.get("policy_evaluation") or {}
        )

        entry = MeshHistoryEntry(
            operation_id=operation_id,
            operation=operation,
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            notes=tuple(notes),
            markers=(WHOLE_MESH_OPERATION_MARKER,),
            metadata={
                "method": "open3d_tensor_fill_holes",
                "hole_size": float(repair_payload.get("hole_size", 0.0) or 0.0),
                "dry_run_report": dry_run_report,
                "policy_evaluation": policy_evaluation,
            },
        )

        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    def _push_mesh_history_entry(self, entry: MeshHistoryEntry) -> None:
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._last_mesh_history_entry = entry




    @staticmethod
    def _preview_selection_summary(preview: ManualEditPreview) -> dict[str, Any]:
        summary = getattr(preview, "selection_summary", None)
        return dict(summary) if isinstance(summary, dict) else {}

    def _commit_low_memory_hole_fill_mesh(
        self,
        preview: ManualEditPreview,
    ) -> trimesh.Trimesh:
        """Return the committed full mesh for a patch-only low-memory preview."""

        self._require_mesh()
        assert self.mesh is not None

        summary = self._preview_selection_summary(preview)
        expected_faces = int(summary.get("source_mesh_face_count") or -1)
        expected_vertices = int(summary.get("source_mesh_vertex_count") or -1)
        current_faces = int(len(self.mesh.faces))
        current_vertices = int(len(self.mesh.vertices))

        if expected_faces != current_faces or expected_vertices != current_vertices:
            raise ValueError(
                "Low-memory hole fill preview is stale; the current mesh changed after preview build. "
                "Re-run Find Hole Candidates and Build Fill Preview."
            )

        encoded_faces = summary.get("low_memory_patch_faces_source")
        generated_vertices = summary.get("low_memory_generated_vertices")
        if not isinstance(encoded_faces, (list, tuple)) or not encoded_faces:
            raise ValueError("Low-memory hole fill preview is missing patch face mapping.")
        if generated_vertices is None:
            generated_vertices = []

        base_vertices = np.asarray(self.mesh.vertices, dtype=float)
        base_faces = np.asarray(self.mesh.faces, dtype=np.int64)[:, :3]
        generated = np.asarray(generated_vertices, dtype=float).reshape((-1, 3))

        if generated.size:
            vertices = np.vstack([base_vertices, generated])
        else:
            vertices = base_vertices.copy()

        patch_faces: list[list[int]] = []
        for raw_face in encoded_faces:
            if not isinstance(raw_face, (list, tuple)) or len(raw_face) < 3:
                raise ValueError("Low-memory patch face mapping contains an invalid face.")
            face: list[int] = []
            for raw_vertex in raw_face[:3]:
                encoded = int(raw_vertex)
                if encoded >= 0:
                    if encoded >= current_vertices:
                        raise ValueError("Low-memory patch references a source vertex outside the current mesh.")
                    face.append(encoded)
                else:
                    generated_index = -encoded - 1
                    if generated_index < 0 or generated_index >= int(len(generated)):
                        raise ValueError("Low-memory patch references a generated vertex outside the patch data.")
                    face.append(current_vertices + generated_index)
            patch_faces.append(face)

        patch_faces_array = np.asarray(patch_faces, dtype=np.int64).reshape((-1, 3))
        faces = np.vstack([base_faces, patch_faces_array])
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def _write_low_memory_hole_fill_history_entry(
        self,
        *,
        preview: ManualEditPreview,
        committed_mesh: trimesh.Trimesh,
        notes: list[str],
    ) -> MeshHistoryEntry:
        """Write history for a patch-only low-memory hole-fill commit."""

        self._require_mesh()
        assert self.mesh is not None

        summary = self._preview_selection_summary(preview)
        operation_id = str(summary.get("operation_id") or self._new_operation_id("hole_fill_low_memory"))
        history_dir = self.project_storage.create_history_dir(operation_id)

        before_snapshot = MeshSnapshot.capture(
            self.mesh,
            history_dir,
            role=SNAPSHOT_ROLE_BEFORE,
            name="before_mesh",
            metadata={
                "operation": "hole_fill",
                "operation_id": operation_id,
                "low_memory_patch_only": True,
            },
        )
        after_snapshot = MeshSnapshot.capture(
            committed_mesh,
            history_dir,
            role=SNAPSHOT_ROLE_AFTER,
            name="after_mesh",
            metadata={
                "operation": "hole_fill",
                "operation_id": operation_id,
                "low_memory_patch_only": True,
            },
        )

        patch_mesh = getattr(preview, "preview_mesh", None)
        output_regions: tuple[Any, ...] = ()
        if isinstance(patch_mesh, trimesh.Trimesh):
            patch_snapshot = MeshSnapshot.capture(
                patch_mesh,
                history_dir,
                role=SNAPSHOT_ROLE_PATCH,
                name="patch_mesh",
                metadata={
                    "operation": "hole_fill",
                    "operation_id": operation_id,
                    "low_memory_patch_only": True,
                },
            )
            output_regions = (
                ToolRegion(
                    name="Generated low-memory hole-fill patch",
                    kind=REGION_KIND_HOLE_PATCH,
                    mesh_snapshot=patch_snapshot,
                    face_ids=(),
                    vertex_ids=(),
                    new_face_ids=tuple(range(int(summary.get("patch_faces") or 0))),
                    new_vertex_ids=tuple(range(int(summary.get("new_vertices") or 0))),
                    source="low_memory_hole_fill_preview",
                    metadata={
                        "method": summary.get("method"),
                        "backend": summary.get("backend"),
                        "low_memory_patch_only": True,
                        "mapping_path": summary.get("low_memory_patch_mapping_path"),
                    },
                ),
            )

        boundary_region = ToolRegion(
            name="Hole boundary",
            kind=REGION_KIND_HOLE_BOUNDARY,
            mesh_snapshot=None,
            face_ids=(),
            vertex_ids=tuple(int(v) for v in summary.get("target_boundary_source_vertex_ids", ()) or ()),
            edge_ids=tuple(tuple(int(x) for x in edge) for edge in summary.get("target_boundary_source_edges", ()) or ()),
            source="low_memory_hole_fill_preview",
            metadata={
                "low_memory_patch_only": True,
                "source_mesh_face_count": summary.get("source_mesh_face_count"),
                "source_mesh_vertex_count": summary.get("source_mesh_vertex_count"),
            },
        )

        entry = MeshHistoryEntry(
            operation_id=operation_id,
            operation="hole_fill",
            history_dir=str(history_dir),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            input_regions=(boundary_region,),
            output_regions=output_regions,
            notes=tuple(notes),
            markers=(HOLE_FILL_PREVIEW_MARKER, LOCAL_REGION_OPERATION_MARKER),
            metadata={
                "low_memory_patch_only": True,
                "source_preview_operation": "hole_fill_preview",
                **summary,
            },
        )
        base_dir = history_dir.parent.parent if history_dir.parent.name == "history" else None
        entry.write_json(history_dir / "history_entry.json", base_dir=base_dir)
        return entry

    def commit_hole_fill_preview(
        self,
        preview: ManualEditPreview,
    ) -> ManualEditResult:
        """
        Commit a validated Phase 2F hole-fill preview into the working mesh.

        Checkpoint F behavior:
        - if an active ToolPreviewState exists, commit from preview_snapshot
        - write MeshHistoryEntry before mutating self.mesh
        - mutate self.mesh only after snapshot/history writes succeed
        - preserve the old ManualEditPreview-only fallback when no state exists
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded; cannot commit hole fill preview")
        if preview is None:
            raise ValueError("No hole fill preview to commit")

        operation = str(getattr(preview, "operation", "") or "")
        if operation not in {"hole_fill_preview", "hole_fill"}:
            raise ValueError(f"Preview is not a hole-fill preview: {operation!r}")

        active_state = self._last_tool_preview_state

        preview_mesh = getattr(preview, "preview_mesh", None)
        if not isinstance(preview_mesh, trimesh.Trimesh) and active_state is None:
            raise ValueError("Hole fill preview does not contain a valid preview mesh")

        base_mesh = getattr(preview, "base_mesh", None)
        before_faces = int(len(self.mesh.faces))
        before_vertices = int(len(self.mesh.vertices))
        summary = self._preview_selection_summary(preview)
        low_memory_patch_only = bool(summary.get("low_memory_patch_only", False))

        if low_memory_patch_only:
            committed_mesh = self._coerce_to_trimesh(
                self._commit_low_memory_hole_fill_mesh(preview)
            )
            notes = [
                "Hole fill preview committed from low-memory patch-only preview.",
                "Large-mesh local work-unit path: source mesh was not sent through Pool.",
            ]
        elif active_state is not None:
            if has_commit_blocking_marker(active_state.markers):
                raise ValueError(
                    "Hole fill preview has a commit-blocking marker; commit refused."
                )

            if int(active_state.base_snapshot.face_count) != before_faces or int(active_state.base_snapshot.vertex_count) != before_vertices:
                raise ValueError(
                    "Hole fill preview is stale; the current mesh changed after the preview was built. "
                    "Re-run Find Hole Candidates and Build Fill Preview."
                )

            committed_mesh = self._coerce_to_trimesh(active_state.preview_snapshot.load())
            notes = [
                "Hole fill preview committed from ToolPreviewState preview_snapshot.",
                "Phase 2F snapshot-backed hole-fill commit path.",
            ]
        else:
            if isinstance(base_mesh, trimesh.Trimesh):
                base_faces = int(len(base_mesh.faces))
                base_vertices = int(len(base_mesh.vertices))
                if base_faces != before_faces or base_vertices != before_vertices:
                    raise ValueError(
                        "Hole fill preview is stale; the current mesh changed after the preview was built. "
                        "Re-run Find Hole Candidates and Build Fill Preview."
                    )

            assert isinstance(preview_mesh, trimesh.Trimesh)
            committed_mesh = self._coerce_to_trimesh(preview_mesh.copy())
            notes = [
                "Hole fill preview committed.",
                "Phase 2F memory-only fallback commit path.",
            ]

        if committed_mesh.is_empty or len(committed_mesh.faces) == 0:
            raise ValueError("Hole fill preview produced an empty mesh; commit refused")

        committed_mesh.remove_unreferenced_vertices()

        for note in getattr(preview, "notes", []) or []:
            note_text = str(note)
            if note_text not in notes and note_text != "__ROI_ONLY_PREVIEW__":
                notes.append(note_text)

        history_entry: MeshHistoryEntry | None = None
        if low_memory_patch_only:
            history_entry = self._write_low_memory_hole_fill_history_entry(
                preview=preview,
                committed_mesh=committed_mesh,
                notes=notes,
            )
            notes.append(f"MeshHistoryEntry written: {history_entry.history_dir}")
        elif active_state is not None:
            # Critical safety rule: all history/snapshot writes happen before
            # self.mesh or undo/redo state is mutated.
            history_entry = self._write_hole_fill_history_entry(
                state=active_state,
                committed_mesh=committed_mesh,
                notes=notes,
            )
            notes.append(f"MeshHistoryEntry written: {history_entry.history_dir}")

        self.mesh = committed_mesh
        self.last_output_path = None

        if history_entry is not None:
            self._push_mesh_history_entry(history_entry)

        self.clear_tool_preview_state()

        result = ManualEditResult(
            operation="hole_fill",
            mesh=self.mesh if low_memory_patch_only else self.mesh.copy(),
            before_faces=before_faces,
            after_faces=int(len(self.mesh.faces)),
            before_vertices=before_vertices,
            after_vertices=int(len(self.mesh.vertices)),
            notes=notes,
        )

        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": history_entry,
            "history_dir": None if history_entry is None else history_entry.history_dir,
        }

        self.sync_project_state_metadata(reason="commit_hole_fill_preview")

        return result

    def build_manual_edit_preview(
        self,
        req: ManualEditRequest,
        *,
        triangle_attrs: dict[str, Any] | None = None,
        vertex_attrs: dict[str, Any] | None = None,
    ) -> ManualEditPreview:
        self._require_mesh()
        return build_manual_edit_preview(
            self.mesh,
            req,
            triangle_attrs=triangle_attrs,
            vertex_attrs=vertex_attrs,
        )

    def build_manual_edit_preview_routed(
        self,
        req: ManualEditRequest,
        *,
        triangle_attrs: dict[str, Any] | None = None,
        vertex_attrs: dict[str, Any] | None = None,
        group_opts: QuadGroupProcessOptions | None = None,
        use_execution_layer: bool = True,
    ) -> ManualEditPreview:
        self._require_mesh()

        if req.operation in {"group_cleanup", "group_reduce"}:
            if req.selection.mode != "faces" or req.selection.face_ids is None:
                raise ValueError("Grouped manual edits require a face selection.")

            opts = group_opts or QuadGroupProcessOptions()
            opts = replace(
                opts,
                reduce=(req.operation == "group_reduce"),
            )

            if use_execution_layer:
                return self.build_group_manual_edit_preview_planned(req.selection.face_ids, opts)

            return self.build_group_manual_edit_preview(req.selection.face_ids, opts)

        if use_execution_layer:
            return self.build_manual_edit_preview_planned(
                req,
                triangle_attrs=triangle_attrs,
                vertex_attrs=vertex_attrs,
            )

        return self.build_manual_edit_preview(
            req,
            triangle_attrs=triangle_attrs,
            vertex_attrs=vertex_attrs,
        )

    def build_manual_edit_preview_planned(
        self,
        req: ManualEditRequest,
        *,
        triangle_attrs: dict[str, Any] | None = None,
        vertex_attrs: dict[str, Any] | None = None,
    ) -> ManualEditPreview:
        """
        Build a normal manual-edit preview through Phase 1.5.

        This routes non-group preview operations through TaskKind.MESH_PREVIEW
        without changing preview/commit semantics. MeshProcessor remains the
        owner of the current mesh and only converts the TaskResult back into
        the established ManualEditPreview contract.
        """

        self._require_mesh()

        if req.operation in {"group_cleanup", "group_reduce"}:
            raise ValueError(
                "Grouped manual edit previews must use "
                "build_group_manual_edit_preview_planned()."
            )

        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        payload: dict[str, Any] = {
            "mesh": self.mesh.copy(),
            "operation": req.operation,
            "selection_mode": req.selection.mode,
            "face_ids": req.selection.face_ids,
            "vertex_ids": req.selection.vertex_ids,
            "parameters": dict(req.parameters),
            "source_path": req.selection.source_path,
            "face_count": face_count,
            "vertex_count": vertex_count,
        }

        if triangle_attrs is not None:
            payload["triangle_attrs"] = triangle_attrs
        if vertex_attrs is not None:
            payload["vertex_attrs"] = vertex_attrs

        request = TaskRequest(
            kind=TaskKind.MESH_PREVIEW,
            payload=payload,
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            source_mesh_ref=self.current_mesh_path,
            description=f"Manual edit preview: {req.operation}",
        )

        result = self.execute_task_request(request)

        if not result.ok:
            raise MeshProcessorError(result.error or "Manual edit preview task failed.")

        result_payload = result.payload

        preview_mesh = result_payload.get("preview_mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise MeshProcessorError("Manual edit preview task did not return preview_mesh.")

        base_mesh = result_payload.get("base_mesh")
        if not isinstance(base_mesh, trimesh.Trimesh):
            base_mesh = self.mesh.copy()

        selection_summary = result_payload.get("selection_summary")
        if not isinstance(selection_summary, dict):
            face_ids = np.asarray(req.selection.face_ids, dtype=np.int64).reshape(-1) if req.selection.face_ids is not None else np.empty((0,), dtype=np.int64)
            vertex_ids = np.asarray(req.selection.vertex_ids, dtype=np.int64).reshape(-1) if req.selection.vertex_ids is not None else np.empty((0,), dtype=np.int64)
            selection_summary = {
                "mode": req.selection.mode,
                "selected_faces": int(face_ids.size),
                "selected_vertices": int(vertex_ids.size),
            }

        notes = result_payload.get("notes")
        if not isinstance(notes, list):
            notes = []
        else:
            notes = list(notes)

        if self.last_execution_plan is not None:
            notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        return ManualEditPreview(
            operation=req.operation,
            preview_mesh=preview_mesh.copy(),
            base_mesh=base_mesh.copy(),
            selection_summary=selection_summary,
            notes=[str(note) for note in notes],
        )

    def build_group_manual_edit_preview_planned(
        self,
        selected_face_ids: Any,
        opts: QuadGroupProcessOptions,
    ) -> ManualEditPreview:
        """
        Build a grouped manual-edit preview through Phase 1.5.

        MeshProcessor still owns the current mesh. The execution layer only
        runs the registered pure-core handler and returns a TaskResult, which
        is converted back into ManualEditPreview here.
        """

        self._require_mesh()

        selected_ids = np.asarray(selected_face_ids, dtype=np.int64).reshape(-1)
        selected_count = int(len(selected_ids))
        face_count = int(len(self.mesh.faces))
        vertex_count = int(len(self.mesh.vertices))

        request = TaskRequest(
            kind=TaskKind.GROUP_REDUCE if opts.reduce else TaskKind.GROUP_CLEANUP,
            payload={
                "mesh": self.mesh.copy(),
                "face_ids": selected_ids,
                "group_options": {
                    "decode_mode": opts.decode_mode,
                    "texture_path": opts.texture_path,
                    "cleanup": opts.cleanup,
                    "reduce": opts.reduce,
                    "target_ratio": opts.target_ratio,
                    "boundary_weight": opts.boundary_weight,
                    "allow_non_manifold_edge_removal": opts.allow_non_manifold_edge_removal,
                },
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            hints={
                "face_count": face_count,
                "vertex_count": vertex_count,
            },
            source_mesh_ref=self.current_mesh_path,
            description="Patch-aware grouped manual edit preview",
        )

        result = self.execute_task_request(request)

        if not result.ok:
            raise MeshProcessorError(result.error or "Grouped manual edit task failed.")

        result_payload = result.payload
        preview_mesh = result_payload.get("preview_mesh") or result_payload.get("mesh")
        if not isinstance(preview_mesh, trimesh.Trimesh):
            raise MeshProcessorError("Grouped manual edit task did not return a preview mesh.")

        base_mesh = result_payload.get("base_mesh")
        if not isinstance(base_mesh, trimesh.Trimesh):
            base_mesh = self.mesh.copy()

        selection_summary = result_payload.get("selection_summary")
        if not isinstance(selection_summary, dict):
            selection_summary = {
                "mode": "faces",
                "selected_faces": selected_count,
                "selected_vertices": 0,
            }

        notes = result_payload.get("notes")
        if not isinstance(notes, list):
            notes = []
        else:
            notes = list(notes)

        final_notes = [
            "Preview shows the full merged mesh after patch-aware group processing.",
            f"Group decode mode: {opts.decode_mode}",
        ]
        final_notes.extend(str(note) for note in notes)

        if self.last_execution_plan is not None:
            final_notes.append(
                f"Phase 1.5 execution: {self.last_execution_plan.mode.value}; "
                f"{self.last_execution_plan.reason}"
            )

        return ManualEditPreview(
            operation="group_reduce" if opts.reduce else "group_cleanup",
            preview_mesh=preview_mesh.copy(),
            base_mesh=base_mesh.copy(),
            selection_summary=selection_summary,
            notes=final_notes,
        )

    def commit_manual_edit_preview(
        self,
        preview: ManualEditPreview,
    ) -> ManualEditResult:
        self._require_mesh()
        assert self.mesh is not None

        result = commit_manual_edit_preview(preview)
        committed_mesh = self._coerce_to_trimesh(result.mesh)
        committed_mesh.remove_unreferenced_vertices()

        history_entry: MeshHistoryEntry | None = None
        if result.operation in {"delete_faces", "delete_vertices", "group_cleanup", "group_reduce"}:
            base_mesh = getattr(preview, "base_mesh", None)
            if isinstance(base_mesh, trimesh.Trimesh):
                if int(len(base_mesh.faces)) != int(len(self.mesh.faces)) or int(len(base_mesh.vertices)) != int(len(self.mesh.vertices)):
                    raise ValueError(
                        f"Manual {result.operation} preview is stale; the current mesh changed after the preview was built. "
                        "Rebuild the preview before committing."
                    )

            notes_for_history = list(result.notes)
            phase_label = "Phase 2G-B" if result.operation in {"group_cleanup", "group_reduce"} else "Phase 2G-A"
            notes_for_history.append(
                f"{phase_label} snapshot-backed manual {result.operation} commit path."
            )
            history_entry = self._write_manual_edit_history_entry(
                preview=preview,
                committed_mesh=committed_mesh,
                operation=str(result.operation),
                notes=notes_for_history,
            )
            result.notes.append(f"MeshHistoryEntry written: {history_entry.history_dir}")

        self.mesh = committed_mesh
        self.last_output_path = None

        if history_entry is not None:
            self._push_mesh_history_entry(history_entry)

        self.clear_tool_preview_state()
        self.last_manual_edit_result = {
            "operation": result.operation,
            "before_faces": result.before_faces,
            "after_faces": result.after_faces,
            "before_vertices": result.before_vertices,
            "after_vertices": result.after_vertices,
            "notes": list(result.notes),
            "mesh": self.mesh,
            "history_entry": history_entry,
            "history_dir": None if history_entry is None else history_entry.history_dir,
        }

        if history_entry is not None:
            self.sync_project_state_metadata(reason=f"commit_manual_edit_preview:{result.operation}")

        return result

    def build_quadwild_group_proxy_from_current_mesh(
        self,
        opts: QuadGroupProcessOptions,
    ) -> QuadProxyMesh:
        self._require_mesh()
        proxy = build_quadwild_proxy_from_mesh(self.mesh, opts)
        self.last_quad_group_proxy = proxy
        return proxy

    def build_group_manual_edit_preview(
        self,
        selected_face_ids: Any,
        opts: QuadGroupProcessOptions,
    ) -> ManualEditPreview:
        self._require_mesh()

        selected_ids = np.asarray(selected_face_ids, dtype=np.int64).reshape(-1)
        selected_count = int(len(selected_ids))

        proxy = self.build_quadwild_group_proxy_from_current_mesh(opts)
        processed = process_selected_groups_locally(proxy, selected_ids, opts)
        self.last_quad_group_proxy = processed

        operation = "group_reduce" if opts.reduce else "group_cleanup"

        return ManualEditPreview(
            operation=operation,
            preview_mesh=processed.proxy_trimesh.copy(),
            base_mesh=self.mesh.copy(),
            selection_summary={
                "mode": "faces",
                "selected_faces": selected_count,
                "selected_vertices": 0,
            },
            notes=[
                "Preview shows the full merged mesh after patch-aware group processing.",
                f"Group decode mode: {opts.decode_mode}",
            ],
        )

    def _remesh_instant_meshes(
        self,
        target_faces: int,
        crease_angle: float,
        smooth_iterations: int,
        deterministic: bool,
    ) -> dict[str, Any]:
        self._require_mesh()

        if hasattr(self.instant_remesher, "is_available") and not self.instant_remesher.is_available():
            raise MeshProcessorError(self.instant_remesher.get_unavailable_reason())

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="far_mesh_instant_") as tmpdir:
            input_path = Path(tmpdir) / "input.obj"
            output_path = Path(tmpdir) / "output.obj"

            self.mesh.export(input_path)

            self.instant_remesher.run(
                input_mesh_path=str(input_path),
                output_mesh_path=str(output_path),
                target_faces=target_faces,
                crease_angle=crease_angle,
                smooth_iterations=smooth_iterations,
                deterministic=deterministic,
                lifecycle=get_lifecycle_manager(),
            )

            if not output_path.exists():
                raise FileNotFoundError(f"Instant Meshes output was not created: {output_path}")

            loaded = trimesh.load(output_path, force="mesh")
            new_mesh = self._coerce_to_trimesh(loaded)
            if new_mesh.is_empty or len(new_mesh.faces) == 0:
                raise ValueError("Instant Meshes produced an empty output mesh.")

            new_mesh.remove_unreferenced_vertices()
            self.mesh = new_mesh
            self.last_output_path = str(output_path)

            elapsed = time.perf_counter() - t0

            return {
                "backend": "instant_meshes",
                "output_path": str(output_path),
                "final_stage": "instant_meshes",
                "generated_files": {"instant_output": str(output_path)},
                "elapsed_seconds": elapsed,
                "stats": self.get_mesh_stats(),
                "mesh": self.mesh,
            }

    def _remesh_quadwild_bimdf(
        self,
        quadwild_stage1_config_rel: str,
        quadwild_stage2_config_rel: str,
        quadwild_do_remesh: bool,
        quadwild_sharp_feature_threshold: float,
        quadwild_alpha: float,
        quadwild_scale_factor: float,
        quadwild_use_original_input_file: bool,
        quadwild_pre_repair_workflow: bool,
        quadwild_cleanup_method: str,
        quadwild_fill_holes: bool,
        auto_reduce_after_quadwild: bool,
        auto_reduce_backend: str,
        auto_reduce_target_faces: int,
        auto_reduce_boundary_weight: float,
        auto_reduce_cleanup: bool,
        post_decimate: bool,
        decimate_target_faces: int,
    ) -> dict[str, Any]:
        self._require_mesh()
        pipeline_t0 = time.perf_counter()

        if not self.quadwild_bimdf_runner.is_available():
            paths = self.quadwild_bimdf_runner.debug_paths()
            raise FileNotFoundError(
                "QuadWild-BiMDF backend is not available.\n"
                f"quadwild repo root: {paths['repo_root']}\n"
                f"quadwild binary: {paths['quadwild']}\n"
                f"quad_from_patches binary: {paths['quad_from_patches']}\n"
                f"lib dir: {paths['lib_dir']}\n"
                f"default stage1 config: {paths['default_stage1_config']}\n"
                f"default stage2 config: {paths['default_stage2_config']}"
            )

        used_original_file = False
        workflow_steps: list[str] = []
        pre_repair_reports: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="far_mesh_quadwild_prepare_") as prep_tmpdir:
            source_name = "input.obj"
            if self.filepath:
                original_name = Path(self.filepath).name
                if original_name:
                    source_name = original_name

            if (
                quadwild_use_original_input_file
                and not quadwild_pre_repair_workflow
                and self.filepath is not None
            ):
                original = Path(self.filepath).expanduser().resolve()
                if original.exists() and original.is_file() and original.suffix.lower() in {".obj", ".ply"}:
                    source_mesh_path = original
                    used_original_file = True
                else:
                    source_mesh_path = Path(prep_tmpdir) / (Path(source_name).stem + ".obj")
                    self.mesh.export(source_mesh_path)
                    workflow_steps.append(
                        "Original file could not be used directly; exported current mesh to OBJ for QuadWild-BiMDF."
                    )
            else:
                working_mesh = self.mesh.copy()

                if quadwild_pre_repair_workflow:
                    try:
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="pymeshfix",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "pymeshfix",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 1: repaired mesh with pymeshfix.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 1 note: {note}")
                    except Exception as exc:
                        workflow_steps.append(
                            f"Pre-workflow step 1: pymeshfix failed ({exc}); falling back to hybrid repair."
                        )
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="hybrid",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "hybrid",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 1 fallback: repaired mesh with hybrid.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 1 fallback note: {note}")

                    try:
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method=quadwild_cleanup_method,
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": quadwild_cleanup_method,
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append(
                            f"Pre-workflow step 2: cleanup pass with {quadwild_cleanup_method}."
                        )
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 2 note: {note}")
                    except Exception as exc:
                        workflow_steps.append(
                            f"Pre-workflow step 2: cleanup with {quadwild_cleanup_method} failed ({exc}); "
                            "falling back to trimesh cleanup."
                        )
                        report = self.repairer.clean_with_report(
                            working_mesh,
                            method="trimesh",
                            join_comp=True,
                            fill_holes=quadwild_fill_holes,
                            collect_inspection=False,
                        )
                        working_mesh = report.mesh
                        pre_repair_reports.append(
                            {
                                "requested_method": "trimesh",
                                "executed_method": report.executed_method,
                                "backend_chain": list(report.backend_chain),
                                "elapsed_seconds": report.elapsed_seconds,
                                "notes": list(report.notes),
                            }
                        )
                        workflow_steps.append("Pre-workflow step 2 fallback: cleanup pass with trimesh.")
                        for note in report.notes:
                            workflow_steps.append(f"Pre-workflow step 2 fallback note: {note}")

                source_mesh_path = Path(prep_tmpdir) / (Path(source_name).stem + ".obj")
                self._coerce_to_trimesh(working_mesh).export(source_mesh_path)
                used_original_file = False

            with tempfile.TemporaryDirectory(prefix="far_mesh_quadwild_bimdf_src_") as run_tmpdir:
                output_dir = Path(run_tmpdir) / "quadwild_bimdf_run"

                result = self.quadwild_bimdf_runner.run(
                    input_mesh_path=str(source_mesh_path),
                    output_dir=str(output_dir),
                    timeout_stage1=None,
                    timeout_stage2=None,
                    overwrite=True,
                    stage1_config_rel=quadwild_stage1_config_rel,
                    stage2_config_rel=quadwild_stage2_config_rel,
                    stage1_overrides={
                        "do_remesh": quadwild_do_remesh,
                        "sharp_feature_thr": quadwild_sharp_feature_threshold,
                        "alpha": quadwild_alpha,
                        "scaleFact": quadwild_scale_factor,
                    },
                    lifecycle=get_lifecycle_manager(),
                )

                if not result.final_mesh_path:
                    raise RuntimeError("QuadWild-BiMDF completed without a final quadrangulation mesh.")

                final_mesh_path = Path(result.final_mesh_path)
                loaded = trimesh.load(final_mesh_path, force="mesh")
                new_mesh = self._coerce_to_trimesh(loaded)

                if new_mesh.is_empty or len(new_mesh.faces) == 0:
                    raise ValueError("QuadWild-BiMDF produced an empty output mesh.")

                quadwild_output_faces = int(len(new_mesh.faces))
                quadwild_output_vertices = int(len(new_mesh.vertices))

                if post_decimate and not auto_reduce_after_quadwild:
                    auto_reduce_after_quadwild = True
                    auto_reduce_backend = "open3d"
                    auto_reduce_target_faces = decimate_target_faces
                    auto_reduce_boundary_weight = 5.0
                    auto_reduce_cleanup = True

                auto_reduce_applied = False
                auto_reduce_payload: dict[str, Any] | None = None
                auto_reduce_note: str | None = None

                if auto_reduce_after_quadwild and auto_reduce_target_faces > 0:
                    reduced_mesh, reduction = self.reducer.reduce(
                        mesh=new_mesh,
                        backend=auto_reduce_backend,
                        target_faces=auto_reduce_target_faces,
                        boundary_weight=auto_reduce_boundary_weight,
                        cleanup=auto_reduce_cleanup,
                    )
                    new_mesh = self._coerce_to_trimesh(reduced_mesh)
                    auto_reduce_applied = True
                    auto_reduce_note = (
                        "Auto reduction was applied after QuadWild-BiMDF using the dedicated reduction stage."
                    )
                    workflow_steps.append(
                        f"Auto reduction: {reduction.before_faces} -> "
                        f"{reduction.after_faces} faces using {auto_reduce_backend} "
                        f"in {reduction.elapsed_seconds:.2f}s."
                    )
                    auto_reduce_payload = {
                        "backend": auto_reduce_backend,
                        "target_faces": auto_reduce_target_faces,
                        "boundary_weight": auto_reduce_boundary_weight,
                        "cleanup": auto_reduce_cleanup,
                        "before_vertices": reduction.before_vertices,
                        "before_faces": reduction.before_faces,
                        "after_vertices": reduction.after_vertices,
                        "after_faces": reduction.after_faces,
                        "reduction_ratio": reduction.reduction_ratio,
                        "elapsed_seconds": reduction.elapsed_seconds,
                        "note": reduction.note,
                    }

                new_mesh.remove_unreferenced_vertices()
                self.mesh = new_mesh

                stable_copy_dir = Path(tempfile.mkdtemp(prefix="far_mesh_quadwild_bimdf_result_"))
                copied_outputs: dict[str, str] = {}

                for key, value in result.generated_files.items():
                    src = Path(value)
                    if src.exists() and src.is_file():
                        dst = stable_copy_dir / src.name
                        shutil.copy2(src, dst)
                        copied_outputs[key] = str(dst)

                if auto_reduce_applied:
                    final_copy = stable_copy_dir / (final_mesh_path.stem + "_reduced.obj")
                    new_mesh.export(final_copy)
                    copied_outputs["reduced_output"] = str(final_copy)
                    final_stage = "reduced_after_quadwild"
                else:
                    final_copy = stable_copy_dir / final_mesh_path.name
                    if final_mesh_path.exists():
                        shutil.copy2(final_mesh_path, final_copy)
                    final_stage = result.final_stage

                pipeline_elapsed = time.perf_counter() - pipeline_t0
                self.last_output_path = str(final_copy)

                return {
                    "backend": "quadwild_bimdf",
                    "output_path": str(final_copy),
                    "final_stage": final_stage,
                    "generated_files": copied_outputs,
                    "working_directory": result.working_directory,
                    "stage1_command": result.stage1_command,
                    "stage2_command": result.stage2_command,
                    "stage1_stdout": result.stage1_stdout,
                    "stage1_stderr": result.stage1_stderr,
                    "stage2_stdout": result.stage2_stdout,
                    "stage2_stderr": result.stage2_stderr,
                    "stage1_returncode": result.stage1_returncode,
                    "stage2_returncode": result.stage2_returncode,
                    "stage1_elapsed_seconds": result.stage1_elapsed_seconds,
                    "stage2_elapsed_seconds": result.stage2_elapsed_seconds,
                    "total_elapsed_seconds": result.total_elapsed_seconds,
                    "pipeline_total_elapsed_seconds": pipeline_elapsed,
                    "stage1_config_requested": result.stage1_config_requested,
                    "stage1_config_used": result.stage1_config_used,
                    "stage2_config_used": result.stage2_config_used,
                    "stage1_overrides_used": result.stage1_overrides_used,
                    "stage1_fallback_used": result.stage1_fallback_used,
                    "stage1_fallback_reason": result.stage1_fallback_reason,
                    "used_original_input_file": used_original_file,
                    "source_mesh_path": str(source_mesh_path),
                    "workflow_steps": workflow_steps,
                    "pre_repair_reports": pre_repair_reports,
                    "quadwild_output_vertices": quadwild_output_vertices,
                    "quadwild_output_faces": quadwild_output_faces,
                    "auto_reduce_applied": auto_reduce_applied,
                    "auto_reduce_payload": auto_reduce_payload,
                    "auto_reduce_note": auto_reduce_note,
                    "stats": self.get_mesh_stats(),
                    "mesh": self.mesh,
                }

    def _require_mesh(self) -> None:
        if self.mesh is None:
            raise ValueError("No mesh loaded.")

    @staticmethod
    def _coerce_to_trimesh(loaded: Any) -> trimesh.Trimesh:
        if isinstance(loaded, trimesh.Trimesh):
            return loaded

        if hasattr(loaded, "geometry") and loaded.geometry:
            geometries = [
                geom
                for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and not geom.is_empty
            ]
            if not geometries:
                raise ValueError("Loaded object does not contain a valid mesh.")
            if len(geometries) == 1:
                return geometries[0].copy()
            return trimesh.util.concatenate(geometries)

        raise ValueError("Loaded object is not a supported mesh.")
