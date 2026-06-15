"""
Disk-backed tool preview state models for FAR MESH Quad 3.

This module is intentionally pure core.

It must not import:
- PySide6
- MainWindow
- MainWindowUI
- viewport classes
- SelectionController
- MeshProcessor
- TaskRouter
- ExecutionManager

Responsibilities:
- MeshSnapshot: disk-backed mesh snapshot metadata + lazy loading
- ToolRegion: named input/output region used or produced by a tool
- ToolPreviewState: complete disk-backed preview record
- marker helpers
- JSON serialization helpers
- atomic mesh and JSON writes
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import trimesh


# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------

SCHEMA_VERSION_TOOL_PREVIEW_STATE = 1


# ---------------------------------------------------------------------------
# Snapshot roles
# ---------------------------------------------------------------------------

SNAPSHOT_ROLE_BASE = "base"
SNAPSHOT_ROLE_PREVIEW = "preview"
SNAPSHOT_ROLE_BEFORE = "before"
SNAPSHOT_ROLE_AFTER = "after"
SNAPSHOT_ROLE_PATCH = "patch"
SNAPSHOT_ROLE_REMOVED_REGION = "removed_region"
SNAPSHOT_ROLE_PROCESSED_REGION = "processed_region"
SNAPSHOT_ROLE_EXTERNAL_INPUT = "external_input"
SNAPSHOT_ROLE_EXTERNAL_OUTPUT = "external_output"


# ---------------------------------------------------------------------------
# Region kinds
# ---------------------------------------------------------------------------

REGION_KIND_WHOLE_MESH = "whole_mesh"
REGION_KIND_HOLE_BOUNDARY = "hole_boundary"
REGION_KIND_HOLE_PATCH = "hole_patch"
REGION_KIND_REMOVED_REGION = "removed_region"
REGION_KIND_SELECTED_FACES = "selected_faces"
REGION_KIND_SELECTED_VERTICES = "selected_vertices"
REGION_KIND_ROI_INPUT = "roi_input"
REGION_KIND_ROI_PROCESSED = "roi_processed"
REGION_KIND_BORE_REGION = "bore_region"
REGION_KIND_ZIPPER_CHAIN = "zipper_chain"
REGION_KIND_ZIPPER_BRIDGE = "zipper_bridge"
REGION_KIND_EXTERNAL_INPUT = "external_input"
REGION_KIND_EXTERNAL_OUTPUT = "external_output"


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

ROI_ONLY_PREVIEW_MARKER = "__ROI_ONLY_PREVIEW__"
HOLE_FILL_PREVIEW_MARKER = "__HOLE_FILL_PREVIEW__"
WHOLE_MESH_OPERATION_MARKER = "__WHOLE_MESH_OPERATION__"
LOCAL_REGION_OPERATION_MARKER = "__LOCAL_REGION_OPERATION__"
OPEN3D_FILL_PREVIEW_MARKER = "__OPEN3D_FILL_PREVIEW__"

# Compatibility only.
# Do not create this marker for new previews.
OLD_HOLE_FILL_PREVIEW_ONLY_MARKER = "__HOLE_FILL_PREVIEW_ONLY__"


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

SUPPORTED_SNAPSHOT_FORMATS = {"ply"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MeshSnapshotError(RuntimeError):
    """Raised when capturing, loading, validating, or serializing a mesh snapshot fails."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalize_file_format(file_format: str) -> str:
    normalized = file_format.strip().lower().lstrip(".")
    if not normalized:
        raise MeshSnapshotError("Snapshot file format is empty.")

    if normalized not in SUPPORTED_SNAPSHOT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_SNAPSHOT_FORMATS))
        raise MeshSnapshotError(
            f"Unsupported internal snapshot format: {file_format!r}. "
            f"Supported formats: {supported}"
        )

    return normalized


def _safe_snapshot_name(
    *,
    role: str,
    name: str | None,
    file_format: str,
) -> str:
    """
    Build a safe snapshot file name.

    If name is provided and has no suffix, the file format suffix is appended.
    If name is omitted, '<role>_mesh.<file_format>' is used.
    """

    suffix = f".{file_format}"

    if name is None:
        base_name = f"{role}_mesh{suffix}"
    else:
        base_name = name.strip()
        if not base_name:
            raise MeshSnapshotError("Snapshot file name is empty.")

        path_name = Path(base_name)
        if path_name.suffix.lower() != suffix:
            base_name = f"{base_name}{suffix}"

    # Keep this conservative. ProjectStorage can do richer sanitization later.
    illegal = {os.sep}
    if os.altsep:
        illegal.add(os.altsep)

    for char in illegal:
        if char in base_name:
            raise MeshSnapshotError(
                f"Snapshot file name must not contain path separators: {base_name!r}"
            )

    return base_name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MeshSnapshotError(f"Could not checksum snapshot {path}: {exc}") from exc

    return digest.hexdigest()


def _path_for_json(path: str | Path, *, base_dir: Path | None = None) -> str:
    """
    Convert a path to a JSON-safe string.

    If base_dir is provided and path is inside base_dir, return a project-relative path.
    Otherwise return the original/absolute path string.
    """

    path_obj = Path(path)

    if base_dir is None:
        return str(path_obj)

    try:
        resolved_path = path_obj.resolve(strict=False)
        resolved_base = base_dir.resolve(strict=False)
        return str(resolved_path.relative_to(resolved_base))
    except ValueError:
        return str(path_obj)


def _path_from_json(path_value: str, *, base_dir: Path | None = None) -> str:
    """
    Resolve a JSON path value.

    If base_dir is provided and the stored path is relative, resolve it under base_dir.
    Absolute paths are preserved.
    """

    path_obj = Path(path_value)

    if base_dir is not None and not path_obj.is_absolute():
        return str((base_dir / path_obj).resolve(strict=False))

    return str(path_obj)


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """
    Atomically write JSON to disk.

    Writes:
        <target>.tmp

    Then renames:
        <target>.tmp -> <target>
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

        tmp_path.replace(path)

    except OSError as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

        raise MeshSnapshotError(f"Could not write JSON file {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise MeshSnapshotError(f"Could not read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MeshSnapshotError(f"Invalid JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MeshSnapshotError(f"Expected JSON object in {path}, got {type(data).__name__}.")

    return data


def _tuple_ints(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()

    if not isinstance(value, (list, tuple)):
        raise MeshSnapshotError(f"{field_name} must be a list/tuple of integers.")

    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise MeshSnapshotError(f"{field_name} contains non-integer values.") from exc


def _tuple_edges(value: Any, *, field_name: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()

    if not isinstance(value, (list, tuple)):
        raise MeshSnapshotError(f"{field_name} must be a list/tuple of edge pairs.")

    edges: list[tuple[int, int]] = []

    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise MeshSnapshotError(
                f"{field_name} must contain pairs like [vertex_a, vertex_b]."
            )

        try:
            edge = (int(item[0]), int(item[1]))
        except (TypeError, ValueError) as exc:
            raise MeshSnapshotError(f"{field_name} contains non-integer edge values.") from exc

        edges.append(edge)

    return tuple(edges)


def _ensure_schema_supported(
    *,
    found: int,
    supported: int,
    label: str,
) -> None:
    if found > supported:
        raise MeshSnapshotError(
            f"{label} schema version {found} is newer than supported version {supported}."
        )


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """
    Load a mesh from disk as a trimesh.Trimesh.

    Uses process=False to preserve stored topology/counts as closely as possible.
    """

    if not path.exists():
        raise MeshSnapshotError(f"Snapshot file does not exist: {path}")

    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise MeshSnapshotError(f"Could not load mesh snapshot {path}: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        try:
            loaded = loaded.dump(concatenate=True)
        except Exception as exc:
            raise MeshSnapshotError(f"Could not convert scene snapshot to mesh: {path}") from exc

    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshSnapshotError(
            f"Loaded snapshot is not a trimesh.Trimesh: {path} "
            f"({type(loaded).__name__})"
        )

    return loaded


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

def is_internal_marker(value: str) -> bool:
    return value.startswith("__") and value.endswith("__")


def is_commit_blocking_marker(value: str) -> bool:
    return value == ROI_ONLY_PREVIEW_MARKER


def has_commit_blocking_marker(markers: Iterable[str]) -> bool:
    return any(is_commit_blocking_marker(marker) for marker in markers)


# ---------------------------------------------------------------------------
# MeshSnapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshSnapshot:
    """
    Disk-backed mesh snapshot.

    The snapshot stores metadata and a path. It does not cache the mesh in memory.
    Use load() to read the mesh from disk when needed.
    """

    path: str
    role: str
    vertex_count: int
    face_count: int
    format: str = "ply"
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        mesh: trimesh.Trimesh,
        storage_dir: Path,
        *,
        role: str,
        name: str | None = None,
        file_format: str = "ply",
        metadata: dict[str, Any] | None = None,
        compute_checksum: bool = False,
    ) -> "MeshSnapshot":
        """
        Write a mesh snapshot to disk and return its metadata record.

        The write is atomic:
            <target>.tmp is exported first
            then replaced into <target>

        If export fails, the final path is left absent or unchanged.
        """

        if not isinstance(mesh, trimesh.Trimesh):
            raise MeshSnapshotError(
                f"MeshSnapshot.capture expected trimesh.Trimesh, got {type(mesh).__name__}."
            )

        normalized_format = _normalize_file_format(file_format)
        filename = _safe_snapshot_name(
            role=role,
            name=name,
            file_format=normalized_format,
        )

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        final_path = storage_dir / filename
        tmp_path = final_path.with_name(final_path.name + ".tmp")

        try:
            mesh.export(tmp_path, file_type=normalized_format)
            tmp_path.replace(final_path)
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

            raise MeshSnapshotError(
                f"Could not capture mesh snapshot {final_path}: {exc}"
            ) from exc

        checksum = _sha256_file(final_path) if compute_checksum else None

        return cls(
            path=str(final_path),
            role=role,
            vertex_count=int(len(mesh.vertices)),
            face_count=int(len(mesh.faces)),
            format=normalized_format,
            checksum=checksum,
            metadata=dict(metadata or {}),
        )

    def load(
        self,
        *,
        validate: bool = True,
        verify_checksum: bool = False,
    ) -> trimesh.Trimesh:
        """
        Load this snapshot from disk.

        By default, validates vertex and face counts against stored metadata.
        Checksum verification is optional and requires checksum to be present.
        """

        file_format = _normalize_file_format(self.format)
        path = Path(self.path)

        if path.suffix.lower() != f".{file_format}":
            raise MeshSnapshotError(
                f"Snapshot path suffix does not match format: "
                f"path={path}, format={self.format!r}"
            )

        if verify_checksum:
            if not self.checksum:
                raise MeshSnapshotError(
                    f"Cannot verify checksum for snapshot without checksum: {path}"
                )

            actual_checksum = _sha256_file(path)
            if actual_checksum != self.checksum:
                raise MeshSnapshotError(
                    f"Snapshot checksum mismatch for {path}: "
                    f"expected {self.checksum}, got {actual_checksum}"
                )

        mesh = _load_trimesh(path)

        if validate:
            actual_vertices = int(len(mesh.vertices))
            actual_faces = int(len(mesh.faces))

            if actual_vertices != int(self.vertex_count):
                raise MeshSnapshotError(
                    f"Snapshot vertex count mismatch for {path}: "
                    f"expected {self.vertex_count}, got {actual_vertices}"
                )

            if actual_faces != int(self.face_count):
                raise MeshSnapshotError(
                    f"Snapshot face count mismatch for {path}: "
                    f"expected {self.face_count}, got {actual_faces}"
                )

        return mesh

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        return {
            "path": _path_for_json(self.path, base_dir=base_dir),
            "role": self.role,
            "vertex_count": int(self.vertex_count),
            "face_count": int(self.face_count),
            "format": self.format,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "MeshSnapshot":
        try:
            raw_path = str(data["path"])
            role = str(data["role"])
            vertex_count = int(data["vertex_count"])
            face_count = int(data["face_count"])
        except KeyError as exc:
            raise MeshSnapshotError(f"Missing MeshSnapshot field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise MeshSnapshotError(f"Invalid MeshSnapshot data: {data!r}") from exc

        file_format = _normalize_file_format(str(data.get("format", "ply")))

        checksum_value = data.get("checksum")
        checksum = None if checksum_value is None else str(checksum_value)

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise MeshSnapshotError("MeshSnapshot metadata must be a JSON object.")

        return cls(
            path=_path_from_json(raw_path, base_dir=base_dir),
            role=role,
            vertex_count=vertex_count,
            face_count=face_count,
            format=file_format,
            checksum=checksum,
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------------
# ToolRegion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolRegion:
    """
    Meaningful local or whole-mesh region used or produced by a tool.

    Examples:
    - selected faces
    - hole boundary
    - generated hole patch
    - removed region
    - external input/output region
    """

    name: str
    kind: str

    mesh_snapshot: MeshSnapshot | None = None

    face_ids: tuple[int, ...] = ()
    vertex_ids: tuple[int, ...] = ()
    edge_ids: tuple[tuple[int, int], ...] = ()

    new_face_ids: tuple[int, ...] = ()
    new_vertex_ids: tuple[int, ...] = ()

    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mesh_snapshot": (
                self.mesh_snapshot.to_dict(base_dir=base_dir)
                if self.mesh_snapshot is not None
                else None
            ),
            "face_ids": list(self.face_ids),
            "vertex_ids": list(self.vertex_ids),
            "edge_ids": [list(edge) for edge in self.edge_ids],
            "new_face_ids": list(self.new_face_ids),
            "new_vertex_ids": list(self.new_vertex_ids),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ToolRegion":
        try:
            name = str(data["name"])
            kind = str(data["kind"])
        except KeyError as exc:
            raise MeshSnapshotError(f"Missing ToolRegion field: {exc.args[0]}") from exc

        snapshot_data = data.get("mesh_snapshot")
        if snapshot_data is not None:
            if not isinstance(snapshot_data, Mapping):
                raise MeshSnapshotError("ToolRegion mesh_snapshot must be an object or null.")

            mesh_snapshot = MeshSnapshot.from_dict(snapshot_data, base_dir=base_dir)
        else:
            mesh_snapshot = None

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise MeshSnapshotError("ToolRegion metadata must be a JSON object.")

        return cls(
            name=name,
            kind=kind,
            mesh_snapshot=mesh_snapshot,
            face_ids=_tuple_ints(data.get("face_ids"), field_name="face_ids"),
            vertex_ids=_tuple_ints(data.get("vertex_ids"), field_name="vertex_ids"),
            edge_ids=_tuple_edges(data.get("edge_ids"), field_name="edge_ids"),
            new_face_ids=_tuple_ints(data.get("new_face_ids"), field_name="new_face_ids"),
            new_vertex_ids=_tuple_ints(data.get("new_vertex_ids"), field_name="new_vertex_ids"),
            source=str(data.get("source", "")),
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------------
# ToolPreviewState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolPreviewState:
    """
    Complete disk-backed preview record before commit.

    The full preview mesh and base mesh are stored as MeshSnapshots.
    Input/output regions describe meaningful local regions such as hole boundary
    and generated patch.
    """

    operation_id: str
    operation: str

    base_snapshot: MeshSnapshot
    preview_snapshot: MeshSnapshot

    input_regions: tuple[ToolRegion, ...] = ()
    output_regions: tuple[ToolRegion, ...] = ()

    committable: bool = False

    markers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: int = SCHEMA_VERSION_TOOL_PREVIEW_STATE

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "operation_id": self.operation_id,
            "operation": self.operation,
            "base_snapshot": self.base_snapshot.to_dict(base_dir=base_dir),
            "preview_snapshot": self.preview_snapshot.to_dict(base_dir=base_dir),
            "input_regions": [
                region.to_dict(base_dir=base_dir)
                for region in self.input_regions
            ],
            "output_regions": [
                region.to_dict(base_dir=base_dir)
                for region in self.output_regions
            ],
            "committable": bool(self.committable),
            "markers": list(self.markers),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ToolPreviewState":
        schema_version = int(
            data.get("schema_version", SCHEMA_VERSION_TOOL_PREVIEW_STATE)
        )
        _ensure_schema_supported(
            found=schema_version,
            supported=SCHEMA_VERSION_TOOL_PREVIEW_STATE,
            label="ToolPreviewState",
        )

        try:
            operation_id = str(data["operation_id"])
            operation = str(data["operation"])
            base_snapshot_data = data["base_snapshot"]
            preview_snapshot_data = data["preview_snapshot"]
        except KeyError as exc:
            raise MeshSnapshotError(
                f"Missing ToolPreviewState field: {exc.args[0]}"
            ) from exc

        if not isinstance(base_snapshot_data, Mapping):
            raise MeshSnapshotError("ToolPreviewState base_snapshot must be an object.")

        if not isinstance(preview_snapshot_data, Mapping):
            raise MeshSnapshotError("ToolPreviewState preview_snapshot must be an object.")

        input_region_data = data.get("input_regions") or []
        output_region_data = data.get("output_regions") or []

        if not isinstance(input_region_data, list):
            raise MeshSnapshotError("ToolPreviewState input_regions must be a list.")

        if not isinstance(output_region_data, list):
            raise MeshSnapshotError("ToolPreviewState output_regions must be a list.")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise MeshSnapshotError("ToolPreviewState metadata must be a JSON object.")

        markers = data.get("markers") or ()
        notes = data.get("notes") or ()

        if not isinstance(markers, (list, tuple)):
            raise MeshSnapshotError("ToolPreviewState markers must be a list/tuple.")

        if not isinstance(notes, (list, tuple)):
            raise MeshSnapshotError("ToolPreviewState notes must be a list/tuple.")

        return cls(
            operation_id=operation_id,
            operation=operation,
            base_snapshot=MeshSnapshot.from_dict(
                base_snapshot_data,
                base_dir=base_dir,
            ),
            preview_snapshot=MeshSnapshot.from_dict(
                preview_snapshot_data,
                base_dir=base_dir,
            ),
            input_regions=tuple(
                ToolRegion.from_dict(region, base_dir=base_dir)
                for region in input_region_data
            ),
            output_regions=tuple(
                ToolRegion.from_dict(region, base_dir=base_dir)
                for region in output_region_data
            ),
            committable=bool(data.get("committable", False)),
            markers=tuple(str(marker) for marker in markers),
            notes=tuple(str(note) for note in notes),
            metadata=dict(metadata),
            schema_version=schema_version,
        )

    def write_json(self, path: Path, *, base_dir: Path | None = None) -> None:
        _atomic_write_json(path, self.to_dict(base_dir=base_dir))

    @classmethod
    def read_json(
        cls,
        path: Path,
        *,
        base_dir: Path | None = None,
    ) -> "ToolPreviewState":
        return cls.from_dict(_read_json(path), base_dir=base_dir)


__all__ = [
    # schema
    "SCHEMA_VERSION_TOOL_PREVIEW_STATE",

    # snapshot roles
    "SNAPSHOT_ROLE_BASE",
    "SNAPSHOT_ROLE_PREVIEW",
    "SNAPSHOT_ROLE_BEFORE",
    "SNAPSHOT_ROLE_AFTER",
    "SNAPSHOT_ROLE_PATCH",
    "SNAPSHOT_ROLE_REMOVED_REGION",
    "SNAPSHOT_ROLE_PROCESSED_REGION",
    "SNAPSHOT_ROLE_EXTERNAL_INPUT",
    "SNAPSHOT_ROLE_EXTERNAL_OUTPUT",

    # region kinds
    "REGION_KIND_WHOLE_MESH",
    "REGION_KIND_HOLE_BOUNDARY",
    "REGION_KIND_HOLE_PATCH",
    "REGION_KIND_REMOVED_REGION",
    "REGION_KIND_SELECTED_FACES",
    "REGION_KIND_SELECTED_VERTICES",
    "REGION_KIND_ROI_INPUT",
    "REGION_KIND_ROI_PROCESSED",
    "REGION_KIND_BORE_REGION",
    "REGION_KIND_ZIPPER_CHAIN",
    "REGION_KIND_ZIPPER_BRIDGE",
    "REGION_KIND_EXTERNAL_INPUT",
    "REGION_KIND_EXTERNAL_OUTPUT",

    # markers
    "ROI_ONLY_PREVIEW_MARKER",
    "HOLE_FILL_PREVIEW_MARKER",
    "WHOLE_MESH_OPERATION_MARKER",
    "LOCAL_REGION_OPERATION_MARKER",
    "OPEN3D_FILL_PREVIEW_MARKER",
    "OLD_HOLE_FILL_PREVIEW_ONLY_MARKER",

    # exceptions
    "MeshSnapshotError",

    # dataclasses
    "MeshSnapshot",
    "ToolRegion",
    "ToolPreviewState",

    # marker helpers
    "is_internal_marker",
    "is_commit_blocking_marker",
    "has_commit_blocking_marker",
]
