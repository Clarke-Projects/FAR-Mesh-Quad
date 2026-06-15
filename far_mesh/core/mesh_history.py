"""
Disk-backed mesh history models for FAR MESH Quad 3.

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
- MeshHistoryEntry: committed operation record for undo/redo
- JSON serialization helpers
- atomic history_entry.json writes

Important authority rule:
MeshHistoryEntry does not mutate application mesh state.
Undo/redo mutation remains MeshProcessor authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .tool_preview_state import MeshSnapshot, MeshSnapshotError, ToolRegion


# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------

SCHEMA_VERSION_MESH_HISTORY_ENTRY = 1


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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

        raise MeshSnapshotError(f"Could not write mesh history JSON file {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise MeshSnapshotError(f"Could not read mesh history JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MeshSnapshotError(f"Invalid mesh history JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MeshSnapshotError(
            f"Expected JSON object in {path}, got {type(data).__name__}."
        )

    return data


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


def _tuple_strs(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()

    if not isinstance(value, (list, tuple)):
        raise MeshSnapshotError(f"{field_name} must be a list/tuple of strings.")

    return tuple(str(item) for item in value)


def _regions_from_json(
    value: Any,
    *,
    base_dir: Path | None,
    field_name: str,
) -> tuple[ToolRegion, ...]:
    if value is None:
        return ()

    if not isinstance(value, list):
        raise MeshSnapshotError(f"{field_name} must be a list.")

    return tuple(ToolRegion.from_dict(item, base_dir=base_dir) for item in value)


# ---------------------------------------------------------------------------
# MeshHistoryEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshHistoryEntry:
    """
    Disk-backed committed mesh operation record.

    This record stores snapshot references and metadata only.
    It does not cache full meshes and does not mutate MeshProcessor.mesh.

    Undo meaning:
        MeshProcessor loads before_snapshot and assigns self.mesh.

    Redo meaning:
        MeshProcessor loads after_snapshot and assigns self.mesh.
    """

    operation_id: str
    operation: str
    history_dir: str

    before_snapshot: MeshSnapshot
    after_snapshot: MeshSnapshot

    input_regions: tuple[ToolRegion, ...] = ()
    output_regions: tuple[ToolRegion, ...] = ()

    notes: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    schema_version: int = SCHEMA_VERSION_MESH_HISTORY_ENTRY

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "operation_id": self.operation_id,
            "operation": self.operation,
            "history_dir": _path_for_json(self.history_dir, base_dir=base_dir),
            "before_snapshot": self.before_snapshot.to_dict(base_dir=base_dir),
            "after_snapshot": self.after_snapshot.to_dict(base_dir=base_dir),
            "input_regions": [
                region.to_dict(base_dir=base_dir)
                for region in self.input_regions
            ],
            "output_regions": [
                region.to_dict(base_dir=base_dir)
                for region in self.output_regions
            ],
            "notes": list(self.notes),
            "markers": list(self.markers),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "MeshHistoryEntry":
        schema_version = int(
            data.get("schema_version", SCHEMA_VERSION_MESH_HISTORY_ENTRY)
        )
        _ensure_schema_supported(
            found=schema_version,
            supported=SCHEMA_VERSION_MESH_HISTORY_ENTRY,
            label="MeshHistoryEntry",
        )

        try:
            operation_id = str(data["operation_id"])
            operation = str(data["operation"])
            history_dir = str(data["history_dir"])
            before_snapshot_data = data["before_snapshot"]
            after_snapshot_data = data["after_snapshot"]
        except KeyError as exc:
            raise MeshSnapshotError(
                f"Missing MeshHistoryEntry field: {exc.args[0]}"
            ) from exc

        if not isinstance(before_snapshot_data, Mapping):
            raise MeshSnapshotError("MeshHistoryEntry before_snapshot must be an object.")

        if not isinstance(after_snapshot_data, Mapping):
            raise MeshSnapshotError("MeshHistoryEntry after_snapshot must be an object.")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise MeshSnapshotError("MeshHistoryEntry metadata must be a JSON object.")

        timestamp = str(data.get("timestamp") or datetime.now().isoformat(timespec="seconds"))

        return cls(
            operation_id=operation_id,
            operation=operation,
            history_dir=_path_from_json(history_dir, base_dir=base_dir),
            before_snapshot=MeshSnapshot.from_dict(
                before_snapshot_data,
                base_dir=base_dir,
            ),
            after_snapshot=MeshSnapshot.from_dict(
                after_snapshot_data,
                base_dir=base_dir,
            ),
            input_regions=_regions_from_json(
                data.get("input_regions"),
                base_dir=base_dir,
                field_name="input_regions",
            ),
            output_regions=_regions_from_json(
                data.get("output_regions"),
                base_dir=base_dir,
                field_name="output_regions",
            ),
            notes=_tuple_strs(data.get("notes"), field_name="notes"),
            markers=_tuple_strs(data.get("markers"), field_name="markers"),
            metadata=dict(metadata),
            timestamp=timestamp,
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
    ) -> "MeshHistoryEntry":
        return cls.from_dict(_read_json(path), base_dir=base_dir)


__all__ = [
    "SCHEMA_VERSION_MESH_HISTORY_ENTRY",
    "MeshHistoryEntry",
]
