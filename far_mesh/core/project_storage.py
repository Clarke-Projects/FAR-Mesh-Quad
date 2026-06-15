"""
Project/session storage helpers for FAR MESH Quad 3.

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
- create unsaved session folders
- create/open saved project folders
- own standard project/session directory layout
- resolve project-relative paths safely
- create per-operation preview/history/external-run folders
- read/write project/session metadata JSON atomically

Non-responsibilities:
- no mesh topology
- no preview generation
- no mesh mutation
- no undo/redo mutation
- no disk cleanup/deletion policy
- no disk usage probing

Disk usage reporting belongs to far_mesh.system.resource_probe.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION_PROJECT_STORAGE = 1

PROJECT_FILE_NAME = "project.json"
SESSION_STATE_FILE_NAME = "project_state.json"

SNAPSHOTS_DIR_NAME = "snapshots"
PREVIEWS_DIR_NAME = "previews"
HISTORY_DIR_NAME = "history"
TEMP_DIR_NAME = "temp"
EXTERNAL_RUNS_DIR_NAME = "external_runs"
LOGS_DIR_NAME = "logs"
EXPORTS_DIR_NAME = "exports"

STANDARD_DIR_NAMES = (
    SNAPSHOTS_DIR_NAME,
    PREVIEWS_DIR_NAME,
    HISTORY_DIR_NAME,
    TEMP_DIR_NAME,
    EXTERNAL_RUNS_DIR_NAME,
    LOGS_DIR_NAME,
    EXPORTS_DIR_NAME,
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ProjectStorageError(RuntimeError):
    """Raised when project/session storage cannot be created, read, or resolved."""


@dataclass(frozen=True)
class ProjectStorageLayout:
    """Resolved standard folder layout for a project/session root."""

    root: str
    metadata_path: str
    snapshots_dir: str
    previews_dir: str
    history_dir: str
    temp_dir: str
    external_runs_dir: str
    logs_dir: str
    exports_dir: str


@dataclass(frozen=True)
class ProjectStorage:
    """
    Core storage owner for one FAR MESH project or unsaved session.

    `root` is either:
    - an unsaved session folder under the user cache directory, or
    - a saved `.farmesh3` project folder.

    This object only manages paths and metadata. It does not write mesh files
    itself except for metadata JSON files.
    """

    root: Path
    is_saved_project: bool = False
    storage_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: _utc_timestamp())
    schema_version: int = SCHEMA_VERSION_PROJECT_STORAGE

    @classmethod
    def create_unsaved_session(
        cls,
        *,
        base_cache_dir: str | Path | None = None,
        session_id: str | None = None,
        timestamp: str | None = None,
        create: bool = True,
    ) -> "ProjectStorage":
        """
        Create a new unsaved session storage root.

        Default layout:
            ~/.cache/far_mesh_3/sessions/session_<timestamp>_<id>/

        If base_cache_dir is supplied, sessions are created under:
            <base_cache_dir>/sessions/session_<timestamp>_<id>/
        """

        sid = _safe_name(session_id or uuid.uuid4().hex[:12], fallback="session")
        ts = _safe_name(timestamp or _timestamp_for_path(), fallback="time")

        base = Path(base_cache_dir).expanduser() if base_cache_dir is not None else _default_cache_root()
        root = base / "sessions" / f"session_{ts}_{sid}"

        storage = cls(
            root=root.resolve(strict=False),
            is_saved_project=False,
            storage_id=sid,
            created_at=_utc_timestamp(),
        )

        if create:
            storage.ensure_layout()
            storage.write_metadata()

        return storage

    @classmethod
    def create_project(
        cls,
        project_root: str | Path,
        *,
        project_id: str | None = None,
        create: bool = True,
    ) -> "ProjectStorage":
        """
        Create or prepare a saved `.farmesh3` project folder.

        The folder name is not forced to end in `.farmesh3`, but callers should
        normally pass a path with that suffix for user-facing saved projects.
        """

        pid = _safe_name(project_id or uuid.uuid4().hex[:12], fallback="project")
        root = Path(project_root).expanduser().resolve(strict=False)

        storage = cls(
            root=root,
            is_saved_project=True,
            storage_id=pid,
            created_at=_utc_timestamp(),
        )

        if create:
            storage.ensure_layout()
            storage.write_metadata()

        return storage

    @classmethod
    def open_existing(
        cls,
        project_root: str | Path,
        *,
        is_saved_project: bool | None = None,
    ) -> "ProjectStorage":
        """
        Open an existing project/session root without creating missing folders.

        Metadata is read when available. Missing metadata is tolerated so older
        or partially-created roots can still be inspected.
        """

        root = Path(project_root).expanduser().resolve(strict=False)
        if not root.exists():
            raise ProjectStorageError(f"Project/session root does not exist: {root}")
        if not root.is_dir():
            raise ProjectStorageError(f"Project/session root is not a directory: {root}")

        saved = bool(is_saved_project) if is_saved_project is not None else (root.suffix == ".farmesh3")
        probe = cls(root=root, is_saved_project=saved)

        metadata = probe.read_metadata(required=False)
        if metadata:
            storage_id = str(metadata.get("storage_id") or probe.storage_id)
            created_at = str(metadata.get("created_at") or probe.created_at)
            schema_version = _schema_version_from_metadata(metadata, path=probe.metadata_path)
            saved = bool(metadata.get("is_saved_project", saved))
            return cls(
                root=root,
                is_saved_project=saved,
                storage_id=storage_id,
                created_at=created_at,
                schema_version=schema_version,
            )

        return probe

    @property
    def metadata_path(self) -> Path:
        return self.root / (PROJECT_FILE_NAME if self.is_saved_project else SESSION_STATE_FILE_NAME)

    @property
    def snapshots_dir(self) -> Path:
        return self.root / SNAPSHOTS_DIR_NAME

    @property
    def previews_dir(self) -> Path:
        return self.root / PREVIEWS_DIR_NAME

    @property
    def history_dir(self) -> Path:
        return self.root / HISTORY_DIR_NAME

    @property
    def temp_dir(self) -> Path:
        return self.root / TEMP_DIR_NAME

    @property
    def external_runs_dir(self) -> Path:
        return self.root / EXTERNAL_RUNS_DIR_NAME

    @property
    def logs_dir(self) -> Path:
        return self.root / LOGS_DIR_NAME

    @property
    def exports_dir(self) -> Path:
        return self.root / EXPORTS_DIR_NAME

    def layout(self) -> ProjectStorageLayout:
        return ProjectStorageLayout(
            root=str(self.root),
            metadata_path=str(self.metadata_path),
            snapshots_dir=str(self.snapshots_dir),
            previews_dir=str(self.previews_dir),
            history_dir=str(self.history_dir),
            temp_dir=str(self.temp_dir),
            external_runs_dir=str(self.external_runs_dir),
            logs_dir=str(self.logs_dir),
            exports_dir=str(self.exports_dir),
        )

    def ensure_layout(self) -> None:
        """Create the standard project/session directory layout."""

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for name in STANDARD_DIR_NAMES:
                (self.root / name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProjectStorageError(f"Could not create project/session layout at {self.root}: {exc}") from exc

    def to_metadata(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "storage_id": self.storage_id,
            "created_at": self.created_at,
            "is_saved_project": bool(self.is_saved_project),
            "root_name": self.root.name,
            "standard_dirs": list(STANDARD_DIR_NAMES),
        }

        if extra:
            metadata["extra"] = dict(extra)

        return metadata

    def write_metadata(self, *, extra: Mapping[str, Any] | None = None) -> None:
        """Atomically write project/session metadata JSON."""

        self.ensure_layout()
        _atomic_write_json(self.metadata_path, self.to_metadata(extra=extra))

    def read_metadata(self, *, required: bool = True) -> dict[str, Any]:
        """Read project/session metadata JSON."""

        path = self.metadata_path
        if not path.exists():
            if required:
                raise ProjectStorageError(f"Project/session metadata does not exist: {path}")
            return {}

        data = _read_json(path)
        _schema_version_from_metadata(data, path=path)
        return data

    def relative_path(self, path: str | Path) -> str:
        """Return a project-root-relative path string."""

        path_obj = Path(path).expanduser().resolve(strict=False)
        root = self.root.resolve(strict=False)

        try:
            return str(path_obj.relative_to(root))
        except ValueError as exc:
            raise ProjectStorageError(f"Path is outside project/session root: {path_obj}") from exc

    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve a project-relative path under this storage root.

        Absolute paths are only accepted if they are still inside the root.
        Relative paths must not escape the root with '..'.
        """

        raw = Path(path).expanduser()
        root = self.root.resolve(strict=False)

        resolved = raw.resolve(strict=False) if raw.is_absolute() else (root / raw).resolve(strict=False)

        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ProjectStorageError(f"Resolved path escapes project/session root: {path!r}") from exc

        return resolved

    def snapshot_path(self, name: str, *, suffix: str = ".ply") -> Path:
        """Return a safe path inside snapshots/. Does not write the file."""

        return self.snapshots_dir / _safe_filename(name, suffix=suffix)

    def create_preview_dir(self, operation_id: str) -> Path:
        """Create and return previews/preview_<operation_id>/ ."""

        name = _safe_name(operation_id, fallback="operation")
        path = self.previews_dir / f"preview_{name}"
        return _mkdir(path)

    def create_history_dir(self, operation_id: str) -> Path:
        """Create and return history/op_<operation_id>/ ."""

        name = _safe_name(operation_id, fallback="operation")
        path = self.history_dir / f"op_{name}"
        return _mkdir(path)

    def create_external_run_dir(self, operation_id: str) -> Path:
        """Create and return external_runs/run_<operation_id>/ ."""

        name = _safe_name(operation_id, fallback="operation")
        path = self.external_runs_dir / f"run_{name}"
        return _mkdir(path)

    def create_temp_dir(self, label: str | None = None) -> Path:
        """Create and return temp/<label>_<uuid>/ ."""

        safe_label = _safe_name(label or "temp", fallback="temp")
        path = self.temp_dir / f"{safe_label}_{uuid.uuid4().hex[:12]}"
        return _mkdir(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_cache_root() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "far_mesh_3"
    return Path.home() / ".cache" / "far_mesh_3"


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str, *, fallback: str) -> str:
    name = str(value or "").strip()
    name = name.replace(os.sep, "_")
    if os.altsep:
        name = name.replace(os.altsep, "_")
    name = _SAFE_NAME_RE.sub("_", name)
    name = name.strip("._-")
    return name or fallback


def _safe_filename(name: str, *, suffix: str) -> str:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    safe_name = _safe_name(name, fallback="file")
    if Path(safe_name).suffix.lower() != safe_suffix.lower():
        safe_name = f"{safe_name}{safe_suffix}"
    return safe_name


def _mkdir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProjectStorageError(f"Could not create directory {path}: {exc}") from exc
    return path


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(data), handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
    except OSError as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise ProjectStorageError(f"Could not write JSON file {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ProjectStorageError(f"Could not read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectStorageError(f"Invalid JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ProjectStorageError(f"Expected JSON object in {path}, got {type(data).__name__}.")

    return data


def _schema_version_from_metadata(metadata: Mapping[str, Any], *, path: Path) -> int:
    """
    Return the validated project-storage schema version from metadata.

    Missing schema_version is treated as the current schema for compatibility
    with early project/session metadata created before version hardening.
    Present schema_version values must be plain integers. Booleans, strings,
    floats, nulls, and other types are rejected so unsupported project metadata
    fails clearly instead of leaking a ValueError from int(...).
    """

    if "schema_version" not in metadata:
        return SCHEMA_VERSION_PROJECT_STORAGE

    raw_version = metadata.get("schema_version")

    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ProjectStorageError(
            "Invalid ProjectStorage schema_version in "
            f"{path}: expected integer, got {type(raw_version).__name__}."
        )

    _ensure_schema_supported(raw_version, path=path)
    return int(raw_version)


def _ensure_schema_supported(found: int, *, path: Path | None = None) -> None:
    location = f" in {path}" if path is not None else ""

    if found < 1:
        raise ProjectStorageError(
            f"ProjectStorage schema version {found}{location} is older than the minimum "
            "supported version 1."
        )

    if found > SCHEMA_VERSION_PROJECT_STORAGE:
        raise ProjectStorageError(
            f"ProjectStorage schema version {found}{location} is newer than supported version "
            f"{SCHEMA_VERSION_PROJECT_STORAGE}."
        )


__all__ = [
    "SCHEMA_VERSION_PROJECT_STORAGE",
    "PROJECT_FILE_NAME",
    "SESSION_STATE_FILE_NAME",
    "SNAPSHOTS_DIR_NAME",
    "PREVIEWS_DIR_NAME",
    "HISTORY_DIR_NAME",
    "TEMP_DIR_NAME",
    "EXTERNAL_RUNS_DIR_NAME",
    "LOGS_DIR_NAME",
    "EXPORTS_DIR_NAME",
    "STANDARD_DIR_NAMES",
    "ProjectStorageError",
    "ProjectStorageLayout",
    "ProjectStorage",
]
