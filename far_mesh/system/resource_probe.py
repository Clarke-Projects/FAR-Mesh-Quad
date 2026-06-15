"""
Launcher-neutral system resource probe for FAR Mesh Quad 3.

Reads effective CPU/memory limits from procfs, affinity, cgroup v1/v2.
Also provides read-only project/session disk usage probing for FAR MESH
project storage.

Important storage rule:
resource_probe.py reports usage only. It must not delete files, clean caches,
trim history, or mutate project/session folders.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass(slots=True)
class CgroupProbe:
    version: str = "none"
    mount_point: Optional[str] = None
    relative_path: Optional[str] = None

    # v2
    cpu_max_raw: Optional[str] = None
    cpu_quota_us: Optional[int] = None
    cpu_period_us: Optional[int] = None

    # v1
    cpu_cfs_quota_us: Optional[int] = None
    cpu_cfs_period_us: Optional[int] = None

    # shared
    quota_cpu_count: Optional[float] = None
    cpuset_raw: Optional[str] = None
    cpuset_cpu_ids: tuple[int, ...] = ()
    cpuset_cpu_count: Optional[int] = None
    memory_max_bytes: Optional[int] = None
    memory_current_bytes: Optional[int] = None
    memory_remaining_bytes: Optional[int] = None
    controller_mounts: dict[str, str] = field(default_factory=dict)
    controller_paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CpuResources:
    host_cpu_count: Optional[int] = None
    process_cpu_count: Optional[int] = None
    affinity_cpu_ids: tuple[int, ...] = ()
    affinity_cpu_count: Optional[int] = None
    cgroup_cpuset_cpu_ids: tuple[int, ...] = ()
    cgroup_cpuset_cpu_count: Optional[int] = None
    cgroup_quota_cpu_count: Optional[float] = None
    effective_cpu_capacity: Optional[float] = None
    recommended_worker_count: Optional[int] = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryResources:
    mem_total_bytes: Optional[int] = None
    mem_available_bytes: Optional[int] = None
    cgroup_memory_max_bytes: Optional[int] = None
    cgroup_memory_current_bytes: Optional[int] = None
    cgroup_memory_remaining_bytes: Optional[int] = None
    effective_available_bytes: Optional[int] = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeContext:
    in_container: bool = False
    systemd_hint: Optional[str] = None
    cgroup_namespace_hint: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SystemResources:
    cpu: CpuResources
    memory: MemoryResources
    cgroup: CgroupProbe
    runtime: RuntimeContext


@dataclass(slots=True, frozen=True)
class DiskUsageFile:
    """Small record for one large file found during disk usage probing."""

    path: str
    bytes: int


@dataclass(slots=True)
class ProjectDiskUsage:
    """
    Read-only disk usage summary for a FAR MESH session/project folder.

    This model intentionally contains only reporting fields. Cleanup policy
    belongs elsewhere and must be reference-aware once undo/redo and project
    storage are fully implemented.
    """

    project_root: str

    project_root_bytes: int = 0
    snapshots_bytes: int = 0
    previews_bytes: int = 0
    history_bytes: int = 0
    temp_bytes: int = 0
    external_runs_bytes: int = 0
    logs_bytes: int = 0
    exports_bytes: int = 0
    other_bytes: int = 0

    file_count: int = 0
    dir_count: int = 0
    symlink_count: int = 0
    unreadable_count: int = 0

    missing: bool = False
    is_file: bool = False

    largest_files: tuple[DiskUsageFile, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def known_bucket_bytes(self) -> int:
        return (
            self.snapshots_bytes
            + self.previews_bytes
            + self.history_bytes
            + self.temp_bytes
            + self.external_runs_bytes
            + self.logs_bytes
            + self.exports_bytes
        )


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------

def probe_system_resources() -> SystemResources:
    """Main function – returns a fully populated SystemResources snapshot."""
    cpu = CpuResources()
    memory = MemoryResources()
    cgroup = CgroupProbe()
    runtime = RuntimeContext()

    _fill_basic_cpu(cpu)
    _fill_meminfo(memory)
    _fill_cgroup_probe(cgroup)
    _derive_cpu_limits(cpu, cgroup)
    _derive_memory_limits(memory, cgroup)
    _detect_runtime_context(runtime, cgroup)

    return SystemResources(cpu=cpu, memory=memory, cgroup=cgroup, runtime=runtime)


def probe_project_disk_usage(
    project_root: str | Path,
    *,
    largest_files_limit: int = 10,
) -> ProjectDiskUsage:
    """
    Return read-only disk usage for a FAR MESH project/session folder.

    The expected top-level storage buckets are:

        snapshots/
        previews/
        history/
        temp/
        external_runs/
        logs/
        exports/

    Any other file/folder is counted as other_bytes.

    This function does not follow symlinked directories. Symlinks are counted
    by their own lstat size and recorded as symlinks so project usage cannot
    accidentally walk outside the project/session folder.
    """

    root = Path(project_root).expanduser()
    usage = ProjectDiskUsage(project_root=str(root.resolve(strict=False)))

    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        usage.missing = True
        usage.notes.append(f"project root does not exist: {root}")
        return usage
    except OSError as exc:
        usage.unreadable_count += 1
        usage.notes.append(f"could not stat project root {root}: {exc}")
        return usage

    if root.is_symlink():
        usage.symlink_count += 1
        usage.project_root_bytes = int(root_stat.st_size)
        usage.other_bytes = usage.project_root_bytes
        usage.notes.append("project root is a symlink; target was not followed")
        return usage

    if root.is_file():
        size = int(root_stat.st_size)
        usage.is_file = True
        usage.file_count = 1
        usage.project_root_bytes = size
        usage.other_bytes = size
        usage.largest_files = (DiskUsageFile(path=root.name, bytes=size),)
        usage.notes.append("project root is a file, not a directory")
        return usage

    if not root.is_dir():
        usage.notes.append(f"project root is neither file nor directory: {root}")
        return usage

    largest_limit = max(0, int(largest_files_limit))
    largest_files: list[DiskUsageFile] = []

    stack: list[Path] = [root]

    while stack:
        current = stack.pop()

        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)

                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        usage.unreadable_count += 1
                        usage.notes.append(f"could not stat {entry_path}: {exc}")
                        continue

                    size = int(stat.st_size)

                    try:
                        is_symlink = entry.is_symlink()
                    except OSError:
                        is_symlink = False

                    if is_symlink:
                        usage.symlink_count += 1
                        usage.file_count += 1
                        _add_project_usage_bytes(usage, root, entry_path, size)
                        _maybe_record_largest_file(
                            largest_files,
                            root,
                            entry_path,
                            size,
                            largest_limit,
                        )
                        continue

                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError as exc:
                        usage.unreadable_count += 1
                        usage.notes.append(f"could not inspect directory flag for {entry_path}: {exc}")
                        continue

                    if is_dir:
                        usage.dir_count += 1
                        stack.append(entry_path)
                        continue

                    try:
                        is_file = entry.is_file(follow_symlinks=False)
                    except OSError:
                        is_file = False

                    if is_file:
                        usage.file_count += 1
                        _add_project_usage_bytes(usage, root, entry_path, size)
                        _maybe_record_largest_file(
                            largest_files,
                            root,
                            entry_path,
                            size,
                            largest_limit,
                        )
                    else:
                        # Socket, device, FIFO, or unusual filesystem node.
                        usage.file_count += 1
                        _add_project_usage_bytes(usage, root, entry_path, size)

        except OSError as exc:
            usage.unreadable_count += 1
            usage.notes.append(f"could not read directory {current}: {exc}")

    usage.largest_files = tuple(
        sorted(largest_files, key=lambda item: item.bytes, reverse=True)
    )

    if usage.unreadable_count:
        usage.notes.append(f"unreadable paths: {usage.unreadable_count}")

    return usage


# ----------------------------------------------------------------------
# Basic probing
# ----------------------------------------------------------------------

def _fill_basic_cpu(cpu: CpuResources) -> None:
    cpu.host_cpu_count = os.cpu_count()
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        try:
            cpu.process_cpu_count = process_cpu_count()
        except Exception as exc:
            cpu.notes.append(f"os.process_cpu_count() failed: {exc}")
    try:
        affinity = sorted(os.sched_getaffinity(0))
        cpu.affinity_cpu_ids = tuple(affinity)
        cpu.affinity_cpu_count = len(affinity)
    except Exception as exc:
        cpu.notes.append(f"os.sched_getaffinity(0) unavailable: {exc}")


def _fill_meminfo(memory: MemoryResources) -> None:
    text = _read_text("/proc/meminfo")
    if text is None:
        memory.notes.append("could not read /proc/meminfo")
        return
    parsed = _parse_meminfo(text)
    memory.mem_total_bytes = parsed.get("MemTotal")
    memory.mem_available_bytes = parsed.get("MemAvailable")
    if memory.mem_available_bytes is None:
        memory.mem_available_bytes = parsed.get("MemFree")
        memory.notes.append("MemAvailable missing; using MemFree as fallback")


# ----------------------------------------------------------------------
# Cgroup probing (v2 + v1 fallback)
# ----------------------------------------------------------------------

@dataclass(slots=True)
class _ProcSelfCgroup:
    v2_path: Optional[str] = None
    v1_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _MountInfo:
    mount_point: str
    fs_type: str
    mount_source: str
    super_options: str
    optional_fields: tuple[str, ...]


def _fill_cgroup_probe(cgroup: CgroupProbe) -> None:
    # Read /proc/self/cgroup
    text = _read_text("/proc/self/cgroup")
    if not text:
        cgroup.notes.append("cannot read /proc/self/cgroup")
        return
    parsed_self = _parse_proc_self_cgroup(text)

    # Read mountinfo
    mnt_text = _read_text("/proc/self/mountinfo")
    if not mnt_text:
        cgroup.notes.append("cannot read /proc/self/mountinfo")
        return
    mounts = _parse_mountinfo(mnt_text)

    # cgroup v2
    v2_mount = _find_cgroup2_mount(mounts)
    if v2_mount and parsed_self.v2_path:
        base = Path(v2_mount.mount_point) / parsed_self.v2_path.lstrip("/")
        cgroup.mount_point = v2_mount.mount_point
        cgroup.relative_path = parsed_self.v2_path
        cgroup.version = "v2"

        cpu_max = _read_text(base / "cpu.max")
        if cpu_max is not None:
            cgroup.cpu_max_raw = cpu_max.strip()
            quota, period = _parse_cpu_max(cgroup.cpu_max_raw)
            cgroup.cpu_quota_us = quota
            cgroup.cpu_period_us = period
            cgroup.quota_cpu_count = _quota_to_cpu_count(quota, period)

        cpuset_text = _read_text(base / "cpuset.cpus.effective")
        if cpuset_text is not None:
            cgroup.cpuset_raw = cpuset_text.strip()
            cgroup.cpuset_cpu_ids = _parse_cpu_set(cgroup.cpuset_raw)
            cgroup.cpuset_cpu_count = len(cgroup.cpuset_cpu_ids)

        mem_max = _read_text(base / "memory.max")
        mem_curr = _read_text(base / "memory.current")
        cgroup.memory_max_bytes = _parse_cgroup_memory_value(mem_max)
        cgroup.memory_current_bytes = _parse_cgroup_memory_value(mem_curr)
        if cgroup.memory_max_bytes is not None and cgroup.memory_current_bytes is not None:
            cgroup.memory_remaining_bytes = max(0, cgroup.memory_max_bytes - cgroup.memory_current_bytes)
        cgroup.notes.append("probed cgroup v2")
        return

    # cgroup v1 fallback
    controller_mounts = {}
    for m in mounts:
        if m.fs_type == "cgroup":
            for tok in _collect_v1_controller_tokens(m.mount_source, m.super_options):
                controller_mounts[tok] = m.mount_point

    if not controller_mounts:
        cgroup.notes.append("no cgroup v2 mount and no v1 controller mounts found")
        return

    cgroup.controller_mounts = controller_mounts
    cgroup.controller_paths = dict(parsed_self.v1_paths)

    cpu_mount = controller_mounts.get("cpu")
    cpu_path = parsed_self.v1_paths.get("cpu")
    if cpu_mount and cpu_path:
        base = Path(cpu_mount) / cpu_path.lstrip("/")
        quota = _read_int(base / "cpu.cfs_quota_us")
        period = _read_int(base / "cpu.cfs_period_us")
        cgroup.cpu_cfs_quota_us = quota
        cgroup.cpu_cfs_period_us = period
        cgroup.quota_cpu_count = _quota_to_cpu_count(quota, period)

    cpuset_mount = controller_mounts.get("cpuset")
    cpuset_path = parsed_self.v1_paths.get("cpuset")
    if cpuset_mount and cpuset_path:
        base = Path(cpuset_mount) / cpuset_path.lstrip("/")
        cpuset_text = _read_text(base / "cpuset.cpus")
        if cpuset_text is not None:
            cgroup.cpuset_raw = cpuset_text.strip()
            cgroup.cpuset_cpu_ids = _parse_cpu_set(cgroup.cpuset_raw)
            cgroup.cpuset_cpu_count = len(cgroup.cpuset_cpu_ids)

    mem_mount = controller_mounts.get("memory")
    mem_path = parsed_self.v1_paths.get("memory")
    if mem_mount and mem_path:
        base = Path(mem_mount) / mem_path.lstrip("/")
        cgroup.memory_max_bytes = _parse_v1_memory_limit(_read_text(base / "memory.limit_in_bytes"))
        cgroup.memory_current_bytes = _parse_cgroup_memory_value(_read_text(base / "memory.usage_in_bytes"))
        if cgroup.memory_max_bytes is not None and cgroup.memory_current_bytes is not None:
            cgroup.memory_remaining_bytes = max(0, cgroup.memory_max_bytes - cgroup.memory_current_bytes)

    if cgroup.mount_point is None and controller_mounts:
        cgroup.mount_point = next(iter(controller_mounts.values()))
    if cgroup.relative_path is None and parsed_self.v1_paths:
        cgroup.relative_path = next(iter(parsed_self.v1_paths.values()))
    cgroup.version = "v1"
    cgroup.notes.append("probed cgroup v1 controllers")


# ----------------------------------------------------------------------
# Derived limits
# ----------------------------------------------------------------------

def _derive_cpu_limits(cpu: CpuResources, cgroup: CgroupProbe) -> None:
    cpu.cgroup_cpuset_cpu_ids = cgroup.cpuset_cpu_ids
    cpu.cgroup_cpuset_cpu_count = cgroup.cpuset_cpu_count
    cpu.cgroup_quota_cpu_count = cgroup.quota_cpu_count

    candidates: list[float] = []
    if cpu.process_cpu_count is not None:
        candidates.append(float(cpu.process_cpu_count))
    elif cpu.affinity_cpu_count is not None:
        candidates.append(float(cpu.affinity_cpu_count))
    elif cpu.host_cpu_count is not None:
        candidates.append(float(cpu.host_cpu_count))
    if cgroup.cpuset_cpu_count is not None and cgroup.cpuset_cpu_count > 0:
        candidates.append(float(cgroup.cpuset_cpu_count))
    if cgroup.quota_cpu_count is not None and cgroup.quota_cpu_count > 0:
        candidates.append(cgroup.quota_cpu_count)

    if candidates:
        cpu.effective_cpu_capacity = min(candidates)
        rec = max(1, math.floor(cpu.effective_cpu_capacity))
        if cpu.host_cpu_count is not None:
            rec = min(rec, cpu.host_cpu_count)
        cpu.recommended_worker_count = rec
    else:
        cpu.notes.append("could not derive effective CPU capacity")


def _derive_memory_limits(memory: MemoryResources, cgroup: CgroupProbe) -> None:
    memory.cgroup_memory_max_bytes = cgroup.memory_max_bytes
    memory.cgroup_memory_current_bytes = cgroup.memory_current_bytes
    memory.cgroup_memory_remaining_bytes = cgroup.memory_remaining_bytes

    candidates: list[int] = []
    if memory.mem_available_bytes is not None and memory.mem_available_bytes >= 0:
        candidates.append(memory.mem_available_bytes)
    if cgroup.memory_remaining_bytes is not None and cgroup.memory_remaining_bytes >= 0:
        candidates.append(cgroup.memory_remaining_bytes)

    if candidates:
        memory.effective_available_bytes = min(candidates)
    elif memory.mem_available_bytes is not None:
        memory.effective_available_bytes = memory.mem_available_bytes
    else:
        memory.notes.append("could not derive effective available memory")


# ----------------------------------------------------------------------
# Runtime context (metadata only)
# ----------------------------------------------------------------------

def _detect_runtime_context(runtime: RuntimeContext, cgroup: CgroupProbe, *args) -> None:
    del args

    if Path("/.dockerenv").exists():
        runtime.in_container = True
        runtime.notes.append("detected /.dockerenv")
    if Path("/run/.containerenv").exists():
        runtime.in_container = True
        runtime.notes.append("detected /run/.containerenv")
    cgroup_path = cgroup.relative_path or ""
    if cgroup_path:
        if "system.slice" in cgroup_path:
            runtime.systemd_hint = "system.slice"
        elif "user.slice" in cgroup_path:
            runtime.systemd_hint = "user.slice"
        elif ".scope" in cgroup_path:
            runtime.systemd_hint = "scope"
    if Path("/proc/self/ns/cgroup").exists():
        runtime.cgroup_namespace_hint = True


# ----------------------------------------------------------------------
# Project disk usage helpers
# ----------------------------------------------------------------------

_PROJECT_STORAGE_BUCKETS = {
    "snapshots",
    "previews",
    "history",
    "temp",
    "external_runs",
    "logs",
    "exports",
}


def _add_project_usage_bytes(
    usage: ProjectDiskUsage,
    root: Path,
    path: Path,
    size: int,
) -> None:
    size = max(0, int(size))
    usage.project_root_bytes += size

    bucket = _project_storage_bucket(root, path)

    if bucket == "snapshots":
        usage.snapshots_bytes += size
    elif bucket == "previews":
        usage.previews_bytes += size
    elif bucket == "history":
        usage.history_bytes += size
    elif bucket == "temp":
        usage.temp_bytes += size
    elif bucket == "external_runs":
        usage.external_runs_bytes += size
    elif bucket == "logs":
        usage.logs_bytes += size
    elif bucket == "exports":
        usage.exports_bytes += size
    else:
        usage.other_bytes += size


def _project_storage_bucket(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "other"

    if not rel.parts:
        return "other"

    first = rel.parts[0]
    if first in _PROJECT_STORAGE_BUCKETS:
        return first

    return "other"


def _maybe_record_largest_file(
    largest_files: list[DiskUsageFile],
    root: Path,
    path: Path,
    size: int,
    limit: int,
) -> None:
    if limit <= 0:
        return

    try:
        rel_path = str(path.relative_to(root))
    except ValueError:
        rel_path = str(path)

    largest_files.append(DiskUsageFile(path=rel_path, bytes=max(0, int(size))))
    largest_files.sort(key=lambda item: item.bytes, reverse=True)
    del largest_files[limit:]


# ----------------------------------------------------------------------
# Parsers / helpers
# ----------------------------------------------------------------------

def _read_text(path: str | Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_int(path: str | Path) -> Optional[int]:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _parse_meminfo(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else ""
        if unit == "kb":
            value *= 1024
        result[key] = value
    return result


def _parse_cpu_max(raw: str) -> tuple[Optional[int], Optional[int]]:
    parts = raw.strip().split()
    if len(parts) != 2:
        return None, None
    quota_s, period_s = parts
    quota = None if quota_s == "max" else _safe_int(quota_s)
    period = _safe_int(period_s)
    return quota, period


def _quota_to_cpu_count(quota_us: Optional[int], period_us: Optional[int]) -> Optional[float]:
    if quota_us is None or period_us is None or quota_us <= 0 or period_us <= 0:
        return None
    return float(quota_us) / float(period_us)


def _parse_cpu_set(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    cpus: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start = _safe_int(left)
            end = _safe_int(right)
            if start is not None and end is not None and start <= end:
                cpus.update(range(start, end + 1))
            elif start is not None and end is not None and end < start:
                cpus.update(range(end, start + 1))
        else:
            v = _safe_int(chunk)
            if v is not None:
                cpus.add(v)
    return tuple(sorted(cpus))


def _parse_cgroup_memory_value(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    value = text.strip()
    if not value or value == "max":
        return None
    parsed = _safe_int(value)
    return parsed


def _parse_v1_memory_limit(text: Optional[str]) -> Optional[int]:
    value = _parse_cgroup_memory_value(text)
    if value is None:
        return None
    if value == 1 << 60:  # sentinel
        return None
    return value


def _collect_v1_controller_tokens(mount_source: str, super_options: str) -> set[str]:
    tokens: set[str] = set()
    for source in (mount_source, super_options):
        for token in source.split(","):
            token = token.strip()
            if token:
                tokens.add(token)
    return tokens


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _parse_proc_self_cgroup(text: str) -> _ProcSelfCgroup:
    result = _ProcSelfCgroup()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            hierarchy, controllers, path = line.split(":", 2)
        except ValueError:
            continue
        if hierarchy == "0" and controllers == "":
            result.v2_path = path
            continue
        for controller in controllers.split(","):
            controller = controller.strip()
            if controller:
                result.v1_paths[controller] = path
    return result


def _parse_mountinfo(text: str) -> list[_MountInfo]:
    mounts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "- " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or len(right_fields) < 3:
            continue
        mount_point = left_fields[4]
        optional_fields = tuple(left_fields[6:])
        fs_type = right_fields[0]
        mount_source = right_fields[1]
        super_options = right_fields[2]
        mounts.append(_MountInfo(mount_point, fs_type, mount_source, super_options, optional_fields))
    return mounts


def _find_cgroup2_mount(mounts: Iterable[_MountInfo]) -> Optional[_MountInfo]:
    for m in mounts:
        if m.fs_type == "cgroup2":
            return m
    return None
