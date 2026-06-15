"""
Phase 1.5 Foundation C: task lifecycle ownership.

Pure system layer:
- no Qt
- no viewport
- no MainWindow
- safe for PROCESS / SUBPROCESS task ownership
"""

from __future__ import annotations

import abc
import os
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from pathlib import Path
from typing import Any, Optional


class TaskRunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    TERMINATING = "terminating"
    KILLING = "killing"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CLEANED = "cleaned"


@dataclass(slots=True)
class TaskLifecycleInfo:
    task_id: str
    label: str = ""
    state: TaskRunState = TaskRunState.PENDING
    temp_paths: list[Path] = field(default_factory=list)


class TaskHandle(abc.ABC):
    def __init__(self, *, task_id: str | None = None, label: str = "") -> None:
        self.info = TaskLifecycleInfo(
            task_id=task_id or uuid.uuid4().hex,
            label=label,
        )
        self._lock = threading.RLock()
        self._cleaned = False

    @abc.abstractmethod
    def cancel(self, *, timeout: float = 5.0) -> None:
        ...

    @abc.abstractmethod
    def wait(self, timeout: Optional[float] = None) -> None:
        ...

    @abc.abstractmethod
    def is_running(self) -> bool:
        ...

    def add_temp_path(self, path: str | Path) -> None:
        with self._lock:
            self.info.temp_paths.append(Path(path))

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return

            for path in self.info.temp_paths:
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    elif path.exists():
                        path.unlink(missing_ok=True)
                except Exception:
                    pass

            self._cleaned = True
            self.info.state = TaskRunState.CLEANED


class ProcessTaskHandle(TaskHandle):
    """Lifecycle handle for multiprocessing.Process tasks."""

    def __init__(
        self,
        process: Process,
        *,
        task_id: str | None = None,
        label: str = "",
        temp_dir: str | Path | None = None,
    ) -> None:
        super().__init__(task_id=task_id, label=label)
        self._process = process
        self.info.state = TaskRunState.RUNNING
        if temp_dir is not None:
            self.add_temp_path(temp_dir)

    def cancel(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._process.is_alive():
                self.info.state = TaskRunState.COMPLETED
                return

            self.info.state = TaskRunState.CANCEL_REQUESTED

            if hasattr(self._process, "terminate"):
                self.info.state = TaskRunState.TERMINATING
                self._process.terminate()

        self._process.join(timeout)

        with self._lock:
            if self._process.is_alive():
                self.info.state = TaskRunState.KILLING
                if hasattr(self._process, "kill"):
                    self._process.kill()
                else:
                    pid = self._process.pid
                    if pid is not None and os.name == "posix":
                        os.kill(pid, signal.SIGKILL)

        self._process.join(1.0)

        with self._lock:
            self.info.state = (
                TaskRunState.CANCELLED
                if not self._process.is_alive()
                else TaskRunState.FAILED
            )

    def wait(self, timeout: Optional[float] = None) -> None:
        self._process.join(timeout)
        with self._lock:
            if not self._process.is_alive() and self.info.state == TaskRunState.RUNNING:
                self.info.state = TaskRunState.COMPLETED

    def is_running(self) -> bool:
        return self._process.is_alive()


class PoolTaskHandle(TaskHandle):
    """Lifecycle handle for multiprocessing.Pool tasks.

    FAR Mesh Quad keeps multiprocessing.Pool as the PROCESS execution socket.
    The lifecycle layer must therefore own the Pool itself, but it must never
    block the GUI forever on Pool.join(): multiprocessing.Pool.join() has no
    timeout argument and can hang during native-library/Open3D shutdown on some
    systems.  This handle uses bounded/timed join helpers and falls back to
    terminating the Pool, matching the old context-manager behavior more closely
    than an unbounded close()+join().
    """

    def __init__(
        self,
        pool: Any,
        async_result: Any | None = None,
        *,
        task_id: str | None = None,
        label: str = "",
        temp_dir: str | Path | None = None,
    ) -> None:
        super().__init__(task_id=task_id, label=label)
        self._pool = pool
        self._async_result = async_result
        self._pool_finalized = False
        self.info.state = TaskRunState.RUNNING
        if temp_dir is not None:
            self.add_temp_path(temp_dir)

    def set_async_result(self, async_result: Any) -> None:
        """Attach the Pool AsyncResult after lifecycle registration."""

        with self._lock:
            self._async_result = async_result

    def finish(self, *, timeout: float = 2.0) -> None:
        """Finalize a completed one-shot Pool without an unbounded join.

        The PROCESS socket currently creates one Pool per task. Once the
        AsyncResult has been read, the worker has already delivered its result.
        Terminating the idle/completed Pool is safer than Pool.close()+join() on
        platforms where native geometry libraries leave non-Python state behind.
        """

        self._finalize_pool(final_state=TaskRunState.COMPLETED, timeout=timeout)

    def close(self, *, join: bool = True, timeout: float = 2.0) -> None:
        """Compatibility helper for normal completion.

        Kept for callers/tests that use close(). It intentionally delegates to
        finish() instead of doing Pool.close()+Pool.join() because Pool.join()
        cannot be bounded by timeout.
        """

        if join:
            self.finish(timeout=timeout)
            return

        self._finalize_pool(final_state=TaskRunState.COMPLETED, timeout=0.0)

    def cancel(self, *, timeout: float = 5.0) -> None:
        """Terminate the Pool and wait only up to the requested timeout."""

        self._finalize_pool(final_state=TaskRunState.CANCELLED, timeout=timeout)

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the AsyncResult only; do not join the Pool here."""

        with self._lock:
            async_result = self._async_result

        if async_result is None:
            return

        try:
            async_result.wait(timeout)
        except Exception:
            with self._lock:
                if self.info.state == TaskRunState.RUNNING:
                    self.info.state = TaskRunState.FAILED
            return

        with self._lock:
            try:
                if async_result.ready() and self.info.state == TaskRunState.RUNNING:
                    self.info.state = TaskRunState.COMPLETED
            except Exception:
                if self.info.state == TaskRunState.RUNNING:
                    self.info.state = TaskRunState.FAILED

    def is_running(self) -> bool:
        with self._lock:
            if self._pool_finalized:
                return False
            if self.info.state in {
                TaskRunState.COMPLETED,
                TaskRunState.CANCELLED,
                TaskRunState.FAILED,
                TaskRunState.CLEANED,
            }:
                return False
            async_result = self._async_result
            if async_result is None:
                return self.info.state == TaskRunState.RUNNING
            try:
                ready = bool(async_result.ready())
            except Exception:
                return False
            if ready and self.info.state == TaskRunState.RUNNING:
                self.info.state = TaskRunState.COMPLETED
                return False
            return not ready

    def cleanup(self) -> None:
        try:
            with self._lock:
                finalized = self._pool_finalized
                async_result = self._async_result

            if not finalized:
                should_cancel = True
                if async_result is not None:
                    try:
                        should_cancel = not bool(async_result.ready())
                    except Exception:
                        should_cancel = True

                if should_cancel:
                    self.cancel(timeout=1.0)
                else:
                    self.finish(timeout=1.0)
        except Exception:
            pass

        super().cleanup()

    def _finalize_pool(self, *, final_state: TaskRunState, timeout: float) -> None:
        with self._lock:
            if self._pool_finalized:
                if self.info.state == TaskRunState.RUNNING:
                    self.info.state = final_state
                return

            self._pool_finalized = True
            self.info.state = (
                TaskRunState.TERMINATING
                if final_state == TaskRunState.CANCELLED
                else final_state
            )

            try:
                self._pool.terminate()
            except Exception:
                pass

        self._join_pool_with_timeout(timeout)

        with self._lock:
            if final_state == TaskRunState.CANCELLED:
                self.info.state = TaskRunState.CANCELLED
            elif self.info.state not in {TaskRunState.CANCELLED, TaskRunState.FAILED}:
                self.info.state = final_state

    def _join_pool_with_timeout(self, timeout: Optional[float]) -> bool:
        """Join Pool workers without risking an unbounded GUI/task hang."""

        done = threading.Event()

        def _join() -> None:
            try:
                self._pool.join()
            except Exception:
                pass
            finally:
                done.set()

        join_thread = threading.Thread(
            target=_join,
            name=f"far-mesh-pool-join-{self.info.task_id[:8]}",
            daemon=True,
        )
        join_thread.start()

        if timeout is None:
            done.wait()
            return True

        try:
            wait_seconds = max(0.0, float(timeout))
        except Exception:
            wait_seconds = 0.0

        return bool(done.wait(wait_seconds))

class SubprocessTaskHandle(TaskHandle):
    """Lifecycle handle for external subprocess tasks."""

    def __init__(
        self,
        popen: subprocess.Popen,
        *,
        task_id: str | None = None,
        label: str = "",
        temp_dir: str | Path | None = None,
        owns_process_group: bool = True,
    ) -> None:
        super().__init__(task_id=task_id, label=label)
        self._popen = popen
        self._owns_process_group = owns_process_group
        self.info.state = TaskRunState.RUNNING
        if temp_dir is not None:
            self.add_temp_path(temp_dir)

    def cancel(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if self._popen.poll() is not None:
                self.info.state = TaskRunState.COMPLETED
                return

            self.info.state = TaskRunState.TERMINATING
            self._terminate()

        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with self._lock:
                self.info.state = TaskRunState.KILLING
                self._kill()
            try:
                self._popen.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

        with self._lock:
            self.info.state = (
                TaskRunState.CANCELLED
                if self._popen.poll() is not None
                else TaskRunState.FAILED
            )

    def wait(self, timeout: Optional[float] = None) -> None:
        self._popen.wait(timeout=timeout)
        with self._lock:
            if self.info.state == TaskRunState.RUNNING:
                self.info.state = TaskRunState.COMPLETED

    def is_running(self) -> bool:
        return self._popen.poll() is None

    def _terminate(self) -> None:
        if os.name == "posix" and self._owns_process_group:
            try:
                os.killpg(os.getpgid(self._popen.pid), signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception:
                pass

        try:
            self._popen.terminate()
        except ProcessLookupError:
            pass

    def _kill(self) -> None:
        if os.name == "posix" and self._owns_process_group:
            try:
                os.killpg(os.getpgid(self._popen.pid), signal.SIGKILL)
                return
            except ProcessLookupError:
                return
            except Exception:
                pass

        try:
            self._popen.kill()
        except ProcessLookupError:
            pass


class LifecycleManager:
    """
    Registry for running isolated tasks.

    This is intentionally GUI-free. MainWindow may call shutdown(), but the
    policy and tracking live here, not in the GUI layer.
    """

    def __init__(self) -> None:
        self._handles: dict[str, TaskHandle] = {}
        self._lock = threading.RLock()
        self._shutting_down = False

    def register(self, handle: TaskHandle) -> str:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("Cannot register task while lifecycle manager is shutting down.")
            self._handles[handle.info.task_id] = handle
            return handle.info.task_id

    def unregister(self, task_id: str, *, cleanup: bool = False) -> None:
        with self._lock:
            handle = self._handles.pop(task_id, None)

        if handle is not None and cleanup:
            handle.cleanup()

    def get(self, task_id: str) -> TaskHandle | None:
        with self._lock:
            return self._handles.get(task_id)

    def active_tasks(self) -> list[TaskLifecycleInfo]:
        with self._lock:
            return [
                handle.info
                for handle in self._handles.values()
                if handle.is_running()
            ]

    def active_count(self) -> int:
        return len(self.active_tasks())

    def cancel(self, task_id: str, *, timeout: float = 5.0, cleanup: bool = False) -> bool:
        handle = self.get(task_id)
        if handle is None:
            return False

        handle.cancel(timeout=timeout)

        if cleanup:
            handle.cleanup()
            self.unregister(task_id)

        return True

    def cancel_all(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            handles = list(self._handles.values())

        for handle in handles:
            try:
                handle.cancel(timeout=timeout)
            except Exception:
                pass

    def wait_all(self, timeout: Optional[float] = None) -> None:
        with self._lock:
            handles = list(self._handles.values())

        for handle in handles:
            try:
                handle.wait(timeout=timeout)
            except Exception:
                pass

    def cleanup_all(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()

        for handle in handles:
            try:
                handle.cleanup()
            except Exception:
                pass

    def shutdown(self, *, cancel_timeout: float = 3.0, wait_timeout: float = 2.0) -> None:
        with self._lock:
            self._shutting_down = True

        self.cancel_all(timeout=cancel_timeout)
        self.wait_all(timeout=wait_timeout)
        self.cleanup_all()
