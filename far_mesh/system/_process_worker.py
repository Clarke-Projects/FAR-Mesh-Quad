"""
Spawned process worker entry point for FAR MESH Quad 3.

This module must stay pure Python / core:
- no Qt
- no viewport
- no MainWindow
- no GUI imports

Important PROCESS rule:
multiprocessing "spawn" starts a fresh Python interpreter. The parent
process task registry is not shared with the child process. Therefore the
child process must register core task handlers before resolving the requested
TaskKind.
"""

from __future__ import annotations

from far_mesh.system.execution_manager import _resolve_task_function
from far_mesh.system.task_protocol import TaskRequest, TaskResult


def _ensure_core_tasks_registered_in_process() -> None:
    """
    Register core-safe task handlers inside the spawned worker process.

    This intentionally imports far_mesh.core.task_registry lazily so importing
    this worker module remains lightweight. The task registry itself remains
    GUI-free and viewport-free.
    """

    from far_mesh.core.task_registry import register_core_tasks

    register_core_tasks()


def run_task_in_process(request: TaskRequest) -> TaskResult:
    """Execute a single registered task in a spawned worker process."""

    try:
        _ensure_core_tasks_registered_in_process()
        func = _resolve_task_function(request.kind)
        result_payload = func(request.payload)
        return TaskResult(ok=True, payload=result_payload)
    except Exception as exc:
        return TaskResult(ok=False, error=str(exc))
