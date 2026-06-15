"""
Execution manager for FAR MESH Quad 3.

The manager executes a TaskRequest according to an ExecutionPlan.

Rules:
- no Qt imports
- no viewport imports
- no MainWindow ownership of scheduling policy
- spawned Pool execution for process-isolated tasks
- lifecycle ownership for isolated PROCESS / SUBPROCESS work
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from far_mesh.system.execution_plan import ExecutionMode, ExecutionPlan
from far_mesh.system.lifecycle import LifecycleManager, PoolTaskHandle, SubprocessTaskHandle
from far_mesh.system.task_protocol import TaskKind, TaskRequest, TaskResult


TaskHandler = Callable[[dict[str, Any]], dict[str, Any]]


_TASK_REGISTRY: dict[str, TaskHandler] = {}
_LIFECYCLE_MANAGER = LifecycleManager()


def get_lifecycle_manager() -> LifecycleManager:
    return _LIFECYCLE_MANAGER


def reset_lifecycle_manager() -> LifecycleManager:
    global _LIFECYCLE_MANAGER

    try:
        _LIFECYCLE_MANAGER.shutdown()
    except Exception:
        pass

    _LIFECYCLE_MANAGER = LifecycleManager()
    return _LIFECYCLE_MANAGER


def register_task(kind: TaskKind | str, func: TaskHandler) -> None:
    key = _task_key(kind)
    _TASK_REGISTRY[key] = func


def unregister_task(kind: TaskKind | str) -> None:
    key = _task_key(kind)
    _TASK_REGISTRY.pop(key, None)


def clear_task_registry() -> None:
    _TASK_REGISTRY.clear()


def execute_task(
    request: TaskRequest,
    plan: ExecutionPlan,
    *,
    lifecycle: LifecycleManager | None = None,
) -> TaskResult:
    """
    Execute a task according to the supplied plan.

    The router decides the plan. The execution manager only follows it.
    """

    lifecycle = lifecycle or _LIFECYCLE_MANAGER

    if plan.mode == ExecutionMode.INLINE:
        return _execute_inline(request)

    if plan.mode == ExecutionMode.THREAD:
        return _execute_thread(request, max_workers=plan.max_workers or 1)

    if plan.mode == ExecutionMode.PROCESS:
        return _execute_process(request, lifecycle=lifecycle)

    if plan.mode == ExecutionMode.SUBPROCESS:
        return _execute_subprocess(request, lifecycle=lifecycle)

    if plan.mode == ExecutionMode.ACCELERATOR:
        return TaskResult(
            ok=False,
            error=(
                "ACCELERATOR execution is a future socket only; no provider "
                "is registered in Phase 1.5."
            ),
            warnings=[plan.reason] if plan.reason else [],
        )

    return TaskResult(
        ok=False,
        error=f"Unsupported execution mode: {plan.mode!r}",
    )


def shutdown_execution_lifecycle() -> None:
    _LIFECYCLE_MANAGER.shutdown()


def _execute_inline(request: TaskRequest) -> TaskResult:
    try:
        func = _resolve_task_function(request.kind)
        payload = func(request.payload)
        return TaskResult(ok=True, payload=payload)
    except Exception as exc:
        return TaskResult(ok=False, error=str(exc))


def _execute_thread(request: TaskRequest, max_workers: int = 1) -> TaskResult:
    workers = max(1, int(max_workers))

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future = executor.submit(_run_registered_task, request)
            return future.result()
    except Exception as exc:
        return TaskResult(ok=False, error=str(exc))


def _execute_process(
    request: TaskRequest,
    *,
    lifecycle: LifecycleManager | None = None,
) -> TaskResult:
    """
    Execute in a lifecycle-owned spawned multiprocessing.Pool.

    FAR Mesh Quad keeps Pool as the PROCESS execution socket. The Pool is
    registered with LifecycleManager, and completion finalization is bounded so
    native/Open3D worker shutdown cannot hang the GUI indefinitely.
    """

    lifecycle = lifecycle or _LIFECYCLE_MANAGER
    pool = None
    handle: PoolTaskHandle | None = None
    task_id: str | None = None

    try:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=1)
        handle = PoolTaskHandle(
            pool,
            label=f"pool:{_task_key(request.kind)}",
        )
        task_id = lifecycle.register(handle)

        async_result = pool.apply_async(_run_process_worker, (request,))
        handle.set_async_result(async_result)

        result = async_result.get()

        handle.finish(timeout=2.0)
        lifecycle.unregister(task_id, cleanup=True)
        task_id = None

        if isinstance(result, TaskResult):
            return result

        return TaskResult(
            ok=False,
            error=f"PROCESS task returned invalid result type: {type(result).__name__}",
        )

    except Exception as exc:
        if handle is not None:
            try:
                handle.cancel(timeout=1.0)
            except Exception:
                pass
        elif pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

        return TaskResult(ok=False, error=str(exc))

    finally:
        if task_id is not None:
            try:
                lifecycle.unregister(task_id, cleanup=True)
            except Exception:
                pass


def _execute_subprocess(
    request: TaskRequest,
    *,
    lifecycle: LifecycleManager,
) -> TaskResult:
    """
    Execute a subprocess-backed task.

    DUMMY_SUBPROCESS is kept as a test-only direct subprocess validation path.
    Real external backend tasks are delegated to registered core handlers.
    Those handlers are responsible for launching lifecycle-aware external
    runners such as Instant Meshes or QuadWild-BiMDF.
    """

    if _task_key(request.kind) == "DUMMY_SUBPROCESS":
        return _execute_dummy_subprocess(request, lifecycle=lifecycle)

    try:
        return _run_registered_task(request)
    except Exception as exc:
        return TaskResult(ok=False, error=str(exc))


def _execute_dummy_subprocess(
    request: TaskRequest,
    *,
    lifecycle: LifecycleManager,
) -> TaskResult:
    """
    Test-only subprocess task.

    This is intentionally not a TaskKind enum member. It validates that the
    execution manager can launch, register, wait for, and allow cancellation of
    a real subprocess through LifecycleManager.
    """

    handle: SubprocessTaskHandle | None = None
    task_id: str | None = None

    try:
        seconds = float(request.payload.get("seconds", 1.0))
        timeout = request.payload.get("timeout")
        wait_timeout = float(timeout) if timeout is not None else seconds + 2.0

        popen = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import time; time.sleep({seconds})",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        handle = SubprocessTaskHandle(
            popen,
            label="dummy-subprocess",
            owns_process_group=True,
        )

        task_id = lifecycle.register(handle)

        try:
            handle.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            handle.cancel(timeout=1.0)
            lifecycle.unregister(task_id, cleanup=True)
            task_id = None
            return TaskResult(ok=False, error="dummy subprocess timeout")

        stdout, stderr = popen.communicate(timeout=1.0)

        lifecycle.unregister(task_id, cleanup=True)
        task_id = None

        return TaskResult(
            ok=(popen.returncode == 0),
            payload={
                "returncode": popen.returncode,
                "stdout": stdout.decode(errors="replace") if stdout else "",
                "stderr": stderr.decode(errors="replace") if stderr else "",
            },
            error=None if popen.returncode == 0 else "dummy subprocess failed",
        )

    except Exception as exc:
        if handle is not None:
            try:
                handle.cancel(timeout=1.0)
                handle.cleanup()
            except Exception:
                pass
        return TaskResult(ok=False, error=str(exc))

    finally:
        if task_id is not None:
            try:
                lifecycle.unregister(task_id, cleanup=True)
            except Exception:
                pass


def _run_registered_task(request: TaskRequest) -> TaskResult:
    try:
        func = _resolve_task_function(request.kind)
        payload = func(request.payload)
        return TaskResult(ok=True, payload=payload)
    except Exception as exc:
        return TaskResult(ok=False, error=str(exc))


def _run_process_worker(request: TaskRequest) -> TaskResult:
    from far_mesh.system._process_worker import run_task_in_process

    return run_task_in_process(request)


def _resolve_task_function(kind: TaskKind | str) -> TaskHandler:
    key = _task_key(kind)

    if key in _TASK_REGISTRY:
        return _TASK_REGISTRY[key]

    raise RuntimeError(f"No handler registered for task kind '{key}'")


def _task_key(kind: TaskKind | str) -> str:
    if isinstance(kind, TaskKind):
        return kind.value

    return str(kind)
