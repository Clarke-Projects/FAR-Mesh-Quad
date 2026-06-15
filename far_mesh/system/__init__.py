"""
System execution layer for FAR MESH Quad 3.

Phase 1.5 provides launcher-neutral resource probing, conservative task
execution sockets, and lifecycle ownership for isolated process/subprocess work.

This package must stay GUI-free:
- no Qt imports
- no viewport imports
- no MainWindow imports
"""

from far_mesh.system.execution_plan import ComputeDeviceKind, ExecutionMode, ExecutionPlan
from far_mesh.system.execution_manager import (
    clear_task_registry,
    execute_task,
    get_lifecycle_manager,
    register_task,
    reset_lifecycle_manager,
    shutdown_execution_lifecycle,
    unregister_task,
)
from far_mesh.system.lifecycle import (
    LifecycleManager,
    PoolTaskHandle,
    ProcessTaskHandle,
    SubprocessTaskHandle,
    TaskHandle,
    TaskLifecycleInfo,
    TaskRunState,
)
from far_mesh.system.resource_probe import SystemResources, probe_system_resources
from far_mesh.system.task_protocol import TaskKind, TaskRequest, TaskResult
from far_mesh.system.task_router import plan_task

__all__ = [
    "ComputeDeviceKind",
    "ExecutionMode",
    "ExecutionPlan",
    "LifecycleManager",
    "PoolTaskHandle",
    "ProcessTaskHandle",
    "SubprocessTaskHandle",
    "SystemResources",
    "TaskHandle",
    "TaskKind",
    "TaskLifecycleInfo",
    "TaskRequest",
    "TaskResult",
    "TaskRunState",
    "clear_task_registry",
    "execute_task",
    "get_lifecycle_manager",
    "plan_task",
    "probe_system_resources",
    "register_task",
    "reset_lifecycle_manager",
    "shutdown_execution_lifecycle",
    "unregister_task",
]
