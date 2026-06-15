from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from far_mesh.system.lifecycle import LifecycleManager, SubprocessTaskHandle


class InstantMeshesRunner:
    def __init__(self, executable_path: str | Path | None = None) -> None:
        if executable_path is None:
            base_dir = Path(__file__).parent.parent.parent
            executable_path = base_dir / "bin" / "Instant Meshes"

        self.executable_path = Path(executable_path)

        if not self.executable_path.exists():
            raise FileNotFoundError(
                f"InstantMeshes executable not found at {self.executable_path}"
            )

    def run(
        self,
        input_mesh_path: str | Path,
        output_mesh_path: str | Path,
        target_faces: int = 5000,
        crease_angle: float = 30.0,
        smooth_iterations: int = 2,
        deterministic: bool = False,
        *,
        lifecycle: LifecycleManager | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run Instant Meshes in batch mode.

        If lifecycle is provided, the external process is registered through
        SubprocessTaskHandle so FAR Mesh can cancel it during shutdown.
        """

        del kwargs

        input_path = Path(input_mesh_path)
        output_path = Path(output_mesh_path)

        cmd = [
            str(self.executable_path),
            str(input_path),
            "-o",
            str(output_path),
            "-f",
            str(int(target_faces)),
            "-c",
            str(float(crease_angle)),
            "-S",
            str(int(smooth_iterations)),
        ]

        if deterministic:
            cmd.append("-d")

        result = _run_external_command(
            cmd,
            lifecycle=lifecycle,
            timeout=timeout,
            label="instant-meshes",
        )

        if result["returncode"] != 0:
            raise RuntimeError(
                "Instant Meshes failed.\n"
                f"Return code: {result['returncode']}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result['stdout']}\n"
                f"STDERR:\n{result['stderr']}"
            )

        return {
            "command": cmd,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
        }


def _run_external_command(
    cmd: list[str],
    *,
    lifecycle: LifecycleManager | None,
    timeout: float | None,
    label: str,
) -> dict[str, Any]:
    """
    Run an external command with optional lifecycle ownership.

    Uses Popen instead of subprocess.run so the process can be tracked and
    terminated through LifecycleManager.
    """

    t0 = time.perf_counter()

    handle: SubprocessTaskHandle | None = None
    task_id: str | None = None

    try:
        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        if lifecycle is not None:
            handle = SubprocessTaskHandle(
                popen,
                label=label,
                owns_process_group=True,
            )
            task_id = lifecycle.register(handle)

        try:
            stdout, stderr = popen.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if handle is not None:
                handle.cancel(timeout=3.0)
            else:
                popen.terminate()
                try:
                    popen.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    popen.kill()
                    popen.wait(timeout=1.0)

            raise TimeoutError(
                f"External command timed out after {timeout} seconds: {' '.join(cmd)}"
            )

        elapsed = time.perf_counter() - t0

        return {
            "returncode": int(popen.returncode),
            "stdout": stdout or "",
            "stderr": stderr or "",
            "elapsed_seconds": elapsed,
        }

    finally:
        if lifecycle is not None and task_id is not None:
            lifecycle.unregister(task_id, cleanup=False)
