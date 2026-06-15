from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import subprocess
import time
from typing import Optional, Any

from far_mesh.system.lifecycle import LifecycleManager, SubprocessTaskHandle


@dataclass(slots=True)
class QuadWildBiMDFRunResult:
    success: bool
    repo_root: str
    working_directory: str
    input_mesh_path: str
    staged_mesh_path: str

    stage1_command: list[str]
    stage2_command: list[str]

    stage1_stdout: str
    stage1_stderr: str
    stage1_returncode: int

    stage2_stdout: str
    stage2_stderr: str
    stage2_returncode: int

    stage1_elapsed_seconds: float = 0.0
    stage2_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0

    stage1_config_requested: str | None = None
    stage1_config_used: str | None = None
    stage2_config_used: str | None = None
    stage1_overrides_used: dict[str, Any] = field(default_factory=dict)
    stage1_fallback_used: bool = False
    stage1_fallback_reason: str | None = None

    final_mesh_path: str | None = None
    final_stage: str | None = None
    generated_files: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _CommandResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed_seconds: float


class QuadWildBiMDFRunner:
    """
    Runs the working two-stage QuadWild-BiMDF CLI pipeline.

    Stage 1:
        quadwild <mesh> 2 <prep_config>

    Stage 2:
        quad_from_patches <mesh_rem_p0.obj> <output_index> <main_config>

    If lifecycle is provided to run(), both external commands are registered
    through SubprocessTaskHandle so FAR Mesh can cancel them during shutdown.
    """

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        stage1_config_rel: str = "config/prep_config/basic_setup.txt",
        stage2_config_rel: str = "config/main_config/flow_noalign_lemon.txt",
        stop_after_step: int = 2,
        output_index: int = 123,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.default_stage1_config_rel = stage1_config_rel
        self.default_stage2_config_rel = stage2_config_rel
        self.stop_after_step = int(stop_after_step)
        self.output_index = int(output_index)

    @property
    def bin_quadwild(self) -> Path:
        return self.repo_root / "build" / "Build" / "bin" / "quadwild"

    @property
    def bin_quad_from_patches(self) -> Path:
        return self.repo_root / "build" / "Build" / "bin" / "quad_from_patches"

    @property
    def lib_dir(self) -> Path:
        return self.repo_root / "build" / "Build" / "lib"

    @property
    def prep_config_dir(self) -> Path:
        return self.repo_root / "config" / "prep_config"

    def _resolve_stage1_config(self, stage1_config_rel: str | None = None) -> Path:
        rel = stage1_config_rel or self.default_stage1_config_rel
        return self.repo_root / rel

    def _resolve_stage2_config(self, stage2_config_rel: str | None = None) -> Path:
        rel = stage2_config_rel or self.default_stage2_config_rel
        return self.repo_root / rel

    def get_available_stage1_presets(self) -> dict[str, str]:
        candidates = [
            ("Basic", "config/prep_config/basic_setup.txt"),
            ("Mechanical", "config/prep_config/basic_setup_mechanical.txt"),
            ("Organic", "config/prep_config/basic_setup_organic.txt"),
        ]

        available: dict[str, str] = {}
        for label, rel in candidates:
            if (self.repo_root / rel).exists():
                available[label] = rel

        if not available and self.prep_config_dir.exists():
            for path in sorted(self.prep_config_dir.glob("*.txt")):
                available[path.stem] = str(path.relative_to(self.repo_root))

        return available

    def _resolve_stage1_config_with_fallback(
        self,
        requested_rel: str | None,
    ) -> tuple[Path, bool, str | None]:
        requested_path = self._resolve_stage1_config(requested_rel)
        if requested_path.exists():
            return requested_path, False, None

        default_path = self._resolve_stage1_config(self.default_stage1_config_rel)
        if default_path.exists():
            return (
                default_path,
                True,
                f"Requested stage1 config was missing: {requested_path}. "
                f"Fell back to default: {default_path}.",
            )

        available = self.get_available_stage1_presets()
        if available:
            first_rel = next(iter(available.values()))
            first_path = self.repo_root / first_rel
            return (
                first_path,
                True,
                f"Requested stage1 config was missing: {requested_path}. "
                f"Fell back to first available preset: {first_path}.",
            )

        raise FileNotFoundError(
            "No usable QuadWild-BiMDF stage1 prep config was found.\n"
            f"Requested: {requested_path}\n"
            f"Prep config directory: {self.prep_config_dir}"
        )

    def is_available(self) -> bool:
        return (
            self.bin_quadwild.exists()
            and self.bin_quadwild.is_file()
            and self.bin_quad_from_patches.exists()
            and self.bin_quad_from_patches.is_file()
            and self.lib_dir.exists()
            and self.lib_dir.is_dir()
        )

    def debug_paths(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "quadwild": str(self.bin_quadwild),
            "quad_from_patches": str(self.bin_quad_from_patches),
            "lib_dir": str(self.lib_dir),
            "default_stage1_config": str(self._resolve_stage1_config()),
            "default_stage2_config": str(self._resolve_stage2_config()),
            "prep_config_dir": str(self.prep_config_dir),
        }

    def run(
        self,
        input_mesh_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        timeout_stage1: Optional[float] = None,
        timeout_stage2: Optional[float] = None,
        overwrite: bool = True,
        stage1_config_rel: str | None = None,
        stage2_config_rel: str | None = None,
        stage1_overrides: dict[str, Any] | None = None,
        *,
        lifecycle: LifecycleManager | None = None,
    ) -> QuadWildBiMDFRunResult:
        self._validate_installation(stage2_config_rel=stage2_config_rel)

        total_t0 = time.perf_counter()

        input_path = Path(input_mesh_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input mesh not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input mesh path is not a file: {input_path}")

        out_dir = Path(output_dir).expanduser().resolve()
        if out_dir.exists() and overwrite:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        staged_mesh_path = out_dir / input_path.name
        shutil.copy2(input_path, staged_mesh_path)

        if input_path.suffix.lower() == ".obj":
            mtl_path = input_path.with_suffix(".mtl")
            if mtl_path.exists() and mtl_path.is_file():
                shutil.copy2(mtl_path, out_dir / mtl_path.name)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

        requested_stage1_config = stage1_config_rel or self.default_stage1_config_rel
        stage1_template, fallback_used, fallback_reason = self._resolve_stage1_config_with_fallback(
            requested_stage1_config
        )
        stage2_config = self._resolve_stage2_config(stage2_config_rel)

        stage1_config = self._prepare_stage1_config(
            template_path=stage1_template,
            output_dir=out_dir,
            overrides=stage1_overrides or {},
        )

        stage1_cmd = [
            str(self.bin_quadwild),
            str(staged_mesh_path),
            str(self.stop_after_step),
            str(stage1_config),
        ]

        stage1 = self._run_external_command(
            stage1_cmd,
            cwd=self.repo_root,
            env=env,
            timeout=timeout_stage1,
            lifecycle=lifecycle,
            label="quadwild-bimdf-stage1",
        )

        generated_after_stage1 = self._collect_generated_files(staged_mesh_path)

        if stage1.returncode != 0:
            raise RuntimeError(
                self._format_stage_error(
                    stage_name="quadwild",
                    command=stage1_cmd,
                    returncode=stage1.returncode,
                    stdout=stage1.stdout,
                    stderr=stage1.stderr,
                    generated=generated_after_stage1,
                )
            )

        rem_p0_path = generated_after_stage1.get("traced_mesh")
        if rem_p0_path is None:
            raise RuntimeError(
                "QuadWild-BiMDF stage 1 finished but did not produce the traced mesh "
                f"'{staged_mesh_path.with_suffix('').name}_rem_p0.obj'."
            )

        stage2_cmd = [
            str(self.bin_quad_from_patches),
            rem_p0_path,
            str(self.output_index),
            str(stage2_config),
        ]

        stage2 = self._run_external_command(
            stage2_cmd,
            cwd=self.repo_root,
            env=env,
            timeout=timeout_stage2,
            lifecycle=lifecycle,
            label="quadwild-bimdf-stage2",
        )

        generated_after_stage2 = self._collect_generated_files(staged_mesh_path)
        final_mesh_path, final_stage = self._pick_final_output(generated_after_stage2)

        total_elapsed = time.perf_counter() - total_t0

        result = QuadWildBiMDFRunResult(
            success=(stage1.returncode == 0 and stage2.returncode == 0 and final_mesh_path is not None),
            repo_root=str(self.repo_root),
            working_directory=str(out_dir),
            input_mesh_path=str(input_path),
            staged_mesh_path=str(staged_mesh_path),
            stage1_command=stage1_cmd,
            stage2_command=stage2_cmd,
            stage1_stdout=stage1.stdout,
            stage1_stderr=stage1.stderr,
            stage1_returncode=stage1.returncode,
            stage2_stdout=stage2.stdout,
            stage2_stderr=stage2.stderr,
            stage2_returncode=stage2.returncode,
            stage1_elapsed_seconds=stage1.elapsed_seconds,
            stage2_elapsed_seconds=stage2.elapsed_seconds,
            total_elapsed_seconds=total_elapsed,
            stage1_config_requested=requested_stage1_config,
            stage1_config_used=str(stage1_config),
            stage2_config_used=str(stage2_config),
            stage1_overrides_used=dict(stage1_overrides or {}),
            stage1_fallback_used=fallback_used,
            stage1_fallback_reason=fallback_reason,
            final_mesh_path=final_mesh_path,
            final_stage=final_stage,
            generated_files=generated_after_stage2,
        )

        if stage2.returncode != 0:
            raise RuntimeError(
                self._format_stage_error(
                    stage_name="quad_from_patches",
                    command=stage2_cmd,
                    returncode=stage2.returncode,
                    stdout=stage2.stdout,
                    stderr=stage2.stderr,
                    generated=generated_after_stage2,
                )
            )

        if final_mesh_path is None:
            raise RuntimeError(
                "QuadWild-BiMDF stage 2 finished but no final quadrangulation OBJ was found.\n"
                f"Generated files:\n{self._format_generated(generated_after_stage2)}"
            )

        return result

    def _run_external_command(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float | None,
        lifecycle: LifecycleManager | None,
        label: str,
    ) -> _CommandResult:
        t0 = time.perf_counter()

        handle: SubprocessTaskHandle | None = None
        task_id: str | None = None

        try:
            popen = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
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
                    f"QuadWild-BiMDF command timed out after {timeout} seconds: {' '.join(cmd)}"
                )

            elapsed = time.perf_counter() - t0

            return _CommandResult(
                stdout=stdout or "",
                stderr=stderr or "",
                returncode=int(popen.returncode),
                elapsed_seconds=elapsed,
            )

        finally:
            if lifecycle is not None and task_id is not None:
                lifecycle.unregister(task_id, cleanup=False)

    def _validate_installation(
        self,
        stage2_config_rel: str | None = None,
    ) -> None:
        missing = []
        stage2_config = self._resolve_stage2_config(stage2_config_rel)

        if not self.bin_quadwild.exists():
            missing.append(f"missing quadwild binary: {self.bin_quadwild}")
        if not self.bin_quad_from_patches.exists():
            missing.append(f"missing quad_from_patches binary: {self.bin_quad_from_patches}")
        if not self.lib_dir.exists():
            missing.append(f"missing lib directory: {self.lib_dir}")
        if not stage2_config.exists():
            missing.append(f"missing stage2 config: {stage2_config}")

        if missing:
            raise FileNotFoundError(
                "QuadWild-BiMDF installation is incomplete:\n- " + "\n- ".join(missing)
            )

    def _prepare_stage1_config(
        self,
        template_path: Path,
        output_dir: Path,
        overrides: dict[str, Any],
    ) -> Path:
        text = template_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        normalized = {
            "do_remesh": overrides.get("do_remesh", None),
            "sharp_feature_thr": overrides.get("sharp_feature_thr", None),
            "alpha": overrides.get("alpha", None),
            "scaleFact": overrides.get("scaleFact", None),
        }

        updated_lines: list[str] = []
        touched: set[str] = set()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                updated_lines.append(line)
                continue

            key = stripped.split()[0]
            if key in normalized and normalized[key] is not None:
                updated_lines.append(f"{key} {self._format_config_value(normalized[key])}")
                touched.add(key)
            else:
                updated_lines.append(line)

        for key, value in normalized.items():
            if value is not None and key not in touched:
                updated_lines.append(f"{key} {self._format_config_value(value)}")

        config_path = output_dir / "quadwild_stage1_generated.txt"
        config_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        return config_path

    @staticmethod
    def _format_config_value(value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _collect_generated_files(self, staged_mesh_path: Path) -> dict[str, str]:
        stem = staged_mesh_path.with_suffix("")
        idx = self.output_index

        candidates = {
            "rem_mesh": stem.with_name(stem.name + "_rem.obj"),
            "rem_field": stem.with_name(stem.name + "_rem.rosy"),
            "rem_sharp": stem.with_name(stem.name + "_rem.sharp"),
            "traced_mesh": stem.with_name(stem.name + "_rem_p0.obj"),
            "patch_file": stem.with_name(stem.name + "_rem_p0.patch"),
            "corners_file": stem.with_name(stem.name + "_rem_p0.corners"),
            "feature_file": stem.with_name(stem.name + "_rem_p0.feature"),
            "c_feature_file": stem.with_name(stem.name + "_rem_p0.c_feature"),
            "quadrangulation": stem.with_name(stem.name + f"_rem_p0_{idx}_quadrangulation.obj"),
            "quadrangulation_smooth": stem.with_name(
                stem.name + f"_rem_p0_{idx}_quadrangulation_smooth.obj"
            ),
        }

        found: dict[str, str] = {}
        for key, path in candidates.items():
            if path.exists() and path.is_file():
                found[key] = str(path)

        return found

    def _pick_final_output(self, generated: dict[str, str]) -> tuple[str | None, str | None]:
        if "quadrangulation_smooth" in generated:
            return generated["quadrangulation_smooth"], "quadrangulation_smooth"
        if "quadrangulation" in generated:
            return generated["quadrangulation"], "quadrangulation"
        return None, None

    @staticmethod
    def _format_generated(generated: dict[str, str]) -> str:
        if not generated:
            return "(none)"
        return "\n".join(f"- {key}: {value}" for key, value in sorted(generated.items()))

    def _format_stage_error(
        self,
        stage_name: str,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        generated: dict[str, str],
    ) -> str:
        parts = [
            f"QuadWild-BiMDF stage '{stage_name}' failed.",
            f"Return code: {returncode}",
            f"Command: {' '.join(command)}",
            f"Repo root: {self.repo_root}",
        ]

        if stdout.strip():
            parts.append("STDOUT:")
            parts.append(stdout.strip())

        if stderr.strip():
            parts.append("STDERR:")
            parts.append(stderr.strip())

        if generated:
            parts.append("Generated files:")
            parts.append(self._format_generated(generated))

        return "\n".join(parts)
