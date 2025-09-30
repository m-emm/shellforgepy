"""Workflow orchestration command line tool for ShellforgePy."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from shellforgepy.slicing.orca_slicer_settings_generator import generate_settings
from shellforgepy.workflow import upload_to_printer

_logger = logging.getLogger(__name__)


CONFIG_ENV_VAR = "SHELLFORGEPY_CONFIG"
CONFIG_DEFAULT_PATH = Path.home() / ".shellforgepy" / "config.json"
DEFAULT_RUNS_DIR_NAME = "runs"
EXPORT_DIR_ENV = "SHELLFORGEPY_EXPORT_DIR"
MANIFEST_ENV = "SHELLFORGEPY_WORKFLOW_MANIFEST"
RUN_ID_ENV = "SHELLFORGEPY_RUN_ID"
RUN_DIR_ENV = "SHELLFORGEPY_RUN_DIRECTORY"
MANIFEST_FILENAME = "workflow_manifest.json"


def _default_config_template() -> Dict[str, object]:
    default_runner = Path(__file__).resolve().parents[2] / "freecad_python.sh"
    runner_path = str(default_runner) if default_runner.exists() else sys.executable

    return {
        "runs_dir": DEFAULT_RUNS_DIR_NAME,
        "python": {
            "runner": runner_path,
        },
        "orca": {
            "master_settings_dir": "~/path/to/orca/settings_master",
            "executable": "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
            "debug_level": 6,
        },
        "upload": {
            "printer": "192.168.0.10:4409",
        },
        "render": {
            "script": "~/path/to/render.sh",
            "executable": "",
            "args": [],
        },
    }


class WorkflowError(Exception):
    """Custom error used for workflow orchestration failures."""


class SubprocessResult:
    """Container for subprocess execution results."""

    def __init__(
        self, returncode: int, stdout_lines: List[str], stderr_lines: List[str]
    ):
        self.returncode = returncode
        self.stdout_lines = stdout_lines
        self.stderr_lines = stderr_lines


def configure_logging(verbose: bool = False) -> None:
    """Configure logger for CLI usage."""

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=format)


def get_config_path(override: Optional[str]) -> Path:
    """Return the config path from override, environment, or default."""

    if override:
        return Path(override).expanduser()

    env_override = os.environ.get(CONFIG_ENV_VAR)
    if env_override:
        return Path(env_override).expanduser()

    return CONFIG_DEFAULT_PATH


def load_config(path: Path) -> Dict[str, object]:
    """Load configuration dictionary from disk."""

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Invalid JSON in config file {path}: {exc}") from exc


def save_config(path: Path, data: Dict[str, object]) -> None:
    """Persist configuration dictionary to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _resolve_dotted_lookup(data: Dict[str, object], dotted_key: str, default=None):
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if not isinstance(current, dict):
            return default
        current = current.get(key, {})
    if not isinstance(current, dict):
        return default
    return current.get(keys[-1], default)


def _set_dotted_key(data: Dict[str, object], dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    current: Dict[str, object] = data
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def _unset_dotted_key(data: Dict[str, object], dotted_key: str) -> None:
    keys = dotted_key.split(".")
    current: Dict[str, object] = data
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(keys[-1], None)


def format_command(cmd: Iterable[str]) -> str:
    """Return a shell-safe representation of a command list."""

    return " ".join(shlex.quote(part) for part in cmd)


def execute_subprocess(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    stdin_data: Optional[str] = None,
) -> SubprocessResult:
    """Execute a subprocess, streaming logs and returning captured output."""

    _logger.debug("Executing command: %s", format_command(cmd))

    combined_env = os.environ.copy()
    if env:
        combined_env.update(env)
    combined_env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        cwd=str(cwd) if cwd else None,
        env=combined_env,
        text=True,
        bufsize=1,
    )

    if stdin_data is not None and process.stdin is not None:
        process.stdin.write(stdin_data)
        process.stdin.flush()
        process.stdin.close()

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def _consume(pipe, accumulator: List[str], level: int) -> None:
        try:
            for raw_line in iter(pipe.readline, ""):
                line = raw_line.rstrip()
                accumulator.append(line)
                _logger.log(level, "%s", line)
        finally:
            pipe.close()

    threads: List[threading.Thread] = []
    if process.stdout is not None:
        threads.append(
            threading.Thread(
                target=_consume,
                args=(process.stdout, stdout_lines, logging.INFO),
                daemon=True,
            )
        )
    if process.stderr is not None:
        threads.append(
            threading.Thread(
                target=_consume,
                args=(process.stderr, stderr_lines, logging.WARNING),
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    returncode = process.wait()

    if returncode != 0:
        raise WorkflowError(
            f"Command failed with exit code {returncode}: {format_command(cmd)}"
        )

    return SubprocessResult(returncode, stdout_lines, stderr_lines)


def _ensure_path(path: Optional[Path], description: str) -> Path:
    if path is None:
        raise WorkflowError(
            f"Could not determine {description}. Provide it explicitly."
        )
    resolved = path.expanduser()
    if not resolved.exists():
        raise WorkflowError(f"{description} does not exist: {resolved}")
    return resolved


def _resolve_manifest_path(base_dir: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _gather_jsons(directory: Path) -> List[Path]:
    json_files = []
    for candidate in directory.glob("*.json"):
        if not candidate.is_file():
            continue
        if candidate.name == MANIFEST_FILENAME:
            continue
        if candidate.name.endswith("_process.json"):
            continue
        json_files.append(candidate)
    return sorted(json_files)


def _gather_filament_jsons(directory: Path) -> List[Path]:
    filaments_dir = directory / "filaments"
    if not filaments_dir.is_dir():
        return []
    return sorted(p for p in filaments_dir.glob("*.json") if p.is_file())


def _list_created_files(directory: Path) -> List[Path]:
    files: List[Path] = []
    for item in sorted(directory.rglob("*")):
        if item.is_file():
            files.append(item)
    return files


def run_workflow(args: argparse.Namespace) -> int:
    config_path = get_config_path(args.config)
    config = load_config(config_path)

    runs_base = Path(
        args.runs_dir
        or _resolve_dotted_lookup(config, "runs_dir")
        or (Path.cwd() / DEFAULT_RUNS_DIR_NAME)
    ).expanduser()
    if not runs_base.is_absolute():
        runs_base = Path.cwd() / runs_base
    runs_base = runs_base.resolve()
    runs_base.mkdir(parents=True, exist_ok=True)

    target_path = Path(args.target).expanduser()
    if not target_path.exists():
        raise WorkflowError(f"Target file does not exist: {target_path}")

    if target_path.suffix.lower() != ".py":
        raise WorkflowError(
            "Only Python scripts (.py) are supported by the workflow tool."
        )

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    production_mode = bool(args.production or args.slice or args.upload)

    run_directory = (runs_base / f"{target_path.stem}_run_{run_id}").resolve()
    run_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = run_directory / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()

    _logger.info("Run ID: %s", run_id)
    _logger.info("Run directory: %s", run_directory)
    _logger.info("Production mode: %s", "yes" if production_mode else "no")

    forwarded_args = list(args.target_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    env = os.environ.copy()
    env[RUN_ID_ENV] = run_id
    env["SHELLFORGEPY_PRODUCTION"] = "1" if production_mode else "0"
    env[RUN_DIR_ENV] = str(run_directory)
    env[EXPORT_DIR_ENV] = str(run_directory)
    env[MANIFEST_ENV] = str(manifest_path)

    default_runner = Path(__file__).resolve().parents[2] / "freecad_python.sh"
    runner = (
        args.python
        or _resolve_dotted_lookup(config, "python.runner")
        or _resolve_dotted_lookup(config, "python_executable")
        or (str(default_runner) if default_runner.exists() else sys.executable)
    )
    runner_path = Path(runner).expanduser()
    if not runner_path.exists():
        raise WorkflowError(f"Configured runner does not exist: {runner_path}")

    cmd = [str(runner_path), str(target_path)] + forwarded_args
    _logger.info("Running target via: %s", format_command(cmd))
    execute_subprocess(cmd, env=env)

    if not manifest_path.exists():
        raise WorkflowError(
            f"No workflow manifest produced at {manifest_path}. "
            "Ensure the target script uses shellforgepy.produce.arrange_and_export_parts."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    part_path = (
        Path(args.part_file).expanduser()
        if args.part_file
        else _resolve_manifest_path(run_directory, manifest.get("assembly_path"))
    )
    process_path = (
        Path(args.process_file).expanduser()
        if args.process_file
        else _resolve_manifest_path(run_directory, manifest.get("process_data_path"))
    )

    part_path = _ensure_path(part_path, "generated STL")
    process_path = _ensure_path(process_path, "generated process data JSON")

    _logger.info("Detected part file: %s", part_path)
    _logger.info("Detected process file: %s", process_path)

    viewer_default = _resolve_dotted_lookup(
        config, "viewer.default_stl_file"
    ) or _resolve_dotted_lookup(config, "default_stl_file")
    if viewer_default:
        try:
            viewer_path = Path(viewer_default).expanduser()
            viewer_path.parent.mkdir(parents=True, exist_ok=True)
            if viewer_path.resolve() != part_path.resolve():
                shutil.copy2(part_path, viewer_path)
            _logger.info("Copied STL to viewer path: %s", viewer_path)
        except Exception as exc:  # pragma: no cover - best effort
            _logger.warning("Failed to update viewer STL %s: %s", viewer_default, exc)

    slice_requested = bool(args.slice or args.upload)

    if not slice_requested:
        _logger.info(
            "No slicing requested; workflow run finished after geometry stage."
        )
        created_files = _list_created_files(run_directory)
        if created_files:
            _logger.info("Artifacts created in run directory:")
            for file_path in created_files:
                _logger.info("  %s", file_path)
        return 0

    master_settings_dir = (
        args.master_settings_dir
        or _resolve_dotted_lookup(config, "orca.master_settings_dir")
        or _resolve_dotted_lookup(config, "orca_master_settings_dir")
    )
    if not master_settings_dir:
        raise WorkflowError(
            "Master settings directory not configured. Use '--master-settings-dir' or configure 'orca.master_settings_dir'."
        )
    master_settings_dir = Path(master_settings_dir).expanduser()
    if not master_settings_dir.is_dir():
        raise WorkflowError(
            f"Master settings directory not found: {master_settings_dir}"
        )

    _logger.info("Generating OrcaSlicer settings in %s", run_directory)
    try:
        generate_settings(
            process_data_file=process_path,
            output_dir=run_directory,
            master_settings_dir=master_settings_dir,
        )
    except Exception as exc:
        raise WorkflowError(f"Failed to generate OrcaSlicer settings: {exc}") from exc

    settings_files = _gather_jsons(run_directory)
    filament_files = _gather_filament_jsons(run_directory)

    if not settings_files:
        raise WorkflowError(
            f"No settings JSON files were generated in {run_directory}."
        )

    slicer_output_dir = run_directory / "slicedata"
    slicer_output_dir.mkdir(exist_ok=True)

    orca_exec = (
        args.orca_executable
        or _resolve_dotted_lookup(config, "orca.executable")
        or _resolve_dotted_lookup(config, "orca_slicer_executable")
    )
    if not orca_exec:
        raise WorkflowError(
            "OrcaSlicer executable not configured. Use '--orca-executable' or set 'orca.executable' in config."
        )
    orca_exec_path = Path(orca_exec).expanduser()
    if not orca_exec_path.exists():
        raise WorkflowError(f"OrcaSlicer executable not found: {orca_exec_path}")

    debug_level = str(
        args.orca_debug or _resolve_dotted_lookup(config, "orca.debug_level") or 6
    )
    project_filename = f"{target_path.stem}.3mf"
    project_path = run_directory / project_filename

    settings_arg = ";".join(str(path) for path in settings_files)
    slicer_cmd = [
        str(orca_exec_path),
        "--debug",
        debug_level,
        "--slice",
        "0",
        "--arrange",
        "0",
        "--outputdir",
        str(run_directory),
        "--export-slicedata",
        str(slicer_output_dir),
        "--export-3mf",
        project_filename,
        "--load-settings",
        settings_arg,
    ]

    if filament_files:
        filament_arg = ";".join(str(path) for path in filament_files)
        slicer_cmd.extend(["--load-filaments", filament_arg])

    slicer_cmd.append(str(part_path))

    _logger.info("Running OrcaSlicer: %s", format_command(slicer_cmd))

    orca_env_settings = _resolve_dotted_lookup(config, "orca.env")
    orca_env: Dict[str, str] = {}
    if isinstance(orca_env_settings, dict):
        for key, value in orca_env_settings.items():
            if value is None:
                continue
            orca_env[str(key)] = str(value)

    if "QT_OPENGL" not in orca_env and "QT_OPENGL" not in os.environ:
        orca_env["QT_OPENGL"] = "legacy"
    if "QT_QPA_PLATFORM" not in orca_env and "QT_QPA_PLATFORM" not in os.environ:
        orca_env["QT_QPA_PLATFORM"] = "offscreen"

    execute_subprocess(slicer_cmd, env=orca_env)

    render_script = _resolve_dotted_lookup(config, "render.script")
    if render_script:
        preview_path = run_directory / "plate_1_preview.png"
        render_exec = _resolve_dotted_lookup(config, "render.executable")
        render_args = _resolve_dotted_lookup(config, "render.args") or []
        if isinstance(render_args, str):
            render_args = shlex.split(render_args)
        elif not isinstance(render_args, list):
            render_args = []
        if render_exec:
            render_cmd = [
                render_exec,
                str(render_script),
                str(part_path),
                str(preview_path),
            ]
        else:
            render_cmd = [
                sys.executable,
                str(render_script),
                str(part_path),
                str(preview_path),
            ]
        render_cmd.extend(str(arg) for arg in render_args)
        _logger.info(
            "Generating preview via render script: %s", format_command(render_cmd)
        )
        try:
            execute_subprocess(render_cmd)
        except WorkflowError as exc:
            _logger.warning("Preview generation failed: %s", exc)

    created_files = _list_created_files(run_directory)
    if created_files:
        _logger.info("Artifacts created in run directory:")
        for file_path in created_files:
            _logger.info("  %s", file_path)

    if args.upload:
        gcode_files = sorted(run_directory.glob("*.gcode"))
        if not gcode_files:
            raise WorkflowError("Upload requested but no G-code files were generated.")

        printer = args.printer or _resolve_dotted_lookup(config, "upload.printer")

        for gcode_file in gcode_files:
            _logger.info("Uploading %s", gcode_file)
            try:
                upload_to_printer.upload_to_printer(gcode_file, printer)
            except Exception as exc:
                raise WorkflowError(f"Failed to upload {gcode_file}: {exc}") from exc

    _logger.info("Workflow completed successfully. Run directory: %s", run_directory)
    return 0


def show_config(config: Dict[str, object]) -> None:
    _logger.info(json.dumps(config, indent=2, sort_keys=True))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ShellforgePy workflow command line tool"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file. Defaults to ~/.shellforgepy/config.json",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config", help="Manage workflow configuration"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )

    config_set = config_subparsers.add_parser("set", help="Set a configuration key")
    config_set.add_argument("key", help="Configuration key (supports dotted notation)")
    config_set.add_argument("value", help="Value to store")

    config_unset = config_subparsers.add_parser(
        "unset", help="Remove a configuration key"
    )
    config_unset.add_argument("key", help="Configuration key to remove")

    config_show = config_subparsers.add_parser(
        "show", help="Display current configuration"
    )

    config_init = config_subparsers.add_parser(
        "init", help="Create a default configuration file"
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing configuration file",
    )

    run_parser = subparsers.add_parser(
        "run", help="Execute a geometry build and slicing workflow"
    )
    run_parser.add_argument("target", help="Path to shellforgepy Python script")
    run_parser.add_argument(
        "--run-id", help="Override automatically generated run identifier"
    )
    run_parser.add_argument("--runs-dir", help="Directory to store run artifacts")
    run_parser.add_argument(
        "--production",
        action="store_true",
        help="Force production mode even if slicing is not requested",
    )
    run_parser.add_argument(
        "--slice",
        action="store_true",
        help="Run OrcaSlicer after generating settings",
    )
    run_parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated G-code files (implies --slice)",
    )
    run_parser.add_argument(
        "--python",
        help="Runner to execute the target script (e.g. freecad_python.sh or python executable)",
    )
    run_parser.add_argument(
        "--master-settings-dir",
        help="Path to Orca master settings directory",
    )
    run_parser.add_argument(
        "--orca-executable",
        help="Path to the OrcaSlicer executable",
    )
    run_parser.add_argument(
        "--orca-debug",
        type=int,
        help="Debug level for OrcaSlicer",
    )
    run_parser.add_argument(
        "--part-file",
        help="Path to the generated STL (override manifest)",
    )
    run_parser.add_argument(
        "--process-file",
        help="Path to the generated process JSON (override manifest)",
    )
    run_parser.add_argument(
        "--printer",
        help="Printer host (ip[:port]) to use for uploads",
    )
    run_parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target (precede with -- to separate)",
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "config":
        config_path = get_config_path(args.config)
        if args.config_command == "init":
            if config_path.exists() and not args.force:
                raise WorkflowError(
                    f"Configuration already exists at {config_path}. Use --force to overwrite."
                )
            template = _default_config_template()
            save_config(config_path, template)
            _logger.info("Wrote default configuration to %s", config_path)
            return 0

        config = load_config(config_path)

        if args.config_command == "set":
            _set_dotted_key(config, args.key, args.value)
            save_config(config_path, config)
            _logger.info("Set %s", args.key)
            return 0
        if args.config_command == "unset":
            _unset_dotted_key(config, args.key)
            save_config(config_path, config)
            _logger.info("Removed %s", args.key)
            return 0
        if args.config_command == "show":
            show_config(config)
            return 0

    if args.command == "run":
        if args.upload and not args.slice:
            args.slice = True
        return run_workflow(args)

    raise AssertionError("Unhandled command")


if __name__ == "__main__":  # pragma: no cover

    sys.exit(main())
