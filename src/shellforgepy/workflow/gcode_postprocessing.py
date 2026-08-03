"""G-code postprocessing support for ShellForgePy workflows."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class GcodePostprocessorError(RuntimeError):
    """Raised when a configured G-code postprocessor cannot be applied safely."""


@dataclass(frozen=True)
class GcodePostprocessorContext:
    """Read-only artifact and workflow context supplied to a postprocessor."""

    gcode_path: Path
    project_path: Path
    stl_path: Path
    obj_path: Optional[Path]
    part_stl_paths: tuple[Path, ...]
    process_data_path: Path
    process_data: Mapping[str, Any]
    plate_name: str
    target_label: str
    run_directory: Path
    workflow_manifest: Mapping[str, Any]
    plate_manifest: Mapping[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _replace_project_gcode(project_path: Path, gcode_bytes: bytes) -> str:
    if not zipfile.is_zipfile(project_path):
        raise GcodePostprocessorError(
            f"Orca project is not a readable 3MF ZIP archive: {project_path}"
        )

    gcode_member = "Metadata/plate_1.gcode"
    checksum_member = "Metadata/plate_1.gcode.md5"
    replacement_checksum = hashlib.md5(gcode_bytes).hexdigest().upper().encode("ascii")
    replacements = {
        gcode_member: gcode_bytes,
        checksum_member: replacement_checksum,
    }

    descriptor, temporary_name = tempfile.mkstemp(
        dir=project_path.parent,
        prefix=f".{project_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    found_members: set[str] = set()
    try:
        with (
            zipfile.ZipFile(project_path, "r") as source,
            zipfile.ZipFile(temporary_path, "w") as destination,
        ):
            for member in source.infolist():
                data = replacements.get(member.filename)
                if data is None:
                    data = source.read(member.filename)
                else:
                    found_members.add(member.filename)
                destination.writestr(member, data)

        missing_members = sorted(set(replacements) - found_members)
        if missing_members:
            raise GcodePostprocessorError(
                "Orca project is missing sliced G-code members: "
                + ", ".join(missing_members)
            )
        os.replace(temporary_path, project_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return replacement_checksum.decode("ascii")


def _load_postprocessor(function_path: str):
    if "." not in function_path:
        raise GcodePostprocessorError(
            f"G-code postprocessor must be a dotted module.function path: {function_path}"
        )
    module_name, function_name = function_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise GcodePostprocessorError(
            f"Could not import G-code postprocessor module '{module_name}': {exc}"
        ) from exc
    try:
        postprocessor = getattr(module, function_name)
    except AttributeError as exc:
        raise GcodePostprocessorError(
            f"G-code postprocessor function '{function_path}' does not exist"
        ) from exc
    if not callable(postprocessor):
        raise GcodePostprocessorError(
            f"G-code postprocessor '{function_path}' is not callable"
        )
    return postprocessor


def _postprocessor_source_report(postprocessor) -> dict[str, Optional[str]]:
    source_path = inspect.getsourcefile(postprocessor) or inspect.getfile(postprocessor)
    if not source_path:
        return {"source_path": None, "source_sha256": None}
    resolved_path = Path(source_path).resolve()
    return {
        "source_path": str(resolved_path),
        "source_sha256": _sha256_bytes(resolved_path.read_bytes()),
    }


def apply_gcode_postprocessor(
    *,
    specification: Mapping[str, Any],
    context: GcodePostprocessorContext,
) -> dict[str, Any]:
    """Apply one configured postprocessor and synchronize its 3MF project."""

    function_path = specification.get("function")
    if not isinstance(function_path, str) or not function_path.strip():
        raise GcodePostprocessorError(
            "gcode_postprocessor.function must be a non-empty dotted path"
        )
    arguments = specification.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise GcodePostprocessorError("gcode_postprocessor.arguments must be a mapping")

    postprocessor = _load_postprocessor(function_path.strip())
    original_text = _read_text_preserving_newlines(context.gcode_path)
    original_bytes = original_text.encode("utf-8")
    try:
        processed_text = postprocessor(
            original_text,
            context=context,
            **deepcopy(dict(arguments)),
        )
    except Exception as exc:
        raise GcodePostprocessorError(
            f"G-code postprocessor '{function_path}' failed: {exc}"
        ) from exc
    if not isinstance(processed_text, str):
        raise GcodePostprocessorError(
            f"G-code postprocessor '{function_path}' must return str, got "
            f"{type(processed_text).__name__}"
        )
    if not processed_text.strip():
        raise GcodePostprocessorError(
            f"G-code postprocessor '{function_path}' returned empty G-code"
        )

    processed_bytes = processed_text.encode("utf-8")
    project_md5 = _replace_project_gcode(context.project_path, processed_bytes)
    _atomic_write_bytes(context.gcode_path, processed_bytes)

    report = {
        "function": function_path.strip(),
        "arguments": deepcopy(dict(arguments)),
        "input_sha256": _sha256_bytes(original_bytes),
        "output_sha256": _sha256_bytes(processed_bytes),
        "changed": original_bytes != processed_bytes,
        "project_path": str(context.project_path),
        "project_gcode_md5": project_md5,
    }
    report.update(_postprocessor_source_report(postprocessor))
    return report


def build_gcode_postprocessor_context(
    *,
    gcode_path: Path,
    project_path: Path,
    stl_path: Path,
    obj_path: Optional[Path],
    part_stl_paths: Sequence[Path],
    process_data_path: Path,
    plate_name: str,
    target_label: str,
    run_directory: Path,
    workflow_manifest: Mapping[str, Any],
    plate_manifest: Optional[Mapping[str, Any]],
) -> GcodePostprocessorContext:
    """Build an isolated context snapshot for a postprocessor invocation."""

    with process_data_path.open("r", encoding="utf-8") as handle:
        process_data = json.load(handle)
    return GcodePostprocessorContext(
        gcode_path=gcode_path,
        project_path=project_path,
        stl_path=stl_path,
        obj_path=obj_path,
        part_stl_paths=tuple(part_stl_paths),
        process_data_path=process_data_path,
        process_data=deepcopy(process_data),
        plate_name=plate_name,
        target_label=target_label,
        run_directory=run_directory,
        workflow_manifest=deepcopy(dict(workflow_manifest)),
        plate_manifest=deepcopy(dict(plate_manifest or {})),
    )


__all__ = [
    "GcodePostprocessorContext",
    "GcodePostprocessorError",
    "apply_gcode_postprocessor",
    "build_gcode_postprocessor_context",
]
