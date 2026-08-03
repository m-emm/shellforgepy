import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
from shellforgepy.workflow.gcode_postprocessing import (
    GcodePostprocessorContext,
    GcodePostprocessorError,
    apply_gcode_postprocessor,
)


def _context(tmp_path: Path, gcode_path: Path, project_path: Path):
    process_path = tmp_path / "process.json"
    process_path.write_text(json.dumps({"process_overrides": {}}), encoding="utf-8")
    stl_path = tmp_path / "part.stl"
    stl_path.write_text("solid part\n", encoding="utf-8")
    return GcodePostprocessorContext(
        gcode_path=gcode_path,
        project_path=project_path,
        stl_path=stl_path,
        obj_path=None,
        part_stl_paths=(),
        process_data_path=process_path,
        process_data={"process_overrides": {}},
        plate_name="test_plate",
        target_label="test_assembly",
        run_directory=tmp_path,
        workflow_manifest={"plates": []},
        plate_manifest={"name": "test_plate"},
    )


def _project(path: Path, gcode: bytes):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Metadata/plate_1.gcode", gcode)
        archive.writestr(
            "Metadata/plate_1.gcode.md5",
            hashlib.md5(gcode).hexdigest().upper(),
        )
        archive.writestr("Metadata/project_settings.config", "{}")


def _install_module(monkeypatch, function):
    module = types.ModuleType("fixture_gcode_postprocessor")
    module.transform = function
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}.transform"


def test_apply_gcode_postprocessor_updates_external_and_project(monkeypatch, tmp_path):
    original = b"G90\nG1 X1 Y1\n"
    gcode_path = tmp_path / "part.gcode"
    gcode_path.write_bytes(original)
    project_path = tmp_path / "part.3mf"
    _project(project_path, original)
    captured = {}

    def transform(gcode_text, *, context, marker):
        captured["context"] = context
        return gcode_text + f"; {marker}\n"

    function_path = _install_module(monkeypatch, transform)
    report = apply_gcode_postprocessor(
        specification={"function": function_path, "arguments": {"marker": "done"}},
        context=_context(tmp_path, gcode_path, project_path),
    )

    expected = original + b"; done\n"
    assert gcode_path.read_bytes() == expected
    with zipfile.ZipFile(project_path) as archive:
        assert archive.read("Metadata/plate_1.gcode") == expected
        assert archive.read("Metadata/plate_1.gcode.md5").decode() == (
            hashlib.md5(expected).hexdigest().upper()
        )
        assert archive.read("Metadata/project_settings.config") == b"{}"
    assert captured["context"].plate_name == "test_plate"
    assert report["changed"] is True
    assert report["arguments"] == {"marker": "done"}
    assert report["output_sha256"] == hashlib.sha256(expected).hexdigest()


def test_postprocessor_failure_leaves_artifacts_unchanged(monkeypatch, tmp_path):
    original = b"G90\n"
    gcode_path = tmp_path / "part.gcode"
    gcode_path.write_bytes(original)
    project_path = tmp_path / "part.3mf"
    _project(project_path, original)
    original_project = project_path.read_bytes()

    def transform(gcode_text, *, context):
        raise ValueError("bad calibration")

    function_path = _install_module(monkeypatch, transform)
    with pytest.raises(GcodePostprocessorError, match="bad calibration"):
        apply_gcode_postprocessor(
            specification={"function": function_path},
            context=_context(tmp_path, gcode_path, project_path),
        )

    assert gcode_path.read_bytes() == original
    assert project_path.read_bytes() == original_project


def test_postprocessor_must_return_nonempty_text(monkeypatch, tmp_path):
    original = b"G90\n"
    gcode_path = tmp_path / "part.gcode"
    gcode_path.write_bytes(original)
    project_path = tmp_path / "part.3mf"
    _project(project_path, original)

    def transform(gcode_text, *, context):
        return b"not text"

    function_path = _install_module(monkeypatch, transform)
    with pytest.raises(GcodePostprocessorError, match="must return str"):
        apply_gcode_postprocessor(
            specification={"function": function_path},
            context=_context(tmp_path, gcode_path, project_path),
        )
