import argparse
import json
import logging
import sys
from pathlib import Path

import pytest
from shellforgepy.workflow.workflow import (
    MANIFEST_ENV,
    SubprocessResult,
    WorkflowError,
    _orca_open_commands,
    complete_workflow_run,
    run_workflow,
)


def _make_args(target: Path, runs_dir: Path, **overrides):
    values = {
        "config": None,
        "runs_dir": str(runs_dir),
        "target": str(target),
        "run_id": "test_run",
        "production": False,
        "slice": False,
        "upload": False,
        "python": sys.executable,
        "master_settings_dir": None,
        "orca_executable": None,
        "orca_debug": None,
        "part_file": None,
        "process_file": None,
        "printer": None,
        "open": False,
        "target_args": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_workflow_allows_obj_only_geometry_runs(monkeypatch, tmp_path):
    target = tmp_path / "design.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        manifest_path = Path(env[MANIFEST_ENV])
        run_directory = manifest_path.parent
        obj_path = run_directory / "design.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"obj_path": str(obj_path)}),
            encoding="utf-8",
        )
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.execute_subprocess",
        fake_execute_subprocess,
    )

    result = run_workflow(_make_args(target, tmp_path))

    assert result == 0


def test_run_workflow_still_requires_stl_for_slicing(monkeypatch, tmp_path):
    target = tmp_path / "design.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        manifest_path = Path(env[MANIFEST_ENV])
        run_directory = manifest_path.parent
        obj_path = run_directory / "design.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"obj_path": str(obj_path)}),
            encoding="utf-8",
        )
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.execute_subprocess",
        fake_execute_subprocess,
    )

    with pytest.raises(WorkflowError, match="Could not determine generated STL"):
        run_workflow(_make_args(target, tmp_path, slice=True))


def test_complete_workflow_run_slices_each_manifest_plate(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plate_a_stl = run_directory / "machine_plate_a.stl"
    plate_b_stl = run_directory / "machine_plate_b.stl"
    plate_a_stl.write_text("solid a\n", encoding="utf-8")
    plate_b_stl.write_text("solid b\n", encoding="utf-8")
    plate_a_process = run_directory / "machine_plate_a_process.json"
    plate_b_process = run_directory / "machine_plate_b_process.json"
    plate_a_process.write_text("{}", encoding="utf-8")
    plate_b_process.write_text("{}", encoding="utf-8")

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    generated_process_inputs = []
    slicer_inputs = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        generated_process_inputs.append(Path(process_data_file).name)
        (Path(output_dir) / "machine_settings.json").write_text("{}", encoding="utf-8")

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            slicer_inputs.append(Path(cmd[-1]).name)
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.generate_settings", fake_generate_settings
    )
    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.execute_subprocess",
        fake_execute_subprocess,
    )

    args = argparse.Namespace(
        slice=True,
        upload=False,
        open=False,
        part_file=None,
        process_file=None,
        master_settings_dir=str(master_settings_dir),
        orca_executable=str(orca_exec),
        orca_debug=None,
        printer=None,
    )
    config = {"orca": {"debug_level": 6}}
    manifest = {
        "plates": [
            {
                "name": "plate_a",
                "assembly_path": str(plate_a_stl),
                "process_data_path": str(plate_a_process),
            },
            {
                "name": "plate_b",
                "assembly_path": str(plate_b_stl),
                "process_data_path": str(plate_b_process),
            },
        ]
    }

    result = complete_workflow_run(
        args,
        config=config,
        run_directory=run_directory,
        manifest=manifest,
        target_label="machine",
    )

    assert result == 0
    assert generated_process_inputs == [
        "machine_plate_a_process.json",
        "machine_plate_b_process.json",
    ]
    assert slicer_inputs == ["machine_plate_a.stl", "machine_plate_b.stl"]


def test_complete_workflow_run_open_starts_orca_for_each_plate(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plate_a_stl = run_directory / "machine_plate_a.stl"
    plate_b_stl = run_directory / "machine_plate_b.stl"
    plate_a_stl.write_text("solid a\n", encoding="utf-8")
    plate_b_stl.write_text("solid b\n", encoding="utf-8")
    plate_a_process = run_directory / "machine_plate_a_process.json"
    plate_b_process = run_directory / "machine_plate_b_process.json"
    plate_a_process.write_text("{}", encoding="utf-8")
    plate_b_process.write_text("{}", encoding="utf-8")

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    popen_calls = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        (Path(output_dir) / "machine_settings.json").write_text("{}", encoding="utf-8")

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--export-3mf" in cmd:
            project_name = cmd[cmd.index("--export-3mf") + 1]
            (run_directory / project_name).write_text("3mf", encoding="utf-8")
        return SubprocessResult(0, [], [])

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return object()

    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.generate_settings", fake_generate_settings
    )
    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.execute_subprocess",
        fake_execute_subprocess,
    )
    monkeypatch.setattr("shellforgepy.workflow.workflow.subprocess.Popen", fake_popen)

    args = argparse.Namespace(
        slice=True,
        upload=False,
        open=True,
        part_file=None,
        process_file=None,
        master_settings_dir=str(master_settings_dir),
        orca_executable=str(orca_exec),
        orca_debug=None,
        printer=None,
    )
    config = {"orca": {"debug_level": 6}}
    manifest = {
        "plates": [
            {
                "name": "plate_a",
                "assembly_path": str(plate_a_stl),
                "process_data_path": str(plate_a_process),
            },
            {
                "name": "plate_b",
                "assembly_path": str(plate_b_stl),
                "process_data_path": str(plate_b_process),
            },
        ]
    }

    result = complete_workflow_run(
        args,
        config=config,
        run_directory=run_directory,
        manifest=manifest,
        target_label="machine",
    )

    assert result == 0
    assert popen_calls == [
        [str(orca_exec), str(run_directory / "machine_plate_a.3mf")],
        [str(orca_exec), str(run_directory / "machine_plate_b.3mf")],
    ]


def test_orca_open_commands_uses_open_with_app_bundle_on_macos(monkeypatch, tmp_path):
    app_dir = tmp_path / "OrcaSlicer.app" / "Contents" / "MacOS"
    app_dir.mkdir(parents=True)
    orca_exec = app_dir / "OrcaSlicer"
    orca_exec.write_text("", encoding="utf-8")
    project_a = tmp_path / "a.3mf"
    project_b = tmp_path / "b.3mf"
    project_a.write_text("", encoding="utf-8")
    project_b.write_text("", encoding="utf-8")

    monkeypatch.setattr("shellforgepy.workflow.workflow.sys.platform", "darwin")

    commands = _orca_open_commands(
        orca_exec_path=orca_exec,
        project_paths=[project_a, project_b],
    )

    assert commands == [
        [
            "open",
            "-a",
            str(tmp_path / "OrcaSlicer.app"),
            str(project_a),
        ],
        [
            "open",
            "-a",
            str(tmp_path / "OrcaSlicer.app"),
            str(project_b),
        ],
    ]


def test_run_workflow_logs_metrics_report_from_manifest(monkeypatch, tmp_path, caplog):
    target = tmp_path / "design.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        manifest_path = Path(env[MANIFEST_ENV])
        run_directory = manifest_path.parent
        obj_path = run_directory / "design.obj"
        metrics_report_path = run_directory / "design_metrics_report.txt"
        obj_path.write_text("# obj\n", encoding="utf-8")
        metrics_report_path.write_text(
            "Weight metrics:\ny_axis_moving_mass: 1.174000 kg\n",
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "obj_path": str(obj_path),
                    "metrics_report_path": str(metrics_report_path),
                    "metrics_report_logged": False,
                }
            ),
            encoding="utf-8",
        )
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(
        "shellforgepy.workflow.workflow.execute_subprocess",
        fake_execute_subprocess,
    )

    caplog.set_level(logging.INFO)

    result = run_workflow(_make_args(target, tmp_path))

    assert result == 0
    assert "Metrics report:" in caplog.text
    assert "Weight metrics:" in caplog.text
    assert "y_axis_moving_mass: 1.174000 kg" in caplog.text
