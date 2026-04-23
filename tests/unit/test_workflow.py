import argparse
import json
import logging
import sys
from pathlib import Path

import pytest
import shellforgepy.render.api as render_api
import shellforgepy.workflow.preview_generator as preview_generator_module
import shellforgepy.workflow.upload_to_printer as upload_to_printer_module
import shellforgepy.workflow.workflow as workflow_module
from shellforgepy.workflow.workflow import (
    MANIFEST_ENV,
    SubprocessResult,
    WorkflowError,
    _normalize_run_argv,
    _orca_open_commands,
    complete_workflow_run,
    main,
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


def _fake_slicer_artifacts(
    output_dir: Path,
    *,
    machine_name: str = "machine_settings.json",
    process_name: str = "process_settings.json",
    filament_names: list[str] | None = None,
):
    if filament_names is None:
        filament_names = []
    return {
        "machine_settings_path": str((output_dir / machine_name).resolve()),
        "process_settings_path": str((output_dir / process_name).resolve()),
        "filament_settings_paths": [
            str((output_dir / "filaments" / name).resolve()) for name in filament_names
        ],
        "print_host_path": None,
        "used_master_settings_files": [],
    }


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

    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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

    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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
        output_dir = Path(output_dir)
        generated_process_inputs.append(Path(process_data_file).name)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            slicer_inputs.append(Path(cmd[-1]).name)
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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


def test_complete_workflow_run_loads_only_current_plate_filament(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    filaments_dir = run_directory / "filaments"
    filaments_dir.mkdir()

    plate_a_stl = run_directory / "machine_plate_a.stl"
    plate_b_stl = run_directory / "machine_plate_b.stl"
    plate_a_stl.write_text("solid a\n", encoding="utf-8")
    plate_b_stl.write_text("solid b\n", encoding="utf-8")

    plate_a_process = run_directory / "machine_plate_a_process.json"
    plate_b_process = run_directory / "machine_plate_b_process.json"
    plate_a_process.write_text(
        json.dumps({"filament": "FilamentPETGCF"}), encoding="utf-8"
    )
    plate_b_process.write_text(
        json.dumps({"filament": "FilamenteSunTPU95A"}), encoding="utf-8"
    )

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    slicer_filament_args = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        process_name = "ProcessMegeMaster.json"
        machine_name = "MegeMasterMachine.json"
        (output_dir / machine_name).write_text("{}", encoding="utf-8")
        (output_dir / process_name).write_text("{}", encoding="utf-8")

        filament_name = json.loads(Path(process_data_file).read_text(encoding="utf-8"))[
            "filament"
        ]
        (output_dir / "filaments" / f"{filament_name}.json").write_text(
            json.dumps({"name": filament_name}),
            encoding="utf-8",
        )
        return _fake_slicer_artifacts(
            output_dir,
            machine_name=machine_name,
            process_name=process_name,
            filament_names=[f"{filament_name}.json"],
        )

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-filaments" in cmd:
            slicer_filament_args.append(cmd[cmd.index("--load-filaments") + 1])
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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
    assert slicer_filament_args == [
        str(run_directory / "filaments" / "FilamentPETGCF.json"),
        str(run_directory / "filaments" / "FilamenteSunTPU95A.json"),
    ]


def test_complete_workflow_run_excludes_builder_debug_json_from_orca_settings(
    monkeypatch, tmp_path
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plate_stl = run_directory / "machine_plate_a.stl"
    plate_stl.write_text("solid a\n", encoding="utf-8")
    plate_process = run_directory / "machine_plate_a_process.json"
    plate_process.write_text("{}", encoding="utf-8")
    (run_directory / "placed_assemblies_bounding_boxes.json").write_text(
        json.dumps({"schema_version": "debug"}),
        encoding="utf-8",
    )

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    slicer_settings_args = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        (output_dir / "MegeMasterMachine.json").write_text("{}", encoding="utf-8")
        (output_dir / "ProcessMegeMaster.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(
            output_dir,
            machine_name="MegeMasterMachine.json",
            process_name="ProcessMegeMaster.json",
        )

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            slicer_settings_args.append(cmd[cmd.index("--load-settings") + 1])
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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
                "assembly_path": str(plate_stl),
                "process_data_path": str(plate_process),
            }
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
    assert slicer_settings_args == [
        ";".join(
            [
                str(run_directory / "MegeMasterMachine.json"),
                str(run_directory / "ProcessMegeMaster.json"),
            ]
        )
    ]


def test_complete_workflow_run_errors_for_manifest_plate_without_process_data(
    monkeypatch, tmp_path
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plate_a_stl = run_directory / "machine_plate_a.stl"
    plate_b_stl = run_directory / "machine_plate_b.stl"
    plate_a_stl.write_text("solid a\n", encoding="utf-8")
    plate_b_stl.write_text("solid b\n", encoding="utf-8")
    plate_a_process = run_directory / "machine_plate_a_process.json"
    plate_a_process.write_text("{}", encoding="utf-8")

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        return SubprocessResult(0, [], [])

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

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
                "process_data_path": None,
            },
        ]
    }

    with pytest.raises(
        WorkflowError,
        match="generated process data JSON for plate 'plate_b'",
    ):
        complete_workflow_run(
            args,
            config=config,
            run_directory=run_directory,
            manifest=manifest,
            target_label="machine",
        )


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
        output_dir = Path(output_dir)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--export-3mf" in cmd:
            project_name = cmd[cmd.index("--export-3mf") + 1]
            (run_directory / project_name).write_text("3mf", encoding="utf-8")
        return SubprocessResult(0, [], [])

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return object()

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)
    monkeypatch.setattr(workflow_module.subprocess, "Popen", fake_popen)

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


def test_complete_workflow_run_preserves_and_uploads_each_plate_gcode(
    monkeypatch, tmp_path
):
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

    upload_calls = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            source_name = Path(cmd[-1]).stem
            (run_directory / "plate_1.gcode").write_text(source_name, encoding="utf-8")
        return SubprocessResult(0, [], [])

    def fake_upload_to_printer(gcode_file, printer):
        upload_calls.append(
            (Path(gcode_file).name, printer, Path(gcode_file).read_text())
        )

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)
    monkeypatch.setattr(
        upload_to_printer_module,
        "upload_to_printer",
        fake_upload_to_printer,
    )

    args = argparse.Namespace(
        slice=True,
        upload=True,
        open=False,
        part_file=None,
        process_file=None,
        master_settings_dir=str(master_settings_dir),
        orca_executable=str(orca_exec),
        orca_debug=None,
        printer="192.168.0.10:4409",
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
    assert upload_calls == [
        ("machine_plate_a.gcode", "192.168.0.10:4409", "machine_plate_a"),
        ("machine_plate_b.gcode", "192.168.0.10:4409", "machine_plate_b"),
    ]
    assert not (run_directory / "plate_1.gcode").exists()


def test_orca_open_commands_uses_open_with_app_bundle_on_macos(monkeypatch, tmp_path):
    app_dir = tmp_path / "OrcaSlicer.app" / "Contents" / "MacOS"
    app_dir.mkdir(parents=True)
    orca_exec = app_dir / "OrcaSlicer"
    orca_exec.write_text("", encoding="utf-8")
    project_a = tmp_path / "a.3mf"
    project_b = tmp_path / "b.3mf"
    project_a.write_text("", encoding="utf-8")
    project_b.write_text("", encoding="utf-8")

    monkeypatch.setattr(workflow_module.sys, "platform", "darwin")

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

    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)

    caplog.set_level(logging.INFO)

    result = run_workflow(_make_args(target, tmp_path))

    assert result == 0
    assert "Metrics report:" in caplog.text
    assert "Weight metrics:" in caplog.text
    assert "y_axis_moving_mass: 1.174000 kg" in caplog.text


def test_complete_workflow_run_generates_obj_previews_when_enabled(
    monkeypatch, tmp_path
):
    from shellforgepy.render.api import PreviewRenderBatchResult, PreviewRenderResult

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    obj_path = run_directory / "design.obj"
    obj_path.write_text("# obj\n", encoding="utf-8")

    render_calls = []

    def fake_render_obj_views_with_stats(
        obj_path_arg,
        *,
        output_dir,
        views=None,
        width=512,
        height=512,
        filename_prefix=None,
        background_color=(250, 250, 250),
    ):
        render_calls.append(
            {
                "obj_path": Path(obj_path_arg),
                "output_dir": Path(output_dir),
                "views": views,
                "width": width,
                "height": height,
                "filename_prefix": filename_prefix,
            }
        )
        preview_path = Path(output_dir) / "design_front_angle.ppm"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text("preview", encoding="utf-8")
        return PreviewRenderBatchResult(
            obj_path=Path(obj_path_arg),
            scene_load_seconds=0.01,
            total_seconds=0.11,
            triangle_count=42,
            vertex_count=21,
            object_count=2,
            results=(
                PreviewRenderResult(
                    view="front_angle",
                    path=preview_path,
                    width=width,
                    height=height,
                    triangle_count=42,
                    vertex_count=21,
                    object_count=2,
                    render_seconds=0.1,
                ),
            ),
        )

    monkeypatch.setattr(
        render_api,
        "render_obj_views_with_stats",
        fake_render_obj_views_with_stats,
    )

    args = argparse.Namespace(
        slice=False,
        upload=False,
        open=False,
        part_file=None,
        process_file=None,
        master_settings_dir=None,
        orca_executable=None,
        orca_debug=None,
        printer=None,
    )
    config = {"render": {"enabled": True}}
    manifest = {"obj_path": str(obj_path)}

    result = complete_workflow_run(
        args,
        config=config,
        run_directory=run_directory,
        manifest=manifest,
        target_label="design",
    )

    assert result == 0
    assert render_calls == [
        {
            "obj_path": obj_path,
            "output_dir": run_directory / "previews",
            "views": None,
            "width": 512,
            "height": 512,
            "filename_prefix": "design",
        }
    ]


def test_complete_workflow_run_uses_obj_renderer_for_single_plate_gcode_preview(
    monkeypatch, tmp_path
):
    from shellforgepy.render.api import PreviewRenderResult

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    obj_path = run_directory / "machine.obj"
    obj_path.write_text("# obj\n", encoding="utf-8")
    stl_path = run_directory / "machine_plate_a.stl"
    stl_path.write_text("solid machine\n", encoding="utf-8")
    process_path = run_directory / "machine_plate_a_process.json"
    process_path.write_text("{}", encoding="utf-8")

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    render_calls = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            (run_directory / "plate_1.gcode").write_text("gcode", encoding="utf-8")
        return SubprocessResult(0, [], [])

    def fake_render_obj_view_to_image_with_stats(
        obj_path_arg,
        *,
        destination,
        view="front_angle",
        width=512,
        height=512,
        background_color=(250, 250, 250),
        exclude_object_name_prefixes=(),
    ):
        output_path = Path(destination)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("preview", encoding="utf-8")
        render_calls.append(
            {
                "obj_path": Path(obj_path_arg),
                "destination": output_path,
                "view": view,
                "width": width,
                "height": height,
                "exclude_object_name_prefixes": tuple(exclude_object_name_prefixes),
            }
        )
        return PreviewRenderResult(
            view=view,
            path=output_path,
            width=width,
            height=height,
            triangle_count=42,
            vertex_count=21,
            object_count=2,
            render_seconds=0.1,
        )

    def fail_render_stl_to_png(**kwargs):
        raise AssertionError("STL preview fallback should not be used")

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)
    monkeypatch.setattr(
        render_api,
        "render_obj_view_to_image_with_stats",
        fake_render_obj_view_to_image_with_stats,
    )
    monkeypatch.setattr(
        preview_generator_module,
        "render_stl_to_png",
        fail_render_stl_to_png,
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
    config = {"orca": {"debug_level": 6}, "render": {"width": 512, "height": 512}}
    manifest = {
        "obj_path": str(obj_path),
        "plates": [
            {
                "name": "plate_a",
                "assembly_path": str(stl_path),
                "process_data_path": str(process_path),
            }
        ],
    }

    result = complete_workflow_run(
        args,
        config=config,
        run_directory=run_directory,
        manifest=manifest,
        target_label="machine",
    )

    assert result == 0
    assert render_calls == [
        {
            "obj_path": obj_path,
            "destination": run_directory / "machine_plate_a_preview.png",
            "view": "front_angle",
            "width": 512,
            "height": 512,
            "exclude_object_name_prefixes": ("plate_boundary_",),
        }
    ]
    assert (run_directory / "machine_plate_a_preview.png").exists()


def test_complete_workflow_run_uses_plate_obj_for_multi_plate_gcode_previews(
    monkeypatch, tmp_path
):
    from shellforgepy.render.api import PreviewRenderResult

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plate_a_stl = run_directory / "machine_plate_a.stl"
    plate_b_stl = run_directory / "machine_plate_b.stl"
    plate_a_stl.write_text("solid a\n", encoding="utf-8")
    plate_b_stl.write_text("solid b\n", encoding="utf-8")
    plate_a_obj = run_directory / "machine_plate_a.obj"
    plate_b_obj = run_directory / "machine_plate_b.obj"
    plate_a_obj.write_text("# obj a\n", encoding="utf-8")
    plate_b_obj.write_text("# obj b\n", encoding="utf-8")
    plate_a_process = run_directory / "machine_plate_a_process.json"
    plate_b_process = run_directory / "machine_plate_b_process.json"
    plate_a_process.write_text("{}", encoding="utf-8")
    plate_b_process.write_text("{}", encoding="utf-8")

    orca_exec = tmp_path / "orca"
    orca_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    master_settings_dir = tmp_path / "masters"
    master_settings_dir.mkdir()

    render_calls = []

    def fake_generate_settings(*, process_data_file, output_dir, master_settings_dir):
        output_dir = Path(output_dir)
        (output_dir / "machine_settings.json").write_text("{}", encoding="utf-8")
        (output_dir / "process_settings.json").write_text("{}", encoding="utf-8")
        return _fake_slicer_artifacts(output_dir)

    def fake_execute_subprocess(cmd, *, env=None, cwd=None, stdin_data=None):
        if "--load-settings" in cmd:
            source_name = Path(cmd[-1]).stem
            (run_directory / "plate_1.gcode").write_text(source_name, encoding="utf-8")
        return SubprocessResult(0, [], [])

    def fake_render_obj_view_to_image_with_stats(
        obj_path_arg,
        *,
        destination,
        view="front_angle",
        width=512,
        height=512,
        background_color=(250, 250, 250),
        exclude_object_name_prefixes=(),
    ):
        output_path = Path(destination)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("preview", encoding="utf-8")
        render_calls.append(
            (
                Path(obj_path_arg).name,
                output_path.name,
                view,
                tuple(exclude_object_name_prefixes),
            )
        )
        return PreviewRenderResult(
            view=view,
            path=output_path,
            width=width,
            height=height,
            triangle_count=42,
            vertex_count=21,
            object_count=2,
            render_seconds=0.1,
        )

    def fail_render_stl_to_png(**kwargs):
        raise AssertionError("STL preview fallback should not be used")

    monkeypatch.setattr(workflow_module, "generate_settings", fake_generate_settings)
    monkeypatch.setattr(workflow_module, "execute_subprocess", fake_execute_subprocess)
    monkeypatch.setattr(
        render_api,
        "render_obj_view_to_image_with_stats",
        fake_render_obj_view_to_image_with_stats,
    )
    monkeypatch.setattr(
        preview_generator_module,
        "render_stl_to_png",
        fail_render_stl_to_png,
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
                "obj_path": str(plate_a_obj),
            },
            {
                "name": "plate_b",
                "assembly_path": str(plate_b_stl),
                "process_data_path": str(plate_b_process),
                "obj_path": str(plate_b_obj),
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
    assert render_calls == [
        (
            "machine_plate_a.obj",
            "machine_plate_a_preview.png",
            "front_angle",
            ("plate_boundary_",),
        ),
        (
            "machine_plate_b.obj",
            "machine_plate_b_preview.png",
            "front_angle",
            ("plate_boundary_",),
        ),
    ]


def test_normalize_run_argv_moves_workflow_flags_before_target():
    argv = ["run", "design.py", "--slice", "--open"]

    assert _normalize_run_argv(argv) == ["run", "--slice", "--open", "design.py"]


def test_normalize_run_argv_preserves_target_args_after_separator():
    argv = ["run", "design.py", "--slice", "--", "--open", "--foo"]

    assert _normalize_run_argv(argv) == [
        "run",
        "--slice",
        "design.py",
        "--",
        "--open",
        "--foo",
    ]


def test_main_accepts_workflow_flags_after_target(monkeypatch, tmp_path):
    target = tmp_path / "design.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    captured = {}

    def fake_run_workflow(args):
        captured["slice"] = args.slice
        captured["open"] = args.open
        captured["target"] = args.target
        captured["target_args"] = list(args.target_args)
        return 0

    monkeypatch.setattr(workflow_module, "run_workflow", fake_run_workflow)

    result = main(["run", str(target), "--slice", "--open"])

    assert result == 0
    assert captured == {
        "slice": True,
        "open": True,
        "target": str(target),
        "target_args": [],
    }
