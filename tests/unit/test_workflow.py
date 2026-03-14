import argparse
import json
import sys
from pathlib import Path

import pytest
from shellforgepy.workflow.workflow import (
    MANIFEST_ENV,
    SubprocessResult,
    WorkflowError,
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
