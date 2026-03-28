import json
from pathlib import Path

import pytest
import yaml
from shellforgepy.slicing.orca_slicer_settings_generator import generate_settings


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_master_settings(
    master_dir: Path,
    *,
    filament_name: str,
    filament_data: dict,
    process_data: dict | None = None,
    machine_data: dict | None = None,
) -> None:
    process_master = {"name": "TestProcess", "type": "process"}
    if process_data:
        process_master.update(process_data)

    machine_master = {"name": "TestMachine", "type": "machine"}
    if machine_data:
        machine_master.update(machine_data)

    filament_master = {"name": filament_name, "type": "filament"}
    filament_master.update(filament_data)

    _write_yaml(master_dir / "machine.yaml", machine_master)
    _write_yaml(master_dir / "process.yaml", process_master)
    _write_yaml(master_dir / "filament.yaml", filament_master)


def _write_process_data(
    path: Path,
    *,
    filament_name: str,
    part_file: Path,
    process_overrides: dict,
) -> None:
    path.write_text(
        json.dumps(
            {
                "filament": filament_name,
                "process_overrides": process_overrides,
                "part_file": str(part_file),
            }
        ),
        encoding="utf-8",
    )


def test_generate_settings_raises_for_unknown_override_key(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={"filament_flow_ratio": ["1.0"]},
        process_data={"layer_height": "0.2"},
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={
            "layer_height": "0.24",
            "enable_pressure_advance": "1",
            "pressure_advance": "0.03",
        },
    )

    with pytest.raises(ValueError, match="Missing keys"):
        generate_settings(process_data_file, output_dir, master_dir)


def test_generate_settings_raises_for_ambiguous_override_key(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={"shared_key": ["1.0"]},
        process_data={"shared_key": "0.2"},
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={"shared_key": "0.24"},
    )

    with pytest.raises(ValueError, match="Ambiguous keys"):
        generate_settings(process_data_file, output_dir, master_dir)


def test_generate_settings_applies_filament_pressure_advance_overrides(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={
            "filament_flow_ratio": ["1.0"],
            "enable_pressure_advance": ["0"],
            "pressure_advance": ["0.02"],
            "adaptive_pressure_advance": ["0"],
            "adaptive_pressure_advance_bridges": ["0"],
            "adaptive_pressure_advance_overhangs": ["0"],
        },
        process_data={"layer_height": "0.2"},
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={
            "layer_height": "0.24",
            "enable_pressure_advance": "1",
            "pressure_advance": "0.03",
        },
    )

    generate_settings(process_data_file, output_dir, master_dir)

    generated_filament = json.loads(
        (output_dir / "filaments" / "TestFilament.json").read_text(encoding="utf-8")
    )
    generated_process = json.loads(
        (output_dir / "TestProcess.json").read_text(encoding="utf-8")
    )

    assert generated_filament["enable_pressure_advance"] == "1"
    assert generated_filament["pressure_advance"] == "0.03"
    assert generated_process["layer_height"] == "0.24"
    assert (output_dir / "part.stl").exists()


def test_generate_settings_applies_process_override_to_source_visible_key(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={"filament_flow_ratio": ["1.0"]},
        process_data={
            "layer_height": "0.2",
            "external_perimeter_speed": "45",
        },
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={
            "layer_height": "0.24",
            "external_perimeter_speed": "60",
        },
    )

    generate_settings(process_data_file, output_dir, master_dir)

    generated_process = json.loads(
        (output_dir / "TestProcess.json").read_text(encoding="utf-8")
    )

    assert generated_process["layer_height"] == "0.24"
    assert generated_process["external_perimeter_speed"] == "60"


def test_generate_settings_stringifies_numeric_process_overrides(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={"filament_flow_ratio": ["1.0"]},
        process_data={
            "support_object_first_layer_gap": "0.2",
        },
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={
            "support_object_first_layer_gap": 0.8,
        },
    )

    generate_settings(process_data_file, output_dir, master_dir)

    generated_process = json.loads(
        (output_dir / "TestProcess.json").read_text(encoding="utf-8")
    )

    assert generated_process["support_object_first_layer_gap"] == "0.8"


def test_generate_settings_allows_disabled_pressure_advance_override(tmp_path):
    master_dir = tmp_path / "masters"
    master_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    part_file = tmp_path / "part.stl"
    part_file.write_text("solid part\nendsolid part\n", encoding="utf-8")
    process_data_file = tmp_path / "process.json"

    _write_master_settings(
        master_dir,
        filament_name="TestFilament",
        filament_data={
            "enable_pressure_advance": ["1"],
            "pressure_advance": ["0.02"],
            "adaptive_pressure_advance": ["0"],
            "adaptive_pressure_advance_bridges": ["0"],
            "adaptive_pressure_advance_overhangs": ["0"],
        },
    )
    _write_process_data(
        process_data_file,
        filament_name="TestFilament",
        part_file=part_file,
        process_overrides={
            "enable_pressure_advance": "0",
            "pressure_advance": "0.07",
        },
    )

    generate_settings(process_data_file, output_dir, master_dir)

    generated_filament = json.loads(
        (output_dir / "filaments" / "TestFilament.json").read_text(encoding="utf-8")
    )

    assert generated_filament["enable_pressure_advance"] == "0"
    assert generated_filament["pressure_advance"] == "0.07"
