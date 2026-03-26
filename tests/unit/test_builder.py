import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest
from shellforgepy.builder import builder
from shellforgepy.construct.part_parameters import PartParameters


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_globals_supports_refs_subs_and_exprs():
    resolved = builder.resolve_globals(
        {
            "base": 12,
            "double": {"$expr": {"$sub": "${base} * 2"}},
            "alias": {"$ref": "double"},
            "label": {"$sub": "frame-${alias}"},
        }
    )

    assert resolved == {
        "base": 12,
        "double": 24,
        "alias": 24,
        "label": "frame-24",
    }


def test_build_from_file_writes_hashed_metadata_and_reuses_cache(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "class FakeComposite:",
                "    def __init__(self, label):",
                "        self.leader = f'leader-{label}'",
                "        self.followers = [f'follower-{label}']",
                "        self.cutters = [f'cutter-{label}']",
                "        self.non_production_parts = [f'np-{label}']",
                "        self.follower_indices_by_name = {'brace': 0}",
                "        self.cutter_indices_by_name = {'hole': 0}",
                "        self.non_production_indices_by_name = {'visual': 0}",
                "    def leaders_followers_fused(self):",
                "        return 'fused-shape'",
                "def make_widget(*, width, height, label):",
                '    return FakeComposite(f"{width}-{height}-{label}")',
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parameters:",
                "  width:",
                "    Type: Float",
                "  height:",
                "    Type: Float",
                "  label:",
                "    Type: String",
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.widget_generator.make_widget",
                "      Properties:",
                "        width:",
                "          $ref: width",
                "        height:",
                "          $ref: height",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                "  base_width: 10",
                "  doubled_width:",
                "    $expr:",
                '      $sub: "${base_width} * 2"',
                "assemblies:",
                "  - name: sample_assembly",
                "    parameters:",
                "      width:",
                "        $ref: base_width",
                "      height:",
                "        $ref: doubled_width",
                "      label:",
                '        $sub: "v${base_width}"',
            ]
        ),
    )

    exported = []

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")
        exported.append(destination)

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)

    results = builder.build_from_file(config_path)

    assert len(results) == 1
    result = results[0]
    expected_hash = PartParameters(
        {"width": 10.0, "height": 20.0, "label": "v10"}
    ).parameters_hash()
    assert result["parameter_hash"] == expected_hash
    assert result["cache_hit"] is False
    assert Path(result["artifacts"]["leader_step"]).exists()
    assert result["artifacts"]["followers"][0]["name"] == "brace"
    assert result["artifacts"]["cutters"][0]["name"] == "hole"
    assert result["artifacts"]["non_production_parts"][0]["name"] == "visual"
    assert Path(result["artifact_dir"]).name == f"sample_assembly__{expected_hash}"

    metadata_path = (
        Path(result["artifact_dir"])
        / f"sample_assembly__{expected_hash}__metadata.json"
    )
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["parameter_hash"] == expected_hash
    assert len(exported) == 5

    monkeypatch.setattr(
        builder,
        "_load_generator_callable",
        lambda path: pytest.fail(
            f"generator should not be reloaded for cache hit: {path}"
        ),
    )

    cached_results = builder.build_from_file(config_path)
    assert cached_results[0]["cache_hit"] is True
    assert cached_results[0]["parameter_hash"] == expected_hash


def test_build_from_file_injects_dependency_part_and_hashes_dependency(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "dependency_generators.py",
        "\n".join(
            [
                "def make_frame(*, width):",
                "    return f'frame-{width}'",
                "def make_feet(*, frame, height):",
                "    return f'feet-on-{frame}-h{height}'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "frame_assembly.yaml",
        "\n".join(
            [
                "Parameters:",
                "  width:",
                "    Type: Float",
                "Parts:",
                "  Frame:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.dependency_generators.make_frame",
                "      Properties:",
                "        width:",
                "          $ref: width",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "feet_assembly.yaml",
        "\n".join(
            [
                "Parameters:",
                "  height:",
                "    Type: Float",
                "Parts:",
                "  Feet:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.dependency_generators.make_feet",
                "      Properties:",
                "        height:",
                "          $ref: height",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: frame_assembly",
                "    parameters:",
                "      width: 42",
                "  - name: feet_assembly",
                "    depends_on:",
                "      - frame_assembly",
                "    inject_parts:",
                "      frame:",
                "        assembly: frame_assembly",
                "        artifact: leader",
                "    parameters:",
                "      height: 9",
            ]
        ),
    )

    exported_parts = {}

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")
        exported_parts[str(destination)] = str(part)

    def fake_import(step_path):
        return Path(step_path).read_text(encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(builder, "_import_dependency_part", fake_import)
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.dependency_generators", raising=False)

    results = builder.build_from_file(config_path)

    assert [result["assembly_name"] for result in results] == [
        "frame_assembly",
        "feet_assembly",
    ]
    frame_result, feet_result = results
    assert feet_result["dependencies"] == [
        {
            "kwarg_name": "frame",
            "assembly_name": "frame_assembly",
            "artifact": "leader",
            "source_parameter_hash": frame_result["parameter_hash"],
            "step_path": frame_result["artifacts"]["leader_step"],
        }
    ]
    expected_feet_hash = PartParameters(
        {
            "height": 9.0,
            "__dependency__frame": f"frame_assembly:leader:{frame_result['parameter_hash']}",
        }
    ).parameters_hash()
    assert feet_result["parameter_hash"] == expected_feet_hash
    assert (
        Path(feet_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "feet-on-frame-42.0-h9.0"
    )


def test_dependency_step_path_supports_dotted_named_artifacts():
    metadata = {
        "assembly_name": "print_bed_assembly",
        "artifacts": {
            "leader_step": "/tmp/leader.step",
            "followers": [{"name": "front_left_uc", "path": "/tmp/follower.step"}],
            "cutters": [{"name": "belt_path_cutter_front", "path": "/tmp/cutter.step"}],
            "non_production_parts": [
                {"name": "damper_left_front", "path": "/tmp/damper.step"}
            ],
        },
    }

    assert (
        builder._dependency_step_path(
            metadata, "non_production_parts.damper_left_front"
        )
        == Path("/tmp/damper.step").resolve()
    )
    assert (
        builder._dependency_step_path(metadata, "followers.front_left_uc")
        == Path("/tmp/follower.step").resolve()
    )
    assert (
        builder._dependency_step_path(metadata, "cutters.belt_path_cutter_front")
        == Path("/tmp/cutter.step").resolve()
    )


def test_run_builder_invokes_build_from_file(monkeypatch, tmp_path):
    captured = {}

    def fake_build_from_file(
        config_file, *, assembly_names=None, repository_dir=None, force=False
    ):
        captured["config_file"] = config_file
        captured["assembly"] = assembly_names
        captured["repository_dir"] = repository_dir
        captured["force"] = force
        return [
            {
                "assembly_name": "frame",
                "artifact_dir": str(tmp_path / "repo" / "frame"),
                "cache_hit": False,
            }
        ]

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)

    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "assemblies:\n  - name: frame\n    depends_on: []\n", encoding="utf-8"
    )

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["frame"],
            repository_dir=str(tmp_path / "repo"),
            force=True,
            visualize=False,
            with_dependents=False,
            production=False,
            slice=False,
            upload=False,
            open=False,
            run_id=None,
            runs_dir=None,
            master_settings_dir=None,
            orca_executable=None,
            orca_debug=None,
            printer=None,
            part_file=None,
            process_file=None,
            config=None,
            verbose=False,
        )
    )

    assert result == 0
    assert captured == {
        "config_file": config_path.resolve(),
        "assembly": ["frame"],
        "repository_dir": str(tmp_path / "repo"),
        "force": True,
    }


def test_run_builder_visualization_expands_dependents_and_exports_scene(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: frame",
                "    depends_on: []",
                "  - name: feet",
                "    depends_on:",
                "      - frame",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_build_from_file(
        config_file, *, assembly_names=None, repository_dir=None, force=False
    ):
        captured["assembly_names"] = assembly_names
        return [
            {
                "assembly_name": "frame",
                "artifact_dir": str(tmp_path / "repo" / "frame"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "frame.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "feet",
                "artifact_dir": str(tmp_path / "repo" / "feet"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "feet.yaml"),
                "cache_hit": False,
            },
        ]

    def fake_export_scene_for_assembly(
        *, args, config_data, build_results, selected_assembly, scene_assembly_names
    ):
        captured["config_data"] = config_data
        captured["selected_assembly"] = selected_assembly
        captured["scene_assembly_names"] = scene_assembly_names
        return 0

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)
    monkeypatch.setattr(
        builder, "_export_scene_for_assembly", fake_export_scene_for_assembly
    )

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["frame"],
            repository_dir=str(tmp_path / "repo"),
            force=False,
            visualize=True,
            with_dependents=True,
            production=False,
            slice=False,
            upload=False,
            open=False,
            run_id=None,
            runs_dir=None,
            master_settings_dir=None,
            orca_executable=None,
            orca_debug=None,
            printer=None,
            part_file=None,
            process_file=None,
            config=None,
            verbose=False,
        )
    )

    assert result == 0
    assert captured["assembly_names"] == ["feet", "frame"]
    assert captured["config_data"] == {
        "assemblies": [
            {"name": "frame", "depends_on": []},
            {"name": "feet", "depends_on": ["frame"]},
        ]
    }
    assert captured["selected_assembly"] == "frame"
    assert captured["scene_assembly_names"] == ["frame", "feet"]


def test_resolve_build_generations_returns_topological_generations():
    generations = builder._resolve_build_generations(
        [
            {"name": "frame", "depends_on": []},
            {"name": "feet", "depends_on": ["frame"]},
            {"name": "gantry", "depends_on": ["frame"]},
            {"name": "printer", "depends_on": ["feet", "gantry"]},
        ],
        ["printer"],
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["frame"],
        ["feet", "gantry"],
        ["printer"],
    ]


def test_resolve_process_data_applies_overrides(monkeypatch):
    class ProcessModule:
        PROCESS_DATA = {
            "filament": "TPU",
            "process_overrides": {"speed": "100"},
        }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.process"
            else real_import_module(module_name)
        ),
    )

    resolved = builder._resolve_process_data(
        {
            "public_parameters": {"bed_temp": 55},
            "generator_kwargs": {"assembly_name": "feet"},
        },
        {
            "Builder": {
                "Production": {
                    "process_data": {
                        "source": "demo.process.PROCESS_DATA",
                        "overrides": {
                            "bed_temperature": {"$ref": "bed_temp"},
                            "process_overrides": {"brim_type": "no_brim"},
                        },
                    }
                }
            }
        },
    )

    assert resolved == {
        "filament": "TPU",
        "bed_temperature": 55,
        "process_overrides": {
            "speed": "100",
            "brim_type": "no_brim",
        },
    }


def test_resolve_export_options_merges_top_level_builder_defaults():
    resolved = builder._resolve_export_options(
        {
            "Builder": {
                "Production": {
                    "arrange": {
                        "export_obj": False,
                    }
                }
            }
        },
        "production",
        {
            "builder_defaults": {
                "Production": {
                    "arrange": {
                        "bed_width": 220,
                        "export_step": True,
                        "export_obj": True,
                        "export_stl": True,
                        "export_individual_parts": True,
                    }
                }
            }
        },
    )

    assert resolved == {
        "prod_gap": 1.0,
        "bed_width": 220,
        "max_build_height": None,
        "export_step": True,
        "export_obj": False,
        "export_individual_parts": True,
        "export_stl": True,
    }
