import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest
from shellforgepy.builder import builder


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
    expected_hash = builder._hash_with_dependencies(
        {"width": 10.0, "height": 20.0, "label": "v10"},
        [],
        None,
        result["version_inputs"],
    )
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
    assert saved_metadata["version_inputs"]["resource_sha256"]
    assert saved_metadata["version_inputs"]["generator_source_sha256"]

    monkeypatch.setattr(
        builder,
        "_export_part_to_step",
        lambda part, destination: pytest.fail(
            f"cache hit should not export artifacts again: {destination}"
        ),
    )

    cached_results = builder.build_from_file(config_path)
    assert cached_results[0]["cache_hit"] is True
    assert cached_results[0]["parameter_hash"] == expected_hash


def test_build_from_file_injects_global_context_when_generator_accepts_it(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "context_generators.py",
        "\n".join(
            [
                "def make_widget(*, width, context=None):",
                "    return f\"widget-{width}-{context.get('BIG_THING', 'missing')}\"",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "context_assembly.yaml",
        "\n".join(
            [
                "Parameters:",
                "  width:",
                "    Type: Float",
                "Parts:",
                "  ContextWidget:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.context_generators.make_widget",
                "      Properties:",
                "        width:",
                "          $ref: width",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                "  BIG_THING: 500",
                "assemblies:",
                "  - name: context_assembly",
                "    parameters:",
                "      width: 10",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(
        builder,
        "_export_part_to_step",
        fake_export,
    )
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.context_generators", raising=False)

    results = builder.build_from_file(config_path)

    assert results[0]["generator_context"] == {"BIG_THING": 500}
    expected_hash = builder._hash_with_dependencies(
        {"width": 10.0},
        [],
        {"BIG_THING": 500},
        results[0]["version_inputs"],
    )
    assert results[0]["parameter_hash"] == expected_hash
    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-500"
    )


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
            "source_assembly_hash": frame_result["parameter_hash"],
            "source_version_inputs": frame_result["version_inputs"],
            "step_path": frame_result["artifacts"]["leader_step"],
        }
    ]
    expected_feet_hash = builder._hash_with_dependencies(
        {"height": 9.0},
        [
            builder._DependencyInjection(
                kwarg_name="frame",
                assembly_name="frame_assembly",
                artifact="leader",
                source_parameter_hash=frame_result["parameter_hash"],
                source_version_inputs=frame_result["version_inputs"],
                step_path=Path(frame_result["artifacts"]["leader_step"]),
                part=None,
            )
        ],
        None,
        feet_result["version_inputs"],
    )
    assert feet_result["parameter_hash"] == expected_feet_hash
    assert (
        feet_result["dependencies"][0]["source_assembly_hash"]
        == frame_result["parameter_hash"]
    )
    assert (
        feet_result["dependencies"][0]["source_version_inputs"]
        == frame_result["version_inputs"]
    )
    assert (
        Path(feet_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "feet-on-frame-42.0-h9.0"
    )


def test_build_from_file_rebuilds_assembly_and_dependents_when_generator_code_changes(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")

    generator_path = src_dir / "dependency_generators.py"
    _write_file(
        generator_path,
        "\n".join(
            [
                "def make_frame(*, width):",
                "    return f'frame-v1-{width}'",
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

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    def fake_import(step_path):
        return Path(step_path).read_text(encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(builder, "_import_dependency_part", fake_import)
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.dependency_generators", raising=False)

    initial_results = builder.build_from_file(config_path)
    initial_frame, initial_feet = initial_results

    _write_file(
        generator_path,
        "\n".join(
            [
                "def make_frame(*, width):",
                "    return f'frame-v2-{width}'",
                "def make_feet(*, frame, height):",
                "    return f'feet-on-{frame}-h{height}'",
            ]
        ),
    )

    rebuilt_results = builder.build_from_file(config_path)
    rebuilt_frame, rebuilt_feet = rebuilt_results

    assert rebuilt_frame["cache_hit"] is False
    assert rebuilt_feet["cache_hit"] is False
    assert rebuilt_frame["parameter_hash"] != initial_frame["parameter_hash"]
    assert rebuilt_feet["parameter_hash"] != initial_feet["parameter_hash"]
    assert (
        rebuilt_feet["dependencies"][0]["source_assembly_hash"]
        == rebuilt_frame["parameter_hash"]
    )
    assert (
        Path(rebuilt_frame["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "frame-v2-42.0"
    )
    assert (
        Path(rebuilt_feet["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "feet-on-frame-v2-42.0-h9.0"
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
    (tmp_path / "frame.yaml").write_text("Builder: {}\n", encoding="utf-8")
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


def test_run_builder_visualization_builds_implicit_placement_dependencies(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: y_axis",
                "    depends_on: []",
                "  - name: printer_frame",
                "    depends_on: []",
                "  - name: print_bed",
                "    depends_on: []",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "y_axis.yaml").write_text(
        "\n".join(
            [
                "placement:",
                "  alignments:",
                "    - part: y_axis.non_production_parts.anchor",
                "      to: print_bed.non_production_parts.buffer_1",
                "      alignment: CENTER",
                "Builder:",
                "  Visualization:",
                "    parts:",
                "      - source: dependencies",
                "        assembly: printer_frame",
                "        artifact: leader",
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
                "assembly_name": "y_axis",
                "artifact_dir": str(tmp_path / "repo" / "y_axis"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "y_axis.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "printer_frame",
                "artifact_dir": str(tmp_path / "repo" / "printer_frame"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "printer_frame.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "print_bed",
                "artifact_dir": str(tmp_path / "repo" / "print_bed"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "print_bed.yaml"),
                "cache_hit": False,
            },
        ]

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)
    monkeypatch.setattr(builder, "_export_scene_for_assembly", lambda **kwargs: 0)

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["y_axis"],
            repository_dir=str(tmp_path / "repo"),
            force=False,
            visualize=True,
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
    assert captured["assembly_names"] == ["print_bed", "printer_frame", "y_axis"]


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


def test_materialize_rule_parts_resolves_declared_scene_dependencies_from_config(
    monkeypatch, tmp_path
):
    dependency_step = tmp_path / "print_bed__leader.step"
    dependency_step.write_text("print-bed", encoding="utf-8")

    metadata = {
        "assembly_name": "y_axis",
        "resource_file": str(tmp_path / "y_axis.yaml"),
        "public_parameters": {},
        "generator_kwargs": {},
        "dependencies": [],
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "dependencies",
                        "assembly": "print_bed",
                        "artifact": "leader",
                        "name": "print_bed",
                    }
                ]
            }
        }
    }
    built_results_by_name = {
        "print_bed": {
            "assembly_name": "print_bed",
            "artifacts": {"leader_step": str(dependency_step)},
        }
    }
    config_data = {
        "assemblies": [
            {"name": "print_bed", "depends_on": []},
            {"name": "y_axis", "depends_on": ["print_bed"]},
        ]
    }

    monkeypatch.setattr(builder, "_import_dependency_part", lambda path: f"part:{path}")

    parts = builder._materialize_rule_parts(
        metadata,
        resource_data,
        "visualization",
        built_results_by_name,
        tmp_path,
        config_data,
    )

    assert len(parts) == 1
    assert parts[0]["name"] == "print_bed"
    assert parts[0]["part"] == f"part:{dependency_step}"


def test_materialize_rule_parts_resolves_animation_from_metadata_context(
    monkeypatch, tmp_path
):
    leader_step = tmp_path / "bed.step"
    leader_step.write_text("bed", encoding="utf-8")

    metadata = {
        "assembly_name": "print_bed",
        "public_parameters": {"print_bed_y_travel": 325},
        "generator_kwargs": {},
        "generator_context": {"BIG_THING": 500},
        "artifacts": {"leader_step": str(leader_step)},
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "self",
                        "artifact": "leader",
                        "name": "print_bed",
                        "animation": {"bed_y": [0, {"$ref": "print_bed_y_travel"}, 0]},
                    }
                ]
            }
        }
    }

    monkeypatch.setattr(builder, "_import_dependency_part", lambda path: f"part:{path}")

    parts = builder._materialize_rule_parts(
        metadata,
        resource_data,
        "visualization",
        {"print_bed": metadata},
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["animation"] == {"bed_y": [0, 325, 0]}


def test_apply_placement_alignments_moves_only_the_owning_assembly(
    monkeypatch, tmp_path
):
    y_axis_leader = tmp_path / "y_axis_leader.step"
    y_axis_leader.write_text("y-leader", encoding="utf-8")
    y_axis_anchor = tmp_path / "y_axis_anchor.step"
    y_axis_anchor.write_text("y-anchor", encoding="utf-8")
    print_bed_leader = tmp_path / "print_bed_leader.step"
    print_bed_leader.write_text("bed-leader", encoding="utf-8")
    print_bed_target = tmp_path / "print_bed_target.step"
    print_bed_target.write_text("bed-target", encoding="utf-8")

    scene_parts = [
        {"assembly_name": "y_axis", "part": "scene-y-axis"},
        {"assembly_name": "print_bed", "part": "scene-print-bed"},
    ]
    selected_metadata = {
        "assembly_name": "y_axis",
        "public_parameters": {"gap": 12},
        "generator_kwargs": {},
    }
    resource_data = {
        "placement": {
            "alignments": [
                {
                    "part": "y_axis.non_production_parts.anchor",
                    "to": "print_bed.non_production_parts.buffer_1",
                    "alignment": "CENTER",
                    "axes": [0, 1],
                },
                {
                    "part": "y_axis",
                    "to": "print_bed",
                    "alignment": "STACK_TOP",
                    "stack_gap": {"$ref": "gap"},
                },
            ]
        }
    }
    built_results_by_name = {
        "y_axis": {
            "assembly_name": "y_axis",
            "artifacts": {
                "leader_step": str(y_axis_leader),
                "non_production_parts": [
                    {"name": "anchor", "path": str(y_axis_anchor)},
                ],
            },
        },
        "print_bed": {
            "assembly_name": "print_bed",
            "artifacts": {
                "leader_step": str(print_bed_leader),
                "non_production_parts": [
                    {"name": "buffer_1", "path": str(print_bed_target)},
                ],
            },
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        label = f"{alignment.name}:{moving_anchor}->{target_anchor}:{axes}:{stack_gap}"
        return lambda part: f"{part}|{label}"

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placed_parts = builder._apply_placement_alignments(
        scene_parts,
        selected_metadata=selected_metadata,
        resource_data=resource_data,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={"assemblies": [{"name": "y_axis"}, {"name": "print_bed"}]},
    )

    assert placed_parts[0]["part"] == (
        "scene-y-axis"
        "|CENTER:y-anchor->bed-target:[0, 1]:0"
        "|STACK_TOP:y-leader->bed-leader:None:12"
    )
    assert placed_parts[1]["part"] == "scene-print-bed"


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
