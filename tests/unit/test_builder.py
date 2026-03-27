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


def test_build_from_file_falls_back_to_same_named_globals_for_missing_parameters(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "fallback_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "fallback_widget_generator.py",
        "\n".join(
            [
                "def make_widget(*, width, height, label):",
                "    return f'widget-{width}-{height}-{label}'",
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
                "      Generator: fallback_demo_pkg.fallback_widget_generator.make_widget",
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
                "  width: 10",
                "  height: 20",
                "  label: global-label",
                "assemblies:",
                "  - name: sample_assembly",
                "    parameters:",
                "      label: explicit-label",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)

    results = builder.build_from_file(config_path)

    monkeypatch.delitem(sys.modules, "fallback_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "fallback_demo_pkg.fallback_widget_generator",
        raising=False,
    )

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-20.0-explicit-label"
    )


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
                "      frame: frame_assembly.leader",
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


def test_build_from_file_injects_dependency_assembly(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "assembly_generators.py",
        "\n".join(
            [
                "class FakeComposite:",
                "    def __init__(self):",
                "        self.leader = 'leader-frame'",
                "        self.followers = ['follower-brace']",
                "        self.cutters = ['cutter-hole']",
                "        self.non_production_parts = ['np-visual']",
                "        self.follower_indices_by_name = {'brace': 0}",
                "        self.cutter_indices_by_name = {'hole': 0}",
                "        self.non_production_indices_by_name = {'visual': 0}",
                "    def leaders_followers_fused(self):",
                "        return 'fused-frame'",
                "def make_frame():",
                "    return FakeComposite()",
                "def make_tool(*, frame):",
                "    return '|'.join([",
                "        frame.leader,",
                "        frame.get_named_follower('brace'),",
                "        frame.get_named_cutter('hole'),",
                "        frame.get_named_non_production_part('visual'),",
                "    ])",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "frame_assembly.yaml",
        "\n".join(
            [
                "Parts:",
                "  Frame:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.assembly_generators.make_frame",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "tool_assembly.yaml",
        "\n".join(
            [
                "Parts:",
                "  Tool:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.assembly_generators.make_tool",
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
                "  - name: tool_assembly",
                "    depends_on:",
                "      - frame_assembly",
                "    inject_parts:",
                "      frame: frame_assembly",
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
    monkeypatch.delitem(sys.modules, "demo_pkg.assembly_generators", raising=False)

    results = builder.build_from_file(config_path)

    frame_result, tool_result = results
    assert tool_result["dependencies"][0]["artifact"] == "assembly"
    assert (
        Path(tool_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "leader-frame|follower-brace|cutter-hole|np-visual"
    )
    assert (
        tool_result["dependencies"][0]["source_assembly_hash"]
        == frame_result["parameter_hash"]
    )


def test_build_from_file_defaults_mapping_injected_part_to_assembly_artifact(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "assembly_generators.py",
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
                "        return 'fused-frame'",
                "def make_frame():",
                "    return FakeComposite('frame')",
                "def make_tool(*, frame):",
                "    return '|'.join([",
                "        frame.leader,",
                "        frame.get_named_follower('brace'),",
                "        frame.get_named_cutter('hole'),",
                "        frame.get_named_non_production_part('visual'),",
                "    ])",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "frame_assembly.yaml",
        "\n".join(
            [
                "Parts:",
                "  Frame:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.assembly_generators.make_frame",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "tool_assembly.yaml",
        "\n".join(
            [
                "Parts:",
                "  Tool:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.assembly_generators.make_tool",
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
                "  - name: tool_assembly",
                "    depends_on:",
                "      - frame_assembly",
                "    inject_parts:",
                "      frame: frame_assembly",
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
    monkeypatch.delitem(sys.modules, "demo_pkg.assembly_generators", raising=False)

    results = builder.build_from_file(config_path)

    _, tool_result = results
    assert tool_result["dependencies"][0]["artifact"] == "assembly"
    assert (
        Path(tool_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "leader-frame|follower-frame|cutter-frame|np-frame"
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
                "      frame: frame_assembly.leader",
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


def test_run_builder_visualization_builds_only_relevant_placement_dependencies(
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
                "  - name: gantry",
                "    depends_on: []",
                "placement:",
                "  alignments:",
                "    - part: printer_frame",
                "      to: gantry",
                "      alignment: CENTER",
                "    - part: y_axis.non_production_parts.anchor",
                "      to: printer_frame.non_production_parts.buffer_1",
                "      alignment: CENTER",
                "    - part: print_bed.non_production_parts.anchor",
                "      to: gantry.non_production_parts.buffer_1",
                "      alignment: CENTER",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "y_axis.yaml").write_text(
        "\n".join(
            [
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
            {
                "assembly_name": "gantry",
                "artifact_dir": str(tmp_path / "repo" / "gantry"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "gantry.yaml"),
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
    assert captured["assembly_names"] == ["gantry", "printer_frame", "y_axis"]


def test_run_builder_visualization_builds_transitive_anchor_dependencies(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: bracket",
                "    depends_on: []",
                "  - name: y_axis",
                "    depends_on: []",
                "  - name: printer_frame",
                "    depends_on: []",
                "  - name: base",
                "    depends_on: []",
                "  - name: unrelated",
                "    depends_on: []",
                "placement:",
                "  alignments:",
                "    - part: printer_frame",
                "      to: base",
                "      alignment: CENTER",
                "    - part: y_axis",
                "      to: printer_frame",
                "      alignment: CENTER",
                "    - part: bracket.non_production_parts.mount",
                "      to: y_axis.non_production_parts.profile_left",
                "      alignment: CENTER",
                "    - part: unrelated",
                "      to: base",
                "      alignment: CENTER",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "bracket.yaml").write_text("Builder: {}\n", encoding="utf-8")

    captured = {}

    def fake_build_from_file(
        config_file, *, assembly_names=None, repository_dir=None, force=False
    ):
        captured["assembly_names"] = assembly_names
        return [
            {
                "assembly_name": "bracket",
                "artifact_dir": str(tmp_path / "repo" / "bracket"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "bracket.yaml"),
                "cache_hit": False,
            },
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
                "assembly_name": "base",
                "artifact_dir": str(tmp_path / "repo" / "base"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "base.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "unrelated",
                "artifact_dir": str(tmp_path / "repo" / "unrelated"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "unrelated.yaml"),
                "cache_hit": False,
            },
        ]

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)
    monkeypatch.setattr(builder, "_export_scene_for_assembly", lambda **kwargs: 0)

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["bracket"],
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
    assert captured["assembly_names"] == ["base", "bracket", "printer_frame", "y_axis"]


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


def test_resolve_build_generations_waits_for_placement_prerequisites_of_injected_parts():
    generations = builder._resolve_build_generations(
        [
            {"name": "printer_frame", "depends_on": []},
            {"name": "print_bed", "depends_on": []},
            {"name": "y_axis", "depends_on": ["printer_frame"]},
            {
                "name": "print_bed_undercarriage",
                "depends_on": ["print_bed"],
            },
            {
                "name": "bracket",
                "depends_on": ["printer_frame", "y_axis"],
                "inject_parts": {
                    "frame": "printer_frame",
                    "y_axis": "y_axis",
                },
            },
        ],
        ["bracket"],
        {
            "placement": {
                "alignments": [
                    {
                        "part": "print_bed.non_production_parts.foil",
                        "to": "printer_frame",
                        "alignment": "TOP",
                    },
                    {
                        "part": "y_axis",
                        "to": "printer_frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "y_axis.non_production_parts.carriages_fused",
                        "to": "print_bed_undercarriage",
                        "alignment": "STACK_BOTTOM",
                    },
                    {
                        "part": "bracket.non_production_parts.mount",
                        "to": "y_axis.non_production_parts.profile_left",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["print_bed", "printer_frame"],
        ["print_bed_undercarriage", "y_axis"],
        ["bracket"],
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
        {"assembly_name": "y_axis", "part": "scene-y-axis", "transform_history": []},
        {
            "assembly_name": "print_bed",
            "part": "scene-print-bed",
            "transform_history": [],
        },
    ]
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
    centers = {
        "y-anchor": (1.0, 2.0, 3.0),
        "bed-target": (10.0, 20.0, 30.0),
        "y-leader": (4.0, 5.0, 6.0),
        "y-leader|CENTER:y-anchor->bed-target:[0, 1]:0": (13.0, 23.0, 6.0),
        "bed-leader": (40.0, 50.0, 60.0),
        "y-anchor|CENTER:y-anchor->bed-target:[0, 1]:0": (10.0, 20.0, 3.0),
        "y-leader|CENTER:y-anchor->bed-target:[0, 1]:0|STACK_TOP:y-leader|CENTER:y-anchor->bed-target:[0, 1]:0->bed-leader:None:12": (
            13.0,
            23.0,
            72.0,
        ),
    }
    captured_logs = []

    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])
    monkeypatch.setattr(
        builder._logger,
        "info",
        lambda message, *args: captured_logs.append(message % args),
    )

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        label = f"{alignment.name}:{moving_anchor}->{target_anchor}:{axes}:{stack_gap}"
        return lambda part: f"{part}|{label}"

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placed_parts = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "globals": {"gap": 12},
            "assemblies": [{"name": "y_axis"}, {"name": "print_bed"}],
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
            },
        },
    )

    assert placed_parts[0]["part"] == (
        "scene-y-axis"
        "|CENTER:y-anchor->bed-target:[0, 1]:0"
        "|STACK_TOP:y-leader|CENTER:y-anchor->bed-target:[0, 1]:0->bed-leader:None:12"
    )
    assert placed_parts[1]["part"] == "scene-print-bed"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [9.0, 18.0, 0.0], "placement_step": 0},
        {"kind": "translate", "vector": [0.0, 0.0, 66.0], "placement_step": 1},
    ]
    assert placed_parts[1]["transform_history"] == []
    assert captured_logs == [
        "Placement step: y_axis.non_production_parts.anchor aligned to print_bed.non_production_parts.buffer_1 via CENTER; moving_anchor_center=(1.0,2.0,3.0); target_anchor_center=(10.0,20.0,30.0); moving_part_position=(4.0,5.0,6.0); target_part_position=(40.0,50.0,60.0); shift=(9.0,18.0,0.0)",
        "Placement step: y_axis aligned to print_bed via STACK_TOP; moving_anchor_center=(13.0,23.0,6.0); target_anchor_center=(40.0,50.0,60.0); moving_part_position=(13.0,23.0,6.0); target_part_position=(40.0,50.0,60.0); shift=(0.0,0.0,66.0)",
    ]


def test_apply_placement_alignments_supports_post_translation(monkeypatch, tmp_path):
    moving_leader = tmp_path / "moving_leader.step"
    moving_leader.write_text("moving-leader", encoding="utf-8")
    moving_anchor = tmp_path / "moving_anchor.step"
    moving_anchor.write_text("moving-anchor", encoding="utf-8")
    target_leader = tmp_path / "target_leader.step"
    target_leader.write_text("target-leader", encoding="utf-8")
    target_anchor = tmp_path / "target_anchor.step"
    target_anchor.write_text("target-anchor", encoding="utf-8")

    scene_parts = [
        {"assembly_name": "moving", "part": "scene-moving"},
        {"assembly_name": "target", "part": "scene-target"},
    ]
    built_results_by_name = {
        "moving": {
            "assembly_name": "moving",
            "artifacts": {
                "leader_step": str(moving_leader),
                "non_production_parts": [
                    {"name": "anchor", "path": str(moving_anchor)},
                ],
            },
        },
        "target": {
            "assembly_name": "target",
            "artifacts": {
                "leader_step": str(target_leader),
                "non_production_parts": [
                    {"name": "anchor", "path": str(target_anchor)},
                ],
            },
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "moving-anchor": (1.0, 2.0, 3.0),
            "target-anchor": (10.0, 20.0, 30.0),
            "moving-leader": (4.0, 5.0, 6.0),
            "target-leader": (40.0, 50.0, 60.0),
            "moving-anchor|ALIGN|POST": (11.0, 20.0, 37.0),
        }[part],
    )

    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|ALIGN"
        ),
    )
    captured_post_translations = []
    monkeypatch.setattr(
        builder,
        "_make_post_translation",
        lambda delta: captured_post_translations.append(delta)
        or (lambda part: f"{part}|POST"),
    )

    placed_parts = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "globals": {"bed_gap": 7},
            "assemblies": [{"name": "moving"}, {"name": "target"}],
            "placement": {
                "alignments": [
                    {
                        "part": "moving.non_production_parts.anchor",
                        "to": "target.non_production_parts.anchor",
                        "alignment": "TOP",
                        "post_translation": [1, 0, {"$ref": "bed_gap"}],
                    }
                ]
            },
        },
    )

    assert captured_post_translations == [(1.0, 0.0, 7.0)]
    assert placed_parts[0]["part"] == "scene-moving|ALIGN|POST"
    assert placed_parts[1]["part"] == "scene-target"


def test_advance_placement_execution_stops_until_missing_assembly_is_built(
    monkeypatch, tmp_path
):
    profile_a_leader = tmp_path / "profile_a.step"
    profile_a_leader.write_text("profile-a", encoding="utf-8")
    profile_b_leader = tmp_path / "profile_b.step"
    profile_b_leader.write_text("profile-b", encoding="utf-8")
    profile_c_leader = tmp_path / "profile_c.step"
    profile_c_leader.write_text("profile-c", encoding="utf-8")

    built_results_by_name = {
        "profile_a": {
            "assembly_name": "profile_a",
            "artifacts": {"leader_step": str(profile_a_leader)},
        },
        "profile_b": {
            "assembly_name": "profile_b",
            "artifacts": {"leader_step": str(profile_b_leader)},
        },
    }
    config_data = {
        "assemblies": [
            {"name": "profile_a"},
            {"name": "profile_b"},
            {"name": "profile_c"},
        ],
        "placement": {
            "alignments": [
                {
                    "part": "profile_b",
                    "to": "profile_a",
                    "alignment": "CENTER",
                },
                {
                    "part": "profile_c",
                    "to": "profile_a",
                    "alignment": "CENTER",
                },
            ]
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "profile-a": (0.0, 0.0, 0.0),
            "profile-b": (10.0, 0.0, 0.0),
            "profile-b|CENTER:profile-b->profile-a": (0.0, 0.0, 0.0),
            "profile-c": (20.0, 0.0, 0.0),
            "profile-c|CENTER:profile-c->profile-a": (0.0, 0.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|{alignment.name}:{moving_anchor}->{target_anchor}"
        ),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data, built_results_by_name
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 1
    assert placement_state.placement_offsets == {"profile_b": (-10.0, 0.0, 0.0)}

    built_results_by_name["profile_c"] = {
        "assembly_name": "profile_c",
        "artifacts": {"leader_step": str(profile_c_leader)},
    }
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 2
    assert placement_state.placement_offsets == {
        "profile_b": (-10.0, 0.0, 0.0),
        "profile_c": (-20.0, 0.0, 0.0),
    }


def test_advance_placement_execution_supports_post_translation(monkeypatch, tmp_path):
    profile_a_leader = tmp_path / "profile_a.step"
    profile_a_leader.write_text("profile-a", encoding="utf-8")
    profile_b_leader = tmp_path / "profile_b.step"
    profile_b_leader.write_text("profile-b", encoding="utf-8")

    built_results_by_name = {
        "profile_a": {
            "assembly_name": "profile_a",
            "artifacts": {"leader_step": str(profile_a_leader)},
        },
        "profile_b": {
            "assembly_name": "profile_b",
            "artifacts": {"leader_step": str(profile_b_leader)},
        },
    }
    config_data = {
        "globals": {"gap": 5},
        "assemblies": [
            {"name": "profile_a"},
            {"name": "profile_b"},
        ],
        "placement": {
            "alignments": [
                {
                    "part": "profile_b",
                    "to": "profile_a",
                    "alignment": "CENTER",
                    "post_translation": [0, {"$ref": "gap"}, 0],
                }
            ]
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "profile-a": (0.0, 0.0, 0.0),
            "profile-b": (10.0, 0.0, 0.0),
            "profile-b|ALIGN|POST": (0.0, 5.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|ALIGN"
        ),
    )
    captured_post_translations = []
    monkeypatch.setattr(
        builder,
        "_make_post_translation",
        lambda delta: captured_post_translations.append(delta)
        or (lambda part: f"{part}|POST"),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data, built_results_by_name
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert captured_post_translations == [(0.0, 5.0, 0.0)]
    assert placement_state.cursor == 1
    assert placement_state.placement_offsets == {"profile_b": (-10.0, 5.0, 0.0)}
    assert placement_state.translation_history["profile_b"][0]("profile-b") == (
        "profile-b|ALIGN|POST"
    )


def test_build_from_file_injects_dependencies_after_eager_placement(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "placement_generators.py",
        "\n".join(
            [
                "def make_profile(*, label):",
                "    return label",
                "def make_corner(*, profile_a, profile_b):",
                "    return f'corner:{profile_a}:{profile_b}'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "profile_a.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  ProfileA:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "profile_b.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  ProfileB:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "corner.yaml",
        "\n".join(
            [
                "Parts:",
                "  Corner:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_corner",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: profile_a",
                "    parameters:",
                "      label: profile-a",
                "  - name: profile_b",
                "    parameters:",
                "      label: profile-b",
                "  - name: corner",
                "    depends_on:",
                "      - profile_a",
                "      - profile_b",
                "    inject_parts:",
                "      profile_a: profile_a.leader",
                "      profile_b: profile_b.leader",
                "placement:",
                "  alignments:",
                "    - part: profile_b",
                "      to: profile_a",
                "      alignment: CENTER",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "profile-a": (0.0, 0.0, 0.0),
            "profile-b": (10.0, 0.0, 0.0),
            "profile-b|CENTER:profile-b->profile-a": (0.0, 0.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|{alignment.name}:{moving_anchor}->{target_anchor}"
        ),
    )
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.placement_generators", raising=False)

    results = builder.build_from_file(config_path)

    assert [result["assembly_name"] for result in results] == [
        "profile_a",
        "profile_b",
        "corner",
    ]
    corner_result = results[2]
    assert (
        Path(corner_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "corner:profile-a:profile-b|CENTER:profile-b->profile-a"
    )
    assert corner_result["dependencies"] == [
        {
            "kwarg_name": "profile_a",
            "assembly_name": "profile_a",
            "artifact": "leader",
            "source_parameter_hash": results[0]["parameter_hash"],
            "source_assembly_hash": results[0]["parameter_hash"],
            "source_version_inputs": results[0]["version_inputs"],
            "step_path": results[0]["artifacts"]["leader_step"],
        },
        {
            "kwarg_name": "profile_b",
            "assembly_name": "profile_b",
            "artifact": "leader",
            "source_parameter_hash": results[1]["parameter_hash"],
            "source_assembly_hash": results[1]["parameter_hash"],
            "source_version_inputs": results[1]["version_inputs"],
            "step_path": results[1]["artifacts"]["leader_step"],
            "source_placement_offset": [-10.0, 0.0, 0.0],
        },
    ]


def test_build_from_file_delays_injected_dependency_consumer_until_placement_prefix_is_ready(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "placement_generators.py",
        "\n".join(
            [
                "def make_profile(*, label):",
                "    return label",
                "def make_bracket(*, y_axis):",
                "    return f'bracket:{y_axis}'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "printer_frame.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  PrinterFrame:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "print_bed.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  PrintBed:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "print_bed_undercarriage.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  PrintBedUndercarriage:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "y_axis.yaml",
        "\n".join(
            [
                "Parameters:",
                "  label:",
                "    Type: String",
                "Parts:",
                "  YAxis:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_profile",
                "      Properties:",
                "        label:",
                "          $ref: label",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "bracket.yaml",
        "\n".join(
            [
                "Parts:",
                "  Bracket:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.placement_generators.make_bracket",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: printer_frame",
                "    parameters:",
                "      label: printer-frame",
                "  - name: print_bed",
                "    parameters:",
                "      label: print-bed",
                "  - name: y_axis",
                "    depends_on:",
                "      - printer_frame",
                "    parameters:",
                "      label: y-axis",
                "  - name: print_bed_undercarriage",
                "    depends_on:",
                "      - print_bed",
                "    parameters:",
                "      label: print-bed-undercarriage",
                "  - name: bracket",
                "    depends_on:",
                "      - printer_frame",
                "      - y_axis",
                "    inject_parts:",
                "      y_axis: y_axis.leader",
                "placement:",
                "  alignments:",
                "    - part: print_bed",
                "      to: printer_frame",
                "      alignment: TOP",
                "    - part: y_axis",
                "      to: printer_frame",
                "      alignment: CENTER",
                "    - part: y_axis",
                "      to: print_bed_undercarriage",
                "      alignment: STACK_BOTTOM",
                "    - part: bracket",
                "      to: y_axis",
                "      alignment: CENTER",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "printer-frame": (0.0, 0.0, 0.0),
            "print-bed": (5.0, 0.0, 0.0),
            "print-bed|TOP:print-bed->printer-frame": (0.0, 0.0, 10.0),
            "print-bed-undercarriage": (20.0, 0.0, 0.0),
            "y-axis": (10.0, 0.0, 0.0),
            "y-axis|CENTER:y-axis->printer-frame": (0.0, 0.0, 0.0),
            "y-axis|CENTER:y-axis->printer-frame|STACK_BOTTOM:y-axis|CENTER:y-axis->printer-frame->print-bed-undercarriage": (
                20.0,
                0.0,
                0.0,
            ),
            "bracket:y-axis|CENTER:y-axis->printer-frame|STACK_BOTTOM:y-axis|CENTER:y-axis->printer-frame->print-bed-undercarriage": (
                30.0,
                0.0,
                0.0,
            ),
        }[part],
    )

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        if isinstance(moving_anchor, str) and moving_anchor.startswith("bracket:"):
            return lambda part: part
        return lambda part: f"{part}|{alignment.name}:{moving_anchor}->{target_anchor}"

    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        fake_make_translation,
    )

    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.placement_generators", raising=False)

    results = builder.build_from_file(config_path)

    assert [result["assembly_name"] for result in results] == [
        "print_bed",
        "printer_frame",
        "print_bed_undercarriage",
        "y_axis",
        "bracket",
    ]
    assert Path(results[-1]["artifacts"]["leader_step"]).read_text(
        encoding="utf-8"
    ) == (
        "bracket:y-axis|CENTER:y-axis->printer-frame|STACK_BOTTOM:y-axis|CENTER:y-axis->printer-frame->print-bed-undercarriage"
    )


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


def test_resolve_export_options_uses_lightweight_visualization_defaults():
    resolved = builder._resolve_export_options({}, "visualization", None)

    assert resolved == {
        "prod_gap": 1.0,
        "bed_width": 200.0,
        "max_build_height": None,
        "export_step": False,
        "export_obj": True,
        "export_individual_parts": False,
        "export_stl": False,
    }
