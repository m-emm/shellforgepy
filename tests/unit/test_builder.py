import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import shellforgepy.builder.builder as builder
import shellforgepy.builder.graph_model as builder_graph_model
import shellforgepy.produce.arrange_and_export as arrange_and_export_module
import shellforgepy.workflow.workflow as workflow_module
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.metrics import (
    Material,
    build_metrics_report_lines,
    record_weight_metric,
    reset_metrics,
    snapshot_metrics,
    using_metrics_snapshot,
)
from shellforgepy.produce.mesh_scene import ObjMesh
from shellforgepy.simple import create_box, get_bounding_box, translate


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_obj_mesh_builder_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = tmp_path / "mesh_project"
    src_dir = project_root / "src" / "mesh_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "mesh_generator.py",
        "\n".join(
            [
                "from pathlib import Path",
                "import numpy as np",
                "from shellforgepy.produce.mesh_scene import ObjMesh",
                "def make_terrain(*, texture_path):",
                "    return ObjMesh(",
                "        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),",
                "        faces=np.asarray([[0, 1, 2]], dtype=np.int64),",
                "        name='terrain_mesh',",
                "        color=(0.1, 0.2, 0.3),",
                "        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),",
                "        texture_path=Path(texture_path),",
                "        material_name='terrain_material',",
                "        metadata={'kind': 'unit-test-terrain'},",
                "    )",
            ]
        ),
    )

    texture_path = project_root / "terrain_texture.png"
    texture_path.write_bytes(b"not-a-real-png-but-good-enough-for-copy")

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "terrain_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Builder:",
                "  Visualization:",
                "    preview:",
                "      enabled: false",
                "    arrange:",
                "      export_obj: true",
                "      export_step: false",
                "      export_stl: false",
                "      export_individual_parts: false",
                "  Production:",
                "    process_data:",
                "      filament: petg",
                "Parameters:",
                "  texture_path:",
                "    Type: String",
                "Parts:",
                "  Terrain:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: mesh_demo_pkg.mesh_generator.make_terrain",
                "      Properties:",
                "        texture_path:",
                "          $ref: texture_path",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                f"  texture_path: {texture_path}",
                "assemblies:",
                "  - name: terrain_assembly",
                "    resource_file: terrain_assembly.yaml",
                "    parameters:",
                "      texture_path: !Ref texture_path",
            ]
        ),
    )
    return config_path, project_root / "repository", texture_path


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


def test_build_from_file_supports_imported_globals(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "import_globals_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
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
                "      Generator: import_globals_demo_pkg.widget_generator.make_widget",
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
    _write_file(
        assemblies_dir / "base_parameters.yaml",
        "\n".join(
            [
                "globals:",
                "  width: 10",
                "  height: 20",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "idex_parameters.yaml",
        "\n".join(
            [
                "globals:",
                "  $import: [base_parameters.yaml]",
                "  height: 30",
                "  label: imported-label",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                "  $import: [idex_parameters.yaml]",
                "  label: local-label",
                "assemblies:",
                "  - name: sample_assembly",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "import_globals_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "import_globals_demo_pkg.widget_generator",
        raising=False,
    )

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-30.0-local-label"
    )


def test_build_from_file_reports_invalid_imported_globals_file(tmp_path):
    project_root = tmp_path / "project"
    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(
        assemblies_dir / "invalid_globals.yaml",
        "\n".join(
            [
                "globals:",
                "  - not",
                "  - a",
                "  - mapping",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                "  $import: [invalid_globals.yaml]",
                "assemblies: []",
            ]
        ),
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "must define a globals mapping" in str(excinfo.value)
    assert "invalid_globals.yaml" in str(excinfo.value)


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
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.metric_generators", raising=False)

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


def test_build_from_file_auto_forwards_required_parameters_with_sparse_properties(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "auto_forward_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_widget(*, width, height, label, record_metrics=False):",
                "    return f'widget-{width}-{height}-{label}-{record_metrics}'",
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
                "      Generator: auto_forward_demo_pkg.widget_generator.make_widget",
                "      Properties:",
                "        record_metrics: true",
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
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "auto_forward_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "auto_forward_demo_pkg.widget_generator",
        raising=False,
    )

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-20.0-global-label-True"
    )


def test_build_from_file_auto_forwards_optional_parameters_with_sparse_properties(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "auto_forward_optional_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_widget(*, width, height=5, label='default-label'):",
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
                "      Generator: auto_forward_optional_demo_pkg.widget_generator.make_widget",
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
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "auto_forward_optional_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "auto_forward_optional_demo_pkg.widget_generator",
        raising=False,
    )

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-20.0-global-label"
    )


def test_build_from_file_ignores_declared_parameters_not_used_by_generator(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "auto_forward_unused_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_widget(*, width, label='default-label'):",
                "    return f'widget-{width}-{label}'",
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
                "  unused_height:",
                "    Type: Float",
                "  label:",
                "    Type: String",
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: auto_forward_unused_demo_pkg.widget_generator.make_widget",
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
                "  unused_height: 20",
                "  label: global-label",
                "assemblies:",
                "  - name: sample_assembly",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "auto_forward_unused_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "auto_forward_unused_demo_pkg.widget_generator",
        raising=False,
    )

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-10.0-global-label"
    )


def test_build_from_file_explicit_properties_override_auto_forwarded_parameters(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "auto_forward_override_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
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
                "      Generator: auto_forward_override_demo_pkg.widget_generator.make_widget",
                "      Properties:",
                "        width:",
                "          $ref: height",
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
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "auto_forward_override_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "auto_forward_override_demo_pkg.widget_generator",
        raising=False,
    )

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-20.0-20.0-global-label"
    )


def test_resolve_assembly_reports_config_and_resource_for_unknown_parameter_override(
    tmp_path,
):
    project_root = tmp_path / "project"
    assemblies_dir = project_root / "assembling" / "assemblies"
    resource_path = assemblies_dir / "sample_assembly.yaml"
    _write_file(
        resource_path,
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parameters:",
                "  width:",
                "    Type: Float",
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.widget_generator.make_widget",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: sample_assembly",
                "    parameters:",
                "      width: 10",
                "      unexpected_parameter: 123",
            ]
        ),
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder._resolve_assembly(
            config_path,
            {
                "name": "sample_assembly",
                "parameters": {"width": 10, "unexpected_parameter": 123},
            },
            {},
        )

    message = str(excinfo.value)
    assert "Unknown parameter override(s)" in message
    assert "sample_assembly" in message
    assert str(config_path) in message
    assert str(resource_path) in message
    assert "unexpected_parameter" in message
    assert "Parameters section" in message


def test_build_from_file_reports_missing_injected_generator_argument_with_yaml_context(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    package_name = "missing_injection_demo_pkg"
    src_dir = project_root / "src" / package_name
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_dependency():",
                "    return 'dependency-part'",
                "",
                "def make_target(*, dependency_assembly):",
                "    return dependency_assembly",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    dependency_resource = assemblies_dir / "dependency_assembly.yaml"
    target_resource = assemblies_dir / "target_assembly.yaml"
    _write_file(
        dependency_resource,
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parts:",
                "  Dependency:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.widget_generator.make_dependency",
            ]
        ),
    )
    _write_file(
        target_resource,
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parts:",
                "  Target:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.widget_generator.make_target",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: dependency_assembly",
                "    resource_file: dependency_assembly.yaml",
                "  - name: target_assembly",
                "    resource_file: target_assembly.yaml",
                "    depends_on:",
                "      - dependency_assembly",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.widget_generator", raising=False)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(
            config_path,
            assembly_names=["target_assembly"],
            force=True,
        )

    message = str(excinfo.value)
    assert (
        "required keyword argument(s) were not provided: dependency_assembly" in message
    )
    assert str(config_path) in message
    assert str(target_resource) in message
    assert "inject_parts" in message


def test_build_from_file_allows_generators_with_var_keyword_arguments(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "var_keyword_generator.py",
        "\n".join(
            [
                "def make_widget(**_kwargs):",
                "    return 'widget-from-var-keyword'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "var_keyword_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parts:",
                "  VarKeywordWidget:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.var_keyword_generator.make_widget",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: var_keyword_assembly",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.var_keyword_generator", raising=False)

    results = builder.build_from_file(config_path)

    assert (
        Path(results[0]["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "widget-from-var-keyword"
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


def test_build_from_file_hashes_declared_file_dependencies(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_widget():",
                "    return 'widget-from-file-dependencies'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    dependency_a = assemblies_dir / "data" / "alpha.json"
    dependency_z = assemblies_dir / "data" / "zeta.json"
    _write_file(dependency_a, '{"value": 1}\n')
    _write_file(dependency_z, '{"value": 2}\n')
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.widget_generator.make_widget",
                "      FileDependencies:",
                "        - path: data/zeta.json",
                "        - data/alpha.json",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(config_path, "assemblies:\n  - name: sample_assembly\n")

    exported = []

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")
        exported.append(destination)

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.widget_generator", raising=False)

    first_results = builder.build_from_file(config_path)
    first_result = first_results[0]
    first_hash = first_result["parameter_hash"]
    expected_dependencies = [
        {
            "path": str(dependency_a.resolve()),
            "sha256": builder._file_sha256(dependency_a.resolve()),
        },
        {
            "path": str(dependency_z.resolve()),
            "sha256": builder._file_sha256(dependency_z.resolve()),
        },
    ]
    assert first_result["file_dependencies"] == expected_dependencies
    assert first_result["version_inputs"]["file_dependency_0000_path"] == str(
        dependency_a.resolve()
    )
    assert first_result["version_inputs"]["file_dependency_0000_sha256"] == (
        expected_dependencies[0]["sha256"]
    )
    assert first_result["cache_hit"] is False
    assert exported

    monkeypatch.setattr(
        builder,
        "_export_part_to_step",
        lambda part, destination: pytest.fail(
            f"cache hit should not export artifacts again: {destination}"
        ),
    )
    cached_results = builder.build_from_file(config_path)
    assert cached_results[0]["cache_hit"] is True
    assert cached_results[0]["parameter_hash"] == first_hash

    _write_file(dependency_a, '{"value": 3}\n')
    rebuilt_exports = []

    def fake_rebuild_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")
        rebuilt_exports.append(destination)

    monkeypatch.setattr(builder, "_export_part_to_step", fake_rebuild_export)
    rebuilt_results = builder.build_from_file(config_path)
    rebuilt_result = rebuilt_results[0]

    assert rebuilt_result["cache_hit"] is False
    assert rebuilt_result["parameter_hash"] != first_hash
    assert rebuilt_result["file_dependencies"][0]["sha256"] == builder._file_sha256(
        dependency_a.resolve()
    )
    assert rebuilt_exports


def test_build_from_file_hashes_architecture_storey_specification(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "architecture_demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "storey_generator.py",
        "\n".join(
            [
                "from shellforgepy.construct.leader_followers_cutters_part import LeaderFollowersCuttersPart",
                "from shellforgepy.simple import create_box",
                "def make_storey(*, architecture):",
                "    spec = architecture['storey_specification']",
                "    return LeaderFollowersCuttersPart(",
                "        create_box(1, 1, 1),",
                "        additional_data={'architecture': {'storey': {'id': spec['id']}}},",
                "    )",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    semantic_file = assemblies_dir / "data" / "storey.yaml"
    _write_file(semantic_file, "schema_version: 2\n")
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parameters:",
                "  storey_id:",
                "    Type: String",
                "  semantic_file:",
                "    Type: String",
                "Parts:",
                "  Storey:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: architecture_demo_pkg.storey_generator.make_storey",
                "      Architecture:",
                "        StoreySpecification:",
                "          id: !Ref storey_id",
                "          storey_index: 3",
                "          path: !Ref semantic_file",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "assemblies.yaml",
        "\n".join(
            [
                "assemblies:",
                "  - name: sample_assembly",
                "    resource_file: sample_assembly.yaml",
                "    parameters:",
                "      storey_id: existing_house_storey_3",
                "      semantic_file: data/storey.yaml",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"

    monkeypatch.setattr(
        builder,
        "_export_part_to_step",
        lambda part, destination: _write_file(destination, str(part)),
    )
    monkeypatch.delitem(sys.modules, "architecture_demo_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "architecture_demo_pkg.storey_generator",
        raising=False,
    )

    result = builder.build_from_file(config_path)[0]
    expected_dependency = {
        "path": str(semantic_file.resolve()),
        "sha256": builder._file_sha256(semantic_file.resolve()),
    }
    assert result["file_dependencies"] == [expected_dependency]
    assert result["architecture"]["storey_specification"] == {
        "id": "existing_house_storey_3",
        "storey_index": 3,
        "path": str(semantic_file.resolve()),
        "sha256": expected_dependency["sha256"],
    }
    assert result["generator_kwargs"]["architecture"] == result["architecture"]
    assert result["additional_data"]["architecture"]["storey"]["id"] == (
        "existing_house_storey_3"
    )
    assert result["version_inputs"]["file_dependency_0000_path"] == str(
        semantic_file.resolve()
    )


def test_architecture_plan_renders_use_injected_storey_metadata(monkeypatch, tmp_path):
    import shellforgepy.architecture.plan_render as plan_render

    selected_metadata = {
        "assembly_name": "scene_assembly",
        "public_parameters": {
            "scene_colors": {"excluded_living_area": [1.0, 0.05, 0.04]}
        },
        "dependencies": [{"kwarg_name": "storey", "assembly_name": "storey_assembly"}],
    }
    storey_metadata = {
        "assembly_name": "storey_assembly",
        "additional_data": {"architecture": {"storey": {"id": "storey_a"}}},
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "architecture_plan_renders": [
                    {
                        "source": "injected",
                        "assembly": "storey",
                        "name": "storey_a_plan",
                        "formats": ["svg"],
                        "excluded_living_area_color": {
                            "$ref": "scene_colors.excluded_living_area"
                        },
                    }
                ]
            }
        }
    }
    calls = []

    def fake_render(metadata, output_path, *, excluded_living_area_color, image_width):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("svg", encoding="utf-8")
        calls.append((metadata, output_path, excluded_living_area_color, image_width))
        return {
            "format": "svg",
            "storey_id": "storey_a",
            "storey_index": 2,
            "semantic_path": "storey.yaml",
        }

    monkeypatch.setattr(plan_render, "render_storey_plan_from_metadata", fake_render)

    rendered = builder._render_architecture_plan_renders(
        metadata=selected_metadata,
        resource_data=resource_data,
        mode="visualization",
        config_data={},
        built_results_by_name={"storey_assembly": storey_metadata},
        repository_dir=tmp_path / "repo",
        run_directory=tmp_path / "run",
    )

    assert calls[0][0] == storey_metadata
    assert calls[0][2] == [1.0, 0.05, 0.04]
    assert rendered == [
        {
            "name": "storey_a_plan",
            "format": "svg",
            "source": "injected",
            "source_assembly": "storey_assembly",
            "storey_id": "storey_a",
            "storey_index": 2,
            "semantic_path": "storey.yaml",
            "path": "architecture_plan_renders/storey_a_plan.svg",
        }
    ]
    assert (tmp_path / "run" / rendered[0]["path"]).read_text(encoding="utf-8") == "svg"


def test_build_from_file_reports_missing_file_dependency(tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "widget_generator.py",
        "\n".join(
            [
                "def make_widget():",
                "    return 'widget'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_pkg.widget_generator.make_widget",
                "      FileDependencies:",
                "        - data/missing.json",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(config_path, "assemblies:\n  - name: sample_assembly\n")

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "File dependency" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


def test_build_from_file_caches_obj_mesh_artifacts(tmp_path):
    config_path, repository_path, texture_path = _write_obj_mesh_builder_project(
        tmp_path
    )

    results = builder.build_from_file(
        config_path,
        assembly_names=["terrain_assembly"],
        repository_dir=str(repository_path),
        force=True,
    )

    result = results[0]
    artifacts = result["artifacts"]
    assert artifacts["leader_step"] is None
    assert artifacts["leader_mesh"]["kind"] == "obj_mesh"
    assert Path(artifacts["leader_mesh"]["path"]).suffix == ".npz"
    assert Path(artifacts["leader_mesh"]["path"]).exists()
    assert Path(artifacts["leader_mesh"]["metadata_path"]).exists()

    entry = builder._artifact_entries_for_selector(result, "leader")[0]
    imported = builder._import_artifact_entry(entry)
    assert isinstance(imported, ObjMesh)
    assert np.allclose(
        imported.vertices,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert np.array_equal(imported.faces, np.asarray([[0, 1, 2]], dtype=np.int64))
    assert np.allclose(
        imported.uvs,
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    assert imported.texture_path == str(texture_path)
    assert imported.material_name == "terrain_material"
    assert imported.color == (0.1, 0.2, 0.3)
    assert imported.metadata == {"kind": "unit-test-terrain"}

    cached_results = builder.build_from_file(
        config_path,
        assembly_names=["terrain_assembly"],
        repository_dir=str(repository_path),
    )
    assert cached_results[0]["cache_hit"] is True
    assert (
        cached_results[0]["artifacts"]["leader_mesh"]["path"]
        == artifacts["leader_mesh"]["path"]
    )


def test_run_builder_visualizes_obj_mesh_artifact_with_texture(tmp_path):
    config_path, repository_path, _texture_path = _write_obj_mesh_builder_project(
        tmp_path
    )
    workflow_config_path = tmp_path / "workflow_config.json"
    workflow_config_path.write_text("{}", encoding="utf-8")
    runs_dir = tmp_path / "runs"

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["terrain_assembly"],
            repository_dir=str(repository_path),
            force=True,
            visualize=True,
            with_dependents=False,
            production=False,
            prototype=False,
            slice=False,
            upload=False,
            open=False,
            run_id="mesh_visual",
            runs_dir=str(runs_dir),
            master_settings_dir=None,
            orca_executable=None,
            orca_debug=None,
            printer=None,
            part_file=None,
            process_file=None,
            plate=None,
            config=str(workflow_config_path),
            verbose=False,
        )
    )

    assert result == 0
    run_dir = runs_dir / "terrain_assembly_run_mesh_visual"
    obj_path = run_dir / "terrain_assembly.obj"
    mtl_path = run_dir / "terrain_assembly.mtl"
    copied_texture_path = run_dir / "terrain_texture.png"
    assert obj_path.exists()
    assert mtl_path.exists()
    assert copied_texture_path.exists()
    assert "usemtl terrain_material" in obj_path.read_text(encoding="utf-8")
    assert "map_Kd terrain_texture.png" in mtl_path.read_text(encoding="utf-8")


def test_run_builder_rejects_obj_mesh_in_production(tmp_path):
    config_path, repository_path, _texture_path = _write_obj_mesh_builder_project(
        tmp_path
    )
    workflow_config_path = tmp_path / "workflow_config.json"
    workflow_config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="ObjMesh passthrough parts"):
        builder.run_builder(
            argparse.Namespace(
                config_file=str(config_path),
                assembly=["terrain_assembly"],
                repository_dir=str(repository_path),
                force=True,
                visualize=False,
                with_dependents=False,
                production=True,
                prototype=False,
                slice=False,
                upload=False,
                open=False,
                run_id="mesh_production",
                runs_dir=str(tmp_path / "runs"),
                master_settings_dir=None,
                orca_executable=None,
                orca_debug=None,
                printer=None,
                part_file=None,
                process_file=None,
                plate=None,
                config=str(workflow_config_path),
                verbose=False,
            )
        )


def test_build_from_file_captures_metrics_snapshot_and_report(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_metrics_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "metric_generators.py",
        "\n".join(
            [
                "from shellforgepy.metrics import record_length_metric",
                "def make_widget(*, width):",
                "    record_length_metric('linear_rail', 'MGN12', 'x_axis_rail', width)",
                "    return f'widget-{width}'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                "Parameters:",
                "  width:",
                "    Type: Float",
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_metrics_pkg.metric_generators.make_widget",
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
                "assemblies:",
                "  - name: sample_assembly",
                "    parameters:",
                "      width: 450",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "demo_metrics_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules, "demo_metrics_pkg.metric_generators", raising=False
    )

    results = builder.build_from_file(config_path)

    metrics = results[0]["metrics"]
    assert metrics["has_metrics"] is True
    assert metrics["snapshot"]["length_metrics"] == [
        {
            "category": "linear_rail",
            "stock_type": "MGN12",
            "part_name": "x_axis_rail",
            "length_mm": 450.0,
        }
    ]
    assert Path(metrics["report_path"]).read_text(encoding="utf-8") == "\n".join(
        [
            "Cut stock metrics:",
            "linear_rail MGN12:",
            "  450 mm x1",
            "    - x_axis_rail",
            "",
        ]
    )


def test_build_from_file_rebuilds_legacy_metadata_schema_for_metrics(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_metrics_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "metric_generators.py",
        "\n".join(
            [
                "from shellforgepy.metrics import record_length_metric",
                "def make_widget(*, width):",
                "    record_length_metric('linear_rail', 'MGN12', 'x_axis_rail', width)",
                "    return f'widget-{width}'",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "sample_assembly.yaml",
        "\n".join(
            [
                "Parameters:",
                "  width:",
                "    Type: Float",
                "Parts:",
                "  Sample:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                "      Generator: demo_metrics_pkg.metric_generators.make_widget",
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
                "assemblies:",
                "  - name: sample_assembly",
                "    parameters:",
                "      width: 450",
            ]
        ),
    )

    exports = []

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")
        exports.append(destination)

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "demo_metrics_pkg", raising=False)
    monkeypatch.delitem(
        sys.modules, "demo_metrics_pkg.metric_generators", raising=False
    )

    initial_results = builder.build_from_file(config_path)
    metadata_path = (
        Path(initial_results[0]["artifact_dir"])
        / f"sample_assembly__{initial_results[0]['parameter_hash']}__metadata.json"
    )
    legacy_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy_metadata["schema_version"] = 1
    legacy_metadata.pop("metrics", None)
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")

    exports.clear()
    rebuilt_results = builder.build_from_file(config_path)

    assert rebuilt_results[0]["cache_hit"] is False
    assert rebuilt_results[0]["schema_version"] == builder.BUILD_METADATA_SCHEMA_VERSION
    assert rebuilt_results[0]["metrics"]["has_metrics"] is True
    assert exports, "legacy schema metadata should force a rebuild"


def test_build_from_file_auto_injects_top_level_context_kwargs(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / "demo_pkg"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "context_generators.py",
        "\n".join(
            [
                "def make_widget(*, width, BIG_THING):",
                '    return f"widget-{width}-{BIG_THING}"',
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
                "  width_source: 10",
                "  unrelated_global: 123",
                "context:",
                "  BIG_THING: 500",
                "assemblies:",
                "  - name: context_assembly",
                "    parameters:",
                "      width: !Ref width_source",
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


def test_build_from_file_ignores_unrelated_globals_for_context_injected_generators(
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
                "def make_widget(*, width, BIG_THING):",
                '    return f"widget-{width}-{BIG_THING}"',
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
                "  width_source: 10",
                "  unrelated_global: 123",
                "context:",
                "  BIG_THING: 500",
                "assemblies:",
                "  - name: context_assembly",
                "    parameters:",
                "      width: !Ref width_source",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.delitem(sys.modules, "demo_pkg", raising=False)
    monkeypatch.delitem(sys.modules, "demo_pkg.context_generators", raising=False)

    initial_results = builder.build_from_file(config_path)
    initial_hash = initial_results[0]["parameter_hash"]

    _write_file(
        config_path,
        "\n".join(
            [
                "globals:",
                "  width_source: 10",
                "  unrelated_global: 456",
                "context:",
                "  BIG_THING: 500",
                "assemblies:",
                "  - name: context_assembly",
                "    parameters:",
                "      width: !Ref width_source",
            ]
        ),
    )

    cached_results = builder.build_from_file(config_path)

    assert cached_results[0]["cache_hit"] is True
    assert cached_results[0]["parameter_hash"] == initial_hash


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


def test_build_from_file_supports_collection_assemblies_without_generators(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "project"
    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(
        assemblies_dir / "collection_assembly.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Builder:",
                "  Collection: true",
                "Parameters:",
                "  label:",
                "    Type: String",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: collection_assembly",
                "    parameters:",
                "      label: alpha",
            ]
        ),
    )

    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)

    initial_results = builder.build_from_file(config_path)

    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: collection_assembly",
                "    parameters:",
                "      label: beta",
            ]
        ),
    )

    rebuilt_results = builder.build_from_file(config_path)

    initial_result = initial_results[0]
    rebuilt_result = rebuilt_results[0]
    assert initial_result["generator"] == builder._COLLECTION_ASSEMBLY_GENERATOR_PATH
    assert initial_result["cache_hit"] is False
    assert rebuilt_result["cache_hit"] is False
    assert initial_result["parameter_hash"] != rebuilt_result["parameter_hash"]
    assert Path(rebuilt_result["artifacts"]["leader_step"]).exists()
    assert Path(rebuilt_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")


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


def test_scene_metrics_aggregation_includes_visualization_dependencies(tmp_path):
    def make_snapshot(assembly_id, material, volume_mm3, part_id):
        reset_metrics()
        record_weight_metric(
            assembly_id,
            material,
            volume_mm3,
            part_id=part_id,
        )
        snapshot = snapshot_metrics()
        reset_metrics()
        return snapshot

    y_axis_resource = tmp_path / "y_axis.yaml"
    y_axis_resource.write_text(
        "\n".join(
            [
                "Builder:",
                "  Visualization:",
                "    parts:",
                "      - source: dependencies",
                "        assembly: print_bed_assembly",
                "        artifact: leader",
                "      - source: dependencies",
                "        assembly: print_bed_undercarriage_assembly",
                "        artifact: leader",
                "      - source: self",
                "        artifact: leader",
            ]
        ),
        encoding="utf-8",
    )
    print_bed_resource = tmp_path / "print_bed.yaml"
    print_bed_resource.write_text("Builder: {}\n", encoding="utf-8")
    undercarriage_resource = tmp_path / "undercarriage.yaml"
    undercarriage_resource.write_text("Builder: {}\n", encoding="utf-8")

    build_results = [
        {
            "assembly_name": "y_axis_assembly",
            "resource_file": str(y_axis_resource),
            "declared_dependencies": [],
            "dependencies": [],
            "metrics": {
                "snapshot": make_snapshot(
                    "y_axis_moving_mass",
                    Material.STEEL,
                    1000.0,
                    "mgn12ca_carriages",
                )
            },
        },
        {
            "assembly_name": "print_bed_assembly",
            "resource_file": str(print_bed_resource),
            "declared_dependencies": [],
            "dependencies": [],
            "metrics": {
                "snapshot": make_snapshot(
                    "y_axis_moving_mass",
                    Material.ALUMINUM,
                    1000.0,
                    "print_bed_main",
                )
            },
        },
        {
            "assembly_name": "print_bed_undercarriage_assembly",
            "resource_file": str(undercarriage_resource),
            "declared_dependencies": [],
            "dependencies": [],
            "metrics": {
                "snapshot": make_snapshot(
                    "y_axis_moving_mass",
                    Material.PETG_CF,
                    1000.0,
                    "print_bed_undercarriage_fused",
                )
            },
        },
    ]

    built_results_by_name = {
        result["assembly_name"]: dict(result) for result in build_results
    }

    metrics_assemblies = builder._scene_metrics_assembly_names(
        seed_assemblies=["y_axis_assembly"],
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        mode="visualization",
        config_data={},
    )

    assert set(metrics_assemblies) == {
        "y_axis_assembly",
        "print_bed_assembly",
        "print_bed_undercarriage_assembly",
    }

    combined_snapshot = builder._combined_metrics_snapshot_for_results(
        build_results,
        metrics_assemblies,
    )

    with using_metrics_snapshot(combined_snapshot):
        assert build_metrics_report_lines() == [
            "Weight metrics:",
            "y_axis_moving_mass: 0.011850 kg",
            "  ALUMINUM: 0.002700 kg",
            "  PETG_CF: 0.001300 kg",
            "  STEEL: 0.007850 kg",
            "  mgn12ca_carriages (STEEL): 0.007850 kg",
            "  print_bed_main (ALUMINUM): 0.002700 kg",
            "  print_bed_undercarriage_fused (PETG_CF): 0.001300 kg",
        ]


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


def test_consumed_part_metadata_marks_matching_scene_parts(tmp_path):
    scene_parts = [
        {
            "assembly_name": "consumer_assembly",
            "obj_metadata": {"builder_selector": "consumer_assembly.leader"},
        },
        {
            "assembly_name": "provider_assembly",
            "obj_metadata": {
                "builder_selector": "provider_assembly.followers.mount_plate"
            },
        },
        {
            "assembly_name": "provider_assembly",
            "obj_metadata": {"builder_selector": "provider_assembly.leader"},
        },
    ]
    built_results_by_name = {
        "consumer_assembly": {
            "assembly_name": "consumer_assembly",
            "additional_data": {
                "consumed_part_refs": ["provider_assembly.followers.mount_plate"]
            },
        },
        "provider_assembly": {
            "assembly_name": "provider_assembly",
            "additional_data": {},
        },
    }

    builder._apply_consumed_part_metadata_to_scene_parts(
        scene_parts,
        built_results_by_name,
        tmp_path,
    )

    assert scene_parts[0]["obj_metadata"].get("consumption") is None
    assert scene_parts[1]["obj_metadata"]["consumption"] == {"is_consumed": True}
    assert scene_parts[2]["obj_metadata"].get("consumption") is None


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


def test_run_builder_lists_assemblies_alphabetically_without_building(
    monkeypatch, tmp_path, capsys
):
    def fail_build_from_file(*args, **kwargs):
        raise AssertionError("list mode must not build assemblies")

    monkeypatch.setattr(builder, "build_from_file", fail_build_from_file)

    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: zed",
                "  - name: alpha",
                "  - name: middle",
            ]
        ),
        encoding="utf-8",
    )

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["does_not_matter"],
            list_assemblies=True,
        )
    )

    assert result == 0
    assert capsys.readouterr().out == "alpha\nmiddle\nzed\n"


def test_run_builder_unknown_assembly_error_includes_available_assemblies(
    monkeypatch, tmp_path
):
    def fail_build_from_file(*args, **kwargs):
        raise AssertionError("unknown assemblies must fail before building")

    monkeypatch.setattr(builder, "build_from_file", fail_build_from_file)

    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: zed",
                "  - name: alpha",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.run_builder(
            argparse.Namespace(
                config_file=str(config_path),
                assembly=["missing"],
                list_assemblies=False,
                repository_dir=None,
                force=False,
                visualize=False,
                with_dependents=False,
                production=False,
                prototype=False,
                slice=False,
                upload=False,
            )
        )

    message = str(excinfo.value)
    assert "Unknown assembly name(s): missing" in message
    assert "Available assemblies:\n  - alpha\n  - zed" in message


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


def test_resolve_preview_options_from_visualization_section():
    metadata = {
        "public_parameters": {
            "default_view": "front_angle",
        }
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "preview": {
                    "enabled": "true",
                    "views": ["top", {"$ref": "default_view"}],
                    "width": 512,
                    "height": 384,
                }
            }
        }
    }

    resolved = builder._resolve_preview_options(
        metadata,
        resource_data,
        "visualization",
    )

    assert resolved == {
        "enabled": True,
        "views": ["top", "front_angle"],
        "width": 512,
        "height": 384,
    }


def test_export_scene_for_assembly_applies_preview_overrides_to_workflow_config(
    monkeypatch, tmp_path
):
    import os

    resource_path = tmp_path / "frame.yaml"
    resource_path.write_text(
        "\n".join(
            [
                "Builder:",
                "  Visualization:",
                "    preview:",
                "      enabled: true",
                "      views:",
                "        - front_angle",
                "        - top",
                "      width: 512",
                "      height: 384",
            ]
        ),
        encoding="utf-8",
    )

    build_results = [
        {
            "assembly_name": "frame",
            "artifact_dir": str(tmp_path / "repo" / "frame"),
            "repository_dir": str(tmp_path / "repo"),
            "resource_file": str(resource_path),
            "public_parameters": {},
            "generator_kwargs": {},
            "generator_context": {},
        }
    ]

    captured = {}

    def fake_materialize_rule_parts(
        metadata,
        resource_data,
        mode,
        built_results_by_name,
        repository_dir,
        config_data,
    ):
        return [
            {
                "name": "frame_part",
                "part": "dummy-part",
                "source_path": str(tmp_path / "repo" / "frame.step"),
            }
        ]

    def fake_arrange_and_export_parts(*args, **kwargs):
        run_directory = Path(kwargs["export_directory"])
        obj_path = run_directory / "frame.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path = Path(os.environ["SHELLFORGEPY_WORKFLOW_MANIFEST"])
        manifest_path.write_text(
            json.dumps({"obj_path": str(obj_path)}),
            encoding="utf-8",
        )

    def fake_complete_workflow_run(
        args,
        *,
        config,
        run_directory,
        manifest,
        target_label,
    ):
        captured["config"] = config
        captured["run_directory"] = run_directory
        captured["manifest"] = manifest
        captured["target_label"] = target_label
        return 0

    monkeypatch.setattr(builder, "_materialize_rule_parts", fake_materialize_rule_parts)
    monkeypatch.setattr(builder, "_apply_placement_alignments", lambda *a, **k: a[0])
    monkeypatch.setattr(builder, "_scene_metrics_assembly_names", lambda **kwargs: [])
    monkeypatch.setattr(
        builder,
        "_combined_metrics_snapshot_for_results",
        lambda build_results, assembly_names: None,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )
    monkeypatch.setattr(
        workflow_module,
        "get_config_path",
        lambda override=None: tmp_path / "config.json",
    )
    monkeypatch.setattr(
        workflow_module,
        "load_config",
        lambda path: {"render": {"enabled": False, "views": ["right"]}},
    )
    monkeypatch.setattr(
        workflow_module,
        "complete_workflow_run",
        fake_complete_workflow_run,
    )

    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test",
            runs_dir=str(tmp_path / "runs"),
            production=False,
            slice=False,
            upload=False,
            visualize=True,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "frame"}]},
        build_results=build_results,
        selected_assembly="frame",
        scene_assembly_names=["frame"],
    )

    assert result == 0
    assert captured["target_label"] == "frame"
    assert captured["config"]["render"] == {
        "enabled": True,
        "views": ["front_angle", "top"],
        "width": 512,
        "height": 384,
    }


def test_export_scene_for_assembly_passes_merged_plate_process_data_to_export(
    monkeypatch, tmp_path
):
    import os

    resource_path = tmp_path / "frame.yaml"
    resource_path.write_text(
        "\n".join(
            [
                "Builder:",
                "  Production:",
                "    process_data:",
                "      filament: default_petg",
                "      process_overrides:",
                "        wall_loops: 3",
                "        brim_type: outer_and_inner",
                "    arrange:",
                "      bed_width: 220",
                "      prod_gap: 4",
                "      plates:",
                "        - name: motion",
                "          parts: [left_part]",
                "          process_data:",
                "            overrides:",
                "              process_overrides:",
                "                wall_loops: 2",
                "                enable_support: 1",
                "        - name: frame",
                "          parts: [right_part]",
            ]
        ),
        encoding="utf-8",
    )

    build_results = [
        {
            "assembly_name": "frame",
            "artifact_dir": str(tmp_path / "repo" / "frame"),
            "repository_dir": str(tmp_path / "repo"),
            "resource_file": str(resource_path),
            "public_parameters": {},
            "generator_kwargs": {},
            "generator_context": {},
        }
    ]

    captured = {}

    def fake_materialize_rule_parts(
        metadata,
        resource_data,
        mode,
        built_results_by_name,
        repository_dir,
        config_data,
    ):
        return [
            {
                "name": "left_part",
                "part": "left-dummy",
                "source_path": str(tmp_path / "repo" / "left.step"),
            },
            {
                "name": "right_part",
                "part": "right-dummy",
                "source_path": str(tmp_path / "repo" / "right.step"),
            },
        ]

    def fake_arrange_and_export_parts(*args, **kwargs):
        captured["process_data"] = kwargs["process_data"]
        captured["plate_process_data_map"] = kwargs["plate_process_data_map"]
        run_directory = Path(kwargs["export_directory"])
        obj_path = run_directory / "frame.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path = Path(os.environ["SHELLFORGEPY_WORKFLOW_MANIFEST"])
        manifest_path.write_text(
            json.dumps({"obj_path": str(obj_path)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(builder, "_materialize_rule_parts", fake_materialize_rule_parts)
    monkeypatch.setattr(builder, "_apply_placement_alignments", lambda *a, **k: a[0])
    monkeypatch.setattr(builder, "_scene_metrics_assembly_names", lambda **kwargs: [])
    monkeypatch.setattr(
        builder,
        "_combined_metrics_snapshot_for_results",
        lambda build_results, assembly_names: None,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )
    monkeypatch.setattr(
        workflow_module,
        "get_config_path",
        lambda override=None: tmp_path / "config.json",
    )
    monkeypatch.setattr(
        workflow_module,
        "load_config",
        lambda path: {"render": {"enabled": False}},
    )
    monkeypatch.setattr(
        workflow_module,
        "complete_workflow_run",
        lambda *args, **kwargs: 0,
    )

    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test",
            runs_dir=str(tmp_path / "runs"),
            production=True,
            slice=False,
            upload=False,
            visualize=False,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "frame"}]},
        build_results=build_results,
        selected_assembly="frame",
        scene_assembly_names=["frame"],
    )

    assert result == 0
    assert captured["process_data"] == {
        "filament": "default_petg",
        "process_overrides": {
            "wall_loops": "3",
            "brim_type": "outer_and_inner",
        },
    }
    assert captured["plate_process_data_map"] == {
        "motion": {
            "filament": "default_petg",
            "process_overrides": {
                "wall_loops": "2",
                "brim_type": "outer_and_inner",
                "enable_support": "1",
            },
        }
    }


def test_export_scene_for_assembly_allows_step_only_production_without_process_data(
    monkeypatch, tmp_path
):
    import os

    resource_path = tmp_path / "machined.yaml"
    resource_path.write_text(
        "\n".join(
            [
                "Builder:",
                "  Production:",
                "    parts:",
                "      - source: self",
                "        artifact: leader",
                "        name: machined_plate",
                "      - source: self",
                "        artifact: followers",
                "        names: [spacer]",
                "        name_template: '{name}'",
                "    arrange:",
                "      export_step: true",
                "      export_stl: false",
                "      export_obj: false",
                "      export_individual_parts: false",
                "      plates:",
                "        - name: machined_plate",
                "          filename: mh_plate",
                "          parts: [machined_plate]",
                "        - name: spacer",
                "          filename: mh_spacer.step",
                "          parts: [spacer]",
            ]
        ),
        encoding="utf-8",
    )
    leader_step = tmp_path / "repo" / "machined" / "leader.step"
    follower_step = tmp_path / "repo" / "machined" / "spacer.step"
    leader_step.parent.mkdir(parents=True)
    leader_step.write_text("leader", encoding="utf-8")
    follower_step.write_text("spacer", encoding="utf-8")

    build_results = [
        {
            "assembly_name": "machined",
            "artifact_dir": str(tmp_path / "repo" / "machined"),
            "repository_dir": str(tmp_path / "repo"),
            "resource_file": str(resource_path),
            "public_parameters": {},
            "generator_kwargs": {},
            "generator_context": {},
            "artifacts": {
                "leader_step": str(leader_step),
                "followers": [
                    {
                        "name": "spacer",
                        "path": str(follower_step),
                        "kind": "step",
                        "index": 0,
                    }
                ],
            },
        }
    ]

    captured = {}

    def fake_arrange_and_export_parts(*args, **kwargs):
        captured["part_names"] = [part["name"] for part in args[0]]
        captured["process_data"] = kwargs["process_data"]
        captured["plate_process_data_map"] = kwargs["plate_process_data_map"]
        captured["export_step"] = kwargs["export_step"]
        captured["export_stl"] = kwargs["export_stl"]
        captured["export_obj"] = kwargs["export_obj"]
        captured["export_individual_parts"] = kwargs["export_individual_parts"]
        captured["plates"] = kwargs["plates"]
        run_directory = Path(kwargs["export_directory"])
        step_path = run_directory / "machined_machined_plate.step"
        step_path.write_text("ISO-10303-21;\n", encoding="utf-8")
        manifest_path = Path(os.environ["SHELLFORGEPY_WORKFLOW_MANIFEST"])
        manifest_path.write_text(
            json.dumps(
                {
                    "step_files": [
                        {
                            "name": "machined_plate",
                            "path": str(step_path),
                            "parts": ["machined_plate"],
                        },
                        {
                            "name": "spacer",
                            "path": str(run_directory / "machined_spacer.step"),
                            "parts": ["spacer"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: create_box(1, 1, 1),
    )
    monkeypatch.setattr(builder, "_apply_placement_alignments", lambda *a, **k: a[0])
    monkeypatch.setattr(builder, "_scene_metrics_assembly_names", lambda **kwargs: [])
    monkeypatch.setattr(
        builder,
        "_combined_metrics_snapshot_for_results",
        lambda build_results, assembly_names: None,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )
    monkeypatch.setattr(
        workflow_module,
        "get_config_path",
        lambda override=None: tmp_path / "config.json",
    )
    monkeypatch.setattr(
        workflow_module,
        "load_config",
        lambda path: {"render": {"enabled": False}},
    )
    monkeypatch.setattr(
        workflow_module,
        "complete_workflow_run",
        lambda *args, **kwargs: 0,
    )

    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test",
            runs_dir=str(tmp_path / "runs"),
            production=True,
            slice=False,
            upload=False,
            visualize=False,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "machined"}]},
        build_results=build_results,
        selected_assembly="machined",
        scene_assembly_names=["machined"],
    )

    assert result == 0
    assert captured["part_names"] == ["machined_plate", "spacer"]
    assert captured["process_data"] is None
    assert captured["plate_process_data_map"] is None
    assert captured["export_step"] is True
    assert captured["export_stl"] is False
    assert captured["export_obj"] is False
    assert captured["export_individual_parts"] is False
    assert captured["plates"] == [
        {
            "name": "machined_plate",
            "filename": "mh_plate",
            "parts": ["machined_plate"],
        },
        {
            "name": "spacer",
            "filename": "mh_spacer.step",
            "parts": ["spacer"],
        },
    ]

    captured.clear()
    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test_visualize",
            runs_dir=str(tmp_path / "runs"),
            production=True,
            slice=False,
            upload=False,
            visualize=True,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "machined"}]},
        build_results=build_results,
        selected_assembly="machined",
        scene_assembly_names=["machined"],
    )

    assert result == 0
    assert captured["export_step"] is True
    assert captured["export_stl"] is False
    assert captured["export_obj"] is True


def test_export_scene_for_assembly_skips_runtime_placement_in_production(
    monkeypatch, tmp_path
):
    import os

    resource_path = tmp_path / "holder.yaml"
    resource_path.write_text(
        "\n".join(
            [
                "Builder:",
                "  Production:",
                "    process_data:",
                "      filament: petg",
                "    parts:",
                "      - source: self",
                "        artifact: leader",
                "        name: holder",
                "        prod_rotation_angle: 90",
                "        prod_rotation_axis: [0, 1, 0]",
            ]
        ),
        encoding="utf-8",
    )

    build_results = [
        {
            "assembly_name": "holder",
            "artifact_dir": str(tmp_path / "repo" / "holder"),
            "repository_dir": str(tmp_path / "repo"),
            "resource_file": str(resource_path),
            "public_parameters": {},
            "generator_kwargs": {},
            "generator_context": {},
            "artifacts": {},
        }
    ]
    build_pool = builder._BuildArtifactPool(["holder"])
    build_pool.insert_available("holder", build_results[0], source="built")
    build_pool.apply_runtime_transforms(
        "holder",
        [lambda part: f"{part}|placed"],
        [{"kind": "rotate", "angle": 90.0, "axis": [0.0, 1.0, 0.0]}],
        (0.0, 0.0, 0.0),
    )

    captured = {}

    def fake_materialize_rule_parts(*args, **kwargs):
        return [
            {
                "name": "holder",
                "part": "holder-local",
                "assembly_name": "holder",
                "source_path": str(tmp_path / "repo" / "holder.step"),
                "transform_history": [],
                "prod_rotation_angle": 90,
                "prod_rotation_axis": [0, 1, 0],
            }
        ]

    def fake_arrange_and_export_parts(parts, *args, **kwargs):
        captured["parts"] = [dict(part) for part in parts]
        run_directory = Path(kwargs["export_directory"])
        obj_path = run_directory / "holder.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path = Path(os.environ["SHELLFORGEPY_WORKFLOW_MANIFEST"])
        manifest_path.write_text(json.dumps({"obj_path": str(obj_path)}))

    monkeypatch.setattr(builder, "_materialize_rule_parts", fake_materialize_rule_parts)
    monkeypatch.setattr(
        builder,
        "_materialize_placed_leaders_by_assembly",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(builder, "_scene_metrics_assembly_names", lambda **kwargs: [])
    monkeypatch.setattr(
        builder,
        "_combined_metrics_snapshot_for_results",
        lambda build_results, assembly_names: None,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )
    monkeypatch.setattr(
        workflow_module,
        "get_config_path",
        lambda override=None: tmp_path / "config.json",
    )
    monkeypatch.setattr(
        workflow_module,
        "load_config",
        lambda path: {"render": {"enabled": False}},
    )
    monkeypatch.setattr(
        workflow_module,
        "complete_workflow_run",
        lambda *args, **kwargs: 0,
    )

    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test",
            runs_dir=str(tmp_path / "runs"),
            production=True,
            slice=False,
            upload=False,
            visualize=True,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "holder"}]},
        build_results=build_results,
        selected_assembly="holder",
        scene_assembly_names=["holder"],
        build_pool=build_pool,
    )

    assert result == 0
    assert captured["parts"][0]["part"] == "holder-local"
    assert captured["parts"][0]["transform_history"] == []
    assert captured["parts"][0]["prod_rotation_angle"] == 90


def test_export_scene_for_assembly_uses_selected_assembly_for_export_base_name(
    monkeypatch, tmp_path
):
    import os

    resource_path = tmp_path / "x_axis_endstop_assembly.yaml"
    resource_path.write_text(
        "Builder:\n  Visualization:\n    parts: []\n",
        encoding="utf-8",
    )

    build_results = [
        {
            "assembly_name": "x_axis_endstop_left_assembly",
            "artifact_dir": str(tmp_path / "repo" / "x_axis_endstop_left_assembly"),
            "repository_dir": str(tmp_path / "repo"),
            "resource_file": str(resource_path),
            "public_parameters": {},
            "generator_kwargs": {},
            "generator_context": {},
        }
    ]

    captured = {}

    def fake_materialize_rule_parts(
        metadata,
        resource_data,
        mode,
        built_results_by_name,
        repository_dir,
        config_data,
    ):
        return [
            {
                "name": "x_axis_endstop",
                "part": "dummy-part",
                "source_path": str(tmp_path / "repo" / "x_axis_endstop.step"),
            }
        ]

    def fake_arrange_and_export_parts(*args, **kwargs):
        captured["script_file"] = kwargs["script_file"]
        captured["export_base_name"] = kwargs["export_base_name"]
        run_directory = Path(kwargs["export_directory"])
        obj_path = run_directory / f"{kwargs['export_base_name']}.obj"
        obj_path.write_text("# obj\n", encoding="utf-8")
        manifest_path = Path(os.environ["SHELLFORGEPY_WORKFLOW_MANIFEST"])
        manifest_path.write_text(
            json.dumps({"obj_path": str(obj_path)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(builder, "_materialize_rule_parts", fake_materialize_rule_parts)
    monkeypatch.setattr(builder, "_apply_placement_alignments", lambda *a, **k: a[0])
    monkeypatch.setattr(builder, "_scene_metrics_assembly_names", lambda **kwargs: [])
    monkeypatch.setattr(
        builder,
        "_combined_metrics_snapshot_for_results",
        lambda build_results, assembly_names: None,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )
    monkeypatch.setattr(
        workflow_module,
        "get_config_path",
        lambda override=None: tmp_path / "config.json",
    )
    monkeypatch.setattr(
        workflow_module,
        "load_config",
        lambda path: {"render": {"enabled": False}},
    )
    monkeypatch.setattr(
        workflow_module,
        "complete_workflow_run",
        lambda *args, **kwargs: 0,
    )

    result = builder._export_scene_for_assembly(
        args=argparse.Namespace(
            config=None,
            run_id="test",
            runs_dir=str(tmp_path / "runs"),
            production=False,
            slice=False,
            upload=False,
            visualize=True,
            prototype=False,
            verbose=False,
            plate=None,
        ),
        config_data={"assemblies": [{"name": "x_axis_endstop_left_assembly"}]},
        build_results=build_results,
        selected_assembly="x_axis_endstop_left_assembly",
        scene_assembly_names=["x_axis_endstop_left_assembly"],
    )

    assert result == 0
    assert captured["script_file"] == str(resource_path.resolve())
    assert captured["export_base_name"] == "x_axis_endstop_left_assembly"


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


def test_run_builder_visualization_ignores_injected_scene_sources_for_dependency_expansion(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: sprite_extruder_left_assembly",
                "    depends_on: []",
                "  - name: nitehawk_holder_left_assembly",
                "    resource_file: nitehawk_holder.yaml",
                "    depends_on:",
                "      - sprite_extruder_left_assembly",
                "    inject_parts:",
                "      sprite_extruder: sprite_extruder_left_assembly",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "nitehawk_holder.yaml").write_text(
        "\n".join(
            [
                "Builder:",
                "  Visualization:",
                "    parts:",
                "      - source: injected",
                "        assembly: sprite_extruder",
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
                "assembly_name": "nitehawk_holder_left_assembly",
                "artifact_dir": str(
                    tmp_path / "repo" / "nitehawk_holder_left_assembly"
                ),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "nitehawk_holder.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "sprite_extruder_left_assembly",
                "artifact_dir": str(
                    tmp_path / "repo" / "sprite_extruder_left_assembly"
                ),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "sprite_extruder.yaml"),
                "cache_hit": False,
            },
        ]

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)
    monkeypatch.setattr(builder, "_export_scene_for_assembly", lambda **kwargs: 0)

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["nitehawk_holder_left_assembly"],
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
    assert captured["assembly_names"] == ["nitehawk_holder_left_assembly"]


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


def test_resolve_build_generations_skips_prefix_when_consumer_has_no_placement_boundary():
    generations = builder._resolve_build_generations(
        [
            {"name": "printer_frame", "depends_on": []},
            {"name": "y_axis", "depends_on": ["printer_frame"]},
            {
                "name": "y_axis_rail_carrier_brackets",
                "depends_on": ["printer_frame", "y_axis"],
            },
            {
                "name": "left_z_axis_profile",
                "depends_on": ["printer_frame"],
            },
            {
                "name": "left_z_axis",
                "depends_on": ["left_z_axis_profile"],
                "inject_parts": {
                    "z_axis_profile": "left_z_axis_profile",
                },
            },
        ],
        ["left_z_axis"],
        {
            "placement": {
                "alignments": [
                    {
                        "part": "y_axis",
                        "to": "printer_frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "y_axis_rail_carrier_brackets.non_production_parts.mount",
                        "to": "y_axis.non_production_parts.profile_left",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "left_z_axis_profile",
                        "to": "printer_frame",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["printer_frame"],
        ["left_z_axis_profile"],
        ["left_z_axis"],
    ]


def test_build_graph_model_uses_first_rigid_group_as_injected_part_boundary():
    model = builder_graph_model.build_graph_model(
        [
            {"name": "frame"},
            {"name": "profile"},
            {
                "name": "mount",
                "inject_parts": {
                    "profile": "profile",
                },
            },
        ],
        {
            "placement": {
                "alignments": [
                    {
                        "part": "profile",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                    {
                        "to": "profile",
                        "rigid_group": ["mount"],
                    },
                ]
            }
        },
    )

    assert model.first_moving_alignment_index == {"profile": 0}
    assert model.first_involved_alignment_index["mount"] == 1
    assert model.placement_build_dependencies["mount"] == ["frame", "profile"]


def test_resolve_build_generations_uses_sequential_prefix_for_injected_rigid_group_parts():
    generations = builder._resolve_build_generations(
        [
            {"name": "frame", "depends_on": []},
            {"name": "x_axis", "depends_on": []},
            {"name": "extruder", "depends_on": []},
            {
                "name": "bracket",
                "depends_on": [],
                "inject_parts": {"extruder": "extruder"},
            },
        ],
        ["bracket"],
        {
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "bracket",
                        "to": "extruder",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["extruder", "frame", "x_axis"],
        ["bracket"],
    ]


def test_resolve_build_generations_rigid_group_prefix_includes_all_group_members():
    generations = builder._resolve_build_generations(
        [
            {"name": "frame", "depends_on": []},
            {"name": "x_axis", "depends_on": []},
            {"name": "extruder", "depends_on": []},
            {"name": "tool_head", "depends_on": []},
            {
                "name": "bracket",
                "depends_on": [],
                "inject_parts": {"extruder": "extruder"},
            },
        ],
        ["bracket"],
        {
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder", "tool_head"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "bracket",
                        "to": "extruder",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["extruder", "frame", "tool_head", "x_axis"],
        ["bracket"],
    ]


def test_resolve_build_generations_two_member_rigid_group_uses_sequential_prefix():
    generations = builder._resolve_build_generations(
        [
            {"name": "frame", "depends_on": []},
            {"name": "x_axis", "depends_on": []},
            {"name": "rail", "depends_on": []},
            {
                "name": "bracket",
                "depends_on": [],
                "inject_parts": {"x_axis": "x_axis"},
            },
        ],
        ["bracket"],
        {
            "placement": {
                "alignments": [
                    {
                        "to": "rail",
                        "rigid_group": ["x_axis"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "bracket",
                        "to": "x_axis",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert [[entry["name"] for entry in generation] for generation in generations] == [
        ["frame", "rail", "x_axis"],
        ["bracket"],
    ]


def test_resolve_build_generation_names_reports_declared_dependency_cycles():
    model = builder_graph_model.build_graph_model(
        [
            {"name": "alpha", "depends_on": ["beta"]},
            {"name": "beta", "depends_on": ["alpha"]},
        ]
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder_graph_model.resolve_build_generation_names(model)

    message = str(excinfo.value)
    assert "Cyclic dependency detected in assembly graph:" in message
    assert (
        "cycle 1: alpha -[declared_dependency]-> beta " "-[declared_dependency]-> alpha"
    ) in message


def test_resolve_build_generation_names_reports_placement_build_cycles():
    model = builder_graph_model.build_graph_model(
        [
            {
                "name": "bracket",
                "depends_on": [],
                "inject_parts": {"carrier": "carrier"},
            },
            {"name": "carrier", "depends_on": []},
            {"name": "frame", "depends_on": ["bracket"]},
        ],
        {
            "placement": {
                "alignments": [
                    {
                        "part": "carrier",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "bracket",
                        "to": "carrier",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder_graph_model.resolve_build_generation_names(model)

    message = str(excinfo.value)
    assert "Cyclic dependency detected in assembly graph:" in message
    assert (
        "cycle 1: bracket -[declared_dependency]-> frame "
        "-[placement_build]-> bracket"
    ) in message


def test_run_builder_visualization_rigid_group_does_not_widen_build_subset(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "assemblies.yaml"
    config_path.write_text(
        "\n".join(
            [
                "assemblies:",
                "  - name: extruder",
                "    depends_on: []",
                "  - name: x_axis",
                "    depends_on: []",
                "  - name: frame",
                "    depends_on: []",
                "placement:",
                "  alignments:",
                "    - rigid_group:",
                "        - extruder",
                "      to: x_axis",
                "    - part: x_axis",
                "      to: frame",
                "      alignment: CENTER",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "extruder.yaml").write_text("Builder: {}\n", encoding="utf-8")

    captured = {}

    def fake_build_from_file(
        config_file, *, assembly_names=None, repository_dir=None, force=False
    ):
        captured["assembly_names"] = assembly_names
        return [
            {
                "assembly_name": "extruder",
                "artifact_dir": str(tmp_path / "repo" / "extruder"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "extruder.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "x_axis",
                "artifact_dir": str(tmp_path / "repo" / "x_axis"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "x_axis.yaml"),
                "cache_hit": False,
            },
            {
                "assembly_name": "frame",
                "artifact_dir": str(tmp_path / "repo" / "frame"),
                "repository_dir": str(tmp_path / "repo"),
                "resource_file": str(tmp_path / "frame.yaml"),
                "cache_hit": False,
            },
        ]

    monkeypatch.setattr(builder, "build_from_file", fake_build_from_file)
    monkeypatch.setattr(builder, "_export_scene_for_assembly", lambda **kwargs: 0)

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["extruder"],
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
    assert captured["assembly_names"] == ["extruder", "x_axis"]


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
                            "process_overrides": {
                                "brim_type": "no_brim",
                                "support_object_first_layer_gap": 0.8,
                            },
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
            "support_object_first_layer_gap": "0.8",
        },
    }


def test_resolve_process_data_applies_prototype_overrides(monkeypatch):
    class ProcessModule:
        PROCESS_DATA = {
            "filament": "TPU",
            "process_overrides": {
                "wall_loops": "3",
                "sparse_infill_density": "75%",
            },
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
            "public_parameters": {},
            "generator_kwargs": {},
        },
        {
            "Builder": {
                "Production": {
                    "process_data": {
                        "source": "demo.process.PROCESS_DATA",
                    },
                    "prototype": {
                        "process_data": {
                            "overrides": {
                                "process_overrides": {
                                    "wall_loops": 2,
                                    "sparse_infill_density": "40%",
                                }
                            }
                        }
                    },
                }
            }
        },
        include_prototype_overrides=True,
    )

    assert resolved == {
        "filament": "TPU",
        "process_overrides": {
            "wall_loops": "2",
            "sparse_infill_density": "40%",
        },
    }


def test_resolve_process_data_allows_prototype_preset_without_production_process_data(
    monkeypatch,
):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    resolved = builder._resolve_process_data(
        {
            "public_parameters": {},
            "generator_kwargs": {},
        },
        {
            "Builder": {
                "Production": {
                    "prototype": {
                        "process_data_preset": "petgcf_max_strength",
                    },
                }
            }
        },
        include_prototype_overrides=True,
        config_data={
            "process_data_generators": {
                "demo_parametric": {
                    "function": "demo.parametric.generate_settings",
                }
            },
            "process_data_presets": {
                "petgcf_max_strength": {
                    "generator": "demo_parametric",
                    "arguments": {
                        "label": "PETGCF",
                        "wall_loops": 4,
                    },
                }
            },
        },
    )

    assert resolved == {
        "filament": "PETGCF",
        "process_overrides": {
            "wall_loops": "4",
        },
    }


def test_resolve_process_data_applies_config_named_base_and_entry_overrides():
    resolved = builder._resolve_process_data(
        {
            "public_parameters": {"bed_temp": 55},
            "generator_kwargs": {"assembly_name": "feet"},
        },
        {
            "Builder": {
                "Production": {
                    "parts": [
                        {
                            "source": "self",
                            "artifact": "leader",
                            "name": "feet",
                        }
                    ]
                }
            }
        },
        config_data={
            "process_data_definitions": {
                "tpu_base": {
                    "filament": "TPU",
                    "process_overrides": {
                        "speed": "100",
                        "support_top_z_distance": "0.4",
                    },
                }
            },
            "assemblies": [
                {
                    "name": "feet",
                    "process_data": {
                        "base": "tpu_base",
                        "overrides": {
                            "bed_temperature": {"$ref": "bed_temp"},
                            "process_overrides": {
                                "brim_type": "no_brim",
                                "support_top_z_distance": 0.25,
                            },
                        },
                    },
                }
            ],
        },
        selected_assembly_name="feet",
    )

    assert resolved == {
        "filament": "TPU",
        "bed_temperature": 55,
        "process_overrides": {
            "speed": "100",
            "brim_type": "no_brim",
            "support_top_z_distance": "0.25",
        },
    }


def test_resolve_process_data_supports_named_generator_presets(monkeypatch):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, printer_id, material, nozzle_diameter_mm, strength):
            return {
                "filament": f"{material}@{printer_id}",
                "process_overrides": {
                    "nozzle_diameter": nozzle_diameter_mm,
                    "wall_loops": 2 if strength < 0.6 else 3,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    resolved = builder._resolve_process_data(
        {
            "public_parameters": {},
            "generator_kwargs": {},
        },
        {
            "Builder": {
                "Production": {
                    "process_data": {
                        "preset": "petgcf_medium",
                        "overrides": {
                            "process_overrides": {
                                "brim_type": "no_brim",
                            }
                        },
                    }
                }
            }
        },
        config_data={
            "process_data_generators": {
                "demo_parametric": {
                    "function": "demo.parametric.generate_settings",
                    "arguments": {
                        "printer_id": "demo_printer",
                    },
                }
            },
            "process_data_presets": {
                "petgcf_medium": {
                    "generator": "demo_parametric",
                    "arguments": {
                        "material": "PETGCF",
                        "nozzle_diameter_mm": 0.6,
                        "strength": 0.5,
                    },
                }
            },
        },
    )

    assert resolved == {
        "filament": "PETGCF@demo_printer",
        "process_overrides": {
            "nozzle_diameter": "0.6",
            "wall_loops": "2",
            "brim_type": "no_brim",
        },
    }


def test_resolve_process_data_supports_production_process_data_preset(monkeypatch):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    resolved = builder._resolve_process_data(
        {
            "public_parameters": {},
            "generator_kwargs": {},
        },
        {
            "Builder": {
                "Production": {
                    "process_data_preset": "petgcf_medium",
                }
            }
        },
        config_data={
            "process_data_generators": {
                "demo_parametric": {
                    "function": "demo.parametric.generate_settings",
                }
            },
            "process_data_presets": {
                "petgcf_medium": {
                    "generator": "demo_parametric",
                    "arguments": {
                        "label": "medium",
                        "wall_loops": 2,
                    },
                }
            },
        },
    )

    assert resolved == {
        "filament": "medium",
        "process_overrides": {
            "wall_loops": "2",
        },
    }


def test_resolve_process_data_supports_preset_shorthand_with_overrides(monkeypatch):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    resolved = builder._resolve_process_data(
        {
            "public_parameters": {},
            "generator_kwargs": {},
        },
        {
            "Builder": {
                "Production": {
                    "process_data_preset": "petgcf_medium",
                    "process_data": {
                        "overrides": {
                            "process_overrides": {
                                "enable_support": 1,
                                "support_threshold_angle": 30,
                            }
                        }
                    },
                }
            }
        },
        config_data={
            "process_data_generators": {
                "demo_parametric": {
                    "function": "demo.parametric.generate_settings",
                }
            },
            "process_data_presets": {
                "petgcf_medium": {
                    "generator": "demo_parametric",
                    "arguments": {
                        "label": "medium",
                        "wall_loops": 2,
                    },
                }
            },
        },
    )

    assert resolved == {
        "filament": "medium",
        "process_overrides": {
            "wall_loops": "2",
            "enable_support": "1",
            "support_threshold_angle": "30",
        },
    }


def test_resolve_plate_process_data_map_supports_plate_presets(monkeypatch):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    metadata = {
        "public_parameters": {},
        "generator_kwargs": {},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "process_data_preset": "default_petgcf",
                "arrange": {
                    "plates": [
                        {
                            "name": "left",
                            "parts": ["left_part"],
                            "process_data": {
                                "preset": "petgcf_medium",
                            },
                        },
                        {
                            "name": "right",
                            "parts": ["right_part"],
                        },
                    ]
                },
            }
        }
    }
    config_data = {
        "process_data_generators": {
            "demo_parametric": {
                "function": "demo.parametric.generate_settings",
            }
        },
        "process_data_presets": {
            "default_petgcf": {
                "generator": "demo_parametric",
                "arguments": {
                    "label": "default",
                    "wall_loops": 3,
                },
            },
            "petgcf_medium": {
                "generator": "demo_parametric",
                "arguments": {
                    "label": "medium",
                    "wall_loops": 2,
                },
            },
        },
    }

    default_process_data = builder._resolve_process_data(
        metadata,
        resource_data,
        config_data=config_data,
    )
    resolved = builder._resolve_plate_process_data_map(
        metadata,
        resource_data,
        default_process_data=default_process_data,
        config_data=config_data,
    )

    assert resolved == {
        "left": {
            "filament": "medium",
            "process_overrides": {
                "wall_loops": "2",
            },
        }
    }


def test_resolve_plate_process_data_map_supports_plate_preset_shorthand_with_overrides(
    monkeypatch,
):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    metadata = {
        "public_parameters": {},
        "generator_kwargs": {},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "arrange": {
                    "plates": [
                        {
                            "name": "left",
                            "parts": ["left_part"],
                            "process_data_preset": "petgcf_medium",
                            "process_data": {
                                "overrides": {
                                    "process_overrides": {
                                        "enable_support": 1,
                                        "support_threshold_angle": 30,
                                    }
                                }
                            },
                        }
                    ]
                },
            }
        }
    }
    config_data = {
        "process_data_generators": {
            "demo_parametric": {
                "function": "demo.parametric.generate_settings",
            }
        },
        "process_data_presets": {
            "petgcf_medium": {
                "generator": "demo_parametric",
                "arguments": {
                    "label": "medium",
                    "wall_loops": 2,
                },
            },
        },
    }

    resolved = builder._resolve_plate_process_data_map(
        metadata,
        resource_data,
        default_process_data=None,
        config_data=config_data,
    )

    assert resolved == {
        "left": {
            "filament": "medium",
            "process_overrides": {
                "wall_loops": "2",
                "enable_support": "1",
                "support_threshold_angle": "30",
            },
        }
    }


def test_resolve_plate_process_data_map_requires_process_data_without_global_default(
    monkeypatch,
):
    class ProcessModule:
        @staticmethod
        def generate_settings(*, label, wall_loops):
            return {
                "filament": label,
                "process_overrides": {
                    "wall_loops": wall_loops,
                },
            }

    real_import_module = importlib.import_module
    monkeypatch.setattr(
        builder.importlib,
        "import_module",
        lambda module_name: (
            ProcessModule
            if module_name == "demo.parametric"
            else real_import_module(module_name)
        ),
    )

    metadata = {
        "public_parameters": {},
        "generator_kwargs": {},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "arrange": {
                    "plates": [
                        {
                            "name": "left",
                            "parts": ["left_part"],
                            "process_data_preset": "petgcf_medium",
                        },
                        {
                            "name": "right",
                            "parts": ["right_part"],
                        },
                    ]
                },
            }
        }
    }
    config_data = {
        "process_data_generators": {
            "demo_parametric": {
                "function": "demo.parametric.generate_settings",
            }
        },
        "process_data_presets": {
            "petgcf_medium": {
                "generator": "demo_parametric",
                "arguments": {
                    "label": "medium",
                    "wall_loops": 2,
                },
            },
        },
    }

    with pytest.raises(
        builder.BuilderError,
        match="Missing process data for plates: right",
    ):
        builder._resolve_plate_process_data_map(
            metadata,
            resource_data,
            default_process_data=None,
            config_data=config_data,
        )


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


def test_materialize_rule_parts_resolves_injected_source_by_kwarg_name(
    monkeypatch, tmp_path
):
    dependency_step = tmp_path / "sprite_extruder_left__leader.step"
    dependency_step.write_text("sprite-extruder-left", encoding="utf-8")

    metadata = {
        "assembly_name": "nitehawk_holder_left_assembly",
        "resource_file": str(tmp_path / "nitehawk_holder.yaml"),
        "public_parameters": {},
        "generator_kwargs": {},
        "dependencies": [
            {
                "kwarg_name": "sprite_extruder",
                "assembly_name": "sprite_extruder_left_assembly",
                "artifact": "assembly",
            }
        ],
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "injected",
                        "assembly": "sprite_extruder",
                        "artifact": "leader",
                        "name": "sprite_extruder",
                    }
                ]
            }
        }
    }
    built_results_by_name = {
        "sprite_extruder_left_assembly": {
            "assembly_name": "sprite_extruder_left_assembly",
            "artifacts": {"leader_step": str(dependency_step)},
        }
    }

    monkeypatch.setattr(builder, "_import_dependency_part", lambda path: f"part:{path}")

    parts = builder._materialize_rule_parts(
        metadata,
        resource_data,
        "visualization",
        built_results_by_name,
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["name"] == "sprite_extruder"
    assert parts[0]["part"] == f"part:{dependency_step}"


def test_artifact_entries_for_selector_all_expands_visualization_artifacts(tmp_path):
    leader_step = tmp_path / "leader.step"
    follower_step = tmp_path / "follower.step"
    non_production_step = tmp_path / "non_production.step"
    leader_step.write_text("leader", encoding="utf-8")
    follower_step.write_text("follower", encoding="utf-8")
    non_production_step.write_text("non-production", encoding="utf-8")

    metadata = {
        "assembly_name": "sprite_extruder_left_assembly",
        "artifacts": {
            "leader_step": str(leader_step),
            "followers": [
                {
                    "path": str(follower_step),
                    "name": "fan_shroud",
                    "index": 0,
                }
            ],
            "non_production_parts": [
                {
                    "path": str(non_production_step),
                    "name": "hotend",
                    "index": 0,
                }
            ],
        },
    }

    entries = builder._artifact_entries_for_selector(metadata, "all")

    assert [entry["artifact"] for entry in entries] == [
        "leader",
        "followers",
        "non_production_parts",
    ]
    assert [entry["name"] for entry in entries] == [None, "fan_shroud", "hotend"]


def test_materialize_rule_parts_resolves_injected_source_all_artifacts(
    monkeypatch, tmp_path
):
    leader_step = tmp_path / "leader.step"
    follower_step = tmp_path / "follower.step"
    non_production_step = tmp_path / "non_production.step"
    leader_step.write_text("leader", encoding="utf-8")
    follower_step.write_text("follower", encoding="utf-8")
    non_production_step.write_text("non-production", encoding="utf-8")

    metadata = {
        "assembly_name": "nitehawk_holder_left_assembly",
        "resource_file": str(tmp_path / "nitehawk_holder.yaml"),
        "public_parameters": {},
        "generator_kwargs": {},
        "dependencies": [
            {
                "kwarg_name": "sprite_extruder",
                "assembly_name": "sprite_extruder_left_assembly",
                "artifact": "assembly",
            }
        ],
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "injected",
                        "assembly": "sprite_extruder",
                        "artifact": "all",
                    }
                ]
            }
        }
    }
    built_results_by_name = {
        "sprite_extruder_left_assembly": {
            "assembly_name": "sprite_extruder_left_assembly",
            "artifacts": {
                "leader_step": str(leader_step),
                "followers": [
                    {
                        "path": str(follower_step),
                        "name": "fan_shroud",
                        "index": 0,
                    }
                ],
                "non_production_parts": [
                    {
                        "path": str(non_production_step),
                        "name": "hotend",
                        "index": 0,
                    }
                ],
            },
        }
    }

    monkeypatch.setattr(builder, "_import_dependency_part", lambda path: f"part:{path}")

    parts = builder._materialize_rule_parts(
        metadata,
        resource_data,
        "visualization",
        built_results_by_name,
        tmp_path,
        None,
    )

    assert len(parts) == 3
    assert [part["artifact"] for part in parts] == [
        "leader",
        "followers",
        "non_production_parts",
    ]
    assert [part["name"] for part in parts] == [
        "sprite_extruder_left_assembly",
        "fan_shroud",
        "hotend",
    ]


def test_materialize_rule_parts_deduplicates_injected_dependency_aliases(
    monkeypatch, tmp_path
):
    dependency_step = tmp_path / "sprite_extruder_left__leader.step"
    dependency_step.write_text("sprite-extruder-left", encoding="utf-8")

    metadata = {
        "assembly_name": "nitehawk_holder_left_assembly",
        "resource_file": str(tmp_path / "nitehawk_holder.yaml"),
        "public_parameters": {},
        "generator_kwargs": {},
        "dependencies": [
            {
                "kwarg_name": "sprite_extruder",
                "assembly_name": "sprite_extruder_left_assembly",
                "artifact": "assembly",
            }
        ],
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "injected",
                        "artifact": "leader",
                    }
                ]
            }
        }
    }
    built_results_by_name = {
        "sprite_extruder_left_assembly": {
            "assembly_name": "sprite_extruder_left_assembly",
            "artifacts": {"leader_step": str(dependency_step)},
        }
    }

    monkeypatch.setattr(builder, "_import_dependency_part", lambda path: f"part:{path}")

    parts = builder._materialize_rule_parts(
        metadata,
        resource_data,
        "visualization",
        built_results_by_name,
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["obj_metadata"]["assembly_name"] == "sprite_extruder_left_assembly"


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


def test_normalize_generated_part_includes_direction_vectors():
    generated = LeaderFollowersCuttersPart(
        create_box(1, 1, 1),
        direction_vectors=[(0, -2, 0)],
        direction_vector_names=["drive_out"],
    )

    normalized = builder._normalize_generated_part(generated)

    assert normalized["direction_vectors"] == [
        {"index": 0, "name": "drive_out", "vector": [0.0, -2.0, 0.0]}
    ]


def test_export_artifacts_includes_direction_vectors(monkeypatch, tmp_path):
    generated = LeaderFollowersCuttersPart(
        create_box(1, 1, 1),
        direction_vectors=[(0, -2, 0)],
        direction_vector_names=["drive_out"],
    )

    monkeypatch.setattr(
        builder,
        "_export_artifact_part",
        lambda part, destination: {"kind": "step", "path": str(destination)},
    )

    artifacts = builder._export_artifacts("fan_garage", "abc123", generated, tmp_path)

    assert artifacts["direction_vectors"] == [
        {"index": 0, "name": "drive_out", "vector": [0.0, -2.0, 0.0]}
    ]


def test_materialize_rule_parts_resolves_direction_vector_animation(
    monkeypatch, tmp_path
):
    car_step = tmp_path / "car.step"
    car_step.write_text("car", encoding="utf-8")

    metadata = {
        "assembly_name": "fan_garage",
        "public_parameters": {},
        "generator_kwargs": {},
        "generator_context": {},
        "artifacts": {
            "followers": [
                {
                    "index": 0,
                    "name": "bay_1_car",
                    "path": str(car_step),
                    "kind": "step",
                }
            ],
            "direction_vectors": [
                {"index": 0, "name": "bay_1_drive_out", "vector": [0, -2, 0]}
            ],
        },
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "self",
                        "artifact": "followers",
                        "names": ["bay_1_car"],
                        "animation": {
                            "fan_bay_1_drive_out": {
                                "direction_vector": "bay_1_drive_out",
                                "distance": 16.0,
                            }
                        },
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
        {"fan_garage": metadata},
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["animation"] == {"fan_bay_1_drive_out": [0.0, -16.0, 0.0]}
    assert parts[0]["_direction_vector_animation_keys"] == {"fan_bay_1_drive_out"}


def test_apply_transform_to_scene_parts_rotates_only_direction_vector_animations():
    scene_parts = [
        {
            "assembly_name": "fan_garage",
            "part": "part",
            "animation": {
                "drive": [10.0, 0.0, 0.0],
                "literal": [0.0, 5.0, 0.0],
            },
            "_direction_vector_animation_keys": {"drive"},
        }
    ]

    builder._apply_transform_to_scene_parts(
        scene_parts,
        "fan_garage",
        lambda part: part,
        transform_record={"kind": "rotate", "angle": 90.0, "axis": [0, 0, 1]},
    )

    assert scene_parts[0]["animation"]["drive"] == pytest.approx(
        [0.0, 10.0, 0.0], abs=1e-6
    )
    assert scene_parts[0]["animation"]["literal"] == [0.0, 5.0, 0.0]

    builder._apply_transform_to_scene_parts(
        scene_parts,
        "fan_garage",
        lambda part: part,
        transform_record={"kind": "translate", "vector": [1.0, 2.0, 3.0]},
    )

    assert scene_parts[0]["animation"]["drive"] == pytest.approx(
        [0.0, 10.0, 0.0], abs=1e-6
    )
    assert scene_parts[0]["animation"]["literal"] == [0.0, 5.0, 0.0]


def test_materialize_rule_parts_resolves_color_from_metadata_context(
    monkeypatch, tmp_path
):
    leader_step = tmp_path / "house.step"
    leader_step.write_text("house", encoding="utf-8")

    metadata = {
        "assembly_name": "rw8_local_scene",
        "public_parameters": {
            "scene_colors": {
                "existing_house": [0.74, 0.88, 1.0],
            }
        },
        "generator_kwargs": {},
        "artifacts": {"leader_step": str(leader_step)},
    }
    resource_data = {
        "Builder": {
            "Visualization": {
                "parts": [
                    {
                        "source": "self",
                        "artifact": "leader",
                        "name": "existing_house",
                        "color": {"$ref": "scene_colors.existing_house"},
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
        {"rw8_local_scene": metadata},
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["color"] == [0.74, 0.88, 1.0]


def test_materialize_rule_parts_resolves_rotation_animation_from_metadata_context(
    monkeypatch, tmp_path
):
    leader_step = tmp_path / "lid.step"
    leader_step.write_text("lid", encoding="utf-8")

    metadata = {
        "assembly_name": "creality_psu",
        "public_parameters": {"lid_open_angle": 90},
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
                        "name": "lid",
                        "animation": {
                            "lid_open": {
                                "type": "rotation",
                                "axis": [1, 0, 0],
                                "angle_degrees": {"$ref": "lid_open_angle"},
                                "center": ["BACK", "BOTTOM"],
                            }
                        },
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
        {"creality_psu": metadata},
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["animation"] == {
        "lid_open": {
            "type": "rotation",
            "axis": [1, 0, 0],
            "angle_degrees": 90,
            "center": ["BACK", "BOTTOM"],
        }
    }


def test_materialize_rule_parts_attaches_obj_metadata_for_assembly_grouping(
    monkeypatch, tmp_path
):
    leader_step = tmp_path / "bed.step"
    leader_step.write_text("bed", encoding="utf-8")

    metadata = {
        "assembly_name": "print_bed_assembly",
        "public_parameters": {},
        "generator_kwargs": {},
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
        {"print_bed_assembly": metadata},
        tmp_path,
        None,
    )

    assert len(parts) == 1
    assert parts[0]["obj_metadata"] == {
        "assembly_name": "print_bed_assembly",
        "assembly_label": "print_bed",
        "builder_selector": "print_bed_assembly.leader",
        "hierarchy": ["print_bed_assembly"],
        "hierarchy_labels": ["print_bed"],
    }


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

    placement_result = builder._apply_placement_alignments(
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
    placed_parts = placement_result.scene_parts

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
        "Placement replay execute: source_step=0; y_axis.non_production_parts.anchor aligned to print_bed.non_production_parts.buffer_1 via CENTER; moving_anchor_center=(1.0,2.0,3.0); target_anchor_center=(10.0,20.0,30.0); moving_part_position=(4.0,5.0,6.0); target_part_position=(40.0,50.0,60.0); shift=(9.0,18.0,0.0)",
        "Placement replay execute: source_step=1; y_axis aligned to print_bed via STACK_TOP; moving_anchor_center=(13.0,23.0,6.0); target_anchor_center=(40.0,50.0,60.0); moving_part_position=(13.0,23.0,6.0); target_part_position=(40.0,50.0,60.0); shift=(0.0,0.0,66.0)",
    ]


def test_apply_placement_alignments_rigid_group_moves_group_when_target_moves_later(
    monkeypatch, tmp_path
):
    extruder_leader = tmp_path / "extruder_leader.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    x_axis_leader = tmp_path / "x_axis_leader.step"
    x_axis_leader.write_text("x-axis-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame_leader.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "x_axis", "part": "scene-x-axis", "transform_history": []},
        {"assembly_name": "frame", "part": "scene-frame", "transform_history": []},
    ]
    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "x_axis": {
            "assembly_name": "x_axis",
            "artifacts": {"leader_step": str(x_axis_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "extruder-leader": (1.0, 2.0, 3.0),
        "x-axis-leader": (10.0, 20.0, 30.0),
        "frame-leader": (40.0, 50.0, 60.0),
        "extruder-leader|CENTER:extruder-leader->x-axis-leader:None:0": (
            10.0,
            20.0,
            30.0,
        ),
        "x-axis-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            10.0,
            20.0,
            70.0,
        ),
        "extruder-leader|CENTER:extruder-leader->x-axis-leader:None:0|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            10.0,
            20.0,
            70.0,
        ),
    }

    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        label = f"{alignment.name}:{moving_anchor}->{target_anchor}:{axes}:{stack_gap}"
        return lambda part: f"{part}|{label}"

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "extruder"},
                {"name": "x_axis"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "part": "extruder",
                        "to": "x_axis",
                        "alignment": "CENTER",
                    },
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "STACK_TOP",
                        "stack_gap": 10,
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == (
        "scene-extruder"
        "|CENTER:extruder-leader->x-axis-leader:None:0"
        "|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[1]["part"] == (
        "scene-x-axis|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[2]["part"] == "scene-frame"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [9.0, 18.0, 27.0], "placement_step": 0},
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 2},
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 2}
    ]
    assert placed_parts[2]["transform_history"] == []


def test_build_graph_model_rejects_rigid_attach_syntax():
    with pytest.raises(builder.BuilderError, match="rigid_attach.*rigid_group"):
        builder_graph_model.build_graph_model(
            [{"name": "extruder"}, {"name": "x_axis"}],
            {
                "placement": {
                    "alignments": [
                        {
                            "part": "extruder",
                            "to": "x_axis",
                            "rigid_attach": True,
                        }
                    ]
                }
            },
        )


def test_build_graph_model_supports_rigid_group():
    model = builder_graph_model.build_graph_model(
        [
            {"name": "extruder"},
            {"name": "x_axis"},
            {"name": "tool_head"},
            {"name": "frame"},
        ],
        {
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder", "tool_head"],
                    },
                    {
                        "part": "extruder",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                ]
            }
        },
    )

    assert len(model.placement_steps) == 2
    assert model.placement_steps[0].is_rigid_group is True
    assert model.placement_steps[0].target_assembly_name == "x_axis"
    assert model.placement_steps[0].rigid_group_assembly_names == (
        "extruder",
        "tool_head",
    )
    assert model.first_moving_alignment_index == {"extruder": 1}
    assert model.first_involved_alignment_index == {
        "extruder": 0,
        "x_axis": 0,
        "tool_head": 0,
        "frame": 1,
    }
    assert model.placement_steps[0].affected_assembly_names == ("extruder", "tool_head")
    assert set(model.placement_execution_dag.predecessors(1)) == {0}


def test_build_graph_model_reports_malformed_placement_step_context():
    with pytest.raises(builder.BuilderError) as exc_info:
        builder_graph_model.build_graph_model(
            [{"name": "extruder"}, {"name": "x_axis"}],
            {
                "placement": {
                    "alignments": [
                        {
                            "to": "x_axis",
                            "rigd_group": ["extruder"],
                        }
                    ]
                }
            },
        )

    message = str(exc_info.value)
    assert "placement.alignments[0]" in message
    assert "requires 'part' or 'rigid_group'" in message
    assert "'rigd_group'" in message
    assert "did you mean 'rigid_group' instead of 'rigd_group'?" in message


def test_build_graph_model_reports_rigid_motion_conflict_step_context():
    with pytest.raises(builder.BuilderError) as exc_info:
        builder_graph_model.build_graph_model(
            [{"name": "extruder"}, {"name": "board"}],
            {
                "placement": {
                    "alignments": [
                        {
                            "to": "extruder",
                            "rigid_group": ["board"],
                        },
                        {
                            "to": "extruder",
                            "rigid_group": ["board"],
                        },
                    ]
                }
            },
        )

    message = str(exc_info.value)
    assert "placement.alignments[1]" in message
    assert "rigid_group ['board'] to extruder" in message
    assert "keys present: 'rigid_group', 'to'" in message
    assert "placement spec: {'to': 'extruder', 'rigid_group': ['board']}" in message
    assert "'board' is already rigidly connected with ['board', 'extruder']" in message


def test_build_graph_model_requires_to_on_rigid_group():
    with pytest.raises(builder.BuilderError, match="rigid_group"):
        builder_graph_model.build_graph_model(
            [{"name": "extruder"}, {"name": "x_axis"}],
            {
                "placement": {
                    "alignments": [
                        {
                            "rigid_group": ["extruder", "x_axis"],
                        }
                    ]
                }
            },
        )


def test_build_graph_model_rejects_part_key_on_rigid_group():
    with pytest.raises(builder.BuilderError, match="rigid_group"):
        builder_graph_model.build_graph_model(
            [{"name": "extruder"}, {"name": "x_axis"}, {"name": "frame"}],
            {
                "placement": {
                    "alignments": [
                        {
                            "part": "extruder",
                            "to": "frame",
                            "rigid_group": ["x_axis"],
                        }
                    ]
                }
            },
        )


def test_apply_placement_alignments_supports_rigid_group(monkeypatch, tmp_path):
    extruder_leader = tmp_path / "extruder_leader.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    x_axis_leader = tmp_path / "x_axis_leader.step"
    x_axis_leader.write_text("x-axis-leader", encoding="utf-8")
    tool_head_leader = tmp_path / "tool_head_leader.step"
    tool_head_leader.write_text("tool-head-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame_leader.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "x_axis", "part": "scene-x-axis", "transform_history": []},
        {
            "assembly_name": "tool_head",
            "part": "scene-tool-head",
            "transform_history": [],
        },
        {"assembly_name": "frame", "part": "scene-frame", "transform_history": []},
    ]
    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "x_axis": {
            "assembly_name": "x_axis",
            "artifacts": {"leader_step": str(x_axis_leader)},
        },
        "tool_head": {
            "assembly_name": "tool_head",
            "artifacts": {"leader_step": str(tool_head_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "extruder-leader": (1.0, 2.0, 3.0),
        "x-axis-leader": (10.0, 20.0, 30.0),
        "tool-head-leader": (4.0, 5.0, 6.0),
        "frame-leader": (40.0, 50.0, 60.0),
        "x-axis-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            10.0,
            20.0,
            70.0,
        ),
        "extruder-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            1.0,
            2.0,
            43.0,
        ),
        "tool-head-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            4.0,
            5.0,
            46.0,
        ),
    }

    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|{alignment.name}:{moving_anchor}->{target_anchor}:{axes}:{stack_gap}"
        ),
    )

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "extruder"},
                {"name": "x_axis"},
                {"name": "tool_head"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder", "tool_head"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "STACK_TOP",
                        "stack_gap": 10,
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == (
        "scene-extruder|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[1]["part"] == (
        "scene-x-axis|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[2]["part"] == (
        "scene-tool-head|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[3]["part"] == "scene-frame"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placed_parts[2]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placed_parts[3]["transform_history"] == []


def test_apply_placement_alignments_supports_rigid_group_without_alignment(
    monkeypatch, tmp_path
):
    extruder_leader = tmp_path / "extruder_leader.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    x_axis_leader = tmp_path / "x_axis_leader.step"
    x_axis_leader.write_text("x-axis-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame_leader.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "x_axis", "part": "scene-x-axis", "transform_history": []},
        {"assembly_name": "frame", "part": "scene-frame", "transform_history": []},
    ]
    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "x_axis": {
            "assembly_name": "x_axis",
            "artifacts": {"leader_step": str(x_axis_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "extruder-leader": (1.0, 2.0, 3.0),
        "x-axis-leader": (10.0, 20.0, 30.0),
        "frame-leader": (40.0, 50.0, 60.0),
        "x-axis-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            10.0,
            20.0,
            70.0,
        ),
        "extruder-leader|STACK_TOP:x-axis-leader->frame-leader:None:10": (
            1.0,
            2.0,
            43.0,
        ),
    }

    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|{alignment.name}:{moving_anchor}->{target_anchor}:{axes}:{stack_gap}"
        ),
    )

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "extruder"},
                {"name": "x_axis"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder"],
                    },
                    {
                        "part": "x_axis",
                        "to": "frame",
                        "alignment": "STACK_TOP",
                        "stack_gap": 10,
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == (
        "scene-extruder|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[1]["part"] == (
        "scene-x-axis|STACK_TOP:x-axis-leader->frame-leader:None:10"
    )
    assert placed_parts[2]["part"] == "scene-frame"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placed_parts[2]["transform_history"] == []


def test_apply_placement_alignments_replays_hidden_target_steps_for_active_scene_assemblies(
    monkeypatch, tmp_path
):
    extruder_leader = tmp_path / "extruder_leader.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    mount_leader = tmp_path / "mount_leader.step"
    mount_leader.write_text("mount-leader", encoding="utf-8")
    carriage_leader = tmp_path / "carriage_leader.step"
    carriage_leader.write_text("carriage-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame_leader.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "mount", "part": "scene-mount", "transform_history": []},
        {"assembly_name": "frame", "part": "scene-frame", "transform_history": []},
    ]
    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "mount": {
            "assembly_name": "mount",
            "artifacts": {"leader_step": str(mount_leader)},
            "dependencies": [
                {
                    "assembly_name": "extruder",
                    "source_placement_transforms": [
                        {
                            "kind": "translate",
                            "vector": [-50.0, 0.0, 0.0],
                            "placement_step": 0,
                        }
                    ],
                }
            ],
        },
        "carriage": {
            "assembly_name": "carriage",
            "artifacts": {"leader_step": str(carriage_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "extruder-leader": (0.0, 0.0, 0.0),
        "extruder-leader|TO_CARRIAGE": (-50.0, 0.0, 0.0),
        "extruder-leader|TO_FRAME": (100.0, 0.0, 0.0),
        "extruder-leader|TO_CARRIAGE|TO_FRAME": (100.0, 0.0, 0.0),
        "carriage-leader": (-50.0, 0.0, 0.0),
        "frame-leader": (100.0, 0.0, 0.0),
        "mount-leader": (-60.0, 0.0, 0.0),
    }
    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        if target_anchor == "carriage-leader":
            return lambda part: f"{part}|TO_CARRIAGE"
        if target_anchor == "frame-leader":
            return lambda part: f"{part}|TO_FRAME"
        raise AssertionError(
            f"Unexpected placement translation {moving_anchor!r} -> {target_anchor!r}"
        )

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "extruder"},
                {"name": "mount"},
                {"name": "carriage"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "part": "extruder",
                        "to": "carriage",
                        "alignment": "CENTER",
                    },
                    {
                        "to": "extruder",
                        "rigid_group": ["mount"],
                    },
                    {
                        "part": "extruder",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == "scene-extruder|TO_FRAME"
    assert placed_parts[1]["part"] == "scene-mount|TO_FRAME"
    assert placed_parts[2]["part"] == "scene-frame"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [100.0, 0.0, 0.0], "placement_step": 2},
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [100.0, 0.0, 0.0], "placement_step": 2}
    ]
    assert placed_parts[2]["transform_history"] == []


def test_apply_placement_alignments_replays_hidden_target_steps_when_only_moving_group_is_visible(
    monkeypatch, tmp_path
):
    extruder_leader = tmp_path / "extruder_leader.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    mount_leader = tmp_path / "mount_leader.step"
    mount_leader.write_text("mount-leader", encoding="utf-8")
    carriage_leader = tmp_path / "carriage_leader.step"
    carriage_leader.write_text("carriage-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame_leader.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "mount", "part": "scene-mount", "transform_history": []},
    ]
    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "mount": {
            "assembly_name": "mount",
            "artifacts": {"leader_step": str(mount_leader)},
        },
        "carriage": {
            "assembly_name": "carriage",
            "artifacts": {"leader_step": str(carriage_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "extruder-leader": (0.0, 0.0, 0.0),
        "extruder-leader|TO_CARRIAGE": (-50.0, 0.0, 0.0),
        "extruder-leader|TO_CARRIAGE|TO_FRAME": (100.0, 0.0, 0.0),
        "mount-leader": (-10.0, 0.0, 0.0),
        "mount-leader|TO_CARRIAGE": (-60.0, 0.0, 0.0),
        "mount-leader|TO_CARRIAGE|TO_FRAME": (90.0, 0.0, 0.0),
        "carriage-leader": (-50.0, 0.0, 0.0),
        "frame-leader": (100.0, 0.0, 0.0),
    }
    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        if target_anchor == "carriage-leader":
            return lambda part: f"{part}|TO_CARRIAGE"
        if target_anchor == "frame-leader":
            return lambda part: f"{part}|TO_FRAME"
        raise AssertionError(
            f"Unexpected placement translation {moving_anchor!r} -> {target_anchor!r}"
        )

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "extruder"},
                {"name": "mount"},
                {"name": "fan"},
                {"name": "carriage"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "to": "extruder",
                        "rigid_group": ["mount", "fan"],
                    },
                    {
                        "part": "extruder",
                        "to": "carriage",
                        "alignment": "CENTER",
                    },
                    {
                        "part": "extruder",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == "scene-extruder"
    assert placed_parts[1]["part"] == "scene-mount"
    assert placed_parts[0]["transform_history"] == []
    assert placed_parts[1]["transform_history"] == []


def test_apply_placement_alignments_replays_hidden_target_rigid_groups(
    monkeypatch, tmp_path
):
    z_profile_leader = tmp_path / "z_profile.step"
    z_profile_leader.write_text("z-profile-leader", encoding="utf-8")
    x_axis_leader = tmp_path / "x_axis.step"
    x_axis_leader.write_text("x-axis-leader", encoding="utf-8")
    extruder_leader = tmp_path / "extruder.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    bed_leader = tmp_path / "bed.step"
    bed_leader.write_text("bed-leader", encoding="utf-8")

    scene_parts = [
        {
            "assembly_name": "z_profile",
            "part": "scene-z-profile",
            "transform_history": [],
        },
        {
            "assembly_name": "extruder",
            "part": "scene-extruder",
            "transform_history": [],
        },
        {"assembly_name": "bed", "part": "scene-bed", "transform_history": []},
    ]
    built_results_by_name = {
        "z_profile": {
            "assembly_name": "z_profile",
            "artifacts": {"leader_step": str(z_profile_leader)},
        },
        "x_axis": {
            "assembly_name": "x_axis",
            "artifacts": {"leader_step": str(x_axis_leader)},
        },
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "bed": {
            "assembly_name": "bed",
            "artifacts": {"leader_step": str(bed_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    centers = {
        "z-profile-leader": (0.0, 0.0, 0.0),
        "x-axis-leader": (10.0, 0.0, 0.0),
        "extruder-leader": (20.0, 0.0, 0.0),
        "bed-leader": (100.0, 0.0, 0.0),
        "extruder-leader|TO_BED": (100.0, 0.0, 0.0),
    }
    monkeypatch.setattr(builder, "_part_center", lambda part: centers[part])

    def fake_make_translation(
        moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0
    ):
        assert moving_anchor == "extruder-leader"
        assert target_anchor == "bed-leader"
        return lambda part: f"{part}|TO_BED"

    monkeypatch.setattr(builder, "_make_placement_translation", fake_make_translation)

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "z_profile"},
                {"name": "x_axis"},
                {"name": "extruder"},
                {"name": "bed"},
            ],
            "placement": {
                "alignments": [
                    {
                        "to": "x_axis",
                        "rigid_group": ["z_profile"],
                    },
                    {
                        "to": "x_axis",
                        "rigid_group": ["extruder"],
                    },
                    {
                        "part": "extruder",
                        "to": "bed",
                        "alignment": "CENTER",
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == "scene-z-profile|TO_BED"
    assert placed_parts[1]["part"] == "scene-extruder|TO_BED"
    assert placed_parts[2]["part"] == "scene-bed"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [80.0, 0.0, 0.0], "placement_step": 2}
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [80.0, 0.0, 0.0], "placement_step": 2}
    ]
    assert placed_parts[2]["transform_history"] == []


def test_apply_placement_alignments_ignores_missing_rigid_predecessors_outside_scene_subset(
    monkeypatch, tmp_path
):
    left_leader = tmp_path / "left.step"
    left_leader.write_text("left-leader", encoding="utf-8")
    right_leader = tmp_path / "right.step"
    right_leader.write_text("right-leader", encoding="utf-8")
    rail_leader = tmp_path / "rail.step"
    rail_leader.write_text("rail-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    scene_parts = [
        {"assembly_name": "right", "part": "scene-right", "transform_history": []},
        {"assembly_name": "rail", "part": "scene-rail", "transform_history": []},
        {"assembly_name": "frame", "part": "scene-frame", "transform_history": []},
    ]
    built_results_by_name = {
        "left": {
            "assembly_name": "left",
            "artifacts": {"leader_step": str(left_leader)},
        },
        "right": {
            "assembly_name": "right",
            "artifacts": {"leader_step": str(right_leader)},
        },
        "rail": {
            "assembly_name": "rail",
            "artifacts": {"leader_step": str(rail_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    base_centers = {
        "left-leader": (0.0, 0.0, 0.0),
        "right-leader": (10.0, 0.0, 0.0),
        "rail-leader": (20.0, 0.0, 0.0),
        "frame-leader": (30.0, 0.0, 0.0),
    }
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: (
            base_centers[part.split("|", 1)[0]][0] + 10.0 * part.count("|MOVE"),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|MOVE"
        ),
    )
    captured_logs = []
    monkeypatch.setattr(
        builder._logger,
        "info",
        lambda message, *args: captured_logs.append(message % args),
    )

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [
                {"name": "left"},
                {"name": "right"},
                {"name": "rail"},
                {"name": "frame"},
            ],
            "placement": {
                "alignments": [
                    {
                        "part": "left",
                        "to": "rail",
                        "alignment": "CENTER",
                    },
                    {
                        "to": "rail",
                        "rigid_group": ["left"],
                    },
                    {
                        "part": "right",
                        "to": "rail",
                        "alignment": "CENTER",
                    },
                    {
                        "to": "rail",
                        "rigid_group": ["right"],
                    },
                    {
                        "part": "rail",
                        "to": "frame",
                        "alignment": "CENTER",
                    },
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert placed_parts[0]["part"] == "scene-right|MOVE|MOVE"
    assert placed_parts[1]["part"] == "scene-rail|MOVE"
    assert placed_parts[2]["part"] == "scene-frame"
    assert placed_parts[0]["transform_history"] == [
        {"kind": "translate", "vector": [10.0, 0.0, 0.0], "placement_step": 2},
        {"kind": "translate", "vector": [10.0, 0.0, 0.0], "placement_step": 4},
    ]
    assert placed_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [10.0, 0.0, 0.0], "placement_step": 4}
    ]
    assert placed_parts[2]["transform_history"] == []
    assert any(
        "Placement replay start: visible_assemblies=['frame', 'rail', 'right']; active_source_steps=2-4; skipped_source_steps=2 outside visible subset"
        in message
        for message in captured_logs
    )
    assert all("left aligned" not in message for message in captured_logs)
    assert any(
        "Placement replay execute: source_step=2; right aligned to rail via CENTER"
        for message in captured_logs
    )
    assert any(
        "Placement replay execute: source_step=4; rail aligned to frame via CENTER"
        for message in captured_logs
    )


def test_apply_placement_alignments_rigid_group_rejects_relative_motion_inside_group(
    monkeypatch, tmp_path
):
    left_leader = tmp_path / "left.step"
    left_leader.write_text("left-leader", encoding="utf-8")
    right_leader = tmp_path / "right.step"
    right_leader.write_text("right-leader", encoding="utf-8")

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "left-leader": (0.0, 0.0, 0.0),
            "right-leader": (10.0, 0.0, 0.0),
            "left-leader|ALIGN": (10.0, 0.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|ALIGN"
        ),
    )

    with pytest.raises(builder.BuilderError) as exc_info:
        builder._apply_placement_alignments(
            [
                {
                    "assembly_name": "left",
                    "part": "scene-left",
                    "transform_history": [],
                },
                {
                    "assembly_name": "right",
                    "part": "scene-right",
                    "transform_history": [],
                },
            ],
            built_results_by_name={
                "left": {
                    "assembly_name": "left",
                    "artifacts": {"leader_step": str(left_leader)},
                },
                "right": {
                    "assembly_name": "right",
                    "artifacts": {"leader_step": str(right_leader)},
                },
            },
            repository_dir=tmp_path,
            config_data={
                "assemblies": [{"name": "left"}, {"name": "right"}],
                "placement": {
                    "alignments": [
                        {
                            "to": "right",
                            "rigid_group": ["left"],
                        },
                        {
                            "part": "right",
                            "to": "left",
                            "alignment": "TOP",
                        },
                    ]
                },
            },
        )

    message = str(exc_info.value)
    assert "placement.alignments[1]" in message
    assert "right to left" in message
    assert "rigidly attached assemblies 'right' and 'left'" in message
    assert (
        "placement spec: {'part': 'right', 'to': 'left', 'alignment': 'TOP'}" in message
    )
    assert "'right' is already rigidly connected with ['left', 'right']" in message
    assert "keys present: 'alignment', 'part', 'to'" in message


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
            "moving-anchor|ALIGN": (10.0, 20.0, 30.0),
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

    placement_result = builder._apply_placement_alignments(
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
    placed_parts = placement_result.scene_parts

    assert captured_post_translations == [(1.0, 0.0, 7.0)]
    assert placed_parts[0]["part"] == "scene-moving|ALIGN|POST"
    assert placed_parts[1]["part"] == "scene-target"


def test_resolve_post_rotation_supports_explicit_vector_center():
    resolved = builder._resolve_post_rotation(
        {
            "post_rotation": {
                "angle": 90,
                "axis": [1, 0, 0],
                "center": [1, {"$expr": {"$sub": "2 + ${gap}"}}, 3],
            }
        },
        {"gap": 5},
        resolve_anchor=lambda reference: (_ for _ in ()).throw(
            AssertionError(f"Unexpected anchor lookup for {reference}")
        ),
    )

    assert resolved == {
        "angle": 90.0,
        "axis": (1.0, 0.0, 0.0),
        "center": (1.0, 7.0, 3.0),
    }


def test_apply_placement_alignments_supports_post_rotation_without_alignment(
    monkeypatch, tmp_path
):
    moving_leader = tmp_path / "moving_leader.step"
    moving_leader.write_text("moving-leader", encoding="utf-8")
    target_anchor = tmp_path / "target_anchor.step"
    target_anchor.write_text("target-anchor", encoding="utf-8")

    scene_parts = [
        {"assembly_name": "moving", "part": "scene-moving", "transform_history": []},
        {"assembly_name": "target", "part": "scene-target", "transform_history": []},
    ]
    built_results_by_name = {
        "moving": {
            "assembly_name": "moving",
            "artifacts": {"leader_step": str(moving_leader)},
        },
        "target": {
            "assembly_name": "target",
            "artifacts": {
                "leader_step": str(target_anchor),
                "non_production_parts": [
                    {"name": "anchor", "path": str(target_anchor)}
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
            "moving-leader": (1.0, 2.0, 3.0),
            "target-anchor": (10.0, 20.0, 30.0),
            "moving-leader|ROTATE": (4.0, 5.0, 6.0),
        }[part],
    )
    captured_rotations = []
    monkeypatch.setattr(
        builder,
        "_make_post_rotation",
        lambda angle, *, axis=None, center=None: captured_rotations.append(
            (angle, axis, center)
        )
        or (lambda part: f"{part}|ROTATE"),
    )

    placement_result = builder._apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        config_data={
            "assemblies": [{"name": "moving"}, {"name": "target"}],
            "placement": {
                "alignments": [
                    {
                        "part": "moving",
                        "post_rotation": {
                            "angle": 90,
                            "center": "target.non_production_parts.anchor.CENTER",
                        },
                    }
                ]
            },
        },
    )
    placed_parts = placement_result.scene_parts

    assert captured_rotations == [(90.0, (0.0, 0.0, 1.0), (10.0, 20.0, 30.0))]
    assert placed_parts[0]["part"] == "scene-moving|ROTATE"
    assert placed_parts[1]["part"] == "scene-target"
    assert placed_parts[0]["transform_history"] == [
        {
            "kind": "rotate",
            "angle": 90.0,
            "axis": [0.0, 0.0, 1.0],
            "center": [10.0, 20.0, 30.0],
            "placement_step": 0,
        }
    ]


def test_build_placed_assemblies_debug_payload_uses_leader_bounding_boxes_only():
    payload = builder._build_placed_assemblies_debug_payload(
        {
            "alpha_assembly": create_box(10, 20, 30),
            "beta_assembly": translate(-2, -3, 4)(create_box(2, 3, 4)),
        },
        build_results=[
            {"assembly_name": "beta_assembly"},
            {"assembly_name": "gamma_assembly"},
            {"assembly_name": "alpha_assembly"},
        ],
        target_assembly="whole_printer_assembly",
        mode="visualization",
    )

    assert payload["schema_version"] == 1
    assert payload["target_assembly"] == "whole_printer_assembly"
    assert payload["mode"] == "visualization"
    assert payload["built_assembly_names"] == [
        "alpha_assembly",
        "beta_assembly",
        "gamma_assembly",
    ]

    alpha_entry = next(
        entry
        for entry in payload["placed_assemblies"]
        if entry["assembly_name"] == "alpha_assembly"
    )
    assert alpha_entry["bounding_box"]["min"] == [0.0, 0.0, 0.0]
    assert alpha_entry["bounding_box"]["max"] == [10.0, 20.0, 30.0]

    beta_entry = next(
        entry
        for entry in payload["placed_assemblies"]
        if entry["assembly_name"] == "beta_assembly"
    )
    assert beta_entry["bounding_box"]["min"] == [-2.0, -3.0, 4.0]
    assert beta_entry["bounding_box"]["max"] == [0.0, 0.0, 8.0]


def test_build_placed_assemblies_debug_payload_rounds_floats_to_four_decimals():
    payload = builder._build_placed_assemblies_debug_payload(
        {
            "rounded_assembly": translate(0.123456, 0.987654, 0.000049)(
                create_box(1.234567, 2.345678, 3.456789)
            ),
        },
        build_results=[{"assembly_name": "rounded_assembly"}],
        target_assembly="rounded_assembly",
        mode="visualization",
    )

    entry = payload["placed_assemblies"][0]
    assert entry["bounding_box"]["min"] == [0.1235, 0.9877, 0.0]
    assert entry["bounding_box"]["max"] == [1.358, 3.3333, 3.4568]
    assert entry["bounding_box"]["center"] == [0.7407, 2.1605, 1.7284]
    assert entry["bounding_box"]["size"] == [1.2346, 2.3457, 3.4568]


def test_materialize_placed_leaders_by_assembly_applies_recorded_transforms(
    monkeypatch, tmp_path
):
    alpha_leader = tmp_path / "alpha.step"
    alpha_leader.write_text("alpha", encoding="utf-8")
    beta_leader = tmp_path / "beta.step"
    beta_leader.write_text("beta", encoding="utf-8")

    built_results_by_name = {
        "alpha": {
            "assembly_name": "alpha",
            "artifacts": {"leader_step": str(alpha_leader)},
        },
        "beta": {
            "assembly_name": "beta",
            "artifacts": {"leader_step": str(beta_leader)},
        },
    }

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: Path(path).read_text(encoding="utf-8"),
    )

    placed_leaders = builder._materialize_placed_leaders_by_assembly(
        ["alpha", "beta"],
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
        assembly_transforms={
            "alpha": [lambda part: f"{part}|MOVE_A", lambda part: f"{part}|MOVE_B"],
        },
    )

    assert placed_leaders == {
        "alpha": "alpha|MOVE_A|MOVE_B",
        "beta": "beta",
    }


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
        config_data,
        built_results_by_name,
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


def test_resolve_dependency_injections_requires_available_build_pool_slot(tmp_path):
    entry = {
        "name": "consumer",
        "inject_parts": {
            "provider": "provider.leader",
        },
    }
    pool_without_provider = builder._BuildArtifactPool(["consumer"])

    with pytest.raises(builder.BuilderError, match="not part of this build pool"):
        builder._resolve_dependency_injections(
            entry,
            {},
            tmp_path,
            build_pool=pool_without_provider,
        )

    pool_with_unavailable_provider = builder._BuildArtifactPool(
        ["consumer", "provider"]
    )
    with pytest.raises(builder.BuilderError, match="not available in the build pool"):
        builder._resolve_dependency_injections(
            entry,
            {},
            tmp_path,
            build_pool=pool_with_unavailable_provider,
        )


def test_unresolved_artifact_diagnostic_skips_unbuilt_matching_scan_assemblies(
    tmp_path,
):
    provider_metadata = {
        "assembly_name": "provider",
        "artifacts": {
            "leader_step": str(tmp_path / "provider.step"),
            "non_production_parts": [
                {
                    "index": 0,
                    "name": "board",
                    "path": str(tmp_path / "provider_board.step"),
                }
            ],
        },
    }
    matching_metadata = {
        "assembly_name": "matching_provider",
        "artifacts": {
            "leader_step": str(tmp_path / "matching_provider.step"),
            "non_production_parts": [
                {
                    "index": 0,
                    "name": "mount_screw",
                    "path": str(tmp_path / "matching_mount_screw.step"),
                }
            ],
        },
    }

    with pytest.raises(builder.BuilderError) as excinfo:
        builder._raise_unresolved_artifact_reference(
            label="Placement reference",
            reference="provider.non_production_parts.mount_screw",
            resolved_reference=None,
            assembly_name="provider",
            selector="non_production_parts.mount_screw",
            entries=[],
            metadata=provider_metadata,
            known_assembly_names=[
                "provider",
                "calibration_assembly",
                "matching_provider",
            ],
            built_results_by_name={
                "provider": provider_metadata,
                "matching_provider": matching_metadata,
            },
            repository_dir=tmp_path / "repository",
        )

    message = str(excinfo.value)
    assert (
        "Placement reference 'provider.non_production_parts.mount_screw' "
        "matched no artifacts in assembly 'provider'."
    ) in message
    assert "Valid references for this assembly: provider.leader" in message
    assert "provider.non_production_parts.board" in message
    assert (
        "Matching references in other assemblies: "
        "matching_provider.non_production_parts.mount_screw"
    ) in message
    assert "calibration_assembly" not in message
    assert "has no repository directory" not in message


def _write_dirty_guard_project(
    tmp_path: Path,
    *,
    package_name: str,
    generator_lines: list[str],
    resources: Mapping[str, str],
    config_lines: list[str],
) -> Path:
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / package_name
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(src_dir / "generators.py", "\n".join(generator_lines))

    assemblies_dir = project_root / "assembling" / "assemblies"
    for assembly_name, generator_name in resources.items():
        _write_file(
            assemblies_dir / f"{assembly_name}.yaml",
            "\n".join(
                [
                    "Parts:",
                    f"  {assembly_name}:",
                    "    Type: Shellforgepy::Assembly",
                    "    Properties:",
                    f"      Generator: {package_name}.generators.{generator_name}",
                ]
            ),
        )

    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(config_path, "\n".join(config_lines))
    return config_path


def _install_string_builder_io(monkeypatch):
    def fake_export(part, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(part), encoding="utf-8")

    def fake_import(step_path):
        return Path(step_path).read_text(encoding="utf-8")

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(builder, "_import_dependency_part", fake_import)


def _install_geometry_builder_io(monkeypatch):
    exported_parts = {}

    def fake_export(part, destination):
        resolved = str(destination.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(resolved, encoding="utf-8")
        exported_parts[resolved] = part

    def fake_import(step_path):
        return exported_parts[str(Path(step_path).resolve())]

    monkeypatch.setattr(builder, "_export_part_to_step", fake_export)
    monkeypatch.setattr(builder, "_import_dependency_part", fake_import)
    return exported_parts


def _clear_demo_package(monkeypatch, package_name: str) -> None:
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.generators", raising=False)


def _write_declarative_composite_demo_project(
    tmp_path: Path,
    *,
    package_name: str,
    composite_resource_lines: list[str],
    config_lines: list[str] | None = None,
) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    src_dir = project_root / "src" / package_name
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "generators.py",
        "\n".join(
            [
                "from shellforgepy.construct.leader_followers_cutters_part import LeaderFollowersCuttersPart",
                "from shellforgepy.simple import create_box, translate",
                "",
                "def make_source_left():",
                "    part = LeaderFollowersCuttersPart(",
                "        leader=translate(0, 0, 0)(create_box(10, 10, 10))",
                "    )",
                "    part.add_named_follower(",
                "        translate(0, 0, 12)(create_box(2, 2, 2)),",
                "        'guide',",
                "    )",
                "    part.followers.append(translate(4, 0, 12)(create_box(1, 1, 1)))",
                "    part.add_named_cutter(",
                "        translate(1, 1, 0)(create_box(1, 1, 12)),",
                "        'mount_hole',",
                "    )",
                "    part.add_named_non_production_part(",
                "        translate(0, 0, -2)(create_box(2, 2, 2)),",
                "        'screw',",
                "    )",
                "    part.non_production_parts.append(",
                "        translate(3, 0, -2)(create_box(1, 1, 1))",
                "    )",
                "    return part",
                "",
                "def make_source_right():",
                "    part = LeaderFollowersCuttersPart(",
                "        leader=translate(20, 0, 0)(create_box(8, 8, 8))",
                "    )",
                "    part.add_named_follower(",
                "        translate(20, 0, 10)(create_box(2, 2, 2)),",
                "        'flange',",
                "    )",
                "    part.followers.append(translate(24, 0, 10)(create_box(1, 1, 1)))",
                "    part.add_named_cutter(",
                "        translate(21, 1, 0)(create_box(1, 1, 10)),",
                "        'slot',",
                "    )",
                "    part.add_named_non_production_part(",
                "        translate(20, 0, -2)(create_box(2, 2, 2)),",
                "        'bolt',",
                "    )",
                "    part.non_production_parts.append(",
                "        translate(24, 0, -2)(create_box(1, 1, 1))",
                "    )",
                "    return part",
                "",
                "def inspect_composite(*, composite):",
                "    follower_names = sorted(composite.follower_indices_by_name)",
                "    cutter_names = sorted(composite.cutter_indices_by_name)",
                "    nonprod_names = sorted(composite.non_production_indices_by_name)",
                "    return (",
                "        f\"followers={len(composite.followers)}:{','.join(follower_names)};\"",
                "        f\"cutters={len(composite.cutters)}:{','.join(cutter_names)};\"",
                "        f\"nonprod={len(composite.non_production_parts)}:{','.join(nonprod_names)}\"",
                "    )",
                "",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "source_left.yaml",
        "\n".join(
            [
                "Parts:",
                "  SourceLeft:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.make_source_left",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "source_right.yaml",
        "\n".join(
            [
                "Parts:",
                "  SourceRight:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.make_source_right",
            ]
        ),
    )
    composite_resource_path = assemblies_dir / "composite.yaml"
    _write_file(composite_resource_path, "\n".join(composite_resource_lines))
    _write_file(
        assemblies_dir / "consumer.yaml",
        "\n".join(
            [
                "Parts:",
                "  Consumer:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.inspect_composite",
            ]
        ),
    )

    if config_lines is None:
        config_lines = [
            "assemblies:",
            "  - name: source_left",
            "  - name: source_right",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "      - source_right",
            "    inject_parts:",
            "      left: source_left",
            "      right: source_right",
            "  - name: consumer",
            "    depends_on:",
            "      - composite",
            "    inject_parts:",
            "      composite: composite",
        ]
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(config_path, "\n".join(config_lines))
    return config_path, composite_resource_path


def _base_composite_resource_lines() -> list[str]:
    return [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "            - source: injected",
        "              assembly: right",
        "              artifact: fused",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: followers",
        "            names:",
        "              - guide",
        "            name_template: '{inject_name}_{name}'",
        "          - source: injected",
        "            assembly: right",
        "            artifact: followers",
        "            exclude_names:",
        "              - flange",
        "            name_template: '{inject_name}_follower_{index}'",
        "          - source: injected",
        "            assembly: left",
        "            artifact: leader",
        "            name: left_body",
        "        Cutters:",
        "          - source: injected",
        "            assembly: right",
        "            artifact: cutters.slot",
        "            name: right_slot",
        "        NonProductionParts:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: non_production_parts.screw",
        "            name_template: '{inject_name}_{name}'",
        "          - source: injected",
        "            assembly: right",
        "            artifact: non_production_parts",
        "            exclude_names:",
        "              - bolt",
        "            name_template: '{inject_name}_nonprod_{index}'",
    ]


def test_build_from_file_builds_declarative_composite_assembly_from_injected_assemblies(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_demo_pkg"
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=_base_composite_resource_lines(),
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    results = builder.build_from_file(config_path)

    composite_result = next(
        result for result in results if result["assembly_name"] == "composite"
    )
    assert composite_result["generator"] == builder._COMPOSITE_ASSEMBLY_GENERATOR_PATH
    assert "composite_spec_sha256" in composite_result["version_inputs"]
    assert "resource_sha256" not in composite_result["version_inputs"]

    composite_part = builder._import_dependency_assembly(composite_result)
    assert sorted(composite_part.follower_indices_by_name) == [
        "left_body",
        "left_guide",
        "right_follower_1",
    ]
    assert sorted(composite_part.cutter_indices_by_name) == ["right_slot"]
    assert sorted(composite_part.non_production_indices_by_name) == [
        "left_screw",
        "right_nonprod_1",
    ]
    assert len(composite_part.followers) == 3
    assert len(composite_part.cutters) == 1
    assert len(composite_part.non_production_parts) == 2

    composite_bbox = get_bounding_box(composite_part.leader)
    assert composite_bbox[0][0] <= 0.0
    assert composite_bbox[1][0] >= 28.0
    assert composite_bbox[1][2] >= 12.0

    consumer_result = next(
        result for result in results if result["assembly_name"] == "consumer"
    )
    consumer_payload = builder._import_dependency_part(
        Path(consumer_result["artifacts"]["leader_step"])
    )
    assert (
        consumer_payload == "followers=3:left_body,left_guide,right_follower_1;"
        "cutters=1:right_slot;"
        "nonprod=2:left_screw,right_nonprod_1"
    )


def test_build_from_file_allows_wildcard_composite_name_filters(monkeypatch, tmp_path):
    package_name = "declarative_composite_wildcard_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: followers",
        "            names:",
        "              - gui*",
        "            name_template: '{name}'",
        "          - source: injected",
        "            assembly: right",
        "            artifact: followers",
        "            exclude_names:",
        "              - flang*",
        "            name_template: 'right_{index}'",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    results = builder.build_from_file(config_path)

    composite_result = next(
        result for result in results if result["assembly_name"] == "composite"
    )
    composite_part = builder._import_dependency_assembly(composite_result)
    assert sorted(composite_part.follower_indices_by_name) == ["guide", "right_1"]


def test_scene_artifact_filters_allow_wildcard_names():
    entries = [
        {"name": "elko_sleeve_plate_1"},
        {"name": "elko_sleeve_plate_2"},
        {"name": "tpu_cover"},
        {"name": None},
    ]

    included = builder._filter_artifact_entries(
        entries,
        {"names": ["elko_sleeve_plate_*"]},
    )
    excluded = builder._filter_artifact_entries(
        entries,
        {"exclude_names": ["elko_sleeve_plate_*"]},
    )

    assert [entry["name"] for entry in included] == [
        "elko_sleeve_plate_1",
        "elko_sleeve_plate_2",
    ]
    assert [entry["name"] for entry in excluded] == ["tpu_cover", None]


def test_materialize_production_rule_expands_num_copies(monkeypatch, tmp_path):
    source_path = tmp_path / "component.step"
    source_path.write_text("step", encoding="utf-8")
    imported_part = {"source": "component"}
    copied_parts = []

    monkeypatch.setattr(
        builder,
        "_import_dependency_part",
        lambda path: imported_part,
    )

    def fake_copy_part(part):
        copied = {"copy_of": part}
        copied_parts.append(copied)
        return copied

    monkeypatch.setattr(builder, "copy_part", fake_copy_part)

    scene_parts = builder._materialize_rule_parts(
        {
            "assembly_name": "component_assembly",
            "artifacts": {"leader_step": str(source_path)},
            "parameter_hash": "abc123",
            "version_inputs": {},
        },
        {
            "Builder": {
                "Production": {
                    "parts": [
                        {
                            "source": "self",
                            "artifact": "leader",
                            "name": "component",
                            "num_copies": 3,
                        }
                    ]
                }
            }
        },
        "production",
        {},
        tmp_path,
    )

    assert [part["name"] for part in scene_parts] == [
        "component",
        "component_copy_2",
        "component_copy_3",
    ]
    assert [part["copy_index"] for part in scene_parts] == [1, 2, 3]
    assert [part["copy_source_name"] for part in scene_parts] == [
        "component",
        "component",
        "component",
    ]
    assert scene_parts[0]["part"] is imported_part
    assert scene_parts[1]["part"] is copied_parts[0]
    assert scene_parts[2]["part"] is copied_parts[1]


def test_scene_rules_use_production_section_num_copies_as_default():
    metadata = {
        "artifacts": {"leader_step": "component.step"},
        "public_parameters": {"num_copies": 3},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "num_copies": {"$ref": "num_copies"},
                "parts": [
                    {
                        "source": "self",
                        "artifact": "leader",
                        "name": "component",
                    }
                ],
            }
        }
    }

    rules = builder._scene_rules_for_mode(metadata, resource_data, "production")

    assert rules == [
        {
            "source": "self",
            "artifact": "leader",
            "name": "component",
            "num_copies": {"$ref": "num_copies"},
        }
    ]

    assert (
        builder._normalize_scene_num_copies(
            rules[0],
            builder._metadata_resolution_context(metadata),
        )
        == 3
    )


def test_declared_plate_parts_expand_exact_copy_source_names():
    plates = [{"name": "component_plate", "parts": ["component"]}]
    scene_parts = [
        {"name": "component", "copy_source_name": "component"},
        {"name": "component_copy_2", "copy_source_name": "component"},
        {"name": "component_copy_3", "copy_source_name": "component"},
    ]

    expanded = builder._expand_declared_plate_parts_for_copies(plates, scene_parts)

    assert expanded == [
        {
            "name": "component_plate",
            "parts": ["component", "component_copy_2", "component_copy_3"],
        }
    ]


def test_build_from_file_rebuilds_composite_when_composite_spec_changes_but_not_for_visualization_changes(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_cache_demo_pkg"
    config_path, composite_resource_path = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=_base_composite_resource_lines(),
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    first_results = builder.build_from_file(config_path)
    first_composite = next(
        result for result in first_results if result["assembly_name"] == "composite"
    )

    _write_file(
        composite_resource_path,
        "\n".join(
            _base_composite_resource_lines()
            + [
                "Builder:",
                "  Visualization:",
                "    parts:",
                "      - source: injected",
                "        assembly: left",
                "        artifact: all",
            ]
        ),
    )
    second_results = builder.build_from_file(config_path)
    second_composite = next(
        result for result in second_results if result["assembly_name"] == "composite"
    )

    assert second_composite["cache_hit"] is True
    assert second_composite["parameter_hash"] == first_composite["parameter_hash"]

    changed_lines = _base_composite_resource_lines()
    changed_lines[19] = "            name_template: '{inject_name}_renamed_{name}'"
    _write_file(composite_resource_path, "\n".join(changed_lines))
    third_results = builder.build_from_file(config_path)
    third_composite = next(
        result for result in third_results if result["assembly_name"] == "composite"
    )

    assert third_composite["cache_hit"] is False
    assert third_composite["parameter_hash"] != first_composite["parameter_hash"]


def test_build_from_file_allows_name_template_for_unnamed_entries_when_template_does_not_use_name(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_unnamed_template_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: followers",
        "            exclude_names:",
        "              - guide",
        "            name_template: '{inject_name}_{index}'",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "    inject_parts:",
            "      left: source_left",
        ],
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    results = builder.build_from_file(config_path)
    composite_result = next(
        result for result in results if result["assembly_name"] == "composite"
    )
    composite_part = builder._import_dependency_assembly(composite_result)

    assert sorted(composite_part.follower_indices_by_name) == ["left_1"]
    assert len(composite_part.followers) == 1


def test_build_from_file_rejects_composite_name_template_using_name_for_unnamed_entries(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_bad_template_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: followers",
        "            exclude_names:",
        "              - guide",
        "            name_template: '{inject_name}_{name}'",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "    inject_parts:",
            "      left: source_left",
        ],
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "requires 'name'" in str(excinfo.value)


def test_build_from_file_rejects_composite_name_override_for_multi_match_selector(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_multi_name_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: followers",
        "            name: all_followers",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "    inject_parts:",
            "      left: source_left",
        ],
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "defines 'name' but resolves to more than one part" in str(excinfo.value)


def test_build_from_file_rejects_composite_alias_that_is_not_full_assembly(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_bad_injection_demo_pkg"
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=_base_composite_resource_lines(),
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: source_right",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "      - source_right",
            "    inject_parts:",
            "      left: source_left.leader",
            "      right: source_right",
        ],
    )
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "inject_parts.left must inject the full assembly" in str(excinfo.value)


def test_build_from_file_rejects_composite_selector_with_unknown_injected_alias(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_unknown_alias_demo_pkg"
    composite_lines = _base_composite_resource_lines()
    composite_lines = [
        (
            "              assembly: missing"
            if line == "              assembly: left"
            else line
        )
        for line in composite_lines
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
    )
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "references injected alias 'missing'" in str(excinfo.value)


def test_build_from_file_rejects_composite_selector_with_unsupported_artifact_all(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_artifact_all_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: all",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "    inject_parts:",
            "      left: source_left",
        ],
    )
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "unsupported artifact 'all'" in str(excinfo.value)


def test_build_from_file_rejects_composite_name_collisions_across_channels(
    monkeypatch, tmp_path
):
    package_name = "declarative_composite_collision_demo_pkg"
    composite_lines = [
        "Parts:",
        "  Composite:",
        "    Type: Shellforgepy::Assembly",
        "    Properties:",
        "      Composite:",
        "        Leader:",
        "          Fused:",
        "            - source: injected",
        "              assembly: left",
        "              artifact: leader",
        "        Followers:",
        "          - source: injected",
        "            assembly: left",
        "            artifact: leader",
        "            name: duplicate",
        "        NonProductionParts:",
        "          - source: injected",
        "            assembly: right",
        "            artifact: non_production_parts.bolt",
        "            name: duplicate",
    ]
    config_path, _ = _write_declarative_composite_demo_project(
        tmp_path,
        package_name=package_name,
        composite_resource_lines=composite_lines,
        config_lines=[
            "assemblies:",
            "  - name: source_left",
            "  - name: source_right",
            "  - name: composite",
            "    depends_on:",
            "      - source_left",
            "      - source_right",
            "    inject_parts:",
            "      left: source_left",
            "      right: source_right",
        ],
    )
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(config_path)

    assert "Composite part name collision for 'duplicate'" in str(excinfo.value)


def _write_join_demo_project(tmp_path: Path, *, package_name: str) -> Path:
    project_root = tmp_path / "join_project"
    src_dir = project_root / "src" / package_name
    _write_file(project_root / "pyproject.toml", "[build-system]\nrequires=[]\n")
    _write_file(src_dir / "__init__.py", "")
    _write_file(
        src_dir / "generators.py",
        "\n".join(
            [
                "from shellforgepy.construct.leader_followers_cutters_part import LeaderFollowersCuttersPart",
                "from shellforgepy.simple import create_box, translate",
                "",
                "JOIN_CALLS = []",
                "",
                "def make_left():",
                "    part = LeaderFollowersCuttersPart(create_box(10, 10, 2))",
                "    part.add_named_non_production_part(",
                "        translate(0, 0, 3)(create_box(1, 1, 1)),",
                "        'left_marker',",
                "    )",
                "    return part",
                "",
                "def make_right():",
                "    part = LeaderFollowersCuttersPart(translate(20, 0, 0)(create_box(8, 8, 2)))",
                "    part.add_named_non_production_part(",
                "        translate(20, 0, 3)(create_box(1, 1, 1)),",
                "        'right_marker',",
                "    )",
                "    return part",
                "",
                "def join_pair(*, part_a, part_b, flange_thickness):",
                "    JOIN_CALLS.append(flange_thickness)",
                "    joined_a = part_a.copy()",
                "    joined_b = part_b.copy()",
                "    joined_a.add_named_follower(",
                "        translate(0, 0, 4)(create_box(2, 2, flange_thickness)),",
                "        'left_join_flange',",
                "    )",
                "    joined_b.add_named_follower(",
                "        translate(20, 0, 4)(create_box(2, 2, flange_thickness)),",
                "        'right_join_flange',",
                "    )",
                "    return {'part_a': joined_a, 'part_b': joined_b}",
                "",
                "def inspect_joined(*, left, right):",
                "    return (",
                '        f"left={sorted(left.follower_indices_by_name)};"',
                '        f"right={sorted(right.follower_indices_by_name)}"',
                "    )",
            ]
        ),
    )

    assemblies_dir = project_root / "assembling" / "assemblies"
    _write_file(
        assemblies_dir / "left_source.yaml",
        "\n".join(
            [
                "Parts:",
                "  Left:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.make_left",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "right_source.yaml",
        "\n".join(
            [
                "Parts:",
                "  Right:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.make_right",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "pair_joiner.yaml",
        "\n".join(
            [
                'ShellforgepyBuilderVersion: "2026-03-27"',
                "Parameters:",
                "  flange_thickness:",
                "    Type: Float",
                "    Default: 3.0",
                "Parts:",
                "  PairJoiner:",
                "    Type: Shellforgepy::AssemblyJoiner",
                "    Properties:",
                f"      Joiner: {package_name}.generators.join_pair",
                "      Properties:",
                "        flange_thickness:",
                "          $ref: flange_thickness",
                "Builder:",
                "  Outputs:",
                "    part_a:",
                "      Visualization:",
                "        parts:",
                "          - source: self",
                "            artifact: followers",
                "            name_template: 'self_{default_name}'",
                "          - source: output",
                "            assembly: part_b",
                "            artifact: followers",
                "            name_template: 'mate_{default_name}'",
                "          - source: injected",
                "            assembly: part_a",
                "            artifact: non_production_parts",
                "            name_template: 'source_{default_name}'",
                "      Production:",
                "        parts:",
                "          - source: self",
                "            artifact: leader",
                "            name: left_prod",
                "    part_b:",
                "      Visualization:",
                "        parts:",
                "          - source: self",
                "            artifact: followers",
                "      Production:",
                "        parts:",
                "          - source: self",
                "            artifact: leader",
            ]
        ),
    )
    _write_file(
        assemblies_dir / "consumer.yaml",
        "\n".join(
            [
                "Parts:",
                "  Consumer:",
                "    Type: Shellforgepy::Assembly",
                "    Properties:",
                f"      Generator: {package_name}.generators.inspect_joined",
            ]
        ),
    )
    config_path = assemblies_dir / "assemblies.yaml"
    _write_file(
        config_path,
        "\n".join(
            [
                "assemblies:",
                "  - name: left_source",
                "  - name: right_source",
                "  - name: pair_join",
                "    kind: join",
                "    resource_file: pair_joiner.yaml",
                "    inject_parts:",
                "      part_a: left_source",
                "      part_b: right_source",
                "    outputs:",
                "      part_a: left_joined",
                "      part_b: right_joined",
                "  - name: consumer",
                "    depends_on:",
                "      - left_joined",
                "      - right_joined",
                "    inject_parts:",
                "      left: left_joined",
                "      right: right_joined",
            ]
        ),
    )
    return config_path


def test_build_graph_model_adds_join_operation_nodes_and_public_outputs():
    model = builder_graph_model.build_graph_model(
        [
            {"name": "left_source"},
            {"name": "right_source"},
            {
                "name": "pair_join",
                "kind": "join",
                "resource_file": "pair_joiner.yaml",
                "inject_parts": {
                    "part_a": "left_source",
                    "part_b": "right_source",
                },
                "outputs": {
                    "part_a": "left_joined",
                    "part_b": "right_joined",
                },
            },
        ]
    )

    assert "pair_join" not in model.assemblies_by_name
    assert sorted(model.assemblies_by_name) == [
        "left_joined",
        "left_source",
        "right_joined",
        "right_source",
    ]
    assert model.join_output_to_operation["left_joined"] == "pair_join"
    assert builder_graph_model.resolve_build_generation_names(
        model,
        ["left_joined"],
    ) == [
        ["left_source", "right_source"],
        ["join:pair_join"],
        ["left_joined", "right_joined"],
    ]


def test_build_graph_model_rejects_join_inject_artifact_selector():
    with pytest.raises(builder.BuilderError) as excinfo:
        builder_graph_model.build_graph_model(
            [
                {"name": "left_source"},
                {"name": "right_source"},
                {
                    "name": "pair_join",
                    "kind": "join",
                    "resource_file": "pair_joiner.yaml",
                    "inject_parts": {
                        "part_a": "left_source.leader",
                        "part_b": "right_source",
                    },
                    "outputs": {
                        "part_a": "left_joined",
                        "part_b": "right_joined",
                    },
                },
            ]
        )

    assert "inject_parts.part_a must inject the full assembly" in str(excinfo.value)


def test_build_from_file_builds_join_outputs_and_downstream_injection(
    monkeypatch, tmp_path
):
    package_name = "assembly_join_demo_pkg"
    config_path = _write_join_demo_project(tmp_path, package_name=package_name)
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    results = builder.build_from_file(config_path)
    by_name = {result["assembly_name"]: result for result in results}

    assert "pair_join" not in by_name
    assert {"left_joined", "right_joined", "consumer"}.issubset(by_name)
    assert (
        by_name["left_joined"]["generator"] == builder._JOINED_ASSEMBLY_GENERATOR_PATH
    )
    assert by_name["left_joined"]["join_operation_name"] == "pair_join"
    assert by_name["left_joined"]["join_output_alias"] == "part_a"
    assert by_name["left_joined"]["join_output_assembly_names"] == {
        "part_a": "left_joined",
        "part_b": "right_joined",
    }

    consumer_payload = builder._import_dependency_part(
        Path(by_name["consumer"]["artifacts"]["leader_step"])
    )
    assert consumer_payload == ("left=['left_join_flange'];right=['right_join_flange']")

    resource_data = builder._load_yaml(Path(by_name["left_joined"]["resource_file"]))
    scene_parts = builder._materialize_rule_parts(
        by_name["left_joined"],
        resource_data,
        "visualization",
        by_name,
        Path(by_name["left_joined"]["repository_dir"]),
    )
    assert [part["name"] for part in scene_parts] == [
        "self_left_join_flange",
        "mate_right_join_flange",
        "source_left_marker",
    ]


def test_build_from_file_selected_join_output_materializes_all_outputs_and_reuses_cache(
    monkeypatch, tmp_path
):
    package_name = "assembly_join_cache_demo_pkg"
    config_path = _write_join_demo_project(tmp_path, package_name=package_name)
    _install_geometry_builder_io(monkeypatch)
    _clear_demo_package(monkeypatch, package_name)

    first_results = builder.build_from_file(config_path, assembly_names=["left_joined"])
    first_by_name = {result["assembly_name"]: result for result in first_results}

    assert set(first_by_name) == {
        "left_source",
        "right_source",
        "left_joined",
        "right_joined",
    }
    assert first_by_name["left_joined"]["cache_hit"] is False
    assert first_by_name["right_joined"]["cache_hit"] is False

    monkeypatch.setattr(
        builder,
        "_export_part_to_step",
        lambda part, destination: pytest.fail(
            f"cache hit should not export artifacts again: {destination}"
        ),
    )
    cached_results = builder.build_from_file(
        config_path,
        assembly_names=["right_joined"],
    )
    cached_by_name = {result["assembly_name"]: result for result in cached_results}

    assert cached_by_name["left_joined"]["cache_hit"] is True
    assert cached_by_name["right_joined"]["cache_hit"] is True
    assert (
        cached_by_name["left_joined"]["join_operation_hash"]
        == first_by_name["left_joined"]["join_operation_hash"]
    )


def test_build_from_file_rejects_injection_after_skipped_provider_post_translation(
    monkeypatch, tmp_path
):
    package_name = "dirty_guard_post_translation_demo_pkg"
    config_path = _write_dirty_guard_project(
        tmp_path,
        package_name=package_name,
        generator_lines=[
            "def make_provider():",
            "    return 'provider'",
            "def make_consumer(*, provider):",
            "    return f'consumer-on-{provider}'",
        ],
        resources={
            "provider": "make_provider",
            "consumer": "make_consumer",
        },
        config_lines=[
            "assemblies:",
            "  - name: frame",
            "  - name: provider",
            "  - name: consumer",
            "    inject_parts:",
            "      provider: provider.leader",
            "placement:",
            "  alignments:",
            "    - part: provider",
            "      to: frame",
            "      post_translation: [1, 2, 3]",
        ],
    )

    _install_string_builder_io(monkeypatch)
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.generators", raising=False)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(
            config_path,
            assembly_names=["consumer"],
            repository_dir=tmp_path / "repository",
        )

    message = str(excinfo.value)
    assert "placement-dirty assembly 'provider'" in message
    assert "into 'consumer'" in message
    assert "inject_parts.provider (leader)" in message
    assert "source_step=0" in message
    assert "post_translation=[1, 2, 3]" in message
    assert "inactive assemblies: frame" in message
    assert "dedicated template assembly" in message
    assert "global placement steps" in message


def test_build_from_file_rejects_injection_after_skipped_provider_post_rotation(
    monkeypatch, tmp_path
):
    package_name = "dirty_guard_post_rotation_demo_pkg"
    config_path = _write_dirty_guard_project(
        tmp_path,
        package_name=package_name,
        generator_lines=[
            "def make_provider():",
            "    return 'provider'",
            "def make_rail():",
            "    return 'rail'",
            "def make_consumer(*, provider):",
            "    return f'consumer-on-{provider}'",
        ],
        resources={
            "provider": "make_provider",
            "rail": "make_rail",
            "consumer": "make_consumer",
        },
        config_lines=[
            "assemblies:",
            "  - name: frame",
            "  - name: provider",
            "  - name: rail",
            "  - name: consumer",
            "    depends_on:",
            "      - rail",
            "    inject_parts:",
            "      provider: provider.leader",
            "placement:",
            "  alignments:",
            "    - part: provider",
            "      to: frame",
            "      post_rotation:",
            "        angle: 90",
            "        axis: [0, 0, 1]",
            "        center: provider.CENTER",
            "    - part: consumer",
            "      to: rail",
            "      alignment: CENTER",
        ],
    )

    _install_string_builder_io(monkeypatch)
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.generators", raising=False)

    with pytest.raises(builder.BuilderError) as excinfo:
        builder.build_from_file(
            config_path,
            assembly_names=["consumer"],
            repository_dir=tmp_path / "repository",
        )

    message = str(excinfo.value)
    assert "placement-dirty assembly 'provider'" in message
    assert "source_step=0" in message
    assert (
        "post_rotation={'angle': 90, 'axis': [0, 0, 1], 'center': 'provider.CENTER'}"
        in message
    )
    assert "inactive assemblies: frame" in message


def test_build_from_file_allows_injection_before_later_skipped_provider_post_transform(
    monkeypatch, tmp_path
):
    package_name = "dirty_guard_late_post_transform_demo_pkg"
    config_path = _write_dirty_guard_project(
        tmp_path,
        package_name=package_name,
        generator_lines=[
            "def make_provider():",
            "    return 'provider'",
            "def make_rail():",
            "    return 'rail'",
            "def make_consumer(*, provider):",
            "    return f'consumer-on-{provider}'",
        ],
        resources={
            "provider": "make_provider",
            "rail": "make_rail",
            "consumer": "make_consumer",
        },
        config_lines=[
            "assemblies:",
            "  - name: frame",
            "  - name: provider",
            "  - name: rail",
            "  - name: consumer",
            "    depends_on:",
            "      - rail",
            "    inject_parts:",
            "      provider: provider.leader",
            "placement:",
            "  alignments:",
            "    - part: consumer",
            "      to: rail",
            "      alignment: CENTER",
            "    - part: provider",
            "      to: frame",
            "      post_rotation:",
            "        angle: 90",
            "        axis: [0, 0, 1]",
            "        center: provider.CENTER",
        ],
    )

    _install_string_builder_io(monkeypatch)
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "consumer-on-provider": (10.0, 0.0, 0.0),
            "consumer-on-provider|CENTER": (0.0, 0.0, 0.0),
            "rail": (0.0, 0.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|CENTER"
        ),
    )
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.generators", raising=False)

    results = builder.build_from_file(
        config_path,
        assembly_names=["consumer"],
        repository_dir=tmp_path / "repository",
    )

    consumer_result = next(
        result for result in results if result["assembly_name"] == "consumer"
    )
    assert (
        Path(consumer_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "consumer-on-provider"
    )
    assert "source_placement_transforms" not in consumer_result["dependencies"][0]


def test_build_from_file_allows_injection_after_active_provider_post_translation(
    monkeypatch, tmp_path
):
    package_name = "dirty_guard_active_post_translation_demo_pkg"
    config_path = _write_dirty_guard_project(
        tmp_path,
        package_name=package_name,
        generator_lines=[
            "def make_provider():",
            "    return 'provider'",
            "def make_consumer(*, provider):",
            "    return f'consumer-on-{provider}'",
        ],
        resources={
            "provider": "make_provider",
            "consumer": "make_consumer",
        },
        config_lines=[
            "assemblies:",
            "  - name: provider",
            "  - name: consumer",
            "    inject_parts:",
            "      provider: provider.leader",
            "placement:",
            "  alignments:",
            "    - part: provider",
            "      post_translation: [1, 2, 3]",
        ],
    )

    captured_post_translations = []
    _install_string_builder_io(monkeypatch)
    monkeypatch.setattr(
        builder,
        "_part_center",
        lambda part: {
            "provider": (0.0, 0.0, 0.0),
            "provider|POST": (1.0, 2.0, 3.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_post_translation",
        lambda delta: captured_post_translations.append(delta)
        or (lambda part: f"{part}|POST"),
    )
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.generators", raising=False)

    results = builder.build_from_file(
        config_path,
        assembly_names=["consumer"],
        repository_dir=tmp_path / "repository",
    )

    consumer_result = next(
        result for result in results if result["assembly_name"] == "consumer"
    )
    assert captured_post_translations == [(1.0, 2.0, 3.0)]
    assert (
        Path(consumer_result["artifacts"]["leader_step"]).read_text(encoding="utf-8")
        == "consumer-on-provider|POST"
    )
    assert consumer_result["dependencies"][0]["source_placement_offset"] == [
        1.0,
        2.0,
        3.0,
    ]
    assert consumer_result["dependencies"][0]["source_placement_transforms"] == [
        {
            "kind": "translate",
            "vector": [1.0, 2.0, 3.0],
            "placement_step": 0,
        }
    ]


def test_advance_placement_execution_stops_at_missing_prefix_even_when_later_step_is_ready(
    monkeypatch, tmp_path
):
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
        "assemblies": [
            {"name": "profile_a"},
            {"name": "profile_b"},
            {"name": "profile_c"},
        ],
        "placement": {
            "alignments": [
                {
                    "part": "profile_c",
                    "to": "profile_a",
                    "alignment": "CENTER",
                },
                {
                    "part": "profile_b",
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
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 0
    assert placement_state.executed_alignment_indices == set()
    assert placement_state.placement_offsets == {}


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
            "profile-b|ALIGN": (0.0, 0.0, 0.0),
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
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert captured_post_translations == [(0.0, 5.0, 0.0)]
    assert placement_state.cursor == 1
    assert placement_state.placement_offsets == {"profile_b": (-10.0, 5.0, 0.0)}
    assert (
        placement_state.build_pool.slots["profile_b"].runtime_anchor_cache["leader"]
        == "profile-b|ALIGN|POST"
    )
    assert builder._apply_translation_sequence(
        "profile-b", placement_state.translation_history["profile_b"]
    ) == ("profile-b|ALIGN|POST")


def test_apply_build_pool_runtime_placement_uses_final_slot_state():
    build_pool = builder._BuildArtifactPool(["frame", "carriage"])
    build_pool.insert_available(
        "frame",
        {"assembly_name": "frame", "artifacts": {}},
        source="built",
    )
    build_pool.insert_available(
        "carriage",
        {"assembly_name": "carriage", "artifacts": {}},
        source="built",
    )
    build_pool.apply_runtime_transforms(
        "carriage",
        [lambda part: f"{part}|placed"],
        [{"kind": "translate", "vector": [1.0, 2.0, 3.0], "placement_step": 4}],
        (1.0, 2.0, 3.0),
    )
    scene_parts = [
        {
            "name": "frame",
            "part": "frame-part",
            "assembly_name": "frame",
            "transform_history": [],
        },
        {
            "name": "carriage",
            "part": "carriage-part",
            "assembly_name": "carriage",
            "transform_history": [],
        },
    ]

    result = builder._apply_build_pool_runtime_placement(scene_parts, build_pool)

    assert result.scene_parts[0]["part"] == "frame-part"
    assert result.scene_parts[1]["part"] == "carriage-part|placed"
    assert result.scene_parts[1]["transform_history"] == [
        {"kind": "translate", "vector": [1.0, 2.0, 3.0], "placement_step": 4}
    ]
    assert result.assembly_transforms["carriage"][0]("x") == "x|placed"


def test_advance_placement_execution_supports_post_rotation_without_alignment(
    monkeypatch, tmp_path
):
    moving_leader = tmp_path / "moving.step"
    moving_leader.write_text("moving-leader", encoding="utf-8")
    target_anchor = tmp_path / "target.step"
    target_anchor.write_text("target-anchor", encoding="utf-8")

    built_results_by_name = {
        "moving": {
            "assembly_name": "moving",
            "artifacts": {"leader_step": str(moving_leader)},
        },
        "target": {
            "assembly_name": "target",
            "artifacts": {
                "leader_step": str(target_anchor),
                "non_production_parts": [
                    {"name": "anchor", "path": str(target_anchor)}
                ],
            },
        },
    }
    config_data = {
        "assemblies": [{"name": "moving"}, {"name": "target"}],
        "placement": {
            "alignments": [
                {
                    "part": "moving",
                    "post_rotation": {
                        "angle": 90,
                        "center": "target.non_production_parts.anchor.CENTER",
                    },
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
            "moving-leader": (1.0, 2.0, 3.0),
            "target-anchor": (10.0, 20.0, 30.0),
            "moving-leader|ROTATE": (4.0, 6.0, 8.0),
        }[part],
    )
    captured_rotations = []
    monkeypatch.setattr(
        builder,
        "_make_post_rotation",
        lambda angle, *, axis=None, center=None: captured_rotations.append(
            (angle, axis, center)
        )
        or (lambda part: f"{part}|ROTATE"),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert captured_rotations == [(90.0, (0.0, 0.0, 1.0), (10.0, 20.0, 30.0))]
    assert placement_state.cursor == 1
    assert placement_state.placement_offsets == {"moving": (3.0, 4.0, 5.0)}
    assert (
        builder._apply_translation_sequence(
            "moving-leader", placement_state.translation_history["moving"]
        )
        == "moving-leader|ROTATE"
    )
    assert placement_state.transform_records["moving"] == [
        {
            "kind": "rotate",
            "angle": 90.0,
            "axis": [0.0, 0.0, 1.0],
            "center": [10.0, 20.0, 30.0],
            "placement_step": 0,
        }
    ]


def test_advance_placement_execution_rigid_groups_move_transitively(
    monkeypatch, tmp_path
):
    leaders = {}
    for name, label in {
        "a": "a-leader",
        "b": "b-leader",
        "c": "c-leader",
        "frame": "frame-leader",
    }.items():
        step_path = tmp_path / f"{name}.step"
        step_path.write_text(label, encoding="utf-8")
        leaders[name] = step_path

    built_results_by_name = {
        name: {
            "assembly_name": name,
            "artifacts": {"leader_step": str(path)},
        }
        for name, path in leaders.items()
    }
    config_data = {
        "assemblies": [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
            {"name": "frame"},
        ],
        "placement": {
            "alignments": [
                {
                    "to": "b",
                    "rigid_group": ["a"],
                },
                {
                    "to": "c",
                    "rigid_group": ["b"],
                },
                {
                    "part": "c",
                    "to": "frame",
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

    base_centers = {
        "a-leader": (0.0, 0.0, 0.0),
        "b-leader": (10.0, 0.0, 0.0),
        "c-leader": (20.0, 0.0, 0.0),
        "frame-leader": (30.0, 0.0, 0.0),
    }

    def fake_center(part):
        move_count = part.count("|MOVE")
        root = part.split("|", 1)[0]
        base_x, base_y, base_z = base_centers[root]
        return (base_x + move_count, base_y, base_z)

    monkeypatch.setattr(builder, "_part_center", fake_center)
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|MOVE"
        ),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 3
    assert placement_state.placement_offsets == {
        "a": (1.0, 0.0, 0.0),
        "b": (1.0, 0.0, 0.0),
        "c": (1.0, 0.0, 0.0),
    }
    assert (
        builder._apply_translation_sequence(
            "a-leader", placement_state.translation_history["a"]
        )
        == "a-leader|MOVE"
    )
    assert (
        builder._apply_translation_sequence(
            "b-leader", placement_state.translation_history["b"]
        )
        == "b-leader|MOVE"
    )
    assert (
        builder._apply_translation_sequence(
            "c-leader", placement_state.translation_history["c"]
        )
        == "c-leader|MOVE"
    )


def test_advance_placement_execution_supports_rigid_group_without_alignment(
    monkeypatch, tmp_path
):
    extruder_leader = tmp_path / "extruder.step"
    extruder_leader.write_text("extruder-leader", encoding="utf-8")
    x_axis_leader = tmp_path / "x_axis.step"
    x_axis_leader.write_text("x-axis-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    built_results_by_name = {
        "extruder": {
            "assembly_name": "extruder",
            "artifacts": {"leader_step": str(extruder_leader)},
        },
        "x_axis": {
            "assembly_name": "x_axis",
            "artifacts": {"leader_step": str(x_axis_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }
    config_data = {
        "assemblies": [
            {"name": "extruder"},
            {"name": "x_axis"},
            {"name": "frame"},
        ],
        "placement": {
            "alignments": [
                {
                    "to": "x_axis",
                    "rigid_group": ["extruder"],
                },
                {
                    "part": "x_axis",
                    "to": "frame",
                    "alignment": "STACK_TOP",
                    "stack_gap": 10,
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
            "extruder-leader": (1.0, 2.0, 3.0),
            "x-axis-leader": (10.0, 20.0, 30.0),
            "frame-leader": (40.0, 50.0, 60.0),
            "x-axis-leader|STACK": (10.0, 20.0, 70.0),
            "extruder-leader|STACK": (1.0, 2.0, 43.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|STACK"
        ),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 2
    assert placement_state.executed_alignment_indices == {0, 1}
    assert placement_state.placement_offsets == {
        "extruder": (0.0, 0.0, 40.0),
        "x_axis": (0.0, 0.0, 40.0),
    }
    assert (
        builder._apply_translation_sequence(
            "extruder-leader", placement_state.translation_history["extruder"]
        )
        == "extruder-leader|STACK"
    )
    assert (
        builder._apply_translation_sequence(
            "x-axis-leader", placement_state.translation_history["x_axis"]
        )
        == "x-axis-leader|STACK"
    )
    assert placement_state.transform_records["extruder"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]
    assert placement_state.transform_records["x_axis"] == [
        {"kind": "translate", "vector": [0.0, 0.0, 40.0], "placement_step": 1}
    ]


def test_advance_placement_execution_rigid_group_waits_for_all_members(
    monkeypatch, tmp_path
):
    leaders = {}
    for name, label in {
        "a": "a-leader",
        "b": "b-leader",
        "c": "c-leader",
        "frame": "frame-leader",
    }.items():
        step_path = tmp_path / f"{name}.step"
        step_path.write_text(label, encoding="utf-8")
        leaders[name] = step_path

    built_results_by_name = {
        "a": {
            "assembly_name": "a",
            "artifacts": {"leader_step": str(leaders["a"])},
        },
        "b": {
            "assembly_name": "b",
            "artifacts": {"leader_step": str(leaders["b"])},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(leaders["frame"])},
        },
    }
    config_data = {
        "assemblies": [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
            {"name": "frame"},
        ],
        "placement": {
            "alignments": [
                {
                    "to": "a",
                    "rigid_group": ["b", "c"],
                },
                {
                    "part": "a",
                    "to": "frame",
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
            "a-leader": (0.0, 0.0, 0.0),
            "b-leader": (10.0, 0.0, 0.0),
            "c-leader": (20.0, 0.0, 0.0),
            "frame-leader": (30.0, 0.0, 0.0),
            "a-leader|MOVE": (10.0, 0.0, 0.0),
            "b-leader|MOVE": (20.0, 0.0, 0.0),
            "c-leader|MOVE": (30.0, 0.0, 0.0),
        }[part],
    )
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|MOVE"
        ),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data,
        built_results_by_name,
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 0
    assert placement_state.executed_alignment_indices == set()
    assert placement_state.placement_offsets == {}

    built_results_by_name["c"] = {
        "assembly_name": "c",
        "artifacts": {"leader_step": str(leaders["c"])},
    }
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 2
    assert placement_state.executed_alignment_indices == {0, 1}
    assert placement_state.placement_offsets == {
        "a": (10.0, 0.0, 0.0),
        "b": (10.0, 0.0, 0.0),
        "c": (10.0, 0.0, 0.0),
    }


def test_advance_placement_execution_ignores_missing_rigid_predecessors_outside_active_subset(
    monkeypatch, tmp_path
):
    left_leader = tmp_path / "left.step"
    left_leader.write_text("left-leader", encoding="utf-8")
    right_leader = tmp_path / "right.step"
    right_leader.write_text("right-leader", encoding="utf-8")
    rail_leader = tmp_path / "rail.step"
    rail_leader.write_text("rail-leader", encoding="utf-8")
    frame_leader = tmp_path / "frame.step"
    frame_leader.write_text("frame-leader", encoding="utf-8")

    built_results_by_name = {
        "right": {
            "assembly_name": "right",
            "artifacts": {"leader_step": str(right_leader)},
        },
        "rail": {
            "assembly_name": "rail",
            "artifacts": {"leader_step": str(rail_leader)},
        },
        "frame": {
            "assembly_name": "frame",
            "artifacts": {"leader_step": str(frame_leader)},
        },
    }
    config_data = {
        "assemblies": [
            {"name": "left"},
            {"name": "right"},
            {"name": "rail"},
            {"name": "frame"},
        ],
        "placement": {
            "alignments": [
                {
                    "part": "left",
                    "to": "rail",
                    "alignment": "CENTER",
                },
                {
                    "to": "rail",
                    "rigid_group": ["left"],
                },
                {
                    "part": "right",
                    "to": "rail",
                    "alignment": "CENTER",
                },
                {
                    "to": "rail",
                    "rigid_group": ["right"],
                },
                {
                    "part": "rail",
                    "to": "frame",
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

    base_centers = {
        "left-leader": (0.0, 0.0, 0.0),
        "right-leader": (10.0, 0.0, 0.0),
        "rail-leader": (20.0, 0.0, 0.0),
        "frame-leader": (30.0, 0.0, 0.0),
    }

    def fake_center(part):
        move_count = part.count("|MOVE")
        root = part.split("|", 1)[0]
        base_x, base_y, base_z = base_centers[root]
        return (base_x + 10.0 * move_count, base_y, base_z)

    monkeypatch.setattr(builder, "_part_center", fake_center)
    monkeypatch.setattr(
        builder,
        "_make_placement_translation",
        lambda moving_anchor, target_anchor, *, alignment, axes=None, stack_gap=0: (
            lambda part: f"{part}|MOVE"
        ),
    )
    captured_logs = []
    monkeypatch.setattr(
        builder._logger,
        "info",
        lambda message, *args: captured_logs.append(message % args),
    )

    placement_state = builder._initialize_placement_execution_state(
        config_data,
        built_results_by_name,
        active_assembly_names={"right", "rail", "frame"},
    )
    placement_state = builder._advance_placement_execution(
        placement_state,
        built_results_by_name=built_results_by_name,
        repository_dir=tmp_path,
    )

    assert placement_state.cursor == 3
    assert placement_state.executed_alignment_indices == {2, 3, 4}
    assert placement_state.placement_offsets == {
        "right": (20.0, 0.0, 0.0),
        "rail": (10.0, 0.0, 0.0),
    }
    assert (
        builder._apply_translation_sequence(
            "right-leader", placement_state.translation_history["right"]
        )
        == "right-leader|MOVE|MOVE"
    )
    assert (
        builder._apply_translation_sequence(
            "rail-leader", placement_state.translation_history["rail"]
        )
        == "rail-leader|MOVE"
    )
    assert any(
        "Placement cursor advance: active_cursor=0/3; active_source_steps=2-4; skipped_source_steps=2 outside active subset; executed_source_steps=<none>; built_assemblies=['frame', 'rail', 'right']"
        in message
        for message in captured_logs
    )
    assert any(
        "Placement cursor execute: active_step=0/3; source_step=2; right aligned to rail via CENTER"
        in message
        for message in captured_logs
    )
    assert any(
        "Placement cursor applied: source_step=2; assemblies=['right']" in message
        for message in captured_logs
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
            "source_placement_transforms": [
                {
                    "kind": "translate",
                    "vector": [-10.0, 0.0, 0.0],
                    "placement_step": 0,
                }
            ],
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
        "prod_gap": 4.0,
        "bed_width": 220,
        "max_build_height": None,
        "export_step": True,
        "export_obj": False,
        "export_individual_parts": True,
        "export_stl": True,
        "plates": None,
        "auto_assign_plates": False,
        "plate_scene_gap": 20.0,
        "visualize_plate_boundaries": True,
    }


def test_resolve_export_options_uses_lightweight_visualization_defaults():
    resolved = builder._resolve_export_options({}, "visualization", None)

    assert resolved == {
        "prod_gap": 4.0,
        "bed_width": 200.0,
        "max_build_height": None,
        "export_step": False,
        "export_obj": True,
        "export_individual_parts": False,
        "export_stl": False,
        "plates": None,
        "auto_assign_plates": False,
        "plate_scene_gap": 20.0,
        "visualize_plate_boundaries": True,
    }


def test_resolve_export_options_supports_plate_configuration():
    resolved = builder._resolve_export_options(
        {
            "Builder": {
                "Production": {
                    "arrange": {
                        "auto_assign_plates": True,
                        "plates": [
                            {"name": "plate_a", "parts": ["frame"]},
                            {"name": "plate_b", "parts": ["feet"]},
                        ],
                    }
                }
            }
        },
        "production",
        None,
    )

    assert resolved["auto_assign_plates"] is True
    assert resolved["plates"] == [
        {"name": "plate_a", "parts": ["frame"]},
        {"name": "plate_b", "parts": ["feet"]},
    ]


def test_apply_prototype_arrange_overrides_merges_prototype_only_settings():
    resolved = builder._apply_prototype_arrange_overrides(
        {
            "prod_gap": 4.0,
            "bed_width": 220,
            "max_build_height": None,
            "export_step": True,
            "export_obj": True,
            "export_individual_parts": False,
            "export_stl": True,
            "plates": [
                {"name": "front_left", "parts": ["front_left_uc"]},
                {"name": "front_right", "parts": ["front_right_uc"]},
            ],
            "auto_assign_plates": False,
        },
        selected_metadata={
            "public_parameters": {"holder_margin": 6},
            "generator_kwargs": {},
            "generator_context": {},
        },
        selected_resource_data={
            "Builder": {
                "Production": {
                    "prototype": {
                        "arrange": {
                            "bed_width": {"$expr": {"$sub": "200 + ${holder_margin}"}},
                            "plates": [
                                {
                                    "name": "prototype_pair",
                                    "parts": ["front_left_uc", "front_right_uc"],
                                }
                            ],
                        }
                    }
                }
            }
        },
        config_data=None,
    )

    assert resolved == {
        "prod_gap": 4.0,
        "bed_width": 206,
        "max_build_height": None,
        "export_step": True,
        "export_obj": True,
        "export_individual_parts": False,
        "export_stl": True,
        "plates": [
            {
                "name": "prototype_pair",
                "parts": ["front_left_uc", "front_right_uc"],
            }
        ],
        "auto_assign_plates": False,
    }


def test_resolve_prototype_reference_supports_self_and_relative_selectors():
    assert (
        builder._resolve_prototype_reference(
            "self.non_production_parts.mount_tower_left_front",
            "print_bed_undercarriage_assembly",
        )
        == "print_bed_undercarriage_assembly.non_production_parts.mount_tower_left_front"
    )
    assert (
        builder._resolve_prototype_reference(
            "followers.front_left_uc",
            "print_bed_undercarriage_assembly",
        )
        == "print_bed_undercarriage_assembly.followers.front_left_uc"
    )
    assert (
        builder._resolve_prototype_reference(
            "print_bed_assembly.leader",
            "print_bed_undercarriage_assembly",
        )
        == "print_bed_assembly.leader"
    )


def test_resolve_prototype_anchor_error_lists_valid_and_cross_assembly_refs(
    monkeypatch, tmp_path
):
    metadata_by_name = {
        "print_bed_undercarriage_assembly": {
            "assembly_name": "print_bed_undercarriage_assembly",
            "artifacts": {
                "leader_step": str(tmp_path / "pbuc_leader.step"),
                "fused_step": None,
                "followers": [],
                "cutters": [],
                "non_production_parts": [
                    {
                        "index": 0,
                        "name": "mount_tower_left_front",
                        "path": str(tmp_path / "mount_tower_left_front.step"),
                    }
                ],
            },
        },
        "print_bed_assembly": {
            "assembly_name": "print_bed_assembly",
            "artifacts": {
                "leader_step": str(tmp_path / "pb_leader.step"),
                "fused_step": None,
                "followers": [],
                "cutters": [],
                "non_production_parts": [
                    {
                        "index": 0,
                        "name": "print_bed_undercarriage_belt_clamp_torsion_screw_back",
                        "path": str(tmp_path / "torsion_screw_back.step"),
                    }
                ],
            },
        },
    }

    monkeypatch.setattr(
        builder,
        "_dependency_metadata_for_assembly",
        lambda assembly_name, built_results_by_name, repository_dir: metadata_by_name[
            assembly_name
        ],
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder._resolve_prototype_anchor_part(
            "self.non_production_parts.print_bed_undercarriage_belt_clamp_torsion_screw_back",
            selected_assembly="print_bed_undercarriage_assembly",
            built_results_by_name=metadata_by_name,
            repository_dir=tmp_path,
            placement_state=None,
            config_data={
                "assemblies": [
                    {"name": "print_bed_undercarriage_assembly"},
                    {"name": "print_bed_assembly"},
                ]
            },
        )

    message = str(excinfo.value)
    assert "matched no artifacts" in message
    assert (
        "resolved to 'print_bed_undercarriage_assembly.non_production_parts."
        "print_bed_undercarriage_belt_clamp_torsion_screw_back'"
    ) in message
    assert "self.non_production_parts.mount_tower_left_front" in message
    assert (
        "print_bed_assembly.non_production_parts."
        "print_bed_undercarriage_belt_clamp_torsion_screw_back"
    ) in message


def test_resolve_placement_anchor_error_lists_valid_references(monkeypatch, tmp_path):
    metadata_by_name = {
        "print_bed_undercarriage_assembly": {
            "assembly_name": "print_bed_undercarriage_assembly",
            "artifacts": {
                "leader_step": str(tmp_path / "pbuc_leader.step"),
                "fused_step": None,
                "followers": [],
                "cutters": [],
                "non_production_parts": [
                    {
                        "index": 0,
                        "name": "mount_tower_left_front",
                        "path": str(tmp_path / "mount_tower_left_front.step"),
                    }
                ],
            },
        },
        "print_bed_assembly": {
            "assembly_name": "print_bed_assembly",
            "artifacts": {
                "leader_step": str(tmp_path / "pb_leader.step"),
                "fused_step": None,
                "followers": [],
                "cutters": [],
                "non_production_parts": [
                    {
                        "index": 0,
                        "name": "print_bed_undercarriage_belt_clamp_torsion_screw_back",
                        "path": str(tmp_path / "torsion_screw_back.step"),
                    }
                ],
            },
        },
    }

    monkeypatch.setattr(
        builder,
        "_dependency_metadata_for_assembly",
        lambda assembly_name, built_results_by_name, repository_dir: metadata_by_name[
            assembly_name
        ],
    )

    placement_state = builder._PlacementExecutionState(
        alignments=[],
        known_assembly_names=[
            "print_bed_assembly",
            "print_bed_undercarriage_assembly",
        ],
        placement_context={},
    )

    with pytest.raises(builder.BuilderError) as excinfo:
        builder._resolve_placement_anchor(
            "print_bed_undercarriage_assembly.non_production_parts.print_bed_undercarriage_belt_clamp_torsion_screw_back",
            placement_state=placement_state,
            built_results_by_name=metadata_by_name,
            repository_dir=tmp_path,
        )

    message = str(excinfo.value)
    assert "Placement reference" in message
    assert "matched no artifacts" in message
    assert (
        "print_bed_undercarriage_assembly.non_production_parts.mount_tower_left_front"
        in message
    )
    assert (
        "print_bed_assembly.non_production_parts."
        "print_bed_undercarriage_belt_clamp_torsion_screw_back"
    ) in message


def test_apply_prototype_configuration_filters_and_clips_parts(monkeypatch, tmp_path):
    import shellforgepy.geometry.keepouts as keepouts
    import shellforgepy.simple as simple

    metadata = {
        "assembly_name": "print_bed_undercarriage_assembly",
        "public_parameters": {"holder_margin": 4},
        "generator_kwargs": {},
        "generator_context": {"BIG_THING": 500},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "prototype": {
                    "include_parts": ["front_left_uc", "front_right_uc"],
                    "exclude_parts": ["front_right_uc"],
                    "box_cutters": [
                        {
                            "part": "front_left_uc",
                            "around": "self.non_production_parts.mount_tower_left_front",
                            "size": [
                                34,
                                {"$expr": {"$sub": "30 + ${holder_margin}"}},
                                70,
                            ],
                            "offset": [1, 2, 3],
                        }
                    ],
                }
            }
        }
    }
    scene_parts = [
        {"name": "front_left_uc", "part": "front-left", "transform_history": []},
        {"name": "front_right_uc", "part": "front-right", "transform_history": []},
        {"name": "belt_clamp_clamp_front", "part": "clamp", "transform_history": []},
    ]

    class FakeKeepVolume:
        def __init__(self, size, cutter_size):
            self.size = size
            self.cutter_size = cutter_size

        def use_as_cutter_on(self, part):
            return f"clipped:{part}:{self.size}:{self.cutter_size}"

    captured = {}

    def fake_create_box_hole_cutter(x, y, z, *, cutter_size):
        captured["size"] = (x, y, z)
        captured["cutter_size"] = cutter_size
        return FakeKeepVolume((x, y, z), cutter_size)

    def fake_align(part, anchor, alignment):
        captured["anchor"] = anchor
        captured["alignment"] = alignment
        return part

    def fake_translate(x, y, z):
        captured["offset"] = (x, y, z)
        return lambda part: part

    monkeypatch.setattr(keepouts, "create_box_hole_cutter", fake_create_box_hole_cutter)
    monkeypatch.setattr(simple, "align", fake_align)
    monkeypatch.setattr(simple, "translate", fake_translate)
    monkeypatch.setattr(
        builder,
        "_resolve_prototype_anchor_part",
        lambda *args, **kwargs: "anchor-part",
    )

    prototype_parts = builder._apply_prototype_configuration(
        scene_parts,
        selected_metadata=metadata,
        selected_resource_data=resource_data,
        selected_assembly="print_bed_undercarriage_assembly",
        built_results_by_name={"print_bed_undercarriage_assembly": metadata},
        repository_dir=tmp_path,
        placement_state=None,
        config_data=None,
    )

    assert [part["name"] for part in prototype_parts] == ["front_left_uc"]
    assert prototype_parts[0]["part"] == "clipped:front-left:(34.0, 34.0, 70.0):500.0"
    assert prototype_parts[0]["transform_history"][-1] == {
        "kind": "prototype_box_cutter",
        "part": "front_left_uc",
        "around": "self.non_production_parts.mount_tower_left_front",
        "size": [34.0, 34.0, 70.0],
        "cutter_size": 500.0,
        "offset": [1.0, 2.0, 3.0],
    }
    assert captured == {
        "size": (34.0, 34.0, 70.0),
        "cutter_size": 500.0,
        "anchor": "anchor-part",
        "alignment": simple.Alignment.CENTER,
        "offset": (1.0, 2.0, 3.0),
    }


def test_apply_prototype_configuration_orients_anchor_like_target(
    monkeypatch, tmp_path
):
    import shellforgepy.geometry.keepouts as keepouts
    import shellforgepy.simple as simple

    metadata = {
        "assembly_name": "sample_assembly",
        "public_parameters": {},
        "generator_kwargs": {},
        "generator_context": {"BIG_THING": 500},
    }
    resource_data = {
        "Builder": {
            "Production": {
                "prototype": {
                    "include_parts": ["hinge"],
                    "box_cutters": [
                        {
                            "part": "hinge",
                            "around": "self.non_production_parts.hinge_pin",
                            "size": [10, 20, 30],
                        }
                    ],
                }
            }
        }
    }
    scene_parts = [
        {
            "name": "hinge",
            "part": "hinge-local",
            "transform_history": [],
            "prod_rotation_angle": 90,
            "prod_rotation_axis": [0, 1, 0],
        }
    ]

    class FakeKeepVolume:
        def use_as_cutter_on(self, part):
            return f"clipped:{part}"

    captured = {}

    def fake_rotate_part(part, *, angle, axis):
        return f"rotated({part},{angle},{tuple(axis)})"

    def fake_align(part, anchor, alignment):
        captured["anchor"] = anchor
        captured["alignment"] = alignment
        return part

    monkeypatch.setattr(arrange_and_export_module, "rotate_part", fake_rotate_part)
    monkeypatch.setattr(
        keepouts,
        "create_box_hole_cutter",
        lambda *args, **kwargs: FakeKeepVolume(),
    )
    monkeypatch.setattr(simple, "align", fake_align)
    monkeypatch.setattr(
        builder,
        "_resolve_prototype_anchor_part",
        lambda *args, **kwargs: "anchor-local",
    )

    prototype_parts = builder._apply_prototype_configuration(
        scene_parts,
        selected_metadata=metadata,
        selected_resource_data=resource_data,
        selected_assembly="sample_assembly",
        built_results_by_name={"sample_assembly": metadata},
        repository_dir=tmp_path,
        placement_state=None,
        config_data=None,
    )

    assert prototype_parts[0]["part"] == "clipped:rotated(hinge-local,90,(0, 1, 0))"
    assert captured["anchor"] == "rotated(anchor-local,90,(0, 1, 0))"
    assert captured["alignment"] == simple.Alignment.CENTER
