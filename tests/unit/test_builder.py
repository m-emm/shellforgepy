import argparse
import json
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
                "    return FakeComposite(f\"{width}-{height}-{label}\")",
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
        lambda path: pytest.fail(f"generator should not be reloaded for cache hit: {path}"),
    )

    cached_results = builder.build_from_file(config_path)
    assert cached_results[0]["cache_hit"] is True
    assert cached_results[0]["parameter_hash"] == expected_hash


def test_run_builder_invokes_build_from_file(monkeypatch, tmp_path):
    captured = {}

    def fake_build_from_file(config_file, *, assembly_names=None, repository_dir=None, force=False):
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
    config_path.write_text("assemblies: []\n", encoding="utf-8")

    result = builder.run_builder(
        argparse.Namespace(
            config_file=str(config_path),
            assembly=["frame"],
            repository_dir=str(tmp_path / "repo"),
            force=True,
        )
    )

    assert result == 0
    assert captured == {
        "config_file": str(config_path),
        "assembly": ["frame"],
        "repository_dir": str(tmp_path / "repo"),
        "force": True,
    }
