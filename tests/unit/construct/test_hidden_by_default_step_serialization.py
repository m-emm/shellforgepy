import json

import pytest

pytest.importorskip("cadquery")

from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.part_parameters import PartParameters
from shellforgepy.construct.step_serialization import (
    deserialize_to_leader_followers_cutters_part,
    serialize_to_step,
    step_cached,
)
from shellforgepy.simple import create_box


def _hidden_reference_assembly():
    return LeaderFollowersCuttersPart(
        create_box(3, 3, 3),
        non_production_parts=[create_box(1, 1, 1)],
        non_production_names=["datum"],
        hidden_by_default_names=["datum"],
    )


def test_step_round_trip_preserves_hidden_by_default_names(tmp_path):
    step_path = tmp_path / "hidden-reference.step"

    serialize_to_step(_hidden_reference_assembly(), step_path)
    restored = deserialize_to_leader_followers_cutters_part(step_path)
    metadata = json.loads(
        (tmp_path / "hidden-reference.lfcp.json").read_text(encoding="utf-8")
    )

    assert metadata["version"] == 4
    assert metadata["hidden_by_default_names"] == ["datum"]
    assert restored.hidden_by_default_names == ["datum"]


@pytest.mark.parametrize("legacy_version", [2, 3])
def test_legacy_step_metadata_defaults_to_no_hidden_parts(tmp_path, legacy_version):
    step_path = tmp_path / "legacy.step"
    metadata_path = tmp_path / "legacy.lfcp.json"
    serialize_to_step(_hidden_reference_assembly(), step_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = legacy_version
    metadata.pop("hidden_by_default_names")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    restored = deserialize_to_leader_followers_cutters_part(step_path)

    assert restored.hidden_by_default_names == []


def test_step_cached_preserves_hidden_by_default_names(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"count": 0}

    @step_cached
    def build(parameters):
        calls["count"] += 1
        return _hidden_reference_assembly()

    parameters = PartParameters({"size": 3.0})
    assert build(parameters).hidden_by_default_names == ["datum"]
    assert build(parameters).hidden_by_default_names == ["datum"]
    assert calls["count"] == 1
