from pathlib import Path

from shellforgepy.construct import step_serialization
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.part_parameters import PartParameters


def test_part_parameters_hash_consistency():
    """Test that PartParameters generates consistent hashes for the same parameters."""
    params1 = PartParameters(
        {
            "length": 10.0,
            "width": 5,
            "name": "test_part",
            "enabled": True,
        }
    )

    params2 = PartParameters(
        {
            "length": 10.0,
            "width": 5,
            "name": "test_part",
            "enabled": True,
        }
    )

    assert params1.parameters_hash() == params2.parameters_hash()


def test_part_parameters_attrs():
    """Test that PartParameters generates consistent hashes for the same parameters."""
    params1 = PartParameters(
        {
            "length": 10.0,
            "width": 5,
            "name": "test_part",
            "enabled": True,
        }
    )

    assert params1.length == 10.0
    assert params1.width == 5
    assert params1.name == "test_part"
    assert params1.enabled is True


def test_step_cached_uses_cache(tmp_path, monkeypatch):
    params = PartParameters({"length": 10.0})
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(cache_dir))

    calls = {"func": 0, "serialize": 0, "deserialize": 0}

    def fake_serialize(part, path):
        calls["serialize"] += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("cached")

    def fake_deserialize(path):
        calls["deserialize"] += 1
        return "cached-part"

    monkeypatch.setattr(step_serialization, "serialize_to_step", fake_serialize)
    monkeypatch.setattr(
        step_serialization,
        "deserialize_to_leader_followers_cutters_part",
        fake_deserialize,
    )

    @step_serialization.step_cached
    def create_part(parameters):
        calls["func"] += 1
        return "fresh-part"

    assert create_part(params) == "fresh-part"
    assert calls["func"] == 1
    assert calls["serialize"] == 1

    assert create_part(params) == "cached-part"
    assert calls["func"] == 1
    assert calls["deserialize"] == 1


def test_step_cached_plain_part_round_trip(tmp_path, monkeypatch):
    params = PartParameters({"length": 10.0})
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(cache_dir))

    calls = {"serialize": 0}

    def fake_serialize(part, path):
        calls["serialize"] += 1
        assert isinstance(part, LeaderFollowersCuttersPart)
        assert part.additional_data.get("_step_cache_plain_part") is True
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("cached")

    class FakeRestored:
        def __init__(self):
            self.additional_data = {"_step_cache_plain_part": True}
            self.leader = "cached-plain"

    monkeypatch.setattr(step_serialization, "serialize_to_step", fake_serialize)
    monkeypatch.setattr(
        step_serialization,
        "deserialize_to_leader_followers_cutters_part",
        lambda path: FakeRestored(),
    )

    @step_serialization.step_cached
    def create_part(parameters):
        return "fresh-plain"

    assert create_part(params) == "fresh-plain"
    assert calls["serialize"] == 1
    assert create_part(params) == "cached-plain"


def test_step_cached_lfcp_round_trip(tmp_path, monkeypatch):
    params = PartParameters({"length": 10.0})
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(cache_dir))

    def fake_serialize(part, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("cached")

    class FakeRestored:
        def __init__(self):
            self.additional_data = {}
            self.leader = "leader"

    restored = FakeRestored()

    monkeypatch.setattr(step_serialization, "serialize_to_step", fake_serialize)
    monkeypatch.setattr(
        step_serialization,
        "deserialize_to_leader_followers_cutters_part",
        lambda path: restored,
    )

    @step_serialization.step_cached
    def create_part(parameters):
        return LeaderFollowersCuttersPart(leader="leader")

    assert create_part(params).leader == "leader"
    assert create_part(params) is restored


def test_step_cached_includes_source_hash(tmp_path, monkeypatch):
    params = PartParameters({"length": 10.0})
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(cache_dir))

    def fake_serialize(part, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("cached")

    monkeypatch.setattr(step_serialization, "serialize_to_step", fake_serialize)
    monkeypatch.setattr(
        step_serialization,
        "_get_function_source_hash",
        lambda func: "hash_a" if func.__name__ == "create_a" else "hash_b",
    )

    @step_serialization.step_cached(include_source_hash=True)
    def create_a(parameters):
        return "part-a"

    @step_serialization.step_cached(include_source_hash=True)
    def create_b(parameters):
        return "part-b"

    create_a(params)
    create_b(params)

    base_hash = params.parameters_hash()
    assert (cache_dir / f"{base_hash}-hash_a.step").is_file()
    assert (cache_dir / f"{base_hash}-hash_b.step").is_file()


def test_step_cached_no_env(monkeypatch):
    params = PartParameters({"length": 10.0})
    monkeypatch.delenv("SHELLFORGEPY_STEP_CACHE_DIR", raising=False)

    calls = {"func": 0}

    @step_serialization.step_cached
    def create_part(parameters):
        calls["func"] += 1
        return "fresh-part"

    assert create_part(params) == "fresh-part"
    assert calls["func"] == 1
