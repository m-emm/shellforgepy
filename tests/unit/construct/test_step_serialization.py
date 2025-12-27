import pytest

pytest.importorskip("cadquery")

from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.step_serialization import (
    _get_metadata_path,
    deserialize_to_leader_followers_cutters_part,
    serialize_to_step,
)
from shellforgepy.simple import *


def test_serialize_to_step(tmp_path):
    """Test serialization of LeaderFollowersCuttersPart to STEP file."""

    leader = create_box(10, 10, 10)
    follower = create_cylinder(5, 15)

    lfcp = LeaderFollowersCuttersPart(leader=leader, followers=[follower])
    step_file_path = tmp_path / "test_part.step"

    serialize_to_step(lfcp, str(step_file_path))

    assert step_file_path.is_file()
    assert step_file_path.stat().st_size > 0

    # Check that metadata file was created
    metadata_path = tmp_path / "test_part.lfcp.json"
    assert metadata_path.is_file()


def test_serialize_deserialize_round_trip_with_names(tmp_path):
    """Test full round-trip serialization preserving structure and names."""
    leader = create_box(10, 10, 10)
    follower_a = create_cylinder(2, 8)
    follower_b = create_cylinder(3, 6)
    cutter = create_cylinder(1, 12)
    non_prod = create_box(4, 4, 2)

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        followers=[follower_a, follower_b],
        cutters=[cutter],
        non_production_parts=[non_prod],
        follower_names=["follower_a", "follower_b"],
        cutter_names=["cutter_a"],
        non_production_names=["non_prod_a"],
    )

    step_file_path = tmp_path / "round_trip.step"
    serialize_to_step(lfcp, str(step_file_path))

    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Verify structure is preserved
    assert restored.leader is not None
    assert len(restored.followers) == 2
    assert len(restored.cutters) == 1
    assert len(restored.non_production_parts) == 1

    # Verify names are preserved
    assert restored.get_follower_index_by_name("follower_a") == 0
    assert restored.get_follower_index_by_name("follower_b") == 1
    assert restored.get_cutter_index_by_name("cutter_a") == 0
    assert restored.get_non_production_index_by_name("non_prod_a") == 0

    # Verify volumes are approximately correct
    assert get_volume(restored.leader) == pytest.approx(10 * 10 * 10, rel=1e-4)
    assert get_volume(restored.followers[0]) == pytest.approx(
        3.141592653589793 * 2**2 * 8, rel=1e-4
    )
    assert get_volume(restored.followers[1]) == pytest.approx(
        3.141592653589793 * 3**2 * 6, rel=1e-4
    )


def test_serialize_to_step_with_path_object(tmp_path):
    """Test serialization works when passing Path object instead of str.

    Reproduces bug where CadQuery's Write() method expects str but receives PosixPath.
    See: TypeError: Write(): incompatible function arguments.
    """
    leader = create_box(10, 10, 10)
    follower = create_cylinder(5, 15)

    lfcp = LeaderFollowersCuttersPart(leader=leader, followers=[follower])

    # Pass Path object directly (not str) - this was causing the bug
    step_file_path = tmp_path / "test_path_object.step"

    # This should work without needing str() conversion
    serialize_to_step(lfcp, step_file_path)

    assert step_file_path.is_file()
    assert step_file_path.stat().st_size > 0


def test_deserialize_legacy_step_without_metadata(tmp_path):
    """Test that we can still import a plain STEP file without metadata."""
    # Create a simple STEP file without using our serialization
    box = create_box(10, 10, 10)
    step_file_path = tmp_path / "legacy.step"

    # Export directly using the adapter
    from shellforgepy.adapters._adapter import export_solid_to_step

    export_solid_to_step(box, str(step_file_path))

    # No metadata file should exist
    metadata_path = tmp_path / "legacy.lfcp.json"
    assert not metadata_path.exists()

    # Should still import as leader-only
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    assert restored.leader is not None
    assert not restored.followers
    assert not restored.cutters
    assert not restored.non_production_parts
    assert get_volume(restored.leader) == pytest.approx(10 * 10 * 10, rel=1e-4)


def test_serialize_leader_only(tmp_path):
    """Test serialization of a leader-only part."""
    leader = create_box(20, 15, 10)

    lfcp = LeaderFollowersCuttersPart(leader=leader)
    step_file_path = tmp_path / "leader_only.step"

    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    assert restored.leader is not None
    assert not restored.followers
    assert not restored.cutters
    assert not restored.non_production_parts
    assert get_volume(restored.leader) == pytest.approx(20 * 15 * 10, rel=1e-4)


def test_fuse_works_after_deserialization(tmp_path):
    """Test that fusing works correctly after deserializing from STEP."""
    leader = create_box(10, 10, 10)

    lfcp = LeaderFollowersCuttersPart(leader=leader)
    step_file_path = tmp_path / "for_fuse.step"

    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Create another part and fuse
    other_box = create_box(5, 5, 5)
    other_lfcp = LeaderFollowersCuttersPart(leader=other_box)

    # This should not raise an error
    result = restored.fuse(other_lfcp)

    assert result.leader is not None
    # The fused volume should be the sum (since they're at the same position,
    # it might be less due to overlap, but should be at least one box volume)
    assert get_volume(result.leader) >= 5 * 5 * 5


def test_metadata_path_helper():
    """Test the metadata path helper function."""
    assert _get_metadata_path("/path/to/file.step") == "/path/to/file.lfcp.json"
    assert _get_metadata_path("/path/to/file.STEP") == "/path/to/file.lfcp.json"
    assert _get_metadata_path("file.step") == "file.lfcp.json"
