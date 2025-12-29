import pytest

pytest.importorskip("cadquery")

from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.part_parameters import PartParameters
from shellforgepy.construct.step_serialization import (
    _get_metadata_path,
    deserialize_to_leader_followers_cutters_part,
    serialize_to_step,
    step_cached,
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


def test_step_cached_uses_step_cache(tmp_path, monkeypatch):
    params = PartParameters({"length": 10.0})
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("SHELLFORGEPY_STEP_CACHE_DIR", str(cache_dir))

    calls = {"func": 0}

    @step_cached
    def create_part(parameters):
        calls["func"] += 1
        return create_box(parameters.length, 10, 5)

    part_first = create_part(params)
    assert calls["func"] == 1
    assert get_volume(part_first) == pytest.approx(10 * 10 * 5, rel=1e-6)

    step_path = cache_dir / f"{params.parameters_hash()}.step"
    assert step_path.is_file()
    assert (cache_dir / f"{params.parameters_hash()}.lfcp.json").is_file()

    part_second = create_part(params)
    assert calls["func"] == 1
    assert get_volume(part_second) == pytest.approx(10 * 10 * 5, rel=1e-6)


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

    bboxes = [
        get_bounding_box(p) for p in [leader, follower_a, follower_b, cutter, non_prod]
    ]

    step_file_path = tmp_path / "round_trip.step"
    serialize_to_step(lfcp, str(step_file_path))

    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    bboxes_restored = [
        get_bounding_box(p)
        for p in [restored.leader]
        + restored.followers
        + restored.cutters
        + restored.non_production_parts
    ]
    # Verify bounding boxes match (parts are in same position/scale)
    for bb_orig, bb_restored in zip(bboxes, bboxes_restored):
        for v_orig, v_restored in zip(bb_orig, bb_restored):
            assert all(
                pytest.approx(a, rel=1e-6) == b for a, b in zip(v_orig, v_restored)
            )

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


def test_deserialize_step_without_metadata_raises_error(tmp_path):
    """Test that importing a STEP file without metadata raises an error."""
    # Create a simple STEP file without using our serialization
    box = create_box(10, 10, 10)
    step_file_path = tmp_path / "no_metadata.step"

    # Export directly using the adapter
    from shellforgepy.adapters._adapter import export_solid_to_step

    export_solid_to_step(box, str(step_file_path))

    # No metadata file should exist
    metadata_path = tmp_path / "no_metadata.lfcp.json"
    assert not metadata_path.exists()

    # Should raise ValueError
    with pytest.raises(ValueError, match="Metadata file not found"):
        deserialize_to_leader_followers_cutters_part(str(step_file_path))


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


def _assert_bboxes_equal(bbox1, bbox2, rel_tol=1e-6, description=""):
    """Helper to assert two bounding boxes are equal within tolerance."""
    for i, (corner1, corner2) in enumerate(zip(bbox1, bbox2)):
        for j, (v1, v2) in enumerate(zip(corner1, corner2)):
            assert v1 == pytest.approx(
                v2, rel=rel_tol
            ), f"{description} bbox corner {i} coord {j}: {v1} != {v2}"


def test_round_trip_preserves_exact_geometry_for_multiple_cutters(tmp_path):
    """Test that multiple cutters maintain their exact geometry after round-trip.

    This test creates cutters at different positions and verifies each one
    is restored with the exact same bounding box.
    """
    leader = create_box(100, 100, 10)

    # Create multiple cutters at different positions
    cutter1 = create_cylinder(5, 20)
    cutter1 = translate(10, 10, 0)(cutter1)

    cutter2 = create_cylinder(3, 15)
    cutter2 = translate(50, 50, 0)(cutter2)

    cutter3 = create_box(8, 8, 25)
    cutter3 = translate(80, 20, 0)(cutter3)

    cutters = [cutter1, cutter2, cutter3]
    original_bboxes = [get_bounding_box(c) for c in cutters]

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        cutters=cutters,
        cutter_names=["hole1", "hole2", "slot1"],
    )

    step_file_path = tmp_path / "multiple_cutters.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Verify we got the same number of cutters back
    assert (
        len(restored.cutters) == 3
    ), f"Expected 3 cutters, got {len(restored.cutters)}"

    # Verify each cutter has the exact same bounding box
    for i, (orig_bbox, restored_cutter) in enumerate(
        zip(original_bboxes, restored.cutters)
    ):
        restored_bbox = get_bounding_box(restored_cutter)
        _assert_bboxes_equal(orig_bbox, restored_bbox, description=f"cutter {i}")


def test_round_trip_preserves_geometry_for_overlapping_cutters(tmp_path):
    """Test that overlapping cutters are NOT merged during serialization.

    This is a critical test - if cutters overlap and get fused together,
    they become a single solid and can't be separated on deserialization.
    """
    leader = create_box(100, 100, 10)

    # Create two overlapping cutters (cylinders that intersect)
    cutter1 = create_cylinder(10, 30)
    cutter1 = translate(25, 25, 0)(cutter1)

    cutter2 = create_cylinder(10, 30)
    cutter2 = translate(30, 25, 0)(cutter2)  # Overlaps with cutter1

    original_bbox1 = get_bounding_box(cutter1)
    original_bbox2 = get_bounding_box(cutter2)
    original_volume1 = get_volume(cutter1)
    original_volume2 = get_volume(cutter2)

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        cutters=[cutter1, cutter2],
    )

    step_file_path = tmp_path / "overlapping_cutters.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Must have exactly 2 cutters, not 1 merged cutter
    assert len(restored.cutters) == 2, (
        f"Expected 2 separate cutters, got {len(restored.cutters)}. "
        "Overlapping cutters may have been merged during serialization!"
    )

    # Each cutter should have its original bounding box
    _assert_bboxes_equal(
        original_bbox1,
        get_bounding_box(restored.cutters[0]),
        description="overlapping cutter 0",
    )
    _assert_bboxes_equal(
        original_bbox2,
        get_bounding_box(restored.cutters[1]),
        description="overlapping cutter 1",
    )

    # Each cutter should have its original volume
    assert get_volume(restored.cutters[0]) == pytest.approx(original_volume1, rel=1e-4)
    assert get_volume(restored.cutters[1]) == pytest.approx(original_volume2, rel=1e-4)


def test_round_trip_preserves_geometry_for_touching_parts(tmp_path):
    """Test that parts sharing a face/edge are correctly separated."""
    leader = create_box(10, 10, 10)

    # Create followers that touch the leader (share a face)
    follower1 = create_box(10, 10, 5)
    follower1 = translate(0, 0, 10)(follower1)  # Sits on top of leader

    follower2 = create_box(5, 10, 10)
    follower2 = translate(10, 0, 0)(follower2)  # Sits next to leader

    original_leader_bbox = get_bounding_box(leader)
    original_f1_bbox = get_bounding_box(follower1)
    original_f2_bbox = get_bounding_box(follower2)

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        followers=[follower1, follower2],
    )

    step_file_path = tmp_path / "touching_parts.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    assert len(restored.followers) == 2, "Expected 2 followers"

    _assert_bboxes_equal(
        original_leader_bbox, get_bounding_box(restored.leader), description="leader"
    )
    _assert_bboxes_equal(
        original_f1_bbox,
        get_bounding_box(restored.followers[0]),
        description="follower 0",
    )
    _assert_bboxes_equal(
        original_f2_bbox,
        get_bounding_box(restored.followers[1]),
        description="follower 1",
    )


def test_round_trip_with_complex_cutter_geometry(tmp_path):
    """Test serialization of a cutter that is itself a compound of multiple solids.

    KNOWN ISSUE: This test exposes a bug where disconnected solids within a single
    cutter are not all preserved during serialization. Only one solid (matching
    the expected Z position) is restored.
    """
    from shellforgepy.simple import PartCollector

    leader = create_box(100, 100, 20)

    # Create a complex cutter by fusing multiple shapes
    cutter_parts = PartCollector()
    for x in [10, 30, 50, 70]:
        hole = create_cylinder(3, 30)
        hole = translate(x, 50, 0)(hole)
        cutter_parts = cutter_parts.fuse(hole)

    # The cutter is now a compound with multiple solids fused together
    original_bbox = get_bounding_box(cutter_parts)
    original_volume = get_volume(cutter_parts)

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        cutters=[cutter_parts],
    )

    step_file_path = tmp_path / "complex_cutter.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    assert len(restored.cutters) == 1
    _assert_bboxes_equal(
        original_bbox,
        get_bounding_box(restored.cutters[0]),
        description="complex cutter",
    )
    assert get_volume(restored.cutters[0]) == pytest.approx(original_volume, rel=1e-4)


def test_round_trip_all_part_types_with_positions(tmp_path):
    """Comprehensive test verifying all part types maintain exact positions."""
    # Create parts at specific, non-zero positions
    leader = create_box(20, 20, 10)
    leader = translate(100, 200, 50)(leader)

    follower1 = create_cylinder(5, 15)
    follower1 = translate(150, 250, 60)(follower1)

    follower2 = create_box(8, 8, 8)
    follower2 = translate(120, 180, 40)(follower2)

    cutter1 = create_cylinder(2, 20)
    cutter1 = translate(105, 205, 45)(cutter1)

    cutter2 = create_cylinder(3, 25)
    cutter2 = translate(115, 215, 45)(cutter2)

    non_prod = create_box(5, 5, 5)
    non_prod = translate(130, 220, 55)(non_prod)

    all_parts = [leader, follower1, follower2, cutter1, cutter2, non_prod]
    original_bboxes = [get_bounding_box(p) for p in all_parts]
    original_volumes = [get_volume(p) for p in all_parts]

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        followers=[follower1, follower2],
        cutters=[cutter1, cutter2],
        non_production_parts=[non_prod],
        follower_names=["f1", "f2"],
        cutter_names=["c1", "c2"],
        non_production_names=["np1"],
    )

    step_file_path = tmp_path / "all_types_positioned.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Reconstruct the list in same order
    restored_parts = (
        [restored.leader]
        + restored.followers
        + restored.cutters
        + restored.non_production_parts
    )

    assert len(restored_parts) == len(all_parts)

    part_names = ["leader", "follower1", "follower2", "cutter1", "cutter2", "non_prod"]
    for i, (orig_bbox, orig_vol, restored_part, name) in enumerate(
        zip(original_bboxes, original_volumes, restored_parts, part_names)
    ):
        restored_bbox = get_bounding_box(restored_part)
        _assert_bboxes_equal(orig_bbox, restored_bbox, description=name)
        assert get_volume(restored_part) == pytest.approx(
            orig_vol, rel=1e-4
        ), f"{name} volume mismatch"


def test_serialize_deserialize_additional_data_with_numpy_arrays(tmp_path):
    """Test that additional_data including numpy arrays survives round-trip.

    This tests the pattern from hair2.py where transformation data is stored:
    - applied_transformation (dict with numpy arrays)
    - hair_position (tuple/array)
    - hair_normal (numpy array)
    - hair_index (int)
    - label_text (str)
    - hair_2d_position (numpy array)
    """
    import numpy as np

    leader = create_box(10, 10, 10)

    # Create additional_data similar to hair2.py pattern
    additional_data = {
        "applied_transformation": {
            "rotation_angle": np.float64(0.7853981633974483),  # pi/4
            "rotation_axis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "translation": np.array([10.5, 20.3, 5.0], dtype=np.float64),
        },
        "hair_position": np.array([15.2, 25.4, 8.0], dtype=np.float64),
        "hair_normal": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "hair_index": 42,
        "label_text": "test_hair_99",
        "hair_2d_position": np.array([12.5, 18.3], dtype=np.float64),
        "nested_list": [[1, 2, 3], [4, 5, 6]],
        "nested_arrays": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }

    lfcp = LeaderFollowersCuttersPart(
        leader=leader,
        additional_data=additional_data,
    )

    step_file_path = tmp_path / "with_additional_data.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Verify additional_data was restored
    assert restored.additional_data is not None
    assert len(restored.additional_data) == len(additional_data)

    # Check scalar values
    assert restored.additional_data["hair_index"] == 42
    assert restored.additional_data["label_text"] == "test_hair_99"

    # Check numpy arrays are restored
    assert isinstance(restored.additional_data["hair_position"], np.ndarray)
    assert np.allclose(
        restored.additional_data["hair_position"], additional_data["hair_position"]
    )

    assert isinstance(restored.additional_data["hair_normal"], np.ndarray)
    assert np.allclose(
        restored.additional_data["hair_normal"], additional_data["hair_normal"]
    )

    assert isinstance(restored.additional_data["hair_2d_position"], np.ndarray)
    assert np.allclose(
        restored.additional_data["hair_2d_position"],
        additional_data["hair_2d_position"],
    )

    # Check nested dict with numpy arrays
    trans = restored.additional_data["applied_transformation"]
    assert isinstance(trans["rotation_axis"], np.ndarray)
    assert np.allclose(
        trans["rotation_axis"],
        additional_data["applied_transformation"]["rotation_axis"],
    )
    assert isinstance(trans["translation"], np.ndarray)
    assert np.allclose(
        trans["translation"],
        additional_data["applied_transformation"]["translation"],
    )
    # Check numpy scalar was converted to Python float
    assert trans["rotation_angle"] == pytest.approx(0.7853981633974483, rel=1e-10)

    # Check nested lists with arrays
    assert len(restored.additional_data["nested_arrays"]) == 2
    assert np.allclose(restored.additional_data["nested_arrays"][0], [1.0, 2.0])
    assert np.allclose(restored.additional_data["nested_arrays"][1], [3.0, 4.0])


def test_serialize_deserialize_empty_additional_data(tmp_path):
    """Test that empty additional_data is handled correctly."""
    leader = create_box(10, 10, 10)

    lfcp = LeaderFollowersCuttersPart(leader=leader)
    assert lfcp.additional_data == {}  # Default is empty dict

    step_file_path = tmp_path / "empty_additional_data.step"
    serialize_to_step(lfcp, step_file_path)
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))

    # Should have empty additional_data after round-trip
    assert restored.additional_data == {} or restored.additional_data is None


def test_version_2_metadata_still_works(tmp_path):
    """Test that v2 metadata (without additional_data) still deserializes correctly."""
    import json

    leader = create_box(10, 10, 10)
    lfcp = LeaderFollowersCuttersPart(leader=leader)

    step_file_path = tmp_path / "v2_compat.step"
    serialize_to_step(lfcp, step_file_path)

    # Manually downgrade metadata to v2 (without additional_data key)
    metadata_path = tmp_path / "v2_compat.lfcp.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["version"] = 2
    if "additional_data" in metadata:
        del metadata["additional_data"]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Should still deserialize correctly
    restored = deserialize_to_leader_followers_cutters_part(str(step_file_path))
    assert restored.leader is not None
    assert restored.additional_data is None or restored.additional_data == {}
