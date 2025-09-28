import numpy as np
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.construct.leaders_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box,
    get_bounding_box_center,
)


def test_leader_followers_creation():
    """Test basic LeaderFollowersCuttersPart creation."""
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    cutter = NamedPart("cutter", create_basic_box(0.5, 0.5, 0.5))
    aux = NamedPart("aux", create_basic_box(0.2, 0.2, 0.2))

    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower],
        cutters=[cutter],
        non_production_parts=[aux],
    )

    assert group.leader is not None
    assert len(group.followers) == 1
    assert len(group.cutters) == 1
    assert len(group.non_production_parts) == 1
    assert group.followers[0].name == "follower"


def test_leader_followers_copy():
    """Test LeaderFollowersCuttersPart copying."""
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    copied_group = group.copy()

    assert copied_group is not group
    assert copied_group.leader is not group.leader
    assert copied_group.followers[0] is not group.followers[0]
    assert copied_group.followers[0].name == "follower"


def test_leader_followers_translate_method():
    """Test LeaderFollowersCuttersPart.translate() method."""
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    original_leader_center = get_bounding_box_center(group.leader)
    original_follower_center = get_bounding_box_center(group.followers[0].part)

    translated_group = group.translate((5, 7, 13))

    # Should return self
    assert translated_group is group

    translated_leader_center = get_bounding_box_center(group.leader)
    translated_follower_center = get_bounding_box_center(group.followers[0].part)

    assert np.allclose(
        translated_leader_center,
        (
            original_leader_center[0] + 5,
            original_leader_center[1] + 7,
            original_leader_center[2] + 13,
        ),
    )
    assert np.allclose(
        translated_follower_center,
        (
            original_follower_center[0] + 5,
            original_follower_center[1] + 7,
            original_follower_center[2] + 13,
        ),
    )


def test_leader_followers_rotate_method():
    """Test LeaderFollowersCuttersPart with functional rotate interface."""
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    # Use functional interface for framework-standardized parameters
    rotated_group = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(group)

    # Should return a new group (functional interface returns new objects)
    assert rotated_group is not group
    assert isinstance(rotated_group, LeaderFollowersCuttersPart)

    # Check that rotation occurred (dimensions should swap)
    leader_bbox = get_bounding_box(group.leader)
    len_x = leader_bbox[1][0] - leader_bbox[0][0]
    len_y = leader_bbox[1][1] - leader_bbox[0][1]
    len_z = leader_bbox[1][2] - leader_bbox[0][2]

    # After 90° rotation around Z, a 2x2x2 box should still be 2x2x2
    assert np.allclose(len_x, 2)
    assert np.allclose(len_y, 2)
    assert np.allclose(len_z, 2)


def test_leader_followers_vs_native_part_translate_consistency():
    """Test that translate()(group) behaves consistently."""
    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    # Get original centers
    original_leader_center = get_bounding_box_center(group.leader)
    original_follower_center = get_bounding_box_center(group.followers[0].part)

    # Apply functional translation
    translated_group = translate(5, 7, 13)(group)

    # Should still be a LeaderFollowersCuttersPart
    assert isinstance(translated_group, LeaderFollowersCuttersPart)

    # Check that translation was applied
    translated_leader_center = get_bounding_box_center(translated_group.leader)
    translated_follower_center = get_bounding_box_center(
        translated_group.followers[0].part
    )

    expected_leader_center = (
        original_leader_center[0] + 5,
        original_leader_center[1] + 7,
        original_leader_center[2] + 13,
    )
    expected_follower_center = (
        original_follower_center[0] + 5,
        original_follower_center[1] + 7,
        original_follower_center[2] + 13,
    )

    assert np.allclose(translated_leader_center, expected_leader_center)
    assert np.allclose(translated_follower_center, expected_follower_center)


def test_leader_followers_vs_native_part_rotate_consistency():
    """Test that rotate()(group) behaves consistently."""
    leader = create_basic_box(10, 20, 30)
    follower = NamedPart("follower", create_basic_box(5, 10, 15))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    # Apply functional rotation
    rotated_group = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(group)

    # Should still be a LeaderFollowersCuttersPart
    assert isinstance(rotated_group, LeaderFollowersCuttersPart)

    # Check that rotation was applied - dimensions should swap for X and Y
    leader_bbox = get_bounding_box(rotated_group.leader)
    len_x = leader_bbox[1][0] - leader_bbox[0][0]
    len_y = leader_bbox[1][1] - leader_bbox[0][1]
    len_z = leader_bbox[1][2] - leader_bbox[0][2]

    # After 90° rotation around Z: (10, 20, 30) -> (20, 10, 30)
    assert np.allclose(len_x, 20)
    assert np.allclose(len_y, 10)
    assert np.allclose(len_z, 30)


def test_leader_followers_chained_transformations():
    """Test chaining transformations on LeaderFollowersCuttersPart."""
    leader = create_basic_box(10, 10, 10)
    group = LeaderFollowersCuttersPart(leader)

    # Chain: translate then rotate
    result = rotate(45, axis=(0, 0, 1), center=(0, 0, 0))(translate(10, 0, 0)(group))

    assert isinstance(result, LeaderFollowersCuttersPart)

    # Check that transformations were applied
    center = get_bounding_box_center(result.leader)
    # After translate(10, 0, 0) and rotate(45°), the center should have moved
    assert not np.allclose(center, (0, 0, 0))  # Should not be at origin


def test_leader_followers_parameter_order_consistency():
    """Test that LeaderFollowersCuttersPart handles different parameter orders consistently."""
    leader = create_basic_box(10, 20, 30)
    group = LeaderFollowersCuttersPart(leader)

    # Test different parameter orders for rotation
    result1 = rotate(45, axis=(0, 0, 1), center=(0, 0, 0))(group)
    result2 = rotate(45, center=(0, 0, 0), axis=(0, 0, 1))(group)

    # Both should give the same result
    bbox1 = get_bounding_box(result1.leader)
    bbox2 = get_bounding_box(result2.leader)

    assert np.allclose(bbox1[0], bbox2[0])  # min bounds
    assert np.allclose(bbox1[1], bbox2[1])  # max bounds


def test_leader_followers_fuse_operations():
    """Test fuse operations with LeaderFollowersCuttersPart."""
    leader1 = create_basic_box(2, 2, 2)
    follower1 = NamedPart("follower1", create_basic_box(1, 1, 1))
    group1 = LeaderFollowersCuttersPart(leader1, followers=[follower1])

    leader2 = create_basic_box(2, 2, 2)
    follower2 = NamedPart("follower2", create_basic_box(1, 1, 1))
    group2 = LeaderFollowersCuttersPart(leader2, followers=[follower2])

    # Test fusing with another group
    fused_group = group1.fuse(group2)

    assert isinstance(fused_group, LeaderFollowersCuttersPart)
    assert len(fused_group.followers) == 2  # Should have both followers

    # Test fusing with a regular part
    regular_part = create_basic_box(1, 1, 1)
    fused_with_part = group1.fuse(regular_part)

    assert isinstance(fused_with_part, LeaderFollowersCuttersPart)
    assert (
        len(fused_with_part.followers) == 1
    )  # Should still have the original follower


def test_leader_followers_complex_nested_operations():
    """Test complex nested operations to ensure consistency."""
    # Create a complex group with properly constructed NamedParts
    leader = create_basic_box(5, 5, 5)
    follower_part = translate(10, 0, 0)(create_basic_box(2, 2, 2))
    follower = NamedPart("follower", follower_part)
    cutter_part = translate(0, 10, 0)(create_basic_box(1, 1, 1))
    cutter = NamedPart("cutter", cutter_part)
    group = LeaderFollowersCuttersPart(leader, followers=[follower], cutters=[cutter])

    # Apply a complex chain of transformations
    result = rotate(90, axis=(0, 0, 1))(
        translate(5, 5, 0)(rotate(45, axis=(1, 0, 0))(group))
    )

    assert isinstance(result, LeaderFollowersCuttersPart)
    assert len(result.followers) == 1
    assert len(result.cutters) == 1
    assert result.followers[0].name == "follower"
    assert result.cutters[0].name == "cutter"

    # Verify that all parts are still valid
    assert result.leader is not None
    assert result.followers[0].part is not None
    assert result.cutters[0].part is not None
