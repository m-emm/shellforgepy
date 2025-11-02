import numpy as np
import pytest
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
    reset_to_original_orientation,
)
from shellforgepy.simple import (
    create_box,
    create_cylinder,
    get_bounding_box_center,
    get_bounding_box_size,
)


def test_reset_to_original_orientation_identity():
    """Test that a part with no transformations returns unchanged when reset."""
    # Create a complex part with leader, followers, cutters
    leader = create_box(2, 2, 2)
    follower = create_cylinder(0.5, 3)
    cutter = create_box(0.5, 0.5, 4)

    part = LeaderFollowersCuttersPart(
        leader=leader,
        followers=[follower],
        cutters=[cutter],
        additional_data={"test": "data"},
    )

    # Get original properties
    original_center = get_bounding_box_center(part.leader)
    original_matrix = part.cumulative_transform_matrix.copy()

    # Reset should return a copy with same properties
    reset_part = reset_to_original_orientation(part)

    # Check that it's a different object
    assert reset_part is not part

    # Check that transformation matrix is still identity
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))
    assert np.allclose(original_matrix, np.eye(4))  # Original was identity

    # Check that geometry is preserved
    reset_center = get_bounding_box_center(reset_part.leader)
    assert np.allclose(original_center, reset_center, atol=1e-6)

    # Check that additional data is preserved
    assert reset_part.additional_data == {"test": "data"}


def test_reset_orientation_after_translation():
    """Test reset after translating a complex part."""
    # Create complex geometry: cube fused with cylinder
    cube = create_box(2, 2, 2)
    cylinder = create_cylinder(0.8, 1)

    # Create the part with additional geometry
    part = LeaderFollowersCuttersPart(
        leader=cube.fuse(cylinder),
        followers=[create_box(0.5, 0.5, 0.5)],
        cutters=[create_cylinder(0.2, 3)],
        additional_data={"original": True, "id": 123},
    )

    # Get original bounding box properties
    original_center = get_bounding_box_center(part.leader)
    original_size = get_bounding_box_size(part.leader)

    # Apply translation
    translation = (5, 3, -2)
    part.translate(*translation)

    # Verify translation worked
    translated_center = get_bounding_box_center(part.leader)
    expected_center = np.array(original_center) + np.array(translation)
    assert np.allclose(translated_center, expected_center, atol=1e-6)

    # Verify transformation matrix is not identity
    assert not np.allclose(part.cumulative_transform_matrix, np.eye(4))

    # Reset to original orientation
    reset_part = reset_to_original_orientation(part)

    # Check that geometry is back to original position
    reset_center = get_bounding_box_center(reset_part.leader)
    reset_size = get_bounding_box_size(reset_part.leader)

    assert np.allclose(reset_center, original_center, atol=1e-6)
    assert np.allclose(reset_size, original_size, atol=1e-6)

    # Check that transformation matrix is reset to identity
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))

    # Check that additional data and modifications are preserved
    assert reset_part.additional_data == {"original": True, "id": 123}
    assert len(reset_part.followers) == 1
    assert len(reset_part.cutters) == 1


def test_reset_orientation_after_rotation():
    """Test reset after rotating a complex part."""
    # Create distinctive asymmetric geometry
    box1 = create_box(3, 1, 1)  # Elongated in X
    box2 = create_box(1, 1, 2)  # Elongated in Z
    complex_leader = box1.fuse(box2)

    part = LeaderFollowersCuttersPart(
        leader=complex_leader,
        followers=[create_cylinder(0.3, 2)],
        additional_data={"rotated": False},
    )

    # Get original properties
    original_center = get_bounding_box_center(part.leader)
    original_size = get_bounding_box_size(part.leader)

    # Apply 90-degree rotation around Z-axis
    part.rotate((0, 0, 0), (0, 0, 1), 90)

    # Verify rotation changed the geometry
    rotated_center = get_bounding_box_center(part.leader)
    rotated_size = get_bounding_box_size(part.leader)

    # After 90° rotation around Z, X and Y dimensions should swap
    assert np.allclose(rotated_size[0], original_size[1], atol=1e-6)  # X becomes Y
    assert np.allclose(rotated_size[1], original_size[0], atol=1e-6)  # Y becomes X
    assert np.allclose(rotated_size[2], original_size[2], atol=1e-6)  # Z unchanged

    # Reset to original orientation
    reset_part = reset_to_original_orientation(part)

    # Check that geometry is back to original
    reset_center = get_bounding_box_center(reset_part.leader)
    reset_size = get_bounding_box_size(reset_part.leader)

    assert np.allclose(reset_center, original_center, atol=1e-6)
    assert np.allclose(reset_size, original_size, atol=1e-6)

    # Check transformation matrix reset
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))


def test_reset_orientation_after_multiple_transformations():
    """Test reset after multiple transformations (translation + rotation)."""
    # Create complex geometry
    main_body = create_box(2, 3, 1)
    attachment = create_cylinder(0.5, 2)

    part = LeaderFollowersCuttersPart(
        leader=main_body,
        followers=[attachment],
        cutters=[create_box(0.2, 0.2, 3)],
        non_production_parts=[create_cylinder(0.1, 1)],
        additional_data={"complex": True, "transforms": []},
    )

    # Get original properties
    original_center = get_bounding_box_center(part.leader)
    original_size = get_bounding_box_size(part.leader)
    original_follower_center = get_bounding_box_center(part.followers[0])

    # Apply multiple transformations
    part.translate(2, -1, 3)
    part.rotate((0, 0, 0), (1, 0, 0), 45)
    part.translate(-1, 2, 0)
    part.rotate((0, 0, 0), (0, 0, 1), -30)

    # Verify transformations accumulated
    assert not np.allclose(part.cumulative_transform_matrix, np.eye(4))

    # Reset to original orientation
    reset_part = reset_to_original_orientation(part)

    # Check that all parts are back to original positions
    reset_center = get_bounding_box_center(reset_part.leader)
    reset_size = get_bounding_box_size(reset_part.leader)
    reset_follower_center = get_bounding_box_center(reset_part.followers[0])

    assert np.allclose(reset_center, original_center, atol=1e-6)
    assert np.allclose(reset_size, original_size, atol=1e-6)
    assert np.allclose(reset_follower_center, original_follower_center, atol=1e-6)

    # Check that all parts were transformed together
    assert len(reset_part.followers) == 1
    assert len(reset_part.cutters) == 1
    assert len(reset_part.non_production_parts) == 1

    # Check transformation matrix reset
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))

    # Check metadata preserved
    assert reset_part.additional_data == {"complex": True, "transforms": []}


def test_reset_orientation_with_mirror():
    """Test reset after mirroring."""
    # Create asymmetric geometry for clear mirror effect
    box = create_box(3, 2, 1)
    small_attachment = create_box(0.5, 0.5, 0.5)

    # Position attachment off-center
    small_attachment = small_attachment.translate((1, 0.5, 0))

    part = LeaderFollowersCuttersPart(
        leader=box.fuse(small_attachment), additional_data={"mirrored": False}
    )

    # Get original properties
    original_center = get_bounding_box_center(part.leader)
    original_size = get_bounding_box_size(part.leader)

    # Apply mirror across YZ plane (normal=(1,0,0))
    part.mirror((1, 0, 0), (0, 0, 0))

    # Reset to original orientation
    reset_part = reset_to_original_orientation(part)

    # Check that geometry is back to original
    reset_center = get_bounding_box_center(reset_part.leader)
    reset_size = get_bounding_box_size(reset_part.leader)

    assert np.allclose(reset_center, original_center, atol=1e-6)
    assert np.allclose(reset_size, original_size, atol=1e-6)

    # Check transformation matrix reset
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))


def test_reset_orientation_preserves_fused_geometry():
    """Test that reset preserves geometry modifications (fuse/cut) while resetting transformations."""
    # Create initial parts
    base = create_box(2, 2, 1)
    addition = create_cylinder(0.5, 2)

    part = LeaderFollowersCuttersPart(leader=base)

    # Fuse additional geometry
    part = part.fuse(LeaderFollowersCuttersPart(addition))

    # Get properties after fusion
    fused_center = get_bounding_box_center(part.leader)
    fused_size = get_bounding_box_size(part.leader)

    # Apply transformations
    part.translate(3, 2, 1)
    part.rotate((0, 0, 0), (0, 0, 1), 60)

    # Reset orientation
    reset_part = reset_to_original_orientation(part)

    # Check that the fused geometry is preserved
    reset_center = get_bounding_box_center(reset_part.leader)
    reset_size = get_bounding_box_size(reset_part.leader)

    assert np.allclose(reset_center, fused_center, atol=1e-6)
    assert np.allclose(reset_size, fused_size, atol=1e-6)

    # Check transformation matrix reset
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))


def test_reset_orientation_non_leader_followers_part():
    """Test that non-LeaderFollowersCuttersPart objects are returned as copies unchanged."""
    # Create a simple box
    box = create_box(1, 1, 1)
    original_center = get_bounding_box_center(box)

    # Reset should return a copy unchanged
    reset_box = reset_to_original_orientation(box)

    # Check it's a different object
    assert reset_box is not box

    # Check geometry is unchanged
    reset_center = get_bounding_box_center(reset_box)
    assert np.allclose(reset_center, original_center, atol=1e-6)


def test_cumulative_transform_matrix_tracking():
    """Test that the cumulative transformation matrix is correctly tracked."""
    part = LeaderFollowersCuttersPart(leader=create_box(1, 1, 1))

    # Initially should be identity
    assert np.allclose(part.cumulative_transform_matrix, np.eye(4))

    # After translation
    part.translate(1, 2, 3)
    expected_translation = np.eye(4)
    expected_translation[0:3, 3] = [1, 2, 3]
    assert np.allclose(part.cumulative_transform_matrix, expected_translation)

    # After additional rotation (should compose with previous)
    part.rotate((0, 0, 0), (0, 0, 1), 90)
    # Matrix should no longer be just a translation
    assert not np.allclose(part.cumulative_transform_matrix[:3, :3], np.eye(3))

    # After reset, should be identity again
    reset_part = reset_to_original_orientation(part)
    assert np.allclose(reset_part.cumulative_transform_matrix, np.eye(4))


if __name__ == "__main__":
    pytest.main([__file__])
