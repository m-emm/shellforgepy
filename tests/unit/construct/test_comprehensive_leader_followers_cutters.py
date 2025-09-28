import numpy as np
from shellforgepy.construct.leaders_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box,
    get_bounding_box_center,
    rotate,
    translate,
)


def test_comprehensive_leader_followers_cutters_rotation():
    """Test that ALL components (leader, followers, cutters, non_production_parts) are properly rotated."""

    # Create parts with different dimensions so rotation is visible
    leader = create_basic_box(10, 20, 30)  # Different dimensions
    follower1 = NamedPart("follower1", create_basic_box(5, 15, 25))
    follower2 = NamedPart("follower2", create_basic_box(8, 12, 18))
    cutter1 = NamedPart("cutter1", create_basic_box(3, 9, 21))
    cutter2 = NamedPart("cutter2", create_basic_box(6, 11, 17))
    non_prod1 = NamedPart("non_prod1", create_basic_box(4, 8, 16))
    non_prod2 = NamedPart("non_prod2", create_basic_box(7, 13, 19))

    # Create group with all component types
    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1, follower2],
        cutters=[cutter1, cutter2],
        non_production_parts=[non_prod1, non_prod2],
    )

    # Get original dimensions of all components
    def get_dimensions(part):
        bbox = get_bounding_box(part)
        return (
            bbox[1][0] - bbox[0][0],  # X size
            bbox[1][1] - bbox[0][1],  # Y size
            bbox[1][2] - bbox[0][2],  # Z size
        )

    # Original dimensions
    original_leader_dims = get_dimensions(group.leader)
    original_follower1_dims = get_dimensions(group.followers[0].part)
    original_follower2_dims = get_dimensions(group.followers[1].part)
    original_cutter1_dims = get_dimensions(group.cutters[0].part)
    original_cutter2_dims = get_dimensions(group.cutters[1].part)
    original_non_prod1_dims = get_dimensions(group.non_production_parts[0].part)
    original_non_prod2_dims = get_dimensions(group.non_production_parts[1].part)

    # Apply 90° Z rotation - should swap X and Y dimensions
    rotated_group = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(group)

    # Check that all components were rotated (X and Y should swap, Z unchanged)
    rotated_leader_dims = get_dimensions(rotated_group.leader)
    rotated_follower1_dims = get_dimensions(rotated_group.followers[0].part)
    rotated_follower2_dims = get_dimensions(rotated_group.followers[1].part)
    rotated_cutter1_dims = get_dimensions(rotated_group.cutters[0].part)
    rotated_cutter2_dims = get_dimensions(rotated_group.cutters[1].part)
    rotated_non_prod1_dims = get_dimensions(rotated_group.non_production_parts[0].part)
    rotated_non_prod2_dims = get_dimensions(rotated_group.non_production_parts[1].part)

    # After 90° rotation around Z, X and Y should swap
    assert np.allclose(
        rotated_leader_dims,
        (original_leader_dims[1], original_leader_dims[0], original_leader_dims[2]),
        atol=1e-10,
    ), f"Leader rotation failed: {original_leader_dims} -> {rotated_leader_dims}"

    assert np.allclose(
        rotated_follower1_dims,
        (
            original_follower1_dims[1],
            original_follower1_dims[0],
            original_follower1_dims[2],
        ),
        atol=1e-10,
    ), f"Follower1 rotation failed: {original_follower1_dims} -> {rotated_follower1_dims}"

    assert np.allclose(
        rotated_follower2_dims,
        (
            original_follower2_dims[1],
            original_follower2_dims[0],
            original_follower2_dims[2],
        ),
        atol=1e-10,
    ), f"Follower2 rotation failed: {original_follower2_dims} -> {rotated_follower2_dims}"

    assert np.allclose(
        rotated_cutter1_dims,
        (original_cutter1_dims[1], original_cutter1_dims[0], original_cutter1_dims[2]),
        atol=1e-10,
    ), f"Cutter1 rotation failed: {original_cutter1_dims} -> {rotated_cutter1_dims}"

    assert np.allclose(
        rotated_cutter2_dims,
        (original_cutter2_dims[1], original_cutter2_dims[0], original_cutter2_dims[2]),
        atol=1e-10,
    ), f"Cutter2 rotation failed: {original_cutter2_dims} -> {rotated_cutter2_dims}"

    assert np.allclose(
        rotated_non_prod1_dims,
        (
            original_non_prod1_dims[1],
            original_non_prod1_dims[0],
            original_non_prod1_dims[2],
        ),
        atol=1e-10,
    ), f"NonProd1 rotation failed: {original_non_prod1_dims} -> {rotated_non_prod1_dims}"

    assert np.allclose(
        rotated_non_prod2_dims,
        (
            original_non_prod2_dims[1],
            original_non_prod2_dims[0],
            original_non_prod2_dims[2],
        ),
        atol=1e-10,
    ), f"NonProd2 rotation failed: {original_non_prod2_dims} -> {rotated_non_prod2_dims}"


def test_comprehensive_leader_followers_cutters_translation():
    """Test that ALL components (leader, followers, cutters, non_production_parts) are properly translated."""

    # Create parts at different positions so translation is visible
    leader = create_basic_box(10, 20, 30)
    follower1 = NamedPart("follower1", create_basic_box(5, 15, 25))
    follower2 = NamedPart("follower2", create_basic_box(8, 12, 18))
    cutter1 = NamedPart("cutter1", create_basic_box(3, 9, 21))
    cutter2 = NamedPart("cutter2", create_basic_box(6, 11, 17))
    non_prod1 = NamedPart("non_prod1", create_basic_box(4, 8, 16))
    non_prod2 = NamedPart("non_prod2", create_basic_box(7, 13, 19))

    # Create group with all component types
    group = LeaderFollowersCuttersPart(
        leader,
        followers=[follower1, follower2],
        cutters=[cutter1, cutter2],
        non_production_parts=[non_prod1, non_prod2],
    )

    # Get original centers of all components
    original_leader_center = get_bounding_box_center(group.leader)
    original_follower1_center = get_bounding_box_center(group.followers[0].part)
    original_follower2_center = get_bounding_box_center(group.followers[1].part)
    original_cutter1_center = get_bounding_box_center(group.cutters[0].part)
    original_cutter2_center = get_bounding_box_center(group.cutters[1].part)
    original_non_prod1_center = get_bounding_box_center(
        group.non_production_parts[0].part
    )
    original_non_prod2_center = get_bounding_box_center(
        group.non_production_parts[1].part
    )

    # Apply translation by (5, 7, 11)
    translated_group = translate(5, 7, 11)(group)

    # Check that all components were translated
    translated_leader_center = get_bounding_box_center(translated_group.leader)
    translated_follower1_center = get_bounding_box_center(
        translated_group.followers[0].part
    )
    translated_follower2_center = get_bounding_box_center(
        translated_group.followers[1].part
    )
    translated_cutter1_center = get_bounding_box_center(
        translated_group.cutters[0].part
    )
    translated_cutter2_center = get_bounding_box_center(
        translated_group.cutters[1].part
    )
    translated_non_prod1_center = get_bounding_box_center(
        translated_group.non_production_parts[0].part
    )
    translated_non_prod2_center = get_bounding_box_center(
        translated_group.non_production_parts[1].part
    )

    # All centers should have moved by (5, 7, 11)
    expected_leader_center = (
        original_leader_center[0] + 5,
        original_leader_center[1] + 7,
        original_leader_center[2] + 11,
    )
    expected_follower1_center = (
        original_follower1_center[0] + 5,
        original_follower1_center[1] + 7,
        original_follower1_center[2] + 11,
    )
    expected_follower2_center = (
        original_follower2_center[0] + 5,
        original_follower2_center[1] + 7,
        original_follower2_center[2] + 11,
    )
    expected_cutter1_center = (
        original_cutter1_center[0] + 5,
        original_cutter1_center[1] + 7,
        original_cutter1_center[2] + 11,
    )
    expected_cutter2_center = (
        original_cutter2_center[0] + 5,
        original_cutter2_center[1] + 7,
        original_cutter2_center[2] + 11,
    )
    expected_non_prod1_center = (
        original_non_prod1_center[0] + 5,
        original_non_prod1_center[1] + 7,
        original_non_prod1_center[2] + 11,
    )
    expected_non_prod2_center = (
        original_non_prod2_center[0] + 5,
        original_non_prod2_center[1] + 7,
        original_non_prod2_center[2] + 11,
    )

    assert np.allclose(
        translated_leader_center, expected_leader_center
    ), f"Leader translation failed: {original_leader_center} -> {translated_leader_center}, expected {expected_leader_center}"

    assert np.allclose(
        translated_follower1_center, expected_follower1_center
    ), f"Follower1 translation failed: {original_follower1_center} -> {translated_follower1_center}, expected {expected_follower1_center}"

    assert np.allclose(
        translated_follower2_center, expected_follower2_center
    ), f"Follower2 translation failed: {original_follower2_center} -> {translated_follower2_center}, expected {expected_follower2_center}"

    assert np.allclose(
        translated_cutter1_center, expected_cutter1_center
    ), f"Cutter1 translation failed: {original_cutter1_center} -> {translated_cutter1_center}, expected {expected_cutter1_center}"

    assert np.allclose(
        translated_cutter2_center, expected_cutter2_center
    ), f"Cutter2 translation failed: {original_cutter2_center} -> {translated_cutter2_center}, expected {expected_cutter2_center}"

    assert np.allclose(
        translated_non_prod1_center, expected_non_prod1_center
    ), f"NonProd1 translation failed: {original_non_prod1_center} -> {translated_non_prod1_center}, expected {expected_non_prod1_center}"

    assert np.allclose(
        translated_non_prod2_center, expected_non_prod2_center
    ), f"NonProd2 translation failed: {original_non_prod2_center} -> {translated_non_prod2_center}, expected {expected_non_prod2_center}"
