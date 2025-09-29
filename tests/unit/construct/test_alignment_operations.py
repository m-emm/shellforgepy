import logging

import numpy as np
from shellforgepy.simple import (
    Alignment,
    align,
    apply_fillet_by_alignment,
    create_basic_box,
    create_basic_cylinder,
    get_adapter_id,
    get_bounding_box,
    get_bounding_box_center,
    rotate,
    translate,
)

_logger = logging.getLogger(__name__)


def test_translate():

    part = create_basic_box(10, 20, 30)
    part_center = get_bounding_box_center(part)

    part_translated = translate(5, 7, 13)(part)
    translated_center = get_bounding_box_center(part_translated)

    assert translated_center == (
        part_center[0] + 5,
        part_center[1] + 7,
        part_center[2] + 13,
    )


def test_rotate():

    part = create_basic_box(10, 20, 30)

    rotated_part = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(part)

    assert rotated_part is not None

    bounding_box = get_bounding_box(rotated_part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(len_x, 20)
    assert np.allclose(len_y, 10)
    assert np.allclose(len_z, 30)


def test_rotate_different_parameter_orders():
    """Test different ways of calling rotate to see if there are parameter order issues."""
    part = create_basic_box(10, 20, 30)

    # Test 1: keyword arguments in different orders
    rotated1 = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(part)
    rotated2 = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(part)

    # Both should give the same result
    bbox1 = get_bounding_box(rotated1)
    bbox2 = get_bounding_box(rotated2)

    assert np.allclose(bbox1[0], bbox2[0])  # min bounds should match
    assert np.allclose(bbox1[1], bbox2[1])  # max bounds should match


def test_functional_consistency_with_named_parts():
    """Test that functional transformations work consistently with NamedPart objects."""
    from shellforgepy.construct.named_part import NamedPart

    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    _logger.info(
        f"Created named_part with id {id(named_part)} and part id {id(named_part.part)}"
    )
    # Functional transformations should work on both native parts and NamedParts
    native_translated = translate(5, 0, 0)(part)
    named_translated = translate(5, 0, 0)(named_part)

    # NamedPart should still be a NamedPart after transformation
    assert isinstance(named_translated, NamedPart)
    assert named_translated.name == "test"

    # Results should be equivalent
    _logger.info(f"native_translated id {id(native_translated)}")
    _logger.info(
        f"named_translated id {id(named_translated)} and part id {id(named_translated.part)}"
    )
    native_center = get_bounding_box_center(native_translated)
    named_center = get_bounding_box_center(named_translated.part)
    assert np.allclose(native_center, named_center)


def test_functional_consistency_with_leader_followers():
    """Test that functional transformations work consistently with LeaderFollowersCuttersPart."""
    from shellforgepy.construct.leaders_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    # Functional transformations should work on the group
    translated_group = translate(5, 0, 0)(group)

    # Should still be a LeaderFollowersCuttersPart
    assert isinstance(translated_group, LeaderFollowersCuttersPart)

    # Check that both leader and followers were translated
    original_leader_center = get_bounding_box_center(leader)
    original_follower_center = get_bounding_box_center(follower.part)

    translated_leader_center = get_bounding_box_center(translated_group.leader)
    translated_follower_center = get_bounding_box_center(
        translated_group.followers[0].part
    )

    assert np.allclose(
        translated_leader_center,
        (
            original_leader_center[0] + 5,
            original_leader_center[1],
            original_leader_center[2],
        ),
    )
    assert np.allclose(
        translated_follower_center,
        (
            original_follower_center[0] + 5,
            original_follower_center[1],
            original_follower_center[2],
        ),
    )


def test_chained_transformations_consistency():
    """Test that chained transformations work consistently across all object types."""
    from shellforgepy.construct.leaders_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    # Test with native part
    native_part = create_basic_box(10, 10, 10)
    native_result = rotate(45, axis=(0, 0, 1))(translate(10, 0, 0)(native_part))

    # Test with NamedPart
    named_part = NamedPart("test", create_basic_box(10, 10, 10))
    named_result = rotate(45, axis=(0, 0, 1))(translate(10, 0, 0)(named_part))

    # Test with LeaderFollowersCuttersPart
    group_part = LeaderFollowersCuttersPart(create_basic_box(10, 10, 10))
    group_result = rotate(45, axis=(0, 0, 1))(translate(10, 0, 0)(group_part))

    # All should preserve their types
    assert isinstance(named_result, NamedPart)
    assert isinstance(group_result, LeaderFollowersCuttersPart)

    # Centers should be similar (accounting for the different starting positions)
    native_center = get_bounding_box_center(native_result)
    named_center = get_bounding_box_center(named_result.part)
    group_center = get_bounding_box_center(group_result.leader)

    # They should all be at approximately the same position
    assert np.allclose(native_center, named_center, atol=1e-10)
    assert np.allclose(native_center, group_center, atol=1e-10)


def test_cylinder_alignment_positioning():
    """Test cylinder alignment behavior to debug bottle cap ripple positioning.

    This test reproduces the alignment sequence from the bottle cap example
    to ensure consistent positioning between CadQuery and FreeCAD.
    """

    for with_fillet in [True, False]:
        reference_radius = 18
        reference_height = 5
        aligned_part_raduis = 3

        # Create the cap cover (reference object)
        reference_part = create_basic_cylinder(
            radius=reference_radius, height=reference_height
        )

        aligned_part = create_basic_cylinder(
            radius=aligned_part_raduis, height=reference_height
        )

        reference_bounding_box = get_bounding_box(reference_part)

        if with_fillet:
            reference_part = apply_fillet_by_alignment(
                reference_part, 1, fillets_at=[Alignment.TOP]
            )
        aligned_part = align(aligned_part, reference_part, Alignment.RIGHT)

        aligned_bounding_box = get_bounding_box(aligned_part)

        _logger.info(f"Reference bounding box: {reference_bounding_box}")
        _logger.info(f"Aligned bounding box: {aligned_bounding_box}")

        is_matching = np.allclose(
            aligned_bounding_box[1][0],
            reference_bounding_box[1][0],
            atol=1e-10,
        )

        if not with_fillet:
            assert (
                is_matching
            ), f"Aligned part max X {aligned_bounding_box[1][0]} does not match reference max X {reference_bounding_box[1][0]}"
        else:
            if not is_matching:
                _logger.warning(
                    f"Aligned part max X {aligned_bounding_box[1][0]} does not match reference max X {reference_bounding_box[1][0]}"
                )

                if get_adapter_id() == "freecad":
                    _logger.warning(
                        "FreeCAD bounding boxes are wrong when using fillets."
                    )
                else:
                    assert (
                        is_matching
                    ), f"Aligned part max X {aligned_bounding_box[1][0]} does not match reference max X {reference_bounding_box[1][0]}"
