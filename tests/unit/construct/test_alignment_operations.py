import logging

import numpy as np
import pytest
from shellforgepy.simple import (
    Alignment,
    align,
    apply_fillet_by_alignment,
    create_box,
    create_cylinder,
    create_extruded_polygon,
    get_adapter_id,
    get_bounding_box,
    get_bounding_box_center,
    get_vertex_coordinates,
    mirror,
    rotate,
    scale,
    translate,
)

_logger = logging.getLogger(__name__)


def test_translate():

    part = create_box(10, 20, 30)
    part_center = get_bounding_box_center(part)

    part_translated = translate(5, 7, 13)(part)
    translated_center = get_bounding_box_center(part_translated)

    assert translated_center == (
        part_center[0] + 5,
        part_center[1] + 7,
        part_center[2] + 13,
    )


def test_rotate():

    part = create_box(10, 20, 30)

    rotated_part = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(part)

    assert rotated_part is not None

    bounding_box = get_bounding_box(rotated_part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(len_x, 20)
    assert np.allclose(len_y, 10)
    assert np.allclose(len_z, 30)


def test_scale_around_center():

    part = create_box(10, 20, 30)
    part_center = get_bounding_box_center(part)

    scaled_part = scale(2, center=part_center)(part)

    scaled_center = get_bounding_box_center(scaled_part)
    assert np.allclose(scaled_center, part_center)

    bounding_box = get_bounding_box(scaled_part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(len_x, 20)
    assert np.allclose(len_y, 40)
    assert np.allclose(len_z, 60)


def test_rotate_different_parameter_orders():
    """Test different ways of calling rotate to see if there are parameter order issues."""
    part = create_box(10, 20, 30)

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

    part = create_box(10, 20, 30)
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


def test_scale_consistency_with_named_parts():
    """Test that scale works consistently with NamedPart objects."""
    from shellforgepy.construct.named_part import NamedPart

    part = create_box(10, 20, 30)
    named_part = NamedPart("test", part)
    center = get_bounding_box_center(part)

    native_scaled = scale(1.5, center=center)(part)
    named_scaled = scale(1.5, center=center)(named_part)

    assert isinstance(named_scaled, NamedPart)

    native_center = get_bounding_box_center(native_scaled)
    named_center = get_bounding_box_center(named_scaled.part)
    assert np.allclose(native_center, named_center)


def test_functional_consistency_with_leader_followers():
    """Test that functional transformations work consistently with LeaderFollowersCuttersPart."""
    from shellforgepy.construct.leader_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    leader = create_box(2, 2, 2)
    follower = NamedPart("follower", create_box(1, 1, 1))
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


def test_scale_consistency_with_leader_followers():
    """Test that scale works consistently with LeaderFollowersCuttersPart."""
    from shellforgepy.construct.leader_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    leader = create_box(4, 6, 8)
    follower = NamedPart("follower", create_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])
    center = get_bounding_box_center(leader)

    scaled_group = scale(2, center=center)(group)

    assert isinstance(scaled_group, LeaderFollowersCuttersPart)

    scaled_leader_bb = get_bounding_box(scaled_group.leader)
    scaled_len = (
        scaled_leader_bb[1][0] - scaled_leader_bb[0][0],
        scaled_leader_bb[1][1] - scaled_leader_bb[0][1],
        scaled_leader_bb[1][2] - scaled_leader_bb[0][2],
    )
    assert np.allclose(scaled_len, (8, 12, 16))


def test_chained_transformations_consistency():
    """Test that chained transformations work consistently across all object types."""
    from shellforgepy.construct.leader_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    # Test with native part
    native_part = create_box(10, 10, 10)
    native_result = rotate(45, axis=(0, 0, 1))(translate(10, 0, 0)(native_part))

    # Test with NamedPart
    named_part = NamedPart("test", create_box(10, 10, 10))
    named_result = rotate(45, axis=(0, 0, 1))(translate(10, 0, 0)(named_part))

    # Test with LeaderFollowersCuttersPart
    group_part = LeaderFollowersCuttersPart(create_box(10, 10, 10))
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


@pytest.mark.parametrize(
    "alignment, axis, sign",
    [
        (Alignment.STACK_TOP, 2, 1),
        (Alignment.STACK_BOTTOM, 2, -1),
        (Alignment.STACK_RIGHT, 0, 1),
        (Alignment.STACK_LEFT, 0, -1),
    ],
)
def test_stack_alignment_positions_without_gap(alignment, axis, sign):
    """Stacking without gap should butt faces exactly together along the axis."""

    reference = create_box(10, 10, 10)
    part = create_box(5, 5, 5)

    stacked = align(part, reference, alignment)

    ref_min, ref_max = get_bounding_box(reference)
    stacked_min, stacked_max = get_bounding_box(stacked)

    if sign > 0:
        assert np.isclose(stacked_min[axis], ref_max[axis])
    else:
        assert np.isclose(stacked_max[axis], ref_min[axis])


@pytest.mark.parametrize(
    "alignment, axis, sign",
    [
        (Alignment.STACK_TOP, 2, 1),
        (Alignment.STACK_BOTTOM, 2, -1),
        (Alignment.STACK_RIGHT, 0, 1),
        (Alignment.STACK_LEFT, 0, -1),
    ],
)
def test_stack_alignment_positions_with_gap(alignment, axis, sign):
    """Stacking with a gap should offset by the given distance along the stack axis."""

    reference = create_box(10, 10, 10)
    part = create_box(5, 5, 5)
    gap = 1.5

    stacked = align(part, reference, alignment, stack_gap=gap)

    ref_min, ref_max = get_bounding_box(reference)
    stacked_min, stacked_max = get_bounding_box(stacked)

    if sign > 0:
        assert np.isclose(stacked_min[axis], ref_max[axis] + gap)
    else:
        assert np.isclose(stacked_max[axis], ref_min[axis] - gap)


def test_mirror_reflects_across_plane_without_mutation():
    """Mirror should produce a true reflection of asymmetric geometry without mutating the source."""

    mirror_normal = (1, 0, 0)
    mirror_point = (0, 0, 0)

    f_outline = [
        (0, 0),
        (3, 0),
        (3, 0.5),
        (1, 0.5),
        (1, 1.5),
        (2.5, 1.5),
        (2.5, 2.0),
        (1, 2.0),
        (1, 3.5),
        (3, 3.5),
        (3, 4.0),
        (0, 4.0),
    ]

    part = create_extruded_polygon(f_outline, thickness=2)

    original_vertices = get_vertex_coordinates(part)

    mirrored_part = mirror(normal=mirror_normal, point=mirror_point)(part)

    mirrored_vertices = get_vertex_coordinates(mirrored_part)

    def normalize(vertices):
        return {
            (
                round(x, 6),
                round(y, 6),
                round(z, 6),
            )
            for x, y, z in vertices
        }

    original_normalized = normalize(original_vertices)
    mirrored_normalized = normalize(mirrored_vertices)

    expected_reflection = {
        (round(2 * mirror_point[0] - x, 6), y, z) for x, y, z in original_normalized
    }

    assert mirrored_normalized == expected_reflection

    # The source geometry should remain unchanged by the mirror transform
    assert normalize(get_vertex_coordinates(part)) == original_normalized


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
        reference_part = create_cylinder(
            radius=reference_radius, height=reference_height
        )

        aligned_part = create_cylinder(
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
