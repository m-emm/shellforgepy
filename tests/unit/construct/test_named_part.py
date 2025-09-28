import numpy as np
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box,
    get_bounding_box_center,
)


def test_named_part_creation():
    """Test basic NamedPart creation."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test_box", part)

    assert named_part.name == "test_box"
    assert named_part.part is not None
    assert get_bounding_box(named_part.part) == get_bounding_box(part)


def test_named_part_copy():
    """Test NamedPart copying."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("original", part)

    copied_part = named_part.copy()

    assert copied_part.name == "original"
    assert copied_part.part is not named_part.part  # Should be a different object
    assert get_bounding_box(copied_part.part) == get_bounding_box(named_part.part)


def test_named_part_translate_method():
    """Test NamedPart.translate() method."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    original_center = get_bounding_box_center(named_part.part)
    translated_named_part = named_part.translate((5, 7, 13))
    translated_center = get_bounding_box_center(translated_named_part.part)

    assert translated_center == (
        original_center[0] + 5,
        original_center[1] + 7,
        original_center[2] + 13,
    )
    assert translated_named_part.name == "test"


def test_named_part_rotate_method():
    """Test NamedPart with functional rotate interface."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Use functional interface for framework-standardized parameters
    rotated_named_part = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(named_part)

    assert rotated_named_part is not None
    assert rotated_named_part.name == "test"

    bounding_box = get_bounding_box(rotated_named_part.part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(len_x, 20)  # X and Y dimensions swapped after 90° rotation
    assert np.allclose(len_y, 10)
    assert np.allclose(len_z, 30)


def test_named_part_vs_native_part_translate_consistency():
    """Test that translate()(named_part) behaves like translate()(native_part)."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Apply translation using the functional approach
    translated_native = translate(5, 7, 13)(part)
    translated_named = translate(5, 7, 13)(named_part)

    # Should get a NamedPart back
    assert isinstance(translated_named, NamedPart)
    assert translated_named.name == "test"

    # Centers should be the same
    native_center = get_bounding_box_center(translated_native)
    named_center = get_bounding_box_center(translated_named.part)

    assert np.allclose(native_center, named_center)


def test_named_part_vs_native_part_rotate_consistency():
    """Test that rotate()(named_part) behaves like rotate()(native_part)."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Apply rotation using the functional approach
    rotated_native = rotate(45, axis=(0, 0, 1), center=(0, 0, 0))(part)
    rotated_named = rotate(45, axis=(0, 0, 1), center=(0, 0, 0))(named_part)

    # Should get a NamedPart back
    assert isinstance(rotated_named, NamedPart)
    assert rotated_named.name == "test"

    # Bounding boxes should be equivalent
    native_bbox = get_bounding_box(rotated_native)
    named_bbox = get_bounding_box(rotated_named.part)

    assert np.allclose(native_bbox[0], named_bbox[0])  # min bounds
    assert np.allclose(native_bbox[1], named_bbox[1])  # max bounds


def test_named_part_chained_transformations():
    """Test chaining transformations on NamedPart."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Chain: translate then rotate
    result = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(
        translate(10, 0, 0)(named_part)
    )

    assert isinstance(result, NamedPart)
    assert result.name == "test"

    # The center should have moved according to translation then rotation
    # translate(10, 0, 0) moves center, then rotate(90°) around origin
    center = get_bounding_box_center(result.part)
    # After translate: center was at (original + (10, 0, 0))
    # After rotate 90°: (x, y) -> (-y, x)
    # The exact values depend on the original center, but we can verify it's not at origin
    assert not np.allclose(center, (0, 0, 0))


def test_named_part_parameter_order_consistency():
    """Test that NamedPart handles different parameter orders consistently."""
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Test different parameter orders for rotation
    result1 = rotate(45, axis=(0, 0, 1), center=(0, 0, 0))(named_part)
    result2 = rotate(45, center=(0, 0, 0), axis=(0, 0, 1))(named_part)

    # Both should give the same result
    bbox1 = get_bounding_box(result1.part)
    bbox2 = get_bounding_box(result2.part)

    assert np.allclose(bbox1[0], bbox2[0])  # min bounds
    assert np.allclose(bbox1[1], bbox2[1])  # max bounds
