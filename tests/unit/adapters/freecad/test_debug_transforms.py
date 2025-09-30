import numpy as np
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box_center,
    rotate,
    translate,
)


def test_debug_translation():
    """Debug what's happening with translations."""
    # Create parts
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Check original centers
    original_native_center = get_bounding_box_center(part)
    original_named_center = get_bounding_box_center(named_part.part)

    print(f"Original native center: {original_native_center}")
    print(f"Original named center: {original_named_center}")

    # Should be the same
    assert np.allclose(original_native_center, original_named_center)

    # Debug: Check if NamedPart.copy() works
    print(f"Can NamedPart.copy()? {hasattr(named_part, 'copy')}")
    try:
        copied_named = named_part.copy()
        print(f"NamedPart.copy() succeeded: {type(copied_named)}")
    except Exception as e:
        print(f"NamedPart.copy() failed: {e}")

    # Debug: Check what NamedPart.copy() returns
    copied_named = named_part.copy()
    print(f"NamedPart.copy() type: {type(copied_named)}")
    print(f"copied_named.part type: {type(copied_named.part)}")

    # Debug: Check what underlying part's copy returns
    underlying_copy = named_part.part.copy()
    print(f"named_part.part.copy() type: {type(underlying_copy)}")

    # Debug: Check what translate_part does to a NamedPart (using a fresh copy)
    from shellforgepy.adapters._adapter import translate_part

    fresh_named_part_1 = NamedPart("test", create_basic_box(10, 20, 30))
    try:
        translated_by_translate_part = translate_part(fresh_named_part_1, (5, 0, 0))
        print(
            f"translate_part on NamedPart succeeded: {type(translated_by_translate_part)}"
        )
        center_after_translate_part = get_bounding_box_center(
            translated_by_translate_part.part
        )
        print(f"Center after translate_part: {center_after_translate_part}")
    except Exception as e:
        print(f"translate_part on NamedPart failed: {e}")

    # Apply translation (using fresh objects to avoid interference)
    fresh_part = create_basic_box(10, 20, 30)
    fresh_named_part_2 = NamedPart("test", create_basic_box(10, 20, 30))
    translated_native = translate(5, 0, 0)(fresh_part)
    translated_named = translate(5, 0, 0)(fresh_named_part_2)  # Check results
    native_center = get_bounding_box_center(translated_native)
    named_center = get_bounding_box_center(translated_named.part)

    print(f"Translated native center: {native_center}")
    print(f"Translated named center: {named_center}")
    print(
        f"Expected: ({original_native_center[0] + 5}, {original_native_center[1]}, {original_native_center[2]})"
    )

    # Both should have moved by (5, 0, 0)
    expected_center = (
        original_native_center[0] + 5,
        original_native_center[1],
        original_native_center[2],
    )

    assert np.allclose(
        native_center, expected_center
    ), f"Native didn't translate correctly: {native_center} vs {expected_center}"
    assert np.allclose(
        named_center, expected_center
    ), f"Named didn't translate correctly: {named_center} vs {expected_center}"


def test_debug_rotation():
    """Debug what's happening with rotations."""
    # Create parts
    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Apply rotation
    rotated_native = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(part)
    rotated_named = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(named_part)

    # Check dimensions - after 90° rotation around Z, X and Y should swap
    from shellforgepy.simple import get_bounding_box

    native_bbox = get_bounding_box(rotated_native)
    named_bbox = get_bounding_box(rotated_named.part)

    native_dims = (
        native_bbox[1][0] - native_bbox[0][0],  # X size
        native_bbox[1][1] - native_bbox[0][1],  # Y size
        native_bbox[1][2] - native_bbox[0][2],  # Z size
    )

    named_dims = (
        named_bbox[1][0] - named_bbox[0][0],
        named_bbox[1][1] - named_bbox[0][1],
        named_bbox[1][2] - named_bbox[0][2],
    )

    print(f"Original box: 10x20x30")
    print(f"Native rotated dims: {native_dims}")
    print(f"Named rotated dims: {named_dims}")
    print(f"Expected after 90° Z rotation: ~20x10x30")

    # After 90° rotation around Z, X(10) and Y(20) should swap
    expected_dims = (20, 10, 30)  # X and Y swapped

    assert np.allclose(
        native_dims, expected_dims, atol=1e-10
    ), f"Native rotation failed: {native_dims}"
    assert np.allclose(
        named_dims, expected_dims, atol=1e-10
    ), f"Named rotation failed: {named_dims}"
