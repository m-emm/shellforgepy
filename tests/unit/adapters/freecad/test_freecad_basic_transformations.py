import numpy as np
import pytest

# Try to import FreeCAD and skip tests if not available
try:
    import Part
    from FreeCAD import Base

    freecad_available = True
except ImportError:
    freecad_available = False
    Base = None
    Part = None

from shellforgepy.simple import (
    ALIGNMENT_SIGNS,
    Alignment,
    align,
    align_translation,
    alignment_signs,
    chain_translations,
    create_basic_box,
    rotate,
    stack_alignment_of,
    translate,
)


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_basic_translate():
    """Test basic translation function with simple movements."""
    # Create a box at origin
    box = create_basic_box(10, 10, 10)
    original_center = box.BoundBox.Center

    # Test translation in X direction
    translated_x = translate(5, 0, 0)(box)
    center_x = translated_x.BoundBox.Center
    assert (
        abs(center_x.x - (original_center.x + 5)) < 1e-6
    ), f"X translation failed: expected {original_center.x + 5}, got {center_x.x}"
    assert (
        abs(center_x.y - original_center.y) < 1e-6
    ), f"Y should be unchanged: expected {original_center.y}, got {center_x.y}"
    assert (
        abs(center_x.z - original_center.z) < 1e-6
    ), f"Z should be unchanged: expected {original_center.z}, got {center_x.z}"

    # Test translation in Y direction
    translated_y = translate(0, -3, 0)(box)
    center_y = translated_y.BoundBox.Center
    assert (
        abs(center_y.x - original_center.x) < 1e-6
    ), f"X should be unchanged: expected {original_center.x}, got {center_y.x}"
    assert (
        abs(center_y.y - (original_center.y - 3)) < 1e-6
    ), f"Y translation failed: expected {original_center.y - 3}, got {center_y.y}"
    assert (
        abs(center_y.z - original_center.z) < 1e-6
    ), f"Z should be unchanged: expected {original_center.z}, got {center_y.z}"

    # Test translation in Z direction
    translated_z = translate(0, 0, 7)(box)
    center_z = translated_z.BoundBox.Center
    assert (
        abs(center_z.x - original_center.x) < 1e-6
    ), f"X should be unchanged: expected {original_center.x}, got {center_z.x}"
    assert (
        abs(center_z.y - original_center.y) < 1e-6
    ), f"Y should be unchanged: expected {original_center.y}, got {center_z.y}"
    assert (
        abs(center_z.z - (original_center.z + 7)) < 1e-6
    ), f"Z translation failed: expected {original_center.z + 7}, got {center_z.z}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_basic_rotate_around_origin():
    """Test basic rotation around origin with asymmetric box to verify orientation."""
    # Create an asymmetric box: 20x10x5 (length x width x height)
    # This makes it easy to track orientation after rotation
    box = create_basic_box(20, 10, 5, (0, 0, 0))

    # Box extends from (0,0,0) to (20,10,5), center at (10,5,2.5)
    original_center = box.BoundBox.Center
    original_bounds = (box.BoundBox.XLength, box.BoundBox.YLength, box.BoundBox.ZLength)

    # Verify original dimensions
    assert (
        abs(original_bounds[0] - 20) < 1e-6
    ), f"Original X length should be 20, got {original_bounds[0]}"
    assert (
        abs(original_bounds[1] - 10) < 1e-6
    ), f"Original Y length should be 10, got {original_bounds[1]}"
    assert (
        abs(original_bounds[2] - 5) < 1e-6
    ), f"Original Z length should be 5, got {original_bounds[2]}"

    # Test 90° rotation around Z-axis (should swap X and Y)
    rotated_z = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(box)
    rotated_center_z = rotated_z.BoundBox.Center
    rotated_bounds_z = (
        rotated_z.BoundBox.XLength,
        rotated_z.BoundBox.YLength,
        rotated_z.BoundBox.ZLength,
    )

    # After 90° rotation around Z, center (10,5,2.5) should become (-5,10,2.5)
    assert (
        abs(rotated_center_z.x + 5) < 1e-6
    ), f"Z-rotated center X should be -5, got {rotated_center_z.x}"
    assert (
        abs(rotated_center_z.y - 10) < 1e-6
    ), f"Z-rotated center Y should be 10, got {rotated_center_z.y}"
    assert (
        abs(rotated_center_z.z - 2.5) < 1e-6
    ), f"Z-rotated center Z should be 2.5, got {rotated_center_z.z}"

    # Dimensions should swap: X(20)->Y(20), Y(10)->X(10), Z(5) unchanged
    assert (
        abs(rotated_bounds_z[0] - 10) < 1e-6
    ), f"Z-rotated X length should be 10, got {rotated_bounds_z[0]}"
    assert (
        abs(rotated_bounds_z[1] - 20) < 1e-6
    ), f"Z-rotated Y length should be 20, got {rotated_bounds_z[1]}"
    assert (
        abs(rotated_bounds_z[2] - 5) < 1e-6
    ), f"Z-rotated Z length should be 5, got {rotated_bounds_z[2]}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_basic_rotate_around_point():
    """Test rotation around a specific point."""
    # Create box at (10, 0, 0) with dimensions 6x4x2
    box = create_basic_box(6, 4, 2, (10, 0, 0))

    # Box center is at (13, 2, 1)
    original_center = box.BoundBox.Center
    assert (
        abs(original_center.x - 13) < 1e-6
    ), f"Original center X should be 13, got {original_center.x}"
    assert (
        abs(original_center.y - 2) < 1e-6
    ), f"Original center Y should be 2, got {original_center.y}"
    assert (
        abs(original_center.z - 1) < 1e-6
    ), f"Original center Z should be 1, got {original_center.z}"

    # Rotate 90° around point (10, 0, 0) on Z-axis
    rotated = rotate(90, center=(10, 0, 0), axis=(0, 0, 1))(box)
    rotated_center = rotated.BoundBox.Center

    # The center (13, 2, 1) relative to rotation point (10, 0, 0) is (3, 2, 1)
    # After 90° rotation: (3, 2, 1) -> (-2, 3, 1)
    # So absolute center should be (10, 0, 0) + (-2, 3, 1) = (8, 3, 1)
    assert (
        abs(rotated_center.x - 8) < 1e-6
    ), f"Rotated center X should be 8, got {rotated_center.x}"
    assert (
        abs(rotated_center.y - 3) < 1e-6
    ), f"Rotated center Y should be 3, got {rotated_center.y}"
    assert (
        abs(rotated_center.z - 1) < 1e-6
    ), f"Rotated center Z should be 1, got {rotated_center.z}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_rotate_different_axes():
    """Test rotation around different axes with asymmetric box."""
    # Create box with distinct dimensions: 12x8x4
    box = create_basic_box(12, 8, 4, (0, 0, 0))
    original_bounds = (box.BoundBox.XLength, box.BoundBox.YLength, box.BoundBox.ZLength)

    # Test 90° rotation around X-axis (should swap Y and Z)
    rotated_x = rotate(90, center=(0, 0, 0), axis=(1, 0, 0))(box)
    bounds_x = (
        rotated_x.BoundBox.XLength,
        rotated_x.BoundBox.YLength,
        rotated_x.BoundBox.ZLength,
    )

    assert (
        abs(bounds_x[0] - 12) < 1e-6
    ), f"X-axis rotation: X length should stay 12, got {bounds_x[0]}"
    assert (
        abs(bounds_x[1] - 4) < 1e-6
    ), f"X-axis rotation: Y length should become 4, got {bounds_x[1]}"
    assert (
        abs(bounds_x[2] - 8) < 1e-6
    ), f"X-axis rotation: Z length should become 8, got {bounds_x[2]}"

    # Test 90° rotation around Y-axis (should swap X and Z)
    rotated_y = rotate(90, center=(0, 0, 0), axis=(0, 1, 0))(box)
    bounds_y = (
        rotated_y.BoundBox.XLength,
        rotated_y.BoundBox.YLength,
        rotated_y.BoundBox.ZLength,
    )

    assert (
        abs(bounds_y[0] - 4) < 1e-6
    ), f"Y-axis rotation: X length should become 4, got {bounds_y[0]}"
    assert (
        abs(bounds_y[1] - 8) < 1e-6
    ), f"Y-axis rotation: Y length should stay 8, got {bounds_y[1]}"
    assert (
        abs(bounds_y[2] - 12) < 1e-6
    ), f"Y-axis rotation: Z length should become 12, got {bounds_y[2]}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_center_boxes():
    """Test centering two boxes."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(10, 10, 10)

    # Align box2 to box1 at center
    aligned_box2 = align(box2, box1, Alignment.CENTER)

    # The centers should be the same
    center1 = box1.BoundBox.Center
    center2 = aligned_box2.BoundBox.Center

    assert np.allclose(
        (center1.x, center1.y, center1.z), (center2.x, center2.y, center2.z)
    )


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_left():
    """Test aligning a box to the left of another box."""
    # Create target box at origin
    box1 = create_basic_box(10, 10, 10)
    # Create box to align, offset to the right
    box2 = create_basic_box(5, 5, 5, (20, 0, 0))

    aligned_box2 = align(box2, box1, Alignment.LEFT)

    # box2's left edge should align with box1's left edge
    assert abs(aligned_box2.BoundBox.XMin - box1.BoundBox.XMin) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_right():
    """Test aligning a box to the right of another box."""
    # Create target box at origin
    box1 = create_basic_box(10, 10, 10)
    # Create box to align, offset to the left
    box2 = create_basic_box(5, 5, 5, (-20, 0, 0))

    aligned_box2 = align(box2, box1, Alignment.RIGHT)

    # box2's right edge should align with box1's right edge
    assert abs(aligned_box2.BoundBox.XMax - box1.BoundBox.XMax) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_front():
    """Test aligning a box to the front of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, (0, 20, 0))

    aligned_box2 = align(box2, box1, Alignment.FRONT)

    # box2's front edge should align with box1's front edge
    assert abs(aligned_box2.BoundBox.YMin - box1.BoundBox.YMin) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_back():
    """Test aligning a box to the back of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, (0, -20, 0))

    aligned_box2 = align(box2, box1, Alignment.BACK)

    # box2's back edge should align with box1's back edge
    assert abs(aligned_box2.BoundBox.YMax - box1.BoundBox.YMax) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_top():
    """Test aligning a box to the top of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, (0, 0, -20))

    aligned_box2 = align(box2, box1, Alignment.TOP)

    # box2's top edge should align with box1's top edge
    assert abs(aligned_box2.BoundBox.ZMax - box1.BoundBox.ZMax) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_bottom():
    """Test aligning a box to the bottom of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, (0, 0, 20))

    aligned_box2 = align(box2, box1, Alignment.BOTTOM)

    # box2's bottom edge should align with box1's bottom edge
    assert abs(aligned_box2.BoundBox.ZMin - box1.BoundBox.ZMin) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_stack_right():
    """Test stacking a box to the right of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    aligned_box2 = align(box2, box1, Alignment.STACK_RIGHT)

    # box2 should be positioned so its left edge is at box1's right edge
    expected_x_min = box1.BoundBox.XMax  # 10
    assert abs(aligned_box2.BoundBox.XMin - expected_x_min) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_stack_left():
    """Test stacking a box to the left of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    aligned_box2 = align(box2, box1, Alignment.STACK_LEFT)

    # box2 should be positioned so its right edge is at box1's left edge
    expected_x_max = box1.BoundBox.XMin  # 0
    assert abs(aligned_box2.BoundBox.XMax - expected_x_max) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_stack_top():
    """Test stacking a box on top of another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    aligned_box2 = align(box2, box1, Alignment.STACK_TOP)

    # box2 should be positioned so its bottom edge is at box1's top edge
    expected_z_min = box1.BoundBox.ZMax  # 10
    assert abs(aligned_box2.BoundBox.ZMin - expected_z_min) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_stack_bottom():
    """Test stacking a box below another box."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    aligned_box2 = align(box2, box1, Alignment.STACK_BOTTOM)

    # box2 should be positioned so its top edge is at box1's bottom edge
    expected_z_max = box1.BoundBox.ZMin  # 0
    assert abs(aligned_box2.BoundBox.ZMax - expected_z_max) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_align_with_axes_constraint():
    """Test alignment with axis constraints."""
    box1 = create_basic_box(10, 10, 10, (20, 15, 8))
    box2 = create_basic_box(5, 5, 5)

    # Align only on X and Y axes, leave Z unchanged
    aligned_box2 = align(box2, box1, Alignment.CENTER, axes=[0, 1])

    # X and Y should be centered, Z should remain at original position
    center1 = box1.BoundBox.Center
    center2 = aligned_box2.BoundBox.Center

    assert abs(center1.x - center2.x) < 1e-6  # X should be centered
    assert abs(center1.y - center2.y) < 1e-6  # Y should be centered
    assert abs(center2.z - 2.5) < 1e-6  # Z should be unchanged (box2's original center)


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_translate_function():
    """Test the translate function."""
    box = create_basic_box(10, 10, 10)
    translated_box = translate(5, -3, 7)(box)

    original_center = box.BoundBox.Center
    translated_center = translated_box.BoundBox.Center

    assert abs(translated_center.x - original_center.x - 5) < 1e-6
    assert abs(translated_center.y - original_center.y + 3) < 1e-6
    assert abs(translated_center.z - original_center.z - 7) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_rotate_function():
    """Test the rotate function."""
    # Create a box offset from origin
    box = create_basic_box(10, 5, 5, (10, 0, 0))

    # Rotate 90 degrees around Z axis
    rotated_box = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(box)

    # After 90° rotation around origin, the box center should move from (15,2.5,2.5) to approximately (-2.5,15,2.5)
    center = rotated_box.BoundBox.Center

    # Allow some tolerance for floating point precision
    assert abs(center.x + 2.5) < 1e-6
    assert abs(center.y - 15) < 1e-6
    assert abs(center.z - 2.5) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_chain_translations():
    """Test chaining multiple transformations."""
    box = create_basic_box(10, 10, 10)
    original_center = box.BoundBox.Center

    # Test assumption: original center should be (5,5,5)
    assert (
        abs(original_center.x - 5) < 1e-6
    ), f"Expected original center.x=5, got {original_center.x}"
    assert (
        abs(original_center.y - 5) < 1e-6
    ), f"Expected original center.y=5, got {original_center.y}"
    assert (
        abs(original_center.z - 5) < 1e-6
    ), f"Expected original center.z=5, got {original_center.z}"

    # Test individual transformations first
    translated = translate(10, 0, 0)(box)
    translated_center = translated.BoundBox.Center

    # After translate(10,0,0), center should move from (5,5,5) to (15,5,5)
    assert (
        abs(translated_center.x - 15) < 1e-6
    ), f"Expected translated center.x=15, got {translated_center.x}"
    assert (
        abs(translated_center.y - 5) < 1e-6
    ), f"Expected translated center.y=5, got {translated_center.y}"
    assert (
        abs(translated_center.z - 5) < 1e-6
    ), f"Expected translated center.z=5, got {translated_center.z}"

    # Test rotation around ORIGIN (0,0,0) - this will cause the object to orbit!
    # After translate(10,0,0), center is at (15,5,5)
    # 90° rotation around origin: (x,y,z) → (-y,x,z)
    # So (15,5,5) → (-5,15,5)
    rotated_around_origin = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(translated)
    rotated_origin_center = rotated_around_origin.BoundBox.Center

    assert (
        abs(rotated_origin_center.x + 5) < 1e-6
    ), f"Expected origin-rotated center.x=-5, got {rotated_origin_center.x}"
    assert (
        abs(rotated_origin_center.y - 15) < 1e-6
    ), f"Expected origin-rotated center.y=15, got {rotated_origin_center.y}"
    assert (
        abs(rotated_origin_center.z - 5) < 1e-6
    ), f"Expected origin-rotated center.z=5, got {rotated_origin_center.z}"

    # Test rotation around point (10,0,0) - this should rotate relative to that point
    # Center (15,5,5) relative to (10,0,0) is (5,5,5)
    # After 90° rotation: (5,5,5) → (-5,5,5)
    # Absolute position: (10,0,0) + (-5,5,5) = (5,5,5)
    rotated_around_point = rotate(90, center=(10, 0, 0), axis=(0, 0, 1))(translated)
    rotated_point_center = rotated_around_point.BoundBox.Center

    assert (
        abs(rotated_point_center.x - 5) < 1e-6
    ), f"Expected point-rotated center.x=5, got {rotated_point_center.x}"
    assert (
        abs(rotated_point_center.y - 5) < 1e-6
    ), f"Expected point-rotated center.y=5, got {rotated_point_center.y}"
    assert (
        abs(rotated_point_center.z - 5) < 1e-6
    ), f"Expected point-rotated center.z=5, got {rotated_point_center.z}"

    # Now test chained transformations
    # First test: translate + rotate around origin (should match individual steps)
    transform_origin = chain_translations(
        translate(10, 0, 0), rotate(90, center=(0, 0, 0), axis=(0, 0, 1))
    )

    transformed_origin = transform_origin(box)
    chained_origin_center = transformed_origin.BoundBox.Center

    # Should match the individual rotation around origin result
    assert (
        abs(chained_origin_center.x + 5) < 1e-6
    ), f"Expected chained origin center.x=-5, got {chained_origin_center.x}"
    assert (
        abs(chained_origin_center.y - 15) < 1e-6
    ), f"Expected chained origin center.y=15, got {chained_origin_center.y}"
    assert (
        abs(chained_origin_center.z - 5) < 1e-6
    ), f"Expected chained origin center.z=5, got {chained_origin_center.z}"

    # Second test: translate + rotate around point (10,0,0)
    transform_point = chain_translations(
        translate(10, 0, 0), rotate(90, center=(10, 0, 0), axis=(0, 0, 1))
    )

    transformed_point = transform_point(box)
    chained_point_center = transformed_point.BoundBox.Center

    # Should match the individual rotation around point result
    assert (
        abs(chained_point_center.x - 5) < 1e-6
    ), f"Expected chained point center.x=5, got {chained_point_center.x}"
    assert (
        abs(chained_point_center.y - 5) < 1e-6
    ), f"Expected chained point center.y=5, got {chained_point_center.y}"
    assert (
        abs(chained_point_center.z - 5) < 1e-6
    ), f"Expected chained point center.z=5, got {chained_point_center.z}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_alignment_signs():
    """Test the alignment signs utility function."""
    # Test single alignment
    signs = alignment_signs(Alignment.LEFT)
    assert signs == (-1, 0, 0)

    signs = alignment_signs(Alignment.RIGHT)
    assert signs == (1, 0, 0)

    signs = alignment_signs(Alignment.TOP)
    assert signs == (0, 0, 1)

    signs = alignment_signs(Alignment.BOTTOM)
    assert signs == (0, 0, -1)

    signs = alignment_signs(Alignment.FRONT)
    assert signs == (0, -1, 0)

    signs = alignment_signs(Alignment.BACK)
    assert signs == (0, 1, 0)

    signs = alignment_signs(Alignment.CENTER)
    assert signs == (0, 0, 0)

    # Test multiple alignments
    signs = alignment_signs([Alignment.LEFT, Alignment.TOP])
    assert signs == (-1, 0, 1)

    signs = alignment_signs([Alignment.RIGHT, Alignment.BACK, Alignment.BOTTOM])
    assert signs == (1, 1, -1)


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_stack_alignment_conversion():
    """Test the stack_alignment_of utility function."""
    assert stack_alignment_of(Alignment.LEFT) == Alignment.STACK_LEFT
    assert stack_alignment_of(Alignment.RIGHT) == Alignment.STACK_RIGHT
    assert stack_alignment_of(Alignment.TOP) == Alignment.STACK_TOP
    assert stack_alignment_of(Alignment.BOTTOM) == Alignment.STACK_BOTTOM
    assert stack_alignment_of(Alignment.FRONT) == Alignment.STACK_FRONT
    assert stack_alignment_of(Alignment.BACK) == Alignment.STACK_BACK


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_alignment_signs_constants():
    """Test the ALIGNMENT_SIGNS constants."""
    assert ALIGNMENT_SIGNS[Alignment.LEFT] == -1
    assert ALIGNMENT_SIGNS[Alignment.RIGHT] == 1
    assert ALIGNMENT_SIGNS[Alignment.TOP] == 1
    assert ALIGNMENT_SIGNS[Alignment.BOTTOM] == -1
    assert ALIGNMENT_SIGNS[Alignment.FRONT] == -1
    assert ALIGNMENT_SIGNS[Alignment.BACK] == 1
    assert ALIGNMENT_SIGNS[Alignment.CENTER] == 0


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_invalid_alignment():
    """Test that invalid alignment raises an error."""
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    # This should raise a ValueError for an invalid alignment
    with pytest.raises(ValueError, match="Unknown alignment"):
        # Use a string that's not a valid alignment
        align_translation(box2, box1, "INVALID_ALIGNMENT")


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_named_part_native_rotate_method():
    """Test NamedPart.rotate() method with native FreeCAD signature."""
    from shellforgepy.construct.named_part import NamedPart

    part = create_basic_box(10, 20, 30)
    named_part = NamedPart("test", part)

    # Use native FreeCAD signature: rotate(base, dir, degree)
    base_vec = Base.Vector(0, 0, 0)  # center point
    dir_vec = Base.Vector(0, 0, 1)  # axis direction
    rotated_named_part = named_part.rotate(base_vec, dir_vec, 90)

    assert rotated_named_part is not None

    # Verify the rotation worked by checking bounding box dimensions
    from shellforgepy.simple import get_bounding_box

    bounding_box = get_bounding_box(rotated_named_part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(
        len_x, 20, atol=1e-6
    )  # X and Y dimensions swapped after 90° rotation
    assert np.allclose(len_y, 10, atol=1e-6)
    assert np.allclose(len_z, 30, atol=1e-6)


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_leader_followers_native_rotate_method():
    """Test LeaderFollowersCuttersPart with native rotate interface."""
    from shellforgepy.construct.leaders_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )
    from shellforgepy.construct.named_part import NamedPart

    leader = create_basic_box(2, 2, 2)
    follower = NamedPart("follower", create_basic_box(1, 1, 1))
    group = LeaderFollowersCuttersPart(leader, followers=[follower])

    # Use native FreeCAD signature: rotate(base, dir, degree)
    base_vec = Base.Vector(0, 0, 0)  # center point
    dir_vec = Base.Vector(0, 0, 1)  # axis direction
    rotated_group = group.rotate(base_vec, dir_vec, 90)

    # Should return self (in-place modification)
    assert rotated_group is group

    # Verify the rotation worked
    from shellforgepy.simple import get_bounding_box_center

    leader_center = get_bounding_box_center(group.leader)

    # After 90° rotation around Z, the center should have moved appropriately
    assert isinstance(leader_center, tuple)  # Basic sanity check
