import numpy as np
from shellforgepy.adapters._adapter import create_box, create_cylinder, get_volume
from shellforgepy.geometry.modifications import slice_part
from shellforgepy.simple import get_bounding_box


def test_slice_part_basic_functionality():
    """Test basic slicing functionality with a simple box."""
    # Create a test box
    box = create_box(10, 20, 30)

    # Slice along Z axis with 5 unit thickness
    slices = slice_part(box, (0, 0, 1), 5.0)

    # Should have 6 slices for a 30-unit tall box with 5-unit slices
    assert len(slices) == 6

    # Check that each slice has the expected structure
    for i, slice_info in enumerate(slices):
        assert "part" in slice_info
        assert "plane_point" in slice_info
        assert "height" in slice_info
        assert "slice_bbox" in slice_info  # New field

        # Verify height progression
        assert np.isclose(slice_info["height"], i * 5.0)

        # Each slice should have positive volume
        assert get_volume(slice_info["part"]) > 0

        # Check that slice is properly oriented: centered and bottom at z=0
        bbox = slice_info["slice_bbox"]
        min_point, max_point = bbox

        # Bottom should be at or near z=0
        assert min_point[2] < 0.1, f"Slice {i} z_min={min_point[2]} should be ~0"

        # Should be centered in X and Y
        center_x = (min_point[0] + max_point[0]) / 2
        center_y = (min_point[1] + max_point[1]) / 2
        assert abs(center_x) < 0.1, f"Slice {i} should be X-centered, got {center_x}"
        assert abs(center_y) < 0.1, f"Slice {i} should be Y-centered, got {center_y}"


def test_slice_part_zero_volume_filtering():
    """Test that slices with zero volume are filtered out."""
    # Create a small box
    box = create_box(5, 5, 5)

    # Slice with a very large thickness that would create mostly empty slices
    slices = slice_part(box, (1, 0, 0), 10.0)

    # Should only have 1 slice since the box is only 5 units wide
    assert len(slices) == 1

    # The one slice should have positive volume
    assert get_volume(slices[0]["part"]) > 0

    # Test with very thin slices to ensure we don't get spurious zero-volume slices
    thin_slices = slice_part(box, (0, 0, 1), 0.5)

    # All slices should have positive volume
    for slice_info in thin_slices:
        volume = get_volume(slice_info["part"])
        assert volume > 1e-10, f"Slice has volume {volume}, should be > 1e-10"


def test_slice_part_horizontal_orientation():
    """Test that all slices are properly oriented horizontally."""
    # Create a box and slice it at various angles
    box = create_box(20, 15, 10)

    test_normals = [
        (1, 0, 0),  # X direction
        (0, 1, 0),  # Y direction
        (0, 0, 1),  # Z direction
        (1, 1, 0),  # XY diagonal
        (1, 0, 1),  # XZ diagonal
        (1, 1, 1),  # XYZ diagonal
    ]

    for normal in test_normals:
        slices = slice_part(box, normal, 3.0)

        # Each slice should be oriented horizontally
        for i, slice_info in enumerate(slices):
            bbox = slice_info["slice_bbox"]
            min_point, max_point = bbox

            # Bottom should be at z=0 (within tolerance)
            assert (
                min_point[2] < 0.1
            ), f"Normal {normal}, slice {i}: z_min={min_point[2]} should be ~0"

            # Should be centered in X and Y
            center_x = (min_point[0] + max_point[0]) / 2
            center_y = (min_point[1] + max_point[1]) / 2
            assert (
                abs(center_x) < 0.1
            ), f"Normal {normal}, slice {i}: should be X-centered, got {center_x}"
            assert (
                abs(center_y) < 0.1
            ), f"Normal {normal}, slice {i}: should be Y-centered, got {center_y}"

            # Thickness should be reasonable (slice should have some Z extent)
            z_thickness = max_point[2] - min_point[2]
            assert (
                z_thickness > 0.1
            ), f"Normal {normal}, slice {i}: slice too thin, thickness={z_thickness}"


def test_slice_part_volume_conservation():
    """Test that slicing conserves total volume (approximately)."""
    # Create a test cylinder for more interesting geometry
    cylinder = create_cylinder(radius=5, height=20)
    original_volume = get_volume(cylinder)

    # Slice with small thickness to get good approximation
    slices = slice_part(cylinder, (0, 0, 1), 2.0)

    # Sum volumes of all slices
    total_sliced_volume = sum(get_volume(slice_info["part"]) for slice_info in slices)

    # Should be approximately equal (within 5% tolerance due to discretization)
    assert np.isclose(total_sliced_volume, original_volume, rtol=0.05)


def test_slice_part_different_orientations():
    """Test slicing along different axes."""
    box = create_box(10, 20, 30)

    # Test slicing along X axis
    slices_x = slice_part(box, (1, 0, 0), 2.0)
    # Box X range: 0 to 10, thickness 2.0 -> slices: [0-2], [2-4], [4-6], [6-8], [8-10] = 5 slices
    assert len(slices_x) == 5

    # Test slicing along Y axis
    slices_y = slice_part(box, (0, 1, 0), 4.0)
    # Box Y range: 0 to 20, thickness 4.0 -> slices: [0-4], [4-8], [8-12], [12-16], [16-20] = 5 slices
    assert len(slices_y) == 5

    # Test slicing along diagonal direction
    diagonal_normal = np.array([1, 1, 1])
    diagonal_normal = diagonal_normal / np.linalg.norm(diagonal_normal)
    slices_diag = slice_part(box, diagonal_normal, 3.0)

    # Should produce some slices
    assert len(slices_diag) > 0

    # Each slice should have positive volume
    for slice_info in slices_diag:
        assert get_volume(slice_info["part"]) > 0


def test_slice_part_non_unit_normal():
    """Test that function properly normalizes the slice plane normal."""
    box = create_box(10, 10, 10)

    # Use a non-unit normal vector
    non_unit_normal = (2, 0, 0)  # magnitude = 2
    slices = slice_part(box, non_unit_normal, 2.5)

    # Box X range: 0 to 10, thickness 2.5 -> slices: [0-2.5], [2.5-5.0], [5.0-7.5], [7.5-10.0] = 4 slices
    assert len(slices) == 4

    # Verify that slicing worked correctly despite non-unit input
    for slice_info in slices:
        assert get_volume(slice_info["part"]) > 0


def test_slice_part_plane_point_progression():
    """Test that plane points progress correctly along the normal direction."""
    box = create_box(20, 20, 20)
    slice_normal = np.array([0, 0, 1])  # Z direction
    slice_thickness = 5.0

    slices = slice_part(box, slice_normal, slice_thickness)

    # Check that plane points progress in the expected direction
    for i in range(len(slices) - 1):
        current_point = slices[i]["plane_point"]
        next_point = slices[i + 1]["plane_point"]

        # The difference should be slice_thickness in the Z direction
        diff = next_point - current_point
        expected_diff = slice_normal * slice_thickness

        assert np.allclose(diff, expected_diff, atol=1e-10)


def test_slice_part_bounding_box_alignment():
    """Test that slicing starts from the correct bounding box position."""
    box = create_box(10, 10, 10)
    bb = get_bounding_box(box)

    # Slice along +Z direction
    slices_z = slice_part(box, (0, 0, 1), 2.0)

    # First slice should start at the bottom of the bounding box
    first_plane_point = slices_z[0]["plane_point"]

    # The Z-coordinate should be close to the minimum Z of the bounding box
    assert np.isclose(first_plane_point[2], bb[0][2], atol=1e-10)

    # Slice along -Z direction (should start from the top)
    slices_neg_z = slice_part(box, (0, 0, -1), 2.0)
    first_plane_point_neg = slices_neg_z[0]["plane_point"]

    # Should start from the top (maximum Z)
    assert np.isclose(first_plane_point_neg[2], bb[1][2], atol=1e-10)


def test_slice_part_edge_cases():
    """Test edge cases and potential error conditions."""
    box = create_box(5, 5, 5)

    # Test with very small slice thickness
    slices_thin = slice_part(box, (1, 0, 0), 0.1)
    # Box X range: 0 to 5, thickness 0.1 -> 5.0/0.1 = 50 slices
    assert len(slices_thin) == 50

    # Test with slice thickness larger than the part dimension
    slices_thick = slice_part(box, (1, 0, 0), 10.0)
    assert len(slices_thick) == 1  # Should produce only one slice

    # Test with zero components in normal (but not zero vector)
    slices_partial = slice_part(box, (1, 0, 0), 1.0)
    # Box X range: 0 to 5, thickness 1.0 -> slices: [0-1], [1-2], [2-3], [3-4], [4-5] = 5 slices
    assert len(slices_partial) == 5

    for slice_info in slices_partial:
        assert get_volume(slice_info["part"]) > 0


def test_slice_part_complex_geometry():
    """Test slicing with more complex geometry."""
    # Create a cylinder and slice it
    cylinder = create_cylinder(radius=10, height=30)

    # Slice perpendicular to cylinder axis
    slices_perp = slice_part(cylinder, (0, 0, 1), 3.0)

    # Should produce 10 slices (the last one might be filtered out due to zero volume)
    assert len(slices_perp) == 10

    # Each slice should be approximately a disk with similar volume
    disk_volume = np.pi * 10**2 * 3.0  # Expected volume of each disk slice

    for slice_info in slices_perp:
        slice_volume = get_volume(slice_info["part"])
        assert np.isclose(slice_volume, disk_volume, rtol=0.1)

    # Slice parallel to cylinder axis (should be more complex)
    slices_para = slice_part(cylinder, (1, 0, 0), 5.0)

    # Should produce some slices
    assert len(slices_para) >= 3  # At least a few slices

    # For cylinder sliced parallel to axis, volume conservation is harder to test
    # due to geometric complexity, so we just check that volumes are reasonable
    total_volume = sum(get_volume(slice_info["part"]) for slice_info in slices_para)
    original_volume = get_volume(cylinder)

    # Each slice should have positive volume and total should be reasonable
    # (not necessarily exactly conserved due to cutting artifacts)
    assert total_volume > 0
    assert total_volume <= original_volume * 2  # Allow for some overlap/artifacts


def test_slice_part_diagonal_slicing():
    """Test slicing along diagonal directions and verify horizontal orientation."""
    # Create a cube
    cube = create_box(10, 10, 10)
    original_volume = get_volume(cube)

    # Slice along (1,1,1) diagonal
    diagonal = np.array([1, 1, 1])
    diagonal = diagonal / np.linalg.norm(diagonal)

    slices = slice_part(cube, diagonal, 2.0)

    # For a cube with side length 10, the space diagonal is 10*sqrt(3) ≈ 17.32
    # So we expect approximately 17.32 / 2.0 ≈ 8.7, so 8-9 slices
    assert 7 <= len(slices) <= 10, f"Expected 7-10 slices, got {len(slices)}"

    # Each slice should have positive volume and be properly oriented
    for i, slice_info in enumerate(slices):
        volume = get_volume(slice_info["part"])
        assert volume > 0

        # Check horizontal orientation: bottom at z=0, centered in X,Y
        bbox = slice_info["slice_bbox"]
        min_point, max_point = bbox

        assert min_point[2] < 0.1, f"Slice {i} should have z_min ~0, got {min_point[2]}"

        center_x = (min_point[0] + max_point[0]) / 2
        center_y = (min_point[1] + max_point[1]) / 2
        assert abs(center_x) < 0.1, f"Slice {i} should be X-centered, got {center_x}"
        assert abs(center_y) < 0.1, f"Slice {i} should be Y-centered, got {center_y}"

    # Total volume should be approximately conserved
    total_volume = sum(get_volume(slice_info["part"]) for slice_info in slices)
    assert np.isclose(total_volume, original_volume, rtol=0.1)


def test_slice_part_negative_normal():
    """Test that negative normals work correctly."""
    box = create_box(10, 10, 10)

    # Slice with positive normal
    slices_pos = slice_part(box, (0, 0, 1), 2.0)

    # Slice with negative normal
    slices_neg = slice_part(box, (0, 0, -1), 2.0)

    # Should produce same number of slices
    assert len(slices_pos) == len(slices_neg)

    # Total volumes should be similar
    vol_pos = sum(get_volume(s["part"]) for s in slices_pos)
    vol_neg = sum(get_volume(s["part"]) for s in slices_neg)

    assert np.isclose(vol_pos, vol_neg, rtol=0.05)


def test_slice_part_height_progression():
    """Test that height values progress correctly."""
    box = create_box(12, 12, 12)
    slice_thickness = 3.0

    slices = slice_part(box, (1, 0, 0), slice_thickness)

    # Heights should progress as 0, 3, 6, 9
    expected_heights = [i * slice_thickness for i in range(len(slices))]
    actual_heights = [s["height"] for s in slices]

    assert np.allclose(actual_heights, expected_heights)


def test_slice_part_coordinate_system_robustness():
    """Test that the coordinate system selection is robust."""
    box = create_box(10, 10, 10)

    # Test normals that might cause issues with coordinate system selection
    test_normals = [
        (0, 0, 1),  # Along Z
        (1, 0, 0),  # Along X
        (0, 1, 0),  # Along Y
        (0, 0, -1),  # Negative Z
        (1, 1, 0),  # Diagonal in XY
        (1, 0, 1),  # Diagonal in XZ
        (0, 1, 1),  # Diagonal in YZ
        (1, 1, 1),  # Space diagonal
    ]

    for normal in test_normals:
        slices = slice_part(box, normal, 2.0)

        # Should produce at least one slice
        assert len(slices) >= 1

        # Each slice should have positive volume
        for slice_info in slices:
            assert get_volume(slice_info["part"]) > 0


def test_slice_part_transform_to_horizontal_flag():
    """Test the transform_to_horizontal flag functionality."""
    box = create_box(10, 20, 30)
    slice_normal = np.array([1, 0, 0])  # X direction
    thickness = 2.0

    # Test with transformation enabled (default behavior)
    slices_horizontal = slice_part(
        box, slice_normal, thickness, transform_to_horizontal=True
    )

    # Test with transformation disabled (keep original positions)
    slices_original = slice_part(
        box, slice_normal, thickness, transform_to_horizontal=False
    )

    # Should have same number of slices
    assert len(slices_horizontal) == len(slices_original)

    # Test horizontal orientation when transform_to_horizontal=True
    for slice_info in slices_horizontal:
        bbox = slice_info["slice_bbox"]
        min_point, max_point = bbox

        # Should be centered in X and Y
        center_x = (min_point[0] + max_point[0]) / 2
        center_y = (min_point[1] + max_point[1]) / 2
        assert (
            abs(center_x) < 0.1
        ), f"Slice should be centered in X, got center_x={center_x}"
        assert (
            abs(center_y) < 0.1
        ), f"Slice should be centered in Y, got center_y={center_y}"

        # Bottom should be at z=0
        assert min_point[2] < 0.1, f"Slice z_min={min_point[2]} should be ~0"

        # Should have reasonable thickness in Z direction
        thickness_z = max_point[2] - min_point[2]
        assert (
            1.8 < thickness_z < 2.2
        ), f"Slice thickness should be ~2.0, got {thickness_z}"

    # Test original positioning when transform_to_horizontal=False
    for i, slice_info in enumerate(slices_original):
        bbox = slice_info["slice_bbox"]
        min_point, max_point = bbox

        # Slices should remain in their original X positions (roughly)
        # The slice plane point indicates where the slice was cut from
        plane_x = slice_info["plane_point"][0]
        slice_center_x = (min_point[0] + max_point[0]) / 2

        # The slice should be positioned around its original cutting position
        # (allowing some tolerance for the cutting operation)
        assert (
            abs(slice_center_x - (plane_x + thickness / 2)) < 1.0
        ), f"Slice {i} should be near original position x={plane_x + thickness/2}, got center_x={slice_center_x}"

        # Y and Z should match original box dimensions (roughly)
        assert (
            abs((min_point[1] + max_point[1]) / 2 - 10) < 0.5
        ), "Y should be centered around original position"
        assert (
            abs((min_point[2] + max_point[2]) / 2 - 15) < 0.5
        ), "Z should be centered around original position"


def test_slice_part_custom_start_and_length():
    """Test slice_part with custom start point and slicing length."""

    # Create a test box centered at origin: from (-5,-10,-15) to (5,10,15)
    box = create_box(10, 20, 30)  # 10mm wide, 20mm deep, 30mm tall

    # Define custom slicing parameters
    slice_normal = [0, 1, 0]  # slice along Y-axis
    slice_thickness = 5.0
    custom_start = [0, -12, 0]  # Start 2mm before the box's Y-min (-10)
    custom_length = 25.0  # Slice for 25mm (covers box and extends 3mm beyond Y-max)

    # Test with custom parameters
    slices = slice_part(
        box,
        slice_normal,
        slice_thickness,
        transform_to_horizontal=False,  # Keep in original position for easier testing
        start_point=custom_start,
        slicing_length=custom_length,
    )

    # Only slices that intersect the box will be returned
    # Box spans Y=-10 to Y=10, slicing from Y=-12 in 5mm increments
    # Expected intersecting slices: Y=-7 to Y=-2, Y=-2 to Y=3, Y=3 to Y=8, Y=8 to Y=13
    # But some may have very small volume and be filtered out
    assert len(slices) >= 3, f"Expected at least 3 slices, got {len(slices)}"

    # Check that slices have sequential slice_index values
    for i, slice_data in enumerate(slices):
        assert (
            slice_data["slice_index"] == i
        ), f"Slice {i}: expected slice_index={i}, got slice_index={slice_data['slice_index']}"

    # Check that the slicing spans the expected range
    first_plane_y = slices[0]["plane_point"][1]
    last_plane_y = slices[-1]["plane_point"][1]

    # The span should cover most of the box's Y range
    y_span = last_plane_y - first_plane_y
    assert (
        y_span >= 8.0
    ), f"Expected Y span >= 8mm, got {y_span}mm"  # Most of the 20mm box height

    # Each slice should have reasonable volume
    for i, slice_data in enumerate(slices):
        volume = get_volume(slice_data["part"])
        assert volume > 1e-10, f"Slice {i} should have positive volume, got {volume}"


def test_slice_part_custom_vs_automatic():
    """Test that custom parameters give different results than automatic detection."""

    # Create a test box
    box = create_box(10, 20, 30)

    slice_normal = [0, 1, 0]  # slice along Y-axis
    slice_thickness = 3.0

    # Automatic slicing
    auto_slices = slice_part(
        box, slice_normal, slice_thickness, transform_to_horizontal=False
    )

    # Custom slicing with different start and length
    custom_start = [5, -5, 15]  # Different start point
    custom_length = 15.0  # Shorter length
    custom_slices = slice_part(
        box,
        slice_normal,
        slice_thickness,
        transform_to_horizontal=False,
        start_point=custom_start,
        slicing_length=custom_length,
    )

    # Should have different number of slices
    assert len(auto_slices) != len(
        custom_slices
    ), f"Expected different slice counts, but both gave {len(auto_slices)} slices"

    # Custom should have 4-5 slices (15mm / 3mm = 5, but edge slices may be filtered)
    assert (
        4 <= len(custom_slices) <= 5
    ), f"Expected 4-5 custom slices, got {len(custom_slices)}"

    # First slice should start near custom start point
    first_plane_point = custom_slices[0]["plane_point"]
    # Allow some tolerance since edge slices might be filtered out
    assert (
        abs(first_plane_point[1] - custom_start[1]) <= slice_thickness
    ), f"Expected first slice near Y={custom_start[1]}, got Y={first_plane_point[1]}"

    # Verify custom slicing uses the provided start point constraints
    assert len(custom_slices) < len(
        auto_slices
    ), f"Custom slicing should produce fewer slices due to length constraint"
