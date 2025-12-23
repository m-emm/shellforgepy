import math

import numpy as np
import pytest
from shellforgepy.adapters._adapter import get_volume
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.higher_order_solids import (
    create_distorted_cube,
    create_hex_prism,
    create_isoceles_triangle,
    create_pyramid_stump,
    create_right_triangle,
    create_ring,
    create_ring_segment_between_points,
    create_rounded_slab,
    create_screw_thread,
    directed_box_at,
    directed_cone_at,
    directed_cylinder_at,
    materialize_bounding_box,
)
from shellforgepy.simple import create_box, get_bounding_box, get_vertex_coordinates


def _min_angle_delta(angles, target):
    if len(angles) == 0:
        return float("inf")
    deltas = (angles - target + 180.0) % 360.0 - 180.0
    return float(np.min(np.abs(deltas)))


def _vertex_angles_radii_heights(vertices, center, plane_normal, start_direction):
    center = np.asarray(center, dtype=float)
    normal = normalize(plane_normal)
    x_axis = normalize(start_direction)
    y_axis = normalize(np.cross(normal, x_axis))

    angles = []
    radii = []
    heights = []

    for vertex in vertices:
        vec = np.asarray(vertex, dtype=float) - center
        heights.append(np.dot(vec, normal))
        vec_proj = vec - np.dot(vec, normal) * normal
        radius = np.linalg.norm(vec_proj)
        if radius <= 1e-8:
            continue
        radii.append(radius)
        angle = math.degrees(
            math.atan2(np.dot(vec_proj, y_axis), np.dot(vec_proj, x_axis))
        )
        if angle < 0:
            angle += 360.0
        angles.append(angle)

    return np.array(angles), np.array(radii), np.array(heights)


def test_create_hex_prism():
    prism = create_hex_prism(diameter=10, thickness=5, origin=(0, 0, 0))
    assert prism is not None
    # Further assertions can be added based on the expected properties of the prism


def test_create_isoceles_triangle_geometry_and_volume():
    side_length = 10.0
    tip_angle = 60.0
    thickness = 4.0

    triangle = create_isoceles_triangle(side_length, tip_angle, thickness)
    base_length = 2 * side_length * math.sin(math.radians(tip_angle / 2.0))
    height = side_length * math.cos(math.radians(tip_angle / 2.0))

    bb_min, bb_max = get_bounding_box(triangle)
    assert np.isclose(bb_min[0], -base_length / 2.0, atol=1e-6)
    assert np.isclose(bb_max[0], base_length / 2.0, atol=1e-6)
    assert np.isclose(bb_min[1], 0.0, atol=1e-6)
    assert np.isclose(bb_max[1], height, atol=1e-6)
    assert np.isclose(bb_min[2], 0.0, atol=1e-6)
    assert np.isclose(bb_max[2], thickness, atol=1e-6)

    expected_volume = 0.5 * base_length * height * thickness
    assert np.isclose(get_volume(triangle), expected_volume, rtol=1e-6)


def test_create_isoceles_triangle_invalid_inputs():
    with pytest.raises(ValueError):
        create_isoceles_triangle(0, 30, 1)
    with pytest.raises(ValueError):
        create_isoceles_triangle(5, 0, 1)
    with pytest.raises(ValueError):
        create_isoceles_triangle(5, 200, 1)
    with pytest.raises(ValueError):
        create_isoceles_triangle(5, 30, -1)


def test_create_right_triangle_default_orientation():
    triangle = create_right_triangle(a=10, b=20, thickness=5)

    bb = get_bounding_box(triangle)
    x_len = bb[1][0] - bb[0][0]
    y_len = bb[1][1] - bb[0][1]
    z_len = bb[1][2] - bb[0][2]

    assert np.isclose(x_len, 20, rtol=1e-6)
    assert np.isclose(y_len, 10, rtol=1e-6)
    assert np.isclose(z_len, 5, rtol=1e-6)

    expected_volume = 0.5 * 10 * 20 * 5
    assert np.isclose(get_volume(triangle), expected_volume, rtol=1e-6)


def test_create_right_triangle_with_orientation_vectors():
    triangle = create_right_triangle(
        a=10,
        b=20,
        thickness=5,
        extrusion_direction=(1, 0, 0),
        a_normal=(0, -1, 0),
    )

    bb = get_bounding_box(triangle)
    axis_lengths = np.array(bb[1]) - np.array(bb[0])

    # Extrusion along +X, base dimensions rotate into Y/Z axes respectively.
    assert np.isclose(axis_lengths[0], 5, rtol=1e-6)
    assert np.isclose(axis_lengths[1], 10, rtol=1e-6)
    assert np.isclose(axis_lengths[2], 20, rtol=1e-6)

    assert np.isclose(get_volume(triangle), 0.5 * 10 * 20 * 5, rtol=1e-6)


def test_create_right_triangle_with_inferred_extrusion_direction():
    triangle = create_right_triangle(
        a=6,
        b=4,
        thickness=3,
        a_normal=(0, -1, 0),
        b_normal=(0, 0, 1),
    )

    bb = get_bounding_box(triangle)
    axis_lengths = np.array(bb[1]) - np.array(bb[0])

    # Cross product of provided normals is (-1, 0, 0).
    assert np.isclose(axis_lengths[0], 3, rtol=1e-6)
    assert np.isclose(axis_lengths[1], 6, rtol=1e-6)
    assert np.isclose(axis_lengths[2], 4, rtol=1e-6)

    assert np.isclose(get_volume(triangle), 0.5 * 6 * 4 * 3, rtol=1e-6)


def test_create_right_triangle_invalid_inputs():
    with pytest.raises(ValueError):
        create_right_triangle(a=0, b=2, thickness=1)

    with pytest.raises(ValueError):
        create_right_triangle(
            a=1,
            b=1,
            thickness=1,
            extrusion_direction=(0, 0, 1),
            a_normal=(0, 0, 0),
        )


def test_create_screw_thread():
    """Test screw thread creation with trapezoidal snake geometry."""
    # Test basic screw thread creation
    screw = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=2,
        resolution=20,
        with_core=True,
    )

    assert screw is not None
    # Check that the screw has reasonable volume
    assert screw.Volume() > 0

    # Test without core
    screw_no_core = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=1,
        resolution=16,
        with_core=False,
    )

    assert screw_no_core is not None
    assert get_volume(screw_no_core) > 0

    # Screw with core should have more volume than without
    screw_with_core = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=1,
        resolution=16,
        with_core=True,
    )

    assert screw_with_core.Volume() > screw_no_core.Volume()


def test_create_screw_thread_parameters():
    """Test screw thread with various parameters."""
    # Test with optimization
    screw_optimized = create_screw_thread(
        pitch=1.5,
        inner_radius=5.0,
        outer_radius=7.0,
        outer_thickness=0.3,
        num_turns=3,
        resolution=24,
        optimize_start=True,
        optimize_start_angle=30,
    )

    assert screw_optimized is not None
    assert get_volume(screw_optimized) > 0

    # Test with custom inner thickness
    screw_custom = create_screw_thread(
        pitch=2.5,
        inner_radius=6.0,
        outer_radius=8.5,
        outer_thickness=0.4,
        inner_thickness=0.6,
        num_turns=1.5,
        resolution=30,
    )

    assert screw_custom is not None
    assert get_volume(screw_custom) > 0

    # Test with core offset
    screw_offset = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.3,
        num_turns=2,
        resolution=16,
        core_offset=0.2,  # Smaller core offset
        core_height=7.0,  # Larger core height to accommodate offset
    )

    assert screw_offset is not None
    assert get_volume(screw_offset) > 0


def test_create_screw_thread_edge_cases():
    """Test screw thread edge cases and validation."""
    # Test minimal parameters
    minimal_screw = create_screw_thread(
        pitch=1.0,
        inner_radius=2.0,
        outer_radius=3.0,
        outer_thickness=0.2,
        num_turns=0.5,
        resolution=8,
    )

    assert minimal_screw is not None
    assert get_volume(minimal_screw) > 0

    # Test high resolution
    high_res_screw = create_screw_thread(
        pitch=2.0,
        inner_radius=5.0,
        outer_radius=7.0,
        outer_thickness=0.4,
        num_turns=1,
        resolution=60,
    )

    assert high_res_screw is not None
    assert get_volume(high_res_screw) > 0
    # @todo add further assertions based on expected properties of the trapezoid


def test_directed_cylinder():
    """Test directed cylinder creation."""

    # Test cylinder along Z axis (should be same as basic cylinder)
    cyl_z = directed_cylinder_at(
        base_point=(0, 0, 0), direction=(0, 0, 1), radius=5, height=10
    )
    expected_volume = np.pi * 5**2 * 10
    assert np.allclose(get_volume(cyl_z), expected_volume, rtol=1e-5)

    # Test cylinder along X axis
    cyl_x = directed_cylinder_at(
        base_point=(0, 0, 0), direction=(1, 0, 0), radius=5, height=10
    )
    assert np.allclose(get_volume(cyl_x), expected_volume, rtol=1e-5)

    # Test cylinder along arbitrary direction
    direction = (1, 1, 1)  # diagonal direction
    cyl_diag = directed_cylinder_at(
        base_point=(5, 5, 5), direction=direction, radius=3, height=8
    )
    expected_volume = np.pi * 3**2 * 8
    assert np.allclose(get_volume(cyl_diag), expected_volume, rtol=1e-5)


def test_directed_cone_at():
    """Test directed cone creation."""

    base_radius = 5
    top_radius = 2
    height = 10
    expected_volume = (
        math.pi
        * height
        * (base_radius**2 + base_radius * top_radius + top_radius**2)
        / 3
    )

    cone_z = directed_cone_at(
        base_point=(0, 0, 0),
        direction=(0, 0, 1),
        radius1=base_radius,
        radius2=top_radius,
        height=height,
    )
    assert np.allclose(get_volume(cone_z), expected_volume, rtol=1e-5)
    bb_z = get_bounding_box(cone_z)
    assert np.isclose(bb_z[0][2], 0, atol=1e-6)
    assert np.isclose(bb_z[1][2], height, atol=1e-6)

    cone_x = directed_cone_at(
        base_point=(0, 0, 0),
        direction=(1, 0, 0),
        radius1=base_radius,
        radius2=top_radius,
        height=height,
    )
    assert np.allclose(get_volume(cone_x), expected_volume, rtol=1e-5)
    bb_x = get_bounding_box(cone_x)
    assert np.isclose(bb_x[0][0], 0, atol=1e-6)
    assert np.isclose(bb_x[1][0], height, atol=1e-6)

    direction = (1, 1, 1)
    cone_diag = directed_cone_at(
        base_point=(2, 3, 4),
        direction=direction,
        radius1=3,
        radius2=0,
        height=9,
    )
    expected_volume_diag = math.pi * 9 * (3**2) / 3
    assert np.allclose(get_volume(cone_diag), expected_volume_diag, rtol=1e-5)


def test_directed_cone_at_errors():
    with pytest.raises(ValueError, match="Direction vector cannot be zero"):
        directed_cone_at(
            base_point=(0, 0, 0),
            direction=(0, 0, 0),
            radius1=1,
            radius2=0,
            height=5,
        )


def test_create_ring():
    """Test ring creation."""
    # Basic ring
    ring = create_ring(outer_radius=10, inner_radius=5, height=8)
    assert ring is not None

    # Calculate expected volume: outer cylinder - inner cylinder
    outer_volume = np.pi * 10**2 * 8
    inner_volume = np.pi * 5**2 * 8
    expected_volume = outer_volume - inner_volume
    assert np.allclose(get_volume(ring), expected_volume, rtol=1e-3)

    # Test with origin offset
    ring_offset = create_ring(
        outer_radius=10, inner_radius=5, height=8, origin=(5, 5, 5)
    )
    assert ring is not None
    assert np.allclose(get_volume(ring_offset), expected_volume, rtol=1e-3)

    # Test with partial angle
    ring_partial = create_ring(outer_radius=10, inner_radius=5, height=8, angle=180)
    assert ring_partial is not None
    # Half the volume for 180 degrees
    assert np.allclose(get_volume(ring_partial), expected_volume / 2, rtol=1e-3)

    # Test error case: inner radius >= outer radius
    try:
        create_ring(outer_radius=5, inner_radius=10, height=8)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_create_ring_segment_between_points_xy_plane_alignment():
    inner_radius = 5.0
    outer_radius = 7.0
    height = 4.0
    middle_radius = (inner_radius + outer_radius) / 2.0

    p1 = (middle_radius, 0.0, 0.0)
    p2 = (0.0, middle_radius, 0.0)
    third_point_on_plane = (0.0, 0.0, 0.0)

    ring_segment = create_ring_segment_between_points(
        p1, p2, third_point_on_plane, inner_radius, outer_radius, height
    )

    vertices = get_vertex_coordinates(ring_segment)
    angles, radii, heights = _vertex_angles_radii_heights(
        vertices,
        center=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        start_direction=np.array(p1),
    )

    expected_angle = 90.0
    assert _min_angle_delta(angles, 0.0) < 0.5
    assert _min_angle_delta(angles, expected_angle) < 0.5

    assert np.min(np.abs(radii - inner_radius)) < 1e-5
    assert np.min(np.abs(radii - outer_radius)) < 1e-5

    assert np.min(np.abs(heights - height / 2.0)) < 1e-6
    assert np.min(np.abs(heights + height / 2.0)) < 1e-6


def test_create_ring_segment_between_points_tilted_plane_alignment():
    inner_radius = 6.0
    outer_radius = 8.0
    height = 3.0
    middle_radius = (inner_radius + outer_radius) / 2.0
    theta_deg = 60.0

    center = np.array((10.0, -5.0, 3.0))
    plane_normal = normalize((1.0, 1.0, 2.0))
    x_axis = normalize((1.0, -1.0, 0.0))
    y_axis = normalize(np.cross(plane_normal, x_axis))

    p1 = center + x_axis * middle_radius
    p2 = (
        center
        + (
            x_axis * math.cos(math.radians(theta_deg))
            + y_axis * math.sin(math.radians(theta_deg))
        )
        * middle_radius
    )
    third_point_on_plane = center + y_axis * middle_radius

    ring_segment = create_ring_segment_between_points(
        p1, p2, third_point_on_plane, inner_radius, outer_radius, height
    )

    vertices = get_vertex_coordinates(ring_segment)
    angles, radii, heights = _vertex_angles_radii_heights(
        vertices,
        center=center,
        plane_normal=plane_normal,
        start_direction=p1 - center,
    )

    assert _min_angle_delta(angles, 0.0) < 0.5
    assert _min_angle_delta(angles, theta_deg) < 0.5
    assert np.min(np.abs(radii - inner_radius)) < 1e-5
    assert np.min(np.abs(radii - outer_radius)) < 1e-5
    assert np.min(np.abs(heights - height / 2.0)) < 1e-6
    assert np.min(np.abs(heights + height / 2.0)) < 1e-6


def test_create_ring_segment_between_points_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        create_ring_segment_between_points(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            inner_radius=3.0,
            outer_radius=4.0,
            height=2.0,
        )


def test_create_ring_segment_between_points_rejects_impossible_radius():
    with pytest.raises(ValueError):
        create_ring_segment_between_points(
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            inner_radius=2.0,
            outer_radius=4.0,
            height=2.0,
        )


def test_create_rounded_slab_respects_rounding_flags():
    """Rounded slab should remove only the flagged corners and preserve bounding box."""

    length = 40.0
    width = 20.0
    thick = 6.0
    round_radius = 5.0
    rounding_flags = {
        (1, 1): True,
        (-1, 1): False,
        (-1, -1): True,
        (1, -1): False,
    }

    slab = create_rounded_slab(
        length,
        width,
        thick,
        round_radius,
        rounding_flags=rounding_flags,
    )

    min_corner, max_corner = get_bounding_box(slab)
    assert np.allclose(min_corner, (0.0, 0.0, 0.0))
    assert np.allclose(max_corner, (length, width, thick))

    num_rounded = sum(1 for value in rounding_flags.values() if value)
    square_area = round_radius**2
    quarter_circle_area = (np.pi * round_radius**2) / 4.0
    expected_volume = (
        length * width * thick
        - num_rounded * (square_area - quarter_circle_area) * thick
    )
    assert np.allclose(get_volume(slab), expected_volume, rtol=1e-6, atol=1e-6)

    vertices = get_vertex_coordinates(slab)

    def has_corner(x_target, y_target):
        return any(
            np.isclose(x, x_target, atol=1e-6) and np.isclose(y, y_target, atol=1e-6)
            for x, y, _ in vertices
        )

    assert not has_corner(length, width)  # (1, 1) corner should be rounded
    assert has_corner(0.0, width)  # (-1, 1) corner should remain sharp
    assert not has_corner(0.0, 0.0)  # (-1, -1) corner should be rounded
    assert has_corner(length, 0.0)  # (1, -1) corner should remain sharp


def test_create_screw_thread():
    """Test screw thread creation."""
    # Basic thread
    thread = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=2,
        resolution=20,  # Lower resolution for faster testing
    )
    assert thread is not None

    # Thread should have some volume
    volume = get_volume(thread)
    assert volume > 0

    # Test thread without core
    thread_no_core = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=1,
        with_core=False,
        resolution=20,
    )
    assert thread_no_core is not None

    # Thread without core should have less volume than with core
    volume_no_core = get_volume(thread_no_core)
    assert volume_no_core > 0

    # Test with optimize_start
    thread_optimized = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=1,
        optimize_start=True,
        resolution=20,
    )
    assert thread_optimized is not None


def test_directed_box_at():
    """Test directed box creation with various orientations."""

    # Test basic box along Z axis (should match basic box behavior)
    box_z = directed_box_at(
        base_point=(0, 0, 0), height_direction=(0, 0, 1), width=10, depth=20, height=5
    )

    # Check volume
    expected_volume = 10 * 20 * 5
    assert np.allclose(get_volume(box_z), expected_volume, rtol=1e-5)

    # Check bounding box - box should extend from z=0 to z=5
    bb = get_bounding_box(box_z)
    assert np.isclose(bb[0][2], 0, atol=1e-6)  # bottom at z=0
    assert np.isclose(bb[1][2], 5, atol=1e-6)  # top at z=5

    # Test box along X axis
    box_x = directed_box_at(
        base_point=(0, 0, 0), height_direction=(1, 0, 0), width=10, depth=20, height=5
    )

    # Volume should be the same
    assert np.allclose(get_volume(box_x), expected_volume, rtol=1e-5)

    # Check that height dimension is now along X axis
    bb_x = get_bounding_box(box_x)
    x_extent = bb_x[1][0] - bb_x[0][0]
    assert np.isclose(x_extent, 5, atol=1e-6)  # height=5 now along X

    # Test box along arbitrary direction with width direction specified
    height_dir = np.array([1, 1, 0])  # diagonal in XY plane
    height_dir = height_dir / np.linalg.norm(height_dir)
    width_dir = np.array([0, 0, 1])  # along Z axis

    box_diag = directed_box_at(
        base_point=(5, 5, 5),
        height_direction=height_dir,
        width=8,
        depth=12,
        height=6,
        width_direction=width_dir,
    )

    # Volume should still be correct
    expected_volume_diag = 8 * 12 * 6
    assert np.allclose(get_volume(box_diag), expected_volume_diag, rtol=1e-5)

    # Test automatic width direction selection
    box_auto = directed_box_at(
        base_point=(1, 2, 3),
        height_direction=(0, 1, 0),  # along Y axis
        width=5,
        depth=7,
        height=9,
    )

    expected_volume_auto = 5 * 7 * 9
    assert np.allclose(get_volume(box_auto), expected_volume_auto, rtol=1e-5)

    # Test that box is positioned correctly - base_point should be center of bottom face
    bb_auto = get_bounding_box(box_auto)
    y_min = bb_auto[0][1]
    y_max = bb_auto[1][1]
    assert np.isclose(y_min, 2, atol=1e-6)  # base_point y-coordinate
    assert np.isclose(y_max, 2 + 9, atol=1e-6)  # base_point + height


def test_directed_box_at_error_cases():
    """Test error cases for directed_box_at function."""

    # Test zero height direction
    with pytest.raises(ValueError, match="Height direction vector cannot be zero"):
        directed_box_at(
            base_point=(0, 0, 0), height_direction=(0, 0, 0), width=5, depth=5, height=5
        )

    # Test zero width direction
    with pytest.raises(ValueError, match="Width direction vector cannot be zero"):
        directed_box_at(
            base_point=(0, 0, 0),
            height_direction=(0, 0, 1),
            width=5,
            depth=5,
            height=5,
            width_direction=(0, 0, 0),
        )

    # Test parallel height and width directions
    with pytest.raises(
        ValueError, match="Width direction cannot be parallel to height direction"
    ):
        directed_box_at(
            base_point=(0, 0, 0),
            height_direction=(1, 0, 0),
            width=5,
            depth=5,
            height=5,
            width_direction=(2, 0, 0),  # parallel to height_direction
        )


def test_directed_box_at_vs_directed_cylinder_positioning():
    """Test that directed_box_at and directed_cylinder_at position similarly."""

    base_point = (10, 20, 30)
    direction = (0, 1, 0)  # along Y axis

    # Create a box and cylinder with same height and positioned at same base_point
    box = directed_box_at(
        base_point=base_point, height_direction=direction, width=6, depth=6, height=8
    )

    cylinder = directed_cylinder_at(
        base_point=base_point, direction=direction, radius=3, height=8
    )

    # Both should start at the same Y coordinate (base_point[1])
    bb_box = get_bounding_box(box)
    bb_cyl = get_bounding_box(cylinder)

    assert np.isclose(bb_box[0][1], bb_cyl[0][1], atol=1e-6)  # same Y minimum
    assert np.isclose(bb_box[1][1], bb_cyl[1][1], atol=1e-6)  # same Y maximum


def test_cube_topology():

    test_box = create_box(10, 20, 30)

    bbox = materialize_bounding_box(test_box)

    assert np.allclose(get_bounding_box(bbox), get_bounding_box(test_box))


def test_create_distorted_cube_geometry_and_validation():
    # Simple axis-aligned box description should produce matching bounding box and volume
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 2, 0),
        (0, 2, 0),
        (0, 0, 3),
        (1, 0, 3),
        (1, 2, 3),
        (0, 2, 3),
    ]
    solid = create_distorted_cube(corners)
    bb = get_bounding_box(solid)

    assert np.allclose(bb[0], (0.0, 0.0, 0.0), atol=1e-8)
    assert np.allclose(bb[1], (1.0, 2.0, 3.0), atol=1e-8)
    assert np.isclose(get_volume(solid), 6.0, atol=1e-8)

    # Invalid vertex count should raise
    with pytest.raises(ValueError):
        create_distorted_cube(corners[:7])


def test_create_pyramid_stump_shape_and_volume():
    bottom_width, top_width = 10.0, 4.0
    bottom_depth, top_depth = 8.0, 2.0
    height = 6.0

    stump = create_pyramid_stump(
        bottom_width=bottom_width,
        top_width=top_width,
        bottom_depth=bottom_depth,
        top_depth=top_depth,
        height=height,
    )

    bb_min, bb_max = get_bounding_box(stump)
    assert np.allclose(bb_min, (-bottom_width / 2, -bottom_depth / 2, 0.0), atol=1e-8)
    assert np.allclose(bb_max, (bottom_width / 2, bottom_depth / 2, height), atol=1e-8)

    # Volume should match the analytic integral of linearly tapering width/depth
    base_area = bottom_width * bottom_depth
    a = top_width - bottom_width
    c = top_depth - bottom_depth
    expected_cross_section_integral = (
        base_area + (bottom_width * c + bottom_depth * a) / 2 + (a * c) / 3
    )
    expected_volume = height * expected_cross_section_integral
    assert np.isclose(get_volume(stump), expected_volume, rtol=1e-6, atol=1e-6)
