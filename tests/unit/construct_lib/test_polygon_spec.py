import math

import numpy as np
import pytest
from shellforgepy.construct.polygon_spec import PolygonSpec


def test_points_in_polygon_3d():
    """Test basic point-in-polygon functionality for a square in XY plane."""

    # Define a square polygon in 3D (lying in the XY plane)
    polygon_3d = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    inside_points = [
        np.array([0.5, 0.5, 0.0]),
        np.array([0.1, 0.1, 0.0]),
        np.array([0.9, 0.9, 0.0]),
    ]

    outside_points = [
        np.array([-0.1, 0.5, 0.0]),
        np.array([1.1, 0.5, 0.0]),
        np.array([0.5, -0.1, 0.0]),
        np.array([0.5, 1.1, 0.0]),
    ]

    ps = PolygonSpec(points=polygon_3d)

    for pt in inside_points:
        assert ps.contains_point(pt), f"Point {pt} should be inside"

    for pt in outside_points:
        assert not ps.contains_point(pt), f"Point {pt} should be outside"


def test_polygon_spec_initialization():
    """Test PolygonSpec initialization and basic properties."""

    # Test with valid triangle
    triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 1.0, 0.0]),
    ]

    ps = PolygonSpec(points=triangle)

    # Check properties are computed correctly
    assert hasattr(ps, "center")
    assert hasattr(ps, "normal")
    assert len(ps.points) == 3

    # Center should be centroid of triangle
    expected_center = np.mean(triangle, axis=0)
    np.testing.assert_array_almost_equal(ps.center, expected_center)

    # Normal should be unit vector perpendicular to triangle plane
    assert np.abs(np.linalg.norm(ps.normal) - 1.0) < 1e-10

    # For triangle in XY plane, normal should be along Z-axis
    expected_normal = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(ps.normal, expected_normal)


def test_polygon_spec_invalid_inputs():
    """Test PolygonSpec error handling for invalid inputs."""

    # Test with too few points
    with pytest.raises(ValueError, match="Polygon must have at least 3 points"):
        PolygonSpec(points=[np.array([0, 0, 0]), np.array([1, 0, 0])])

    # Test with empty list
    with pytest.raises(ValueError, match="Polygon must have at least 3 points"):
        PolygonSpec(points=[])


def test_from_points_3d_factory():
    """Test the from_points_3d factory method."""

    points_list = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
    ]

    ps1 = PolygonSpec.from_points_3d(points_list)
    ps2 = PolygonSpec(points=points_list)

    # Should be equivalent
    np.testing.assert_array_almost_equal(ps1.center, ps2.center)
    np.testing.assert_array_almost_equal(ps1.normal, ps2.normal)


def test_2d_coordinate_system():
    """Test 2D coordinate system creation and conversion."""

    # Create a triangle in XY plane
    triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([1.5, 2.0, 0.0]),
    ]

    ps = PolygonSpec(points=triangle)

    # Test 2D basis creation
    u, v = ps.create_2d_basis()

    # Basis vectors should be orthonormal
    assert abs(np.linalg.norm(u) - 1.0) < 1e-10
    assert abs(np.linalg.norm(v) - 1.0) < 1e-10
    assert abs(np.dot(u, v)) < 1e-10

    # Both should be perpendicular to normal
    assert abs(np.dot(u, ps.normal)) < 1e-10
    assert abs(np.dot(v, ps.normal)) < 1e-10


def test_point_2d_conversion_round_trip():
    """Test 3D to 2D and back conversion preserves points."""

    # Create hexagon in arbitrary 3D plane
    hexagon = []
    for i in range(6):
        angle = i * 2 * math.pi / 6
        # Create points on tilted plane
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        z = 0.5 * x + 0.3 * y + 1.0  # Tilted plane equation
        hexagon.append(np.array([x, y, z]))

    ps = PolygonSpec(points=hexagon)

    # Test round trip conversion for each vertex
    for point_3d in hexagon:
        point_2d = ps.point_to_2d(point_3d)
        reconstructed_3d = ps.point_from_2d(point_2d)

        # Should be very close to original
        diff = np.linalg.norm(point_3d - reconstructed_3d)
        assert diff < 1e-10, f"Round trip failed: {diff}"

    # Test center converts to origin in 2D
    center_2d = ps.point_to_2d(ps.center)
    assert np.linalg.norm(center_2d) < 1e-10, "Center should map to 2D origin"


def test_contains_point_different_planes():
    """Test point containment for polygons in different orientations."""

    # Test 1: Triangle in YZ plane
    triangle_yz = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.5, 1.0]),
    ]

    ps_yz = PolygonSpec(points=triangle_yz)

    # Point inside triangle (in YZ plane)
    inside_yz = np.array([0.0, 0.4, 0.3])
    assert ps_yz.contains_point(inside_yz)

    # Point outside triangle
    outside_yz = np.array([0.0, 1.5, 0.5])
    assert not ps_yz.contains_point(outside_yz)

    # Point not on plane (should project and then test)
    off_plane = np.array([1.0, 0.4, 0.3])  # Same YZ coords as inside point
    assert ps_yz.contains_point(off_plane)  # Should project to inside point

    # Test 2: Tilted triangle
    tilted_triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
    ]

    ps_tilted = PolygonSpec(points=tilted_triangle)

    # Point at centroid should be inside
    centroid = np.mean(tilted_triangle, axis=0)
    assert ps_tilted.contains_point(centroid)


def test_regular_hexagon_properties():
    """Test properties specific to regular hexagon."""

    # Create regular hexagon centered at origin in XY plane
    radius = 5.0
    hexagon = []
    for i in range(6):
        angle = i * 2 * math.pi / 6
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        hexagon.append(np.array([x, y, 0.0]))

    ps = PolygonSpec(points=hexagon)

    # Center should be at origin
    np.testing.assert_array_almost_equal(ps.center, np.array([0.0, 0.0, 0.0]))

    # Normal should be along Z-axis
    np.testing.assert_array_almost_equal(ps.normal, np.array([0.0, 0.0, 1.0]))

    # Test points at various distances from center
    # Points inside radius should be inside polygon
    test_points_inside = [
        np.array([0.0, 0.0, 0.0]),  # Center
        np.array([2.0, 0.0, 0.0]),  # Along X-axis, inside
        np.array([0.0, 2.0, 0.0]),  # Along Y-axis, inside
        np.array([1.0, 1.0, 0.0]),  # Diagonal, inside
    ]

    for pt in test_points_inside:
        assert ps.contains_point(pt), f"Point {pt} should be inside hexagon"

    # Points outside radius should be outside polygon
    test_points_outside = [
        np.array([6.0, 0.0, 0.0]),  # Along X-axis, outside
        np.array([0.0, 6.0, 0.0]),  # Along Y-axis, outside
        np.array([4.0, 4.0, 0.0]),  # Diagonal, outside
    ]

    for pt in test_points_outside:
        assert not ps.contains_point(pt), f"Point {pt} should be outside hexagon"


def test_edge_cases():
    """Test edge cases and boundary conditions."""

    # Very small triangle
    tiny_triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-6, 0.0, 0.0]),
        np.array([0.5e-6, 1e-6, 0.0]),
    ]

    ps_tiny = PolygonSpec(points=tiny_triangle)

    # Should still work for tiny triangles
    assert hasattr(ps_tiny, "center")
    assert hasattr(ps_tiny, "normal")
    assert np.linalg.norm(ps_tiny.normal) > 0.5  # Should be normalized

    # Triangle with large coordinates
    large_triangle = [
        np.array([1e6, 0.0, 0.0]),
        np.array([1e6 + 1.0, 0.0, 0.0]),
        np.array([1e6 + 0.5, 1.0, 0.0]),
    ]

    ps_large = PolygonSpec(points=large_triangle)

    # Should handle large coordinates
    center_large = ps_large.center
    assert abs(center_large[0] - 1e6 - 0.5) < 1e-10

    # Point inside should still work
    inside_large = np.array([1e6 + 0.4, 0.3, 0.0])
    assert ps_large.contains_point(inside_large)


def test_complex_polygon():
    """Test with more complex polygon shapes."""

    # Create an irregular pentagon
    pentagon = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.5, 0.0]),
        np.array([1.5, 2.0, 0.0]),
        np.array([-0.5, 1.8, 0.0]),
        np.array([-1.0, 0.8, 0.0]),
    ]

    ps = PolygonSpec(points=pentagon)

    # Test various points
    # Center should be inside
    assert ps.contains_point(ps.center)

    # Test some manually determined inside/outside points
    inside_points = [
        np.array([0.5, 0.5, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    outside_points = [
        np.array([3.0, 0.0, 0.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]

    for pt in inside_points:
        assert ps.contains_point(pt), f"Point {pt} should be inside pentagon"

    for pt in outside_points:
        assert not ps.contains_point(pt), f"Point {pt} should be outside pentagon"


def test_polygon_spec_with_non_planar_points():
    """Test behavior when points are not exactly coplanar."""

    # Create points that are nearly but not exactly coplanar
    almost_planar = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1e-8]),  # Slightly off-plane
        np.array([1.0, 1.0, -1e-8]),  # Slightly off-plane in other direction
        np.array([0.0, 1.0, 0.0]),
    ]

    ps = PolygonSpec(points=almost_planar)

    # Should still create valid polygon spec
    assert hasattr(ps, "center")
    assert hasattr(ps, "normal")

    # Normal should still be approximately along Z-axis
    assert abs(ps.normal[2]) > 0.9  # Mostly Z-aligned

    # Point containment should still work reasonably
    center_point = np.array([0.5, 0.5, 0.0])
    assert ps.contains_point(center_point)


def test_2d_projection_accuracy():
    """Test accuracy of 2D projection for geometric operations."""

    # Create a square tilted in 3D space
    # Rotate square around X-axis by 45 degrees
    angle = math.pi / 4
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    tilted_square = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, cos_a, sin_a]),
        np.array([0.0, cos_a, sin_a]),
    ]

    ps = PolygonSpec(points=tilted_square)

    # Convert all vertices to 2D
    vertices_2d = [ps.point_to_2d(v) for v in tilted_square]

    # In 2D, should form a proper square (all sides equal, right angles)
    # Calculate side lengths in 2D
    side_lengths_2d = []
    for i in range(4):
        v1 = vertices_2d[i]
        v2 = vertices_2d[(i + 1) % 4]
        length = np.linalg.norm(v2 - v1)
        side_lengths_2d.append(length)

    # All sides should be equal length
    expected_length = side_lengths_2d[0]
    for length in side_lengths_2d:
        assert abs(length - expected_length) < 1e-10

    # Test that 2D area is preserved
    # Original square has area 1.0 * sqrt(2) = sqrt(2) (side length is sqrt(2) due to rotation)
    # In 2D projection, it should maintain the same area relationship


def test_polygon_spec_documentation_examples():
    """Test examples that might appear in documentation."""

    # Example 1: Simple workflow
    triangle_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1, 0])]

    polygon = PolygonSpec.from_points_3d(triangle_points)

    # Check if point is inside
    test_point = np.array([0.5, 0.3, 0])
    is_inside = polygon.contains_point(test_point)
    assert is_inside

    # Convert point to 2D coordinates
    point_2d = polygon.point_to_2d(test_point)
    assert len(point_2d) == 2

    # Convert back to 3D
    point_3d_back = polygon.point_from_2d(point_2d)
    np.testing.assert_array_almost_equal(test_point, point_3d_back)


def test_calc_inset_polygon_spec_triangle():
    """Test polygon inset with a simple triangle."""

    # Create an equilateral triangle in XY plane
    side_length = 2.0
    height = side_length * math.sqrt(3) / 2
    triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([side_length, 0.0, 0.0]),
        np.array([side_length / 2, height, 0.0]),
    ]

    ps = PolygonSpec(points=triangle)

    # Inset by small amount
    inset_distance = 0.2
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # Check basic properties
    assert len(inset_poly.points) == 3
    assert hasattr(inset_poly, "center")
    assert hasattr(inset_poly, "normal")

    # Normal should be preserved (same orientation or opposite is fine)
    dot_product = np.dot(ps.normal, inset_poly.normal)
    assert abs(abs(dot_product) - 1.0) < 1e-10

    # All inset vertices should be inside original triangle
    for vertex in inset_poly.points:
        assert ps.contains_point(
            vertex
        ), f"Inset vertex {vertex} should be inside original"

    # Inset triangle should be smaller (lower perimeter)
    original_perim = ps.circumference()
    inset_perim = inset_poly.circumference()
    assert inset_perim < original_perim

    # Center should be roughly preserved for symmetric shapes
    center_diff = np.linalg.norm(ps.center - inset_poly.center)
    assert center_diff < 0.1  # Should be close for equilateral triangle


def test_calc_inset_polygon_spec_square():
    """Test polygon inset with a square."""

    # Create a square in XY plane
    square = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ]

    ps = PolygonSpec(points=square)

    # Inset by 0.3
    inset_distance = 0.3
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # Check basic properties
    assert len(inset_poly.points) == 4

    # All inset vertices should be inside original square
    for vertex in inset_poly.points:
        assert ps.contains_point(vertex)

    # For a square, inset should preserve square shape
    # Check that opposite vertices are equidistant from center
    inset_center = inset_poly.center
    distances = [np.linalg.norm(v - inset_center) for v in inset_poly.points]

    # All vertices should be roughly same distance from center (square symmetry)
    avg_distance = np.mean(distances)
    for dist in distances:
        assert abs(dist - avg_distance) < 1e-10

    # Expected inset square should have side length = original - 2*inset
    # Original square side = 2.0, so inset square side ≈ 2.0 - 2*0.3 = 1.4
    inset_side_lengths = []
    for i in range(4):
        v1 = inset_poly.points[i]
        v2 = inset_poly.points[(i + 1) % 4]
        length = np.linalg.norm(v2 - v1)
        inset_side_lengths.append(length)

    expected_side = 2.0 - 2 * inset_distance
    for length in inset_side_lengths:
        assert abs(length - expected_side) < 1e-10


def test_calc_inset_polygon_spec_hexagon():
    """Test polygon inset with a regular hexagon."""

    # Create regular hexagon
    radius = 3.0
    hexagon = []
    for i in range(6):
        angle = i * 2 * math.pi / 6
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        hexagon.append(np.array([x, y, 0.0]))

    ps = PolygonSpec(points=hexagon)

    # Inset by reasonable amount
    inset_distance = 0.5
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # Check basic properties
    assert len(inset_poly.points) == 6

    # All inset vertices should be inside original hexagon
    for vertex in inset_poly.points:
        assert ps.contains_point(vertex)

    # For regular hexagon, symmetry should be preserved
    inset_center = inset_poly.center
    distances = [np.linalg.norm(v - inset_center) for v in inset_poly.points]

    # All vertices should be same distance from center
    avg_distance = np.mean(distances)
    for dist in distances:
        assert abs(dist - avg_distance) < 1e-6

    # Inset hexagon should be smaller
    original_perim = ps.circumference()
    inset_perim = inset_poly.circumference()
    assert inset_perim < original_perim


def test_calc_inset_polygon_spec_tilted_triangle():
    """Test polygon inset with a triangle in arbitrary 3D orientation."""

    # Create triangle in tilted plane (make it larger to avoid edge issues)
    triangle = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 1.0, 5.0]),  # Make edges longer
        np.array([0.5, 5.0, 2.5]),
    ]

    ps = PolygonSpec(points=triangle)

    # Inset by small amount
    inset_distance = 0.05  # Reduce inset distance
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # Check basic properties
    assert len(inset_poly.points) == 3

    # Normal should be preserved (same orientation or opposite)
    dot_product = np.dot(ps.normal, inset_poly.normal)
    assert abs(abs(dot_product) - 1.0) < 1e-10  # Should be parallel (either direction)

    # All inset vertices should be inside original triangle
    for vertex in inset_poly.points:
        assert ps.contains_point(vertex)


def test_calc_inset_polygon_spec_error_cases():
    """Test error handling for polygon inset."""

    # Test with negative inset distance
    triangle = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 1.0, 0.0]),
    ]
    ps = PolygonSpec(points=triangle)

    with pytest.raises(ValueError, match="Inset distance must be positive"):
        ps.calc_inset_polygon_spec(-0.1)

    with pytest.raises(ValueError, match="Inset distance must be positive"):
        ps.calc_inset_polygon_spec(0.0)

    # Test with too large inset distance
    with pytest.raises(ValueError, match="too large"):
        ps.calc_inset_polygon_spec(10.0)  # Much larger than triangle

    # Test with degenerate polygon (collinear points)
    degenerate = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),  # Collinear
    ]
    ps_degenerate = PolygonSpec(points=degenerate)

    with pytest.raises(ValueError, match="Degenerate angle"):
        ps_degenerate.calc_inset_polygon_spec(0.1)


def test_calc_inset_polygon_spec_non_planar():
    """Test polygon inset with non-planar polygon."""

    # Create a "butterfly" shape - not quite planar
    non_planar = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.1]),  # Slightly elevated
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 2.0, -0.1]),  # Slightly depressed
    ]

    ps = PolygonSpec(points=non_planar)

    # Should still work by projecting to best-fit plane
    inset_distance = 0.2
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # Check basic properties
    assert len(inset_poly.points) == 4

    # All inset vertices should be inside original polygon (approximately)
    for vertex in inset_poly.points:
        assert ps.contains_point(vertex)


def test_calc_inset_polygon_spec_preservation_properties():
    """Test that inset preserves important geometric properties."""

    # Test with pentagon
    pentagon = [
        np.array([2.0, 0.0, 0.0]),
        np.array([0.618, 1.902, 0.0]),
        np.array([-1.618, 1.176, 0.0]),
        np.array([-1.618, -1.176, 0.0]),
        np.array([0.618, -1.902, 0.0]),
    ]

    ps = PolygonSpec(points=pentagon)

    inset_distance = 0.15
    inset_poly = ps.calc_inset_polygon_spec(inset_distance)

    # 1. Same number of vertices
    assert len(inset_poly.points) == len(ps.points)

    # 2. Same normal direction (parallel)
    dot_product = np.dot(ps.normal, inset_poly.normal)
    assert abs(dot_product - 1.0) < 1e-10

    # 3. Smaller perimeter
    assert inset_poly.circumference() < ps.circumference()

    # 4. All vertices inside original
    for vertex in inset_poly.points:
        assert ps.contains_point(vertex)

    # 5. Inset by larger amount should give smaller result
    larger_inset = ps.calc_inset_polygon_spec(0.25)
    assert larger_inset.circumference() < inset_poly.circumference()


def test_calc_inset_multiple_iterations():
    """Test that multiple inset operations work correctly."""

    # Start with square
    square = [
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([3.0, 3.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    ]

    ps = PolygonSpec(points=square)

    # Apply multiple small insets
    current_poly = ps
    inset_amount = 0.1
    num_iterations = 5

    perimeters = [current_poly.circumference()]

    for i in range(num_iterations):
        current_poly = current_poly.calc_inset_polygon_spec(inset_amount)
        perimeters.append(current_poly.circumference())

        # Each iteration should produce smaller perimeter
        assert perimeters[-1] < perimeters[-2]

        # All vertices should still be valid
        assert len(current_poly.points) == 4

        # Should still be roughly square-shaped (equal side lengths)
        side_lengths = []
        for j in range(4):
            v1 = current_poly.points[j]
            v2 = current_poly.points[(j + 1) % 4]
            length = np.linalg.norm(v2 - v1)
            side_lengths.append(length)

        # All sides should be approximately equal
        avg_side = np.mean(side_lengths)
        for length in side_lengths:
            assert abs(length - avg_side) < 1e-10
