import numpy as np
from shellforgepy.construct.construct_utils import rotation_matrix_from_vectors
from shellforgepy.geometry.spherical_tools import (
    cartesian_to_spherical_jackson,
    coordinate_system_transformation_function,
    ray_plane_polygon_intersect,
    ray_triangle_intersect,
    spherical_to_cartesian_jackson,
)


def test_cartesian_to_spherical_jackson():
    np.random.seed(42)  # For reproducibility
    size = 2000

    for _ in range(100):
        x, y, z = np.random.uniform(-size, size, 3)
        r, theta, phi = cartesian_to_spherical_jackson((x, y, z))

        xyz = spherical_to_cartesian_jackson((r, theta, phi))
        assert np.allclose((x, y, z), xyz, atol=1e-6), f"Failed for input: {(x, y, z)}"


def test_rotation_matrix_from_vectors_opposite_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = -a

    R = rotation_matrix_from_vectors(a, b)

    # It should rotate a to b
    a_rotated = R @ a
    assert np.allclose(
        a_rotated, b, atol=1e-6
    ), f"Rotation failed: got {a_rotated}, expected {b}"

    # It should be a proper rotation matrix
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "Matrix is not orthogonal"
    assert np.isclose(
        np.linalg.det(R), 1.0, atol=1e-6
    ), "Determinant is not 1 (not a proper rotation)"


def test_ray_triangle_intersect():
    # Define a triangle in 3D space
    triangle_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Define a ray that intersects the triangle
    ray_origin = np.array([0.5, 0.5, -1.0])
    ray_direction = np.array([0.0, 0.0, 1.0])  # Pointing upwards

    intersection_point = ray_triangle_intersect(
        ray_origin, ray_direction, triangle_vertices
    )

    assert intersection_point is not None, "Expected an intersection point"
    assert np.allclose(
        intersection_point, [0.5, 0.5, 0.0]
    ), "Intersection point mismatch"


class Rotated:
    def __init__(self, object, angle, axis):
        self.object = object
        self.angle = angle
        self.axis = axis

    def __repr__(self):
        return f"Rotated({self.object}, {self.angle}, {self.axis})"


class Translated:

    def __init__(self, object, x, y, z):
        self.object = object
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Translated({self.object}, {self.x}, {self.y}, {self.z})"


def test_coordinate_system_transformation_function():

    origin_a = (0, 0, 0)
    up_a = (0, 0, 1)
    out_a = (1, 0, 0)
    origin_b = [-91.34176083, 62.75628474, 100.00960039]
    up_b = (1, 0, 0)
    out_b = [-0.85301514, 0.08469599, 0.51496773]

    def rotation_function_generator(angle, axis):
        def retval(x):
            return Rotated(x, angle, axis)

        return retval

    def translation_function_generator(x, y, z):
        def retval(obj):
            return Translated(obj, x, y, z)

        return retval

    cstf = coordinate_system_transformation_function(
        origin_a,
        up_a,
        out_a,
        origin_b,
        up_b,
        out_b,
        rotation_function_generator,
        translation_function_generator,
    )

    print(cstf("object"))


def test_ray_plane_polygon_intersect_triangle_hit():
    """Test ray intersecting with a triangle polygon."""
    # Define a triangle in the xy-plane
    polygon = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])

    # Ray from below pointing upward, hitting the center of the triangle
    ray_origin = np.array([1.0, 0.5, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection with triangle"
    expected_point = np.array([1.0, 0.5, 0.0])
    assert np.allclose(
        intersection, expected_point, atol=1e-6
    ), f"Expected {expected_point}, got {intersection}"


def test_ray_plane_polygon_intersect_triangle_miss():
    """Test ray missing a triangle polygon."""
    # Define a triangle in the xy-plane
    polygon = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])

    # Ray from below pointing upward, but missing the triangle
    ray_origin = np.array([5.0, 5.0, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is None, "Expected no intersection with triangle"


def test_ray_plane_polygon_intersect_square_hit():
    """Test ray intersecting with a square polygon."""
    # Define a square in the xy-plane
    polygon = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    )

    # Ray from below pointing upward, hitting the center of the square
    ray_origin = np.array([1.0, 1.0, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection with square"
    expected_point = np.array([1.0, 1.0, 0.0])
    assert np.allclose(
        intersection, expected_point, atol=1e-6
    ), f"Expected {expected_point}, got {intersection}"


def test_ray_plane_polygon_intersect_square_edge():
    """Test ray intersecting with the edge of a square polygon."""
    # Define a square in the xy-plane
    polygon = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    )

    # Ray hitting the edge of the square
    ray_origin = np.array([2.0, 1.0, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection at square edge"
    expected_point = np.array([2.0, 1.0, 0.0])
    assert np.allclose(
        intersection, expected_point, atol=1e-6
    ), f"Expected {expected_point}, got {intersection}"


def test_ray_plane_polygon_intersect_parallel_ray():
    """Test ray parallel to the polygon plane."""
    # Define a triangle in the xy-plane
    polygon = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])

    # Ray parallel to the xy-plane (parallel to polygon)
    ray_origin = np.array([1.0, 1.0, 1.0])
    ray_vector = np.array([1.0, 0.0, 0.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is None, "Expected no intersection for parallel ray"


def test_ray_plane_polygon_intersect_behind_origin():
    """Test ray where intersection would be behind the ray origin."""
    # Define a triangle in the xy-plane
    polygon = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])

    # Ray pointing away from the polygon
    ray_origin = np.array([1.0, 1.0, 1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])  # pointing away from polygon

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is None, "Expected no intersection for ray pointing away"


def test_ray_plane_polygon_intersect_angled_plane():
    """Test ray intersecting with a polygon in an angled plane."""
    # Define a triangle in a plane tilted 45 degrees around x-axis
    # Original triangle: [(0,0,0), (2,0,0), (1,2,0)]
    # Rotated 45 degrees around x-axis
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    polygon = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0 * cos45, 2.0 * sin45]]
    )

    # Ray from below pointing upward
    ray_origin = np.array([1.0, 0.5 * cos45, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection with angled polygon"
    # Verify intersection is on the correct plane
    assert intersection[0] == 1.0, "X coordinate should match ray origin"
    assert intersection[1] == 0.5 * cos45, "Y coordinate should match ray origin"


def test_ray_plane_polygon_intersect_pentagon():
    """Test ray intersecting with a pentagon polygon."""
    # Define a regular pentagon in the xy-plane
    n_sides = 5
    radius = 1.0
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    polygon = np.array(
        [[radius * np.cos(angle), radius * np.sin(angle), 0.0] for angle in angles]
    )

    # Ray from below pointing upward, hitting the center
    ray_origin = np.array([0.0, 0.0, -1.0])
    ray_vector = np.array([0.0, 0.0, 1.0])

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection with pentagon"
    expected_point = np.array([0.0, 0.0, 0.0])
    assert np.allclose(
        intersection, expected_point, atol=1e-6
    ), f"Expected {expected_point}, got {intersection}"


def test_ray_plane_polygon_intersect_oblique_ray():
    """Test ray with oblique direction intersecting polygon."""
    # Define a square in the xy-plane
    polygon = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    )

    # Oblique ray that should hit the center of the square
    ray_origin = np.array([0.0, 0.0, -1.0])
    ray_vector = np.array([1.0, 1.0, 1.0])  # normalized direction
    ray_vector = ray_vector / np.linalg.norm(ray_vector)

    intersection = ray_plane_polygon_intersect(ray_origin, ray_vector, polygon)

    assert intersection is not None, "Expected intersection with oblique ray"
    # The intersection should be at (1, 1, 0) when the ray hits the plane
    expected_point = np.array([1.0, 1.0, 0.0])
    assert np.allclose(
        intersection, expected_point, atol=1e-6
    ), f"Expected {expected_point}, got {intersection}"


def test_shrink_triangle():
    """Test triangle shrinking functionality."""
    from shellforgepy.geometry.spherical_tools import shrink_triangle

    # Define a simple triangle
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.5, 1.0, 0.0])

    border_width = 0.1

    A_shrunk, B_shrunk, C_shrunk = shrink_triangle(A, B, C, border_width)

    # Check that shrunk triangle vertices are arrays
    assert isinstance(A_shrunk, np.ndarray)
    assert isinstance(B_shrunk, np.ndarray)
    assert isinstance(C_shrunk, np.ndarray)

    # Check that all coordinates are finite
    assert np.all(np.isfinite(A_shrunk))
    assert np.all(np.isfinite(B_shrunk))
    assert np.all(np.isfinite(C_shrunk))

    # Check that the shrunk triangle is smaller than the original
    original_area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    shrunk_area = 0.5 * np.linalg.norm(
        np.cross(B_shrunk - A_shrunk, C_shrunk - A_shrunk)
    )

    assert shrunk_area < original_area, "Shrunk triangle should have smaller area"


def test_shrink_triangle_edge_cases():
    """Test edge cases for triangle shrinking."""
    from shellforgepy.geometry.spherical_tools import shrink_triangle

    # Test with very small border width
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])

    small_border = 1e-6
    A_s, B_s, C_s = shrink_triangle(A, B, C, small_border)

    # Results should be close to original but slightly smaller
    assert np.allclose(A_s, A, atol=1e-4)
    assert np.allclose(B_s, B, atol=1e-4)
    assert np.allclose(C_s, C, atol=1e-4)

    # Test with degenerate triangle (should raise error)
    A_deg = np.array([0.0, 0.0, 0.0])
    B_deg = np.array([0.0, 0.0, 0.0])  # Same as A
    C_deg = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        shrink_triangle(A_deg, B_deg, C_deg, 0.1)


def test_create_shell_triangle_geometry():
    """Test shell triangle geometry creation."""
    from shellforgepy.geometry.spherical_tools import create_shell_triangle_geometry

    # Define triangle vertices in spherical coordinates (r, theta, phi)
    triangle_spherical_vertexes = [
        (1.0, 0.5, 0.0),  # vertex 1
        (1.0, 0.5, 1.0),  # vertex 2
        (1.0, 1.0, 0.5),  # vertex 3
    ]

    sphere_center = np.array([0.0, 0.0, 0.0])
    shell_thickness = 0.1

    result = create_shell_triangle_geometry(
        triangle_spherical_vertexes, sphere_center, shell_thickness
    )

    # Should return a dictionary with vertex and face information
    assert isinstance(result, dict)
    assert "vertexes" in result
    assert "faces" in result

    vertexes = result["vertexes"]
    faces = result["faces"]

    # Should have 6 vertices (3 inner + 3 outer)
    assert len(vertexes) == 6
    # Should have 8 faces for the prism
    assert len(faces) == 8

    # Check vertex structure
    for vertex_id, vertex in vertexes.items():
        assert isinstance(vertex_id, int)
        assert isinstance(vertex, np.ndarray)
        assert np.all(np.isfinite(vertex))
        assert vertex.shape == (3,)

    # Check face structure
    for face_id, face in faces.items():
        assert isinstance(face_id, int)
        assert isinstance(face, list)
        assert len(face) == 3  # triangular faces
        assert all(v in vertexes for v in face)  # face vertices should exist


def test_is_inside_convex_polygon_2d():
    """Test 2D point-in-polygon detection."""
    from shellforgepy.geometry.spherical_tools import is_inside_convex_polygon_2d

    # Define a square polygon
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # Test points inside and outside
    test_points = np.array(
        [
            [0.5, 0.5],  # inside
            [0.1, 0.1],  # inside
            [0.9, 0.9],  # inside
            [1.5, 0.5],  # outside
            [-0.1, 0.5],  # outside
            [0.5, 1.5],  # outside
            [0.5, -0.1],  # outside
        ]
    )

    expected_inside = np.array([True, True, True, False, False, False, False])

    result = is_inside_convex_polygon_2d(polygon, test_points)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == len(test_points)

    np.testing.assert_array_equal(result, expected_inside)


def test_spherical_conversions():
    """Test basic spherical coordinate conversions."""
    from shellforgepy.geometry.spherical_tools import (
        cartesian_to_spherical,
        spherical_to_cartesian,
    )

    # Test with arrays (these functions expect arrays)
    thetas = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi])
    phis = np.array([0.0, 0.0, np.pi / 2, np.pi / 4, np.pi])

    # Convert to cartesian
    xyz = spherical_to_cartesian(thetas, phis)

    assert isinstance(xyz, np.ndarray)
    assert xyz.shape == (5, 3)
    assert np.all(np.isfinite(xyz))

    # Convert back to spherical
    theta_phi_back = cartesian_to_spherical(xyz)

    assert isinstance(theta_phi_back, np.ndarray)
    assert theta_phi_back.shape == (5, 2)

    thetas_back = theta_phi_back[:, 0]
    phis_back = theta_phi_back[:, 1]

    # Check consistency (handling special cases)
    for i, (theta, phi, theta_back, phi_back) in enumerate(
        zip(thetas, phis, thetas_back, phis_back)
    ):
        if abs(theta) < 1e-10:  # At poles, phi is undefined
            assert abs(theta_back) < 1e-10
        else:
            assert np.isclose(theta, theta_back, atol=1e-10)
            # Handle phi periodicity
            phi_diff = abs(phi - phi_back)
            phi_diff_mod = min(phi_diff, abs(phi_diff - 2 * np.pi))
            assert phi_diff_mod < 1e-10


def test_azimuthal_projection():
    """Test azimuthal projection."""
    from shellforgepy.geometry.spherical_tools import azimuthal_projection

    # Test with array of theta-phi pairs
    theta_phi = np.array(
        [
            [0.5, 1.0],
            [0.1, 0.5],
            [1.0, 2.0],
            [1.5, 0.0],
        ]
    )

    # Basic projection
    xy = azimuthal_projection(theta_phi)

    assert isinstance(xy, np.ndarray)
    assert xy.shape == (4, 2)  # 4 points, 2 coordinates each
    assert np.all(np.isfinite(xy))

    # With extension
    xy_ext = azimuthal_projection(theta_phi, extension=0.1)

    assert isinstance(xy_ext, np.ndarray)
    assert xy_ext.shape == (4, 2)
    assert np.all(np.isfinite(xy_ext))


def test_filter_outside_spherical_cap():
    """Test spherical cap filtering."""
    from shellforgepy.geometry.spherical_tools import filter_outside_spherical_cap

    # Define multiple cap points (need at least 3 for convex hull)
    cap_theta_phi = np.array(
        [
            [0.0, 0.0],  # North pole
            [0.1, 0.0],  # Close point 1
            [0.1, np.pi / 2],  # Close point 2
            [0.1, np.pi],  # Close point 3
        ]
    )

    # Test points at various locations
    test_points = np.array(
        [
            [0.05, 0.0],  # Very close to cap center
            [0.5, 0.0],  # Moderate distance
            [1.0, 0.0],  # Farther away
            [1.5, 0.0],  # Very far
            [np.pi / 2, 0.0],  # On equator
        ]
    )

    # Function returns (filtered_points, mask, hull_vertices)
    filtered_points, mask_outside, hull_vertices = filter_outside_spherical_cap(
        cap_theta_phi, test_points
    )

    assert isinstance(filtered_points, np.ndarray)
    assert filtered_points.ndim == 2  # Should return points, not boolean mask
    assert filtered_points.shape[1] == 2  # theta, phi coordinates

    assert isinstance(mask_outside, np.ndarray)
    assert isinstance(hull_vertices, np.ndarray)

    # The function returns points that are outside the cap
    # Farther points should be more likely to be in the result
    assert filtered_points.shape[0] <= test_points.shape[0]


def test_coordinate_system_transform_to_matrix():
    """Test coordinate system transform to matrix conversion."""
    from shellforgepy.geometry.spherical_tools import (
        coordinate_system_transform_to_matrix,
    )

    # Define a simple transform (identity rotation)
    transform = {
        "translation": [1.0, 2.0, 3.0],
        "rotation_axis": [1.0, 0.0, 0.0],
        "rotation_angle": 0.0,
    }

    matrix = coordinate_system_transform_to_matrix(transform)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)

    # Check that it's a valid transformation matrix
    expected_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(matrix, expected_matrix)


def test_matrix_to_coordinate_system_transform():
    """Test matrix to coordinate system transform conversion."""
    from shellforgepy.geometry.spherical_tools import (
        matrix_to_coordinate_system_transform,
    )

    # Define a transformation matrix
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transform = matrix_to_coordinate_system_transform(matrix)

    assert isinstance(transform, dict)
    assert "translation" in transform
    assert "rotation_axis" in transform
    assert "rotation_angle" in transform

    # Check values
    np.testing.assert_array_almost_equal(transform["translation"], [1.0, 2.0, 3.0])
    assert np.isclose(transform["rotation_angle"], 0.0)
    assert isinstance(transform["rotation_axis"], tuple)


import pytest
