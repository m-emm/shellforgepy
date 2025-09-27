import numpy as np
from shellforgepy.construct.construct_utils import is_valid_rigid_transform
from shellforgepy.geometry.spherical_tools import (
    cartesian_to_spherical_jackson,
    coordinate_system_transform,
    coordinate_system_transformation_function,
    create_shell_triangle_geometry,
    matrix_to_coordinate_system_transform,
    ray_plane_polygon_intersect,
    ray_triangle_intersect,
    rotation_matrix_from_vectors,
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
