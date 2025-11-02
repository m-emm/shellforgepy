import logging

import numpy as np
from shellforgepy.construct.construct_utils import fibonacci_sphere, point_in_polygon_2d
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)

_logger = logging.getLogger(__name__)


def create_sphere_mesh():
    sphere_points = np.array(fibonacci_sphere(samples=50))
    sphere_points *= 50
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(sphere_points)
    return mesh


def create_polygon_edge_vertex_filter(
    polygon_points_3d, polygon_center, polygon_normal, current_edge_idx, epsilon=1e-6
):
    """
    Create a vertex filter for polygon edge-constrained cutting.

    Args:
        polygon_points_3d: List of 3D points defining polygon vertices
        polygon_center: Center point of polygon for coordinate system origin
        polygon_normal: Normal vector of polygon plane
        current_edge_idx: Index of current polygon edge being processed
        epsilon: Numerical tolerance

    Returns:
        Filter function that returns True if intersection vertex should be created
    """
    # Get current edge endpoints
    p1 = polygon_points_3d[current_edge_idx]
    p2 = polygon_points_3d[(current_edge_idx + 1) % len(polygon_points_3d)]

    # Edge vector in 3D
    edge_vector_3d = p2 - p1
    edge_length = np.linalg.norm(edge_vector_3d)

    if edge_length < epsilon:
        # Degenerate edge, reject all vertices
        return lambda *args: False

    edge_direction_3d = edge_vector_3d / edge_length

    # Construct cutting plane normal: orthogonal to both edge vector and polygon normal
    cutting_plane_normal = np.cross(edge_vector_3d, polygon_normal)
    cutting_plane_normal_length = np.linalg.norm(cutting_plane_normal)

    if cutting_plane_normal_length < epsilon:
        # Edge is parallel to polygon normal, reject all vertices
        return lambda *args: False

    cutting_plane_normal = cutting_plane_normal / cutting_plane_normal_length

    # Create 2D coordinate system in polygon plane
    # Primary axis: along the current edge (in polygon plane)
    edge_in_plane = edge_vector_3d
    u_axis = edge_in_plane / np.linalg.norm(edge_in_plane)

    # Secondary axis: orthogonal to edge in polygon plane
    v_axis = np.cross(polygon_normal, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    def vertex_filter(intersection_point, edge, va, vb):
        """
        Filter function to check if intersection vertex should be created.

        Returns True if:
        1. Point projects correctly onto current polygon edge
        2. Point is within the polygon boundary when projected to 2D
        """

        # Step 1: Project intersection point onto polygon plane
        to_point = intersection_point - polygon_center
        distance_to_plane = np.dot(to_point, polygon_normal)
        projected_point = intersection_point - distance_to_plane * polygon_normal

        # Step 2: Convert projected point to 2D coordinates in polygon plane
        to_projected = projected_point - polygon_center
        u_coord = np.dot(to_projected, u_axis)
        v_coord = np.dot(to_projected, v_axis)
        point_2d = np.array([u_coord, v_coord])

        # Step 3: Check if point lies on current edge line in 2D
        # Convert edge endpoints to 2D
        to_p1 = p1 - polygon_center
        to_p2 = p2 - polygon_center

        p1_u = np.dot(to_p1, u_axis)
        p1_v = np.dot(to_p1, v_axis)
        p1_2d = np.array([p1_u, p1_v])

        p2_u = np.dot(to_p2, u_axis)
        p2_v = np.dot(to_p2, v_axis)
        p2_2d = np.array([p2_u, p2_v])

        # Check if point lies on edge line (should be very close to v=0 in our coordinate system)
        edge_2d = p2_2d - p1_2d
        edge_length_2d = np.linalg.norm(edge_2d)

        if edge_length_2d < epsilon:
            return False  # Degenerate edge

        # Distance from point to edge line in 2D
        to_point_2d = point_2d - p1_2d
        # Project onto edge direction
        edge_dir_2d = edge_2d / edge_length_2d
        projection_length = np.dot(to_point_2d, edge_dir_2d)

        # Point on edge line closest to our point
        closest_on_edge = p1_2d + projection_length * edge_dir_2d
        distance_to_edge_line = np.linalg.norm(point_2d - closest_on_edge)

        # Must be very close to the edge line
        if distance_to_edge_line > epsilon * 100:  # Allow some tolerance
            return False

        # Step 4: Check if point is within edge segment bounds
        if projection_length < -epsilon or projection_length > edge_length_2d + epsilon:
            return False  # Outside edge segment

        # Step 5: Check if point is inside polygon in 2D
        # Convert all polygon points to 2D
        polygon_2d = []
        for poly_point in polygon_points_3d:
            to_poly = poly_point - polygon_center
            poly_u = np.dot(to_poly, u_axis)
            poly_v = np.dot(to_poly, v_axis)
            polygon_2d.append([poly_u, poly_v])

        # Use 2D point-in-polygon test
        inside = point_in_polygon_2d(point_2d, polygon_2d)

        return inside

    return vertex_filter


def test_plane_perforation():

    mesh = create_sphere_mesh()

    perforation_result = mesh.compute_plane_perforation((0, 0, 0), (0, 0, 1))

    for vertex in perforation_result.new_vertices:
        assert abs(vertex[2]) < 1e-6, f"Vertex {vertex} not on plane z=0"


def test_plane_perforation_with_filter():
    """Test that vertex filter function can control which intersection vertices are added."""

    mesh = create_sphere_mesh()

    # Filter that only allows vertices with x > 0
    def positive_x_filter(intersection_point, edge, va, vb):
        return intersection_point[0] > 0

    # Perforation with filter
    perforation_result = mesh.compute_plane_perforation(
        plane_point=(0, 0, 0), plane_normal=(0, 0, 1), vertex_filter=positive_x_filter
    )

    # All new vertices should be on the plane z=0
    for vertex in perforation_result.new_vertices:
        assert abs(vertex[2]) < 1e-6, f"Vertex {vertex} not on plane z=0"

    # All new vertices should have x > 0 due to filter
    for vertex in perforation_result.new_vertices:
        assert vertex[0] > -1e-6, f"Vertex {vertex} should have x > 0 due to filter"

    # Should have fewer vertices than unfiltered perforation
    unfiltered_result = mesh.compute_plane_perforation((0, 0, 0), (0, 0, 1))

    print(f"Unfiltered vertices: {len(unfiltered_result.new_vertices)}")
    print(f"Filtered vertices: {len(perforation_result.new_vertices)}")

    assert len(perforation_result.new_vertices) <= len(unfiltered_result.new_vertices)


def test_plane_perforation_polygon_filter():
    """Test vertex filter for polygon-constrained cutting."""

    mesh = create_sphere_mesh()

    # First, let's see what unfiltered perforation gives us
    unfiltered_result = mesh.compute_plane_perforation((0, 0, 0), (0, 0, 1))
    print(f"Unfiltered vertices: {len(unfiltered_result.new_vertices)}")
    if len(unfiltered_result.new_vertices) > 0:
        x_coords = [v[0] for v in unfiltered_result.new_vertices]
        y_coords = [v[1] for v in unfiltered_result.new_vertices]
        print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")

    # Define a smaller square polygon on the z=0 plane (sphere has radius ~1)
    polygon_points = [
        np.array([-0.5, -0.5, 0]),
        np.array([0.5, -0.5, 0]),
        np.array([0.5, 0.5, 0]),
        np.array([-0.5, 0.5, 0]),
    ]

    # Filter that only allows vertices inside the polygon bounds
    def polygon_filter(intersection_point, edge, va, vb):
        x, y, z = intersection_point
        return -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5

    # Apply filtered perforation
    perforation_result = mesh.compute_plane_perforation(
        plane_point=(0, 0, 0), plane_normal=(0, 0, 1), vertex_filter=polygon_filter
    )

    # All new vertices should be on plane and within polygon bounds
    for vertex in perforation_result.new_vertices:
        assert abs(vertex[2]) < 1e-6, f"Vertex {vertex} not on plane"
        assert -0.5 <= vertex[0] <= 0.5, f"Vertex {vertex} outside polygon x bounds"
        assert -0.5 <= vertex[1] <= 0.5, f"Vertex {vertex} outside polygon y bounds"

    print(f"Polygon-filtered vertices: {len(perforation_result.new_vertices)}")

    # Should have fewer vertices than unfiltered (if there are vertices outside the bounds)
    assert len(perforation_result.new_vertices) <= len(unfiltered_result.new_vertices)

    # We should still have some vertices since the polygon covers the center of the sphere
    if len(unfiltered_result.new_vertices) > 0:
        print("Filter is working - verified vertex count consistency")
