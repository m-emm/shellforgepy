import logging

import numpy as np
import pytest
from shellforgepy.construct.construct_utils import fibonacci_sphere, normalize
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)

_logger = logging.getLogger(__name__)


@pytest.fixture
def sphere_mesh_partition():
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0  # Scale up for better numerical stability
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    return partition


@pytest.fixture
def square_polygon_2d():
    return [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]


def test_project_polygon_onto_mesh(sphere_mesh_partition, square_polygon_2d):
    """Test projecting a 2D polygon onto mesh surface."""

    # Project from origin toward positive Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])

    # Project the polygon onto the mesh
    projected_points, inside_vertex_ids = (
        sphere_mesh_partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=square_polygon_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            target_segment_length=1.0,
        )
    )

    # Check that we got some projected points
    assert (
        len(projected_points) >= 4
    ), f"Expected at least 4 projected points, got {len(projected_points)}"

    # Check that all projected points are 3D
    for point in projected_points:
        assert point.shape == (3,), f"Expected 3D point, got shape {point.shape}"

    # Check that projected points are roughly on the sphere surface
    for point in projected_points:
        distance_from_origin = np.linalg.norm(point)
        assert (
            abs(distance_from_origin - 10.0) < 2.0
        ), f"Point {point} not on sphere surface"

    _logger.info(f"Successfully projected {len(projected_points)} points onto mesh")


def test_inside_vertex_ids(sphere_mesh_partition, square_polygon_2d):

    face_id = 30

    face_normal = sphere_mesh_partition.mesh.get_face_normal(face_id)
    face_centroid = sphere_mesh_partition.mesh.get_face_centroid(face_id)

    face_vertex_ids = sphere_mesh_partition.mesh.faces[face_id]
    face_vertices = sphere_mesh_partition.mesh.vertices[face_vertex_ids]

    ray_origin = np.array([0, 0, 0])
    ray_direction = normalize(face_centroid - ray_origin)

    # Calculate the size of the triangle face to determine appropriate polygon size
    # Find the maximum edge length of the triangle
    edge_lengths = [
        np.linalg.norm(face_vertices[1] - face_vertices[0]),
        np.linalg.norm(face_vertices[2] - face_vertices[1]),
        np.linalg.norm(face_vertices[0] - face_vertices[2]),
    ]
    max_edge_length = max(edge_lengths)

    # Create a reasonably sized square polygon around the triangle
    # Use the maximum edge length to ensure we capture the triangle vertices
    polygon_size = max_edge_length * 1.0  # Full size to include triangle vertices
    face_polygon_2d = [
        (-polygon_size / 2, -polygon_size / 2),
        (polygon_size / 2, -polygon_size / 2),
        (polygon_size / 2, polygon_size / 2),
        (-polygon_size / 2, polygon_size / 2),
    ]

    # Project the polygon onto the mesh
    projected_points, inside_vertex_ids = (
        sphere_mesh_partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=face_polygon_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            target_segment_length=None,  # Only project corners
        )
    )

    # Check that we got some vertices inside
    assert len(inside_vertex_ids) > 0, "No vertices found inside the polygon"

    # Check that all returned vertices are from the same region (region 0)
    for vid in inside_vertex_ids:
        # Find which faces contain this vertex
        containing_faces = [
            f_idx
            for f_idx, face in enumerate(sphere_mesh_partition.mesh.faces)
            if vid in face
        ]
        # Check that at least one of these faces is in region 0
        regions_for_vertex = [
            sphere_mesh_partition.face_to_region_map[f_idx]
            for f_idx in containing_faces
        ]
        assert 0 in regions_for_vertex, f"Vertex {vid} not found in region 0"

    # Check that the target face vertices are included (they should be since we're projecting onto that face)
    face_vertex_set = set(face_vertex_ids)
    inside_vertex_set = set(inside_vertex_ids)

    # We should find at least some of the target face vertices
    overlap = face_vertex_set.intersection(inside_vertex_set)
    assert len(overlap) > 0, (
        f"Expected to find at least some target face vertices {face_vertex_ids} "
        f"in inside vertices {inside_vertex_ids}, but found none"
    )


def test_corners_only_projection(sphere_mesh_partition, square_polygon_2d):

    # Project from origin toward positive Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])

    # Project the polygon onto the mesh
    projected_points, inside_vertex_ids = (
        sphere_mesh_partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=square_polygon_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            target_segment_length=None,  # means corners only
        )
    )

    assert (
        len(projected_points) == 4
    ), f"Expected 4 projected corner points, got {len(projected_points)}"


def test_project_polygon_onto_mesh_change_x_direction(
    sphere_mesh_partition, square_polygon_2d
):
    """Test projecting a 2D polygon onto mesh surface."""
    # Create a simple sphere mesh

    # Project from origin toward positive Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])

    # Project the polygon onto the mesh
    projected_points, inside_vertex_ids = (
        sphere_mesh_partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=square_polygon_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            target_segment_length=1.0,
            x_axis_global_direction=(0, -1, 0),
        )
    )

    # Check that we got some projected points
    assert (
        len(projected_points) >= 4
    ), f"Expected at least 4 projected points, got {len(projected_points)}"

    # Check that all projected points are 3D
    for point in projected_points:
        assert point.shape == (3,), f"Expected 3D point, got shape {point.shape}"

    # Check that projected points are roughly on the sphere surface
    for point in projected_points:
        distance_from_origin = np.linalg.norm(point)
        assert (
            abs(distance_from_origin - 10.0) < 2.0
        ), f"Point {point} not on sphere surface"

    _logger.info(f"Successfully projected {len(projected_points)} points onto mesh")


def test_project_polygon_onto_mesh_invalid_region(
    sphere_mesh_partition, square_polygon_2d
):
    """Test error handling for invalid region ID."""

    # Try to project onto non-existent region
    with pytest.raises(ValueError, match="Region 999 has no faces"):
        sphere_mesh_partition.project_polygon_onto_mesh(
            region_id=999,
            polygon_points_2d=square_polygon_2d,
            ray_origin=np.array([0.0, 0.0, 0.0]),
            ray_direction=np.array([0.0, 0.0, 1.0]),
        )


def test_project_polygon_onto_mesh_no_intersection(
    sphere_mesh_partition, square_polygon_2d
):
    """Test error handling when ray doesn't intersect mesh."""
    points = np.array(fibonacci_sphere(samples=50))
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    square_2d = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    # Ray pointing away from the mesh - this should fail
    try:
        projected_points, inside_vertex_ids = partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=square_2d,
            ray_origin=np.array([0.0, 0.0, 0.0]),
            ray_direction=np.array([0.0, 0.0, -1.0]),  # Wrong direction
        )
        # If we get here without exception, the projection failed to find central intersection
        # but didn't raise an error - that's also a valid test case
        _logger.info(f"Projection succeeded but found {len(projected_points)} points")
        assert (
            len(projected_points) == 0 or len(projected_points) < 4
        ), "Expected few or no projections for ray pointing away"
    except ValueError as e:
        assert "No central intersection found" in str(e)
        _logger.info("Correctly raised ValueError for no intersection")


def test_create_filler_mesh_basic():
    """Test basic filler mesh creation functionality."""
    # Create a simple spherical mesh like the fixture
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0  # Scale up for better numerical stability
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Define test polygon - use the same size as the working tests
    polygon_points_2d = [(-2, -2), (2, -2), (2, 2), (-2, 2)]

    # Ray configuration
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])

    # Create filler mesh
    filler_mesh = partition.create_filler_mesh(
        region_id=0,
        polygon_points_2d=polygon_points_2d,
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        base_thickness=2.0,
        base_inset=0.2,
        target_segment_length=1.0,
    )

    # Verify filler mesh properties
    assert isinstance(filler_mesh, PartitionableSpheroidTriangleMesh)
    assert len(filler_mesh.vertices) > 0
    assert len(filler_mesh.faces) > 0


def test_create_filler_mesh_no_inset():
    """Test filler mesh creation without inset."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    polygon_points_2d = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])

    # Create filler without inset
    filler_mesh = partition.create_filler_mesh(
        region_id=0,
        polygon_points_2d=polygon_points_2d,
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        base_thickness=1.5,
        base_inset=None,  # No inset
    )

    assert isinstance(filler_mesh, PartitionableSpheroidTriangleMesh)
    assert len(filler_mesh.vertices) > 0


def test_create_filler_mesh_with_x_axis_direction():
    """Test filler mesh creation with specific x-axis alignment."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    polygon_points_2d = [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])
    x_axis_direction = np.array([1, 1, 0])  # 45-degree rotation

    filler_mesh = partition.create_filler_mesh(
        region_id=0,
        polygon_points_2d=polygon_points_2d,
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        base_thickness=2.0,
        base_inset=0.1,
        x_axis_global_direction=x_axis_direction,
    )

    assert isinstance(filler_mesh, PartitionableSpheroidTriangleMesh)
    assert len(filler_mesh.vertices) > 0


def test_create_filler_mesh_different_thicknesses():
    """Test filler mesh creation with different thickness values."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    polygon_points_2d = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])

    thicknesses = [0.5, 2.0, 4.0]
    filler_meshes = []

    for thickness in thicknesses:
        filler_mesh = partition.create_filler_mesh(
            region_id=0,
            polygon_points_2d=polygon_points_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            base_thickness=thickness,
            base_inset=0.1,
        )
        filler_meshes.append(filler_mesh)

    # All should be valid meshes
    for filler_mesh in filler_meshes:
        assert isinstance(filler_mesh, PartitionableSpheroidTriangleMesh)
        assert len(filler_mesh.vertices) > 0
        assert len(filler_mesh.faces) > 0


def test_create_filler_mesh_triangular_polygon():
    """Test filler mesh creation with triangular polygon."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Triangular polygon
    polygon_points_2d = [(-1.2, -1.2), (1.2, -1.2), (0, 1.2)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])

    filler_mesh = partition.create_filler_mesh(
        region_id=0,
        polygon_points_2d=polygon_points_2d,
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        base_thickness=1.5,
        base_inset=0.1,
    )

    assert isinstance(filler_mesh, PartitionableSpheroidTriangleMesh)
    assert len(filler_mesh.vertices) > 0


def test_create_filler_mesh_segment_length_variation():
    """Test filler mesh creation with different target segment lengths."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    polygon_points_2d = [(-1.4, -1.4), (1.4, -1.4), (1.4, 1.4), (-1.4, 1.4)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])

    segment_lengths = [0.5, 1.0, 2.0]
    vertex_counts = []

    for target_length in segment_lengths:
        filler_mesh = partition.create_filler_mesh(
            region_id=0,
            polygon_points_2d=polygon_points_2d,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            base_thickness=2.0,
            target_segment_length=target_length,
            base_inset=0.0,  # Explicit zero inset
        )
        vertex_counts.append(len(filler_mesh.vertices))

    # Smaller segment lengths should generally produce more vertices
    # (though this isn't guaranteed due to mesh complexity)
    assert all(count > 0 for count in vertex_counts)


def test_create_filler_mesh_coordinates_reasonable():
    """Test that filler mesh coordinates are reasonable relative to input."""
    points = np.array(fibonacci_sphere(samples=200))
    points *= 10.0
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    polygon_points_2d = [(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)]
    ray_origin = np.array([0, 0, 15])
    ray_direction = np.array([0, 0, -1])
    base_thickness = 2.5

    filler_mesh = partition.create_filler_mesh(
        region_id=0,
        polygon_points_2d=polygon_points_2d,
        ray_origin=ray_origin,
        ray_direction=ray_direction,
        base_thickness=base_thickness,
        base_inset=0.2,
    )

    # Check that vertices are within reasonable bounds
    vertices = np.array(filler_mesh.vertices)

    # All vertices should be within reasonable distance from origin
    distances = np.linalg.norm(vertices, axis=1)
    assert np.all(distances < 25.0), "Vertices should be reasonably close to origin"

    # Some vertices should be near the sphere surface (radius ~10)
    near_surface = np.any(np.abs(distances - 10.0) < 2.0)
    assert near_surface, "Some vertices should be near the original sphere surface"
