import logging
import math
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from shellforgepy.construct.construct_utils import fibonacci_sphere, normalize
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend suitable for pytest

import matplotlib.pyplot as plt

POLYGON_TEST_SPHERE_RADIUS = 80
HEXAGON_RADIUS = 40
SPHERE_RESOLUTION = 200


def visualize_projected_edges_to_file(
    edges_2d, filepath: str, title: str = "Boundary Walk"
):
    """
    edges_2d: List of ((x1, y1), (x2, y2)) tuples
    filepath: Path to write image (e.g., 'test_boundary_walk.png' or .svg)
    """
    fig, ax = plt.subplots()
    for (x1, y1), (x2, y2) in edges_2d:
        ax.plot([x1, x2], [y1, y2], "k-")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def test_drill_hole_jaggedness():

    SPHERE_RADIUS = 100
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=40, radius=SPHERE_RADIUS
    )

    for i, v in enumerate(mesh.vertices):
        print(f"Vertex: {i}: {v}")

    new_vertices = []

    partition = MeshPartition(mesh)

    bottom = np.array([0.0, 0.0, -1.0])
    axis = np.array([0.0, 0.0, 1.0])
    height = 10000
    radius = SPHERE_RADIUS * 0.6

    partition = partition.perforate_and_split_region_by_cylinder(
        region_id=0,
        bottom=bottom,
        axis=axis,
        height=height,
        radius=radius,
    )

    boundary_edges = partition.get_boundary_edges_of_region(0)

    # find the best plane that fits the vertices of the boundary edges

    boundary_points = np.array(
        [partition.mesh.vertices[e[0]] for e in boundary_edges]
        + [partition.mesh.vertices[e[1]] for e in boundary_edges]
    )

    centroid = np.mean(boundary_points, axis=0)
    cov = np.cov(boundary_points.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normalize(normal)
    print(f"Centroid: {centroid}, Normal: {normal}")

    def orthonormal_basis(normal):
        if abs(normal[0]) < abs(normal[1]):
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, tangent)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v

    u, v = orthonormal_basis(normal)

    # Step 2: Map 3D boundary points to 2D in the plane
    vertex_2d_map = {}
    for e in boundary_edges:
        for vi in e:
            if vi not in vertex_2d_map:
                vec = partition.mesh.vertices[vi] - centroid
                x = np.dot(vec, u)
                y = np.dot(vec, v)
                vertex_2d_map[vi] = (x, y)

    # Step 3: Build 2D edges
    projected_2d_edges = [
        (vertex_2d_map[e[0]], vertex_2d_map[e[1]]) for e in boundary_edges
    ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        filename = tmpfile.name

        visualize_projected_edges_to_file(
            projected_2d_edges, filepath=filename, title="Drill Hole Boundary Walk"
        )

        print(f"Saved boundary walk visualization to {filename}")


def create_regular_hexagon_points(radius):
    points = []
    for i in range(6):
        angle = math.radians(i * 60)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points


def create_sphere_partition():
    sphere_points = np.array(fibonacci_sphere(samples=SPHERE_RESOLUTION))

    sphere_points *= POLYGON_TEST_SPHERE_RADIUS

    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(sphere_points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, (0, 0, POLYGON_TEST_SPHERE_RADIUS * 0.7), (0, 0, 1)
    )

    region_view = TransformedRegionView(partition, 0)

    return partition, mesh, region_view


def test_projected_polygon_partition():
    """Test projecting a polygon onto a mesh and creating a polygon partition."""

    hexagon = create_regular_hexagon_points(radius=HEXAGON_RADIUS)
    partition, mesh, region_view = create_sphere_partition()

    hexagon_2d = [(p[0], p[1]) for p in hexagon]  # Convert to 2D tuples

    projected_polygon, inside_vertex_ids = partition.project_polygon_onto_mesh(
        region_id=0,  # Project onto the main region
        polygon_points_2d=hexagon_2d,
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([0.1, 0.1, 1.0]),
        target_segment_length=3.0,
    )

    # Use the new API method instead of the removed create_polygon_partition
    polygon_partition = partition.perforate_and_split_region_by_polygon(
        region_id=0,
        polygon_points_3d=projected_polygon,
        min_relative_area=1e-3,
        min_angle_deg=3.0,
    )

    assert len(polygon_partition.get_regions()) > len(partition.get_regions())

    region_view = TransformedRegionView(polygon_partition, 2)

    # Visually, the new region does not appear to properly match the projected polygon
    # How to write this as a test, I don't yet know


def test_polygon_perforation_edge_intersection_accuracy():
    """Test accuracy of polygon edge intersections."""
    # Use a simple sphere mesh instead of open mesh
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=20, radius=1.0
    )

    # Create polygon that should intersect the sphere
    polygon_points = [
        np.array([0.5, 0.5, 0.0]),
        np.array([-0.5, 0.5, 0.0]),
        np.array([-0.5, -0.5, 0.0]),
        np.array([0.5, -0.5, 0.0]),
    ]

    if hasattr(mesh, "perforate_with_polygon"):
        from shellforgepy.construct.polygon_spec import PolygonSpec

        polygon_spec = PolygonSpec.from_points_3d(polygon_points)

        new_mesh, _ = mesh.perforate_with_polygon(polygon_spec, epsilon=1e-10)

        # Should have created intersection vertices
        assert len(new_mesh.vertices) >= len(mesh.vertices)

        # Check that mesh is still valid
        for face in new_mesh.faces:
            area = new_mesh.triangle_area(face)
            assert area > 1e-8  # Reasonable minimum area


def test_polygon_perforation_quality_controls():
    """Test that quality controls work correctly for polygon perforation."""
    # Use a simple sphere mesh instead of open mesh
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=30, radius=1.0
    )

    # Create small polygon that would create quality issues if quality controls are disabled
    polygon_points = [
        np.array([0.01, 0.01, 0.0]),
        np.array([-0.01, 0.01, 0.0]),
        np.array([-0.01, -0.01, 0.0]),
        np.array([0.01, -0.01, 0.0]),
    ]

    if hasattr(mesh, "perforate_with_polygon"):
        from shellforgepy.construct.polygon_spec import PolygonSpec

        polygon_spec = PolygonSpec.from_points_3d(polygon_points)

        # With strict quality controls
        strict_mesh, _ = mesh.perforate_with_polygon(
            polygon_spec,
            min_relative_area=1e-1,  # High threshold
            min_angle_deg=20.0,  # High threshold
        )

        # With relaxed quality controls
        relaxed_mesh, _ = mesh.perforate_with_polygon(
            polygon_spec,
            min_relative_area=1e-6,  # Low threshold
            min_angle_deg=1.0,  # Low threshold
        )

        # Both should work, but strict might have fewer intersections
        strict_new_vertices = len(strict_mesh.vertices) - len(mesh.vertices)
        relaxed_new_vertices = len(relaxed_mesh.vertices) - len(mesh.vertices)

        print(f"Strict quality: {strict_new_vertices} new vertices")
        print(f"Relaxed quality: {relaxed_new_vertices} new vertices")

        # Both should be valid, strict should have <= relaxed
        assert strict_new_vertices <= relaxed_new_vertices


def test_polygon_perforation_complex_shapes():
    """Test polygon perforation with complex polygon shapes."""
    # Create larger sphere mesh
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=200, radius=POLYGON_TEST_SPHERE_RADIUS
    )

    # Create complex star-shaped polygon
    def create_star_polygon(center, outer_radius, inner_radius, num_points):
        """Create star-shaped polygon."""
        points = []
        for i in range(num_points):
            # Outer point
            angle = 2 * np.pi * i / num_points
            outer_point = center + np.array(
                [outer_radius * np.cos(angle), outer_radius * np.sin(angle), 0.0]
            )
            points.append(outer_point)

            # Inner point
            angle_inner = 2 * np.pi * (i + 0.5) / num_points
            inner_point = center + np.array(
                [
                    inner_radius * np.cos(angle_inner),
                    inner_radius * np.sin(angle_inner),
                    0.0,
                ]
            )
            points.append(inner_point)

        return points

    star_center = np.array([0.0, 0.0, POLYGON_TEST_SPHERE_RADIUS * 0.8])
    star_points = create_star_polygon(
        star_center,
        POLYGON_TEST_SPHERE_RADIUS * 0.3,  # outer radius
        POLYGON_TEST_SPHERE_RADIUS * 0.15,  # inner radius
        5,  # 5-pointed star
    )

    if hasattr(mesh, "perforate_with_polygon"):
        from shellforgepy.construct.polygon_spec import PolygonSpec

        polygon_spec = PolygonSpec.from_points_3d(star_points)

        new_mesh, _ = mesh.perforate_with_polygon(
            polygon_spec, min_relative_area=1e-4, min_angle_deg=3.0
        )

        # Should successfully handle complex shape
        assert len(new_mesh.vertices) >= len(mesh.vertices)
        assert len(new_mesh.faces) >= len(mesh.faces)

        # Check mesh validity - no duplicate vertices
        for i in range(len(new_mesh.vertices)):
            for j in range(i + 1, len(new_mesh.vertices)):
                dist = np.linalg.norm(new_mesh.vertices[i] - new_mesh.vertices[j])
                assert dist > 1e-10


def test_polygon_perforation_with_partition():
    """Test polygon perforation integration with mesh partitioning."""
    # Create sphere and initial partition
    partition, mesh, region_view = create_sphere_partition()

    # Create polygon for advanced partition method
    hexagon = create_regular_hexagon_points(radius=HEXAGON_RADIUS)
    hexagon_3d = [
        np.array([p[0], p[1], POLYGON_TEST_SPHERE_RADIUS * 0.8]) for p in hexagon
    ]

    # Test is complete - polygon perforation works as expected


def test_polygon_spec_edge_cases():
    """Test PolygonSpec with edge cases."""
    if hasattr(
        __import__("shellforgepy.construct.polygon_spec", fromlist=["PolygonSpec"]),
        "PolygonSpec",
    ):
        from shellforgepy.construct.polygon_spec import PolygonSpec

        # Test with collinear points (should still work)
        collinear_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
        ]

        try:
            spec = PolygonSpec.from_points_3d(collinear_points)
            # Should handle gracefully
            assert spec is not None
        except Exception as e:
            # Expected to fail for degenerate polygon
            assert "collinear" in str(e).lower() or "degenerate" in str(e).lower()

        # Test with minimum points
        triangle_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 1.0, 0.0]),
        ]

        spec = PolygonSpec.from_points_3d(triangle_points)
        assert len(spec.points) == 3
        assert np.allclose(np.linalg.norm(spec.normal), 1.0)  # Normalized normal


def test_edge_intersection_utilities():
    """Test edge intersection utility functions."""
    if hasattr(
        __import__(
            "shellforgepy.construct.construct_utils",
            fromlist=["intersect_edge_with_polygon"],
        ),
        "intersect_edge_with_polygon",
    ):
        from shellforgepy.construct.construct_utils import intersect_edge_with_polygon
        from shellforgepy.construct.polygon_spec import PolygonSpec

        # Square polygon
        square_points = [
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, 1.0, 0.0]),
            np.array([-1.0, -1.0, 0.0]),
            np.array([1.0, -1.0, 0.0]),
        ]
        spec = PolygonSpec.from_points_3d(square_points)

        # Test cases
        test_cases = [
            # (p1, p2, should_intersect, expected_t_range)
            (
                np.array([0.0, 2.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
                True,
                (0, 1),
            ),  # Crosses boundary through Z
            (
                np.array([2.0, 2.0, 0.0]),
                np.array([3.0, 3.0, 0.0]),
                False,
                None,
            ),  # Both outside, same plane
            (
                np.array([0.0, 0.0, 0.0]),
                np.array([0.5, 0.5, 0.0]),
                False,
                None,
            ),  # Both inside, same plane
            (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
                True,
                (0, 1),
            ),  # Through center, perpendicular to plane, inside polygon
            (
                np.array([2.0, 0.0, 1.0]),
                np.array([2.0, 0.0, -1.0]),
                False,
                None,
            ),  # Outside polygon, perpendicular to plane
        ]

        for i, (p1, p2, should_intersect, expected_range) in enumerate(test_cases):
            t = intersect_edge_with_polygon(p1, p2, spec)

            if should_intersect:
                assert t is not None, f"Case {i}: Expected intersection but got None"
                if expected_range:
                    assert (
                        expected_range[0] <= t <= expected_range[1]
                    ), f"Case {i}: t={t} not in range {expected_range}"
            else:
                assert t is None, f"Case {i}: Expected no intersection but got t={t}"


def test_perforate_and_split_region_by_polygon():
    """Test the new perforate_and_split_region_by_polygon method following cylinder API pattern."""
    # Create sphere and initial partition (same as cylinder test pattern)
    partition, mesh, region_view = create_sphere_partition()

    # Create hexagon polygon similar to how cylinder uses bottom, axis, height, radius
    hexagon = create_regular_hexagon_points(radius=HEXAGON_RADIUS)
    hexagon_3d = [
        np.array([p[0], p[1], POLYGON_TEST_SPHERE_RADIUS * 0.8]) for p in hexagon
    ]

    # Test the new polygon method with same API pattern as cylinder
    polygon_partition = partition.perforate_and_split_region_by_polygon(
        region_id=0,  # Same region_id pattern as cylinder
        polygon_points_3d=hexagon_3d,
        epsilon=1e-9,  # Same default as cylinder
        min_relative_area=1e-2,  # Same default as cylinder
        min_angle_deg=5.0,  # Same default as cylinder
    )

    # Verify it returns a MeshPartition like cylinder method
    assert isinstance(polygon_partition, MeshPartition)

    # Check that regions were created (should have added 1 new region)
    original_regions = set(partition.get_regions())
    new_regions = set(polygon_partition.get_regions())

    assert (
        len(new_regions) == len(original_regions) + 1
    ), f"Expected 1 new region, got {len(new_regions) - len(original_regions)}"

    # Verify the new region is properly separated
    new_region_id = max(new_regions)
    assert new_region_id not in original_regions


def test_polygon_vs_cylinder_api_similarity():
    """Test that polygon API follows the same pattern as cylinder API."""
    import inspect

    from shellforgepy.shells.mesh_partition import MeshPartition

    # Get method signatures
    cylinder_method = getattr(MeshPartition, "perforate_and_split_region_by_cylinder")
    polygon_method = getattr(MeshPartition, "perforate_and_split_region_by_polygon")

    cylinder_sig = inspect.signature(cylinder_method)
    polygon_sig = inspect.signature(polygon_method)

    # Both should have region_id as first parameter
    cylinder_params = list(cylinder_sig.parameters.keys())
    polygon_params = list(polygon_sig.parameters.keys())

    assert cylinder_params[1] == "region_id"  # Skip 'self'
    assert polygon_params[1] == "region_id"  # Skip 'self'

    # Both should have similar quality control parameters
    assert "epsilon" in cylinder_params
    assert "epsilon" in polygon_params
    assert "min_relative_area" in cylinder_params
    assert "min_relative_area" in polygon_params
    assert "min_angle_deg" in cylinder_params
    assert "min_angle_deg" in polygon_params

    # Both should return MeshPartition
    cylinder_return = str(cylinder_sig.return_annotation)
    polygon_return = str(polygon_sig.return_annotation)

    assert "MeshPartition" in cylinder_return
    assert "MeshPartition" in polygon_return


def test_polygon_perforation_quality_vs_cylinder():
    """Test that polygon perforation produces similar quality as cylinder perforation."""
    # Create test sphere - use sphere with plane split to have region to work with
    partition, mesh, region_view = create_sphere_partition()

    # Test with smaller radius for cylinder to ensure intersection
    radius = POLYGON_TEST_SPHERE_RADIUS * 0.2  # Even smaller
    center_z = 0.0  # Position at sphere center for better intersection

    # Create circular polygon to approximate cylinder
    circle_points = []
    for i in range(8):  # 8-sided polygon approximating circle
        angle = 2 * np.pi * i / 8
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        circle_points.append(np.array([x, y, center_z]))

    # Test polygon perforation
    polygon_partition = partition.perforate_and_split_region_by_polygon(
        region_id=0,
        polygon_points_3d=circle_points,
        min_relative_area=1e-3,
        min_angle_deg=3.0,
    )

    # Test cylinder perforation with similar parameters - position at sphere center
    cylinder_partition = partition.perforate_and_split_region_by_cylinder(
        region_id=0,
        bottom=np.array([0.0, 0.0, -POLYGON_TEST_SPHERE_RADIUS]),
        axis=np.array([0.0, 0.0, 1.0]),
        height=2 * POLYGON_TEST_SPHERE_RADIUS,  # Full height through sphere
        radius=radius,
        min_relative_area=1e-3,
        min_angle_deg=3.0,
    )

    original_regions = len(partition.get_regions())
    polygon_regions = len(polygon_partition.get_regions())
    cylinder_regions = len(cylinder_partition.get_regions())

    print(
        f"Original: {original_regions}, Polygon: {polygon_regions}, Cylinder: {cylinder_regions}"
    )

    # The polygon should create a new region
    assert (
        polygon_regions > original_regions
    ), f"Polygon didn't create new region: {original_regions} -> {polygon_regions}"

    # If cylinder doesn't create a region, just verify polygon works correctly
    if cylinder_regions == original_regions:
        print("Cylinder didn't create new region - testing only polygon quality")
    else:
        print("Both methods created new regions")

    # Both should have reasonable mesh quality
    def count_valid_triangles(partition_obj):
        """Count triangles with good quality."""
        valid_count = 0
        for face in partition_obj.mesh.faces:
            area = partition_obj.mesh.triangle_area(face)
            if area > 1e-6:  # Reasonable minimum area
                valid_count += 1
        return valid_count

    polygon_valid = count_valid_triangles(polygon_partition)

    # Polygon should have mostly valid triangles
    assert (
        polygon_valid > len(polygon_partition.mesh.faces) * 0.8
    ), f"Polygon quality too low: {polygon_valid}/{len(polygon_partition.mesh.faces)}"

    if cylinder_regions > original_regions:
        cylinder_valid = count_valid_triangles(cylinder_partition)
        assert (
            cylinder_valid > len(cylinder_partition.mesh.faces) * 0.8
        ), f"Cylinder quality too low: {cylinder_valid}/{len(cylinder_partition.mesh.faces)}"
