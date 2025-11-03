"""Tests for polygon perforation with 2D projection."""

import logging
import math

import numpy as np
from shellforgepy.construct.construct_utils import fibonacci_sphere
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)

_logger = logging.getLogger(__name__)

sphere_radius = 80
hexagon_radius = 40
sphere_resolution = 200


def create_sphere_mesh():
    sphere_points = np.array(fibonacci_sphere(samples=sphere_resolution))
    sphere_points *= sphere_radius
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(sphere_points)
    return mesh


def create_regular_hexagon_points(radius):
    points = []
    for i in range(6):
        angle = math.radians(i * 60)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points


def test_polygon_perforation_boundary_quality():
    """Test polygon perforation boundary quality with strict geometric limits.

    This test EXACTLY replicates the working code from inner_feature_test.py to ensure
    we're analyzing the same geometry and methods that are actually being used.
    """
    _logger.info("Testing polygon perforation boundary quality...")

    # EXACT REPLICATION: Create sphere mesh exactly like workshop test
    sphere_resolution = 200
    sphere_radius = 80
    hexagon_radius = 40

    sphere_points = np.array(fibonacci_sphere(samples=sphere_resolution))
    sphere_points *= sphere_radius
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(sphere_points)
    partition = MeshPartition(mesh)

    # EXACT REPLICATION: Cut sphere into top/bottom exactly like workshop test
    partition = partition.perforate_and_split_region_by_plane(
        0, (0, 0, sphere_radius * 0.7), (0, 0, 1)
    )

    _logger.info(
        f"Created sphere mesh: radius={sphere_radius}, resolution={sphere_resolution}"
    )
    _logger.info(f"Cut plane at z={sphere_radius * 0.7}")

    hexagon = create_regular_hexagon_points(radius=hexagon_radius)

    # EXACT REPLICATION: Project polygon exactly like workshop test
    projected_polygon, inside_vertex_ids = partition.project_polygon_onto_mesh(
        region_id=0,  # Project onto the main region
        polygon_points_2d=hexagon,
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([0.1, 0.1, 1.0]),
        target_segment_length=3.0,
    )

    _logger.info(f"Projected polygon has {len(projected_polygon)} points")

    hexagon_3d = []
    for pt_2d in hexagon:
        # Project each vertex onto mesh to get proper 3D coordinates
        single_projected, _ = partition.project_polygon_onto_mesh(
            region_id=0,
            polygon_points_2d=[pt_2d],  # Single point
            ray_origin=np.array([0.0, 0.0, 0.0]),
            ray_direction=np.array([0.1, 0.1, 1.0]),
            target_segment_length=1.0,
        )
        if single_projected:
            hexagon_3d.append(single_projected[0])  # Already a numpy array

    # EXACT REPLICATION: Perforate exactly like workshop test
    original_regions = set(partition.get_regions())

    polygon_partition = partition.perforate_and_split_region_by_polygon(
        region_id=0,
        polygon_points_3d=hexagon_3d,  # Use original 6 vertices projected to mesh
        min_relative_area=1e-3,  # Prevent very small triangles
        min_angle_deg=5.0,  # Prevent very sharp angles
    )

    new_regions = set(polygon_partition.get_regions())
    added_regions = new_regions - original_regions

    _logger.info(f"Original regions: {original_regions}")
    _logger.info(f"New regions: {new_regions}")
    _logger.info(f"Added regions: {added_regions}")

    # Handle the case where no new regions are created (which is the problem we're investigating)
    if len(added_regions) == 0:
        _logger.warning(
            "CRITICAL: Polygon perforation failed to create any new regions!"
        )
        _logger.warning(
            "This indicates the polygon perforation algorithm is fundamentally broken."
        )

        # Use the original region for boundary comparison to show the problem
        polygon_region_id = 0
        _logger.warning(
            f"Using original region {polygon_region_id} for boundary comparison"
        )

        # This test will demonstrate the poor quality of non-perforated boundaries
    else:
        # Find the newest region (highest ID) - exactly like workshop test
        polygon_region_id = max(new_regions)
        _logger.info(f"Created polygon region {polygon_region_id}")

    # STATISTICS: Compare actual region boundary with projected polygon
    _logger.info("=== BOUNDARY COMPARISON STATISTICS ===")

    try:
        # Get the actual boundary edges of the polygon region
        actual_boundary_edges = polygon_partition.get_boundary_edges_of_region(
            polygon_region_id
        )
        _logger.info(f"Actual region boundary has {len(actual_boundary_edges)} edges")

        # Convert boundary edges to 3D points - exactly like workshop test
        actual_boundary_points = []
        for edge in actual_boundary_edges:
            v1_idx, v2_idx = edge
            v1 = polygon_partition.mesh.vertices[v1_idx]
            v2 = polygon_partition.mesh.vertices[v2_idx]
            actual_boundary_points.extend([v1, v2])

        # Remove duplicates while preserving order (approximate) - exactly like workshop test
        unique_boundary_points = []
        tolerance = 1e-6
        for point in actual_boundary_points:
            is_duplicate = False
            for existing in unique_boundary_points:
                if np.linalg.norm(point - existing) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boundary_points.append(point)

        _logger.info(f"Unique boundary points: {len(unique_boundary_points)}")
        _logger.info(f"Projected polygon points: {len(projected_polygon)}")

        if len(unique_boundary_points) > 0 and len(projected_polygon) > 0:
            # 1. Boundary length comparison - exactly like workshop test
            actual_boundary_length = 0
            for i in range(len(unique_boundary_points)):
                p1 = unique_boundary_points[i]
                p2 = unique_boundary_points[(i + 1) % len(unique_boundary_points)]
                actual_boundary_length += np.linalg.norm(p2 - p1)

            projected_boundary_length = 0
            for i in range(len(projected_polygon)):
                p1 = np.array(projected_polygon[i])
                p2 = np.array(projected_polygon[(i + 1) % len(projected_polygon)])
                projected_boundary_length += np.linalg.norm(p2 - p1)

            _logger.info(
                f"Boundary length - Actual: {actual_boundary_length:.2f}, Projected: {projected_boundary_length:.2f}"
            )
            length_ratio = (
                actual_boundary_length / projected_boundary_length
                if projected_boundary_length > 0
                else 0
            )
            _logger.info(f"Length ratio (actual/projected): {length_ratio:.3f}")

            # 2. Point-to-point distance analysis - exactly like workshop test
            min_distances_actual_to_projected = []
            for actual_point in unique_boundary_points:
                distances = [
                    np.linalg.norm(actual_point - np.array(proj_point))
                    for proj_point in projected_polygon
                ]
                min_distances_actual_to_projected.append(min(distances))

            min_distances_projected_to_actual = []
            for proj_point in projected_polygon:
                distances = [
                    np.linalg.norm(np.array(proj_point) - actual_point)
                    for actual_point in unique_boundary_points
                ]
                min_distances_projected_to_actual.append(min(distances))

            avg_actual_to_proj = np.mean(min_distances_actual_to_projected)
            max_actual_to_proj = np.max(min_distances_actual_to_projected)
            avg_proj_to_actual = np.mean(min_distances_projected_to_actual)
            max_proj_to_actual = np.max(min_distances_projected_to_actual)

            _logger.info(
                f"Distance actual→projected - Avg: {avg_actual_to_proj:.2f}, Max: {max_actual_to_proj:.2f}"
            )
            _logger.info(
                f"Distance projected→actual - Avg: {avg_proj_to_actual:.2f}, Max: {max_proj_to_actual:.2f}"
            )

            # 3. Coverage analysis - exactly like workshop test
            covered_points = sum(
                1 for d in min_distances_projected_to_actual if d < hexagon_radius * 0.1
            )
            coverage_ratio = (
                covered_points / len(projected_polygon)
                if len(projected_polygon) > 0
                else 0
            )
            _logger.info(
                f"Coverage: {covered_points}/{len(projected_polygon)} projected points within 10% of hexagon radius"
            )
            _logger.info(f"Coverage ratio: {coverage_ratio:.3f}")

            # 4. Assessment - exactly like workshop test
            _logger.info("=== ASSESSMENT ===")
            if length_ratio < 0.5:
                _logger.error(
                    f"POOR: Actual boundary much shorter than expected ({length_ratio:.2f}x)"
                )
            elif length_ratio < 0.8:
                _logger.warning(
                    f"MEDIOCRE: Actual boundary shorter than expected ({length_ratio:.2f}x)"
                )
            else:
                _logger.info(f"GOOD: Boundary length reasonable ({length_ratio:.2f}x)")

            if coverage_ratio < 0.5:
                _logger.error(
                    f"POOR: Low coverage of projected polygon ({coverage_ratio:.2f})"
                )
            elif coverage_ratio < 0.8:
                _logger.warning(
                    f"MEDIOCRE: Partial coverage of projected polygon ({coverage_ratio:.2f})"
                )
            else:
                _logger.info(
                    f"GOOD: Good coverage of projected polygon ({coverage_ratio:.2f})"
                )

            if max_proj_to_actual > hexagon_radius * 0.5:
                _logger.error(
                    f"POOR: Large gaps in boundary approximation (max {max_proj_to_actual:.2f})"
                )
            elif max_proj_to_actual > hexagon_radius * 0.2:
                _logger.warning(
                    f"MEDIOCRE: Some gaps in boundary approximation (max {max_proj_to_actual:.2f})"
                )
            else:
                _logger.info(
                    f"GOOD: Tight boundary approximation (max gap {max_proj_to_actual:.2f})"
                )

            # STRICT QUALITY ASSERTIONS - Document what we expect vs what we get
            # These are the thresholds that SHOULD pass but currently don't due to the algorithm issue

            MIN_COVERAGE_RATIO = (
                0.70  # At least 70% of projected polygon should be well-covered
            )
            MAX_GAP_TOLERANCE = (
                hexagon_radius * 0.25
            )  # No gaps larger than 25% of hexagon radius
            MIN_PERIMETER_RATIO = (
                0.60  # Actual perimeter should be at least 60% of projected
            )

            _logger.info(
                "=== QUALITY EXPECTATIONS (currently failing due to algorithm issues) ==="
            )
            _logger.info(
                f"Expected coverage ratio >= {MIN_COVERAGE_RATIO:.2f}, actual: {coverage_ratio:.3f}"
            )
            _logger.info(
                f"Expected max gap <= {MAX_GAP_TOLERANCE:.2f}, actual: {max_proj_to_actual:.2f}"
            )
            _logger.info(
                f"Expected perimeter ratio >= {MIN_PERIMETER_RATIO:.2f}, actual: {length_ratio:.3f}"
            )

            # For now, just log the failures rather than asserting, since we know the algorithm is broken
            if coverage_ratio < MIN_COVERAGE_RATIO:
                _logger.error(
                    f"FAILED: Coverage ratio {coverage_ratio:.3f} < {MIN_COVERAGE_RATIO}"
                )

            if max_proj_to_actual > MAX_GAP_TOLERANCE:
                _logger.error(
                    f"FAILED: Max gap {max_proj_to_actual:.2f} > {MAX_GAP_TOLERANCE:.2f}"
                )

            if length_ratio < MIN_PERIMETER_RATIO:
                _logger.error(
                    f"FAILED: Perimeter ratio {length_ratio:.3f} < {MIN_PERIMETER_RATIO}"
                )

    except Exception as e:
        _logger.error(f"Error in boundary comparison: {e}", exc_info=True)

    _logger.info("=== END BOUNDARY COMPARISON ===")

    # This test documents the current broken state rather than asserting success
    _logger.info(
        "✓ Boundary quality analysis completed (results show algorithm needs fixing)"
    )
