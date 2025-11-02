"""Tests for polygon perforation vertex filter."""

import logging

import numpy as np
from shellforgepy.construct.construct_utils import fibonacci_sphere
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.polygon_perforation import create_edge_filter

_logger = logging.getLogger(__name__)

sphere_radius = 100
sphere_resolution = 12


def create_sphere_mesh():
    sphere_points = np.array(fibonacci_sphere(samples=sphere_resolution))
    sphere_points *= sphere_radius
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(sphere_points)
    return mesh


def test_peforation_with_edge_fiter():
    """Test basic vertex filter functionality."""

    _logger.info("Creating sphere mesh")
    # Create sphere mesh
    mesh = create_sphere_mesh()

    cut_point = (0, 0, 0)
    cut_normal = (0, 0, 1)
    perforation_result_no_filter = mesh.compute_plane_perforation(cut_point, cut_normal)

    _logger.info(f"New vertices without filter:")
    for new_v in perforation_result_no_filter.new_vertices:

        vertex_text = " ".join(f"{coord:.1f}" for coord in new_v)

        _logger.info(f"  {vertex_text}")

    edge_filter = create_edge_filter(
        p1_local=np.array([-70, 0.0, 0.0]),
        p2_local=np.array([70.0, 0.0, 0.0]),
        normal_local=np.array([0.0, 1.0, 0]),
    )

    perforation_result_with_filter = mesh.compute_plane_perforation(
        cut_point, cut_normal, vertex_filter=edge_filter
    )

    # Check that the perforation with filter created fewer new vertices
    num_new_vertices_no_filter = len(perforation_result_no_filter.new_vertices)
    num_new_vertices_with_filter = len(perforation_result_with_filter.new_vertices)
    _logger.info(f"New vertices without filter: {num_new_vertices_no_filter}")

    _logger.info(f"New vertices with filter: {num_new_vertices_with_filter}")

    assert num_new_vertices_with_filter < num_new_vertices_no_filter
    assert num_new_vertices_with_filter > 0
