import numpy as np
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)


def test_enhanced_polygon_perforation_mesh_refinement():
    """Test that the enhanced polygon perforation creates more vertices for better boundary following."""
    # Create test sphere mesh
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10.0
    )
    partition = MeshPartition(mesh)

    # Create a simple square polygon
    size = 3.0
    z_level = 8.0
    polygon_points = [
        np.array([size, size, z_level]),
        np.array([-size, size, z_level]),
        np.array([-size, -size, z_level]),
        np.array([size, -size, z_level]),
    ]

    # Record original mesh size
    original_face_count = len(mesh.faces)
    original_vertex_count = len(mesh.vertices)

    # Test enhanced polygon perforation
    result_partition = partition.perforate_and_split_region_by_polygon(
        region_id=0,
        polygon_points_3d=polygon_points,
        min_relative_area=1e-3,
        min_angle_deg=3.0,
    )

    # Check that mesh was enhanced (more faces and vertices)
    new_face_count = len(result_partition.mesh.faces)
    new_vertex_count = len(result_partition.mesh.vertices)

    print(
        f"Original mesh: {original_face_count} faces, {original_vertex_count} vertices"
    )
    print(f"Enhanced mesh: {new_face_count} faces, {new_vertex_count} vertices")
    print(
        f"Face increase: {(new_face_count - original_face_count)/original_face_count:.1%}"
    )
    print(
        f"Vertex increase: {(new_vertex_count - original_vertex_count)/original_vertex_count:.1%}"
    )

    # Assert that enhancement created more geometry for better boundary following
    assert (
        new_face_count > original_face_count
    ), "Enhanced perforation should create more faces"
    assert (
        new_vertex_count > original_vertex_count
    ), "Enhanced perforation should create more vertices"

    # Assert reasonable enhancement (not excessive)
    face_increase_ratio = new_face_count / original_face_count
    assert (
        1.1 <= face_increase_ratio <= 3.0
    ), f"Face increase ratio should be reasonable, got {face_increase_ratio:.2f}"

    # Check that new region was created
    regions = result_partition.get_regions()
    assert len(regions) > len(partition.get_regions()), "Should create new region"

    # Check new region has faces
    new_region_id = max(regions)
    region_faces = result_partition.get_faces_of_region(new_region_id)
    assert len(region_faces) > 0, "New region should have faces"

    print(
        f"✅ Enhanced polygon perforation successfully created {len(region_faces)} faces in new region"
    )
    return True
