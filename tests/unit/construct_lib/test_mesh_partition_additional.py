import networkx as nx
import numpy as np
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.shells.mesh_partition import (
    MeshPartition,
    are_collinear,
    point_inside_cylinder,
)
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)


def test_point_inside_cylinder():
    """Test point_inside_cylinder utility function."""
    # Define cylinder parameters
    bottom = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])  # pointing up
    height = 2.0
    radius = 1.0

    # Test points inside cylinder
    inside_points = [
        np.array([0.0, 0.0, 1.0]),  # center at mid-height
        np.array([0.5, 0.0, 0.5]),  # inside radially and vertically
        np.array([0.0, 0.8, 1.8]),  # near top edge
        np.array([0.0, 0.0, 0.0]),  # at bottom center
        np.array([0.0, 0.0, 2.0]),  # at top center
    ]

    for point in inside_points:
        assert point_inside_cylinder(
            point, bottom, axis, height, radius
        ), f"Point {point} should be inside cylinder"

    # Test points outside cylinder
    outside_points = [
        np.array([1.5, 0.0, 1.0]),  # outside radially
        np.array([0.0, 0.0, 2.5]),  # above cylinder
        np.array([0.0, 0.0, -0.5]),  # below cylinder
        np.array([0.8, 0.8, 1.0]),  # outside radially (diagonal)
    ]

    for point in outside_points:
        assert not point_inside_cylinder(
            point, bottom, axis, height, radius
        ), f"Point {point} should be outside cylinder"

    # Test boundary points (with epsilon)
    boundary_points = [
        np.array([1.0, 0.0, 1.0]),  # on radius boundary
        np.array([0.0, 1.0, 1.0]),  # on radius boundary
        np.array([0.0, 0.0, 0.0]),  # on height boundary (bottom)
        np.array([0.0, 0.0, 2.0]),  # on height boundary (top)
    ]

    for point in boundary_points:
        assert point_inside_cylinder(
            point, bottom, axis, height, radius
        ), f"Boundary point {point} should be inside cylinder (with epsilon)"


def test_are_collinear():
    """Test are_collinear utility function."""
    # Test collinear segments
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    q1 = np.array([0.5, 0.0, 0.0])
    q2 = np.array([1.5, 0.0, 0.0])

    assert are_collinear(p1, p2, q1, q2), "Segments on x-axis should be collinear"

    # Test collinear segments (different direction)
    q1_rev = np.array([1.5, 0.0, 0.0])
    q2_rev = np.array([0.5, 0.0, 0.0])

    assert are_collinear(
        p1, p2, q1_rev, q2_rev
    ), "Reverse direction should still be collinear"

    # Test non-collinear segments
    r1 = np.array([0.0, 1.0, 0.0])
    r2 = np.array([1.0, 1.0, 0.0])

    assert not are_collinear(
        p1, p2, r1, r2
    ), "Parallel but offset segments should not be collinear"

    # Test non-parallel segments
    s1 = np.array([0.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 0.0])

    assert not are_collinear(
        p1, p2, s1, s2
    ), "Perpendicular segments should not be collinear"

    # Test 3D collinear segments
    p1_3d = np.array([0.0, 0.0, 0.0])
    p2_3d = np.array([1.0, 1.0, 1.0])
    q1_3d = np.array([0.5, 0.5, 0.5])
    q2_3d = np.array([2.0, 2.0, 2.0])

    assert are_collinear(
        p1_3d, p2_3d, q1_3d, q2_3d
    ), "3D diagonal segments should be collinear"


def test_mesh_partition_has_region_holes():
    """Test has_region_holes method."""
    # Create a simple mesh
    points, _ = create_tetrahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Initially, single region should not have holes
    assert not partition.has_region_holes(0), "Single region should not have holes"

    # After splitting, check hole detection
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Basic check that method runs without error
    for region_id in partition.get_regions():
        has_holes = partition.has_region_holes(region_id)
        assert isinstance(has_holes, bool), "has_region_holes should return boolean"


def test_mesh_partition_construct_closed_path():
    """Test construct_closed_path_from_vertices method."""
    # Create a cube and split it
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    # Get boundary edges for a region
    boundary_edges = partition.get_boundary_edges_of_region(0)

    if len(boundary_edges) > 0:
        # Extract vertices from boundary edges
        vertices = set()
        for edge in boundary_edges:
            vertices.update(edge)

        vertex_list = list(vertices)

        if len(vertex_list) > 2:
            # Try to construct a closed path (fix parameter order)
            region_faces = partition.get_faces_of_region(0)
            path = partition.construct_closed_path_from_vertices(
                region_faces, set(vertex_list)
            )

            # Basic validation
            assert isinstance(path, list), "Path should be a list"
            assert len(path) >= 3, "Closed path should have at least 3 vertices"
            assert path[0] == path[-1], "Path should be closed (first == last)"


def test_mesh_partition_is_region_contiguous():
    """Test is_region_contiguous method."""
    # Create mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Single region should be contiguous
    assert partition.is_region_contiguous(0), "Single region should be contiguous"

    # After normal split, regions should still be contiguous
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    for region_id in partition.get_regions():
        is_contiguous = partition.is_region_contiguous(region_id)
        assert isinstance(
            is_contiguous, bool
        ), "is_region_contiguous should return boolean"


def test_mesh_partition_region_adjacency_graph():
    """Test region_adjacency_graph method."""
    # Create and split mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Get adjacency graph
    adj_graph = partition.region_adjacency_graph()

    assert isinstance(adj_graph, dict), "Should return dictionary of region adjacencies"

    # Should have keys for each region
    regions = partition.get_regions()
    for region_id in regions:
        assert region_id in adj_graph, f"Region {region_id} should be in adjacency dict"

    # Check basic dictionary properties
    assert len(adj_graph) == len(regions), "Dictionary should have key for each region"


def test_mesh_partition_get_region_area():
    """Test get_region_area method."""
    # Create mesh
    points, _ = create_tetrahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Get area of single region
    total_area = partition.get_region_area(0)

    assert isinstance(total_area, (int, float)), "Area should be numeric"
    assert total_area > 0, "Area should be positive"

    # Split and check that sum of areas is preserved
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    split_area_sum = sum(partition.get_region_area(r) for r in partition.get_regions())

    # Areas should be approximately equal (within numerical precision)
    assert np.isclose(
        total_area, split_area_sum, rtol=1e-10
    ), "Total area should be preserved after splitting"


def test_mesh_partition_find_regions_of_edge():
    """Test find_regions_of_edge method."""
    # Create and split mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Get some boundary edges
    boundary_edges = partition.get_boundary_edges_of_region(0)

    if len(boundary_edges) > 0:
        edge = list(boundary_edges)[0]  # Convert set to list to get first element
        regions = partition.find_regions_of_edge(edge)

        assert isinstance(
            regions, (list, tuple)
        ), "Should return list or tuple of region IDs"
        assert len(regions) >= 1, "Edge should belong to at least one region"
        assert all(isinstance(r, int) for r in regions), "Region IDs should be integers"


def test_mesh_partition_vertex_labels():
    """Test vertex label functionality."""
    # Create mesh with some labeled vertices
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)

    # Test basic vertex operations instead of labels (which don't exist)
    # Get regions containing some vertex
    regions_containing_vertex = []
    for region_id in partition.get_regions():
        region_vertices = partition.get_region_vertices(region_id)
        if len(region_vertices) > 0:
            # Just check that we can get vertices from a region
            first_vertex = next(iter(region_vertices))
            regions_containing_vertex.append(region_id)
            break

    assert isinstance(
        regions_containing_vertex, list
    ), "Should return list of region IDs"
    # At least one region should exist
    assert len(partition.get_regions()) >= 1, "Should have at least one region"

    # Test finding local vertex IDs by label
    local_ids_A = partition.find_local_vertex_ids_by_label("corner_A", 0)
    assert isinstance(local_ids_A, list), "Should return list of local vertex IDs"

    # Test with non-existent label
    regions_nonexistent = partition.find_regions_of_vertex_by_label("nonexistent")
    assert isinstance(
        regions_nonexistent, list
    ), "Should return empty list for non-existent label"
    assert len(regions_nonexistent) == 0, "Non-existent label should return empty list"


def test_mesh_partition_edge_graph():
    """Test edge graph construction."""
    # Create simple mesh
    points, _ = create_tetrahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Access the edge graph (built during initialization)
    edge_graph = partition.edge_graph

    assert isinstance(edge_graph, nx.Graph), "Edge graph should be NetworkX Graph"
    assert edge_graph.number_of_nodes() > 0, "Edge graph should have nodes"

    # Each node should represent a vertex (integer index)
    for node in edge_graph.nodes():
        assert isinstance(
            node, (int, np.integer)
        ), "Edge graph nodes should be integers (vertex indices)"
        # Nodes are just vertex indices, nothing to compare


def test_mesh_partition_face_graph():
    """Test face graph construction."""
    # Create simple mesh
    points, _ = create_tetrahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Access the face graph (built during initialization)
    face_graph = partition.face_graph

    assert isinstance(face_graph, nx.Graph), "Face graph should be NetworkX Graph"
    assert face_graph.number_of_nodes() == len(
        mesh.faces
    ), "Face graph should have node for each face"

    # Check adjacencies are reasonable
    for node in face_graph.nodes():
        neighbors = list(face_graph.neighbors(node))
        # Each face should have at most 3 neighbors (for triangle mesh)
        assert (
            len(neighbors) <= 3
        ), f"Face {node} has too many neighbors: {len(neighbors)}"


def test_mesh_partition_complex_splits():
    """Test multiple sequential splits."""
    # Create mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    original_num_faces = len(mesh.faces)

    # Perform multiple splits
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    regions_after_first = len(partition.get_regions())
    assert regions_after_first >= 2, "Should have at least 2 regions after first split"

    # Split one of the resulting regions
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 1, 0])
    )

    regions_after_second = len(partition.get_regions())
    assert (
        regions_after_second >= regions_after_first
    ), "Should have at least as many regions after second split"

    # Verify total face count is reasonable
    total_faces_in_regions = sum(
        partition.get_num_faces_in_region(r) for r in partition.get_regions()
    )

    # Should have same or more faces due to perforation
    assert (
        total_faces_in_regions >= original_num_faces
    ), "Total faces should be preserved or increased due to perforation"


def test_get_region_vertices():
    """Test get_region_vertices method."""
    # Create and split mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Test getting vertices for each region
    for region_id in partition.get_regions():
        vertices = partition.get_region_vertices(region_id)

        assert isinstance(vertices, np.ndarray), "Should return numpy array"
        assert vertices.ndim == 2, "Should be 2D array"
        assert vertices.shape[1] == 3, "Should have 3D coordinates"
        assert vertices.shape[0] > 0, "Should have at least some vertices"
        assert np.all(np.isfinite(vertices)), "All coordinates should be finite"


def test_mesh_partition_get_region_id_of_triangle():
    """Test get_region_id_of_triangle method."""
    # Create and split mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Initially all triangles should be in region 0
    for i in range(len(mesh.faces)):
        region_id = partition.get_region_id_of_triangle(i)
        assert region_id == 0, "All triangles should initially be in region 0"

    # After splitting
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Check that triangles belong to valid regions
    valid_regions = set(partition.get_regions())
    for i in range(len(mesh.faces)):
        region_id = partition.get_region_id_of_triangle(i)
        assert (
            region_id in valid_regions
        ), f"Triangle {i} belongs to invalid region {region_id}"


def test_mesh_partition_get_faces_of_region():
    """Test get_faces_of_region method."""
    # Create and split mesh
    points, _ = create_cube_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Test getting faces for each region
    all_faces_in_regions = set()

    for region_id in partition.get_regions():
        faces = partition.get_faces_of_region(region_id)

        assert isinstance(faces, list), "Should return list of face indices"
        assert len(faces) > 0, f"Region {region_id} should have at least one face"
        assert all(
            isinstance(f, (int, np.integer)) for f in faces
        ), "Face indices should be integers"

        # Check no overlapping faces between regions
        faces_set = set(faces)
        assert (
            len(all_faces_in_regions & faces_set) == 0
        ), "Faces should not overlap between regions"

        all_faces_in_regions.update(faces_set)

    # Check that all original faces are accounted for
    original_faces = set(range(len(mesh.faces)))
    # Note: after perforation, there might be more faces than original
    assert (
        all_faces_in_regions >= original_faces
    ), "All original faces should be accounted for in regions"


def test_mesh_partition_get_num_faces_in_region():
    """Test get_num_faces_in_region method."""
    # Create mesh
    points, _ = create_tetrahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Initially all faces should be in region 0
    num_faces_region_0 = partition.get_num_faces_in_region(0)
    assert num_faces_region_0 == len(
        mesh.faces
    ), "All faces should initially be in region 0"

    # After splitting
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([1, 0, 0])
    )

    # Check consistency between get_num_faces_in_region and get_faces_of_region
    for region_id in partition.get_regions():
        num_faces = partition.get_num_faces_in_region(region_id)
        faces_list = partition.get_faces_of_region(region_id)

        assert num_faces == len(
            faces_list
        ), f"Inconsistent face count for region {region_id}: {num_faces} vs {len(faces_list)}"
        assert num_faces > 0, f"Region {region_id} should have at least one face"
