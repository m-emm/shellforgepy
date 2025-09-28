import numpy as np
from shellforgepy.geometry.face_point_cloud import sphere_radius
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.geometry.spherical_tools import (
    cartesian_to_spherical_jackson,
    spherical_to_cartesian_jackson,
)
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.transformed_region_view import TransformedRegionView


def test_transformed_shell_map():
    # Step 1: Create geometry (a cube approximated as a sphere)
    points, _ = create_cube_geometry(sphere_radius)

    # Step 2: Create and partition the mesh
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Step 3: Perforate and split into two regions
    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    # Step 4: Get a region view and apply a transformation (e.g., small translation)
    region_view = TransformedRegionView(partition, 0).translated(1.0, 0.0, 0.0)

    # Step 5: Compute transformed shell maps
    shell_maps, vertex_index_map = region_view.get_transformed_materialized_shell_maps(
        shell_thickness=0.1
    )

    # Basic checks
    assert isinstance(shell_maps, dict)
    assert isinstance(vertex_index_map, dict)
    assert len(shell_maps) > 0

    for face_id, shell_map in shell_maps.items():
        assert "vertexes" in shell_map
        assert "faces" in shell_map
        verts = shell_map["vertexes"]
        faces = shell_map["faces"]
        assert isinstance(verts, dict)
        assert isinstance(faces, dict)
        for v in verts.values():
            assert v.shape == (3,)
            assert v[0] > 0.5  # confirm that translation x+1.0 took effect

    for face_id, vmap in vertex_index_map.items():
        assert "inner" in vmap and "outer" in vmap
        assert len(vmap["inner"]) == 3
        assert len(vmap["outer"]) == 3


def test_compute_connector_hints_on_transformed_region_view():
    # Step 1: Generate icosahedron geometry
    points, _ = create_cube_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    region_view = TransformedRegionView(partition, 0)

    num_faces = region_view.num_faces()
    print(f"Region {region_view.region_id} has {num_faces} faces")

    # Compute connector hints
    hints = region_view.compute_transformed_connector_hints(shell_thickness=0.02)

    # Basic checks
    assert isinstance(hints, list)
    assert all(h.region_a != h.region_b for h in hints)
    assert all(h.region_a < h.region_b for h in hints)  # canonicalization
    assert all(np.isclose(np.linalg.norm(h.edge_vector), 1.0) for h in hints)

    # Optional debug output
    for h in hints:
        print(
            f"Connector: {h.region_a} -> {h.region_b}, edge at {h.edge_centroid}, normal A {h.triangle_a_normal}, normal B {h.triangle_b_normal}"
        )


def test_compute_connector_hints_merge_tetrahedron():

    sphere_radius = 30
    shell_thickness = sphere_radius * 0.05

    shrink_border = 0.3

    # Step 1: Generate geometry
    points, _ = create_tetrahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, plane_point=np.array([0, 0, 0]), plane_normal=np.array([0, 1, 1])
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    region_views = []

    for region_id in partition.get_regions():
        view = TransformedRegionView(partition, region_id)

        if region_id == 0:
            view = view.rotated(np.deg2rad(180), axis=(1, 0, 0))

        else:
            view = view.rotated(np.deg2rad(0), axis=(1, 0, 0))
        region_views.append(view)

    # Step 5: Fuse solids per region
    parts = {}
    for region_view in region_views:
        region_id = region_view.region_id

        connector_hints = region_view.compute_transformed_connector_hints(
            shell_thickness, merge_connectors=False
        )

        edge_vectors_int = [
            tuple([int(q) for q in 1000 * np.round(h.edge_vector, 3)])
            for h in connector_hints
        ]
        unique_edge_vectors = set(edge_vectors_int)
        print(f"Unique edge vectors: {unique_edge_vectors}")

        assert len(unique_edge_vectors) == len(
            connector_hints
        ), f"Duplicate edge vectors found: {len(unique_edge_vectors)} unique vs {len(connector_hints)} total"

        print(f"connector_hints: \n{connector_hints}")

        connector_hints_merged = region_view.compute_transformed_connector_hints(
            shell_thickness, merge_connectors=True
        )

        assert len(connector_hints) == len(connector_hints_merged)


def test_lay_flat_optimal():
    # Step 1: Generate geometry
    points, _ = create_tetrahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, plane_point=np.array([0, 0, 0]), plane_normal=np.array([0, 1, 1])
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")

    region_views = []

    for region_id in partition.get_regions():
        view = TransformedRegionView(partition, region_id)
        region_views.append(view)

    # Step 5: Fuse solids per region
    tol = 1e-5  # tolerance for printability score
    for region_view in region_views:
        region_view = region_view.lay_flat_optimally_printable()

        assert region_view.printability_score() >= 0.5 - tol


def test_lay_flat_on_edge():
    print("Generating base icosahedron...")

    # Step 1: Generate icosahedron geometry
    points, _ = create_dodecahedron_geometry(sphere_radius)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )

    partition = partition.perforate_and_split_region_by_plane(
        region_id=1,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 1, 0]),
    )

    print(f"Partitioned into {partition.get_regions()}")
    for region_id in partition.get_regions():
        print(f"Region {region_id} Faces: {partition.get_faces_of_region(region_id)}")
    region_views = []

    for region_id in partition.get_regions():
        view = TransformedRegionView(partition, region_id)
        region_views.append(view)

    region_view = region_views[1]

    region_view = region_view.lay_flat_on_boundary_edges_for_printability()

    assert (
        region_view.printability_score() > 0.1
    ), "Printability score should be at least 0.1"


def test_numerical_instability():
    sphere_radius = 30
    shell_thickness = sphere_radius * 0.05
    shrink_border = 0

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=30, radius=sphere_radius
    )

    for i, v in enumerate(mesh.vertices):
        print(f"Vertex: {i}: {v}")

    new_vertices = []

    random_size = sphere_radius * 0.1
    new_vertices = []

    for v in mesh.vertices:
        r, theta, phi = cartesian_to_spherical_jackson(v)
        # r += np.random.uniform(-random_size, random_size)

        print(
            f"Vertex: Cartesian: {v} -> Spherical: (r={r:.2f}, theta={theta:.2f}, phi={phi:.2f})"
        )

        back = spherical_to_cartesian_jackson((r, theta, phi))
        # back = [ 0.0 if abs(coord) < 1e-6 else coord for coord in back ]

        new_vertices.append([back[0], back[1], back[2]])

    new_vertices = np.array(new_vertices)

    # for i, v in enumerate(mesh.vertices):
    #     assert np.allclose(v, new_vertices[i])

    assert len(new_vertices) == len(mesh.vertices)

    for i, v in enumerate(new_vertices):
        print(f"Vertex: {i}: {v}")

    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(new_vertices)

    for f in sorted([tuple(f) for f in mesh.faces]):
        print(f"Mesh face: {f}")

    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )

    shell_maps, _ = partition.mesh.calculate_materialized_shell_maps(
        shell_thickness=shell_thickness,
        shrinkage=0,
        shrink_border=shrink_border,
    )


def test_transformed_edge_features_along_original_edge():
    # Create cube mesh
    vertices, faces = create_cube_geometry()
    mesh = PartitionableSpheroidTriangleMesh(vertices=vertices, faces=faces)
    partition = MeshPartition(mesh)

    # Split with Z=0 plane (splits vertical edges)
    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )

    # Create view for region 0 (lower half of cube)
    view = TransformedRegionView(partition, region_id=0)

    # Vertical edge from bottom to top (will be split)
    v0 = 0  # (-1, -1, -1)
    v1 = 4  # (-1, -1,  1)

    features = view.find_transformed_edge_features_along_original_edge(v0, v1)

    assert isinstance(features, list)
    assert all(hasattr(f, "edge_coords") for f in features)
    assert all(len(f.edge_coords) == 2 for f in features)

    # Check each transformed point lies on the original edge line (after transform)
    v0_trans = view.transform_point(mesh.vertices[v0])
    v1_trans = view.transform_point(mesh.vertices[v1])
    edge_vec = v1_trans - v0_trans
    edge_len = np.linalg.norm(edge_vec)
    edge_dir = edge_vec / edge_len

    for feat in features:
        for pt in feat.edge_coords:
            proj_len = np.dot(pt - v0_trans, edge_dir)
            closest = v0_trans + proj_len * edge_dir
            dist = np.linalg.norm(pt - closest)
            assert dist < 1e-6, f"Point {pt} is not on the transformed edge"

    assert len(features) >= 1, "Expected at least one edge feature on the original edge"


def test_transformed_region_view_apply_transform():
    """Test apply_transform method."""
    # Create a simple mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Create region view
    view = TransformedRegionView(partition, 0)

    # Define a transformation matrix (translation + rotation)
    translation = np.array([1.0, 2.0, 3.0])
    angle = np.pi / 4  # 45 degrees
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation

    # Apply transform
    transformed_view = view.apply_transform(transform_matrix)

    # Check that transform was applied
    assert not np.array_equal(view.transform, transformed_view.transform)

    # Check that vertices are transformed correctly
    original_vertex = mesh.vertices[0]
    original_transformed = view.transform_point(original_vertex)
    new_transformed = transformed_view.transform_point(original_vertex)

    # Manually compute expected transformation
    homogeneous = np.append(original_vertex, 1.0)
    expected = (transform_matrix @ view.transform @ homogeneous)[:3]

    assert np.allclose(new_transformed, expected, atol=1e-10)


def test_transformed_region_view_rotated():
    """Test rotated method."""
    # Create mesh and partition
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test rotation around z-axis
    angle = np.pi / 3  # 60 degrees
    axis = np.array([0, 0, 1])
    center = np.array([0, 0, 0])

    rotated_view = view.rotated(angle, axis, center)

    # Check that rotation was applied
    assert not np.array_equal(view.transform, rotated_view.transform)

    # Test specific point transformation
    test_point = np.array([1.0, 0.0, 0.0])
    original_transformed = view.transform_point(test_point)
    rotated_transformed = rotated_view.transform_point(test_point)

    # The rotated point should be different
    assert not np.allclose(original_transformed, rotated_transformed)

    # Distance from center should be preserved
    original_dist = np.linalg.norm(original_transformed - center)
    rotated_dist = np.linalg.norm(rotated_transformed - center)
    assert np.isclose(original_dist, rotated_dist)


def test_transformed_region_view_translated():
    """Test translated method."""
    # Create mesh and partition
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test translation
    dx, dy, dz = 1.5, -2.0, 3.5
    translated_view = view.translated(dx, dy, dz)

    # Check that translation was applied
    assert not np.array_equal(view.transform, translated_view.transform)

    # Test point transformation
    test_point = np.array([0.0, 0.0, 0.0])
    original_transformed = view.transform_point(test_point)
    translated_transformed = translated_view.transform_point(test_point)

    # The difference should be the translation vector
    diff = translated_transformed - original_transformed
    expected_diff = np.array([dx, dy, dz])
    assert np.allclose(diff, expected_diff)

    # Test alternative calling patterns (provide explicit zeros)
    translated_view_2 = view.translated(dx, 0.0, 0.0)  # Only x
    translated_view_3 = view.translated(dx, dy, 0.0)  # x and y

    # Check that these work without error
    assert isinstance(translated_view_2, TransformedRegionView)
    assert isinstance(translated_view_3, TransformedRegionView)


def test_vertex_indices_closer_than():
    """Test vertex_indices_closer_than method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test with point at origin
    test_point = np.array([0.0, 0.0, 0.0])
    min_distance = 0.5

    close_indices = view.vertex_indices_closer_than(test_point, min_distance)

    assert isinstance(close_indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in close_indices)

    # Verify that returned vertices are actually close
    vertices, _, _ = view.get_transformed_vertices_faces_boundary_edges()

    for idx in close_indices:
        vertex = vertices[idx]
        distance = np.linalg.norm(vertex - test_point)
        assert (
            distance < min_distance
        ), f"Vertex {idx} at distance {distance} should be closer than {min_distance}"

    # Test with very small distance (should return fewer or no vertices)
    very_close_indices = view.vertex_indices_closer_than(test_point, 0.01)
    assert len(very_close_indices) <= len(close_indices)


def test_face_indices_of_vertex_index_set():
    """Test face_indices_of_vertex_index_set method."""
    # Create mesh and partition
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Get some vertex indices
    vertex_indices = {0, 1, 2}  # First few vertices

    face_indices = view.face_indices_of_vertex_index_set(vertex_indices)

    assert isinstance(face_indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in face_indices)

    # Verify that returned faces actually contain the specified vertices
    _, faces, _ = view.get_transformed_vertices_faces_boundary_edges()

    for face_idx in face_indices:
        face = faces[face_idx]
        face_vertices = set(face)
        # The face should contain at least one of the specified vertices
        assert (
            len(face_vertices & vertex_indices) > 0
        ), f"Face {face_idx} should contain at least one vertex from {vertex_indices}"


def test_ray_intersect_faces():
    """Test ray_intersect_faces method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Define ray pointing towards the mesh center
    ray_origin = np.array([0.0, 0.0, -2.0])
    ray_direction = np.array([0.0, 0.0, 1.0])  # pointing up

    intersections = view.ray_intersect_faces(ray_origin, ray_direction)

    assert isinstance(intersections, list)

    # Each intersection should be a tuple (face_index, intersection_point)
    for intersection in intersections:
        assert isinstance(intersection, tuple)
        assert len(intersection) == 2
        face_idx, point = intersection
        assert isinstance(face_idx, (int, np.integer))
        assert isinstance(point, np.ndarray)
        assert point.shape == (3,)
        assert np.all(np.isfinite(point))

        # The intersection point should be on the ray
        t = np.dot(point - ray_origin, ray_direction) / np.dot(
            ray_direction, ray_direction
        )
        closest_on_ray = ray_origin + t * ray_direction
        assert np.allclose(point, closest_on_ray, atol=1e-6)


def test_average_normal_at_vertex():
    """Test average_normal_at_vertex method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test normal computation for a vertex
    vertex_index = 0

    normal = view.average_normal_at_vertex(vertex_index)

    assert isinstance(normal, np.ndarray)
    assert normal.shape == (3,)
    assert np.all(np.isfinite(normal))

    # Normal should be approximately unit length
    normal_length = np.linalg.norm(normal)
    assert np.isclose(
        normal_length, 1.0, atol=1e-6
    ), f"Normal length {normal_length} should be close to 1"


def test_face_centroid():
    """Test face_centroid method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test centroid computation
    face_index = 0

    centroid = view.face_centroid(face_index)

    assert isinstance(centroid, np.ndarray)
    assert centroid.shape == (3,)
    assert np.all(np.isfinite(centroid))

    # Verify centroid is average of face vertices
    face_vertices = view.face_vertices(face_index)
    expected_centroid = np.mean(face_vertices, axis=0)

    assert np.allclose(centroid, expected_centroid, atol=1e-10)


def test_face_normal():
    """Test face_normal method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test normal computation
    face_index = 0

    normal = view.face_normal(face_index)

    assert isinstance(normal, np.ndarray)
    assert normal.shape == (3,)
    assert np.all(np.isfinite(normal))

    # Normal should be unit length
    normal_length = np.linalg.norm(normal)
    assert np.isclose(normal_length, 1.0, atol=1e-6)


def test_face_vertices():
    """Test face_vertices method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test vertex retrieval
    face_index = 0

    vertices = view.face_vertices(face_index)

    assert isinstance(vertices, np.ndarray)
    assert vertices.shape == (3, 3)  # 3 vertices, 3 coordinates each
    assert np.all(np.isfinite(vertices))

    # Vertices should be transformed versions of the original mesh vertices
    original_vertices, faces, _ = view.get_transformed_vertices_faces_boundary_edges()
    face = faces[face_index]

    expected_vertices = np.array([original_vertices[i] for i in face])
    assert np.allclose(vertices, expected_vertices, atol=1e-10)


def test_num_faces():
    """Test num_faces method."""
    # Create mesh and split it
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    partition = partition.perforate_and_split_region_by_plane(
        0, np.array([0, 0, 0]), np.array([0, 0, 1])
    )

    # Test for each region
    for region_id in partition.get_regions():
        view = TransformedRegionView(partition, region_id)

        num_faces = view.num_faces()

        assert isinstance(num_faces, int)
        assert num_faces > 0, f"Region {region_id} should have at least one face"

        # Verify consistency with actual face data
        _, faces, _ = view.get_transformed_vertices_faces_boundary_edges()
        assert num_faces == len(
            faces
        ), f"num_faces() returned {num_faces} but actual faces are {len(faces)}"


def test_lay_flat():
    """Test lay_flat method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Apply lay_flat transformation
    flat_view = view.lay_flat()

    assert isinstance(flat_view, TransformedRegionView)
    assert flat_view.region_id == view.region_id
    assert flat_view.partition is view.partition

    # Transform should be different
    assert not np.array_equal(view.transform, flat_view.transform)

    # Test with custom definition of low
    flat_view_custom = view.lay_flat(definition_of_low=2.0)
    assert isinstance(flat_view_custom, TransformedRegionView)


def test_lay_flat_on_face():
    """Test lay_flat_on_face method."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Get a face to lay flat on
    face_index = 0

    flat_view = view.lay_flat_on_face(face_index)

    assert isinstance(flat_view, TransformedRegionView)
    assert flat_view.region_id == view.region_id
    assert flat_view.partition is view.partition

    # Transform should be different
    assert not np.array_equal(view.transform, flat_view.transform)

    # The specified face should now be approximately flat (parallel to xy-plane)
    face_normal = flat_view.face_normal(face_index)
    z_axis = np.array([0, 0, 1])

    # Normal should be close to pointing up or down
    dot_product = abs(np.dot(face_normal, z_axis))
    assert dot_product > 0.9, f"Face normal {face_normal} should be close to z-axis"


def test_find_local_vertex_ids_by_label():
    """Test find_local_vertex_ids_by_label method."""
    # Create mesh with labeled vertices
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    # Add vertex labels
    mesh.vertex_labels = {0: "corner_A", 1: "corner_B", 2: "corner_C"}

    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test finding labeled vertices
    local_ids_A = view.find_local_vertex_ids_by_label("corner_A")

    assert isinstance(local_ids_A, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in local_ids_A)

    # Test with non-existent label
    local_ids_none = view.find_local_vertex_ids_by_label("nonexistent")
    assert isinstance(local_ids_none, list)
    assert len(local_ids_none) == 0


def test_unprintable_area_fraction():
    """Test unprintable_area_fraction function."""
    from shellforgepy.shells.transformed_region_view import unprintable_area_fraction

    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Test with default angle
    fraction = unprintable_area_fraction(view)

    assert isinstance(fraction, (int, float))
    assert 0.0 <= fraction <= 1.0, f"Fraction {fraction} should be between 0 and 1"

    # Test with custom angle
    fraction_custom = unprintable_area_fraction(view, max_angle_deg=30)

    assert isinstance(fraction_custom, (int, float))
    assert 0.0 <= fraction_custom <= 1.0

    # Stricter angle should generally give higher unprintable fraction
    assert fraction_custom >= fraction or np.isclose(fraction_custom, fraction)


def test_rotation_matrix_about_axis():
    """Test rotation_matrix_about_axis function."""
    from shellforgepy.shells.transformed_region_view import rotation_matrix_about_axis

    # Test rotation about z-axis
    axis = np.array([0, 0, 1])
    angle = np.pi / 2  # 90 degrees

    rotation_matrix = rotation_matrix_about_axis(axis, angle)

    assert isinstance(rotation_matrix, np.ndarray)
    assert rotation_matrix.shape == (3, 3)

    # Should be orthogonal matrix
    assert np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-10)

    # Determinant should be 1
    assert np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-10)

    # Test rotation behavior
    test_vector = np.array([1, 0, 0])
    rotated = rotation_matrix @ test_vector
    expected = np.array(
        [0, -1, 0]
    )  # 90-degree rotation about z (clockwise when looking down)

    # Use a more lenient tolerance for floating point comparisons
    assert np.allclose(rotated, expected, atol=1e-6)


def test_transformed_region_view_caching():
    """Test that caching works correctly."""
    # Create mesh
    points, _ = create_cube_geometry(sphere_radius)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)
    view = TransformedRegionView(partition, 0)

    # Initially cache should be empty
    assert view.face_cache is None
    assert view.vertex_cache is None
    assert view.edge_cache is None

    # First call should populate cache
    vertices1, faces1, edges1 = view.get_transformed_vertices_faces_boundary_edges()

    assert view.face_cache is not None
    assert view.vertex_cache is not None
    assert view.edge_cache is not None

    # Second call should use cache
    vertices2, faces2, edges2 = view.get_transformed_vertices_faces_boundary_edges()

    # Results should be identical (same object references)
    assert vertices1 is vertices2
    assert faces1 is faces2
    assert edges1 is edges2
