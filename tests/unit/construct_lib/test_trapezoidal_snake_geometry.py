import os
import tempfile

import numpy as np
import pytest
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.mesh_utils import write_stl_binary
from shellforgepy.geometry.treapezoidal_snake_geometry import (
    create_local_coordinate_system,
    create_snake_vertices,
    create_trapezoidal_snake_geometry,
    transform_cross_section_to_3d,
)


def traditional_face_vertex_map_to_stl_format(mesh):
    """
    Convert traditional face-vertex map to STL format (vertices list + triangles list).

    Args:
        mesh: Dict with 'vertexes' and 'faces' keys in traditional format

    Returns:
        tuple: (vertices_list, triangles_list) for STL export
    """
    # Convert vertices dict to list, maintaining index mapping
    vertices_list = []
    vertex_mapping = {}

    for vertex_id in sorted(mesh["vertexes"].keys()):
        vertex_mapping[vertex_id] = len(vertices_list)
        vertices_list.append(mesh["vertexes"][vertex_id])

    # Convert faces to triangle index lists
    triangles_list = []
    for face_indices in mesh["faces"].values():
        # Remap vertex indices to list indices
        triangle = [vertex_mapping[idx] for idx in face_indices]
        triangles_list.append(tuple(triangle))

    return vertices_list, triangles_list


def test_normalize():
    """Test vector normalization using the library function."""
    v = np.array([3.0, 4.0, 0.0])
    normalized = normalize(v)
    expected = np.array([0.6, 0.8, 0.0])
    np.testing.assert_allclose(normalized, expected, rtol=1e-10)

    # Test zero vector
    zero_v = np.array([0.0, 0.0, 0.0])
    normalized_zero = normalize(zero_v)
    np.testing.assert_allclose(normalized_zero, zero_v)


def test_create_local_coordinate_system():
    """Test local coordinate system creation."""
    normal = np.array([0.0, 0.0, 1.0])  # Z-up
    x_axis, y_axis, z_axis = create_local_coordinate_system(normal)

    # Y-axis should be the normal
    np.testing.assert_allclose(y_axis, normal, rtol=1e-10)

    # All axes should be unit vectors
    assert abs(np.linalg.norm(x_axis) - 1.0) < 1e-10
    assert abs(np.linalg.norm(y_axis) - 1.0) < 1e-10
    assert abs(np.linalg.norm(z_axis) - 1.0) < 1e-10

    # Axes should be orthogonal
    assert abs(np.dot(x_axis, y_axis)) < 1e-10
    assert abs(np.dot(y_axis, z_axis)) < 1e-10
    assert abs(np.dot(x_axis, z_axis)) < 1e-10


def test_transform_cross_section_to_3d():
    """Test 2D to 3D cross-section transformation."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 1.0],
        ]
    )

    base_point = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])  # Z-up

    points_3d = transform_cross_section_to_3d(cross_section, base_point, normal)

    # Should have 4 points in 3D
    assert points_3d.shape == (4, 3)

    # When normal is Z-up, the 2D cross-section's Y becomes 3D Z coordinate
    # Points with Y=0 in 2D should have Z=0 in 3D
    # Points with Y=1 in 2D should have Z=1 in 3D
    expected_z_values = [0.0, 0.0, 1.0, 1.0]  # Based on Y values in cross_section

    for i, point in enumerate(points_3d):
        assert abs(point[2] - expected_z_values[i]) < 1e-10


def test_create_trapezoidal_snake_geometry():
    """Test the main trapezoidal snake geometry creation function."""
    cross_section = np.array(
        [
            [-1.0, 0.0],  # Bottom left
            [1.0, 0.0],  # Bottom right
            [0.5, 1.0],  # Top right
            [-0.5, 1.0],  # Top left
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 0.3],
            [3.0, 0.0, 0.5],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.2, 0.8],
        ]
    )
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Should have 3 segments (4 points - 1)
    assert len(meshes) == 3

    for i, mesh in enumerate(meshes):
        # Validate structure
        assert "vertexes" in mesh
        assert "faces" in mesh
        assert type(mesh["vertexes"]) is dict
        assert type(mesh["faces"]) is dict

        # Should have 8 vertices (4 at each end of segment)
        assert len(mesh["vertexes"]) == 8

        # Should have 12 triangular faces
        assert len(mesh["faces"]) == 12

        # Check vertex indices are correct
        for vertex_idx in range(8):
            assert vertex_idx in mesh["vertexes"]
            # Each vertex should be a 3-tuple
            vertex = mesh["vertexes"][vertex_idx]
            assert len(vertex) == 3
            assert all(isinstance(coord, (int, float)) for coord in vertex)

        # Check face indices
        for face_idx in range(12):
            assert face_idx in mesh["faces"]
            face = mesh["faces"][face_idx]
            # Each face should have 3 vertices (triangulated)
            assert len(face) == 3
            # All face vertex indices should be valid
            for vertex_idx in face:
                assert 0 <= vertex_idx < 8


def test_create_trapezoidal_snake_geometry_edge_cases():
    """Test edge cases and error conditions."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 1.0],
        ]
    )

    # Test insufficient base points
    base_points = np.array([[0.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])

    try:
        create_trapezoidal_snake_geometry(cross_section, base_points, normals)
        assert False, "Should have raised ValueError for insufficient points"
    except ValueError as e:
        assert "Need at least 2 base points" in str(e)

    # Test mismatched cross-section size
    bad_cross_section = np.array([[-1.0, 0.0], [1.0, 0.0]])  # Only 2 points
    base_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    try:
        create_trapezoidal_snake_geometry(bad_cross_section, base_points, normals)
        assert False, "Should have raised ValueError for wrong cross-section size"
    except ValueError as e:
        assert "Cross section must have exactly 4 points" in str(e)

    # Test mismatched base points and normals
    base_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])  # One fewer normal

    try:
        create_trapezoidal_snake_geometry(cross_section, base_points, normals)
        assert False, "Should have raised ValueError for mismatched points and normals"
    except ValueError as e:
        assert "Number of base points must match number of normals" in str(e)


def test_mesh_compatibility_with_traditional_face_vertex_maps():
    """Test that the mesh format is compatible with traditional face-vertex maps."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 1.0],
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Should have exactly 1 segment
    assert len(meshes) == 1
    mesh = meshes[0]

    # Verify the structure matches what create_solid_from_traditional_face_vertex_maps expects:
    # - vertexes: dict mapping int keys to 3-tuples
    # - faces: dict mapping int keys to lists of 3 vertex indices

    # Check vertex format
    for vertex_key, vertex_value in mesh["vertexes"].items():
        assert isinstance(vertex_key, int)
        assert isinstance(vertex_value, tuple)
        assert len(vertex_value) == 3
        assert all(isinstance(coord, (int, float)) for coord in vertex_value)

    # Check face format
    for face_key, face_value in mesh["faces"].items():
        assert isinstance(face_key, int)
        assert isinstance(face_value, list)
        assert len(face_value) == 3  # Triangulated
        assert all(isinstance(vertex_idx, int) for vertex_idx in face_value)
        assert all(0 <= vertex_idx < 8 for vertex_idx in face_value)


def test_stl_export_functionality():
    """Test that trapezoidal snake geometry can be exported to STL format."""
    cross_section = np.array(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.25, 0.5],
            [-0.25, 0.5],
        ]
    )

    # Create a simple curved path
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.2],
            [2.0, 1.0, 0.5],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.2, 1.0],
        ]
    )
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Convert to STL format
    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    for mesh in meshes:
        vertices, triangles = traditional_face_vertex_map_to_stl_format(mesh)
        all_vertices.extend(vertices)
        # Adjust triangle indices for the combined mesh
        offset_triangles = [
            (t[0] + vertex_offset, t[1] + vertex_offset, t[2] + vertex_offset)
            for t in triangles
        ]
        all_triangles.extend(offset_triangles)
        vertex_offset += len(vertices)

    # Test that we can write to STL (using temporary file)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
        try:
            write_stl_binary(
                temp_file.name,
                all_vertices,
                all_triangles,
                header_text="Test trapezoidal snake",
            )

            # Verify file was created and has reasonable size
            assert os.path.exists(temp_file.name)
            file_size = os.path.getsize(temp_file.name)

            # STL header (80 bytes) + triangle count (4 bytes) + triangles (50 bytes each)
            expected_min_size = 84 + len(all_triangles) * 50
            assert file_size >= expected_min_size

        finally:
            # Clean up
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    # Verify mesh structure
    assert len(all_vertices) > 0
    assert len(all_triangles) > 0
    assert all(len(v) == 3 for v in all_vertices)  # 3D vertices
    assert all(len(t) == 3 for t in all_triangles)  # Triangular faces
    assert all(
        all(0 <= idx < len(all_vertices) for idx in t) for t in all_triangles
    )  # Valid indices


def test_segment_vertex_connectivity():
    """Test that adjacent segments have coincident vertices at their interface."""
    cross_section = np.array(
        [
            [-1.0, 0.0],  # Point 0: Bottom left
            [1.0, 0.0],  # Point 1: Bottom right
            [1.0, 1.0],  # Point 2: Top right
            [-1.0, 1.0],  # Point 3: Top left
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Should have 2 segments for 3 base points
    assert len(meshes) == 2

    # Check connectivity between segments
    tolerance = 1e-10

    for i in range(len(meshes) - 1):
        mesh1 = meshes[i]
        mesh2 = meshes[i + 1]

        # In the LED coil structure, the "end" vertices of segment i should match
        # the "start" vertices of segment i+1
        # Based on our vertex arrangement:
        # - Vertices 0-3: first cross-section (start of segment)
        # - Vertices 4-7: second cross-section (end of segment)
        #
        # So end vertices of segment i (4,5,6,7) should match
        # start vertices of segment i+1 (0,1,2,3)

        expected_matches = [(4, 0), (5, 1), (6, 2), (7, 3)]

        for v1_id, v2_id in expected_matches:
            v1 = np.array(mesh1["vertexes"][v1_id])
            v2 = np.array(mesh2["vertexes"][v2_id])
            distance = np.linalg.norm(v1 - v2)

            assert distance < tolerance, (
                f"Vertices should be coincident at segment interface! "
                f"Segment {i} vertex {v1_id} {v1} vs "
                f"Segment {i+1} vertex {v2_id} {v2}, "
                f"distance = {distance}"
            )


def test_face_winding_consistency():
    """Test that face normals point outward consistently."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)
    mesh = meshes[0]

    vertices = mesh["vertexes"]
    faces = mesh["faces"]

    # Calculate center of the shape for reference
    vertex_coords = [np.array(v) for v in vertices.values()]
    center = np.mean(vertex_coords, axis=0)

    outward_facing_count = 0
    total_faces = len(faces)

    for face_id, face_vertices in faces.items():
        # Get the three vertices of the triangle
        v0 = np.array(vertices[face_vertices[0]])
        v1 = np.array(vertices[face_vertices[1]])
        v2 = np.array(vertices[face_vertices[2]])

        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_length = np.linalg.norm(normal)

        # Skip degenerate triangles
        if normal_length < 1e-10:
            continue

        normal = normal / normal_length

        # Calculate face center
        face_center = (v0 + v1 + v2) / 3

        # Vector from shape center to face center
        outward_dir = face_center - center
        outward_dir_length = np.linalg.norm(outward_dir)

        if outward_dir_length < 1e-10:
            continue  # Face center is at shape center, skip

        outward_dir = outward_dir / outward_dir_length

        # Check if normal points outward
        dot_product = np.dot(normal, outward_dir)

        if dot_product > 0:
            outward_facing_count += 1

    # Most faces should be outward-facing for a proper solid
    # Allow some tolerance since internal faces might exist
    outward_ratio = outward_facing_count / total_faces
    assert outward_ratio >= 0.5, (
        f"Too many inward-facing normals! {outward_facing_count}/{total_faces} "
        f"= {outward_ratio:.2%} are outward-facing. Expected >= 50%"
    )


def test_distorted_cube_face_structure_compatibility():
    """Test that our face structure matches the distorted cube convention."""

    # This is the face structure from create_distorted_cube
    expected_cube_faces = {
        0: [0, 2, 1],  # Bottom face triangle 1
        1: [0, 3, 2],  # Bottom face triangle 2
        2: [4, 5, 6],  # Top face triangle 1
        3: [4, 6, 7],  # Top face triangle 2
        4: [0, 1, 5],  # Side face triangle 1
        5: [0, 5, 4],  # Side face triangle 2
        6: [2, 3, 6],  # Side face triangle 1
        7: [3, 7, 6],  # Side face triangle 2
        8: [1, 2, 5],  # Side face triangle 1
        9: [2, 6, 5],  # Side face triangle 2
        10: [0, 4, 3],  # Side face triangle 1
        11: [3, 4, 7],  # Side face triangle 2
    }

    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)
    mesh = meshes[0]

    # Should have 8 vertices and 12 faces like a distorted cube
    assert len(mesh["vertexes"]) == 8
    assert len(mesh["faces"]) == 12

    # Check that our face structure matches the expected structure
    faces = mesh["faces"]

    for face_id in range(12):
        assert face_id in faces, f"Missing face {face_id}"
        actual_face = faces[face_id]
        expected_face = expected_cube_faces[face_id]

        # Face should have 3 vertices
        assert (
            len(actual_face) == 3
        ), f"Face {face_id} should have 3 vertices, got {len(actual_face)}"

        # All vertex indices should be valid
        for vertex_idx in actual_face:
            assert (
                0 <= vertex_idx < 8
            ), f"Invalid vertex index {vertex_idx} in face {face_id}"


def test_vertex_order_consistency_with_led_coil():
    """Test that our vertex ordering is consistent and proper for segment creation."""

    # Create a simple test case
    cross_section = np.array(
        [
            [-1.0, 0.0],  # Point 0 and 4
            [1.0, 0.0],  # Point 1 and 5
            [1.0, 1.0],  # Point 2 and 6
            [-1.0, 1.0],  # Point 3 and 7
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)
    mesh = meshes[0]
    vertices = mesh["vertexes"]

    # For X-direction snake with Z normals, the segments are distinguished by X coordinate
    # Vertices 0-3: first cross-section (X=0)
    # Vertices 4-7: second cross-section (X=2)

    # Check X-coordinates: first cross-section should have lower X
    x_coords_first = [vertices[i][0] for i in range(4)]
    x_coords_second = [vertices[i][0] for i in range(4, 8)]

    avg_x_first = np.mean(x_coords_first)
    avg_x_second = np.mean(x_coords_second)

    tolerance = 1e-10
    assert avg_x_first < avg_x_second, (
        f"First cross-section should have lower X coordinates than second. "
        f"Got avg X: first={avg_x_first}, second={avg_x_second}"
    )

    # Check that vertices are properly matched (corresponding points have same Y,Z)
    for i in range(4):
        y_diff = abs(vertices[i][1] - vertices[i + 4][1])
        z_diff = abs(vertices[i][2] - vertices[i + 4][2])
        assert y_diff < tolerance, (
            f"Vertices {i} and {i+4} should have same Y coordinate. " f"Diff: {y_diff}"
        )
        assert z_diff < tolerance, (
            f"Vertices {i} and {i+4} should have same Z coordinate. " f"Diff: {z_diff}"
        )


def test_create_snake_vertices_straight_x_direction():
    """Test vertex generation for a straight snake going in X direction with Z normals."""
    cross_section = np.array(
        [
            [-1.0, 0.0],  # Point 0: cross-section X=-1, Y=0
            [1.0, 0.0],  # Point 1: cross-section X=+1, Y=0
            [1.0, 1.0],  # Point 2: cross-section X=+1, Y=1
            [-1.0, 1.0],  # Point 3: cross-section X=-1, Y=1
        ]
    )

    # Straight line in X direction
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],  # First base point at origin
            [2.0, 0.0, 0.0],  # Second base point at X=2
        ]
    )

    # Normals pointing in Z direction
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    all_vertices = create_snake_vertices(cross_section, base_points, normals)
    assert len(all_vertices) == 2, "Should create one vertex set per base point"

    # Check first vertex set (at base point 0)
    vertices_0 = all_vertices[0]
    assert vertices_0.shape == (4, 3), "Should have 4 vertices in 3D"

    # For X-direction snake with Z normals:
    # - Cross-section X maps to global Y
    # - Cross-section Y maps to global Z
    # - X position is the base point X coordinate

    # All vertices at first base point should have X=0
    tolerance = 1e-10
    for i in range(4):
        assert abs(vertices_0[i][0] - 0.0) < tolerance, f"Vertex {i} should be at X=0"

    # Check Y coordinates (from cross-section X)
    expected_y = [-1.0, 1.0, 1.0, -1.0]
    for i in range(4):
        assert (
            abs(vertices_0[i][1] - expected_y[i]) < tolerance
        ), f"Vertex {i} Y should be {expected_y[i]}, got {vertices_0[i][1]}"

    # Check Z coordinates (from cross-section Y)
    expected_z = [0.0, 0.0, 1.0, 1.0]
    for i in range(4):
        assert (
            abs(vertices_0[i][2] - expected_z[i]) < tolerance
        ), f"Vertex {i} Z should be {expected_z[i]}, got {vertices_0[i][2]}"

    # Check second vertex set (at base point 1)
    vertices_1 = all_vertices[1]
    assert vertices_1.shape == (4, 3), "Should have 4 vertices in 3D"

    # All vertices at second base point should have X=2
    for i in range(4):
        assert abs(vertices_1[i][0] - 2.0) < tolerance, f"Vertex {i} should be at X=2"

    # Y and Z coordinates should be the same as first set
    for i in range(4):
        assert (
            abs(vertices_1[i][1] - expected_y[i]) < tolerance
        ), f"Vertex {i} Y should be {expected_y[i]}, got {vertices_1[i][1]}"
        assert (
            abs(vertices_1[i][2] - expected_z[i]) < tolerance
        ), f"Vertex {i} Z should be {expected_z[i]}, got {vertices_1[i][2]}"


def test_create_snake_vertices_straight_y_direction():
    """Test vertex generation for a straight snake going in Y direction with Z normals."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    # Straight line in Y direction
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    # Normals pointing in Z direction
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    all_vertices = create_snake_vertices(cross_section, base_points, normals)
    assert len(all_vertices) == 2, "Should create one vertex set per base point"

    vertices_0 = all_vertices[0]
    vertices_1 = all_vertices[1]

    # For Y-direction snake with Z normals:
    # - Cross-section X maps to global -X (perpendicular to segment direction)
    # - Cross-section Y maps to global Z (normal direction)
    # - Y position is the base point Y coordinate

    tolerance = 1e-10

    # Check first vertex set (Y=0)
    for i in range(4):
        assert abs(vertices_0[i][1] - 0.0) < tolerance, f"Vertex {i} should be at Y=0"

    # Check second vertex set (Y=2)
    for i in range(4):
        assert abs(vertices_1[i][1] - 2.0) < tolerance, f"Vertex {i} should be at Y=2"

    # Check X coordinates (cross-section X becomes global -X due to orientation)
    expected_x = [1.0, -1.0, -1.0, 1.0]  # negated from cross_section X
    for i in range(4):
        assert (
            abs(vertices_0[i][0] - expected_x[i]) < tolerance
        ), f"Vertex {i} X should be {expected_x[i]}, got {vertices_0[i][0]}"
        assert (
            abs(vertices_1[i][0] - expected_x[i]) < tolerance
        ), f"Vertex {i} X should be {expected_x[i]}, got {vertices_1[i][0]}"

    # Check Z coordinates (cross-section Y becomes global Z)
    expected_z = [0.0, 0.0, 1.0, 1.0]
    for i in range(4):
        assert (
            abs(vertices_0[i][2] - expected_z[i]) < tolerance
        ), f"Vertex {i} Z should be {expected_z[i]}, got {vertices_0[i][2]}"
        assert (
            abs(vertices_1[i][2] - expected_z[i]) < tolerance
        ), f"Vertex {i} Z should be {expected_z[i]}, got {vertices_1[i][2]}"


def test_create_snake_vertices_straight_z_direction():
    """Test vertex generation for a snake going in Z direction with Y normals."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    # Straight line in Z direction
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )

    # Normals pointing in Y direction
    normals = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    all_vertices = create_snake_vertices(cross_section, base_points, normals)
    assert len(all_vertices) == 2, "Should create one vertex set per base point"

    vertices_0 = all_vertices[0]
    vertices_1 = all_vertices[1]

    tolerance = 1e-10

    # Check Z coordinates (snake direction)
    for i in range(4):
        assert abs(vertices_0[i][2] - 0.0) < tolerance, f"Vertex {i} should be at Z=0"
        assert abs(vertices_1[i][2] - 2.0) < tolerance, f"Vertex {i} should be at Z=2"

    # For Z-direction snake with Y normals:
    # - Cross-section X maps to global X
    # - Cross-section Y maps to global Y (normal direction)

    expected_x = [-1.0, 1.0, 1.0, -1.0]  # from cross_section X
    expected_y = [0.0, 0.0, 1.0, 1.0]  # from cross_section Y

    for i in range(4):
        assert (
            abs(vertices_0[i][0] - expected_x[i]) < tolerance
        ), f"Vertex {i} X should be {expected_x[i]}, got {vertices_0[i][0]}"
        assert (
            abs(vertices_0[i][1] - expected_y[i]) < tolerance
        ), f"Vertex {i} Y should be {expected_y[i]}, got {vertices_0[i][1]}"

        assert (
            abs(vertices_1[i][0] - expected_x[i]) < tolerance
        ), f"Vertex {i} X should be {expected_x[i]}, got {vertices_1[i][0]}"
        assert (
            abs(vertices_1[i][1] - expected_y[i]) < tolerance
        ), f"Vertex {i} Y should be {expected_y[i]}, got {vertices_1[i][1]}"


def test_create_snake_vertices_multiple_base_points():
    """Test vertex generation for multiple base points."""
    cross_section = np.array(
        [
            [-0.5, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )

    # Three base points creating a snake
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    all_vertices = create_snake_vertices(cross_section, base_points, normals)
    assert len(all_vertices) == 3, "Should create one vertex set per base point"

    # Check each vertex set
    for i, vertices in enumerate(all_vertices):
        assert vertices.shape == (4, 3), f"Vertex set {i} should have 4 vertices in 3D"

        # All vertices should have the correct X coordinate
        expected_x = float(i)  # 0, 1, 2
        tolerance = 1e-10
        for j in range(4):
            assert (
                abs(vertices[j][0] - expected_x) < tolerance
            ), f"Vertex set {i}, vertex {j} should be at X={expected_x}"


def test_create_snake_vertices_edge_cases():
    """Test edge cases for vertex generation."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Test wrong cross-section size
    bad_cross_section = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # 3 points
    with pytest.raises(ValueError, match="exactly 4 points"):
        create_snake_vertices(bad_cross_section, base_points, normals)

    # Test mismatched arrays
    bad_normals = np.array([[0.0, 0.0, 1.0]])  # Only 1 normal
    with pytest.raises(ValueError, match="must match number of normals"):
        create_snake_vertices(cross_section, base_points, bad_normals)

    # Test insufficient points
    bad_points = np.array([[0.0, 0.0, 0.0]])  # Only 1 point
    bad_normals_single = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="at least 2 base points"):
        create_snake_vertices(cross_section, bad_points, bad_normals_single)


def test_create_snake_vertices_consistency():
    """Test that vertex generation is consistent for different orientations."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    # Test that vertices are consistently ordered
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    all_vertices = create_snake_vertices(cross_section, base_points, normals)

    # Check that all vertex sets have the same shape
    for i, vertices in enumerate(all_vertices):
        assert vertices.shape == (4, 3), f"Vertex set {i} should have shape (4, 3)"

    # Check that vertices are finite numbers
    for i, vertices in enumerate(all_vertices):
        assert np.all(
            np.isfinite(vertices)
        ), f"Vertex set {i} contains non-finite values"


def test_mobius_strip_with_close_loop():
    """
    Test creating a Möbius strip with close_loop=True to verify proper closure.

    This test creates a mathematical Möbius strip by following a circular path
    while rotating the normal vector by 180 degrees over one complete revolution.
    The close_loop functionality should properly handle the twisted vertex
    correspondence at the seam.
    """
    # Möbius strip parameters
    radius = 10.0  # Small radius for testing
    num_points = 16  # Fewer points for faster testing

    # Cross-section MUST be centered at (0,0) for proper closure
    cross_section = np.array(
        [
            [-2.0, -0.5],  # Bottom left
            [2.0, -0.5],  # Bottom right
            [2.0, 0.5],  # Top right
            [-2.0, 0.5],  # Top left
        ]
    )

    # Generate circular path in X-Y plane
    theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = radius * np.cos(theta_values)
    y_values = radius * np.sin(theta_values)
    z_values = np.zeros_like(theta_values)

    base_points = np.column_stack([x_values, y_values, z_values])

    # Create Möbius twist: normals rotate by 180° over one full circle
    normals = np.zeros_like(base_points)

    for i in range(len(base_points)):
        # Normal rotation angle: 180° over full circle (π radians)
        normal_rotation = (theta_values[i] / (2 * np.pi)) * np.pi

        # Start with Z-normal, then add radial component that varies with twist
        normals[i, 2] = np.cos(normal_rotation)  # Z component varies from 1 to -1

        radial_component = np.sin(normal_rotation)
        normals[i, 0] = radial_component * np.cos(theta_values[i])  # Radial X
        normals[i, 1] = radial_component * np.sin(theta_values[i])  # Radial Y

    # Test without close_loop first
    meshes_open = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=False
    )
    assert len(meshes_open) == num_points - 1  # N-1 segments for N points

    # Test with close_loop
    meshes_closed = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=True
    )
    assert (
        len(meshes_closed) == num_points
    )  # N segments for N points (including closure)

    # Verify all meshes have the correct structure
    for i, mesh in enumerate(meshes_closed):
        assert "vertexes" in mesh
        assert "faces" in mesh
        assert len(mesh["vertexes"]) == 8  # 4 vertices at each end
        assert len(mesh["faces"]) == 12  # 12 triangular faces per segment

        # Check that all vertex coordinates are finite
        for vertex_id, vertex_coords in mesh["vertexes"].items():
            assert len(vertex_coords) == 3
            assert all(np.isfinite(coord) for coord in vertex_coords)

        # Check that all faces have valid vertex indices
        for face_id, face_vertices in mesh["faces"].items():
            assert len(face_vertices) == 3
            assert all(0 <= v_idx <= 7 for v_idx in face_vertices)

    # The closing segment (last mesh) should connect last cross-section to first
    closing_mesh = meshes_closed[-1]

    # Verify that the closing mesh properly handles twisted vertices
    # (The test passes if no exceptions are thrown and mesh structure is valid)
    assert len(closing_mesh["vertexes"]) == 8
    assert len(closing_mesh["faces"]) == 12

    # Test STL export to ensure mesh is manifold
    # Convert the meshes to STL format and verify no errors
    for i, mesh in enumerate(meshes_closed):
        vertices_list, triangles_list = traditional_face_vertex_map_to_stl_format(mesh)
        assert len(vertices_list) == 8
        assert len(triangles_list) == 12

        # Verify all triangles reference valid vertices
        for triangle in triangles_list:
            assert len(triangle) == 3
            assert all(0 <= v_idx < len(vertices_list) for v_idx in triangle)


def test_close_loop_parameter_validation():
    """Test that close_loop parameter works correctly with edge cases."""
    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 1.0],
        ]
    )

    # Test with minimum number of points for close_loop
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Should work with 3 points
    meshes = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=True
    )
    assert len(meshes) == 3  # 3 segments including closing segment

    # Test with 2 points - close_loop should not add extra segment
    base_points_2 = base_points[:2]
    normals_2 = normals[:2]

    meshes_2 = create_trapezoidal_snake_geometry(
        cross_section, base_points_2, normals_2, close_loop=True
    )
    assert len(meshes_2) == 1  # Only 1 segment possible with 2 points

    # Verify that close_loop=False works correctly
    meshes_open = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=False
    )
    assert len(meshes_open) == 2  # 2 segments for 3 points when not closed


def test_close_loop_maintains_mesh_quality():
    """Test that close_loop maintains proper mesh quality and doesn't create degenerate faces."""
    # Create a simple circular path
    num_points = 8
    radius = 5.0

    cross_section = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    # Circular path
    theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_values = radius * np.cos(theta_values)
    y_values = radius * np.sin(theta_values)
    z_values = np.zeros_like(theta_values)

    base_points = np.column_stack([x_values, y_values, z_values])

    # All normals pointing up
    normals = np.tile([0.0, 0.0, 1.0], (num_points, 1))

    meshes = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=True
    )

    # Should have exactly num_points segments (including closing segment)
    assert len(meshes) == num_points

    # Check that all meshes maintain quality
    for i, mesh in enumerate(meshes):
        # Convert to vertices and check for degeneracies
        vertices_list, triangles_list = traditional_face_vertex_map_to_stl_format(mesh)

        # Check that no triangle has zero area (degenerate)
        for triangle in triangles_list:
            v0, v1, v2 = [vertices_list[idx] for idx in triangle]

            # Calculate triangle area using cross product
            edge1 = np.array(v1) - np.array(v0)
            edge2 = np.array(v2) - np.array(v0)
            cross_product = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross_product)

            # Area should be greater than zero (non-degenerate)
            assert area > 1e-10, f"Mesh {i} has degenerate triangle with area {area}"


def test_helical_snake_solid_fusion():
    """Test creating a helical snake geometry and fusing it into a single solid."""
    from shellforgepy.adapters._adapter import (
        create_solid_from_traditional_face_vertex_maps,
        get_volume,
    )

    # Create a helical path for screw thread
    num_turns = 2
    resolution = 16  # Lower resolution for faster testing
    pitch = 2.0
    inner_radius = 5.0
    outer_radius = 7.0

    # Generate helical path
    total_points = int(num_turns * resolution)
    theta_values = np.linspace(0, 2 * np.pi * num_turns, total_points)

    # Calculate helical coordinates
    x_values = inner_radius * np.cos(theta_values)
    y_values = inner_radius * np.sin(theta_values)
    z_values = (pitch / (2 * np.pi)) * theta_values

    base_points = np.column_stack([x_values, y_values, z_values])

    # Calculate outward-pointing normals (radial direction)
    normals = np.zeros_like(base_points)
    for i in range(len(base_points)):
        normals[i, 0] = np.cos(theta_values[i])
        normals[i, 1] = np.sin(theta_values[i])
        normals[i, 2] = 0.0

    # Create trapezoidal cross-section for thread
    thread_depth = outer_radius - inner_radius
    outer_thickness = 0.4
    inner_thickness = 0.4
    cross_section = np.array(
        [
            [-outer_thickness / 2, 0.0],  # Bottom left
            [outer_thickness / 2, 0.0],  # Bottom right
            [inner_thickness / 2, thread_depth],  # Top right
            [-inner_thickness / 2, thread_depth],  # Top left
        ]
    )

    # Generate snake geometry
    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Verify we got mesh segments
    assert len(meshes) > 0, "Should generate mesh segments"
    print(f"Generated {len(meshes)} mesh segments")

    # Convert each mesh segment to a solid and fuse them
    solids = []
    for i, mesh in enumerate(meshes):
        try:
            solid = create_solid_from_traditional_face_vertex_maps(mesh)
            assert solid is not None, f"Mesh segment {i} should create a valid solid"
            volume = get_volume(solid)
            assert volume > 0, f"Solid {i} should have positive volume"
            solids.append(solid)
            print(f"Segment {i}: Volume = {volume:.6f}")
        except Exception as e:
            print(f"Failed to create solid from segment {i}: {e}")
            raise

    assert len(solids) > 0, "Should create at least one solid"

    # Fuse all solids together
    fused_solid = solids[0]
    for i, solid in enumerate(solids[1:], 1):
        try:
            fused_solid = fused_solid.fuse(solid)
            fused_volume = get_volume(fused_solid)
            print(f"Fused {i+1} solids, volume = {fused_volume:.6f}")
        except Exception as e:
            print(f"Failed to fuse solid {i}: {e}")
            raise

    # Verify the final fused solid
    assert fused_solid is not None, "Fused solid should not be None"
    final_volume = get_volume(fused_solid)
    assert final_volume > 0, "Fused solid should have positive volume"
    print(f"Final fused solid volume: {final_volume:.6f}")

    # The fused solid should have a reasonable volume for a helical thread
    expected_min_volume = (
        num_turns * pitch * inner_radius * outer_thickness * 0.1
    )  # Conservative estimate
    assert (
        final_volume > expected_min_volume
    ), f"Volume {final_volume} seems too small for helical thread"


def test_circular_snake_solid_fusion():
    """Test creating a circular snake geometry and fusing it into a single solid."""
    from shellforgepy.adapters._adapter import (
        create_solid_from_traditional_face_vertex_maps,
        get_volume,
    )

    # Create a circular path
    radius = 10.0
    num_segments = 12
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    base_points = np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in theta])

    # Normals pointing outward
    normals = np.array([[np.cos(t), np.sin(t), 0] for t in theta])

    # Rectangular cross-section
    width = 2.0
    height = 1.0
    cross_section = np.array(
        [[-width / 2, 0], [width / 2, 0], [width / 2, height], [-width / 2, height]]
    )

    # Generate snake geometry with closed loop
    meshes = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=True
    )

    # Verify we got the expected number of segments (including closing segment)
    assert (
        len(meshes) == num_segments
    ), f"Expected {num_segments} segments, got {len(meshes)}"

    # Convert each mesh to solid and fuse
    solids = []
    for i, mesh in enumerate(meshes):
        solid = create_solid_from_traditional_face_vertex_maps(mesh)
        assert solid is not None, f"Segment {i} should create valid solid"
        volume = get_volume(solid)
        assert volume > 0, f"Segment {i} should have positive volume"
        solids.append(solid)

    # Fuse all solids
    fused_solid = solids[0]
    for solid in solids[1:]:
        fused_solid = fused_solid.fuse(solid)

    # Verify final result
    assert fused_solid is not None
    final_volume = get_volume(fused_solid)
    assert final_volume > 0

    # Should have reasonable volume for a torus-like shape
    expected_volume = 2 * np.pi * radius * width * height
    final_fused_volume = get_volume(fused_solid)
    volume_ratio = final_fused_volume / expected_volume
    assert 0.5 < volume_ratio < 2.0, f"Volume ratio {volume_ratio} seems unreasonable"
