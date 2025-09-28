import os
import struct
import tempfile

import numpy as np
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.geometry.mesh_utils import (
    _cross,
    _merge_duplicate_vertices,
    _norm,
    _normalize,
    _sub,
    detect_twisted_vertices,
    fix_twisted_vertex_correspondence,
    merge_meshes,
    shell_maps_to_unified_mesh,
    validate_and_fix_mesh_segment,
    write_shell_maps_to_stl,
    write_stl_binary,
)
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)


def test_vector_operations():
    """Test basic vector operations used in STL export."""
    v1 = (1.0, 2.0, 3.0)
    v2 = (4.0, 5.0, 6.0)

    # Test subtraction
    result = _sub(v1, v2)
    expected = (-3.0, -3.0, -3.0)
    assert result == expected

    # Test cross product
    cross = _cross(v1, v2)
    expected_cross = (
        2.0 * 6.0 - 3.0 * 5.0,
        3.0 * 4.0 - 1.0 * 6.0,
        1.0 * 5.0 - 2.0 * 4.0,
    )
    assert cross == expected_cross

    # Test norm
    norm = _norm(v1)
    expected_norm = (1.0 + 4.0 + 9.0) ** 0.5
    assert abs(norm - expected_norm) < 1e-6

    # Test normalize
    normalized = _normalize(v1)
    normalized_norm = _norm(normalized)
    assert abs(normalized_norm - 1.0) < 1e-6

    # Test zero vector normalization
    zero_normalized = _normalize((0.0, 0.0, 0.0))
    assert zero_normalized == (0.0, 0.0, 0.0)


def test_write_stl_binary_basic():
    """Test basic STL binary writing."""
    # Simple triangle
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    triangles = [(0, 1, 2)]

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles, header_text="test triangle")

        # Verify file was created and has correct size
        assert os.path.exists(output_path)

        # Check file structure
        with open(output_path, "rb") as f:
            # Header (80 bytes)
            header = f.read(80)
            assert len(header) == 80
            assert header.startswith(b"test triangle")

            # Triangle count (4 bytes)
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count == 1

            # Triangle data (50 bytes per triangle)
            triangle_data = f.read(50)
            assert len(triangle_data) == 50

            # Should be no more data
            remaining = f.read()
            assert len(remaining) == 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_write_stl_binary_multiple_triangles():
    """Test STL writing with multiple triangles."""
    # Square made of two triangles
    vertices = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]
    triangles = [(0, 1, 2), (0, 2, 3)]  # First triangle  # Second triangle

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles)

        # Check triangle count
        with open(output_path, "rb") as f:
            f.seek(80)  # Skip header
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count == 2
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_shell_maps_to_unified_mesh_basic():
    """Test basic shell map to unified mesh conversion."""
    # Create simple shell map (like from calculate_materialized_shell_maps)
    simple_shell_maps = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
            },
            "faces": {0: [0, 1, 2]},
        }
    }

    vertices, triangles = shell_maps_to_unified_mesh(
        simple_shell_maps, remove_inner_faces=False, merge_duplicate_vertices=False
    )

    # Check results
    assert len(vertices) == 3
    assert len(triangles) == 1
    assert triangles[0] == (0, 1, 2)

    # Check vertex values
    expected_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    for i, expected in enumerate(expected_vertices):
        assert vertices[i] == expected


def test_shell_maps_to_unified_mesh_multiple_shells():
    """Test shell map conversion with multiple shells."""
    # Create test shell maps with two triangle prisms
    test_shell_maps = {
        0: {  # First shell (triangle prism)
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),  # inner triangle
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
                3: np.array([0.0, 0.0, 1.0]),  # outer triangle
                4: np.array([1.0, 0.0, 1.0]),
                5: np.array([0.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom (inner)
                1: [3, 4, 5],  # top (outer)
                2: [0, 1, 4],  # side faces
                3: [0, 4, 3],
            },
        },
        1: {  # Second shell (adjacent triangle prism)
            "vertexes": {
                0: np.array([1.0, 0.0, 0.0]),  # inner triangle (shared edge)
                1: np.array([2.0, 0.0, 0.0]),
                2: np.array([1.5, 1.0, 0.0]),
                3: np.array([1.0, 0.0, 1.0]),  # outer triangle
                4: np.array([2.0, 0.0, 1.0]),
                5: np.array([1.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom (inner)
                1: [3, 4, 5],  # top (outer)
                2: [0, 1, 4],  # side faces
                3: [0, 4, 3],
            },
        },
    }

    vertices, triangles = shell_maps_to_unified_mesh(
        test_shell_maps, remove_inner_faces=False, merge_duplicate_vertices=False
    )

    # Should have vertices from both shells
    assert len(vertices) == 12  # 6 vertices per shell * 2 shells
    assert len(triangles) == 8  # 4 faces per shell * 2 shells


def test_merge_duplicate_vertices():
    """Test vertex merging functionality."""
    # Create vertices with duplicates
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),  # Duplicate of vertex 0
        (1.0, 0.0, 0.0001),  # Near duplicate of vertex 1
    ]
    triangles = [(0, 1, 2), (1, 2, 3)]

    merged_vertices, merged_triangles = _merge_duplicate_vertices(
        vertices, triangles, tolerance=1e-3
    )

    # Should have fewer vertices after merging
    assert len(merged_vertices) < len(vertices)

    # All triangle indices should be valid
    for triangle in merged_triangles:
        for vertex_idx in triangle:
            assert vertex_idx < len(merged_vertices)


def test_merge_duplicate_vertices_removes_degenerate_triangles():
    """Test that degenerate triangles are removed during vertex merging."""
    vertices = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0001),  # Very close to vertex 0
        (1.0, 0.0, 0.0),
    ]
    triangles = [(0, 1, 2)]  # Will become degenerate after merging

    merged_vertices, merged_triangles = _merge_duplicate_vertices(
        vertices, triangles, tolerance=1e-3
    )

    # Degenerate triangle should be removed
    assert len(merged_triangles) == 0


def test_write_shell_maps_to_stl_integration():
    """Test complete integration from shell maps to STL file."""
    # Create simple test shell map
    test_shell_maps = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
                3: np.array([0.0, 0.0, 1.0]),
                4: np.array([1.0, 0.0, 1.0]),
                5: np.array([0.5, 1.0, 1.0]),
            },
            "faces": {
                0: [0, 2, 1],  # bottom
                1: [3, 4, 5],  # top
                2: [0, 1, 4],  # side
                3: [0, 4, 3],  # side
            },
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_shell_maps_to_stl(
            output_path,
            test_shell_maps,
            header_text="test shell mesh",
            remove_inner_faces=True,
            merge_duplicate_vertices=True,
        )

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify it's a valid STL file
        with open(output_path, "rb") as f:
            # Check header
            header = f.read(80)
            assert len(header) == 80

            # Check triangle count is reasonable
            tri_count_bytes = f.read(4)
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            assert tri_count > 0
            assert tri_count < 100  # Reasonable upper bound for our test data
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_numpy_array_conversion():
    """Test that numpy arrays are properly converted to tuples."""
    shell_maps_with_numpy = {
        0: {
            "vertexes": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0]),
                2: np.array([0.5, 1.0, 0.0]),
            },
            "faces": {0: [0, 1, 2]},
        }
    }

    vertices, triangles = shell_maps_to_unified_mesh(shell_maps_with_numpy)

    # All vertices should be tuples, not numpy arrays
    for vertex in vertices:
        assert isinstance(vertex, tuple)
        assert len(vertex) == 3
        for coord in vertex:
            assert isinstance(coord, float)


def test_empty_shell_maps():
    """Test handling of empty shell maps."""
    empty_shell_maps = {}

    vertices, triangles = shell_maps_to_unified_mesh(empty_shell_maps)

    assert len(vertices) == 0
    assert len(triangles) == 0


def test_compute_normals_false():
    """Test STL writing with compute_normals=False."""
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    triangles = [(0, 1, 2)]

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        output_path = f.name

    try:
        write_stl_binary(output_path, vertices, triangles, compute_normals=False)

        # Verify file was created
        assert os.path.exists(output_path)

        # Check that normals are zero
        with open(output_path, "rb") as f:
            f.seek(84)  # Skip header + triangle count
            normal_bytes = f.read(12)  # 3 floats for normal
            normal = struct.unpack("<3f", normal_bytes)
            assert normal == (0.0, 0.0, 0.0)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_merge_meshes_basic():
    """Test basic mesh merging functionality."""
    # First mesh: a triangle
    vertices_1 = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]
    faces_1 = [(0, 1, 2)]

    # Second mesh: another triangle, sharing one vertex
    vertices_2 = [
        (1.0, 0.0, 0.0),  # 0 (should merge with vertex 1 from first mesh)
        (2.0, 0.0, 0.0),  # 1
        (1.5, 1.0, 0.0),  # 2
    ]
    faces_2 = [(0, 1, 2)]

    tolerance = 1e-6
    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should have 5 unique vertices (one shared)
    assert len(merged_vertices) == 5

    # Should have 2 faces
    assert len(merged_faces) == 2

    # All face indices should be valid
    for face in merged_faces:
        for vertex_idx in face:
            assert 0 <= vertex_idx < len(merged_vertices)

    # Check that shared vertex is properly merged
    # Find the shared vertex (1.0, 0.0, 0.0)
    shared_vertex_found = False
    for vertex in merged_vertices:
        if (
            abs(vertex[0] - 1.0) < tolerance
            and abs(vertex[1] - 0.0) < tolerance
            and abs(vertex[2] - 0.0) < tolerance
        ):
            shared_vertex_found = True
            break
    assert shared_vertex_found


def test_merge_meshes_no_shared_vertices():
    """Test merging meshes with no shared vertices."""
    # First mesh: triangle at origin
    vertices_1 = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 1.0, 0.0),
    ]
    faces_1 = [(0, 1, 2)]

    # Second mesh: triangle far away
    vertices_2 = [
        (10.0, 10.0, 0.0),
        (11.0, 10.0, 0.0),
        (10.5, 11.0, 0.0),
    ]
    faces_2 = [(0, 1, 2)]

    tolerance = 1e-6
    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should have all 6 vertices (no merging)
    assert len(merged_vertices) == 6

    # Should have 2 faces
    assert len(merged_faces) == 2

    # All face indices should be valid
    for face in merged_faces:
        for vertex_idx in face:
            assert 0 <= vertex_idx < len(merged_vertices)


def test_merge_meshes_multiple_shared_vertices():
    """Test merging meshes with multiple shared vertices."""
    # First mesh: a triangle
    vertices_1 = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]
    faces_1 = [(0, 1, 2)]

    # Second mesh: adjacent triangle sharing an edge
    vertices_2 = [
        (1.0, 0.0, 0.0),  # 0 (shares with vertex 1 from first mesh)
        (0.5, 1.0, 0.0),  # 1 (shares with vertex 2 from first mesh)
        (1.5, 1.0, 0.0),  # 2 (new vertex)
    ]
    faces_2 = [(0, 1, 2)]

    tolerance = 1e-6
    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should have 4 unique vertices (2 shared)
    assert len(merged_vertices) == 4

    # Should have 2 faces
    assert len(merged_faces) == 2

    # All face indices should be valid
    for face in merged_faces:
        for vertex_idx in face:
            assert 0 <= vertex_idx < len(merged_vertices)


def test_merge_meshes_tolerance():
    """Test that tolerance parameter works correctly."""
    # First mesh
    vertices_1 = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 1.0, 0.0),
    ]
    faces_1 = [(0, 1, 2)]

    # Second mesh with vertices very close but not exactly the same
    vertices_2 = [
        (1.0001, 0.0, 0.0),  # Very close to (1.0, 0.0, 0.0)
        (2.0, 0.0, 0.0),
        (1.5, 1.0, 0.0),
    ]
    faces_2 = [(0, 1, 2)]

    # Test with tight tolerance - should not merge
    tight_tolerance = 1e-6
    merged_vertices_tight, merged_faces_tight = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tight_tolerance
    )

    # Should have 6 vertices (no merging)
    assert len(merged_vertices_tight) == 6

    # Test with loose tolerance - should merge
    loose_tolerance = 1e-3
    merged_vertices_loose, merged_faces_loose = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, loose_tolerance
    )

    # Should have 5 vertices (one merged)
    assert len(merged_vertices_loose) == 5


def test_merge_meshes_empty_meshes():
    """Test merging with empty meshes."""
    # Test with first mesh empty
    vertices_1 = []
    faces_1 = []
    vertices_2 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    faces_2 = [(0, 1, 2)]

    tolerance = 1e-6
    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should equal second mesh
    assert len(merged_vertices) == 3
    assert len(merged_faces) == 1

    # Test with second mesh empty
    vertices_1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
    faces_1 = [(0, 1, 2)]
    vertices_2 = []
    faces_2 = []

    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should equal first mesh
    assert len(merged_vertices) == 3
    assert len(merged_faces) == 1

    # Test with both meshes empty
    vertices_1 = []
    faces_1 = []
    vertices_2 = []
    faces_2 = []

    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should be empty
    assert len(merged_vertices) == 0
    assert len(merged_faces) == 0


def test_merge_meshes_vertex_order_preservation():
    """Test that vertex order in faces is preserved correctly."""
    # Create two simple triangles
    vertices_1 = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]
    faces_1 = [(0, 1, 2)]  # Counter-clockwise

    vertices_2 = [
        (2.0, 0.0, 0.0),  # 0
        (3.0, 0.0, 0.0),  # 1
        (2.5, 1.0, 0.0),  # 2
    ]
    faces_2 = [(0, 2, 1)]  # Clockwise

    tolerance = 1e-6
    merged_vertices, merged_faces = merge_meshes(
        vertices_1, faces_1, vertices_2, faces_2, tolerance
    )

    # Should have 6 vertices and 2 faces
    assert len(merged_vertices) == 6
    assert len(merged_faces) == 2

    # Check that face vertex orders are maintained
    # (exact indices will depend on implementation, but structure should be preserved)
    for face in merged_faces:
        assert len(face) == 3
        assert len(set(face)) == 3  # No duplicate vertices in face


def test_detect_twisted_vertices_no_twist():
    """Test twisted vertex detection with properly aligned vertices."""
    # Square cross-sections that are aligned (no twist)
    vertices_start = np.array(
        [
            [0.0, 0.0, 0.0],  # Bottom left
            [1.0, 0.0, 0.0],  # Bottom right
            [1.0, 1.0, 0.0],  # Top right
            [0.0, 1.0, 0.0],  # Top left
        ]
    )

    vertices_end = np.array(
        [
            [0.0, 0.0, 1.0],  # Bottom left
            [1.0, 0.0, 1.0],  # Bottom right
            [1.0, 1.0, 1.0],  # Top right
            [0.0, 1.0, 1.0],  # Top left
        ]
    )

    result = detect_twisted_vertices(vertices_start, vertices_end)

    assert not result["is_twisted"]
    assert result["best_rotation"] == 0
    assert len(result["distances"]) == 4
    assert result["best_distance"] == 4.0  # Sum of 4 distances of 1.0 each


def test_detect_twisted_vertices_with_twist():
    """Test twisted vertex detection with rotated vertices (180 degree twist)."""
    # Square cross-sections where end is rotated by 180 degrees
    vertices_start = np.array(
        [
            [0.0, 0.0, 0.0],  # Bottom left
            [1.0, 0.0, 0.0],  # Bottom right
            [1.0, 1.0, 0.0],  # Top right
            [0.0, 1.0, 0.0],  # Top left
        ]
    )

    vertices_end = np.array(
        [
            [1.0, 1.0, 1.0],  # Top right (rotated by 2 positions)
            [0.0, 1.0, 1.0],  # Top left
            [0.0, 0.0, 1.0],  # Bottom left
            [1.0, 0.0, 1.0],  # Bottom right
        ]
    )

    result = detect_twisted_vertices(vertices_start, vertices_end)

    assert result["is_twisted"]
    assert result["best_rotation"] == 2  # 180 degree rotation (2 out of 4 positions)
    assert len(result["distances"]) == 4
    # Best distance should be 4.0 (sum of 4 perfect matches)
    assert result["best_distance"] == 4.0


def test_detect_twisted_vertices_edge_cases():
    """Test edge cases for twisted vertex detection."""
    # Empty vertices
    result = detect_twisted_vertices([], [])
    assert not result["is_twisted"]
    assert result["best_rotation"] == 0
    assert result["distances"] == []
    assert result["best_distance"] == 0.0

    # Single vertex
    vertices_start = np.array([[0.0, 0.0, 0.0]])
    vertices_end = np.array([[0.0, 0.0, 1.0]])

    result = detect_twisted_vertices(vertices_start, vertices_end)
    assert not result["is_twisted"]
    assert result["best_rotation"] == 0
    assert len(result["distances"]) == 1
    assert result["best_distance"] == 1.0


def test_fix_twisted_vertex_correspondence():
    """Test fixing twisted vertex correspondence."""
    # Original vertices
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [1.0, 0.0, 0.0],  # B
            [1.0, 1.0, 0.0],  # C
            [0.0, 1.0, 0.0],  # D
        ]
    )

    # Test no rotation
    fixed = fix_twisted_vertex_correspondence(vertices, 0)
    np.testing.assert_array_equal(fixed, vertices)

    # Test rotation by 2 positions (180 degrees)
    fixed = fix_twisted_vertex_correspondence(vertices, 2)
    expected = np.array(
        [
            [1.0, 1.0, 0.0],  # C
            [0.0, 1.0, 0.0],  # D
            [0.0, 0.0, 0.0],  # A
            [1.0, 0.0, 0.0],  # B
        ]
    )
    np.testing.assert_array_equal(fixed, expected)

    # Test rotation by 1 position
    fixed = fix_twisted_vertex_correspondence(vertices, 1)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # B
            [1.0, 1.0, 0.0],  # C
            [0.0, 1.0, 0.0],  # D
            [0.0, 0.0, 0.0],  # A
        ]
    )
    np.testing.assert_array_equal(fixed, expected)


def test_validate_and_fix_mesh_segment_no_twist():
    """Test validate_and_fix_mesh_segment with no twist needed."""
    vertices_start = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    vertices_end = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    corrected_start, corrected_end, twist_info = validate_and_fix_mesh_segment(
        vertices_start, vertices_end
    )

    # No changes should be made
    np.testing.assert_array_equal(corrected_start, vertices_start)
    np.testing.assert_array_equal(corrected_end, vertices_end)
    assert not twist_info["is_twisted"]


def test_validate_and_fix_mesh_segment_with_twist():
    """Test validate_and_fix_mesh_segment with twist that needs fixing."""
    vertices_start = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [1.0, 0.0, 0.0],  # B
            [1.0, 1.0, 0.0],  # C
            [0.0, 1.0, 0.0],  # D
        ]
    )

    # End vertices rotated by 180 degrees (2 positions)
    vertices_end = np.array(
        [
            [1.0, 1.0, 1.0],  # C (rotated)
            [0.0, 1.0, 1.0],  # D
            [0.0, 0.0, 1.0],  # A
            [1.0, 0.0, 1.0],  # B
        ]
    )

    corrected_start, corrected_end, twist_info = validate_and_fix_mesh_segment(
        vertices_start, vertices_end
    )

    # Start should be unchanged
    np.testing.assert_array_equal(corrected_start, vertices_start)

    # End should be corrected to match start
    expected_corrected_end = np.array(
        [
            [0.0, 0.0, 1.0],  # A
            [1.0, 0.0, 1.0],  # B
            [1.0, 1.0, 1.0],  # C
            [0.0, 1.0, 1.0],  # D
        ]
    )
    np.testing.assert_array_equal(corrected_end, expected_corrected_end)

    # Twist should be detected
    assert twist_info["is_twisted"]
    assert twist_info["best_rotation"] == 2


def test_create_cube_geometry():
    """Test create_cube_geometry function."""
    verts, faces = create_cube_geometry(radius=1.0)

    # Should have 8 vertices and 12 triangular faces
    assert verts.shape == (8, 3)
    assert faces.shape == (12, 3)

    # All vertices should be on sphere of radius 1.0
    distances = np.linalg.norm(verts, axis=1)
    np.testing.assert_allclose(distances, 1.0, rtol=1e-10)

    # Check that faces use valid vertex indices
    assert np.all(faces >= 0)
    assert np.all(faces < 8)

    # Test custom radius
    verts_r2, faces_r2 = create_cube_geometry(radius=2.0)
    distances_r2 = np.linalg.norm(verts_r2, axis=1)
    np.testing.assert_allclose(distances_r2, 2.0, rtol=1e-10)

    # Shape should remain the same
    assert verts_r2.shape == (8, 3)
    assert faces_r2.shape == (12, 3)


def test_create_dodecahedron_geometry():
    """Test create_dodecahedron_geometry function."""
    verts, faces = create_dodecahedron_geometry(radius=1.0)

    # Should have 20 vertices and 12 pentagonal faces
    assert verts.shape == (20, 3)
    assert faces.shape == (12, 5)

    # All vertices should be on sphere of radius 1.0
    distances = np.linalg.norm(verts, axis=1)
    np.testing.assert_allclose(distances, 1.0, rtol=1e-10)

    # Check that faces use valid vertex indices
    assert np.all(faces >= 0)
    assert np.all(faces < 20)

    # Test custom radius
    verts_r3, faces_r3 = create_dodecahedron_geometry(radius=3.0)
    distances_r3 = np.linalg.norm(verts_r3, axis=1)
    np.testing.assert_allclose(distances_r3, 3.0, rtol=1e-10)


def test_create_tetrahedron_geometry():
    """Test create_tetrahedron_geometry function."""
    verts, faces = create_tetrahedron_geometry(radius=1.0)

    # Should have 4 vertices and 4 triangular faces
    assert verts.shape == (4, 3)
    assert faces.shape == (4, 3)

    # All vertices should be on sphere of radius 1.0
    distances = np.linalg.norm(verts, axis=1)
    np.testing.assert_allclose(distances, 1.0, rtol=1e-10)

    # Check that faces use valid vertex indices
    assert np.all(faces >= 0)
    assert np.all(faces < 4)

    # Test custom radius
    verts_r05, faces_r05 = create_tetrahedron_geometry(radius=0.5)
    distances_r05 = np.linalg.norm(verts_r05, axis=1)
    np.testing.assert_allclose(distances_r05, 0.5, rtol=1e-10)


def test_create_fibonacci_sphere_geometry():
    """Test create_fibonacci_sphere_geometry function."""
    # Test with default parameters
    verts, faces = create_fibonacci_sphere_geometry(radius=1.0, samples=100)

    # Should have 100 vertices (sample count)
    assert verts.shape[0] == 100
    assert verts.shape[1] == 3

    # All vertices should be on sphere of radius 1.0
    distances = np.linalg.norm(verts, axis=1)
    np.testing.assert_allclose(distances, 1.0, rtol=1e-10)

    # Should have triangular faces
    assert faces.shape[1] == 3
    assert faces.shape[0] > 0  # Should have some faces from convex hull

    # Check that faces use valid vertex indices
    assert np.all(faces >= 0)
    assert np.all(faces < 100)

    # Test with fewer samples
    verts_small, faces_small = create_fibonacci_sphere_geometry(radius=2.0, samples=20)
    assert verts_small.shape == (20, 3)
    distances_small = np.linalg.norm(verts_small, axis=1)
    np.testing.assert_allclose(distances_small, 2.0, rtol=1e-10)

    # Test with different radius
    verts_r5, faces_r5 = create_fibonacci_sphere_geometry(radius=5.0, samples=50)
    distances_r5 = np.linalg.norm(verts_r5, axis=1)
    np.testing.assert_allclose(distances_r5, 5.0, rtol=1e-10)


def test_partitionable_spheroid_triangle_mesh_basic():
    """Test basic PartitionableSpheroidTriangleMesh functionality."""
    # Create a simple tetrahedron mesh
    verts, faces = create_tetrahedron_geometry(radius=1.0)

    # Convert to list format expected by the class
    vertices_list = [list(v) for v in verts]
    faces_list = [list(f) for f in faces]
    vertex_labels = [f"v{i}" for i in range(len(vertices_list))]

    # Create mesh instance
    mesh = PartitionableSpheroidTriangleMesh(
        vertices=vertices_list, faces=faces_list, vertex_labels=vertex_labels
    )

    # Test basic properties
    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 4
    assert len(mesh.vertex_labels) == 4

    # Test that vertices are preserved
    for i, vertex in enumerate(mesh.vertices):
        np.testing.assert_allclose(vertex, vertices_list[i], rtol=1e-10)


def test_partitionable_spheroid_triangle_mesh_from_point_cloud():
    """Test PartitionableSpheroidTriangleMesh creation from point cloud."""
    # Create a simple cube point cloud
    verts, _ = create_cube_geometry(radius=1.0)

    # Convert to expected format
    points = [list(v) for v in verts]

    # Create mesh from point cloud
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    # Should have 8 vertices
    assert len(mesh.vertices) == 8
    assert len(mesh.vertex_labels) == 8

    # Should have some triangles from convex hull
    assert len(mesh.faces) > 0

    # All vertices should be preserved
    mesh_verts = np.array(mesh.vertices)
    orig_verts = np.array(points)

    # Should contain all original points (order may differ)
    for orig_vert in orig_verts:
        distances = np.linalg.norm(mesh_verts - orig_vert, axis=1)
        assert np.min(distances) < 1e-10  # At least one vertex should match closely


def test_partitionable_spheroid_triangle_mesh_with_fibonacci_sphere():
    """Test PartitionableSpheroidTriangleMesh with fibonacci sphere geometry."""
    # Create fibonacci sphere
    verts, faces = create_fibonacci_sphere_geometry(radius=2.0, samples=50)

    # Convert to expected format
    vertices_list = [list(v) for v in verts]
    faces_list = [list(f) for f in faces]
    vertex_labels = [f"fib_{i}" for i in range(len(vertices_list))]

    # Create mesh
    mesh = PartitionableSpheroidTriangleMesh(
        vertices=vertices_list, faces=faces_list, vertex_labels=vertex_labels
    )

    # Verify properties
    assert len(mesh.vertices) == 50
    assert len(mesh.faces) == len(faces_list)
    assert len(mesh.vertex_labels) == 50

    # All vertices should maintain their distance from origin
    for vertex in mesh.vertices:
        distance = np.linalg.norm(vertex)
        assert abs(distance - 2.0) < 1e-10
