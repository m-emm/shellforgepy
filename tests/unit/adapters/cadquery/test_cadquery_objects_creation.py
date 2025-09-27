import numpy as np
import pytest

# Try to import cadquery and skip tests if not available
try:
    import cadquery as cq

    cadquery_available = True
except ImportError:
    cadquery_available = False
    cq = None

from shellforgepy.construct.alignment import Alignment
from shellforgepy.geometry.mesh_builders import create_dodecahedron_geometry
from shellforgepy.simple import (
    create_basic_box,
    create_basic_cone,
    create_basic_cylinder,
    create_basic_sphere,
    create_filleted_box,
    create_solid_from_traditional_face_vertex_maps,
)


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_create_basic_box():
    """Test creation of a basic box in CadQuery."""
    box = create_basic_box(10, 20, 30)
    assert box is not None
    bbox = box.BoundingBox()
    assert abs(bbox.xlen - 10) < 1e-10
    assert abs(bbox.ylen - 20) < 1e-10
    assert abs(bbox.zlen - 30) < 1e-10


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_create_dodecahedron():
    """Test creation of a dodecahedron from face-vertex maps."""
    vertices, faces = create_dodecahedron_geometry(radius=10.0)

    # Calculate the actual edge length from the geometry
    face = faces[0]  # Use first face to measure edge length
    v1 = vertices[face[0]]
    v2 = vertices[face[1]]
    actual_edge_length = np.linalg.norm(v2 - v1)

    # Volume of regular dodecahedron with edge length a: V = (15 + 7*sqrt(5))/4 * a^3
    expected_volume = (15 + 7 * np.sqrt(5)) / 4 * actual_edge_length**3

    mesh = {
        "vertexes": {i: tuple(v) for i, v in enumerate(vertices)},
        "faces": {i: tuple(f) for i, f in enumerate(faces)},
    }

    solid = create_solid_from_traditional_face_vertex_maps(mesh)

    # CadQuery volume calculation
    actual_volume = solid.Volume()
    assert np.allclose(
        actual_volume, expected_volume, rtol=1e-10
    ), f"Expected volume {expected_volume}, got {actual_volume}"
    assert solid.isValid(), "Dodecahedron solid should be valid"


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_create_basic_shapes():
    """Test creation of basic geometric shapes."""
    # Test cylinder
    cylinder = create_basic_cylinder(radius=5, height=10)
    assert cylinder is not None
    # Cylinder volume = π * r² * h
    expected_cyl_volume = np.pi * 5**2 * 10
    assert np.allclose(cylinder.Volume(), expected_cyl_volume, rtol=1e-5)

    # Test sphere
    sphere = create_basic_sphere(radius=5)
    assert sphere is not None
    # Sphere volume = (4/3) * π * r³
    expected_sphere_volume = (4 / 3) * np.pi * 5**3
    assert np.allclose(sphere.Volume(), expected_sphere_volume, rtol=1e-5)

    # Test cone
    cone = create_basic_cone(radius1=5, radius2=2, height=10)
    assert cone is not None
    # Truncated cone volume = (1/3) * π * h * (r1² + r1*r2 + r2²)
    expected_cone_volume = (1 / 3) * np.pi * 10 * (5**2 + 5 * 2 + 2**2)
    assert np.allclose(cone.Volume(), expected_cone_volume, rtol=1e-5)


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_bounding_box_functions():
    """Test bounding box utility functions."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        get_bounding_box,
        get_bounding_box_center,
        get_bounding_box_max,
        get_bounding_box_min,
        get_bounding_box_size,
        get_z_max,
        get_z_min,
    )

    # Create a box offset from origin
    box = create_basic_box(10, 20, 30, (5, 10, 15))

    # Test bounding box
    min_point, max_point = get_bounding_box(box)
    assert min_point == (5, 10, 15)
    assert max_point == (15, 30, 45)

    # Test center
    center = get_bounding_box_center(box)
    assert center == (10, 20, 30)  # midpoint of each dimension

    # Test size
    size = get_bounding_box_size(box)
    assert size == (10, 20, 30)

    # Test min/max
    assert get_bounding_box_min(box) == (5, 10, 15)
    assert get_bounding_box_max(box) == (15, 30, 45)

    # Test Z functions
    assert get_z_min(box) == 15
    assert get_z_max(box) == 45


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_vertex_functions():
    """Test vertex extraction functions."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        get_vertex_coordinates,
        get_vertex_coordinates_np,
        get_vertices,
    )

    # Create a simple tetrahedron using face-vertex maps
    # All faces oriented counter-clockwise when viewed from outside
    vertices = [
        (0, 0, 0),  # vertex 0
        (1, 0, 0),  # vertex 1
        (0, 1, 0),  # vertex 2
        (0, 0, 1),  # vertex 3
    ]
    faces = [
        [0, 2, 1],  # bottom face (CCW from below)
        [0, 1, 3],  # front face (CCW from outside)
        [0, 3, 2],  # left face (CCW from outside)
        [1, 2, 3],  # right face (CCW from outside)
    ]

    mesh = {
        "vertexes": {i: vertices[i] for i in range(len(vertices))},
        "faces": {i: faces[i] for i in range(len(faces))},
    }

    solid = create_solid_from_traditional_face_vertex_maps(mesh)

    # Test vertex extraction
    extracted_vertices = get_vertices(solid)
    assert len(extracted_vertices) == 4

    # Test coordinate extraction
    coords = get_vertex_coordinates(solid)
    assert len(coords) == 4

    # Test numpy coordinate extraction
    coords_np = get_vertex_coordinates_np(solid)
    assert coords_np.shape == (4, 3)

    # Verify coordinates are approximately correct (may be reordered)
    original_coords = set(vertices)
    extracted_coords = set(tuple(c) for c in coords)

    # Allow for floating point tolerance
    for orig in original_coords:
        found = False
        for ext in extracted_coords:
            if np.allclose(orig, ext, atol=1e-6):
                found = True
                break
        assert found, f"Original coordinate {orig} not found in extracted coordinates"


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_rotation_mechanics():
    """Test rotation mechanics to understand how rotate_part works."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import rotate_part

    # Create a test box and examine rotation behavior
    box = create_basic_box(
        10, 10, 10
    )  # Box from (0,0,0) to (10,10,10), center at (5,5,5)

    original_bbox = box.BoundingBox()
    original_center = original_bbox.center
    assert (
        abs(original_center.x - 5) < 1e-10
    ), f"Expected center x=5, got {original_center.x}"
    assert (
        abs(original_center.y - 5) < 1e-10
    ), f"Expected center y=5, got {original_center.y}"
    assert (
        abs(original_center.z - 5) < 1e-10
    ), f"Expected center z=5, got {original_center.z}"

    # Test 90° rotation around Z-axis at origin (0,0,0)
    # Point (5,5) rotated 90° around origin should become (-5,5)
    rotated = rotate_part(box, 90, center=(0, 0, 0), axis=(0, 0, 1))
    rotated_center = rotated.BoundingBox().center

    # Document actual behavior for future reference
    print(
        f"Original center: ({original_center.x}, {original_center.y}, {original_center.z})"
    )
    print(
        f"Rotated center: ({rotated_center.x}, {rotated_center.y}, {rotated_center.z})"
    )

    # Let's test different angles to understand the pattern
    # Test 0° rotation - should be unchanged
    rotated_0 = rotate_part(box, 0, center=(0, 0, 0), axis=(0, 0, 1))
    center_0 = rotated_0.BoundingBox().center
    print(f"0° rotation center: ({center_0.x}, {center_0.y}, {center_0.z})")

    # Test 45° rotation
    rotated_45 = rotate_part(box, 45, center=(0, 0, 0), axis=(0, 0, 1))
    center_45 = rotated_45.BoundingBox().center
    print(f"45° rotation center: ({center_45.x}, {center_45.y}, {center_45.z})")

    # Test 180° rotation
    rotated_180 = rotate_part(box, 180, center=(0, 0, 0), axis=(0, 0, 1))
    center_180 = rotated_180.BoundingBox().center
    print(f"180° rotation center: ({center_180.x}, {center_180.y}, {center_180.z})")

    # For now, let's just document the actual behavior rather than assuming theoretical
    # Since we measured (5,5,5) -> (4.86..., 5.13..., 5.0) for 90°
    # Let's verify this is consistent
    tolerance = 1e-5

    # Verify 0° gives original position
    assert (
        abs(center_0.x - 5.0) < tolerance
    ), f"0° rotation should preserve x=5, got {center_0.x}"
    assert (
        abs(center_0.y - 5.0) < tolerance
    ), f"0° rotation should preserve y=5, got {center_0.y}"
    assert (
        abs(center_0.z - 5.0) < tolerance
    ), f"0° rotation should preserve z=5, got {center_0.z}"

    # After fixing the radians/degrees bug, verify correct mathematical rotation
    # (5,5,5) rotated 90° around (0,0,0) should give (-5,5,5)
    expected_x, expected_y, expected_z = -5.0, 5.0, 5.0
    assert (
        abs(rotated_center.x - expected_x) < tolerance
    ), f"Expected x={expected_x}, got {rotated_center.x}"
    assert (
        abs(rotated_center.y - expected_y) < tolerance
    ), f"Expected y={expected_y}, got {rotated_center.y}"
    assert (
        abs(rotated_center.z - expected_z) < tolerance
    ), f"Expected z={expected_z}, got {rotated_center.z}"

    # Test rotation around object's own center - should stay in place
    rotated_around_self = rotate_part(box, 90, center=(5, 5, 5), axis=(0, 0, 1))
    self_rotated_center = rotated_around_self.BoundingBox().center

    assert (
        abs(self_rotated_center.x - 5) < tolerance
    ), f"Self-rotation should keep center at x=5, got {self_rotated_center.x}"
    assert (
        abs(self_rotated_center.y - 5) < tolerance
    ), f"Self-rotation should keep center at y=5, got {self_rotated_center.y}"
    assert (
        abs(self_rotated_center.z - 5) < tolerance
    ), f"Self-rotation should keep center at z=5, got {self_rotated_center.z}"


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_rotation_angle_units():
    """Test that our CadQuery adapter rotation gives correct mathematical results."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import rotate_part

    # Create a test box
    box = create_basic_box(10, 10, 10)  # Center at (5,5,5)

    # Test our wrapper with different angles to verify correctness
    # Mathematical expectation for 90° rotation of (5,5,5) around origin: (-5,5,5)
    result_90 = rotate_part(box, 90, center=(0, 0, 0), axis=(0, 0, 1))
    center_90 = result_90.BoundingBox().center
    print(f"90° rotation center: ({center_90.x}, {center_90.y}, {center_90.z})")

    # Test 180° rotation: (5,5,5) -> (-5,-5,5)
    result_180 = rotate_part(box, 180, center=(0, 0, 0), axis=(0, 0, 1))
    center_180 = result_180.BoundingBox().center
    print(f"180° rotation center: ({center_180.x}, {center_180.y}, {center_180.z})")

    # Test 270° rotation: (5,5,5) -> (5,-5,5)
    result_270 = rotate_part(box, 270, center=(0, 0, 0), axis=(0, 0, 1))
    center_270 = result_270.BoundingBox().center
    print(f"270° rotation center: ({center_270.x}, {center_270.y}, {center_270.z})")

    # Verify mathematical correctness
    tolerance = 1e-10

    # 90° rotation: (5,5) -> (-5,5)
    assert (
        abs(center_90.x + 5) < tolerance
    ), f"90° rotation x: expected -5, got {center_90.x}"
    assert (
        abs(center_90.y - 5) < tolerance
    ), f"90° rotation y: expected 5, got {center_90.y}"
    assert (
        abs(center_90.z - 5) < tolerance
    ), f"90° rotation z: expected 5, got {center_90.z}"

    # 180° rotation: (5,5) -> (-5,-5)
    assert (
        abs(center_180.x + 5) < tolerance
    ), f"180° rotation x: expected -5, got {center_180.x}"
    assert (
        abs(center_180.y + 5) < tolerance
    ), f"180° rotation y: expected -5, got {center_180.y}"
    assert (
        abs(center_180.z - 5) < tolerance
    ), f"180° rotation z: expected 5, got {center_180.z}"

    # 270° rotation: (5,5) -> (5,-5)
    assert (
        abs(center_270.x - 5) < tolerance
    ), f"270° rotation x: expected 5, got {center_270.x}"
    assert (
        abs(center_270.y + 5) < tolerance
    ), f"270° rotation y: expected -5, got {center_270.y}"
    assert (
        abs(center_270.z - 5) < tolerance
    ), f"270° rotation z: expected 5, got {center_270.z}"

    print("✓ All CadQuery adapter rotations give correct mathematical results")


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_part_manipulation_functions():
    """Test part manipulation functions."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        copy_part,
        cut_parts,
        fuse_parts,
        rotate_part,
        translate_part,
    )

    # Create test parts
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, (20, 0, 0))

    # Test copy
    box1_copy = copy_part(box1)
    assert box1_copy is not box1
    assert box1_copy.Volume() == box1.Volume()

    # Test translate
    translated = translate_part(box1, (10, 20, 30))
    original_center = box1.BoundingBox().center
    translated_center = translated.BoundingBox().center

    assert abs(translated_center.x - original_center.x - 10) < 1e-6
    assert abs(translated_center.y - original_center.y - 20) < 1e-6
    assert abs(translated_center.z - original_center.z - 30) < 1e-6

    # Test rotate (90 degrees around Z axis)
    # Box center is at (5,5,5), rotating around origin (0,0,0)
    # Point (5,5) rotated 90° around origin becomes (-5,5)
    # So new center should be (-5,5,5)
    rotated = rotate_part(box1, 90, center=(0, 0, 0), axis=(0, 0, 1))
    rotated_center = rotated.BoundingBox().center
    tolerance = 1e-10  # Use tight tolerance for numerical precision

    assert (
        abs(rotated_center.x + 5) < tolerance
    ), f"Expected x=-5, got {rotated_center.x}"
    assert (
        abs(rotated_center.y - 5) < tolerance
    ), f"Expected y=5, got {rotated_center.y}"
    assert (
        abs(rotated_center.z - 5) < tolerance
    ), f"Expected z=5, got {rotated_center.z}"

    # Test fuse
    fused = fuse_parts(box1, box2)
    # Fused volume should be sum of both boxes (they don't overlap)
    expected_volume = box1.Volume() + box2.Volume()
    assert np.allclose(fused.Volume(), expected_volume, rtol=1e-5)

    # Test cut
    # Create two overlapping boxes to test cutting
    box3 = create_basic_box(10, 10, 10)  # Box at origin
    box4 = create_basic_box(
        5, 5, 15, (2.5, 2.5, 0)
    )  # Smaller box overlapping with box3

    original_volume = box3.Volume()
    cut_result = cut_parts(box3, box4)

    # After cutting, the volume should be less than the original
    assert cut_result.Volume() < original_volume
    # Should be positive (not everything was cut away)
    assert cut_result.Volume() > 0


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_export_stl():
    """Test STL export functionality."""
    import os
    import tempfile

    from shellforgepy.adapters.cadquery.cadquery_adapter import export_solid_to_stl

    # Create a test solid
    box = create_basic_box(10, 10, 10)

    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_solid_to_stl(box, tmp_path)

        # Verify file was created and has content
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Basic check that it's an STL file (starts with "solid" for ASCII STL)
        with open(tmp_path, "rb") as f:
            header = f.read(80)  # STL header is 80 bytes
            assert len(header) > 0  # File has content

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_polygon_creation():
    from shellforgepy.adapters.cadquery.cadquery_adapter import create_extruded_polygon

    poly = create_extruded_polygon(
        points=[(0, 0), (10, 0), (10, 10), (0, 10)], thickness=5
    )
    assert poly is not None


@pytest.mark.skipif(not cadquery_available, reason="cadquery not available")
def test_filleted_box_creation():
    """Test creation of filleted boxes with different configurations."""

    # Test 1: Basic filleted box (all edges filleted)
    filleted_box = create_filleted_box(10, 10, 10, 2)
    assert filleted_box is not None

    # Check that the box has the correct dimensions
    bbox = filleted_box.BoundingBox()
    assert abs(bbox.xlen - 10) < 1e-10
    assert abs(bbox.ylen - 10) < 1e-10
    assert abs(bbox.zlen - 10) < 1e-10

    # Check that filleting was applied - a filleted box should have more edges
    # than the original 12 edges of a cube
    original_box = create_basic_box(10, 10, 10)
    original_edge_count = len(original_box.Edges())
    filleted_edge_count = len(filleted_box.Edges())

    # Filleted box should have more edges due to fillet surfaces
    assert (
        filleted_edge_count > original_edge_count
    ), f"Filleted box should have more edges. Original: {original_edge_count}, Filleted: {filleted_edge_count}"

    # Test 2: Box with fillets only at top
    top_filleted_box = create_filleted_box(10, 10, 10, 1, fillets_at=[Alignment.TOP])
    assert top_filleted_box is not None

    # Should still have correct dimensions
    bbox2 = top_filleted_box.BoundingBox()
    assert abs(bbox2.xlen - 10) < 1e-10
    assert abs(bbox2.ylen - 10) < 1e-10
    assert abs(bbox2.zlen - 10) < 1e-10

    # Should have fewer additional edges than fully filleted box
    top_filleted_edge_count = len(top_filleted_box.Edges())
    assert (
        top_filleted_edge_count > original_edge_count
    ), "Top-filleted box should have more edges than original"
    assert (
        top_filleted_edge_count < filleted_edge_count
    ), "Top-filleted box should have fewer edges than fully filleted box"

    # Test 3: Box with no fillets at bottom
    no_bottom_filleted_box = create_filleted_box(
        10, 10, 10, 1, no_fillets_at=[Alignment.BOTTOM]
    )
    assert no_bottom_filleted_box is not None

    # Should still have correct dimensions
    bbox3 = no_bottom_filleted_box.BoundingBox()
    assert abs(bbox3.xlen - 10) < 1e-10
    assert abs(bbox3.ylen - 10) < 1e-10
    assert abs(bbox3.zlen - 10) < 1e-10

    # Should have more edges than original but may be different from top-only
    no_bottom_edge_count = len(no_bottom_filleted_box.Edges())
    assert no_bottom_edge_count > original_edge_count

    # Test 4: Box with fillets at multiple specific locations
    multi_filleted_box = create_filleted_box(
        10, 10, 10, 1, fillets_at=[Alignment.TOP, Alignment.FRONT]
    )
    assert multi_filleted_box is not None

    multi_edge_count = len(multi_filleted_box.Edges())
    assert multi_edge_count > original_edge_count

    # Test 5: Box with no fillets (empty fillets_at list)
    no_fillet_box = create_filleted_box(10, 10, 10, 1, fillets_at=[])
    assert no_fillet_box is not None

    # This should be identical to a basic box
    basic_box = create_basic_box(10, 10, 10)

    # Both should have same bounding box
    bbox_no_fillet = no_fillet_box.BoundingBox()
    bbox_basic = basic_box.BoundingBox()
    assert abs(bbox_no_fillet.xlen - bbox_basic.xlen) < 1e-10
    assert abs(bbox_no_fillet.ylen - bbox_basic.ylen) < 1e-10
    assert abs(bbox_no_fillet.zlen - bbox_basic.zlen) < 1e-10

    # Should have same number of edges as original box
    no_fillet_edge_count = len(no_fillet_box.Edges())
    assert (
        no_fillet_edge_count == original_edge_count
    ), f"No-fillet box should have same edges as original. Got {no_fillet_edge_count}, expected {original_edge_count}"

    # Test 6: Verify different fillet radius works
    large_fillet_box = create_filleted_box(20, 20, 20, 5)
    assert large_fillet_box is not None

    bbox_large = large_fillet_box.BoundingBox()
    assert abs(bbox_large.xlen - 20) < 1e-10
    assert abs(bbox_large.ylen - 20) < 1e-10
    assert abs(bbox_large.zlen - 20) < 1e-10

    # Large fillet should also create additional edges
    large_fillet_edge_count = len(large_fillet_box.Edges())
    large_original_edge_count = len(create_basic_box(20, 20, 20).Edges())
    assert large_fillet_edge_count > large_original_edge_count
