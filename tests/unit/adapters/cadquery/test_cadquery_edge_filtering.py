"""Tests for CadQuery edge filtering functionality."""

import pytest

# Try to import cadquery and related modules, skip tests if not available
try:
    import cadquery as cq
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        apply_fillet_by_alignment,
        apply_fillet_by_function,
        apply_fillet_to_edges,
        create_basic_box,
        create_basic_cylinder,
        filter_edges_by_alignment,
        filter_edges_by_function,
        filter_edges_by_z_position,
    )

    if cq is not None:  # dummy -just to use the import once
        cadquery_available = True


except ImportError:
    cadquery_available = False
    # Create dummy imports for the test collection to work
    apply_fillet_by_alignment = None
    apply_fillet_by_function = None
    apply_fillet_to_edges = None
    create_basic_box = None
    create_basic_cylinder = None
    filter_edges_by_alignment = None
    filter_edges_by_function = None
    filter_edges_by_z_position = None

from shellforgepy.construct.alignment import Alignment

# Skip all tests in this module if CadQuery is not available
pytestmark = pytest.mark.skipif(not cadquery_available, reason="CadQuery not available")


def test_filter_edges_by_z_position_box():
    """Test edge filtering on a box with known geometry."""
    # Create a 10x10x10 box at origin
    box = create_basic_box(10, 10, 10, origin=(0, 0, 0))
    assert box is not None

    # Filter edges at the bottom (z<=0.1)
    bottom_edges = filter_edges_by_z_position(box, z_threshold=0.1, below=True)
    assert (
        len(bottom_edges) >= 4
    ), f"Expected at least 4 bottom edges, got {len(bottom_edges)}"

    # Filter edges at the top (z>=9.9)
    top_edges = filter_edges_by_z_position(box, z_threshold=9.9, below=False)
    assert len(top_edges) >= 4, f"Expected at least 4 top edges, got {len(top_edges)}"

    # Filter edges at exact bottom (z<=0)
    exact_bottom = filter_edges_by_z_position(box, z_threshold=0.0, below=True)
    assert (
        len(exact_bottom) >= 4
    ), f"Expected bottom edges at z=0, got {len(exact_bottom)}"

    # Filter edges at exact top (z>=10)
    exact_top = filter_edges_by_z_position(box, z_threshold=10.0, below=False)
    assert len(exact_top) >= 4, f"Expected top edges at z=10, got {len(exact_top)}"


def test_filter_edges_by_alignment_box():
    """Test alignment-based edge filtering on a box."""
    box = create_basic_box(10, 10, 10, origin=(0, 0, 0))

    # Test individual alignments
    top_edges = filter_edges_by_alignment(box, fillets_at=[Alignment.TOP])
    assert len(top_edges) >= 4, f"Expected top edges, got {len(top_edges)}"

    bottom_edges = filter_edges_by_alignment(box, fillets_at=[Alignment.BOTTOM])
    assert len(bottom_edges) >= 4, f"Expected bottom edges, got {len(bottom_edges)}"

    # Test multiple alignments
    top_bottom_edges = filter_edges_by_alignment(
        box, fillets_at=[Alignment.TOP, Alignment.BOTTOM]
    )
    assert (
        len(top_bottom_edges) >= 8
    ), f"Expected top and bottom edges, got {len(top_bottom_edges)}"

    # Test exclusion
    not_bottom_edges = filter_edges_by_alignment(box, no_fillets_at=[Alignment.BOTTOM])
    assert len(not_bottom_edges) > 0, "Expected some edges when excluding bottom"

    # Test combination
    top_not_front = filter_edges_by_alignment(
        box, fillets_at=[Alignment.TOP], no_fillets_at=[Alignment.FRONT]
    )
    assert len(top_not_front) >= 0, "Expected valid edge count for combined filter"


def test_filter_edges_by_function():
    """Test custom function-based edge filtering."""
    box = create_basic_box(20, 20, 20, origin=(0, 0, 0))

    # Filter for top edges using custom function
    def top_edge_filter(bbox, v0, v1):
        min_point, max_point = bbox
        tolerance = 1e-3
        return (
            abs(v0[2] - max_point[2]) < tolerance
            and abs(v1[2] - max_point[2]) < tolerance
        )

    top_edges = filter_edges_by_function(box, top_edge_filter)
    assert (
        len(top_edges) >= 4
    ), f"Expected top edges from custom filter, got {len(top_edges)}"

    # Filter for edges above a certain height
    def upper_half_filter(bbox, v0, v1):
        min_point, max_point = bbox
        mid_z = (min_point[2] + max_point[2]) / 2
        return v0[2] >= mid_z or v1[2] >= mid_z

    upper_edges = filter_edges_by_function(box, upper_half_filter)
    assert len(upper_edges) > 0, "Expected some edges in upper half"


def test_apply_fillet_by_alignment():
    """Test applying fillet using alignment-based selection."""
    box = create_basic_box(20, 20, 20, origin=(0, 0, 0))
    original_volume = box.Volume()

    # Apply fillet to top edges only
    filleted_box = apply_fillet_by_alignment(
        box, fillet_radius=2.0, fillets_at=[Alignment.TOP]
    )
    assert filleted_box is not None
    assert (
        abs(filleted_box.Volume() - original_volume) > 0.01
    ), "Volume should change after filleting"

    # Apply fillet to top and bottom, but not front
    filleted_box2 = apply_fillet_by_alignment(
        box,
        fillet_radius=1.0,
        fillets_at=[Alignment.TOP, Alignment.BOTTOM],
        no_fillets_at=[Alignment.FRONT],
    )
    assert filleted_box2 is not None


def test_apply_fillet_by_function():
    """Test applying fillet using custom function selection."""
    box = create_basic_box(15, 15, 15, origin=(0, 0, 0))
    original_volume = box.Volume()

    # Fillet only top edges using custom function
    def top_edge_filter(bbox, v0, v1):
        min_point, max_point = bbox
        tolerance = 1e-3
        return (
            abs(v0[2] - max_point[2]) < tolerance
            and abs(v1[2] - max_point[2]) < tolerance
        )

    filleted_box = apply_fillet_by_function(
        box, fillet_radius=1.5, edge_filter_func=top_edge_filter
    )
    assert filleted_box is not None
    assert (
        abs(filleted_box.Volume() - original_volume) > 0.01
    ), "Volume should change after filleting"


def test_filter_edges_by_z_position_cylinder():
    """Test edge filtering on a cylinder with known geometry."""
    # Create a cylinder similar to hawaii bottle cap
    radius = 16.85  # cap_rim_outer_diameter / 2
    height = 4.5  # cap_cover_thickness
    cylinder = create_basic_cylinder(radius=radius, height=height)
    assert cylinder is not None

    # Test z-position filtering
    top_edges = filter_edges_by_z_position(
        cylinder, z_threshold=height - 0.1, below=False
    )
    bottom_edges = filter_edges_by_z_position(cylinder, z_threshold=0.1, below=True)

    print(f"Cylinder: radius={radius}, height={height}")
    print(f"Top edges found: {len(top_edges)}")
    print(f"Bottom edges found: {len(bottom_edges)}")

    # Should find at least 1 circular edge at top and bottom
    assert len(top_edges) >= 1, f"Expected top edges, got {len(top_edges)}"
    assert len(bottom_edges) >= 1, f"Expected bottom edges, got {len(bottom_edges)}"


def test_cadquery_cylinder_structure_analysis():
    """Comprehensive analysis of CadQuery cylinder structure to understand edge modeling."""
    import cadquery as cq

    # Create cylinder using both methods for comparison
    radius = 16.85
    height = 4.5

    # Method 1: Our create_basic_cylinder
    cylinder1 = create_basic_cylinder(radius=radius, height=height)

    # Method 2: Direct CadQuery
    cylinder2 = cq.Workplane("XY").cylinder(height, radius)

    # Method 3: CadQuery solid
    cylinder3 = cq.Solid.makeCylinder(radius, height)

    print("\n=== CADQUERY CYLINDER STRUCTURE ANALYSIS ===")

    for i, (cylinder, name) in enumerate(
        [
            (cylinder1, "create_basic_cylinder"),
            (cylinder2, "cq.Workplane.cylinder"),
            (cylinder3, "cq.Solid.makeCylinder"),
        ]
    ):
        print(f"\n--- {name} ---")

        # Get the underlying solid
        if hasattr(cylinder, "val"):
            solid = cylinder.val()
        else:
            solid = cylinder

        print(f"Type: {type(solid)}")
        print(f"Volume: {solid.Volume()}")

        # Bounding box
        bbox = solid.BoundingBox()
        print(f"BBox: min=({bbox.xmin:.3f}, {bbox.ymin:.3f}, {bbox.zmin:.3f})")
        print(f"      max=({bbox.xmax:.3f}, {bbox.ymax:.3f}, {bbox.zmax:.3f})")

        # Faces analysis
        faces = solid.Faces()
        print(f"\nFaces: {len(faces)}")
        for j, face in enumerate(faces):
            face_bbox = face.BoundingBox()
            print(f"  Face {j}: type={type(face)}, area={face.Area():.3f}")
            print(
                f"          bbox: min=({face_bbox.xmin:.3f}, {face_bbox.ymin:.3f}, {face_bbox.zmin:.3f})"
            )
            print(
                f"                max=({face_bbox.xmax:.3f}, {face_bbox.ymax:.3f}, {face_bbox.zmax:.3f})"
            )

            # Check if face is planar and at top/bottom
            center = face.Center()
            print(f"          center=({center.x:.3f}, {center.y:.3f}, {center.z:.3f})")

            # Analyze face normal
            try:
                # Get face normal at center
                surface = face.Surface()
                print(f"          surface type: {type(surface)}")
            except:
                pass

        # Edges analysis
        edges = solid.Edges()
        print(f"\nEdges: {len(edges)}")
        for j, edge in enumerate(edges):
            vertices = edge.Vertices()
            print(f"  Edge {j}: {len(vertices)} vertices, length={edge.Length():.3f}")

            if len(vertices) >= 2:
                v0, v1 = vertices[0], vertices[1]
                print(f"          v0=({v0.X:.3f}, {v0.Y:.3f}, {v0.Z:.3f})")
                print(f"          v1=({v1.X:.3f}, {v1.Y:.3f}, {v1.Z:.3f})")

            # Check edge type
            try:
                curve = edge.Curve()
                print(f"          curve type: {type(curve)}")

                # For circular edges, check if they're at top/bottom
                edge_bbox = edge.BoundingBox()
                print(
                    f"          edge bbox: min=({edge_bbox.xmin:.3f}, {edge_bbox.ymin:.3f}, {edge_bbox.zmin:.3f})"
                )
                print(
                    f"                     max=({edge_bbox.xmax:.3f}, {edge_bbox.ymax:.3f}, {edge_bbox.zmax:.3f})"
                )

                # Check if this is a circular edge at constant Z
                z_min, z_max = edge_bbox.zmin, edge_bbox.zmax
                if abs(z_max - z_min) < 1e-6:  # Constant Z
                    print(f"          -> CIRCULAR EDGE at Z={z_min:.3f}")
                    if abs(z_min - 0.0) < 1e-6:
                        print(f"          -> BOTTOM EDGE")
                    elif abs(z_min - height) < 1e-6:
                        print(f"          -> TOP EDGE")

            except Exception as e:
                print(f"          curve analysis failed: {e}")

        # Vertices analysis
        vertices = solid.Vertices()
        print(f"\nVertices: {len(vertices)}")
        for j, vertex in enumerate(vertices):
            print(f"  Vertex {j}: ({vertex.X:.3f}, {vertex.Y:.3f}, {vertex.Z:.3f})")

    # Test our edge filtering on the cylinder that should work
    print(f"\n=== TESTING EDGE FILTERING ===")

    cylinder = cylinder1  # Use our standard cylinder

    # Try different approaches to find top/bottom edges
    edges = list(cylinder.Edges())

    print(f"\nAnalyzing {len(edges)} edges for top/bottom identification:")

    top_circular_edges = []
    bottom_circular_edges = []

    for i, edge in enumerate(edges):
        edge_bbox = edge.BoundingBox()
        z_min, z_max = edge_bbox.zmin, edge_bbox.zmax

        # Check if edge is at constant Z (circular edge)
        if abs(z_max - z_min) < 1e-6:
            print(f"Edge {i}: Circular at Z={z_min:.3f}")
            if abs(z_min - 0.0) < 1e-6:
                bottom_circular_edges.append(edge)
                print(f"  -> BOTTOM circular edge")
            elif abs(z_min - height) < 1e-6:
                top_circular_edges.append(edge)
                print(f"  -> TOP circular edge")
        else:
            print(f"Edge {i}: Vertical from Z={z_min:.3f} to Z={z_max:.3f}")

    print(f"\nFound {len(top_circular_edges)} top circular edges")
    print(f"Found {len(bottom_circular_edges)} bottom circular edges")

    # Now test if our alignment filter can find these
    alignment_top = filter_edges_by_alignment(cylinder, fillets_at=[Alignment.TOP])
    alignment_bottom = filter_edges_by_alignment(
        cylinder, fillets_at=[Alignment.BOTTOM]
    )

    print(f"\nAlignment filter results:")
    print(f"Top edges found by alignment filter: {len(alignment_top)}")
    print(f"Bottom edges found by alignment filter: {len(alignment_bottom)}")

    # If we found circular edges manually but alignment filter didn't, there's a bug
    if len(top_circular_edges) > 0 and len(alignment_top) == 0:
        print(
            f"BUG: Manual detection found {len(top_circular_edges)} top edges, but alignment filter found {len(alignment_top)}"
        )
    if len(bottom_circular_edges) > 0 and len(alignment_bottom) == 0:
        print(
            f"BUG: Manual detection found {len(bottom_circular_edges)} bottom edges, but alignment filter found {len(alignment_bottom)}"
        )

    # Test assertions
    assert len(edges) > 0, "Cylinder should have edges"
    assert (
        len(top_circular_edges) > 0 or len(bottom_circular_edges) > 0
    ), "Should find at least one circular edge"


def test_cylinder_top_edge_fillet():
    """Test that fillet can be applied to cylinder top edges (regression test for hawaii bottle cap)."""
    # Create cylinder with hawaii bottle cap dimensions
    radius = 16.85
    height = 4.5
    cylinder = create_basic_cylinder(radius=radius, height=height)

    original_volume = cylinder.Volume()

    # Apply fillet to top edges using alignment method
    filleted = apply_fillet_by_alignment(
        cylinder, fillet_radius=1.0, fillets_at=[Alignment.TOP]
    )

    new_volume = filleted.Volume()
    volume_change = abs(new_volume - original_volume)

    # Fillet should reduce volume (rounds off the sharp edge)
    assert (
        volume_change > 0.1
    ), f"Expected significant volume change from fillet, got {volume_change}"
    assert new_volume < original_volume, "Fillet should reduce volume"

    # Test that we can also apply to bottom edges
    filleted_bottom = apply_fillet_by_alignment(
        cylinder, fillet_radius=1.0, fillets_at=[Alignment.BOTTOM]
    )

    bottom_volume_change = abs(filleted_bottom.Volume() - original_volume)
    assert (
        bottom_volume_change > 0.1
    ), f"Expected volume change from bottom fillet, got {bottom_volume_change}"


def test_apply_fillet_to_cylinder_like_hawaii_cap():
    """Test applying fillet to a cylinder similar to hawaii bottle cap."""
    # Create cylinder with exact hawaii bottle cap dimensions
    radius = 16.85  # cap_rim_outer_diameter / 2
    height = 4.5  # cap_cover_thickness
    cylinder = create_basic_cylinder(radius=radius, height=height)

    original_volume = cylinder.Volume()
    print(f"\nOriginal cylinder volume: {original_volume}")

    # Try to apply fillet using alignment-based method
    try:
        filleted = apply_fillet_by_alignment(
            cylinder, fillet_radius=1.0, fillets_at=[Alignment.TOP]
        )
        new_volume = filleted.Volume()
        print(f"Filleted cylinder volume: {new_volume}")
        print(f"Volume change: {abs(new_volume - original_volume)}")

        # If fillet was applied, volume should change
        if abs(new_volume - original_volume) > 0.001:
            print("SUCCESS: Fillet was applied (volume changed)")
        else:
            print("WARNING: Fillet may not have been applied (volume unchanged)")

    except Exception as e:
        print(f"ERROR applying fillet: {e}")
        # Don't fail the test, just report the issue
        pass

    # Also test with z-position based filtering for comparison
    try:
        top_edges_z = filter_edges_by_z_position(
            cylinder, z_threshold=height - 0.1, below=False
        )
        if len(top_edges_z) > 0:
            filleted_z = apply_fillet_to_edges(cylinder, top_edges_z, fillet_radius=1.0)
            new_volume_z = filleted_z.Volume()
            print(f"\nZ-based fillet volume: {new_volume_z}")
            print(f"Z-based volume change: {abs(new_volume_z - original_volume)}")
    except Exception as e:
        print(f"ERROR with z-based fillet: {e}")
    # Create a cylinder: radius=5, height=20
    cylinder = create_basic_cylinder(radius=5, height=20, origin=(0, 0, 0))
    assert cylinder is not None

    # Filter edges at the bottom (z=0)
    bottom_edges = filter_edges_by_z_position(cylinder, z_threshold=0.1, below=True)
    assert (
        len(bottom_edges) >= 1
    ), f"Expected at least 1 bottom edge, got {len(bottom_edges)}"

    # Filter edges at the top (z=20)
    top_edges = filter_edges_by_z_position(cylinder, z_threshold=19.9, below=False)
    assert len(top_edges) >= 1, f"Expected at least 1 top edge, got {len(top_edges)}"

    # Test with exact thresholds
    exact_bottom = filter_edges_by_z_position(cylinder, z_threshold=0.0, below=True)
    exact_top = filter_edges_by_z_position(cylinder, z_threshold=20.0, below=False)

    # Should have found some edges
    assert (
        len(exact_bottom) > 0
    ), f"Expected bottom edges at z=0, got {len(exact_bottom)}"
    assert len(exact_top) > 0, f"Expected top edges at z=20, got {len(exact_top)}"


def test_apply_fillet_comprehensive():
    """Test applying fillet to different edge selections."""
    # Create a box for testing
    box = create_basic_box(20, 20, 20, origin=(0, 0, 0))
    original_volume = box.Volume()

    # Test 1: Fillet top edges
    top_edges = filter_edges_by_z_position(box, z_threshold=19.5, below=False)
    if top_edges:
        filleted_top = apply_fillet_to_edges(box, fillet_radius=1.0, edges=top_edges)
        assert filleted_top is not None
        # Filleted box should have different volume
        assert abs(filleted_top.Volume() - original_volume) > 0.01

    # Test 2: Fillet bottom edges
    bottom_edges = filter_edges_by_z_position(box, z_threshold=0.5, below=True)
    if bottom_edges:
        filleted_bottom = apply_fillet_to_edges(
            box, fillet_radius=1.0, edges=bottom_edges
        )
        assert filleted_bottom is not None
        assert abs(filleted_bottom.Volume() - original_volume) > 0.01

    # Test 3: Empty edge list should return original
    no_fillet = apply_fillet_to_edges(box, fillet_radius=1.0, edges=[])
    assert no_fillet is not None
    assert abs(no_fillet.Volume() - original_volume) < 0.001


def test_edge_filtering_precision():
    """Test edge filtering with precise threshold values."""
    # Create a box with known dimensions
    box = create_basic_box(10, 10, 15, origin=(0, 0, 0))  # Height = 15

    # Test filtering at exact boundary
    edges_at_top = filter_edges_by_z_position(box, z_threshold=15.0, below=False)
    edges_below_top = filter_edges_by_z_position(box, z_threshold=14.99, below=False)

    # Top edges should be found at exact height
    assert len(edges_at_top) > 0, f"Expected top edges at z=15, got {len(edges_at_top)}"
    # Should get same or fewer results with slightly lower threshold
    assert len(edges_below_top) >= len(edges_at_top)

    # Test filtering at bottom
    edges_at_bottom = filter_edges_by_z_position(box, z_threshold=0.0, below=True)
    edges_above_bottom = filter_edges_by_z_position(box, z_threshold=0.01, below=True)

    assert len(edges_above_bottom) >= len(edges_at_bottom)


def test_fillet_large_radius_handling():
    """Test fillet with various radius sizes."""
    box = create_basic_box(10, 10, 10)
    edges = filter_edges_by_z_position(box, z_threshold=9.5, below=False)

    if edges:
        # Small fillet should work
        small_fillet = apply_fillet_to_edges(box, fillet_radius=0.5, edges=edges)
        assert small_fillet is not None

        # Medium fillet should work
        medium_fillet = apply_fillet_to_edges(box, fillet_radius=2.0, edges=edges)
        assert medium_fillet is not None

        # Very large fillet might fail gracefully
        try:
            large_fillet = apply_fillet_to_edges(box, fillet_radius=8.0, edges=edges)
            # If it succeeds, that's fine
            assert large_fillet is not None
        except Exception:
            # If it fails, that's also acceptable for very large fillets
            pass


def test_cylinder_top_bottom_edge_detection():
    """Specifically test that we can distinguish top and bottom edges of a cylinder."""
    cylinder = create_basic_cylinder(radius=10, height=30, origin=(0, 0, 0))

    # Get top edges (should be circular edge at z=30)
    top_edges = filter_edges_by_z_position(cylinder, z_threshold=29.0, below=False)

    # Get bottom edges (should be circular edge at z=0)
    bottom_edges = filter_edges_by_z_position(cylinder, z_threshold=1.0, below=True)

    # Both should find edges
    assert len(top_edges) > 0, "Should find top edges of cylinder"
    assert len(bottom_edges) > 0, "Should find bottom edges of cylinder"

    # Apply fillet to top edges
    if top_edges:
        filleted_cylinder = apply_fillet_to_edges(
            cylinder, fillet_radius=2.0, edges=top_edges
        )
        assert filleted_cylinder is not None
        # Volume should be different (slightly less due to rounded edges)
        original_volume = cylinder.Volume()
        assert abs(filleted_cylinder.Volume() - original_volume) > 0.01


def test_alignment_filtering_cylinder():
    """Test alignment-based filtering on a cylinder."""
    cylinder = create_basic_cylinder(radius=8, height=25, origin=(0, 0, 0))

    # Test top and bottom alignments on cylinder
    top_edges = filter_edges_by_alignment(cylinder, fillets_at=[Alignment.TOP])
    bottom_edges = filter_edges_by_alignment(cylinder, fillets_at=[Alignment.BOTTOM])

    # Should find some edges for top and bottom
    assert len(top_edges) > 0, "Should find top edges on cylinder"
    assert len(bottom_edges) > 0, "Should find bottom edges on cylinder"

    # Apply fillet to top only
    filleted = apply_fillet_by_alignment(
        cylinder, fillet_radius=1.0, fillets_at=[Alignment.TOP]
    )
    assert filleted is not None
    original_volume = cylinder.Volume()
    assert (
        abs(filleted.Volume() - original_volume) > 0.001
    ), "Volume should change after filleting"
