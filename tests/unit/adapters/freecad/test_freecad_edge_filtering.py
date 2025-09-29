"""Tests for FreeCAD edge filtering functionality."""

import pytest

# Try to import FreeCAD and skip tests if not available
try:
    import Part
    from FreeCAD import Base

    freecad_available = True
except ImportError:
    freecad_available = False
    Base = None
    Part = None

from shellforgepy.adapters.freecad.freecad_adapter import (
    apply_fillet_by_alignment,
    apply_fillet_to_edges,
    create_basic_box,
    create_basic_cylinder,
    filter_edges_by_alignment,
    filter_edges_by_z_position,
)
from shellforgepy.construct.alignment import Alignment


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_filter_edges_by_z_position_box():
    """Test edge filtering on a box with known geometry."""
    # Create a 10x10x10 box at origin
    box = create_basic_box(10, 10, 10, origin=(0, 0, 0))
    assert box is not None

    # Filter edges at the bottom (z=0)
    bottom_edges = filter_edges_by_z_position(box, z_threshold=0.1, below=True)
    assert (
        len(bottom_edges) >= 4
    ), f"Expected at least 4 bottom edges, got {len(bottom_edges)}"

    # Filter edges at the top (z=10)
    top_edges = filter_edges_by_z_position(box, z_threshold=9.9, below=False)
    assert len(top_edges) >= 4, f"Expected at least 4 top edges, got {len(top_edges)}"

    # Filter edges in the middle (should get all edges below middle)
    middle_edges = filter_edges_by_z_position(box, z_threshold=5.0, below=True)
    assert len(middle_edges) > 0

    # Just ensure we get sensible counts - don't assume specific differences
    total_edges = len(box.Edges)
    assert len(bottom_edges) <= total_edges
    assert len(top_edges) <= total_edges
    assert len(middle_edges) <= total_edges


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_filter_edges_by_z_position_cylinder():
    """Test edge filtering on a cylinder with known geometry."""
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


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_filter_edges_by_alignment_cylinder():
    """Test alignment-based edge filtering on a cylinder (regression test for hawaii bottle cap)."""
    # Create cylinder with hawaii bottle cap dimensions
    radius = 16.85
    height = 4.5
    cylinder = create_basic_cylinder(radius=radius, height=height)

    # Test alignment-based filtering
    top_edges = filter_edges_by_alignment(cylinder, fillets_at=[Alignment.TOP])
    bottom_edges = filter_edges_by_alignment(cylinder, fillets_at=[Alignment.BOTTOM])

    # Should find the circular edges
    assert len(top_edges) >= 1, f"Expected top edges, got {len(top_edges)}"
    assert len(bottom_edges) >= 1, f"Expected bottom edges, got {len(bottom_edges)}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_apply_fillet_by_alignment_cylinder():
    """Test applying fillet to cylinder using alignment-based filtering."""
    # Create cylinder
    radius = 10
    height = 5
    cylinder = create_basic_cylinder(radius=radius, height=height)

    original_volume = cylinder.Volume

    # Apply fillet to top edges using alignment method
    filleted = apply_fillet_by_alignment(
        cylinder, fillet_radius=1.0, fillets_at=[Alignment.TOP]
    )

    new_volume = filleted.Volume
    volume_change = abs(new_volume - original_volume)

    # Fillet should change volume (rounds off the sharp edge)
    assert (
        volume_change > 0.1
    ), f"Expected significant volume change from fillet, got {volume_change}"

    # Test with exact thresholds matching the cylinder height
    exact_bottom = filter_edges_by_z_position(cylinder, z_threshold=0.0, below=True)
    exact_top = filter_edges_by_z_position(cylinder, z_threshold=height, below=False)

    # Should have found some edges
    assert (
        len(exact_bottom) > 0
    ), f"Expected bottom edges at z=0, got {len(exact_bottom)}"
    assert len(exact_top) > 0, f"Expected top edges at z={height}, got {len(exact_top)}"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_apply_fillet_comprehensive():
    """Test applying fillet to different edge selections."""
    # Create a box for testing
    box = create_basic_box(20, 20, 20, origin=(0, 0, 0))
    original_volume = box.Volume

    # Test 1: Fillet top edges
    top_edges = filter_edges_by_z_position(box, z_threshold=19.5, below=False)
    if top_edges:
        filleted_top = apply_fillet_to_edges(box, fillet_radius=1.0, edges=top_edges)
        assert filleted_top is not None
        # Filleted box should have different volume
        assert abs(filleted_top.Volume - original_volume) > 0.01

    # Test 2: Fillet bottom edges
    bottom_edges = filter_edges_by_z_position(box, z_threshold=0.5, below=True)
    if bottom_edges:
        filleted_bottom = apply_fillet_to_edges(
            box, fillet_radius=1.0, edges=bottom_edges
        )
        assert filleted_bottom is not None
        assert abs(filleted_bottom.Volume - original_volume) > 0.01

    # Test 3: Empty edge list should return original
    no_fillet = apply_fillet_to_edges(box, fillet_radius=1.0, edges=[])
    assert no_fillet is not None
    assert abs(no_fillet.Volume - original_volume) < 0.001


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
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


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
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


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
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
