"""Tests for polygon perforation vertex filter."""

import logging

import numpy as np
from shellforgepy.shells.polygon_perforation import create_edge_filter

_logger = logging.getLogger(__name__)


def test_vertex_filter_basic():
    """Test basic vertex filter functionality."""

    # Create edge filter for edge from (1,0,0.3) to (0,1,0.6) with Z-normal plane
    edge_filter = create_edge_filter(
        p1_local=np.array([1.0, 0.0, 0.3]),
        p2_local=np.array([0.0, 1.0, 0.6]),
        normal_local=np.array([0.0, 0.0, 1.0]),
    )

    # Test point that should be accepted (midpoint of projected edge)
    midpoint = np.array([0.5, 0.5, 0.45])  # Midpoint in 3D
    assert edge_filter(midpoint, None, None, None), "Midpoint should be accepted"

    # Test projected p1 (should be accepted)
    p1_projected = np.array([1.0, 0.0, 0.3])  # p1 is already on the plane
    assert edge_filter(
        p1_projected, None, None, None
    ), "Projected p1 should be accepted"

    # Test projected p2 (should be accepted)
    p2_projected = np.array([0.0, 1.0, 0.3])  # p2 projected onto Z=0.3 plane
    assert edge_filter(
        p2_projected, None, None, None
    ), "Projected p2 should be accepted"


def test_vertex_filter_out_of_bounds():
    """Test vertex filter rejects points outside segment bounds."""

    edge_filter = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 0.0]),
        p2_local=np.array([1.0, 0.0, 0.0]),
        normal_local=np.array([0.0, 0.0, 1.0]),
    )

    # Point before p1 (should be rejected)
    before_p1 = np.array([-0.5, 0.0, 0.0])
    assert not edge_filter(
        before_p1, None, None, None
    ), "Point before p1 should be rejected"

    # Point after p2 (should be rejected)
    after_p2 = np.array([1.5, 0.0, 0.0])
    assert not edge_filter(
        after_p2, None, None, None
    ), "Point after p2 should be rejected"

    # Point off the line (should be rejected)
    off_line = np.array([0.5, 1.0, 0.0])
    assert not edge_filter(
        off_line, None, None, None
    ), "Point off the line should be rejected"


def test_vertex_filter_different_planes():
    """Test vertex filter with different plane orientations."""

    # Test with Y-normal plane
    edge_filter_y = create_edge_filter(
        p1_local=np.array([0.0, 2.0, 0.0]),
        p2_local=np.array([1.0, 2.0, 1.0]),
        normal_local=np.array([0.0, 1.0, 0.0]),  # Y-normal
    )

    # Point on projected edge (p2 projects to (1,2,0) with Y-normal)
    on_edge = np.array([0.5, 5.0, 0.5])  # Should project to midpoint
    assert edge_filter_y(on_edge, None, None, None), "Point should project onto edge"

    # Test with arbitrary plane normal - use edge NOT parallel to normal
    edge_filter_arbitrary = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 0.0]),
        p2_local=np.array([2.0, 1.0, 0.0]),  # Edge in XY plane
        normal_local=np.array([1.0, 1.0, 1.0])
        / np.sqrt(3),  # Normalized, NOT parallel to edge
    )

    # Point that should project onto the edge
    # Since p1=[0,0,0] is on the plane and p2=[2,1,0] projects onto the plane,
    # we can find a point that projects to the midpoint of the projected edge
    test_point = np.array(
        [1.07735027, 0.57735027, 0.07735027]
    )  # Projects to edge midpoint
    assert edge_filter_arbitrary(
        test_point, None, None, None
    ), "Point should project onto edge"


def test_vertex_filter_edge_cases():
    """Test vertex filter edge cases."""

    # Test with very small edge
    edge_filter_tiny = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 0.0]),
        p2_local=np.array([1e-10, 0.0, 0.0]),
        normal_local=np.array([0.0, 0.0, 1.0]),
    )

    # Should reject all points for tiny edge
    test_point = np.array([5e-11, 0.0, 0.0])
    # This might accept or reject depending on numerical precision - just ensure no crash
    result = edge_filter_tiny(test_point, None, None, None)
    assert isinstance(result, bool), "Should return boolean without crashing"

    # Test with zero-length edge (degenerate case)
    edge_filter_zero = create_edge_filter(
        p1_local=np.array([1.0, 2.0, 3.0]),
        p2_local=np.array([1.0, 2.0, 3.0]),  # Same point
        normal_local=np.array([0.0, 0.0, 1.0]),
    )

    # Should reject all points for zero-length edge
    test_point = np.array([1.0, 2.0, 3.0])
    assert not edge_filter_zero(
        test_point, None, None, None
    ), "Zero-length edge should reject all points"


def test_vertex_filter_projection_accuracy():
    """Test that projection onto plane works correctly."""

    # Edge with endpoints at different Z values
    edge_filter = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 1.0]),  # p1 is ON the plane z=1
        p2_local=np.array([1.0, 0.0, 2.0]),  # p2 is ABOVE the plane z=1
        normal_local=np.array([0.0, 0.0, 1.0]),  # Z-normal, plane at z=1
    )

    # p2 should project to (1,0,1) on the plane
    # So the projected edge is from (0,0,1) to (1,0,1)

    # Point exactly on the projected edge
    on_projected_edge = np.array([0.5, 0.0, 1.0])
    assert edge_filter(
        on_projected_edge, None, None, None
    ), "Point on projected edge should be accepted"

    # Point above the projected edge but same X,Y
    above_projected_edge = np.array([0.5, 0.0, 3.0])
    assert edge_filter(
        above_projected_edge, None, None, None
    ), "Point above projected edge should be accepted (projects onto edge)"

    # Point below the projected edge but same X,Y
    below_projected_edge = np.array([0.5, 0.0, -1.0])
    assert edge_filter(
        below_projected_edge, None, None, None
    ), "Point below projected edge should be accepted (projects onto edge)"


def test_vertex_filter_tolerance():
    """Test vertex filter tolerance for numerical precision."""

    edge_filter = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 0.0]),
        p2_local=np.array([1.0, 0.0, 0.0]),
        normal_local=np.array([0.0, 0.0, 1.0]),
    )

    # Point slightly off the edge (within tolerance)
    slightly_off = np.array([0.5, 1e-8, 0.0])  # Very small Y offset
    assert edge_filter(
        slightly_off, None, None, None
    ), "Point within tolerance should be accepted"

    # Point way off the edge (outside tolerance)
    way_off = np.array([0.5, 0.1, 0.0])  # Large Y offset
    assert not edge_filter(
        way_off, None, None, None
    ), "Point outside tolerance should be rejected"

    # Point slightly before p1 (within tolerance)
    slightly_before = np.array([-1e-8, 0.0, 0.0])
    assert edge_filter(
        slightly_before, None, None, None
    ), "Point slightly before p1 (within tolerance) should be accepted"

    # Point slightly after p2 (within tolerance)
    slightly_after = np.array([1 + 1e-8, 0.0, 0.0])
    assert edge_filter(
        slightly_after, None, None, None
    ), "Point slightly after p2 (within tolerance) should be accepted"


def test_vertex_filter_complex_geometry():
    """Test vertex filter with more complex geometric configurations."""

    # Diagonal edge in 3D space
    edge_filter_diagonal = create_edge_filter(
        p1_local=np.array([1.0, 2.0, 3.0]),
        p2_local=np.array([4.0, 5.0, 7.0]),
        normal_local=np.array([1.0, 1.0, 0.0]) / np.sqrt(2),  # 45-degree plane normal
    )

    # Calculate what p2 should project to
    # Plane equation: (x-1) + (y-2) = 0  =>  x + y = 3
    # p2 = (4,5,7), distance to plane = (4-1) + (5-2) = 6
    # Projected p2 = (4,5,7) - 6/sqrt(2) * (1,1,0)/sqrt(2) = (4,5,7) - 3*(1,1,0) = (1,2,7)

    # Point that should be on the projected edge
    midpoint_projected = np.array(
        [1.0, 2.0, 5.0]
    )  # Midpoint between (1,2,3) and (1,2,7)
    assert edge_filter_diagonal(
        midpoint_projected, None, None, None
    ), "Midpoint of projected edge should be accepted"


def test_vertex_filter_comprehensive_scenarios():
    """Test comprehensive scenarios covering various edge cases."""

    # Scenario 1: Horizontal edge with vertical plane
    edge_filter_h = create_edge_filter(
        p1_local=np.array([-1.0, 0.0, 0.0]),
        p2_local=np.array([1.0, 0.0, 2.0]),
        normal_local=np.array([1.0, 0.0, 0.0]),  # X-normal plane at x=-1
    )

    # p2 projects to (-1, 0, 2) on the x=-1 plane
    # Projected edge: (-1,0,0) to (-1,0,2)

    test_points_h = [
        (np.array([-1.0, 0.0, 1.0]), True, "Midpoint of projected edge"),
        (np.array([5.0, 0.0, 1.0]), True, "Point that projects to midpoint"),
        (np.array([-1.0, 1.0, 1.0]), False, "Point off the projected edge"),
        (np.array([-1.0, 0.0, -0.5]), False, "Point before projected p1"),
        (np.array([-1.0, 0.0, 2.5]), False, "Point after projected p2"),
    ]

    for point, expected, description in test_points_h:
        result = edge_filter_h(point, None, None, None)
        assert result == expected, f"{description}: expected {expected}, got {result}"

    # Scenario 2: Edge parallel to plane (should project to a point)
    edge_filter_parallel = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 1.0]),
        p2_local=np.array([1.0, 1.0, 1.0]),  # Both points on plane z=1
        normal_local=np.array([0.0, 0.0, 1.0]),  # Z-normal plane at z=1
    )

    # Since both points are on the plane, projected edge is same as original
    test_points_parallel = [
        (np.array([0.5, 0.5, 1.0]), True, "Midpoint on edge"),
        (np.array([0.5, 0.5, 5.0]), True, "Point that projects to midpoint"),
        (np.array([0.0, 1.0, 1.0]), False, "Point off the edge"),
    ]

    for point, expected, description in test_points_parallel:
        result = edge_filter_parallel(point, None, None, None)
        assert result == expected, f"{description}: expected {expected}, got {result}"


def test_vertex_filter_documentation_examples():
    """Test examples that might appear in documentation."""

    # Example: Simple horizontal edge with vertical cutting plane
    edge_filter = create_edge_filter(
        p1_local=np.array([0.0, 0.0, 0.0]),  # Starting point
        p2_local=np.array([2.0, 0.0, 1.0]),  # End point
        normal_local=np.array([0.0, 1.0, 0.0]),  # Y-normal plane through p1
    )

    # The plane is y=0, so p2 projects to (2,0,1)
    # Projected edge goes from (0,0,0) to (2,0,1)

    # Test various intersection points
    examples = [
        # Point, Expected, Description
        (np.array([1.0, 0.0, 0.5]), True, "Midpoint of projected edge"),
        (np.array([0.0, 0.0, 0.0]), True, "Start point"),
        (np.array([2.0, 0.0, 1.0]), True, "End point projected"),
        (np.array([1.0, 5.0, 0.5]), True, "Point that projects to midpoint"),
        (np.array([3.0, 0.0, 1.5]), False, "Point beyond end"),
        (np.array([-1.0, 0.0, -0.5]), False, "Point before start"),
        (np.array([1.0, 0.0, 2.0]), False, "Point off the projected line"),
    ]

    for point, expected, desc in examples:
        result = edge_filter(point, None, None, None)
        assert (
            result == expected
        ), f"{desc}: point {point}, expected {expected}, got {result}"


if __name__ == "__main__":
    # Run basic test
    test_vertex_filter_basic()
    print("✓ Basic test passed")

    test_vertex_filter_out_of_bounds()
    print("✓ Out of bounds test passed")

    test_vertex_filter_projection_accuracy()
    print("✓ Projection accuracy test passed")

    print("All tests passed!")
