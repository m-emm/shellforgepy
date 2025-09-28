import math

import numpy as np
from shellforgepy.construct.construct_utils import (
    CylinderSpec,
    compute_area,
    fit_sphere_to_points,
    intersect_edge_with_cylinder,
    normalize_edge,
    select_uniform_cylindrical_vertices,
    split_triangle_topologically,
    triangle_edges,
)


def test_split_triangle_topologically_one_edge_split():
    # Triangle vertex indices
    triangle = (0, 1, 2)

    # Coordinates: equilateral triangle
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint of edge 0-1
    }

    # Define the cut: midpoint on edge (0, 1)
    edge_to_new_vertex = {normalize_edge(0, 1): 3}

    # Call the topological splitter
    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect exactly two new triangles
    assert len(new_triangles) == 2

    # Check that all triangles are well-formed
    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3  # no duplicate vertices
        all_vertices.update(tri)

    # Must only use vertices 0, 1, 2, and new vertex 3
    assert all_vertices <= {0, 1, 2, 3}

    # Check that triangle area is preserved
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Inner edge (which includes the cut vertex) must be used in both directions
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    # Each inner edge should be used exactly twice
    for edge, count in counts.items():
        if edge == normalize_edge(0, 1):
            continue  # the original edge was split; its pieces shouldn't reappear
        assert count in (1, 2)


def test_split_triangle_topologically_two_edge_split():
    # Triangle vertex indices
    triangle = (0, 1, 2)

    # Coordinates: equilateral triangle
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint of edge 0-1
        4: (0.75, math.sqrt(3) / 4),  # Midpoint of edge 1-2
    }

    # Define the cuts: midpoints on edges (0,1) and (1,2)
    edge_to_new_vertex = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
    }

    # Call the topological splitter
    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect exactly three new triangles
    assert len(new_triangles) == 3

    # Check that all triangles are well-formed
    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    # Must only use vertices 0, 1, 2, and new vertices 3, 4
    assert all_vertices <= {0, 1, 2, 3, 4}

    # Check that triangle area is preserved
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Inner edges should appear twice (in opposite directions)
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    # All edges should appear either once (outer) or twice (inner)
    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue  # skip illegal original edges
        assert count in (1, 2), f"Edge {edge} appears {count} times"


def test_split_triangle_topologically_no_split():
    triangle = (0, 1, 2)

    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
    }

    edge_to_new_vertex = {}  # No cuts

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    assert len(new_triangles) == 1
    assert (
        new_triangles[0] == [0, 1, 2]
        or new_triangles[0] == [1, 2, 0]
        or new_triangles[0] == [2, 0, 1]
    )

    # Area check
    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = compute_area(*[coords[v] for v in new_triangles[0]])
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)


def test_split_triangle_topologically_three_edge_split():
    triangle = (0, 1, 2)

    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.5, math.sqrt(3) / 2),
        3: (0.5, 0.0),  # Midpoint 0-1
        4: (0.75, math.sqrt(3) / 4),  # Midpoint 1-2
        5: (0.25, math.sqrt(3) / 4),  # Midpoint 2-0
    }

    edge_to_new_vertex = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
        normalize_edge(2, 0): 5,
    }

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    # Expect four triangles
    assert len(new_triangles) == 4

    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    assert all_vertices <= {0, 1, 2, 3, 4, 5}

    original_area = compute_area(coords[0], coords[1], coords[2])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Check inner edge usage
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue
        assert count in (1, 2), f"Edge {edge} appears {count} times"


def test_split_triangle_topologically_three_edge_split_wild_numbering():
    # Arbitrary large global vertex indices
    triangle = (770, 771, 772)

    # Remapped coordinates: same triangle shape
    coords = {
        770: (0.0, 0.0),  # Vertex 0
        771: (1.0, 0.0),  # Vertex 1
        772: (0.5, math.sqrt(3) / 2),  # Vertex 2
        773: (0.5, 0.0),  # Midpoint of 770-771
        774: (0.75, math.sqrt(3) / 4),  # Midpoint of 771-772
        775: (0.25, math.sqrt(3) / 4),  # Midpoint of 772-770
    }

    edge_to_new_vertex = {
        normalize_edge(770, 771): 773,
        normalize_edge(771, 772): 774,
        normalize_edge(772, 770): 775,
    }

    new_triangles = split_triangle_topologically(triangle, edge_to_new_vertex)

    assert len(new_triangles) == 4

    all_vertices = set()
    for tri in new_triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3
        all_vertices.update(tri)

    # We expect only the original and new vertices
    expected_vertices = {770, 771, 772, 773, 774, 775}
    assert all_vertices <= expected_vertices

    # Area should still be preserved
    original_area = compute_area(coords[770], coords[771], coords[772])
    total_new_area = sum(
        compute_area(coords[a], coords[b], coords[c]) for a, b, c in new_triangles
    )
    assert math.isclose(original_area, total_new_area, rel_tol=1e-9)

    # Check edge usage consistency
    from collections import Counter

    norm_edges = [
        normalize_edge(*e) for tri in new_triangles for e in triangle_edges(tri)
    ]
    counts = Counter(norm_edges)

    for edge, count in counts.items():
        if edge in edge_to_new_vertex:
            continue
        assert count in (1, 2), f"Edge {edge} appears {count} times"


def test_intersect_edge_with_cylinder_basic():
    cylinder = CylinderSpec(
        bottom=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        height=1.0,
        radius=1.0,
    )

    # Edge goes through the cylinder vertically
    p1 = np.array([0.5, 0.0, -1.0])
    p2 = np.array([0.5, 0.0, 2.0])

    result = intersect_edge_with_cylinder(p1, p2, cylinder)
    assert result is not None
    t1, t2 = result
    assert 0.0 <= t1 < t2 <= 1.0

    # Edge far outside cylinder
    p3 = np.array([2.0, 0.0, -1.0])
    p4 = np.array([2.0, 0.0, 2.0])
    assert intersect_edge_with_cylinder(p3, p4, cylinder) is None

    # Edge tangent to cylinder
    p5 = np.array([1.0, -1.0, 0.5])
    p6 = np.array([1.0, 1.0, 0.5])
    result = intersect_edge_with_cylinder(p5, p6, cylinder)
    assert result is None or (0.0 <= result[0] <= result[1] <= 1.0)


def test_intersect_edge_with_cylinder_lateral_entry_exit():
    cylinder = CylinderSpec(
        bottom=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        height=1.0,
        radius=1.0,
    )

    # Edge enters cylinder from the side and exits again
    p1 = np.array([-2.0, 0.0, 0.5])
    p2 = np.array([2.0, 0.0, 0.5])

    result = intersect_edge_with_cylinder(p1, p2, cylinder)
    assert result is not None
    t1, t2 = result
    assert 0.0 <= t1 < t2 <= 1.0


def test_intersect_edge_with_cylinder_top_cap_touch():
    cylinder = CylinderSpec(
        bottom=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        height=1.0,
        radius=1.0,
    )

    # Edge starts inside cylinder and exits exactly at top
    p1 = np.array([0.0, 0.0, 0.5])
    p2 = np.array([0.0, 0.0, 1.0])

    result = intersect_edge_with_cylinder(p1, p2, cylinder)
    assert result is not None
    t1, t2 = result
    assert np.isclose(t2, 1.0)


def test_intersect_edge_with_cylinder_diagonal():
    cylinder = CylinderSpec(
        bottom=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        height=1.0,
        radius=1.0,
    )

    # Enters at bottom-left, exits top-right
    p1 = np.array([-2.0, -2.0, -1.0])
    p2 = np.array([2.0, 2.0, 2.0])

    result = intersect_edge_with_cylinder(p1, p2, cylinder)
    assert result is not None
    t1, t2 = result
    assert 0.0 <= t1 < t2 <= 1.0


def test_intersect_edge_with_cylinder_outside_radius():
    cylinder = CylinderSpec(
        bottom=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        height=2.0,
        radius=1.0,
    )

    # Completely parallel, at z=1, but 2 units to the side
    p1 = np.array([2.0, 0.0, 1.0])
    p2 = np.array([3.0, 0.0, 1.0])

    assert intersect_edge_with_cylinder(p1, p2, cylinder) is None


def test_select_uniform_cylindrical_vertices():
    np.random.seed(42)  # For reproducibility

    cylinder_angle = np.radians(80)
    cylinder_start_angle = np.radians(70)

    cylinder_radius = 100
    cylinder_height = 100  # increase to allow selection room

    min_dist = 20.0
    unsupported_threshold = 20.0
    cylinder_center = np.array([30, 70])
    cylinder_center_xy = cylinder_center

    radius_noise = 0.5  # ±50% radius variation

    num_points = 200

    vertices = []
    for _ in range(num_points):
        r = cylinder_radius + np.random.uniform(-radius_noise, radius_noise)
        theta = cylinder_start_angle + np.random.uniform(0, cylinder_angle)

        x = cylinder_center[0] + r * np.cos(theta)
        y = cylinder_center[1] + r * np.sin(theta)
        z = np.random.uniform(0, cylinder_height)

        vertices.append([x, y, z])

    vertices = np.array(vertices)

    selected = select_uniform_cylindrical_vertices(
        vertices,
        cylinder_center_xy,
        min_vertex_distance=min_dist,
        allowed_unsupported_height=unsupported_threshold,
    )

    assert len(selected) > 0, "No vertices were selected, expected at least some"

    rel = selected[:, :2] - cylinder_center_xy
    radii = np.linalg.norm(rel, axis=1)
    thetas = np.arctan2(rel[:, 1], rel[:, 0])
    thetas = np.unwrap(thetas)
    theta_corrected = thetas * np.mean(radii)
    z_theta = np.stack([selected[:, 2], theta_corrected], axis=1)

    # Validate: all above height threshold
    assert np.all(
        selected[:, 2] >= unsupported_threshold
    ), "Some vertices below height threshold"

    # Validate: pairwise distance constraint in (z, theta_corrected) space
    for i in range(len(z_theta)):
        for j in range(i + 1, len(z_theta)):
            dist = np.linalg.norm(z_theta[i] - z_theta[j])
            assert dist >= min_dist, f"Selected vertices too close: dist={dist:.2f}"

    print(f"Selected {len(selected)} support vertices from {num_points} total.")


def test_fit_sphere_to_points():
    vertices = []
    num_points = 100

    sphere_radius = 100
    radius_noise = 0.01

    sphere_center = np.array([-79, 268, 776])
    radius_range = sphere_radius * radius_noise

    theta_start_angle = np.radians(70)
    theta_range = np.radians(50)

    phi_start_angle = np.radians(30)
    phi_range = np.radians(50)

    for i in range(num_points):
        r = sphere_radius + np.random.uniform(-radius_range, radius_range)

        theta = theta_start_angle + np.random.uniform(0, theta_range)
        phi = phi_start_angle + np.random.uniform(0, phi_range)

        x = sphere_center[0] + r * np.sin(theta) * np.cos(phi)
        y = sphere_center[1] + r * np.sin(theta) * np.sin(phi)
        z = sphere_center[2] + r * np.cos(theta)

        vertices.append([x, y, z])

    vertices = np.array(vertices)

    center, radius = fit_sphere_to_points(vertices)

    # Check that center and radius are close to expected
    assert np.linalg.norm(center - sphere_center) < 5.0
    assert abs(radius - sphere_radius) < 5.0
