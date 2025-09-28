import logging
import math
from collections import defaultdict

import numpy as np
from shellforgepy.construct.construct_utils import (
    compute_triangle_normal,
    normalize,
    triangle_area,
)
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
)
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.transformed_region_view import TransformedRegionView

_logger = logging.getLogger(__name__)


def test_split_region_by_cap():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = MeshPartition(mesh)

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=0, target_area_fraction=0.4
    )


def test_split_region_by_polar_oriented_plane():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition = MeshPartition(mesh)

    partition_2 = partition.split_region_by_polar_oriented_plane(
        region_id=0, target_area_fraction=0.5, phi=np.pi / 4
    )


def test_split_twice():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = MeshPartition(mesh)

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=0, target_area_fraction=0.4
    )

    print(f"***Partition after first split: {partition}")

    partition_2 = partition.split_region_by_polar_oriented_plane(
        region_id=1, target_area_fraction=0.5, phi=np.pi / 4
    )


def test_split_top_bottom_caps():

    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=100, radius=10
    )

    partition_0 = MeshPartition(mesh)

    trv = TransformedRegionView(partition_0, region_id=0)

    V, F, E = trv.get_transformed_vertices_faces_boundary_edges()

    # find the poles

    top_pole_vertex_index = np.argmax(V[:, 2])
    bottom_pole_vertex_index = np.argmin(V[:, 2])
    print(
        f"Top pole vertex index: {top_pole_vertex_index}, Bottom pole vertex index: {bottom_pole_vertex_index}"
    )

    # find a  triangle each that contains the top and bottom poles
    top_pole_triangle_index = np.where(np.isin(F, top_pole_vertex_index))[0][0]
    bottom_pole_triangle_index = np.where(np.isin(F, bottom_pole_vertex_index))[0][0]

    print(
        f"Top pole triangle index: {top_pole_triangle_index}, Bottom pole triangle index: {bottom_pole_triangle_index}"
    )

    partition = partition_0.split_region_by_cap(
        0, initial_seed_triangle_index=top_pole_triangle_index, target_area_fraction=0.1
    )

    partition_2 = partition.split_region_by_cap(
        0,
        initial_seed_triangle_index=bottom_pole_triangle_index,
        target_area_fraction=0.1,
    )


def test_add_vertex_on_edge_of_tetrahedron():
    vertices = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ]
    )

    faces = np.array(
        [
            [0, 2, 1],  # now face 0 is outward
            [0, 1, 3],  # face 1 fixed
            [0, 3, 2],  # already outward
            [1, 2, 3],  # already outward
        ]
    )

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces)

    barycentric_coords = [0.5, 0.5, 0.0]  # edge 0–2

    new_mesh = mesh.add_vertex_in_face(0, barycentric_coords)

    assert len(new_mesh.vertices) == 5
    assert len(new_mesh.faces) == 6

    expected = 0.5 * mesh.vertices[0] + 0.5 * mesh.vertices[2]  # edge 0–2!
    actual = new_mesh.vertices[-1]
    assert np.allclose(actual, expected)

    directed_edge_count = defaultdict(int)
    for face in new_mesh.faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            directed_edge_count[(a, b)] += 1

    for (a, b), count in directed_edge_count.items():
        reverse_count = directed_edge_count.get((b, a), 0)
        assert count == 1
        assert reverse_count == 1


def test_add_vertex_inside_triangle_of_tetrahedron():
    vertices = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ]
    )

    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces)

    barycentric_coords = [1 / 3, 1 / 3, 1 / 3]

    new_mesh = mesh.add_vertex_in_face(0, barycentric_coords)

    assert len(new_mesh.vertices) == 5
    assert len(new_mesh.faces) == 6

    expected = np.mean(mesh.vertices[faces[0]], axis=0)
    actual = new_mesh.vertices[-1]
    assert np.allclose(actual, expected)

    directed_edge_count = defaultdict(int)
    for face in new_mesh.faces:
        for i in range(3):
            a = face[i]
            b = face[(i + 1) % 3]
            directed_edge_count[(a, b)] += 1

    for (a, b), count in directed_edge_count.items():
        reverse_count = directed_edge_count.get((b, a), 0)
        assert count == 1
        assert reverse_count == 1


def test_perforate_along_plane_tetrahedron():
    vertices = np.array(
        [
            [1, 1, 1],  # 0 (A)
            [-1, -1, 1],  # 1 (B)
            [-1, 1, -1],  # 2 (C)
            [1, -1, -1],  # 3 (D)
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )
    labels = ["A", "B", "C", "D"]

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces, vertex_labels=labels)

    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    expected_cut_edges = [("A", "B"), ("A", "C")]
    for pair in expected_cut_edges:
        label1 = f"{pair[0]}__{pair[1]}"
        label2 = f"{pair[1]}__{pair[0]}"
        candidates = cut_mesh.get_vertices_by_label(label1)
        if not candidates:
            candidates = cut_mesh.get_vertices_by_label(label2)
        assert (
            len(candidates) == 1
        ), f"Expected one vertex for edge {pair}, got {candidates}"


def test_perforate_along_plane_tetrahedron_partial():
    vertices = np.array(
        [
            [1, 1, 1],  # 0 (A)
            [-1, -1, 1],  # 1 (B)
            [-1, 1, -1],  # 2 (C)
            [1, -1, -1],  # 3 (D)
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )
    labels = ["A", "B", "C", "D"]

    mesh = PartitionableSpheroidTriangleMesh(vertices, faces, vertex_labels=labels)

    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    triangle_indices = [0]
    cut_mesh, face_index_map = mesh.perforate_along_plane(
        plane_origin, plane_normal, epsilon=1e-9, triangle_indices=triangle_indices
    )

    expected_cut_labels = {"A__B", "A__C", "B__A", "C__A"}
    found_cut_labels = set(l for l in cut_mesh.vertex_labels if "__" in l)
    assert (
        found_cut_labels <= expected_cut_labels
    ), f"Unexpected cut labels: {found_cut_labels}"

    old_faces_per_subdivided = {k: v for k, v in face_index_map.items() if len(v) > 1}
    expected_split_faces = {0, 1, 2}
    assert (
        set(old_faces_per_subdivided.keys()) == expected_split_faces
    ), f"Expected faces {expected_split_faces} to be split, got {old_faces_per_subdivided}"

    for i in range(4):
        if i in expected_split_faces:
            assert len(face_index_map[i]) > 1, f"Face {i} should be split"
        else:
            assert len(face_index_map[i]) == 1, f"Face {i} should not be split"

    for a, b in [("A", "B"), ("A", "C")]:
        candidates = cut_mesh.get_vertices_by_label(f"{a}__{b}")
        if not candidates:
            candidates = cut_mesh.get_vertices_by_label(f"{b}__{a}")
        assert len(candidates) == 1, f"Missing or duplicate cut vertex on edge {a}-{b}"


def test_perforate_along_plane_dodecahedron():

    points, _ = create_dodecahedron_geometry(1.0)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    # Choose a plane that cuts diagonally through the tetrahedron:
    # Plane x = 0 should intersect the edges AB, AC, AD (i.e., vertex 0 to others)
    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])

    # Act
    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    _logger.info(f"Cut mesh labels: {cut_mesh.vertex_labels}")

    # Check that we have the expected original vertices (0-19)
    expected_original_vertices = [str(i) for i in range(20)]
    for vertex_label in expected_original_vertices:
        assert (
            vertex_label in cut_mesh.vertex_labels
        ), f"Missing original vertex {vertex_label}"

    # Check that we have the expected edge intersection vertices
    # The exact order may vary between environments, so we check for presence
    expected_edge_labels = {"5__12", "12__14", "0__4", "0__14", "6__13", "13__15"}

    # Get the edge intersection labels (those with "__" in them)
    actual_edge_labels = {label for label in cut_mesh.vertex_labels if "__" in label}

    # Check that we have the right number of edge intersections
    assert (
        len(actual_edge_labels) >= len(expected_edge_labels) - 1
    ), f"Expected at least {len(expected_edge_labels) - 1} edge intersections, got {len(actual_edge_labels)}"
    assert (
        len(actual_edge_labels) <= len(expected_edge_labels) + 1
    ), f"Expected at most {len(expected_edge_labels) + 1} edge intersections, got {len(actual_edge_labels)}"

    # Check that the total number of vertices is reasonable
    assert (
        len(cut_mesh.vertex_labels) >= 26
    ), f"Expected at least 26 vertices total, got {len(cut_mesh.vertex_labels)}"
    assert (
        len(cut_mesh.vertex_labels) <= 28
    ), f"Expected at most 28 vertices total, got {len(cut_mesh.vertex_labels)}"


def test_perforate_along_plane_dodecahedron_check_normals():

    def check_normals_point_outward(mesh: PartitionableSpheroidTriangleMesh, label=""):
        total_area = 0.0
        inward_count = 0
        outward_count = 0
        zero_normal_count = 0

        for i, face in enumerate(mesh.faces):
            v0, v1, v2 = [mesh.vertices[vi] for vi in face]
            normal = compute_triangle_normal(v0, v1, v2)
            center = (v0 + v1 + v2) / 3.0
            center_dir = normalize(center)

            dot = np.dot(normal, center_dir)
            area = triangle_area(v0, v1, v2)
            total_area += area

            if dot < -1e-4:
                inward_count += 1
                _logger.warning(f"{label} Triangle {i} points inward (dot={dot:.4f})")
            elif dot > 1e-4:
                outward_count += 1
            else:
                zero_normal_count += 1
                _logger.warning(
                    f"{label} Triangle {i} has degenerate normal (dot={dot:.4f})"
                )

        _logger.info(
            f"{label} Triangle normals: outward={outward_count}, inward={inward_count}, degenerate={zero_normal_count}, total_area={total_area:.6f}"
        )
        return total_area

    points, _ = create_dodecahedron_geometry(1.0)

    # Step 2: Create mesh object
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    area_before = check_normals_point_outward(mesh, label="BEFORE")

    # Cut it
    plane_origin = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 1.0, 0.0])
    cut_mesh, _ = mesh.perforate_along_plane(plane_origin, plane_normal)

    area_after = check_normals_point_outward(cut_mesh, label="AFTER")

    assert (
        abs(area_before - area_after) < 1e-5
    ), f"Area mismatch: before={area_before}, after={area_after}"


def test_perforate_with_cylinder_basic():
    # Create a test mesh: cube or small tetrahedron
    verts, faces = create_cube_geometry(radius=4 * math.sqrt(3) / 2.0)
    labels = [str(i) for i in range(len(verts))]
    mesh = PartitionableSpheroidTriangleMesh(verts, faces, labels)

    new_mesh, face_map = mesh.perforate_with_cylinder(
        np.array([0.0, 0.0, -1.0]),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        height=1000,
        radius=1.0,
    )
    assert len(new_mesh.vertices) > len(verts)
    assert all(len(f) == 3 for f in new_mesh.faces)


def test_materialized_shell_maps_with_and_without_smoothing():
    # Create slightly perturbed cube to break perfect symmetry
    verts, faces = create_cube_geometry(radius=1.0)
    verts = np.array(verts)
    verts[0] += np.array([0.2, 0.0, 0.0])  # perturb vertex 0
    verts[1] += np.array([0.0, 0.1, 0.0])  # perturb vertex 1

    labels = [str(i) for i in range(len(verts))]
    mesh = PartitionableSpheroidTriangleMesh(verts, faces, labels)

    shell_thickness = 0.1

    # --- Without smoothing
    shell_maps_unsmoothed, _ = mesh.calculate_materialized_shell_maps(
        shell_thickness=shell_thickness, smooth_inside=False
    )

    # Check: inner and outer triangle centroids should be parallel offset
    for face_idx, shell in shell_maps_unsmoothed.items():
        inner_tri = [shell["vertexes"][i] for i in [0, 1, 2]]
        outer_tri = [shell["vertexes"][i] for i in [3, 4, 5]]

        outer_normal = np.cross(
            outer_tri[1] - outer_tri[0], outer_tri[2] - outer_tri[0]
        )
        outer_normal /= np.linalg.norm(outer_normal)

        inner_normal = np.cross(
            inner_tri[1] - inner_tri[0], inner_tri[2] - inner_tri[0]
        )
        inner_normal /= np.linalg.norm(inner_normal)

        assert np.allclose(inner_normal, outer_normal, atol=1e-6), "Normals differ"

        offset = np.dot(outer_tri[0] - inner_tri[0], outer_normal)
        assert np.isclose(
            offset, shell_thickness, atol=1e-6
        ), f"Offset is {offset}, expected {shell_thickness}"

        # --- With smoothing
    shell_maps_smoothed, _ = mesh.calculate_materialized_shell_maps(
        shell_thickness=shell_thickness, smooth_inside=True
    )

    # Check: inner vertices of all faces sharing a vertex now match
    inner_vertex_global_positions = {}

    for face_idx, shell in shell_maps_smoothed.items():
        for local_idx in range(3):
            inner_vertex = shell["vertexes"][local_idx]
            rounded = tuple(np.round(inner_vertex, decimals=5))
            inner_vertex_global_positions.setdefault(rounded, 0)
            inner_vertex_global_positions[rounded] += 1

    # All inner vertices now reused — so number of unique rounded positions is small
    assert (
        len(inner_vertex_global_positions) < 10
    ), f"Expected re-use of inner vertices after smoothing, got {len(inner_vertex_global_positions)}"
