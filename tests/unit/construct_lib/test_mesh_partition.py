import logging
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
)
from shellforgepy.shells.connector_hint import ConnectorHint
from shellforgepy.shells.connector_utils import merge_collinear_connectors
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.region_edge_feature import RegionEdgeFeature

_logger = logging.getLogger(__name__)


import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend suitable for pytest

import matplotlib.pyplot as plt


def visualize_projected_edges_to_file(
    edges_2d, filepath: str, title: str = "Boundary Walk"
):
    """
    edges_2d: List of ((x1, y1), (x2, y2)) tuples
    filepath: Path to write image (e.g., 'test_boundary_walk.png' or .svg)
    """
    fig, ax = plt.subplots()
    for (x1, y1), (x2, y2) in edges_2d:
        ax.plot([x1, x2], [y1, y2], "k-")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def test_get_submesh_maps():

    points, _ = create_dodecahedron_geometry(1.0)

    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)

    partition = MeshPartition(mesh)
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])  # vertical yz-plane

    new_partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    submesh_maps_0 = new_partition.get_submesh_maps(0)
    submesh_maps_1 = new_partition.get_submesh_maps(1)

    assert len(submesh_maps_0["faces"]) + len(submesh_maps_1["faces"]) == len(
        new_partition.mesh.faces
    ), "Total faces in submesh maps should equal total faces in mesh"

    local_to_global_vertex_map_0 = submesh_maps_0["local_to_global_vertex_map"]
    for local_face_idx, global_face_idx in submesh_maps_0[
        "local_to_global_face_map"
    ].items():

        global_face = new_partition.mesh.faces[global_face_idx]

        local_face = submesh_maps_0["faces"][local_face_idx]

        global_vertices_of_local_face = [
            local_to_global_vertex_map_0[v] for v in local_face
        ]

        assert tuple(global_face) == tuple(global_vertices_of_local_face)


def test_perforated():
    points, _ = create_dodecahedron_geometry(1.0)

    # Step 1: Create the initial mesh and trivial partition (one region)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Step 2: Define a plane (cut through origin, normal in x-direction)
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])  # vertical yz-plane

    # Step 3: Perforate and split
    new_partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    # Step 4: Analyze the new face-to-region mapping
    regions = defaultdict(list)
    for face_idx, region_id in new_partition.face_to_region_map.items():
        regions[region_id].append(face_idx)

    # Step 5: Check that we got two distinct regions
    assert len(regions) == 2, f"Expected 2 regions, got {len(regions)}"
    sizes = {rid: len(faces) for rid, faces in regions.items()}
    _logger.info(f"Perforated region sizes: {sizes}")

    # Step 6: Check that no triangle was lost
    total_faces = sum(sizes.values())
    assert total_faces == len(
        new_partition.mesh.faces
    ), f"Expected {len(new_partition.mesh.faces)} faces assigned, got {total_faces}"

    # Optional: check spatial separation using centroids
    centroids_by_region = {
        rid: np.array(
            [
                new_partition.mesh.vertices[new_partition.mesh.faces[f]].mean(axis=0)
                for f in face_indices
            ]
        )
        for rid, face_indices in regions.items()
    }

    for rid, centroids in centroids_by_region.items():
        avg = centroids.mean(axis=0)
        _logger.info(f"Region {rid} avg centroid: {avg}")


def test_merge_two_collinear_connector_hints():
    # Define two connector hints with a common endpoint
    # They should be merged into a single connector
    a1 = np.array([0.0, 0.0, 0.0])
    b1 = np.array([1.0, 0.0, 0.0])
    b2 = np.array([2.0, 0.0, 0.0])

    # Triangle normals are identical
    normal = np.array([0.0, 0.0, 1.0])

    # Apex vertices not on the edge
    apex_a1 = np.array([0.5, 1.0, 0.0])
    apex_a2 = np.array([1.5, 1.0, 0.0])
    apex_b1 = np.array([0.5, -1.0, 0.0])
    apex_b2 = np.array([1.5, -1.0, 0.0])

    ch1 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a1, b1, apex_a1),
        triangle_b_vertices=(b1, a1, apex_b1),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=b1 - a1,
        edge_centroid=(a1 + b1) / 2,
        original_edges=[],
        face_pair_ids=[(1, 1)],
        start_vertex=a1,
        end_vertex=b1,
        start_vertex_index=0,
        end_vertex_index=1,
    )

    ch2 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(b1, b2, apex_a2),
        triangle_b_vertices=(b2, b1, apex_b2),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=b2 - b1,
        edge_centroid=(b1 + b2) / 2,
        original_edges=[],
        face_pair_ids=[(2, 2)],
        start_vertex=b1,
        end_vertex=b2,
        start_vertex_index=0,
        end_vertex_index=1,
    )

    result = merge_collinear_connectors([ch1, ch2])

    # Expect one merged hint
    assert len(result) == 1, f"Expected 1 merged connector hint, got {len(result)}"

    merged = result[0]
    expected_vec = b2 - a1
    assert np.allclose(
        merged.edge_vector, expected_vec / np.linalg.norm(expected_vec)
    ), f"Expected edge_vector to be normalized {expected_vec}, got {merged.edge_vector}"

    expected_mid = (a1 + b2) / 2
    assert np.allclose(
        merged.edge_centroid, expected_mid
    ), f"Expected edge_centroid {expected_mid}, got {merged.edge_centroid}"

    # Vertex chain must match start and end
    tri_a = merged.triangle_a_vertices
    tri_b = merged.triangle_b_vertices
    assert np.allclose(tri_a[0], a1), "Merged triangle_a should start at a1"
    assert np.allclose(tri_a[1], b2), "Merged triangle_a should end at b2"
    assert np.allclose(tri_b[0], b2), "Merged triangle_b should start at b2"
    assert np.allclose(tri_b[1], a1), "Merged triangle_b should end at a1"

    # Check that face IDs were merged
    assert sorted(merged.face_pair_ids) == [
        (1, 1),
        (2, 2),
    ], "face_pair_ids not merged properly"

    print("test_merge_two_collinear_connector_hints passed.")


def test_merge_three_collinear_connector_hints():
    a0 = np.array([-1.0, 0.0, 0.0])
    a1 = np.array([0.0, 0.0, 0.0])
    a2 = np.array([1.0, 0.0, 0.0])
    a3 = np.array([2.0, 0.0, 0.0])

    normal = np.array([0.0, 0.0, 1.0])

    apex_a1 = np.array([-0.5, 1.0, 0.0])
    apex_a2 = np.array([0.5, 1.0, 0.0])
    apex_a3 = np.array([1.5, 1.0, 0.0])
    apex_b1 = np.array([-0.5, -1.0, 0.0])
    apex_b2 = np.array([0.5, -1.0, 0.0])
    apex_b3 = np.array([1.5, -1.0, 0.0])

    ch1 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a0, a1, apex_a1),
        triangle_b_vertices=(a1, a0, apex_b1),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a1 - a0,
        edge_centroid=(a0 + a1) / 2,
        original_edges=[],
        face_pair_ids=[(1, 1)],
        start_vertex=a0,
        end_vertex=a1,
        start_vertex_index=0,
        end_vertex_index=1,
    )

    ch2 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a1, a2, apex_a2),
        triangle_b_vertices=(a2, a1, apex_b2),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a2 - a1,
        edge_centroid=(a1 + a2) / 2,
        original_edges=[],
        face_pair_ids=[(2, 2)],
        start_vertex=a1,
        end_vertex=a2,
        start_vertex_index=0,
        end_vertex_index=1,
    )

    ch3 = ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(a2, a3, apex_a3),
        triangle_b_vertices=(a3, a2, apex_b3),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=a3 - a2,
        edge_centroid=(a2 + a3) / 2,
        original_edges=[],
        face_pair_ids=[(3, 3)],
        start_vertex=a2,
        end_vertex=a3,
        start_vertex_index=0,
        end_vertex_index=1,
    )

    result = merge_collinear_connectors([ch1, ch2, ch3])
    assert len(result) == 1, f"Expected 1 merged connector hint, got {len(result)}"
    merged = result[0]

    expected_vec = a3 - a0
    expected_mid = (a0 + a3) / 2

    assert np.allclose(
        merged.edge_vector, expected_vec / np.linalg.norm(expected_vec)
    ), f"Expected edge_vector {expected_vec}, got {merged.edge_vector}"
    assert np.allclose(
        merged.edge_centroid, expected_mid
    ), f"Expected edge_centroid {expected_mid}, got {merged.edge_centroid}"

    tri_a = merged.triangle_a_vertices
    tri_b = merged.triangle_b_vertices
    assert np.allclose(tri_a[0], a0), "Merged triangle_a should start at a0"
    assert np.allclose(tri_a[1], a3), "Merged triangle_a should end at a3"
    assert np.allclose(tri_b[0], a3), "Merged triangle_b should start at a3"
    assert np.allclose(tri_b[1], a0), "Merged triangle_b should end at a0"
    assert sorted(merged.face_pair_ids) == [
        (1, 1),
        (2, 2),
        (3, 3),
    ], "face_pair_ids not merged properly"

    print("test_merge_three_collinear_connector_hints passed.")


def make_connector_hint(start, end, fid):
    normal = np.array([0.0, 0.0, 1.0])
    apex_a = (start + end) / 2 + np.array([0.0, 1.0, 0.0])
    apex_b = (start + end) / 2 + np.array([0.0, -1.0, 0.0])
    return ConnectorHint(
        region_a=0,
        region_b=1,
        triangle_a_vertices=(start, end, apex_a),
        triangle_b_vertices=(end, start, apex_b),
        triangle_a_vertex_indices=(0, 1, 2),
        triangle_b_vertex_indices=(2, 1, 0),
        triangle_a_normal=normal,
        triangle_b_normal=normal,
        edge_vector=end - start,
        edge_centroid=(start + end) / 2,
        original_edges=[],
        face_pair_ids=[(fid, fid)],
        start_vertex=start,
        end_vertex=end,
        start_vertex_index=0,
        end_vertex_index=1,
    )


def test_merge_three_collinear_connector_hints_backward():
    # This time we list the hints in reverse order to force backward chaining
    hints = [
        make_connector_hint(
            start=np.array([1.0, 0.0, 0.0]), end=np.array([2.0, 0.0, 0.0]), fid=3
        ),
        make_connector_hint(
            start=np.array([0.0, 0.0, 0.0]), end=np.array([1.0, 0.0, 0.0]), fid=2
        ),
        make_connector_hint(
            start=np.array([-1.0, 0.0, 0.0]), end=np.array([0.0, 0.0, 0.0]), fid=1
        ),
    ]

    merged = merge_collinear_connectors(hints)
    assert len(merged) == 1
    hint = merged[0]
    assert np.allclose(hint.edge_vector, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(hint.edge_centroid, np.array([0.5, 0.0, 0.0]))
    assert sorted(hint.face_pair_ids) == [(1, 1), (2, 2), (3, 3)]
    print("test_merge_three_collinear_connector_hints_backward passed.")


def test_compute_connector_hints_on_partition():
    # Create initial mesh
    points, _ = create_dodecahedron_geometry(1.0)
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    partition = MeshPartition(mesh)

    # Perforate and split to create two regions
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 0.0, 0.0])
    new_partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    # Compute connector hints
    hints = new_partition.compute_connector_hints(shell_thickness=0.02)

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


def round_array(arr):
    arr = np.where(np.abs(arr) < 1e-3, 0.0, arr)  # set near-zero values to zero
    return np.round(arr, 2)


def format_array(arr):
    return f"np.array({round_array(arr).tolist()})"


def test_merge_connector_hints_tetrahedron():

    connector_hints = [
        ConnectorHint(
            region_a=0,
            region_b=1,
            triangle_a_vertices=(
                np.array([15.59, -15.16, -15.16]),
                np.array([-14.72, -15.16, 15.16]),
                np.array([-14.72, 15.16, -15.16]),
            ),
            triangle_b_vertices=(
                np.array([-14.72, 15.16, -15.16]),
                np.array([-14.72, -15.16, 15.16]),
                np.array([15.59, 15.16, 15.16]),
            ),
            triangle_a_vertex_indices=(0, 1, 2),
            triangle_b_vertex_indices=(2, 1, 0),
            triangle_a_normal=np.array([-0.58, -0.58, -0.58]),
            triangle_b_normal=np.array([-0.58, 0.58, 0.58]),
            edge_vector=np.array([0.0, 0.71, -0.71]),
            edge_centroid=np.array([-14.72, 0.0, 0.0]),
            start_vertex=np.array([-14.72, -15.16, 15.16]),
            end_vertex=np.array([-14.72, 15.16, -15.16]),
            start_vertex_index=1,
            end_vertex_index=2,
            original_edges=[],
            face_pair_ids=[],
        ),
        ConnectorHint(
            region_a=0,
            region_b=1,
            triangle_a_vertices=(
                np.array([14.72, -14.07, -14.07]),
                np.array([14.72, 0.0, 0.0]),
                np.array([-13.42, -14.07, 14.07]),
            ),
            triangle_b_vertices=(
                np.array([-13.42, -14.07, 14.07]),
                np.array([14.72, 0.0, 0.0]),
                np.array([14.72, 14.07, 14.07]),
            ),
            triangle_a_vertex_indices=(0, 1, 2),
            triangle_b_vertex_indices=(2, 1, 0),
            triangle_a_normal=np.array([0.58, -0.58, 0.58]),
            triangle_b_normal=np.array([0.58, -0.58, 0.58]),
            edge_vector=np.array([-0.82, -0.41, 0.41]),
            edge_centroid=np.array([0.65, -7.04, 7.04]),
            start_vertex=np.array([14.72, 0.0, 0.0]),
            end_vertex=np.array([-13.42, -14.07, 14.07]),
            start_vertex_index=1,
            end_vertex_index=2,
            original_edges=[],
            face_pair_ids=[],
        ),
        ConnectorHint(
            region_a=0,
            region_b=1,
            triangle_a_vertices=(
                np.array([14.72, -14.07, -14.07]),
                np.array([-13.42, 14.07, -14.07]),
                np.array([14.72, 0.0, 0.0]),
            ),
            triangle_b_vertices=(
                np.array([-13.42, 14.07, -14.07]),
                np.array([14.72, 14.07, 14.07]),
                np.array([14.72, 0.0, 0.0]),
            ),
            triangle_a_vertex_indices=(0, 1, 2),
            triangle_b_vertex_indices=(1, 2, 0),
            triangle_a_normal=np.array([0.58, 0.58, -0.58]),
            triangle_b_normal=np.array([0.58, 0.58, -0.58]),
            edge_vector=np.array([0.82, -0.41, 0.41]),
            edge_centroid=np.array([0.65, 7.04, -7.04]),
            start_vertex=np.array([-13.42, 14.07, -14.07]),
            end_vertex=np.array([14.72, 0.0, 0.0]),
            start_vertex_index=1,
            end_vertex_index=2,
            original_edges=[],
            face_pair_ids=[],
        ),
    ]

    assert len(merge_collinear_connectors(connector_hints)) == 3


def test_drill_hole():

    points, vertices = create_fibonacci_sphere_geometry(1.0, 50)

    vertex_labels = [str(i) for i in range(len(points))]

    vertex_labels[0] = "drill_here"
    vertex_labels[len(points) // 2] = "drill_here"

    mesh = PartitionableSpheroidTriangleMesh(
        points, vertices, vertex_labels=vertex_labels
    )

    partition = MeshPartition(mesh)

    # find plane to cut through the sphere

    drill_vertex_indices = partition.mesh.get_vertices_by_label("drill_here")

    assert len(drill_vertex_indices) == 2, "Expected exactly two drill points"

    drill_points = partition.mesh.vertices[drill_vertex_indices]

    cut_center = np.mean(drill_points, axis=0)

    cut_normal = normalize(drill_points[1] - drill_points[0])

    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=cut_center,
        plane_normal=cut_normal,
    )

    partition = partition.drill_holes_by_label(
        "drill_here",
        radius=0.3,
    )

    assert len(partition.get_regions()) == 4


def test_drill_hole_jaggedness():

    sphere_radius = 100
    mesh = PartitionableSpheroidTriangleMesh.create_fibonacci_sphere_mesh(
        num_points=40, radius=sphere_radius
    )

    for i, v in enumerate(mesh.vertices):
        print(f"Vertex: {i}: {v}")

    new_vertices = []

    partition = MeshPartition(mesh)

    bottom = np.array([0.0, 0.0, -1.0])
    axis = np.array([0.0, 0.0, 1.0])
    height = 10000
    radius = sphere_radius * 0.6

    partition = partition.perforate_and_split_region_by_cylinder(
        region_id=0,
        bottom=bottom,
        axis=axis,
        height=height,
        radius=radius,
    )

    boundary_edges = partition.get_boundary_edges_of_region(0)

    # find the best plane that fits the vertices of the boundary edges

    boundary_points = np.array(
        [partition.mesh.vertices[e[0]] for e in boundary_edges]
        + [partition.mesh.vertices[e[1]] for e in boundary_edges]
    )

    centroid = np.mean(boundary_points, axis=0)
    cov = np.cov(boundary_points.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normalize(normal)
    print(f"Centroid: {centroid}, Normal: {normal}")

    def orthonormal_basis(normal):
        if abs(normal[0]) < abs(normal[1]):
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, tangent)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v

    u, v = orthonormal_basis(normal)

    # Step 2: Map 3D boundary points to 2D in the plane
    vertex_2d_map = {}
    for e in boundary_edges:
        for vi in e:
            if vi not in vertex_2d_map:
                vec = partition.mesh.vertices[vi] - centroid
                x = np.dot(vec, u)
                y = np.dot(vec, v)
                vertex_2d_map[vi] = (x, y)

    # Step 3: Build 2D edges
    projected_2d_edges = [
        (vertex_2d_map[e[0]], vertex_2d_map[e[1]]) for e in boundary_edges
    ]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        filename = tmpfile.name

        visualize_projected_edges_to_file(
            projected_2d_edges, filepath=filename, title="Drill Hole Boundary Walk"
        )

        print(f"Saved boundary walk visualization to {filename}")


def test_find_region_subedges_along_original_edge():
    vertices, faces = create_cube_geometry()
    mesh = PartitionableSpheroidTriangleMesh(vertices=vertices, faces=faces)
    partition = MeshPartition(mesh)

    center = np.array([0.0, 0.0, 0.0])  # z=0 plane
    normal = np.array([0.0, 0.0, 1.0])
    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=center,
        plane_normal=normal,
    )

    # Pick a vertical edge intersected by the plane
    v0 = 0  # (-1, -1, -1)
    v1 = 4  # (-1, -1,  1)

    subedges = partition.find_region_subedges_along_original_edge(
        0,
        v0,
        v1,
    )

    assert isinstance(subedges, list)
    for a, b in subedges:
        assert a.shape == (3,)
        assert b.shape == (3,)
        edge_vec = mesh.vertices[v1] - mesh.vertices[v0]
        edge_len = np.linalg.norm(edge_vec)
        edge_dir = edge_vec / edge_len

        for pt in [a, b]:
            proj_len = np.dot(pt - mesh.vertices[v0], edge_dir)
            closest = mesh.vertices[v0] + proj_len * edge_dir
            dist = np.linalg.norm(pt - closest)
            assert dist < 1e-6, f"Point {pt} is not on the original edge"

    assert len(subedges) >= 1, "Expected at least one subsegment on the original edge"


def test_find_region_subedges_along_original_edge_indices():
    # Create a cube mesh
    vertices, faces = create_cube_geometry()
    mesh = PartitionableSpheroidTriangleMesh(vertices=vertices, faces=faces)
    partition = MeshPartition(mesh)

    # Perform a plane perforation to split some vertical edges
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=center,
        plane_normal=normal,
    )

    # Pick a vertical edge that is cut by the z=0 plane
    v0 = 0  # (-1, -1, -1)
    v1 = 4  # (-1, -1,  1)

    subedge_indices = partition.find_region_subedges_along_original_edge_indices(
        region_id=0, v0=v0, v1=v1
    )

    assert isinstance(subedge_indices, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in subedge_indices)

    # Geometry check: each subedge must lie on the original line
    v0_coord = mesh.vertices[v0]
    v1_coord = mesh.vertices[v1]
    edge_vec = v1_coord - v0_coord
    edge_len = np.linalg.norm(edge_vec)
    edge_dir = edge_vec / edge_len

    for vi_a, vi_b in subedge_indices:
        for vi in (vi_a, vi_b):
            pt = partition.mesh.vertices[vi]
            proj_len = np.dot(pt - v0_coord, edge_dir)
            closest = v0_coord + proj_len * edge_dir
            dist = np.linalg.norm(pt - closest)
            assert dist < 1e-6, f"Vertex {vi} ({pt}) not on original edge"

    # Sanity check: we expect at least one subedge if the original edge was split
    assert (
        len(subedge_indices) >= 1
    ), "Expected at least one subedge along original edge"


def test_find_region_edge_features():
    vertices, faces = create_cube_geometry()
    mesh = PartitionableSpheroidTriangleMesh(vertices=vertices, faces=faces)
    partition = MeshPartition(mesh)

    center = np.array([0.0, 0.0, 0.0])  # z=0 plane
    normal = np.array([0.0, 0.0, 1.0])
    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=center,
        plane_normal=normal,
    )

    features = partition.find_region_edge_features(region_id=0)

    assert isinstance(features, list)
    assert all(isinstance(f, RegionEdgeFeature) for f in features)
    assert all(f.region_id == 0 for f in features)

    for f in features:
        # check geometry dimensions
        assert f.edge_coords[0].shape == (3,)
        assert f.edge_coords[1].shape == (3,)
        assert f.edge_vector.shape == (3,)
        assert f.edge_centroid.shape == (3,)
        assert all(n.shape == (3,) for n in f.face_normals)
        assert all(len(verts) == 3 for verts in f.face_vertices)

        # edge vector must align with the segment
        actual_vec = f.edge_coords[1] - f.edge_coords[0]
        actual_len = np.linalg.norm(actual_vec)
        if actual_len > 0:
            actual_dir = actual_vec / actual_len
            dot = np.dot(actual_dir, f.edge_vector)
            assert (
                abs(abs(dot) - 1.0) < 1e-6
            ), "edge_vector is not aligned with actual edge"

        # centroid must be midpoint
        midpoint = (f.edge_coords[0] + f.edge_coords[1]) / 2
        dist = np.linalg.norm(midpoint - f.edge_centroid)
        assert dist < 1e-6, "edge_centroid is not midpoint of edge"

    assert len(features) >= 1, "Expected at least one RegionEdgeFeature"


def test_find_region_edge_features_along_original_edge():
    vertices, faces = create_cube_geometry()
    mesh = PartitionableSpheroidTriangleMesh(vertices=vertices, faces=faces)
    partition = MeshPartition(mesh)

    # Perforate through Z=0 to split vertical edges
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    partition = partition.perforate_and_split_region_by_plane(
        region_id=0,
        plane_point=center,
        plane_normal=normal,
    )

    # Pick original vertical edge cut by the plane
    v0 = 0  # (-1, -1, -1)
    v1 = 4  # (-1, -1,  1)

    # Now find edge features on subedges from this original edge
    features = partition.find_region_edge_features_along_original_edge(
        region_id=0,
        v0=v0,
        v1=v1,
    )

    assert isinstance(features, list)
    assert all(isinstance(f, RegionEdgeFeature) for f in features)

    assert len(features) >= 1, "Expected at least one edge feature for subedges"

    for f in features:
        a, b = f.edge_coords
        edge_vec = mesh.vertices[v1] - mesh.vertices[v0]
        edge_len = np.linalg.norm(edge_vec)
        edge_dir = edge_vec / edge_len

        for pt in [a, b]:
            proj_len = np.dot(pt - mesh.vertices[v0], edge_dir)
            closest = mesh.vertices[v0] + proj_len * edge_dir
            dist = np.linalg.norm(pt - closest)
            assert dist < 1e-6, f"Point {pt} is not on the original edge"
