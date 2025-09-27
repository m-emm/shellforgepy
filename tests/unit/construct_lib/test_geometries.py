import numpy as np
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_icosahedron_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)


def test_create_cube_geometry():
    points, faces = create_cube_geometry(1.0)
    assert len(points) == 8
    assert len(faces) == 12  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_tetrahedron_geometry():
    points, faces = create_tetrahedron_geometry(1.0)
    assert len(points) == 4
    assert len(faces) == 4  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_dodecahedron_geometry():

    points, faces = create_dodecahedron_geometry(1.0)

    points = points.tolist()
    faces = faces.tolist()

    assert len(points) == 20
    assert len(faces) == 12  # pentagons
    assert all(len(face) == 5 for face in faces)

    # Test face connectivity - every edge should appear in exactly 2 faces (forward and reverse)
    from collections import defaultdict

    edge_count = defaultdict(int)

    for face in faces:
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            edge_count[edge] += 1

    # Check that every edge has its reverse edge
    for edge, count in edge_count.items():
        edge = (int(edge[0]), int(edge[1]))
        reverse_edge = (int(edge[1]), int(edge[0]))
        assert (
            reverse_edge in edge_count
        ), f"Edge {edge} has no reverse edge {reverse_edge}"
        assert count == 1, f"Edge {edge} appears {count} times, should be exactly 1"

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh.from_point_cloud(points)


def test_create_icosahedron_geometry():
    points, faces = create_icosahedron_geometry(1.0)
    assert len(points) == 12
    assert len(faces) == 20  # triangles
    assert all(len(face) == 3 for face in faces)

    # this will crash if the triangles are not wound correctly and have inward pointing normals
    _ = PartitionableSpheroidTriangleMesh(points, faces)


def test_create_fibonacci_sphere_geometry():
    points, faces = create_fibonacci_sphere_geometry(1.0, samples=100)

    assert len(points) == 100
    assert all(len(p) == 3 for p in points)

    # Check that the points are uniformly distributed on the sphere
    norms = [np.linalg.norm(p) for p in points]
    assert all(
        np.isclose(norm, 1.0) for norm in norms
    )  # All points should be on the unit sphere

    _ = PartitionableSpheroidTriangleMesh(points, faces)
