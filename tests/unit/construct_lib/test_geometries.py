import numpy as np
import pytest
import shellforgepy.simple as sf
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_icosahedron_geometry,
    create_regular_polygon_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.geometry.mesh_utils import convert_to_traditional_face_vertex_maps
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.simple import (
    create_solid_from_traditional_face_vertex_maps,
    get_vertex_coordinates,
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


def test_create_regular_polygon_geometry():
    sides = 6
    radius = 2.0
    thickness = 0.5

    points, faces = create_regular_polygon_geometry(radius, sides, thickness)

    assert points.shape == (2 * sides, 3)
    assert faces.shape == (4 * sides - 4, 3)
    assert np.all(faces >= 0)
    assert np.all(faces < len(points))
    assert np.allclose(points[:sides, 2], 0.0)
    assert np.allclose(points[sides:, 2], thickness)
    assert np.allclose(np.linalg.norm(points[:, :2], axis=1), radius)

    mesh_center = np.mean(points, axis=0)
    for face in faces:
        a, b, c = points[face]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        assert np.dot(normal, face_center - mesh_center) > 0.0


def test_create_regular_polygon_geometry_rejects_too_few_sides():
    with pytest.raises(ValueError, match="at least 3 sides"):
        create_regular_polygon_geometry(sides=2)


def test_create_regular_polygon_geometry_is_exported_from_simple():
    assert sf.create_regular_polygon_geometry is create_regular_polygon_geometry
    assert "create_regular_polygon_geometry" in sf.__all__


def test_tetrahedron_traditional_face_vertex_maps():
    points, faces = create_tetrahedron_geometry(1.0)

    face_vertex_maps = convert_to_traditional_face_vertex_maps(points, faces)

    solid = create_solid_from_traditional_face_vertex_maps(face_vertex_maps)

    vertices = get_vertex_coordinates(solid)

    assert len(vertices) == 4

    for original in points:
        assert any(np.allclose(original, v) for v in vertices)


def test_dodecahedron_traditional_face_vertex_maps():
    points, faces = create_dodecahedron_geometry(1.0)

    face_vertex_maps = convert_to_traditional_face_vertex_maps(points, faces)

    solid = create_solid_from_traditional_face_vertex_maps(face_vertex_maps)

    vertices = get_vertex_coordinates(solid)

    assert len(vertices) == 20

    for original in points:
        assert any(np.allclose(original, v) for v in vertices)


def test_icosahedron_traditional_face_vertex_maps():
    points, faces = create_icosahedron_geometry(1.0)

    face_vertex_maps = convert_to_traditional_face_vertex_maps(points, faces)

    solid = create_solid_from_traditional_face_vertex_maps(face_vertex_maps)

    vertices = get_vertex_coordinates(solid)

    assert len(vertices) == 12

    for original in points:
        assert any(np.allclose(original, v) for v in vertices)


def test_cube_traditional_face_vertex_maps():
    points, faces = create_cube_geometry(1.0)

    face_vertex_maps = convert_to_traditional_face_vertex_maps(points, faces)

    solid = create_solid_from_traditional_face_vertex_maps(face_vertex_maps)

    vertices = get_vertex_coordinates(solid)

    assert len(vertices) == 8

    for original in points:
        assert any(np.allclose(original, v) for v in vertices)


def test_fibonacci_sphere_traditional_face_vertex_maps():
    points, faces = create_fibonacci_sphere_geometry(1.0, samples=100)

    face_vertex_maps = convert_to_traditional_face_vertex_maps(points, faces)

    solid = create_solid_from_traditional_face_vertex_maps(face_vertex_maps)

    vertices = get_vertex_coordinates(solid)

    assert len(vertices) == 100

    for original in points:
        assert any(np.allclose(original, v) for v in vertices)
