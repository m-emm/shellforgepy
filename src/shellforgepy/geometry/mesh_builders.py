import math

import numpy as np
from scipy.spatial import ConvexHull


def create_icosahedron_geometry(radius=1.0):
    """
    Returns:
      verts: (12,3) numpy array of vertex coordinates on sphere of given radius
      faces: (20,3) numpy array of triangle indices into verts
    """

    # golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 12 un-scaled verts in the "three.js" / Wikipedia order:
    raw_verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )

    # normalize so each lies on sphere of radius `radius`
    lengths = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts * (radius / lengths)[:, None]

    # the 20 faces (triangles), CCW when viewed from outside
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )

    return verts, faces


def create_dodecahedron_geometry(radius=1.0):
    """
    Returns:
      verts: (20,3) numpy array of vertex coordinates on sphere of given radius
      faces: (12,5) numpy array of pentagon indices into verts (CCW)
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Create the 20 vertices
    raw_verts = np.array(
        [
            # 8 vertices at (±1, ±1, ±1)
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
            # 12 vertices at even permutations of (0, ±1/phi, ±phi)
            [0, -1 / phi, -phi],
            [0, -1 / phi, phi],
            [0, 1 / phi, -phi],
            [0, 1 / phi, phi],
            [-1 / phi, -phi, 0],
            [-1 / phi, phi, 0],
            [1 / phi, -phi, 0],
            [1 / phi, phi, 0],
            [-phi, 0, -1 / phi],
            [phi, 0, -1 / phi],
            [-phi, 0, 1 / phi],
            [phi, 0, 1 / phi],
        ],
        dtype=np.float64,
    )

    # Normalize to lie on sphere
    lengths = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts * (radius / lengths)[:, None]

    # Define the 12 pentagonal faces (indices into verts array)
    faces = np.array(
        [
            [0, 8, 4, 14, 12],  # Face 0
            [0, 12, 1, 18, 16],  # Face 1
            [0, 16, 2, 10, 8],  # Face 2
            [1, 12, 14, 5, 9],  # Face 3
            [1, 9, 11, 3, 18],  # Face 4
            [2, 16, 18, 3, 13],  # Face 5
            [2, 13, 15, 6, 10],  # Face 6
            [3, 11, 7, 15, 13],  # Face 7
            [4, 8, 10, 6, 17],  # Face 8
            [4, 17, 19, 5, 14],  # Face 9
            [5, 19, 7, 11, 9],  # Face 10
            [6, 15, 7, 19, 17],  # Face 11
        ],
        dtype=int,
    )

    return verts, faces


def create_cube_geometry(radius=1.0):
    """
    Returns:
        verts: (8,3) numpy array — vertices on a sphere of given radius
        faces: (12,3) numpy array — triangles with outward-facing normals
    """
    raw_verts = np.array(
        [
            [-1, -1, -1],  # 0
            [1, -1, -1],  # 1
            [1, 1, -1],  # 2
            [-1, 1, -1],  # 3
            [-1, -1, 1],  # 4
            [1, -1, 1],  # 5
            [1, 1, 1],  # 6
            [-1, 1, 1],  # 7
        ],
        dtype=np.float64,
    )

    verts = raw_verts / np.linalg.norm(raw_verts, axis=1)[:, None] * radius

    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom (-Z)
            [4, 5, 6],
            [4, 6, 7],  # top    (+Z)
            [0, 1, 5],
            [0, 5, 4],  # front  (-Y)
            [2, 3, 7],
            [2, 7, 6],  # back   (+Y)
            [1, 2, 6],
            [1, 6, 5],  # right  (+X)
            [3, 0, 4],
            [3, 4, 7],  # left   (-X)
        ],
        dtype=int,
    )

    return verts, faces


def create_tetrahedron_geometry(radius=1.0):
    """
    Returns:
        verts: (4,3) numpy array of vertex coordinates on sphere of given radius
        faces: (4,3) numpy array of triangle indices with outward normals
    """
    raw_verts = np.array(
        [
            [1, 1, 1],  # 0
            [-1, -1, 1],  # 1
            [-1, 1, -1],  # 2
            [1, -1, -1],  # 3
        ],
        dtype=np.float64,
    )

    verts = raw_verts / np.linalg.norm(raw_verts, axis=1)[:, None] * radius

    faces = np.array(
        [
            [0, 2, 1],  # base face (bottom)
            [0, 1, 3],  # side face
            [0, 3, 2],  # side face
            [1, 2, 3],  # back face
        ],
        dtype=int,
    )

    return verts, faces


def create_fibonacci_sphere_geometry(radius=1.0, samples=100):
    """
    Returns:
        verts: (N, 3) array of points evenly distributed on a sphere of given radius
        faces: (M, 3) array of triangle indices forming a convex hull with outward-pointing normals
    """
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    points = []
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y from 1 to -1
        r = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append([x, y, z])

    points = np.array(points) * radius
    hull = ConvexHull(points)

    verts = points
    faces = []

    sphere_center = np.mean(points, axis=0)  # should be near (0, 0, 0)

    for simplex in hull.simplices:
        a, b, c = verts[simplex[0]], verts[simplex[1]], verts[simplex[2]]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3
        if np.dot(normal, face_center - sphere_center) < 0:
            # Flip face to make it outward-facing
            faces.append([simplex[0], simplex[2], simplex[1]])
        else:
            faces.append([simplex[0], simplex[1], simplex[2]])

    faces = np.array(faces, dtype=int)

    return verts, faces
