import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from shellforgepy.construct.cylinder_spec import CylinderSpec


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix that rotates vector `a` to vector `b`.
    Both a and b must be normalized.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)  # No rotation needed
        else:
            # 180° rotation around any axis perpendicular to a
            # Find a vector orthogonal to a
            if abs(a[0]) < abs(a[1]):
                ortho = np.array([1, 0, 0])
            else:
                ortho = np.array([0, 1, 0])
            axis = np.cross(a, ortho)
            axis = axis / np.linalg.norm(axis)

            x, y, z = axis
            R = np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y, 2 * x * z],
                    [2 * x * y, 1 - 2 * x * x - 2 * z * z, 2 * y * z],
                    [2 * x * z, 2 * y * z, 1 - 2 * x * x - 2 * y * y],
                ]
            )
            return R

    # General case: Rodrigues' rotation formula
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return R


def is_valid_rigid_transform(T: np.ndarray, tol=1e-6) -> bool:
    """
    Checks if a 4x4 transformation matrix is a valid rigid-body transform
    (no scaling, no mirroring, no shear).

    Returns True if the transform preserves orientation and lengths.
    """
    if T.shape != (4, 4):
        raise ValueError("Expected a 4x4 transformation matrix")

    R = T[:3, :3]
    should_be_identity = R.T @ R
    identity = np.eye(3)

    # Check orthonormality
    if not np.allclose(should_be_identity, identity, atol=tol):
        return False

    # Check determinant (should be +1 for rotation, -1 means mirroring)
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tol):
        return False

    return True


def normalize_edge(a, b):
    return tuple(sorted((a, b)))


def triangle_edges(tri):
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def compute_triangle_normal(v0, v1, v2):
    return np.cross(v1 - v0, v2 - v0)


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def are_normals_similar(n1: np.ndarray, n2: np.ndarray, tol: float = 1e-3) -> bool:
    """
    Checks if two normals are nearly aligned (dot product close to 1.0).
    """
    n1 = normalize(n1)
    n2 = normalize(n2)
    return np.dot(n1, n2) > 1.0 - tol


def fibonacci_sphere_spherical_coordinates(samples=100):
    """
    Generates spherical coordinates (theta, phi) for points evenly distributed on a sphere
    using the Fibonacci spiral method.

    Returns:
        List of (theta, phi) tuples where:
            - theta ∈ [0, 2π) is the azimuthal angle
            - phi ∈ [0, π] is the polar angle
    """
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.399963

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y ∈ [1, -1]
        theta = golden_angle * i
        phi = math.acos(y)  # polar angle from +Z
        points.append((theta, phi))

    return points


def fibonacci_sphere(samples=100):
    """
    Generates 3D Cartesian coordinates for points evenly distributed on a unit sphere,
    using spherical coordinates from Fibonacci spiral sampling.

    Returns:
        List of np.array([x, y, z]) points on the unit sphere.
    """
    points = []
    spherical_coords = fibonacci_sphere_spherical_coordinates(samples)

    for theta, phi in spherical_coords:
        x = math.sin(phi) * math.cos(theta)
        y = math.cos(phi)
        z = math.sin(phi) * math.sin(theta)
        points.append(np.array([x, y, z]))

    return points


def compute_barycentric_coords(p, tri):
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return None  # degenerate triangle

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


def is_point_in_triangle(p, v0, v1, v2, tol=1e-6):
    """
    Check if point p is inside the triangle defined by vertices v0, v1, v2.
    Uses barycentric coordinates to determine if the point lies within the triangle.
    """
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)
    p = np.array(p)

    bary_coords = compute_barycentric_coords(p, (v0, v1, v2))
    if bary_coords is None:
        return False
    u, v, w = bary_coords
    return u >= -tol and v >= -tol and w >= -tol and u + v + w <= 1 + tol


Vertex = int
Triangle = Tuple[Vertex, Vertex, Vertex]
Edge = Tuple[Vertex, Vertex]
NewVertexMapping = Dict[Edge, Vertex]


def triangle_edges(tri: Triangle) -> List[Edge]:
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def compute_area(p1, p2, p3):
    """Returns area of triangle with vertices p1, p2, p3 in 2D"""
    return abs(
        (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        / 2
    )


def rotate_triangle(tri: Tuple[int, int, int], k: int) -> Tuple[int, int, int]:
    return (tri[k % 3], tri[(k + 1) % 3], tri[(k + 2) % 3])


def split_triangle_topologically(tri, edge_to_new_vertex, perform_area_check=True):
    original_edges = [
        (tri[0], tri[1]),
        (tri[1], tri[2]),
        (tri[2], tri[0]),
    ]

    split_flags = [normalize_edge(*e) in edge_to_new_vertex for e in original_edges]

    # Find the rotation offset so that all split edges come first
    def count_split_flags(flags):  # how many split edges from the front
        count = 0
        for f in flags:
            if f:
                count += 1
            else:
                break
        return count

    best_offset = max(
        range(3), key=lambda k: count_split_flags(split_flags[k:] + split_flags[:k])
    )

    tri_rot = rotate_triangle(tri, best_offset)
    edge_rot = [
        (tri_rot[0], tri_rot[1]),
        (tri_rot[1], tri_rot[2]),
        (tri_rot[2], tri_rot[0]),
    ]

    # Assign local indices 0, 1, 2 to rotated triangle
    local_to_global = {0: tri_rot[0], 1: tri_rot[1], 2: tri_rot[2]}

    edge_to_local = {
        normalize_edge(0, 1): 3,
        normalize_edge(1, 2): 4,
        normalize_edge(2, 0): 5,
    }

    for local_edge, new_local_index in edge_to_local.items():
        # Map back to global edge
        global_edge = normalize_edge(
            local_to_global[local_edge[0]], local_to_global[local_edge[1]]
        )
        if global_edge in edge_to_new_vertex:
            v_new = edge_to_new_vertex[global_edge]
            local_to_global[new_local_index] = v_new

    # Determine case and return triangles as before
    num_splits = sum([normalize_edge(*e) in edge_to_new_vertex for e in edge_rot])

    CASE_TO_LOCAL_TRIANGLES = {
        0: [[0, 1, 2]],
        1: [[0, 3, 2], [3, 1, 2]],
        2: [[0, 3, 4], [3, 1, 4], [4, 2, 0]],
        3: [[0, 3, 5], [3, 4, 5], [3, 1, 4], [4, 2, 5]],
    }

    local_tris = CASE_TO_LOCAL_TRIANGLES[num_splits]
    final_tris = [[local_to_global[i] for i in tri] for tri in local_tris]

    # Optionally check area
    if perform_area_check:
        coords = {
            tri_rot[0]: (0.0, 0.0),
            tri_rot[1]: (1.0, 0.0),
            tri_rot[2]: (0.5, math.sqrt(3) / 2),
        }
        for i in range(3):
            a, b = edge_rot[i]
            canon = normalize_edge(a, b)
            if canon in edge_to_new_vertex:
                mid = edge_to_new_vertex[canon]
                pa = coords[a]
                pb = coords[b]
                coords[mid] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

        original_area = compute_area(
            coords[tri_rot[0]], coords[tri_rot[1]], coords[tri_rot[2]]
        )
        new_area = sum(
            compute_area(coords[a], coords[b], coords[c]) for (a, b, c) in final_tris
        )
        if not math.isclose(original_area, new_area, rel_tol=1e-9):
            raise ValueError(f"Area mismatch: original {original_area}, new {new_area}")

    return final_tris


def compute_lay_flat_transform(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Compute a 4×4 transformation matrix that lays the triangle (a, b, c) flat on the XY plane.
    The triangle will be rotated such that its normal aligns with +Z and translated so it lies at Z=0.

    Parameters:
        a, b, c: np.ndarray
            The 3D coordinates of the triangle vertices.

    Returns:
        A 4×4 np.ndarray representing the affine transform.
    """
    centroid = (a + b + c) / 3.0
    normal_vec = np.cross(b - a, c - a)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    target_normal = np.array([0.0, 0.0, 1.0])

    # Rotation matrix to align the normal to +Z
    R3 = rotation_matrix_from_vectors(normal_vec, target_normal)

    # Build the full transform: T_z * T(+centroid) * R * T(-centroid)
    T1 = np.eye(4)
    T1[:3, 3] = -centroid

    R4 = np.eye(4)
    R4[:3, :3] = R3

    T2 = np.eye(4)
    T2[:3, 3] = centroid

    M = T2 @ R4 @ T1

    # Apply to triangle and compute average z after transform
    pts_m = [(M @ np.hstack([v, 1.0]))[:3] for v in (a, b, c)]
    z_face = sum(p[2] for p in pts_m) / 3.0

    # Final shift to bring to Z=0
    T3 = np.eye(4)
    T3[2, 3] = -z_face

    return T3 @ M


def intersect_edge_with_cylinder(p1, p2, cylinder: CylinderSpec, epsilon=1e-8):
    """
    Returns the (t1, t2) parameters along the edge p1→p2 where it enters/exits the cylinder.
    If no intersection, returns None.
    """
    from numpy.linalg import norm

    d = p2 - p1  # direction of the edge
    h = cylinder.normal / norm(cylinder.normal)  # normalized cylinder axis
    m = p1 - cylinder.bottom

    # Vector components orthogonal to cylinder axis
    d_perp = d - np.dot(d, h) * h
    m_perp = m - np.dot(m, h) * h

    A = np.dot(d_perp, d_perp)

    if A < epsilon:
        # Edge is parallel to axis; check if it's within radius
        dist_to_axis = np.linalg.norm(m_perp)
        if dist_to_axis > cylinder.radius + epsilon:
            return None  # Edge is outside the cylinder

        # Compute t values where edge enters/leaves via height
        t1 = (0.0 - np.dot(m, h)) / np.dot(d, h)
        t2 = (cylinder.height - np.dot(m, h)) / np.dot(d, h)

        t_enter = min(t1, t2)
        t_exit = max(t1, t2)

        if t_exit < 0 or t_enter > 1:
            return None

        return max(t_enter, 0.0), min(t_exit, 1.0)

    B = 2 * np.dot(d_perp, m_perp)
    C = np.dot(m_perp, m_perp) - cylinder.radius**2

    discriminant = B**2 - 4 * A * C

    if discriminant < -epsilon:
        return None  # no real roots, no intersection
    elif abs(discriminant) <= epsilon:
        discriminant = 0.0

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    t_enter = min(t1, t2)
    t_exit = max(t1, t2)

    # Clamp to edge segment
    if t_exit < 0 or t_enter > 1:
        return None

    return max(t_enter, 0.0), min(t_exit, 1.0)


def triangle_min_angle(p0, p1, p2):
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p0)
    c = np.linalg.norm(p0 - p1)
    angles = []
    for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
        cos_angle = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    return min(angles)


def compute_out_vector(normal, triangle, edge_centroid, edge_vector):
    # Compute triangle centroid
    tri_vertices = [np.array(v) for v in triangle]
    tri_centroid = sum(tri_vertices) / 3.0

    # Initial out vector
    out = np.cross(edge_vector, normal)
    if np.linalg.norm(out) < 1e-6:
        raise ValueError("Degenerate orientation: edge_vector and normal are parallel")

    out = normalize(out)

    # Ensure it points from edge_centroid toward triangle centroid
    to_centroid = tri_centroid - edge_centroid
    if np.dot(out, to_centroid) < 0:
        out = -out
    return out


def select_uniform_cylindrical_vertices(
    vertices,
    cylinder_center_xy,
    min_vertex_distance=20.0,
    allowed_unsupported_height=60.0,
):
    """
    Select a uniform distribution of vertices from a roughly cylindrical point cloud.

    This function takes a 3D point cloud and selects a subset of vertices that are
    uniformly distributed across the cylindrical surface. It transforms the vertices
    into a cylindrical coordinate system (z, theta) and uses recursive spatial
    subdivision to ensure uniform sampling while maintaining minimum distances
    between selected points.

    The algorithm:
    1. Converts 3D vertices to cylindrical coordinates (z, theta) relative to the cylinder axis
    2. Filters vertices based on height and angular constraints
    3. Uses recursive quadtree-like subdivision in (z, theta) space
    4. Selects corner points and center points from each subdivision region
    5. Enforces minimum distance constraints between selected vertices

    Args:
        vertices (array-like): Input vertices as (N, 3) array of 3D points [x, y, z]
        cylinder_center_xy (array-like): Center of cylinder in XY plane as [x, y]
        min_vertex_distance (float, optional): Minimum distance between selected
            vertices in the transformed (z, theta) space. Defaults to 20.0.
        allowed_unsupported_height (float, optional): Minimum Z height to consider
            for vertex selection (filters out low vertices). Defaults to 60.0.

    Returns:
        np.ndarray: Selected vertices as (M, 3) array where M <= N, containing
            the uniformly distributed subset of input vertices

    Notes:
        - The function assumes vertices roughly follow a cylindrical distribution
        - Angular coordinates are scaled by mean radius for uniform spacing
        - A 10% margin is applied to angular bounds to avoid edge effects
        - The recursive subdivision stops when regions become smaller than min_vertex_distance
        - Corner and center point selection ensures good spatial coverage

    Example:
        >>> vertices = np.random.rand(1000, 3) * 100  # Random 3D points
        >>> center = np.array([50, 50])  # Cylinder center in XY
        >>> selected = select_uniform_cylindrical_vertices(
        ...     vertices, center, min_vertex_distance=15.0, allowed_unsupported_height=40.0
        ... )
        >>> print(f"Selected {len(selected)} from {len(vertices)} vertices")
    """
    points = np.array(vertices)

    radii = np.linalg.norm(points[:, :2] - cylinder_center_xy, axis=1)
    mean_radius = np.mean(radii)

    xy = points[:, :2]
    z = points[:, 2]
    rel = xy - cylinder_center_xy
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    theta = np.unwrap(theta)
    theta_corrected = theta * mean_radius

    z_theta = np.stack([z, theta_corrected], axis=1)

    z_min = allowed_unsupported_height
    z_max = np.max(z)
    theta_min = np.min(theta)
    theta_max = np.max(theta)
    theta_margin = 0.1 * (theta_max - theta_min)
    theta_lo = theta_min + theta_margin
    theta_hi = theta_max - theta_margin

    valid_mask = (z >= z_min) & (theta >= theta_lo) & (theta <= theta_hi)
    candidate_indices = np.where(valid_mask)[0]
    candidate_points = z_theta[valid_mask]

    selected_indices = []

    def recurse(candidates, selected, bbox_min, bbox_max):
        if len(candidates) == 0:
            return

        zc_min, tc_min = bbox_min
        zc_max, tc_max = bbox_max

        in_bounds = (
            (candidates[:, 0] >= zc_min)
            & (candidates[:, 0] <= zc_max)
            & (candidates[:, 1] >= tc_min)
            & (candidates[:, 1] <= tc_max)
        )
        bounded_candidates = candidates[in_bounds]
        if len(bounded_candidates) == 0:
            return

        def try_add(point):
            for j in selected:
                if np.linalg.norm(point - candidate_points[j]) < min_vertex_distance:
                    return
            idx = np.where((candidate_points == point).all(axis=1))[0][0]
            selected.append(idx)

        # Add four corners
        try_add(
            bounded_candidates[
                np.argmax(bounded_candidates[:, 0] + bounded_candidates[:, 1])
            ]
        )  # top-right
        try_add(
            bounded_candidates[
                np.argmax(bounded_candidates[:, 0] - bounded_candidates[:, 1])
            ]
        )  # top-left
        try_add(
            bounded_candidates[
                np.argmin(bounded_candidates[:, 0] - bounded_candidates[:, 1])
            ]
        )  # bottom-left
        try_add(
            bounded_candidates[
                np.argmin(bounded_candidates[:, 0] + bounded_candidates[:, 1])
            ]
        )  # bottom-right

        # Center
        center = (np.array(bbox_min) + np.array(bbox_max)) / 2
        dists = np.linalg.norm(bounded_candidates - center, axis=1)
        center_point = bounded_candidates[np.argmin(dists)]
        try_add(center_point)

        # Don't recurse below a minimum box size
        if tc_max - tc_min < min_vertex_distance:
            return

        # Recurse into 4 quadrants
        mid_z = (zc_min + zc_max) / 2
        mid_t = (tc_min + tc_max) / 2
        recurse(candidates, selected, (zc_min, tc_min), (mid_z, mid_t))
        recurse(candidates, selected, (zc_min, mid_t), (mid_z, tc_max))
        recurse(candidates, selected, (mid_z, tc_min), (zc_max, mid_t))
        recurse(candidates, selected, (mid_z, mid_t), (zc_max, tc_max))

    z_max = np.max(z)
    bbox_min = [z_min, theta_lo * mean_radius]
    bbox_max = [z_max, theta_hi * mean_radius]
    recurse(candidate_points, selected_indices, bbox_min, bbox_max)

    return vertices[candidate_indices[selected_indices]]


def fit_sphere_to_points(points: np.ndarray, weights: np.ndarray = None):
    """
    Fit a sphere to a set of 3D points using algebraic least squares.

    Args:
        points: (N, 3) array of triangle centroids
        weights: (N,) array of triangle areas (optional)

    Returns:
        center: (3,) array — estimated sphere center
        radius: float — estimated radius
    """
    assert points.shape[1] == 3, "Expecting (N, 3) array for points"

    A = np.hstack((2 * points, np.ones((len(points), 1))))
    b = np.sum(points**2, axis=1)

    if weights is not None:
        W = np.diag(weights)
        A = W @ A
        b = W @ b

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = x[:3]
    radius = np.sqrt(np.sum(center**2) + x[3])

    return center, radius


def fit_plane(points: np.ndarray):
    """Fit plane to points via PCA. Returns (centroid, normal)."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]  # Normal is the last right-singular vector
    return centroid, normal


def fit_sphere_center_along_plane_normal(points: np.ndarray):
    """
    Fit sphere center constrained to lie on the normal line of the best-fit plane.
    Returns (best_center, best_radius).
    """
    centroid, normal = fit_plane(points)

    def sphere_fit_error(t):
        center = centroid + t * normal
        dists = np.linalg.norm(points - center, axis=1)
        r_mean = np.mean(dists)
        return np.mean((dists - r_mean) ** 2)  # Variance of radius

    # Optimize along t axis
    result = minimize_scalar(sphere_fit_error, bounds=(-1000, 1000), method="bounded")
    best_t = result.x
    best_center = centroid + best_t * normal

    # Final best radius
    dists = np.linalg.norm(points - best_center, axis=1)
    best_radius = np.mean(dists)

    return best_center, best_radius
