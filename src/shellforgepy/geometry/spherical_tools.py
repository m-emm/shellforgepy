import numpy as np
from scipy.spatial import ConvexHull
from shellforgepy.construct.construct_utils import rotation_matrix_from_vectors


def spherical_to_cartesian_jackson(sph: tuple, radius_offset=0, sphere_center=None):
    r, theta, phi = sph
    r += radius_offset
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    if sphere_center is not None:
        x += sphere_center[0]
        y += sphere_center[1]
        z += sphere_center[2]
    return np.array([x, y, z])


def cartesian_to_spherical_jackson(v: np.ndarray, center=None):
    if center is not None:
        v = v - center
    x, y, z = v
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def shrink_triangle(A, B, C, border_width, epsilon=1e-6):
    def compute_offset_point(p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < epsilon or norm2 < epsilon:
            raise ValueError("Degenerate triangle corner with zero-length edge.")

        v1n = v1 / norm1
        v2n = v2 / norm2

        dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
        angle = np.arccos(dot)
        sin_half_angle = np.sin(angle / 2)

        if sin_half_angle < epsilon:
            raise ValueError(
                f"Corner too sharp (angle={np.degrees(angle):.2f}°), cannot safely shrink."
            )

        offset_length = border_width / sin_half_angle

        # Require offset to be smaller than both adjacent edge lengths
        if offset_length > min(norm1, norm2):
            raise ValueError(
                f"Offset {offset_length:.4f} too large for triangle at corner with edge lengths "
                f"{norm1:.4f} and {norm2:.4f}."
            )

        bisector = v1n + v2n
        bisector /= np.linalg.norm(bisector)
        return p0 + bisector * offset_length

    A_new = compute_offset_point(A, B, C)
    B_new = compute_offset_point(B, C, A)
    C_new = compute_offset_point(C, A, B)

    return A_new, B_new, C_new


def create_shell_triangle_geometry(
    triangle_spherical_vertexes,
    sphere_center,
    shell_thickness,
    shrinkage=0.1,
    shrink_border=0,
):

    if len(triangle_spherical_vertexes) != 3:
        raise ValueError("triangle_spherical_vertexes must have 3 elements")

    for i in range(3):
        if len(triangle_spherical_vertexes[i]) != 3:
            raise ValueError(
                "Each element of triangle_spherical_vertexes must have 3 elements (r, theta, phi)"
            )

    cartesian_vertexes = [
        spherical_to_cartesian_jackson(v, sphere_center=sphere_center)
        for v in triangle_spherical_vertexes
    ]
    outside_cartesian_vertexes = [
        spherical_to_cartesian_jackson(
            v, radius_offset=shell_thickness, sphere_center=sphere_center
        )
        for v in triangle_spherical_vertexes
    ]

    # check if the vertexes are in the right order
    # if not, reverse the order

    if (
        np.cross(
            cartesian_vertexes[1] - cartesian_vertexes[0],
            cartesian_vertexes[2] - cartesian_vertexes[0],
        )[2]
        < 0
    ):
        cartesian_vertexes[1], cartesian_vertexes[2] = (
            cartesian_vertexes[2],
            cartesian_vertexes[1],
        )
        outside_cartesian_vertexes[1], outside_cartesian_vertexes[2] = (
            outside_cartesian_vertexes[2],
            outside_cartesian_vertexes[1],
        )

    centroid = np.sum(cartesian_vertexes, axis=0) / 6
    centroid += np.sum(outside_cartesian_vertexes, axis=0) / 6

    for i in range(3):
        cartesian_vertexes[i] = cartesian_vertexes[i] - shrinkage * (
            cartesian_vertexes[i] - centroid
        )
        outside_cartesian_vertexes[i] = outside_cartesian_vertexes[i] - shrinkage * (
            outside_cartesian_vertexes[i] - centroid
        )

    # shrink with border
    if shrink_border > 0:
        cartesian_vertexes[0], cartesian_vertexes[1], cartesian_vertexes[2] = (
            shrink_triangle(
                cartesian_vertexes[0],
                cartesian_vertexes[1],
                cartesian_vertexes[2],
                border_width=shrink_border,
            )
        )
        (
            outside_cartesian_vertexes[0],
            outside_cartesian_vertexes[1],
            outside_cartesian_vertexes[2],
        ) = shrink_triangle(
            outside_cartesian_vertexes[0],
            outside_cartesian_vertexes[1],
            outside_cartesian_vertexes[2],
            border_width=shrink_border,
        )

    # Now use these six points to define a prism
    vertexes = {i: v for i, v in enumerate(cartesian_vertexes)}
    outside_vertexes = {i + 3: v for i, v in enumerate(outside_cartesian_vertexes)}
    cartesian_vertexes = {**vertexes, **outside_vertexes}
    maps = {
        "vertexes": cartesian_vertexes,
        "faces": {
            0: [0, 2, 1],  # bottom
            1: [3, 4, 5],  # top
            2: [0, 1, 4],
            3: [0, 4, 3],
            4: [1, 2, 5],
            5: [1, 5, 4],
            6: [2, 0, 3],
            7: [2, 3, 5],
        },
    }

    return maps


def ray_triangle_intersect(ray_origin, ray_vector, triangle):
    EPSILON = 1e-8
    vertex0, vertex1, vertex2 = triangle
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return None  # Ray is parallel to triangle
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > EPSILON:
        return ray_origin + ray_vector * t  # Intersection point
    else:
        return None  # Line intersects but not the ray


def ray_plane_polygon_intersect(ray_origin, ray_vector, polygon):
    # Compute plane normal
    polygon = np.array(polygon)
    ray_origin = np.array(ray_origin)
    ray_vector = np.array(ray_vector)
    v0, v1, v2 = polygon[:3]

    plane_normal = np.cross(v1 - v0, v2 - v0)
    plane_normal /= np.linalg.norm(plane_normal)

    # Check if ray and plane are parallel
    denom = np.dot(plane_normal, ray_vector)
    if abs(denom) < 1e-8:
        return None  # Parallel

    # Compute intersection point with the plane
    d = np.dot(plane_normal, v0)
    t = (d - np.dot(plane_normal, ray_origin)) / denom
    if t < 0:
        return None  # Intersection behind the ray origin

    intersection_point = ray_origin + t * ray_vector

    # Check if the intersection point is inside the polygon using barycentric technique
    total_area = 0.0
    area_sum = 0.0
    for i in range(len(polygon)):
        v_curr = polygon[i]
        v_next = polygon[(i + 1) % len(polygon)]
        edge = v_next - v_curr
        to_point = intersection_point - v_curr
        cross_prod = np.cross(edge, to_point)
        area = np.linalg.norm(cross_prod) / 2.0
        area_sum += area
        total_area += (
            np.linalg.norm(np.cross(v_next - v_curr, v_curr - polygon[0])) / 2.0
        )

    if abs(area_sum - total_area) < 1e-6:
        return intersection_point
    else:
        return None  # Outside polygon


def is_inside_convex_polygon_2d(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Test if each point in `points` lies inside the convex polygon defined by `polygon`.

    Parameters:
        polygon: (N, 2) array of 2D points (ordered, counterclockwise)
        points:  (M, 2) array of 2D test points

    Returns:
        mask: (M,) boolean array, True if point is inside the polygon
    """
    n_edges = polygon.shape[0]
    n_points = points.shape[0]

    inside = np.ones(n_points, dtype=bool)

    for i in range(n_edges):
        a = polygon[i]
        b = polygon[(i + 1) % n_edges]
        edge = b - a
        to_point = points - a  # (M, 2)

        # 2D cross product: edge_x * point_y - edge_y * point_x
        cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]

        # For CCW polygon: point must be on the left side of edge (cross >= 0)
        inside &= cross >= 0

    return inside


def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def cartesian_to_spherical(xyz):
    x, y, z = xyz.T
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.stack([theta, phi], axis=1)


def azimuthal_projection(theta_phi, extension=0):
    theta, phi = theta_phi.T
    theta = theta + extension
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.stack([x, y], axis=1)


def filter_outside_spherical_cap(cap_theta_phi, other_theta_phi, border_extension=0):
    """
    Filters points outside a spherical cap region defined by `cap_theta_phi`.

    Parameters:
        cap_theta_phi: (N, 2) array of [theta, phi] in radians
        other_theta_phi: (M, 2) array of [theta, phi] in radians

    Returns:
        (K, 2) array of [theta, phi] points from `other_theta_phi` that are outside the cap.
    """
    cap_xyz = spherical_to_cartesian(cap_theta_phi[:, 0], cap_theta_phi[:, 1])
    center_vec = cap_xyz.mean(axis=0)
    center_vec /= np.linalg.norm(center_vec)

    R = rotation_matrix_from_vectors(center_vec, np.array([0, 0, 1]))

    cap_rotated = cap_xyz @ R.T
    cap_rotated_theta_phi = cartesian_to_spherical(cap_rotated)

    # project rotated cap to plane
    cap_proj = azimuthal_projection(cap_rotated_theta_phi, extension=border_extension)

    hull = ConvexHull(cap_proj)

    hull_vertices = cap_proj[hull.vertices]

    # rotate and project other points
    other_xyz = spherical_to_cartesian(other_theta_phi[:, 0], other_theta_phi[:, 1])
    other_rotated = other_xyz @ R.T
    other_rotated_theta_phi = cartesian_to_spherical(other_rotated)
    other_proj = azimuthal_projection(other_rotated_theta_phi)

    mask_inside = is_inside_convex_polygon_2d(hull_vertices, other_proj)

    mask_outside = ~mask_inside

    return other_theta_phi[mask_outside], mask_outside, hull.vertices


def coordinate_system_transform(origin_a, up_a, out_a, origin_b, up_b, out_b):
    """
    Compute the rigid transformation (rotation and translation) needed
    to align coordinate system A to coordinate system B in 3D space.
    This version re-orthogonalizes the up/out vectors using a Gram-Schmidt process.

    Parameters:
    ----------
    origin_a, origin_b : array-like, shape (3,)
        Origins of the source and target coordinate systems.
    up_a, out_a : array-like, shape (3,)
        Up and out vectors of coordinate system A.
    up_b, out_b : array-like, shape (3,)
        Up and out vectors of coordinate system B.

    Returns:
    -------
    transform : dict
        Dictionary with keys:
        - "rotation_axis": tuple of 3 floats
        - "rotation_angle": float (radians)
        - "translation": tuple of 3 floats
    """

    def orthonormalize(u, v):
        """Given two vectors, returns an orthonormal basis (û, v̂, ŵ)."""
        u = u / np.linalg.norm(u)
        v_othogonalized = v - np.dot(v, u) * u
        v_norm = np.linalg.norm(v_othogonalized)
        if v_norm < 1e-8:
            raise ValueError(
                f"Provided 'out' vector is collinear with 'up' vector: u={u}, v={v}"
            )
        v_othogonalized = v_othogonalized / v_norm
        w = np.cross(u, v_othogonalized)
        return np.column_stack((u, v_othogonalized, w))

    # Convert inputs to numpy arrays
    origin_a = np.asarray(origin_a, dtype=float)
    origin_b = np.asarray(origin_b, dtype=float)
    up_a = np.asarray(up_a, dtype=float)
    assert np.linalg.norm(up_a) > 1e-8, "Up vector A cannot be zero"

    out_a = np.asarray(out_a, dtype=float)
    assert np.linalg.norm(out_a) > 1e-8, "Out vector A cannot be zero"
    up_b = np.asarray(up_b, dtype=float)
    assert np.linalg.norm(up_b) > 1e-8, "Up vector B cannot be zero"
    out_b = np.asarray(out_b, dtype=float)
    assert np.linalg.norm(out_b) > 1e-8, "Out vector B cannot be zero"

    # Build orthonormal bases using Gram-Schmidt
    R_a = orthonormalize(up_a, out_a)
    R_b = orthonormalize(up_b, out_b)

    # Rotation matrix from A to B
    R = R_b @ R_a.T

    # Compute axis-angle representation
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0.0):
        axis = np.array([1.0, 0.0, 0.0])  # arbitrary
    elif np.isclose(angle, np.pi):
        # Edge case: rotation by pi → find axis from eigenvector
        eigvals, eigvecs = np.linalg.eigh(R)
        mask = np.isclose(eigvals, 1.0, atol=1e-5)
        if not np.any(mask):
            raise ValueError(
                f"No eigenvector with eigenvalue ~1 found. Eigenvalues: {eigvals}"
            )
        axis = eigvecs[:, mask][:, 0]
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2.0 * np.sin(angle)
        )

    axis = tuple(axis.tolist())
    translation = tuple((origin_b - origin_a).tolist())

    return {
        "rotation_angle": float(angle),
        "rotation_axis": axis,
        "translation": translation,
    }


def coordinate_system_transformation_function(
    origin_a,
    up_a,
    out_a,
    origin_b,
    up_b,
    out_b,
    degree_rotation_function_generator,
    translation_function_generator,
    verbose=False,
):
    """
    Create a transformation function that applies a rigid body transformation
    from coordinate system A to coordinate system B.
    The transformation consists of a rotation and a translation.
    The rotation is defined by the angle and axis of rotation,
    and the translation is defined by the translation vector.
    The returned function takes a part (anything that than can be rotated by using the a function generated by the degreee_rotation_function_generator and translated by the translation_function_generator) and applies the transformation to it.
    Parameters:
    ----------
    origin_a : array-like, shape (3,)
        Origin of the source coordinate system A.
    up_a : array-like, shape (3,)
        "Up" vector of coordinate system A (must be orthogonal to out_a).
    out_a : array-like, shape (3,)
        "Out" vector of coordinate system A (must be orthogonal to up_a).
    origin_b : array-like, shape (3,)
        Origin of the target coordinate system B.
    up_b : array-like, shape (3,)
        "Up" vector of coordinate system B (must be orthogonal to out_b).
    out_b : array-like, shape (3,)
        "Out" vector of coordinate system B (must be orthogonal to up_b).
    degreee_rotation_function_generator : callable
        Function that generates a rotation function given the rotation angle and axis.
    translation_function_generator : callable
        Function that generates a translation function given the translation vector.


    Returns:
    -------
    retval : callable
        A function that takes a part and applies the transformation to it.
        The part is expected to be compatible with the rotation and translation functions.
    """

    if verbose:
        print(f"Creating coordinate system transformation function with parameters:")
        print(f"  Origin A: {origin_a}")
        print(f"  Up A: {up_a}")
        print(f"  Out A: {out_a}")
        print(f"  Origin B: {origin_b}")
        print(f"  Up B: {up_b}")
        print(f"  Out B: {out_b}")

    origin_a = np.asarray(origin_a, dtype=float)
    origin_b = np.asarray(origin_b, dtype=float)
    up_a = np.asarray(up_a, dtype=float) / np.linalg.norm(up_a)
    out_a = np.asarray(out_a, dtype=float) / np.linalg.norm(out_a)
    up_b = np.asarray(up_b, dtype=float) / np.linalg.norm(up_b)
    out_b = np.asarray(out_b, dtype=float) / np.linalg.norm(out_b)

    transformation = coordinate_system_transform(
        origin_a, up_a, out_a, origin_b, up_b, out_b
    )

    rotation_function = degree_rotation_function_generator(
        np.degrees(transformation["rotation_angle"]),
        axis=transformation["rotation_axis"],
    )
    translation_function = translation_function_generator(
        *transformation["translation"]
    )

    def retval(part):

        rotated_part = rotation_function(part)
        translated_part = translation_function(rotated_part)

        return translated_part

    return retval


def coordinate_system_transform_to_matrix(transform: dict) -> np.ndarray:
    angle = transform["rotation_angle"]
    axis = np.array(transform["rotation_axis"])
    translation = np.array(transform["translation"])

    # Build 3x3 rotation matrix from axis/angle
    R = rotation_matrix_from_vectors(axis, axis)  # identity if angle == 0
    if angle != 0:
        axis /= np.linalg.norm(axis)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = translation
    return A


def matrix_to_coordinate_system_transform(matrix: np.ndarray) -> dict:
    """
    Given a 4x4 rigid body transform matrix (rotation + translation),
    extract the transform dictionary used in `coordinate_system_transform`.

    Parameters:
    ----------
    matrix : np.ndarray
        4x4 numpy array representing a homogeneous transform matrix.

    Returns:
    -------
    transform : dict
        Dictionary with:
        - 'rotation_axis': tuple[float, float, float]
        - 'rotation_angle': float (in radians)
        - 'translation': tuple[float, float, float]

    Raises:
    -------
    ValueError if the matrix is not a valid rigid transformation (no scaling/shear).
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix")

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Check orthogonality and determinant
    should_be_identity = R.T @ R
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        raise ValueError("Rotation part is not orthogonal (may contain shear or scale)")

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Rotation part must have determinant 1 (pure rotation)")

    # Compute angle
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0):
        axis = np.array([1, 0, 0])  # arbitrary axis
    elif np.isclose(angle, np.pi):
        # Use eigenvector with eigenvalue 1
        eigvals, eigvecs = np.linalg.eigh(R)
        axis = eigvecs[:, np.isclose(eigvals, 1)].flatten()
        axis /= np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * np.sin(angle)
        )

    return {
        "rotation_axis": tuple(axis),
        "rotation_angle": float(angle),
        "translation": tuple(t),
    }
