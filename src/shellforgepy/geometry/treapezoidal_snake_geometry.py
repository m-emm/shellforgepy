import numpy as np
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.mesh_utils import (
    propagate_consistent_winding,
    validate_and_fix_mesh_segment,
)
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
)


def create_snake_vertices(cross_section, base_points, normals):
    """
    Create vertices for a snake geometry by transforming 2D cross-section to 3D
    at each base point using proper coordinate system transformation.

    Args:
        cross_section (np.ndarray): (4, 2) array of 2D trapezoid points
        base_points (np.ndarray): (N, 3) array of 3D base points
        normals (np.ndarray): (N, 3) array of normal vectors at each base point

    Returns:
        list: List of vertex arrays, one per segment. Each is (8, 3) for 8 vertices.
    """
    if len(cross_section) != 4:
        raise ValueError("Cross section must have exactly 4 points for a trapezoid")

    if len(base_points) != len(normals):
        raise ValueError("Number of base points must match number of normals")

    if len(base_points) < 2:
        raise ValueError("Need at least 2 base points to create segments")

    base_points = np.array(base_points)
    normals = np.array(normals)
    cross_section = np.array(cross_section)

    all_vertices = []

    for i, (base_point, normal) in enumerate(zip(base_points, normals)):

        if i == 0:
            snake_direction = normalize(base_points[1] - base_points[0])
        elif i == len(base_points) - 1:
            snake_direction = normalize(base_points[-1] - base_points[-2])
        else:
            snake_direction = normalize(base_points[i + 1] - base_points[i - 1])

        transform = coordinate_system_transform(
            origin_a=[0, 0, 0],  # 2D origin
            up_a=[0, 1, 0],  # 2D Y axis (cross-section Y)
            out_a=[
                0,
                0,
                1,
            ],  # Z-axis -  will be rotated to point in the snake_direction
            origin_b=base_point,  # 3D position
            up_b=normal,  # Normal becomes the "up" direction
            out_b=snake_direction,  # Snake direction becomes "out"
        )
        matrix = coordinate_system_transform_to_matrix(transform)

        cross_section_3d = np.concatenate(
            [cross_section, np.zeros((4, 1))], axis=1
        )  # Add z=0
        cross_section_homo = np.concatenate(
            [cross_section_3d, np.ones((4, 1))], axis=1
        )  # Add w=1
        transformed_cross_section = (
            matrix @ cross_section_homo.T
        )  # (4,4) @ (4,4) -> (4,4)
        all_vertices.append(
            transformed_cross_section[:3, :].T
        )  # Take only XYZ, transpose back

    return all_vertices


def create_local_coordinate_system(normal, direction=None):
    """
    Create a local coordinate system from a normal vector.

    Uses Gram-Schmidt orthogonalization similar to spherical_tools.orthonormalize.

    Args:
        normal (np.ndarray): The normal vector (will be aligned with local Y axis)
        direction (np.ndarray, optional): Preferred direction for local X axis

    Returns:
        tuple: (x_axis, y_axis, z_axis) unit vectors
    """
    y_axis = normalize(normal)

    # Choose an arbitrary vector that's not parallel to normal
    if direction is not None:
        temp = normalize(direction)
    else:
        # Use a vector that's least aligned with normal
        abs_normal = np.abs(y_axis)
        min_idx = np.argmin(abs_normal)
        temp = np.zeros(3)
        temp[min_idx] = 1.0

    # Create orthogonal axes using Gram-Schmidt process
    # First, make temp orthogonal to y_axis
    temp_orthogonal = temp - np.dot(temp, y_axis) * y_axis
    temp_norm = np.linalg.norm(temp_orthogonal)

    if temp_norm < 1e-8:
        # temp is collinear with normal, try a different approach
        if abs(y_axis[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        temp_orthogonal = temp - np.dot(temp, y_axis) * y_axis
        temp_norm = np.linalg.norm(temp_orthogonal)

    x_axis = temp_orthogonal / temp_norm
    z_axis = np.cross(x_axis, y_axis)  # No need to normalize, already unit

    return x_axis, y_axis, z_axis


def transform_cross_section_to_3d(cross_section, base_point, normal, direction=None):
    """
    Transform a 2D cross-section to 3D space using a base point and normal.

    Args:
        cross_section (np.ndarray): (N, 2) array of 2D points
        base_point (np.ndarray): 3D point where cross-section is positioned
        normal (np.ndarray): Normal vector (aligned with cross-section's Y axis)
        direction (np.ndarray, optional): Preferred direction for X axis

    Returns:
        np.ndarray: (N, 3) array of 3D points
    """
    x_axis, y_axis, z_axis = create_local_coordinate_system(normal, direction)

    # Transform each 2D point to 3D
    points_3d = []
    for point_2d in cross_section:
        # cross_section coordinates: (x, y) -> (x_axis, y_axis) in 3D
        point_3d = base_point + point_2d[0] * x_axis + point_2d[1] * y_axis
        points_3d.append(point_3d)

    return np.array(points_3d)


def _is_degenerate_segment(start_vertices, end_vertices, tolerance=1e-10):
    """
    Check if a segment is degenerate (start and end cross-sections are essentially identical).

    Args:
        start_vertices (np.ndarray): (4, 3) array of start cross-section vertices
        end_vertices (np.ndarray): (4, 3) array of end cross-section vertices
        tolerance (float): Tolerance for considering vertices identical

    Returns:
        bool: True if the segment is degenerate
    """
    for i in range(4):
        distance = np.linalg.norm(end_vertices[i] - start_vertices[i])
        if distance > tolerance:
            return False
    return True


def create_trapezoidal_snake_geometry(
    cross_section, base_points, normals, close_loop=False
):
    """
    Create a 3D mesh of a trapezoidal snake-like structure by extruding a given cross-sectional shape
    along a specified path defined by base points and normals.

    The cross-section is assumed to be a trapeze given in 2D (x, y) coordinates in the XY plane.
    The trapeze will be oriented such that the (0,0) point of the cross-section will be at the base point,
    and the positive Y axis of the cross-section will be aligned with the normal vector at that base point.

    The function returns, for each segment between two consecutive base points, the vertices and faces
    of the trapezoidal mesh, which can then be converted to solids using any computational solid geometry library.

    Args:
        cross_section (np.ndarray): An (4, 2) array of 2D points defining the cross-sectional trapeze shape.
        base_points (np.ndarray): An (N, 3) array of points defining the path along which to extrude the cross-section.
        normals (np.ndarray): An (N, 3) array of normal vectors at each base point.
        close_loop (bool): If True, creates an additional segment connecting the last cross-section back to the first.
                          This is essential for creating closed loops like Möbius strips or circular paths.
                          Uses propagate_consistent_winding to handle potential vertex correspondence issues
                          from twisting (e.g., 180° rotation in Möbius strips).

    Returns:
        list of dicts: Each dict contains:
            "vertexes": a dict with keys 0-7 for the vertex coordinates of the trapezoid corners (as tuples)
            "faces": a dict with keys 0-11 with faces defined by vertex indices (triangulated faces)

    Note:
        When close_loop=True, the last segment connects the final cross-section to the first one.
        For geometries like Möbius strips where the cross-sections may be rotated relative to each other,
        the propagate_consistent_winding function automatically handles vertex correspondence to ensure
        proper mesh topology without gaps or overlaps.
    """
    # First, generate all vertices for each base point
    all_vertex_sets = create_snake_vertices(cross_section, base_points, normals)

    # Create segments by pairing consecutive vertex sets
    num_segments = len(base_points) - 1
    meshes = []

    for i in range(num_segments):
        # Get vertices for start and end of this segment
        start_vertices = all_vertex_sets[i]  # (4, 3) array
        end_vertices = all_vertex_sets[i + 1]  # (4, 3) array

        # Skip degenerate segments where start and end are essentially identical
        if _is_degenerate_segment(start_vertices, end_vertices):
            continue

        # Create vertex map (8 vertices: 4 at start + 4 at end)
        vertices = {}
        for j in range(4):
            vertices[j] = tuple(start_vertices[j])  # First cross-section (indices 0-3)
            vertices[j + 4] = tuple(
                end_vertices[j]
            )  # Second cross-section (indices 4-7)

        # Create face map (12 triangular faces for a trapezoidal prism)
        # Faces are wound counterclockwise when viewed from outside (right-hand rule)
        faces = {
            # Bottom face (cross-section 1) - normal pointing backward from segment (negative along segment direction)
            0: [
                0,
                2,
                1,
            ],  # Reversed to point outward (negative X for X-direction segment)
            1: [0, 3, 2],  # Reversed to point outward
            # Top face (cross-section 2) - normal pointing forward from segment (positive along segment direction)
            2: [
                4,
                5,
                6,
            ],  # Normal order to point outward (positive X for X-direction segment)
            3: [4, 6, 7],  # Normal order to point outward
            # Side faces connecting the cross-sections
            # Side 0-1
            4: [0, 1, 5],  # Reversed to point outward (negative Y direction)
            5: [0, 5, 4],  # Reversed to point outward
            # Side 1-2
            6: [1, 2, 6],  # Normal order to point outward (positive Y direction)
            7: [1, 6, 5],  # Normal order to point outward
            # Side 2-3
            8: [2, 3, 7],  # Normal order to point outward (positive Z direction)
            9: [2, 7, 6],  # Normal order to point outward
            # Side 3-0
            10: [3, 0, 4],  # Reversed to point outward (negative Z direction)
            11: [3, 4, 7],  # Reversed to point outward
        }

        mesh = {"vertexes": vertices, "faces": faces}
        meshes.append(mesh)

    # Handle loop closing if requested
    if close_loop and len(base_points) >= 3:
        # Create final segment connecting last cross-section to first cross-section
        last_vertices = all_vertex_sets[-1]  # Last cross-section (4, 3) array
        first_vertices = all_vertex_sets[0]  # First cross-section (4, 3) array

        # Skip closing segment if it would be degenerate
        if not _is_degenerate_segment(last_vertices, first_vertices):
            # Detect and fix any twisted vertex correspondence (e.g., in Möbius strips)
            corrected_last, corrected_first, twist_info = validate_and_fix_mesh_segment(
                last_vertices, first_vertices, tolerance=1e-6
            )

            # Create vertex map for closing segment (8 vertices: 4 at end + 4 at start)
            vertices = {}
            for j in range(4):
                vertices[j] = tuple(
                    corrected_last[j]
                )  # Last cross-section (indices 0-3)
                vertices[j + 4] = tuple(
                    corrected_first[j]
                )  # First cross-section (indices 4-7)

            # Create initial face map using standard winding
            faces = {
                # Bottom face (last cross-section) - normal pointing backward
                0: [0, 2, 1],
                1: [0, 3, 2],
                # Top face (first cross-section) - normal pointing forward
                2: [4, 5, 6],
                3: [4, 6, 7],
                # Side faces connecting the cross-sections
                # Side 0-1
                4: [0, 1, 5],
                5: [0, 5, 4],
                # Side 1-2
                6: [1, 2, 6],
                7: [1, 6, 5],
                # Side 2-3
                8: [2, 3, 7],
                9: [2, 7, 6],
                # Side 3-0
                10: [3, 0, 4],
                11: [3, 4, 7],
            }

            # Create triangles list for winding correction
            triangles = [list(face) for face in faces.values()]

            # Use propagate_consistent_winding to handle potential twist
            # This will ensure proper closure even for Möbius strips
            corrected_triangles = propagate_consistent_winding(triangles)

            # Update faces with corrected winding
            corrected_faces = {}
            for i, triangle in enumerate(corrected_triangles):
                corrected_faces[i] = triangle

            closing_mesh = {"vertexes": vertices, "faces": corrected_faces}
            meshes.append(closing_mesh)

    return meshes


# ------------------------------------------------------------
# 1. Cubic Bezier evaluation (3D)
# ------------------------------------------------------------
def _bez_eval_3d(b, t):
    b0, b1, b2, b3 = [np.asarray(p).reshape(1, 3) for p in b]
    t = np.asarray(t).reshape(-1, 1)
    mt = 1 - t
    return b0 * (mt**3) + 3 * b1 * (mt**2) * t + 3 * b2 * mt * (t**2) + b3 * (t**3)


# ------------------------------------------------------------
# 2. Build poly-Bezier chain in 3D (Illustrator-style)
# ------------------------------------------------------------
def _build_bezier_chain_3d(points, tau=0.5):
    """
    points = [
        {"p": (x,y,z), "in": (dx,dy,dz), "out": (dx,dy,dz)},
        ...
    ]
    """
    P = np.array([p["p"] for p in points], dtype=float)
    n = len(P)

    # Catmull-Rom central tangents for auto handles
    M = np.zeros_like(P)
    for i in range(n):
        if i == 0:
            t = P[1] - P[0]
        elif i == n - 1:
            t = P[i] - P[i - 1]
        else:
            t = 0.5 * (P[i + 1] - P[i - 1])
        M[i] = tau * t

    out = M.copy()
    in_ = -M.copy()

    # overrides
    for i, p in enumerate(points):
        if "out" in p:
            out[i] = np.asarray(p["out"], float)
        if "in" in p:
            in_[i] = np.asarray(p["in"], float)

    # segments
    segments = []
    for i in range(n - 1):
        b0 = P[i]
        b1 = P[i] + out[i]
        b2 = P[i + 1] + in_[i + 1]
        b3 = P[i + 1]
        segments.append((b0, b1, b2, b3))
    return segments


# ------------------------------------------------------------
# 3. Sample poly-Bezier curve
# ------------------------------------------------------------
def _sample_bezier_chain_3d(segments, samples_per_segment=40):
    pts = []
    for seg in segments:
        t = np.linspace(0, 1, samples_per_segment)
        pts.append(_bez_eval_3d(seg, t))
    return np.concatenate(pts, axis=0)


# ------------------------------------------------------------
# 4. Bishop frame normals (rotation-minimizing)
# ------------------------------------------------------------
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def compute_bishop_normals(base_points, initial_normal=(0, 0, 1)):
    P = np.asarray(base_points, float)
    n = len(P)
    T = np.zeros_like(P)

    # Tangents
    T[:-1] = P[1:] - P[:-1]
    T[-1] = T[-2]
    T = np.array([normalize(v) for v in T])

    normals = np.zeros_like(P)

    # Initial
    init_n = np.asarray(initial_normal, float)
    init_n = init_n - np.dot(init_n, T[0]) * T[0]
    if np.linalg.norm(init_n) < 1e-12:
        init_n = np.array([1, 0, 0]) - np.dot([1, 0, 0], T[0]) * T[0]
    normals[0] = normalize(init_n)

    # Transport
    for i in range(n - 1):
        t_i = T[i]
        t_j = T[i + 1]

        dot_tt = np.clip(np.dot(t_i, t_j), -1.0, 1.0)
        if dot_tt > 1.0 - 1e-9:
            normals[i + 1] = normals[i]
            continue

        axis = np.cross(t_i, t_j)
        normA = np.linalg.norm(axis)
        if normA < 1e-12:
            normals[i + 1] = normals[i]
            continue
        axis /= normA
        angle = np.arccos(dot_tt)

        n_i = normals[i]
        normals[i + 1] = (
            n_i * np.cos(angle)
            + np.cross(axis, n_i) * np.sin(angle)
            + axis * np.dot(axis, n_i) * (1 - np.cos(angle))
        )
        normals[i + 1] = normalize(normals[i + 1])

    return normals


# ------------------------------------------------------------
# 5. Unified interface: create_bezier_snake_geometry
# ------------------------------------------------------------
def create_bezier_snake_geometry(
    points,
    cross_section,
    samples_per_segment=40,
    tau=0.5,
    initial_normal=(0, 0, 1),
    close_loop=False,
):
    """
    points = [{"p": (x,y,z), "in":..., "out":...}, ...]
    cross_section = 4x2 array
    """

    # Build & sample curve
    segments = _build_bezier_chain_3d(points, tau=tau)
    base_pts = _sample_bezier_chain_3d(segments, samples_per_segment)

    # Compute normals
    normals = compute_bishop_normals(base_pts, initial_normal=initial_normal)

    # Call trapezoidal snake generator
    return create_trapezoidal_snake_geometry(
        cross_section=cross_section,
        base_points=base_pts,
        normals=normals,
        close_loop=close_loop,
    )
