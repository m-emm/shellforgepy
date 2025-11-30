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


def create_snake_vertices(cross_section, base_points, normals, base_scales=None):
    """
    Create vertices for a snake geometry by transforming 2D cross-section to 3D
    at each base point using proper coordinate system transformation.

    Args:
        cross_section (np.ndarray): (4, 2) array of 2D trapezoid points
        base_points (np.ndarray): (N, 3) array of 3D base points
        normals (np.ndarray): (N, 3) array of normal vectors at each base point
        base_scales (np.ndarray | list | None): Optional per-point scale factors
            applied to the cross-section before transformation.

    Returns:
        list: List of vertex arrays, one per segment. Each is (8, 3) for 8 vertices.
    """
    if len(cross_section) != 4:
        raise ValueError("Cross section must have exactly 4 points for a trapezoid")

    if len(base_points) != len(normals):
        raise ValueError("Number of base points must match number of normals")

    if base_scales is not None and len(base_scales) != len(base_points):
        raise ValueError("Number of base scales must match number of base points")

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

        scale_factor = (
            base_scales[i] if base_scales is not None else 1.0
        )  # Per-point scale for the cross-section
        scaled_cross_section = cross_section * scale_factor

        cross_section_3d = np.concatenate(
            [scaled_cross_section, np.zeros((4, 1))], axis=1
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
    cross_section, base_points, normals, close_loop=False, base_scales=None
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
        base_scales (np.ndarray | list | None): Optional per-point scale factors
            applied to the cross-section before transformation. If None, all
            sections use scale 1.0.

    Returns:
        list of dicts: Each dict contains:
            "vertexes": a dict with keys 0-7 for the vertex coordinates of the trapezoid corners (as tuples)
            "faces": a dict with keys 0-11 with faces defined by vertex indices (triangulated faces)

    Note:
        When close_loop=True, the last segment connects the final cross-section to the first one.
        For geometries like Möbius strips where the cross-sections may be rotated relative to each other,
        the propagate_consistent_winding function automatically handles vertex correspondence to ensure
        proper mesh topology without gaps or overlaps.
        Any provided base_scales are respected for the closing segment as well.
    """
    # First, generate all vertices for each base point
    all_vertex_sets = create_snake_vertices(
        cross_section, base_points, normals, base_scales=base_scales
    )

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
def _sample_bezier_chain_3d(segments, samples_per_segment=40, return_segments=False):
    pts = []
    segment_samples = []
    for seg in segments:
        t = np.linspace(0, 1, samples_per_segment)
        seg_pts = _bez_eval_3d(seg, t)
        pts.append(seg_pts)
        if return_segments:
            segment_samples.append(seg_pts)
    concatenated = np.concatenate(pts, axis=0)
    if return_segments:
        return concatenated, segment_samples
    return concatenated


# ------------------------------------------------------------
# 4. Scale interpolation along the sampled curve
# ------------------------------------------------------------
def _interpolate_scales_along_segment(seg_points, start_scale, end_scale):
    """Linearly interpolate scale along a segment using arc-length parameterization."""
    seg_points = np.asarray(seg_points, float)
    if len(seg_points) == 0:
        return np.array([], dtype=float)
    if len(seg_points) == 1:
        return np.array([start_scale], dtype=float)

    diffs = np.diff(seg_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-12:
        progress = np.zeros_like(cumulative)
    else:
        progress = cumulative / total_length

    return start_scale + (end_scale - start_scale) * progress


def _interpolate_scales_along_chain(segment_samples, control_scales):
    """Generate per-sample scales for the full chain, matching sampled points."""
    expected_points = len(segment_samples) + 1
    if len(control_scales) != expected_points:
        raise ValueError(
            f"Expected {expected_points} control scales, got {len(control_scales)}"
        )

    per_segment_scales = []
    for idx, seg_pts in enumerate(segment_samples):
        start_scale = control_scales[idx]
        end_scale = control_scales[idx + 1]
        per_segment_scales.append(
            _interpolate_scales_along_segment(seg_pts, start_scale, end_scale)
        )

    return np.concatenate(per_segment_scales, axis=0)


# ------------------------------------------------------------
# 5. Bishop frame normals (rotation-minimizing)
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


def create_bezier_snake_geometry(
    points,
    cross_section,
    samples_per_segment=40,
    tau=0.5,
    initial_normal=(0, 0, 1),
    close_loop=False,
):
    """
    Generate a trapezoidal "snake" mesh that follows a poly-Bezier path.

    Builds an Illustrator-style cubic Bezier chain from the provided control
    points, samples it into base points, computes rotation-minimizing (Bishop)
    normals, and feeds everything into `create_trapezoidal_snake_geometry` to
    produce a list of mesh segments.

    Args:
        points (list[dict]): Control points in order along the path, each with a
            required `p` (x, y, z) coordinate and optional `in`/`out` handle
            vectors. `in` and `out` are world-space offsets from `p` (not
            absolute positions) that set the incoming and outgoing tangents for
            cubic Beziers; set them to (0, 0, 0) for a sharp corner. Missing
            handles are auto-generated using Catmull-Rom tangents scaled by
            `tau`. An optional `scale` factor (default 1.0) scales the
            cross-section at that control point; scales are interpolated in arc
            length between points.
        cross_section (np.ndarray): (4, 2) trapezoid in the local XY plane that
            will be swept along the Bezier chain.
        samples_per_segment (int): Number of samples per Bezier segment used to
            build the path that the cross-section is extruded along.
        tau (float): Tension factor for the auto-generated Bezier handles.
            Higher values produce tighter curves.
        initial_normal (tuple): Starting normal for the Bishop frame that keeps
            the cross-section's orientation stable along the path. Use a vector
            that is not collinear with the first segment direction.
        close_loop (bool): When True, connects the final sampled cross-section
            back to the first one and fixes winding so the mesh closes cleanly.

    Returns:
        list[dict]: Mesh segment dictionaries exactly as returned by
            `create_trapezoidal_snake_geometry`, ready for conversion into solids.

    Examples:
        # Simple smooth chain (handles auto-generated)
        cross = np.array([[-2, 0], [2, 0], [1.5, 1.5], [-1.5, 1.5]], dtype=float)
        pts = [{"p": (0, 0, 0)}, {"p": (20, 0, 10)}, {"p": (40, 10, 0)}]
        meshes = create_bezier_snake_geometry(pts, cross_section=cross, samples_per_segment=30)

        # Force a vertical lift before turning right: start tangent points straight up
        pts = [
            {"p": (0, 0, 0), "out": (0, 0, 20)},          # exit upward 20 units
            {"p": (40, 0, 20)},                           # first bend anchor
            {"p": (80, 40, 20)},                          # continues right
        ]
        meshes = create_bezier_snake_geometry(pts, cross_section=cross)

        # Land horizontally at the end even when coming from above:
        pts = [
            {"p": (0, 0, 0)},
            {"p": (40, 0, 40)},                           # high approach
            {"p": (80, 0, 0), "in": (-20, 0, 0)},         # arrive with flat (XY) tangent
        ]
        meshes = create_bezier_snake_geometry(pts, cross_section=cross)

        # Scale cross-section along the path (1x -> 2x):
        pts = [
            {"p": (0, 0, 0), "scale": 1.0},
            {"p": (60, 0, 0), "scale": 2.0},
        ]
        meshes = create_bezier_snake_geometry(
            pts, cross_section=cross, samples_per_segment=20
        )
    """

    # Build & sample curve
    segments = _build_bezier_chain_3d(points, tau=tau)
    control_scales = np.array([p.get("scale", 1.0) for p in points], dtype=float)
    base_pts, segment_samples = _sample_bezier_chain_3d(
        segments, samples_per_segment, return_segments=True
    )
    base_scales = _interpolate_scales_along_chain(
        segment_samples, control_scales=control_scales
    )

    # Compute normals
    normals = compute_bishop_normals(base_pts, initial_normal=initial_normal)

    # Call trapezoidal snake generator
    return create_trapezoidal_snake_geometry(
        cross_section=cross_section,
        base_points=base_pts,
        normals=normals,
        close_loop=close_loop,
        base_scales=base_scales,
    )
