import numpy as np
from shellforgepy.geometry.mesh_utils import validate_and_fix_mesh_segment
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
)
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    propagate_consistent_winding,
)


def normalize(v):
    """Normalize a vector. Local copy to avoid circular imports."""
    vec = np.asarray(v, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


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

        # Detect and fix any twisted vertex correspondence (e.g., in Möbius strips)
        corrected_last, corrected_first, twist_info = validate_and_fix_mesh_segment(
            last_vertices, first_vertices, tolerance=1e-6
        )

        # Create vertex map for closing segment (8 vertices: 4 at end + 4 at start)
        vertices = {}
        for j in range(4):
            vertices[j] = tuple(corrected_last[j])  # Last cross-section (indices 0-3)
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
