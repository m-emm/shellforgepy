"""
Minimal binary STL writer from plain Python lists.

- vertices: list[tuple[float, float, float]]
- triangles: list[tuple[int, int, int]]  (indices into vertices)
"""

from collections import defaultdict, deque
from math import sqrt
from struct import pack
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from shellforgepy.construct.construct_utils import normalize_edge, polygon_edges

Vec3 = Tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    n = _norm(v)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / n
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def merge_meshes(vertices_1, faces_1, vertices_2, faces_2, tol):
    """
    Merge two meshes, combining vertices that are within tolerance.

    Args:
        vertices_1: List of vertices for first mesh [(x, y, z), ...]
        faces_1: List of faces for first mesh [(i0, i1, i2), ...]
        vertices_2: List of vertices for second mesh [(x, y, z), ...]
        faces_2: List of faces for second mesh [(i0, i1, i2), ...]
        tol: Distance tolerance for merging vertices

    Returns:
        Tuple of (merged_vertices, merged_faces)
    """
    merged_vertices = []
    merged_faces = []

    # Map from (mesh_id, original_vertex_index) to new_vertex_index
    vertex_index_map = {}

    def find_or_add_vertex(vertex, mesh_id, original_index):
        """Find existing vertex within tolerance or add new one."""
        # Check if this vertex is close to any existing vertex
        for i, existing_vertex in enumerate(merged_vertices):
            dist_sq = (
                (vertex[0] - existing_vertex[0]) ** 2
                + (vertex[1] - existing_vertex[1]) ** 2
                + (vertex[2] - existing_vertex[2]) ** 2
            )
            if dist_sq <= tol * tol:
                vertex_index_map[(mesh_id, original_index)] = i
                return i

        # No close vertex found, add new one
        new_index = len(merged_vertices)
        merged_vertices.append(vertex)
        vertex_index_map[(mesh_id, original_index)] = new_index
        return new_index

    # Process first mesh
    for i, vertex in enumerate(vertices_1):
        find_or_add_vertex(vertex, 1, i)

    # Process second mesh
    for i, vertex in enumerate(vertices_2):
        find_or_add_vertex(vertex, 2, i)

    # Process faces from first mesh
    for face in faces_1:
        new_face = tuple(vertex_index_map[(1, vertex_idx)] for vertex_idx in face)
        merged_faces.append(new_face)

    # Process faces from second mesh
    for face in faces_2:
        new_face = tuple(vertex_index_map[(2, vertex_idx)] for vertex_idx in face)
        merged_faces.append(new_face)

    return merged_vertices, merged_faces


def write_stl_binary(
    path: str,
    vertices: Sequence[Vec3],
    triangles: Iterable[Tuple[int, int, int]],
    header_text: str = "created by plain-python stl",
    compute_normals: bool = True,
):
    """
    Write a binary STL file.

    Notes:
      - Triangle winding should be counter-clockwise when viewed from outside.
      - Units are arbitrary (STL has no unit metadata).
      - If compute_normals=False, normals are written as zeroes; many tools re-compute.
    """
    # 80-byte header
    header = (header_text[:80]).ljust(80, "\0").encode("ascii", errors="replace")

    # We may need to count triangles; collect them into a list if not already
    tris: List[Tuple[int, int, int]] = list(triangles)
    tri_count = len(tris)

    with open(path, "wb") as f:
        f.write(header)
        f.write(pack("<I", tri_count))  # uint32 number of triangles

        for i0, i1, i2 in tris:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]

            if compute_normals:
                # normal = normalized( (v1-v0) x (v2-v0) )
                n = _normalize(_cross(_sub(v1, v0), _sub(v2, v0)))
            else:
                n = (0.0, 0.0, 0.0)

            # Write: normal (3 floats), v0 (3f), v1 (3f), v2 (3f), attribute uint16
            f.write(pack("<3f", *n))
            f.write(pack("<3f", *v0))
            f.write(pack("<3f", *v1))
            f.write(pack("<3f", *v2))
            f.write(pack("<H", 0))  # attribute byte count (unused)


def shell_maps_to_unified_mesh(
    shell_maps: Dict[int, Dict[str, Any]],
    remove_inner_faces: bool = True,
    merge_duplicate_vertices: bool = True,
    tolerance: float = 1e-6,
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """
    Convert shell maps from get_transformed_materialized_shell_maps to a unified mesh.

    This function processes the output of PartitionableSpheroidTriangleMesh.calculate_materialized_shell_maps()
    and creates a single unified mesh suitable for STL export.

    Args:
        shell_maps: Dictionary mapping face indices to shell geometry data.
                   Each shell_map contains 'vertexes' (dict of vertex positions)
                   and 'faces' (dict of triangle indices).
        remove_inner_faces: If True, removes internal faces between adjacent shells.
                          This creates a hollow shell suitable for 3D printing.
        merge_duplicate_vertices: If True, merges vertices that are closer than tolerance.
        tolerance: Distance threshold for considering vertices as duplicates.

    Returns:
        Tuple of (vertices, triangles) where:
        - vertices: List of (x, y, z) coordinate tuples
        - triangles: List of (i0, i1, i2) index tuples referencing vertices

    Examples:
        >>> shell_maps, _ = mesh.calculate_materialized_shell_maps(shell_thickness=2.0)
        >>> vertices, triangles = shell_maps_to_unified_mesh(shell_maps)
        >>> write_stl_binary("output.stl", vertices, triangles)
    """
    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    # Track which faces are internal (shared between adjacent shells)
    internal_faces = set()

    if remove_inner_faces:
        # Build mapping of face vertex sets to identify internal faces
        face_vertex_sets = {}
        for shell_idx, shell_data in shell_maps.items():
            vertices_dict = shell_data["vertexes"]
            faces_dict = shell_data["faces"]

            for face_idx, face_vertices in faces_dict.items():
                # Get actual vertex positions for this face
                face_positions = tuple(
                    sorted(tuple(vertices_dict[v_idx]) for v_idx in face_vertices)
                )

                if face_positions in face_vertex_sets:
                    # This face geometry appears twice - mark both as internal
                    prev_shell, prev_face = face_vertex_sets[face_positions]
                    internal_faces.add((prev_shell, prev_face))
                    internal_faces.add((shell_idx, face_idx))
                else:
                    face_vertex_sets[face_positions] = (shell_idx, face_idx)

    # Process each shell map
    for shell_idx, shell_data in shell_maps.items():
        vertices_dict = shell_data["vertexes"]
        faces_dict = shell_data["faces"]

        # Convert vertices to list format and build index mapping
        vertex_list = []
        vertex_index_map = {}

        for v_idx, vertex_pos in vertices_dict.items():
            # Convert numpy array to tuple if needed
            if hasattr(vertex_pos, "tolist"):
                vertex_pos = tuple(vertex_pos.tolist())
            elif isinstance(vertex_pos, (list, np.ndarray)):
                vertex_pos = tuple(float(x) for x in vertex_pos)

            vertex_index_map[v_idx] = len(vertex_list)
            vertex_list.append(vertex_pos)

        # Add triangles, skipping internal faces if requested
        for face_idx, face_vertices in faces_dict.items():
            if remove_inner_faces and (shell_idx, face_idx) in internal_faces:
                continue

            # Map local vertex indices to global indices
            global_indices = tuple(
                vertex_offset + vertex_index_map[v_idx] for v_idx in face_vertices
            )
            all_triangles.append(global_indices)

        # Add vertices to global list
        all_vertices.extend(vertex_list)
        vertex_offset += len(vertex_list)

    # Merge duplicate vertices if requested
    if merge_duplicate_vertices and all_vertices:
        all_vertices, all_triangles = _merge_duplicate_vertices(
            all_vertices, all_triangles, tolerance
        )

    return all_vertices, all_triangles


def _merge_duplicate_vertices(
    vertices: List[Vec3], triangles: List[Tuple[int, int, int]], tolerance: float
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """
    Merge vertices that are closer than tolerance and update triangle indices.

    Args:
        vertices: List of vertex positions
        triangles: List of triangle vertex indices
        tolerance: Distance threshold for merging vertices

    Returns:
        Tuple of (merged_vertices, updated_triangles)
    """
    if not vertices:
        return vertices, triangles

    # Build spatial hash for efficient duplicate detection
    vertex_groups = {}

    # Group vertices by spatial hash
    for i, vertex in enumerate(vertices):
        # Create hash key based on rounded coordinates
        hash_key = tuple(int(coord / tolerance) for coord in vertex)
        if hash_key not in vertex_groups:
            vertex_groups[hash_key] = []
        vertex_groups[hash_key].append((i, vertex))

    # Find duplicates within each group
    vertex_mapping = {}  # old_index -> new_index
    merged_vertices = []

    for group in vertex_groups.values():
        if len(group) == 1:
            # No duplicates in this group
            old_idx, vertex = group[0]
            vertex_mapping[old_idx] = len(merged_vertices)
            merged_vertices.append(vertex)
        else:
            # Check for actual duplicates within tolerance
            processed = set()
            for i, (idx1, v1) in enumerate(group):
                if idx1 in processed:
                    continue

                # This vertex becomes the representative
                new_idx = len(merged_vertices)
                merged_vertices.append(v1)
                vertex_mapping[idx1] = new_idx
                processed.add(idx1)

                # Find all vertices within tolerance of this one
                for j, (idx2, v2) in enumerate(group[i + 1 :], i + 1):
                    if idx2 in processed:
                        continue

                    dist_sq = sum((a - b) ** 2 for a, b in zip(v1, v2))
                    if dist_sq <= tolerance * tolerance:
                        vertex_mapping[idx2] = new_idx
                        processed.add(idx2)

    # Update triangle indices
    updated_triangles = []
    for triangle in triangles:
        new_triangle = tuple(vertex_mapping[old_idx] for old_idx in triangle)
        # Skip degenerate triangles (where vertices collapsed to same point)
        if len(set(new_triangle)) == 3:
            updated_triangles.append(new_triangle)

    return merged_vertices, updated_triangles


def write_shell_maps_to_stl(
    path: str,
    shell_maps: Dict[int, Dict[str, Any]],
    header_text: str = "3D printed shell mesh",
    remove_inner_faces: bool = True,
    merge_duplicate_vertices: bool = True,
    tolerance: float = 1e-6,
    compute_normals: bool = True,
):
    """
    Convenience function to write shell maps directly to STL file.

    This combines shell_maps_to_unified_mesh() and write_stl_binary() for easy STL export
    of shell geometries generated by PartitionableSpheroidTriangleMesh.

    Args:
        path: Output STL file path
        shell_maps: Output from calculate_materialized_shell_maps()
        header_text: STL header text (max 80 characters)
        remove_inner_faces: Remove internal faces for hollow shells
        merge_duplicate_vertices: Merge nearby vertices to reduce file size
        tolerance: Distance threshold for vertex merging
        compute_normals: Whether to compute triangle normals

    Examples:
        >>> mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
        >>> shell_maps, _ = mesh.calculate_materialized_shell_maps(shell_thickness=2.0)
        >>> write_shell_maps_to_stl("hollow_sphere.stl", shell_maps)
    """
    vertices, triangles = shell_maps_to_unified_mesh(
        shell_maps,
        remove_inner_faces=remove_inner_faces,
        merge_duplicate_vertices=merge_duplicate_vertices,
        tolerance=tolerance,
    )

    write_stl_binary(
        path,
        vertices,
        triangles,
        header_text=header_text,
        compute_normals=compute_normals,
    )


def detect_twisted_vertices(vertices_start, vertices_end, tolerance=1e-6):
    """
    Detect if two cross-sections have twisted vertex correspondence.

    This function checks if the vertices in two cross-sections correspond properly
    or if they are rotated/twisted relative to each other. This is common in
    geometries like Möbius strips where the cross-section rotates as it follows
    the path.

    Args:
        vertices_start (array-like): First cross-section vertices (N, 3)
        vertices_end (array-like): Second cross-section vertices (N, 3)
        tolerance (float): Distance tolerance for detecting correspondence

    Returns:
        dict: Contains:
            - 'is_twisted': bool indicating if vertices are twisted
            - 'best_rotation': int indicating the rotation offset (0 to N-1)
            - 'distances': list of minimum distances for each rotation
            - 'best_distance': float indicating the best match distance

    Algorithm:
        For each possible rotation (0 to N-1), compute the sum of distances
        between corresponding vertices. The rotation with minimum total distance
        indicates the best correspondence. If this rotation is non-zero, the
        vertices are twisted.
    """
    vertices_start = np.array(vertices_start)
    vertices_end = np.array(vertices_end)

    if len(vertices_start) != len(vertices_end):
        raise ValueError("Both cross-sections must have the same number of vertices")

    n_vertices = len(vertices_start)
    if n_vertices == 0:
        return {
            "is_twisted": False,
            "best_rotation": 0,
            "distances": [],
            "best_distance": 0.0,
        }

    # Try all possible rotations
    rotation_distances = []

    for rotation in range(n_vertices):
        total_distance = 0.0
        for i in range(n_vertices):
            # Compare vertex i from start with vertex (i + rotation) % n from end
            rotated_index = (i + rotation) % n_vertices
            distance = np.linalg.norm(vertices_start[i] - vertices_end[rotated_index])
            total_distance += distance
        rotation_distances.append(total_distance)

    # Find the rotation with minimum total distance
    best_rotation = np.argmin(rotation_distances)
    best_distance = rotation_distances[best_rotation]

    # Check if vertices are twisted (best rotation is not 0)
    is_twisted = (best_rotation != 0) and (best_distance < rotation_distances[0])

    return {
        "is_twisted": is_twisted,
        "best_rotation": best_rotation,
        "distances": rotation_distances,
        "best_distance": best_distance,
    }


def fix_twisted_vertex_correspondence(vertices, best_rotation):
    """
    Fix twisted vertex correspondence by rotating vertex indices.

    This function reorders vertices to fix twisted correspondence detected by
    detect_twisted_vertices(). It rotates the vertex order by the specified
    rotation amount.

    Args:
        vertices (array-like): Vertices to reorder (N, 3)
        best_rotation (int): Rotation amount from detect_twisted_vertices()

    Returns:
        np.ndarray: Reordered vertices with corrected correspondence

    Example:
        If vertices = [A, B, C, D] and best_rotation = 2,
        returns [C, D, A, B] (rotated by 2 positions)
    """
    vertices = np.array(vertices)
    n_vertices = len(vertices)

    if best_rotation == 0 or n_vertices == 0:
        return vertices  # No rotation needed

    # Rotate the vertex order
    rotated_vertices = np.zeros_like(vertices)
    for i in range(n_vertices):
        rotated_index = (i + best_rotation) % n_vertices
        rotated_vertices[i] = vertices[rotated_index]

    return rotated_vertices


def validate_and_fix_mesh_segment(vertices_start, vertices_end, tolerance=1e-6):
    """
    Validate and fix vertex correspondence in a mesh segment.

    This is the main function to use when connecting two cross-sections that
    might have twisted vertex correspondence. It detects the twist and automatically
    fixes it by reordering the vertices.

    Args:
        vertices_start (array-like): First cross-section vertices (N, 3)
        vertices_end (array-like): Second cross-section vertices (N, 3)
        tolerance (float): Distance tolerance for twist detection

    Returns:
        tuple: (corrected_vertices_start, corrected_vertices_end, twist_info)
            - corrected_vertices_start: Original or corrected start vertices
            - corrected_vertices_end: Original or corrected end vertices
            - twist_info: Dictionary from detect_twisted_vertices()

    Example:
        >>> start_verts = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        >>> end_verts = [[1,1,1], [0,1,1], [0,0,1], [1,0,1]]  # Rotated by 180°
        >>> fixed_start, fixed_end, info = validate_and_fix_mesh_segment(start_verts, end_verts)
        >>> print(f"Was twisted: {info['is_twisted']}, rotation: {info['best_rotation']}")
    """
    twist_info = detect_twisted_vertices(vertices_start, vertices_end, tolerance)

    corrected_vertices_start = np.array(vertices_start)
    corrected_vertices_end = np.array(vertices_end)

    if twist_info["is_twisted"]:
        # Fix the twist by rotating the end vertices
        corrected_vertices_end = fix_twisted_vertex_correspondence(
            vertices_end, twist_info["best_rotation"]
        )

    return corrected_vertices_start, corrected_vertices_end, twist_info


def convert_to_traditional_face_vertex_maps(vertices, faces):
    """
    Convert vertices and faces to traditional face-vertex maps.
    The resulting maps are JSON-friendly, they use string keys.

    Args:
        vertices: List of (x, y, z) vertex tuples
        faces: List of (i0, i1, i2..., iN) index tuples referencing vertices

    Returns:
        dict: {
            'vertexes': {vertex_index: (x, y, z), ...},
            'faces': {face_index: (i0, i1, i2), ...}
        }
    """
    vertex_map = {str(i): tuple(v) for i, v in enumerate(vertices)}
    face_map = {str(i): tuple(face) for i, face in enumerate(faces)}

    return {"vertexes": vertex_map, "faces": face_map}


def calc_edge_to_polygon_map(polygons):
    edge_to_tri = defaultdict(list)

    for i, tri in enumerate(polygons):
        for edge in polygon_edges(tri):
            edge_to_tri[normalize_edge(*edge)].append(i)

    return edge_to_tri


def propagate_consistent_winding(polygons):
    """
    Propagate consistent winding order across all polygons in a mesh using breadth-first traversal.

    This function ensures that all polygons in a mesh have consistent vertex winding order
    (all clockwise or all counter-clockwise) by propagating the winding direction from a
    starting polygon to its neighbors through shared edges. This is essential for ensuring
    consistent surface normal directions in 3D meshes.

    Algorithm Overview:
    1. Start with polygon 0 as the reference winding direction
    2. Use breadth-first search to traverse all connected polygons
    3. For each shared edge between current and neighbor polygons:
       - If the edge appears in the same direction in both polygons, flip the neighbor
       - If the edge appears in opposite directions, keep the neighbor unchanged
    4. Continue until all connected polygons are processed

    Args:
        polygons (array-like): Collection of polygons, where each polygon is represented
                              as a sequence of 3 vertex indices [v0, v1, v2]. The input
                              can be a numpy array, list of lists, or any nested sequence.

    Returns:
        numpy.ndarray: Array of polygons with consistent winding order. The output has
                      the same shape as the input but with corrected vertex ordering.
                      polygon 0 retains its original winding direction, and all other
                      polygons are adjusted to maintain consistency.

    Implementation Details:
        - Uses an edge-to-polygon mapping for efficient neighbor lookup
        - Employs breadth-first search to ensure systematic propagation
        - Handles disconnected mesh components by starting from polygon 0
        - Preserves the original winding direction of the first polygon
        - Shared edges must appear in opposite directions for consistent winding

    Edge Direction Logic:
        For two polygons sharing an edge (a,b):
        - If both polygons traverse the edge as (a→b), they have the same winding
        - If one traverses (a→b) and the other (b→a), they have opposite winding
        - Consistent winding requires opposite edge directions between neighbors

    Example:
        >>> polygons = [[0, 1, 2], [1, 3, 2], [2, 3, 4]]
        >>> consistent = propagate_consistent_winding(polygons)
        >>> # Result ensures all polygons have consistent surface normal direction

    Note:
        This function assumes the mesh is manifold (each edge shared by at most 2 polygons).
        For non-manifold meshes, the behavior is undefined and may not produce the expected
        consistent winding across all components.
    """
    polygons = [list(tri) for tri in polygons]
    edge_to_tri = calc_edge_to_polygon_map(polygons)

    visited = set()
    queue = deque([0])  # Start with polygon 0
    visited.add(0)

    while queue:
        current = queue.popleft()
        current_polygon = polygons[current]

        n = len(current_polygon)
        if n < 3:
            raise ValueError("Polygons must have at least 3 vertices")
        for i in range(n):
            a, b = current_polygon[i], current_polygon[(i + 1) % n]
            edge_key = normalize_edge(a, b)
            neighbors = edge_to_tri[edge_key]

            for neighbor in neighbors:
                if neighbor == current or neighbor in visited:
                    continue

                neighbor_polygon = polygons[neighbor]

                # Find how this edge appears in neighbor
                n_neighbor = len(neighbor_polygon)
                if n_neighbor < 3:
                    raise ValueError("Polygons must have at least 3 vertices")
                for j in range(n_neighbor):
                    na, nb = neighbor_polygon[j], neighbor_polygon[(j + 1) % n_neighbor]
                    if {a, b} == {na, nb}:
                        if (a, b) == (na, nb):
                            # Same direction: flip neighbor
                            polygons[neighbor] = neighbor_polygon[::-1]
                        break

                visited.add(neighbor)
                queue.append(neighbor)

    return [list(v) for v in polygons]


def calc_distance_to_path(vertices, edges_on_path, point):
    """Calc the minimum distance from a point to a path defined by vertices and edges."""
    min_dist_sq = float("inf")
    point = np.array(point)

    for edge in edges_on_path:
        v0 = np.array(vertices[edge[0]])
        v1 = np.array(vertices[edge[1]])

        edge_vec = v1 - v0
        edge_len_sq = np.dot(edge_vec, edge_vec)
        if edge_len_sq == 0.0:
            # Degenerate edge, treat as point
            dist_sq = np.dot(point - v0, point - v0)
        else:
            t = np.dot(point - v0, edge_vec) / edge_len_sq
            t = max(0.0, min(1.0, t))  # Clamp to segment
            projection = v0 + t * edge_vec
            dist_sq = np.dot(point - projection, point - projection)

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return sqrt(min_dist_sq)
