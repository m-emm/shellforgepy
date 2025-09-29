import logging
from collections import Counter, defaultdict, deque
from typing import List, Optional

import networkx as nx
import numpy as np
from shellforgepy.construct.construct_utils import (
    compute_triangle_normal,
    fibonacci_sphere,
    normalize,
    normalize_edge,
    rotation_matrix_from_vectors,
    triangle_edges,
)
from shellforgepy.geometry.spherical_tools import cartesian_to_spherical_jackson
from shellforgepy.shells.connector_hint import ConnectorHint
from shellforgepy.shells.connector_utils import (
    compute_connector_hints_from_shell_maps,
    merge_collinear_connectors,
)
from shellforgepy.shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)
from shellforgepy.shells.region_edge_feature import RegionEdgeFeature

_logger = logging.getLogger(__name__)


def point_inside_cylinder(point, bottom, axis, height, radius, epsilon=1e-9):
    """
    Return True if the point lies inside or on the surface of the finite cylinder.
    """
    v = point - bottom
    axis = normalize(axis)
    height_proj = np.dot(v, axis)
    if not (0 - epsilon <= height_proj <= height + epsilon):
        return False
    radial_vec = v - height_proj * axis
    return np.linalg.norm(radial_vec) <= radius + epsilon


def are_collinear(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, tol: float = 1e-3
) -> bool:
    """
    Checks if two line segments lie on the same line (direction and position).
    """
    dir1 = normalize(p2 - p1)
    dir2 = normalize(q2 - q1)

    if abs(np.dot(dir1, dir2)) < 1.0 - tol:
        return False

    for point in [q1, q2]:
        to_point = point - p1
        if np.linalg.norm(np.cross(dir1, to_point)) > tol:
            return False

    return True


class MeshPartition:

    def __init__(
        self, mesh: PartitionableSpheroidTriangleMesh, face_to_region_map=None
    ):

        self.mesh = mesh
        if face_to_region_map is None:
            # Trivial partition: all faces in region 0
            self.face_to_region_map = {i: 0 for i in range(len(mesh.faces))}
        else:
            self.face_to_region_map = face_to_region_map

        for f_idx in range(len(mesh.faces)):
            if f_idx not in self.face_to_region_map:
                raise ValueError(
                    f"Face index {f_idx} is not mapped to any region. "
                    "Ensure all faces are assigned a region."
                )

        self.boundary_edges = self._compute_boundary_edges()
        self.face_graph = self._build_face_graph()
        self.edge_graph = self._build_edge_graph()

    def _build_face_graph(self):
        G = nx.Graph()
        region_faces = self.face_to_region_map.keys()
        edge_to_faces = defaultdict(list)

        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_to_faces[(a, b)].append(f_idx)

        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                a, b = faces
                G.add_edge(a, b)

        return G

    def _build_edge_graph(self):
        G = nx.Graph()
        region_faces = self.face_to_region_map.keys()
        V = self.mesh.vertices

        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                dist = np.linalg.norm(V[a] - V[b])
                G.add_edge(a, b, weight=dist)

        return G

    def has_region_holes(self, region_id):
        region_faces = set(self.get_faces_of_region(region_id))
        edge_count = defaultdict(int)

        for f in region_faces:
            face = self.mesh.faces[f]
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_count[(a, b)] += 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        G = nx.Graph()
        for a, b in boundary_edges:
            G.add_edge(a, b)

        return nx.number_connected_components(G) > 1

    def construct_closed_path_from_vertices(
        self,
        region_face_indices: set[int],
        vertex_set: set[int],
    ) -> list[int]:
        """
        Given a set of vertex indices assumed to lie on a closed boundary,
        attempt to construct an ordered closed path using available edges
        in the region. If gaps exist, shortest paths are inserted.

        Returns:
            A list of vertex indices forming a closed path (first == last),
            covering all vertices in vertex_set and any minimal fillers.
        """
        from heapq import heappop, heappush

        V = self.mesh.vertices
        vertex_set = set(vertex_set)

        # Step 1: Build adjacency graph from region faces
        edge_graph = defaultdict(set)
        for face_idx in region_face_indices:
            face = self.mesh.faces[face_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge_graph[a].add(b)
                edge_graph[b].add(a)

        # Step 2: Identify subpaths in the input vertex set
        unused = set(vertex_set)
        subpaths = []

        while unused:
            start = unused.pop()
            path = [start]
            current = start
            while True:
                neighbors = edge_graph[current].intersection(unused)
                if not neighbors:
                    break
                next_v = neighbors.pop()
                path.append(next_v)
                unused.remove(next_v)
                current = next_v
            subpaths.append(path)

        # Step 3: Connect subpaths by shortest paths (greedy)
        def dijkstra_path(start, end):
            dist = {start: 0.0}
            prev = {}
            heap = [(0.0, start)]
            while heap:
                d, u = heappop(heap)
                if u == end:
                    break
                for v in edge_graph[u]:
                    nd = d + np.linalg.norm(V[u] - V[v])
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heappush(heap, (nd, v))
            # Reconstruct
            if end not in prev:
                raise ValueError(f"No path found from {start} to {end}")
            path = [end]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return list(reversed(path))

        while len(subpaths) > 1:
            best_pair = None
            best_cost = float("inf")
            best_connection = None
            for i in range(len(subpaths)):
                for j in range(i + 1, len(subpaths)):
                    a_tail, a_head = subpaths[i][0], subpaths[i][-1]
                    b_tail, b_head = subpaths[j][0], subpaths[j][-1]
                    # Try head to head, tail to tail, etc.
                    for u, udir in [(a_head, "f"), (a_tail, "r")]:
                        for v, vdir in [(b_head, "f"), (b_tail, "r")]:
                            try:
                                path = dijkstra_path(u, v)
                                cost = sum(
                                    np.linalg.norm(V[path[i + 1]] - V[path[i]])
                                    for i in range(len(path) - 1)
                                )
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pair = (i, j, udir, vdir)
                                    best_connection = path
                            except ValueError:
                                continue

            # Merge the best two
            i, j, udir, vdir = best_pair
            pi, pj = subpaths[i], subpaths[j]

            # Adjust directions
            if udir == "r":
                pi = pi[::-1]
            if vdir == "r":
                pj = pj[::-1]

            merged = pi + best_connection[1:] + pj
            new_subpaths = [
                subpaths[k] for k in range(len(subpaths)) if k not in [i, j]
            ]
            new_subpaths.append(merged)
            subpaths = new_subpaths

        final_path = subpaths[0]
        if final_path[0] != final_path[-1]:
            final_path.append(final_path[0])

        return final_path

    def is_region_contiguous(self, region_id):
        region_faces = self.get_faces_of_region(region_id)
        subgraph = self.face_graph.subgraph(region_faces)
        return nx.is_connected(subgraph)

    def _compute_boundary_edges(self):
        edge_to_faces = defaultdict(set)
        for i, face in enumerate(self.mesh.faces):
            for j in range(3):
                edge = normalize_edge(face[j], face[(j + 1) % 3])
                edge_to_faces[edge].add(i)

        boundary_edges = defaultdict(set)

        for edge, adjacent_faces in edge_to_faces.items():
            if len(adjacent_faces) != 2:
                continue  # might be mesh border or degenerate

            face_list = list(adjacent_faces)
            region1 = self.face_to_region_map[face_list[0]]
            region2 = self.face_to_region_map[face_list[1]]

            if region1 != region2:
                key = frozenset([region1, region2])
                boundary_edges[key].add(edge)

        return boundary_edges

    def get_boundary_edges_of_region(self, region_id):
        edge_set = set()
        for key, edges in self.boundary_edges.items():
            if region_id in key:
                edge_set.update(edges)

        faces_in_region = self.get_faces_of_region(region_id)
        edges_in_region = set()

        for face_index in faces_in_region:
            face = self.mesh.faces[face_index]
            for i in range(3):
                edge = (face[i], face[(i + 1) % 3])
                edges_in_region.add(edge)

        decanicalized_edge_set = set()
        for edge in edge_set:
            if edge in edges_in_region:
                decanicalized_edge_set.add(edge)
            elif (edge[1], edge[0]) in edges_in_region:
                decanicalized_edge_set.add((edge[1], edge[0]))
            else:
                raise ValueError(
                    f"Edge {edge} is not part of the region {region_id} faces."
                )

        return decanicalized_edge_set

    def region_adjacency_graph(self):
        adj = defaultdict(set)
        for key in self.boundary_edges:
            a, b = tuple(key)
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def get_faces_of_region(self, region):
        return [face for face, reg in self.face_to_region_map.items() if reg == region]

    def get_region_id_of_triangle(self, triangle_index):
        """
        Returns the region ID of the triangle with the given index.
        """
        if triangle_index < 0 or triangle_index >= len(self.mesh.faces):
            raise IndexError("Triangle index out of bounds")
        return self.face_to_region_map.get(triangle_index, None)

    def get_regions(self):
        return sorted(set(self.face_to_region_map.values()))

    def get_region_area(self, region):
        faces_of_region = self.get_faces_of_region(region)
        return sum(
            self.mesh.triangle_area(self.mesh.faces[face]) for face in faces_of_region
        )

    def find_regions_of_edge(self, edge: tuple[int, int]) -> List[int]:
        """
        Returns a list of region IDs that contain the edge defined by the vertex indices.
        """
        regions = set()
        for face_index, region in self.face_to_region_map.items():
            face = self.mesh.faces[face_index]
            if edge[0] in face and edge[1] in face:
                regions.add(region)
        return tuple(sorted(regions))

    def find_regions_of_vertex_by_label(self, vertex_label: str) -> List[int]:
        """
        Returns a list of region IDs that contain the vertex with the given label.
        """
        regions = set()
        vertex_indices = self.mesh.get_vertices_by_label(vertex_label)

        # Check all faces to see if this vertex is part of them
        for face_index, region in self.face_to_region_map.items():
            face = self.mesh.faces[face_index]
            if any(v in vertex_indices for v in face):
                regions.add(region)

        return sorted(regions)

    def find_local_vertex_ids_by_label(self, vertex_label: str, region_id: int):
        """
        Returns the local vertex index in the specified region for the vertex with the given label.
        If the vertex is not found in the region, returns None.
        """

        submesh_maps = self.get_submesh_maps(region_id)

        global_vertex_indices = self.mesh.get_vertices_by_label(vertex_label)
        local_vertex_indices = [
            submesh_maps["global_to_local_vertex_map"][v]
            for v in global_vertex_indices
            if v in submesh_maps["global_to_local_vertex_map"]
        ]

        return local_vertex_indices

    def compute_connector_hints(
        self,
        shell_thickness,
        merge_connectors=False,
        min_connector_distance=None,
        min_corner_distance=None,
    ) -> list[ConnectorHint]:
        shell_maps, vertex_index_map = self.mesh.calculate_materialized_shell_maps(
            shell_thickness
        )

        connector_hints = compute_connector_hints_from_shell_maps(
            mesh_faces=self.mesh.faces,
            face_to_region=self.face_to_region_map,
            shell_maps=shell_maps,
            vertex_index_map=vertex_index_map,
        )

        if merge_connectors:
            connector_hints = merge_collinear_connectors(connector_hints)

        # filter out corner connectors

        vertex_regions = defaultdict(set)
        for face_index, region in self.face_to_region_map.items():
            face = self.mesh.faces[face_index]
            for vertex_index in face:
                vertex_regions[vertex_index].add(region)

        corner_vertices = {
            v for v, regions in vertex_regions.items() if len(regions) > 2
        }
        corner_vertex_coordinates = np.array(
            [self.mesh.vertices[v] for v in corner_vertices]
        )

        print(f"There are {len(corner_vertices)} corner vertices.")

        connector_hints = [
            hint
            for hint in connector_hints
            if not any(v in corner_vertices for v in hint.triangle_a_vertex_indices)
            and not any(v in corner_vertices for v in hint.triangle_b_vertex_indices)
        ]

        if min_corner_distance is not None:

            connector_hints = [
                hint
                for hint in connector_hints
                if all(
                    np.linalg.norm(hint.edge_centroid - corner_vertex)
                    >= min_corner_distance
                    for corner_vertex in corner_vertex_coordinates
                )
            ]

        if (min_connector_distance is not None) or (min_corner_distance is not None):
            if min_connector_distance is None:
                min_connector_distance = 0  # no limit for connector distance
            filtered_hints = []

            hints_by_region_pair = defaultdict(list)
            for hint_index, hint in enumerate(connector_hints):
                key = tuple(sorted((hint.region_a, hint.region_b)))
                hints_by_region_pair[key].append((hint_index, hint))

            for region_pair, hints_for_pair in hints_by_region_pair.items():

                edge_centroids = np.array(
                    [hint.edge_centroid for _, hint in hints_for_pair]
                )
                region_pair_centroid = np.mean(edge_centroids, axis=0)

                # find the closest hint to the centroid
                closest_hint = None
                closest_distance = np.inf
                for hint_index, hint in hints_for_pair:
                    distance = np.linalg.norm(hint.edge_centroid - region_pair_centroid)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_hint = (hint_index, hint)

                filtered_hints.append(closest_hint)

            # now try to add more hints, but only if they are sufficiently far apart and not too close to corners
            for hint_index, hint in enumerate(connector_hints):
                if hint_index in [h[0] for h in filtered_hints]:
                    continue
                far_enough_from_others = all(
                    np.linalg.norm(hint.edge_centroid - h.edge_centroid)
                    >= min_connector_distance
                    for h in [h[1] for h in filtered_hints]
                )
                far_enough_from_corners = (
                    True
                    if min_corner_distance is None
                    else all(
                        np.linalg.norm(hint.edge_centroid - corner_vertex)
                        >= min_corner_distance
                        for corner_vertex in corner_vertex_coordinates
                    )
                )
                if not far_enough_from_corners:
                    print(f"Skipping hint {hint_index} due to corner proximity.")

                if far_enough_from_others and far_enough_from_corners:
                    filtered_hints.append((hint_index, hint))

            connector_hints = [h[1] for h in filtered_hints]

        return sorted(
            connector_hints,
            key=lambda h: (h.region_a, h.region_b, tuple(h.edge_centroid)),
        )

    def get_num_faces_in_region(self, region_id):
        """
        Returns the number of faces in the specified region.
        """
        return len(self.get_faces_of_region(region_id))

    def get_submesh_maps(self, region_id):

        faces_of_region = self.get_faces_of_region(region_id)

        vertex_index_faces_of_region = [
            self.mesh.faces[face] for face in faces_of_region
        ]

        vertex_index_set = set()
        for face in vertex_index_faces_of_region:
            vertex_index_set.update(face)

        old_to_new_vertex_index_mapping = {}
        new_to_old_vertex_index_mapping = {}

        for new_vertex_index, old_vertex_index in enumerate(sorted(vertex_index_set)):
            old_to_new_vertex_index_mapping[old_vertex_index] = new_vertex_index
            new_to_old_vertex_index_mapping[new_vertex_index] = old_vertex_index

        new_faces = {}
        new_to_old_faces_index_mapping = {}
        for new_face_index, old_face_index in enumerate(faces_of_region):
            new_to_old_faces_index_mapping[new_face_index] = old_face_index
            new_faces[new_face_index] = tuple(
                old_to_new_vertex_index_mapping[vertex_index]
                for vertex_index in vertex_index_faces_of_region[new_face_index]
            )

        new_vertices = {}
        for (
            new_vertex_index,
            old_vertex_index,
        ) in new_to_old_vertex_index_mapping.items():
            new_vertices[new_vertex_index] = self.mesh.vertices[old_vertex_index]

        boundary_edges = self.get_boundary_edges_of_region(region_id)

        boundary_edges_new = {}
        for edge_number, edge in enumerate(boundary_edges):
            a, b = edge
            boundary_edges_new[edge_number] = (
                old_to_new_vertex_index_mapping[a],
                old_to_new_vertex_index_mapping[b],
            )

        maps = {
            "vertexes": new_vertices,
            "faces": new_faces,
            "boundary_edges": boundary_edges_new,
            "local_to_global_vertex_map": new_to_old_vertex_index_mapping,
            "local_to_global_face_map": new_to_old_faces_index_mapping,
            "global_to_local_vertex_map": old_to_new_vertex_index_mapping,
            "local_to_global_face_map": new_to_old_faces_index_mapping,
        }

        sorted_vertex_keys = list(sorted(maps["vertexes"].keys()))
        expected_vertex_keys = list(range(len(maps["vertexes"])))
        if not sorted_vertex_keys == expected_vertex_keys:
            raise ValueError(
                f"Vertex keys are not numbered correctly. Expected {expected_vertex_keys}, got {sorted_vertex_keys}"
            )

        return maps

    def get_region_vertices(self, region_id: int) -> np.ndarray:
        """
        Returns the vertices of the specified region as a numpy array.
        """
        faces_of_region = self.get_faces_of_region(region_id)
        if not faces_of_region:
            raise ValueError(f"Region {region_id} has no faces.")

        vertex_indices = set()
        for face_index in faces_of_region:
            face = self.mesh.faces[face_index]
            vertex_indices.update(face)

        if not vertex_indices:
            raise ValueError(f"Region {region_id} has no vertices.")

        return np.array([self.mesh.vertices[v] for v in sorted(vertex_indices)])

    def get_region_centroid(self, region_id: int) -> np.ndarray:
        """
        Computes the centroid of the specified region by averaging all vertices in the region
        """
        faces_of_region = self.get_faces_of_region(region_id)
        if not faces_of_region:
            raise ValueError(f"Region {region_id} has no faces.")

        vertex_indices = set()
        for face_index in faces_of_region:
            face = self.mesh.faces[face_index]
            vertex_indices.update(face)
        if not vertex_indices:
            raise ValueError(f"Region {region_id} has no vertices.")
        vertices = np.array([self.mesh.vertices[v] for v in vertex_indices])
        centroid = np.mean(vertices, axis=0)
        return centroid

    def split_region_by_cap(
        self,
        region_id: int,
        initial_seed_triangle_index: int,
        target_area_fraction: float,
        verbose: bool = False,
    ) -> "MeshPartition":
        """
        Splits a region by growing a spherical cap from a seed triangle centroid,
        and bisecting until the desired area fraction is reached.

        Parameters:
        - region_id: the region to split
        - initial_seed_triangle_index: triangle index (in global mesh) inside the region
        - target_area_fraction: area fraction to enclose in the cap (between 0 and 1)
        """
        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        mesh = self.mesh

        # Filter to relevant faces
        region_faces = [f for f, r in self.face_to_region_map.items() if r == region_id]
        region_set = set(region_faces)

        if initial_seed_triangle_index not in region_set:
            raise ValueError("Seed triangle must be part of the given region.")

        target_area = target_area_fraction * sum(
            mesh.triangle_area(mesh.faces[f]) for f in region_faces
        )

        seed_centroid = mesh.triangle_centroid(mesh.faces[initial_seed_triangle_index])
        R = rotation_matrix_from_vectors(seed_centroid, np.array([0, 0, 1]))

        rotated_coords = (R @ mesh.vertices.T).T
        rotated_r_theta_phi = np.array(
            [cartesian_to_spherical_jackson(v) for v in rotated_coords]
        )
        rotated_theta_phi = rotated_r_theta_phi[:, 1:3]

        # Bisection loop
        min_angle = 0
        max_angle = np.pi
        best_masked_faces = []

        while max_angle - min_angle > 1e-3:
            current_angle = (max_angle + min_angle) / 2

            # Vertex mask (by theta)
            vertex_mask = rotated_theta_phi[:, 0] <= current_angle

            # Find triangles where all 3 vertices are inside the cap
            masked_faces = []
            for f_idx in region_faces:
                tri = mesh.faces[f_idx]
                if all(vertex_mask[v] for v in tri):
                    masked_faces.append(f_idx)

            area = sum(mesh.triangle_area(mesh.faces[f]) for f in masked_faces)

            if verbose:
                print(
                    f"Cap angle: {current_angle:.4f}, area: {area:.4f}, target: {target_area:.4f}"
                )

            if area < target_area:
                min_angle = current_angle
            else:
                max_angle = current_angle
                best_masked_faces = masked_faces

        # Assign region ids
        new_region_id = max(self.get_regions()) + 1

        initial_faces_A = set(best_masked_faces)
        initial_faces_B = region_set - initial_faces_A

        print(f"Cap split with {len(initial_faces_A)} vs {len(initial_faces_B)} faces.")
        try:
            walk = self.extract_boundary_walk_between_face_sets(
                self.mesh, initial_faces_A, initial_faces_B
            )
            print(f"Extracted boundary walk: {walk}")
            tightened_A, tightened_B = self.tighten_boundary_walk(
                self.mesh,
                walk,
                segment_length=4,
                shorten_factor=0.99,
                allowed_faces=initial_faces_A | initial_faces_B,
            )
            print(f"Tightened A: {tightened_A}\nTightened B: {tightened_B}")
        except ValueError as e:
            if verbose:
                print(f"Skipping boundary tightening: {e}")
            tightened_A = initial_faces_A
            tightened_B = initial_faces_B

        final_face_to_region = dict(self.face_to_region_map)
        for f in tightened_A:
            final_face_to_region[f] = region_id
        for f in tightened_B:
            final_face_to_region[f] = new_region_id

        if verbose:
            a1 = sum(mesh.triangle_area(mesh.faces[f]) for f in tightened_A)
            a2 = sum(mesh.triangle_area(mesh.faces[f]) for f in tightened_B)
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] (tightened) | "
                f"area A: {a1:.2f}, B: {a2:.2f}, target: {target_area:.2f}"
            )

        region_sizes = Counter(final_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(mesh, final_face_to_region)

    def split_region_along_boundary_walk(
        self,
        region_id: int,
        target_area_fraction: float,
        seed_vertex: Optional[int] = None,
        verbose: bool = True,
    ) -> "MeshPartition":
        import heapq

        if not (0 < target_area_fraction < 1):
            raise ValueError("target_area_fraction must be between 0 and 1 (exclusive)")

        # Compute the absolute target area
        target_area = target_area_fraction * self.get_region_area(region_id)
        region_faces = set(self.get_faces_of_region(region_id))
        V = self.mesh.vertices

        # 1) Build edge→face map for this region
        edge_to_faces = defaultdict(list)
        for f in region_faces:
            verts = self.mesh.faces[f]
            for i in range(3):
                a, b = sorted((verts[i], verts[(i + 1) % 3]))
                edge_to_faces[(a, b)].append(f)

        # 2) Extract the closed boundary loop of vertices
        def extract_boundary_loop():
            boundary_edges = [e for e, fs in edge_to_faces.items() if len(fs) == 1]
            adj = defaultdict(list)
            for a, b in boundary_edges:
                adj[a].append(b)
                adj[b].append(a)

            start = boundary_edges[0][0]
            loop = [start]
            prev = None
            cur = start
            while True:
                nbrs = [v for v in adj[cur] if v != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == start:
                    loop.append(start)
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
            return loop

        boundary_loop = extract_boundary_loop()
        # rotate so that boundary_loop[0] == seed_vertex
        if seed_vertex is None:
            seed_vertex = boundary_loop[0]
        i0 = boundary_loop.index(seed_vertex)
        boundary_loop = boundary_loop[i0:] + boundary_loop[1 : i0 + 1]  # keep closed

        # 3) Find opposite vertex: balances left/right boundary lengths
        def path_length(path):
            return sum(
                np.linalg.norm(V[path[i + 1]] - V[path[i]])
                for i in range(len(path) - 1)
            )

        def find_opposite(loop, seed):
            idx = loop.index(seed)
            best, bd = None, float("inf")
            n = len(loop) - 1  # last repeats seed
            for j in range(1, n):
                opp = loop[j]
                # two boundary branches
                right = loop[idx : j + 1] if j > idx else loop[idx:] + loop[: j + 1]
                left = loop[j : idx + 1] if j < idx else loop[j:] + loop[: idx + 1]
                diff = abs(path_length(left) - path_length(right))
                if diff < bd:
                    bd, best = diff, opp
            return best

        opposite = find_opposite(boundary_loop, seed_vertex)

        # 4) Precompute the two boundary branches
        idx_seed = boundary_loop.index(seed_vertex)
        idx_opp = boundary_loop.index(opposite)
        if idx_seed < idx_opp:
            branch_right = boundary_loop[idx_seed : idx_opp + 1]
            branch_left = boundary_loop[idx_opp:] + boundary_loop[: idx_seed + 1]
        else:
            branch_right = boundary_loop[idx_seed:] + boundary_loop[: idx_opp + 1]
            branch_left = boundary_loop[idx_opp : idx_seed + 1]

        # 5) Build interior graph for shortest path (only internal edges)
        internal_edges = [e for e, fs in edge_to_faces.items() if len(fs) == 2]
        graph = defaultdict(list)
        for a, b in internal_edges:
            d = np.linalg.norm(V[a] - V[b])
            graph[a].append((b, d))
            graph[b].append((a, d))

        def dijkstra(s, t):
            dist = {s: 0}
            prev = {}
            heap = [(0, s)]
            while heap:
                cd, u = heapq.heappop(heap)
                if u == t:
                    break
                if cd > dist[u]:
                    continue
                for v, w in graph[u]:
                    nd = cd + w
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            # reconstruct
            path = [t]
            while path[-1] != s:
                path.append(prev[path[-1]])
            return path[::-1]

        # Helper to walk a branch up to fraction t
        def walk_frac(path, t):
            tot = path_length(path)
            goal = t * tot
            acc = 0.0
            for i in range(len(path) - 1):
                step = np.linalg.norm(V[path[i + 1]] - V[path[i]])
                if acc + step >= goal:
                    return path[i + 1]
                acc += step
            return path[-1]

        new_id = max(self.get_regions()) + 1
        min_t, max_t = 0.0, 1.0
        best_assign = set()

        # 6) Bisection on t
        for _ in range(30):
            t = 0.5 * (min_t + max_t)
            lv = walk_frac(branch_left, t)
            rv = walk_frac(branch_right, t)

            # build the 3‐segment loop: seed→…→lv, lv→…→rv (interior), rv→…→seed
            iL = boundary_loop.index(lv)
            seg1 = boundary_loop[: iL + 1]  # seed→…→lv
            seg2 = dijkstra(lv, rv)  # lv→…→rv
            iR = boundary_loop.index(rv)
            seg3 = boundary_loop[iR:]  # rv→…→seed (last element is seed)

            # assemble cut‐loop edges
            def norm(e):
                return e if e[0] < e[1] else (e[1], e[0])

            cut = set()
            for seg in (seg1, seg2, seg3):
                for a, b in zip(seg, seg[1:]):
                    cut.add(norm((a, b)))

            # build triangle adjacency skipping cut edges
            tri_adj = defaultdict(list)
            for edge, fs in edge_to_faces.items():
                ne = norm(edge)
                if len(fs) == 2 and ne not in cut:
                    f1, f2 = fs
                    tri_adj[f1].append(f2)
                    tri_adj[f2].append(f1)

            # flood‐fill from a seed triangle
            def tri_for_vert(v):
                for f in region_faces:
                    if v in self.mesh.faces[f]:
                        return f
                raise RuntimeError

            start_tri = tri_for_vert(seed_vertex)
            visited = {start_tri}
            dq = deque([start_tri])
            while dq:
                u = dq.popleft()
                for w in tri_adj[u]:
                    if w not in visited:
                        visited.add(w)
                        dq.append(w)

            area = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in visited)
            if verbose:
                print(f" t={t:.4f}  areaA={area:.1f}  target={target_area:.1f}")

            if abs(area - target_area) < 1e-3 * target_area:
                best_assign = visited
                break
            if area < target_area:
                min_t = t
            else:
                max_t = t
            best_assign = visited

        # 7) Build new face→region map
        new_map = dict(self.face_to_region_map)
        for f in best_assign:
            new_map[f] = new_id

        if verbose:
            print(
                f" New region {new_id}: {len(best_assign)} faces vs {len(region_faces)-len(best_assign)}"
            )

        return MeshPartition(self.mesh, new_map)

    def is_path_well_formed(self, region_face_indices, path):
        if len(path) < 2:
            return False

        region_vertices = set()
        edge_set = set()
        for f_idx in region_face_indices:
            face = self.mesh.faces[f_idx]
            region_vertices.update(face)
            for i in range(3):
                a, b = sorted((face[i], face[(i + 1) % 3]))
                edge_set.add((a, b))

        for a, b in zip(path, path[1:]):
            if a not in region_vertices or b not in region_vertices:
                return False
            if (a, b) not in edge_set and (b, a) not in edge_set:
                return False

        return True

    def vertex_shortest_path(self, u, v):
        return nx.shortest_path(self.edge_graph, source=u, target=v, weight="weight")

    def split_region_by_fibonacci_plane(
        self,
        region_id: int,
        target_area_fraction: float,
        samples: int = 300,
        verbose: bool = False,
    ) -> "MeshPartition":
        """
        Split a region by rotating it via Fibonacci sphere directions and slicing along the x=0 plane.
        Chooses the orientation that results in the closest match to the target area fraction.
        """

        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        view = self.region_view(region_id)
        directions = fibonacci_sphere(samples)

        best_diff = float("inf")
        best_split = None

        for d in directions:
            R = rotation_matrix_from_vectors(d, np.array([1.0, 0.0, 0.0]))  # x-axis
            A = np.eye(4)
            A[:3, :3] = R
            rotated_view = view.apply_transform(A)

            V, F, _ = rotated_view.get_transformed_vertices_faces_boundary_edges()

            area_total = 0.0
            area_A = 0.0
            faces_A = []
            faces_B = []

            for i, face in enumerate(F):
                verts = V[face]
                A_face = 0.5 * np.linalg.norm(
                    np.cross(verts[1] - verts[0], verts[2] - verts[0])
                )
                area_total += A_face
                if all(v[0] > 0 for v in verts):
                    faces_A.append(i)
                    area_A += A_face
                else:
                    faces_B.append(i)

            area_fraction = area_A / area_total if area_total > 0 else 0
            diff = abs(area_fraction - target_area_fraction)

            if diff < best_diff:
                best_diff = diff
                best_split = (faces_A, faces_B)

        # --- Build new face_to_region_map ---
        region_faces = self.get_faces_of_region(region_id)
        region_to_global_face = {i: f for i, f in enumerate(region_faces)}

        new_region_id = max(self.get_regions()) + 1
        new_face_to_region = dict(self.face_to_region_map)

        for i in best_split[0]:
            new_face_to_region[region_to_global_face[i]] = (
                region_id  # stay in original region
            )
        for i in best_split[1]:
            new_face_to_region[region_to_global_face[i]] = (
                new_region_id  # move to new region
            )

        if verbose:
            a1 = sum(
                self.mesh.triangle_area(self.mesh.faces[region_to_global_face[i]])
                for i in best_split[0]
            )
            a2 = sum(
                self.mesh.triangle_area(self.mesh.faces[region_to_global_face[i]])
                for i in best_split[1]
            )
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] | "
                f"area A: {a1:.2f}, B: {a2:.2f}, diff: {best_diff:.4f}"
            )

        region_sizes = Counter(new_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(self.mesh, new_face_to_region)

    @staticmethod
    def extract_boundary_walk_between_face_sets(mesh, faces_a, faces_b):
        """
        Given two disjoint sets of face indices, extract the boundary walk separating them.

        Returns:
            walk: list of ordered edges [(v0, v1), (v1, v2), ...]
        """
        # Collect which face owns which edge
        edge_to_faces = defaultdict(list)

        for f_idx in faces_a:
            face = mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_to_faces[edge].append(("A", f_idx))

        for f_idx in faces_b:
            face = mesh.faces[f_idx]
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_to_faces[edge].append(("B", f_idx))

        # Boundary edges: shared exactly between one face in A and one in B
        boundary_edges = []
        for edge, owners in edge_to_faces.items():
            if len(owners) == 2 and {owners[0][0], owners[1][0]} == {"A", "B"}:
                boundary_edges.append(edge)

        print(f"Boundary edges: {boundary_edges}")

        # Build adjacency graph of boundary edges
        vertex_adj = defaultdict(list)
        for a, b in boundary_edges:
            vertex_adj[a].append(b)
            vertex_adj[b].append(a)

        # Find endpoints (degree 1 vertices)
        endpoints = [v for v, neighbors in vertex_adj.items() if len(neighbors) == 1]
        if len(endpoints) == 0:
            print(f"Boundary is closed loop")
            start = next(iter(vertex_adj))  # just pick any vertex
        elif len(endpoints) > 2:
            raise ValueError(
                "Boundary walk has multiple disconnected components or branches."
            )
        else:
            start = endpoints[0]

        # Traverse the walk

        walk = []
        visited_edges = set()
        current = start

        while True:
            neighbors = [
                n
                for n in vertex_adj[current]
                if (min(current, n), max(current, n)) not in visited_edges
            ]
            if not neighbors:
                break
            next_v = neighbors[0]
            edge = (min(current, next_v), max(current, next_v))
            walk.append((current, next_v))
            visited_edges.add(edge)
            current = next_v

        return walk

    @staticmethod
    def try_shorten_segment(vertex_segment, edge_graph, shorten_factor):
        """
        Try to shorten a segment of a walk using the shortest path in the edge graph.
        If the path is shorter by the given factor, return the replacement path.
        Otherwise, return None.
        """
        if len(vertex_segment) < 2:
            return vertex_segment

        start, end = vertex_segment[0], vertex_segment[-1]

        try:
            path = nx.shortest_path(
                edge_graph, source=start, target=end, weight="weight"
            )
        except nx.NetworkXNoPath:
            print(
                f"No path found between {start} and {end}. Returning original segment."
            )
            raise

        def path_length(vs):
            return sum(
                edge_graph[vs[i]][vs[i + 1]]["weight"] for i in range(len(vs) - 1)
            )

        if path_length(path) < path_length(vertex_segment) * shorten_factor:
            return path
        return vertex_segment

    @classmethod
    def tighten_boundary_walk(
        cls, mesh, walk, allowed_faces, segment_length, shorten_factor
    ):
        """
        Tightens a walk by replacing segments with shortest paths in the edge graph,
        represented as an ordered list of vertices. Then flood-fills the triangle mesh
        into two disjoint regions using the walk as a separator.

        Returns:
            (faces_a, faces_b): sets of triangle indices assigned to each side of the walk.
        """

        # --- Build edge graph from allowed faces ---
        edge_graph = nx.Graph()
        for f_idx in allowed_faces:
            tri = mesh.faces[f_idx]
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                dist = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
                edge_graph.add_edge(a, b, weight=dist)

        # --- Convert original walk to ordered vertex list ---
        vertex_walk = [walk[0][0], walk[0][1]]
        for edge in walk[1:]:
            if edge[0] == vertex_walk[-1]:
                vertex_walk.append(edge[1])
            elif edge[1] == vertex_walk[-1]:
                vertex_walk.append(edge[0])
            else:
                raise ValueError(f"Non-contiguous edge in original walk: {edge}")

        def is_valid_vertex_walk(vertex_list):
            return all(
                edge_graph.has_edge(vertex_list[i], vertex_list[i + 1])
                for i in range(len(vertex_list) - 1)
            )

        def walk_length(vertex_list):
            return sum(
                edge_graph[vertex_list[i]][vertex_list[i + 1]]["weight"]
                for i in range(len(vertex_list) - 1)
            )

        original_walk_length = walk_length(vertex_walk)
        is_closed = vertex_walk[0] == vertex_walk[-1]
        print(
            f"Original vertex walk: {vertex_walk}, length: {original_walk_length}, closed: {is_closed}"
        )
        n = len(vertex_walk)

        segments = []

        num_segments = (n - 1) // segment_length + 1

        if num_segments >= 2:

            for i in range(num_segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, n - 1)
                segment = vertex_walk[start : end + 1]
                segments.append(segment)
            new_walk = []

            for segment in segments:

                if len(segment) < 2:
                    shortened_segment = segment
                else:

                    shortened_segment = cls.try_shorten_segment(
                        segment, edge_graph, shorten_factor
                    )

                if len(new_walk) > 0 and shortened_segment[0] == new_walk[-1]:
                    shortened_segment = shortened_segment[1:]

                new_walk.extend(shortened_segment)

            vertex_walk = new_walk

        print(f"Vertex walk after: {vertex_walk}")
        # --- Extract edge set for forbidden boundary ---
        forbidden_edges = {
            tuple(sorted((vertex_walk[i], vertex_walk[i + 1])))
            for i in range(len(vertex_walk) - 1)
        }

        # --- Build face adjacency map ---
        face_edges = {}
        face_adjacency = defaultdict(set)
        for f_idx in allowed_faces:
            tri = mesh.faces[f_idx]
            edges = {tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)}
            face_edges[f_idx] = edges

        allowed_list = list(allowed_faces)
        for i, f1 in enumerate(allowed_list):
            for j in range(i + 1, len(allowed_list)):
                f2 = allowed_list[j]
                if face_edges[f1] & face_edges[f2]:
                    face_adjacency[f1].add(f2)
                    face_adjacency[f2].add(f1)

        # --- Find seed triangles adjacent to the first walk edge ---
        first_edge = tuple(sorted((vertex_walk[0], vertex_walk[1])))
        edge_to_faces = defaultdict(set)
        for f_idx in allowed_faces:
            for e in face_edges[f_idx]:
                edge_to_faces[e].add(f_idx)

        seeds = list(edge_to_faces[first_edge])
        if len(seeds) != 2:
            raise ValueError("Cannot find two seed triangles for the first edge.")
        seed_a, seed_b = seeds

        # --- Flood fill using triangle adjacency, stopping at forbidden edges ---
        visited = set()

        def flood_fill(seed, forbidden_faces):
            region = set()
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                if current in region or current in visited:
                    continue
                visited.add(current)
                region.add(current)
                for neighbor in face_adjacency[current]:
                    shared = face_edges[current] & face_edges[neighbor]
                    if any(e in forbidden_edges for e in shared):
                        continue
                    if neighbor in forbidden_faces:
                        raise ValueError(
                            f"Flood fill leaked into forbidden face {neighbor}"
                        )

                    queue.append(neighbor)
            return region

        region_a = flood_fill(seed_a, forbidden_faces={seed_b})
        region_b = flood_fill(seed_b, forbidden_faces={seed_a})

        if not region_a or not region_b:
            raise ValueError("Flood fill failed: one region is empty.")

        assert region_a.isdisjoint(region_b), "Regions are not disjoint."
        assert (
            region_a | region_b == allowed_faces
        ), "Flood fill did not cover all allowed faces."

        print(
            f"Tightened walk length: {walk_length(vertex_walk)}, original: {original_walk_length}"
        )
        print(f"Tightened vertex walk: {vertex_walk}")
        print(f"Region A: {len(region_a)} faces, Region B: {len(region_b)} faces")

        # Use lexicographic heuristic for naming A vs B
        if min(region_a) < min(region_b):
            return region_a, region_b
        else:
            return region_b, region_a

    def split_region_by_polar_oriented_plane(
        self,
        region_id: int,
        target_area_fraction: float,
        phi: float = 0.0,
        verbose: bool = False,
        up_direction: Optional[np.ndarray] = None,
    ) -> "MeshPartition":
        """
        Rotate region to align its mean direction with Z+, apply additional Z-rotation by `phi`,
        and then split it by sweeping an x-cut plane until `target_area_fraction` is reached.

        Parameters:
        - region_id: ID of region to split
        - target_area_fraction: desired area ratio (between 0 and 1)
        - phi: extra rotation angle (in radians) around the Z-axis
        - steps: number of candidate x-cuts to try
        """

        from shellforgepy.shells.transformed_region_view import TransformedRegionView

        if not (0.0 < target_area_fraction < 1.0):
            raise ValueError("target_area_fraction must be between 0 and 1.")

        view = TransformedRegionView(self, region_id)

        # Step 1: Compute average direction of region (mean of vertex positions, normalized)
        V, F, _ = view.get_transformed_vertices_faces_boundary_edges()

        if up_direction is not None:
            up_direction = np.asarray(up_direction, dtype=np.float64)

            mean_vec = up_direction
            mean_vec /= np.linalg.norm(mean_vec)

        else:
            mean_vec = V.mean(axis=0)

            if np.linalg.norm(mean_vec) < 1e-6:
                mean_vec = np.array([0, 0, 1])  # Fallback to Z+ if region is empty
            else:
                mean_vec /= np.linalg.norm(mean_vec)

        # Step 2: Rotate mean_vec to point "up" (to Z+)
        R_align = rotation_matrix_from_vectors(mean_vec, np.array([0, 0, 1]))
        A_align = np.eye(4)
        A_align[:3, :3] = R_align

        # Step 3: Rotate around Z-axis by phi
        c, s = np.cos(phi), np.sin(phi)
        R_phi = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        A_phi = np.eye(4)
        A_phi[:3, :3] = R_phi

        # Combined transform
        A = A_phi @ A_align
        rotated_view = view.apply_transform(A)

        V_rot, F_rot, _ = rotated_view.get_transformed_vertices_faces_boundary_edges()
        region_faces = self.get_faces_of_region(region_id)
        region_to_global_face = {i: f for i, f in enumerate(region_faces)}

        min_x = np.min(V_rot[:, 0])
        max_x = np.max(V_rot[:, 0])

        best_diff = float("inf")
        best_split = None

        low = min_x
        high = max_x
        best_diff = float("inf")
        best_split = None
        epsilon = 1e-6  # convergence threshold

        while high - low > epsilon:
            x_cut = 0.5 * (low + high)

            faces_A, faces_B = [], []
            area_A, area_total = 0.0, 0.0

            for j, face in enumerate(F_rot):
                verts = V_rot[face]
                area = 0.5 * np.linalg.norm(
                    np.cross(verts[1] - verts[0], verts[2] - verts[0])
                )
                area_total += area

                triangle_centroid = np.mean(verts, axis=0)
                if triangle_centroid[0] > x_cut:
                    faces_A.append(j)
                    area_A += area
                else:
                    faces_B.append(j)

            area_fraction = area_A / area_total if area_total > 0 else 0
            diff = abs(area_fraction - target_area_fraction)

            if diff < best_diff:
                best_diff = diff
                best_split = (faces_A, faces_B)

            # Update interval
            if area_fraction < target_area_fraction:
                high = x_cut
            else:
                low = x_cut
        # Build new region mapping
        new_region_id = max(self.get_regions()) + 1

        # Reconstruct initial face sets
        initial_faces_A = {region_to_global_face[i] for i in best_split[0]}
        initial_faces_B = {region_to_global_face[i] for i in best_split[1]}

        print(f"Best split at x={0.5 * (low + high):.4f} with diff={best_diff:.4f}")
        print(f"Faces A: {initial_faces_A},\nFaces B: {initial_faces_B}")

        # Step 4: Tighten the boundary between them
        try:
            walk = self.extract_boundary_walk_between_face_sets(
                self.mesh, initial_faces_A, initial_faces_B
            )
            print(f"Extracted boundary walk: {walk}")
            tightened_A, tightened_B = self.tighten_boundary_walk(
                self.mesh,
                walk,
                initial_faces_A | initial_faces_B,
                segment_length=4,
                shorten_factor=0.9,
            )
            print(f"Tightened A: {tightened_A},\nTightened B: {tightened_B}")
        except ValueError as e:
            print(f"Skipping boundary tightening: {e}")

            tightened_A = initial_faces_A
            tightened_B = initial_faces_B

        # Step 5: Construct final face_to_region_map
        final_face_to_region = dict(self.face_to_region_map)
        for f in tightened_A:
            final_face_to_region[f] = region_id
        for f in tightened_B:
            final_face_to_region[f] = new_region_id

        if verbose:
            a1 = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in tightened_A)
            a2 = sum(self.mesh.triangle_area(self.mesh.faces[f]) for f in tightened_B)
            print(
                f"Split region {region_id} → [{region_id}, {new_region_id}] (tightened) | "
                f"area A: {a1:.2f}, B: {a2:.2f}, diff: {best_diff:.4f}"
            )

        region_sizes = Counter(final_face_to_region.values())

        print(f"Final region sizes: {region_sizes}")

        return MeshPartition(self.mesh, final_face_to_region)

    def perforated(
        self, plane_point: np.ndarray, plane_normal: np.ndarray
    ) -> "MeshPartition":
        new_mesh, face_index_mapping = self.mesh.perforate_along_plane(
            plane_point, plane_normal
        )

        new_face_to_region_map = {}

        for old_face, new_faces in face_index_mapping.items():
            region = self.face_to_region_map[old_face]
            for new_face in new_faces:
                new_face_to_region_map[new_face] = region

        return MeshPartition(new_mesh, new_face_to_region_map)

    def perforate_and_split_region_by_plane(
        self,
        region_id: int,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
    ) -> "MeshPartition":
        """
        Perforate and split a specific region by a plane.

        Parameters
        ----------
        region_id : int
            The region to be split.
        plane_point : np.ndarray
            A point on the plane.
        plane_normal : np.ndarray
            The normal vector of the plane.

        Returns
        -------
        MeshPartition
            A new mesh partition with the region split along the given plane.
        """
        # Step 1: Collect triangle indices of this region
        region_faces = [
            face_idx
            for face_idx, r in self.face_to_region_map.items()
            if r == region_id
        ]

        # Step 2: Perforate only that region
        new_mesh, face_index_map = self.mesh.perforate_along_plane(
            plane_point, plane_normal, epsilon=1e-9, triangle_indices=region_faces
        )

        # Step 3: Classify faces by side of plane
        new_face_to_region = {}
        max_region_id = max(self.face_to_region_map.values())
        new_region_id = max_region_id + 1

        for old_face_idx, new_face_indices in face_index_map.items():
            old_region = self.face_to_region_map[old_face_idx]
            if old_region != region_id:
                # This face was not part of the region to split → copy directly
                for new_face_idx in new_face_indices:
                    new_face_to_region[new_face_idx] = old_region
                continue

            # Classify new faces by centroid
            for new_face_idx in new_face_indices:
                face = new_mesh.faces[new_face_idx]
                centroid = new_mesh.vertices[face].mean(axis=0)
                signed_distance = np.dot(centroid - plane_point, plane_normal)

                if signed_distance >= 0:
                    new_face_to_region[new_face_idx] = region_id
                else:
                    new_face_to_region[new_face_idx] = new_region_id

        return MeshPartition(new_mesh, new_face_to_region)

    def perforate_and_split_region_by_cylinder(
        self,
        region_id: int,
        bottom: np.ndarray,
        axis: np.ndarray,
        height: float,
        radius: float,
        epsilon: float = 1e-9,
        min_relative_area=1e-2,
        min_angle_deg=5.0,
    ) -> "MeshPartition":

        def triangle_area_2d(p0, p1, p2):
            return 0.5 * abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            )

        def select_faces_inside_cylinder_projected_area(
            mesh,
            faces_to_consider,
            bottom,
            axis,
            height,
            radius,
            epsilon=1e-2,
            area_slack=0.0,
        ):

            radius_slack = 0.1
            # Step 1: Vertex classification
            axis = axis / np.linalg.norm(axis)
            inside_vertex_indices = set()
            for i, v in enumerate(mesh.vertices):
                vec = v - bottom
                t = np.dot(vec, axis)
                if -epsilon <= t <= height + epsilon:
                    radial = vec - axis * t
                    if (
                        np.linalg.norm(radial)
                        <= radius + epsilon + radius * radius_slack
                    ):
                        inside_vertex_indices.add(i)

            # Step 2: Collect candidate triangles
            candidate_faces = set()
            full_inside_faces = set()
            for f_idx in faces_to_consider:
                verts = mesh.faces[f_idx]
                verts_set = set(verts)
                if verts_set <= inside_vertex_indices:
                    full_inside_faces.add(f_idx)
                elif verts_set & inside_vertex_indices:
                    candidate_faces.add(f_idx)

            return full_inside_faces

        region_faces = [
            idx for idx, r in self.face_to_region_map.items() if r == region_id
        ]

        new_mesh, face_index_map = self.mesh.perforate_with_cylinder(
            bottom,
            axis,
            height,
            radius,
            epsilon=epsilon,
            triangle_indices=region_faces,
            min_relative_area=min_relative_area,
            min_angle_deg=min_angle_deg,
        )

        new_face_to_region = {}
        max_region_id = max(self.face_to_region_map.values())
        new_region_id = max_region_id + 1

        all_new_faces = set()
        for face_list in face_index_map.values():
            all_new_faces.update(face_list)

        # add all faces in the region that were not perforated
        all_new_faces.update(
            idx for idx, r in self.face_to_region_map.items() if r == region_id
        )

        selected_faces = select_faces_inside_cylinder_projected_area(
            mesh=new_mesh,
            faces_to_consider=all_new_faces,
            bottom=bottom,
            axis=axis,
            height=height,
            radius=radius,
            epsilon=epsilon,
            area_slack=0.0,
        )

        for old_face_idx, new_face_indices in face_index_map.items():
            old_region = self.face_to_region_map[old_face_idx]
            for new_face in new_face_indices:
                if old_region != region_id:
                    new_face_to_region[new_face] = old_region
                elif new_face in selected_faces:
                    new_face_to_region[new_face] = new_region_id
                else:
                    new_face_to_region[new_face] = region_id

        return MeshPartition(new_mesh, new_face_to_region)

    def find_regions_of_vertex_by_index(self, vertex_index: int) -> List[int]:
        """
        Find all regions that contain a specific vertex by its index.

        Parameters
        ----------
        vertex_index : int
            The index of the vertex to search for.

        Returns
        -------
        List[int]
            A list of region IDs that contain the specified vertex.
        """
        regions = set()
        for face_idx, region_id in self.face_to_region_map.items():
            if vertex_index in self.mesh.faces[face_idx]:
                regions.add(region_id)
        return sorted(regions)

    def drill_holes_by_label(
        self,
        center_vertex_label: str,
        radius: float,
        height: float = 1000.0,
        epsilon: float = 1e-8,
        min_relative_area=1e-2,
        min_angle_deg=5.0,
        explicit_vertex_index: Optional[int] = None,
    ) -> "MeshPartition":
        """
        Drill a cylindrical hole in all regions by removing faces within a certain radius
        from the center vertices, which are given by label, along the averaged local normal.

        Parameters
        ----------
        center_vertex_label : str
            The label of the vertex at the center of the hole.
        radius : float
            The radius of the hole (cylinder radius).
        height : float
            The height of the cylinder used to drill.
        epsilon : float
            Numerical tolerance for perforation.

        Returns
        -------
        MeshPartition
            A new mesh partition with the hole drilled (split into two regions).
        """
        # Step 1: Find the regions which contain any of the vertices with the given label
        if explicit_vertex_index is not None:
            regions = self.find_regions_of_vertex_by_index(explicit_vertex_index)
        else:
            regions = self.find_regions_of_vertex_by_label(center_vertex_label)

        if not regions:
            raise ValueError(
                f"No regions found containing vertices with label '{center_vertex_label}'"
            )

        current_partition = self

        vertex_indices = self.mesh.get_vertices_by_label(center_vertex_label)

        for center_idx in vertex_indices:

            regions = self.find_regions_of_vertex_by_index(center_idx)

            for region_id in regions:
                print(
                    f"Drilling hole at vertex index {center_idx} with label '{center_vertex_label}' into region {region_id}."
                )
                center_point = self.mesh.vertices[center_idx]

                # Step 2: Find adjacent faces in the region
                adjacent_faces = [
                    idx
                    for idx, face in enumerate(self.mesh.faces)
                    if center_idx in face
                    and self.face_to_region_map.get(idx) == region_id
                ]
                if not adjacent_faces:
                    print(
                        f"No adjacent faces found for vertex {center_idx} in region {region_id}."
                    )
                    continue
                # Step 3: Compute average normal from adjacent faces
                normals = []
                for face_idx in adjacent_faces:
                    v0, v1, v2 = self.mesh.vertices[self.mesh.faces[face_idx]]
                    normal = np.cross(v1 - v0, v2 - v0)
                    normals.append(normalize(normal))
                avg_normal = normalize(np.mean(normals, axis=0))

                # Step 4: Build the drilling cylinder
                axis_start = center_point - 0.5 * height * avg_normal

                # Step 5: Perforate and split region by cylinder
                current_partition = (
                    current_partition.perforate_and_split_region_by_cylinder(
                        region_id=region_id,
                        bottom=axis_start,
                        axis=avg_normal,
                        height=height,
                        radius=radius,
                        epsilon=epsilon,
                        min_relative_area=min_relative_area,
                        min_angle_deg=min_angle_deg,
                    )
                )

                print(
                    f"Current partition has now {len(current_partition.get_regions())} regions."
                )

        return current_partition

    def find_region_subedges_along_original_edge(
        self,
        region_id: int,
        v0: int,
        v1: int,
        epsilon: float = 1e-6,
    ):
        """
        Given two original vertex indices (v0, v1), finds the maximal connected subsegments
        in the current mesh that lie on the original edge and belong entirely to the given region.

        Returns a list of (point_a, point_b) segments, with coordinates.
        """
        V = self.mesh.vertices
        v0_coord = V[v0]
        v1_coord = V[v1]
        edge_vec = v1_coord - v0_coord
        edge_len = np.linalg.norm(edge_vec)

        if edge_len < epsilon:
            return []

        edge_dir = edge_vec / edge_len

        def project_t(p):
            """Return the t-value (0 to 1) of p along the original edge"""
            return np.dot(p - v0_coord, edge_dir) / edge_len

        # Collect all edges from triangles in the region
        region_faces = [
            idx for idx, r in self.face_to_region_map.items() if r == region_id
        ]
        region_edges = set()
        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for a, b in triangle_edges(face):
                region_edges.add(normalize_edge(a, b))

        # Check which of those edges lie on the original edge (geometrically)
        subsegments = []
        for a, b in region_edges:
            pa, pb = V[a], V[b]

            # Project both endpoints to the line
            for pt in [pa, pb]:
                proj_len = np.dot(pt - v0_coord, edge_dir)
                closest_pt = v0_coord + proj_len * edge_dir
                if np.linalg.norm(pt - closest_pt) > epsilon:
                    break
            else:
                ta = project_t(pa)
                tb = project_t(pb)
                subsegments.append((min(ta, tb), max(ta, tb), pa, pb))

        # Sort by t and merge contiguous subsegments
        subsegments.sort()
        merged_segments = []

        if not subsegments:
            return []

        current_start_t, current_end_t, current_pa, current_pb = subsegments[0]

        for next_start_t, next_end_t, next_pa, next_pb in subsegments[1:]:
            if next_start_t <= current_end_t + epsilon:
                # Extend current segment
                current_end_t = max(current_end_t, next_end_t)
                current_pb = next_pb if next_end_t >= current_end_t else current_pb
            else:
                merged_segments.append((current_pa, current_pb))
                current_start_t, current_end_t = next_start_t, next_end_t
                current_pa, current_pb = next_pa, next_pb

        merged_segments.append((current_pa, current_pb))

        return merged_segments

    def find_region_subedges_along_original_edge_indices(
        self,
        region_id: int,
        v0: int,
        v1: int,
        epsilon: float = 1e-6,
    ) -> list[tuple[int, int]]:
        """
        Like find_region_subedges_along_original_edge, but returns merged subsegments
        as vertex index pairs (vi_a, vi_b) instead of coordinate tuples.

        These are edges in the current mesh that lie on the original edge and belong
        to the given region, merged into maximal contiguous segments.
        """
        V = self.mesh.vertices
        v0_coord = V[v0]
        v1_coord = V[v1]
        edge_vec = v1_coord - v0_coord
        edge_len = np.linalg.norm(edge_vec)

        if edge_len < epsilon:
            return []

        edge_dir = edge_vec / edge_len

        def project_t(p):
            return np.dot(p - v0_coord, edge_dir) / edge_len

        # Gather edges from the region
        region_faces = [
            idx for idx, r in self.face_to_region_map.items() if r == region_id
        ]
        region_edges = set()
        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for a, b in triangle_edges(face):
                region_edges.add(normalize_edge(a, b))

        # Check which edges lie on the original edge
        subsegments = []
        for a, b in region_edges:
            pa, pb = V[a], V[b]

            if all(
                np.linalg.norm(
                    pt - (v0_coord + np.dot(pt - v0_coord, edge_dir) * edge_dir)
                )
                < epsilon
                for pt in (pa, pb)
            ):
                ta = project_t(pa)
                tb = project_t(pb)
                subsegments.append((min(ta, tb), max(ta, tb), a, b))

        # Sort and merge contiguous subsegments
        subsegments.sort()
        merged_segments = []

        if not subsegments:
            return []

        _, current_end_t, current_a, current_b = subsegments[0]

        for next_start_t, next_end_t, next_a, next_b in subsegments[1:]:
            if next_start_t <= current_end_t + epsilon:
                current_end_t = max(current_end_t, next_end_t)
                current_b = next_b if next_end_t >= current_end_t else current_b
            else:
                merged_segments.append((current_a, current_b))
                _, current_end_t, current_a, current_b = (
                    next_start_t,
                    next_end_t,
                    next_a,
                    next_b,
                )

        merged_segments.append((current_a, current_b))

        return merged_segments

    def find_region_edge_features(
        self, region_id: int, epsilon=1e-6
    ) -> list[RegionEdgeFeature]:
        region_faces = [f for f, r in self.face_to_region_map.items() if r == region_id]

        edge_to_faces = defaultdict(list)
        for f_idx in region_faces:
            face = self.mesh.faces[f_idx]
            for a, b in triangle_edges(face):
                edge = normalize_edge(a, b)
                edge_to_faces[edge].append(f_idx)

        features = []

        for (vi1, vi2), face_ids in edge_to_faces.items():
            p1, p2 = self.mesh.vertices[vi1], self.mesh.vertices[vi2]
            edge_vec = normalize(p2 - p1)
            edge_mid = (p1 + p2) / 2

            face_geometry = []
            face_normals = []
            for f_idx in face_ids:
                verts = self.mesh.faces[f_idx]
                tri = tuple(self.mesh.vertices[i] for i in verts)
                face_geometry.append(tri)
                normal = normalize(compute_triangle_normal(*tri))
                face_normals.append(normal)

            features.append(
                RegionEdgeFeature(
                    region_id=region_id,
                    edge_vertices=(vi1, vi2),
                    edge_coords=(p1, p2),
                    edge_vector=edge_vec,
                    edge_centroid=edge_mid,
                    face_ids=face_ids,
                    face_vertices=face_geometry,
                    face_normals=face_normals,
                )
            )

        return features

    def find_region_edge_features_along_original_edge(
        self,
        region_id: int,
        v0: int,
        v1: int,
        epsilon: float = 1e-6,
    ) -> list[RegionEdgeFeature]:
        """
        Returns RegionEdgeFeature objects for all subedges of the given original edge (v0, v1)
        that lie within the given region.
        """
        # First, get subedge vertex index pairs along the original edge
        subedges = self.find_region_subedges_along_original_edge_indices(
            region_id=region_id,
            v0=v0,
            v1=v1,
            epsilon=epsilon,
        )

        if not subedges:
            return []

        # Build edge -> feature map for all edges in region
        all_features = self.find_region_edge_features(region_id)
        edge_feature_map = {
            normalize_edge(*feature.edge_vertices): feature for feature in all_features
        }

        # Collect only features corresponding to subedges
        matched_features = []
        for a, b in subedges:
            key = normalize_edge(a, b)
            if key in edge_feature_map:
                matched_features.append(edge_feature_map[key])

        return matched_features
