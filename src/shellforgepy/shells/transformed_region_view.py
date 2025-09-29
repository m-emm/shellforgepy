import math
from typing import Optional

import numpy as np
from shellforgepy.construct.construct_utils import (
    compute_lay_flat_transform,
    fibonacci_sphere,
    is_valid_rigid_transform,
    normalize,
    rotation_matrix_from_vectors,
    triangle_area,
)
from shellforgepy.geometry.spherical_tools import ray_triangle_intersect
from shellforgepy.shells.connector_utils import transform_connector_hint
from shellforgepy.shells.mesh_partition import MeshPartition
from shellforgepy.shells.region_edge_feature import RegionEdgeFeature


class TransformedRegionView:
    def __init__(
        self,
        partition,
        region_id: int,
        transform: Optional[np.ndarray] = None,
    ):
        self.partition: MeshPartition = partition
        self.region_id = region_id
        self.transform = transform if transform is not None else np.eye(4)
        assert is_valid_rigid_transform(self.transform), "Transform is not rigid!"

        self.face_cache = None
        self.vertex_cache = None
        self.edge_cache = None

    def get_transformed_vertices_faces_boundary_edges(self):
        """Return transformed vertex array and face index list."""

        if (
            self.face_cache is not None
            and self.vertex_cache is not None
            and self.edge_cache is not None
        ):

            # Return cached values if available
            return self.vertex_cache, self.face_cache, self.edge_cache

        maps = self.partition.get_submesh_maps(self.region_id)

        V = np.array([maps["vertexes"][i] for i in sorted(maps["vertexes"])])
        F = np.array([maps["faces"][i] for i in sorted(maps["faces"])])
        E = np.array(
            sorted([maps["boundary_edges"][i] for i in sorted(maps["boundary_edges"])])
        )

        vertex_indices_in_edges = set()
        for a, b in E:
            vertex_indices_in_edges.add(a)
            vertex_indices_in_edges.add(b)

        assert all(
            j < len(V) for j in vertex_indices_in_edges
        ), "Vertex indices in edges are out of bounds"

        # Apply affine transformation to homogeneous coords
        V_homo = np.concatenate([V, np.ones((len(V), 1))], axis=1)
        V_transformed = (self.transform @ V_homo.T).T[:, :3]

        self.vertex_cache = V_transformed
        self.face_cache = F
        self.edge_cache = E

        return V_transformed, F, E

    def apply_transform(self, mat4x4: np.ndarray):
        """Returns a new view with the composed transformation applied."""
        new_transform = mat4x4 @ self.transform
        return TransformedRegionView(self.partition, self.region_id, new_transform)

    def rotated(
        self, angle: float, axis: np.ndarray = None, center: np.ndarray = None
    ) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView rotated around a given axis and center point.

        Parameters:
        -----------
        angle : float
            Rotation angle in radians.
        axis : np.ndarray
            3-element vector defining the rotation axis.
        center : np.ndarray
            3-element point around which the rotation is applied.
        """
        if axis is None:
            axis = np.array([0, 0, 1])
        if center is None:
            center = np.array([0, 0, 0])
        if isinstance(axis, list) or isinstance(axis, tuple):
            axis = np.array(axis)
        if isinstance(center, list) or isinstance(center, tuple):
            center = np.array(center)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        # Rotation matrix using Rodrigues' formula
        R = np.array(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ]
        )

        # Compose affine 4x4 rotation matrix around `center`
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = center - R @ center

        return self.apply_transform(A)

    def translated(self, x, y=None, z=None) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView translated by (x, y, z).

        Parameters:
        -----------
        x : float or array-like
            X component of translation or a 3-element vector.
        y : float, optional
            Y component of translation.
        z : float, optional
            Z component of translation.
        """
        if isinstance(x, (list, tuple, np.ndarray)) and y is None and z is None:
            vec = np.array(x, dtype=float)
        else:
            vec = np.array([x, y, z], dtype=float)

        T = np.eye(4)
        T[:3, 3] = vec
        return self.apply_transform(T)

    def transform_point(self, vertex):
        """Apply the current transformation to a single vertex."""
        vertex_homo = np.concatenate([vertex, [1]])
        transformed = self.transform @ vertex_homo
        return transformed[:3]

    def transformed_mesh_vertex_by_index(self, mesh_vertex_index):
        vertex_coords = self.partition.mesh.vertices[mesh_vertex_index]
        return self.transform_point(vertex_coords)

    def get_transformed_materialized_shell_maps(
        self, shell_thickness, shrinkage=0, shrink_border=0, smooth_inside=False
    ):
        """
        Return a dict of shell maps (face_id -> vertex/face map),
        where all vertex coordinates are transformed by the current affine matrix.
        """
        shell_maps, vertex_index_map = (
            self.partition.mesh.calculate_materialized_shell_maps(
                shell_thickness=shell_thickness,
                shrinkage=shrinkage,
                shrink_border=shrink_border,
                smooth_inside=smooth_inside,
            )
        )
        region_faces = self.partition.get_faces_of_region(self.region_id)

        result = {}
        for face_id in region_faces:
            face_map = shell_maps[face_id]
            V = face_map["vertexes"]
            V_arr = np.array([V[k] for k in sorted(V)])
            V_homo = np.concatenate([V_arr, np.ones((len(V_arr), 1))], axis=1)
            V_transformed = (self.transform @ V_homo.T).T[:, :3]
            transformed_vertexes = {k: v for k, v in zip(sorted(V), V_transformed)}
            result[face_id] = {
                "vertexes": transformed_vertexes,
                "faces": face_map["faces"],
            }

        return result, vertex_index_map

    def vertex_indices_closer_than(self, point, min_distance):
        V, _, _ = self.get_transformed_vertices_faces_boundary_edges()
        point = np.array(point)
        close_vertices = []
        for i, v in enumerate(V):
            if np.linalg.norm(v - point) < min_distance:
                close_vertices.append(i)
        return close_vertices

    def find_local_vertex_ids_by_label(self, vertex_label: str):
        return self.partition.find_local_vertex_ids_by_label(
            vertex_label, self.region_id
        )

    def face_indices_of_vertex_index_set(self, vertex_indices):
        _, F, _ = self.get_transformed_vertices_faces_boundary_edges()
        face_indices = set()
        for i, face in enumerate(F):
            if any(v in vertex_indices for v in face):
                face_indices.add(i)
        return sorted(face_indices)

    def ray_intersect_faces(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        """
        Given a ray, return a list of tuples (face_id, intersection_point)
        for all intersected triangles in the transformed region.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        hits = []

        for face_id, (i0, i1, i2) in enumerate(F):
            triangle = np.array([V[i0], V[i1], V[i2]])
            hit = ray_triangle_intersect(ray_origin, ray_direction, triangle)
            if hit is not None:
                hits.append((face_id, hit))

        return hits

    def compute_transformed_connector_hints(
        self,
        shell_thickness,
        merge_connectors=False,
        min_connector_distance=None,
        min_corner_distance=None,
    ):
        """
        Compute connector hints for this transformed region view.

        Parameters:
        -----------
        shell_thickness : float
            Thickness of the shell to use when computing materialized prisms.
        merge_connectors : bool
            Whether to merge collinear connectors after computation.

        Returns:
        --------
        List[ConnectorHint]
            List of connector hints on the transformed region.
        """

        connector_hints = self.partition.compute_connector_hints(
            shell_thickness,
            merge_connectors,
            min_connector_distance=min_connector_distance,
            min_corner_distance=min_corner_distance,
        )
        return [
            transform_connector_hint(h, self.transform)
            for h in connector_hints
            if h.region_a == self.region_id or h.region_b == self.region_id
        ]

    def average_normal_at_vertex(self, vertex_index: int) -> np.ndarray:
        """
        Compute the average normal vector at a vertex in the transformed region.

        Parameters:
        -----------
        vertex_index : int
            Index of the vertex in the region.
        Returns:
        --------
        np.ndarray
            Average normal vector at the vertex.
        """

        faces_at_vertex = self.face_indices_of_vertex_index_set([vertex_index])
        if not faces_at_vertex:

            raise ValueError(
                f"Vertex index {vertex_index} is not part of any face in the region."
            )

        face_normals = []
        for face_index in faces_at_vertex:
            face_normals.append(self.face_normal(face_index))

        face_normals = np.array(face_normals)

        return normalize(np.mean(face_normals, axis=0))

    def face_centroid(self, face_index: int) -> np.ndarray:
        """
        Compute the centroid of a face in the transformed region.

        Parameters:
        -----------
        face_index : int
            Index of the face in the region.

        Returns:
        --------
        np.ndarray
            Centroid of the face.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        if face_index < 0 or face_index >= len(F):
            raise ValueError(
                f"face_index {face_index} is out of bounds for region with {len(F)} faces."
            )

        face = F[face_index]
        vertices = [V[i] for i in face]
        centroid = np.mean(vertices, axis=0)

        return centroid

    def face_normal(self, face_index: int) -> np.ndarray:
        """
        Compute the normal vector of a face in the transformed region.

        Parameters:
        -----------
        face_index : int
            Index of the face in the region.

        Returns:
        --------
        np.ndarray
            Normal vector of the face.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        if face_index < 0 or face_index >= len(F):
            raise ValueError(
                f"face_index {face_index} is out of bounds for region with {len(F)} faces."
            )

        face = F[face_index]
        a, b, c = [V[i] for i in face]
        n = np.cross(b - a, c - a)

        return normalize(n)

    def face_vertices(self, face_index: int) -> np.ndarray:
        """
        Get the vertices of a face in the transformed region.

        Parameters:
        -----------
        face_index : int
            Index of the face in the region.

        Returns:
        --------
        np.ndarray
            Array of vertices for the specified face.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()
        if face_index < 0 or face_index >= len(F):
            raise ValueError(
                f"face_index {face_index} is out of bounds for region with {len(F)} faces."
            )
        face = F[face_index]
        return np.array([V[i] for i in face])

    def lay_flat(self, definition_of_low: float = 1) -> "TransformedRegionView":
        """
        Return a new TransformedRegionView where the region is rotated and translated so that
        one of the low triangles lies flat on the XY plane.
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        z_min = np.min(V[:, 2])
        z_max = np.max(V[:, 2])
        low_thresh = z_min + definition_of_low * (z_max - z_min)

        low_faces = [f for f in F if any(V[i][2] <= low_thresh for i in f)]
        if not low_faces:
            raise ValueError("No low faces found.")
        else:
            print(f"Low faces found: {low_faces}")

        def normal(face):
            a, b, c = [V[i] for i in face]
            n = np.cross(b - a, c - a)
            return n / np.linalg.norm(n)

        good_faces = []
        for face in low_faces:

            a, b, c = [V[i] for i in face]

            print(f"Low face: {face}, vertices: {a}, {b}, {c}")
            # 2) Compute centroid pivot
            centroid = (a + b + c) / 3

            # 3) Build rotation R that carries face_normal → [0,0,1]
            fn = -normal(face)
            target = np.array([0.0, 0.0, 1.0])
            R3 = rotation_matrix_from_vectors(fn, target)  # your existing utility

            # 4) Assemble the full affine A = T_z * T( +centroid ) * R * T( -centroid )
            # 4×4 identity:
            A = np.eye(4)

            # T1 = translate(-centroid)
            T1 = np.eye(4)
            T1[:3, 3] = -centroid

            # R4 = rotation about origin
            R4 = np.eye(4)
            R4[:3, :3] = R3

            # T2 = translate(+centroid)
            T2 = np.eye(4)
            T2[:3, 3] = centroid

            # Combine: first T1, then R4, then T2
            M = T2 @ R4 @ T1

            pts_face_m = [(M @ np.hstack([v, 1]).T)[:3] for v in (a, b, c)]
            z_face = (
                sum(p[2] for p in pts_face_m) / 3.0
            )  # they should all be equal up to FP noise

            # Build T3 to drop *that* face to Z=0
            T3 = np.eye(4)
            T3[2, 3] = -z_face

            # Final composite
            A = T3 @ M

            to_flatten = [a, b, c]
            to_flatten_transformed = [(A @ np.hstack([v, 1]).T) for v in to_flatten]

            for v in to_flatten_transformed:
                if not np.isclose(v[2], 0, atol=1e-5):
                    print(f"WARNING: vertex not flat: {v}")

            # check if at least one triangle lies flat on the floor
            new_view = self.apply_transform(A)

            V_flat, F_flat, _ = new_view.get_transformed_vertices_faces_boundary_edges()

            # # no vertex should be below the floor
            if not np.any(V_flat[:, 2] < -1e-5):
                print(
                    f"Found good face: {face.tolist()}, vertices: {[V_flat[i].tolist() for i in face]}"
                )
                good_faces.append({"face": face, "transform": A})

        if not good_faces:
            print(f"******** NO GOOD FACES FOUND ********")
            raise ValueError("No good faces found.")

        A = good_faces[0]["transform"]

        new_view = self.apply_transform(A)

        V_flat, F_flat, _ = new_view.get_transformed_vertices_faces_boundary_edges()

        found_face_number = None
        for face_number, faces in enumerate(F_flat):
            a, b, c = [V_flat[i] for i in faces]
            if np.isclose(a[2], 0) and np.isclose(b[2], 0) and np.isclose(c[2], 0):
                found_face_number = face_number
                break

        if found_face_number is None:
            raise ValueError("No flat face found after transformation.")

        else:
            print(
                f"Flat face found: {found_face_number}, vertex_indixes: {F_flat[found_face_number]} vertices: {V_flat[F_flat[found_face_number]]}, selected face_number: {face}"
            )

        return new_view

    def num_faces(self) -> int:
        """
        Return the number of faces in the region.
        """
        return self.partition.get_num_faces_in_region(self.region_id)

    def lay_flat_on_face(self, face_index_in_region: int) -> "TransformedRegionView":
        """
        Lay the region flat on a specific face by aligning it with the XY plane.
        Parameters:
        -----------
        face_index : int
            Index of the face to lay flat on the XY plane.
        Returns:
        --------
        TransformedRegionView
            A new view of the region with the specified face laid flat.
        """

        def normal(face):
            a, b, c = [V[i] for i in face]
            n = np.cross(b - a, c - a)
            return n / np.linalg.norm(n)

        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()

        if face_index_in_region < 0 or face_index_in_region >= len(F):
            raise ValueError(
                f"face_index_in_region {face_index_in_region} is out of bounds for region with {len(F)} faces."
            )
        face = F[face_index_in_region]
        a, b, c = [V[i] for i in face]
        # print(f"Laying flat on face: {face}, vertices: {a}, {b}, {c}")
        # 2) Compute centroid pivot
        centroid = (a + b + c) / 3
        # 3) Build rotation R that carries face_normal → [0,0,1]
        fn = -normal(face)

        target = np.array([0.0, 0.0, 1.0])

        R3 = rotation_matrix_from_vectors(fn, target)  # your existing utility

        # 4) Assemble the full affine A = T_z * T( +centroid ) * R * T( -centroid )
        # 4×4 identity:
        A = np.eye(4)

        # T1 = translate(-centroid)
        T1 = np.eye(4)
        T1[:3, 3] = -centroid

        # R4 = rotation about origin
        R4 = np.eye(4)
        R4[:3, :3] = R3

        # T2 = translate(+centroid)
        T2 = np.eye(4)
        T2[:3, 3] = centroid

        # Combine: first T1, then R4, then T2
        M = T2 @ R4 @ T1

        pts_face_m = [(M @ np.hstack([v, 1]).T)[:3] for v in (a, b, c)]
        z_face = (
            sum(p[2] for p in pts_face_m) / 3.0
        )  # they should all be equal up to FP noise

        # Build T3 to drop *that* face to Z=0
        T3 = np.eye(4)
        T3[2, 3] = -z_face

        # Final composite
        A = T3 @ M

        to_flatten = [a, b, c]
        to_flatten_transformed = [(A @ np.hstack([v, 1]).T) for v in to_flatten]

        for v in to_flatten_transformed:
            if not np.isclose(v[2], 0, atol=1e-5):
                print(f"WARNING: vertex not flat: {v}")

        assert is_valid_rigid_transform(A), "Computed transform is not valid!"

        new_view = self.apply_transform(A)
        return new_view

    def check_printability(self, overhang_threshold_deg: float = 45.0):
        """
        Check 3D printability of the region: identify triangles with too-steep overhangs.

        Parameters:
        -----------
        overhang_threshold_deg : float
            Maximum allowed angle (in degrees) between the triangle normal and the Z-axis.
            Triangles with greater angles are considered non-printable.

        Returns:
        --------
        dict with:
            - total_area: float
            - printable_area: float
            - unprintable_area: float
            - bad_faces: list of (face_index, angle_in_degrees)
        """
        V, F, _ = self.get_transformed_vertices_faces_boundary_edges()
        z_axis = np.array([0, 0, 1])
        threshold_rad = np.radians(overhang_threshold_deg)

        total_area = 0.0
        unprintable_area = 0.0
        bad_faces = []

        for idx, face in enumerate(F):
            a, b, c = V[face[0]], V[face[1]], V[face[2]]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-8:
                continue  # skip degenerate
            n /= np.linalg.norm(n)
            angle = np.arccos(np.clip(np.dot(n, z_axis), -1.0, 1.0))
            area = triangle_area(a, b, c)
            total_area += area

            if angle > threshold_rad:
                unprintable_area += area
                bad_faces.append((idx, np.degrees(angle)))

        return {
            "total_area": total_area,
            "printable_area": total_area - unprintable_area,
            "unprintable_area": unprintable_area,
            "bad_faces": bad_faces,
            "bad_fraction": unprintable_area / total_area if total_area > 0 else 0.0,
        }

    def find_overhanging_boundary_edges(
        self,
        angle_threshold_deg=45,
        vertical_edge_tolerance_deg=10,
        triangle_downward_threshold_deg=45,
    ) -> list[tuple[int, int]]:
        """
        Find boundary edges that need support due to overhang.

        An edge is overhanging if its triangle's "inward" direction (orthogonal to the edge and triangle normal),
        properly disambiguated to point inward, points upward more than `angle_threshold_deg` from horizontal.

        Parameters:
        -----------
        angle_threshold_deg : float
            Maximum allowable angle from horizontal. Edges exceeding this upward are marked.
        vertical_edge_tolerance_deg : float
            Edges more vertical than this angle are skipped entirely.
        """
        angle_threshold_sin = -np.sin(np.radians(triangle_downward_threshold_deg))
        vertical_edge_cos = np.cos(np.radians(vertical_edge_tolerance_deg))

        V_trans, region_faces, boundary_edges = (
            self.get_transformed_vertices_faces_boundary_edges()
        )
        V = V_trans
        result = []

        print(
            f"Found {len(boundary_edges)} boundary edges and {len(region_faces)} faces in region {self.region_id}"
        )

        for a, b in boundary_edges:
            # Find triangle in region that contains this edge
            tri = next((face for face in region_faces if a in face and b in face), None)
            if tri is None:
                raise ValueError(f"Edge {a}-{b} not found in region faces")

            c = next(i for i in tri if i not in (a, b))
            va, vb, vc = V[a], V[b], V[c]

            edge_vec = vb - va
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-8:
                continue  # degenerate edge
            edge_dir = edge_vec / edge_len

            # Skip nearly vertical edges — they print fine
            if abs(edge_dir[2]) > vertical_edge_cos:
                continue

            triangle_normal = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
            triangle_normal /= np.linalg.norm(triangle_normal)

            # Check if triangle is downward-facing
            downward_problematic = False
            if triangle_normal[2] < angle_threshold_sin:
                print(f"Triangle {tri} is downward-facing")
                downward_problematic = True

            inward = np.cross(triangle_normal, edge_dir)
            inward /= np.linalg.norm(inward)

            edge_center = (va + vb) / 2
            tri_centroid = (va + vb + vc) / 3

            to_centroid = tri_centroid - edge_center
            to_centroid /= np.linalg.norm(to_centroid)

            # Flip inward if it points outward
            if np.dot(to_centroid, inward) < 0:
                inward = -inward

            if downward_problematic:
                print(f"Edge {a}-{b} is downward-facing, adding")
                result.append((a, b))
            elif inward[2] > 0:
                print(f"Edge {a}-{b} is overhanging: inward.z = {inward[2]:.3f}")
                result.append((a, b))
            else:
                print(f"Edge {a}-{b} is supported: inward.z = {inward[2]:.3f}")

        return result

    def find_best_orientation(self, max_angle_deg=45.0, samples=100):
        best_score = float("inf")
        best_view = self

        directions = fibonacci_sphere(samples)
        up = np.array([0, 0, 1])
        for d in directions:
            R3 = rotation_matrix_from_vectors(d, up)
            A = np.eye(4)
            A[:3, :3] = R3
            candidate = self.apply_transform(A)
            score = unprintable_area_fraction(candidate, max_angle_deg=max_angle_deg)
            if score < best_score:
                best_score = score
                best_view = candidate

        return best_view, best_score

    def printability_score(self, angle_threshold_rad=np.radians(45)):

        def elevation_angle(normal):
            # Angle between normal and the XY plane
            xy_norm = np.linalg.norm(normal[:2])
            return np.arctan2(normal[2], xy_norm)

        def smooth_score(elev_angle, threshold_rad):
            angle = abs(elev_angle)
            if angle >= threshold_rad:
                return 0.0
            return 1.0 - (angle / threshold_rad) ** 2

        V, F, E = self.get_transformed_vertices_faces_boundary_edges()

        z_coords = V[:, 2]
        all_above_or_on_plane = np.all(z_coords >= -1e-6)
        if not all_above_or_on_plane:
            return 0.0

        # Condition 1: triangle(s) lying flat on z=0
        has_flat_triangle = any(np.all(np.abs(V[face][:, 2]) < 1e-6) for face in F)

        # Condition 2: multiple non-collinear edges at z=0
        flat_edges = [
            (V[a], V[b]) for a, b in E if abs(V[a][2]) < 1e-6 and abs(V[b][2]) < 1e-6
        ]
        non_collinear_pairs = 0
        for i in range(len(flat_edges)):
            for j in range(i + 1, len(flat_edges)):
                va1, vb1 = flat_edges[i]
                va2, vb2 = flat_edges[j]
                dir1 = vb1 - va1
                dir2 = vb2 - va2
                dir1 /= np.linalg.norm(dir1)
                dir2 /= np.linalg.norm(dir2)
                if np.linalg.norm(np.cross(dir1, dir2)) > 0.1:  # not collinear
                    non_collinear_pairs += 1
                    break
            if non_collinear_pairs > 0:
                break

        if not has_flat_triangle and non_collinear_pairs == 0:
            return 0.0

        total_area = 0.0
        printable_area = 0.0
        for face in F:
            v0, v1, v2 = V[face]
            area = triangle_area(v0, v1, v2)
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normalize(normal)

            # Special case: exactly flat triangle at z=0 → always printable
            if np.all(np.abs(V[face][:, 2]) < 1e-6):
                printable_area += area
            else:
                elev = elevation_angle(normal)
                score = smooth_score(elev, angle_threshold_rad)
                printable_area += area * score

            total_area += area

        return printable_area / total_area if total_area > 0 else 0.0

    def lay_flat_optimally_printable(self, angle_threshold_rad=np.radians(45)):
        """
        Lay the region flat in the most printable orientation.

        This method finds the best orientation that minimizes unprintable area fraction
        based on the specified angle threshold.
        """

        best_printability_score = 0
        best_view = None
        for i in range(self.num_faces()):

            optimized_view = self.lay_flat_on_face(i)

            score = optimized_view.printability_score(angle_threshold_rad)

            if score > best_printability_score:
                print(f"New best printability score: {score} for face {i}")
                best_printability_score = score
                best_view = optimized_view

        if best_view is None:
            return self
        else:
            print(f"Best printability score: {best_printability_score}")
            return best_view

    def lay_flat_on_boundary_edges_for_printability(
        self, angle_threshold_rad=np.radians(45), desired_region_pairs=None
    ):
        """
        Try to lay flat all non-collinear pairs of boundary edges by rotating the region
        so that the first edge lies along X-axis on the Z=0 plane, and the first point
        of the second edge lies also on Z=0.

        Returns the transformed view with the highest printability score.
        """
        from itertools import combinations

        V, F, edge_list = self.get_transformed_vertices_faces_boundary_edges()

        edge_walk = []

        remaining = [(a, b) for a, b in edge_list]

        while len(remaining) > 0:
            progress = False
            for current in remaining:
                if not edge_walk:
                    edge_walk.append(current)
                    remaining.remove(current)
                    progress = True
                    break

                first_vertex = edge_walk[0][0]
                last_vertex = edge_walk[-1][1]

                if current[0] == last_vertex:
                    edge_walk.append((current[0], current[1]))
                    remaining.remove(current)
                    progress = True
                    break
                elif current[1] == first_vertex:
                    edge_walk.insert(0, (current[0], current[1]))
                    remaining.remove(current)
                    progress = True
                    break

            if not progress:
                print("No progress in edge walk, remaining edges:", remaining)
                break

        edge_walk_is_closed = edge_walk[0][0] == edge_walk[-1][1]
        print(f"Edge walk: {edge_walk}, is closed: {edge_walk_is_closed}")

        submesh_maps = self.partition.get_submesh_maps(self.region_id)
        local_to_global_vertex_map = submesh_maps["local_to_global_vertex_map"]

        if desired_region_pairs is not None:
            desired_region_pairs = set(
                tuple(sorted(pair)) for pair in desired_region_pairs
            )

        all_edge_regions = set()
        for (i1, j1), (i2, j2) in combinations(edge_walk, 2):

            global_i1 = local_to_global_vertex_map[i1]
            global_j1 = local_to_global_vertex_map[j1]

            edge_regions = self.partition.find_regions_of_edge((global_i1, global_j1))
            assert len(edge_regions) == 2

            all_edge_regions.add(tuple(sorted(edge_regions)))

        print(f"Edge regions: {all_edge_regions}")

        best_score = 0.0
        best_view = None
        best_region_pair = None
        found_candidate = False

        for (i1, j1), (i2, j2) in combinations(edge_walk, 2):

            global_i1 = local_to_global_vertex_map[i1]
            global_j1 = local_to_global_vertex_map[j1]

            edge_regions = self.partition.find_regions_of_edge((global_i1, global_j1))

            assert (
                self.region_id in edge_regions
            ), f"Edge ({global_i1}, {global_j1}) is not in region {self.region_id}."
            assert len(edge_regions) == 2

            if desired_region_pairs is not None:
                if not tuple(sorted(edge_regions)) in desired_region_pairs:
                    print(
                        f"Skipping edge pair {edge_regions} as it is not in desired pairs."
                    )
                    continue

            p1, p2 = V[i1], V[j1]
            q1, q2 = V[i2], V[j2]

            # Compute triangle areas to check if we can form meaningful triangle
            area1 = triangle_area(p1, p2, q1)
            area2 = triangle_area(p1, p2, q2)

            if area1 < 1e-8 and area2 < 1e-8:
                continue

            found_candidate = True

            # Choose the larger triangle for stability
            if area1 >= area2:
                base_triangle = (p1, p2, q1)
                check_vertices = [i2, j2]
            else:
                base_triangle = (p1, p2, q2)
                check_vertices = [i2, j2]

            A = compute_lay_flat_transform(*base_triangle)

            # Apply transform to the check vertices
            v_check_1 = (A @ np.hstack([V[check_vertices[0]], 1.0]))[:3]
            v_check_2 = (A @ np.hstack([V[check_vertices[1]], 1.0]))[:3]

            if not np.isclose(v_check_1[2], 0.0, atol=1e-5) or not np.isclose(
                v_check_2[2], 0.0, atol=1e-5
            ):
                continue

            new_view = self.apply_transform(A)
            new_V, _, _ = new_view.get_transformed_vertices_faces_boundary_edges()

            if np.any(new_V[:, 2] < -1e-6):
                continue

            score = new_view.printability_score(angle_threshold_rad)
            if score > best_score:
                print(
                    f"New best printability score (edge-based): {score}, for region pair  {edge_regions}"
                )
                best_score = score
                best_view = new_view
                best_region_pair = edge_regions

        if not found_candidate:
            print("No suitable edge pairs found for laying flat.")
        else:
            print(
                f"Best printability score (edge-based): {best_score}, for region pair {best_region_pair}"
            )

        return best_view if best_view is not None else self

    def find_transformed_edge_features_along_original_edge(
        self, v0: int, v1: int
    ) -> list[RegionEdgeFeature]:
        raw_features = self.partition.find_region_edge_features_along_original_edge(
            region_id=self.region_id,
            v0=v0,
            v1=v1,
        )

        transformed_features = []
        for feat in raw_features:
            # Transform edge endpoints and centroid
            p1_trans = self.transform_point(feat.edge_coords[0])
            p2_trans = self.transform_point(feat.edge_coords[1])
            centroid_trans = self.transform_point(feat.edge_centroid)

            # Transform all triangle face vertices
            face_vertices_trans = []
            for tri in feat.face_vertices:
                tri_trans = tuple(self.transform_point(v) for v in tri)
                face_vertices_trans.append(tri_trans)

            # Transform face normals (only rotate, don't translate)
            R = self.transform[:3, :3]
            face_normals_trans = [normalize(R @ n) for n in feat.face_normals]

            transformed_features.append(
                RegionEdgeFeature(
                    region_id=feat.region_id,
                    edge_vertices=feat.edge_vertices,  # mesh indices stay the same
                    edge_coords=(p1_trans, p2_trans),
                    edge_vector=normalize(p2_trans - p1_trans),
                    edge_centroid=centroid_trans,
                    face_ids=feat.face_ids,
                    face_vertices=face_vertices_trans,
                    face_normals=face_normals_trans,
                )
            )

        return transformed_features


def rotation_matrix_about_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2)
    b, c, d = -axis * math.sin(angle / 2)
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def unprintable_area_fraction(view: TransformedRegionView, max_angle_deg=45):
    V, F, _ = view.get_transformed_vertices_faces_boundary_edges()
    threshold = math.cos(math.radians(max_angle_deg))  # angle from vertical

    def normal(face):
        a, b, c = [V[i] for i in face]
        n = np.cross(b - a, c - a)
        return n / np.linalg.norm(n)

    def area(face):
        a, b, c = [V[i] for i in face]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    total = 0.0
    unprintable = 0.0
    for face in F:
        A = area(face)
        N = normal(face)

        vertical = abs(N[2])  # 1 = vertical, 0 = horizontal
        if vertical > threshold:
            unprintable += A
        total += A

    return unprintable / total if total > 0 else 1.0
