import logging
from collections import defaultdict
from typing import List

import numpy as np
from shellforgepy.construct.construct_utils import (
    are_normals_similar,
    compute_triangle_normal,
    normalize,
)
from shellforgepy.shells.connector_hint import ConnectorHint

_logger = logging.getLogger(__name__)


def merge_collinear_connectors(
    hints: List[ConnectorHint], angle_tol: float = 1e-3
) -> List[ConnectorHint]:

    tol = 1e-6
    original_count = len(hints)

    def get_endpoints(h: ConnectorHint):
        return h.start_vertex, h.end_vertex

    grouped = defaultdict(list)
    for h in hints:
        grouped[(h.region_a, h.region_b)].append(h)

    merged_hints = []

    for (region_a, region_b), group in grouped.items():
        remaining = list(enumerate(group))

        while remaining:
            idx, base = remaining.pop(0)
            base_start, base_end = get_endpoints(base)
            chain = [(idx, base_start, base_end)]

            progress = True

            while progress:
                progress = False
                new_remaining = []
                for j, hint in remaining:
                    a, b = get_endpoints(hint)
                    end_chain_vertex = chain[-1][2]
                    start_chain_vertex = chain[0][1]

                    # check if the hint is collinear with the chain
                    is_collinear = np.allclose(
                        normalize(b - a),
                        normalize(end_chain_vertex - start_chain_vertex),
                        atol=tol,
                    )

                    if np.allclose(end_chain_vertex, a, atol=tol) and is_collinear:
                        # print(f"Found forward match: {j} to {chain[-1][0]}")
                        if are_normals_similar(
                            base.triangle_a_normal, hint.triangle_a_normal, angle_tol
                        ) and are_normals_similar(
                            base.triangle_b_normal, hint.triangle_b_normal, angle_tol
                        ):

                            # print(f"Appending forward: {j} to {chain[-1][0]}")
                            chain.append((j, a, b))
                            progress = True
                    elif np.allclose(start_chain_vertex, b, atol=tol) and is_collinear:
                        # print(f"Found backward match: {j} to {chain[0][0]}")
                        if are_normals_similar(
                            base.triangle_a_normal, hint.triangle_a_normal, angle_tol
                        ) and are_normals_similar(
                            base.triangle_b_normal, hint.triangle_b_normal, angle_tol
                        ):

                            # print(f"Prepending forward: {j} to {chain[0][0]}")
                            chain.insert(0, (j, a, b))
                            progress = True
                    else:
                        new_remaining.append((j, hint))  # keep only unmerged
                remaining = new_remaining

            if len(chain) == 1:
                # print(f"Found chain of length 1: {chain[0][0]}")
                merged_hints.append(group[chain[0][0]])
                continue

            # Build merged hint
            start = chain[0][1]
            end = chain[-1][2]
            edge_vec = normalize(end - start)
            edge_mid = (start + end) / 2

            if np.allclose(start, end):
                raise ValueError(
                    f"Degenerate connector edge in chain (start == end)\n"
                    f"  region_a = {region_a}, region_b = {region_b}\n"
                    f"  start = {start}, end = {end}"
                )

            first_hint = group[chain[0][0]]

            # Get apex vertices by excluding edge from triangle vertex set

            tri_a = np.array(first_hint.triangle_a_vertices)

            for i in range(3):
                if np.allclose(tri_a[i], first_hint.start_vertex, atol=tol):
                    tri_a[i] = start
                elif np.allclose(tri_a[i], first_hint.end_vertex, atol=tol):
                    tri_a[i] = end

            tri_b = np.array(first_hint.triangle_b_vertices)
            for i in range(3):
                if np.allclose(tri_b[i], first_hint.start_vertex, atol=tol):
                    tri_b[i] = start
                elif np.allclose(tri_b[i], first_hint.end_vertex, atol=tol):
                    tri_b[i] = end

            new_hint = ConnectorHint(
                region_a=first_hint.region_a,
                region_b=first_hint.region_b,
                triangle_a_vertices=tri_a,
                triangle_b_vertices=tri_b,
                triangle_a_vertex_indices=first_hint.triangle_a_vertex_indices,
                triangle_b_vertex_indices=first_hint.triangle_b_vertex_indices,
                triangle_a_normal=first_hint.triangle_a_normal,
                triangle_b_normal=first_hint.triangle_b_normal,
                edge_vector=first_hint.edge_vector,
                edge_centroid=edge_mid,
                start_vertex=start,
                end_vertex=end,
                start_vertex_index=first_hint.start_vertex_index,
                end_vertex_index=first_hint.end_vertex_index,
                face_pair_ids=[
                    fid
                    for j, *_ in chain
                    for fid in getattr(group[j], "face_pair_ids", [])
                ],
                original_edges=[
                    e
                    for j, *_ in chain
                    for e in getattr(group[j], "original_edges", [])
                ],
            )

            if len(chain) > 1:
                for chain_member in chain:
                    current_hint = group[chain_member[0]]
                    original_edge_vec = (
                        current_hint.end_vertex - current_hint.start_vertex
                    )

                    assert np.dot(original_edge_vec, new_hint.edge_vector) > 0

            merged_hints.append(new_hint)
    print(f"Merged into {len(merged_hints)} connector hints (from {original_count})")

    return merged_hints


def compute_connector_hints_from_shell_maps(
    mesh_faces: np.ndarray,
    face_to_region: dict[int, int],
    shell_maps: dict[int, dict],
    vertex_index_map: dict[int, dict],
) -> List[ConnectorHint]:
    edge_to_faces = defaultdict(list)

    # 1. Build edge -> [(face_index, region)] mapping
    for f_idx, region in face_to_region.items():
        face = mesh_faces[f_idx]
        for i in range(3):
            a, b = sorted((face[i], face[(i + 1) % 3]))
            edge_to_faces[(a, b)].append((f_idx, region))

    connector_hints = []

    for edge, face_region_pairs in edge_to_faces.items():
        if len(face_region_pairs) != 2:
            raise ValueError(f"The edge {edge} is not shared by exactly two faces.")

        (f_a, r_a), (f_b, r_b) = face_region_pairs
        if r_a == r_b:
            continue

        if r_a > r_b:
            (f_a, r_a), (f_b, r_b) = (f_b, r_b), (f_a, r_a)

        face_a = mesh_faces[f_a]
        shared_indices = set(face_a) & set(mesh_faces[f_b])

        # preserve winding from face_a
        ordered_shared = []
        for i in range(3):
            a, b = face_a[i], face_a[(i + 1) % 3]
            if a in shared_indices and b in shared_indices:
                ordered_shared = [a, b]
                break

        if len(ordered_shared) != 2:
            raise ValueError(f"Could not find shared edge between {f_a} and {f_b}")

        vi1, vi2 = ordered_shared

        def get_vertex(face_id, vi):
            vi_local = vertex_index_map[face_id]["inner"][vi]
            return shell_maps[face_id]["vertexes"][vi_local]

        p1 = get_vertex(f_a, vi1)
        p2 = get_vertex(f_a, vi2)

        edge_vec = normalize(p2 - p1)
        edge_mid = (p1 + p2) / 2

        tri_a = [get_vertex(f_a, vi) for vi in mesh_faces[f_a]]
        tri_b = [get_vertex(f_b, vi) for vi in mesh_faces[f_b]]

        n_a = normalize(compute_triangle_normal(*tri_a))
        n_b = normalize(compute_triangle_normal(*tri_b))

        hint = ConnectorHint(
            region_a=r_a,
            region_b=r_b,
            triangle_a_vertices=tuple(tri_a),
            triangle_b_vertices=tuple(tri_b),
            triangle_a_vertex_indices=tuple(mesh_faces[f_a]),
            triangle_b_vertex_indices=tuple(mesh_faces[f_b]),
            triangle_a_normal=n_a,
            triangle_b_normal=n_b,
            edge_vector=edge_vec,
            edge_centroid=edge_mid,
            start_vertex=p1,
            end_vertex=p2,
            start_vertex_index=vi1,
            end_vertex_index=vi2,
            original_edges=[(vi1, vi2)],
            face_pair_ids=[(f_a, f_b)],
        )
        connector_hints.append(hint)

    return connector_hints


def transform_connector_hint(
    hint: ConnectorHint, transform: np.ndarray
) -> ConnectorHint:
    """
    Apply an affine transformation to a ConnectorHint.
    Positions (vertices, centroids) are fully transformed.
    Vectors (normals, edge vectors) are only rotated.

    Parameters:
    -----------
    hint : ConnectorHint
        The original connector hint.
    transform : np.ndarray
        A 4x4 affine transformation matrix.

    Returns:
    --------
    ConnectorHint
        A new connector hint with transformed data.
    """

    def transform_point(p):
        p_h = np.append(p, 1.0)
        return (transform @ p_h)[:3]

    def rotate_vector(v):
        R = transform[:3, :3]
        return R @ v

    return ConnectorHint(
        region_a=hint.region_a,
        region_b=hint.region_b,
        triangle_a_vertices=tuple(transform_point(v) for v in hint.triangle_a_vertices),
        triangle_b_vertices=tuple(transform_point(v) for v in hint.triangle_b_vertices),
        triangle_a_vertex_indices=tuple(hint.triangle_a_vertex_indices),
        triangle_b_vertex_indices=tuple(hint.triangle_b_vertex_indices),
        triangle_a_normal=rotate_vector(hint.triangle_a_normal),
        triangle_b_normal=rotate_vector(hint.triangle_b_normal),
        edge_vector=rotate_vector(hint.edge_vector),
        edge_centroid=transform_point(hint.edge_centroid),
        start_vertex=transform_point(hint.start_vertex),
        end_vertex=transform_point(hint.end_vertex),
        start_vertex_index=hint.start_vertex_index,
        end_vertex_index=hint.end_vertex_index,
        original_edges=list(hint.original_edges),
        face_pair_ids=list(hint.face_pair_ids),
    )
