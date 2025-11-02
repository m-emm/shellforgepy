import logging

import numpy as np

_logger = logging.getLogger(__name__)


def create_edge_filter(p1_local, p2_local, normal_local):
    def edge_vertex_filter(intersection_point, mesh_edge, va, vb):
        """
        Here, we try to figure out the following:
        - We have a plane going through p1_local, with the normal vector normal_local
        - We project p1 trivially onto this plane
        - We project the p2 point onto this plane, along the normal_local direction
        - We also project the intersection_pont onto this plane, along the normal_local direction
        - We then check if the projected intersection_point lies within the segment defined by the projected p1  and p2
        We only return true if this is the case
        """

        # Step 1: Project p1 trivially onto the plane (it's already on the plane)
        p1_projected = p1_local

        # Step 2: Project p2 onto the plane along the normal direction
        # Distance from p2 to the plane
        p2_to_plane_vec = p2_local - p1_local  # Vector from plane point to p2
        distance_to_plane = np.dot(p2_to_plane_vec, normal_local)
        p2_projected = p2_local - distance_to_plane * normal_local

        # Step 3: Project intersection_point onto the plane along the normal direction
        intersection_to_plane_vec = (
            intersection_point - p1_local
        )  # Vector from plane point to intersection
        intersection_distance_to_plane = np.dot(intersection_to_plane_vec, normal_local)
        intersection_projected = (
            intersection_point - intersection_distance_to_plane * normal_local
        )

        # Step 4: Check if the projected intersection_point lies within the segment
        # defined by projected p1 and p2

        # Vector from p1_projected to p2_projected
        edge_vector = p2_projected - p1_projected
        edge_length_squared = np.dot(edge_vector, edge_vector)

        # If edge has zero length, reject
        if edge_length_squared < 1e-12:
            _logger.debug("Edge has zero length in edge filter")
            return False

        # Vector from p1_projected to intersection_projected
        to_intersection = intersection_projected - p1_projected

        # Parameter t such that intersection_projected = p1_projected + t * edge_vector
        t = np.dot(to_intersection, edge_vector) / edge_length_squared

        # Check if intersection lies on the segment (with small tolerance)
        if not (-1e-6 <= t <= 1 + 1e-6):
            _logger.debug(
                f"Intersection point {intersection_projected} not on edge segment {p1_projected} to {p2_projected}"
            )
            return False

        # Point on segment closest to intersection_projected
        closest_point_on_segment = p1_projected + t * edge_vector

        # Distance from intersection_projected to the segment
        distance_to_segment = np.linalg.norm(
            intersection_projected - closest_point_on_segment
        )

        # Accept if close enough to the segment
        tolerance = max(1e-6, 1e-8 * np.sqrt(edge_length_squared))
        is_close_enough = distance_to_segment <= tolerance
        if not is_close_enough:
            _logger.debug(
                f"Intersection point {intersection_projected} too far from edge segment {p1_projected} to {p2_projected}, distance {distance_to_segment}, tolerance {tolerance}"
            )
        return is_close_enough

    return edge_vertex_filter
