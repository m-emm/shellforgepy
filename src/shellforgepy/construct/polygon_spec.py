"""Polygon specification for perforation operations."""

from dataclasses import dataclass
from typing import List

import numpy as np
from shellforgepy.construct.construct_utils import (
    compute_polygon_normal,
    normalize,
    point_in_polygon_2d,
)


@dataclass
class PolygonSpec:
    """Specification for a 3D polygon used in perforation operations."""

    points: List[np.ndarray]  # List of 3D points defining the polygon boundary

    def __post_init__(self):
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points)
        if len(self.points) < 3:
            raise ValueError("Polygon must have at least 3 points")

        # check no duplicate points
        unique_points = {tuple(p) for p in self.points}
        if len(unique_points) != len(self.points):
            raise ValueError("Polygon points must be unique")

        self.center = np.mean(self.points, axis=0)
        self.normal = compute_polygon_normal(self.points)

    @classmethod
    def from_points_3d(cls, points: List[np.ndarray]):
        """Create a PolygonSpec from 3D points, automatically computing center and normal."""
        points = [np.asarray(p) for p in points]

        return cls(points=points)

    def create_2d_basis(self):
        """Create orthonormal basis vectors for the polygon's 2D coordinate system."""
        # Create two orthogonal vectors in the polygon plane
        # Start with an arbitrary vector not parallel to normal
        if abs(self.normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])

        # First basis vector: temp projected onto polygon plane
        u = temp - np.dot(temp, self.normal) * self.normal
        u = normalize(u)

        # Second basis vector: cross product of normal and first basis
        v = np.cross(self.normal, u)
        v = normalize(v)

        return u, v

    def point_to_2d(self, point_3d: np.ndarray) -> np.ndarray:
        """Convert a 3D point to polygon's 2D coordinate system."""
        u, v = self.create_2d_basis()

        # Vector from polygon center to the point
        rel_point = point_3d - self.center

        # Project onto 2D basis vectors
        x = np.dot(rel_point, u)
        y = np.dot(rel_point, v)

        return np.array([x, y])

    def point_from_2d(self, point_2d: np.ndarray) -> np.ndarray:
        """Convert a 2D point in polygon coordinates back to 3D."""
        u, v = self.create_2d_basis()

        # Reconstruct 3D point
        point_3d = self.center + point_2d[0] * u + point_2d[1] * v

        return point_3d

    def contains_point(self, point):
        """Test if a 3D point is inside the 3D polygon using 2D projection."""
        # Project point onto polygon plane

        to_point = point - self.center

        distance_to_plane = np.dot(to_point, self.normal)
        projected_point = point - distance_to_plane * self.normal

        # Create 2D coordinate system on the polygon plane
        if abs(self.normal[2]) < 0.9:
            plane_x = normalize(np.cross(self.normal, np.array([0, 0, 1])))
        else:
            plane_x = normalize(np.cross(self.normal, np.array([1, 0, 0])))
        plane_y = normalize(np.cross(plane_x, self.normal))

        # Convert polygon and point to 2D
        polygon_2d = []
        for poly_point in self.points:
            relative = poly_point - self.center
            x_coord = np.dot(relative, plane_x)
            y_coord = np.dot(relative, plane_y)
            polygon_2d.append((x_coord, y_coord))

        point_relative = projected_point - self.center
        point_2d = (np.dot(point_relative, plane_x), np.dot(point_relative, plane_y))

        # Use ray casting algorithm for 2D point-in-polygon test
        return point_in_polygon_2d(point_2d, polygon_2d)

    def circumference(self) -> float:
        """Calculate the circumference (perimeter) of the polygon."""
        circ = 0.0
        num_points = len(self.points)
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            circ += np.linalg.norm(p2 - p1)
        return circ

    def point_on_perimeter(self, s: float):
        """Get a point on the polygon perimeter given a parameter s in [0, circumference]."""
        circumference = self.circumference()
        s = s % circumference  # Wrap around

        accumulated_length = 0.0
        num_points = len(self.points)
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            edge_length = np.linalg.norm(p2 - p1)

            if accumulated_length + edge_length >= s:
                t = (s - accumulated_length) / edge_length
                point_on_edge = (1 - t) * p1 + t * p2
                return point_on_edge

            accumulated_length += edge_length
        # Fallback (should not reach here)
        return self.points[-1]

    def calc_inset_polygon_spec(self, inset_distance: float):
        """
        Inset a 3D polygon by moving each edge inward by 'inset_distance'.

        For each vertex, the new position lies on the angle bisector between
        the two adjacent edges, at a distance that creates the specified
        perpendicular offset to both edges.

        Args:
            inset_distance: Distance to move edges inward (must be > 0)

        Returns:
            PolygonSpec: New polygon with inset vertices

        Raises:
            ValueError: If inset_distance is too large or polygon is degenerate
        """
        if inset_distance <= 0:
            raise ValueError("Inset distance must be positive")

        num_points = len(self.points)
        if num_points < 3:
            raise ValueError("Polygon must have at least 3 points")

        # Check if polygon is roughly planar
        max_deviation = 0.0
        for point in self.points:
            deviation = abs(np.dot(point - self.center, self.normal))
            max_deviation = max(max_deviation, deviation)

        # For non-planar polygons, we'll work in the best-fit plane
        if max_deviation > 1e-6:
            # Project all points onto the best-fit plane
            projected_points = []
            for point in self.points:
                to_point = point - self.center
                distance_to_plane = np.dot(to_point, self.normal)
                projected_point = point - distance_to_plane * self.normal
                projected_points.append(projected_point)
            working_points = projected_points
        else:
            working_points = list(self.points)

        new_points = []

        for i in range(num_points):
            vi = working_points[i]
            vim = working_points[(i - 1) % num_points]  # previous vertex
            vip = working_points[(i + 1) % num_points]  # next vertex

            # Edge vectors from vi
            edge_prev = vim - vi  # vector to previous vertex
            edge_next = vip - vi  # vector to next vertex

            # Normalize edge vectors
            edge_prev_norm = normalize(edge_prev)
            edge_next_norm = normalize(edge_next)

            # Interior angle at vi
            dot = float(np.clip(np.dot(edge_prev_norm, edge_next_norm), -1.0, 1.0))
            theta = np.arccos(dot)  # angle between edges

            # Check for degenerate case (straight line)
            if abs(theta) < 1e-10 or abs(theta - np.pi) < 1e-10:
                raise ValueError(f"Degenerate angle at vertex {i}: {theta}")

            # Direction of internal angle bisector
            # The bisector direction is the sum of normalized edge vectors
            bis_dir = normalize(edge_prev_norm + edge_next_norm)

            # Determine if we need to flip the bisector direction
            # We want the bisector to point "inward" relative to the polygon
            # Test this by checking if moving along bisector brings us closer to polygon center
            test_point = vi + 0.001 * bis_dir
            to_center_original = self.center - vi
            to_center_test = self.center - test_point

            # If test point is farther from center, flip the bisector
            if np.linalg.norm(to_center_test) > np.linalg.norm(to_center_original):
                bis_dir = -bis_dir

            # Distance along bisector to achieve perpendicular offset = inset_distance
            # to both adjacent edges
            sin_half_theta = np.sin(theta / 2.0)
            if sin_half_theta < 1e-10:
                raise ValueError(
                    f"Cannot compute inset for very sharp angle at vertex {i}"
                )

            t = inset_distance / sin_half_theta

            # Check if inset is too large (would create invalid polygon)
            # Use more reasonable bounds based on triangle geometry
            min_edge = min(np.linalg.norm(edge_prev), np.linalg.norm(edge_next))
            max_reasonable_t = min_edge * 0.4  # Allow up to 40% of shortest edge
            if t > max_reasonable_t:
                raise ValueError(
                    f"Inset distance {inset_distance} too large at vertex {i} "
                    f"(would move {t:.3f} but max is {max_reasonable_t:.3f})"
                )

            new_vertex = vi + t * bis_dir
            new_points.append(new_vertex)

        return PolygonSpec(points=new_points)
