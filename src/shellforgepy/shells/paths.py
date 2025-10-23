from dataclasses import dataclass

import numpy as np


@dataclass
class PointOnEdge:
    start_vertex_index: int
    end_vertex_index: int
    position: float  # Ranges from 0 (at start vertex) to 1 (at end vertex)

    # validation
    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("Position must be between 0 and 1")


class VertexPath:

    def __init__(self, vertices, path_vertex_indices):
        self.vertices = vertices
        self.path_vertex_indices = path_vertex_indices

    def path_point_at_length(self, length):
        vertices = [self.vertices[p] for p in self.path_vertex_indices]

        path_length_progressive = self.calculate_path_length_progressive(vertices)

        for i in range(1, len(vertices)):
            if path_length_progressive[i] >= length:
                ratio = (length - path_length_progressive[i - 1]) / (
                    path_length_progressive[i] - path_length_progressive[i - 1]
                )
                point_on_edge = PointOnEdge(
                    start_vertex_index=self.path_vertex_indices[i - 1],
                    end_vertex_index=self.path_vertex_indices[i],
                    position=float(ratio),
                )
                return point_on_edge
        return None

    def point_on_edge_coordinates(self, point_on_edge: PointOnEdge):
        start_vertex = self.vertices[point_on_edge.start_vertex_index]
        end_vertex = self.vertices[point_on_edge.end_vertex_index]
        point = start_vertex + point_on_edge.position * (end_vertex - start_vertex)
        return point

    def point_on_edge_length(self, point_on_edge: PointOnEdge):
        path_length_progressive = self.calculate_path_length_progressive(
            [self.vertices[p] for p in self.path_vertex_indices]
        )

        start_index_in_path = self.path_vertex_indices.index(
            point_on_edge.start_vertex_index
        )
        length_up_to_start = path_length_progressive[start_index_in_path]

        start_vertex = self.vertices[point_on_edge.start_vertex_index]
        end_vertex = self.vertices[point_on_edge.end_vertex_index]
        edge_length = np.linalg.norm(end_vertex - start_vertex)

        total_length = length_up_to_start + point_on_edge.position * edge_length

        return total_length

    def length(self):
        path_length_progressive = self.calculate_path_length_progressive(
            [self.vertices[p] for p in self.path_vertex_indices]
        )
        return path_length_progressive[-1]

    @staticmethod
    def calculate_path_length_progressive(vertices):
        retval = []
        last_vertex = None
        for current_vertex in vertices:
            if last_vertex is None:
                retval.append(0)
            else:
                retval.append(np.linalg.norm(current_vertex - last_vertex) + retval[-1])

            last_vertex = current_vertex
        return retval

    def recurse_partition(self, left, right, min_distance, is_first=False):

        mid_length = (left + right) / 2
        left_point = self.path_point_at_length(left)
        right_point = self.path_point_at_length(right)
        mid_point = self.path_point_at_length(mid_length)

        if mid_point is None:
            raise ValueError("Mid point calculation failed in partitioning.")

        partitions = []

        if (
            np.linalg.norm(
                self.point_on_edge_coordinates(mid_point)
                - self.point_on_edge_coordinates(left_point)
            )
            < min_distance
            and not is_first
        ):
            return []

        if (
            np.linalg.norm(
                self.point_on_edge_coordinates(mid_point)
                - self.point_on_edge_coordinates(right_point)
            )
            < min_distance
            and not is_first
        ):
            return []

        partitions += self.recurse_partition(left, mid_length, min_distance)

        partitions.append(mid_point)

        partitions += self.recurse_partition(mid_length, right, min_distance)

        return partitions

    def calc_partitioning(self, min_distance) -> list[PointOnEdge]:
        return self.recurse_partition(
            0,
            self.calculate_path_length_progressive(
                [self.vertices[p] for p in self.path_vertex_indices]
            )[-1],
            min_distance,
            is_first=True,
        )

    def average_vertex_function(self, start_length, end_length, func, resolution=100):
        """
        Calculate the average value of a vertex function along the path segment
        defined by start_length and end_length.
        Works also for functions which return non-scalar values (np.arrays, such as normals / colors/ etc)
        Args:
            start_length (float): The starting length along the path.
            end_length (float): The ending length along the path.
            func (callable): A function that takes a vertex index and returns a value.
            resolution (int): The number of samples to take along the path segment.

        """

        total_value = None

        for i in range(resolution + 1):
            sample_length = start_length + (end_length - start_length) * (
                i / resolution
            )
            point_on_edge = self.path_point_at_length(sample_length)

            start_vertex_index = point_on_edge.start_vertex_index
            end_vertex_index = point_on_edge.end_vertex_index

            start_vertex_value = func(start_vertex_index)
            end_vertex_value = func(end_vertex_index)

            value_at_point = (
                start_vertex_value
                + (end_vertex_value - start_vertex_value) * point_on_edge.position
            )

            if total_value is None:
                total_value = value_at_point
            else:
                total_value += value_at_point

        average_value = total_value / (resolution + 1)
        return average_value
