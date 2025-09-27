from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class RegionEdgeFeature:
    region_id: int
    edge_vertices: Tuple[int, int]  # Global vertex indices
    edge_coords: Tuple[np.ndarray, np.ndarray]  # (start, end) as 3D coordinates
    edge_vector: np.ndarray  # Normalized edge vector
    edge_centroid: np.ndarray  # Midpoint of the edge
    face_ids: List[int]  # Face indices that share this edge
    face_vertices: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]  # Actual face geometry
    face_normals: List[np.ndarray]  # Normals of the faces

    def edge_length(self) -> float:
        return np.linalg.norm(self.edge_coords[1] - self.edge_coords[0])
