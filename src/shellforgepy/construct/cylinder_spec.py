from dataclasses import dataclass

import numpy as np


@dataclass
class CylinderSpec:
    bottom: np.ndarray  # shape (3,)
    normal: np.ndarray  # shape (3,), must be normalized
    height: float
    radius: float

    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)  # ensure unit
