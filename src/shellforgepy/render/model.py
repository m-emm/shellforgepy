from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MeshObject:
    """Single mesh object with a solid display color."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(self.vertices, dtype=np.float32)
        return vertices.min(axis=0), vertices.max(axis=0)


@dataclass(slots=True)
class Scene:
    """Collection of independently colored mesh objects."""

    objects: list[MeshObject]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.objects:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero

        mins = []
        maxs = []
        for obj in self.objects:
            obj_min, obj_max = obj.bounds()
            mins.append(obj_min)
            maxs.append(obj_max)
        return np.min(np.stack(mins, axis=0), axis=0), np.max(
            np.stack(maxs, axis=0), axis=0
        )

    def triangle_count(self) -> int:
        return int(sum(len(obj.faces) for obj in self.objects))

    def vertex_count(self) -> int:
        return int(sum(len(obj.vertices) for obj in self.objects))

    def object_count(self) -> int:
        return int(len(self.objects))
