"""Small mesh scene objects for OBJ passthrough export."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ObjMesh:
    """A plain triangle mesh that is already ready for OBJ scene export."""

    vertices: np.ndarray
    faces: np.ndarray
    name: str | None = None
    color: tuple[float, float, float] | None = None
    uvs: np.ndarray | None = None
    texture_path: str | Path | None = None
    material_name: str | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> "ObjMesh":
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("ObjMesh.vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("ObjMesh.faces must have shape (M, 3)")
        if len(vertices) == 0:
            raise ValueError("ObjMesh.vertices must not be empty")
        if len(faces) == 0:
            raise ValueError("ObjMesh.faces must not be empty")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("ObjMesh.vertices must contain only finite values")
        if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
            raise ValueError("ObjMesh.faces contains vertex indices out of range")

        self.vertices = vertices
        self.faces = faces

        if self.uvs is not None:
            uvs = np.asarray(self.uvs, dtype=float)
            if uvs.ndim != 2 or uvs.shape != (len(vertices), 2):
                raise ValueError("ObjMesh.uvs must have shape (N, 2)")
            if not np.all(np.isfinite(uvs)):
                raise ValueError("ObjMesh.uvs must contain only finite values")
            self.uvs = uvs

        if self.color is not None:
            if len(self.color) != 3:
                raise ValueError("ObjMesh.color must contain three RGB values")
            color = tuple(float(component) for component in self.color)
            if not all(0.0 <= component <= 1.0 for component in color):
                raise ValueError("ObjMesh.color RGB values must be in 0.0-1.0")
            self.color = color

        if self.metadata is not None and not hasattr(self.metadata, "items"):
            raise ValueError("ObjMesh.metadata must be a mapping")

        return self

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        self.validate()
        return (
            tuple(float(value) for value in np.min(self.vertices, axis=0)),
            tuple(float(value) for value in np.max(self.vertices, axis=0)),
        )

    def transformed(self, matrix) -> "ObjMesh":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("ObjMesh.transformed() expects a 4x4 affine matrix")

        self.validate()
        homogeneous = np.column_stack(
            (self.vertices, np.ones(len(self.vertices), dtype=float))
        )
        transformed_vertices = homogeneous @ matrix.T
        w = transformed_vertices[:, 3]
        if np.any(np.isclose(w, 0.0)):
            raise ValueError("ObjMesh transform produced points with zero W")
        transformed_vertices = transformed_vertices[:, :3] / w[:, None]

        return ObjMesh(
            vertices=transformed_vertices,
            faces=np.array(self.faces, copy=True),
            name=self.name,
            color=self.color,
            uvs=None if self.uvs is None else np.array(self.uvs, copy=True),
            texture_path=self.texture_path,
            material_name=self.material_name,
            metadata=deepcopy(self.metadata),
        )
