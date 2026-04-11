from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shellforgepy.render.model import MeshObject, Scene

DEFAULT_COLOR = (0.72, 0.76, 0.82)


@dataclass
class _PendingObject:
    name: str
    material_name: str | None = None
    face_indices: list[list[int]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def _parse_material_color_tokens(tokens: list[str]) -> tuple[float, float, float]:
    if len(tokens) < 4:
        raise ValueError("Invalid Kd line in MTL file")
    return (float(tokens[1]), float(tokens[2]), float(tokens[3]))


def load_mtl_colors(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Load diffuse material colors from an MTL file."""

    colors: dict[str, tuple[float, float, float]] = {}
    current_material: str | None = None

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            keyword = tokens[0]
            if keyword == "newmtl" and len(tokens) >= 2:
                current_material = tokens[1]
            elif keyword == "Kd" and current_material is not None:
                colors[current_material] = _parse_material_color_tokens(tokens)

    return colors


def _flush_object(
    pending: _PendingObject | None,
    vertices: np.ndarray,
    material_colors: dict[str, tuple[float, float, float]],
    objects: list[MeshObject],
) -> None:
    if pending is None or not pending.face_indices:
        return

    flat_indices = sorted({idx for face in pending.face_indices for idx in face})
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(flat_indices)}
    object_vertices = vertices[flat_indices]
    object_faces = np.asarray(
        [[old_to_new[idx] for idx in face] for face in pending.face_indices],
        dtype=np.int32,
    )
    color = material_colors.get(pending.material_name or "", DEFAULT_COLOR)
    objects.append(
        MeshObject(
            name=pending.name,
            vertices=np.asarray(object_vertices, dtype=np.float32),
            faces=object_faces,
            color=color,
            metadata=dict(pending.metadata),
        )
    )


def _parse_obj_face_index(token: str, vertex_count: int) -> int:
    raw_index = token.split("/")[0]
    if not raw_index:
        raise ValueError(f"Unsupported OBJ face token: {token!r}")
    idx = int(raw_index)
    if idx > 0:
        return idx - 1
    return vertex_count + idx


def _parse_shellforgepy_comment(line: str, metadata: dict[str, object]) -> bool:
    payload = line[1:].strip()
    if payload.startswith("shellforgepy_hierarchy_labels "):
        metadata["hierarchy_labels"] = payload.split(" ", 1)[1].split("/")
        return True
    if payload.startswith("shellforgepy_hierarchy "):
        metadata["hierarchy"] = payload.split(" ", 1)[1].split("/")
        return True
    if payload.startswith("shellforgepy_builder_selector "):
        metadata["builder_selector"] = payload.split(" ", 1)[1]
        return True
    return False


def load_obj_scene(path: str | Path) -> Scene:
    """Load a ShellForgePy-style OBJ scene with per-object colors."""

    obj_path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    pending_objects: list[_PendingObject] = []
    pending_metadata: dict[str, object] = {}
    pending_obj: _PendingObject | None = None
    material_lib_name: str | None = None

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                _parse_shellforgepy_comment(line, pending_metadata)
                continue

            tokens = line.split()
            keyword = tokens[0]

            if keyword == "mtllib" and len(tokens) >= 2:
                material_lib_name = tokens[1]
                continue

            if keyword == "v" and len(tokens) >= 4:
                vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
                continue

            if keyword == "o":
                if pending_obj is not None:
                    pending_objects.append(pending_obj)
                pending_obj = _PendingObject(
                    name=(
                        tokens[1]
                        if len(tokens) >= 2
                        else f"object_{len(pending_objects) + 1}"
                    ),
                    metadata=dict(pending_metadata),
                )
                pending_metadata.clear()
                continue

            if keyword == "usemtl":
                if pending_obj is None:
                    pending_obj = _PendingObject(
                        name=f"object_{len(pending_objects) + 1}",
                        metadata=dict(pending_metadata),
                    )
                    pending_metadata.clear()
                pending_obj.material_name = tokens[1] if len(tokens) >= 2 else None
                continue

            if keyword == "f":
                if pending_obj is None:
                    pending_obj = _PendingObject(
                        name=f"object_{len(pending_objects) + 1}",
                        metadata=dict(pending_metadata),
                    )
                    pending_metadata.clear()

                face = [
                    _parse_obj_face_index(token, len(vertices)) for token in tokens[1:]
                ]
                if len(face) < 3:
                    continue
                if len(face) == 3:
                    pending_obj.face_indices.append(face)
                else:
                    for idx in range(1, len(face) - 1):
                        pending_obj.face_indices.append(
                            [face[0], face[idx], face[idx + 1]]
                        )

    vertex_array = np.asarray(vertices, dtype=np.float32)
    if pending_obj is not None:
        pending_objects.append(pending_obj)

    material_colors = {}
    if material_lib_name:
        mtl_path = obj_path.with_name(material_lib_name)
        if mtl_path.exists():
            material_colors = load_mtl_colors(mtl_path)

    objects_out: list[MeshObject] = []
    for pending in pending_objects:
        _flush_object(pending, vertex_array, material_colors, objects_out)
    return Scene(objects_out)
