from dataclasses import dataclass
from typing import Any, Optional, Tuple

from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)


@dataclass
class PartInfo:
    """Information about a part for production arrangement."""

    name: str
    part: Any  # CAD object type depends on the adapter
    flip: bool = False
    skip_in_production: bool = False
    prod_rotation_angle: Optional[float] = None
    prod_rotation_axis: Optional[Tuple[float, float, float]] = None
    color: Optional[Tuple[float, float, float]] = None  # RGB tuple (0.0-1.0)
    animation: Optional[dict[str, Tuple[float, float, float]]] = None


def _normalize_xyz_tuple(value, *, field_name: str) -> Tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return tuple(float(component) for component in value)


def _normalize_animation(animation) -> Optional[dict[str, Tuple[float, float, float]]]:
    if animation is None:
        return None
    if not hasattr(animation, "items"):
        raise ValueError("animation must be a dict of animation key to XYZ vector")

    normalized_animation: dict[str, Tuple[float, float, float]] = {}
    for key, vector in animation.items():
        if not isinstance(key, str) or not key:
            raise ValueError("animation keys must be non-empty strings")
        normalized_animation[key] = _normalize_xyz_tuple(
            vector, field_name=f"animation vector for '{key}'"
        )
    return normalized_animation


class PartList:
    """Container for managing named CadQuery parts."""

    def __init__(self):
        self.parts = []

    def add(
        self,
        part,
        name,
        *,
        flip=False,
        skip_in_production=False,
        prod_rotation_angle=None,
        prod_rotation_axis=None,
        color=None,
        animation=None,
    ):
        if isinstance(part, LeaderFollowersCuttersPart):
            shape = part.get_leader_as_part()
        else:
            shape = part

        if any(existing.name == name for existing in self.parts):
            raise ValueError(f"Part with name '{name}' already exists")

        axis_tuple = None
        if prod_rotation_axis is not None:
            axis_tuple = _normalize_xyz_tuple(
                prod_rotation_axis, field_name="prod_rotation_axis"
            )

        color_tuple = None
        if color is not None:
            color_tuple = _normalize_xyz_tuple(color, field_name="color")
            # Validate range
            if not all(0.0 <= c <= 1.0 for c in color_tuple):
                raise ValueError("color RGB values must be in the range 0.0-1.0")

        normalized_animation = _normalize_animation(animation)

        self.parts.append(
            PartInfo(
                name=name,
                part=shape,
                flip=flip,
                skip_in_production=skip_in_production,
                prod_rotation_angle=prod_rotation_angle,
                prod_rotation_axis=axis_tuple,
                color=color_tuple,
                animation=normalized_animation,
            )
        )

    def as_list(self):
        return [
            {
                "name": info.name,
                "part": info.part,
                "flip": info.flip,
                "skip_in_production": info.skip_in_production,
                "prod_rotation_angle": info.prod_rotation_angle,
                "prod_rotation_axis": (
                    list(info.prod_rotation_axis)
                    if info.prod_rotation_axis is not None
                    else None
                ),
                "color": (list(info.color) if info.color is not None else None),
                "animation": (
                    {key: list(vector) for key, vector in info.animation.items()}
                    if info.animation is not None
                    else None
                ),
            }
            for info in self.parts
        ]

    def __iter__(self):
        return iter(self.parts)

    def __len__(self):  # pragma: no cover - trivial
        return len(self.parts)

    def __getitem__(self, key):  # pragma: no cover - trivial
        return self.parts[key]
