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
    animation: Optional[dict[str, Any]] = None


def _normalize_xyz_tuple(value, *, field_name: str) -> Tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return tuple(float(component) for component in value)


def _normalize_alignment_list(value, *, field_name: str) -> tuple[str, ...]:
    if not value:
        raise ValueError(f"{field_name} must contain at least one alignment value")

    allowed_values = {"left", "right", "front", "back", "bottom", "top", "center"}
    normalized = tuple(str(component).strip().lower() for component in value)
    for index, component in enumerate(normalized):
        if component not in allowed_values:
            raise ValueError(
                f"{field_name}[{index}] must be one of {sorted(allowed_values)}"
            )
    return normalized


def _normalize_rotation_center(value, *, field_name: str):
    if len(value) == 3:
        try:
            return _normalize_xyz_tuple(value, field_name=field_name)
        except (TypeError, ValueError):
            pass

    return _normalize_alignment_list(value, field_name=field_name)


def _normalize_animation_entry(value, *, field_name: str):
    if hasattr(value, "items"):
        animation_type = str(value.get("type", "")).strip().lower()
        if animation_type != "rotation":
            raise ValueError(
                f"{field_name} must be an XYZ vector or a rotation animation"
            )

        center_field_names = [
            key for key in ("center", "origin", "origin_anchor") if key in value
        ]
        if len(center_field_names) != 1:
            raise ValueError(
                f"{field_name} rotation must define exactly one center, origin, or origin_anchor field"
            )

        normalized = {
            "type": "rotation",
            "axis": _normalize_xyz_tuple(
                value.get("axis"), field_name=f"{field_name} axis"
            ),
            "angle_degrees": float(value.get("angle_degrees")),
        }

        if "center" in value:
            center = _normalize_rotation_center(
                value.get("center"), field_name=f"{field_name} center"
            )
            if (
                isinstance(center, tuple)
                and len(center) == 3
                and all(isinstance(component, float) for component in center)
            ):
                normalized["center"] = center
            else:
                normalized["center_alignments"] = center
        elif "origin" in value:
            normalized["center"] = _normalize_xyz_tuple(
                value.get("origin"), field_name=f"{field_name} origin"
            )
        else:
            normalized["center_alignments"] = _normalize_alignment_list(
                value.get("origin_anchor"),
                field_name=f"{field_name} origin_anchor",
            )

        return normalized

    return _normalize_xyz_tuple(value, field_name=field_name)


def _serialize_animation_entry(value):
    if hasattr(value, "items"):
        serialized = {}
        for key, item_value in value.items():
            if isinstance(item_value, tuple):
                serialized[key] = list(item_value)
            else:
                serialized[key] = item_value
        return serialized
    return list(value)


def _normalize_animation(animation) -> Optional[dict[str, Any]]:
    if animation is None:
        return None
    if not hasattr(animation, "items"):
        raise ValueError(
            "animation must be a dict of animation key to XYZ vector or rotation animation"
        )

    normalized_animation: dict[str, Any] = {}
    for key, value in animation.items():
        if not isinstance(key, str) or not key:
            raise ValueError("animation keys must be non-empty strings")
        normalized_animation[key] = _normalize_animation_entry(
            value, field_name=f"animation vector for '{key}'"
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
                    {
                        key: _serialize_animation_entry(value)
                        for key, value in info.animation.items()
                    }
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
