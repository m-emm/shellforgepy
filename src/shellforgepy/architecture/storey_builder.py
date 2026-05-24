"""Builder generator for semantic architecture storey assemblies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml
from shellforgepy.architecture.semantic_storey import (
    frame_depth,
    frame_width,
    resolve_storey_geometry,
    validate_semantic_storey,
)
from shellforgepy.architecture.storey import (
    calculate_inner_floor_area_breakdown,
    create_storey,
    resolve_floor_cutouts_for_storey,
)
from shellforgepy.simple import translate


def _storey_specification(architecture: Mapping[str, Any]) -> dict[str, Any]:
    spec = architecture.get("storey_specification")
    if spec is None:
        spec = architecture.get("StoreySpecification")
    if not isinstance(spec, Mapping):
        raise ValueError("architecture.storey_specification must be provided")
    return dict(spec)


def _load_semantic_storey(path: str) -> dict[str, Any]:
    semantic_path = Path(path).expanduser().resolve()
    storey = yaml.safe_load(semantic_path.read_text(encoding="utf-8"))
    validate_semantic_storey(storey)
    return storey


def _storey_stack_value(
    storey_stack: Mapping[str, Any],
    key: str,
    semantic_storey: Mapping[str, Any],
    semantic_default_key: str | None = None,
):
    if key in storey_stack:
        return storey_stack[key]
    if semantic_default_key is not None:
        defaults = semantic_storey.get("defaults") or {}
        if semantic_default_key in defaults:
            return defaults[semantic_default_key]
    raise ValueError(f"storey_stack.{key} is required")


def _storey_height(storey_stack: Mapping[str, Any], storey_index: int) -> float:
    floor_heights = list(storey_stack["floor_heights"])
    floor_index = storey_index - 1
    if floor_index < 0 or floor_index >= len(floor_heights):
        raise ValueError(f"storey_index {storey_index} has no floor height")
    return float(floor_heights[floor_index])


def _storey_z_base(storey_stack: Mapping[str, Any], storey_index: int) -> float:
    floor_bases = list(storey_stack.get("floor_bases", []))
    floor_index = storey_index - 1
    if floor_index < 0 or floor_index >= len(floor_bases):
        return 0.0
    return float(floor_bases[floor_index]) + float(
        storey_stack.get("floor_bases_z_offset", 0.0)
    )


def _storey_metadata(
    *,
    spec: Mapping[str, Any],
    semantic_storey: Mapping[str, Any],
    semantic_geometry: Mapping[str, Any],
    dimensions: Mapping[str, Any],
    floor_cutouts: list[dict[str, Any]],
    living_area_breakdown,
) -> dict[str, Any]:
    return {
        "type": "semantic_storey",
        "id": spec["id"],
        "storey_index": int(spec["storey_index"]),
        "semantic_path": spec["path"],
        "semantic_sha256": spec.get("sha256"),
        "semantic": dict(semantic_storey),
        "geometry": {
            **dict(semantic_geometry),
            "floor_cutouts": floor_cutouts,
        },
        "dimensions": dict(dimensions),
        "living_area_breakdown": asdict(living_area_breakdown),
    }


def build_storey(
    *,
    architecture: Mapping[str, Any],
    storey_stack: Mapping[str, Any],
    living_space: Mapping[str, Any] | None = None,
):
    """Build a positioned storey assembly from a semantic storey YAML file."""

    spec = _storey_specification(architecture)
    semantic_storey = _load_semantic_storey(spec["path"])
    semantic_geometry = resolve_storey_geometry(semantic_storey)
    storey_index = int(spec["storey_index"])

    width = frame_width(semantic_storey)
    depth = frame_depth(semantic_storey)
    height = _storey_height(storey_stack, storey_index)
    z_base = _storey_z_base(storey_stack, storey_index)
    outer_wall_thickness = float(
        _storey_stack_value(
            storey_stack,
            "outer_wall_thickness",
            semantic_storey,
            "outer_wall_thickness",
        )
    )
    floor_thickness = float(
        _storey_stack_value(storey_stack, "floor_thickness", semantic_storey)
    )
    inner_wall_thickness = float(
        storey_stack.get(
            "inner_wall_thickness",
            (semantic_storey.get("defaults") or {}).get("inner_wall_thickness", 0.0),
        )
    )

    floor_cutouts = [
        *resolve_floor_cutouts_for_storey(
            width,
            depth,
            outer_wall_thickness,
            living_space,
            storey_index,
        ),
        *semantic_geometry["floor_cutouts"],
    ]
    living_area_breakdown = calculate_inner_floor_area_breakdown(
        width,
        depth,
        outer_wall_thickness,
        inner_wall_thickness=inner_wall_thickness,
        wall_specs=semantic_geometry["wall_specs"],
        floor_cutouts=floor_cutouts,
        living_area_exclusions=semantic_geometry["living_area_exclusions"],
        outer_footprint=semantic_geometry["outer_footprint"],
    )

    storey = create_storey(
        width=width,
        depth=depth,
        height=height,
        outer_wall_thickness=outer_wall_thickness,
        floor_thickness=floor_thickness,
        windows=semantic_geometry["outer_cuts"],
        inner_wall_thickness=inner_wall_thickness,
        floor_cutouts=floor_cutouts,
        living_area_exclusions=semantic_geometry["living_area_exclusions"],
        outer_footprint=semantic_geometry["outer_footprint"],
        wall_specs=semantic_geometry["wall_specs"],
    )
    if z_base:
        storey = translate(0.0, 0.0, z_base)(storey)

    dimensions = {
        "width": width,
        "depth": depth,
        "height": height,
        "z_base": z_base,
        "outer_wall_thickness": outer_wall_thickness,
        "floor_thickness": floor_thickness,
        "inner_wall_thickness": inner_wall_thickness,
    }
    storey.additional_data.setdefault("architecture", {})
    storey.additional_data["architecture"]["storey"] = _storey_metadata(
        spec=spec,
        semantic_storey=semantic_storey,
        semantic_geometry=semantic_geometry,
        dimensions=dimensions,
        floor_cutouts=floor_cutouts,
        living_area_breakdown=living_area_breakdown,
    )
    return storey
