"""Resolve human-readable semantic storey YAML into model geometry."""

from __future__ import annotations

import math
import re
from typing import Any

from shellforgepy.construct.alignment import Alignment

SCHEMA_VERSION = 2
V1_ONLY_KEYS = {
    "outer_footprint",
    "outer_wall_openings",
    "opening_candidates",
    "wall_specs",
    "wall_acceptance_gate",
    "excluded_measurements",
}

HORIZONTAL = "HORIZONTAL"
VERTICAL = "VERTICAL"
START = "START"
END = "END"
OUTSIDE_LEFT = "OUTSIDE_LEFT"

WALL_SIDES = {"LEFT", "RIGHT", "FRONT", "BACK"}
SWING_DIRECTIONS = WALL_SIDES
SEMANTIC_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_semantic_storey(storey: dict[str, Any]) -> None:
    """Validate semantic storey v2 structure enough for deterministic use."""

    if int(storey.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Expected semantic storey schema_version {SCHEMA_VERSION}")
    forbidden = V1_ONLY_KEYS.intersection(storey)
    if forbidden:
        raise ValueError(f"v1 semantic keys are not allowed in v2: {sorted(forbidden)}")

    refs = _reference_lines(storey)
    resolve_outer_footprint(storey)
    walls = resolve_inner_walls(storey)
    wall_by_id = {wall["id"]: wall for wall in walls}
    outer_cut_by_id = {cut["id"]: cut for cut in storey.get("outer_cuts", [])}

    for cutout in storey.get("floor_cutouts", []):
        _resolve_outline(cutout["outline"], refs)

    seen_exclusion_ids = set()
    for exclusion in storey.get("living_area_exclusions", []):
        _validate_living_area_exclusion_id(exclusion)
        if exclusion["id"] in seen_exclusion_ids:
            raise ValueError(f"Duplicate living area exclusion id: {exclusion['id']}")
        seen_exclusion_ids.add(exclusion["id"])
        _resolve_living_area_exclusion_polygon(exclusion, refs)

    for cut in storey.get("outer_cuts", []):
        _validate_positive(cut, "width")
        _validate_positive(cut, "height")
        if cut["wall"] not in WALL_SIDES:
            raise ValueError(f"Unsupported outer cut wall side: {cut['wall']}")
        if cut.get("offset_from", OUTSIDE_LEFT) != OUTSIDE_LEFT:
            raise ValueError("outer_cuts currently support only OUTSIDE_LEFT offsets")
        _outer_cut_segment(storey, cut)

    for door in storey.get("doors", []):
        _validate_positive(door, "width")
        _validate_positive(door, "height")
        if door.get("offset_from", START) not in {START, END}:
            raise ValueError(f"Unsupported door offset_from: {door.get('offset_from')}")
        if door.get("hinge_at", START) not in {START, END}:
            raise ValueError(f"Unsupported door hinge_at: {door.get('hinge_at')}")
        if door["swings_toward"] not in SWING_DIRECTIONS:
            raise ValueError(
                f"Unsupported door swing direction: {door['swings_toward']}"
            )

        wall_id = door["wall"]
        if wall_id in wall_by_id:
            wall = wall_by_id[wall_id]
            _door_opening_for_wall(wall, door)
            _validate_swing_direction(wall["orientation"], door["swings_toward"])
        elif wall_id in outer_cut_by_id:
            cut = outer_cut_by_id[wall_id]
            orientation = HORIZONTAL if cut["wall"] in {"FRONT", "BACK"} else VERTICAL
            _validate_swing_direction(orientation, door["swings_toward"])
        else:
            raise ValueError(f"Door references unknown wall or outer cut: {wall_id}")


def resolve_storey_geometry(storey: dict[str, Any]) -> dict[str, Any]:
    """Return geometry dictionaries consumed by CAD primitives and renderers."""

    validate_semantic_storey(storey)
    inner_walls = resolve_inner_walls(storey)
    wall_by_id = {wall["id"]: dict(wall, openings=[]) for wall in inner_walls}
    for door in storey.get("doors", []):
        wall_id = door["wall"]
        if wall_id not in wall_by_id:
            continue
        wall_by_id[wall_id]["openings"].append(
            _door_opening_for_wall(wall_by_id[wall_id], door)
        )

    wall_specs = []
    for wall in inner_walls:
        wall_spec = wall_by_id[wall["id"]]
        if not wall_spec["openings"]:
            wall_spec.pop("openings")
        wall_spec.pop("orientation")
        wall_specs.append(wall_spec)

    return {
        "outer_footprint": resolve_outer_footprint(storey),
        "outer_cuts": resolve_outer_cuts_as_windows(storey),
        "floor_cutouts": resolve_floor_cutouts(storey),
        "living_area_exclusions": resolve_living_area_exclusions(storey),
        "wall_specs": wall_specs,
        "door_symbols": resolve_door_symbols(storey),
    }


def resolve_outer_footprint(storey: dict[str, Any]) -> list[list[float]]:
    return _resolve_outline(storey["outer_walls"]["outline"], _reference_lines(storey))


def resolve_floor_cutouts(storey: dict[str, Any]) -> list[dict[str, Any]]:
    refs = _reference_lines(storey)
    return [
        {
            "id": cutout["id"],
            "kind": cutout.get("type", "cutout"),
            "polygon": _resolve_outline(cutout["outline"], refs),
        }
        for cutout in storey.get("floor_cutouts", [])
    ]


def resolve_living_area_exclusions(storey: dict[str, Any]) -> list[dict[str, Any]]:
    refs = _reference_lines(storey)
    return [
        {
            "id": exclusion["id"],
            "kind": exclusion.get("type", "excluded"),
            "polygon": _resolve_living_area_exclusion_polygon(exclusion, refs),
        }
        for exclusion in storey.get("living_area_exclusions", [])
    ]


def resolve_inner_walls(storey: dict[str, Any]) -> list[dict[str, Any]]:
    refs = _reference_lines(storey)
    default_thickness = float(storey["defaults"]["inner_wall_thickness"])
    walls = []
    for wall in storey.get("inner_walls", []):
        orientation = wall["orientation"]
        if orientation == HORIZONTAL:
            y = _ref_value(refs, "y", wall["at"])
            start = [_ref_value(refs, "x", wall["from"]), y]
            end = [_ref_value(refs, "x", wall["to"]), y]
        elif orientation == VERTICAL:
            x = _ref_value(refs, "x", wall["at"])
            start = [x, _ref_value(refs, "y", wall["from"])]
            end = [x, _ref_value(refs, "y", wall["to"])]
        else:
            raise ValueError(f"Unsupported wall orientation: {orientation}")
        if start == end:
            raise ValueError(f"Semantic wall has zero length: {wall['id']}")
        wall_spec = {
            "id": wall["id"],
            "orientation": orientation,
            "start": start,
            "end": end,
        }
        thickness = float(wall.get("thickness", default_thickness))
        if thickness != default_thickness:
            wall_spec["thickness"] = thickness
        walls.append(wall_spec)
    return walls


def resolve_outer_cuts_as_windows(storey: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for cut in storey.get("outer_cuts", []):
        windows.append(
            {
                "id": cut["id"],
                "kind": cut.get("type", "opening"),
                "location": getattr(Alignment, cut["wall"]),
                "width": float(cut["width"]),
                "height": float(cut["height"]),
                "offset_from_floor": float(cut.get("bottom", 0.0)),
                "offset_from_outside_left": float(cut["offset"]),
            }
        )
    return windows


def resolve_outer_cuts_as_openings(storey: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": cut["id"],
            "kind": cut.get("type", "opening"),
            "location": cut["wall"],
            "width": float(cut["width"]),
            "height": float(cut["height"]),
            "offset_from_floor": float(cut.get("bottom", 0.0)),
            "offset_from_outside_left": float(cut["offset"]),
        }
        for cut in storey.get("outer_cuts", [])
    ]


def resolve_door_symbols(storey: dict[str, Any]) -> list[dict[str, Any]]:
    refs = _reference_lines(storey)
    wall_by_id = {wall["id"]: wall for wall in resolve_inner_walls(storey)}
    outer_cut_by_id = {cut["id"]: cut for cut in storey.get("outer_cuts", [])}
    symbols = []
    for door in storey.get("doors", []):
        wall_id = door["wall"]
        if wall_id in wall_by_id:
            opening = _door_opening_for_wall(wall_by_id[wall_id], door)
            start, end = _opening_segment(wall_by_id[wall_id], opening)
            orientation = wall_by_id[wall_id]["orientation"]
        else:
            start, end = _outer_cut_segment(storey, outer_cut_by_id[wall_id])
            orientation = (
                HORIZONTAL
                if outer_cut_by_id[wall_id]["wall"] in {"FRONT", "BACK"}
                else VERTICAL
            )
        hinge = start if door.get("hinge_at", START) == START else end
        leaf_end = _swing_leaf_end(hinge, door["swings_toward"], float(door["width"]))
        symbols.append(
            {
                "id": door["id"],
                "wall": wall_id,
                "orientation": orientation,
                "start": start,
                "end": end,
                "hinge": hinge,
                "leaf_end": leaf_end,
                "swings_toward": door["swings_toward"],
                "source_detection": door.get("source_detection"),
            }
        )
    return symbols


def frame_width(storey: dict[str, Any]) -> float:
    return float(storey["frame"]["width"])


def frame_depth(storey: dict[str, Any]) -> float:
    return float(storey["frame"]["depth"])


def _reference_lines(storey: dict[str, Any]) -> dict[str, dict[str, float]]:
    refs = {
        "x": {
            key: float(value) for key, value in storey["reference_lines"]["x"].items()
        },
        "y": {
            key: float(value) for key, value in storey["reference_lines"]["y"].items()
        },
    }
    refs["x"].setdefault("LEFT", 0.0)
    refs["x"].setdefault("RIGHT", frame_width(storey))
    refs["y"].setdefault("FRONT", 0.0)
    refs["y"].setdefault("BACK", frame_depth(storey))
    return refs


def _resolve_outline(
    outline: list[dict[str, Any]], refs: dict[str, dict[str, float]]
) -> list[list[float]]:
    return [
        [_ref_value(refs, "x", point["x"]), _ref_value(refs, "y", point["y"])]
        for point in outline
    ]


def _resolve_living_area_exclusion_polygon(
    exclusion: dict[str, Any], refs: dict[str, dict[str, float]]
) -> list[list[float]]:
    if "outline" in exclusion:
        return _resolve_outline(exclusion["outline"], refs)
    if "rectangle" not in exclusion:
        raise ValueError(
            f"Living area exclusion {exclusion.get('id', exclusion)} requires rectangle or outline"
        )

    rectangle = exclusion["rectangle"]
    x_min = _ref_value(refs, "x", rectangle["x_min"])
    x_max = _ref_value(refs, "x", rectangle["x_max"])
    y_min = _ref_value(refs, "y", rectangle["y_min"])
    y_max = _ref_value(refs, "y", rectangle["y_max"])
    if x_max <= x_min:
        raise ValueError(
            f"Living area exclusion {exclusion['id']} has invalid x bounds"
        )
    if y_max <= y_min:
        raise ValueError(
            f"Living area exclusion {exclusion['id']} has invalid y bounds"
        )
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def _validate_living_area_exclusion_id(exclusion: dict[str, Any]) -> None:
    exclusion_id = str(exclusion.get("id", "")).strip()
    if not exclusion_id:
        raise ValueError("Living area exclusion id must be non-empty")
    if not SEMANTIC_ID_RE.fullmatch(exclusion_id):
        raise ValueError(
            f"Living area exclusion id must use lowercase letters, numbers, and underscores: {exclusion_id}"
        )


def _ref_value(refs: dict[str, dict[str, float]], axis: str, value: Any) -> float:
    if isinstance(value, str):
        try:
            return refs[axis][value]
        except KeyError as exc:
            raise ValueError(f"Unknown {axis} reference line: {value}") from exc
    return float(value)


def _validate_positive(item: dict[str, Any], key: str) -> None:
    if float(item[key]) <= 0:
        raise ValueError(f"{item.get('id', item)} {key} must be positive")


def _wall_length(wall: dict[str, Any]) -> float:
    return math.hypot(
        wall["end"][0] - wall["start"][0],
        wall["end"][1] - wall["start"][1],
    )


def _wall_unit(wall: dict[str, Any]) -> tuple[float, float]:
    length = _wall_length(wall)
    if length <= 0:
        raise ValueError(f"Wall has zero length: {wall['id']}")
    return (
        (wall["end"][0] - wall["start"][0]) / length,
        (wall["end"][1] - wall["start"][1]) / length,
    )


def _door_opening_for_wall(
    wall: dict[str, Any], door: dict[str, Any]
) -> dict[str, Any]:
    length = _wall_length(wall)
    width = float(door["width"])
    raw_offset = float(door["offset"])
    offset = (
        raw_offset
        if door.get("offset_from", START) == START
        else length - raw_offset - width
    )
    if offset < -1e-9 or offset + width > length + 1e-9:
        raise ValueError(f"Door {door['id']} does not fit inside wall {wall['id']}")
    return {
        "id": door["id"],
        "offset": round(max(0.0, offset), 6),
        "width": width,
        "height": float(door["height"]),
        "offset_from_floor": float(door.get("bottom", 0.0)),
        "source_detection": door.get("source_detection"),
    }


def _opening_segment(
    wall: dict[str, Any], opening: dict[str, Any]
) -> tuple[list[float], list[float]]:
    ux, uy = _wall_unit(wall)
    offset = float(opening["offset"])
    width = float(opening["width"])
    return (
        [
            wall["start"][0] + ux * offset,
            wall["start"][1] + uy * offset,
        ],
        [
            wall["start"][0] + ux * (offset + width),
            wall["start"][1] + uy * (offset + width),
        ],
    )


def _outer_cut_segment(
    storey: dict[str, Any], cut: dict[str, Any]
) -> tuple[list[float], list[float]]:
    width = frame_width(storey)
    depth = frame_depth(storey)
    offset = float(cut["offset"])
    cut_width = float(cut["width"])
    side = cut["wall"]
    if side == "FRONT":
        return [offset, 0.0], [offset + cut_width, 0.0]
    if side == "BACK":
        return [width - offset - cut_width, depth], [width - offset, depth]
    if side == "LEFT":
        y_min = depth - offset - cut_width
        return [0.0, y_min], [0.0, y_min + cut_width]
    if side == "RIGHT":
        return [width, offset], [width, offset + cut_width]
    raise ValueError(f"Unsupported outer cut wall side: {side}")


def _validate_swing_direction(orientation: str, swing: str) -> None:
    if orientation == HORIZONTAL and swing not in {"FRONT", "BACK"}:
        raise ValueError(f"Horizontal doors must swing FRONT or BACK, got {swing}")
    if orientation == VERTICAL and swing not in {"LEFT", "RIGHT"}:
        raise ValueError(f"Vertical doors must swing LEFT or RIGHT, got {swing}")


def _swing_leaf_end(
    hinge: list[float], swings_toward: str, radius: float
) -> list[float]:
    dx, dy = {
        "LEFT": (-radius, 0.0),
        "RIGHT": (radius, 0.0),
        "FRONT": (0.0, -radius),
        "BACK": (0.0, radius),
    }[swings_toward]
    return [hinge[0] + dx, hinge[1] + dy]
