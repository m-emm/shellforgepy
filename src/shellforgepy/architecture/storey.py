"""CAD primitives and area calculations for semantic architecture storeys."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shellforgepy.architecture.polygon_clipping import (
    clean_polygon_points,
    inset_polygon,
    polygon_area,
    polygon_covers,
    union_area,
)
from shellforgepy.simple import *

LIVING_AREA_EXCLUSION_PREFIX = "living_area_exclusion_"
LIVING_AREA_EXCLUSION_MARKER_THICKNESS = 0.02


@dataclass(frozen=True)
class InnerFloorAreaBreakdown:
    square_meters: float
    inner_floor_area: float
    inner_wall_area: float
    floor_cutout_area: float
    excluded_area: float
    non_living_floor_area: float
    excluded_areas_by_id: dict[str, float]


def create_storey_hull(width, depth, height, outer_wall_thickness, floor_thickness):
    inner_width = width - 2 * outer_wall_thickness
    inner_depth = depth - 2 * outer_wall_thickness
    inner_height = height - floor_thickness

    outer_box = create_box(width, depth, height)
    inner_box = create_box(inner_width, inner_depth, inner_height)

    inner_box = align(inner_box, outer_box, Alignment.CENTER)
    inner_box = translate(0, 0, floor_thickness)(inner_box)

    return outer_box.cut(inner_box)


def _clean_polygon_points(points):
    return clean_polygon_points(points)


def _polygon_from_points(points, *, label):
    try:
        polygon = _clean_polygon_points(points)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid non-empty polygon")
    return polygon


def _single_polygon_from_geometry(geometry, *, label):
    if not geometry:
        raise ValueError(f"{label} produced an empty polygon")
    return _clean_polygon_points(geometry)


def _polygon_exterior_points(polygon):
    return _clean_polygon_points(polygon)


def _create_extruded_polygon_at(points, thickness, z_origin=0.0):
    part = create_extruded_polygon(_clean_polygon_points(points), thickness)
    if z_origin:
        part = translate(0.0, 0.0, z_origin)(part)
    return part


def _create_storey_hull_from_footprint(
    outer_footprint,
    height,
    outer_wall_thickness,
    floor_thickness,
):
    outer_polygon = _polygon_from_points(outer_footprint, label="outer_footprint")
    inner_polygon = inset_polygon(
        outer_polygon,
        float(outer_wall_thickness),
        label="outer_footprint inset",
    )

    outer = _create_extruded_polygon_at(outer_polygon, height)
    inner_cut_overhang = 0.005
    inner = _create_extruded_polygon_at(
        inner_polygon,
        height - floor_thickness + inner_cut_overhang,
        floor_thickness,
    )
    return outer.cut(inner)


def _inner_space_bounds(width, depth, height, outer_wall_thickness, floor_thickness):
    return {
        "x_min": outer_wall_thickness,
        "x_max": width - outer_wall_thickness,
        "y_min": outer_wall_thickness,
        "y_max": depth - outer_wall_thickness,
        "z_min": floor_thickness,
        "z_max": height,
    }


def _inner_floor_bounds(width, depth, outer_wall_thickness):
    return {
        "x_min": outer_wall_thickness,
        "x_max": width - outer_wall_thickness,
        "y_min": outer_wall_thickness,
        "y_max": depth - outer_wall_thickness,
    }


def _inner_floor_polygon(width, depth, outer_wall_thickness, outer_footprint=None):
    if outer_footprint is None:
        inner_width = width - 2.0 * outer_wall_thickness
        inner_depth = depth - 2.0 * outer_wall_thickness
        if inner_width <= 0 or inner_depth <= 0:
            raise ValueError("outer_wall_thickness leaves no positive inner floor area")
        bounds = _inner_floor_bounds(width, depth, outer_wall_thickness)
        return _clean_polygon_points(
            [
                (bounds["x_min"], bounds["y_min"]),
                (bounds["x_max"], bounds["y_min"]),
                (bounds["x_max"], bounds["y_max"]),
                (bounds["x_min"], bounds["y_max"]),
            ]
        )

    outer_polygon = _polygon_from_points(outer_footprint, label="outer_footprint")
    return inset_polygon(
        outer_polygon,
        float(outer_wall_thickness),
        label="outer_footprint inset",
    )


def _floor_cutout_origin(floor_cutout):
    if "origin" in floor_cutout:
        origin = floor_cutout["origin"]
        return float(origin[0]), float(origin[1])
    return float(floor_cutout["x"]), float(floor_cutout["y"])


def _is_polygon_floor_cutout(floor_cutout):
    return "polygon" in floor_cutout


def _floor_cutout_bounds(floor_cutout):
    x_origin, y_origin = _floor_cutout_origin(floor_cutout)
    length = float(floor_cutout["length"])
    cutout_depth = float(floor_cutout["depth"])
    return {
        "x_min": x_origin,
        "x_max": x_origin + length,
        "y_min": y_origin,
        "y_max": y_origin + cutout_depth,
        "length": length,
        "depth": cutout_depth,
    }


def _floor_cutout_origin_from_location(location, length, depth, bounds):
    if location == "FRONT_LEFT":
        return bounds["x_min"], bounds["y_min"]
    if location == "FRONT_RIGHT":
        return bounds["x_max"] - length, bounds["y_min"]
    if location == "BACK_LEFT":
        return bounds["x_min"], bounds["y_max"] - depth
    if location == "BACK_RIGHT":
        return bounds["x_max"] - length, bounds["y_max"] - depth
    if location == "CENTER":
        return (
            (bounds["x_min"] + bounds["x_max"] - length) * 0.5,
            (bounds["y_min"] + bounds["y_max"] - depth) * 0.5,
        )
    raise ValueError(f"Unsupported floor cutout location: {location}")


def resolve_floor_cutout(width, depth, outer_wall_thickness, floor_cutout):
    length = float(floor_cutout["length"])
    cutout_depth = float(floor_cutout["depth"])
    if "origin" in floor_cutout or ("x" in floor_cutout and "y" in floor_cutout):
        x_origin, y_origin = _floor_cutout_origin(floor_cutout)
    else:
        x_origin, y_origin = _floor_cutout_origin_from_location(
            floor_cutout["location"],
            length,
            cutout_depth,
            _inner_floor_bounds(width, depth, outer_wall_thickness),
        )

    return {
        "origin": (x_origin, y_origin),
        "length": length,
        "depth": cutout_depth,
    }


def resolve_floor_cutouts_for_storey(
    width, depth, outer_wall_thickness, living_space, storey_index
):
    if not living_space:
        return []

    floor_cutouts = []
    for floor_cutout in living_space.get("floor_cutouts", []):
        storey_indices = floor_cutout.get("storey_indices", [])
        if storey_index not in storey_indices:
            continue
        floor_cutouts.append(
            resolve_floor_cutout(width, depth, outer_wall_thickness, floor_cutout)
        )
    return floor_cutouts


def _validate_floor_cutout(
    width, depth, outer_wall_thickness, floor_cutout, outer_footprint=None
):
    if _is_polygon_floor_cutout(floor_cutout):
        return _validate_polygon_floor_cutout(
            width, depth, outer_wall_thickness, floor_cutout, outer_footprint
        )

    cutout_bounds = _floor_cutout_bounds(floor_cutout)
    if cutout_bounds["length"] <= 0:
        raise ValueError(
            f"Floor cutout length must be positive, got {cutout_bounds['length']}"
        )
    if cutout_bounds["depth"] <= 0:
        raise ValueError(
            f"Floor cutout depth must be positive, got {cutout_bounds['depth']}"
        )

    bounds = _inner_floor_bounds(width, depth, outer_wall_thickness)
    if (
        cutout_bounds["x_min"] < bounds["x_min"]
        or cutout_bounds["x_max"] > bounds["x_max"]
        or cutout_bounds["y_min"] < bounds["y_min"]
        or cutout_bounds["y_max"] > bounds["y_max"]
    ):
        raise ValueError("Floor cutout must stay inside inner floor area")

    return cutout_bounds


def _validate_polygon_floor_cutout(
    width, depth, outer_wall_thickness, floor_cutout, outer_footprint=None
):
    polygon = _polygon_from_points(floor_cutout["polygon"], label="floor cutout")
    inner_floor_polygon = _inner_floor_polygon(
        width, depth, outer_wall_thickness, outer_footprint
    )
    if not polygon_covers(inner_floor_polygon, polygon):
        raise ValueError("Floor cutout must stay inside inner floor area")
    return polygon


def _floor_region_polygon(
    width,
    depth,
    outer_wall_thickness,
    floor_region,
    outer_footprint=None,
    *,
    label,
    inside_error,
):
    if _is_polygon_floor_cutout(floor_region):
        polygon = _polygon_from_points(floor_region["polygon"], label=label)
    else:
        bounds = _floor_cutout_bounds(floor_region)
        if bounds["length"] <= 0:
            raise ValueError(f"{label} length must be positive, got {bounds['length']}")
        if bounds["depth"] <= 0:
            raise ValueError(f"{label} depth must be positive, got {bounds['depth']}")
        polygon = _clean_polygon_points(
            [
                (bounds["x_min"], bounds["y_min"]),
                (bounds["x_max"], bounds["y_min"]),
                (bounds["x_max"], bounds["y_max"]),
                (bounds["x_min"], bounds["y_max"]),
            ]
        )

    inner_floor_polygon = _inner_floor_polygon(
        width, depth, outer_wall_thickness, outer_footprint
    )
    if not polygon_covers(inner_floor_polygon, polygon):
        raise ValueError(inside_error)
    return polygon


def _create_floor_cutout(
    width,
    depth,
    outer_wall_thickness,
    floor_thickness,
    floor_cutout,
    outer_footprint=None,
):
    if _is_polygon_floor_cutout(floor_cutout):
        _validate_polygon_floor_cutout(
            width, depth, outer_wall_thickness, floor_cutout, outer_footprint
        )
        cut_overhang = 0.005
        return _create_extruded_polygon_at(
            floor_cutout["polygon"], floor_thickness + cut_overhang, -cut_overhang
        )

    cutout_bounds = _validate_floor_cutout(
        width, depth, outer_wall_thickness, floor_cutout, outer_footprint
    )
    cut_overhang = 0.005
    return create_box(
        cutout_bounds["length"],
        cutout_bounds["depth"],
        floor_thickness + cut_overhang,
        origin=(
            cutout_bounds["x_min"],
            cutout_bounds["y_min"],
            -cut_overhang,
        ),
    )


def _create_living_area_exclusion_marker(
    width,
    depth,
    outer_wall_thickness,
    floor_thickness,
    exclusion,
    outer_footprint=None,
):
    polygon = _floor_region_polygon(
        width,
        depth,
        outer_wall_thickness,
        exclusion,
        outer_footprint,
        label="living area exclusion",
        inside_error="Living-area exclusion must stay inside inner floor area",
    )
    return _create_extruded_polygon_at(
        polygon,
        LIVING_AREA_EXCLUSION_MARKER_THICKNESS,
        floor_thickness,
    )


def _is_segment_wall_spec(wall_spec):
    return "start" in wall_spec and "end" in wall_spec


def _inner_wall_spec_thickness(wall_spec, default_thickness):
    return float(wall_spec.get("thickness", default_thickness))


def _inner_wall_spec_length(wall_spec):
    if _is_segment_wall_spec(wall_spec):
        start = wall_spec["start"]
        end = wall_spec["end"]
        return math.hypot(
            float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        )
    return float(wall_spec["length"])


def _calculate_inner_wall_area(wall_specs, inner_wall_thickness):
    wall_specs = [] if wall_specs is None else wall_specs
    if wall_specs and inner_wall_thickness <= 0:
        raise ValueError(
            f"inner_wall_thickness must be positive when wall_specs are provided, got {inner_wall_thickness}"
        )

    inner_wall_area = 0.0
    for wall_spec in wall_specs:
        length = _inner_wall_spec_length(wall_spec)
        if length <= 0:
            raise ValueError(f"Inner wall length must be positive, got {length}")
        thickness = _inner_wall_spec_thickness(wall_spec, inner_wall_thickness)
        opening_area = 0.0
        for opening in wall_spec.get("openings", []):
            opening_area += float(opening["width"]) * thickness
        inner_wall_area += max(0.0, float(length) * thickness - opening_area)
    return inner_wall_area


def _non_empty_living_area_exclusion_id(exclusion):
    exclusion_id = str(exclusion.get("id", "")).strip()
    if not exclusion_id:
        raise ValueError("Living-area exclusion id must be non-empty")
    return exclusion_id


def _union_area(polygons):
    return union_area(polygons)


def calculate_inner_floor_area_breakdown(
    width,
    depth,
    outer_wall_thickness,
    *,
    inner_wall_thickness=0.0,
    wall_specs=None,
    floor_cutouts=None,
    living_area_exclusions=None,
    outer_footprint=None,
):
    inner_floor_polygon = _inner_floor_polygon(
        width, depth, outer_wall_thickness, outer_footprint
    )
    inner_floor_area = polygon_area(inner_floor_polygon)
    inner_wall_area = _calculate_inner_wall_area(wall_specs, inner_wall_thickness)

    floor_cutout_polygons = []
    floor_cutouts = [] if floor_cutouts is None else floor_cutouts
    for floor_cutout in floor_cutouts:
        floor_cutout_polygons.append(
            _floor_region_polygon(
                width,
                depth,
                outer_wall_thickness,
                floor_cutout,
                outer_footprint,
                label="floor cutout",
                inside_error="Floor cutout must stay inside inner floor area",
            )
        )

    living_area_exclusions = (
        [] if living_area_exclusions is None else living_area_exclusions
    )
    exclusion_polygons = []
    excluded_areas_by_id = {}
    for exclusion in living_area_exclusions:
        exclusion_id = _non_empty_living_area_exclusion_id(exclusion)
        polygon = _floor_region_polygon(
            width,
            depth,
            outer_wall_thickness,
            exclusion,
            outer_footprint,
            label="living area exclusion",
            inside_error="Living-area exclusion must stay inside inner floor area",
        )
        exclusion_polygons.append(polygon)
        excluded_areas_by_id[exclusion_id] = excluded_areas_by_id.get(
            exclusion_id, 0.0
        ) + polygon_area(polygon)

    floor_cutout_area = _union_area(floor_cutout_polygons)
    excluded_area = sum(excluded_areas_by_id.values())
    non_living_floor_area = _union_area([*floor_cutout_polygons, *exclusion_polygons])
    floor_area = inner_floor_area - inner_wall_area - non_living_floor_area
    if floor_area < -1e-9:
        raise ValueError("non-living footprints exceed inner floor area")
    floor_area = max(0.0, floor_area)
    return InnerFloorAreaBreakdown(
        square_meters=floor_area,
        inner_floor_area=inner_floor_area,
        inner_wall_area=inner_wall_area,
        floor_cutout_area=floor_cutout_area,
        excluded_area=excluded_area,
        non_living_floor_area=non_living_floor_area,
        excluded_areas_by_id=excluded_areas_by_id,
    )


def calculate_inner_floor_area(
    width,
    depth,
    outer_wall_thickness,
    *,
    inner_wall_thickness=0.0,
    wall_specs=None,
    floor_cutouts=None,
    living_area_exclusions=None,
    outer_footprint=None,
):
    return calculate_inner_floor_area_breakdown(
        width,
        depth,
        outer_wall_thickness,
        inner_wall_thickness=inner_wall_thickness,
        wall_specs=wall_specs,
        floor_cutouts=floor_cutouts,
        living_area_exclusions=living_area_exclusions,
        outer_footprint=outer_footprint,
    ).square_meters


def _segment_angle_degrees(start, end):
    return math.degrees(
        math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    )


def _create_segment_aligned_box(
    start,
    end,
    width,
    height,
    z_origin,
    *,
    offset_along=0.0,
    length=None,
    overhang=0.0,
):
    segment_length = math.hypot(
        float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    )
    if segment_length <= 0:
        raise ValueError("Segment length must be positive")
    box_length = segment_length if length is None else float(length)
    local = create_box(
        box_length + 2.0 * overhang,
        width,
        height,
        origin=(float(offset_along) - overhang, -width * 0.5, z_origin),
    )
    return translate(float(start[0]), float(start[1]), 0.0)(
        rotate(_segment_angle_degrees(start, end), axis=(0, 0, 1))(local)
    )


def _create_segment_wall_opening_cutter(wall_spec, thickness, floor_thickness, opening):
    wall_length = _inner_wall_spec_length(wall_spec)
    offset = float(opening["offset"])
    width = float(opening["width"])
    height = float(opening["height"])
    offset_from_floor = float(opening.get("offset_from_floor", 0.0))
    if offset < 0 or offset + width > wall_length:
        raise ValueError("Inner wall opening must stay inside wall segment")
    if width <= 0 or height <= 0:
        raise ValueError("Inner wall opening width and height must be positive")
    return _create_segment_aligned_box(
        wall_spec["start"],
        wall_spec["end"],
        thickness + 0.02,
        height,
        floor_thickness + offset_from_floor,
        offset_along=offset,
        length=width,
        overhang=0.005,
    )


def _create_outer_wall_opening_cutter(
    outer_footprint,
    outer_wall_thickness,
    floor_thickness,
    opening,
):
    points = _clean_polygon_points(outer_footprint)
    edge_index = int(opening["edge_index"])
    if edge_index < 0 or edge_index >= len(points):
        raise ValueError(f"outer wall opening edge_index out of range: {edge_index}")
    start = points[edge_index]
    end = points[(edge_index + 1) % len(points)]
    edge_length = math.hypot(end[0] - start[0], end[1] - start[1])
    offset = float(opening["offset"])
    width = float(opening["width"])
    height = float(opening["height"])
    offset_from_floor = float(opening.get("offset_from_floor", 0.0))
    if offset < 0 or offset + width > edge_length:
        raise ValueError("Outer wall opening must stay inside footprint edge")
    if width <= 0 or height <= 0:
        raise ValueError("Outer wall opening width and height must be positive")
    return _create_segment_aligned_box(
        start,
        end,
        2.0 * outer_wall_thickness + 0.04,
        height,
        floor_thickness + offset_from_floor,
        offset_along=offset,
        length=width,
        overhang=0.005,
    )


def _create_window_cutout(
    width, depth, height, outer_wall_thickness, floor_thickness, window
):
    window_width = window["width"]
    window_height = window["height"]
    offset_from_floor = window["offset_from_floor"]
    offset_from_outside_left = window["offset_from_outside_left"]
    window_location = window["location"]

    if window_location not in [
        Alignment.LEFT,
        Alignment.RIGHT,
        Alignment.FRONT,
        Alignment.BACK,
    ]:
        raise ValueError(f"Unsupported window location: {window_location}")
    if window_width <= 0:
        raise ValueError(f"Window width must be positive, got {window_width}")
    if window_height <= 0:
        raise ValueError(f"Window height must be positive, got {window_height}")

    cut_depth = outer_wall_thickness + 0.01
    cut_overhang = 0.005
    z_origin = floor_thickness + offset_from_floor

    if window_location in [Alignment.FRONT, Alignment.BACK]:
        if (
            offset_from_outside_left < 0
            or offset_from_outside_left + window_width > width
        ):
            raise ValueError(
                f"Window offset {offset_from_outside_left} keeps a {window_width} wide window outside storey width {width}"
            )

        x_origin = (
            width - offset_from_outside_left - window_width
            if window_location == Alignment.BACK
            else offset_from_outside_left
        )
        y_origin = (
            depth - outer_wall_thickness - cut_overhang
            if window_location == Alignment.BACK
            else -cut_overhang
        )

        return create_box(
            window_width,
            cut_depth,
            window_height,
            origin=(x_origin, y_origin, z_origin),
        )

    if offset_from_outside_left < 0 or offset_from_outside_left + window_width > depth:
        raise ValueError(
            f"Window offset {offset_from_outside_left} keeps a {window_width} wide window outside storey depth {depth}"
        )

    y_origin = (
        depth - offset_from_outside_left - window_width
        if window_location == Alignment.LEFT
        else offset_from_outside_left
    )
    x_origin = (
        width - outer_wall_thickness - cut_overhang
        if window_location == Alignment.RIGHT
        else -cut_overhang
    )

    return create_box(
        cut_depth,
        window_width,
        window_height,
        origin=(x_origin, y_origin, z_origin),
    )


def _create_inner_wall(
    width, depth, height, outer_wall_thickness, floor_thickness, thickness, wall_spec
):
    if _is_segment_wall_spec(wall_spec):
        segment_thickness = _inner_wall_spec_thickness(wall_spec, thickness)
        wall_height = height - floor_thickness
        if segment_thickness <= 0:
            raise ValueError(
                f"Inner wall thickness must be positive, got {segment_thickness}"
            )
        if wall_height <= 0:
            raise ValueError("floor_thickness leaves no positive wall height")
        inner_wall = _create_segment_aligned_box(
            wall_spec["start"],
            wall_spec["end"],
            segment_thickness,
            wall_height,
            floor_thickness,
        )
        for opening in wall_spec.get("openings", []):
            inner_wall = inner_wall.cut(
                _create_segment_wall_opening_cutter(
                    wall_spec, segment_thickness, floor_thickness, opening
                )
            )
        return inner_wall

    bounds = _inner_space_bounds(
        width, depth, height, outer_wall_thickness, floor_thickness
    )
    inner_width = bounds["x_max"] - bounds["x_min"]
    inner_depth = bounds["y_max"] - bounds["y_min"]
    wall_height = bounds["z_max"] - bounds["z_min"]

    outer_wall_side = wall_spec["outer_wall_side"]
    length = wall_spec["length"]
    offset = wall_spec["offset_from_outside_left"]

    if outer_wall_side not in [
        Alignment.LEFT,
        Alignment.RIGHT,
        Alignment.FRONT,
        Alignment.BACK,
    ]:
        raise ValueError(f"Unsupported outer_wall_side: {outer_wall_side}")
    if length <= 0:
        raise ValueError(f"Inner wall length must be positive, got {length}")

    half_thickness = thickness / 2

    if outer_wall_side in [Alignment.LEFT, Alignment.RIGHT]:
        if length > inner_width:
            raise ValueError(
                f"Inner wall length {length} exceeds inner width {inner_width}"
            )
        if offset < half_thickness or offset > inner_depth - half_thickness:
            raise ValueError(
                f"Inner wall offset {offset} keeps a {thickness} thick wall outside inner depth {inner_depth}"
            )

        y_center = (
            bounds["y_max"] - offset
            if outer_wall_side == Alignment.LEFT
            else bounds["y_min"] + offset
        )
        x_origin = (
            bounds["x_min"]
            if outer_wall_side == Alignment.LEFT
            else bounds["x_max"] - length
        )

        return create_box(
            length,
            thickness,
            wall_height,
            origin=(x_origin, y_center - half_thickness, bounds["z_min"]),
        )

    if length > inner_depth:
        raise ValueError(
            f"Inner wall length {length} exceeds inner depth {inner_depth}"
        )
    if offset < half_thickness or offset > inner_width - half_thickness:
        raise ValueError(
            f"Inner wall offset {offset} keeps a {thickness} thick wall outside inner width {inner_width}"
        )

    x_center = (
        bounds["x_max"] - offset
        if outer_wall_side == Alignment.BACK
        else bounds["x_min"] + offset
    )
    y_origin = (
        bounds["y_min"]
        if outer_wall_side == Alignment.FRONT
        else bounds["y_max"] - length
    )

    return create_box(
        thickness,
        length,
        wall_height,
        origin=(x_center - half_thickness, y_origin, bounds["z_min"]),
    )


def create_storey(
    width,
    depth,
    height,
    outer_wall_thickness,
    floor_thickness,
    windows,
    *,
    inner_wall_thickness=0.0,
    wall_specs=None,
    floor_cutouts=None,
    living_area_exclusions=None,
    outer_footprint=None,
    outer_wall_openings=None,
):
    if outer_footprint is None:
        hull = create_storey_hull(
            width, depth, height, outer_wall_thickness, floor_thickness
        )
    else:
        hull = _create_storey_hull_from_footprint(
            outer_footprint, height, outer_wall_thickness, floor_thickness
        )

    for window in windows:
        window_cutout = _create_window_cutout(
            width, depth, height, outer_wall_thickness, floor_thickness, window
        )
        hull = hull.cut(window_cutout)

    outer_wall_openings = [] if outer_wall_openings is None else outer_wall_openings
    for opening in outer_wall_openings:
        if outer_footprint is None:
            raise ValueError("outer_wall_openings require outer_footprint")
        hull = hull.cut(
            _create_outer_wall_opening_cutter(
                outer_footprint,
                outer_wall_thickness,
                floor_thickness,
                opening,
            )
        )

    floor_cutouts = [] if floor_cutouts is None else floor_cutouts
    floor_cutout_markers = []
    for floor_cutout in floor_cutouts:
        cutter = _create_floor_cutout(
            width,
            depth,
            outer_wall_thickness,
            floor_thickness,
            floor_cutout,
            outer_footprint,
        )
        hull = hull.cut(cutter)
        floor_cutout_markers.append(cutter)

    living_area_exclusions = (
        [] if living_area_exclusions is None else living_area_exclusions
    )
    living_area_exclusion_markers = []
    for exclusion in living_area_exclusions:
        living_area_exclusion_markers.append(
            (
                _non_empty_living_area_exclusion_id(exclusion),
                _create_living_area_exclusion_marker(
                    width,
                    depth,
                    outer_wall_thickness,
                    floor_thickness,
                    exclusion,
                    outer_footprint,
                ),
            )
        )

    wall_specs = [] if wall_specs is None else wall_specs
    if wall_specs and inner_wall_thickness <= 0:
        raise ValueError(
            f"inner_wall_thickness must be positive when wall_specs are provided, got {inner_wall_thickness}"
        )

    result = LeaderFollowersCuttersPart(hull)

    if outer_footprint is not None:
        result.add_named_non_production_part(
            _create_extruded_polygon_at(outer_footprint, 0.01),
            "semantic_footprint_marker",
        )

    for index, marker in enumerate(floor_cutout_markers):
        result.add_named_non_production_part(
            marker,
            f"floor_cutout_{index + 1}",
        )

    for exclusion_id, marker in living_area_exclusion_markers:
        result.add_named_non_production_part(
            marker,
            f"{LIVING_AREA_EXCLUSION_PREFIX}{exclusion_id}",
        )

    if wall_specs:
        inner_walls = PartCollector()
        for wall_spec in wall_specs:
            inner_wall = _create_inner_wall(
                width,
                depth,
                height,
                outer_wall_thickness,
                floor_thickness,
                inner_wall_thickness,
                wall_spec,
            )
            inner_walls = inner_walls.fuse(inner_wall)
        result.add_named_follower(inner_walls, "inner_walls")

    return result
