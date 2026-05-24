"""SVG renderers for semantic architecture storey plans."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from shellforgepy.architecture.semantic_storey import (
    frame_depth,
    frame_width,
    resolve_storey_geometry,
    validate_semantic_storey,
)

DEFAULT_EXCLUDED_LIVING_AREA_COLOR = "#dc2d2d"
TITLE_HEIGHT_PX = 74
LEGEND_HEIGHT_PX = 74
SEMANTIC_MARGIN_PX = 42


def _hex_color(value: Any, default: str = DEFAULT_EXCLUDED_LIVING_AREA_COLOR) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            return text.lower()
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 3:
            return _hex_from_components(float(part) for part in parts)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return _hex_from_components(float(component) for component in value[:3])
    return default


def _hex_from_components(components: Sequence[float]) -> str:
    values = list(components)
    if all(0.0 <= component <= 1.0 for component in values):
        values = [round(component * 255.0) for component in values]
    return "#" + "".join(
        f"{max(0, min(255, int(round(value)))):02x}" for value in values
    )


def _storey_payload_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    additional_data = metadata.get("additional_data") or {}
    architecture = additional_data.get("architecture") or {}
    storey = architecture.get("storey")
    if isinstance(storey, Mapping):
        payload = dict(storey)
        if "geometry" not in payload and "semantic" in payload:
            payload["geometry"] = resolve_storey_geometry(dict(payload["semantic"]))
        return payload

    generator_architecture = (metadata.get("generator_kwargs") or {}).get(
        "architecture"
    ) or metadata.get("architecture")
    spec = None
    if isinstance(generator_architecture, Mapping):
        spec = generator_architecture.get("storey_specification")
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"Assembly {metadata.get('assembly_name')} has no semantic storey metadata"
        )
    semantic_path = Path(str(spec["path"])).expanduser().resolve()
    semantic = yaml.safe_load(semantic_path.read_text(encoding="utf-8"))
    validate_semantic_storey(semantic)
    return {
        "id": spec["id"],
        "storey_index": int(spec["storey_index"]),
        "semantic_path": str(semantic_path),
        "semantic_sha256": spec.get("sha256"),
        "semantic": semantic,
        "geometry": resolve_storey_geometry(semantic),
        "dimensions": {
            "width": frame_width(semantic),
            "depth": frame_depth(semantic),
        },
    }


def _semantic_transform(
    semantic_plan: Mapping[str, Any],
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> dict[str, float]:
    x0, y0 = origin
    width, height = size
    model_width = frame_width(dict(semantic_plan))
    model_depth = frame_depth(dict(semantic_plan))
    scale = min(
        (width - 2.0 * SEMANTIC_MARGIN_PX) / max(model_width, 1e-9),
        (height - 2.0 * SEMANTIC_MARGIN_PX) / max(model_depth, 1e-9),
    )
    drawing_width = model_width * scale
    drawing_height = model_depth * scale
    return {
        "x0": x0 + (width - drawing_width) * 0.5,
        "y0": y0 + (height - drawing_height) * 0.5,
        "model_depth": model_depth,
        "scale": scale,
    }


def _region_polygon(region: Mapping[str, Any]) -> list[list[float]]:
    if "polygon" in region:
        return [[float(point[0]), float(point[1])] for point in region["polygon"]]
    if "origin" in region:
        x_origin = float(region["origin"][0])
        y_origin = float(region["origin"][1])
    else:
        x_origin = float(region["x"])
        y_origin = float(region["y"])
    length = float(region["length"])
    depth = float(region["depth"])
    return [
        [x_origin, y_origin],
        [x_origin + length, y_origin],
        [x_origin + length, y_origin + depth],
        [x_origin, y_origin + depth],
    ]


def _point_float(
    point: Sequence[float], transform: Mapping[str, float]
) -> tuple[float, float]:
    x = transform["x0"] + float(point[0]) * transform["scale"]
    y = (
        transform["y0"]
        + (transform["model_depth"] - float(point[1])) * transform["scale"]
    )
    return x, y


def _svg_points(
    points: Sequence[Sequence[float]], transform: Mapping[str, float]
) -> str:
    return " ".join(
        f"{point[0]:.2f},{point[1]:.2f}"
        for point in (_point_float(point, transform) for point in points)
    )


def _opening_marker_float(
    wall_spec: Mapping[str, Any],
    opening: Mapping[str, Any],
    transform: Mapping[str, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    start = wall_spec["start"]
    end = wall_spec["end"]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 0:
        raise ValueError("Cannot mark opening on zero-length wall segment")
    ux = dx / length
    uy = dy / length
    offset = float(opening["offset"])
    width = float(opening["width"])
    return (
        _point_float([start[0] + ux * offset, start[1] + uy * offset], transform),
        _point_float(
            [start[0] + ux * (offset + width), start[1] + uy * (offset + width)],
            transform,
        ),
    )


def _side_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _outer_cut_segment_points(
    semantic_plan: Mapping[str, Any], outer_cut: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    width = frame_width(dict(semantic_plan))
    depth = frame_depth(dict(semantic_plan))
    offset = float(outer_cut["offset_from_outside_left"])
    cut_width = float(outer_cut["width"])
    side = _side_name(outer_cut["location"])
    if side == "FRONT":
        return [offset, 0.0], [offset + cut_width, 0.0]
    if side == "BACK":
        return [width - offset - cut_width, depth], [width - offset, depth]
    if side == "LEFT":
        y_min = depth - offset - cut_width
        return [0.0, y_min], [0.0, y_min + cut_width]
    if side == "RIGHT":
        return [width, offset], [width, offset + cut_width]
    raise ValueError(f"Unsupported outer cut side: {side}")


def _outer_cut_marker_float(
    semantic_plan: Mapping[str, Any],
    outer_cut: Mapping[str, Any],
    transform: Mapping[str, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(
        _point_float(point, transform)
        for point in _outer_cut_segment_points(semantic_plan, outer_cut)
    )


def _door_arc_points(
    door_symbol: Mapping[str, Any], steps: int = 16
) -> list[list[float]]:
    hinge = [float(door_symbol["hinge"][0]), float(door_symbol["hinge"][1])]
    closed_end = (
        door_symbol["end"]
        if door_symbol["hinge"] == door_symbol["start"]
        else door_symbol["start"]
    )
    closed_end = [float(closed_end[0]), float(closed_end[1])]
    leaf_end = [float(door_symbol["leaf_end"][0]), float(door_symbol["leaf_end"][1])]
    radius = math.hypot(leaf_end[0] - hinge[0], leaf_end[1] - hinge[1])
    if radius <= 0:
        return []
    start_angle = math.atan2(closed_end[1] - hinge[1], closed_end[0] - hinge[0])
    end_angle = math.atan2(leaf_end[1] - hinge[1], leaf_end[0] - hinge[0])
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    return [
        [
            hinge[0] + math.cos(start_angle + delta * index / (steps - 1)) * radius,
            hinge[1] + math.sin(start_angle + delta * index / (steps - 1)) * radius,
        ]
        for index in range(steps)
    ]


def _door_symbol_svg(
    door_symbol: Mapping[str, Any],
    transform: Mapping[str, float],
    *,
    index: int | None = None,
) -> str:
    hinge = _point_float(door_symbol["hinge"], transform)
    leaf_end = _point_float(door_symbol["leaf_end"], transform)
    arc_points = [
        _point_float(point, transform) for point in _door_arc_points(door_symbol)
    ]
    arc = ""
    if arc_points:
        path_data = " ".join(
            f"{'M' if point_index == 0 else 'L'} {point[0]:.2f} {point[1]:.2f}"
            for point_index, point in enumerate(arc_points)
        )
        arc = (
            f'<path class="door-symbol door-arc-halo" d="{path_data}" fill="none" '
            'stroke="white" stroke-width="7" stroke-linecap="round" />'
            f'<path class="door-symbol door-arc" d="{path_data}" fill="none" '
            'stroke="#2daa4b" stroke-width="3" stroke-linecap="round" />'
        )
    label = ""
    if index is not None:
        source_id = (door_symbol.get("source_detection") or {}).get(
            "id", door_symbol["id"]
        )
        label = (
            f'<text x="{hinge[0] + 5:.2f}" y="{hinge[1] - 7:.2f}" '
            'font-family="Arial, sans-serif" font-size="14" font-weight="700" '
            'fill="#25dc4b" stroke="#000000" stroke-width="3" '
            f'paint-order="stroke">{_xml_escape(f"{index}:{source_id}")}</text>'
        )
    return (
        f'<g class="door-symbol" data-door-id="{_xml_escape(str(door_symbol["id"]))}">'
        f'<line x1="{hinge[0]:.2f}" y1="{hinge[1]:.2f}" '
        f'x2="{leaf_end[0]:.2f}" y2="{leaf_end[1]:.2f}" '
        'stroke="white" stroke-width="7" stroke-linecap="round" />'
        f'<line x1="{hinge[0]:.2f}" y1="{hinge[1]:.2f}" '
        f'x2="{leaf_end[0]:.2f}" y2="{leaf_end[1]:.2f}" '
        'stroke="#2daa4b" stroke-width="3" stroke-linecap="round" />'
        f"{arc}"
        f'<circle cx="{hinge[0]:.2f}" cy="{hinge[1]:.2f}" r="4" fill="#2daa4b" />'
        f"{label}"
        "</g>"
    )


def _semantic_svg_elements(
    semantic_plan: Mapping[str, Any],
    geometry: Mapping[str, Any],
    transform: Mapping[str, float],
    *,
    excluded_living_area_color: str,
) -> str:
    defaults = semantic_plan["defaults"]
    outer_wall_width = max(
        4.0,
        float(defaults["outer_wall_thickness"]) * transform["scale"],
    )
    inner_wall_thickness = float(defaults["inner_wall_thickness"])
    parts = [
        f'<polygon class="outer-footprint" points="{_svg_points(geometry["outer_footprint"], transform)}" '
        f'fill="#f8fafc" stroke="#141414" stroke-width="{outer_wall_width:.2f}" '
        'stroke-linejoin="miter" />'
    ]

    for cutout in geometry.get("floor_cutouts", []):
        polygon = _region_polygon(cutout)
        parts.append(
            f'<polygon class="floor-cutout" data-cutout-id="{_xml_escape(str(cutout.get("id", "")))}" '
            f'points="{_svg_points(polygon, transform)}" fill="#d8ecff" '
            'fill-opacity="0.85" stroke="#195fb4" stroke-width="4" />'
        )

    for exclusion in geometry.get("living_area_exclusions", []):
        polygon = _region_polygon(exclusion)
        exclusion_id = _xml_escape(str(exclusion.get("id", "")))
        points = _svg_points(polygon, transform)
        parts.append(
            f'<polygon class="living-area-exclusion" data-exclusion-id="{exclusion_id}" '
            f'points="{points}" fill="{excluded_living_area_color}" fill-opacity="0.18" '
            f'stroke="{excluded_living_area_color}" stroke-width="3" />'
        )
        parts.append(
            f'<polygon class="living-area-exclusion-hatch" data-exclusion-id="{exclusion_id}" '
            f'points="{points}" fill="url(#excludedLivingAreaHatch)" '
            f'stroke="{excluded_living_area_color}" stroke-width="2" />'
        )

    for outer_cut in geometry.get("outer_cuts", []):
        opening_start, opening_end = _outer_cut_marker_float(
            semantic_plan, outer_cut, transform
        )
        color = "#2daa4b" if outer_cut.get("kind") == "door" else "#dc7828"
        parts.append(
            f'<line class="outer-opening-erase" x1="{opening_start[0]:.2f}" '
            f'y1="{opening_start[1]:.2f}" x2="{opening_end[0]:.2f}" '
            f'y2="{opening_end[1]:.2f}" stroke="white" '
            f'stroke-width="{outer_wall_width + 8:.2f}" stroke-linecap="butt" />'
        )
        parts.append(
            f'<line class="outer-opening-marker" x1="{opening_start[0]:.2f}" '
            f'y1="{opening_start[1]:.2f}" x2="{opening_end[0]:.2f}" '
            f'y2="{opening_end[1]:.2f}" stroke="{color}" '
            f'stroke-width="{max(3.0, outer_wall_width / 3.0):.2f}" '
            'stroke-linecap="butt" />'
        )

    for wall_spec in geometry.get("wall_specs", []):
        start = _point_float(wall_spec["start"], transform)
        end = _point_float(wall_spec["end"], transform)
        wall_width = max(
            3.0,
            float(wall_spec.get("thickness", inner_wall_thickness))
            * transform["scale"],
        )
        parts.append(
            f'<line class="inner-wall" data-wall-id="{_xml_escape(str(wall_spec.get("id", "")))}" '
            f'x1="{start[0]:.2f}" y1="{start[1]:.2f}" '
            f'x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#0f0f0f" '
            f'stroke-width="{wall_width:.2f}" stroke-linecap="butt" />'
        )
        for opening in wall_spec.get("openings", []):
            opening_start, opening_end = _opening_marker_float(
                wall_spec, opening, transform
            )
            parts.append(
                f'<line class="inner-opening-erase" x1="{opening_start[0]:.2f}" '
                f'y1="{opening_start[1]:.2f}" x2="{opening_end[0]:.2f}" '
                f'y2="{opening_end[1]:.2f}" stroke="white" '
                f'stroke-width="{wall_width + 6:.2f}" stroke-linecap="butt" />'
            )

    for index, door_symbol in enumerate(geometry.get("door_symbols", []), start=1):
        parts.append(_door_symbol_svg(door_symbol, transform, index=index))

    return "\n    ".join(parts)


def _legend_svg(y: int, excluded_living_area_color: str) -> str:
    entries = [
        ("#0f0f0f", "walls"),
        ("#195fb4", "floor/stair cutout"),
        ("#2daa4b", "door symbol"),
        (excluded_living_area_color, "excluded living area"),
    ]
    x = 22
    parts = []
    for color, label in entries:
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 54}" y2="{y}" '
            f'stroke="{color}" stroke-width="9" stroke-linecap="round" />'
        )
        parts.append(
            f'<text x="{x + 68}" y="{y + 9}" font-family="Arial, sans-serif" '
            'font-size="19" font-weight="700" fill="#232323">'
            f"{_xml_escape(label)}</text>"
        )
        x += 330
    return "\n  ".join(parts)


def _render_storey_payload_svg(
    storey_payload: Mapping[str, Any],
    output_path: Path,
    *,
    excluded_living_area_color=None,
    image_width: int = 1800,
) -> None:
    semantic = storey_payload["semantic"]
    geometry = storey_payload["geometry"]
    width_m = float(
        (storey_payload.get("dimensions") or {}).get("width") or frame_width(semantic)
    )
    depth_m = float(
        (storey_payload.get("dimensions") or {}).get("depth") or frame_depth(semantic)
    )
    panel_width = max(600, int(image_width))
    panel_height = max(
        500, int(round(panel_width * max(depth_m, 1e-9) / max(width_m, 1e-9)))
    )
    canvas_width = panel_width
    canvas_height = TITLE_HEIGHT_PX + panel_height + LEGEND_HEIGHT_PX
    transform = _semantic_transform(
        semantic,
        origin=(0, TITLE_HEIGHT_PX),
        size=(panel_width, panel_height),
    )
    exclusion_color = _hex_color(excluded_living_area_color)
    title = f"{storey_payload.get('id', 'storey')} semantic plan"
    semantic_svg = _semantic_svg_elements(
        semantic,
        geometry,
        transform,
        excluded_living_area_color=exclusion_color,
    )
    legend_y = TITLE_HEIGHT_PX + panel_height + 35
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">
  <defs>
    <pattern id="excludedLivingAreaHatch" patternUnits="userSpaceOnUse" width="12" height="12" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="12" stroke="{exclusion_color}" stroke-width="3" stroke-opacity="0.75" />
    </pattern>
  </defs>
  <rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" fill="white" />
  <text x="18" y="45" font-family="Arial, sans-serif" font-size="31" font-weight="700" fill="#111111">{_xml_escape(title)}</text>
  <line x1="0" y1="{TITLE_HEIGHT_PX - 1}" x2="{panel_width}" y2="{TITLE_HEIGHT_PX - 1}" stroke="#d2d2d2" stroke-width="2" />
  <rect x="0" y="{TITLE_HEIGHT_PX}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#282828" stroke-width="2" />
  <g>
    {semantic_svg}
  </g>
  {_legend_svg(legend_y, exclusion_color)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def render_storey_plan_from_metadata(
    metadata: Mapping[str, Any],
    output_path,
    *,
    excluded_living_area_color=None,
    image_width: int = 1800,
) -> dict[str, Any]:
    """Render one semantic storey plan SVG from builder assembly metadata."""

    output_path = Path(output_path)
    storey_payload = _storey_payload_from_metadata(metadata)
    _render_storey_payload_svg(
        storey_payload,
        output_path,
        excluded_living_area_color=excluded_living_area_color,
        image_width=image_width,
    )
    return {
        "format": "svg",
        "storey_id": storey_payload.get("id"),
        "storey_index": storey_payload.get("storey_index"),
        "semantic_path": storey_payload.get("semantic_path"),
        "path": str(output_path),
    }


def render_semantic_storey_plan(
    semantic_storey: Mapping[str, Any],
    output_path,
    *,
    storey_id: str = "storey",
    excluded_living_area_color=None,
    image_width: int = 1800,
) -> dict[str, Any]:
    """Render one semantic storey plan SVG directly from semantic YAML data."""

    validate_semantic_storey(dict(semantic_storey))
    payload = {
        "id": storey_id,
        "semantic": dict(semantic_storey),
        "geometry": resolve_storey_geometry(dict(semantic_storey)),
        "dimensions": {
            "width": frame_width(dict(semantic_storey)),
            "depth": frame_depth(dict(semantic_storey)),
        },
    }
    output_path = Path(output_path)
    _render_storey_payload_svg(
        payload,
        output_path,
        excluded_living_area_color=excluded_living_area_color,
        image_width=image_width,
    )
    return {"format": "svg", "storey_id": storey_id, "path": str(output_path)}


render_storey_plan_svg_from_metadata = render_storey_plan_from_metadata
render_semantic_storey_plan_svg = render_semantic_storey_plan


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
