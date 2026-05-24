"""Small pyclipper-backed helpers for architecture 2D polygons."""

from __future__ import annotations

from typing import Iterable, Sequence

import pyclipper

CLIPPER_SCALE = 1_000_000


Point = tuple[float, float]
PolygonPoints = list[Point]


def clean_polygon_points(points: Sequence[Sequence[float]]) -> PolygonPoints:
    cleaned = [(float(point[0]), float(point[1])) for point in points]
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    if len(set(cleaned)) < 3:
        raise ValueError("Polygon must contain at least three unique points")
    if polygon_area(cleaned) <= 0:
        raise ValueError("Polygon must have positive area")
    return cleaned


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    scaled = _scale_path(points, orient_positive=False)
    return abs(float(pyclipper.Area(scaled))) / float(CLIPPER_SCALE * CLIPPER_SCALE)


def union_area(polygons: Iterable[Sequence[Sequence[float]]]) -> float:
    scaled_paths = [
        _scale_path(polygon, orient_positive=True)
        for polygon in polygons
        if len(polygon) >= 3
    ]
    if not scaled_paths:
        return 0.0

    clipper = pyclipper.Pyclipper()
    clipper.AddPaths(scaled_paths, pyclipper.PT_SUBJECT, True)
    solution = clipper.Execute(
        pyclipper.CT_UNION,
        pyclipper.PFT_NONZERO,
        pyclipper.PFT_NONZERO,
    )
    return sum(abs(float(pyclipper.Area(path))) for path in solution) / float(
        CLIPPER_SCALE * CLIPPER_SCALE
    )


def polygon_covers(
    container: Sequence[Sequence[float]],
    polygon: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-9,
) -> bool:
    container_path = _scale_path(container, orient_positive=True)
    polygon_path = _scale_path(polygon, orient_positive=True)
    if tolerance > 0:
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(container_path, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(max(1, int(round(tolerance * CLIPPER_SCALE))))
        if expanded:
            container_paths = expanded
        else:
            container_paths = [container_path]
    else:
        container_paths = [container_path]

    clipper = pyclipper.Pyclipper()
    clipper.AddPath(polygon_path, pyclipper.PT_SUBJECT, True)
    clipper.AddPaths(container_paths, pyclipper.PT_CLIP, True)
    remainder = clipper.Execute(
        pyclipper.CT_DIFFERENCE,
        pyclipper.PFT_NONZERO,
        pyclipper.PFT_NONZERO,
    )
    remainder_area = sum(abs(float(pyclipper.Area(path))) for path in remainder)
    return remainder_area <= max(1.0, tolerance * CLIPPER_SCALE * CLIPPER_SCALE)


def inset_polygon(
    points: Sequence[Sequence[float]],
    distance: float,
    *,
    label: str,
) -> PolygonPoints:
    source = clean_polygon_points(points)
    scaled = _scale_path(source, orient_positive=True)
    offset = pyclipper.PyclipperOffset(miter_limit=4.0)
    offset.AddPath(scaled, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = offset.Execute(-float(distance) * CLIPPER_SCALE)
    if not solution:
        raise ValueError(f"{label} produced an empty polygon")
    positive_solution = [path for path in solution if abs(pyclipper.Area(path)) > 0]
    if len(positive_solution) != 1:
        raise ValueError(
            f"{label} must produce one polygon, got {len(positive_solution)}"
        )
    return clean_polygon_points(_unscale_path(positive_solution[0]))


def _scale_path(
    points: Sequence[Sequence[float]], *, orient_positive: bool
) -> list[tuple[int, int]]:
    scaled = [
        (
            int(round(float(point[0]) * CLIPPER_SCALE)),
            int(round(float(point[1]) * CLIPPER_SCALE)),
        )
        for point in points
    ]
    if len(scaled) > 1 and scaled[0] == scaled[-1]:
        scaled = scaled[:-1]
    if len(set(scaled)) < 3:
        raise ValueError("Polygon must contain at least three unique points")
    if orient_positive and pyclipper.Area(scaled) < 0:
        scaled = list(reversed(scaled))
    return scaled


def _unscale_path(path: Sequence[Sequence[int]]) -> PolygonPoints:
    return [
        (float(point[0]) / CLIPPER_SCALE, float(point[1]) / CLIPPER_SCALE)
        for point in path
    ]
