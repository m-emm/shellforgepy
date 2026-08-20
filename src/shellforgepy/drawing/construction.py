"""Construction-drawing request, SVG, and sheet rendering helpers.

The request and frame contracts are backend independent. Exact CAD section
elements remain in model-derived drawing coordinates. A drawing frame maps
model points to drawing ``(x, y)`` coordinates where positive Y points upward;
the SVG geometry group applies the Y-axis inversion needed by SVG's native
coordinate system. A technical sheet wraps those same geometry groups in a
viewport transform and adds reusable border/title/metadata elements.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict

try:  # FreeCAD 0.21 embeds Python 3.10, where these live in typing_extensions.
    from typing import NotRequired, Required
except ImportError:  # pragma: no cover - exercised by the FreeCAD runner.
    from typing_extensions import NotRequired, Required

from shellforgepy.construct.alignment import Alignment
from shellforgepy.drawing.layout import (
    PLANAR_ALIGNMENTS,
    Bounds2D,
    align_bounds_sequence_2d,
)

Vector3 = tuple[float, float, float]
ModelBounds = tuple[Vector3, Vector3]
DrawingBounds = tuple[float, float, float, float]
ViewPreset = Literal["top", "bottom", "front", "back", "left", "right"]

SVG_NS = "http://www.w3.org/2000/svg"
_GEOMETRY_STYLE = {
    "fill": "none",
    "stroke": "#000000",
    "stroke-width": "0.2",
}
_HIDDEN_GEOMETRY_STYLE = {
    **_GEOMETRY_STYLE,
    "stroke-dasharray": "1.2 0.8",
}
DEFAULT_SECTION_VIEW: ViewPreset = "top"
SUPPORTED_VIEW_PRESETS: tuple[ViewPreset, ...] = (
    "top",
    "bottom",
    "front",
    "back",
    "left",
    "right",
)

# Request/frame/SVG structure is backend independent.  Both existing CAD
# adapters expose the Stage 1 exact section seam; CadQuery is the verified
# backend for the checked-in fixture/demo in this repository.
CONSTRUCTION_DRAWING_BACKEND_SUPPORT = {
    "cadquery": {
        "fixture_construction": True,
        "step_artifact": True,
        "section_extraction": True,
        "projection_extraction": True,
    },
    "freecad": {
        "fixture_construction": True,
        "step_artifact": True,
        "section_extraction": True,
        "projection_extraction": True,
    },
}


class PartSelector(TypedDict, total=False):
    """Existing builder visualization/production selector syntax."""

    source: Required[str]
    artifact: Required[str]
    assembly: NotRequired[str]
    name: NotRequired[str]
    names: NotRequired[list[str]]
    exclude_names: NotRequired[list[str]]
    name_template: NotRequired[str]


class ProjectionRepresentation(TypedDict, total=False):
    """Exact orthographic-projection policy for one construction view."""

    mode: Required[Literal["section", "projection"]]
    include: NotRequired[list[str]]


class ConstructionDrawingRequest(TypedDict, total=False):
    """Plain-data request for a construction section drawing.

    This is intentionally a TypedDict rather than a drawing-document or
    geometry-primitive object.  ``parts`` uses the existing builder selector
    vocabulary and ``view`` may be a named preset or an explicit plane mapping.
    ``parts`` selects visible scene geometry. ``annotations`` uses canonical
    named-part references as measurement targets and can therefore refer to a
    non-visible cutter or non-production part.
    """

    name: Required[str]
    parts: Required[list[PartSelector]]
    units: Required[str]
    scale: Required[float]
    precision: Required[int]
    view: NotRequired[ViewPreset | Mapping[str, object]]
    views: NotRequired[list["ConstructionDrawingView"]]
    representation: NotRequired[ProjectionRepresentation]
    section_thickness: NotRequired[float]
    visibility: NotRequired[str]
    tolerance: NotRequired[float]
    curve_approximation: NotRequired[str]
    metadata: NotRequired[dict[str, str]]
    sheet: NotRequired[Mapping[str, object]]
    annotations: NotRequired[list["DimensionAnnotation"]]


class ConstructionDrawingView(TypedDict, total=False):
    """One independently sectioned and annotated view on a shared drawing."""

    id: Required[str]
    view: Required[ViewPreset | Mapping[str, object]]
    representation: NotRequired[ProjectionRepresentation]
    placement: NotRequired[AnnotationPlacement]
    annotations: NotRequired[list["DimensionAnnotation"]]


class AnnotationAlignment(TypedDict, total=False):
    """One 2D ShellForgePy alignment operation for an annotation."""

    alignment: Required[str]
    stack_gap: NotRequired[float]


class AnnotationPlacement(TypedDict):
    """Visible-drawing-relative placement using the existing Alignment vocabulary."""

    alignments: list[AnnotationAlignment]


class LinearDimensionEndpoint(TypedDict):
    """One explicit projected-envelope endpoint for a linear dimension."""

    target: str
    edge: str


class DimensionAnnotation(TypedDict, total=False):
    """Explicit construction-drawing dimension declaration.

    The callout fields deliberately live on ``circle_diameter`` annotations
    rather than introducing a second target or selector language.  They
    describe how the exact, resolved circular target is labelled; they never
    change the measured geometry.
    """

    id: Required[str]
    operation: Required[str]
    target: NotRequired[str]
    from_: NotRequired[LinearDimensionEndpoint]
    to: NotRequired[LinearDimensionEndpoint]
    dimension_direction: NotRequired[str]
    placement: NotRequired[AnnotationPlacement]
    quantity: NotRequired[int]
    diameter_tolerance: NotRequired[str]
    thread_size: NotRequired[str]
    thread_tolerance_class: NotRequired[str]
    depth: NotRequired[float]
    through: NotRequired[bool]
    leader_tilt_degrees: NotRequired[float]
    leader_elbow_length: NotRequired[float]


class SectionViewFrame(TypedDict):
    """Resolved right-handed model-to-drawing frame."""

    origin: Vector3
    normal: Vector3
    up: Vector3
    right: Vector3


_VIEW_AXES: dict[ViewPreset, tuple[Vector3, Vector3]] = {
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}

_SELECTOR_KEYS = {
    "source",
    "artifact",
    "assembly",
    "name",
    "names",
    "exclude_names",
    "name_template",
}
_VECTOR_KEYS = ("normal", "up", "origin")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_ANNOTATION_OPERATIONS = {
    "bounding_box_x_dimension",
    "bounding_box_y_dimension",
    "circle_diameter",
    "linear_dimension",
}
_CIRCLE_DIAMETER_CALLOUT_KEYS = {
    "quantity",
    "diameter_tolerance",
    "thread_size",
    "thread_tolerance_class",
    "depth",
    "through",
    "leader_tilt_degrees",
    "leader_elbow_length",
}
_DEFAULT_CIRCLE_LEADER_TILT_DEGREES = 30.0
_DEFAULT_CIRCLE_LEADER_ELBOW_LENGTH = 6.0
_CIRCLE_ARROW_CLEARANCE = 0.0
_EXTENSION_LINE_PART_GAP = 0.8
_EXTENSION_LINE_DIMENSION_OVERRUN = 1.5
_EXTENSION_LINE_STROKE_WIDTH = 0.12
_ANNOTATION_2D_ALIGNMENTS = PLANAR_ALIGNMENTS
_LINEAR_DIMENSION_DIRECTIONS = frozenset({"RIGHT", "BACK"})
_LINEAR_DIMENSION_EDGES_BY_DIRECTION = {
    "RIGHT": frozenset({"EDGE_LEFT", "EDGE_RIGHT"}),
    "BACK": frozenset({"EDGE_FRONT", "EDGE_BACK"}),
}
_LINEAR_DIMENSION_STACK_PITCH = 4.0
_MULTI_VIEW_STACK_GAP = 12.0
_PROJECTION_INCLUDE_DEFAULT = ("visible_outline", "visible_feature_edges")
_PROJECTION_INCLUDE_VALUES = frozenset(
    {
        "visible_outline",
        "visible_feature_edges",
        "hidden_feature_edges",
        "tangent_edges",
    }
)


def make_construction_drawing_request(
    *,
    name: str,
    parts: Sequence[Mapping[str, object]],
    units: str = "mm",
    scale: float = 1.0,
    precision: int = 2,
    view: ViewPreset | Mapping[str, object] | None = None,
    representation: Mapping[str, object] | None = None,
    section_thickness: float = 0.0,
    visibility: str = "visible_edges",
    tolerance: float = 1e-6,
    curve_approximation: str = "reject",
    metadata: Mapping[str, str] | None = None,
    sheet: Mapping[str, object] | None = None,
    annotations: Sequence[Mapping[str, object]] | None = None,
    views: Sequence[Mapping[str, object]] | None = None,
) -> ConstructionDrawingRequest:
    """Create and validate a construction-drawing request."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Construction drawing name must be a non-empty string")
    if units != "mm":
        raise ValueError("Stage 0 construction drawings support units='mm' only")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Construction drawing scale must be a positive finite number")
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError("Construction drawing precision must be an integer")
    if precision < 0 or precision > 8:
        raise ValueError("Construction drawing precision must be between 0 and 8")
    if not parts:
        raise ValueError("Construction drawing requires at least one part selector")

    normalized_parts: list[PartSelector] = []
    for index, selector in enumerate(parts):
        if not isinstance(selector, Mapping):
            raise TypeError(f"Part selector {index} must be a mapping")
        unknown_keys = set(selector) - _SELECTOR_KEYS
        if unknown_keys:
            raise ValueError(
                f"Part selector {index} contains unsupported keys: "
                f"{sorted(unknown_keys)!r}"
            )
        if not selector.get("source") or not selector.get("artifact"):
            raise ValueError(
                f"Part selector {index} requires non-empty 'source' and 'artifact'"
            )
        normalized_parts.append(dict(selector))  # type: ignore[arg-type]

    if views is not None and (
        view is not None or annotations is not None or representation is not None
    ):
        raise ValueError(
            "Construction drawing views are an alternative to single view/annotations"
        )
    if views is None:
        resolved_view = DEFAULT_SECTION_VIEW if view is None else view
        _validate_view_spec(resolved_view)
    else:
        resolved_view = None
    _validate_nonnegative("section_thickness", section_thickness)
    _validate_nonnegative("tolerance", tolerance)
    if not isinstance(visibility, str) or not visibility.strip():
        raise ValueError("visibility must be a non-empty string")
    if not isinstance(curve_approximation, str) or not curve_approximation.strip():
        raise ValueError("curve_approximation must be a non-empty string")
    if metadata is not None and any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise TypeError("Construction drawing metadata must contain string pairs")

    request: ConstructionDrawingRequest = {
        "name": name,
        "parts": normalized_parts,
        "units": units,
        "scale": float(scale),
        "precision": precision,
        "section_thickness": float(section_thickness),
        "visibility": visibility,
        "tolerance": float(tolerance),
        "curve_approximation": curve_approximation,
    }
    if metadata:
        request["metadata"] = dict(metadata)
    if sheet is not None:
        request["sheet"] = _normalize_sheet_spec(sheet)
    if views is not None:
        request["views"] = _normalize_construction_drawing_views(views)
    else:
        request["view"] = (
            dict(resolved_view) if isinstance(resolved_view, Mapping) else resolved_view
        )
        request["representation"] = _normalize_representation(representation)
    if annotations is not None:
        request["annotations"] = _normalize_annotations(annotations)
    return request


def _normalize_construction_drawing_views(
    views: Sequence[Mapping[str, object]],
) -> list[ConstructionDrawingView]:
    if isinstance(views, (str, bytes)) or not views:
        raise TypeError("Construction drawing views must be a non-empty list")

    normalized: list[ConstructionDrawingView] = []
    ids: set[str] = set()
    for index, raw_view in enumerate(views):
        if not isinstance(raw_view, Mapping):
            raise TypeError(f"Construction drawing view {index} must be a mapping")
        unknown_keys = set(raw_view) - {
            "id",
            "view",
            "representation",
            "placement",
            "annotations",
        }
        if unknown_keys:
            raise ValueError(
                f"Construction drawing view {index} contains unsupported keys: "
                f"{sorted(unknown_keys)!r}"
            )
        view_id = str(raw_view.get("id") or "").strip()
        if not view_id:
            raise ValueError(f"Construction drawing view {index} requires an id")
        if view_id in ids:
            raise ValueError(
                f"Construction drawing view ids must be unique; duplicate {view_id!r}"
            )
        if "view" not in raw_view:
            raise ValueError(f"Construction drawing view {view_id!r} requires a view")
        view = raw_view["view"]
        _validate_view_spec(view)
        normalized_view: ConstructionDrawingView = {
            "id": view_id,
            "view": dict(view) if isinstance(view, Mapping) else view,
        }
        if "representation" in raw_view:
            normalized_view["representation"] = _normalize_representation(
                raw_view["representation"]  # type: ignore[arg-type]
            )
        if "placement" in raw_view:
            if index == 0:
                raise ValueError(
                    "The first construction drawing view cannot have placement"
                )
            normalized_view["placement"] = _normalize_annotation_placement(
                raw_view["placement"], annotation_id=f"view {view_id!r}"
            )
        annotations = raw_view.get("annotations")
        if annotations is not None:
            normalized_view["annotations"] = _normalize_annotations(annotations)  # type: ignore[arg-type]
        normalized.append(normalized_view)
        ids.add(view_id)
    return normalized


def _normalize_representation(
    representation: Mapping[str, object] | None,
) -> ProjectionRepresentation:
    """Normalize the explicit Stage 7 representation without changing sections."""

    if representation is None:
        return {"mode": "section"}
    if not isinstance(representation, Mapping):
        raise TypeError("Construction drawing representation must be a mapping")
    unknown = set(representation) - {"mode", "include"}
    if unknown:
        raise ValueError(
            "Construction drawing representation contains unsupported keys: "
            f"{sorted(unknown)!r}"
        )
    mode = str(representation.get("mode") or "").strip()
    if mode not in {"section", "projection"}:
        raise ValueError(
            "Construction drawing representation mode must be 'section' or 'projection'"
        )
    include = representation.get("include")
    if mode == "section":
        if include is not None:
            raise ValueError(
                "Section representation does not accept projection include categories"
            )
        return {"mode": "section"}
    if include is None:
        normalized_include = list(_PROJECTION_INCLUDE_DEFAULT)
    elif isinstance(include, Sequence) and not isinstance(include, (str, bytes)):
        normalized_include = [str(value).strip() for value in include]
    else:
        raise TypeError("Projection representation include must be a list")
    if not normalized_include or any(not value for value in normalized_include):
        raise ValueError(
            "Projection representation include must contain non-empty categories"
        )
    if len(set(normalized_include)) != len(normalized_include):
        raise ValueError("Projection representation include categories must be unique")
    unsupported = set(normalized_include) - _PROJECTION_INCLUDE_VALUES
    if unsupported:
        raise ValueError(
            f"Projection representation includes unsupported categories: {sorted(unsupported)!r}"
        )
    if "visible_outline" not in normalized_include:
        raise ValueError("Projection representation requires 'visible_outline'")
    return {"mode": "projection", "include": normalized_include}


def _normalize_annotations(
    annotations: Sequence[Mapping[str, object]],
) -> list[DimensionAnnotation]:
    if isinstance(annotations, (str, bytes)):
        raise TypeError("Construction drawing annotations must be a list")

    normalized: list[DimensionAnnotation] = []
    annotation_ids: set[str] = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise TypeError(
                f"Construction drawing annotation {index} must be a mapping"
            )
        unknown_keys = set(annotation) - {
            "id",
            "operation",
            "target",
            "from",
            "to",
            "dimension_direction",
            "placement",
            *_CIRCLE_DIAMETER_CALLOUT_KEYS,
        }
        if unknown_keys:
            raise ValueError(
                f"Construction drawing annotation {index} contains unsupported keys: "
                f"{sorted(unknown_keys)!r}"
            )
        annotation_id = str(annotation.get("id") or "").strip()
        if not annotation_id:
            raise ValueError(f"Construction drawing annotation {index} requires an id")
        if annotation_id in annotation_ids:
            raise ValueError(
                f"Construction drawing annotation ids must be unique; duplicate {annotation_id!r}"
            )
        operation = str(annotation.get("operation") or "").strip()
        if operation not in _ANNOTATION_OPERATIONS:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} has unsupported "
                f"operation {operation!r}; expected one of {sorted(_ANNOTATION_OPERATIONS)!r}"
            )
        normalized_annotation: DimensionAnnotation = {
            "id": annotation_id,
            "operation": operation,
        }
        if operation == "linear_dimension":
            if "target" in annotation:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} linear_dimension "
                    "uses explicit from/to endpoints instead of target"
                )
            normalized_annotation.update(
                _normalize_linear_dimension_declaration(
                    annotation, annotation_id=annotation_id
                )
            )
        else:
            target = str(annotation.get("target") or "").strip()
            if not target:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} requires a target"
                )
            if any(key in annotation for key in ("from", "to", "dimension_direction")):
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} endpoint fields only "
                    "apply to operation 'linear_dimension'"
                )
            normalized_annotation["target"] = target
        if "placement" in annotation:
            normalized_annotation["placement"] = _normalize_annotation_placement(
                annotation["placement"], annotation_id=annotation_id
            )
        _normalize_circle_diameter_callout(
            annotation,
            normalized_annotation,
            annotation_id=annotation_id,
            operation=operation,
        )
        if operation == "circle_diameter":
            _resolve_circle_callout_quadrant(
                normalized_annotation.get("placement"),
                annotation_id=annotation_id,
            )
        normalized.append(normalized_annotation)
        annotation_ids.add(annotation_id)
    return normalized


def _normalize_linear_dimension_declaration(
    annotation: Mapping[str, object], *, annotation_id: str
) -> dict[str, object]:
    direction = str(annotation.get("dimension_direction") or "").strip()
    if direction not in _LINEAR_DIMENSION_DIRECTIONS:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} linear_dimension "
            "dimension_direction must be one of "
            f"{sorted(_LINEAR_DIMENSION_DIRECTIONS)!r}"
        )
    return {
        "from": _normalize_linear_dimension_endpoint(
            annotation.get("from"),
            annotation_id=annotation_id,
            endpoint_name="from",
            dimension_direction=direction,
        ),
        "to": _normalize_linear_dimension_endpoint(
            annotation.get("to"),
            annotation_id=annotation_id,
            endpoint_name="to",
            dimension_direction=direction,
        ),
        "dimension_direction": direction,
    }


def _normalize_linear_dimension_endpoint(
    value: object,
    *,
    annotation_id: str,
    endpoint_name: str,
    dimension_direction: str,
) -> LinearDimensionEndpoint:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Construction drawing annotation {annotation_id!r} {endpoint_name} endpoint "
            "must be a mapping"
        )
    unknown_keys = set(value) - {"target", "edge"}
    if unknown_keys:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} {endpoint_name} endpoint "
            f"has unsupported keys: {sorted(unknown_keys)!r}"
        )
    target = str(value.get("target") or "").strip()
    if not target:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} {endpoint_name} endpoint "
            "requires a target"
        )
    edge = str(value.get("edge") or "").strip()
    allowed_edges = _LINEAR_DIMENSION_EDGES_BY_DIRECTION[dimension_direction]
    if edge not in allowed_edges:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} {endpoint_name} endpoint "
            f"edge must be one of {sorted(allowed_edges)!r} for {dimension_direction}"
        )
    return {"target": target, "edge": edge}


def _normalize_circle_diameter_callout(
    annotation: Mapping[str, object],
    normalized_annotation: DimensionAnnotation,
    *,
    annotation_id: str,
    operation: str,
) -> None:
    supplied_keys = set(annotation) & _CIRCLE_DIAMETER_CALLOUT_KEYS
    if supplied_keys and operation != "circle_diameter":
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} callout fields only "
            "apply to operation 'circle_diameter'"
        )
    if not supplied_keys:
        return

    if "quantity" in annotation:
        quantity = annotation["quantity"]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} quantity must be a positive integer"
            )
        normalized_annotation["quantity"] = quantity

    for key in ("diameter_tolerance", "thread_size", "thread_tolerance_class"):
        if key not in annotation:
            continue
        value = annotation[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} {key} must be a non-empty string"
            )
        normalized_annotation[key] = value.strip()  # type: ignore[literal-required]

    if "thread_tolerance_class" in annotation and "thread_size" not in annotation:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} thread_tolerance_class requires thread_size"
        )

    if "depth" in annotation:
        depth = annotation["depth"]
        if isinstance(depth, bool) or not isinstance(depth, (int, float)):
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} depth must be a positive finite number"
            )
        if not math.isfinite(float(depth)) or float(depth) <= 0:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} depth must be a positive finite number"
            )
        normalized_annotation["depth"] = float(depth)

    if "through" in annotation:
        through = annotation["through"]
        if not isinstance(through, bool):
            raise TypeError(
                f"Construction drawing annotation {annotation_id!r} through must be a boolean"
            )
        normalized_annotation["through"] = through
    if annotation.get("through") and "depth" in annotation:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} depth and through are mutually exclusive"
        )

    for key, default, allow_zero in (
        ("leader_tilt_degrees", _DEFAULT_CIRCLE_LEADER_TILT_DEGREES, True),
        ("leader_elbow_length", _DEFAULT_CIRCLE_LEADER_ELBOW_LENGTH, False),
    ):
        if key not in annotation:
            continue
        value = annotation[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} {key} must be finite"
            )
        value = float(value)
        if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} {key} must be a {qualifier} finite number"
            )
        if key == "leader_tilt_degrees" and value >= 90:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} leader_tilt_degrees must be below 90"
            )
        normalized_annotation[key] = value  # type: ignore[literal-required]


def _resolve_circle_callout_quadrant(
    placement: object,
    *,
    annotation_id: str,
) -> tuple[Alignment, float, Alignment, float]:
    """Resolve exactly one horizontal and vertical placement for a callout."""

    vertical: tuple[Alignment, float] | None = None
    horizontal: tuple[Alignment, float] | None = None
    operations = (
        _annotation_layout_operations(
            placement,
            default_alignment=Alignment.STACK_BACK,
        )
        if placement is not None
        else ()
    )
    for alignment, stack_gap in operations:
        if alignment in {Alignment.STACK_BACK, Alignment.STACK_FRONT}:
            if vertical is not None:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} circle_diameter "
                    "placement cannot specify more than one vertical alignment"
                )
            vertical = (alignment, stack_gap)
        elif alignment in {
            Alignment.BACK,
            Alignment.FRONT,
            Alignment.EDGE_BACK,
            Alignment.EDGE_FRONT,
        }:
            if vertical is not None:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} circle_diameter "
                    "placement cannot specify more than one vertical alignment"
                )
            vertical = (alignment, stack_gap)
        elif alignment in {Alignment.STACK_LEFT, Alignment.STACK_RIGHT}:
            if horizontal is not None:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} circle_diameter "
                    "placement cannot specify more than one horizontal alignment"
                )
            horizontal = (alignment, stack_gap)
        elif alignment in {
            Alignment.LEFT,
            Alignment.RIGHT,
            Alignment.EDGE_LEFT,
            Alignment.EDGE_RIGHT,
        }:
            if horizontal is not None:
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} circle_diameter "
                    "placement cannot specify more than one horizontal alignment"
                )
            horizontal = (alignment, stack_gap)
        else:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} circle_diameter "
                "placement only supports one LEFT/RIGHT and one FRONT/BACK "
                "planar alignment; CENTER is not supported"
            )
    vertical = vertical or (Alignment.STACK_BACK, 5.0)
    horizontal = horizontal or (Alignment.STACK_RIGHT, 5.0)
    if vertical[0] not in {Alignment.STACK_BACK, Alignment.STACK_FRONT} and horizontal[
        0
    ] not in {Alignment.STACK_LEFT, Alignment.STACK_RIGHT}:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} circle_diameter "
            "placement requires at least one STACK_* alignment"
        )
    return vertical[0], vertical[1], horizontal[0], horizontal[1]


def _normalize_annotation_placement(
    placement: object,
    *,
    annotation_id: str,
) -> AnnotationPlacement:
    if not isinstance(placement, Mapping):
        raise TypeError(
            f"Construction drawing annotation {annotation_id!r} placement must be a mapping"
        )
    unknown_keys = set(placement) - {"alignments"}
    if unknown_keys:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} placement has unsupported keys: "
            f"{sorted(unknown_keys)!r}"
        )
    raw_alignments = placement.get("alignments")
    if not isinstance(raw_alignments, Sequence) or isinstance(
        raw_alignments, (str, bytes)
    ):
        raise TypeError(
            f"Construction drawing annotation {annotation_id!r} placement requires alignments"
        )
    if not raw_alignments:
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} placement requires at least one alignment"
        )

    alignments: list[AnnotationAlignment] = []
    for alignment_index, raw_alignment in enumerate(raw_alignments):
        if not isinstance(raw_alignment, Mapping):
            raise TypeError(
                f"Construction drawing annotation {annotation_id!r} alignment "
                f"{alignment_index} must be a mapping"
            )
        unknown_alignment_keys = set(raw_alignment) - {"alignment", "stack_gap"}
        if unknown_alignment_keys:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} alignment "
                f"{alignment_index} has unsupported keys: "
                f"{sorted(unknown_alignment_keys)!r}"
            )
        alignment_name = str(raw_alignment.get("alignment") or "").strip()
        try:
            alignment = Alignment[alignment_name]
        except KeyError as exc:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} alignment "
                f"{alignment_index} has unknown Alignment.{alignment_name}"
            ) from exc
        if alignment not in _ANNOTATION_2D_ALIGNMENTS:
            raise ValueError(
                f"Construction drawing annotation {annotation_id!r} alignment "
                f"{alignment_index} only supports 2D alignments; got Alignment.{alignment.name}"
            )
        normalized_alignment: AnnotationAlignment = {"alignment": alignment.name}
        if "stack_gap" in raw_alignment:
            stack_gap = float(raw_alignment["stack_gap"])
            if not math.isfinite(stack_gap):
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} alignment "
                    f"{alignment_index} stack_gap must be finite"
                )
            normalized_alignment["stack_gap"] = stack_gap
        alignments.append(normalized_alignment)
    return {"alignments": alignments}


def resolve_view_frame(
    request: Mapping[str, object],
    model_bounds: ModelBounds,
) -> SectionViewFrame:
    """Resolve a named or explicit view against combined model-space bounds."""

    bounds = _normalize_model_bounds(model_bounds)
    view = request.get("view", DEFAULT_SECTION_VIEW)
    if view is None:
        view = DEFAULT_SECTION_VIEW

    if isinstance(view, str):
        if view not in _VIEW_AXES:
            raise ValueError(
                f"Unsupported section view {view!r}; expected one of "
                f"{SUPPORTED_VIEW_PRESETS!r}"
            )
        normal, up = _VIEW_AXES[view]
        explicit_origin = None
    elif isinstance(view, Mapping):
        normal = _vector_from_mapping(view, "normal")
        up = _vector_from_mapping(view, "up")
        explicit_origin = (
            _vector_from_mapping(view, "origin") if "origin" in view else None
        )
    else:
        raise TypeError("view must be a named preset or a mapping")

    normal = _normalize(normal, "normal")
    up = _normalize(up, "up")
    if abs(_dot(normal, up)) > 1e-9:
        raise ValueError("View normal and up vectors must be orthogonal")

    # right = up x normal gives top-view +X for normal +Z and up +Y.
    right = _normalize(_cross(up, normal), "derived right")
    origin = explicit_origin or _bounds_center(bounds)
    return {
        "origin": origin,
        "normal": normal,
        "up": up,
        "right": right,
    }


def model_point_to_drawing(
    point: Sequence[float], frame: SectionViewFrame
) -> tuple[float, float]:
    """Map a model-space point into the frame's positive-up drawing axes."""

    if len(point) != 3:
        raise ValueError("Model point must have exactly three coordinates")
    relative = tuple(float(point[index]) - frame["origin"][index] for index in range(3))
    return (_dot(relative, frame["right"]), _dot(relative, frame["up"]))


def drawing_bounds_from_model_bounds(
    model_bounds: ModelBounds,
    frame: SectionViewFrame,
) -> DrawingBounds:
    """Project all model-bounds corners and return ``min_x, min_y, w, h``."""

    bounds = _normalize_model_bounds(model_bounds)
    points = []
    for x in (bounds[0][0], bounds[1][0]):
        for y in (bounds[0][1], bounds[1][1]):
            for z in (bounds[0][2], bounds[1][2]):
                points.append(model_point_to_drawing((x, y, z), frame))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _merge_model_bounds(
    bounds: Sequence[ModelBounds],
) -> ModelBounds:
    if not bounds:
        raise ValueError("Cannot merge an empty model-bounds sequence")
    normalized = [_normalize_model_bounds(item) for item in bounds]
    return (
        tuple(min(item[0][axis] for item in normalized) for axis in range(3)),
        tuple(max(item[1][axis] for item in normalized) for axis in range(3)),
    )  # type: ignore[return-value]


def create_svg_document(
    request: Mapping[str, object],
    frame: SectionViewFrame,
    drawing_bounds: DrawingBounds,
    *,
    adapter_id: str | None = None,
    source_assembly: str | None = None,
) -> tuple[ET.ElementTree, ET.Element]:
    """Create the SVG tree and its geometry insertion group.

    The returned group is where exact section extractors append part groups. No
    CAD geometry is inferred or serialized here. When ``request['sheet']`` is
    present, the geometry group is placed in a technical drawing viewport and
    the sheet frame is emitted before it.
    """

    min_x, min_y, width, height = _normalize_drawing_bounds(drawing_bounds)
    if width <= 0 or height <= 0:
        raise ValueError("SVG drawing bounds must have positive width and height")
    ET.register_namespace("", SVG_NS)

    sheet = request.get("sheet")
    sheet_spec = _normalize_sheet_spec(sheet) if sheet is not None else None
    effective_scale = float(request.get("scale", 1.0))
    sheet_request = request
    if sheet_spec is None:
        view_box = (min_x, min_y, width, height)
    else:
        effective_scale = _select_discrete_sheet_scale(
            sheet_spec,
            drawing_bounds=(min_x, min_y, width, height),
            requested_scale=float(request.get("scale", 1.0)),
        )
        sheet_request = dict(request)
        sheet_request["_effective_scale"] = effective_scale
        view_box = (
            0.0,
            0.0,
            sheet_spec["width"],
            sheet_spec["height"],
        )
    attrs = {
        "viewBox": _format_view_box(view_box),
        "data-shellforgepy-units": str(request.get("units", "mm")),
        "data-shellforgepy-view": _view_metadata(request.get("view")),
        "data-shellforgepy-section-normal": _format_vector(frame["normal"]),
        "data-shellforgepy-section-up": _format_vector(frame["up"]),
        "data-shellforgepy-section-origin": _format_vector(frame["origin"]),
        "data-shellforgepy-scale": _format_number(effective_scale),
        "data-shellforgepy-representation": _normalize_representation(
            request.get("representation")
        )["mode"],
    }
    if sheet_spec is not None:
        attrs["data-shellforgepy-scale-ratio"] = _scale_ratio(effective_scale)
        attrs["data-shellforgepy-scale-equivalence"] = _scale_equivalence(
            effective_scale
        )
    if adapter_id is not None:
        attrs["data-shellforgepy-adapter"] = adapter_id
    if source_assembly is not None:
        attrs["data-shellforgepy-source-assembly"] = source_assembly
    metadata = request.get("metadata")
    if isinstance(metadata, Mapping):
        for key in sorted(metadata):
            attrs[f"data-shellforgepy-metadata-{_safe_svg_id(str(key))}"] = str(
                metadata[key]
            )

    root = ET.Element(_svg_tag("svg"), attrs)
    root.set("id", _safe_svg_id(str(request.get("name", "construction-drawing"))))
    if sheet_spec is not None:
        _append_technical_sheet(
            root,
            sheet_spec,
            request=sheet_request,
        )
        transform = _sheet_geometry_transform(
            sheet_spec,
            drawing_bounds=(min_x, min_y, width, height),
            requested_scale=float(request.get("scale", 1.0)),
            effective_scale=effective_scale,
        )
    else:
        transform = _svg_y_up_transform(min_y, height)
    geometry = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": "shellforgepy-geometry",
            "data-shellforgepy-role": "geometry",
            "transform": transform,
        },
    )
    return ET.ElementTree(root), geometry


def append_line(
    parent: ET.Element,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source_edge: str | None = None,
    projection_metadata: Mapping[str, str] | None = None,
) -> ET.Element:
    """Append a standard SVG line with exact-geometry provenance."""

    attrs = {
        "x1": _format_number(x1),
        "y1": _format_number(y1),
        "x2": _format_number(x2),
        "y2": _format_number(y2),
        "data-shellforgepy-geometry": "exact",
        **_GEOMETRY_STYLE,
    }
    if source_edge is not None:
        attrs["data-shellforgepy-source-edge"] = source_edge
    if projection_metadata:
        attrs.update(projection_metadata)
    return ET.SubElement(parent, _svg_tag("line"), attrs)


def append_circle(
    parent: ET.Element,
    *,
    cx: float,
    cy: float,
    radius: float,
    source_edge: str | None = None,
    projection_metadata: Mapping[str, str] | None = None,
) -> ET.Element:
    """Append a standard analytic SVG circle with exact provenance."""

    attrs = {
        "cx": _format_number(cx),
        "cy": _format_number(cy),
        "r": _format_number(radius),
        "data-shellforgepy-geometry": "exact",
        **_GEOMETRY_STYLE,
    }
    if source_edge is not None:
        attrs["data-shellforgepy-source-edge"] = source_edge
    if projection_metadata:
        attrs.update(projection_metadata)
    return ET.SubElement(parent, _svg_tag("circle"), attrs)


def append_arc(
    parent: ET.Element,
    *,
    cx: float,
    cy: float,
    radius: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    large_arc: bool,
    sweep: bool,
    source_edge: str | None = None,
    projection_metadata: Mapping[str, str] | None = None,
) -> ET.Element:
    """Append an exact circular arc using the SVG elliptical-arc command."""

    attrs = {
        "d": (
            f"M {_format_number(start_x)} {_format_number(start_y)} "
            f"A {_format_number(radius)} {_format_number(radius)} 0 "
            f"{int(bool(large_arc))} {int(bool(sweep))} "
            f"{_format_number(end_x)} {_format_number(end_y)}"
        ),
        "data-shellforgepy-geometry": "exact",
        "data-shellforgepy-center": f"{_format_number(cx)},{_format_number(cy)}",
        "data-shellforgepy-radius": _format_number(radius),
        "data-shellforgepy-start": f"{_format_number(start_x)},{_format_number(start_y)}",
        "data-shellforgepy-end": f"{_format_number(end_x)},{_format_number(end_y)}",
        "data-shellforgepy-large-arc": str(int(bool(large_arc))),
        "data-shellforgepy-sweep": str(int(bool(sweep))),
        **_GEOMETRY_STYLE,
    }
    if source_edge is not None:
        attrs["data-shellforgepy-source-edge"] = source_edge
    if projection_metadata:
        attrs.update(projection_metadata)
    return ET.SubElement(parent, _svg_tag("path"), attrs)


def append_ellipse(
    parent: ET.Element,
    *,
    cx: float,
    cy: float,
    radius_x: float,
    radius_y: float,
    rotation_degrees: float = 0.0,
    source_edge: str | None = None,
    projection_metadata: Mapping[str, str] | None = None,
) -> ET.Element:
    """Append an exact projected ellipse, retaining analytic provenance."""

    attrs = {
        "cx": _format_number(cx),
        "cy": _format_number(cy),
        "rx": _format_number(radius_x),
        "ry": _format_number(radius_y),
        "data-shellforgepy-geometry": "exact",
        **_GEOMETRY_STYLE,
    }
    if rotation_degrees:
        attrs["transform"] = (
            f"rotate({_format_number(rotation_degrees)} {_format_number(cx)} {_format_number(cy)})"
        )
    if source_edge is not None:
        attrs["data-shellforgepy-source-edge"] = source_edge
    if projection_metadata:
        attrs.update(projection_metadata)
    return ET.SubElement(parent, _svg_tag("ellipse"), attrs)


def render_construction_drawing(
    solid,
    request: Mapping[str, object],
    destination,
    *,
    part_identity: str,
    source: str | None = None,
    annotation_targets: Mapping[str, Mapping[str, object]] | None = None,
    annotation_records: list[dict[str, object]] | None = None,
) -> Path:
    """Render one selected solid as a borderless exact construction SVG.

    The direct API accepts resolved geometry and optional canonical annotation
    targets. Builder selector resolution remains outside this function; the
    adapter writes standard SVG primitives directly into the selected part
    group.
    """

    return render_construction_drawing_parts(
        [
            {
                "part": solid,
                "name": part_identity,
                "source": source,
            }
        ],
        request,
        destination,
        annotation_targets=annotation_targets,
        annotation_records=annotation_records,
    )


def render_construction_drawing_parts(
    scene_parts: Sequence[Mapping[str, object]],
    request: Mapping[str, object],
    destination,
    *,
    annotation_targets: Mapping[str, Mapping[str, object]] | None = None,
    annotation_records: list[dict[str, object]] | None = None,
    render_metadata: dict[str, object] | None = None,
) -> Path:
    """Render one or more selected views into one construction-drawing SVG."""

    if request.get("views"):
        tree = _render_multi_view_construction_drawing_tree(
            scene_parts,
            request,
            annotation_targets=annotation_targets,
            annotation_records=annotation_records,
        )
    else:
        tree = _render_single_construction_drawing_tree(
            scene_parts,
            request,
            annotation_targets=annotation_targets,
            annotation_records=annotation_records,
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialize_svg(tree))
    if render_metadata is not None:
        render_metadata.update(_drawing_scale_metadata(tree.getroot()))
    return destination


def _render_single_construction_drawing_tree(
    scene_parts: Sequence[Mapping[str, object]],
    request: Mapping[str, object],
    *,
    annotation_targets: Mapping[str, Mapping[str, object]] | None = None,
    annotation_records: list[dict[str, object]] | None = None,
) -> ET.ElementTree:
    """Render selected, already-placed scene parts into one SVG tree."""

    from shellforgepy.adapters._adapter import (
        emit_projection_svg as adapter_emit_projection_svg,
    )
    from shellforgepy.adapters._adapter import (
        emit_section_svg as adapter_emit_section_svg,
    )
    from shellforgepy.adapters._adapter import get_adapter_id, get_bounding_box

    if not scene_parts:
        raise ValueError("Construction drawing requires at least one scene part")
    solids = []
    model_bounds = []
    for index, scene_part in enumerate(scene_parts):
        if "part" not in scene_part:
            raise ValueError(f"Construction drawing scene part {index} has no geometry")
        solid = scene_part["part"]
        solids.append((solid, scene_part))
        model_bounds.append(get_bounding_box(solid))

    combined_bounds = _merge_model_bounds(model_bounds)
    frame = resolve_view_frame(request, combined_bounds)
    drawing_bounds = drawing_bounds_from_model_bounds(combined_bounds, frame)
    tree, geometry = create_svg_document(
        request,
        frame,
        drawing_bounds,
        adapter_id=get_adapter_id(),
        source_assembly=(
            str(request["source_assembly"])
            if request.get("source_assembly") is not None
            else None
        ),
    )
    representation = _normalize_representation(request.get("representation"))
    representation_mode = representation["mode"]
    section_thickness = float(request.get("section_thickness", 0.0))
    if representation_mode == "projection" and section_thickness != 0.0:
        raise ValueError("Projection representation does not accept section_thickness")
    if representation_mode == "section" and section_thickness != 0.0:
        raise ValueError("Stage 1 supports infinitely thin sections only")
    if request.get("curve_approximation", "reject") != "reject":
        raise ValueError("Stage 1 supports exact curves only")

    if representation_mode == "projection":
        adapter_emit_projection_svg(
            [
                {
                    "part": solid,
                    "name": str(scene_part.get("name") or f"part_{index + 1}"),
                    "source": _scene_part_reference(scene_part),
                }
                for index, (solid, scene_part) in enumerate(solids)
            ],
            frame,
            geometry,
            include=representation["include"],
            tolerance=float(request.get("tolerance", 1e-6)),
        )
    else:
        for index, (solid, scene_part) in enumerate(solids):
            part_name = str(scene_part.get("name") or f"part_{index + 1}")
            part_group = append_part_group(
                geometry,
                part_identity=part_name,
                source=_scene_part_reference(scene_part),
                exact=True,
            )
            adapter_emit_section_svg(
                solid,
                frame,
                part_group,
                section_thickness=section_thickness,
                tolerance=float(request.get("tolerance", 1e-6)),
            )

    # Layout deliberately follows what is visible in this drawing, rather than
    # an annotation target which may be a hidden cutter or a named sub-part.
    visible_bounds = _drawing_elements_bounds(
        [
            element
            for element in geometry.iter()
            if _local_name(element.tag) in {"line", "circle", "ellipse", "path"}
        ]
    )

    raw_annotations = request.get("annotations", [])
    if raw_annotations:
        if not isinstance(raw_annotations, Sequence) or isinstance(
            raw_annotations, (str, bytes)
        ):
            raise TypeError("Construction drawing annotations must be a list")
        resolved_targets = _annotation_target_map(
            scene_parts,
            annotation_targets=annotation_targets,
        )
        _append_dimension_annotations(
            tree.getroot(),
            geometry_transform=geometry.attrib["transform"],
            request=request,
            frame=frame,
            annotations=raw_annotations,
            targets=resolved_targets,
            visible_bounds=visible_bounds,
            section_thickness=section_thickness,
            tolerance=float(request.get("tolerance", 1e-6)),
            records=annotation_records,
        )

    _update_technical_sheet_scale(tree, request=request)
    return tree


def _render_multi_view_construction_drawing_tree(
    scene_parts: Sequence[Mapping[str, object]],
    request: Mapping[str, object],
    *,
    annotation_targets: Mapping[str, Mapping[str, object]] | None,
    annotation_records: list[dict[str, object]] | None,
) -> ET.ElementTree:
    """Compose independently sectioned views into one uniformly scaled SVG."""

    raw_views = request.get("views")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise TypeError("Construction drawing views must be a list")
    if not raw_views:
        raise ValueError("Construction drawing views must not be empty")

    from shellforgepy.adapters._adapter import get_adapter_id, get_bounding_box

    model_bounds = [get_bounding_box(scene_part["part"]) for scene_part in scene_parts]
    combined_model_bounds = _merge_model_bounds(model_bounds)
    rendered_views: list[dict[str, object]] = []
    for raw_view in raw_views:
        assert isinstance(raw_view, Mapping)
        view_id = str(raw_view["id"])
        local_request = {
            key: value
            for key, value in request.items()
            if key not in {"views", "sheet", "view", "annotations"}
        }
        local_request["name"] = f"{request['name']}_{view_id}"
        local_request["view"] = raw_view["view"]
        local_request["representation"] = raw_view.get("representation")
        if "annotations" in raw_view:
            local_request["annotations"] = raw_view["annotations"]
        local_records: list[dict[str, object]] = []
        local_tree = _render_single_construction_drawing_tree(
            scene_parts,
            local_request,
            annotation_targets=annotation_targets,
            annotation_records=local_records,
        )
        local_root = local_tree.getroot()
        local_geometry = local_root.find(
            f"./{{{SVG_NS}}}g[@id='shellforgepy-geometry']"
        )
        if local_geometry is None:
            raise RuntimeError(
                "Construction drawing view geometry group was not emitted"
            )
        local_annotations = local_root.find(
            f"./{{{SVG_NS}}}g[@id='shellforgepy-annotations']"
        )
        bounds = _multi_view_content_bounds(local_geometry, local_annotations)
        frame = resolve_view_frame(local_request, combined_model_bounds)
        rendered_views.append(
            {
                "id": view_id,
                "frame": frame,
                "geometry": local_geometry,
                "annotations": local_annotations,
                "bounds": bounds,
                "records": local_records,
                "view": raw_view["view"],
                "representation": _normalize_representation(
                    raw_view.get("representation")
                ),
                "placement": raw_view.get("placement"),
            }
        )

    previous_bounds: Bounds2D | None = None
    for index, rendered_view in enumerate(rendered_views):
        bounds = rendered_view["bounds"]
        assert isinstance(bounds, Bounds2D)
        if index == 0:
            placed_bounds = bounds.translated(-bounds.center_x, -bounds.center_y)
        else:
            assert previous_bounds is not None
            placement = rendered_view.get("placement") or {
                "alignments": [
                    {
                        "alignment": Alignment.STACK_RIGHT.name,
                        "stack_gap": _MULTI_VIEW_STACK_GAP,
                    }
                ]
            }
            operations = _annotation_layout_operations(
                placement,
                default_alignment=Alignment.STACK_RIGHT,
            )
            placed_bounds = align_bounds_sequence_2d(
                bounds, previous_bounds, operations
            )
        dx = placed_bounds.min_x - bounds.min_x
        dy = placed_bounds.min_y - bounds.min_y
        rendered_view["translation"] = (dx, dy)
        rendered_view["placed_bounds"] = placed_bounds
        previous_bounds = placed_bounds

    placed_bounds = [item["placed_bounds"] for item in rendered_views]
    assert all(isinstance(item, Bounds2D) for item in placed_bounds)
    merged_bounds = _merge_bounds_2d(*placed_bounds)  # type: ignore[arg-type]
    # Treat all placed views as one drawing, rather than preserving the first
    # view as an implicit origin.  This makes the group intrinsically centered
    # before the common sheet transform centers it in the framed viewport.
    for rendered_view in rendered_views:
        dx, dy = rendered_view["translation"]  # type: ignore[misc]
        rendered_view["translation"] = (
            dx - merged_bounds.center_x,
            dy - merged_bounds.center_y,
        )
        rendered_view["placed_bounds"] = rendered_view["placed_bounds"].translated(
            -merged_bounds.center_x,
            -merged_bounds.center_y,
        )
    placed_bounds = [item["placed_bounds"] for item in rendered_views]
    assert all(isinstance(item, Bounds2D) for item in placed_bounds)
    drawing_bounds_2d = _merge_bounds_2d(*placed_bounds)  # type: ignore[arg-type]
    drawing_bounds = (
        drawing_bounds_2d.min_x,
        drawing_bounds_2d.min_y,
        drawing_bounds_2d.max_x - drawing_bounds_2d.min_x,
        drawing_bounds_2d.max_y - drawing_bounds_2d.min_y,
    )
    first_frame = rendered_views[0]["frame"]
    assert isinstance(first_frame, Mapping)
    root_request = {
        key: value for key, value in request.items() if key not in {"views", "view"}
    }
    root_request["view"] = "multiple"
    tree, root_geometry = create_svg_document(
        root_request,
        first_frame,  # type: ignore[arg-type]
        drawing_bounds,
        adapter_id=get_adapter_id(),
        source_assembly=(
            str(request["source_assembly"])
            if request.get("source_assembly") is not None
            else None
        ),
    )
    root = tree.getroot()
    representations = {
        item["representation"]["mode"] for item in rendered_views  # type: ignore[index]
    }
    root.set(
        "data-shellforgepy-representation",
        representations.pop() if len(representations) == 1 else "multiple",
    )
    root.set(
        "data-shellforgepy-views", ",".join(str(item["id"]) for item in rendered_views)
    )
    root.set("data-shellforgepy-view-count", str(len(rendered_views)))
    if any(item["annotations"] is not None for item in rendered_views):
        _ensure_dimension_arrow_marker(root)

    for rendered_view in rendered_views:
        view_id = str(rendered_view["id"])
        dx, dy = rendered_view["translation"]  # type: ignore[misc]
        group = ET.SubElement(
            root_geometry,
            _svg_tag("g"),
            {
                "id": _safe_svg_id(f"view-{view_id}"),
                "data-shellforgepy-role": "view",
                "data-shellforgepy-view-id": view_id,
                "data-shellforgepy-view": _view_metadata(rendered_view["view"]),
                "data-shellforgepy-representation": rendered_view["representation"][  # type: ignore[index]
                    "mode"
                ],
                "data-shellforgepy-section-normal": _format_vector(
                    rendered_view["frame"]["normal"]  # type: ignore[index]
                ),
                "data-shellforgepy-section-up": _format_vector(
                    rendered_view["frame"]["up"]  # type: ignore[index]
                ),
                "data-shellforgepy-section-origin": _format_vector(
                    rendered_view["frame"]["origin"]  # type: ignore[index]
                ),
                "transform": f"translate({_format_number(dx)} {_format_number(dy)})",
            },
        )
        view_content = [rendered_view["geometry"]]
        if rendered_view["annotations"] is not None:
            view_content.append(rendered_view["annotations"])
        _prefix_svg_ids(view_content, prefix=f"view-{view_id}-")
        for content in view_content:
            assert isinstance(content, ET.Element)
            content.attrib.pop("transform", None)
            group.append(content)
        if annotation_records is not None:
            for record in rendered_view["records"]:  # type: ignore[union-attr]
                record["view_id"] = view_id
                annotation_records.append(record)

    # ``drawing_bounds`` above already represents the placed union of every
    # view and annotation.  ``create_svg_document`` therefore centers that
    # complete, merged footprint in the technical viewport.  Do not recompute
    # it from nested SVG elements here: their per-view translations are local
    # transforms and would incorrectly center only their unplaced coordinates.
    return tree


def _multi_view_content_bounds(
    geometry: ET.Element,
    annotations: ET.Element | None,
) -> Bounds2D:
    elements = [
        element
        for element in geometry.iter()
        if _local_name(element.tag) in {"line", "circle", "ellipse", "path"}
    ]
    bounds = [Bounds2D(*_drawing_elements_bounds(elements))]
    if annotations is not None:
        for annotation in annotations.findall(
            f".//{{{SVG_NS}}}g[@data-shellforgepy-role='dimension']"
        ):
            placed = annotation.attrib.get("data-shellforgepy-placed-bounds")
            if placed is not None:
                bounds.append(Bounds2D(*_parse_bounds(placed)))
    return _merge_bounds_2d(*bounds)


def _prefix_svg_ids(elements: Sequence[ET.Element], *, prefix: str) -> None:
    replacements: dict[str, str] = {}
    for element in elements:
        for child in element.iter():
            identifier = child.attrib.get("id")
            if identifier:
                replacements[identifier] = _safe_svg_id(f"{prefix}{identifier}")
    for element in elements:
        for child in element.iter():
            identifier = child.attrib.get("id")
            if identifier:
                child.attrib["id"] = replacements[identifier]
            for key, value in tuple(child.attrib.items()):
                for old, new in replacements.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})")
                child.attrib[key] = value


def _annotation_target_map(
    scene_parts: Sequence[Mapping[str, object]],
    *,
    annotation_targets: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, Mapping[str, object]]:
    targets: dict[str, Mapping[str, object]] = {}
    for scene_part in scene_parts:
        reference = _scene_part_reference(scene_part)
        if reference:
            targets.setdefault(reference, scene_part)
    for reference, scene_part in (annotation_targets or {}).items():
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                "Construction drawing annotation target keys must be strings"
            )
        if "part" not in scene_part:
            raise ValueError(
                f"Construction drawing annotation target {reference!r} has no geometry"
            )
        targets[reference] = scene_part
    return targets


def _scene_part_reference(scene_part: Mapping[str, object]) -> str | None:
    source = scene_part.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    obj_metadata = scene_part.get("obj_metadata")
    if isinstance(obj_metadata, Mapping):
        builder_selector = obj_metadata.get("builder_selector")
        if isinstance(builder_selector, str) and builder_selector.strip():
            return builder_selector.strip()
    return None


def _append_dimension_annotations(
    root: ET.Element,
    *,
    geometry_transform: str,
    request: Mapping[str, object],
    frame: SectionViewFrame,
    annotations: Sequence[object],
    targets: Mapping[str, Mapping[str, object]],
    visible_bounds: tuple[float, float, float, float],
    section_thickness: float,
    tolerance: float,
    records: list[dict[str, object]] | None,
) -> None:
    from shellforgepy.adapters._adapter import (
        emit_projection_svg as adapter_emit_projection_svg,
    )
    from shellforgepy.adapters._adapter import (
        emit_section_svg as adapter_emit_section_svg,
    )

    normalized_annotations = _normalize_annotations(annotations)  # type: ignore[arg-type]
    marker_id = _ensure_dimension_arrow_marker(root)
    annotation_layer = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": "shellforgepy-annotations",
            "data-shellforgepy-role": "annotations",
            "transform": geometry_transform,
        },
    )
    precision = int(request.get("precision", 2))
    units = str(request.get("units", "mm"))

    resolved_annotations: list[dict[str, object]] = []
    for annotation in normalized_annotations:
        annotation_id = annotation["id"]
        operation = annotation["operation"]
        if operation == "linear_dimension":
            from_endpoint = annotation["from"]
            to_endpoint = annotation["to"]
            assert isinstance(from_endpoint, Mapping)
            assert isinstance(to_endpoint, Mapping)
            from_target, from_bounds, _ = _resolve_annotation_target_geometry(
                targets,
                str(from_endpoint["target"]),
                annotation_id=annotation_id,
                endpoint_name="from",
                adapter_emit_section_svg=adapter_emit_section_svg,
                adapter_emit_projection_svg=adapter_emit_projection_svg,
                representation=_normalize_representation(request.get("representation")),
                frame=frame,
                section_thickness=section_thickness,
                tolerance=tolerance,
            )
            to_target, to_bounds, _ = _resolve_annotation_target_geometry(
                targets,
                str(to_endpoint["target"]),
                annotation_id=annotation_id,
                endpoint_name="to",
                adapter_emit_section_svg=adapter_emit_section_svg,
                adapter_emit_projection_svg=adapter_emit_projection_svg,
                representation=_normalize_representation(request.get("representation")),
                frame=frame,
                section_thickness=section_thickness,
                tolerance=tolerance,
            )
            dimension_direction = str(annotation["dimension_direction"])
            from_coordinate = _projected_endpoint_coordinate(
                from_bounds, str(from_endpoint["edge"]), dimension_direction
            )
            to_coordinate = _projected_endpoint_coordinate(
                to_bounds, str(to_endpoint["edge"]), dimension_direction
            )
            if dimension_direction == "RIGHT":
                value = to_coordinate - from_coordinate
            else:
                value = from_coordinate - to_coordinate
            if value < 0:
                comparison = (
                    "from_x <= to_x"
                    if dimension_direction == "RIGHT"
                    else "from_y >= to_y"
                )
                raise ValueError(
                    f"Construction drawing annotation {annotation_id!r} "
                    f"{dimension_direction} dimension requires {comparison}; got "
                    f"{from_coordinate:g}, {to_coordinate:g}"
                )
            resolved_annotations.append(
                {
                    "annotation": annotation,
                    "from_target": from_target,
                    "to_target": to_target,
                    "from_bounds": from_bounds,
                    "to_bounds": to_bounds,
                    "dimension_direction": dimension_direction,
                    "from_coordinate": from_coordinate,
                    "to_coordinate": to_coordinate,
                    "value": value,
                }
            )
            continue

        target_ref = annotation["target"]
        target, target_bounds, target_elements = _resolve_annotation_target_geometry(
            targets,
            target_ref,
            annotation_id=annotation_id,
            endpoint_name="target",
            adapter_emit_section_svg=adapter_emit_section_svg,
            adapter_emit_projection_svg=adapter_emit_projection_svg,
            representation=_normalize_representation(request.get("representation")),
            frame=frame,
            section_thickness=section_thickness,
            tolerance=tolerance,
        )
        resolved_annotations.append(
            {
                "annotation": annotation,
                "target": target,
                "target_bounds": target_bounds,
                "target_elements": target_elements,
            }
        )

    dimension_stack_offsets = _linear_dimension_stack_offsets(resolved_annotations)
    for annotation_index, resolved in enumerate(resolved_annotations):
        annotation = resolved["annotation"]
        assert isinstance(annotation, Mapping)
        annotation_id = str(annotation["id"])
        operation = str(annotation["operation"])
        stack_side, stack_offset = dimension_stack_offsets.get(
            annotation_index, (None, 0.0)
        )
        placement = annotation.get("placement")
        if stack_side is not None and stack_offset:
            default_alignment = (
                Alignment.STACK_FRONT
                if operation == "bounding_box_x_dimension"
                or (
                    operation == "linear_dimension"
                    and annotation["dimension_direction"] == "RIGHT"
                )
                else Alignment.STACK_RIGHT
            )
            placement = _placement_with_extra_stack_gap(
                placement,
                default_alignment=default_alignment,
                stack_alignment=stack_side,
                extra_stack_gap=stack_offset,
            )
        annotation_group = ET.SubElement(
            annotation_layer,
            _svg_tag("g"),
            {
                "id": _safe_svg_id(f"annotation-{annotation_id}"),
                "data-shellforgepy-role": "dimension",
                "data-shellforgepy-annotation-id": annotation_id,
                "data-shellforgepy-operation": operation,
                "data-shellforgepy-value-source": "exact-geometry",
                "data-shellforgepy-layout-bounds": _format_bounds(visible_bounds),
            },
        )

        if operation == "bounding_box_x_dimension":
            target_bounds = resolved["target_bounds"]
            assert isinstance(target_bounds, tuple)
            target_ref = str(annotation["target"])
            annotation_group.set("data-shellforgepy-target", target_ref)
            annotation_group.set(
                "data-shellforgepy-target-bounds", _format_bounds(target_bounds)
            )
            value, placed_bounds = _append_x_dimension(
                annotation_group,
                target_bounds=target_bounds,
                layout_bounds=visible_bounds,
                placement=placement,
                marker_id=marker_id,
                precision=precision,
            )
        elif operation == "bounding_box_y_dimension":
            target_bounds = resolved["target_bounds"]
            assert isinstance(target_bounds, tuple)
            target_ref = str(annotation["target"])
            annotation_group.set("data-shellforgepy-target", target_ref)
            annotation_group.set(
                "data-shellforgepy-target-bounds", _format_bounds(target_bounds)
            )
            value, placed_bounds = _append_y_dimension(
                annotation_group,
                target_bounds=target_bounds,
                layout_bounds=visible_bounds,
                placement=placement,
                marker_id=marker_id,
                precision=precision,
            )
        elif operation == "circle_diameter":
            target_bounds = resolved["target_bounds"]
            target_elements = resolved["target_elements"]
            assert isinstance(target_bounds, tuple)
            assert isinstance(target_elements, list)
            target_ref = str(annotation["target"])
            annotation_group.set("data-shellforgepy-target", target_ref)
            annotation_group.set(
                "data-shellforgepy-target-bounds", _format_bounds(target_bounds)
            )
            value, placed_bounds = _append_circle_diameter_dimension(
                annotation_group,
                target_elements=target_elements,
                target_bounds=target_bounds,
                layout_bounds=visible_bounds,
                placement=placement,
                marker_id=marker_id,
                precision=precision,
                annotation=annotation,
            )
        elif operation == "linear_dimension":
            from_endpoint = annotation["from"]
            to_endpoint = annotation["to"]
            from_bounds = resolved["from_bounds"]
            to_bounds = resolved["to_bounds"]
            dimension_direction = str(resolved["dimension_direction"])
            from_coordinate = float(resolved["from_coordinate"])
            to_coordinate = float(resolved["to_coordinate"])
            assert isinstance(from_endpoint, Mapping)
            assert isinstance(to_endpoint, Mapping)
            assert isinstance(from_bounds, tuple)
            assert isinstance(to_bounds, tuple)
            annotation_group.set(
                "data-shellforgepy-from-target", str(from_endpoint["target"])
            )
            annotation_group.set(
                "data-shellforgepy-from-edge", str(from_endpoint["edge"])
            )
            annotation_group.set(
                "data-shellforgepy-to-target", str(to_endpoint["target"])
            )
            annotation_group.set("data-shellforgepy-to-edge", str(to_endpoint["edge"]))
            annotation_group.set(
                "data-shellforgepy-dimension-direction",
                dimension_direction,
            )
            annotation_group.set(
                "data-shellforgepy-from-target-bounds", _format_bounds(from_bounds)
            )
            annotation_group.set(
                "data-shellforgepy-to-target-bounds", _format_bounds(to_bounds)
            )
            annotation_group.set(
                "data-shellforgepy-from-coordinate",
                _format_number(from_coordinate),
            )
            annotation_group.set(
                "data-shellforgepy-to-coordinate", _format_number(to_coordinate)
            )
            if dimension_direction == "RIGHT":
                value, placed_bounds, rule_bounds, label_bounds = (
                    _append_linear_x_dimension(
                        annotation_group,
                        from_x=from_coordinate,
                        to_x=to_coordinate,
                        from_bounds=from_bounds,
                        to_bounds=to_bounds,
                        layout_bounds=visible_bounds,
                        placement=placement,
                        marker_id=marker_id,
                        precision=precision,
                    )
                )
            else:
                value, placed_bounds, rule_bounds, label_bounds = (
                    _append_linear_y_dimension(
                        annotation_group,
                        from_y=from_coordinate,
                        to_y=to_coordinate,
                        from_bounds=from_bounds,
                        to_bounds=to_bounds,
                        layout_bounds=visible_bounds,
                        placement=placement,
                        marker_id=marker_id,
                        precision=precision,
                    )
                )
            annotation_group.set(
                "data-shellforgepy-rule-bounds", _format_bounds(rule_bounds)
            )
            annotation_group.set(
                "data-shellforgepy-label-bounds", _format_bounds(label_bounds)
            )
        else:  # _normalize_annotations keeps this defensive branch unreachable.
            raise ValueError(
                f"Unsupported construction drawing operation {operation!r}"
            )

        formatted_value = _format_dimension_value(value, precision)
        annotation_group.set("data-shellforgepy-value", formatted_value)
        annotation_group.set("data-shellforgepy-units", units)
        annotation_group.set("data-shellforgepy-precision", str(precision))
        annotation_group.set(
            "data-shellforgepy-placed-bounds", _format_bounds(placed_bounds)
        )
        callout = _circle_diameter_callout_data(annotation)
        for key, callout_value in callout.items():
            annotation_group.set(
                f"data-shellforgepy-{key.replace('_', '-')}",
                _format_annotation_callout_value(callout_value),
            )
        if records is not None:
            record: dict[str, object] = {
                "id": annotation_id,
                "operation": operation,
                "value": value,
                "formatted_value": formatted_value,
                "units": units,
                "precision": precision,
                "layout_bounds": list(visible_bounds),
                "placed_bounds": list(placed_bounds),
            }
            if operation == "linear_dimension":
                record.update(
                    {
                        "from": {
                            "target": annotation["from"]["target"],
                            "edge": annotation["from"]["edge"],
                            "projected_coordinate": float(resolved["from_coordinate"]),
                        },
                        "to": {
                            "target": annotation["to"]["target"],
                            "edge": annotation["to"]["edge"],
                            "projected_coordinate": float(resolved["to_coordinate"]),
                        },
                        "dimension_direction": annotation["dimension_direction"],
                        "from_target_bounds": list(from_bounds),
                        "to_target_bounds": list(to_bounds),
                        "visible_scene_layout_bounds": list(visible_bounds),
                        "rule_bounds": list(rule_bounds),
                        "label_bounds": list(label_bounds),
                    }
                )
            else:
                record["target"] = annotation["target"]
                record["target_bounds"] = list(target_bounds)
            if callout:
                record["callout"] = callout
            records.append(record)


def _resolve_annotation_target_geometry(
    targets: Mapping[str, Mapping[str, object]],
    target_ref: str,
    *,
    annotation_id: str,
    endpoint_name: str,
    adapter_emit_section_svg,
    adapter_emit_projection_svg,
    representation: ProjectionRepresentation,
    frame: SectionViewFrame,
    section_thickness: float,
    tolerance: float,
) -> tuple[Mapping[str, object], tuple[float, float, float, float], list[ET.Element]]:
    target = targets.get(target_ref)
    if target is None:
        available = ", ".join(sorted(targets)) or "none"
        raise ValueError(
            f"Construction drawing annotation {annotation_id!r} {endpoint_name} target "
            f"{target_ref!r} did not resolve; available targets: {available}"
        )
    target_elements_parent = ET.Element(_svg_tag("g"))
    if representation["mode"] == "projection":
        adapter_emit_projection_svg(
            [{"part": target["part"], "name": target_ref, "source": target_ref}],
            frame,
            target_elements_parent,
            include=representation["include"],
            tolerance=tolerance,
        )
    else:
        adapter_emit_section_svg(
            target["part"],
            frame,
            target_elements_parent,
            section_thickness=section_thickness,
            tolerance=tolerance,
        )
    target_elements = [
        element
        for element in target_elements_parent.iter()
        if _local_name(element.tag) in {"line", "circle", "ellipse", "path"}
    ]
    return target, _drawing_elements_bounds(target_elements), target_elements


def _projected_endpoint_coordinate(
    bounds: tuple[float, float, float, float], edge: str, dimension_direction: str
) -> float:
    if dimension_direction == "RIGHT":
        if edge == "EDGE_LEFT":
            return bounds[0]
        if edge == "EDGE_RIGHT":
            return bounds[2]
    elif dimension_direction == "BACK":
        if edge == "EDGE_FRONT":
            return bounds[1]
        if edge == "EDGE_BACK":
            return bounds[3]
    raise ValueError(
        f"{dimension_direction} linear dimensions do not support edge {edge!r}"
    )


def _linear_dimension_stack_offsets(
    resolved_annotations: Sequence[Mapping[str, object]],
) -> dict[int, tuple[Alignment, float]]:
    """Assign deterministic outside nesting to same-side axial dimensions."""

    grouped: dict[Alignment, list[tuple[int, float]]] = {}
    for index, resolved in enumerate(resolved_annotations):
        annotation = resolved["annotation"]
        assert isinstance(annotation, Mapping)
        operation = annotation["operation"]
        if operation == "bounding_box_x_dimension":
            bounds = resolved["target_bounds"]
            assert isinstance(bounds, tuple)
            span = bounds[2] - bounds[0]
            dimension_direction = "RIGHT"
        elif operation == "bounding_box_y_dimension":
            bounds = resolved["target_bounds"]
            assert isinstance(bounds, tuple)
            span = bounds[3] - bounds[1]
            dimension_direction = "BACK"
        elif operation == "linear_dimension":
            span = float(resolved["value"])
            dimension_direction = str(resolved["dimension_direction"])
        else:
            continue
        stack_side = _linear_dimension_stack_side(annotation, dimension_direction)
        if stack_side is not None:
            grouped.setdefault(stack_side, []).append((index, span))

    offsets: dict[int, tuple[Alignment, float]] = {}
    for stack_side, items in grouped.items():
        for nesting_index, (annotation_index, _) in enumerate(
            sorted(items, key=lambda item: (item[1], item[0]))
        ):
            offsets[annotation_index] = (
                stack_side,
                nesting_index * _LINEAR_DIMENSION_STACK_PITCH,
            )
    return offsets


def _linear_dimension_stack_side(
    annotation: Mapping[str, object], dimension_direction: str
) -> Alignment | None:
    operations = _annotation_layout_operations(
        annotation.get("placement"),
        default_alignment=(
            Alignment.STACK_FRONT
            if dimension_direction == "RIGHT"
            else Alignment.STACK_RIGHT
        ),
    )
    eligible_sides = (
        {Alignment.STACK_FRONT, Alignment.STACK_BACK}
        if dimension_direction == "RIGHT"
        else {Alignment.STACK_LEFT, Alignment.STACK_RIGHT}
    )
    stack_sides = [
        alignment for alignment, _ in operations if alignment in eligible_sides
    ]
    return stack_sides[-1] if stack_sides else None


def _placement_with_extra_stack_gap(
    placement: object,
    *,
    default_alignment: Alignment,
    stack_alignment: Alignment,
    extra_stack_gap: float,
) -> AnnotationPlacement:
    if placement is None:
        return {
            "alignments": [
                {
                    "alignment": default_alignment.name,
                    "stack_gap": 5.0 + extra_stack_gap,
                }
            ]
        }
    assert isinstance(placement, Mapping)
    alignments = placement["alignments"]
    assert isinstance(alignments, Sequence)
    normalized_alignments = [dict(item) for item in alignments]
    for alignment in reversed(normalized_alignments):
        if alignment.get("alignment") == stack_alignment.name:
            alignment["stack_gap"] = (
                float(alignment.get("stack_gap", 0.0)) + extra_stack_gap
            )
            break
    return {"alignments": normalized_alignments}  # type: ignore[return-value]


def _append_x_dimension(
    parent: ET.Element,
    *,
    target_bounds: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    marker_id: str,
    precision: int,
) -> tuple[float, tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = target_bounds
    placed_line = _place_annotation_bounds(
        Bounds2D(min_x, 0.0, max_x, 0.0),
        layout_bounds,
        placement,
        default_alignment=Alignment.STACK_FRONT,
    )
    line_y = placed_line.min_y
    line_min_x, line_max_x = placed_line.min_x, placed_line.max_x
    witness_y = min_y if line_y <= (min_y + max_y) / 2.0 else max_y
    extension_start_y = witness_y + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_y - witness_y
    )
    extension_end_y = line_y + math.copysign(
        _EXTENSION_LINE_DIMENSION_OVERRUN, line_y - witness_y
    )
    _append_extension_line(
        parent, min_x, extension_start_y, line_min_x, extension_end_y
    )
    _append_extension_line(
        parent, max_x, extension_start_y, line_max_x, extension_end_y
    )
    _append_annotation_line(
        parent,
        line_min_x,
        line_y,
        line_max_x,
        line_y,
        marker_start=marker_id,
        marker_end=marker_id,
    )
    text_y = (
        line_y + _annotation_text_baseline_below_line()
        if line_y <= witness_y
        else line_y + _annotation_text_baseline_above_line()
    )
    value = _format_dimension_value(max_x - min_x, precision)
    _append_annotation_text(
        parent,
        x=(line_min_x + line_max_x) / 2.0,
        y=text_y,
        value=value,
        anchor="middle",
    )
    return (
        max_x - min_x,
        _merge_bounds_2d(
            placed_line,
            _annotation_text_bounds(
                x=(line_min_x + line_max_x) / 2.0,
                y=text_y,
                value=value,
                anchor="middle",
            ),
        ).as_tuple(),
    )


def _append_linear_x_dimension(
    parent: ET.Element,
    *,
    from_x: float,
    to_x: float,
    from_bounds: tuple[float, float, float, float],
    to_bounds: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    marker_id: str,
    precision: int,
) -> tuple[
    float,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Render a RIGHT dimension from two explicitly selected projected edges."""

    placed_line = _place_annotation_bounds(
        Bounds2D(from_x, 0.0, to_x, 0.0),
        layout_bounds,
        placement,
        default_alignment=Alignment.STACK_FRONT,
    )
    line_y = placed_line.min_y
    witness_y_from = (
        from_bounds[1]
        if line_y <= (from_bounds[1] + from_bounds[3]) / 2
        else from_bounds[3]
    )
    witness_y_to = (
        to_bounds[1] if line_y <= (to_bounds[1] + to_bounds[3]) / 2 else to_bounds[3]
    )
    from_start_y = witness_y_from + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_y - witness_y_from
    )
    to_start_y = witness_y_to + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_y - witness_y_to
    )
    extension_end_y = line_y + math.copysign(
        _EXTENSION_LINE_DIMENSION_OVERRUN, line_y - witness_y_from
    )
    _append_extension_line(parent, from_x, from_start_y, from_x, extension_end_y)
    _append_extension_line(parent, to_x, to_start_y, to_x, extension_end_y)
    _append_annotation_line(
        parent,
        from_x,
        line_y,
        to_x,
        line_y,
        marker_start=marker_id,
        marker_end=marker_id,
    )
    text_y = (
        line_y + _annotation_text_baseline_below_line()
        if line_y <= min(witness_y_from, witness_y_to)
        else line_y + _annotation_text_baseline_above_line()
    )
    value = to_x - from_x
    formatted_value = _format_dimension_value(value, precision)
    label_bounds = _annotation_text_bounds(
        x=(from_x + to_x) / 2.0,
        y=text_y,
        value=formatted_value,
        anchor="middle",
    )
    _append_annotation_text(
        parent,
        x=(from_x + to_x) / 2.0,
        y=text_y,
        value=formatted_value,
        anchor="middle",
    )
    return (
        value,
        _merge_bounds_2d(placed_line, label_bounds).as_tuple(),
        placed_line.as_tuple(),
        label_bounds.as_tuple(),
    )


def _append_linear_y_dimension(
    parent: ET.Element,
    *,
    from_y: float,
    to_y: float,
    from_bounds: tuple[float, float, float, float],
    to_bounds: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    marker_id: str,
    precision: int,
) -> tuple[
    float,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Render a BACK dimension from named back and front projected edges."""

    placed_line = _place_annotation_bounds(
        Bounds2D(0.0, to_y, 0.0, from_y),
        layout_bounds,
        placement,
        default_alignment=Alignment.STACK_RIGHT,
    )
    line_x = placed_line.min_x
    witness_x_from = (
        from_bounds[0]
        if line_x <= (from_bounds[0] + from_bounds[2]) / 2
        else from_bounds[2]
    )
    witness_x_to = (
        to_bounds[0] if line_x <= (to_bounds[0] + to_bounds[2]) / 2 else to_bounds[2]
    )
    from_start_x = witness_x_from + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_x - witness_x_from
    )
    to_start_x = witness_x_to + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_x - witness_x_to
    )
    extension_end_x = line_x + math.copysign(
        _EXTENSION_LINE_DIMENSION_OVERRUN, line_x - witness_x_from
    )
    _append_extension_line(parent, from_start_x, from_y, extension_end_x, from_y)
    _append_extension_line(parent, to_start_x, to_y, extension_end_x, to_y)
    _append_annotation_line(
        parent,
        line_x,
        from_y,
        line_x,
        to_y,
        marker_start=marker_id,
        marker_end=marker_id,
    )
    text_anchor = "end" if line_x <= min(witness_x_from, witness_x_to) else "start"
    text_x = line_x - 2.0 if text_anchor == "end" else line_x + 2.0
    value = from_y - to_y
    formatted_value = _format_dimension_value(value, precision)
    text_y = (from_y + to_y) / 2.0
    label_bounds = _annotation_text_bounds(
        x=text_x,
        y=text_y,
        value=formatted_value,
        anchor=text_anchor,
    )
    _append_annotation_text(
        parent,
        x=text_x,
        y=text_y,
        value=formatted_value,
        anchor=text_anchor,
    )
    return (
        value,
        _merge_bounds_2d(placed_line, label_bounds).as_tuple(),
        placed_line.as_tuple(),
        label_bounds.as_tuple(),
    )


def _append_y_dimension(
    parent: ET.Element,
    *,
    target_bounds: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    marker_id: str,
    precision: int,
) -> tuple[float, tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = target_bounds
    placed_line = _place_annotation_bounds(
        Bounds2D(0.0, min_y, 0.0, max_y),
        layout_bounds,
        placement,
        default_alignment=Alignment.STACK_RIGHT,
    )
    line_x = placed_line.min_x
    line_min_y, line_max_y = placed_line.min_y, placed_line.max_y
    witness_x = min_x if line_x <= (min_x + max_x) / 2.0 else max_x
    extension_start_x = witness_x + math.copysign(
        _EXTENSION_LINE_PART_GAP, line_x - witness_x
    )
    extension_end_x = line_x + math.copysign(
        _EXTENSION_LINE_DIMENSION_OVERRUN, line_x - witness_x
    )
    _append_extension_line(
        parent, extension_start_x, min_y, extension_end_x, line_min_y
    )
    _append_extension_line(
        parent, extension_start_x, max_y, extension_end_x, line_max_y
    )
    _append_annotation_line(
        parent,
        line_x,
        line_min_y,
        line_x,
        line_max_y,
        marker_start=marker_id,
        marker_end=marker_id,
    )
    text_anchor = "end" if line_x <= witness_x else "start"
    text_x = line_x - 2.0 if text_anchor == "end" else line_x + 2.0
    value = _format_dimension_value(max_y - min_y, precision)
    text_y = (line_min_y + line_max_y) / 2.0
    _append_annotation_text(
        parent,
        x=text_x,
        y=text_y,
        value=value,
        anchor=text_anchor,
    )
    return (
        max_y - min_y,
        _merge_bounds_2d(
            placed_line,
            _annotation_text_bounds(
                x=text_x,
                y=text_y,
                value=value,
                anchor=text_anchor,
            ),
        ).as_tuple(),
    )


def _append_circle_diameter_dimension(
    parent: ET.Element,
    *,
    target_elements: Sequence[ET.Element],
    target_bounds: tuple[float, float, float, float],
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    marker_id: str,
    precision: int,
    annotation: Mapping[str, object],
) -> tuple[float, tuple[float, float, float, float]]:
    circle = _circle_or_arc_geometry_from_elements(target_elements)
    if circle is None:
        raise ValueError(
            "circle_diameter requires the target to produce exactly one 2D circle or arc"
        )
    center_x, center_y, radius, is_arc = circle
    labels = _circle_diameter_callout_lines(
        2.0 * radius,
        annotation=annotation,
        precision=precision,
    )
    thread_label = labels[0] if isinstance(annotation.get("thread_size"), str) else None
    diameter_label = labels[1] if thread_label is not None else labels[0]
    callout_width = max(
        _estimated_annotation_text_width(label)
        for label in (thread_label, diameter_label)
        if label is not None
    )
    lower_offset = _annotation_text_baseline_below_line() - _ANNOTATION_TEXT_DESCENT
    upper_offset = _annotation_text_baseline_above_line() + _ANNOTATION_TEXT_ASCENT
    vertical, vertical_gap, horizontal, horizontal_gap = (
        _resolve_circle_callout_quadrant(
            placement,
            annotation_id=str(annotation.get("id", "<unnamed>")),
        )
    )
    callout_bounds = align_bounds_sequence_2d(
        Bounds2D(
            center_x,
            center_y + lower_offset,
            center_x + callout_width,
            center_y + upper_offset,
        ),
        Bounds2D(*layout_bounds),
        ((vertical, vertical_gap), (horizontal, horizontal_gap)),
    )
    rule_y = callout_bounds.min_y - lower_offset
    horizontal_delta = callout_bounds.center_x - center_x
    if abs(horizontal_delta) > 1e-9:
        line_direction = 1.0 if horizontal_delta > 0 else -1.0
    else:
        line_direction = (
            1.0
            if horizontal
            in {Alignment.RIGHT, Alignment.EDGE_RIGHT, Alignment.STACK_RIGHT}
            else -1.0
        )
    elbow_x = callout_bounds.min_x if line_direction > 0 else callout_bounds.max_x
    elbow_y = rule_y
    direction_x = elbow_x - center_x
    direction_y = elbow_y - center_y
    direction_length = math.hypot(direction_x, direction_y)
    if direction_length <= 1e-9:
        direction_x, direction_y, direction_length = 1.0, 0.0, 1.0
    unit_x, unit_y = direction_x / direction_length, direction_y / direction_length
    if is_arc:
        arrow_x, arrow_y = _arc_anchor_toward_elements(target_elements, unit_x, unit_y)
        arrow_x += unit_x * _CIRCLE_ARROW_CLEARANCE
        arrow_y += unit_y * _CIRCLE_ARROW_CLEARANCE
    else:
        arrow_x, arrow_y = _circle_anchor_toward_point(
            center_x, center_y, radius, elbow_x, elbow_y
        )
    elbow_length = float(
        annotation.get("leader_elbow_length", _DEFAULT_CIRCLE_LEADER_ELBOW_LENGTH)
    )
    leader_length = math.hypot(elbow_x - arrow_x, elbow_y - arrow_y)
    if leader_length < elbow_length:
        if leader_length <= 1e-9:
            shift_x, shift_y = unit_x * elbow_length, unit_y * elbow_length
        else:
            extension = elbow_length - leader_length
            shift_x = (elbow_x - arrow_x) / leader_length * extension
            shift_y = (elbow_y - arrow_y) / leader_length * extension
        callout_bounds = callout_bounds.translated(shift_x, shift_y)
        elbow_x += shift_x
        elbow_y += shift_y
        if is_arc:
            shifted_direction_x = elbow_x - center_x
            shifted_direction_y = elbow_y - center_y
            shifted_length = math.hypot(shifted_direction_x, shifted_direction_y)
            arrow_x, arrow_y = _arc_anchor_toward_elements(
                target_elements,
                shifted_direction_x / shifted_length,
                shifted_direction_y / shifted_length,
            )
        else:
            arrow_x, arrow_y = _circle_anchor_toward_point(
                center_x,
                center_y,
                radius,
                elbow_x,
                elbow_y,
            )
    _append_annotation_line(
        parent,
        elbow_x,
        elbow_y,
        arrow_x,
        arrow_y,
        marker_end=marker_id,
    )
    text_anchor = "start" if line_direction > 0 else "end"
    text_x = callout_bounds.min_x if line_direction > 0 else callout_bounds.max_x
    _append_annotation_line(
        parent,
        elbow_x,
        elbow_y,
        callout_bounds.max_x if line_direction > 0 else callout_bounds.min_x,
        elbow_y,
    )
    if thread_label is not None:
        _append_annotation_text(
            parent,
            x=text_x,
            y=elbow_y + _annotation_text_baseline_above_line(),
            value=thread_label,
            anchor=text_anchor,
        )
    _append_annotation_text(
        parent,
        x=text_x,
        y=elbow_y + _annotation_text_baseline_below_line(),
        value=diameter_label,
        anchor=text_anchor,
    )
    return 2.0 * radius, callout_bounds.as_tuple()


def _circle_anchor_toward_point(
    center_x: float,
    center_y: float,
    radius: float,
    point_x: float,
    point_y: float,
) -> tuple[float, float]:
    """Return the rim point on the radial line from a circle to ``point``."""

    delta_x = point_x - center_x
    delta_y = point_y - center_y
    distance = math.hypot(delta_x, delta_y)
    if distance <= 1e-9:
        return center_x + radius, center_y
    return (
        center_x + radius * delta_x / distance,
        center_y + radius * delta_y / distance,
    )


def _estimated_annotation_text_width(value: str) -> float:
    """Conservative width for the fixed 3 mm Arial callout text style."""

    return max(6.0, len(value) * 1.74)


def _circle_diameter_callout_lines(
    value: float,
    *,
    annotation: Mapping[str, object],
    precision: int,
) -> tuple[str, ...]:
    quantity = annotation.get("quantity")
    quantity_prefix = (
        f"{quantity} X " if isinstance(quantity, int) and quantity > 1 else ""
    )
    tolerance = annotation.get("diameter_tolerance")
    diameter_line = f"⌀{_format_dimension_value(value, precision)}"
    if isinstance(tolerance, str):
        diameter_line = f"{diameter_line} {tolerance}"

    thread_size = annotation.get("thread_size")
    depth = annotation.get("depth")
    through = annotation.get("through") is True
    suffix = ""
    if isinstance(depth, (int, float)) and not isinstance(depth, bool):
        suffix = f" ↧ {_format_dimension_value(float(depth), precision)}"
    elif through:
        suffix = " THRU"

    if isinstance(thread_size, str):
        thread_line = f"{quantity_prefix}{thread_size}"
        tolerance_class = annotation.get("thread_tolerance_class")
        if isinstance(tolerance_class, str):
            thread_line = f"{thread_line} - {tolerance_class}"
        return (f"{thread_line}{suffix}", diameter_line)
    return (f"{quantity_prefix}{diameter_line}{suffix}",)


def _circle_diameter_callout_data(
    annotation: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: annotation[key]
        for key in sorted(_CIRCLE_DIAMETER_CALLOUT_KEYS)
        if key in annotation
    }


def _format_annotation_callout_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _place_annotation_bounds(
    bounds: Bounds2D,
    layout_bounds: tuple[float, float, float, float],
    placement: object,
    *,
    default_alignment: Alignment,
) -> Bounds2D:
    """Place an annotation footprint against the visible drawing bounds."""

    operations = _annotation_layout_operations(
        placement,
        default_alignment=default_alignment,
    )
    return align_bounds_sequence_2d(
        bounds,
        Bounds2D(*layout_bounds),
        operations,
    )


def _annotation_layout_operations(
    placement: object,
    *,
    default_alignment: Alignment,
) -> tuple[tuple[Alignment, float], ...]:
    if placement is None:
        return ((default_alignment, 5.0),)
    elif isinstance(placement, Mapping):
        raw_alignments = placement.get("alignments")
        if not isinstance(raw_alignments, Sequence) or isinstance(
            raw_alignments, (str, bytes)
        ):
            raise TypeError(
                "Construction drawing annotation placement requires alignments"
            )
        alignments = raw_alignments
    else:
        raise TypeError("Construction drawing annotation placement must be a mapping")

    operations: list[tuple[Alignment, float]] = []
    for raw_alignment in alignments:
        if not isinstance(raw_alignment, Mapping):
            raise TypeError(
                "Construction drawing annotation alignments must be mappings"
            )
        alignment = Alignment[str(raw_alignment["alignment"])]
        stack_gap = float(raw_alignment.get("stack_gap", 0.0))
        operations.append((alignment, stack_gap))
    return tuple(operations)


_ANNOTATION_TEXT_ASCENT = 2.25
_ANNOTATION_TEXT_DESCENT = 0.75
_ANNOTATION_TEXT_RULE_GAP = 0.8


def _annotation_text_baseline_above_line() -> float:
    """Baseline for text above a rule, leaving a fixed visible-ink gap."""

    return _ANNOTATION_TEXT_DESCENT + _ANNOTATION_TEXT_RULE_GAP


def _annotation_text_baseline_below_line() -> float:
    """Baseline for text below a rule, leaving a fixed visible-ink gap."""

    return -(_ANNOTATION_TEXT_ASCENT + _ANNOTATION_TEXT_RULE_GAP)


def _annotation_text_bounds(
    *,
    x: float,
    y: float,
    value: str,
    anchor: str,
) -> Bounds2D:
    width = _estimated_annotation_text_width(value)
    if anchor == "start":
        min_x, max_x = x, x + width
    elif anchor == "end":
        min_x, max_x = x - width, x
    elif anchor == "middle":
        min_x, max_x = x - width / 2.0, x + width / 2.0
    else:
        raise ValueError(f"Unsupported annotation text anchor {anchor!r}")
    return Bounds2D(
        min_x,
        y - _ANNOTATION_TEXT_DESCENT,
        max_x,
        y + _ANNOTATION_TEXT_ASCENT,
    )


def _merge_bounds_2d(*bounds: Bounds2D) -> Bounds2D:
    return Bounds2D(
        min(item.min_x for item in bounds),
        min(item.min_y for item in bounds),
        max(item.max_x for item in bounds),
        max(item.max_y for item in bounds),
    )


def _append_annotation_line(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    marker_start: str | None = None,
    marker_end: str | None = None,
) -> ET.Element:
    attrs = {
        "x1": _format_number(x1),
        "y1": _format_number(y1),
        "x2": _format_number(x2),
        "y2": _format_number(y2),
        "fill": "none",
        "stroke": "#000000",
        "stroke-width": "0.18",
        "data-shellforgepy-role": "dimension-line",
    }
    if marker_start:
        attrs["marker-start"] = f"url(#{marker_start})"
    if marker_end:
        attrs["marker-end"] = f"url(#{marker_end})"
    return ET.SubElement(parent, _svg_tag("line"), attrs)


def _append_extension_line(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> ET.Element:
    """Emit a thin witness line that stops short of the measured contour."""

    return ET.SubElement(
        parent,
        _svg_tag("line"),
        {
            "x1": _format_number(x1),
            "y1": _format_number(y1),
            "x2": _format_number(x2),
            "y2": _format_number(y2),
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": _format_number(_EXTENSION_LINE_STROKE_WIDTH),
            "data-shellforgepy-role": "dimension-extension",
        },
    )


def _append_annotation_text(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    value: str,
    anchor: str,
) -> ET.Element:
    element = ET.SubElement(
        parent,
        _svg_tag("text"),
        {
            "x": _format_number(x),
            "y": _format_number(-y),
            "transform": "scale(1 -1)",
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": "3",
            "fill": "#000000",
            "text-anchor": anchor,
            "data-shellforgepy-role": "dimension-text",
        },
    )
    element.text = value
    return element


def _ensure_dimension_arrow_marker(root: ET.Element) -> str:
    marker_id = "shellforgepy-dimension-arrow"
    if root.find(f".//{{{SVG_NS}}}marker[@id='{marker_id}']") is not None:
        return marker_id
    defs = ET.Element(_svg_tag("defs"))
    marker = ET.SubElement(
        defs,
        _svg_tag("marker"),
        {
            "id": marker_id,
            "viewBox": "0 0 6 6",
            # Reference the triangle tip, so the line endpoint is the visual
            # arrow tip rather than a point inside the marker body.
            "refX": "6",
            "refY": "3",
            "markerWidth": "2.4",
            "markerHeight": "2.4",
            "orient": "auto-start-reverse",
            "markerUnits": "userSpaceOnUse",
        },
    )
    ET.SubElement(
        marker,
        _svg_tag("path"),
        {"d": "M 0 0 L 6 3 L 0 6 z", "fill": "#000000"},
    )
    root.insert(0, defs)
    return marker_id


def _drawing_elements_bounds(
    elements: Sequence[ET.Element],
) -> tuple[float, float, float, float]:
    bounds: list[tuple[float, float, float, float]] = []
    for element in elements:
        tag = _local_name(element.tag)
        if tag == "line":
            x1 = float(element.attrib["x1"])
            y1 = float(element.attrib["y1"])
            x2 = float(element.attrib["x2"])
            y2 = float(element.attrib["y2"])
            bounds.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        elif tag == "circle":
            cx = float(element.attrib["cx"])
            cy = float(element.attrib["cy"])
            radius = float(element.attrib["r"])
            bounds.append((cx - radius, cy - radius, cx + radius, cy + radius))
        elif tag == "ellipse":
            cx = float(element.attrib["cx"])
            cy = float(element.attrib["cy"])
            radius = max(float(element.attrib["rx"]), float(element.attrib["ry"]))
            bounds.append((cx - radius, cy - radius, cx + radius, cy + radius))
        elif tag == "path" and _circle_or_arc_geometry(element) is not None:
            bounds.append(_arc_bounds(element))
        else:
            raise ValueError(
                f"Construction drawing annotation target has unsupported 2D element {tag!r}"
            )
    if not bounds:
        raise ValueError(
            "Construction drawing annotation target has no 2D section result"
        )
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


# Existing annotation helpers and Stage 0-6 tests use this private name.
_section_elements_bounds = _drawing_elements_bounds


def _circle_or_arc_geometry(
    element: ET.Element,
) -> tuple[float, float, float, bool] | None:
    tag = _local_name(element.tag)
    if tag == "circle":
        return (
            float(element.attrib["cx"]),
            float(element.attrib["cy"]),
            float(element.attrib["r"]),
            False,
        )
    if tag != "path":
        return None
    center = element.attrib.get("data-shellforgepy-center")
    radius = element.attrib.get("data-shellforgepy-radius")
    if center is None or radius is None:
        return None
    center_x, center_y = _parse_point(center)
    return center_x, center_y, float(radius), True


def _circle_or_arc_geometry_from_elements(
    elements: Sequence[ET.Element],
) -> tuple[float, float, float, bool] | None:
    """Recognize one circle, one arc, or one circle split into co-circular arcs.

    OpenCascade can represent the section of one cylindrical drill as several
    adjacent analytic arc edges.  They remain one circular target, not a
    compound feature.  A collection containing any line, circle mix, or arcs
    with distinct centres/radii remains invalid for ``circle_diameter``.
    """

    if len(elements) == 1:
        return _circle_or_arc_geometry(elements[0])
    geometries = [_circle_or_arc_geometry(element) for element in elements]
    if not geometries or any(geometry is None for geometry in geometries):
        return None
    first = geometries[0]
    assert first is not None
    center_x, center_y, radius, is_arc = first
    if not is_arc:
        return None
    for geometry in geometries[1:]:
        assert geometry is not None
        other_x, other_y, other_radius, other_is_arc = geometry
        if (
            not other_is_arc
            or not math.isclose(other_x, center_x, abs_tol=1e-6)
            or not math.isclose(other_y, center_y, abs_tol=1e-6)
            or not math.isclose(other_radius, radius, abs_tol=1e-6)
        ):
            return None
    return center_x, center_y, radius, True


def _arc_bounds(element: ET.Element) -> tuple[float, float, float, float]:
    center_x, center_y, radius, is_arc = _circle_or_arc_geometry(element) or (
        0.0,
        0.0,
        0.0,
        False,
    )
    if not is_arc:
        raise ValueError("Expected an SVG arc")
    start = element.attrib.get("data-shellforgepy-start")
    end = element.attrib.get("data-shellforgepy-end")
    if start is None or end is None:
        return (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
    points = [_parse_point(start), _parse_point(end)]
    start_angle = math.atan2(points[0][1] - center_y, points[0][0] - center_x)
    for cardinal in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        if _angle_is_on_arc(element, start_angle, cardinal):
            points.append(
                (
                    center_x + radius * math.cos(cardinal),
                    center_y + radius * math.sin(cardinal),
                )
            )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _arc_anchor_toward(
    element: ET.Element, unit_x: float, unit_y: float
) -> tuple[float, float]:
    center_x, center_y, radius, _ = _circle_or_arc_geometry(element)  # type: ignore[misc]
    desired_angle = math.atan2(unit_y, unit_x)
    start = element.attrib.get("data-shellforgepy-start")
    if start is not None:
        start_x, start_y = _parse_point(start)
        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        if _angle_is_on_arc(element, start_angle, desired_angle):
            return (
                center_x + radius * math.cos(desired_angle),
                center_y + radius * math.sin(desired_angle),
            )
    candidates = [
        _parse_point(value)
        for value in (
            element.attrib.get("data-shellforgepy-start"),
            element.attrib.get("data-shellforgepy-end"),
        )
        if value
    ]
    if not candidates:
        return center_x + radius * unit_x, center_y + radius * unit_y
    return max(
        candidates,
        key=lambda point: (point[0] - center_x) * unit_x
        + (point[1] - center_y) * unit_y,
    )


def _arc_anchor_toward_elements(
    elements: Sequence[ET.Element], unit_x: float, unit_y: float
) -> tuple[float, float]:
    return max(
        (_arc_anchor_toward(element, unit_x, unit_y) for element in elements),
        key=lambda point: point[0] * unit_x + point[1] * unit_y,
    )


def _angle_is_on_arc(element: ET.Element, start_angle: float, candidate: float) -> bool:
    end = element.attrib.get("data-shellforgepy-end")
    center = element.attrib.get("data-shellforgepy-center")
    if end is None or center is None:
        return True
    center_x, center_y = _parse_point(center)
    end_x, end_y = _parse_point(end)
    end_angle = math.atan2(end_y - center_y, end_x - center_x)
    sweep = element.attrib.get("data-shellforgepy-sweep", "0") == "1"
    large_arc = element.attrib.get("data-shellforgepy-large-arc", "0") == "1"
    raw_span = (end_angle - start_angle) % (2.0 * math.pi)
    if not sweep:
        raw_span = (start_angle - end_angle) % (2.0 * math.pi)
    span = raw_span
    if large_arc and span < math.pi:
        span = 2.0 * math.pi - span
    elif not large_arc and span > math.pi:
        span = 2.0 * math.pi - span
    delta = (candidate - start_angle) % (2.0 * math.pi)
    if not sweep:
        delta = (start_angle - candidate) % (2.0 * math.pi)
    return delta <= span + 1e-9


def _parse_point(value: str) -> tuple[float, float]:
    x, y = value.split(",", 1)
    return float(x), float(y)


def _format_dimension_value(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _format_bounds(bounds: tuple[float, float, float, float]) -> str:
    return ",".join(_format_number(value) for value in bounds)


def _parse_bounds(value: str) -> tuple[float, float, float, float]:
    parsed = tuple(float(item) for item in value.split(","))
    if len(parsed) != 4:
        raise ValueError(f"Expected four drawing bounds values; got {value!r}")
    return parsed  # type: ignore[return-value]


def _effective_scale_label(request: Mapping[str, object]) -> str:
    scale = float(request.get("_effective_scale", request.get("scale", 1.0)))
    return f"{_scale_ratio(scale)} ({_scale_equivalence(scale)})"


def _drawing_scale_metadata(root: ET.Element) -> dict[str, object]:
    return {
        "effective_scale": float(root.attrib["data-shellforgepy-scale"]),
        "scale_ratio": root.attrib.get("data-shellforgepy-scale-ratio", "1:1"),
        "scale_equivalence": root.attrib.get(
            "data-shellforgepy-scale-equivalence", "1 mm = 1 mm"
        ),
    }


def _update_technical_sheet_scale(
    tree: ET.ElementTree,
    *,
    request: Mapping[str, object],
) -> None:
    """Apply the shared discrete scale after all annotation footprints exist."""

    sheet = request.get("sheet")
    if sheet is None:
        return
    sheet_spec = _normalize_sheet_spec(sheet)
    root = tree.getroot()
    geometry = root.find(f"./{{{SVG_NS}}}g[@id='shellforgepy-geometry']")
    if geometry is None:
        return
    content_bounds = _drawing_tree_content_bounds(root, geometry)
    drawing_bounds = (
        content_bounds.min_x,
        content_bounds.min_y,
        content_bounds.max_x - content_bounds.min_x,
        content_bounds.max_y - content_bounds.min_y,
    )
    scale = _select_discrete_sheet_scale(
        sheet_spec,
        drawing_bounds=drawing_bounds,
        requested_scale=float(request.get("scale", 1.0)),
    )
    transform = _sheet_geometry_transform(
        sheet_spec,
        drawing_bounds=drawing_bounds,
        requested_scale=float(request.get("scale", 1.0)),
        effective_scale=scale,
    )
    geometry.attrib["transform"] = transform
    annotations = root.find(f"./{{{SVG_NS}}}g[@id='shellforgepy-annotations']")
    if annotations is not None:
        annotations.attrib["transform"] = transform
    root.attrib["data-shellforgepy-scale"] = _format_number(scale)
    root.attrib["data-shellforgepy-scale-ratio"] = _scale_ratio(scale)
    root.attrib["data-shellforgepy-scale-equivalence"] = _scale_equivalence(scale)
    effective_request = dict(request)
    effective_request["_effective_scale"] = scale
    for text in root.findall(f".//{{{SVG_NS}}}text"):
        role = text.attrib.get("data-shellforgepy-role")
        if role == "metadata-text" and (text.text or "").startswith("UNITS:"):
            text.text = (
                f"UNITS: {request.get('units', 'mm')}   "
                f"SCALE: {_effective_scale_label(effective_request)}"
            )
        elif role == "title-block-field" and (text.text or "").startswith("SCALE:"):
            text.text = f"SCALE: {_effective_scale_label(effective_request)}"


def _drawing_tree_content_bounds(root: ET.Element, geometry: ET.Element) -> Bounds2D:
    elements = [
        element
        for element in geometry.iter()
        if _local_name(element.tag) in {"line", "circle", "path"}
    ]
    bounds = [Bounds2D(*_section_elements_bounds(elements))]
    for annotation in root.findall(
        f".//{{{SVG_NS}}}g[@data-shellforgepy-role='dimension']"
    ):
        placed = annotation.attrib.get("data-shellforgepy-placed-bounds")
        if placed is not None:
            bounds.append(Bounds2D(*_parse_bounds(placed)))
    return _merge_bounds_2d(*bounds)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def append_part_group(
    parent: ET.Element,
    *,
    part_identity: str,
    role: str = "section-contour",
    source: str | None = None,
    exact: bool | None = None,
    representation: str | None = None,
) -> ET.Element:
    """Append the semantic group that will contain one selected part's edges."""

    attrs = {
        "id": _safe_svg_id(f"part-{part_identity}"),
        "data-shellforgepy-role": role,
        "data-shellforgepy-part": part_identity,
    }
    if source is not None:
        attrs["data-shellforgepy-source"] = source
    if exact is not None:
        attrs["data-shellforgepy-geometry"] = "exact" if exact else "approximate"
    if representation is not None:
        attrs["data-shellforgepy-representation"] = representation
    return ET.SubElement(parent, _svg_tag("g"), attrs)


def serialize_svg(tree: ET.ElementTree) -> bytes:
    """Serialize the canonical SVG tree deterministically as UTF-8 XML."""

    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def _normalize_sheet_spec(sheet: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(sheet, Mapping):
        raise TypeError("Technical drawing sheet must be a mapping")
    allowed_keys = {
        "format",
        "orientation",
        "margin",
        "border",
        "title_block",
        "width",
        "height",
        "drawing_margin",
    }
    unknown = set(sheet) - allowed_keys
    if unknown:
        raise ValueError(
            f"Technical drawing sheet has unsupported keys: {sorted(unknown)!r}"
        )
    sheet_format = str(sheet.get("format", "A4")).upper()
    if sheet_format != "A4":
        raise ValueError("Technical drawing sheets currently support format='A4' only")
    orientation = str(sheet.get("orientation", "landscape")).lower()
    if orientation not in {"landscape", "portrait"}:
        raise ValueError(
            "Technical drawing sheet orientation must be landscape or portrait"
        )
    margin = float(sheet.get("margin", 10.0))
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Technical drawing sheet margin must be positive and finite")
    border = sheet.get("border", True)
    if not isinstance(border, bool):
        raise TypeError("Technical drawing sheet border must be boolean")
    drawing_margin = float(sheet.get("drawing_margin", 5.0))
    if not math.isfinite(drawing_margin) or drawing_margin < 0:
        raise ValueError(
            "Technical drawing sheet drawing_margin must be non-negative and finite"
        )
    title_block = sheet.get("title_block", {})
    if not isinstance(title_block, Mapping):
        raise TypeError("Technical drawing sheet title_block must be a mapping")
    normalized_title_block = {
        str(key): str(value) for key, value in title_block.items()
    }
    if orientation == "landscape":
        width, height = 297.0, 210.0
    else:
        width, height = 210.0, 297.0
    return {
        "format": sheet_format,
        "orientation": orientation,
        "margin": margin,
        "border": border,
        "drawing_margin": drawing_margin,
        "title_block": normalized_title_block,
        "width": width,
        "height": height,
    }


def _append_technical_sheet(
    root: ET.Element,
    sheet: Mapping[str, object],
    *,
    request: Mapping[str, object],
) -> ET.Element:
    width = float(sheet["width"])
    height = float(sheet["height"])
    margin = float(sheet["margin"])
    title_block = dict(sheet["title_block"])
    group = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": "shellforgepy-sheet",
            "data-shellforgepy-role": "sheet",
            "data-shellforgepy-format": str(sheet["format"]),
            "data-shellforgepy-orientation": str(sheet["orientation"]),
        },
    )
    if bool(sheet["border"]):
        _append_sheet_rect(
            group,
            x=margin / 2.0,
            y=margin / 2.0,
            width=width - margin,
            height=height - margin,
            role="outer-border",
            stroke_width=0.7,
        )
        _append_sheet_rect(
            group,
            x=margin,
            y=margin,
            width=width - 2.0 * margin,
            height=height - 2.0 * margin,
            role="inner-border",
            stroke_width=0.25,
        )

    viewport = _technical_viewport(sheet)
    _append_sheet_rect(
        group,
        x=viewport[0],
        y=viewport[1],
        width=viewport[2],
        height=viewport[3],
        role="drawing-viewport",
        stroke_width=0.18,
    )
    _append_sheet_text(
        group,
        x=viewport[0],
        y=viewport[1] - 2.5,
        value=f"VIEW: {_view_metadata(request.get('view')).upper()}",
        role="view-label",
        size=3.2,
        anchor="start",
    )

    info_x = margin + 4.0
    info_y = margin + 4.0
    info_width = min(82.0, viewport[2] * 0.32)
    info_height = 16.0
    _append_sheet_rect(
        group,
        x=info_x,
        y=info_y,
        width=info_width,
        height=info_height,
        role="metadata-frame",
        stroke_width=0.18,
    )
    _append_sheet_text(
        group,
        x=info_x + 2.0,
        y=info_y + 4.5,
        value="CONSTRUCTION DRAWING",
        role="metadata-heading",
        size=3.0,
        anchor="start",
    )
    _append_sheet_text(
        group,
        x=info_x + 2.0,
        y=info_y + 9.0,
        value=(
            f"UNITS: {request.get('units', 'mm')}   "
            f"SCALE: {_effective_scale_label(request)}"
        ),
        role="metadata-text",
        size=2.7,
        anchor="start",
    )
    _append_sheet_text(
        group,
        x=info_x + 2.0,
        y=info_y + 13.0,
        value=f"SECTION: {_view_metadata(request.get('view')).upper()}",
        role="metadata-text",
        size=2.7,
        anchor="start",
    )

    block_width = min(135.0, width - 2.0 * margin - 4.0)
    block_height = 38.0
    block_x = width - margin - block_width
    block_y = height - margin - block_height
    _append_sheet_title_block(
        group,
        x=block_x,
        y=block_y,
        width=block_width,
        height=block_height,
        title_block=title_block,
        request=request,
    )
    return group


def _technical_viewport(
    sheet: Mapping[str, object],
) -> tuple[float, float, float, float]:
    width = float(sheet["width"])
    height = float(sheet["height"])
    margin = float(sheet["margin"])
    title_block_height = 38.0
    x = margin + 4.0
    y = margin + 28.0
    viewport_width = width - 2.0 * margin - 8.0
    viewport_height = height - 2.0 * margin - title_block_height - 34.0
    return (x, y, viewport_width, viewport_height)


def _technical_content_viewport(
    sheet: Mapping[str, object],
) -> tuple[float, float, float, float]:
    """Return the white-margin inset where drawing content may be placed."""

    x, y, width, height = _technical_viewport(sheet)
    drawing_margin = float(sheet["drawing_margin"])
    content_width = width - 2.0 * drawing_margin
    content_height = height - 2.0 * drawing_margin
    if content_width <= 0 or content_height <= 0:
        raise ValueError(
            "Technical drawing sheet drawing_margin leaves no content area"
        )
    return (
        x + drawing_margin,
        y + drawing_margin,
        content_width,
        content_height,
    )


def _sheet_geometry_transform(
    sheet: Mapping[str, object],
    *,
    drawing_bounds: DrawingBounds,
    requested_scale: float,
    effective_scale: float | None = None,
) -> str:
    min_x, min_y, drawing_width, drawing_height = drawing_bounds
    viewport_x, viewport_y, viewport_width, viewport_height = (
        _technical_content_viewport(sheet)
    )
    scale = effective_scale
    if scale is None:
        scale = _select_discrete_sheet_scale(
            sheet,
            drawing_bounds=drawing_bounds,
            requested_scale=requested_scale,
        )
    offset_x = viewport_x + (viewport_width - drawing_width * scale) / 2.0
    offset_y = viewport_y + (viewport_height - drawing_height * scale) / 2.0
    return (
        f"translate({_format_number(offset_x - scale * min_x)} "
        f"{_format_number(offset_y + scale * (min_y + drawing_height))}) "
        f"scale({_format_number(scale)} {_format_number(-scale)})"
    )


def _select_discrete_sheet_scale(
    sheet: Mapping[str, object],
    *,
    drawing_bounds: DrawingBounds,
    requested_scale: float,
) -> float:
    """Select the largest fitting reduction from the preferred 1-2-5 series."""

    _, _, drawing_width, drawing_height = drawing_bounds
    viewport = _technical_content_viewport(sheet)
    fit_scale = min(viewport[2] / drawing_width, viewport[3] / drawing_height)
    limit = min(float(requested_scale), fit_scale)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Construction drawing cannot fit the technical sheet")
    multiplier = 1
    while True:
        for base in (1, 2, 5):
            denominator = base * multiplier
            scale = 1.0 / denominator
            if scale <= limit + 1e-12:
                return scale
        multiplier *= 10


def _scale_ratio(scale: float) -> str:
    denominator = round(1.0 / scale)
    return f"1:{denominator}"


def _scale_equivalence(scale: float) -> str:
    denominator = round(1.0 / scale)
    return f"1 mm = {denominator} mm"


def _append_sheet_rect(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    role: str,
    stroke_width: float,
) -> ET.Element:
    return ET.SubElement(
        parent,
        _svg_tag("rect"),
        {
            "x": _format_number(x),
            "y": _format_number(y),
            "width": _format_number(width),
            "height": _format_number(height),
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": _format_number(stroke_width),
            "data-shellforgepy-role": role,
        },
    )


def _append_sheet_text(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    value: str,
    role: str,
    size: float,
    anchor: str,
) -> ET.Element:
    element = ET.SubElement(
        parent,
        _svg_tag("text"),
        {
            "x": _format_number(x),
            "y": _format_number(y),
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": _format_number(size),
            "fill": "#000000",
            "text-anchor": anchor,
            "data-shellforgepy-role": role,
        },
    )
    element.text = value
    return element


def _append_sheet_title_block(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title_block: Mapping[str, str],
    request: Mapping[str, object],
) -> ET.Element:
    group = ET.SubElement(
        parent,
        _svg_tag("g"),
        {"data-shellforgepy-role": "title-block"},
    )
    _append_sheet_rect(
        group,
        x=x,
        y=y,
        width=width,
        height=height,
        role="title-block-frame",
        stroke_width=0.35,
    )
    title = title_block.get("title", str(request.get("name", "Construction drawing")))
    drawing_number = title_block.get("drawing_number", str(request.get("name", "-")))
    revision = title_block.get("revision", "-")
    material = title_block.get("material", "-")
    units = title_block.get("units", str(request.get("units", "mm")))
    scale = _effective_scale_label(request)
    source = _truncate_sheet_value(
        title_block.get("source", str(request.get("source_assembly", "-"))),
        max_chars=24,
    )

    title_height = 10.0
    _append_sheet_line(
        group, x1=x, y1=y + title_height, x2=x + width, y2=y + title_height
    )
    _append_sheet_text(
        group,
        x=x + 2.0,
        y=y + 6.5,
        value=title,
        role="title-block-title",
        size=4.0,
        anchor="start",
    )
    row_height = (height - title_height) / 3.0
    labels = (
        ("DRAWING", drawing_number, "REVISION", revision),
        ("MATERIAL", material, "UNITS", units),
        ("SCALE", scale, "SOURCE", source),
    )
    split_x = x + width * 0.64
    _append_sheet_line(
        group,
        x1=split_x,
        y1=y + title_height,
        x2=split_x,
        y2=y + height,
    )
    for index, (left_label, left_value, right_label, right_value) in enumerate(labels):
        row_y = y + title_height + row_height * index
        if index:
            _append_sheet_line(
                group,
                x1=x,
                y1=row_y,
                x2=x + width,
                y2=row_y,
            )
        _append_sheet_text(
            group,
            x=x + 1.5,
            y=row_y + 3.7,
            value=f"{left_label}: {left_value}",
            role="title-block-field",
            size=2.5,
            anchor="start",
        )
        _append_sheet_text(
            group,
            x=split_x + 1.5,
            y=row_y + 3.7,
            value=f"{right_label}: {right_value}",
            role="title-block-field",
            size=2.5,
            anchor="start",
        )
    return group


def _append_sheet_line(
    parent: ET.Element,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> ET.Element:
    return ET.SubElement(
        parent,
        _svg_tag("line"),
        {
            "x1": _format_number(x1),
            "y1": _format_number(y1),
            "x2": _format_number(x2),
            "y2": _format_number(y2),
            "stroke": "#000000",
            "stroke-width": "0.18",
            "data-shellforgepy-role": "sheet-frame-line",
        },
    )


def _truncate_sheet_value(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _validate_view_spec(view: object) -> None:
    if isinstance(view, str):
        if view not in _VIEW_AXES:
            raise ValueError(
                f"Unsupported section view {view!r}; expected one of "
                f"{SUPPORTED_VIEW_PRESETS!r}"
            )
        return
    if not isinstance(view, Mapping):
        raise TypeError("view must be a named preset or a mapping")
    unknown = set(view) - set(_VECTOR_KEYS)
    if unknown:
        raise ValueError(
            f"Explicit view contains unsupported keys: {sorted(unknown)!r}"
        )
    for key in ("normal", "up"):
        _vector_from_mapping(view, key)
    if "origin" in view:
        _vector_from_mapping(view, "origin")


def _validate_nonnegative(name: str, value: object) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _vector_from_mapping(mapping: Mapping[str, object], key: str) -> Vector3:
    value = mapping.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"View {key!r} must be a 3-element coordinate sequence")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"View {key!r} must contain finite coordinates")
    return result  # type: ignore[return-value]


def _normalize(vector: Vector3, label: str) -> Vector3:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-12:
        raise ValueError(f"View {label} vector must not be zero length")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _normalize_model_bounds(bounds: ModelBounds) -> ModelBounds:
    if len(bounds) != 2 or len(bounds[0]) != 3 or len(bounds[1]) != 3:
        raise ValueError("Model bounds must contain two 3-element points")
    minimum = tuple(float(component) for component in bounds[0])
    maximum = tuple(float(component) for component in bounds[1])
    if any(not math.isfinite(component) for component in (*minimum, *maximum)):
        raise ValueError("Model bounds must contain finite coordinates")
    if any(minimum[index] > maximum[index] for index in range(3)):
        raise ValueError("Model bounds minimum must not exceed maximum")
    return minimum, maximum  # type: ignore[return-value]


def _normalize_drawing_bounds(bounds: DrawingBounds) -> DrawingBounds:
    if len(bounds) != 4:
        raise ValueError("Drawing bounds must be (min_x, min_y, width, height)")
    normalized = tuple(float(value) for value in bounds)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Drawing bounds must contain finite values")
    return normalized  # type: ignore[return-value]


def _bounds_center(bounds: ModelBounds) -> Vector3:
    return tuple(
        (bounds[0][index] + bounds[1][index]) / 2 for index in range(3)
    )  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return format(value, ".12g")


def _format_vector(vector: Sequence[float]) -> str:
    return ",".join(_format_number(float(component)) for component in vector)


def _format_view_box(bounds: DrawingBounds) -> str:
    return " ".join(_format_number(value) for value in bounds)


def _view_metadata(view: object) -> str:
    if isinstance(view, str):
        return view
    if isinstance(view, Mapping):
        return "explicit"
    return DEFAULT_SECTION_VIEW


def _svg_y_up_transform(min_y: float, height: float) -> str:
    return f"translate(0 {_format_number(2 * min_y + height)}) scale(1 -1)"


def _safe_svg_id(value: str) -> str:
    safe = _SAFE_ID_RE.sub("-", value).strip("-") or "construction-drawing"
    if safe[0].isdigit():
        safe = f"id-{safe}"
    return safe
