"""Small, CAD-free 2D layout helpers for construction drawings.

The operations intentionally mirror the planar subset of ShellForgePy's
``Alignment`` vocabulary.  They work on already-projected drawing bounds, so
they can place annotation footprints without importing geometry adapters or
SVG rendering code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shellforgepy.construct.alignment import Alignment


@dataclass(frozen=True)
class Bounds2D:
    """Axis-aligned bounds in drawing coordinates (positive Y is drawing-up)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        values = (self.min_x, self.min_y, self.max_x, self.max_y)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("2D layout bounds must be finite")
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("2D layout bounds must have ordered minima and maxima")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    def translated(self, dx: float, dy: float) -> "Bounds2D":
        return Bounds2D(
            self.min_x + dx,
            self.min_y + dy,
            self.max_x + dx,
            self.max_y + dy,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)


PLANAR_ALIGNMENTS = frozenset(
    {
        Alignment.LEFT,
        Alignment.RIGHT,
        Alignment.FRONT,
        Alignment.BACK,
        Alignment.CENTER,
        Alignment.EDGE_LEFT,
        Alignment.EDGE_RIGHT,
        Alignment.EDGE_FRONT,
        Alignment.EDGE_BACK,
        Alignment.STACK_LEFT,
        Alignment.STACK_RIGHT,
        Alignment.STACK_FRONT,
        Alignment.STACK_BACK,
    }
)


def alignment_delta_2d(
    moving: Bounds2D,
    target: Bounds2D,
    *,
    alignment: Alignment,
    stack_gap: float = 0.0,
) -> tuple[float, float]:
    """Return the translation for one planar ShellForgePy alignment.

    ``FRONT`` is drawing-down and ``BACK`` is drawing-up.  ``STACK_*`` keeps
    the same touching-face and additional-gap semantics as 3D alignment.
    """

    if alignment not in PLANAR_ALIGNMENTS:
        raise ValueError(f"Unsupported 2D drawing alignment Alignment.{alignment.name}")
    if not math.isfinite(stack_gap):
        raise ValueError("2D layout stack_gap must be finite")

    if alignment == Alignment.LEFT:
        return target.min_x - moving.min_x, 0.0
    if alignment == Alignment.RIGHT:
        return target.max_x - moving.max_x, 0.0
    if alignment == Alignment.FRONT:
        return 0.0, target.min_y - moving.min_y
    if alignment == Alignment.BACK:
        return 0.0, target.max_y - moving.max_y
    if alignment == Alignment.CENTER:
        return target.center_x - moving.center_x, target.center_y - moving.center_y
    if alignment == Alignment.EDGE_LEFT:
        return target.min_x - moving.center_x, 0.0
    if alignment == Alignment.EDGE_RIGHT:
        return target.max_x - moving.center_x, 0.0
    if alignment == Alignment.EDGE_FRONT:
        return 0.0, target.min_y - moving.center_y
    if alignment == Alignment.EDGE_BACK:
        return 0.0, target.max_y - moving.center_y
    if alignment == Alignment.STACK_LEFT:
        return target.min_x - moving.max_x - stack_gap, 0.0
    if alignment == Alignment.STACK_RIGHT:
        return target.max_x - moving.min_x + stack_gap, 0.0
    if alignment == Alignment.STACK_FRONT:
        return 0.0, target.min_y - moving.max_y - stack_gap
    if alignment == Alignment.STACK_BACK:
        return 0.0, target.max_y - moving.min_y + stack_gap
    raise AssertionError(f"Unhandled 2D drawing alignment {alignment!r}")


def align_bounds_2d(
    moving: Bounds2D,
    target: Bounds2D,
    *,
    alignment: Alignment,
    stack_gap: float = 0.0,
) -> Bounds2D:
    """Return ``moving`` translated for one planar alignment operation."""

    dx, dy = alignment_delta_2d(
        moving,
        target,
        alignment=alignment,
        stack_gap=stack_gap,
    )
    return moving.translated(dx, dy)


def align_bounds_sequence_2d(
    moving: Bounds2D,
    target: Bounds2D,
    operations: Sequence[tuple[Alignment, float]],
) -> Bounds2D:
    """Apply an ordered sequence of planar alignment operations."""

    placed = moving
    for alignment, stack_gap in operations:
        placed = align_bounds_2d(
            placed,
            target,
            alignment=alignment,
            stack_gap=stack_gap,
        )
    return placed
