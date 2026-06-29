"""Dependency-free straight-line glyphs for fast vector text creation."""

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

Point2D = tuple[float, float]
Segment2D = tuple[Point2D, Point2D]
StrokePolygon2D = tuple[Point2D, ...]

DEFAULT_STROKE_WIDTH_RATIO = 0.08
STROKE_CAP_ARC_SEGMENTS = 8
SPACE_ADVANCE = 0.55
LINE_ADVANCE = 1.25
PUNCTUATION_CHARACTERS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


@dataclass(frozen=True)
class VectorGlyph:
    """A normalized single-line glyph definition."""

    advance: float
    strokes: tuple[Segment2D, ...]


@dataclass(frozen=True)
class VectorTextSegment:
    """A scaled centerline segment ready for adapter materialization."""

    start: Point2D
    end: Point2D


@dataclass(frozen=True)
class VectorTextLayout:
    """Backend-free vector text layout result."""

    text: str
    size: float
    stroke_width: float
    segments: tuple[VectorTextSegment, ...]


@dataclass(frozen=True)
class _VectorGlyphPlacement:
    char: str
    origin: Point2D


VECTOR_GLYPHS: dict[str, VectorGlyph] = {
    "A": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.35, 1.00)),
            ((0.70, 0.00), (0.35, 1.00)),
            ((0.15, 0.50), (0.55, 0.50)),
        ),
    ),
    "B": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.62)),
            ((0.70, 0.62), (0.55, 0.50)),
            ((0.00, 0.50), (0.55, 0.50)),
            ((0.55, 0.50), (0.70, 0.38)),
            ((0.70, 0.38), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.00, 0.00)),
        ),
    ),
    "C": VectorGlyph(
        0.90,
        (
            ((0.70, 1.00), (0.10, 1.00)),
            ((0.10, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.15)),
            ((0.00, 0.15), (0.10, 0.00)),
            ((0.10, 0.00), (0.70, 0.00)),
        ),
    ),
    "D": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.50, 1.00)),
            ((0.50, 1.00), (0.70, 0.80)),
            ((0.70, 0.80), (0.70, 0.20)),
            ((0.70, 0.20), (0.50, 0.00)),
            ((0.50, 0.00), (0.00, 0.00)),
        ),
    ),
    "E": VectorGlyph(
        0.85,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.70, 1.00)),
            ((0.00, 0.50), (0.55, 0.50)),
            ((0.00, 0.00), (0.70, 0.00)),
        ),
    ),
    "F": VectorGlyph(
        0.85,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.70, 1.00)),
            ((0.00, 0.50), (0.55, 0.50)),
        ),
    ),
    "G": VectorGlyph(
        0.95,
        (
            ((0.70, 1.00), (0.10, 1.00)),
            ((0.10, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.15)),
            ((0.00, 0.15), (0.10, 0.00)),
            ((0.10, 0.00), (0.70, 0.00)),
            ((0.70, 0.00), (0.70, 0.45)),
            ((0.70, 0.45), (0.40, 0.45)),
        ),
    ),
    "H": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.70, 0.00), (0.70, 1.00)),
            ((0.00, 0.50), (0.70, 0.50)),
        ),
    ),
    "I": VectorGlyph(
        0.65,
        (
            ((0.00, 1.00), (0.50, 1.00)),
            ((0.25, 1.00), (0.25, 0.00)),
            ((0.00, 0.00), (0.50, 0.00)),
        ),
    ),
    "J": VectorGlyph(
        0.80,
        (
            ((0.00, 1.00), (0.65, 1.00)),
            ((0.50, 1.00), (0.50, 0.15)),
            ((0.50, 0.15), (0.35, 0.00)),
            ((0.35, 0.00), (0.10, 0.00)),
            ((0.10, 0.00), (0.00, 0.20)),
        ),
    ),
    "K": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.70, 1.00), (0.00, 0.50)),
            ((0.00, 0.50), (0.70, 0.00)),
        ),
    ),
    "L": VectorGlyph(
        0.80,
        (
            ((0.00, 1.00), (0.00, 0.00)),
            ((0.00, 0.00), (0.70, 0.00)),
        ),
    ),
    "M": VectorGlyph(
        1.05,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.45, 0.45)),
            ((0.45, 0.45), (0.90, 1.00)),
            ((0.90, 1.00), (0.90, 0.00)),
        ),
    ),
    "N": VectorGlyph(
        0.95,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.75, 0.00)),
            ((0.75, 0.00), (0.75, 1.00)),
        ),
    ),
    "O": VectorGlyph(
        0.95,
        (
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 0.85)),
            ((0.70, 0.85), (0.55, 1.00)),
            ((0.55, 1.00), (0.15, 1.00)),
            ((0.15, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
        ),
    ),
    "P": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.60)),
            ((0.70, 0.60), (0.55, 0.50)),
            ((0.55, 0.50), (0.00, 0.50)),
        ),
    ),
    "Q": VectorGlyph(
        0.95,
        (
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 0.85)),
            ((0.70, 0.85), (0.55, 1.00)),
            ((0.55, 1.00), (0.15, 1.00)),
            ((0.15, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
            ((0.42, 0.25), (0.78, -0.10)),
        ),
    ),
    "R": VectorGlyph(
        0.90,
        (
            ((0.00, 0.00), (0.00, 1.00)),
            ((0.00, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.60)),
            ((0.70, 0.60), (0.55, 0.50)),
            ((0.55, 0.50), (0.00, 0.50)),
            ((0.25, 0.50), (0.75, 0.00)),
        ),
    ),
    "S": VectorGlyph(
        0.90,
        (
            ((0.70, 1.00), (0.10, 1.00)),
            ((0.10, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.60)),
            ((0.00, 0.60), (0.15, 0.50)),
            ((0.15, 0.50), (0.55, 0.50)),
            ((0.55, 0.50), (0.70, 0.38)),
            ((0.70, 0.38), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.00, 0.00)),
        ),
    ),
    "T": VectorGlyph(
        0.85,
        (
            ((0.00, 1.00), (0.70, 1.00)),
            ((0.35, 1.00), (0.35, 0.00)),
        ),
    ),
    "U": VectorGlyph(
        0.95,
        (
            ((0.00, 1.00), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 1.00)),
        ),
    ),
    "V": VectorGlyph(
        0.90,
        (
            ((0.00, 1.00), (0.35, 0.00)),
            ((0.35, 0.00), (0.70, 1.00)),
        ),
    ),
    "W": VectorGlyph(
        1.10,
        (
            ((0.00, 1.00), (0.20, 0.00)),
            ((0.20, 0.00), (0.45, 0.55)),
            ((0.45, 0.55), (0.70, 0.00)),
            ((0.70, 0.00), (0.90, 1.00)),
        ),
    ),
    "X": VectorGlyph(
        0.90,
        (
            ((0.00, 1.00), (0.70, 0.00)),
            ((0.70, 1.00), (0.00, 0.00)),
        ),
    ),
    "Y": VectorGlyph(
        0.90,
        (
            ((0.00, 1.00), (0.35, 0.55)),
            ((0.70, 1.00), (0.35, 0.55)),
            ((0.35, 0.55), (0.35, 0.00)),
        ),
    ),
    "Z": VectorGlyph(
        0.90,
        (
            ((0.00, 1.00), (0.70, 1.00)),
            ((0.70, 1.00), (0.00, 0.00)),
            ((0.00, 0.00), (0.70, 0.00)),
        ),
    ),
    "0": VectorGlyph(
        0.90,
        (
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 0.85)),
            ((0.70, 0.85), (0.55, 1.00)),
            ((0.55, 1.00), (0.15, 1.00)),
            ((0.15, 1.00), (0.00, 0.85)),
            ((0.00, 0.85), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
            ((0.18, 0.15), (0.52, 0.85)),
        ),
    ),
    "1": VectorGlyph(
        0.65,
        (
            ((0.10, 0.80), (0.30, 1.00)),
            ((0.30, 1.00), (0.30, 0.00)),
            ((0.05, 0.00), (0.55, 0.00)),
        ),
    ),
    "2": VectorGlyph(
        0.90,
        (
            ((0.00, 0.85), (0.15, 1.00)),
            ((0.15, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.65)),
            ((0.70, 0.65), (0.00, 0.00)),
            ((0.00, 0.00), (0.70, 0.00)),
        ),
    ),
    "3": VectorGlyph(
        0.90,
        (
            ((0.00, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.60)),
            ((0.70, 0.60), (0.55, 0.50)),
            ((0.55, 0.50), (0.25, 0.50)),
            ((0.55, 0.50), (0.70, 0.38)),
            ((0.70, 0.38), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.00, 0.00)),
        ),
    ),
    "4": VectorGlyph(
        0.90,
        (
            ((0.62, 1.00), (0.62, 0.00)),
            ((0.08, 0.42), (0.72, 0.42)),
            ((0.08, 0.42), (0.62, 1.00)),
        ),
    ),
    "5": VectorGlyph(
        0.90,
        (
            ((0.70, 1.00), (0.05, 1.00)),
            ((0.05, 1.00), (0.00, 0.52)),
            ((0.00, 0.52), (0.55, 0.52)),
            ((0.55, 0.52), (0.70, 0.38)),
            ((0.70, 0.38), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.00, 0.00)),
        ),
    ),
    "6": VectorGlyph(
        0.90,
        (
            ((0.65, 1.00), (0.15, 1.00)),
            ((0.15, 1.00), (0.00, 0.75)),
            ((0.00, 0.75), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 0.38)),
            ((0.70, 0.38), (0.55, 0.50)),
            ((0.55, 0.50), (0.00, 0.50)),
        ),
    ),
    "7": VectorGlyph(
        0.85,
        (
            ((0.00, 1.00), (0.70, 1.00)),
            ((0.70, 1.00), (0.15, 0.00)),
        ),
    ),
    "8": VectorGlyph(
        0.90,
        (
            ((0.15, 0.00), (0.55, 0.00)),
            ((0.55, 0.00), (0.70, 0.15)),
            ((0.70, 0.15), (0.70, 0.38)),
            ((0.70, 0.38), (0.55, 0.50)),
            ((0.55, 0.50), (0.15, 0.50)),
            ((0.15, 0.50), (0.00, 0.38)),
            ((0.00, 0.38), (0.00, 0.15)),
            ((0.00, 0.15), (0.15, 0.00)),
            ((0.15, 0.50), (0.00, 0.62)),
            ((0.00, 0.62), (0.00, 0.85)),
            ((0.00, 0.85), (0.15, 1.00)),
            ((0.15, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.62)),
            ((0.70, 0.62), (0.55, 0.50)),
        ),
    ),
    "9": VectorGlyph(
        0.90,
        (
            ((0.70, 0.50), (0.15, 0.50)),
            ((0.15, 0.50), (0.00, 0.62)),
            ((0.00, 0.62), (0.00, 0.85)),
            ((0.00, 0.85), (0.15, 1.00)),
            ((0.15, 1.00), (0.55, 1.00)),
            ((0.55, 1.00), (0.70, 0.85)),
            ((0.70, 0.85), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.05, 0.00)),
        ),
    ),
    "!": VectorGlyph(
        0.45,
        (
            ((0.20, 1.00), (0.20, 0.30)),
            ((0.14, 0.05), (0.26, 0.05)),
        ),
    ),
    '"': VectorGlyph(
        0.55,
        (
            ((0.12, 1.00), (0.12, 0.70)),
            ((0.38, 1.00), (0.38, 0.70)),
        ),
    ),
    "#": VectorGlyph(
        0.90,
        (
            ((0.20, 0.00), (0.35, 1.00)),
            ((0.50, 0.00), (0.65, 1.00)),
            ((0.05, 0.35), (0.75, 0.35)),
            ((0.00, 0.65), (0.70, 0.65)),
        ),
    ),
    "$": VectorGlyph(
        0.90,
        (
            ((0.35, 1.08), (0.35, -0.08)),
            ((0.70, 1.00), (0.10, 1.00)),
            ((0.10, 1.00), (0.00, 0.82)),
            ((0.00, 0.82), (0.00, 0.62)),
            ((0.00, 0.62), (0.15, 0.50)),
            ((0.15, 0.50), (0.55, 0.50)),
            ((0.55, 0.50), (0.70, 0.38)),
            ((0.70, 0.38), (0.70, 0.15)),
            ((0.70, 0.15), (0.55, 0.00)),
            ((0.55, 0.00), (0.00, 0.00)),
        ),
    ),
    "%": VectorGlyph(
        1.00,
        (
            ((0.05, 0.00), (0.75, 1.00)),
            ((0.08, 0.78), (0.20, 0.90)),
            ((0.20, 0.90), (0.32, 0.78)),
            ((0.32, 0.78), (0.20, 0.66)),
            ((0.20, 0.66), (0.08, 0.78)),
            ((0.48, 0.22), (0.60, 0.34)),
            ((0.60, 0.34), (0.72, 0.22)),
            ((0.72, 0.22), (0.60, 0.10)),
            ((0.60, 0.10), (0.48, 0.22)),
        ),
    ),
    "&": VectorGlyph(
        0.95,
        (
            ((0.60, 0.00), (0.20, 0.45)),
            ((0.20, 0.45), (0.12, 0.65)),
            ((0.12, 0.65), (0.25, 0.85)),
            ((0.25, 0.85), (0.45, 0.85)),
            ((0.45, 0.85), (0.55, 0.70)),
            ((0.55, 0.70), (0.15, 0.15)),
            ((0.15, 0.15), (0.30, 0.00)),
            ((0.30, 0.00), (0.72, 0.00)),
            ((0.72, 0.00), (0.20, 0.55)),
        ),
    ),
    "'": VectorGlyph(
        0.35,
        (((0.15, 1.00), (0.15, 0.70)),),
    ),
    "(": VectorGlyph(
        0.55,
        (
            ((0.38, 1.00), (0.10, 0.70)),
            ((0.10, 0.70), (0.10, 0.30)),
            ((0.10, 0.30), (0.38, 0.00)),
        ),
    ),
    ")": VectorGlyph(
        0.55,
        (
            ((0.10, 1.00), (0.38, 0.70)),
            ((0.38, 0.70), (0.38, 0.30)),
            ((0.38, 0.30), (0.10, 0.00)),
        ),
    ),
    "*": VectorGlyph(
        0.70,
        (
            ((0.30, 0.95), (0.30, 0.35)),
            ((0.00, 0.65), (0.60, 0.65)),
            ((0.08, 0.90), (0.52, 0.40)),
            ((0.52, 0.90), (0.08, 0.40)),
        ),
    ),
    "+": VectorGlyph(
        0.80,
        (
            ((0.35, 0.80), (0.35, 0.20)),
            ((0.05, 0.50), (0.65, 0.50)),
        ),
    ),
    ",": VectorGlyph(
        0.40,
        (((0.24, 0.10), (0.08, -0.18)),),
    ),
    "-": VectorGlyph(
        0.75,
        (((0.05, 0.50), (0.60, 0.50)),),
    ),
    ".": VectorGlyph(
        0.40,
        (
            ((0.10, 0.04), (0.26, 0.04)),
            ((0.10, 0.12), (0.26, 0.12)),
        ),
    ),
    "/": VectorGlyph(
        0.75,
        (((0.00, 0.00), (0.60, 1.00)),),
    ),
    ":": VectorGlyph(
        0.40,
        (
            ((0.10, 0.72), (0.24, 0.72)),
            ((0.10, 0.20), (0.24, 0.20)),
        ),
    ),
    ";": VectorGlyph(
        0.40,
        (
            ((0.10, 0.72), (0.24, 0.72)),
            ((0.24, 0.10), (0.08, -0.18)),
        ),
    ),
    "<": VectorGlyph(
        0.80,
        (
            ((0.65, 0.85), (0.05, 0.50)),
            ((0.05, 0.50), (0.65, 0.15)),
        ),
    ),
    "=": VectorGlyph(
        0.80,
        (
            ((0.05, 0.65), (0.65, 0.65)),
            ((0.05, 0.35), (0.65, 0.35)),
        ),
    ),
    ">": VectorGlyph(
        0.80,
        (
            ((0.05, 0.85), (0.65, 0.50)),
            ((0.65, 0.50), (0.05, 0.15)),
        ),
    ),
    "?": VectorGlyph(
        0.80,
        (
            ((0.00, 0.82), (0.12, 1.00)),
            ((0.12, 1.00), (0.52, 1.00)),
            ((0.52, 1.00), (0.65, 0.85)),
            ((0.65, 0.85), (0.65, 0.65)),
            ((0.65, 0.65), (0.35, 0.45)),
            ((0.35, 0.45), (0.35, 0.30)),
            ((0.29, 0.05), (0.41, 0.05)),
        ),
    ),
    "@": VectorGlyph(
        1.05,
        (
            ((0.20, 0.00), (0.65, 0.00)),
            ((0.65, 0.00), (0.85, 0.20)),
            ((0.85, 0.20), (0.85, 0.78)),
            ((0.85, 0.78), (0.65, 1.00)),
            ((0.65, 1.00), (0.20, 1.00)),
            ((0.20, 1.00), (0.00, 0.78)),
            ((0.00, 0.78), (0.00, 0.20)),
            ((0.00, 0.20), (0.20, 0.00)),
            ((0.60, 0.25), (0.60, 0.65)),
            ((0.60, 0.65), (0.45, 0.78)),
            ((0.45, 0.78), (0.25, 0.70)),
            ((0.25, 0.70), (0.20, 0.45)),
            ((0.20, 0.45), (0.35, 0.25)),
            ((0.35, 0.25), (0.60, 0.25)),
            ((0.60, 0.25), (0.80, 0.40)),
        ),
    ),
    "[": VectorGlyph(
        0.55,
        (
            ((0.35, 1.00), (0.05, 1.00)),
            ((0.05, 1.00), (0.05, 0.00)),
            ((0.05, 0.00), (0.35, 0.00)),
        ),
    ),
    "\\": VectorGlyph(
        0.75,
        (((0.00, 1.00), (0.60, 0.00)),),
    ),
    "]": VectorGlyph(
        0.55,
        (
            ((0.05, 1.00), (0.35, 1.00)),
            ((0.35, 1.00), (0.35, 0.00)),
            ((0.35, 0.00), (0.05, 0.00)),
        ),
    ),
    "^": VectorGlyph(
        0.80,
        (
            ((0.05, 0.55), (0.35, 1.00)),
            ((0.35, 1.00), (0.65, 0.55)),
        ),
    ),
    "_": VectorGlyph(
        0.80,
        (((0.00, 0.00), (0.70, 0.00)),),
    ),
    "`": VectorGlyph(
        0.45,
        (((0.05, 1.00), (0.25, 0.78)),),
    ),
    "{": VectorGlyph(
        0.65,
        (
            ((0.45, 1.00), (0.20, 0.82)),
            ((0.20, 0.82), (0.20, 0.58)),
            ((0.20, 0.58), (0.05, 0.50)),
            ((0.05, 0.50), (0.20, 0.42)),
            ((0.20, 0.42), (0.20, 0.18)),
            ((0.20, 0.18), (0.45, 0.00)),
        ),
    ),
    "|": VectorGlyph(
        0.35,
        (((0.15, 1.00), (0.15, 0.00)),),
    ),
    "}": VectorGlyph(
        0.65,
        (
            ((0.05, 1.00), (0.30, 0.82)),
            ((0.30, 0.82), (0.30, 0.58)),
            ((0.30, 0.58), (0.45, 0.50)),
            ((0.45, 0.50), (0.30, 0.42)),
            ((0.30, 0.42), (0.30, 0.18)),
            ((0.30, 0.18), (0.05, 0.00)),
        ),
    ),
    "~": VectorGlyph(
        0.85,
        (
            ((0.05, 0.45), (0.22, 0.58)),
            ((0.22, 0.58), (0.45, 0.42)),
            ((0.45, 0.42), (0.65, 0.55)),
        ),
    ),
}


def supported_vector_text_characters() -> frozenset[str]:
    """Return all renderable non-whitespace vector glyph characters."""

    return frozenset(VECTOR_GLYPHS)


def resolve_stroke_width(size, stroke_width=None) -> float:
    """Validate and resolve the exact stroke width for vector text."""

    size_value = _positive_float(size, "Size")
    if stroke_width is None:
        stroke_value = size_value * DEFAULT_STROKE_WIDTH_RATIO
    else:
        stroke_value = _positive_float(stroke_width, "Stroke width")

    if stroke_value >= size_value:
        raise ValueError("Stroke width must be positive and smaller than size")
    return stroke_value


def _decimal_point_advance(size_value: float, stroke_value: float) -> float:
    glyph = VECTOR_GLYPHS["."]
    return max(glyph.advance * size_value, 2.0 * stroke_value + 0.10 * size_value)


@lru_cache(maxsize=512)
def _scaled_vector_glyph_segments(
    char: str,
    size_value: float,
    stroke_value: float,
) -> tuple[VectorTextSegment, ...]:
    """Return origin-local scaled centerline segments for one glyph."""

    if char == ".":
        advance = _decimal_point_advance(size_value, stroke_value)
        center_x = advance / 2.0
        half_centerline_length = stroke_value

        return tuple(
            VectorTextSegment(
                (center_x - half_centerline_length, center_y),
                (center_x + half_centerline_length, center_y),
            )
            for center_y in (
                stroke_value / 2.0,
                1.5 * stroke_value,
            )
        )

    glyph = VECTOR_GLYPHS[char]
    return tuple(
        VectorTextSegment(
            (start[0] * size_value, start[1] * size_value),
            (end[0] * size_value, end[1] * size_value),
        )
        for start, end in glyph.strokes
    )


def _translate_segment(
    segment: VectorTextSegment, origin: Point2D
) -> VectorTextSegment:
    origin_x, origin_y = origin
    return VectorTextSegment(
        (segment.start[0] + origin_x, segment.start[1] + origin_y),
        (segment.end[0] + origin_x, segment.end[1] + origin_y),
    )


def _vector_text_glyph_placements(
    text: str,
    size_value: float,
    stroke_value: float,
) -> tuple[str, tuple[_VectorGlyphPlacement, ...]]:
    normalized_text = text.upper()
    placements: list[_VectorGlyphPlacement] = []
    cursor_x = 0.0
    cursor_y = 0.0

    for char in normalized_text:
        if char == " ":
            cursor_x += SPACE_ADVANCE * size_value
            continue
        if char == "\n":
            cursor_x = 0.0
            cursor_y -= LINE_ADVANCE * size_value
            continue

        glyph = VECTOR_GLYPHS.get(char)
        if glyph is None:
            raise ValueError(f"Unsupported vector text character: {char!r}")

        placements.append(_VectorGlyphPlacement(char, (cursor_x, cursor_y)))
        if char == ".":
            cursor_x += _decimal_point_advance(size_value, stroke_value)
        else:
            cursor_x += glyph.advance * size_value

    if not placements:
        raise ValueError("Text contains no renderable vector glyphs")

    return normalized_text, tuple(placements)


def layout_vector_text(text: str, size, *, stroke_width=None) -> VectorTextLayout:
    """Layout text as scaled centerline stroke segments.

    Lowercase ASCII input is normalized to uppercase. Spaces and newlines
    advance the cursor but do not produce strokes.
    """

    if not isinstance(text, str) or text == "":
        raise ValueError("Text must be a non-empty string")

    size_value = _positive_float(size, "Size")
    stroke_value = resolve_stroke_width(size_value, stroke_width)

    segments: list[VectorTextSegment] = []
    normalized_text, placements = _vector_text_glyph_placements(
        text,
        size_value,
        stroke_value,
    )
    for placement in placements:
        segments.extend(
            _translate_segment(segment, placement.origin)
            for segment in _scaled_vector_glyph_segments(
                placement.char,
                size_value,
                stroke_value,
            )
        )

    return VectorTextLayout(
        text=normalized_text,
        size=size_value,
        stroke_width=stroke_value,
        segments=tuple(segments),
    )


def segment_stroke_polygon(
    segment: VectorTextSegment,
    stroke_width: float,
) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    """Return a rectangle around a centerline segment."""

    x1, y1 = segment.start
    x2, y2 = segment.end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        raise ValueError("Cannot materialize a zero-length vector text stroke")

    half_width = stroke_width / 2.0
    normal_x = -dy / length * half_width
    normal_y = dx / length * half_width

    return (
        (x1 + normal_x, y1 + normal_y),
        (x2 + normal_x, y2 + normal_y),
        (x2 - normal_x, y2 - normal_y),
        (x1 - normal_x, y1 - normal_y),
    )


def vector_text_stroke_polygons(
    segments: tuple[VectorTextSegment, ...],
    stroke_width: float,
) -> tuple[StrokePolygon2D, ...]:
    """Return stroke rectangles plus rounded overlap caps at shared endpoints."""

    touching_endpoints = _touching_endpoint_keys(segments)
    polygons: list[StrokePolygon2D] = []

    for segment in segments:
        polygons.append(segment_stroke_polygon(segment, stroke_width))
        if _endpoint_key(segment.start) in touching_endpoints:
            polygons.append(
                _segment_stroke_cap_polygon(segment, stroke_width, at_start=True)
            )
        if _endpoint_key(segment.end) in touching_endpoints:
            polygons.append(
                _segment_stroke_cap_polygon(segment, stroke_width, at_start=False)
            )

    return tuple(polygons)


def _segment_stroke_cap_polygon(
    segment: VectorTextSegment,
    stroke_width: float,
    *,
    at_start: bool,
) -> StrokePolygon2D:
    x1, y1 = segment.start
    x2, y2 = segment.end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        raise ValueError("Cannot materialize a zero-length vector text stroke")

    radius = stroke_width / 2.0
    tangent_x = dx / length
    tangent_y = dy / length
    normal_x = -dy / length
    normal_y = dx / length
    center_x, center_y = segment.start if at_start else segment.end
    outward_x = -tangent_x if at_start else tangent_x
    outward_y = -tangent_y if at_start else tangent_y

    return tuple(
        (
            center_x
            + (outward_x * math.cos(angle) + normal_x * math.sin(angle)) * radius,
            center_y
            + (outward_y * math.cos(angle) + normal_y * math.sin(angle)) * radius,
        )
        for angle in (
            math.pi / 2.0 - math.pi * index / STROKE_CAP_ARC_SEGMENTS
            for index in range(STROKE_CAP_ARC_SEGMENTS + 1)
        )
    )


def _touching_endpoint_keys(
    segments: tuple[VectorTextSegment, ...],
) -> set[tuple[float, float]]:
    endpoint_counts = Counter(
        _endpoint_key(point)
        for segment in segments
        for point in (segment.start, segment.end)
    )
    return {key for key, count in endpoint_counts.items() if count > 1}


def _endpoint_key(point: Point2D) -> tuple[float, float]:
    return (round(point[0], 9), round(point[1], 9))


@lru_cache(maxsize=512)
def _materialized_vector_glyph_solid(
    adapter_id: str,
    char: str,
    size_value: float,
    stroke_value: float,
    thickness_value: float,
):
    from shellforgepy.adapters._adapter import create_extruded_polygon, fuse_parts

    segments = _scaled_vector_glyph_segments(char, size_value, stroke_value)
    solids = [
        create_extruded_polygon(polygon, thickness_value)
        for polygon in vector_text_stroke_polygons(segments, stroke_value)
    ]
    if not solids:
        raise RuntimeError("Vector glyph layout produced no solids")

    solid = solids[0]
    for stroke_solid in solids[1:]:
        solid = fuse_parts(solid, stroke_solid)
    return solid


def create_vector_text_object(
    text: str,
    size,
    thickness,
    *,
    stroke_width=None,
    padding=0.0,
):
    """Create extruded straight-line vector text anchored to the XY origin.

    This dependency-free renderer uses simple built-in glyph strokes instead of
    resolving a system font. CAD primitives are imported lazily so the glyph
    layout helpers remain backend-independent until materialization is needed.
    """

    from shellforgepy.adapters._adapter import (
        fuse_parts,
        get_adapter_id,
        get_bounding_box,
        translate_part,
    )

    thickness_value = _positive_float(thickness, "Thickness")
    padding_value = _non_negative_float(padding, "Padding")
    if not isinstance(text, str) or text == "":
        raise ValueError("Text must be a non-empty string")

    size_value = _positive_float(size, "Size")
    stroke_value = resolve_stroke_width(size_value, stroke_width)
    _, placements = _vector_text_glyph_placements(text, size_value, stroke_value)
    adapter_id = get_adapter_id()
    solids = [
        translate_part(
            _materialized_vector_glyph_solid(
                adapter_id,
                placement.char,
                size_value,
                stroke_value,
                thickness_value,
            ),
            (placement.origin[0], placement.origin[1], 0.0),
        )
        for placement in placements
    ]
    if not solids:
        raise RuntimeError("Vector text placement produced no solids")

    solid = solids[0]
    for stroke_solid in solids[1:]:
        solid = fuse_parts(solid, stroke_solid)

    min_point, _ = get_bounding_box(solid)
    return translate_part(
        solid,
        (
            padding_value - min_point[0],
            padding_value - min_point[1],
            -min_point[2],
        ),
    )


def _positive_float(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be positive") from None

    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be positive")
    return numeric


def _non_negative_float(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} cannot be negative") from None

    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} cannot be negative")
    return numeric


__all__ = [
    "DEFAULT_STROKE_WIDTH_RATIO",
    "LINE_ADVANCE",
    "PUNCTUATION_CHARACTERS",
    "SPACE_ADVANCE",
    "VECTOR_GLYPHS",
    "VectorGlyph",
    "VectorTextLayout",
    "VectorTextSegment",
    "create_vector_text_object",
    "layout_vector_text",
    "resolve_stroke_width",
    "segment_stroke_polygon",
    "supported_vector_text_characters",
    "vector_text_stroke_polygons",
]
