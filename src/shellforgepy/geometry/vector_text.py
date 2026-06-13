"""Dependency-free straight-line glyphs for fast vector text creation."""

import math
from dataclasses import dataclass

Point2D = tuple[float, float]
Segment2D = tuple[Point2D, Point2D]

DEFAULT_STROKE_WIDTH_RATIO = 0.08
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
            ((0.60, 1.00), (0.60, 0.00)),
            ((0.00, 0.65), (0.60, 0.65)),
            ((0.00, 0.65), (0.45, 1.00)),
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


def _append_decimal_point_segments(
    segments: list[VectorTextSegment],
    cursor_x: float,
    cursor_y: float,
    size_value: float,
    stroke_value: float,
) -> float:
    """Append a decimal point whose materialized dot is two stroke widths square."""

    glyph = VECTOR_GLYPHS["."]
    advance = max(glyph.advance * size_value, 2.0 * stroke_value + 0.10 * size_value)
    center_x = cursor_x + advance / 2.0
    half_centerline_length = stroke_value

    for center_y in (
        cursor_y + stroke_value / 2.0,
        cursor_y + 1.5 * stroke_value,
    ):
        segments.append(
            VectorTextSegment(
                (center_x - half_centerline_length, center_y),
                (center_x + half_centerline_length, center_y),
            )
        )

    return advance


def layout_vector_text(text: str, size, *, stroke_width=None) -> VectorTextLayout:
    """Layout text as scaled centerline stroke segments.

    Lowercase ASCII input is normalized to uppercase. Spaces and newlines
    advance the cursor but do not produce strokes.
    """

    if not isinstance(text, str) or text == "":
        raise ValueError("Text must be a non-empty string")

    size_value = _positive_float(size, "Size")
    stroke_value = resolve_stroke_width(size_value, stroke_width)

    normalized_text = text.upper()
    segments: list[VectorTextSegment] = []
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

        if char == ".":
            cursor_x += _append_decimal_point_segments(
                segments,
                cursor_x,
                cursor_y,
                size_value,
                stroke_value,
            )
        else:
            for start, end in glyph.strokes:
                segments.append(
                    VectorTextSegment(
                        (
                            cursor_x + start[0] * size_value,
                            cursor_y + start[1] * size_value,
                        ),
                        (
                            cursor_x + end[0] * size_value,
                            cursor_y + end[1] * size_value,
                        ),
                    )
                )
            cursor_x += glyph.advance * size_value

    if not segments:
        raise ValueError("Text contains no renderable vector glyphs")

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
        create_extruded_polygon,
        fuse_parts,
        get_bounding_box,
        translate_part,
    )

    thickness_value = _positive_float(thickness, "Thickness")
    padding_value = _non_negative_float(padding, "Padding")
    layout = layout_vector_text(text, size, stroke_width=stroke_width)

    solids = [
        create_extruded_polygon(
            segment_stroke_polygon(segment, layout.stroke_width),
            thickness_value,
        )
        for segment in layout.segments
    ]
    if not solids:
        raise RuntimeError("Vector text layout produced no solids")

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
]
