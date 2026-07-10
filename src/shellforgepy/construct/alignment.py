import math
from enum import Enum
from numbers import Real


class Alignment(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4
    FRONT = 5
    BACK = 6
    CENTER = 7
    STACK_LEFT = 8
    STACK_RIGHT = 9
    STACK_TOP = 10
    STACK_BOTTOM = 11
    STACK_FRONT = 12
    STACK_BACK = 13
    EDGE_LEFT = 14
    EDGE_RIGHT = 15
    EDGE_TOP = 16
    EDGE_BOTTOM = 17
    EDGE_FRONT = 18
    EDGE_BACK = 19


ALIGNMENT_SIGNS = {
    Alignment.LEFT: -1,
    Alignment.RIGHT: 1,
    Alignment.TOP: 1,
    Alignment.BOTTOM: -1,
    Alignment.FRONT: -1,
    Alignment.BACK: 1,
    Alignment.CENTER: 0,
    Alignment.STACK_LEFT: -1,
    Alignment.STACK_RIGHT: 1,
    Alignment.STACK_TOP: 1,
    Alignment.STACK_BOTTOM: -1,
    Alignment.STACK_FRONT: -1,
    Alignment.STACK_BACK: 1,
    Alignment.EDGE_LEFT: -1,
    Alignment.EDGE_RIGHT: 1,
    Alignment.EDGE_TOP: 1,
    Alignment.EDGE_BOTTOM: -1,
    Alignment.EDGE_FRONT: -1,
    Alignment.EDGE_BACK: 1,
}

for k, v in ALIGNMENT_SIGNS.items():
    setattr(k, "sign", v)

ALIGNMENT_AXIS = {
    Alignment.LEFT: 0,
    Alignment.RIGHT: 0,
    Alignment.TOP: 2,
    Alignment.BOTTOM: 2,
    Alignment.FRONT: 1,
    Alignment.BACK: 1,
    Alignment.CENTER: None,
    Alignment.STACK_LEFT: 0,
    Alignment.STACK_RIGHT: 0,
    Alignment.STACK_TOP: 2,
    Alignment.STACK_BOTTOM: 2,
    Alignment.STACK_FRONT: 1,
    Alignment.STACK_BACK: 1,
    Alignment.EDGE_LEFT: 0,
    Alignment.EDGE_RIGHT: 0,
    Alignment.EDGE_TOP: 2,
    Alignment.EDGE_BOTTOM: 2,
    Alignment.EDGE_FRONT: 1,
    Alignment.EDGE_BACK: 1,
}

ALIGNMENT_STACK_ALIGNMENT = {
    Alignment.LEFT: Alignment.STACK_LEFT,
    Alignment.RIGHT: Alignment.STACK_RIGHT,
    Alignment.TOP: Alignment.STACK_TOP,
    Alignment.BOTTOM: Alignment.STACK_BOTTOM,
    Alignment.FRONT: Alignment.STACK_FRONT,
    Alignment.BACK: Alignment.STACK_BACK,
    Alignment.CENTER: None,
    Alignment.STACK_LEFT: Alignment.STACK_LEFT,
    Alignment.STACK_RIGHT: Alignment.STACK_RIGHT,
    Alignment.STACK_TOP: Alignment.STACK_TOP,
    Alignment.STACK_BOTTOM: Alignment.STACK_BOTTOM,
    Alignment.STACK_FRONT: Alignment.STACK_FRONT,
    Alignment.STACK_BACK: Alignment.STACK_BACK,
    Alignment.EDGE_LEFT: Alignment.STACK_LEFT,
    Alignment.EDGE_RIGHT: Alignment.STACK_RIGHT,
    Alignment.EDGE_TOP: Alignment.STACK_TOP,
    Alignment.EDGE_BOTTOM: Alignment.STACK_BOTTOM,
    Alignment.EDGE_FRONT: Alignment.STACK_FRONT,
    Alignment.EDGE_BACK: Alignment.STACK_BACK,
}

ALIGNMENT_EDGE_ALIGNMENT = {
    Alignment.LEFT: Alignment.EDGE_LEFT,
    Alignment.RIGHT: Alignment.EDGE_RIGHT,
    Alignment.TOP: Alignment.EDGE_TOP,
    Alignment.BOTTOM: Alignment.EDGE_BOTTOM,
    Alignment.FRONT: Alignment.EDGE_FRONT,
    Alignment.BACK: Alignment.EDGE_BACK,
    Alignment.CENTER: None,
    Alignment.STACK_LEFT: Alignment.EDGE_LEFT,
    Alignment.STACK_RIGHT: Alignment.EDGE_RIGHT,
    Alignment.STACK_TOP: Alignment.EDGE_TOP,
    Alignment.STACK_BOTTOM: Alignment.EDGE_BOTTOM,
    Alignment.STACK_FRONT: Alignment.EDGE_FRONT,
    Alignment.STACK_BACK: Alignment.EDGE_BACK,
    Alignment.EDGE_LEFT: Alignment.EDGE_LEFT,
    Alignment.EDGE_RIGHT: Alignment.EDGE_RIGHT,
    Alignment.EDGE_TOP: Alignment.EDGE_TOP,
    Alignment.EDGE_BOTTOM: Alignment.EDGE_BOTTOM,
    Alignment.EDGE_FRONT: Alignment.EDGE_FRONT,
    Alignment.EDGE_BACK: Alignment.EDGE_BACK,
}

OPPOSITE_ALIGNMENTS = {
    Alignment.LEFT: Alignment.RIGHT,
    Alignment.RIGHT: Alignment.LEFT,
    Alignment.TOP: Alignment.BOTTOM,
    Alignment.BOTTOM: Alignment.TOP,
    Alignment.FRONT: Alignment.BACK,
    Alignment.BACK: Alignment.FRONT,
    Alignment.CENTER: Alignment.CENTER,
    Alignment.STACK_LEFT: Alignment.STACK_RIGHT,
    Alignment.STACK_RIGHT: Alignment.STACK_LEFT,
    Alignment.STACK_TOP: Alignment.STACK_BOTTOM,
    Alignment.STACK_BOTTOM: Alignment.STACK_TOP,
    Alignment.STACK_FRONT: Alignment.STACK_BACK,
    Alignment.STACK_BACK: Alignment.STACK_FRONT,
    Alignment.EDGE_LEFT: Alignment.EDGE_RIGHT,
    Alignment.EDGE_RIGHT: Alignment.EDGE_LEFT,
    Alignment.EDGE_TOP: Alignment.EDGE_BOTTOM,
    Alignment.EDGE_BOTTOM: Alignment.EDGE_TOP,
    Alignment.EDGE_FRONT: Alignment.EDGE_BACK,
    Alignment.EDGE_BACK: Alignment.EDGE_FRONT,
}

ALIGNMENT_VECTORS = {
    Alignment.LEFT: (-1, 0, 0),
    Alignment.RIGHT: (1, 0, 0),
    Alignment.TOP: (0, 0, 1),
    Alignment.BOTTOM: (0, 0, -1),
    Alignment.FRONT: (0, -1, 0),
    Alignment.BACK: (0, 1, 0),
    Alignment.CENTER: (0, 0, 0),
    Alignment.STACK_LEFT: (-1, 0, 0),
    Alignment.STACK_RIGHT: (1, 0, 0),
    Alignment.STACK_TOP: (0, 0, 1),
    Alignment.STACK_BOTTOM: (0, 0, -1),
    Alignment.STACK_FRONT: (0, -1, 0),
    Alignment.STACK_BACK: (0, 1, 0),
    Alignment.EDGE_LEFT: (-1, 0, 0),
    Alignment.EDGE_RIGHT: (1, 0, 0),
    Alignment.EDGE_TOP: (0, 0, 1),
    Alignment.EDGE_BOTTOM: (0, 0, -1),
    Alignment.EDGE_FRONT: (0, -1, 0),
    Alignment.EDGE_BACK: (0, 1, 0),
}

_BASE_ALIGNMENT_BY_VECTOR = {
    (-1, 0, 0): Alignment.LEFT,
    (1, 0, 0): Alignment.RIGHT,
    (0, 0, 1): Alignment.TOP,
    (0, 0, -1): Alignment.BOTTOM,
    (0, -1, 0): Alignment.FRONT,
    (0, 1, 0): Alignment.BACK,
    (0, 0, 0): Alignment.CENTER,
}

_STACK_ALIGNMENT_BY_VECTOR = {
    (-1, 0, 0): Alignment.STACK_LEFT,
    (1, 0, 0): Alignment.STACK_RIGHT,
    (0, 0, 1): Alignment.STACK_TOP,
    (0, 0, -1): Alignment.STACK_BOTTOM,
    (0, -1, 0): Alignment.STACK_FRONT,
    (0, 1, 0): Alignment.STACK_BACK,
}

_EDGE_ALIGNMENT_BY_VECTOR = {
    (-1, 0, 0): Alignment.EDGE_LEFT,
    (1, 0, 0): Alignment.EDGE_RIGHT,
    (0, 0, 1): Alignment.EDGE_TOP,
    (0, 0, -1): Alignment.EDGE_BOTTOM,
    (0, -1, 0): Alignment.EDGE_FRONT,
    (0, 1, 0): Alignment.EDGE_BACK,
}

_ROTATION_AXES = {
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
}


def _normalize_quarter_turns(angle):
    if not isinstance(angle, Real) or not math.isfinite(angle):
        raise ValueError("Alignment rotation angle must be a finite number")

    quarter_turns_exact = angle / 90
    quarter_turns = round(quarter_turns_exact)
    if not math.isclose(quarter_turns_exact, quarter_turns, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Alignment rotation angle must be a multiple of 90 degrees")

    return quarter_turns % 4


def _normalize_rotation_axis(axis):
    try:
        normalized = tuple(axis)
    except TypeError as exc:
        raise ValueError(
            "Alignment rotation axis must be one of the coordinate axes"
        ) from exc

    if normalized not in _ROTATION_AXES:
        raise ValueError(
            "Alignment rotation axis must be one of (1, 0, 0), (0, 1, 0), "
            "or (0, 0, 1)"
        )

    return normalized


def _rotate_alignment_vector_once(vector, axis):
    x, y, z = vector
    if axis == (1, 0, 0):
        return (x, -z, y)
    if axis == (0, 1, 0):
        return (z, y, -x)
    if axis == (0, 0, 1):
        return (-y, x, z)

    raise ValueError(f"Unsupported alignment rotation axis: {axis}")


def _alignment_by_rotated_vector(alignment, vector):
    if alignment == Alignment.CENTER:
        return Alignment.CENTER

    if alignment.name.startswith("STACK_"):
        return _STACK_ALIGNMENT_BY_VECTOR[vector]

    if alignment.name.startswith("EDGE_"):
        return _EDGE_ALIGNMENT_BY_VECTOR[vector]

    return _BASE_ALIGNMENT_BY_VECTOR[vector]


def rotate_alignment(angle, axis=(0, 0, 1)):
    """
    Return a function that rotates an ``Alignment`` by right-hand 90-degree turns.

    The rotation follows the same active convention as ``rotate()``. With the
    default Z axis, ``rotate_alignment(90)(Alignment.RIGHT)`` becomes
    ``Alignment.BACK`` and ``Alignment.LEFT`` becomes ``Alignment.FRONT``.
    """

    quarter_turns = _normalize_quarter_turns(angle)
    axis = _normalize_rotation_axis(axis)

    def retval(alignment):
        if not isinstance(alignment, Alignment):
            raise TypeError("rotate_alignment() can only rotate Alignment values")

        vector = ALIGNMENT_VECTORS[alignment]
        for _ in range(quarter_turns):
            vector = _rotate_alignment_vector_once(vector, axis)

        return _alignment_by_rotated_vector(alignment, vector)

    return retval


for k, v in ALIGNMENT_AXIS.items():
    setattr(k, "axis", v)

for k, v in ALIGNMENT_STACK_ALIGNMENT.items():
    setattr(k, "stack_alignment", v)

for k, v in ALIGNMENT_EDGE_ALIGNMENT.items():
    setattr(k, "edge_alignment", v)


for k, v in OPPOSITE_ALIGNMENTS.items():
    setattr(k, "opposite", v)
