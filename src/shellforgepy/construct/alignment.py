from enum import Enum


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
}

for k, v in ALIGNMENT_AXIS.items():
    setattr(k, "axis", v)

for k, v in ALIGNMENT_STACK_ALIGNMENT.items():
    setattr(k, "stack_alignment", v)


for k, v in OPPOSITE_ALIGNMENTS.items():
    setattr(k, "opposite", v)
