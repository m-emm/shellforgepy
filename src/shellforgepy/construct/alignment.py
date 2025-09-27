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
