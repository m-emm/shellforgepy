from webbrowser import get

import numpy as np
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.simple import (
    create_basic_box,
    get_bounding_box,
    get_bounding_box_center,
)


def test_translate():

    part = create_basic_box(10, 20, 30)
    part_center = get_bounding_box_center(part)

    part_translated = translate(5, 7, 13)(part)
    translated_center = get_bounding_box_center(part_translated)

    assert translated_center == (
        part_center[0] + 5,
        part_center[1] + 7,
        part_center[2] + 13,
    )


def test_rotate():

    part = create_basic_box(10, 20, 30)

    rotated_part = rotate(90, axis=(0, 0, 1), center=(0, 0, 0))(part)

    assert rotated_part is not None

    bounding_box = get_bounding_box(rotated_part)
    len_x = bounding_box[1][0] - bounding_box[0][0]
    len_y = bounding_box[1][1] - bounding_box[0][1]
    len_z = bounding_box[1][2] - bounding_box[0][2]

    assert np.allclose(len_x, 20)
    assert np.allclose(len_y, 10)
    assert np.allclose(len_z, 30)
