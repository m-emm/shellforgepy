import logging

import numpy as np
import pytest
from shellforgepy.construct.bounding_box_helpers import (
    bottom_bounding_box_point,
    get_bounding_box_center,
    get_xlen,
    get_xmax,
    get_xmin,
    get_ylen,
    get_ymax,
    get_ymin,
    get_zlen,
    get_zmax,
    get_zmin,
)

_logger = logging.getLogger(__name__)


def test_bottom_point_with_z_positive_normal():
    """Test bottom_bounding_box_point with normal (0,0,1) - should return bottom leftmost/frontmost (lowest y)."""

    # Create a bounding box from (0,0,0) to (10,10,10)
    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (0, 0, 1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (0,0,1), the "lowest" points are those with minimum z
    # Among those, we want leftmost (min x) and frontmost (min y)
    expected = (0, 0, 0)  # bottom-left-front corner

    assert result == expected


def test_bottom_point_with_z_negative_normal():
    """Test bottom_bounding_box_point with normal (0,0,-1) - should return top leftmost/frontmost."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (0, 0, -1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (0,0,-1), the "lowest" points are those with maximum z
    # Among those, we want leftmost (min x) and frontmost (min y)
    expected = (0, 0, 10)  # top-left-front corner

    assert result == expected


def test_bottom_point_with_positive_diagonal_normal():
    """Test bottom_bounding_box_point with normal (1,1,1) - should return bottom leftmost."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (1, 1, 1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (1,1,1), the "lowest" point is the one with minimum dot product
    # This should be the corner with minimum x, y, and z coordinates
    expected = (0, 0, 0)

    assert result == expected


def test_bottom_point_with_negative_diagonal_normal():
    """Test bottom_bounding_box_point with normal (-1,-1,-1) - should return top rightmost back (biggest x, biggest y)."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (-1, -1, -1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (-1,-1,-1), the "lowest" point is the one with minimum dot product
    # This should be the corner with maximum x, y, and z coordinates
    expected = (10, 10, 10)

    assert result == expected


def test_bottom_point_with_x_positive_normal():
    """Test bottom_bounding_box_point with normal (1,0,0) - should return leftmost point."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (1, 0, 0)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (1,0,0), the "lowest" points are those with minimum x
    # Among those, we want smallest y and z for tie-breaking
    expected = (0, 0, 0)

    assert result == expected


def test_bottom_point_with_x_negative_normal():
    """Test bottom_bounding_box_point with normal (-1,0,0) - should return rightmost point."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (-1, 0, 0)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (-1,0,0), the "lowest" points are those with maximum x
    # Among those, we want smallest y and z for tie-breaking
    expected = (10, 0, 0)

    assert result == expected


def test_bottom_point_with_y_positive_normal():
    """Test bottom_bounding_box_point with normal (0,1,0) - should return frontmost point."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (0, 1, 0)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (0,1,0), the "lowest" points are those with minimum y
    # Among those, we want smallest x and z for tie-breaking
    expected = (0, 0, 0)

    assert result == expected


def test_bottom_point_with_y_negative_normal():
    """Test bottom_bounding_box_point with normal (0,-1,0) - should return backmost point."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (0, -1, 0)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (0,-1,0), the "lowest" points are those with maximum y
    # Among those, we want smallest x and z for tie-breaking
    expected = (0, 10, 0)

    assert result == expected


def test_bottom_point_with_non_unit_normal():
    """Test that bottom_bounding_box_point works correctly with non-unit normal vectors."""

    bounding_box = ((0, 0, 0), (10, 10, 10))
    normal = (2, 2, 2)  # Non-unit vector, same direction as (1,1,1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # Should give same result as (1,1,1) after normalization
    expected = (0, 0, 0)

    assert result == expected


def test_bottom_point_with_offset_bounding_box():
    """Test bottom_bounding_box_point with a bounding box that doesn't start at origin."""

    bounding_box = ((5, 3, 2), (15, 13, 12))
    normal = (0, 0, 1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (0,0,1), want minimum z, then minimum x, then minimum y
    expected = (5, 3, 2)

    assert result == expected


def test_bottom_point_with_negative_coordinates():
    """Test bottom_bounding_box_point with negative coordinates in bounding box."""

    bounding_box = ((-5, -5, -5), (5, 5, 5))
    normal = (1, 1, 1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # With normal (1,1,1), want minimum dot product
    expected = (-5, -5, -5)

    assert result == expected


@pytest.mark.parametrize(
    "normal, expected_corner",
    [
        ((1, 0, 0), (0, 0, 0)),  # leftmost
        ((-1, 0, 0), (10, 0, 0)),  # rightmost
        ((0, 1, 0), (0, 0, 0)),  # frontmost
        ((0, -1, 0), (0, 10, 0)),  # backmost
        ((0, 0, 1), (0, 0, 0)),  # bottom
        ((0, 0, -1), (0, 0, 10)),  # top
        ((1, 1, 0), (0, 0, 0)),  # front-left
        ((-1, -1, 0), (10, 10, 0)),  # back-right
        ((1, 0, 1), (0, 0, 0)),  # bottom-left
        ((-1, 0, -1), (10, 0, 10)),  # top-right
    ],
)
def test_bottom_point_parametrized_normals(normal, expected_corner):
    """Parametrized test for various normal directions."""

    bounding_box = ((0, 0, 0), (10, 10, 10))

    result = bottom_bounding_box_point(bounding_box, normal)

    assert result == expected_corner


def test_bottom_point_with_zero_size_dimension():
    """Test bottom_bounding_box_point with a bounding box that has zero size in one dimension."""

    # Flat box in XY plane
    bounding_box = ((0, 0, 5), (10, 10, 5))
    normal = (0, 0, 1)

    result = bottom_bounding_box_point(bounding_box, normal)

    # All corners have same z, so tie-breaking should pick leftmost/frontmost
    expected = (0, 0, 5)

    assert result == expected


def test_bottom_point_numerical_precision():
    """Test that bottom_bounding_box_point handles numerical precision correctly."""

    bounding_box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    normal = (1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3))  # Normalized (1,1,1)

    result = bottom_bounding_box_point(bounding_box, normal)

    expected = (0.0, 0.0, 0.0)

    assert np.allclose(result, expected, atol=1e-10)


def test_get_bounding_box_center():
    """Test get_bounding_box_center function."""

    bounding_box = ((0, 0, 0), (10, 20, 30))

    result = get_bounding_box_center(bounding_box)

    expected = (5.0, 10.0, 15.0)

    assert result == expected


def test_bounding_box_dimension_functions():
    """Test the various bounding box dimension and coordinate functions."""

    bounding_box = ((1, 2, 3), (11, 22, 33))

    assert get_xlen(bounding_box) == 10
    assert get_ylen(bounding_box) == 20
    assert get_zlen(bounding_box) == 30

    assert get_xmin(bounding_box) == 1
    assert get_ymin(bounding_box) == 2
    assert get_zmin(bounding_box) == 3

    assert get_xmax(bounding_box) == 11
    assert get_ymax(bounding_box) == 22
    assert get_zmax(bounding_box) == 33


def test_bottom_point_consistency_across_transformations():
    """Test that bottom_bounding_box_point behaves consistently when bounding box is transformed."""

    # Original bounding box
    original_bb = ((0, 0, 0), (10, 10, 10))
    normal = (1, 1, 1)

    original_result = bottom_bounding_box_point(original_bb, normal)

    # Translated bounding box
    offset = (5, 3, 2)
    translated_bb = (
        (
            original_bb[0][0] + offset[0],
            original_bb[0][1] + offset[1],
            original_bb[0][2] + offset[2],
        ),
        (
            original_bb[1][0] + offset[0],
            original_bb[1][1] + offset[1],
            original_bb[1][2] + offset[2],
        ),
    )

    translated_result = bottom_bounding_box_point(translated_bb, normal)

    # The result should be the original result plus the offset
    expected = (
        original_result[0] + offset[0],
        original_result[1] + offset[1],
        original_result[2] + offset[2],
    )

    assert translated_result == expected
