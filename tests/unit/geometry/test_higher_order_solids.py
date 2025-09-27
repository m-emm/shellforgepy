import numpy as np
from shellforgepy.adapters.simple import get_volume
from shellforgepy.geometry.higher_order_solids import (
    create_hex_prism,
    create_trapezoid,
    directed_cylinder_at,
)


def test_create_hex_prism():
    prism = create_hex_prism(diameter=10, thickness=5, origin=(0, 0, 0))
    assert prism is not None
    # Further assertions can be added based on the expected properties of the prism


def test_create_trapezoid():
    trapezoid = create_trapezoid(
        base_length=10,
        top_length=6,
        height=5,
        thickness=3,
        top_shift=1.0,
    )
    assert trapezoid is not None
    # @todo add further assertions based on expected properties of the trapezoid


def test_directed_cylinder():
    """Test directed cylinder creation."""

    # Test cylinder along Z axis (should be same as basic cylinder)
    cyl_z = directed_cylinder_at(
        base_point=(0, 0, 0), direction=(0, 0, 1), radius=5, height=10
    )
    expected_volume = np.pi * 5**2 * 10
    assert np.allclose(get_volume(cyl_z), expected_volume, rtol=1e-5)

    # Test cylinder along X axis
    cyl_x = directed_cylinder_at(
        base_point=(0, 0, 0), direction=(1, 0, 0), radius=5, height=10
    )
    assert np.allclose(get_volume(cyl_x), expected_volume, rtol=1e-5)

    # Test cylinder along arbitrary direction
    direction = (1, 1, 1)  # diagonal direction
    cyl_diag = directed_cylinder_at(
        base_point=(5, 5, 5), direction=direction, radius=3, height=8
    )
    expected_volume = np.pi * 3**2 * 8
    assert np.allclose(get_volume(cyl_diag), expected_volume, rtol=1e-5)
