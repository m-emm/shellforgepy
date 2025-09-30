import numpy as np
from shellforgepy.adapters._adapter import get_volume
from shellforgepy.geometry.higher_order_solids import (
    create_hex_prism,
    create_ring,
    create_screw_thread,
    directed_cylinder_at,
)


def test_create_hex_prism():
    prism = create_hex_prism(diameter=10, thickness=5, origin=(0, 0, 0))
    assert prism is not None
    # Further assertions can be added based on the expected properties of the prism


def test_create_screw_thread():
    """Test screw thread creation with trapezoidal snake geometry."""
    # Test basic screw thread creation
    screw = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=2,
        resolution=20,
        with_core=True,
    )

    assert screw is not None
    # Check that the screw has reasonable volume
    assert screw.Volume() > 0

    # Test without core
    screw_no_core = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=1,
        resolution=16,
        with_core=False,
    )

    assert screw_no_core is not None
    assert get_volume(screw_no_core) > 0

    # Screw with core should have more volume than without
    screw_with_core = create_screw_thread(
        pitch=2.0,
        inner_radius=8.0,
        outer_radius=10.0,
        outer_thickness=0.5,
        num_turns=1,
        resolution=16,
        with_core=True,
    )

    assert screw_with_core.Volume() > screw_no_core.Volume()


def test_create_screw_thread_parameters():
    """Test screw thread with various parameters."""
    # Test with optimization
    screw_optimized = create_screw_thread(
        pitch=1.5,
        inner_radius=5.0,
        outer_radius=7.0,
        outer_thickness=0.3,
        num_turns=3,
        resolution=24,
        optimize_start=True,
        optimize_start_angle=30,
    )

    assert screw_optimized is not None
    assert get_volume(screw_optimized) > 0

    # Test with custom inner thickness
    screw_custom = create_screw_thread(
        pitch=2.5,
        inner_radius=6.0,
        outer_radius=8.5,
        outer_thickness=0.4,
        inner_thickness=0.6,
        num_turns=1.5,
        resolution=30,
    )

    assert screw_custom is not None
    assert get_volume(screw_custom) > 0

    # Test with core offset
    screw_offset = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.3,
        num_turns=2,
        resolution=16,
        core_offset=0.2,  # Smaller core offset
        core_height=7.0,  # Larger core height to accommodate offset
    )

    assert screw_offset is not None
    assert get_volume(screw_offset) > 0


def test_create_screw_thread_edge_cases():
    """Test screw thread edge cases and validation."""
    # Test minimal parameters
    minimal_screw = create_screw_thread(
        pitch=1.0,
        inner_radius=2.0,
        outer_radius=3.0,
        outer_thickness=0.2,
        num_turns=0.5,
        resolution=8,
    )

    assert minimal_screw is not None
    assert get_volume(minimal_screw) > 0

    # Test high resolution
    high_res_screw = create_screw_thread(
        pitch=2.0,
        inner_radius=5.0,
        outer_radius=7.0,
        outer_thickness=0.4,
        num_turns=1,
        resolution=60,
    )

    assert high_res_screw is not None
    assert get_volume(high_res_screw) > 0
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


def test_create_ring():
    """Test ring creation."""
    # Basic ring
    ring = create_ring(outer_radius=10, inner_radius=5, height=8)
    assert ring is not None

    # Calculate expected volume: outer cylinder - inner cylinder
    outer_volume = np.pi * 10**2 * 8
    inner_volume = np.pi * 5**2 * 8
    expected_volume = outer_volume - inner_volume
    assert np.allclose(get_volume(ring), expected_volume, rtol=1e-3)

    # Test with origin offset
    ring_offset = create_ring(
        outer_radius=10, inner_radius=5, height=8, origin=(5, 5, 5)
    )
    assert ring is not None
    assert np.allclose(get_volume(ring_offset), expected_volume, rtol=1e-3)

    # Test with partial angle
    ring_partial = create_ring(outer_radius=10, inner_radius=5, height=8, angle=180)
    assert ring_partial is not None
    # Half the volume for 180 degrees
    assert np.allclose(get_volume(ring_partial), expected_volume / 2, rtol=1e-3)

    # Test error case: inner radius >= outer radius
    try:
        create_ring(outer_radius=5, inner_radius=10, height=8)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_create_screw_thread():
    """Test screw thread creation."""
    # Basic thread
    thread = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=2,
        resolution=20,  # Lower resolution for faster testing
    )
    assert thread is not None

    # Thread should have some volume
    volume = get_volume(thread)
    assert volume > 0

    # Test thread without core
    thread_no_core = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=1,
        with_core=False,
        resolution=20,
    )
    assert thread_no_core is not None

    # Thread without core should have less volume than with core
    volume_no_core = get_volume(thread_no_core)
    assert volume_no_core > 0

    # Test with optimize_start
    thread_optimized = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=5.0,
        outer_thickness=0.5,
        num_turns=1,
        optimize_start=True,
        resolution=20,
    )
    assert thread_optimized is not None
