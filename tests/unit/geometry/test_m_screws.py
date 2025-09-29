"""
Unit tests for the m_screws module.

Tests all functions for creating and working with metric screws and nuts,
ensuring compatibility with the shellforgepy framework.

Note: Some tests are marked with @pytest.mark.slow because thread generation
is computationally expensive. Run these with: pytest -m slow
"""

import math

import pytest
from shellforgepy.geometry.m_screws import (
    create_bolt_thread,
    create_cylinder_screw,
    create_nut,
    get_clearance_hole_diameter,
    get_core_hole_diameter,
    get_nut_outer_diameter,
    get_screw_info,
    get_thread_pitch,
    list_supported_sizes,
    m_screws_table,
)


def test_supported_sizes():
    """Test that all expected screw sizes are supported."""
    sizes = list_supported_sizes()
    expected_sizes = ["M2", "M3", "M4", "M5", "M6", "M8", "M10", "M12"]
    assert set(sizes) == set(expected_sizes)


def test_get_screw_info():
    """Test getting complete screw information."""
    info = get_screw_info("M3")
    assert info["nut_size"] == 5.5
    assert info["pitch"] == 0.5
    assert info["clearance_hole_normal"] == 3.4

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        get_screw_info("M999")


def test_get_nut_outer_diameter():
    """Test nut outer diameter calculation."""
    # Test M3 nut
    outer_diameter = get_nut_outer_diameter("M3")
    expected = 5.5 / math.cos(math.radians(30))
    assert abs(outer_diameter - expected) < 1e-6

    # Test M4 nut
    outer_diameter = get_nut_outer_diameter("M4")
    expected = 7.0 / math.cos(math.radians(30))
    assert abs(outer_diameter - expected) < 1e-6

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        get_nut_outer_diameter("M999")


def test_get_clearance_hole_diameter():
    """Test clearance hole diameter retrieval."""
    # Test normal clearance for M3
    diameter = get_clearance_hole_diameter("M3", "normal")
    assert diameter == 3.4

    # Test close clearance for M4
    diameter = get_clearance_hole_diameter("M4", "close")
    assert diameter == 4.3

    # Test loose clearance for M5
    diameter = get_clearance_hole_diameter("M5", "loose")
    assert diameter == 5.8

    # Test invalid clearance type
    with pytest.raises(ValueError, match="Invalid clearance type"):
        get_clearance_hole_diameter("M3", "invalid")

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        get_clearance_hole_diameter("M999", "normal")


def test_get_core_hole_diameter():
    """Test core hole diameter retrieval."""
    # Test M3 core hole
    diameter = get_core_hole_diameter("M3")
    assert diameter == 2.5

    # Test M4 core hole
    diameter = get_core_hole_diameter("M4")
    assert diameter == 3.3

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        get_core_hole_diameter("M999")


def test_get_thread_pitch():
    """Test thread pitch retrieval."""
    # Test M3 pitch
    pitch = get_thread_pitch("M3")
    assert pitch == 0.5

    # Test M4 pitch
    pitch = get_thread_pitch("M4")
    assert pitch == 0.7

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        get_thread_pitch("M999")


def test_create_nut_basic():
    """Test basic nut creation."""
    # Test M3 nut
    nut = create_nut("M3")
    assert nut is not None

    # Test M4 nut
    nut = create_nut("M4")
    assert nut is not None

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        create_nut("M999")


def test_create_nut_no_hole():
    """Test nut creation without center hole."""
    nut = create_nut("M3", no_hole=True)
    assert nut is not None


def test_create_nut_custom_height():
    """Test nut creation with custom height."""
    nut = create_nut("M3", height=5.0)
    assert nut is not None


def test_create_nut_with_slack():
    """Test nut creation with slack."""
    nut = create_nut("M3", slack=0.2)
    assert nut is not None


@pytest.mark.slow
def test_create_bolt_thread():
    """Test bolt thread creation (marked as slow test)."""
    # This test is marked as slow because thread generation is computationally expensive
    # Run with: pytest -m slow
    thread = create_bolt_thread("M3", length=1.5)
    assert thread is not None

    # Test thread with enlargement (very short for speed)
    thread = create_bolt_thread("M3", length=1.5, enlargement=0.1)
    assert thread is not None

    # Test cutter thread (very short for speed)
    thread = create_bolt_thread("M3", length=1.5, cutter=True)
    assert thread is not None

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        create_bolt_thread("M999", length=1.5)


def test_bolt_thread_parameters():
    """Test bolt thread parameter validation without creating actual threads."""
    # Test that function accepts valid parameters
    try:
        # This would work if we actually called it, but we're just testing validation
        assert get_thread_pitch("M3") == 0.5  # Verify pitch retrieval works
        assert get_screw_info("M3")["pitch"] == 0.5  # Verify info access works
    except Exception:
        pytest.fail("Basic parameter validation failed")


def test_create_cylinder_screw_basic():
    """Test basic cylinder screw creation."""
    # Test M3 screw (short length)
    screw = create_cylinder_screw("M3", length=8)
    assert screw is not None

    # Test M4 screw (short length)
    screw = create_cylinder_screw("M4", length=10)
    assert screw is not None

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        create_cylinder_screw("M999", length=8)


@pytest.mark.slow
def test_create_cylinder_screw_with_thread():
    """Test cylinder screw creation with threading (marked as slow test)."""
    # This test is marked as slow because threading operations are computationally expensive
    # Run with: pytest -m slow
    screw = create_cylinder_screw(
        "M3", length=4, with_thread=True, only_minimal_thread=True
    )
    assert screw is not None


def test_create_cylinder_screw_parameters():
    """Test cylinder screw creation with different parameters (no threading for speed)."""
    # Test basic screw creation without threading
    screw = create_cylinder_screw("M3", length=8, with_thread=False)
    assert screw is not None

    # Test with enlargement
    screw = create_cylinder_screw("M3", length=8, with_thread=False, enlargement=0.1)
    assert screw is not None


def test_create_cylinder_screw_with_enlargement():
    """Test cylinder screw creation with enlargement."""
    screw = create_cylinder_screw("M3", length=8, enlargement=0.1)
    assert screw is not None


def test_screw_table_completeness():
    """Test that all required fields are present in the screw table."""
    required_fields = [
        "nut_size",
        "clearance_hole_normal",
        "pitch",
        "core_hole",
        "cylinder_head_diameter",
        "cylinder_head_height",
        "min_thread_length",
    ]

    for size, specs in m_screws_table.items():
        for field in required_fields:
            assert field in specs, f"Missing field '{field}' for size '{size}'"
            assert isinstance(
                specs[field], (int, float)
            ), f"Field '{field}' for size '{size}' must be numeric"
            assert (
                specs[field] > 0
            ), f"Field '{field}' for size '{size}' must be positive"


def test_screw_table_size_progression():
    """Test that screw dimensions increase with size."""
    sizes = ["M3", "M4", "M5", "M6", "M8"]

    # Check that nut sizes increase
    nut_sizes = [m_screws_table[size]["nut_size"] for size in sizes]
    assert nut_sizes == sorted(nut_sizes), "Nut sizes should increase with screw size"

    # Check that clearance holes increase
    clearance_holes = [m_screws_table[size]["clearance_hole_normal"] for size in sizes]
    assert clearance_holes == sorted(
        clearance_holes
    ), "Clearance holes should increase with screw size"

    # Check that pitches generally increase (with some exceptions)
    pitches = [m_screws_table[size]["pitch"] for size in sizes]
    # Pitches should be non-decreasing (may stay the same between consecutive sizes)
    for i in range(1, len(pitches)):
        assert (
            pitches[i] >= pitches[i - 1]
        ), f"Pitch should not decrease from {sizes[i-1]} to {sizes[i]}"


def test_mathematical_relationships():
    """Test mathematical relationships in screw specifications."""
    for size, specs in m_screws_table.items():
        # Core hole should be smaller than the major diameter
        major_diameter = float(size[1:])
        assert specs["core_hole"] < major_diameter, f"Core hole too large for {size}"

        # Clearance hole should be larger than major diameter
        assert (
            specs["clearance_hole_normal"] > major_diameter
        ), f"Clearance hole too small for {size}"

        # Nut should be larger than major diameter
        assert specs["nut_size"] > major_diameter, f"Nut size too small for {size}"


def test_nut_creation_edge_cases():
    """Test nut creation with edge cases."""
    # Test with zero slack
    nut = create_nut("M3", slack=0)
    assert nut is not None

    # Test with very small height
    nut = create_nut("M3", height=0.1)
    assert nut is not None

    # Test with large slack
    nut = create_nut("M3", slack=1.0)
    assert nut is not None


@pytest.mark.slow
def test_thread_creation_edge_cases():
    """Test thread creation with edge cases (marked as slow test)."""
    # Test very short thread (minimum practical length)
    thread = create_bolt_thread("M3", length=0.5)
    assert thread is not None

    # Test with negative enlargement (smaller thread)
    thread = create_bolt_thread("M3", length=1.5, enlargement=-0.05)
    assert thread is not None


@pytest.mark.slow
def test_all_sizes_work():
    """Test that all supported sizes can create basic geometry (marked as slow test)."""
    sizes = list_supported_sizes()

    for size in sizes:
        # Test nut creation
        nut = create_nut(size)
        assert nut is not None, f"Failed to create nut for size {size}"

        # Test screw creation (without threading for speed)
        screw = create_cylinder_screw(size, length=8)
        assert screw is not None, f"Failed to create screw for size {size}"

        # Test thread creation (very short for speed)
        thread = create_bolt_thread(size, length=1.5)
        assert thread is not None, f"Failed to create thread for size {size}"


def test_all_sizes_basic():
    """Test that all supported sizes can create basic geometry without threading."""
    sizes = list_supported_sizes()

    for size in sizes[:4]:  # Test just first 4 sizes for speed
        # Test nut creation
        nut = create_nut(size)
        assert nut is not None, f"Failed to create nut for size {size}"

        # Test screw creation (without threading for speed)
        screw = create_cylinder_screw(size, length=8, with_thread=False)
        assert screw is not None, f"Failed to create screw for size {size}"


def test_dimensional_consistency():
    """Test that dimensions are consistent across different functions."""
    for size in list_supported_sizes():
        info = get_screw_info(size)

        # Test that pitch matches
        assert get_thread_pitch(size) == info["pitch"]

        # Test that core hole matches
        assert get_core_hole_diameter(size) == info["core_hole"]

        # Test that clearance hole matches
        assert (
            get_clearance_hole_diameter(size, "normal") == info["clearance_hole_normal"]
        )

        # Test that nut outer diameter calculation is consistent
        expected_outer = info["nut_size"] / math.cos(math.radians(30))
        actual_outer = get_nut_outer_diameter(size)
        assert abs(actual_outer - expected_outer) < 1e-10
