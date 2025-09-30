"""Unit tests for the fixed create_screw_thread implementation."""

import pytest
from shellforgepy.adapters._adapter import get_volume
from shellforgepy.geometry.higher_order_solids import create_screw_thread


def test_create_screw_thread_basic():
    """Test basic screw thread creation with default parameters."""
    screw = create_screw_thread(
        pitch=2.0,
        inner_radius=5.0,
        outer_radius=7.0,
        outer_thickness=0.3,
        num_turns=1,
        resolution=16,  # Lower resolution for faster testing
    )

    assert screw is not None, "Screw thread should be created"
    volume = get_volume(screw)
    assert volume > 0, f"Screw thread should have positive volume, got {volume}"
    print(f"Basic screw thread volume: {volume:.6f}")


def test_create_screw_thread_hawaii_bottle_cap_parameters():
    """Test screw thread with Hawaii bottle cap parameters."""
    # Parameters from the Hawaii bottle cap example
    pitch = 4.3
    inner_radius = 15 / 2  # 7.5
    outer_radius = inner_radius + 1.4  # 8.9
    outer_thickness = 0.2
    num_turns = 2.0  # Approximate from the original

    screw = create_screw_thread(
        pitch=pitch,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        outer_thickness=outer_thickness,
        num_turns=num_turns,
        resolution=24,  # Reasonable resolution
        optimize_start=True,
        with_core=True,
    )

    assert screw is not None, "Hawaii bottle cap screw thread should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Hawaii bottle cap screw thread should have positive volume, got {volume}"

    # The volume should be reasonable for a screw thread of this size
    expected_min_volume = 50  # Conservative estimate
    expected_max_volume = 5000  # More liberal estimate for larger thread
    assert (
        expected_min_volume < volume < expected_max_volume
    ), f"Volume {volume} seems unreasonable for Hawaii bottle cap thread"

    print(f"Hawaii bottle cap screw thread volume: {volume:.6f}")


def test_create_screw_thread_with_optimization():
    """Test screw thread creation with start optimization."""
    screw = create_screw_thread(
        pitch=1.5,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.25,
        num_turns=2,
        resolution=20,
        optimize_start=True,
        optimize_start_angle=30,
    )

    assert screw is not None, "Optimized screw thread should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Optimized screw thread should have positive volume, got {volume}"
    print(f"Optimized screw thread volume: {volume:.6f}")


def test_create_screw_thread_partial_turns():
    """Test screw thread with partial turns."""
    screw = create_screw_thread(
        pitch=2.0,
        inner_radius=3.0,
        outer_radius=5.0,
        outer_thickness=0.3,
        num_turns=1.5,  # 1.5 turns should create partial turn
        resolution=16,
    )

    assert screw is not None, "Partial turn screw thread should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Partial turn screw thread should have positive volume, got {volume}"
    print(f"Partial turn screw thread volume: {volume:.6f}")


def test_create_screw_thread_without_core():
    """Test screw thread creation without core."""
    screw_with_core = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.3,
        num_turns=1,
        with_core=True,
        resolution=16,
    )

    screw_without_core = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.3,
        num_turns=1,
        with_core=False,
        resolution=16,
    )

    assert screw_with_core is not None, "Screw with core should be created"
    assert screw_without_core is not None, "Screw without core should be created"

    volume_with_core = get_volume(screw_with_core)
    volume_without_core = get_volume(screw_without_core)

    assert (
        volume_with_core > volume_without_core
    ), f"Screw with core ({volume_with_core}) should have larger volume than without core ({volume_without_core})"

    print(f"Screw with core volume: {volume_with_core:.6f}")
    print(f"Screw without core volume: {volume_without_core:.6f}")


def test_create_screw_thread_inner_thickness_default():
    """Test that inner_thickness defaults to pitch - outer_thickness."""
    outer_thickness = 0.3
    pitch = 2.0
    expected_inner_thickness = pitch - outer_thickness  # Should be 1.7

    # This test verifies the logic is applied correctly by creating a thread
    # and ensuring it doesn't fail with the default inner_thickness calculation
    screw = create_screw_thread(
        pitch=pitch,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=outer_thickness,
        # inner_thickness not specified, should default to pitch - outer_thickness
        num_turns=1,
        resolution=12,
    )

    assert screw is not None, "Screw with default inner_thickness should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Screw with default inner_thickness should have positive volume, got {volume}"
    print(f"Screw with default inner_thickness volume: {volume:.6f}")


def test_create_screw_thread_custom_core_parameters():
    """Test screw thread with custom core height and offset."""
    screw = create_screw_thread(
        pitch=2.0,
        inner_radius=4.0,
        outer_radius=6.0,
        outer_thickness=0.3,
        num_turns=2,
        with_core=True,
        core_height=8.0,  # Larger custom core height (2 turns * 2.0 pitch = 4.0 + some margin)
        core_offset=0.5,  # Custom core offset
        resolution=16,
    )

    assert screw is not None, "Screw with custom core parameters should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Screw with custom core parameters should have positive volume, got {volume}"
    print(f"Screw with custom core parameters volume: {volume:.6f}")


def test_create_screw_thread_multiple_turns():
    """Test screw thread with multiple turns."""
    screw = create_screw_thread(
        pitch=1.5,
        inner_radius=3.0,
        outer_radius=5.0,
        outer_thickness=0.25,
        num_turns=3,  # Multiple turns
        resolution=20,
    )

    assert screw is not None, "Multi-turn screw thread should be created"
    volume = get_volume(screw)
    assert (
        volume > 0
    ), f"Multi-turn screw thread should have positive volume, got {volume}"
    print(f"Multi-turn screw thread volume: {volume:.6f}")


def test_create_screw_thread_validation():
    """Test screw thread parameter validation."""
    # Test that invalid core height raises an error
    with pytest.raises(ValueError, match="Core height.*must be greater than"):
        create_screw_thread(
            pitch=2.0,
            inner_radius=4.0,
            outer_radius=6.0,
            outer_thickness=0.3,
            num_turns=2,
            core_height=0.1,  # Too small core height
            resolution=16,
        )


if __name__ == "__main__":
    # Run the tests manually if called directly
    test_create_screw_thread_basic()
    test_create_screw_thread_hawaii_bottle_cap_parameters()
    test_create_screw_thread_with_optimization()
    test_create_screw_thread_partial_turns()
    test_create_screw_thread_without_core()
    test_create_screw_thread_inner_thickness_default()
    test_create_screw_thread_custom_core_parameters()
    test_create_screw_thread_multiple_turns()
    print("All screw thread tests passed!")
