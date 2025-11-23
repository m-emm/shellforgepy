"""Unit tests for sheet metal geometry functions."""

import math

import pytest
from shellforgepy.adapters._adapter import get_bounding_box, get_volume
from shellforgepy.geometry.sheet_metal import (
    create_sheet_metal_bend,
    create_sheet_metal_bracket,
    create_sheet_metal_hem,
    create_sheet_metal_wall,
)


class TestSheetMetalBend:
    """Tests for create_sheet_metal_bend function."""

    def test_basic_bend_creation(self):
        """Test basic 90° bend creation."""
        bend = create_sheet_metal_bend(thickness=2.0, length=50.0)

        # Should create a valid solid
        assert bend is not None

        # Check that it has some volume
        volume = get_volume(bend)
        assert volume > 0

        # For a 90° bend with thickness 2, inner radius 2, outer radius 4
        # Volume should be approximately: (π/4) * (4² - 2²) * 50 = 150π/2
        # The 90° bend is a quarter of a full annulus
        expected_volume = math.pi * 0.25 * (4**2 - 2**2) * 50
        assert (
            abs(volume - expected_volume) < expected_volume * 0.2
        )  # Allow 20% tolerance

    def test_custom_inner_radius(self):
        """Test bend with custom inner radius."""
        bend = create_sheet_metal_bend(thickness=1.5, length=30.0, inner_radius=3.0)

        volume = get_volume(bend)
        assert volume > 0

        # Volume with inner radius 3, outer radius 4.5
        # 90° sector volume: (π/4) * (4.5² - 3.0²) * 30
        expected_volume = math.pi * 0.25 * (4.5**2 - 3.0**2) * 30
        assert (
            abs(volume - expected_volume) < expected_volume * 0.2
        )  # Allow 20% tolerance

    def test_custom_bend_angle(self):
        """Test bend with custom angle."""
        bend_45 = create_sheet_metal_bend(thickness=2.0, length=40.0, bend_angle=45.0)

        bend_90 = create_sheet_metal_bend(thickness=2.0, length=40.0, bend_angle=90.0)

        volume_45 = get_volume(bend_45)
        volume_90 = get_volume(bend_90)

        # 45° bend should have half the volume of 90° bend
        assert abs(volume_45 * 2 - volume_90) < volume_90 * 0.1

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="Thickness must be positive"):
            create_sheet_metal_bend(thickness=0, length=10)

        with pytest.raises(ValueError, match="Length must be positive"):
            create_sheet_metal_bend(thickness=1, length=0)

        with pytest.raises(ValueError, match="Bend angle must be between 0 and 180"):
            create_sheet_metal_bend(thickness=1, length=10, bend_angle=200)

        with pytest.raises(ValueError, match="Inner radius must be positive"):
            create_sheet_metal_bend(thickness=1, length=10, inner_radius=0)


class TestSheetMetalWall:
    """Tests for create_sheet_metal_wall function."""

    def test_basic_wall_creation(self):
        """Test basic wall with bend creation."""
        wall = create_sheet_metal_wall(thickness=1.5, length=80.0, height=20.0)

        assert wall is not None
        volume = get_volume(wall)
        assert volume > 0

        # Wall should have significant volume
        # Approximate: flat section + bend section
        flat_volume = 1.5 * 80.0 * 20.0  # thickness * length * height
        assert volume > flat_volume * 0.5  # Should be at least half of flat section

    def test_wall_without_bend(self):
        """Test wall creation without bend."""
        wall_with_bend = create_sheet_metal_wall(
            thickness=2.0, length=60.0, height=15.0, with_bend=True
        )

        wall_without_bend = create_sheet_metal_wall(
            thickness=2.0, length=60.0, height=15.0, with_bend=False
        )

        volume_with = get_volume(wall_with_bend)
        volume_without = get_volume(wall_without_bend)

        # Wall with bend should have more volume
        assert volume_with > volume_without

    def test_custom_bend_angle(self):
        """Test wall with custom bend angle."""
        wall = create_sheet_metal_wall(
            thickness=1.0, length=50.0, height=10.0, bend_angle=60.0
        )

        assert wall is not None
        assert get_volume(wall) > 0

    def test_wall_bounding_box(self):
        """Test that wall has reasonable bounding box dimensions."""
        thickness = 2.0
        length = 100.0
        height = 25.0

        wall = create_sheet_metal_wall(
            thickness=thickness, length=length, height=height
        )

        bbox = get_bounding_box(wall)

        # Check that bounding box makes sense
        x_size = bbox[1][0] - bbox[0][0]
        y_size = bbox[1][1] - bbox[0][1]
        z_size = bbox[1][2] - bbox[0][2]

        # Should have dimensions roughly related to input parameters
        assert x_size > 0
        assert y_size > 0
        assert z_size > 0

    def test_invalid_wall_parameters(self):
        """Test invalid parameters for wall creation."""
        with pytest.raises(ValueError, match="Thickness must be positive"):
            create_sheet_metal_wall(thickness=-1, length=10, height=5)

        with pytest.raises(ValueError, match="Length must be positive"):
            create_sheet_metal_wall(thickness=1, length=0, height=5)

        with pytest.raises(ValueError, match="Height must be positive"):
            create_sheet_metal_wall(thickness=1, length=10, height=0)


class TestSheetMetalBracket:
    """Tests for create_sheet_metal_bracket function."""

    def test_basic_bracket_creation(self):
        """Test basic L-bracket creation."""
        bracket = create_sheet_metal_bracket(
            thickness=3.0, width=40.0, height=30.0, flange_width=20.0
        )

        assert bracket is not None
        volume = get_volume(bracket)
        assert volume > 0

        # Approximate volume: vertical section + horizontal section
        vertical_vol = 3.0 * 40.0 * 30.0
        horizontal_vol = 20.0 * 40.0 * 3.0
        expected_vol = vertical_vol + horizontal_vol

        # Should be close to expected (allowing for overlap at joint)
        assert volume > expected_vol * 0.8
        assert volume < expected_vol * 1.2

    def test_bracket_with_bend_relief(self):
        """Test bracket with bend relief."""
        bracket_no_relief = create_sheet_metal_bracket(
            thickness=2.0, width=30.0, height=25.0, flange_width=15.0
        )

        bracket_with_relief = create_sheet_metal_bracket(
            thickness=2.0, width=30.0, height=25.0, flange_width=15.0, bend_relief=1.0
        )

        vol_no_relief = get_volume(bracket_no_relief)
        vol_with_relief = get_volume(bracket_with_relief)

        # With relief should have slightly less volume
        assert vol_with_relief < vol_no_relief

    def test_invalid_bracket_parameters(self):
        """Test invalid parameters for bracket creation."""
        with pytest.raises(ValueError, match="Thickness must be positive"):
            create_sheet_metal_bracket(thickness=0, width=10, height=10, flange_width=5)

        with pytest.raises(ValueError, match="Bend relief must be non-negative"):
            create_sheet_metal_bracket(
                thickness=1, width=10, height=10, flange_width=5, bend_relief=-1
            )


class TestSheetMetalHem:
    """Tests for create_sheet_metal_hem function."""

    def test_open_hem_creation(self):
        """Test open hem creation."""
        hem = create_sheet_metal_hem(
            thickness=1.0, length=60.0, hem_width=4.0, hem_type="open"
        )

        assert hem is not None
        assert get_volume(hem) > 0

    def test_closed_hem_creation(self):
        """Test closed hem creation."""
        hem = create_sheet_metal_hem(
            thickness=1.5, length=50.0, hem_width=5.0, hem_type="closed"
        )

        assert hem is not None
        assert get_volume(hem) > 0

    def test_teardrop_hem_creation(self):
        """Test teardrop hem creation."""
        hem = create_sheet_metal_hem(
            thickness=1.2, length=40.0, hem_width=3.0, hem_type="teardrop"
        )

        assert hem is not None
        assert get_volume(hem) > 0

    def test_hem_volume_comparison(self):
        """Test that different hem types produce different volumes."""
        params = dict(thickness=1.0, length=30.0, hem_width=4.0)

        open_hem = create_sheet_metal_hem(**params, hem_type="open")
        closed_hem = create_sheet_metal_hem(**params, hem_type="closed")
        teardrop_hem = create_sheet_metal_hem(**params, hem_type="teardrop")

        open_vol = get_volume(open_hem)
        closed_vol = get_volume(closed_hem)
        teardrop_vol = get_volume(teardrop_hem)

        # All should have positive volume
        assert open_vol > 0
        assert closed_vol > 0
        assert teardrop_vol > 0

        # Closed should generally have more volume than open
        assert closed_vol > open_vol

    def test_invalid_hem_parameters(self):
        """Test invalid parameters for hem creation."""
        with pytest.raises(ValueError, match="Hem type must be"):
            create_sheet_metal_hem(
                thickness=1, length=10, hem_width=3, hem_type="invalid"
            )

        with pytest.raises(ValueError, match="Hem width must be positive"):
            create_sheet_metal_hem(thickness=1, length=10, hem_width=0, hem_type="open")


class TestSheetMetalIntegration:
    """Integration tests combining multiple sheet metal functions."""

    def test_wall_and_bend_integration(self):
        """Test that wall and bend functions work well together."""
        # Create individual components
        bend = create_sheet_metal_bend(thickness=2.0, length=50.0)
        wall = create_sheet_metal_wall(
            thickness=2.0, length=50.0, height=20.0, with_bend=False
        )

        # Both should be valid
        assert bend is not None
        assert wall is not None
        assert get_volume(bend) > 0
        assert get_volume(wall) > 0

        # Should be able to fuse them (basic compatibility test)
        combined = wall.fuse(bend)
        assert get_volume(combined) > 0

    def test_bracket_dimensions_consistency(self):
        """Test that bracket dimensions are consistent with input parameters."""
        thickness = 2.5
        width = 50.0
        height = 35.0
        flange_width = 25.0

        bracket = create_sheet_metal_bracket(
            thickness=thickness, width=width, height=height, flange_width=flange_width
        )

        bbox = get_bounding_box(bracket)

        # Check that bounding box dimensions make sense
        x_size = bbox[1][0] - bbox[0][0]
        y_size = bbox[1][1] - bbox[0][1]
        z_size = bbox[1][2] - bbox[0][2]

        # Should have reasonable dimensions relative to inputs
        assert x_size >= thickness  # At least the thickness
        assert y_size >= width * 0.8  # Most of the width (allowing for tolerances)
        assert z_size >= height * 0.8  # Most of the height

    def test_manufacturing_realistic_dimensions(self):
        """Test with realistic manufacturing dimensions."""
        # Test with typical sheet metal thicknesses and dimensions
        test_cases = [
            {"thickness": 0.8, "length": 100, "height": 25},  # Thin sheet
            {"thickness": 1.5, "length": 150, "height": 40},  # Medium sheet
            {"thickness": 3.0, "length": 200, "height": 50},  # Thick sheet
        ]

        for case in test_cases:
            wall = create_sheet_metal_wall(**case)
            bend = create_sheet_metal_bend(
                thickness=case["thickness"], length=case["length"]
            )

            # Both should create valid geometry
            assert wall is not None
            assert bend is not None
            assert get_volume(wall) > 0
            assert get_volume(bend) > 0
