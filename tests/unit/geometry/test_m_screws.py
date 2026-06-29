"""
Unit tests for the m_screws module.

Tests all functions for creating and working with metric screws and nuts,
ensuring compatibility with the shellforgepy framework.

Note: Some tests are marked with @pytest.mark.slow because thread generation
is computationally expensive. Run these with: pytest -m slow
"""

import math
from pathlib import Path

import pytest
from shellforgepy.adapters._adapter import (
    create_box,
    create_cylinder,
    cut_parts,
    get_bounding_box,
    get_bounding_box_size,
    get_volume,
)
from shellforgepy.construct.bounding_box_helpers import get_zlen, get_zmax, get_zmin
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.geometry.m_screws import (
    MScrew,
    create_bolt_thread,
    create_conical_head_screw,
    create_cylinder_screw,
    create_hidden_nut_pocket_cutter,
    create_nut,
    create_self_threading_hole_cutter,
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
    expected_sizes = ["M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10", "M12"]
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


def test_create_hidden_nut_pocket_cutter_defaults():
    """Hidden nut pocket cutter should return a composite part with the expected height."""
    result = create_hidden_nut_pocket_cutter("M3")

    assert isinstance(result, LeaderFollowersCuttersPart)
    assert len(result.cutters) == 1

    expected_nut_height = m_screws_table["M3"]["nut_thickness"] + 0.4  # slack*2

    leader_bb = get_bounding_box(result.leader)
    assert math.isclose(
        get_zlen(leader_bb), expected_nut_height, rel_tol=1e-6, abs_tol=1e-6
    )

    cutter_bb = get_bounding_box(result.cutters[0])
    # Without a bottom cutter the geometry should start at z=0
    assert get_zmin(cutter_bb) >= -1e-6
    expected_total_height = expected_nut_height + 500  # default top_cutter_length
    assert math.isclose(
        get_zlen(cutter_bb), expected_total_height, rel_tol=1e-6, abs_tol=1e-4
    )


def test_create_hidden_nut_pocket_cutter_with_bottom():
    """Custom cutter lengths should change the overall bounding box as expected."""
    top_length = 10
    bottom_length = 5
    slack = 0.0

    result = create_hidden_nut_pocket_cutter(
        "M3",
        bottom_cutter_length=bottom_length,
        top_cutter_length=top_length,
        slack=slack,
    )

    assert isinstance(result, LeaderFollowersCuttersPart)
    assert len(result.cutters) == 1

    expected_nut_height = m_screws_table["M3"]["nut_thickness"] + slack * 2
    expected_total_height = bottom_length + expected_nut_height + top_length

    cutter_bb = get_bounding_box(result.cutters[0])
    assert math.isclose(
        get_zlen(cutter_bb), expected_total_height, rel_tol=1e-6, abs_tol=1e-5
    )
    assert math.isclose(get_zmin(cutter_bb), -bottom_length, abs_tol=1e-6)
    assert math.isclose(
        get_zmax(cutter_bb), expected_nut_height + top_length, abs_tol=1e-6
    )


def test_create_self_threading_hole_cutter_volume_and_height():
    """Self-threading cutter should sit between core and clearance cylinder volumes."""
    length = 8.0
    cutter = create_self_threading_hole_cutter("M3", length)

    assert cutter is not None

    cutter_bb = get_bounding_box(cutter)
    assert math.isclose(get_zlen(cutter_bb), length, rel_tol=1e-6, abs_tol=1e-6)

    core_cylinder = create_cylinder(get_core_hole_diameter("M3") / 2, length)
    clearance_cylinder = create_cylinder(
        get_clearance_hole_diameter("M3", "close") / 2, length
    )

    cutter_volume = get_volume(cutter)
    assert get_volume(core_cylinder) < cutter_volume
    assert cutter_volume < get_volume(clearance_cylinder)


def test_create_self_threading_hole_cutter_default_arguments_match_default_shape():
    """Explicit new defaults should preserve the original public behavior."""
    length = 8.0
    implicit = create_self_threading_hole_cutter("M3", length)
    explicit = create_self_threading_hole_cutter(
        "M3",
        length,
        core_radius_adjustment=0.0,
        lead_in=False,
    )

    implicit_bbox = get_bounding_box(implicit)
    explicit_bbox = get_bounding_box(explicit)
    assert tuple(implicit_bbox[0] + implicit_bbox[1]) == pytest.approx(
        explicit_bbox[0] + explicit_bbox[1],
        abs=1e-6,
    )
    assert get_volume(implicit) == pytest.approx(get_volume(explicit), rel=1e-6)


def test_create_self_threading_hole_cutter_core_radius_adjustment_tightens_valleys():
    """Negative core radius adjustment should tighten the lobes only."""
    length = 8.0
    clearance = get_clearance_hole_diameter("M3", "close")
    default = create_self_threading_hole_cutter("M3", length)
    tightened = create_self_threading_hole_cutter(
        "M3",
        length,
        core_radius_adjustment=-0.15,
    )

    tightened_size = get_bounding_box_size(tightened)
    assert get_zlen(get_bounding_box(tightened)) == pytest.approx(length)
    assert max(tightened_size[:2]) == pytest.approx(clearance, abs=0.005)
    assert get_volume(tightened) < get_volume(default)


def test_create_self_threading_hole_cutter_lead_in_preserves_length_and_rounds_entry():
    """The optional lead-in should stay inside length and round the local +Z entry."""
    length = 8.0
    clearance_radius = get_clearance_hole_diameter("M3", "close") / 2
    adjusted_core_radius = get_core_hole_diameter("M3") / 2 - 0.15
    lead_in_height = clearance_radius - adjusted_core_radius

    tightened = create_self_threading_hole_cutter(
        "M3",
        length,
        core_radius_adjustment=-0.15,
    )
    lead_in = create_self_threading_hole_cutter(
        "M3",
        length,
        core_radius_adjustment=-0.15,
        lead_in=True,
    )

    lead_in_bbox = get_bounding_box(lead_in)
    lead_in_size = get_bounding_box_size(lead_in)

    assert get_zmin(lead_in_bbox) == pytest.approx(0, abs=1e-6)
    assert get_zmax(lead_in_bbox) == pytest.approx(length, abs=1e-6)
    assert get_zlen(lead_in_bbox) == pytest.approx(length)
    assert lead_in_size[:2] == pytest.approx([2 * clearance_radius] * 2, abs=0.005)
    assert lead_in_height == pytest.approx(0.5)
    assert get_volume(lead_in) > get_volume(tightened)


def test_create_self_threading_hole_cutter_rejects_invalid_radius_adjustments():
    """Adjusted core radius must stay between zero and clearance radius."""
    with pytest.raises(ValueError, match="Adjusted core radius"):
        create_self_threading_hole_cutter("M3", 8, core_radius_adjustment=-2)

    with pytest.raises(ValueError, match="Adjusted core radius"):
        create_self_threading_hole_cutter("M3", 8, core_radius_adjustment=1)


def test_create_self_threading_hole_cutter_preview_artifacts():
    """Generate deterministic preview artifacts for visual inspection."""
    from shellforgepy.construct.alignment_operations import Alignment, align, translate
    from shellforgepy.produce.arrange_and_export import arrange_and_export
    from shellforgepy.produce.production_parts_model import PartList
    from shellforgepy.render.api import render_obj_views
    from shellforgepy.render.image import preferred_image_suffix

    if preferred_image_suffix() != ".png":
        pytest.skip("PNG preview generation requires Pillow")

    output_dir = (
        Path(__file__).resolve().parents[3]
        / "output"
        / "self_threading_hole_cutter_preview"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def export_obj_scene(part, name, color):
        parts = PartList()
        parts.add(part, name, color=color)
        return arrange_and_export(
            parts,
            script_file=__file__,
            export_base_name=name,
            export_directory=output_dir,
            export_stl=False,
            export_step=False,
            export_obj=True,
            export_individual_parts=False,
            preserve_model_coordinates=True,
        )

    cutter_length = 8
    block_height = 6
    preview_specs = [
        (
            create_self_threading_hole_cutter("M3", cutter_length),
            "self_threading_hole_cutter_default",
            (0.95, 0.38, 0.18),
        ),
        (
            create_self_threading_hole_cutter(
                "M3",
                cutter_length,
                core_radius_adjustment=-0.15,
            ),
            "self_threading_hole_cutter_tightened",
            (0.95, 0.52, 0.18),
        ),
        (
            create_self_threading_hole_cutter(
                "M3",
                cutter_length,
                core_radius_adjustment=-0.15,
                lead_in=True,
            ),
            "self_threading_hole_cutter_tightened_lead_in",
            (0.25, 0.62, 0.95),
        ),
    ]
    lead_in_cutaway = cut_parts(
        preview_specs[-1][0],
        create_box(4, 4, cutter_length + 2, origin=(-2, 0, -1)),
    )
    preview_specs.append(
        (
            lead_in_cutaway,
            "self_threading_hole_cutter_tightened_lead_in_cutaway",
            (0.25, 0.62, 0.95),
        )
    )

    block = create_box(16, 16, block_height)
    drilling_cutter = align(
        preview_specs[2][0],
        block,
        Alignment.CENTER,
        axes=[0, 1],
    )
    drilling_cutter = translate(0, 0, block_height - cutter_length)(drilling_cutter)
    drilled_block = cut_parts(block, drilling_cutter)
    drilled_block_cutaway = cut_parts(
        drilled_block,
        create_box(20, 10, 8, origin=(-2, -2, -1)),
    )
    preview_specs.append(
        (
            drilled_block,
            "self_threading_hole_drilled_block_lead_in",
            (0.86, 0.90, 0.78),
        )
    )
    preview_specs.append(
        (
            drilled_block_cutaway,
            "self_threading_hole_drilled_block_lead_in_cutaway",
            (0.86, 0.90, 0.78),
        )
    )

    preview_paths = []
    for part, name, color in preview_specs:
        obj_path = export_obj_scene(part, name, color=color)
        preview_paths.extend(
            render_obj_views(
                obj_path,
                output_dir=output_dir,
                views=("top", "front_angle"),
                width=768,
                height=768,
                filename_prefix=name,
            )
        )

    assert {path.suffix for path in preview_paths} == {".png"}
    assert {path.name for path in preview_paths} >= {
        "self_threading_hole_cutter_default_top.png",
        "self_threading_hole_cutter_tightened_top.png",
        "self_threading_hole_cutter_tightened_lead_in_front_angle.png",
        "self_threading_hole_cutter_tightened_lead_in_cutaway_front_angle.png",
        "self_threading_hole_drilled_block_lead_in_top.png",
        "self_threading_hole_drilled_block_lead_in_cutaway_front_angle.png",
    }


def test_create_self_threading_hole_cutter_preserves_lookup_errors():
    """Self-threading cutter should use the same size and clearance validation."""
    with pytest.raises(KeyError, match="Unsupported screw size"):
        create_self_threading_hole_cutter("M999", 8)

    with pytest.raises(ValueError, match="Invalid clearance type"):
        create_self_threading_hole_cutter("M3", 8, clearance_type="invalid")


def test_create_self_threading_hole_cutter_is_exported_from_simple():
    """The convenience facade should expose the public cutter helper."""
    from shellforgepy.simple import (
        create_self_threading_hole_cutter as simple_self_threading_hole_cutter,
    )

    cutter = simple_self_threading_hole_cutter(
        "M3",
        2,
        core_radius_adjustment=-0.05,
        lead_in=True,
    )
    assert cutter is not None


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


def test_m_screw_class():
    """Test the MScrew class functionality."""

    for size in list_supported_sizes():
        screw = MScrew.from_size(size)
        assert screw.size == size
        assert screw.pitch == get_thread_pitch(size)
        assert screw.nut_size == m_screws_table[size]["nut_size"]
    screw = MScrew.from_size("M3")
    assert screw.size == "M3"
    assert screw.pitch == 0.5
    assert screw.nut_size == 5.5

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        MScrew.from_size("M999")


def test_create_conical_head_screw_basic():
    """Test basic conical head screw creation without threading."""
    screw = create_conical_head_screw("M4", length=16)
    assert screw is not None

    screw = create_conical_head_screw("M5", length=16)
    assert screw is not None

    # Test with enlargement
    screw = create_conical_head_screw("M5", length=16, enlargement=0.1)
    assert screw is not None

    # Test unsupported size
    with pytest.raises(KeyError, match="Unsupported screw size"):
        create_conical_head_screw("M999", length=10)

    with pytest.raises(
        ValueError, match="Conical head dimensions are not defined for screw size"
    ):
        create_conical_head_screw("M12", length=10)


@pytest.mark.slow
def test_create_conical_head_screw_with_minimal_thread():
    """Test conical head screw with only_minimal_thread=True (marked as slow)."""
    screw = create_conical_head_screw(
        "M5", length=16, with_thread=True, only_minimal_thread=True
    )
    assert screw is not None
