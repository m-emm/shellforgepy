import json
import tempfile
from pathlib import Path

import pytest
from shellforgepy.adapters._adapter import get_volume
from shellforgepy.simple import *

try:
    import FreeCAD

    if FreeCAD is not None:
        freecad_available = True
    else:
        freecad_available = False
except ImportError:
    freecad_available = False


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_collector_basic():
    """Test PartCollector basic functionality."""
    collector = PartCollector()
    assert collector.part is None

    # Create a simple box and add it
    box1 = create_basic_box(10, 10, 10)
    result = collector.fuse(box1)
    assert collector.part is not None
    assert result is collector.part


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_collector_multiple_fuse():
    """Test PartCollector with multiple parts."""
    collector = PartCollector()

    # Create two boxes
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5, origin=(15, 0, 0))

    collector.fuse(box1)
    collector.fuse(box2)

    assert collector.part is not None
    # The fused part should be larger than individual parts
    min_point, max_point = get_bounding_box(collector.part)
    assert max_point[0] > 15  # Should extend past the second box


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_collector_cut():
    """Test PartCollector cut functionality."""
    collector = PartCollector()

    # Create two overlapping boxes
    box1 = create_basic_box(20, 20, 20)
    box2 = create_basic_box(10, 10, 10, origin=(5, 5, 5))  # Overlapping with box1

    # Add the first box
    collector.fuse(box1)
    # Use get_volume function to handle both FreeCAD and CadQuery properly
    original_volume = get_volume(collector.part)

    # Cut the second box from the first
    result = collector.cut(box2)

    # The result should be the part and should have less volume
    assert result is collector.part
    cut_volume = get_volume(collector.part)
    assert cut_volume < original_volume


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_named_part_basic():
    """Test NamedPart basic functionality."""
    box = create_basic_box(10, 10, 10)
    named_part = NamedPart("test_box", box)

    assert named_part.name == "test_box"
    assert named_part.part is not None


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_named_part_copy():
    """Test NamedPart copy functionality."""
    box = create_basic_box(10, 10, 10)
    named_part = NamedPart("test_box", box)

    copied_part = named_part.copy()
    assert copied_part.name == named_part.name
    assert copied_part.part is not named_part.part  # Different objects


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_named_part_translate():
    """Test NamedPart translation."""
    box = create_basic_box(10, 10, 10)
    named_part = NamedPart("test_box", box)

    # Get original position
    min_point_orig, _ = get_bounding_box(named_part.part)

    # Translate the part
    translated_part = named_part.translate((5, 10, 15))

    # Check that it moved
    min_point_new, _ = get_bounding_box(translated_part.part)
    assert abs(min_point_new[0] - (min_point_orig[0] + 5)) < 1e-6
    assert abs(min_point_new[1] - (min_point_orig[1] + 10)) < 1e-6
    assert abs(min_point_new[2] - (min_point_orig[2] + 15)) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_named_part_rotate():
    """Test NamedPart rotation."""
    box = create_basic_box(20, 10, 5)  # Longer in X direction
    named_part = NamedPart("test_box", box)

    # Rotate 90 degrees around Z axis using native FreeCAD signature: rotate(base, dir, degree)
    if freecad_available:
        from FreeCAD import Base

        base_vec = Base.Vector(0, 0, 0)  # center point
        dir_vec = Base.Vector(0, 0, 1)  # axis direction
        rotated_part = named_part.rotate(base_vec, dir_vec, 90)
    else:
        # For CadQuery compatibility, use different rotation approach
        rotated_part = named_part.rotate((0, 0, 0), (0, 0, 1), 90)

    # After rotation, dimensions should swap (roughly)
    min_point, max_point = get_bounding_box(rotated_part)
    width = max_point[0] - min_point[0]
    depth = max_point[1] - min_point[1]

    # Original was 20x10, after 90° rotation should be roughly 10x20
    assert abs(width - 10) < 1e-6
    assert abs(depth - 20) < 1e-6


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_list_basic():
    """Test PartList basic functionality."""
    part_list = PartList()
    assert len(part_list) == 0

    box = create_basic_box(10, 10, 10)
    part_list.add(box, "test_box")

    assert len(part_list) == 1
    assert part_list[0].name == "test_box"


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_list_duplicate_name():
    """Test PartList prevents duplicate names."""
    part_list = PartList()
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    part_list.add(box1, "test_box")

    with pytest.raises(ValueError, match="already exists"):
        part_list.add(box2, "test_box")


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_list_with_options():
    """Test PartList with various options."""
    part_list = PartList()
    box = create_basic_box(10, 10, 10)

    part_list.add(
        box,
        "test_box",
        flip=True,
        skip_in_production=True,
        prod_rotation_angle=45.0,
        prod_rotation_axis=(0, 0, 1),
    )

    part_info = part_list[0]
    assert part_info.flip is True
    assert part_info.skip_in_production is True
    assert part_info.prod_rotation_angle == 45.0
    assert part_info.prod_rotation_axis == (0.0, 0.0, 1.0)


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_part_list_as_list():
    """Test PartList as_list conversion."""
    part_list = PartList()
    box = create_basic_box(10, 10, 10)
    part_list.add(box, "test_box", flip=True)

    parts_dict_list = part_list.as_list()
    assert len(parts_dict_list) == 1
    assert parts_dict_list[0]["name"] == "test_box"
    assert parts_dict_list[0]["flip"] is True


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_export_solid_to_stl():
    """Test STL export functionality."""
    box = create_basic_box(10, 10, 10)

    with tempfile.TemporaryDirectory() as temp_dir:
        stl_path = Path(temp_dir) / "test_box.stl"
        export_solid_to_stl(box, stl_path)

        assert stl_path.exists()
        assert stl_path.stat().st_size > 0


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_arrange_and_export_parts_basic():
    """Test basic arrange and export functionality."""
    # Create some test parts
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    parts_list = [{"name": "box1", "part": box1}, {"name": "box2", "part": box2}]

    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = arrange_and_export_parts(
            parts_list,
            prod_gap=2.0,
            bed_width=50.0,
            script_file="test_script.py",
            export_directory=temp_dir,
        )

        assert result_path.exists()
        assert result_path.name == "test_script.stl"

        # Check individual part files were created
        box1_path = Path(temp_dir) / "test_script_box1.stl"
        box2_path = Path(temp_dir) / "test_script_box2.stl"
        assert box1_path.exists()
        assert box2_path.exists()


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_arrange_and_export_with_part_list():
    """Test arrange and export with PartList."""
    part_list = PartList()
    box1 = create_basic_box(10, 10, 10)
    box2 = create_basic_box(5, 5, 5)

    part_list.add(box1, "box1")
    part_list.add(box2, "box2", skip_in_production=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test normal mode
        result_path = arrange_and_export_parts(
            part_list,
            prod_gap=2.0,
            bed_width=50.0,
            script_file="test_script.py",
            export_directory=temp_dir,
        )

        assert result_path.exists()

        # Test production mode (should skip box2)
        result_path_prod = arrange_and_export_parts(
            part_list,
            prod_gap=2.0,
            bed_width=50.0,
            script_file="test_prod.py",
            export_directory=temp_dir,
            prod=True,
        )

        assert result_path_prod.exists()
        # Only box1 should be exported in production mode
        box1_prod_path = Path(temp_dir) / "test_prod_box1.stl"
        box2_prod_path = Path(temp_dir) / "test_prod_box2.stl"
        assert box1_prod_path.exists()
        assert not box2_prod_path.exists()


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_arrange_and_export_with_process_data():
    """Test arrange and export with process data."""
    box = create_basic_box(10, 10, 10)
    parts_list = [{"name": "box", "part": box}]

    process_data = {"temperature": 200, "speed": 50}

    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = arrange_and_export_parts(
            parts_list,
            prod_gap=2.0,
            bed_width=50.0,
            script_file="test_script.py",
            export_directory=temp_dir,
            process_data=process_data,
        )

        assert result_path.exists()

        # Check process data file was created
        process_path = Path(temp_dir) / "test_script_process.json"
        assert process_path.exists()

        with process_path.open() as f:
            saved_data = json.load(f)

        assert saved_data["temperature"] == 200
        assert saved_data["speed"] == 50
        assert "part_file" in saved_data


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_arrange_and_export_max_build_height():
    """Test arrange and export with max build height constraint."""
    # Create a tall box
    tall_box = create_basic_box(10, 10, 30)
    parts_list = [{"name": "tall_box", "part": tall_box}]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Should fail with max_build_height constraint
        with pytest.raises(ValueError, match="exceeds max_build_height"):
            arrange_and_export_parts(
                parts_list,
                prod_gap=2.0,
                bed_width=50.0,
                script_file="test_script.py",
                export_directory=temp_dir,
                prod=True,
                max_build_height=20.0,
            )


@pytest.mark.skipif(not freecad_available, reason="FreeCAD not available")
def test_arrange_and_export_empty_parts():
    """Test arrange and export with empty parts list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="No parts provided"):
            arrange_and_export_parts(
                [],
                prod_gap=2.0,
                bed_width=50.0,
                script_file="test_script.py",
                export_directory=temp_dir,
            )
