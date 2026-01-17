"""Tests for OBJ export functionality in the CadQuery adapter."""

import os
import tempfile


def test_export_solid_to_obj_simple():
    """Test exporting a simple solid to OBJ without color."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        create_box,
        export_solid_to_obj,
    )

    box = create_box(10, 20, 30)

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "box.obj")
        export_solid_to_obj(box, obj_path)

        assert os.path.exists(obj_path)
        with open(obj_path) as f:
            content = f.read()
        assert "v " in content  # Has vertices
        assert "f " in content  # Has faces
        assert "mtllib" not in content  # No material reference without color


def test_export_solid_to_obj_with_color():
    """Test exporting a solid to OBJ with a color creates MTL file."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        create_box,
        export_solid_to_obj,
    )

    box = create_box(10, 20, 30)
    color = (1.0, 0.0, 0.0)  # Red

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "red_box.obj")
        mtl_path = os.path.join(tmpdir, "red_box.mtl")

        export_solid_to_obj(box, obj_path, color=color, material_name="red_material")

        assert os.path.exists(obj_path)
        assert os.path.exists(mtl_path)

        # Check OBJ references MTL
        with open(obj_path) as f:
            obj_content = f.read()
        assert "mtllib red_box.mtl" in obj_content
        assert "usemtl red_material" in obj_content

        # Check MTL has correct color
        with open(mtl_path) as f:
            mtl_content = f.read()
        assert "newmtl red_material" in mtl_content
        assert "Kd 1.000000 0.000000 0.000000" in mtl_content


def test_export_colored_parts_to_obj():
    """Test exporting multiple parts with different colors."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        create_box,
        create_cylinder,
        export_colored_parts_to_obj,
        translate_part,
    )

    box = create_box(10, 10, 10)
    cylinder = translate_part(create_cylinder(5, 20), (20, 0, 0))

    parts = [
        (box, "red_box", (1.0, 0.0, 0.0)),
        (cylinder, "green_cylinder", (0.0, 1.0, 0.0)),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "assembly.obj")
        mtl_path = os.path.join(tmpdir, "assembly.mtl")

        export_colored_parts_to_obj(parts, obj_path)

        assert os.path.exists(obj_path)
        assert os.path.exists(mtl_path)

        # Check OBJ content
        with open(obj_path) as f:
            obj_content = f.read()
        assert "mtllib assembly.mtl" in obj_content
        assert "o red_box" in obj_content
        assert "o green_cylinder" in obj_content
        assert "usemtl red_box" in obj_content
        assert "usemtl green_cylinder" in obj_content

        # Check MTL content
        with open(mtl_path) as f:
            mtl_content = f.read()
        assert "newmtl red_box" in mtl_content
        assert "newmtl green_cylinder" in mtl_content
        assert "Kd 1.000000 0.000000 0.000000" in mtl_content  # Red
        assert "Kd 0.000000 1.000000 0.000000" in mtl_content  # Green


def test_obj_vertex_indices_are_correct():
    """Test that face vertex indices are correctly offset for multiple parts."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import (
        create_box,
        export_colored_parts_to_obj,
        translate_part,
    )

    # Two boxes
    box1 = create_box(10, 10, 10)
    box2 = translate_part(create_box(10, 10, 10), (20, 0, 0))

    parts = [
        (box1, "box1", (1.0, 0.0, 0.0)),
        (box2, "box2", (0.0, 0.0, 1.0)),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "two_boxes.obj")

        export_colored_parts_to_obj(parts, obj_path)

        with open(obj_path) as f:
            lines = f.readlines()

        # Count vertices and check face indices
        vertex_count = sum(1 for line in lines if line.startswith("v "))
        face_lines = [line for line in lines if line.startswith("f ")]

        # All face indices should be valid (between 1 and vertex_count)
        for face_line in face_lines:
            indices = [int(idx) for idx in face_line.strip().split()[1:]]
            for idx in indices:
                assert (
                    1 <= idx <= vertex_count
                ), f"Invalid vertex index {idx} (max {vertex_count})"
