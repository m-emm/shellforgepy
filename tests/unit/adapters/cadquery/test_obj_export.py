"""Tests for OBJ export functionality in the CadQuery adapter."""

import os
import tempfile

import numpy as np


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


def test_cadquery_tessellate_returns_normalized_numpy_arrays():
    """CadQuery tessellation should expose backend-agnostic NumPy arrays."""
    from shellforgepy.adapters.cadquery.cadquery_adapter import create_box, tesellate

    vertices, triangles = tesellate(create_box(10, 10, 10))

    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert np.issubdtype(vertices.dtype, np.floating)

    assert isinstance(triangles, np.ndarray)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert np.issubdtype(triangles.dtype, np.integer)


def test_export_colored_meshes_to_obj_accepts_numpy_mesh_data():
    """OBJ writer should work with plain numeric arrays from any backend."""
    from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    meshes = [(vertices, triangles, "flat/mesh", (0.25, 0.5, 0.75))]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "mesh.obj")
        mtl_path = os.path.join(tmpdir, "mesh.mtl")
        export_colored_meshes_to_obj(meshes, obj_path)

        assert os.path.exists(obj_path)
        assert os.path.exists(mtl_path)

        with open(obj_path) as f:
            obj_content = f.read()
        assert "o flat_mesh" in obj_content
        assert "usemtl flat_mesh" in obj_content
        assert obj_content.count("\nv ") == 4
        assert obj_content.count("\nf ") == 2

        with open(mtl_path) as f:
            mtl_content = f.read()
        assert "newmtl flat_mesh" in mtl_content
        assert "Kd 0.250000 0.500000 0.750000" in mtl_content


def test_export_colored_meshes_to_obj_writes_animation_comments():
    """OBJ writer should emit ShellForgePy animation comments per object."""
    from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    meshes = [
        (
            vertices,
            triangles,
            "print bed",
            (0.1, 0.2, 0.3),
            {"bed_y": (0, 300, 0), "bed_x": (100, 0, 0)},
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "animated.obj")

        export_colored_meshes_to_obj(meshes, obj_path)

        with open(obj_path) as f:
            obj_content = f.read()

        assert "# Object: print bed" in obj_content
        assert "o print_bed" in obj_content
        assert "# shellforgepy_anim bed_y 0.0 300.0 0.0" in obj_content
        assert "# shellforgepy_anim bed_x 100.0 0.0 0.0" in obj_content
        assert obj_content.index("o print_bed") < obj_content.index(
            "# shellforgepy_anim bed_y 0.0 300.0 0.0"
        )
        assert obj_content.index(
            "# shellforgepy_anim bed_x 100.0 0.0 0.0"
        ) < obj_content.index("\nv 0.0 0.0 0.0")


def test_export_colored_meshes_to_obj_writes_hierarchy_comments():
    """OBJ writer should emit hierarchy comments when provided."""
    from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    meshes = [
        (
            vertices,
            triangles,
            "bed spacer",
            (0.5, 0.5, 0.5),
            None,
            {
                "hierarchy": ["print_bed_assembly", "hardware"],
                "hierarchy_labels": ["print_bed", "hardware"],
            },
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "hierarchy.obj")

        export_colored_meshes_to_obj(meshes, obj_path)

        with open(obj_path) as f:
            obj_content = f.read()

        assert "# shellforgepy_hierarchy print_bed_assembly/hardware" in obj_content
        assert "# shellforgepy_hierarchy_labels print_bed/hardware" in obj_content
        assert obj_content.index(
            "# shellforgepy_hierarchy print_bed_assembly/hardware"
        ) < obj_content.index("o bed_spacer")


def test_export_colored_meshes_to_obj_writes_builder_selector_comment():
    from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    meshes = [
        (
            vertices,
            triangles,
            "belt clamp screw",
            (0.5, 0.5, 0.5),
            None,
            {
                "builder_selector": "print_bed_assembly.non_production_parts.print_bed_undercarriage_belt_clamp_torsion_screw_back",
            },
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "selector.obj")

        export_colored_meshes_to_obj(meshes, obj_path)

        with open(obj_path) as f:
            obj_content = f.read()

        assert (
            "# shellforgepy_builder_selector "
            "print_bed_assembly.non_production_parts."
            "print_bed_undercarriage_belt_clamp_torsion_screw_back" in obj_content
        )
        assert obj_content.index(
            "# shellforgepy_builder_selector "
            "print_bed_assembly.non_production_parts."
            "print_bed_undercarriage_belt_clamp_torsion_screw_back"
        ) < obj_content.index("o belt_clamp_screw")
