import json
from pathlib import Path

import numpy as np
import pytest
from shellforgepy.produce.arrange_and_export import (
    _arrange_parts_for_production,
    _arrange_plate_parts_for_bed,
    _build_colored_meshes,
    _build_production_obj_scene_parts,
    _center_plate_parts_on_bed,
    _render_signature,
    _split_parts_into_plates,
    arrange_and_export,
    arrange_and_export_parts,
)
from shellforgepy.produce.mesh_scene import ObjMesh
from shellforgepy.produce.production_parts_model import PartList
from shellforgepy.simple import create_box, get_bounding_box, translate


def test_split_parts_into_declared_plates_and_auto_assignment():
    arranged_parts = [
        {"name": "frame", "width": 80, "height": 80},
        {"name": "left_bracket", "width": 30, "height": 20},
        {"name": "right_bracket", "width": 25, "height": 20},
    ]

    plates = _split_parts_into_plates(
        arranged_parts,
        declared_plates=[{"name": "critical", "parts": ["frame"]}],
        auto_assign_plates=True,
        bed_width=120,
        bed_depth=120,
        gap=5,
    )

    assert [name for name, _ in plates] == ["critical", "plate_2"]
    assert [[part["name"] for part in members] for _, members in plates] == [
        ["frame"],
        ["left_bracket", "right_bracket"],
    ]


def test_obj_mesh_validate_rejects_invalid_meshes():
    with pytest.raises(ValueError, match="vertices must not be empty"):
        ObjMesh(
            vertices=np.empty((0, 3)),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        ).validate()

    with pytest.raises(ValueError, match="vertices must have shape"):
        ObjMesh(
            vertices=np.asarray([0.0, 0.0, 0.0]),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        ).validate()

    with pytest.raises(ValueError, match="out of range"):
        ObjMesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.asarray([[0, 1, 3]], dtype=np.int64),
        ).validate()


def test_part_list_preserves_obj_mesh_metadata_fields():
    parts = PartList()
    parts.add(
        object(),
        "terrain",
        obj_metadata={"hierarchy": ["rw8", "terrain"]},
        source_path="/tmp/terrain.npz",
        source_parameter_hash="abc123",
        source_version_inputs={"dtm_stride": 4},
    )

    entry = parts.as_list()[0]

    assert entry["obj_metadata"] == {"hierarchy": ["rw8", "terrain"]}
    assert entry["source_path"] == "/tmp/terrain.npz"
    assert entry["source_parameter_hash"] == "abc123"
    assert entry["source_version_inputs"] == {"dtm_stride": 4}


def test_build_colored_meshes_passes_obj_mesh_through_without_tessellation(
    monkeypatch,
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    monkeypatch.setattr(
        arrange_and_export_module,
        "adapter_tessellate",
        lambda *args, **kwargs: pytest.fail("ObjMesh should not be tessellated"),
    )

    mesh = ObjMesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        color=(0.9, 0.8, 0.7),
        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        texture_path="terrain.png",
        material_name="terrain_orthophoto",
        metadata={"hierarchy": ["rw8", "terrain"]},
    )

    meshes = _build_colored_meshes(
        [{"name": "terrain", "part": mesh, "animation": None}]
    )

    assert len(meshes) == 1
    vertices, faces, name, color, animation, metadata, material = meshes[0]
    assert vertices is mesh.vertices
    assert faces is mesh.faces
    assert name == "terrain"
    assert color == (0.9, 0.8, 0.7)
    assert animation is None
    assert metadata == {"hierarchy": ["rw8", "terrain"]}
    assert np.asarray(material["uvs"]).shape == (3, 2)
    assert material["texture_path"] == "terrain.png"
    assert material["material_name"] == "terrain_orthophoto"


def test_split_parts_into_declared_plates_accepts_wildcard_part_names():
    arranged_parts = [
        {"name": "elko_sleeve_plate_1", "width": 20, "height": 20},
        {"name": "elko_sleeve_plate_2", "width": 20, "height": 20},
        {"name": "side_walls", "width": 80, "height": 80},
    ]

    plates = _split_parts_into_plates(
        arranged_parts,
        declared_plates=[
            {
                "name": "board_holder_elko_sleeve_plates",
                "parts": ["elko_sleeve_plate_*"],
            },
            {"name": "board_holder_side_walls", "parts": ["side_walls"]},
        ],
        auto_assign_plates=False,
        gap=5.0,
    )

    assert [name for name, _ in plates] == [
        "board_holder_elko_sleeve_plates",
        "board_holder_side_walls",
    ]
    assert [[part["name"] for part in members] for _, members in plates] == [
        ["elko_sleeve_plate_1", "elko_sleeve_plate_2"],
        ["side_walls"],
    ]


def test_split_parts_into_auto_plates_spills_when_next_part_no_longer_fits():
    arranged_parts = [
        {"name": "large", "width": 70, "height": 70},
        {"name": "medium", "width": 45, "height": 45},
        {"name": "small", "width": 40, "height": 40},
    ]

    plates = _split_parts_into_plates(
        arranged_parts,
        auto_assign_plates=True,
        bed_width=100,
        bed_depth=100,
        gap=5,
    )

    assert [name for name, _ in plates] == ["plate_1", "plate_2"]
    assert [[part["name"] for part in members] for _, members in plates] == [
        ["large"],
        ["medium", "small"],
    ]


def test_split_parts_into_declared_plates_requires_full_assignment_without_auto():
    arranged_parts = [{"name": "frame"}, {"name": "bracket"}]

    with pytest.raises(ValueError, match="Unassigned parts remain"):
        _split_parts_into_plates(
            arranged_parts,
            declared_plates=[{"name": "plate_a", "parts": ["frame"]}],
            auto_assign_plates=False,
            gap=5.0,
        )


def test_split_parts_into_declared_plates_rejects_duplicate_assignments():
    arranged_parts = [{"name": "frame"}, {"name": "bracket"}]

    with pytest.raises(ValueError, match="multiple declared plates"):
        _split_parts_into_plates(
            arranged_parts,
            declared_plates=[
                {"name": "plate_a", "parts": ["frame"]},
                {"name": "plate_b", "parts": ["frame"]},
            ],
            auto_assign_plates=True,
            gap=5.0,
        )


def test_center_plate_parts_on_bed_recenters_subset_independently():
    plate_parts = [
        {
            "name": "frame",
            "part": translate(0, 290, 0)(create_box(160, 140, 10)),
            "transform_history": [],
        },
        {
            "name": "clamp",
            "part": translate(170, 290, 0)(create_box(10, 30, 10)),
            "transform_history": [],
        },
    ]

    centered = _center_plate_parts_on_bed(
        plate_parts,
        bed_width=220.0,
        bed_depth=220.0,
    )

    mins = []
    maxs = []
    for entry in centered:
        min_point, max_point = get_bounding_box(entry["part"])
        mins.append(min_point)
        maxs.append(max_point)

    min_x = min(point[0] for point in mins)
    min_y = min(point[1] for point in mins)
    max_x = max(point[0] for point in maxs)
    max_y = max(point[1] for point in maxs)

    assert min_x == pytest.approx(20.0)
    assert max_x == pytest.approx(200.0)
    assert min_y == pytest.approx(40.0)
    assert max_y == pytest.approx(180.0)
    assert centered[0]["transform_history"][-1]["kind"] == "translate"


def test_arrange_parts_for_production_allows_overflow_for_visualization():
    arranged = _arrange_parts_for_production(
        [{"name": "oversize", "part": create_box(270.4, 144.0, 10.0)}],
        gap=1.0,
        bed_width=220.0,
        enforce_bed_size=False,
    )

    assert [entry["name"] for entry in arranged] == ["oversize"]

    min_point, max_point = get_bounding_box(arranged[0]["part"])
    assert max_point[0] - min_point[0] == pytest.approx(270.4)
    assert max_point[1] - min_point[1] == pytest.approx(144.0)


def test_arrange_plate_parts_for_bed_matches_legacy_production_layout():
    parts = [
        {"name": "tall", "part": create_box(60.0, 60.0, 10.0)},
        {"name": "wide", "part": create_box(60.0, 20.0, 10.0)},
        {"name": "small", "part": create_box(35.0, 20.0, 10.0)},
    ]

    arranged_production = _arrange_parts_for_production(
        parts,
        gap=5.0,
        bed_width=100.0,
        bed_depth=100.0,
    )
    arranged_plate = _arrange_plate_parts_for_bed(
        parts,
        bed_width=100.0,
        bed_depth=100.0,
        gap=5.0,
    )

    production_bounds = {
        entry["name"]: get_bounding_box(entry["part"]) for entry in arranged_production
    }
    plate_bounds = {
        entry["name"]: get_bounding_box(entry["part"]) for entry in arranged_plate
    }

    for part_name in production_bounds:
        production_min, production_max = production_bounds[part_name]
        plate_min, plate_max = plate_bounds[part_name]
        assert plate_min == pytest.approx(production_min)
        assert plate_max == pytest.approx(production_max)

    assert plate_bounds["small"][0][1] == pytest.approx(72.5)


def test_arrange_plate_parts_for_bed_supports_rectangular_origin_offset():
    arranged = _arrange_plate_parts_for_bed(
        [{"name": "token", "part": create_box(20.0, 20.0, 1.0)}],
        bed_width=140.0,
        bed_depth=250.0,
        gap=5.0,
        prod_origin=(155.0, 55.0),
    )

    min_point, max_point = get_bounding_box(arranged[0]["part"])

    assert min_point[0] == pytest.approx(215.0)
    assert max_point[0] == pytest.approx(235.0)
    assert min_point[1] == pytest.approx(170.0)
    assert max_point[1] == pytest.approx(190.0)
    assert min_point[2] == pytest.approx(0.0)
    assert max_point[2] == pytest.approx(1.0)


def test_arrange_plate_parts_for_bed_keeps_production_group_together():
    grouped_metadata = {"production_group": "offset_cross"}
    arranged = _arrange_plate_parts_for_bed(
        [
            {
                "name": "left_material",
                "part": create_box(10.0, 10.0, 1.0, origin=(0.0, 0.0, 0.0)),
                "obj_metadata": grouped_metadata,
            },
            {
                "name": "right_material",
                "part": create_box(10.0, 10.0, 1.0, origin=(20.0, 0.0, 0.0)),
                "obj_metadata": grouped_metadata,
            },
        ],
        bed_width=100.0,
        bed_depth=100.0,
        gap=5.0,
    )

    bounds = {entry["name"]: get_bounding_box(entry["part"]) for entry in arranged}

    assert bounds["left_material"][0][0] == pytest.approx(35.0)
    assert bounds["left_material"][1][0] == pytest.approx(45.0)
    assert bounds["right_material"][0][0] == pytest.approx(55.0)
    assert bounds["right_material"][1][0] == pytest.approx(65.0)
    assert arranged[0]["obj_metadata"] == grouped_metadata
    assert arranged[1]["obj_metadata"] == grouped_metadata


def test_build_production_obj_scene_parts_offsets_each_plate_independently():
    prepared_plate_groups = [
        (
            "left_plate",
            _arrange_plate_parts_for_bed(
                [{"name": "left_box", "part": create_box(10, 10, 10)}],
                bed_width=100.0,
                bed_depth=100.0,
                gap=5.0,
            ),
        ),
        (
            "right_plate",
            _arrange_plate_parts_for_bed(
                [{"name": "right_box", "part": create_box(10, 10, 10)}],
                bed_width=100.0,
                bed_depth=100.0,
                gap=5.0,
            ),
        ),
    ]

    scene_parts = _build_production_obj_scene_parts(
        prepared_plate_groups,
        bed_width=100.0,
        bed_depth=100.0,
        plate_scene_gap=20.0,
        visualize_plate_boundaries=False,
    )

    left_part = next(entry for entry in scene_parts if entry["name"] == "left_box")
    right_part = next(entry for entry in scene_parts if entry["name"] == "right_box")

    left_min, left_max = get_bounding_box(left_part["part"])
    right_min, right_max = get_bounding_box(right_part["part"])

    assert left_min[0] == pytest.approx(45.0)
    assert left_max[0] == pytest.approx(55.0)
    assert right_min[0] == pytest.approx(165.0)
    assert right_max[0] == pytest.approx(175.0)
    assert left_part["obj_metadata"]["plate_name"] == "left_plate"
    assert right_part["obj_metadata"]["plate_name"] == "right_plate"


def test_arrange_and_export_parts_can_add_plate_boundaries_to_obj_scene(
    monkeypatch, tmp_path
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    captured_parts = []

    def fake_build_colored_meshes(parts, **kwargs):
        captured_parts[:] = list(parts)
        return []

    def fake_export_colored_meshes_to_obj(meshes, destination):
        destination_path = Path(destination)
        destination_path.write_text("obj", encoding="utf-8")
        destination_path.with_suffix(".mtl").write_text("mtl", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "_build_colored_meshes",
        fake_build_colored_meshes,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "export_colored_meshes_to_obj",
        fake_export_colored_meshes_to_obj,
    )

    result_path = arrange_and_export_parts(
        [{"name": "box", "part": create_box(10, 10, 10)}],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="test_prod_view.py",
        export_directory=tmp_path,
        prod=True,
        export_stl=False,
        export_step=False,
        export_obj=True,
        plates=[{"name": "build_plate", "parts": ["box"]}],
    )

    assert result_path == tmp_path / "test_prod_view_prod.obj"

    boundary_parts = [
        entry for entry in captured_parts if entry["name"].startswith("plate_boundary_")
    ]
    assert len(boundary_parts) == 1

    min_x = min(get_bounding_box(entry["part"])[0][0] for entry in boundary_parts)
    min_y = min(get_bounding_box(entry["part"])[0][1] for entry in boundary_parts)
    max_x = max(get_bounding_box(entry["part"])[1][0] for entry in boundary_parts)
    max_y = max(get_bounding_box(entry["part"])[1][1] for entry in boundary_parts)

    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(100.0)
    assert max_y == pytest.approx(100.0)


def test_arrange_and_export_rejects_obj_mesh_in_production(tmp_path):
    mesh = ObjMesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="visualization-only"):
        arrange_and_export_parts(
            [{"name": "terrain", "part": mesh}],
            prod_gap=2.0,
            bed_width=100.0,
            script_file="terrain.py",
            export_directory=tmp_path,
            prod=True,
            export_stl=False,
            export_step=False,
            export_obj=True,
        )


def test_arrange_and_export_exports_plate_obj_files_and_manifest_entries(
    monkeypatch, tmp_path
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    export_calls = []
    manifest_path = tmp_path / "workflow_manifest.json"

    def fake_build_colored_meshes(parts, **kwargs):
        return [{"parts": [entry["name"] for entry in parts]}]

    def fake_export_colored_meshes_to_obj(meshes, destination):
        destination_path = Path(destination)
        export_calls.append((destination_path.name, meshes[0]["parts"]))
        destination_path.write_text("obj", encoding="utf-8")
        destination_path.with_suffix(".mtl").write_text("mtl", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "_build_colored_meshes",
        fake_build_colored_meshes,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "export_colored_meshes_to_obj",
        fake_export_colored_meshes_to_obj,
    )

    monkeypatch.setenv("SHELLFORGEPY_WORKFLOW_MANIFEST", str(manifest_path))

    arrange_and_export_parts(
        [
            {"name": "left_box", "part": create_box(10, 10, 10)},
            {"name": "right_box", "part": create_box(10, 10, 10)},
        ],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="test_multi_plate.py",
        export_directory=tmp_path,
        export_stl=False,
        export_step=False,
        export_obj=True,
        plates=[
            {"name": "left_plate", "parts": ["left_box"]},
            {"name": "right_plate", "parts": ["right_box"]},
        ],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert export_calls == [
        ("test_multi_plate_left_plate.obj", ["left_box"]),
        ("test_multi_plate_right_plate.obj", ["right_box"]),
        ("test_multi_plate.obj", ["left_box", "right_box"]),
    ]
    assert manifest["obj_path"].endswith("test_multi_plate.obj")
    assert manifest["mtl_path"].endswith("test_multi_plate.mtl")
    assert manifest["plates"][0]["obj_path"].endswith("test_multi_plate_left_plate.obj")
    assert manifest["plates"][0]["mtl_path"].endswith("test_multi_plate_left_plate.mtl")
    assert manifest["plates"][1]["obj_path"].endswith(
        "test_multi_plate_right_plate.obj"
    )
    assert manifest["plates"][1]["mtl_path"].endswith(
        "test_multi_plate_right_plate.mtl"
    )


def test_arrange_and_export_manifest_records_plate_local_part_files(
    monkeypatch, tmp_path
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    manifest_path = tmp_path / "workflow_manifest.json"

    def fake_export_solid_to_stl(part, destination):
        Path(destination).write_text("solid test\nendsolid test\n", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "export_solid_to_stl",
        fake_export_solid_to_stl,
    )
    monkeypatch.setenv("SHELLFORGEPY_WORKFLOW_MANIFEST", str(manifest_path))

    arrange_and_export_parts(
        [
            {
                "name": "tool_0_body",
                "part": create_box(10, 10, 1),
                "obj_metadata": {"slicer_filament_id": 1},
            },
            {
                "name": "tool_1_body",
                "part": create_box(10, 10, 1),
                "obj_metadata": {"slicer_filament_id": 2},
            },
        ],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="test_multi_material.py",
        export_directory=tmp_path,
        export_stl=True,
        export_step=False,
        export_obj=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plate_part_files = manifest["plates"][0]["part_files"]

    assert [entry["name"] for entry in plate_part_files] == [
        "tool_0_body",
        "tool_1_body",
    ]
    assert [
        entry["obj_metadata"]["slicer_filament_id"] for entry in plate_part_files
    ] == [
        1,
        2,
    ]
    assert all(Path(entry["path"]).exists() for entry in plate_part_files)


def test_arrange_and_export_manifest_records_plate_step_files(monkeypatch, tmp_path):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    manifest_path = tmp_path / "workflow_manifest.json"

    def fake_export_solid_to_step(part, destination):
        Path(destination).write_text("ISO-10303-21;\n", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "export_solid_to_step",
        fake_export_solid_to_step,
    )
    monkeypatch.setenv("SHELLFORGEPY_WORKFLOW_MANIFEST", str(manifest_path))

    result = arrange_and_export_parts(
        [
            {"name": "left_bracket", "part": create_box(10, 10, 1)},
            {"name": "right_bracket", "part": create_box(10, 10, 1)},
        ],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="test_cnc_bundle.py",
        export_directory=tmp_path,
        export_base_name="cnc_bundle",
        export_stl=False,
        export_step=True,
        export_obj=False,
        export_individual_parts=False,
        plates=[
            {
                "name": "left_part",
                "filename": "left_upload.step",
                "parts": ["left_bracket"],
            },
            {
                "name": "right_part",
                "filename": "right_upload",
                "parts": ["right_bracket"],
            },
        ],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result == tmp_path / "left_upload.step"
    assert manifest["assembly_step_path"].endswith("left_upload.step")
    assert [entry["name"] for entry in manifest["step_files"]] == [
        "left_part",
        "right_part",
    ]
    assert [entry["parts"] for entry in manifest["step_files"]] == [
        ["left_bracket"],
        ["right_bracket"],
    ]
    assert manifest["plates"][0]["step_path"].endswith("left_upload.step")
    assert manifest["plates"][1]["step_path"].endswith("right_upload.step")
    assert all(Path(entry["path"]).exists() for entry in manifest["step_files"])


def test_arrange_and_export_parts_uses_export_base_name_for_filenames(
    monkeypatch, tmp_path
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    manifest_path = tmp_path / "workflow_manifest.json"

    def fake_build_colored_meshes(parts, **kwargs):
        return [{"parts": [entry["name"] for entry in parts]}]

    def fake_export_colored_meshes_to_obj(meshes, destination):
        destination_path = Path(destination)
        destination_path.write_text("obj", encoding="utf-8")
        destination_path.with_suffix(".mtl").write_text("mtl", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "_build_colored_meshes",
        fake_build_colored_meshes,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "export_colored_meshes_to_obj",
        fake_export_colored_meshes_to_obj,
    )
    monkeypatch.setenv("SHELLFORGEPY_WORKFLOW_MANIFEST", str(manifest_path))

    result = arrange_and_export_parts(
        [{"name": "left_box", "part": create_box(10, 10, 10)}],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="x_axis_endstop_assembly.py",
        export_base_name="x_axis_endstop_left_assembly",
        export_directory=tmp_path,
        export_stl=False,
        export_step=False,
        export_obj=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result == tmp_path / "x_axis_endstop_left_assembly.obj"
    assert manifest["script_file"] == "x_axis_endstop_assembly.py"
    assert manifest["obj_path"].endswith("x_axis_endstop_left_assembly.obj")
    assert manifest["mtl_path"].endswith("x_axis_endstop_left_assembly.mtl")


def test_arrange_and_export_uses_single_selected_plate_name_for_primary_obj(
    monkeypatch, tmp_path
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    export_calls = []
    manifest_path = tmp_path / "workflow_manifest.json"

    def fake_build_colored_meshes(parts, **kwargs):
        return [{"parts": [entry["name"] for entry in parts]}]

    def fake_export_colored_meshes_to_obj(meshes, destination):
        destination_path = Path(destination)
        export_calls.append(destination_path.name)
        destination_path.write_text("obj", encoding="utf-8")
        destination_path.with_suffix(".mtl").write_text("mtl", encoding="utf-8")

    monkeypatch.setattr(
        arrange_and_export_module,
        "_build_colored_meshes",
        fake_build_colored_meshes,
    )
    monkeypatch.setattr(
        arrange_and_export_module,
        "export_colored_meshes_to_obj",
        fake_export_colored_meshes_to_obj,
    )
    monkeypatch.setenv("SHELLFORGEPY_WORKFLOW_MANIFEST", str(manifest_path))

    result = arrange_and_export_parts(
        [
            {"name": "left_box", "part": create_box(10, 10, 10)},
            {"name": "right_box", "part": create_box(10, 10, 10)},
        ],
        prod_gap=2.0,
        bed_width=100.0,
        script_file="test_multi_plate.py",
        export_directory=tmp_path,
        prod=True,
        export_stl=False,
        export_step=False,
        export_obj=True,
        plates=[
            {"name": "left_plate", "parts": ["left_box"]},
            {"name": "right_plate", "parts": ["right_box"]},
        ],
        selected_plates=["left_plate"],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert export_calls == ["test_multi_plate_left_plate.obj"]
    assert result == tmp_path / "test_multi_plate_left_plate.obj"
    assert manifest["obj_path"].endswith("test_multi_plate_left_plate.obj")
    assert manifest["mtl_path"].endswith("test_multi_plate_left_plate.mtl")
    assert manifest["plates"][0]["obj_path"].endswith("test_multi_plate_left_plate.obj")
    assert manifest["plates"][0]["mtl_path"].endswith("test_multi_plate_left_plate.mtl")


def test_arrange_and_export_uses_four_millimeter_default_gap(monkeypatch):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    captured = {}

    def fake_arrange_and_export_parts(
        parts, prod_gap, bed_width, script_file, **kwargs
    ):
        captured["prod_gap"] = prod_gap
        captured["bed_width"] = bed_width
        captured["script_file"] = script_file
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(
        arrange_and_export_module,
        "arrange_and_export_parts",
        fake_arrange_and_export_parts,
    )

    result = arrange_and_export([], script_file="demo.py")

    assert result == "ok"
    assert captured["prod_gap"] == pytest.approx(4.0)


def test_render_signature_uses_step_data_hash_for_source_artifacts(tmp_path):
    first_step = tmp_path / "first.step"
    second_step = tmp_path / "second.step"
    changed_step = tmp_path / "changed.step"
    first_step.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_NAME('shape','2026-05-15T13:17:36',('Author'),('Open CASCADE'),"
        "'processor','system','Unknown');\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#1 = CARTESIAN_POINT('',(1.0,2.0,3.0));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    second_step.write_text(
        first_step.read_text(encoding="utf-8").replace(
            "2026-05-15T13:17:36", "2026-05-15T13:21:06"
        ),
        encoding="utf-8",
    )
    changed_step.write_text(
        first_step.read_text(encoding="utf-8").replace(
            "(1.0,2.0,3.0)", "(1.0,2.0,4.0)"
        ),
        encoding="utf-8",
    )

    base_entry = {
        "name": "box",
        "part": "shape-1",
        "source_parameter_hash": "old-build-hash",
        "source_version_inputs": {"generator_source_sha256": "old-source"},
        "transform_history": [{"kind": "translate", "vector": [1.0, 2.0, 3.0]}],
    }
    first_signature, first_payload = _render_signature(
        {**base_entry, "source_path": str(first_step)},
        tolerance=0.1,
        angular_tolerance=0.1,
    )
    second_signature, second_payload = _render_signature(
        {
            **base_entry,
            "source_path": str(second_step),
            "source_parameter_hash": "new-build-hash",
            "source_version_inputs": {"generator_source_sha256": "new-source"},
        },
        tolerance=0.1,
        angular_tolerance=0.1,
    )
    changed_signature, _ = _render_signature(
        {**base_entry, "source_path": str(changed_step)},
        tolerance=0.1,
        angular_tolerance=0.1,
    )

    assert first_signature == second_signature
    assert changed_signature != first_signature
    assert "source_artifact" in first_payload
    assert first_payload["source_artifact"] == second_payload["source_artifact"]


def test_build_colored_meshes_migrates_legacy_mesh_cache(monkeypatch, tmp_path):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    monkeypatch.setattr(
        arrange_and_export_module,
        "adapter_tessellate",
        lambda *args, **kwargs: pytest.fail("legacy cache should be reused"),
    )
    monkeypatch.setattr(
        arrange_and_export_module, "adapter_get_adapter_id", lambda: "test-adapter"
    )

    source_step = tmp_path / "source.step"
    source_step.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_NAME('shape','now',(),(),'','','');\n"
        "ENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('',(1.0,2.0,3.0));\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    part_entry = {
        "name": "box",
        "part": "shape-1",
        "source_path": str(source_step),
        "source_parameter_hash": "abc123",
        "source_version_inputs": {"resource_sha256": "deadbeef"},
        "transform_history": [{"kind": "translate", "vector": [1.0, 2.0, 3.0]}],
    }
    legacy_signature, legacy_payload = _render_signature(
        part_entry,
        tolerance=0.1,
        angular_tolerance=0.1,
        use_source_artifact_signature=False,
    )
    new_signature, _ = _render_signature(
        part_entry,
        tolerance=0.1,
        angular_tolerance=0.1,
    )
    assert legacy_signature != new_signature

    np.savez_compressed(
        tmp_path / f"{legacy_signature}.npz",
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    (tmp_path / f"{legacy_signature}.json").write_text(
        json.dumps(legacy_payload),
        encoding="utf-8",
    )

    meshes = _build_colored_meshes([part_entry], mesh_cache_dir=tmp_path)

    assert len(meshes) == 1
    assert (tmp_path / f"{new_signature}.npz").exists()
    assert (tmp_path / f"{new_signature}.json").exists()


def test_build_colored_meshes_recovers_from_truncated_cache(
    monkeypatch, tmp_path, caplog
):
    import shellforgepy.produce.arrange_and_export as arrange_and_export_module

    tessellated_parts = []

    def fake_tessellate(part, tolerance=0.1, angular_tolerance=0.1):
        tessellated_parts.append(part)
        return (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

    monkeypatch.setattr(
        arrange_and_export_module, "adapter_tessellate", fake_tessellate
    )
    monkeypatch.setattr(
        arrange_and_export_module, "adapter_get_adapter_id", lambda: "test-adapter"
    )

    part_entry = {
        "name": "box",
        "part": "shape-1",
        "source_path": "/tmp/source.step",
        "source_parameter_hash": "abc123",
        "source_version_inputs": {"resource_sha256": "deadbeef"},
        "transform_history": [{"kind": "translate", "vector": [1.0, 2.0, 3.0]}],
    }

    signature, _ = _render_signature(
        part_entry,
        tolerance=0.1,
        angular_tolerance=0.1,
    )
    assert signature is not None

    mesh_path = tmp_path / f"{signature}.npz"
    metadata_path = tmp_path / f"{signature}.json"
    mesh_path.write_bytes(b"broken-cache")
    metadata_path.write_text('{"stale": true}', encoding="utf-8")

    with caplog.at_level("WARNING"):
        meshes = _build_colored_meshes([part_entry], mesh_cache_dir=tmp_path)

    assert len(meshes) == 1
    assert tessellated_parts == ["shape-1"]
    assert any(
        "Discarding unreadable OBJ mesh cache entry" in message
        for message in caplog.messages
    )

    with np.load(mesh_path) as data:
        assert np.asarray(data["vertices"]).shape == (3, 3)
        assert np.asarray(data["triangles"]).shape == (1, 3)

    tessellated_parts.clear()
    meshes = _build_colored_meshes([part_entry], mesh_cache_dir=tmp_path)

    assert len(meshes) == 1
    assert tessellated_parts == []
