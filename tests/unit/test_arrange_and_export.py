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

    monkeypatch.setattr(arrange_and_export_module, "adapter_tesellate", fake_tessellate)
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
