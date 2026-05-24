import yaml
from shellforgepy.architecture.plan_render import render_semantic_storey_plan
from shellforgepy.architecture.polygon_clipping import (
    inset_polygon,
    polygon_area,
    polygon_covers,
    union_area,
)
from shellforgepy.architecture.storey_builder import build_storey


def _semantic_storey():
    return {
        "schema_version": 2,
        "name": "unit_test_storey",
        "frame": {
            "width": 4.0,
            "depth": 4.0,
            "origin": "FRONT_LEFT_OUTER",
            "axes": {"x": "RIGHT", "y": "BACK"},
        },
        "defaults": {
            "outer_wall_thickness": 0.2,
            "inner_wall_thickness": 0.1,
            "door_height": 2.0,
            "door_bottom": 0.0,
        },
        "reference_lines": {
            "x": {"LEFT": 0.0, "RIGHT": 4.0, "EX_LEFT": 1.0, "EX_RIGHT": 2.0},
            "y": {"FRONT": 0.0, "BACK": 4.0, "EX_FRONT": 1.0, "EX_BACK": 2.0},
        },
        "outer_walls": {
            "thickness": 0.2,
            "outline": [
                {"x": "LEFT", "y": "FRONT"},
                {"x": "RIGHT", "y": "FRONT"},
                {"x": "RIGHT", "y": "BACK"},
                {"x": "LEFT", "y": "BACK"},
            ],
        },
        "outer_cuts": [],
        "floor_cutouts": [],
        "living_area_exclusions": [
            {
                "id": "excluded",
                "type": "test",
                "rectangle": {
                    "x_min": "EX_LEFT",
                    "x_max": "EX_RIGHT",
                    "y_min": "EX_FRONT",
                    "y_max": "EX_BACK",
                },
            }
        ],
        "inner_walls": [
            {
                "id": "divider",
                "orientation": "VERTICAL",
                "at": "EX_RIGHT",
                "from": "FRONT",
                "to": "BACK",
            }
        ],
        "doors": [],
    }


def test_build_storey_adds_exclusion_marker_and_metadata(tmp_path):
    semantic_file = tmp_path / "storey.yaml"
    semantic_file.write_text(yaml.safe_dump(_semantic_storey()), encoding="utf-8")

    storey = build_storey(
        architecture={
            "storey_specification": {
                "id": "storey_unit",
                "storey_index": 1,
                "path": str(semantic_file),
                "sha256": "abc123",
            }
        },
        storey_stack={
            "floor_bases": [0.5],
            "floor_bases_z_offset": 1.0,
            "floor_heights": [3.0],
            "outer_wall_thickness": 0.2,
            "floor_thickness": 0.25,
            "inner_wall_thickness": 0.1,
        },
        living_space=None,
    )

    assert storey.get_named_non_production_part("living_area_exclusion_excluded")
    assert storey.get_named_follower("inner_walls")
    metadata = storey.additional_data["architecture"]["storey"]
    assert metadata["id"] == "storey_unit"
    assert metadata["dimensions"]["z_base"] == 1.5
    assert metadata["living_area_breakdown"]["excluded_areas_by_id"] == {
        "excluded": 1.0
    }


def test_semantic_storey_plan_render_hatches_exclusions(tmp_path):
    output_path = tmp_path / "storey_plan.svg"
    render_semantic_storey_plan(
        _semantic_storey(),
        output_path,
        storey_id="storey_unit",
        excluded_living_area_color=[1.0, 0.0, 0.0],
    )

    svg = output_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert 'id="excludedLivingAreaHatch"' in svg
    assert 'class="living-area-exclusion-hatch"' in svg
    assert 'class="inner-wall"' in svg
    assert "#ff0000" in svg


def test_semantic_storey_plan_render_draws_door_arcs_and_cutouts(tmp_path):
    semantic = _semantic_storey()
    semantic["floor_cutouts"] = [
        {
            "id": "stair_cutout",
            "outline": [
                {"x": "EX_LEFT", "y": "EX_FRONT"},
                {"x": "EX_RIGHT", "y": "EX_FRONT"},
                {"x": "EX_RIGHT", "y": "EX_BACK"},
                {"x": "EX_LEFT", "y": "EX_BACK"},
            ],
        }
    ]
    semantic["doors"] = [
        {
            "id": "divider_door",
            "wall": "divider",
            "width": 0.8,
            "height": 2.0,
            "offset": 1.0,
            "swings_toward": "RIGHT",
        }
    ]
    output_path = tmp_path / "storey_plan.svg"

    render_semantic_storey_plan(semantic, output_path, storey_id="storey_unit")

    svg = output_path.read_text(encoding="utf-8")
    assert 'class="floor-cutout"' in svg
    assert 'class="door-symbol door-arc"' in svg
    assert 'data-door-id="divider_door"' in svg


def test_polygon_clipping_helpers_cover_area_union_and_inset():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    inner = [(1, 1), (3, 1), (3, 3), (1, 3)]
    overlap_a = [(0, 0), (2, 0), (2, 2), (0, 2)]
    overlap_b = [(1, 0), (3, 0), (3, 2), (1, 2)]

    assert polygon_area(outer) == 16.0
    assert polygon_covers(outer, inner)
    assert not polygon_covers(inner, outer)
    assert union_area([overlap_a, overlap_b]) == 6.0
    assert inset_polygon(outer, 1.0, label="test inset") == [
        (3.0, 3.0),
        (1.0, 3.0),
        (1.0, 1.0),
        (3.0, 1.0),
    ]
