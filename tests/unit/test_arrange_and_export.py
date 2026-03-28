import pytest
from shellforgepy.produce.arrange_and_export import (
    _arrange_parts_for_production,
    _center_plate_parts_on_bed,
    _split_parts_into_plates,
)
from shellforgepy.simple import create_box, get_bounding_box, translate


def test_split_parts_into_declared_plates_and_auto_assignment():
    arranged_parts = [
        {"name": "frame"},
        {"name": "left_bracket"},
        {"name": "right_bracket"},
    ]

    plates = _split_parts_into_plates(
        arranged_parts,
        declared_plates=[{"name": "critical", "parts": ["frame"]}],
        auto_assign_plates=True,
    )

    assert [name for name, _ in plates] == ["critical", "plate_2", "plate_3"]
    assert [[part["name"] for part in members] for _, members in plates] == [
        ["frame"],
        ["left_bracket"],
        ["right_bracket"],
    ]


def test_split_parts_into_declared_plates_requires_full_assignment_without_auto():
    arranged_parts = [{"name": "frame"}, {"name": "bracket"}]

    with pytest.raises(ValueError, match="Unassigned parts remain"):
        _split_parts_into_plates(
            arranged_parts,
            declared_plates=[{"name": "plate_a", "parts": ["frame"]}],
            auto_assign_plates=False,
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
