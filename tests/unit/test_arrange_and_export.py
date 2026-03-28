import pytest

from shellforgepy.produce.arrange_and_export import _split_parts_into_plates


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
