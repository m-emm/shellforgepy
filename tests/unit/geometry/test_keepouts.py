import pytest
from shellforgepy.adapters._adapter import create_box, get_bounding_box, get_volume
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.simple import Alignment, align, create_box_hole_cutter


def _assert_bboxes_close(actual, expected, tol=1e-6):
    for actual_point, expected_point in zip(actual, expected):
        for actual_coord, expected_coord in zip(actual_point, expected_point):
            assert actual_coord == pytest.approx(expected_coord, abs=tol)


def test_create_box_hole_cutter_returns_box_leader_and_single_outer_cutter():
    assembly = create_box_hole_cutter(12.0, 34.0, 56.0, cutter_size=100.0)

    assert isinstance(assembly, LeaderFollowersCuttersPart)
    assert len(assembly.followers) == 0
    assert len(assembly.cutters) == 1

    _assert_bboxes_close(
        get_bounding_box(assembly.leader),
        ((0.0, 0.0, 0.0), (12.0, 34.0, 56.0)),
    )
    _assert_bboxes_close(
        get_bounding_box(assembly.cutters[0]),
        ((-100.0, -100.0, -100.0), (112.0, 134.0, 156.0)),
    )


def test_create_box_hole_cutter_uses_default_500_cutter_size():
    assembly = create_box_hole_cutter(10.0, 20.0, 30.0)

    _assert_bboxes_close(
        get_bounding_box(assembly.cutters[0]),
        ((-500.0, -500.0, -500.0), (510.0, 520.0, 530.0)),
    )


def test_create_box_hole_cutter_clips_part_to_aligned_box():
    part = create_box(200.0, 200.0, 200.0, origin=(-100.0, -100.0, -100.0))

    keep_volume = create_box_hole_cutter(40.0, 60.0, 80.0, cutter_size=250.0)
    keep_volume = align(keep_volume, part, Alignment.CENTER)

    trimmed_part = keep_volume.use_as_cutter_on(part)

    _assert_bboxes_close(
        get_bounding_box(trimmed_part),
        get_bounding_box(keep_volume.leader),
    )
    assert get_volume(trimmed_part) == pytest.approx(
        get_volume(keep_volume.leader),
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"box_width": 0.0, "box_length": 20.0, "box_height": 30.0}, "box_width"),
        ({"box_width": 10.0, "box_length": -1.0, "box_height": 30.0}, "box_length"),
        ({"box_width": 10.0, "box_length": 20.0, "box_height": 0.0}, "box_height"),
        (
            {
                "box_width": 10.0,
                "box_length": 20.0,
                "box_height": 30.0,
                "cutter_size": 0.0,
            },
            "cutter_size",
        ),
    ],
)
def test_create_box_hole_cutter_validates_positive_dimensions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_box_hole_cutter(**kwargs)
