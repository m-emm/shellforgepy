import math

import numpy as np
import pytest
from shellforgepy.adapters._adapter import (
    create_box,
    get_bounding_box,
    get_vertex_coordinates,
    get_volume,
)
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.geometry.dovetails import (
    DEFAULT_DOVETAIL_CUTTER_NAME,
    DEFAULT_DOVETAIL_FOLLOWER_NAME,
    create_dovetail_tongue_and_groove,
)


def _span(values):
    return float(np.max(values) - np.min(values))


def _assert_bboxes_close(actual, expected, tol=1e-6):
    for actual_point, expected_point in zip(actual, expected):
        for actual_coord, expected_coord in zip(actual_point, expected_point):
            assert actual_coord == pytest.approx(expected_coord, abs=tol)


def test_create_dovetail_tongue_and_groove_returns_named_assembly():
    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=10.0,
        length=30.0,
        box_size_x=18.0,
        box_size_y=6.0,
        taper_per_side=1.5,
        dovetail_clearance=0.2,
        parts_clearance=0.5,
    )

    assert isinstance(assembly, LeaderFollowersCuttersPart)
    assert len(assembly.followers) == 1
    assert len(assembly.cutters) == 1
    assert assembly.get_follower_index_by_name(DEFAULT_DOVETAIL_FOLLOWER_NAME) == 0
    assert assembly.get_cutter_index_by_name(DEFAULT_DOVETAIL_CUTTER_NAME) == 0


def test_dovetail_cutter_has_expected_orientation_and_widths():
    dovetail_width = 10.0
    length = 40.0
    box_size_y = 8.0
    taper_per_side = 2.0
    dovetail_clearance = 0.3

    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=dovetail_width,
        length=length,
        box_size_x=20.0,
        box_size_y=box_size_y,
        taper_per_side=taper_per_side,
        dovetail_clearance=dovetail_clearance,
    )

    cutter = assembly.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME)
    vertices = np.asarray(get_vertex_coordinates(cutter), dtype=float)

    opening_vertices = vertices[np.isclose(vertices[:, 1], 0.0, atol=1e-6)]
    back_vertices = vertices[np.isclose(vertices[:, 1], box_size_y, atol=1e-6)]

    assert len(opening_vertices) >= 4
    assert len(back_vertices) >= 4
    assert float(np.min(vertices[:, 1])) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(vertices[:, 1])) == pytest.approx(box_size_y, abs=1e-6)

    expected_opening_width = dovetail_width + 2 * dovetail_clearance
    expected_back_width = dovetail_width + 2 * taper_per_side + 2 * dovetail_clearance

    assert _span(opening_vertices[:, 0]) == pytest.approx(
        expected_opening_width, abs=1e-6
    )
    assert _span(back_vertices[:, 0]) == pytest.approx(expected_back_width, abs=1e-6)
    assert _span(opening_vertices[:, 2]) == pytest.approx(length, abs=1e-6)
    assert _span(back_vertices[:, 2]) == pytest.approx(length, abs=1e-6)


def test_parts_clearance_adds_narrow_groove_entry_relief():
    dovetail_width = 10.0
    length = 30.0
    box_size_y = 6.0
    taper_per_side = 1.5
    dovetail_clearance = 0.2
    parts_clearance = 0.75

    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=dovetail_width,
        length=length,
        box_size_x=18.0,
        box_size_y=box_size_y,
        taper_per_side=taper_per_side,
        dovetail_clearance=dovetail_clearance,
        parts_clearance=parts_clearance,
    )

    cutter = assembly.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME)
    vertices = np.asarray(get_vertex_coordinates(cutter), dtype=float)
    front_vertices = vertices[np.isclose(vertices[:, 1], 0.0, atol=1e-6)]

    expected_relief_width = dovetail_width + 2 * dovetail_clearance

    assert len(front_vertices) >= 4
    assert _span(front_vertices[:, 0]) == pytest.approx(expected_relief_width, abs=1e-6)


def test_follower_geometry_matches_named_groove_cutter():
    box_size_x = 22.0
    box_size_y = 7.0
    length = 35.0

    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=9.0,
        length=length,
        box_size_x=box_size_x,
        box_size_y=box_size_y,
        taper_per_side=1.0,
        dovetail_clearance=0.25,
    )

    expected_follower = create_box(
        box_size_x,
        box_size_y,
        length,
        origin=(-box_size_x / 2, 0.0, -length / 2),
    ).cut(assembly.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME))

    follower = assembly.get_follower_part_by_name(DEFAULT_DOVETAIL_FOLLOWER_NAME)

    assert get_volume(follower) == pytest.approx(
        get_volume(expected_follower), rel=1e-6, abs=1e-6
    )
    _assert_bboxes_close(
        get_bounding_box(follower), get_bounding_box(expected_follower)
    )


def test_parts_clearance_extends_the_leader_backwards():
    box_size_x = 20.0
    box_size_y = 6.0
    length = 32.0
    parts_clearance = 1.25

    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=8.0,
        length=length,
        box_size_x=box_size_x,
        box_size_y=box_size_y,
        taper_per_side=1.5,
        parts_clearance=parts_clearance,
    )

    leader_bb = get_bounding_box(assembly.leader)
    follower_bb = get_bounding_box(
        assembly.get_follower_part_by_name(DEFAULT_DOVETAIL_FOLLOWER_NAME)
    )

    assert leader_bb[0][1] == pytest.approx(-(box_size_y + parts_clearance), abs=1e-6)
    assert leader_bb[1][1] == pytest.approx(box_size_y, abs=1e-6)
    assert follower_bb[0][1] == pytest.approx(0.0, abs=1e-6)
    assert follower_bb[1][1] == pytest.approx(box_size_y, abs=1e-6)


def test_groove_block_depth_and_front_wall_clearance_are_independent():
    box_size_x = 20.0
    box_size_y = 6.0
    groove_box_size_y = 10.0
    front_wall_clearance = 2.5
    length = 32.0
    dovetail_width = 8.0
    taper_per_side = 1.5
    dovetail_clearance = 0.2

    assembly = create_dovetail_tongue_and_groove(
        dovetail_width=dovetail_width,
        length=length,
        box_size_x=box_size_x,
        box_size_y=box_size_y,
        groove_box_size_y=groove_box_size_y,
        front_wall_clearance=front_wall_clearance,
        taper_per_side=taper_per_side,
        dovetail_clearance=dovetail_clearance,
    )

    cutter_bb = get_bounding_box(
        assembly.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME)
    )
    follower_bb = get_bounding_box(
        assembly.get_follower_part_by_name(DEFAULT_DOVETAIL_FOLLOWER_NAME)
    )

    expected_cutter_depth = box_size_y + front_wall_clearance
    expected_front_wall_thickness = groove_box_size_y - expected_cutter_depth

    assert cutter_bb[0][1] == pytest.approx(0.0, abs=1e-6)
    assert cutter_bb[1][1] == pytest.approx(expected_cutter_depth, abs=1e-6)
    assert follower_bb[0][1] == pytest.approx(0.0, abs=1e-6)
    assert follower_bb[1][1] == pytest.approx(groove_box_size_y, abs=1e-6)
    assert follower_bb[1][1] - cutter_bb[1][1] == pytest.approx(
        expected_front_wall_thickness, abs=1e-6
    )


def test_angle_based_taper_matches_equivalent_taper_per_side():
    box_size_y = 10.0
    taper_per_side = 2.0
    angle_deg = math.degrees(math.atan(taper_per_side / box_size_y))

    with_taper = create_dovetail_tongue_and_groove(
        dovetail_width=12.0,
        length=36.0,
        box_size_x=24.0,
        box_size_y=box_size_y,
        taper_per_side=taper_per_side,
        dovetail_clearance=0.2,
    )
    with_angle = create_dovetail_tongue_and_groove(
        dovetail_width=12.0,
        length=36.0,
        box_size_x=24.0,
        box_size_y=box_size_y,
        angle_deg=angle_deg,
        dovetail_clearance=0.2,
    )

    _assert_bboxes_close(
        get_bounding_box(
            with_taper.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME)
        ),
        get_bounding_box(
            with_angle.get_cutter_part_by_name(DEFAULT_DOVETAIL_CUTTER_NAME)
        ),
    )


def test_dovetail_parameter_validation():
    with pytest.raises(ValueError, match="Specify either angle_deg or taper_per_side"):
        create_dovetail_tongue_and_groove(
            dovetail_width=10.0,
            length=30.0,
            box_size_x=25.0,
            box_size_y=5.0,
            taper_per_side=1.0,
            angle_deg=10.0,
        )

    with pytest.raises(
        ValueError, match="box_size_x must be larger than the widened groove root width"
    ):
        create_dovetail_tongue_and_groove(
            dovetail_width=10.0,
            length=30.0,
            box_size_x=11.0,
            box_size_y=5.0,
            taper_per_side=1.0,
            dovetail_clearance=0.5,
        )

    with pytest.raises(ValueError, match="parts_clearance must be non-negative"):
        create_dovetail_tongue_and_groove(
            dovetail_width=10.0,
            length=30.0,
            box_size_x=20.0,
            box_size_y=5.0,
            taper_per_side=1.0,
            parts_clearance=-0.1,
        )

    with pytest.raises(
        ValueError,
        match="groove_box_size_y must be at least box_size_y \\+ front_wall_clearance",
    ):
        create_dovetail_tongue_and_groove(
            dovetail_width=10.0,
            length=30.0,
            box_size_x=20.0,
            box_size_y=5.0,
            groove_box_size_y=6.0,
            front_wall_clearance=2.0,
            taper_per_side=1.0,
        )
