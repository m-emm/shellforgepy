import math

import pytest
from shellforgepy.simple import (
    LeaderFollowersCuttersPart,
    PartCollector,
    create_box,
    create_cylinder,
    get_bounding_box,
    get_clearance_hole_diameter,
    get_volume,
)

# Keep this fixture local to the tests. The example intentionally duplicates
# the geometry so user-facing examples and tests remain independent.
PLATE_LENGTH_X_MM = 60.0
PLATE_WIDTH_Y_MM = 20.0
PLATE_THICKNESS_Z_MM = 5.0
M3_CLEARANCE_DIAMETER_MM = get_clearance_hole_diameter("M3", "normal")
HOLE_CENTERS_MM = ((15.0, 10.0), (45.0, 10.0))


def _create_plate_assembly_for_test():
    plate = create_box(
        PLATE_LENGTH_X_MM,
        PLATE_WIDTH_Y_MM,
        PLATE_THICKNESS_Z_MM,
        origin=(0.0, 0.0, 0.0),
    )
    hole_cutters = PartCollector()
    for x, y in HOLE_CENTERS_MM:
        hole_cutters = hole_cutters.fuse(
            create_cylinder(
                M3_CLEARANCE_DIAMETER_MM / 2,
                PLATE_THICKNESS_Z_MM + 2.0,
                origin=(x, y, -1.0),
            )
        )
    return LeaderFollowersCuttersPart(
        leader=plate.cut(hole_cutters),
        additional_data={
            "m3_clearance_diameter_mm": M3_CLEARANCE_DIAMETER_MM,
            "hole_centers_mm": HOLE_CENTERS_MM,
        },
    )


def test_stage0_plate_fixture_has_explicit_concept_dimensions_and_holes():
    assembly = _create_plate_assembly_for_test()

    assert get_bounding_box(assembly.leader) == (
        (0.0, 0.0, 0.0),
        (60.0, 20.0, 5.0),
    )
    assert assembly.additional_data["m3_clearance_diameter_mm"] == 3.4
    assert assembly.additional_data["hole_centers_mm"] == HOLE_CENTERS_MM
    assert get_volume(assembly.leader) == pytest.approx(
        60.0 * 20.0 * 5.0
        - len(HOLE_CENTERS_MM) * math.pi * (M3_CLEARANCE_DIAMETER_MM / 2) ** 2 * 5.0,
        rel=1e-6,
    )
