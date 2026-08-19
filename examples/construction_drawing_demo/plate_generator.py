"""Shared 60-by-20-by-5 mm plate fixture for construction-drawing Stage 0."""

from shellforgepy.simple import (
    LeaderFollowersCuttersPart,
    PartCollector,
    create_box,
    create_cylinder,
    get_clearance_hole_diameter,
)

PLATE_LENGTH_X_MM = 60.0
PLATE_WIDTH_Y_MM = 20.0
PLATE_THICKNESS_Z_MM = 5.0
M3_CLEARANCE_SIZE = "M3"
M3_CLEARANCE_TYPE = "normal"
M3_CLEARANCE_DIAMETER_MM = get_clearance_hole_diameter(
    M3_CLEARANCE_SIZE, M3_CLEARANCE_TYPE
)
HOLE_CENTERS_MM = ((15.0, 10.0), (45.0, 10.0))
PLATE_ORIGIN_MM = (0.0, 0.0, 0.0)


def create_plate_assembly():
    """Build the canonical Stage 0 plate as a production assembly leader."""

    plate = create_box(
        PLATE_LENGTH_X_MM,
        PLATE_WIDTH_Y_MM,
        PLATE_THICKNESS_Z_MM,
        origin=PLATE_ORIGIN_MM,
    )
    hole_cutters = PartCollector()
    for x, y in HOLE_CENTERS_MM:
        cutter = create_cylinder(
            M3_CLEARANCE_DIAMETER_MM / 2,
            PLATE_THICKNESS_Z_MM + 2.0,
            origin=(x, y, -1.0),
        )
        hole_cutters = hole_cutters.fuse(cutter)

    plate = plate.cut(hole_cutters)
    return LeaderFollowersCuttersPart(
        leader=plate,
        additional_data={
            "fixture": "construction_drawing_stage0_plate",
            "units": "mm",
            "coordinate_convention": "origin at plate minimum XYZ; X length, Y width, Z thickness",
            "m3_clearance_size": M3_CLEARANCE_SIZE,
            "m3_clearance_type": M3_CLEARANCE_TYPE,
            "m3_clearance_diameter_mm": M3_CLEARANCE_DIAMETER_MM,
            "hole_centers_mm": HOLE_CENTERS_MM,
        },
    )


def construction_drawing_request():
    """Return the Stage 0 request for the same plate artifact."""

    from shellforgepy.drawing import make_construction_drawing_request

    return make_construction_drawing_request(
        name="plate_top",
        parts=[
            {
                "source": "self",
                "artifact": "leader",
                "name": "plate",
            }
        ],
        units="mm",
        scale=1.0,
        view="top",
    )
