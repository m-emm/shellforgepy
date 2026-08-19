"""M3 tapped-hole plate used only by the user-facing drawing example."""

from shellforgepy.geometry.m_screws import get_core_hole_diameter
from shellforgepy.simple import LeaderFollowersCuttersPart, create_box, create_cylinder

PLATE_WIDTH_MM = 80.0
PLATE_HEIGHT_MM = 50.0
PLATE_THICKNESS_MM = 8.0
M3_SIZE = "M3"
M3_CORE_DIAMETER_MM = get_core_hole_diameter(M3_SIZE)
M3_THREAD_DEPTH_MM = 6.0
M3_FRONT_THREAD_DEPTH_MM = 5.0


def _m3_blind_tap_drill(x: float, y: float):
    """Create a 6 mm top-entry M3 tap-drill cut with a small top overshoot."""

    return create_cylinder(
        M3_CORE_DIAMETER_MM / 2.0,
        M3_THREAD_DEPTH_MM + 1.0,
        origin=(x, y, PLATE_THICKNESS_MM - M3_THREAD_DEPTH_MM),
    )


def _m3_through_tap_drill(x: float, y: float):
    return create_cylinder(
        M3_CORE_DIAMETER_MM / 2.0,
        PLATE_THICKNESS_MM + 2.0,
        origin=(x, y, -1.0),
    )


def _m3_front_tap_drill():
    """Create a 5 mm M3 tap drill from the Y-min front face into the plate."""

    return create_cylinder(
        M3_CORE_DIAMETER_MM / 2.0,
        M3_FRONT_THREAD_DEPTH_MM + 1.0,
        origin=(PLATE_WIDTH_MM / 2.0, -1.0, PLATE_THICKNESS_MM / 2.0),
        direction=(0.0, 1.0, 0.0),
    )


def create_m3_threaded_plate_assembly():
    """Create top-entry M3 pairs and one centered front-entry M3 blind hole."""

    plate = create_box(PLATE_WIDTH_MM, PLATE_HEIGHT_MM, PLATE_THICKNESS_MM)
    cutters = {
        "m3_blind_left": _m3_blind_tap_drill(20.0, 35.0),
        "m3_blind_right": _m3_blind_tap_drill(60.0, 35.0),
        "m3_through_left": _m3_through_tap_drill(20.0, 15.0),
        "m3_through_right": _m3_through_tap_drill(60.0, 15.0),
        "m3_front_center": _m3_front_tap_drill(),
    }
    for cutter in cutters.values():
        plate = plate.cut(cutter)

    assembly = LeaderFollowersCuttersPart(leader=plate)
    for name, cutter in cutters.items():
        assembly.add_named_cutter(cutter, name)
    return assembly
