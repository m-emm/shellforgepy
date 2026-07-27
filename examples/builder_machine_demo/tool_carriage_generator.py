"""Geometry generator for the shape-aware inspection-gantry carriage."""

from shellforgepy.simple import (
    LeaderFollowersCuttersPart,
    create_box,
    create_filleted_box,
    get_bounding_box_size,
    translate,
)


def create_tool_carriage(
    *,
    bridge,
    carriage_width,
    wall_thickness,
    running_clearance,
    tool_plate_thickness,
    tool_plate_drop,
):
    """Create a slide whose opening is derived from the injected bridge."""

    _, bridge_depth, bridge_height = get_bounding_box_size(bridge)
    outer_depth = bridge_depth + 2 * wall_thickness
    outer_height = bridge_height + 2 * wall_thickness

    outer = create_filleted_box(
        carriage_width,
        outer_depth,
        outer_height,
        fillet_radius=2,
        no_fillets_at=[],
    )
    opening = create_box(
        carriage_width + 2,
        bridge_depth + 2 * running_clearance,
        bridge_height + 2 * running_clearance,
        origin=(
            -1,
            wall_thickness - running_clearance,
            wall_thickness - running_clearance,
        ),
    )
    slide = outer.cut(opening)

    tool_plate_height = bridge_height + tool_plate_drop
    tool_plate = create_filleted_box(
        carriage_width,
        tool_plate_thickness,
        tool_plate_height,
        fillet_radius=2,
        no_fillets_at=[],
    )
    tool_plate = translate(
        0,
        -tool_plate_thickness,
        -tool_plate_drop,
    )(tool_plate)

    return LeaderFollowersCuttersPart(
        leader=slide,
        followers=[tool_plate],
        follower_names=["tool_plate"],
    )
