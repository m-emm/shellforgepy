"""Geometry generator for the inspection-gantry base assembly."""

from shellforgepy.simple import (
    LeaderFollowersCuttersPart,
    create_box,
    create_filleted_box,
    translate,
)


def create_machine_base(
    *,
    length,
    depth,
    thickness,
    profile_size,
    upright_inset,
):
    """Create a base with named pads that act as assembly interfaces."""

    base = create_filleted_box(
        length,
        depth,
        thickness,
        fillet_radius=3,
        no_fillets_at=[],
    )

    pad_border = 3
    pad_height = 3
    pad_size = profile_size + 2 * pad_border
    pad = create_filleted_box(
        pad_size,
        pad_size,
        pad_height,
        fillet_radius=0.8,
        no_fillets_at=[],
    )
    pad_y = (depth - pad_size) / 2
    left_pad = translate(
        upright_inset - pad_border,
        pad_y,
        thickness,
    )(pad)
    right_pad = translate(
        length - upright_inset - profile_size - pad_border,
        pad_y,
        thickness,
    )(pad)

    workpiece_length = length * 0.38
    workpiece_depth = depth * 0.42
    workpiece_height = 18
    workpiece = create_box(workpiece_length, workpiece_depth, workpiece_height)
    workpiece = translate(
        (length - workpiece_length) / 2,
        (depth - workpiece_depth) / 2,
        thickness,
    )(workpiece)

    return LeaderFollowersCuttersPart(
        leader=base,
        followers=[left_pad, right_pad],
        follower_names=["left_upright_pad", "right_upright_pad"],
        non_production_parts=[workpiece],
        non_production_names=["workpiece_reference"],
    )
