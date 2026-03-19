"""Dovetail tongue-and-groove helpers."""

import math

from shellforgepy.adapters._adapter import create_box, cut_parts, fuse_parts
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.geometry.higher_order_solids import create_pyramid_stump

DEFAULT_DOVETAIL_FOLLOWER_NAME = "groove_part"
DEFAULT_DOVETAIL_CUTTER_NAME = "groove_cutter"


def _require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _resolve_taper_per_side(box_size_y, taper_per_side, angle_deg):
    if angle_deg is not None and taper_per_side != 0:
        raise ValueError("Specify either angle_deg or taper_per_side, not both")

    if angle_deg is None:
        taper = taper_per_side
    else:
        if angle_deg <= 0 or angle_deg >= 90:
            raise ValueError("angle_deg must be between 0 and 90 degrees")
        taper = math.tan(math.radians(angle_deg)) * box_size_y

    if taper < 0:
        raise ValueError("taper_per_side must be non-negative")

    return taper


def _create_vertical_dovetail(width_at_opening, width_at_back, length, depth):
    """Create a dovetail tongue that runs along Z and points towards +Y."""

    # The opening sits at Y=0 and the undercut widens towards +Y so the
    # mating parts are retained when the tongue sits inside the groove.
    dovetail = create_pyramid_stump(
        bottom_width=width_at_back,
        top_width=width_at_opening,
        bottom_depth=length,
        top_depth=length,
        height=depth,
    )
    dovetail = rotate(90, axis=(1, 0, 0))(dovetail)
    return translate(0, depth, 0)(dovetail)


def create_dovetail_tongue_and_groove(
    dovetail_width,
    length,
    box_size_x,
    box_size_y,
    *,
    taper_per_side=0.0,
    angle_deg=None,
    dovetail_clearance=0.0,
    parts_clearance=0.0,
    groove_box_size_y=None,
    front_wall_clearance=0.0,
) -> LeaderFollowersCuttersPart:
    """Create a dovetail tongue, matching groove part, and groove cutter.

    Args:
        dovetail_width: Width of the tongue at the groove opening.
        length: Length of the dovetail along the Z axis.
        box_size_x: Shared X width of both supporting boxes.
        box_size_y: Depth of the tongue side and of the dovetail engagement.
        taper_per_side: Extra width added per side at the dovetail back.
        angle_deg: Optional alternative to ``taper_per_side``.
        dovetail_clearance: Lateral clearance between tongue and groove walls.
        parts_clearance: Clearance between the two mating flat faces at the opening.
        groove_box_size_y: Optional independent depth of the groove-side box.
        front_wall_clearance: Extra clearance behind the tongue tip inside the groove.

    The returned composite uses:
    - leader: a tongue part with its support box on the negative Y side
    - follower: a box with a matching groove opening towards -Y
    - cutter: the named groove cutter used to create that follower

    The dovetail runs along the Z axis, the tongue points towards +Y, and the
    groove opens towards -Y.
    """

    _require_positive("dovetail_width", dovetail_width)
    _require_positive("length", length)
    _require_positive("box_size_x", box_size_x)
    _require_positive("box_size_y", box_size_y)

    if dovetail_clearance < 0:
        raise ValueError("dovetail_clearance must be non-negative")
    if parts_clearance < 0:
        raise ValueError("parts_clearance must be non-negative")
    if front_wall_clearance < 0:
        raise ValueError("front_wall_clearance must be non-negative")

    if groove_box_size_y is None:
        groove_box_size_y = box_size_y
    _require_positive("groove_box_size_y", groove_box_size_y)

    minimum_required_groove_depth = box_size_y + front_wall_clearance
    if groove_box_size_y < minimum_required_groove_depth:
        raise ValueError(
            "groove_box_size_y must be at least box_size_y + front_wall_clearance"
        )

    taper = _resolve_taper_per_side(box_size_y, taper_per_side, angle_deg)

    tongue_back_width = dovetail_width + 2 * taper
    groove_opening_width = dovetail_width + 2 * dovetail_clearance
    groove_back_width = tongue_back_width + 2 * dovetail_clearance

    if groove_back_width >= box_size_x:
        raise ValueError("box_size_x must be larger than the widened groove root width")

    tongue = _create_vertical_dovetail(
        width_at_opening=dovetail_width,
        width_at_back=tongue_back_width,
        length=length,
        depth=box_size_y,
    )

    tongue_part = create_box(
        box_size_x,
        box_size_y,
        length,
        origin=(-box_size_x / 2, -(box_size_y + parts_clearance), -length / 2),
    )
    tongue_part = fuse_parts(tongue_part, tongue)

    if parts_clearance > 0:
        clearance_bridge = create_box(
            dovetail_width,
            parts_clearance,
            length,
            origin=(-dovetail_width / 2, -parts_clearance, -length / 2),
        )
        tongue_part = fuse_parts(tongue_part, clearance_bridge)

    groove_cutter = _create_vertical_dovetail(
        width_at_opening=groove_opening_width,
        width_at_back=groove_back_width,
        length=length,
        depth=box_size_y,
    )

    if parts_clearance > 0:
        groove_entry_clearance = create_box(
            groove_opening_width,
            parts_clearance,
            length,
            origin=(-groove_opening_width / 2, 0.0, -length / 2),
        )
        groove_cutter = fuse_parts(groove_cutter, groove_entry_clearance)

    if front_wall_clearance > 0:
        groove_front_wall_clearance = create_box(
            groove_back_width,
            front_wall_clearance,
            length,
            origin=(-groove_back_width / 2, box_size_y, -length / 2),
        )
        groove_cutter = fuse_parts(groove_cutter, groove_front_wall_clearance)

    groove_part = create_box(
        box_size_x,
        groove_box_size_y,
        length,
        origin=(-box_size_x / 2, 0.0, -length / 2),
    )
    groove_part = cut_parts(groove_part, groove_cutter)

    return LeaderFollowersCuttersPart(
        leader=tongue_part,
        followers=[groove_part],
        cutters=[groove_cutter],
        follower_names=[DEFAULT_DOVETAIL_FOLLOWER_NAME],
        cutter_names=[DEFAULT_DOVETAIL_CUTTER_NAME],
    )


__all__ = [
    "DEFAULT_DOVETAIL_CUTTER_NAME",
    "DEFAULT_DOVETAIL_FOLLOWER_NAME",
    "create_dovetail_tongue_and_groove",
]
