"""Geometry generator for the inspection-gantry probe reference."""

from shellforgepy.simple import LeaderFollowersCuttersPart, create_cylinder


def create_inspection_probe(*, radius, length, tip_length):
    """Create a simple purchased-part reference with a named probe tip."""

    body = create_cylinder(radius, length)
    tip = create_cylinder(
        radius * 0.28,
        tip_length,
        origin=(0, 0, -tip_length),
    )
    return LeaderFollowersCuttersPart(
        leader=body,
        followers=[tip],
        follower_names=["probe_tip"],
    )
