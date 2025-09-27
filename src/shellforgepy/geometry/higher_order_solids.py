import math

import numpy as np
from shellforgepy.adapters.simple import create_basic_cylinder, create_extruded_polygon
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.geometry.spherical_tools import coordinate_system_transform


def create_hex_prism(diameter, thickness, origin=(0, 0, 0)):
    """Create a hexagonal prism."""

    # Create hexagonal wire
    points = []
    for i in range(6):
        angle = i * math.pi / 3
        x = diameter * 0.5 * math.cos(angle)
        y = diameter * 0.5 * math.sin(angle)
        points.append((x, y))

    prism = create_extruded_polygon(points, thickness=thickness)

    # Translate to origin
    if origin != (0, 0, 0):
        prism = translate(*origin)(prism)

    return prism


def create_trapezoid(
    base_length,
    top_length,
    height,
    thickness,
    top_shift=0.0,
):
    """Create a trapezoidal prism using CAD-agnostic functions."""
    p1 = (-base_length / 2, 0)
    p2 = (base_length / 2, 0)
    p3 = (top_length / 2 + top_shift, height)
    p4 = (-top_length / 2 + top_shift, height)
    points = [p1, p2, p3, p4]
    return create_extruded_polygon(points, thickness=thickness)


def directed_cylinder_at(
    base_point,
    direction,
    radius,
    height,
):
    """Create a cylinder oriented along ``direction`` starting at ``base_point``.

    Args:
        base_point: XYZ coordinates of the cylinder's base centre in millimetres.
        direction: Vector indicating the extrusion direction. Must be non-zero.
        radius: Cylinder radius.
        height: Cylinder height measured along ``direction``.

    Returns:
        ``cadquery.Solid`` positioned and oriented as requested.
    """

    cylinder = create_basic_cylinder(radius=radius, height=height)

    direction = np.array(direction, dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError("Direction vector cannot be zero")
    direction /= np.linalg.norm(direction)

    if not np.allclose(direction, [0, 0, 1]):

        out_1 = np.array([0, 0, 1], dtype=np.float64)
        if np.allclose(direction, out_1):
            out_1 = np.array([1, 0, 0], dtype=np.float64)

        transformation = coordinate_system_transform(
            (0, 0, 0), (0, 0, 1), (1, 0, 0), base_point, direction, out_1
        )

        rotation = rotate(
            np.degrees(transformation["rotation_angle"]),
            axis=transformation["rotation_axis"],
        )
        the_translation = translate(
            transformation["translation"][0],
            transformation["translation"][1],
            transformation["translation"][2],
        )

        cylinder = rotation(cylinder)
        cylinder = the_translation(cylinder)

        return cylinder
    else:
        # If the direction is already aligned with Z, just translate
        cylinder = translate(base_point[0], base_point[1], base_point[2])(cylinder)
        return cylinder
