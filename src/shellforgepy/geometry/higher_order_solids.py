import math
from typing import Optional

import numpy as np
from shellforgepy.adapters.simple import (
    create_basic_cylinder,
    create_extruded_polygon,
    create_solid_from_traditional_face_vertex_maps,
)
from shellforgepy.construct.alignment_operations import rotate, translate
from shellforgepy.geometry.spherical_tools import coordinate_system_transform
from shellforgepy.geometry.treapezoidal_snake_geometry import (
    create_trapezoidal_snake_geometry,
)


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


def create_ring(
    outer_radius,
    inner_radius,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    angle: Optional[float] = None,
):
    """Create a ring (hollow cylinder) using CadQuery.

    Args:
        outer_radius: Outer radius of the ring
        inner_radius: Inner radius of the ring (must be less than outer_radius)
        height: Height of the ring
        origin: Origin point as (x, y, z), defaults to (0, 0, 0)
        direction: Direction vector as (x, y, z), defaults to (0, 0, 1)
        angle: Optional angle in degrees for partial ring

    Returns:
        CadQuery solid representing the ring
    """
    if outer_radius <= inner_radius:
        raise ValueError("Outer radius must be greater than inner radius")

    # Create outer cylinder
    outer_cyl = create_basic_cylinder(outer_radius, height, origin, direction, angle)

    # Create inner cylinder to subtract
    inner_cyl = create_basic_cylinder(inner_radius, height, origin, direction, angle)

    # Cut inner from outer to create ring
    return outer_cyl.cut(inner_cyl)


def create_screw_thread(
    pitch,
    inner_radius,
    outer_radius,
    outer_thickness,
    num_turns=1,
    with_core=True,
    inner_thickness=None,
    core_height=None,
    resolution=30,
    optimize_start=False,
    optimize_start_angle=15,
    core_offset=0,
):
    """Create a helical screw thread using trapezoidal snake geometry.

    Creates a realistic helical thread by generating a trapezoidal cross-section
    and sweeping it along a helical path.

    Args:
        pitch: Distance between thread peaks
        inner_radius: Inner radius of the thread
        outer_radius: Outer radius of the thread
        outer_thickness: Thickness of the thread at outer radius
        num_turns: Number of complete turns
        with_core: Whether to include a solid core
        inner_thickness: Thickness of thread at inner radius (defaults to outer_thickness)
        core_height: Height of the core (defaults to thread height)
        resolution: Number of segments per turn
        optimize_start: Whether to optimize the thread start
        optimize_start_angle: Angle over which to optimize start (degrees)
        core_offset: Z offset for the core

    Returns:
        Solid representing the screw thread
    """
    if inner_thickness is None:
        inner_thickness = outer_thickness

    # Calculate thread geometry
    thread_height = num_turns * pitch
    total_points = int(num_turns * resolution)  # Ensure integer

    # Create trapezoidal cross-section for the thread
    thread_depth = outer_radius - inner_radius
    cross_section = np.array(
        [
            [-outer_thickness / 2, 0.0],  # Bottom left
            [outer_thickness / 2, 0.0],  # Bottom right
            [inner_thickness / 2, thread_depth],  # Top right
            [-inner_thickness / 2, thread_depth],  # Top left
        ]
    )

    # Generate helical path
    theta_values = np.linspace(0, 2 * np.pi * num_turns, total_points)

    # Apply start optimization if requested
    if optimize_start and total_points > 0:
        optimize_points = int(
            resolution * optimize_start_angle / 360.0
        )  # Ensure integer
        optimize_points = min(
            optimize_points, total_points // 4
        )  # Don't optimize more than 1/4 turn

        # Create a smooth transition for the first few points
        for i in range(optimize_points):
            scale_factor = i / optimize_points
            theta_values[i] *= scale_factor

    # Calculate helical coordinates
    x_values = inner_radius * np.cos(theta_values)
    y_values = inner_radius * np.sin(theta_values)
    z_values = (pitch / (2 * np.pi)) * theta_values

    base_points = np.column_stack([x_values, y_values, z_values])

    # Calculate outward-pointing normals (radial direction from thread axis)
    normals = np.zeros_like(base_points)
    for i in range(len(base_points)):
        normals[i, 0] = np.cos(theta_values[i])  # X component
        normals[i, 1] = np.sin(theta_values[i])  # Y component
        normals[i, 2] = 0.0  # Z component (purely radial)

    # Generate thread geometry using snake geometry
    thread_meshes = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals
    )

    # Convert each mesh segment to a solid and fuse them
    thread_solids = []
    for mesh in thread_meshes:
        mesh_data = {"vertexes": mesh["vertexes"], "faces": mesh["faces"]}
        solid = create_solid_from_traditional_face_vertex_maps(mesh_data)
        thread_solids.append(solid)

    # Fuse all thread segments together
    thread_solid = thread_solids[0]
    for solid in thread_solids[1:]:
        thread_solid = thread_solid.fuse(solid)

    # Add core if requested
    if with_core:
        if core_height is None:
            core_height = thread_height

        core = create_basic_cylinder(
            radius=inner_radius, height=core_height, origin=(0, 0, core_offset)
        )

        thread_solid = thread_solid.fuse(core)

    return thread_solid
