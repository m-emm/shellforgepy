import math
from typing import Optional

import numpy as np
from shellforgepy.adapters._adapter import (
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
    and sweeping it along a helical path, following the original FreeCAD implementation.

    Args:
        pitch: Distance between thread peaks
        inner_radius: Inner radius of the thread
        outer_radius: Outer radius of the thread
        outer_thickness: Thickness of the thread at outer radius
        num_turns: Number of complete turns
        with_core: Whether to include a solid core
        inner_thickness: Thickness of thread at inner radius (defaults to pitch - outer_thickness)
        core_height: Height of the core (defaults to calculated minimum)
        resolution: Number of segments per turn
        optimize_start: Whether to optimize the thread start
        optimize_start_angle: Angle over which to optimize start (degrees)
        core_offset: Z offset for the core

    Returns:
        Solid representing the screw thread
    """
    # Fix the default inner_thickness to match original implementation
    if inner_thickness is None:
        inner_thickness = pitch - outer_thickness

    # Calculate turn structure like the original
    whole_turns = int(num_turns)
    partial_turn = num_turns - whole_turns
    partial_turn_segments = 0
    if partial_turn > 0:
        partial_turn_segments = int(resolution * partial_turn)

    # Convert angles to radians
    optimize_start_angle_rad = math.radians(optimize_start_angle)

    def construct_thread_for_turn(turn_index, is_partial=False, num_segments=None):
        """Construct thread geometry for one turn using snake geometry."""
        if num_segments is None:
            num_segments = resolution

        # Create path points for this turn
        base_points = []
        normals = []

        for i in range(num_segments + 1):
            # Calculate angle for this segment
            angle = 2 * math.pi * i / resolution

            # Apply optimization for first turn if requested
            current_outer_radius = outer_radius
            if turn_index == 0 and optimize_start and angle < optimize_start_angle_rad:
                # Gradually transition from inner to outer radius
                radius_factor = angle / optimize_start_angle_rad
                current_outer_radius = (inner_radius + outer_radius) / 2 + (
                    outer_radius - inner_radius
                ) / 2 * radius_factor

            # Use the middle radius between inner and outer for the helical path
            # This matches the original's approach of having separate inner/outer paths
            path_radius = (inner_radius + current_outer_radius) / 2

            x = path_radius * math.cos(angle)
            y = path_radius * math.sin(angle)
            z = pitch * i / resolution + turn_index * pitch

            base_points.append([x, y, z])

            # Normal points radially outward
            normals.append([math.cos(angle), math.sin(angle), 0.0])

        base_points = np.array(base_points)
        normals = np.array(normals)

        # Create proper trapezoidal cross-section for thread profile
        # The cross-section represents the thread's shape in the radial-axial plane
        thread_radial_extent = outer_radius - inner_radius

        cross_section = np.array(
            [
                # Bottom of thread (at inner radius side)
                [-inner_thickness / 2, -thread_radial_extent / 2],  # Bottom left
                [inner_thickness / 2, -thread_radial_extent / 2],  # Bottom right
                # Top of thread (at outer radius side)
                [outer_thickness / 2, thread_radial_extent / 2],  # Top right
                [-outer_thickness / 2, thread_radial_extent / 2],  # Top left
            ]
        )

        # Generate mesh using snake geometry
        try:
            thread_meshes = create_trapezoidal_snake_geometry(
                cross_section, base_points, normals
            )

            # Convert meshes to solids and fuse
            turn_solids = []
            for mesh in thread_meshes:
                mesh_data = {"vertexes": mesh["vertexes"], "faces": mesh["faces"]}
                solid = create_solid_from_traditional_face_vertex_maps(mesh_data)
                if solid is not None:
                    turn_solids.append(solid)

            if not turn_solids:
                return None

            # Fuse all segments for this turn
            turn_solid = turn_solids[0]
            for solid in turn_solids[1:]:
                turn_solid = turn_solid.fuse(solid)

            return turn_solid

        except Exception as e:
            print(f"Error creating thread turn {turn_index}: {e}")
            return None

    # Build the complete thread
    thread_parts = []

    # Create full turns
    for turn_index in range(whole_turns):
        if turn_index == 0 and optimize_start:
            # First turn with optimization
            turn_solid = construct_thread_for_turn(turn_index, is_partial=False)
        else:
            # Regular turn
            turn_solid = construct_thread_for_turn(turn_index, is_partial=False)

        if turn_solid is not None:
            thread_parts.append(turn_solid)

    # Create partial turn if needed
    if partial_turn > 0:
        partial_solid = construct_thread_for_turn(
            whole_turns, is_partial=True, num_segments=partial_turn_segments
        )
        if partial_solid is not None:
            thread_parts.append(partial_solid)

    if not thread_parts:
        raise ValueError("Failed to create any thread geometry")

    # Fuse all thread parts
    final_thread = thread_parts[0]
    for part in thread_parts[1:]:
        final_thread = final_thread.fuse(part)

    # Add core if requested (following original logic)
    if with_core:
        from shellforgepy.adapters._adapter import get_bounding_box

        bbox = get_bounding_box(final_thread)
        lowest_z = bbox[0][2]  # (xmin, ymin, zmin) -> zmin
        highest_z = bbox[1][2]  # (xmax, ymax, zmax) -> zmax

        core_height_tolerance = 0.05
        min_core_height = highest_z - lowest_z

        if core_height is None:
            core_height = min_core_height + core_height_tolerance

        if core_height < min_core_height:
            raise ValueError(
                f"Core height ({core_height}) must be greater than the minimum core height ({min_core_height})"
            )

        core_top = lowest_z + core_height - core_offset
        core_bottom = lowest_z - core_offset

        if core_top < highest_z:
            raise ValueError(
                f"Core top ({core_top}) must be greater than the highest point ({highest_z})"
            )

        # Create and position the core
        core = create_basic_cylinder(
            radius=inner_radius, height=core_height, origin=(0, 0, core_bottom)
        )

        final_thread = final_thread.fuse(core)

    return final_thread
