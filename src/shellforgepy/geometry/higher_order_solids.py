import logging
import math
from typing import Optional

import numpy as np
from shellforgepy.adapters._adapter import (
    copy_part,
    create_box,
    create_cone,
    create_cylinder,
    create_extruded_polygon,
    create_solid_from_traditional_face_vertex_maps,
    get_bounding_box,
)
from shellforgepy.construct.alignment_operations import mirror, rotate, translate
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.mesh_builders import create_cube_geometry
from shellforgepy.geometry.mesh_utils import convert_to_traditional_face_vertex_maps
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transformation_function,
)
from shellforgepy.geometry.treapezoidal_snake_geometry import (
    create_trapezoidal_snake_geometry,
)

_logger = logging.getLogger(__name__)


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


def create_isoceles_triangle(side_length, tip_angle, thickness):
    """Create an isoceles triangular prism extruded along +Z.

    The base is centered on the X axis at Y=0, and the tip lies on the positive
    Y axis at X=0.
    """
    if side_length <= 0 or thickness <= 0:
        raise ValueError("side_length and thickness must be positive")
    if tip_angle <= 0 or tip_angle >= 180:
        raise ValueError("tip_angle must be between 0 and 180 degrees")

    half_angle_rad = math.radians(tip_angle / 2.0)
    base_length = 2.0 * side_length * math.sin(half_angle_rad)
    height = side_length * math.cos(half_angle_rad)

    points = [
        (-base_length / 2.0, 0.0),
        (base_length / 2.0, 0.0),
        (0.0, height),
    ]

    return create_extruded_polygon(points, thickness=thickness)


def create_right_triangle(
    a,
    b,
    thickness,
    extrusion_direction=None,
    a_normal=None,
    b_normal=None,
):
    """Create and optionally orient a right triangular prism.

    The base triangle sits on the XY plane with vertices ``(0, 0)``, ``(b, 0)``,
    and ``(0, a)`` before any re-orientation is applied. The prism extrudes along
    the +Z axis by ``thickness``.

    Args:
        a: Length of the leg parallel to the Y axis.
        b: Length of the leg parallel to the X axis.
        thickness: Extrusion depth along the initial Z axis.
        extrusion_direction: Optional 3-vector describing the desired up direction
            of the extruded prism.
        a_normal: Optional 3-vector describing the outward normal of the edge
            running along the ``a`` leg. When supplied together with
            ``extrusion_direction`` this defines a complete target frame.
        b_normal: Optional 3-vector describing the outward normal of the edge
            running along the ``b`` leg. When supplied alongside ``a_normal`` the
            extrusion direction is inferred via the right-hand rule.

    Returns:
        CAD solid representing the triangular prism.
    """

    if a <= 0 or b <= 0 or thickness <= 0:
        raise ValueError("a, b, and thickness must be positive")

    base_points = [(0.0, 0.0), (float(b), 0.0), (0.0, float(a))]
    triangle = create_extruded_polygon(base_points, thickness=float(thickness))

    extrusion_vec = None

    # Handle the different parameter combinations
    if b_normal is not None and a_normal is not None:
        # Both normals specified - compute extrusion direction automatically
        a_vec = normalize(a_normal)
        b_vec = normalize(b_normal)

        # Compute extrusion direction as cross product (right-hand rule)
        # The extrusion direction should be perpendicular to both normals
        extrusion_vec = np.cross(a_vec, b_vec)
        extrusion_norm = np.linalg.norm(extrusion_vec)

        # Verify the vectors are not parallel
        if extrusion_norm < 1e-6:
            raise ValueError("a_normal and b_normal cannot be parallel")

        extrusion_vec = extrusion_vec / extrusion_norm
        extrusion_direction = tuple(extrusion_vec)

    elif extrusion_direction is not None and a_normal is not None:
        # Extrusion direction and a_normal specified - compute b_normal
        extrusion_vec = normalize(extrusion_direction)
        a_vec = normalize(a_normal)

        # Compute b_normal as cross product
        b_vec = np.cross(extrusion_vec, a_vec)
        b_normal = tuple(b_vec)

        # Verify the vectors are not parallel
        if np.linalg.norm(b_vec) < 1e-6:
            raise ValueError("extrusion_direction and a_normal cannot be parallel")

    elif extrusion_direction is not None:
        extrusion_vec = normalize(extrusion_direction)

    if a_normal is not None and extrusion_direction is None and extrusion_vec is None:
        # a_normal without an extrusion direction is insufficient to orient the part
        a_vec = normalize(a_normal)
        a_normal = tuple(a_vec)
    elif a_normal is not None:
        a_normal = tuple(normalize(a_normal))

    if extrusion_vec is not None:
        extrusion_direction = tuple(extrusion_vec)
    # Apply rotation if orientation vectors are specified
    if extrusion_direction is not None or a_normal is not None:
        # Default coordinate system for the triangle
        # Triangle vertices: (0,0,0), (b,0,0), (0,a,0)
        # - up (extrusion direction): (0, 0, 1) - along z-axis
        # - out (a-side normal): (0, 1, 0) - along y-axis (a-side goes from (0,0) to (0,a))
        default_origin = [0, 0, 0]
        default_up = [0, 0, 1]  # extrusion direction
        default_out = [0, -1, 0]  # a-side normal (a-side is along y-axis)

        # Target coordinate system
        target_origin = [0, 0, 0]  # keep at origin
        target_up = (
            list(extrusion_direction) if extrusion_direction is not None else default_up
        )
        target_out = list(a_normal) if a_normal is not None else default_out

        # Compute the transformation
        transform = coordinate_system_transform(
            default_origin,
            default_up,
            default_out,
            target_origin,
            target_up,
            target_out,
        )

        rotation_angle = transform["rotation_angle"]

        if abs(rotation_angle) > 1e-6:
            bb = get_bounding_box(triangle)
            center = (
                (bb[0][0] + bb[1][0]) / 2.0,
                (bb[0][1] + bb[1][1]) / 2.0,
                (bb[0][2] + bb[1][2]) / 2.0,
            )

            triangle = rotate(
                math.degrees(rotation_angle),
                center=center,
                axis=transform["rotation_axis"],
            )(triangle)

    return triangle


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

    cylinder = create_cylinder(radius=radius, height=height)

    direction = np.array(direction, dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError("Direction vector cannot be zero")
    direction /= np.linalg.norm(direction)

    if not np.allclose(direction, [0, 0, 1]):

        out_1 = np.array([0, 0, 1], dtype=np.float64)
        if np.allclose(direction, out_1):
            out_1 = np.array([1, 0, 0], dtype=np.float64)

        # check if direction is collinear with out_1
        if np.abs(np.dot(direction, out_1)) > 0.99:
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


def directed_cone_at(
    base_point,
    direction,
    radius1,
    radius2,
    height,
):
    """Create a cone oriented along ``direction`` starting at ``base_point``."""

    cone = create_cone(radius1=radius1, radius2=radius2, height=height)

    direction = np.array(direction, dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError("Direction vector cannot be zero")
    direction /= np.linalg.norm(direction)

    if not np.allclose(direction, [0, 0, 1]):

        out_1 = np.array([0, 0, 1], dtype=np.float64)
        if np.allclose(direction, out_1):
            out_1 = np.array([1, 0, 0], dtype=np.float64)

        if np.abs(np.dot(direction, out_1)) > 0.99:
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

        cone = rotation(cone)
        cone = the_translation(cone)

        return cone
    else:
        cone = translate(base_point[0], base_point[1], base_point[2])(cone)
        return cone


def directed_box_at(
    base_point,
    height_direction,
    width,
    depth,
    height,
    width_direction=None,
):
    """Create a box oriented with its height along ``height_direction`` starting at ``base_point``.

    The box is created and positioned so that its bottom face center is at ``base_point``
    and its height dimension aligns with ``height_direction``.

    Args:
        base_point: XYZ coordinates of the box's bottom face center in millimetres.
        height_direction: Vector indicating the height direction. Must be non-zero.
        width: Box width (dimension in the local X direction before rotation).
        depth: Box depth (dimension in the local Y direction before rotation).
        height: Box height measured along ``height_direction``.
        width_direction: Optional vector to specify the orientation of the width dimension.
            If None, an appropriate orthogonal direction is automatically chosen.

    Returns:
        CAD solid positioned and oriented as requested.
    """

    # Create box at origin (extends from (0,0,0) to (width,depth,height))
    box = create_box(width, depth, height)

    # First, center the box so its center is at origin
    box = translate(-width / 2, -depth / 2, -height / 2)(box)

    height_direction = np.array(height_direction, dtype=np.float64)
    if np.linalg.norm(height_direction) < 1e-8:
        raise ValueError("Height direction vector cannot be zero")
    height_direction /= np.linalg.norm(height_direction)

    # Validate and determine width direction first (even for Z-aligned case)
    if width_direction is not None:
        width_direction = np.array(width_direction, dtype=np.float64)
        if np.linalg.norm(width_direction) < 1e-8:
            raise ValueError("Width direction vector cannot be zero")
        width_direction /= np.linalg.norm(width_direction)

        # Check that width_direction is not parallel to height_direction
        if np.abs(np.dot(width_direction, height_direction)) > 0.99:
            raise ValueError("Width direction cannot be parallel to height direction")
    else:
        # Automatically choose a width direction orthogonal to height_direction
        # Try to use [1, 0, 0] first, then [0, 1, 0] if that's too parallel
        candidate_width = np.array([1, 0, 0], dtype=np.float64)
        if np.abs(np.dot(candidate_width, height_direction)) > 0.9:
            candidate_width = np.array([0, 1, 0], dtype=np.float64)

        # Make it orthogonal to height_direction using Gram-Schmidt
        width_direction = (
            candidate_width
            - np.dot(candidate_width, height_direction) * height_direction
        )
        width_direction /= np.linalg.norm(width_direction)

    # Check if we need to rotate at all
    if np.allclose(height_direction, [0, 0, 1]):
        # If height direction is already aligned with Z, position so bottom face center is at base_point
        box = translate(base_point[0], base_point[1], base_point[2] + height / 2)(box)
        return box

    # Use coordinate system transform to position the box
    # Default centered box coordinate system: up = Z (height), out = X (width)
    default_origin = (0, 0, 0)
    default_up = (0, 0, 1)  # height direction in default box
    default_out = (1, 0, 0)  # width direction in default box

    # Target coordinate system - we want the box center offset by +height/2 from base_point in height direction
    target_offset = height / 2 * height_direction
    target_origin = (
        base_point[0] + target_offset[0],
        base_point[1] + target_offset[1],
        base_point[2] + target_offset[2],
    )
    target_up = tuple(height_direction.tolist())
    target_out = tuple(width_direction.tolist())

    transformation = coordinate_system_transform(
        default_origin, default_up, default_out, target_origin, target_up, target_out
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

    box = rotation(box)
    box = the_translation(box)

    return box


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
        solid representing the ring
    """
    if outer_radius <= inner_radius:
        raise ValueError("Outer radius must be greater than inner radius")

    # Create outer cylinder
    outer_cyl = create_cylinder(outer_radius, height, origin, direction, angle)

    # Create inner cylinder to subtract
    inner_cyl = create_cylinder(inner_radius, height, origin, direction, angle)

    # Cut inner from outer to create ring
    return outer_cyl.cut(inner_cyl)


def create_ring_segment_between_points(
    p1, p2, third_point_on_plane, inner_radius, outer_radius, height
):
    """Create a ring segment between two points, where the ring plane is defined by the normal.

    Args:
        p1: First point on the ring segment (3-tuple)
        p2: Second point on the ring segment (3-tuple)
        third_point_on_plane: A third point on the desired ring plane
        inner_radius: Inner radius of the ring segment
        outer_radius: Outer radius of the ring segment
        height: Height of the ring segment
    Returns:
        Solid representing the ring segment
    """

    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    third_point_on_plane = np.array(third_point_on_plane, dtype=np.float64)
    edge = p2 - p1
    edge_length = np.linalg.norm(edge)
    if edge_length <= 0:
        raise ValueError("p1 and p2 must be distinct points")

    plane_normal = np.cross(edge, third_point_on_plane - p1)
    if np.linalg.norm(plane_normal) <= 1e-8:
        raise ValueError("p1, p2, and third_point_on_plane must not be colinear")

    plane_normal = normalize(plane_normal)

    edge_direction = normalize(edge)
    edge_centroid = (p1 + p2) / 2.0

    radius_ray_direction = normalize(np.cross(plane_normal, edge_direction))

    middle_radius = (inner_radius + outer_radius) / 2.0
    if middle_radius < edge_length / 2.0:
        raise ValueError("Points are too far apart for the provided radii")

    distance_to_center = math.sqrt(middle_radius**2 - (edge_length / 2.0) ** 2)
    center_point = edge_centroid + radius_ray_direction * (distance_to_center)

    angle = math.degrees(
        math.acos(np.dot(normalize(p1 - center_point), normalize(p2 - center_point)))
    )

    ring_segment = create_ring(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        height=height,
        angle=angle,
    )
    ring_segment = translate(0, 0, -height / 2)(ring_segment)

    cstf = coordinate_system_transformation_function(
        origin_a=(0, 0, 0),
        up_a=(0, 0, 1),
        out_a=(1, 0, 0),
        origin_b=center_point,
        up_b=plane_normal,
        out_b=p1 - center_point,
        degree_rotation_function_generator=rotate,
        translation_function_generator=translate,
    )

    ring_segment = cstf(ring_segment)

    return ring_segment


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
        core = create_cylinder(
            radius=inner_radius, height=core_height, origin=(0, 0, core_bottom)
        )

        final_thread = final_thread.fuse(core)

    return final_thread


def create_rounded_slab(
    length,
    width,
    thick,
    round_radius,
    rounding_flags={(1, 1): True, (-1, 1): True, (-1, -1): True, (1, -1): True},
):
    retval = create_box(length, width, thick)
    retval = translate(-length / 2, -width / 2, 0)(retval)
    stencil = create_box(round_radius, round_radius, thick)
    rounder = create_cylinder(
        round_radius,
        thick,
    )
    stencil = stencil.cut(rounder)

    for i in [-1, 1]:

        moved_stencil = translate(
            length / 2 - round_radius, width / 2 - round_radius, 0
        )(stencil)
        if i == -1:
            moved_stencil = mirror((1, 0, 0), (0, 0, 0))(moved_stencil)

        for j in [-1, 1]:
            moved_stencil2 = copy_part(moved_stencil)
            if j == -1:
                moved_stencil2 = mirror((0, 1, 0), (0, 0, 0))(moved_stencil)

            if rounding_flags[(i, j)]:
                retval = retval.cut(moved_stencil2)
    retval = translate(length / 2, width / 2, 0)(retval)
    return retval


def create_distorted_cube(corners):
    if len(corners) != 8:
        raise ValueError("Distorted cube requires exactly 8 corners")
    maps = {
        "vertexes": {i: tuple(map(float, corners[i])) for i in range(8)},
        "faces": {
            0: [0, 2, 1],
            1: [0, 3, 2],
            2: [4, 5, 6],
            3: [4, 6, 7],
            4: [0, 1, 5],
            5: [0, 5, 4],
            6: [2, 3, 6],
            7: [3, 7, 6],
            8: [1, 2, 5],
            9: [2, 6, 5],
            10: [0, 4, 3],
            11: [3, 4, 7],
        },
    }
    return create_solid_from_traditional_face_vertex_maps(maps)


def create_triangular_prism_geometry(corners):
    if len(corners) != 6:
        raise ValueError("Triangular prism requires exactly 6 corners")
    maps = {
        "vertexes": {i: tuple(map(float, corners[i])) for i in range(6)},
        "faces": {
            0: [0, 2, 1],  # bottom
            1: [3, 4, 5],  # top
            2: [0, 1, 4],
            3: [0, 4, 3],
            4: [1, 2, 5],
            5: [1, 5, 4],
            6: [2, 0, 3],
            7: [2, 3, 5],
        },
    }
    return maps


def create_triangular_prism(corners):
    maps = create_triangular_prism_geometry(corners)
    return create_solid_from_traditional_face_vertex_maps(maps)


def materialize_bounding_box(part):
    bb = get_bounding_box(part)
    corners = [
        (bb[0][0], bb[0][1], bb[0][2]),
        (bb[1][0], bb[0][1], bb[0][2]),
        (bb[1][0], bb[1][1], bb[0][2]),
        (bb[0][0], bb[1][1], bb[0][2]),
        (bb[0][0], bb[0][1], bb[1][2]),
        (bb[1][0], bb[0][1], bb[1][2]),
        (bb[1][0], bb[1][1], bb[1][2]),
        (bb[0][0], bb[1][1], bb[1][2]),
    ]

    return create_distorted_cube(corners)


def create_pyramid_stump(bottom_width, top_width, bottom_depth, top_depth, height):
    # vertices from the “spherical cube” are ±0.5 in all coords with this radius
    vertices, faces = create_cube_geometry(math.sqrt(3) / 2)

    new_vertices = []
    for x, y, z in vertices:
        # Map z ∈ {-0.5, +0.5} → t ∈ {0, 1}
        t = z + 0.5  # since range is exactly 1.0

        # Interpolate half-sizes
        sx = (1 - t) * (bottom_width / 2.0) + t * (top_width / 2.0)
        sy = (1 - t) * (bottom_depth / 2.0) + t * (top_depth / 2.0)

        # Use sign(x), sign(y) to choose corner; z goes 0..height
        nx = np.sign(x) * sx
        ny = np.sign(y) * sy
        nz = t * height

        new_vertices.append([nx, ny, nz])

    solid = create_solid_from_traditional_face_vertex_maps(
        convert_to_traditional_face_vertex_maps(np.asarray(new_vertices), faces)
    )
    return solid


def create_conical_ring(inner_radius, thickness, height, angle_deg):
    angle_inset = math.tan(math.radians(angle_deg)) * height

    outer_cone = create_cone(
        inner_radius + thickness,
        inner_radius + thickness - angle_inset,
        height,
    )
    inner_cone = create_cone(
        inner_radius,
        inner_radius - angle_inset,
        height + 2,
    )
    ring = outer_cone.cut(inner_cone)
    return ring
