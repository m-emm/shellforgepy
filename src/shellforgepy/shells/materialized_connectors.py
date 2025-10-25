"""CAD-agnostic connector generation utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence, Tuple, Union

import numpy as np
from shellforgepy.adapters._adapter import (
    create_box,
    create_cylinder,
    create_extruded_polygon,
    create_solid_from_traditional_face_vertex_maps,
)
from shellforgepy.construct.alignment import Alignment
from shellforgepy.construct.alignment_operations import (
    align,
    chain_translations,
    rotate,
    translate,
)
from shellforgepy.construct.construct_utils import normalize
from shellforgepy.geometry.higher_order_solids import (
    create_trapezoid,
    directed_cylinder_at,
)
from shellforgepy.geometry.m_screws import create_nut, m_screws_table
from shellforgepy.geometry.spherical_tools import (
    coordinate_system_transform as _coordinate_system_transform,
)

_logger = logging.getLogger(__name__)

VectorLike = Union[Sequence[float], np.ndarray, Tuple[float, float, float]]


def _to_tuple3(value):
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):  # pragma: no cover - defensive
        raise ValueError(f"Expected 3D vector, got shape {arr.shape}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def compute_out_vector(
    normal,
    triangle,
    edge_centroid,
    edge_vector,
):
    tri_vertices = [np.asarray(v, dtype=float) for v in triangle]
    tri_centroid = sum(tri_vertices) / 3.0

    out = np.cross(edge_vector, normal)
    if np.linalg.norm(out) < 1e-6:
        raise ValueError("Degenerate orientation: edge and normal are parallel")

    out = normalize(out)
    to_centroid = tri_centroid - np.asarray(edge_centroid, dtype=float)
    if np.dot(out, to_centroid) < 0:
        out = -out
    return out


@dataclass(frozen=True)
class CoordinateTransform:
    rotation_axis: Tuple[float, float, float]
    rotation_angle: float
    translation: Tuple[float, float, float]


def coordinate_system_transform(
    origin_a,
    up_a,
    out_a,
    origin_b,
    up_b,
    out_b,
):
    transform = _coordinate_system_transform(
        origin_a=origin_a,
        up_a=up_a,
        out_a=out_a,
        origin_b=origin_b,
        up_b=up_b,
        out_b=out_b,
    )

    return CoordinateTransform(
        rotation_axis=_to_tuple3(transform["rotation_axis"]),
        rotation_angle=float(transform["rotation_angle"]),
        translation=_to_tuple3(transform["translation"]),
    )


def _apply_transform(shape, transform):
    axis_vec = _to_tuple3(transform.rotation_axis)
    angle_deg = math.degrees(transform.rotation_angle)
    if np.linalg.norm(axis_vec) > 0 and not math.isclose(angle_deg, 0.0, abs_tol=1e-8):
        shape = rotate(angle_deg, axis=axis_vec, center=(0, 0, 0))(shape)
    translation_vec = _to_tuple3(transform.translation)
    return translate(*translation_vec)(shape)


def _create_box(
    length,
    width,
    height,
    base_point,
):
    return create_box(length, width, height, origin=_to_tuple3(base_point))


# create_trapezoid is now imported from higher_order_solids


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


def create_nut(size, height=None, slack=0.0, no_hole=False):
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size {size}")
    nut_size = m_screws_table[size]["nut_size"] / math.cos(math.radians(30)) + slack
    if height is None:
        height = m_screws_table[size]["nut_thickness"]

    # Create hexagonal points
    points = []
    for i in range(6):
        angle = i * math.pi / 3
        x = nut_size * 0.5 * math.cos(angle)
        y = nut_size * 0.5 * math.sin(angle)
        points.append((x, y))

    hex_prism = create_extruded_polygon(points, thickness=height)
    if no_hole:
        return hex_prism
    clearance = m_screws_table[size]["clearance_hole_normal"] / 2
    hole = create_cylinder(clearance, height)
    return hex_prism.cut(hole)


BIG_THING = 200


def compute_transforms_from_hint(
    hint,
    male_female_region_calculator=None,
):
    region_a, region_b = hint.region_a, hint.region_b

    if region_a == region_b:
        raise ValueError("Regions for connector hint must be different")

    if male_female_region_calculator is not None:
        male_region, female_region = male_female_region_calculator(hint)
    else:
        male_region = max(region_a, region_b)
        female_region = min(region_a, region_b)

    male_normal = (
        hint.triangle_a_normal if region_a == male_region else hint.triangle_b_normal
    )
    female_normal = (
        hint.triangle_b_normal if region_a == male_region else hint.triangle_a_normal
    )

    origin = np.asarray(hint.edge_centroid, dtype=float)
    male_tri = (
        hint.triangle_a_vertices
        if region_a == male_region
        else hint.triangle_b_vertices
    )
    female_tri = (
        hint.triangle_b_vertices
        if region_a == male_region
        else hint.triangle_a_vertices
    )

    male_out = compute_out_vector(male_normal, male_tri, origin, hint.edge_vector)
    female_out = compute_out_vector(female_normal, female_tri, origin, hint.edge_vector)

    def build_transform(up_vec, out_vec, origin_vec):
        return coordinate_system_transform(
            origin_a=(0, 0, 0),
            up_a=(0, 0, 1),
            out_a=(0, 1, 0),
            origin_b=origin_vec,
            up_b=up_vec,
            out_b=out_vec,
        )

    tf_male = build_transform(male_normal, male_out, origin)
    tf_female = build_transform(female_normal, female_out, origin)

    def apply_tf(shape, transform):
        return _apply_transform(shape, transform)

    return SimpleNamespace(
        apply_tf=apply_tf,
        tf_male=tf_male,
        tf_female=tf_female,
        male_region=male_region,
        female_region=female_region,
        male_normal=np.asarray(male_normal, dtype=float),
        female_normal=np.asarray(female_normal, dtype=float),
        male_out=male_out,
        female_out=female_out,
    )


def create_connector_parts_from_hint(
    hint,
    connector_length,
    connector_width,
    connector_thickness,
    connector_cyl_radius,
    connector_cylinder_length,
    connector_slack,
    connector_male_side_expansion=0.0,
):
    transforms = compute_transforms_from_hint(hint)

    male_slab = _create_box(
        connector_length + 2 * connector_male_side_expansion,
        connector_width,
        connector_thickness,
        (
            -connector_length / 2 - connector_male_side_expansion,
            0,
            -connector_thickness,
        ),
    )
    male_slab = transforms.apply_tf(male_slab, transforms.tf_male)

    if connector_male_side_expansion > 0:
        female_slab = create_trapezoid(
            connector_length + 2 * connector_male_side_expansion,
            connector_length,
            connector_width,
            connector_thickness,
        )
        female_slab = translate(connector_length / 2, 0, 0)(female_slab)
    else:
        female_slab = _create_box(
            connector_length,
            connector_width,
            connector_thickness,
            (-connector_length / 2, 0, -connector_thickness),
        )

    female_slab = transforms.apply_tf(female_slab, transforms.tf_female)

    knob = directed_cylinder_at(
        (-connector_cylinder_length / 2, connector_width, 0),
        (1, 0, 0),
        connector_cyl_radius,
        connector_cylinder_length,
    )
    knob = transforms.apply_tf(knob, transforms.tf_female)

    cutter = directed_cylinder_at(
        (-connector_cylinder_length / 2, connector_width, 0),
        (1, 0, 0),
        connector_cyl_radius + connector_slack,
        connector_cylinder_length + connector_slack,
    )
    cutter = transforms.apply_tf(cutter, transforms.tf_female)

    male_connector = male_slab.fuse(female_slab).fuse(knob)
    return transforms.male_region, transforms.female_region, male_connector, cutter


def create_nut_holder_cutter(size, slack, drill):
    nut_dimension = m_screws_table[size]["nut_size"]
    corner_distance = nut_dimension / math.cos(math.radians(30))

    height = m_screws_table[size]["nut_thickness"] + 2 * slack
    cutter = create_nut(size, height=height, slack=slack, no_hole=True)
    cutter = rotate(30)(cutter)
    cutter = rotate(90, axis=(1, 0, 0))(cutter)

    cutter = align(cutter, drill, Alignment.CENTER)

    rest = _create_box(
        nut_dimension + 2 * slack,
        height,
        BIG_THING / 10,
        (-(nut_dimension + 2 * slack) / 2, -height / 2, 0),
    )
    rest = align(rest, cutter, Alignment.CENTER)
    rest = align(rest, cutter, Alignment.TOP)
    rest = translate(0, 0, -corner_distance / 2)(rest)

    return cutter.fuse(rest)


def line_dist(alpha, l, d):
    """
    Minimal height y0 so that a horizontal segment of length l, centered at x=0
    and placed at y=y0, has perpendicular distance at least d from both rays
    forming a symmetric wedge of opening angle alpha around the +y axis.

    Parameters
    ----------
    alpha : float
        Angle between the two rays. **Radians** expected.
        (If alpha > pi, we assume degrees and convert.)
    l : float
        Length of the horizontal segment (>= 0).
    d : float
        Required clearance to each ray (>= 0).

    Returns
    -------
    y0 : float
        Minimal height. Returns math.inf for alpha <= 0.

    Formula
    -------
    y0 = ( d + 0.5*l*cos(alpha/2) ) / sin(alpha/2)
       = d*csc(alpha/2) + 0.5*l*cot(alpha/2)
    """
    if alpha < 0:
        raise ValueError("Angle alpha must be non-negative")

    if alpha > math.pi:
        raise ValueError("Angle alpha must be in radians and <= pi")

    s = math.sin(alpha / 2.0)
    if s <= 0:
        return math.inf

    c = math.cos(alpha / 2.0)
    return (d + 0.5 * l * c) / s


def create_screw_connector_normal(
    hint,
    screw_size,
    screw_length,
    screw_length_slack=0.1,
    tongue_slack=1.0,
    male_female_region_calculator=None,
    cover_thickness=None,
    shell_thickness=0.0,
    connector_gap=0.0,
):

    transforms = compute_transforms_from_hint(
        hint, male_female_region_calculator=male_female_region_calculator
    )
    nut_thickness = m_screws_table[screw_size]["nut_thickness"]

    dihedral_dot = np.dot(hint.triangle_a_normal, hint.triangle_b_normal)
    dihedral_dot = float(np.clip(dihedral_dot, -1.0, 1.0))
    dihedral_angle = math.acos(dihedral_dot)

    connector_thickness = m_screws_table[screw_size]["clearance_hole_normal"] * 2
    total_screw_length = (
        screw_length + m_screws_table[screw_size]["cylinder_head_height"]
    )
    total_connector_width = total_screw_length + 2 * nut_thickness
    connector_width = total_connector_width / 2
    dihedral_inset = math.tan(dihedral_angle / 2) * connector_thickness
    connector_length = connector_thickness * 2
    tongue_width = (
        connector_width * 2 + math.tan(dihedral_angle / 2) * connector_thickness
    )

    tongue_thickness = screw_length / 4

    if cover_thickness is None:
        cover_thickness = tongue_thickness * 2

    tongue_length = connector_length - 2 * tongue_thickness

    a_b_ray_angle = math.pi - dihedral_angle

    _logger.debug(f"a_b_normal_angle (degrees) = {math.degrees(a_b_ray_angle)}")

    d = tongue_thickness
    y0 = d
    if a_b_ray_angle < (math.pi) and a_b_ray_angle > 0:

        alpha = a_b_ray_angle
        l = tongue_width
        y0 = line_dist(
            alpha,
            l,
            d,
        )
        _logger.debug(
            f"Calculated line_dist y0 = {y0} from alpha={math.degrees(alpha)} degrees, l={l}, d={d}"
        )
    else:
        _logger.debug(
            f"Skipping line_dist calculation for angle {math.degrees(a_b_ray_angle)} degrees"
        )

    relevant_normal = normalize(transforms.female_normal + transforms.male_normal)
    edge_vec = np.asarray(hint.edge_vector, dtype=float)
    tongue_direction = normalize(transforms.female_out + -transforms.male_out)

    def calc_male_connector_vertices(length, width, thickness):
        bottom_length = length + 2 * thickness
        centroid = np.asarray(hint.edge_centroid, dtype=float)
        verts = [
            centroid
            - edge_vec * length / 2
            - relevant_normal * thickness
            - tongue_direction * (width - thickness),
            centroid
            + edge_vec * length / 2
            - relevant_normal * thickness
            - tongue_direction * (width - thickness),
            centroid + edge_vec * length / 2 - relevant_normal * thickness,
            centroid - edge_vec * length / 2 - relevant_normal * thickness,
            centroid - edge_vec * bottom_length / 2 + transforms.male_out * width,
            centroid + edge_vec * bottom_length / 2 + transforms.male_out * width,
            centroid + edge_vec * bottom_length / 2,
            centroid - edge_vec * bottom_length / 2,
        ]
        return verts

    male_connector_vertices = calc_male_connector_vertices(
        connector_length,
        connector_width + y0 / 2,
        y0 + tongue_thickness + cover_thickness,
    )

    male_connector = create_distorted_cube(male_connector_vertices)
    male_connector = translate(*(-connector_gap / 2 * tongue_direction))(male_connector)

    def calc_female_connector_vertices(length, width, thickness):
        bottom_length = length + 2 * thickness
        centroid = np.asarray(hint.edge_centroid, dtype=float)
        real_thickness = thickness
        real_width = width - connector_width + tongue_width / 2
        verts = [
            centroid
            - edge_vec * length / 2
            - relevant_normal * real_thickness
            + tongue_direction * (real_width),
            centroid
            + edge_vec * length / 2
            - relevant_normal * real_thickness
            + tongue_direction * (real_width),
            centroid + edge_vec * length / 2 - relevant_normal * real_thickness,
            centroid - edge_vec * length / 2 - relevant_normal * real_thickness,
            centroid
            - edge_vec * bottom_length / 2
            + transforms.female_out * (real_width + real_thickness),
            centroid
            + edge_vec * bottom_length / 2
            + transforms.female_out * (real_width + real_thickness),
            centroid + edge_vec * bottom_length / 2,
            centroid - edge_vec * bottom_length / 2,
        ]

        verts_reversed = []
        for i in range(len(verts)):
            verts_reversed.append(verts[len(verts) - 1 - i])

        verts = np.array(verts_reversed)

        return verts

    female_connector_vertices = calc_female_connector_vertices(
        connector_length,
        connector_width + y0 / 2,
        y0 + tongue_thickness + cover_thickness,
    )

    female_connector = create_distorted_cube(female_connector_vertices)

    female_connector = translate(*(connector_gap / 2 * tongue_direction))(
        female_connector
    )

    screw_direction = -relevant_normal

    screw_radius = m_screws_table[screw_size]["clearance_hole_close"] / 2
    total_length = total_screw_length + screw_length_slack + y0
    base_point = _to_tuple3(hint.edge_centroid)
    screw_hole = directed_cylinder_at(
        base_point, _to_tuple3(screw_direction), screw_radius, total_length
    )

    trans1 = translate(*(_to_tuple3(tongue_direction * (connector_width / 2))))
    trans2 = translate(*(_to_tuple3(tongue_direction * (dihedral_inset / 2))))
    trans3 = translate(*(_to_tuple3(-screw_direction * (screw_length_slack / 2))))
    screw_transform = chain_translations(trans1, trans2, trans3)

    screw_hole = screw_transform(screw_hole)
    screw_visualization = screw_transform(
        directed_cylinder_at(
            base_point, _to_tuple3(screw_direction), screw_radius, total_length
        )
    )

    _logger.debug("*" * 20)
    _logger.debug(
        f"connector_thickness={connector_thickness}, connector_width={connector_width}, dihedral_inset={dihedral_inset}, connector_length={connector_length}"
    )
    _logger.debug(
        f"tongue_width={tongue_width}, tongue_thickness={tongue_thickness}, tongue_length={tongue_length}, bottom_length={tongue_length + 2 * tongue_thickness}"
    )

    # calculate angle between the planes. If the normals are collinear, the angle must be 180 degrees
    # so it i is NOT just the acos of the dot product
    # and also not the dihedral angle - if the dihedral angle is 0, this angle must be 180 degrees
    # For line_dist, we need the opening angle of the wedge formed by the two surfaces
    # This is the supplementary angle to the dihedral angle: pi - dihedral_angle

    _logger.debug("+" * 20)

    def calc_tongue_vertices(length, width, thickness):
        bottom_length = length + 2 * thickness
        centroid = np.asarray(hint.edge_centroid, dtype=float)
        verts = [
            centroid - edge_vec * bottom_length / 2 - relevant_normal * thickness / 2,
            centroid + edge_vec * bottom_length / 2 - relevant_normal * thickness / 2,
            centroid
            + edge_vec * length / 2
            - relevant_normal * thickness / 2
            + tongue_direction * width / 2,
            centroid
            - edge_vec * length / 2
            - relevant_normal * thickness / 2
            + tongue_direction * width / 2,
            centroid - edge_vec * bottom_length / 2 + relevant_normal * thickness / 2,
            centroid + edge_vec * bottom_length / 2 + relevant_normal * thickness / 2,
            centroid
            + edge_vec * length / 2
            + tongue_direction * width / 2
            + relevant_normal * thickness / 2,
            centroid
            - edge_vec * length / 2
            + tongue_direction * width / 2
            + relevant_normal * thickness / 2,
        ]

        # Adjust back vertices (0, 1, 4, 5) for connector gap
        gap_offset = -connector_gap / 2 * tongue_direction
        verts[0] += gap_offset
        verts[1] += gap_offset
        verts[4] += gap_offset
        verts[5] += gap_offset

        verts = [v - (y0 + thickness / 2) * relevant_normal for v in verts]
        return verts

    tongue_vertices = calc_tongue_vertices(
        tongue_length, tongue_width, tongue_thickness
    )

    for v in tongue_vertices:
        _logger.debug(f"tongue vertex: {v}")
    tongue = create_distorted_cube(tongue_vertices).cut(screw_hole)

    tongue_cutter_vertices = calc_tongue_vertices(
        tongue_length + tongue_slack,
        tongue_width + tongue_slack,
        tongue_thickness + tongue_slack,
    )

    tongue_cutter = create_distorted_cube(tongue_cutter_vertices)

    male_connector = male_connector.fuse(tongue)
    female_connector = female_connector.cut(screw_hole)
    female_cutter = tongue_cutter.fuse(screw_hole)

    female_connector = female_connector.cut(tongue_cutter)

    male_connector = translate(*(-(shell_thickness - 0.01) * relevant_normal))(
        male_connector
    )
    female_connector = translate(*(-(shell_thickness - 0.01) * relevant_normal))(
        female_connector
    )

    return SimpleNamespace(
        male_region=transforms.male_region,
        female_region=transforms.female_region,
        male_connector=male_connector,
        tongue=tongue,
        male_cutter=None,
        female_connector=female_connector,
        female_cutter=female_cutter,
        additional_parts=None,
        non_production_parts=[screw_visualization],
    )
