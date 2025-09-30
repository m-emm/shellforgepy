"""CAD-agnostic connector generation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence, Tuple, Union

import numpy as np
from shellforgepy.adapters._adapter import (
    create_basic_box,
    create_basic_cylinder,
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
    def orthonormalize(u, v):
        u = u / np.linalg.norm(u)
        v_orth = v - np.dot(v, u) * u
        v_norm = np.linalg.norm(v_orth)
        if v_norm < 1e-8:
            raise ValueError("Provided vectors are collinear")
        v_orth /= v_norm
        w = np.cross(u, v_orth)
        return np.column_stack((u, v_orth, w))

    origin_a = np.asarray(origin_a, dtype=float)
    origin_b = np.asarray(origin_b, dtype=float)
    up_a = np.asarray(up_a, dtype=float)
    out_a = np.asarray(out_a, dtype=float)
    up_b = np.asarray(up_b, dtype=float)
    out_b = np.asarray(out_b, dtype=float)

    R_a = orthonormalize(up_a, out_a)
    R_b = orthonormalize(up_b, out_b)

    R = R_b @ R_a.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = math.acos(np.clip((trace - 1) / 2.0, -1.0, 1.0))

    if np.isclose(angle, 0.0):
        axis = np.array([1.0, 0.0, 0.0])
    elif np.isclose(angle, math.pi):
        eigvals, eigvecs = np.linalg.eigh(R)
        mask = np.isclose(eigvals, 1.0, atol=1e-5)
        if not np.any(mask):
            raise ValueError("Unable to derive rotation axis")
        axis = eigvecs[:, mask][:, 0]
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2.0 * math.sin(angle)
        )

    rotated_origin = R @ (-origin_a)
    translation = origin_b + rotated_origin
    return CoordinateTransform(tuple(axis), angle, tuple(translation))


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
    return create_basic_box(length, width, height, origin=_to_tuple3(base_point))


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


m_screws_table = {
    "M3": {
        "nut_size": 5.5,
        "nut_thickness": 2.3,
        "clearance_hole_close": 3.2,
        "clearance_hole_normal": 3.4,
        "cylinder_head_height": 3.0,
    },
    "M4": {
        "nut_size": 7.0,
        "nut_thickness": 3.0,
        "clearance_hole_close": 4.3,
        "clearance_hole_normal": 4.5,
        "cylinder_head_height": 4.0,
    },
    "M5": {
        "nut_size": 8.0,
        "nut_thickness": 4.6,
        "clearance_hole_close": 5.3,
        "clearance_hole_normal": 5.5,
        "cylinder_head_height": 5.0,
    },
}


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
    hole = create_basic_cylinder(clearance, height)
    return hex_prism.cut(hole)


BIG_THING = 200


def compute_transforms_from_hint(
    hint,
    male_female_region_calculator=None,
):
    region_a, region_b = hint.region_a, hint.region_b
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


def create_screw_connector_normal(
    hint,
    screw_size,
    screw_length,
    screw_length_slack=0.1,
    tongue_slack=1.0,
    male_female_region_calculator=None,
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

    def calc_corners(length, width, thickness, inset):
        return [
            (-length / 2 - thickness, 0, 0),
            (length / 2 + thickness, 0, 0),
            (length / 2 + thickness, width + thickness, 0),
            (-length / 2 - thickness, width + thickness, 0),
            (-length / 2, inset, thickness),
            (length / 2, inset, thickness),
            (length / 2, width, thickness),
            (-length / 2, width, thickness),
        ]

    corners = calc_corners(
        connector_length, connector_width, connector_thickness, dihedral_inset
    )
    male_corners = calc_corners(
        connector_length, connector_width, connector_thickness / 2, dihedral_inset
    )

    male_slab = create_distorted_cube(male_corners)
    male_slab = rotate(180, axis=(0, 1, 0))(male_slab)

    female_slab = create_distorted_cube(corners)
    female_slab = rotate(180, axis=(0, 1, 0))(female_slab)

    male_connector = transforms.apply_tf(male_slab, transforms.tf_male)
    female_connector = transforms.apply_tf(female_slab, transforms.tf_female)

    screw_direction = -np.asarray(transforms.female_normal)
    tongue_direction = transforms.female_out

    screw_radius = m_screws_table[screw_size]["clearance_hole_close"] / 2
    total_length = total_screw_length + screw_length_slack
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

    tongue_width = (
        connector_width * 2 + math.tan(dihedral_angle / 2) * connector_thickness
    )
    tongue_thickness = screw_length / 4
    tongue_length = connector_length - 2 * tongue_thickness
    edge_vec = np.asarray(hint.edge_vector, dtype=float)

    def calc_tongue_vertices(length, width, thickness):
        bottom_length = length + 2 * thickness
        centroid = np.asarray(hint.edge_centroid, dtype=float)
        verts = [
            centroid
            - edge_vec * bottom_length / 2
            - transforms.female_normal * thickness,
            centroid
            + edge_vec * bottom_length / 2
            - transforms.female_normal * thickness,
            centroid
            + edge_vec * length / 2
            - transforms.female_normal * thickness
            + tongue_direction * width / 2,
            centroid
            - edge_vec * length / 2
            - transforms.female_normal * thickness
            + tongue_direction * width / 2,
            centroid - edge_vec * bottom_length / 2,
            centroid + edge_vec * bottom_length / 2,
            centroid + edge_vec * length / 2 + tongue_direction * width / 2,
            centroid - edge_vec * length / 2 + tongue_direction * width / 2,
        ]
        return verts

    tongue_vertices = calc_tongue_vertices(
        tongue_length, tongue_width, tongue_thickness
    )
    tongue = create_distorted_cube(tongue_vertices).cut(screw_hole)

    tongue_cutter_vertices = calc_tongue_vertices(
        tongue_length + tongue_slack,
        tongue_width + tongue_slack,
        tongue_thickness + tongue_slack,
    )
    tongue_cutter = create_distorted_cube(tongue_cutter_vertices)

    male_connector = male_connector.fuse(tongue)
    female_cutter = tongue_cutter.fuse(screw_hole)
    female_connector = female_connector.cut(female_cutter)

    return SimpleNamespace(
        male_region=transforms.male_region,
        female_region=transforms.female_region,
        male_connector=male_connector,
        male_cutter=None,
        female_connector=female_connector,
        female_cutter=female_cutter,
        additional_parts=None,
        non_production_parts=[screw_visualization],
    )
