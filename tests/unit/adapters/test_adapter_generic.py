import os
from pathlib import Path

import numpy as np
import pytest
from shellforgepy.adapters._adapter import (
    apply_fillet_by_alignment,
    apply_fillet_to_edges,
    copy_part,
    create_box,
    create_cone,
    create_cylinder,
    create_extruded_polygon,
    create_filleted_box,
    create_solid_from_traditional_face_vertex_maps,
    create_sphere,
    create_text_object,
    cut_parts,
    export_solid_to_stl,
    fuse_parts,
    get_bounding_box,
    get_bounding_box_center,
    get_bounding_box_size,
    get_vertex_coordinates,
    get_vertex_coordinates_np,
    get_vertices,
    get_volume,
    mirror_part,
    rotate_part,
    translate_part,
)
from shellforgepy.adapters.font_resolver import resolve_font
from shellforgepy.construct.alignment import Alignment

# These tests are on a generic adapter level and do not depend on a specific CAD backend
# like FreeCAD or CadQuery.
# Importing backend-specific code is not allowed here.


def _collect_edges(solid):
    edges_attr = getattr(solid, "Edges", None)
    if callable(edges_attr):
        return list(edges_attr())
    if edges_attr is not None:
        return list(edges_attr)
    raise AttributeError("Solid does not expose edges in a known way")


def _bbox_size_tuple(obj):
    return tuple(float(value) for value in get_bounding_box_size(obj))


def _sorted_bbox_dimensions(obj):
    return tuple(sorted(_bbox_size_tuple(obj)))


def test_box_geometry_and_vertices():
    dimensions = (10.0, 20.0, 30.0)
    box = create_box(*dimensions)

    min_point, max_point = get_bounding_box(box)
    size = _bbox_size_tuple(box)
    assert np.allclose(size, dimensions, rtol=1e-6, atol=1e-6)

    center = get_bounding_box_center(box)
    expected_center = tuple(d / 2.0 for d in dimensions)
    assert np.allclose(center, expected_center, atol=1e-6)

    vertices = list(get_vertices(box))
    assert len(vertices) >= 8

    coord_list = get_vertex_coordinates(box)
    coord_array = get_vertex_coordinates_np(box)
    assert coord_array.shape[0] == len(coord_list)
    assert coord_array.shape[1] == 3

    xs = sorted({round(coord[0], 6) for coord in coord_list})
    ys = sorted({round(coord[1], 6) for coord in coord_list})
    zs = sorted({round(coord[2], 6) for coord in coord_list})

    assert min_point[0] == pytest.approx(xs[0], abs=1e-6)
    assert max_point[0] == pytest.approx(xs[-1], abs=1e-6)
    assert min_point[1] == pytest.approx(ys[0], abs=1e-6)
    assert max_point[1] == pytest.approx(ys[-1], abs=1e-6)
    assert min_point[2] == pytest.approx(zs[0], abs=1e-6)
    assert max_point[2] == pytest.approx(zs[-1], abs=1e-6)


def test_create_solid_from_traditional_face_vertex_maps():

    cube_maps = {
        "faces": {
            "0": [0, 1, 2, 3],
            "1": [7, 6, 5, 4],
            "2": [4, 5, 1, 0],
            "3": [5, 6, 2, 1],
            "4": [6, 7, 3, 2],
            "5": [7, 4, 0, 3],
        },
        "vertexes": {
            "0": [0.0, 0.0, 0.0],
            "1": [1.0, 0.0, 0.0],
            "2": [1.0, 1.0, 0.0],
            "3": [0.0, 1.0, 0.0],
            "4": [0.0, 0.0, 1.0],
            "5": [1.0, 0.0, 1.0],
            "6": [1.0, 1.0, 1.0],
            "7": [0.0, 1.0, 1.0],
        },
    }

    solid = create_solid_from_traditional_face_vertex_maps(cube_maps)
    size = _bbox_size_tuple(solid)

    assert np.allclose(size, (1.0, 1.0, 1.0), atol=1e-6)
    assert get_volume(solid) > 0.0


def test_text_object_padding():
    text = create_text_object("Test", size=4.0, thickness=1.0, padding=1.5)
    min_point, _ = get_bounding_box(text)
    assert min_point[0] == pytest.approx(1.5, abs=1e-3)
    assert min_point[1] == pytest.approx(1.5, abs=1e-3)
    assert min_point[2] >= -1e-6
    assert get_volume(text) > 0.0


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Font availability varies on CI"
)
def test_text_object_bounding_box_dimensions():
    font_spec = resolve_font(font="DejaVu Sans")

    text_value = "Forge"
    glyph_height = 8.0
    extrusion = 2.5

    text = create_text_object(
        text_value,
        size=glyph_height,
        thickness=extrusion,
        font=font_spec.path,
        font_path=font_spec.path,
    )

    width, height, thickness = _bbox_size_tuple(text)

    assert thickness == pytest.approx(extrusion, abs=1e-6)

    assert glyph_height * 0.99 <= height <= glyph_height * 1.01
    assert width >= glyph_height * 1.2
    aspect_ratio = width / max(height, 1e-6)
    assert 1.0 <= aspect_ratio <= 8.0


def test_cylinder_cone_and_sphere_creation():
    cylinder = create_cylinder(2.0, 5.0, angle=180.0)
    cyl_size = _bbox_size_tuple(cylinder)
    assert cyl_size[2] == pytest.approx(5.0, abs=1e-6)
    assert cyl_size[0] > 0 and cyl_size[1] > 0

    cone = create_cone(3.0, 1.0, 4.0)
    cone_size = _bbox_size_tuple(cone)
    assert cone_size[2] == pytest.approx(4.0, abs=1e-6)

    sphere = create_sphere(2.5)
    sphere_size = _bbox_size_tuple(sphere)
    assert np.allclose(sphere_size, (5.0, 5.0, 5.0), atol=1e-6)


def test_polygon_copy_and_translations():
    polygon = create_extruded_polygon([(0, 0), (2, 0), (0, 1)], 3.0)
    polygon_copy = copy_part(polygon)

    base_center = np.array(get_bounding_box_center(polygon_copy))
    offset = (1.0, -0.5, 2.0)

    translated = translate_part(polygon_copy, offset)
    translated_center = np.array(get_bounding_box_center(translated))
    assert np.allclose(translated_center - base_center, offset, atol=1e-6)


def test_rotations_and_mirroring():
    box_for_rotate = create_box(10.0, 20.0, 30.0)
    center = tuple(get_bounding_box_center(box_for_rotate))
    original_dims = _sorted_bbox_dimensions(box_for_rotate)

    rotated = rotate_part(box_for_rotate, 90.0, center=center, axis=(0.0, 0.0, 1.0))
    rotated_dims = _sorted_bbox_dimensions(rotated)
    assert np.allclose(rotated_dims, original_dims, atol=1e-6)
    rotated_center = get_bounding_box_center(rotated)
    assert np.allclose(rotated_center, center, atol=1e-6)

    mirrored = mirror_part(
        create_box(4.0, 5.0, 6.0), normal=(1.0, 0.0, 0.0), point=(0.0, 0.0, 0.0)
    )
    min_point, max_point = get_bounding_box(mirrored)
    assert max_point[0] <= 1e-6
    assert np.isclose(max_point[1] - min_point[1], 5.0, atol=1e-6)


def test_boolean_operations_change_volume():
    base = create_box(10.0, 10.0, 10.0)
    cutter = translate_part(create_box(6.0, 6.0, 12.0), (2.0, 2.0, -1.0))

    fused = fuse_parts(base, cutter)
    cut = cut_parts(base, cutter)

    assert get_volume(fused) > get_volume(base)
    assert get_volume(cut) < get_volume(base)


def test_filleting_helpers_reduce_volume():
    base_box = create_box(10.0, 10.0, 10.0)
    filleted_box = create_filleted_box(10.0, 10.0, 10.0, 1.0)
    assert get_volume(filleted_box) < get_volume(base_box)

    aligned_filleted = apply_fillet_by_alignment(
        create_box(10.0, 10.0, 10.0),
        0.5,
        fillets_at=[Alignment.TOP],
    )
    assert get_volume(aligned_filleted) < get_volume(base_box)

    box_for_edge_fillet = create_box(10.0, 10.0, 10.0)
    edges = _collect_edges(box_for_edge_fillet)
    edge_filleted = apply_fillet_to_edges(box_for_edge_fillet, 0.5, edges)
    assert get_volume(edge_filleted) < get_volume(base_box)


def test_export_solid_to_stl(tmp_path: Path):
    destination = tmp_path / "box.stl"
    solid = create_box(5.0, 5.0, 5.0)
    export_solid_to_stl(solid, str(destination))

    assert destination.exists()
    assert destination.stat().st_size > 0
