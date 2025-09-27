import math
from types import SimpleNamespace

import numpy as np
import pytest
from shellforgepy.shells.materialized_connectors import (
    BIG_THING,
    compute_transforms_from_hint,
    create_connector_parts_from_hint,
    create_nut_holder_cutter,
    create_screw_connector_normal,
)

# Import CAD functions through the adapter system
from shellforgepy.simple import create_basic_cylinder


@pytest.fixture
def simple_hint():
    edge_start = np.array([-5.0, 0.0, 0.0])
    edge_end = np.array([5.0, 0.0, 0.0])
    edge_vector = edge_end - edge_start
    edge_centroid = (edge_start + edge_end) / 2

    tri_a = [edge_start, edge_end, np.array([0.0, 5.0, 5.0])]
    tri_b = [edge_start, edge_end, np.array([0.0, 0.0, 5.0])]

    return SimpleNamespace(
        region_a=1,
        region_b=0,
        triangle_a_normal=np.array([0.0, 0.0, 1.0]),
        triangle_b_normal=np.array([0.0, -1.0, 0.0]),
        triangle_a_vertices=tri_a,
        triangle_b_vertices=tri_b,
        edge_centroid=edge_centroid,
        edge_vector=edge_vector,
    )


def test_compute_transforms(simple_hint):
    transforms = compute_transforms_from_hint(simple_hint)

    assert transforms.male_region == 1
    assert transforms.female_region == 0
    assert np.allclose(
        transforms.male_normal,
        simple_hint.triangle_a_normal,
    )
    assert np.allclose(
        transforms.female_normal,
        simple_hint.triangle_b_normal,
    )


def test_create_connector_parts(simple_hint):
    male_region, female_region, male_connector, cutter = (
        create_connector_parts_from_hint(
            simple_hint,
            connector_length=10.0,
            connector_width=6.0,
            connector_thickness=2.0,
            connector_cyl_radius=2.0,
            connector_cylinder_length=8.0,
            connector_slack=0.5,
            connector_male_side_expansion=1.0,
        )
    )

    assert male_region == 1
    assert female_region == 0
    assert male_connector is not None
    assert cutter is not None
    # Check that objects have volume (works for both CadQuery and FreeCAD)
    assert hasattr(male_connector, "Volume")
    assert hasattr(cutter, "Volume")
    # Get volume (call as method for CadQuery, access as property for FreeCAD)
    male_vol = (
        male_connector.Volume()
        if callable(male_connector.Volume)
        else male_connector.Volume
    )
    cutter_vol = cutter.Volume() if callable(cutter.Volume) else cutter.Volume
    assert male_vol > 0
    assert cutter_vol > 0


def test_create_nut_holder_cutter(simple_hint):
    drill = create_basic_cylinder(2.0, 10.0)
    cutter = create_nut_holder_cutter("M3", slack=0.2, drill=drill)
    assert cutter is not None
    # Get volumes using CAD-agnostic approach
    cutter_vol = cutter.Volume() if callable(cutter.Volume) else cutter.Volume
    drill_vol = drill.Volume() if callable(drill.Volume) else drill.Volume
    assert cutter_vol > drill_vol
    # Get bounding box in a CAD-agnostic way
    if hasattr(cutter, "BoundingBox"):
        bb = (
            cutter.BoundingBox() if callable(cutter.BoundingBox) else cutter.BoundingBox
        )
        if hasattr(bb, "zmax"):
            assert bb.zmax - bb.zmin < BIG_THING
        elif hasattr(bb, "ZMax"):
            assert bb.ZMax - bb.ZMin < BIG_THING


def test_create_screw_connector_normal(simple_hint):
    result = create_screw_connector_normal(
        simple_hint,
        screw_size="M3",
        screw_length=12.0,
        screw_length_slack=0.2,
        tongue_slack=0.5,
    )

    assert result.male_region == 1
    assert result.female_region == 0
    # Get volumes using CAD-agnostic approach
    male_vol = (
        result.male_connector.Volume()
        if callable(result.male_connector.Volume)
        else result.male_connector.Volume
    )
    female_vol = (
        result.female_connector.Volume()
        if callable(result.female_connector.Volume)
        else result.female_connector.Volume
    )
    cutter_vol = (
        result.female_cutter.Volume()
        if callable(result.female_cutter.Volume)
        else result.female_cutter.Volume
    )
    assert male_vol > 0
    assert female_vol > 0
    assert cutter_vol > 0
    assert len(result.non_production_parts) == 1
    screw = result.non_production_parts[0]
    assert screw is not None
    screw_vol = screw.Volume() if callable(screw.Volume) else screw.Volume
    assert not math.isclose(screw_vol, 0.0)
