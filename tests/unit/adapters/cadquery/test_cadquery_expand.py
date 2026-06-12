import pytest

pytest.importorskip("cadquery")

from shellforgepy.adapters.cadquery import cadquery_adapter
from shellforgepy.geometry.higher_order_solids import materialize_bounding_box
from shellforgepy.geometry.modifications import expand


def _bbox_size_tuple(part):
    return tuple(float(value) for value in cadquery_adapter.get_bounding_box_size(part))


def test_expand_part_grows_box_by_distance_on_each_side():
    box = cadquery_adapter.create_box(10.0, 20.0, 30.0)

    expanded = cadquery_adapter.expand_part(box, 1.0)

    assert _bbox_size_tuple(expanded) == pytest.approx((12.0, 22.0, 32.0), abs=1e-6)
    assert cadquery_adapter.get_volume(expanded) > cadquery_adapter.get_volume(box)


def test_expand_materialized_bounding_box_grows_by_distance_on_each_side():
    box = cadquery_adapter.create_box(40.2, 40.2, 10.5)
    bbox = materialize_bounding_box(box)
    distance = 0.04

    expanded = expand(bbox, distance)

    expected_size = tuple(size + 2 * distance for size in _bbox_size_tuple(bbox))
    assert _bbox_size_tuple(expanded) == pytest.approx(expected_size, abs=1e-6)
    assert cadquery_adapter.get_volume(expanded) > cadquery_adapter.get_volume(bbox)


def test_expand_modification_api_preserves_zero_distance_geometry():
    box = cadquery_adapter.create_box(10.0, 20.0, 30.0)

    expanded = expand(box, 0.0)

    assert _bbox_size_tuple(expanded) == pytest.approx(_bbox_size_tuple(box), abs=1e-6)
    assert cadquery_adapter.get_volume(expanded) == pytest.approx(
        cadquery_adapter.get_volume(box), rel=1e-6
    )


def test_expand_rejects_negative_distance():
    box = cadquery_adapter.create_box(10.0, 20.0, 30.0)

    with pytest.raises(ValueError, match="distance must be non-negative"):
        expand(box, -0.1)
