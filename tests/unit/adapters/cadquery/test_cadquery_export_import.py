import os
import tempfile

import pytest

pytest.importorskip("cadquery")

from shellforgepy.adapters.cadquery import cadquery_adapter


def _export_step(tmpdir, part):
    step_path = os.path.join(tmpdir, "test_part.step")
    cadquery_adapter.export_solid_to_step(part, step_path)
    return step_path


def test_export_step():
    part = cadquery_adapter.create_box(10, 10, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = _export_step(tmpdir, part)
        assert os.path.isfile(step_path)
        assert os.path.getsize(step_path) > 0


def test_import_step_round_trip_volume():
    part = cadquery_adapter.create_box(12, 7, 3)
    expected_volume = 12 * 7 * 3

    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = _export_step(tmpdir, part)
        imported = cadquery_adapter.import_solid_from_step(step_path)
        solid = cadquery_adapter.normalize_to_solid(imported)
        assert solid.Volume() == pytest.approx(expected_volume, rel=1e-4)


def test_export_structured_step_round_trip_solids():
    box_a = cadquery_adapter.create_box(10, 10, 10)
    box_b = cadquery_adapter.create_box(5, 8, 6)
    box_b = cadquery_adapter.translate_part(box_b, (25, 0, 0))

    structure = {
        "group_a": [("box_a", box_a)],
        "group_b": [("box_b", box_b)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = os.path.join(tmpdir, "assembly.step")
        cadquery_adapter.export_structured_step(structure, step_path)
        assert os.path.isfile(step_path)

        imported = cadquery_adapter.import_solid_from_step(step_path)
        solids = cadquery_adapter.extract_solids(imported)
        assert len(solids) == 2


def test_deserialize_structured_step_groups_and_volumes():
    box_a = cadquery_adapter.create_box(10, 10, 10)
    box_b = cadquery_adapter.create_box(5, 8, 6)

    structure = {
        "group_a": [("box_a", box_a)],
        "group_b": [("box_b", box_b)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = os.path.join(tmpdir, "assembly.step")
        cadquery_adapter.export_structured_step(structure, step_path)

        loaded = cadquery_adapter.deserialize_structured_step(step_path)
        if "ROOT" in loaded:
            # CadQuery doesn't preserve STEP assembly structure on import.
            # We now extract individual solids from the imported compound.
            # With 2 input solids, we expect 2 entries in ROOT.
            assert len(loaded["ROOT"]) == 2
            volumes = [
                cadquery_adapter.normalize_to_solid(part).Volume()
                for _, part in loaded["ROOT"]
            ]
            volumes.sort()
            assert volumes[0] == pytest.approx(5 * 8 * 6, rel=1e-4)
            assert volumes[1] == pytest.approx(10 * 10 * 10, rel=1e-4)
        else:
            assert set(loaded.keys()) == {"group_a", "group_b"}
            assert len(loaded["group_a"]) == 1
            assert len(loaded["group_b"]) == 1

            volume_a = cadquery_adapter.normalize_to_solid(
                loaded["group_a"][0][1]
            ).Volume()
            volume_b = cadquery_adapter.normalize_to_solid(
                loaded["group_b"][0][1]
            ).Volume()
            assert volume_a == pytest.approx(10 * 10 * 10, rel=1e-4)
            assert volume_b == pytest.approx(5 * 8 * 6, rel=1e-4)
