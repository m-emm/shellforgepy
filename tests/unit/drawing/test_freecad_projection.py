"""FreeCAD-only Stage 7 projection contract tests.

Run through ``./freecad_python.sh -m pytest``.  This avoids the generic
``shellforgepy.simple`` test fixture, which imports CadQuery-only STEP helpers
in the embedded Python environment.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

Part = pytest.importorskip("Part")
Base = pytest.importorskip("FreeCAD").Base

from shellforgepy.drawing import (
    make_construction_drawing_request,
    render_construction_drawing,
)

SVG_NS = "http://www.w3.org/2000/svg"


def _plate_with_front_bore_and_top_taps():
    solid = Part.makeBox(80.0, 50.0, 8.0)
    front_bore = Part.makeCylinder(
        1.25, 6.0, Base.Vector(40.0, -1.0, 4.0), Base.Vector(0.0, 1.0, 0.0)
    )
    left_tap = Part.makeCylinder(1.25, 10.0, Base.Vector(20.0, 15.0, -1.0))
    right_tap = Part.makeCylinder(1.25, 10.0, Base.Vector(60.0, 15.0, -1.0))
    return solid.cut(front_bore.fuse(left_tap).fuse(right_tap))


def _hidden_lines(path: Path):
    root = ET.fromstring(path.read_bytes())
    return root.findall(
        ".//svg:line[@data-shellforgepy-visibility='hidden']", {"svg": SVG_NS}
    )


def test_freecad_projection_shows_edge_on_bore_and_tapped_hole_profiles(tmp_path):
    solid = _plate_with_front_bore_and_top_taps()
    top_request = make_construction_drawing_request(
        name="freecad_top",
        parts=[{"source": "self", "artifact": "leader"}],
        view="top",
        representation={
            "mode": "projection",
            "include": ["visible_outline", "hidden_feature_edges"],
        },
    )
    front_request = make_construction_drawing_request(
        name="freecad_front",
        parts=[{"source": "self", "artifact": "leader"}],
        view="front",
        representation={
            "mode": "projection",
            "include": ["visible_outline", "hidden_feature_edges"],
        },
    )
    top_path = render_construction_drawing(
        solid, top_request, tmp_path / "top.svg", part_identity="plate"
    )
    front_path = render_construction_drawing(
        solid, front_request, tmp_path / "front.svg", part_identity="plate"
    )

    top_side_lines = [
        line
        for line in _hidden_lines(top_path)
        if abs(float(line.attrib["x1"]) - float(line.attrib["x2"])) < 1e-9
        and abs(abs(float(line.attrib["y1"]) - float(line.attrib["y2"])) - 5.0) < 1e-9
    ]
    assert len(top_side_lines) == 2

    front_verticals = [
        line
        for line in _hidden_lines(front_path)
        if abs(float(line.attrib["x1"]) - float(line.attrib["x2"])) < 1e-9
        and abs(abs(float(line.attrib["y1"]) - float(line.attrib["y2"])) - 8.0) < 1e-9
    ]
    assert {round(float(line.attrib["x1"]), 2) for line in front_verticals} == {
        -21.25,
        -18.75,
        18.75,
        21.25,
    }
