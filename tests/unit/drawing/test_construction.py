import math
import xml.etree.ElementTree as ET

import pytest
from shellforgepy.construct.alignment import Alignment
from shellforgepy.drawing import SVG_NS, append_arc, append_part_group
from shellforgepy.drawing import construction as construction_module
from shellforgepy.drawing import (
    create_svg_document,
    drawing_bounds_from_model_bounds,
    make_construction_drawing_request,
    model_point_to_drawing,
    render_construction_drawing,
    render_construction_drawing_parts,
    resolve_view_frame,
    serialize_svg,
)
from shellforgepy.drawing.layout import (
    Bounds2D,
    align_bounds_sequence_2d,
    alignment_delta_2d,
)
from shellforgepy.simple import (
    LeaderFollowersCuttersPart,
    PartCollector,
    create_box,
    create_cylinder,
)

MODEL_BOUNDS = ((0.0, 0.0, 0.0), (60.0, 20.0, 5.0))


def _bounds_attribute(element, name):
    return tuple(float(value) for value in element.attrib[name].split(","))


def _make_plate_for_drawing_test():
    plate = create_box(60.0, 20.0, 5.0)
    cutters = PartCollector()
    for x in (15.0, 45.0):
        cutters = cutters.fuse(create_cylinder(1.7, 7.0, origin=(x, 10.0, -1.0)))
    return plate.cut(cutters)


def _make_annotation_plate_for_drawing_test():
    cutter_left = create_cylinder(1.7, 7.0, origin=(15.0, 10.0, -1.0))
    cutter_right = create_cylinder(1.7, 7.0, origin=(45.0, 10.0, -1.0))
    leader = create_box(60.0, 20.0, 5.0).cut(cutter_left.fuse(cutter_right))
    return LeaderFollowersCuttersPart(
        leader=leader,
        cutters=[cutter_left, cutter_right],
        cutter_names=["hole_left", "hole_right"],
        additional_data={"part_ref_origin": {"assembly_name": "drawing_test"}},
    )


def test_default_top_frame_is_centered_and_right_handed():
    request = make_construction_drawing_request(
        name="plate_top",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
    )

    frame = resolve_view_frame(request, MODEL_BOUNDS)

    assert frame == {
        "origin": (30.0, 10.0, 2.5),
        "normal": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "right": (1.0, 0.0, 0.0),
    }
    assert model_point_to_drawing((0.0, 0.0, 2.5), frame) == (-30.0, -10.0)
    assert model_point_to_drawing((60.0, 20.0, 2.5), frame) == (30.0, 10.0)


def test_explicit_view_normalizes_vectors_and_uses_explicit_origin():
    request = make_construction_drawing_request(
        name="custom",
        parts=[{"source": "self", "artifact": "leader"}],
        view={"normal": (0, 0, 2), "up": (0, 3, 0), "origin": (1, 2, 3)},
    )

    assert resolve_view_frame(request, MODEL_BOUNDS) == {
        "origin": (1.0, 2.0, 3.0),
        "normal": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "right": (1.0, 0.0, 0.0),
    }


@pytest.mark.parametrize(
    "view, message",
    [
        ({"normal": (0, 0, 1), "up": (0, 0, 1)}, "orthogonal"),
        ({"normal": (0, 0, 0), "up": (0, 1, 0)}, "zero length"),
    ],
)
def test_invalid_view_frame_is_rejected(view, message):
    request = make_construction_drawing_request(
        name="invalid",
        parts=[{"source": "self", "artifact": "leader"}],
        view=view,
    )

    with pytest.raises(ValueError, match=message):
        resolve_view_frame(request, MODEL_BOUNDS)


def test_request_keeps_existing_selector_shape_and_has_no_annotation_api():
    request = make_construction_drawing_request(
        name="plate_top",
        parts=[
            {
                "source": "dependencies",
                "assembly": "plate",
                "artifact": "leader",
                "name": "plate",
            }
        ],
    )

    assert request["parts"] == [
        {
            "source": "dependencies",
            "assembly": "plate",
            "artifact": "leader",
            "name": "plate",
        }
    ]
    assert "annotations" not in request


def test_svg_document_has_canonical_groups_metadata_and_stable_xml():
    request = make_construction_drawing_request(
        name="plate<&",
        parts=[{"source": "self", "artifact": "leader", "name": "plate<&"}],
        metadata={"note": "A & B"},
    )
    frame = resolve_view_frame(request, MODEL_BOUNDS)
    bounds = drawing_bounds_from_model_bounds(MODEL_BOUNDS, frame)
    tree, geometry = create_svg_document(
        request,
        frame,
        bounds,
        adapter_id="cadquery",
        source_assembly="plate",
    )
    part = append_part_group(
        geometry,
        part_identity="plate<&",
        source="plate.leader",
    )

    serialized = serialize_svg(tree)
    reloaded = ET.fromstring(serialized)
    assert reloaded.tag == f"{{{SVG_NS}}}svg"
    assert reloaded.attrib["viewBox"] == "-30 -10 60 20"
    assert reloaded.attrib["data-shellforgepy-units"] == "mm"
    assert reloaded.attrib["data-shellforgepy-section-normal"] == "0,0,1"
    assert reloaded.attrib["data-shellforgepy-adapter"] == "cadquery"
    assert reloaded.attrib["data-shellforgepy-metadata-note"] == "A & B"
    assert geometry.attrib["data-shellforgepy-role"] == "geometry"
    assert geometry.attrib["transform"] == "translate(0 0) scale(1 -1)"
    assert part.attrib["data-shellforgepy-role"] == "section-contour"
    assert part.attrib["data-shellforgepy-part"] == "plate<&"
    assert serialize_svg(tree) == serialized


def test_append_arc_uses_exact_svg_arc_command():
    parent = ET.Element("g")

    arc = append_arc(
        parent,
        cx=0,
        cy=0,
        radius=5,
        start_x=5,
        start_y=0,
        end_x=0,
        end_y=5,
        large_arc=False,
        sweep=True,
        source_edge="section-edge-0",
    )

    assert arc.tag == f"{{{SVG_NS}}}path"
    assert arc.attrib["d"] == "M 5 0 A 5 5 0 0 1 0 5"
    assert arc.attrib["data-shellforgepy-geometry"] == "exact"
    assert arc.attrib["data-shellforgepy-source-edge"] == "section-edge-0"


def test_technical_sheet_adds_border_metadata_frames_and_text():
    request = make_construction_drawing_request(
        name="plate_sheet",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
        sheet={
            "format": "A4",
            "orientation": "landscape",
            "margin": 10,
            "border": True,
            "title_block": {
                "title": "Test Plate",
                "drawing_number": "PLATE-001",
                "revision": "A",
                "material": "Aluminum",
                "scale": "1:1",
                "source": "test assembly",
            },
        },
    )
    frame = resolve_view_frame(request, MODEL_BOUNDS)
    tree, geometry = create_svg_document(
        request,
        frame,
        drawing_bounds_from_model_bounds(MODEL_BOUNDS, frame),
        adapter_id="cadquery",
        source_assembly="test_assembly",
    )

    root = tree.getroot()
    roles = [
        element.attrib.get("data-shellforgepy-role")
        for element in root.iter()
        if element.attrib.get("data-shellforgepy-role")
    ]
    text_values = [element.text for element in root.findall(f".//{{{SVG_NS}}}text")]
    assert root.attrib["viewBox"] == "0 0 297 210"
    assert geometry.attrib["transform"].startswith("translate(")
    assert "outer-border" in roles
    assert "inner-border" in roles
    assert "drawing-viewport" in roles
    assert "metadata-frame" in roles
    assert "title-block-frame" in roles
    assert "Test Plate" in text_values
    assert "DRAWING: PLATE-001" in text_values
    assert "REVISION: A" in text_values


def test_stage1_render_emits_four_lines_and_two_analytic_circles(tmp_path):
    request = make_construction_drawing_request(
        name="plate_top",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
    )
    destination = tmp_path / "plate_top.svg"

    render_construction_drawing(
        _make_plate_for_drawing_test(),
        request,
        destination,
        part_identity="plate",
        source="plate.leader",
    )
    first_output = destination.read_bytes()
    render_construction_drawing(
        _make_plate_for_drawing_test(),
        request,
        tmp_path / "plate_top_again.svg",
        part_identity="plate",
        source="plate.leader",
    )
    assert first_output == (tmp_path / "plate_top_again.svg").read_bytes()

    root = ET.fromstring(first_output)
    lines = root.findall(f".//{{{SVG_NS}}}line")
    circles = root.findall(f".//{{{SVG_NS}}}circle")
    assert len(lines) == 4
    assert len(circles) == 2
    assert not root.findall(f".//{{{SVG_NS}}}path")
    assert not root.findall(f".//{{{SVG_NS}}}polyline")
    assert not root.findall(f".//{{{SVG_NS}}}polygon")
    assert root.attrib["data-shellforgepy-adapter"] in {"cadquery", "freecad"}
    assert all(
        element.attrib["data-shellforgepy-geometry"] == "exact"
        for element in [*lines, *circles]
    )
    assert sorted(
        (
            float(circle.attrib["cx"]),
            float(circle.attrib["cy"]),
            float(circle.attrib["r"]),
        )
        for circle in circles
    ) == [(-15.0, 0.0, 1.7), (15.0, 0.0, 1.7)]
    assert sorted(
        (
            float(line.attrib["x1"]),
            float(line.attrib["y1"]),
            float(line.attrib["x2"]),
            float(line.attrib["y2"]),
        )
        for line in lines
    ) == [
        (-30.0, -10.0, -30.0, 10.0),
        (-30.0, -10.0, 30.0, -10.0),
        (-30.0, 10.0, 30.0, 10.0),
        (30.0, -10.0, 30.0, 10.0),
    ]


def test_stage5_dimensions_use_canonical_named_targets_and_alignment(tmp_path):
    assembly = _make_annotation_plate_for_drawing_test()
    leader_ref = assembly.part_ref_for_leader()
    cutter_ref = assembly.part_ref_for_named_cutter("hole_left")
    request = make_construction_drawing_request(
        name="plate_dimensions",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
        annotations=[
            {
                "id": "plate_width",
                "operation": "bounding_box_x_dimension",
                "target": leader_ref,
                "placement": {
                    "alignments": [{"alignment": "STACK_FRONT", "stack_gap": 8.0}]
                },
            },
            {
                "id": "plate_height",
                "operation": "bounding_box_y_dimension",
                "target": leader_ref,
                "placement": {
                    "alignments": [{"alignment": "STACK_RIGHT", "stack_gap": 8.0}]
                },
            },
            {
                "id": "hole_left_diameter",
                "operation": "circle_diameter",
                "target": cutter_ref,
                "placement": {
                    "alignments": [{"alignment": "STACK_BACK", "stack_gap": 5.0}]
                },
            },
        ],
    )
    records = []
    destination = tmp_path / "plate_dimensions.svg"

    render_construction_drawing_parts(
        [{"part": assembly.leader, "name": "plate", "source": leader_ref}],
        request,
        destination,
        annotation_targets={
            cutter_ref: {
                "part": assembly.get_named_cutter("hole_left"),
                "source": cutter_ref,
            }
        },
        annotation_records=records,
    )

    root = ET.fromstring(destination.read_bytes())
    dimensions = root.findall(
        ".//svg:g[@data-shellforgepy-role='dimension']", {"svg": SVG_NS}
    )
    dimension_by_id = {
        element.attrib["data-shellforgepy-annotation-id"]: element
        for element in dimensions
    }
    assert set(dimension_by_id) == {
        "plate_width",
        "plate_height",
        "hole_left_diameter",
    }
    assert (
        dimension_by_id["plate_width"].attrib["data-shellforgepy-target"] == leader_ref
    )
    assert (
        dimension_by_id["hole_left_diameter"].attrib["data-shellforgepy-target"]
        == cutter_ref
    )
    assert dimension_by_id["plate_width"].attrib["data-shellforgepy-value"] == "60.00"
    assert dimension_by_id["plate_height"].attrib["data-shellforgepy-value"] == "20.00"
    assert (
        dimension_by_id["hole_left_diameter"].attrib["data-shellforgepy-value"]
        == "3.40"
    )
    text_values = [element.text for element in root.findall(f".//{{{SVG_NS}}}text")]
    assert "60.00" in text_values
    assert "20.00" in text_values
    assert "⌀3.40" in text_values
    assert len(records) == 3
    assert records[2]["target"] == cutter_ref
    assert records[2]["operation"] == "circle_diameter"
    assert records[2]["layout_bounds"] == [-30.0, -10.0, 30.0, 10.0]
    width_extensions = dimension_by_id["plate_width"].findall(
        ".//svg:line[@data-shellforgepy-role='dimension-extension']",
        {"svg": SVG_NS},
    )
    height_extensions = dimension_by_id["plate_height"].findall(
        ".//svg:line[@data-shellforgepy-role='dimension-extension']",
        {"svg": SVG_NS},
    )
    assert len(width_extensions) == len(height_extensions) == 2
    assert all(
        extension.attrib["stroke-width"] == "0.12" for extension in width_extensions
    )
    assert all(float(extension.attrib["y1"]) < -10.0 for extension in width_extensions)
    assert all(float(extension.attrib["x1"]) > 30.0 for extension in height_extensions)
    width_dimension_line = dimension_by_id["plate_width"].find(
        ".//svg:line[@marker-start]", {"svg": SVG_NS}
    )
    height_dimension_line = dimension_by_id["plate_height"].find(
        ".//svg:line[@marker-start]", {"svg": SVG_NS}
    )
    assert width_dimension_line is not None
    assert height_dimension_line is not None
    assert all(
        float(extension.attrib["y2"]) < float(width_dimension_line.attrib["y1"])
        for extension in width_extensions
    )
    assert all(
        float(extension.attrib["x2"]) > float(height_dimension_line.attrib["x1"])
        for extension in height_extensions
    )
    circle_leader = dimension_by_id["hole_left_diameter"].find(
        ".//svg:line[@marker-end]", {"svg": SVG_NS}
    )
    assert circle_leader is not None
    circle_target_bounds = _bounds_attribute(
        dimension_by_id["hole_left_diameter"], "data-shellforgepy-target-bounds"
    )
    circle_center_x = (circle_target_bounds[0] + circle_target_bounds[2]) / 2.0
    circle_center_y = (circle_target_bounds[1] + circle_target_bounds[3]) / 2.0
    circle_radius = (circle_target_bounds[2] - circle_target_bounds[0]) / 2.0
    assert math.hypot(
        float(circle_leader.attrib["x2"]) - circle_center_x,
        float(circle_leader.attrib["y2"]) - circle_center_y,
    ) == pytest.approx(circle_radius + construction_module._CIRCLE_ARROW_CLEARANCE)
    circle_layout_bounds = _bounds_attribute(
        dimension_by_id["hole_left_diameter"], "data-shellforgepy-layout-bounds"
    )
    circle_placed_bounds = _bounds_attribute(
        dimension_by_id["hole_left_diameter"], "data-shellforgepy-placed-bounds"
    )
    assert circle_placed_bounds[1] > circle_layout_bounds[3]
    marker = root.find(f".//{{{SVG_NS}}}marker")
    assert marker is not None
    assert marker.attrib["refX"] == "6"


def test_stage5_circle_diameter_rejects_compound_target(tmp_path):
    assembly = _make_annotation_plate_for_drawing_test()
    leader_ref = assembly.part_ref_for_leader()
    request = make_construction_drawing_request(
        name="invalid_diameter",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
        annotations=[
            {
                "id": "invalid",
                "operation": "circle_diameter",
                "target": leader_ref,
            }
        ],
    )

    with pytest.raises(ValueError, match="exactly one 2D circle or arc"):
        render_construction_drawing(
            assembly.leader,
            request,
            tmp_path / "invalid_diameter.svg",
            part_identity="plate",
            source=leader_ref,
        )


def test_stage5_annotation_placement_rejects_model_z_alignments():
    with pytest.raises(ValueError, match="only supports 2D alignments"):
        make_construction_drawing_request(
            name="invalid_alignment",
            parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
            annotations=[
                {
                    "id": "invalid",
                    "operation": "bounding_box_x_dimension",
                    "target": "drawing_test.leader",
                    "placement": {"alignments": [{"alignment": "STACK_TOP"}]},
                }
            ],
        )


def test_stage5_threaded_circle_callouts_emit_supplier_style_labels_and_leaders(
    tmp_path,
):
    blind_cutter = create_cylinder(1.25, 7.0, origin=(15.0, 10.0, -1.0))
    through_cutter = create_cylinder(1.25, 10.0, origin=(45.0, 10.0, -1.0))
    leader = create_box(60.0, 24.0, 8.0).cut(blind_cutter.fuse(through_cutter))
    assembly = LeaderFollowersCuttersPart(
        leader=leader,
        cutters=[blind_cutter, through_cutter],
        cutter_names=["m3_blind", "m3_through"],
        additional_data={"part_ref_origin": {"assembly_name": "thread_callout_test"}},
    )
    leader_ref = assembly.part_ref_for_leader()
    blind_ref = assembly.part_ref_for_named_cutter("m3_blind")
    through_ref = assembly.part_ref_for_named_cutter("m3_through")
    request = make_construction_drawing_request(
        name="thread_callouts",
        parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
        annotations=[
            {
                "id": "blind_m3",
                "operation": "circle_diameter",
                "target": blind_ref,
                "quantity": 2,
                "diameter_tolerance": "±0.05",
                "thread_size": "M3",
                "thread_tolerance_class": "6H",
                "depth": 6,
                "leader_tilt_degrees": 20,
                "leader_elbow_length": 7,
                "placement": {
                    "alignments": [{"alignment": "STACK_BACK", "stack_gap": 6.0}]
                },
            },
            {
                "id": "through_m3",
                "operation": "circle_diameter",
                "target": through_ref,
                "quantity": 2,
                "thread_size": "M3",
                "thread_tolerance_class": "6H",
                "through": True,
                "placement": {
                    "alignments": [{"alignment": "STACK_RIGHT", "stack_gap": 8.0}]
                },
            },
        ],
    )
    records = []
    destination = tmp_path / "thread_callouts.svg"

    render_construction_drawing_parts(
        [{"part": assembly.leader, "name": "plate", "source": leader_ref}],
        request,
        destination,
        annotation_targets={
            blind_ref: {"part": assembly.get_named_cutter("m3_blind")},
            through_ref: {"part": assembly.get_named_cutter("m3_through")},
        },
        annotation_records=records,
    )

    root = ET.fromstring(destination.read_bytes())
    dimensions = {
        element.attrib["data-shellforgepy-annotation-id"]: element
        for element in root.findall(
            ".//svg:g[@data-shellforgepy-role='dimension']", {"svg": SVG_NS}
        )
    }
    blind = dimensions["blind_m3"]
    through = dimensions["through_m3"]
    blind_text = [element.text for element in blind.findall(f".//{{{SVG_NS}}}text")]
    through_text = [element.text for element in through.findall(f".//{{{SVG_NS}}}text")]
    assert blind_text == ["2 X M3 - 6H ↧ 6.00", "⌀2.50 ±0.05"]
    assert through_text == ["2 X M3 - 6H THRU", "⌀2.50"]
    assert blind.attrib["data-shellforgepy-thread-size"] == "M3"
    assert blind.attrib["data-shellforgepy-diameter-tolerance"] == "±0.05"
    assert through.attrib["data-shellforgepy-through"] == "true"
    assert records[0]["callout"] == {
        "depth": 6.0,
        "diameter_tolerance": "±0.05",
        "leader_elbow_length": 7.0,
        "leader_tilt_degrees": 20.0,
        "quantity": 2,
        "thread_size": "M3",
        "thread_tolerance_class": "6H",
    }

    blind_lines = blind.findall(f".//{{{SVG_NS}}}line")
    through_lines = through.findall(f".//{{{SVG_NS}}}line")
    assert len(blind_lines) == 2
    assert blind_lines[0].attrib["marker-end"].startswith("url(#")
    assert float(blind_lines[0].attrib["x1"]) != float(blind_lines[0].attrib["x2"])
    assert float(blind_lines[0].attrib["y1"]) != float(blind_lines[0].attrib["y2"])
    assert float(blind_lines[1].attrib["y1"]) == float(blind_lines[1].attrib["y2"])
    assert abs(
        float(blind_lines[1].attrib["x2"]) - float(blind_lines[1].attrib["x1"])
    ) >= construction_module._estimated_annotation_text_width("2 X M3 - 6H ↧ 6.00")
    blind_layout_bounds = _bounds_attribute(blind, "data-shellforgepy-layout-bounds")
    blind_placed_bounds = _bounds_attribute(blind, "data-shellforgepy-placed-bounds")
    through_layout_bounds = _bounds_attribute(
        through, "data-shellforgepy-layout-bounds"
    )
    through_placed_bounds = _bounds_attribute(
        through, "data-shellforgepy-placed-bounds"
    )
    assert blind_placed_bounds[1] > blind_layout_bounds[3]
    assert through_placed_bounds[0] > through_layout_bounds[2]
    assert -float(blind.findall(f".//{{{SVG_NS}}}text")[0].attrib["y"]) - float(
        blind_lines[1].attrib["y1"]
    ) == pytest.approx(construction_module._annotation_text_baseline_above_line())
    assert float(blind_lines[1].attrib["y1"]) - -float(
        blind.findall(f".//{{{SVG_NS}}}text")[1].attrib["y"]
    ) == pytest.approx(-construction_module._annotation_text_baseline_below_line())
    blind_length = math.hypot(
        float(blind_lines[0].attrib["x1"]) - float(blind_lines[0].attrib["x2"]),
        float(blind_lines[0].attrib["y1"]) - float(blind_lines[0].attrib["y2"]),
    )
    through_length = math.hypot(
        float(through_lines[0].attrib["x1"]) - float(through_lines[0].attrib["x2"]),
        float(through_lines[0].attrib["y1"]) - float(through_lines[0].attrib["y2"]),
    )
    assert blind_length >= 7.0
    assert through_length >= 6.0 - 1e-8


def test_stage5_circle_callout_uses_tilted_leader_for_arc_target():
    parent = ET.Element(f"{{{SVG_NS}}}g")
    arc = append_arc(
        parent,
        cx=0.0,
        cy=0.0,
        radius=5.0,
        start_x=5.0,
        start_y=0.0,
        end_x=-5.0,
        end_y=0.0,
        large_arc=False,
        sweep=True,
    )
    annotation_parent = ET.Element(f"{{{SVG_NS}}}g")
    value, _ = construction_module._append_circle_diameter_dimension(
        annotation_parent,
        target_elements=[arc],
        target_bounds=(-5.0, 0.0, 5.0, 5.0),
        layout_bounds=(-5.0, 0.0, 5.0, 5.0),
        placement={"alignments": [{"alignment": "STACK_BACK", "stack_gap": 4.0}]},
        marker_id="arrow",
        precision=2,
        annotation={"operation": "circle_diameter"},
    )

    leader = annotation_parent.findall(f".//{{{SVG_NS}}}line")[0]
    assert value == 10.0
    assert leader.attrib["marker-end"] == "url(#arrow)"
    assert float(leader.attrib["x1"]) != float(leader.attrib["x2"])
    assert float(leader.attrib["y1"]) != float(leader.attrib["y2"])


def test_drawing_layout_uses_shellforgepy_planar_alignment_semantics():
    moving = Bounds2D(2.0, 3.0, 6.0, 5.0)
    target = Bounds2D(10.0, 20.0, 30.0, 40.0)

    assert alignment_delta_2d(
        moving, target, alignment=Alignment.STACK_LEFT, stack_gap=2.0
    ) == (2.0, 0.0)
    assert alignment_delta_2d(
        moving, target, alignment=Alignment.STACK_BACK, stack_gap=2.0
    ) == (0.0, 39.0)
    assert align_bounds_sequence_2d(
        moving,
        target,
        ((Alignment.STACK_RIGHT, 3.0), (Alignment.EDGE_BACK, 0.0)),
    ) == Bounds2D(33.0, 39.0, 37.0, 41.0)
    with pytest.raises(ValueError, match="Unsupported 2D drawing alignment"):
        alignment_delta_2d(moving, target, alignment=Alignment.STACK_TOP)


def test_stage5_plain_circle_callout_supports_through_and_depth_suffixes():
    assert construction_module._circle_diameter_callout_lines(
        3.6,
        annotation={"quantity": 4, "diameter_tolerance": "±0.10", "through": True},
        precision=2,
    ) == ("4 X ⌀3.60 ±0.10 THRU",)
    assert construction_module._circle_diameter_callout_lines(
        3.6,
        annotation={"depth": 4},
        precision=2,
    ) == ("⌀3.60 ↧ 4.00",)


@pytest.mark.parametrize(
    "annotation, message",
    [
        (
            {"operation": "bounding_box_x_dimension", "through": True},
            "callout fields only apply",
        ),
        (
            {"operation": "circle_diameter", "thread_tolerance_class": "6H"},
            "requires thread_size",
        ),
        (
            {"operation": "circle_diameter", "depth": 5, "through": True},
            "mutually exclusive",
        ),
        (
            {"operation": "circle_diameter", "quantity": 0},
            "positive integer",
        ),
    ],
)
def test_stage5_circle_callout_rejects_invalid_declarations(annotation, message):
    with pytest.raises(ValueError, match=message):
        make_construction_drawing_request(
            name="invalid_callout",
            parts=[{"source": "self", "artifact": "leader", "name": "plate"}],
            annotations=[
                {
                    "id": "invalid",
                    "target": "drawing_test.cutters.hole_left",
                    **annotation,
                }
            ],
        )
