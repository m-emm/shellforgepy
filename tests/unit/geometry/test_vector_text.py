import pytest
from shellforgepy.geometry.vector_text import (
    PUNCTUATION_CHARACTERS,
    VectorTextSegment,
    layout_vector_text,
    segment_stroke_polygon,
    supported_vector_text_characters,
)


def test_vector_text_glyph_coverage():
    required = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    required.update(PUNCTUATION_CHARACTERS)

    assert required <= supported_vector_text_characters()


def test_vector_text_layout_normalizes_lowercase():
    layout = layout_vector_text("forge-42", 10.0)

    assert layout.text == "FORGE-42"
    assert layout.size == 10.0
    assert layout.stroke_width == pytest.approx(0.8)
    assert len(layout.segments) > 0


def test_vector_text_layout_handles_spaces_and_newlines():
    layout = layout_vector_text("A A\nB", 10.0, stroke_width=0.5)

    xs = [
        point[0]
        for segment in layout.segments
        for point in (segment.start, segment.end)
    ]
    ys = [
        point[1]
        for segment in layout.segments
        for point in (segment.start, segment.end)
    ]

    assert max(xs) > 10.0
    assert min(ys) < 0.0
    assert layout.stroke_width == 0.5


def test_vector_text_layout_supports_all_punctuation():
    for char in PUNCTUATION_CHARACTERS:
        layout = layout_vector_text(char, 5.0)
        assert layout.segments, char


def test_vector_text_decimal_point_is_two_strokes_tall():
    layout = layout_vector_text(".", 4.5, stroke_width=0.6)
    dot_points = [
        point
        for segment in layout.segments
        for polygon in (segment_stroke_polygon(segment, layout.stroke_width),)
        for point in polygon
    ]
    dot_xs = [point[0] for point in dot_points]
    dot_ys = [point[1] for point in dot_points]

    assert len(layout.segments) == 2
    assert layout.stroke_width == pytest.approx(0.6)
    assert max(dot_xs) - min(dot_xs) == pytest.approx(1.2)
    assert max(dot_ys) - min(dot_ys) == pytest.approx(1.2)


def test_vector_text_layout_rejects_unsupported_characters():
    with pytest.raises(ValueError, match="Unsupported vector text character"):
        layout_vector_text("FORGE\t42", 10.0)


def test_vector_text_layout_rejects_whitespace_only_text():
    with pytest.raises(ValueError, match="no renderable"):
        layout_vector_text(" \n ", 10.0)


def test_vector_text_layout_validates_dimensions():
    with pytest.raises(ValueError, match="Text must be a non-empty string"):
        layout_vector_text("", 10.0)

    with pytest.raises(ValueError, match="Size must be positive"):
        layout_vector_text("A", 0.0)

    with pytest.raises(ValueError, match="Stroke width"):
        layout_vector_text("A", 10.0, stroke_width=10.0)


def test_segment_stroke_polygon_wraps_centerline():
    segment = VectorTextSegment((0.0, 0.0), (2.0, 0.0))
    polygon = segment_stroke_polygon(segment, 0.4)
    coordinates = [coordinate for point in polygon for coordinate in point]

    assert coordinates == pytest.approx([0.0, 0.2, 2.0, 0.2, 2.0, -0.2, 0.0, -0.2])
