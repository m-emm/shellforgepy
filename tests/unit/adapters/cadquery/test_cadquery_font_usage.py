import pytest

pytest.importorskip("cadquery")

from shellforgepy.adapters.cadquery import cadquery_adapter


def test_create_text_object_passes_font_path(monkeypatch, tmp_path):
    from cadquery.occ_impl import shapes

    fake_font = tmp_path / "custom_font.ttf"
    fake_font.write_text("test")

    captured = {}

    def fake_make_text(cls, *args, **kwargs):
        captured["font"] = kwargs.get("font")
        captured["fontPath"] = kwargs.get("fontPath")
        raise RuntimeError("makeText invoked")

    monkeypatch.setattr(shapes.Compound, "makeText", classmethod(fake_make_text))

    with pytest.raises(RuntimeError, match="makeText invoked"):
        cadquery_adapter.create_text_object(
            "Forge",
            size=4.0,
            thickness=1.0,
            font_path=str(fake_font),
        )

    assert captured["fontPath"] == str(fake_font)
    assert captured["font"] == fake_font.stem.replace("_", " ")
