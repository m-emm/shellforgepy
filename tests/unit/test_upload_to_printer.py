from __future__ import annotations

import builtins
from pathlib import Path

from shellforgepy.workflow.upload_to_printer import (
    embed_preview_image,
    find_preview_image_path,
)


def test_find_preview_image_path_prefers_png_then_ppm(tmp_path):
    gcode_path = tmp_path / "sample.gcode"
    gcode_path.write_text("G1 X0 Y0\n", encoding="utf-8")

    ppm_path = tmp_path / "sample_preview.ppm"
    ppm_path.write_text("ppm", encoding="utf-8")
    assert find_preview_image_path(gcode_path) == ppm_path

    png_path = tmp_path / "sample_preview.png"
    png_path.write_text("png", encoding="utf-8")
    assert find_preview_image_path(gcode_path) == png_path


def test_embed_preview_image_skips_when_pillow_missing(monkeypatch, tmp_path):
    gcode_path = tmp_path / "sample.gcode"
    gcode_path.write_text("; HEADER_BLOCK_END\nG1 X0 Y0\n", encoding="utf-8")
    image_path = tmp_path / "sample_preview.ppm"
    image_path.write_text("P6\n1 1\n255\n\x00\x00\x00", encoding="latin-1")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    embed_preview_image(gcode_path, image_path)

    assert gcode_path.read_text(encoding="utf-8") == "; HEADER_BLOCK_END\nG1 X0 Y0\n"
