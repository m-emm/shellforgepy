import pytest
import shellforgepy.adapters.font_resolver as font_resolver
from shellforgepy.adapters.font_resolver import FontResolutionError, resolve_font


def test_resolve_font_prefers_explicit_font_path(tmp_path):
    font_file = tmp_path / "custom.ttf"
    font_file.write_text("")

    spec = resolve_font(font_path=str(font_file))

    assert spec.path == str(font_file)
    assert spec.family == "custom"


def test_resolve_font_prefers_lintsec_when_available(tmp_path, monkeypatch):
    font_file = tmp_path / "lintsec.regular.ttf"
    font_file.write_text("")

    monkeypatch.setattr(font_resolver, "_default_font_dirs", lambda: (tmp_path,))
    monkeypatch.delenv("SHELLFORGEPY_FONT", raising=False)
    monkeypatch.delenv("SHELLFORGEPY_FONT_PATH", raising=False)

    spec = resolve_font()

    assert spec.path == str(font_file)
    assert spec.family == "Lintsec"


def test_resolve_font_require_path_raises_when_missing(monkeypatch):
    monkeypatch.setattr(font_resolver, "_default_font_dirs", lambda: ())
    monkeypatch.delenv("SHELLFORGEPY_FONT", raising=False)
    monkeypatch.delenv("SHELLFORGEPY_FONT_PATH", raising=False)

    with pytest.raises(FontResolutionError):
        resolve_font(require_path=True)
