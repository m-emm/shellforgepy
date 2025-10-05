from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_logger = logging.getLogger(__name__)


class FontResolutionError(RuntimeError):
    """Raised when a suitable font file cannot be resolved."""


@dataclass(frozen=True)
class FontSpec:
    """Container describing the resolved font family and file path."""

    family: Optional[str]
    path: Optional[str]


def resolve_font(
    font: Optional[str] = None,
    font_path: Optional[str] = None,
    *,
    require_path: bool = False,
) -> FontSpec:
    """Resolve a font specification into a usable family name and font file path.

    Resolution order:
        1. Explicit ``font_path`` argument (must exist if provided).
        2. ``font`` argument if it points to a file path.
        3. ``SHELLFORGEPY_FONT_PATH`` environment variable.
        4. ``SHELLFORGEPY_FONT`` environment variable (treated as family first,
           but also accepted as a direct path).
        5. Common system font locations for candidate family names.

    Falls back to portable defaults (e.g. DejaVu Sans, Liberation Sans, Arial)
    to work out-of-the-box on most platforms.
    """

    provided_font = _normalize(font)
    provided_font_path = _normalize(font_path)
    env_font = _normalize(os.environ.get("SHELLFORGEPY_FONT"))
    env_font_path = _normalize(os.environ.get("SHELLFORGEPY_FONT_PATH"))

    resolved_family: Optional[str] = None
    resolved_path: Optional[str] = None
    search_preferences: list[FontPreference] = []

    if provided_font_path:
        resolved_path = _validate_existing_path(provided_font_path, "font_path")
        resolved_family = Path(resolved_path).stem.replace("_", " ")

    if provided_font:
        maybe_path = _maybe_existing_path(provided_font)
        if maybe_path:
            resolved_path = resolved_path or maybe_path
            resolved_family = resolved_family or Path(maybe_path).stem
        else:
            resolved_family = provided_font
            search_preferences.append(_preference_from_family(provided_font))

    if env_font_path:
        env_path = _maybe_existing_path(env_font_path)
        if env_path:
            resolved_path = resolved_path or env_path
            if resolved_family is None:
                resolved_family = Path(env_path).stem.replace("_", " ")

    if env_font:
        maybe_path = _maybe_existing_path(env_font)
        if maybe_path:
            resolved_path = resolved_path or maybe_path
            if resolved_family is None:
                resolved_family = Path(maybe_path).stem
        else:
            if resolved_family is None:
                resolved_family = env_font
            search_preferences.insert(0, _preference_from_family(env_font))

    lookup_order = tuple(search_preferences) + DEFAULT_FONT_PREFERENCES

    if resolved_path is None:
        for preference in lookup_order:
            found = _search_in_font_dirs(preference.filenames)
            if found:
                resolved_path = found
                resolved_family = preference.family
                break

    if resolved_family is None and lookup_order:
        resolved_family = lookup_order[0].family

    if resolved_path is None and require_path:
        raise FontResolutionError(
            "Unable to locate a usable font file for text creation. Set SHELLFORGEPY_FONT "
            "(family name) or SHELLFORGEPY_FONT_PATH (font file path) to specify one."
        )

    if resolved_family is None:
        resolved_family = "Arial"

    if resolved_path is None:
        _logger.debug(
            "Font path unresolved for family '%s'. Falling back to family only.",
            resolved_family,
        )
    else:
        resolved_family = resolved_family or Path(resolved_path).stem.replace("_", " ")

    _logger.debug("Resolved font: family=%s path=%s", resolved_family, resolved_path)
    return FontSpec(family=resolved_family, path=resolved_path)


@dataclass(frozen=True)
class FontPreference:
    family: str
    filenames: tuple[str, ...]


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_existing_path(path_str: str, argument_name: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FontResolutionError(
            f"The {argument_name} '{path}' does not exist or is not a file."
        )
    return str(path)


def _maybe_existing_path(path_str: str) -> Optional[str]:
    try:
        return _validate_existing_path(path_str, argument_name="font")
    except FontResolutionError:
        return None


def _preference_from_family(family: str) -> FontPreference:
    sanitized = family.replace(" ", "")
    dashed = family.replace(" ", "-")
    lower = family.lower().replace(" ", "")
    filenames = {
        f"{family}.ttf",
        f"{family}.TTF",
        f"{sanitized}.ttf",
        f"{sanitized}.TTF",
        f"{dashed}.ttf",
        f"{dashed}.TTF",
        f"{lower}.ttf",
        f"{lower}.TTF",
    }
    return FontPreference(family=family, filenames=tuple(filenames))


def _search_in_font_dirs(filenames: Iterable[str]) -> Optional[str]:
    for filename in filenames:
        for directory in _default_font_dirs():
            path = _find_font_in_directory(directory, filename)
            if path:
                return path
    return None


def _find_font_in_directory(directory: Path, filename: str) -> Optional[str]:
    if not directory.is_dir():
        return None
    try:
        for match in directory.rglob(filename):
            if match.is_file():
                return str(match)
    except (OSError, PermissionError):
        _logger.debug("Unable to search directory %s", directory, exc_info=True)
    return None


def _default_font_dirs() -> tuple[Path, ...]:
    home = Path.home()
    dirs: list[Path] = []
    system = platform.system().lower()

    if system == "windows":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        dirs.append(windir / "Fonts")
    elif system == "darwin":
        dirs.extend(
            [
                Path("/System/Library/Fonts"),
                Path("/System/Library/Fonts/Supplemental"),
                Path("/Library/Fonts"),
                home / "Library" / "Fonts",
            ]
        )
    else:
        dirs.extend(
            [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                home / ".fonts",
                home / ".local/share/fonts",
            ]
        )

    return tuple(dict.fromkeys(dirs))  # deduplicate while preserving order


DEFAULT_FONT_PREFERENCES: tuple[FontPreference, ...] = (
    FontPreference(
        family="Lintsec",
        filenames=(
            "lintsec.regular.ttf",
            "Lintsec.regular.ttf",
            "Lintsec-Regular.ttf",
            "lintsec.ttf",
        ),
    ),
    FontPreference(
        family="DejaVu Sans",
        filenames=(
            "DejaVuSans.ttf",
            "DejaVuSansCondensed.ttf",
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
        ),
    ),
    FontPreference(
        family="Liberation Sans",
        filenames=(
            "LiberationSans-Regular.ttf",
            "LiberationSans.ttf",
        ),
    ),
    FontPreference(
        family="Arial",
        filenames=(
            "Arial.ttf",
            "arial.ttf",
        ),
    ),
    FontPreference(
        family="Noto Sans",
        filenames=(
            "NotoSans-Regular.ttf",
            "NotoSans.ttf",
        ),
    ),
    FontPreference(
        family="FreeSans",
        filenames=(
            "FreeSans.ttf",
            "freesans.ttf",
        ),
    ),
)
