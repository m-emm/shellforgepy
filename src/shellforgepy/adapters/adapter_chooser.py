import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def _try_import_occ_adapter() -> bool:
    """Attempt to import the external OCC adapter, adding local paths if needed."""

    try:
        import shellforgepy_occ_adapter  # noqa: F401

        return True
    except ImportError:
        pass

    # Try environment override
    env_path = os.environ.get("SHELLFORGEPY_OCC_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    # Try sibling checkout (../shellforgepy-occ-adapter)
    here = Path(__file__).resolve()
    repo_root = here.parents[
        3
    ]  # <repo>/src/shellforgepy/adapters/adapter_chooser.py -> repo root
    candidates.extend(
        [
            repo_root / "shellforgepy-occ-adapter",
            repo_root.parent / "shellforgepy-occ-adapter",
        ]
    )

    for cand in candidates:
        cand_src = cand / "src"
        if cand_src.exists() and cand_src.is_dir():
            if str(cand_src) not in sys.path:
                sys.path.insert(0, str(cand_src))
            try:
                import shellforgepy_occ_adapter  # noqa: F401

                return True
            except ImportError:
                continue

    return False


def detect_cad_environment():
    """
    Automatically detect which CAD environment is available.

    Returns:
        str: 'occ', 'cadquery', 'freecad', or raises ImportError if none is available
    """
    # Prefer external OCC adapter first (lightweight dependency)
    if _try_import_occ_adapter():
        return "occ"

    # Try CadQuery next (more common in Python-only environments)
    try:
        import cadquery

        if hasattr(cadquery, "Shape"):
            return "cadquery"
        _logger.debug(
            "cadquery module was importable but lacks expected API; trying FreeCAD",
        )
    except ImportError:
        pass

    # Try FreeCAD
    try:
        import FreeCAD

        return "freecad"
    except ImportError:
        pass

    # If neither is available, raise an informative error
    raise ImportError(
        "None of the supported CAD backends are available. "
        "Please install one of them:\n"
        "  Preferred OCC adapter: pip install shellforgepy-occ-adapter\n"
        "  CadQuery: pip install cadquery\n"
        "  FreeCAD: install FreeCAD application or conda install freecad"
    )


def import_adapter_module():
    """
    Import the appropriate adapter module based on available CAD environment.

    You can override the auto-detection by setting the SHELLFORGEPY_ADAPTER
    environment variable to 'occ', 'cadquery' or 'freecad'.
    """
    import os

    # Allow manual override via environment variable
    adapter_type = os.environ.get("SHELLFORGEPY_ADAPTER")

    if adapter_type:
        if adapter_type not in ["occ", "cadquery", "freecad"]:
            raise ValueError(
                f"Invalid SHELLFORGEPY_ADAPTER value: {adapter_type}. "
                "Must be 'occ', 'cadquery' or 'freecad'"
            )
    else:
        # Auto-detect if not manually specified
        adapter_type = detect_cad_environment()

    if adapter_type == "occ":
        from shellforgepy.adapters.occ import occ_adapter as adapter_module
    elif adapter_type == "cadquery":
        from shellforgepy.adapters.cadquery import cadquery_adapter as adapter_module
    elif adapter_type == "freecad":
        from shellforgepy.adapters.freecad import freecad_adapter as adapter_module
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    return adapter_module


# Cache the adapter to avoid repeated detection/import
_cached_adapter = None


def get_cad_adapter():
    """Get the CAD adapter, caching it for subsequent calls."""
    global _cached_adapter
    if _cached_adapter is None:
        _cached_adapter = import_adapter_module()
    return _cached_adapter


# For backward compatibility
cad_adapter = get_cad_adapter()
