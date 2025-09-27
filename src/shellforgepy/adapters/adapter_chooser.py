def detect_cad_environment():
    """
    Automatically detect which CAD environment is available.

    Returns:
        str: 'cadquery', 'freecad', or raises ImportError if neither is available
    """
    # Try CadQuery first (more common in Python-only environments)
    try:
        import cadquery

        return "cadquery"
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
        "Neither CadQuery nor FreeCAD is available. "
        "Please install one of them:\n"
        "  For CadQuery: pip install cadquery\n"
        "  For FreeCAD: Install FreeCAD application or conda install freecad"
    )


def import_adapter_module():
    """
    Import the appropriate adapter module based on available CAD environment.

    You can override the auto-detection by setting the SHELLFORGEPY_ADAPTER
    environment variable to 'cadquery' or 'freecad'.
    """
    import os

    # Allow manual override via environment variable
    adapter_type = os.environ.get("SHELLFORGEPY_ADAPTER")

    if adapter_type:
        if adapter_type not in ["cadquery", "freecad"]:
            raise ValueError(
                f"Invalid SHELLFORGEPY_ADAPTER value: {adapter_type}. "
                "Must be 'cadquery' or 'freecad'"
            )
    else:
        # Auto-detect if not manually specified
        adapter_type = detect_cad_environment()

    if adapter_type == "cadquery":
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
