"""
Simple import module for shellforgepy.

This module provides convenient access to all key classes and functions from the
shellforgepy package. Import this module to get access to the most
commonly used functionality.

Usage:
    from shellforgepy.simple import *

    # Now you can use:
    # - Alignment enums and functions
    # - Solid building utilities
    # - Part arrangement and export functions
"""

# Core alignment functionality
from .construct.alignment import ALIGNMENT_SIGNS, Alignment
from .construct.alignment_operations import (
    aligment_signs,
    align,
    align_translation,
    chain_translations,
    rotate,
    stack_alignment_of,
    translate,
)

# Part arrangement and export functionality
from .produce.arrange_and_export import (
    LeaderFollowersCuttersPart,
    NamedPart,
    PartCollector,
    PartInfo,
    PartList,
    arrange_and_export_parts,
    export_solid_to_stl,
)

# Note: Materialized connectors functionality is not yet implemented


# Dynamically load CAD adapter functions
def _load_cad_functions():
    """Load CAD adapter functions dynamically to handle import errors gracefully."""
    from .adapters.adapter_chooser import get_cad_adapter

    try:
        adapter = get_cad_adapter()
        return {
            "create_basic_box": adapter.create_basic_box,
            "create_basic_cone": adapter.create_basic_cone,
            "create_basic_cylinder": adapter.create_basic_cylinder,
            "create_basic_sphere": adapter.create_basic_sphere,
            "create_solid_from_traditional_face_vertex_maps": adapter.create_solid_from_traditional_face_vertex_maps,
            "create_text_object": adapter.create_text_object,
            "directed_cylinder_at": adapter.directed_cylinder_at,
            "get_bounding_box": adapter.get_bounding_box,
        }
    except ImportError as e:
        # Return stub functions that provide helpful error messages
        error_message = str(e)  # Capture the error message for use in nested functions

        def _missing_cad_error(func_name):
            def stub(*args, **kwargs):
                raise ImportError(
                    f"Cannot use {func_name}: {error_message}\n"
                    "Please ensure either CadQuery or FreeCAD is properly installed."
                )

            return stub

        return {
            "create_basic_box": _missing_cad_error("create_basic_box"),
            "create_basic_cone": _missing_cad_error("create_basic_cone"),
            "create_basic_cylinder": _missing_cad_error("create_basic_cylinder"),
            "create_basic_sphere": _missing_cad_error("create_basic_sphere"),
            "create_solid_from_traditional_face_vertex_maps": _missing_cad_error(
                "create_solid_from_traditional_face_vertex_maps"
            ),
            "create_text_object": _missing_cad_error("create_text_object"),
            "directed_cylinder_at": _missing_cad_error("directed_cylinder_at"),
            "get_bounding_box": _missing_cad_error("get_bounding_box"),
        }


# Load the CAD functions
_cad_functions = _load_cad_functions()

# Expose them at module level
create_basic_box = _cad_functions["create_basic_box"]
create_basic_cone = _cad_functions["create_basic_cone"]
create_basic_cylinder = _cad_functions["create_basic_cylinder"]
create_basic_sphere = _cad_functions["create_basic_sphere"]
create_solid_from_traditional_face_vertex_maps = _cad_functions[
    "create_solid_from_traditional_face_vertex_maps"
]
create_text_object = _cad_functions["create_text_object"]
directed_cylinder_at = _cad_functions["directed_cylinder_at"]
get_bounding_box = _cad_functions["get_bounding_box"]

# Define what gets exported with "from simple import *"
__all__ = [
    # Alignment
    "Alignment",
    "ALIGNMENT_SIGNS",
    "stack_alignment_of",
    "aligment_signs",
    "translate",
    "rotate",
    "align_translation",
    "align",
    "chain_translations",
    # Solid builders
    "create_solid_from_traditional_face_vertex_maps",
    "create_basic_box",
    "create_basic_cylinder",
    "create_basic_sphere",
    "create_basic_cone",
    "create_text_object",
    "directed_cylinder_at",
    "get_bounding_box",
    # Arrange and export
    "PartCollector",
    "PartInfo",
    "NamedPart",
    "PartList",
    "LeaderFollowersCuttersPart",
    "export_solid_to_stl",
    "arrange_and_export_parts",
]
