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

from shellforgepy.shells.transformed_region_view import TransformedRegionView

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
from .construct.construct_utils import fibonacci_sphere, normalize
from .geometry.face_point_cloud import face_point_cloud
from .geometry.higher_order_solids import (
    create_hex_prism,
    create_trapezoid,
    directed_cylinder_at,
)
from .geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
)
from .geometry.treapezoidal_snake_geometry import create_trapezoidal_snake_geometry

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
from .shells.materialized_connectors import create_screw_connector_normal
from .shells.mesh_partition import MeshPartition
from .shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
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
            "get_bounding_box": adapter.get_bounding_box,
            "get_bounding_box_center": adapter.get_bounding_box_center,
            "get_vertex_coordinates": adapter.get_vertex_coordinates,
            "get_z_min": adapter.get_z_min,
            "create_extruded_polygon": adapter.create_extruded_polygon,
            "create_filleted_box": adapter.create_filleted_box,
            "get_volume": getattr(adapter, "get_volume", None),
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
            "create_extruded_polygon": _missing_cad_error("create_extruded_polygon"),
            "get_bounding_box_center": _missing_cad_error("get_bounding_box_center"),
            "get_volume": _missing_cad_error("get_volume"),
            "create_filleted_box": _missing_cad_error("create_filleted_box"),
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

get_bounding_box = _cad_functions["get_bounding_box"]
get_bounding_box_center = _cad_functions["get_bounding_box_center"]
get_vertex_coordinates = _cad_functions["get_vertex_coordinates"]
get_z_min = _cad_functions.get("get_z_min")
create_extruded_polygon = _cad_functions["create_extruded_polygon"]
create_filleted_box = _cad_functions["create_filleted_box"]
get_volume = _cad_functions.get("get_volume")

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
    "get_bounding_box_center",
    "get_vertex_coordinates",
    "get_z_min",
    "fibonacci_sphere",
    "normalize",
    "face_point_cloud",
    "MeshPartition",
    "PartitionableSpheroidTriangleMesh",
    "coordinate_system_transform",
    "coordinate_system_transform_to_matrix",
    "coordinate_system_transformation_function",
    "TransformedRegionView",
    "create_trapezoidal_snake_geometry",
    "create_hex_prism",
    "create_trapezoid",
    "create_screw_connector_normal",
    "create_extruded_polygon",
    "get_volume",
    "create_filleted_box",
]
