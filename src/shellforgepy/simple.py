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

from shellforgepy.geometry.m_screws import (
    create_bolt_thread,
    create_cylinder_screw,
    create_nut,
    get_clearance_hole_diameter,
    get_screw_info,
    list_supported_sizes,
)
from shellforgepy.geometry.mesh_utils import (
    merge_meshes,
    write_shell_maps_to_stl,
    write_stl_binary,
)
from shellforgepy.shells.transformed_region_view import TransformedRegionView

# Core alignment functionality
from .construct.alignment import ALIGNMENT_SIGNS, Alignment
from .construct.alignment_operations import (
    align,
    align_translation,
    alignment_signs,
    chain_translations,
    rotate,
    stack_alignment_of,
    translate,
)
from .construct.construct_utils import fibonacci_sphere, normalize
from .construct.leaders_followers_cutters_part import LeaderFollowersCuttersPart
from .construct.named_part import NamedPart
from .construct.part_collector import PartCollector
from .geometry.face_point_cloud import face_point_cloud
from .geometry.higher_order_solids import (
    create_hex_prism,
    create_ring,
    create_screw_thread,
    create_trapezoid,
    directed_cylinder_at,
)
from .geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
)
from .geometry.treapezoidal_snake_geometry import create_trapezoidal_snake_geometry
from .produce.arrange_and_export import arrange_and_export_parts, export_solid_to_stl
from .produce.production_parts_model import PartInfo, PartList
from .shells.materialized_connectors import create_screw_connector_normal
from .shells.mesh_partition import MeshPartition
from .shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)

ADAPTER_FUNTIONS = [
    "create_basic_box",
    "create_basic_cone",
    "create_basic_cylinder",
    "create_basic_sphere",
    "create_solid_from_traditional_face_vertex_maps",
    "create_text_object",
    "get_bounding_box",
    "get_bounding_box_center",
    "get_vertex_coordinates",
    "get_z_min",
    "create_extruded_polygon",
    "create_filleted_box",
    "get_volume",
    "filter_edges_by_z_position",
    "filter_edges_by_alignment",
    "filter_edges_by_function",
    "apply_fillet_to_edges",
    "apply_fillet_by_alignment",
    "apply_fillet_by_function",
    "get_adapter_id",
]


# Dynamically load CAD adapter functions
def _load_cad_functions():
    """Load CAD adapter functions dynamically to handle import errors gracefully."""
    from .adapters.adapter_chooser import get_cad_adapter

    try:
        adapter = get_cad_adapter()
        return {
            func_name: getattr(adapter, func_name) for func_name in ADAPTER_FUNTIONS
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
            func_name: _missing_cad_error(func_name) for func_name in ADAPTER_FUNTIONS
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
filter_edges_by_z_position = _cad_functions["filter_edges_by_z_position"]
filter_edges_by_alignment = _cad_functions["filter_edges_by_alignment"]
filter_edges_by_function = _cad_functions["filter_edges_by_function"]
apply_fillet_to_edges = _cad_functions["apply_fillet_to_edges"]
apply_fillet_by_alignment = _cad_functions["apply_fillet_by_alignment"]
apply_fillet_by_function = _cad_functions["apply_fillet_by_function"]
get_adapter_id = _cad_functions["get_adapter_id"]

# Define what gets exported with "from simple import *"
__all__ = [
    # Alignment
    "Alignment",
    "ALIGNMENT_SIGNS",
    "stack_alignment_of",
    "alignment_signs",
    "translate",
    "rotate",
    "align_translation",
    "align",
    "chain_translations",
    # Solid builders
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
    "create_ring",
    "create_screw_thread",
    "create_trapezoid",
    "create_screw_connector_normal",
    "create_extruded_polygon",
    "create_bolt_thread",
    "create_cylinder_screw",
    "create_nut",
    "get_clearance_hole_diameter",
    "get_screw_info",
    "list_supported_sizes",
    "write_stl_binary",
    "merge_meshes",
] + ADAPTER_FUNTIONS
