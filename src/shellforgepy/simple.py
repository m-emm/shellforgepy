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

from shellforgepy.adapters._adapter import (
    apply_fillet_by_alignment,
    apply_fillet_to_edges,
    create_box,
    create_cone,
    create_cylinder,
    create_extruded_polygon,
    create_filleted_box,
    create_solid_from_traditional_face_vertex_maps,
    create_sphere,
    create_text_object,
    get_adapter_id,
    get_bounding_box,
    get_bounding_box_center,
    get_bounding_box_size,
    get_vertex_coordinates,
    get_volume,
)
from shellforgepy.geometry.m_screws import (
    create_bolt_thread,
    create_cylinder_screw,
    create_nut,
    get_clearance_hole_diameter,
    get_screw_info,
    list_supported_sizes,
    m_screws_table,
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
    mirror,
    rotate,
    stack_alignment_of,
    translate,
)
from .construct.construct_utils import fibonacci_sphere, normalize
from .construct.leader_followers_cutters_part import LeaderFollowersCuttersPart
from .construct.named_part import NamedPart
from .construct.part_collector import PartCollector
from .geometry.face_point_cloud import face_point_cloud
from .geometry.higher_order_solids import (
    create_hex_prism,
    create_right_triangle,
    create_ring,
    create_rounded_slab,
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
from .produce.arrange_and_export import (
    arrange_and_export,
    arrange_and_export_parts,
    export_solid_to_stl,
)
from .produce.production_parts_model import PartInfo, PartList
from .shells.materialized_connectors import create_screw_connector_normal
from .shells.mesh_partition import MeshPartition
from .shells.partitionable_spheroid_triangle_mesh import (
    PartitionableSpheroidTriangleMesh,
)

# Define what gets exported with "from simple import *"
__all__ = [
    "align_translation",
    "align",
    "alignment_signs",
    "ALIGNMENT_SIGNS",
    "Alignment",
    "arrange_and_export_parts",
    "arrange_and_export",
    "chain_translations",
    "coordinate_system_transform_to_matrix",
    "coordinate_system_transform",
    "coordinate_system_transformation_function",
    "create_box",
    "create_bolt_thread",
    "create_cylinder_screw",
    "create_extruded_polygon",
    "create_hex_prism",
    "create_nut",
    "create_right_triangle",
    "create_ring",
    "create_rounded_slab",
    "create_screw_connector_normal",
    "create_screw_thread",
    "create_trapezoid",
    "create_trapezoidal_snake_geometry",
    "directed_cylinder_at",
    "export_solid_to_stl",
    "face_point_cloud",
    "fibonacci_sphere",
    "get_bounding_box_center",
    "get_bounding_box",
    "get_clearance_hole_diameter",
    "get_screw_info",
    "get_vertex_coordinates",
    "LeaderFollowersCuttersPart",
    "list_supported_sizes",
    "m_screws_table",
    "merge_meshes",
    "MeshPartition",
    "mirror",
    "NamedPart",
    "normalize",
    "PartCollector",
    "PartInfo",
    "PartitionableSpheroidTriangleMesh",
    "PartList",
    "rotate",
    "stack_alignment_of",
    "TransformedRegionView",
    "translate",
    "write_stl_binary",
    "create_box",
    "create_cone",
    "create_cylinder",
    "create_sphere",
    "create_solid_from_traditional_face_vertex_maps",
    "create_text_object",
    "get_bounding_box",
    "get_bounding_box_center",
    "get_bounding_box_size",
    "get_vertex_coordinates",
    "create_extruded_polygon",
    "create_filleted_box",
    "get_volume",
    "apply_fillet_to_edges",
    "apply_fillet_by_alignment",
    "get_adapter_id",
]
