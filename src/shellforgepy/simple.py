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
from shellforgepy.geometry.mesh_builders import (
    create_cube_geometry,
    create_dodecahedron_geometry,
    create_fibonacci_sphere_geometry,
    create_icosahedron_geometry,
    create_tetrahedron_geometry,
)
from shellforgepy.geometry.mesh_utils import (
    calc_distance_to_path,
    convert_to_traditional_face_vertex_maps,
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
from .construct.construct_utils import (
    compute_triangle_normal,
    fibonacci_sphere,
    normalize,
)
from .construct.leader_followers_cutters_part import LeaderFollowersCuttersPart
from .construct.named_part import NamedPart
from .construct.part_collector import PartCollector
from .geometry.face_point_cloud import face_point_cloud
from .geometry.higher_order_solids import (
    create_distorted_cube,
    create_hex_prism,
    create_right_triangle,
    create_ring,
    create_rounded_slab,
    create_screw_thread,
    create_trapezoid,
    directed_box_at,
    directed_cylinder_at,
)
from .geometry.modifications import (
    orient_for_flatness,
    orient_for_flatness_riemannian,
    slice_part,
)
from .geometry.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    coordinate_system_transformation_function,
    matrix_to_coordinate_system_transform,
    matrix_to_coordinate_system_transformation_function,
    ray_triangle_intersect,
)
from .geometry.treapezoidal_snake_geometry import create_trapezoidal_snake_geometry
from .produce.arrange_and_export import (
    arrange_and_export,
    arrange_and_export_parts,
    export_solid_to_stl,
)
from .produce.production_parts_model import PartInfo, PartList
from .shells.connector_hint import ConnectorHint
from .shells.materialized_connectors import (
    compute_transforms_from_hint,
    create_screw_connector_normal,
)
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
    "apply_fillet_by_alignment",
    "apply_fillet_to_edges",
    "arrange_and_export_parts",
    "arrange_and_export",
    "calc_distance_to_path",
    "chain_translations",
    "compute_transforms_from_hint",
    "ConnectorHint",
    "convert_to_traditional_face_vertex_maps",
    "coordinate_system_transform_to_matrix",
    "coordinate_system_transform",
    "coordinate_system_transformation_function",
    "create_bolt_thread",
    "create_box",
    "create_cone",
    "create_cube_geometry",
    "create_cylinder_screw",
    "create_cylinder",
    "create_distorted_cube",
    "create_dodecahedron_geometry",
    "create_extruded_polygon",
    "create_extruded_polygon",
    "create_fibonacci_sphere_geometry",
    "create_filleted_box",
    "create_hex_prism",
    "create_icosahedron_geometry",
    "create_nut",
    "create_right_triangle",
    "create_ring",
    "create_rounded_slab",
    "create_screw_connector_normal",
    "create_screw_thread",
    "create_solid_from_traditional_face_vertex_maps",
    "create_sphere",
    "create_tetrahedron_geometry",
    "create_text_object",
    "create_trapezoid",
    "create_trapezoidal_snake_geometry",
    "directed_box_at",
    "directed_cylinder_at",
    "export_solid_to_stl",
    "face_point_cloud",
    "fibonacci_sphere",
    "get_adapter_id",
    "get_bounding_box_center",
    "get_bounding_box_size",
    "get_bounding_box",
    "get_clearance_hole_diameter",
    "get_screw_info",
    "get_vertex_coordinates",
    "get_volume",
    "LeaderFollowersCuttersPart",
    "list_supported_sizes",
    "m_screws_table",
    "matrix_to_coordinate_system_transform",
    "matrix_to_coordinate_system_transformation_function",
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
    "slice_part",
    "stack_alignment_of",
    "TransformedRegionView",
    "translate",
    "write_shell_maps_to_stl",
    "write_stl_binary",
    "orient_for_flatness",
    "orient_for_flatness_riemannian",
    "ray_triangle_intersect",
    "compute_triangle_normal",
]


LOGGING_FORMAT = "%(asctime)s - %(name)-60s - %(levelname)-8s - %(message)s"


def _setup_logging():
    import os
    import sys

    def is_running_under_pytest():
        running_under_pytest = (
            bool(os.getenv("PYTEST_VERSION")) or "pytest" in sys.modules
        )
        return running_under_pytest

    # Allow disabling logging setup via environment variable, and do not init if running under pytest
    if (
        not os.environ.get("SHELLFORGEPY_NO_LOGGING_INIT", False)
        and not is_running_under_pytest()
    ):
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format=LOGGING_FORMAT,
        )

        # send all logs to stdout, not stderr
        logging.getLogger().handlers[0].stream = sys.stdout
        logging.getLogger("shellforgepy").setLevel(logging.INFO)


_setup_logging()
