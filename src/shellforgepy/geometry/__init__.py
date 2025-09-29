"""
Geometry module for shellforgepy.

Provides geometric primitives, utilities, and higher-order solids for CAD operations.
"""

from .higher_order_solids import (
    create_hex_prism,
    create_ring,
    create_screw_thread,
    create_trapezoid,
    directed_cylinder_at,
)
from .m_screws import (
    create_bolt_thread,
    create_cylinder_screw,
    create_nut,
    get_clearance_hole_diameter,
    get_core_hole_diameter,
    get_nut_outer_diameter,
    get_screw_info,
    get_thread_pitch,
    list_supported_sizes,
    m_screws_table,
)

__all__ = [
    # Higher order solids
    "create_hex_prism",
    "create_ring",
    "create_screw_thread",
    "create_trapezoid",
    "directed_cylinder_at",
    # M-screws
    "create_bolt_thread",
    "create_cylinder_screw",
    "create_nut",
    "get_clearance_hole_diameter",
    "get_core_hole_diameter",
    "get_nut_outer_diameter",
    "get_screw_info",
    "get_thread_pitch",
    "list_supported_sizes",
    "m_screws_table",
]
