"""
M-Screws module for shellforgepy.

Provides comprehensive screw and nut geometry generation with standard metric dimensions
and tolerances. This module is a port of the original FreeCAD m_screws module to the
CAD-agnostic shellforgepy framework.

The module includes:
- Complete M-screw specification table (M2 through M12)
- Nut creation with configurable slack and hole options
- Bolt thread generation using trapezoidal snake geometry
- Cylinder head screw creation with threading options
- Helper functions for nut and screw dimensions

All functions follow the shellforgepy convention of being adapter-agnostic,
allowing them to work with any supported CAD backend.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from shellforgepy.adapters._adapter import (
    create_box,
    create_cone,
    create_cylinder,
    create_extruded_polygon,
    cut_parts,
    fuse_parts,
    get_bounding_box,
)
from shellforgepy.construct.alignment_operations import (
    Alignment,
    align,
    rotate,
    translate,
)
from shellforgepy.construct.bounding_box_helpers import get_zlen
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.geometry.higher_order_solids import create_screw_thread

# Complete metric screw specifications table
m_screws_table = {
    "M2": {
        "nut_size": 4,
        "cap_screw_size": 1.5,
        "cap_screw_head_size": 1.25,
        "grub_screw_wrench_size": 0.9,
        "clearance_hole_close": 2.2,
        "clearance_hole_normal": 2.4,
        "clearance_hole_loose": 2.6,
        "pitch": 0.4,
        "core_hole": 1.6,
        "nut_circle_diameter": 4.32,
        "nut_thickness": 1.6,
        "cylinder_head_diameter": 3.8,
        "cylinder_head_height": 2,
        "wrench_socket_outer_diameter": 7.0,
        "min_thread_length": 16,
        "thread_inset_hole_diameter": 3.54,
        "thread_inset_length": 4,
    },
    "M2.5": {
        "nut_size": 5,
        "cap_screw_size": 2,
        "cap_screw_head_size": 1.5,
        "grub_screw_wrench_size": 1.3,
        "clearance_hole_close": 2.7,
        "clearance_hole_normal": 2.9,
        "clearance_hole_loose": 3.1,
        "pitch": 0.45,
        "core_hole": 2.05,
        "nut_circle_diameter": 5.45,
        "nut_thickness": 2.0,
        "cylinder_head_diameter": 4.5,
        "cylinder_head_height": 2.5,
        "wrench_socket_outer_diameter": 7.5,
        "min_thread_length": 17,
        "conical_head_diameter": 5.0,
        "conical_head_height": 1.5,
    },
    "M3": {
        "nut_size": 5.5,
        "cap_screw_size": 2.5,
        "cap_screw_head_size": 2,
        "grub_screw_wrench_size": 1.5,
        "clearance_hole_close": 3.2,
        "clearance_hole_normal": 3.4,
        "clearance_hole_loose": 3.6,
        "pitch": 0.5,
        "core_hole": 2.5,
        "nut_circle_diameter": 6.01,
        "nut_thickness": 2.3,
        "cylinder_head_diameter": 5.5,
        "cylinder_head_height": 3,
        "wrench_socket_outer_diameter": 8.0,
        "min_thread_length": 18,
        "thread_inset_hole_diameter": 4.3,
        "thread_inset_length": 6,
        "conical_head_diameter": 6.0,
        "conical_head_height": 1.7,
    },
    "M4": {
        "nut_size": 7,
        "cap_screw_size": 3,
        "cap_screw_head_size": 2.5,
        "grub_screw_wrench_size": 2,
        "clearance_hole_close": 4.3,
        "clearance_hole_normal": 4.5,
        "clearance_hole_loose": 4.8,
        "pitch": 0.7,
        "core_hole": 3.3,
        "nut_circle_diameter": 7.66,
        "nut_thickness": 3.0,
        "cylinder_head_diameter": 7,
        "cylinder_head_height": 4,
        "wrench_socket_outer_diameter": 10.0,
        "min_thread_length": 20,
        "conical_head_diameter": 8.0,
        "conical_head_height": 2.3,
        "thread_inset_hole_diameter": 6.3,
        "thread_inset_length": 8.1,
    },
    "M5": {
        "nut_size": 8,
        "cap_screw_size": 4,
        "cap_screw_head_size": 3,
        "grub_screw_wrench_size": 2.5,
        "clearance_hole_close": 5.3,
        "clearance_hole_normal": 5.5,
        "clearance_hole_loose": 5.8,
        "pitch": 0.8,
        "core_hole": 4.2,
        "nut_circle_diameter": 8.79,
        "nut_thickness": 4.6,
        "cylinder_head_diameter": 8.5,
        "cylinder_head_height": 5,
        "wrench_socket_outer_diameter": 11.5,
        "min_thread_length": 22,
        "thread_inset_hole_diameter": 6.2,
        "thread_inset_length": 8,
        "conical_head_diameter": 10,
        "conical_head_height": 3.2,
    },
    "M6": {
        "nut_size": 10,
        "cap_screw_size": 5,
        "cap_screw_head_size": 4,
        "grub_screw_wrench_size": 3,
        "clearance_hole_close": 6.4,
        "clearance_hole_normal": 6.6,
        "clearance_hole_loose": 7,
        "pitch": 1,
        "core_hole": 5,
        "nut_circle_diameter": 11.05,
        "nut_thickness": 5.1,
        "cylinder_head_diameter": 10,
        "cylinder_head_height": 6,
        "wrench_socket_outer_diameter": 13.5,
        "min_thread_length": 24,
        "conical_head_diameter": 12.0,
        "conical_head_height": 3.3,
    },
    "M8": {
        "nut_size": 13,
        "cap_screw_size": 6,
        "cap_screw_head_size": 5,
        "grub_screw_wrench_size": 4,
        "clearance_hole_close": 8.4,
        "clearance_hole_normal": 9,
        "clearance_hole_loose": 10,
        "pitch": 1.25,
        "core_hole": 6.8,
        "nut_circle_diameter": 14.38,
        "nut_thickness": 6.6,
        "cylinder_head_diameter": 13,
        "cylinder_head_height": 8,
        "wrench_socket_outer_diameter": 16.5,
        "min_thread_length": 28,
        "conical_head_diameter": 16.0,
        "conical_head_height": 4.4,
    },
    "M10": {
        "nut_size": 16,
        "cap_screw_size": 17,
        "cap_screw_head_size": 8,
        "grub_screw_wrench_size": 6,
        "clearance_hole_close": 10.5,
        "clearance_hole_normal": 11,
        "clearance_hole_loose": 12,
        "pitch": 1.5,
        "core_hole": 8.5,
        "nut_circle_diameter": 17.77,
        "nut_thickness": 8.2,
        "cylinder_head_diameter": 16,
        "cylinder_head_height": 10,
        "wrench_socket_outer_diameter": 20.0,
        "min_thread_length": 32,
        "conical_head_diameter": 20.0,
        "conical_head_height": 5.5,
    },
    "M12": {
        "nut_size": 18,
        "cap_screw_size": 19,
        "cap_screw_head_size": 10,
        "grub_screw_wrench_size": 8,
        "clearance_hole_close": 13,
        "clearance_hole_normal": 13.5,
        "clearance_hole_loose": 14.5,
        "pitch": 1.75,
        "core_hole": 10.2,
        "nut_circle_diameter": 20.03,
        "nut_thickness": 10.6,
        "cylinder_head_diameter": 18,
        "cylinder_head_height": 12,
        "wrench_socket_outer_diameter": 22.5,
        "min_thread_length": 36,
    },
}

# DIN 562 low square nuts (Vierkantmuttern), dimensions in millimetres.
# ``width`` is the side length (s) and ``thickness`` is the minimum height (m).
square_nuts_table = {
    "M2": {"width": 4.0, "thickness": 1.2},
    "M2.5": {"width": 5.0, "thickness": 1.4},
    "M3": {"width": 5.5, "thickness": 1.6},
    "M4": {"width": 7.0, "thickness": 1.8},
    "M5": {"width": 8.0, "thickness": 2.3},
    "M6": {"width": 10.0, "thickness": 2.72},
    "M8": {"width": 13.0, "thickness": 3.52},
}


@dataclass
class MScrew:
    size: str
    nut_size: float
    cap_screw_size: float
    cap_screw_head_size: float
    grub_screw_wrench_size: float
    clearance_hole_close: float
    clearance_hole_normal: float
    clearance_hole_loose: float
    pitch: float
    core_hole: float
    nut_thickness: float
    nut_circle_diameter: float
    cylinder_head_diameter: float
    cylinder_head_height: float
    min_thread_length: float
    wrench_socket_outer_diameter: float
    thread_inset_hole_diameter: Optional[float] = None
    thread_inset_length: Optional[float] = None
    conical_head_diameter: Optional[float] = None
    conical_head_height: Optional[float] = None

    @staticmethod
    def from_size(size: str) -> "MScrew":
        """Create an MScrew instance from the global screw table, safely handling optional fields."""
        if size not in m_screws_table:
            raise KeyError(f"Unsupported screw size: {size}")
        specs = m_screws_table[size].copy()
        # Fill missing optional fields
        specs.setdefault("thread_inset_hole_diameter", None)
        specs.setdefault("thread_inset_length", None)
        specs.setdefault("conical_head_diameter", None)
        specs.setdefault("conical_head_height", None)
        return MScrew(size=size, **specs)

    def get_clearance_hole_diameter(self, clearance_type="normal") -> float:
        """Get the clearance hole diameter for this screw based on the specified clearance type."""
        clearance_key = f"clearance_hole_{clearance_type}"
        if not hasattr(self, clearance_key):
            raise ValueError(
                f"Invalid clearance type: {clearance_type}. Must be 'close', 'normal', or 'loose'"
            )
        return getattr(self, clearance_key)


def get_nut_outer_diameter(size):
    """
    Get the diameter of the nut corners for a given size.

    Args:
        size: The size as a string, e.g. "M3", "M4", etc.

    Returns:
        float: Distance between opposite corners of the hexagonal nut

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    nut_size = m_screws_table[size]["nut_size"]
    # The nut is a hexagon, and the size is given as the distance between two opposite sides
    # We need to calculate the distance between two opposite corners
    nut_outer_circle_diameter = nut_size / math.cos(math.radians(30))
    return nut_outer_circle_diameter


def create_nut(size, height=None, slack=None, no_hole=False):
    """
    Create a hexagonal nut for the specified screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        height: Height of the nut (defaults to standard thickness)
        slack: Additional clearance to add to nut dimensions
        no_hole: If True, creates a solid hexagon without the center hole

    Returns:
        Solid: CAD solid representing the nut

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    nut_size = m_screws_table[size]["nut_size"]
    # The nut is a hexagon, and the size is given as the distance between two opposite sides
    # We need to calculate the distance between two opposite corners
    nut_size = nut_size / math.cos(math.radians(30))

    if slack is not None:
        nut_size += slack

    if height is None:
        height = m_screws_table[size]["nut_thickness"]

    # Create hexagonal points
    points = []
    for i in range(6):
        angle = i * math.pi / 3
        x = nut_size * 0.5 * math.cos(angle)
        y = nut_size * 0.5 * math.sin(angle)
        points.append((x, y))

    nut = create_extruded_polygon(points, thickness=height)

    if no_hole:
        return nut

    # Create a hole in the middle
    nut_hole_diameter = m_screws_table[size]["clearance_hole_normal"]
    nut_hole = create_cylinder(nut_hole_diameter / 2, height)
    nut = cut_parts(nut, nut_hole)

    return nut


def create_square_nut(size, height=None, slack=None, no_hole=False):
    """Create a DIN 562 low square nut (Vierkantmutter).

    Args:
        size: Metric thread size from M2 through M8 (including M2.5).
        height: Nut height; defaults to the DIN 562 minimum thickness.
        slack: Clearance added to the overall width for printable fits.
        no_hole: Return a solid square when true.

    Returns:
        Solid: CAD solid representing the square nut.

    Raises:
        KeyError: If no DIN 562 square-nut dimensions are available for ``size``.
    """
    if size not in square_nuts_table:
        raise KeyError(f"Unsupported square nut size: {size}")

    specification = square_nuts_table[size]
    width = specification["width"] + (slack or 0)
    if height is None:
        height = specification["thickness"]

    half_width = width / 2
    nut = create_extruded_polygon(
        [
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width),
        ],
        thickness=height,
    )
    if no_hole:
        return nut

    hole_diameter = m_screws_table[size]["clearance_hole_normal"]
    return cut_parts(nut, create_cylinder(hole_diameter / 2, height))


def create_bolt_thread(size, length, enlargement=0, cutter=False):
    """
    Create a bolt thread for the specified screw size using trapezoidal snake geometry.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        length: Length of the threaded section
        enlargement: Additional radius to add/subtract for fit adjustment
        cutter: If True, creates a cutting thread with different dimensions

    Returns:
        Solid: CAD solid representing the bolt thread

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    pitch = m_screws_table[size]["pitch"]

    major_diameter = float(size[1:])
    outer_radius = major_diameter / 2

    H = 0.8660 * pitch

    inner_radius = outer_radius - 5 * H / 8 + enlargement
    outer_thickness = pitch / 8
    inner_thickness = 3 * pitch / 4

    if cutter:
        outer_thickness = 1e-3
        outer_radius = major_diameter / 2 + H / 8

    outer_radius += enlargement

    thread = create_screw_thread(
        pitch,
        inner_radius,
        outer_radius,
        outer_thickness,
        length / pitch,
        inner_thickness=inner_thickness,
    )

    return thread


def create_self_threading_hole_cutter(
    size,
    length,
    clearance_type="close",
    start_angle=90.0,
    core_radius_adjustment=0.0,
    lead_in=False,
):
    """
    Create a tri-lobed self-threading hole cutter for the specified screw size.

    The cutter starts as a clearance-hole cylinder, then subtracts three broad
    cylindrical bites from the drill shape. The deepest point of each bite lands
    on the adjusted screw core-hole radius, leaving three self-threading contact
    lobes.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        length: Length of the cutter
        clearance_type: Type of clearance ("close", "normal", or "loose")
        start_angle: Rotation angle in degrees for the first reduced-radius lobe
        core_radius_adjustment: Absolute radius adjustment applied to the
            screw-table core-hole radius. Negative values make the valleys
            tighter.
        lead_in: If True, add a 45-degree conical lead-in at the local +Z end

    Returns:
        Solid: CAD solid representing the self-threading hole cutter

    Raises:
        KeyError: If the screw size is not supported
        ValueError: If the clearance type or adjusted core radius is not valid
    """

    clearance_radius = get_clearance_hole_diameter(size, clearance_type) / 2
    core_radius = get_core_hole_diameter(size) / 2
    adjusted_core_radius = core_radius + core_radius_adjustment

    if adjusted_core_radius <= 0 or adjusted_core_radius >= clearance_radius:
        raise ValueError(
            "Adjusted core radius must be greater than 0 and smaller than the "
            "clearance radius"
        )

    cutter = create_cylinder(clearance_radius, length)
    bite_radius = clearance_radius
    bite_center_radius = adjusted_core_radius + bite_radius
    bite_margin = max(length * 0.02, 0.1)

    for index in range(3):
        angle = math.radians(start_angle + index * 120)
        bite = create_cylinder(
            bite_radius,
            length + bite_margin,
            origin=(
                bite_center_radius * math.cos(angle),
                bite_center_radius * math.sin(angle),
                -bite_margin / 2,
            ),
        )
        cutter = cut_parts(cutter, bite)

    if lead_in:
        lead_in_height = clearance_radius - adjusted_core_radius
        if lead_in_height > length:
            raise ValueError("Lead-in cone cannot be longer than the cutter")

        lead_in_cone = create_cone(
            adjusted_core_radius,
            clearance_radius,
            lead_in_height,
            origin=(0, 0, length - lead_in_height),
        )
        cutter = fuse_parts(cutter, lead_in_cone)

    return cutter


def create_cylinder_screw(
    size, length, with_thread=False, only_minimal_thread=True, enlargement=0
):
    """
    Create a cylinder head screw for the specified size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        length: Length of the screw shaft
        with_thread: If True, creates actual threaded geometry
        only_minimal_thread: If True, only creates minimal thread length needed
        enlargement: Additional diameter to add for fit adjustment

    Returns:
        Solid: CAD solid representing the cylinder screw

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    thread_outer_diameter = float(size[1:]) + enlargement * 2

    if with_thread:
        if only_minimal_thread:
            thread_length = min(length, m_screws_table[size]["min_thread_length"])
            thread = create_bolt_thread(size, thread_length, cutter=True)
            thread_cylinder = create_cylinder(
                thread_outer_diameter / 2 + enlargement,
                length - thread_length + enlargement,
            )
            # Stack the cylinder on top of the thread
            thread_cylinder = translate(0, 0, thread_length)(thread_cylinder)
            thread = fuse_parts(thread, thread_cylinder)
        else:
            thread_length = length
            thread = create_bolt_thread(size, thread_length, cutter=True)
    else:
        thread = create_cylinder(thread_outer_diameter / 2, length)

    # Cylinder head
    cylinder_head_diameter = (
        m_screws_table[size]["cylinder_head_diameter"] + enlargement * 2
    )
    cylinder_head_height = m_screws_table[size]["cylinder_head_height"] + enlargement

    cylinder_head = create_cylinder(cylinder_head_diameter / 2, cylinder_head_height)
    # Position head on top of thread
    cylinder_head = translate(0, 0, length)(cylinder_head)

    retval = fuse_parts(thread, cylinder_head)
    return retval


def create_conical_head_screw(
    size, length, with_thread=False, only_minimal_thread=True, enlargement=0
):
    """
    Create a conical head screw for the specified size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        length: Length of the screw shaft
        with_thread: If True, creates actual threaded geometry
        only_minimal_thread: If True, only creates minimal thread length needed
        enlargement: Additional diameter to add for fit adjustment

    Returns:
        Solid: CAD solid representing the conical head screw

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    conical_head_diameter_base = m_screws_table[size].get("conical_head_diameter")
    conical_head_height_base = m_screws_table[size].get("conical_head_height")
    if conical_head_diameter_base is None or conical_head_height_base is None:
        raise ValueError(
            f"Conical head dimensions are not defined for screw size: {size}"
        )

    thread_outer_diameter = float(size[1:]) + enlargement * 2

    if with_thread:
        if only_minimal_thread:
            thread_length = min(length, m_screws_table[size]["min_thread_length"])
            thread = create_bolt_thread(size, thread_length, cutter=True)
            thread_cylinder = create_cylinder(
                thread_outer_diameter / 2 + enlargement,
                length - thread_length + enlargement,
            )
            # Stack the cylinder on top of the thread
            thread_cylinder = translate(0, 0, thread_length)(thread_cylinder)
            thread = fuse_parts(thread, thread_cylinder)
        else:
            thread_length = length
            thread = create_bolt_thread(size, thread_length, cutter=True)
    else:
        thread = create_cylinder(thread_outer_diameter / 2, length)

    # Conical head
    conical_head_diameter = conical_head_diameter_base + enlargement * 2
    conical_head_height = conical_head_height_base + enlargement

    conical_head = create_cone(
        radius1=thread_outer_diameter / 2 + enlargement,
        radius2=conical_head_diameter / 2,
        height=conical_head_height,
    )
    # Position head flush with the top of the thread - conical head screws are typically countersunk
    conical_head = translate(0, 0, length - conical_head_height)(conical_head)

    retval = fuse_parts(thread, conical_head)
    return retval


def get_clearance_hole_diameter(size, clearance_type="normal"):
    """
    Get the clearance hole diameter for a given screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        clearance_type: Type of clearance ("close", "normal", or "loose")

    Returns:
        float: Clearance hole diameter

    Raises:
        KeyError: If the screw size is not supported
        ValueError: If the clearance type is not valid
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    clearance_key = f"clearance_hole_{clearance_type}"
    if clearance_key not in m_screws_table[size]:
        raise ValueError(
            f"Invalid clearance type: {clearance_type}. Must be 'close', 'normal', or 'loose'"
        )

    return m_screws_table[size][clearance_key]


def get_core_hole_diameter(size):
    """
    Get the core hole diameter for threading a given screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)

    Returns:
        float: Core hole diameter for threading

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    return m_screws_table[size]["core_hole"]


def get_thread_pitch(size):
    """
    Get the thread pitch for a given screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)

    Returns:
        float: Thread pitch in millimeters

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    return m_screws_table[size]["pitch"]


def list_supported_sizes():
    """
    Get a list of all supported screw sizes.

    Returns:
        list: List of supported screw size strings
    """
    return list(m_screws_table.keys())


def get_screw_info(size):
    """
    Get complete specification information for a screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)

    Returns:
        dict: Complete specification dictionary for the screw size

    Raises:
        KeyError: If the screw size is not supported
    """
    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    return m_screws_table[size].copy()


def create_hidden_nut_pocket_cutter(
    size,
    nut_height=None,
    bottom_cutter_length=None,
    top_cutter_length=500,
    slack=0.2,
    clearance_hole_diameter=None,
    square_nut=False,
):
    """
    Create a cutter solid for a hidden nut holder for the specified screw size.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        nut_height: Height of the nut pocket
        slack: Additional clearance to add to nut dimensions
        bottom_cutter_length: Length of the bottom cutter section
        top_cutter_length: Length of the top cutter section
        clearance_hole_diameter: Diameter of the clearance hole (defaults to standard)
        square_nut: Use a DIN 562 square nut instead of a hexagonal nut

    Returns:
        A LeaderFollowersCuttersPart which has the nut pocket cutter as leader for easy alignment with a screw, hole or other cylindrical part. It contains a cutter, that it can be used with use_as_cutter_on


    """

    screw_spec = MScrew.from_size(size)

    if clearance_hole_diameter is None:
        clearance_hole_diameter = screw_spec.clearance_hole_normal

    if square_nut:
        if size not in square_nuts_table:
            raise KeyError(f"Unsupported square nut size: {size}")
        nut_width = square_nuts_table[size]["width"]
        default_nut_height = square_nuts_table[size]["thickness"]
    else:
        nut_width = screw_spec.nut_size
        default_nut_height = screw_spec.nut_thickness

    if nut_height is None:
        nut_height = default_nut_height + slack * 2

    if square_nut:
        nut = create_square_nut(size, height=nut_height, slack=slack, no_hole=True)
        insertion_depth = (nut_width + slack) / 2
    else:
        nut = create_nut(size, height=nut_height, slack=slack, no_hole=True)
        nut = rotate(30)(nut)  # align the hexagon's flat sides with the slit
        insertion_depth = screw_spec.nut_circle_diameter / 2

    overall_cutter = nut

    if bottom_cutter_length is not None:
        bottom_cutter = create_cylinder(
            clearance_hole_diameter / 2, bottom_cutter_length
        )
        bottom_cutter = align(bottom_cutter, nut, Alignment.CENTER)
        bottom_cutter = align(bottom_cutter, nut, Alignment.STACK_BOTTOM)
        overall_cutter = overall_cutter.fuse(bottom_cutter)

    # create the top cutter section
    top_cutter = create_cylinder(clearance_hole_diameter / 2, top_cutter_length)
    top_cutter = align(top_cutter, nut, Alignment.CENTER)
    top_cutter = align(top_cutter, nut, Alignment.STACK_TOP)
    overall_cutter = overall_cutter.fuse(top_cutter)

    # create the nut insertion slit cutter

    nut_slit_cutter = create_box(nut_width + slack, 500, nut_height)

    nut_slit_cutter = align(nut_slit_cutter, nut, Alignment.CENTER)
    nut_slit_cutter = align(
        nut_slit_cutter,
        nut,
        Alignment.STACK_BACK,
        stack_gap=-insertion_depth,
    )
    overall_cutter = overall_cutter.fuse(nut_slit_cutter)

    retval = LeaderFollowersCuttersPart(nut, cutters=[overall_cutter])

    return retval


def create_thread_inset_assembly(
    size,
    thickness,
    extra_radius=2,
    clearance_type="normal",
    thread_inset_hole_radius_adjustment=0.0,
):
    """
    Create an assembly for a thread inset for the specified screw size.


    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        thickness: Thickness of the material to be threaded
        extra_radius: Additional radius to add around the thread inset hole for the material around the inset
        clearance_type: Type of clearance hole to create if thickness exceeds thread inset length ("close",
    Returns:
        A LeaderFollowersCuttersPart representing the thread inset assembly, with the main body as the leader and the cutters for the thread inset hole and optional clearance hole.

    Raises:
        ValueError: If the chosen thickness is less than the required thread inset length for the specified screw size


    Example:
        # Create a thread inset assembly for an M4 screw in a 5mm thick material with normal clearance
        my_part = create_box(50,50,20)

        thread_inset_assembly = create_thread_inset_assembly("M3", thickness=5, clearance_type="normal")

        thread_inset_assembly = align(thread_inset_assembly, my_part, Alignment.CENTER)
        thread_inset_assembly = align(thread_inset_assembly, my_part, Alignment.BOTTOM)
        my_part = thread_inset_assembly.use_as_cutter_on(my_part)
        my_part = my_part.fuse(thread_inset_assembly)


    """

    thread_inset_length = m_screws_table[size].get("thread_inset_length")
    if thread_inset_length is None:
        raise ValueError(f"Thread inset is not defined for screw size: {size}")
    if thickness < thread_inset_length:
        raise ValueError(
            f"Thickness {thickness} is too small for thread inset of size {size}. Minimum required is {thread_inset_length}."
        )

    thread_inset_hole_diameter = m_screws_table[size].get("thread_inset_hole_diameter")
    if thread_inset_hole_diameter is None:
        raise ValueError(
            f"Thread inset hole diameter is not defined for screw size: {size}"
        )

    inset_assembly_diameter = thread_inset_hole_diameter + extra_radius * 2

    inset_assembly_cylinder = create_cylinder(inset_assembly_diameter / 2, thickness)
    inset_assembly_leader = inset_assembly_cylinder

    if thickness > thread_inset_length:
        clearance_hole_cutter = create_cylinder(
            get_clearance_hole_diameter(size, clearance_type) / 2, thickness
        )
        clearance_hole_cutter = align(
            clearance_hole_cutter, inset_assembly_cylinder, Alignment.CENTER
        )
        inset_assembly_leader = inset_assembly_leader.cut(clearance_hole_cutter)

    thread_inset_cutter = create_cylinder(
        thread_inset_hole_diameter / 2 + thread_inset_hole_radius_adjustment,
        thread_inset_length,
    )
    thread_inset_cutter = align(
        thread_inset_cutter, inset_assembly_cylinder, Alignment.CENTER
    )
    thread_inset_cutter = align(
        thread_inset_cutter, inset_assembly_cylinder, Alignment.BOTTOM
    )

    inset_assembly_leader = inset_assembly_leader.cut(thread_inset_cutter)

    thread_inset = create_cylinder(thread_inset_hole_diameter / 2, thread_inset_length)
    thread_inset = align(thread_inset, thread_inset_cutter, Alignment.CENTER)

    thread_inset_core_hole_cutter = create_cylinder(
        get_core_hole_diameter(size) / 2, thread_inset_length
    )
    thread_inset_core_hole_cutter = align(
        thread_inset_core_hole_cutter, thread_inset, Alignment.CENTER
    )
    thread_inset = thread_inset.cut(thread_inset_core_hole_cutter)

    retval = LeaderFollowersCuttersPart(inset_assembly_leader)

    retval.add_named_cutter(inset_assembly_cylinder, "assembly_cutter")
    retval.add_named_non_production_part(thread_inset, "thread_inset")

    return retval


class ScrewType(str, Enum):
    """Supported screw-head geometries for complete screw assemblies."""

    CYLINDER_HEAD = "cylinder_head"
    CONICAL_HEAD = "conical_head"


class HoleType(str, Enum):
    """Supported thread-side hole geometries for complete screw assemblies."""

    CLEARANCE = "clearance"
    CORE = "core"
    SELF_THREADING = "self_threading"


def create_complete_screw_assembly(
    size,
    length,
    screw_type=ScrewType.CYLINDER_HEAD,
    with_thread=False,
    hole_type=HoleType.CLEARANCE,
    clearance_type="normal",
    core_radius_adjustment=0.0,
    lead_in=False,
    hole_distance_from_head=0.0,
    extra_hole_length=0.0,
    with_access_hole=False,
    extra_access_hole_length=None,
    access_hole_clearance=0.1,
):
    """
    Create an alignable screw with its thread-side and optional access cutters.

    The headless screw shaft is the assembly leader. Aligning or rotating the
    returned assembly therefore moves the screw, cutters, and reference parts as
    one unit while keeping the head from influencing placement.

    Args:
        size: Screw size string (e.g., "M3", "M4", etc.)
        length: Length of the screw shaft
        screw_type: Type of screw head ("cylinder_head" or "conical_head")
        with_thread: If True, creates actual threaded geometry
        hole_type: Thread-side hole type ("clearance", "core", or
            "self_threading")
        clearance_type: Clearance class used by clearance and self-threading holes
        core_radius_adjustment: Radius adjustment for self-threading-hole valleys
        lead_in: Add a conical lead-in at the head-side end of a self-threading hole
        hole_distance_from_head: Uncut distance between the head and the
            head-side end of the thread-side hole
        extra_hole_length: Distance the thread-side cutter extends beyond the tip
        with_access_hole: Add a head-diameter access cutter above the shaft
        extra_access_hole_length: Access length in addition to the head height
        access_hole_clearance: Radial clearance around the screw head

    Returns:
        LeaderFollowersCuttersPart with the headless shaft as leader. Named
        cutters are ``thread_side_hole_cutter`` and, when requested,
        ``access_hole_cutter``. Named reference parts are ``complete_screw`` and
        ``screw_head``.

    Raises:
        KeyError: If the screw size is not supported
        ValueError: If an enum value or calculated cutter length is invalid
    """

    if size not in m_screws_table:
        raise KeyError(f"Unsupported screw size: {size}")

    if hole_distance_from_head < 0:
        raise ValueError("hole_distance_from_head must be greater than or equal to 0")

    screw_spec = MScrew.from_size(size)

    if screw_type == ScrewType.CYLINDER_HEAD:
        main_screw = create_cylinder_screw(size, length, with_thread=with_thread)
        head_height = screw_spec.cylinder_head_height
        head_radius = screw_spec.cylinder_head_diameter / 2 + access_hole_clearance
    elif screw_type == ScrewType.CONICAL_HEAD:
        main_screw = create_conical_head_screw(size, length, with_thread=with_thread)
        head_height = screw_spec.conical_head_height
        head_radius = screw_spec.conical_head_diameter / 2 + access_hole_clearance
    else:
        raise ValueError(
            f"Invalid screw type: {screw_type}. Must be 'cylinder_head' or "
            "'conical_head'"
        )

    head_cutter = create_cylinder(head_radius, head_height)
    head_cutter = align(head_cutter, main_screw, Alignment.CENTER, axes=[0, 1])
    head_cutter = align(head_cutter, main_screw, Alignment.TOP)
    leader = main_screw.cut(head_cutter)

    leader_length = get_zlen(get_bounding_box(leader))
    effective_hole_length = leader_length - hole_distance_from_head + extra_hole_length
    if effective_hole_length <= 0:
        raise ValueError(
            "Thread-side hole length must be positive after applying "
            "hole_distance_from_head and extra_hole_length"
        )

    if hole_type == HoleType.CLEARANCE:
        hole_diameter = screw_spec.get_clearance_hole_diameter(clearance_type)
        hole_cutter = create_cylinder(hole_diameter / 2, effective_hole_length)
    elif hole_type == HoleType.CORE:
        hole_cutter = create_cylinder(screw_spec.core_hole / 2, effective_hole_length)
    elif hole_type == HoleType.SELF_THREADING:
        hole_cutter = create_self_threading_hole_cutter(
            size,
            effective_hole_length,
            clearance_type=clearance_type,
            core_radius_adjustment=core_radius_adjustment,
            lead_in=lead_in,
        )
    else:
        raise ValueError(
            f"Invalid hole type: {hole_type}. Must be 'clearance', 'core', or "
            "'self_threading'"
        )

    hole_cutter = align(hole_cutter, leader, Alignment.CENTER, axes=[0, 1])
    hole_cutter = align(hole_cutter, leader, Alignment.TOP)
    if hole_distance_from_head:
        hole_cutter = translate(0, 0, -hole_distance_from_head)(hole_cutter)

    access_hole_cutter = None
    if with_access_hole:
        extra_access_length = (
            extra_access_hole_length if extra_access_hole_length is not None else 0.0
        )
        access_hole_length = head_height + extra_access_length
        if access_hole_length <= 0:
            raise ValueError(
                "Access hole length must be positive after applying "
                "extra_access_hole_length"
            )

        access_hole_cutter = create_cylinder(head_radius, access_hole_length)
        access_hole_cutter = align(
            access_hole_cutter, leader, Alignment.CENTER, axes=[0, 1]
        )
        access_hole_cutter = align(access_hole_cutter, leader, Alignment.STACK_TOP)

    assembly = LeaderFollowersCuttersPart(leader)
    assembly.add_named_cutter(hole_cutter, "thread_side_hole_cutter")

    if access_hole_cutter is not None:
        assembly.add_named_cutter(access_hole_cutter, "access_hole_cutter")

    assembly.add_named_non_production_part(main_screw, "complete_screw")
    screw_head_only = main_screw.cut(leader)
    assembly.add_named_non_production_part(screw_head_only, "screw_head")

    return assembly
