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

from shellforgepy.adapters._adapter import (
    create_basic_cylinder,
    create_extruded_polygon,
    cut_parts,
    fuse_parts,
)
from shellforgepy.construct.alignment_operations import translate
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
        "nut_thickness": 1.6,
        "cylinder_head_diameter": 3.8,
        "cylinder_head_height": 2,
        "min_thread_length": 16,
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
        "min_thread_length": 18,
        "thread_inset_hole_diameter": 4.3,
        "thread_inset_length": 6,
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
        "wrench_socket_outer_diameter": 12.5,
        "min_thread_length": 20,
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
        "wrench_socket_outer_diameter": 13,
        "min_thread_length": 22,
        "thread_inset_hole_diameter": 6.2,
        "thread_inset_length": 8,
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
        "min_thread_length": 24,
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
        "min_thread_length": 28,
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
        "min_thread_length": 32,
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
        "min_thread_length": 36,
    },
}


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
    nut_hole = create_basic_cylinder(nut_hole_diameter / 2, height)
    nut = cut_parts(nut, nut_hole)

    return nut


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

    thread_outer_diameter = int(size[1:]) + enlargement * 2

    if with_thread:
        if only_minimal_thread:
            thread_length = min(length, m_screws_table[size]["min_thread_length"])
            thread = create_bolt_thread(size, thread_length, cutter=True)
            thread_cylinder = create_basic_cylinder(
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
        thread = create_basic_cylinder(thread_outer_diameter / 2, length)

    # Cylinder head
    cylinder_head_diameter = (
        m_screws_table[size]["cylinder_head_diameter"] + enlargement * 2
    )
    cylinder_head_height = m_screws_table[size]["cylinder_head_height"] + enlargement

    cylinder_head = create_basic_cylinder(
        cylinder_head_diameter / 2, cylinder_head_height
    )
    # Position head on top of thread
    cylinder_head = translate(0, 0, length)(cylinder_head)

    retval = fuse_parts(thread, cylinder_head)
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
