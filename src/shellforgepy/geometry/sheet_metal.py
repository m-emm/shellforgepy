"""Sheet metal geometry utilities for ShellForgePy.

This module provides functions for creating common sheet metal geometries
like bends and walls, useful for fabrication-oriented designs.
"""

import logging

from shellforgepy.adapters._adapter import create_box, create_cylinder
from shellforgepy.construct.alignment_operations import rotate, translate

_logger = logging.getLogger(__name__)


def create_sheet_metal_bend(
    thickness: float,
    length: float,
    bend_angle: float = 90.0,
    inner_radius: float = None,
) -> object:
    """Create a sheet metal bend geometry.

    Creates a 90° (or custom angle) cylindrical bend suitable for sheet metal work.
    The bend has an inner radius equal to the thickness and outer radius of 2*thickness
    by default, which is typical for sheet metal bending.

    Args:
        thickness: Material thickness (also determines inner radius if not specified)
        length: Length of the bend along the bend axis
        bend_angle: Bend angle in degrees (default 90°)
        inner_radius: Inner radius of bend (defaults to thickness)

    Returns:
        CAD solid representing the bent section

    Example:
        >>> bend = create_sheet_metal_bend(thickness=2.0, length=50.0)
        >>> # Creates a 90° bend with 2mm thickness, 50mm length
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if length <= 0:
        raise ValueError("Length must be positive")
    if bend_angle <= 0 or bend_angle > 180:
        raise ValueError("Bend angle must be between 0 and 180 degrees")

    if inner_radius is None:
        inner_radius = thickness

    if inner_radius <= 0:
        raise ValueError("Inner radius must be positive")

    outer_radius = inner_radius + thickness

    # Create outer cylinder section
    outer = create_cylinder(
        radius=outer_radius,
        height=length,
        origin=(0, 0, 0),
        direction=(0, 1, 0),
        angle=bend_angle,
    )

    # Create inner cylinder to subtract
    inner = create_cylinder(
        radius=inner_radius,
        height=length,
        origin=(0, 0, 0),
        direction=(0, 1, 0),
        angle=bend_angle,
    )

    # Cut inner from outer to create the bend
    bend = outer.cut(inner)

    # Rotate the bend to match original orientation (180° around Y axis)
    # This positions it so it can be naturally connected to flat sections
    bend = rotate(180, center=(0, 0, inner_radius), axis=(0, 1, 0))(bend)

    return bend


def create_sheet_metal_wall(
    thickness: float,
    length: float,
    height: float,
    with_bend: bool = True,
    bend_angle: float = 90.0,
) -> object:
    """Create a sheet metal wall with optional bend at the bottom.

    Creates a wall structure consisting of a vertical flat section and an optional
    horizontal bend at the bottom. This is common in sheet metal fabrication
    for creating flanges, mounting tabs, or structural elements.

    Args:
        thickness: Material thickness
        length: Length of the wall (along the bend axis)
        height: Height of the vertical wall section
        with_bend: Whether to include the bend at the bottom (default True)
        bend_angle: Angle of the bend in degrees (default 90°)

    Returns:
        CAD solid representing the wall with optional bend

    Example:
        >>> wall = create_sheet_metal_wall(thickness=1.5, length=100, height=25)
        >>> # Creates a 25mm high wall with 90° bend at bottom
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if length <= 0:
        raise ValueError("Length must be positive")
    if height <= 0:
        raise ValueError("Height must be positive")
    if bend_angle <= 0 or bend_angle > 180:
        raise ValueError("Bend angle must be between 0 and 180 degrees")

    # Create the vertical wall section
    wall = create_box(thickness, length, height)

    # Position the wall properly - move it to align with bend coordinate system
    wall = translate(-2 * thickness, 0, 2 * thickness)(wall)

    if with_bend:
        # Create the bend section
        bend_object = create_sheet_metal_bend(
            thickness=thickness, length=length, bend_angle=bend_angle
        )

        # Fuse wall and bend
        result = wall.fuse(bend_object)
    else:
        result = wall

    # Rotate the entire assembly to standard orientation
    result = rotate(90, center=(0, 0, 0), axis=(0, 0, 1))(result)

    # Translate to final position
    result = translate(length, 0, 0)(result)

    return result


def create_sheet_metal_bracket(
    thickness: float,
    width: float,
    height: float,
    flange_width: float,
    bend_relief: float = 0.0,
) -> object:
    """Create an L-shaped sheet metal bracket.

    Creates a simple L-bracket with vertical and horizontal sections,
    commonly used for mounting and structural applications.

    Args:
        thickness: Material thickness
        width: Width of both vertical and horizontal sections
        height: Height of the vertical section
        flange_width: Width of the horizontal flange
        bend_relief: Optional bend relief radius (default 0)

    Returns:
        CAD solid representing the L-bracket

    Example:
        >>> bracket = create_sheet_metal_bracket(
        ...     thickness=2.0, width=50, height=30, flange_width=20
        ... )
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if width <= 0:
        raise ValueError("Width must be positive")
    if height <= 0:
        raise ValueError("Height must be positive")
    if flange_width <= 0:
        raise ValueError("Flange width must be positive")
    if bend_relief < 0:
        raise ValueError("Bend relief must be non-negative")

    # Create vertical section
    vertical = create_box(thickness, width, height)

    # Create horizontal flange
    horizontal = create_box(flange_width, width, thickness)

    # Position horizontal section at bottom of vertical, extending outward
    horizontal = translate(thickness, 0, -thickness)(horizontal)

    # Fuse the sections
    bracket = vertical.fuse(horizontal)

    # Add bend relief if specified
    if bend_relief > 0:
        # Create cylindrical relief at the inside corner
        relief = create_cylinder(
            radius=bend_relief,
            height=width,
            origin=(thickness, 0, 0),
            direction=(0, 1, 0),
        )

        # Cut the relief from the bracket
        bracket = bracket.cut(relief)

    return bracket


def create_sheet_metal_hem(
    thickness: float,
    length: float,
    hem_width: float,
    hem_type: str = "open",
) -> object:
    """Create a hemmed edge for sheet metal.

    Creates a folded edge (hem) which is commonly used to strengthen edges
    and eliminate sharp edges in sheet metal work.

    Args:
        thickness: Material thickness
        length: Length of the hem
        hem_width: Width of the folded section
        hem_type: Type of hem - "open", "closed", or "teardrop"

    Returns:
        CAD solid representing the hemmed edge

    Example:
        >>> hem = create_sheet_metal_hem(
        ...     thickness=1.0, length=100, hem_width=5, hem_type="closed"
        ... )
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if length <= 0:
        raise ValueError("Length must be positive")
    if hem_width <= 0:
        raise ValueError("Hem width must be positive")
    if hem_type not in ["open", "closed", "teardrop"]:
        raise ValueError("Hem type must be 'open', 'closed', or 'teardrop'")

    # Start with base flat section
    base = create_box(hem_width + thickness, length, thickness)

    if hem_type == "open":
        # Simple 180° fold with gap
        fold = create_box(thickness, length, hem_width - thickness)
        fold = translate(0, 0, thickness)(fold)
        result = base.fuse(fold)

    elif hem_type == "closed":
        # 180° fold that touches the base
        fold = create_box(thickness, length, hem_width)
        fold = translate(thickness, 0, thickness)(fold)

        # Create bend section
        bend = create_sheet_metal_bend(thickness, length, 180.0, thickness / 2)
        bend = translate(hem_width, 0, thickness)(bend)
        bend = rotate(90, center=(hem_width, 0, thickness), axis=(0, 1, 0))(bend)

        result = base.fuse(fold).fuse(bend)

    else:  # teardrop
        # Rounded hem with teardrop shape
        radius = thickness

        # Main fold section
        fold = create_box(thickness, length, hem_width - radius)
        fold = translate(thickness, 0, thickness)(fold)

        # Rounded end
        rounded_end = create_cylinder(
            radius=radius,
            height=length,
            origin=(thickness + radius, 0, thickness + hem_width - radius),
            direction=(0, 1, 0),
            angle=180,
        )

        result = base.fuse(fold).fuse(rounded_end)

    return result
