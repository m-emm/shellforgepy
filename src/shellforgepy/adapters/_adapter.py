from typing import Optional

from shellforgepy.adapters._adapter_bridge import (
    apply_fillet_by_alignment as adapter_apply_fillet_by_alignment,
)
from shellforgepy.adapters._adapter_bridge import (
    apply_fillet_to_edges as adapter_apply_fillet_to_edges,
)
from shellforgepy.adapters._adapter_bridge import copy_part as adapter_copy_part
from shellforgepy.adapters._adapter_bridge import create_box as adapter_create_box
from shellforgepy.adapters._adapter_bridge import create_cone as adapter_create_cone
from shellforgepy.adapters._adapter_bridge import (
    create_cylinder as adapter_create_cylinder,
)
from shellforgepy.adapters._adapter_bridge import (
    create_extruded_polygon as adapter_create_extruded_polygon,
)
from shellforgepy.adapters._adapter_bridge import (
    create_filleted_box as adapter_create_filleted_box,
)
from shellforgepy.adapters._adapter_bridge import (
    create_solid_from_traditional_face_vertex_maps as adapter_create_solid_from_traditional_face_vertex_maps,
)
from shellforgepy.adapters._adapter_bridge import create_sphere as adapter_create_sphere
from shellforgepy.adapters._adapter_bridge import (
    create_text_object as adapter_create_text_object,
)
from shellforgepy.adapters._adapter_bridge import cut_parts as adapter_cut_parts
from shellforgepy.adapters._adapter_bridge import (
    deserialize_structured_step as adapter_deserialize_structured_step,
)
from shellforgepy.adapters._adapter_bridge import (
    export_colored_parts_to_obj as adapter_export_colored_parts_to_obj,
)
from shellforgepy.adapters._adapter_bridge import (
    export_solid_to_obj as adapter_export_solid_to_obj,
)
from shellforgepy.adapters._adapter_bridge import (
    export_solid_to_step as adapter_export_solid_to_step,
)
from shellforgepy.adapters._adapter_bridge import (
    export_solid_to_stl as adapter_export_solid_to_stl,
)
from shellforgepy.adapters._adapter_bridge import (
    export_structured_step as adapter_export_structured_step,
)
from shellforgepy.adapters._adapter_bridge import (
    filter_edges_by_function as adapter_filter_edges_by_function,
)
from shellforgepy.adapters._adapter_bridge import fuse_parts as adapter_fuse_parts
from shellforgepy.adapters._adapter_bridge import (
    get_adapter_id as adapter_get_adapter_id,
)
from shellforgepy.adapters._adapter_bridge import (
    get_bounding_box as adapter_get_bounding_box,
)
from shellforgepy.adapters._adapter_bridge import (
    get_bounding_box_center as adapter_get_bounding_box_center,
)
from shellforgepy.adapters._adapter_bridge import (
    get_bounding_box_size as adapter_get_bounding_box_size,
)
from shellforgepy.adapters._adapter_bridge import (
    get_vertex_coordinates as adapter_get_vertex_coordinates,
)
from shellforgepy.adapters._adapter_bridge import (
    get_vertex_coordinates_np as adapter_get_vertex_coordinates_np,
)
from shellforgepy.adapters._adapter_bridge import get_vertices as adapter_get_vertices
from shellforgepy.adapters._adapter_bridge import get_volume as adapter_get_volume
from shellforgepy.adapters._adapter_bridge import (
    import_solid_from_step as adapter_import_solid_from_step,
)
from shellforgepy.adapters._adapter_bridge import mirror_part as adapter_mirror_part
from shellforgepy.adapters._adapter_bridge import (
    mirror_part_native as adapter_mirror_part_native,
)
from shellforgepy.adapters._adapter_bridge import rotate_part as adapter_rotate_part
from shellforgepy.adapters._adapter_bridge import (
    rotate_part_native as adapter_rotate_part_native,
)
from shellforgepy.adapters._adapter_bridge import scale_part as adapter_scale_part
from shellforgepy.adapters._adapter_bridge import (
    scale_part_native as adapter_scale_part_native,
)
from shellforgepy.adapters._adapter_bridge import (
    translate_part as adapter_translate_part,
)
from shellforgepy.adapters._adapter_bridge import (
    translate_part_native as adapter_translate_part_native,
)

"""Adapter functions that wrap the selected CAD backend. 
These functions should not contain any backend-specific code. They document the  api that users of shellforgepy should use, independent of the CAD backend.
"""


def create_box(length, width, height, origin=(0.0, 0.0, 0.0)):
    """Create a basic box solid.

    Args:
        length: Length of the box along the X-axis.
        width: Width of the box along the Y-axis.
        height: Height of the box along the Z-axis.

    Returns:
        Solid representing the box.
    """
    return adapter_create_box(length, width, height, origin=origin)


def get_bounding_box(
    obj,
):
    """
    Get the bounding box of a geometry object in a portable way.

    Args:
        obj: A CadQuery geometry object (Shape, Compound, etc.)

    Returns:
        Tuple of (min_point, max_point) where each point is (x, y, z)
    """
    return adapter_get_bounding_box(obj)


def get_bounding_box_center(obj):
    """
    Get the center point of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (x, y, z) coordinates of the center
    """
    return adapter_get_bounding_box_center(obj)


def get_bounding_box_size(obj):
    """
    Get the size (dimensions) of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (width, height, depth) - the size in x, y, z directions
    """

    return adapter_get_bounding_box_size(obj)


def get_vertices(obj):
    """
    Get vertices from a geometry object in a portable way.

    Args:
        obj: A CadQuery geometry object (Shape, Compound, etc.)

    Returns:
        List of vertex objects that have coordinate access
    """

    return adapter_get_vertices(obj)


def get_vertex_coordinates(obj) -> list:
    """
    Get all vertex coordinates from a geometry object.

    Args:
        obj: A CadQuery geometry object

    Returns:
        List of (x, y, z) tuples representing vertex coordinates
    """
    return adapter_get_vertex_coordinates(obj)


def get_vertex_coordinates_np(obj):
    """
    Get all vertex coordinates from a geometry object as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of shape (n_vertices, 3) with coordinates
    """
    return adapter_get_vertex_coordinates_np(obj)


def create_solid_from_traditional_face_vertex_maps(
    maps,
):
    """Create a CadQuery solid from a face-vertex map.

    Args:
        maps: A mapping with ``"vertexes"`` and ``"faces"`` entries. The vertex
            data may be provided as either a sequence (ordered by index) or a
            mapping whose keys can be converted to integers. Each vertex value
            is interpreted as an ``(x, y, z)`` coordinate triple. Face data can
            likewise be a sequence or mapping of integer-convertible keys to a
            sequence of vertex indices that define the perimeter of the face.

    Returns:
        ``cadquery.Solid`` constructed from the supplied topology.

    Raises:
        KeyError: if required keys are missing.
        ValueError: if the topology is invalid or does not describe a closed
            volume.
    """
    return adapter_create_solid_from_traditional_face_vertex_maps(maps)


def create_text_object(
    text: str,
    size,
    thickness,
    font=None,
    *,
    padding=0.0,
    font_path=None,
):
    """Create an extruded text solid anchored to the XY origin.

    The resulting solid is translated so its minimum X/Y lie ``padding``
    millimetres from the origin and its minimum Z sits on ``Z = 0``.
    """
    return adapter_create_text_object(
        text,
        size,
        thickness,
        font=font,
        padding=padding,
        font_path=font_path,
    )


def create_cylinder(
    radius,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    angle: Optional[float] = None,
):
    """Create a cylinder, optionally using ``angle`` for partial segments."""
    return adapter_create_cylinder(
        radius,
        height,
        origin=origin,
        direction=direction,
        angle=angle,
    )


def create_extruded_polygon(points, thickness):
    """Create an extruded polygon from a list of (x, y) coordinates."""
    return adapter_create_extruded_polygon(points, thickness)


def copy_part(part):
    """Create a copy of a CadQuery part."""
    return adapter_copy_part(part)


def translate_part(part, vector):
    """Translate a CadQuery part by the given vector."""
    return adapter_translate_part(part, vector)


def translate_part_native(part, *args):
    """Translate using native CadQuery signature. Used by composite objects."""
    return adapter_translate_part_native(part, *args)


def rotate_part(part, angle, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!"
    # if something is needed like this, do it in reconstruct

    """Rotate a CadQuery part around the given axis."""
    return adapter_rotate_part(part, angle, center=center, axis=axis)


def scale_part(part, factor, center=(0.0, 0.0, 0.0)):
    """Scale a CAD part around the given center."""
    return adapter_scale_part(part, factor, center=center)


def mirror_part(part, normal=(1, 0, 0), point=(0, 0, 0)):
    """Mirror a CadQuery part across a plane defined by normal and point."""
    return adapter_mirror_part(part, normal=normal, point=point)


def translate_part_native(part, *args):
    """Translate using native CadQuery signature. Used by composite objects."""
    return adapter_translate_part(part, *args)


def rotate_part_native(part, v1, v2, angle):
    return adapter_rotate_part_native(part, v1, v2, angle)


def scale_part_native(part, *args, **kwargs):
    """Scale using native CAD signature. Used by composite objects."""
    return adapter_scale_part_native(part, *args, **kwargs)


def mirror_part_native(part, *args, **kwargs):
    """Mirror using native CAD signature. Used by composite objects."""
    return adapter_mirror_part_native(part, *args, **kwargs)


def fuse_parts(part1, part2):
    """Fuse two CadQuery parts together."""
    return adapter_fuse_parts(part1, part2)


def cut_parts(part1, part2):
    """Cut part2 from part1."""
    return adapter_cut_parts(part1, part2)


def create_filleted_box(
    length, width, height, fillet_radius, fillets_at=None, no_fillets_at=None
):
    """
    Create a filleted box using CadQuery.

    Args:
        length: Box length (X dimension)
        width: Box width (Y dimension)
        height: Box height (Z dimension)
        fillet_radius: Radius of the fillets
        fillets_at: List of Alignment values indicating which faces/edges to fillet
        no_fillets_at: List of Alignment values indicating which faces/edges NOT to fillet

    Returns:
        CadQuery Shape (solid) with the filleted box
    """
    return adapter_create_filleted_box(
        length,
        width,
        height,
        fillet_radius,
        fillets_at=fillets_at,
        no_fillets_at=no_fillets_at,
    )


def get_volume(solid):
    """Get the volume of a CadQuery solid."""

    return adapter_get_volume(solid)


def fuse_parts(part1, part2):
    """Fuse two CadQuery parts together."""
    return adapter_fuse_parts(part1, part2)


def export_solid_to_stl(
    solid,
    destination: str,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
) -> None:
    """Export a CadQuery solid or workplane to an STL file.

    Args:
        solid: CadQuery solid or workplane to export.
        destination: Path to write the STL file to.
        tolerance: Linear deflection tolerance in model units (defaults to
            0.1 mm, suitable for most 3D printing previews).
        angular_tolerance: Angular deflection tolerance in radians.
    """
    return adapter_export_solid_to_stl(
        solid,
        destination,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def export_solid_to_step(
    solid,
    destination: str,
) -> None:
    """Export a solid to a STEP file.

    Args:
        solid: Solid to export.
        destination: Path to write the STEP file to.
    """
    return adapter_export_solid_to_step(
        solid,
        destination,
    )


def export_solid_to_obj(
    solid,
    destination: str,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
    color: Optional[tuple] = None,
    material_name: str = "material_0",
) -> None:
    """Export a solid to an OBJ file with optional color via MTL.

    Args:
        solid: Solid to export.
        destination: Path to write the OBJ file to.
        tolerance: Linear deflection tolerance in model units.
        angular_tolerance: Angular deflection tolerance in radians.
        color: Optional RGB color tuple (0.0-1.0 range). If provided, creates an MTL file.
        material_name: Name of the material in the MTL file.
    """
    return adapter_export_solid_to_obj(
        solid,
        destination,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
        color=color,
        material_name=material_name,
    )


def export_colored_parts_to_obj(
    parts,
    destination: str,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
) -> None:
    """Export multiple parts with different colors to a single OBJ file.

    Args:
        parts: List of tuples (solid, name, color) where:
            - solid: CAD solid
            - name: Part/material name
            - color: RGB tuple (0.0-1.0 range)
        destination: Path to write the OBJ file to.
        tolerance: Linear deflection tolerance in model units.
        angular_tolerance: Angular deflection tolerance in radians.
    """
    return adapter_export_colored_parts_to_obj(
        parts,
        destination,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def export_structured_step(
    structure,
    path: str,
) -> None:
    """Export a structured STEP assembly.

    Args:
        structure: Mapping of group name -> list of solids.
        path: Output STEP file path.
    """
    return adapter_export_structured_step(structure, path)


def deserialize_structured_step(path: str):
    """Load a structured STEP file and return grouped solids.

    Args:
        path: Path to the STEP file.
    """
    return adapter_deserialize_structured_step(path)


def import_solid_from_step(
    source: str,
):
    """Import a solid or assembly from a STEP file.

    Args:
        source: Path to read the STEP file from.
    """
    return adapter_import_solid_from_step(source)


def apply_fillet_by_alignment(
    solid, fillet_radius, fillets_at=None, no_fillets_at=None
):
    """Apply fillet to edges based on alignment positions.

    Args:
        solid: CadQuery solid
        fillet_radius: Radius of the fillet
        fillets_at: List of Alignment values indicating which edges to fillet
        no_fillets_at: List of Alignment values indicating which edges NOT to fillet

    Returns:
        Filleted solid
    """

    return adapter_apply_fillet_by_alignment(
        solid,
        fillet_radius,
        fillets_at=fillets_at,
        no_fillets_at=no_fillets_at,
    )


def create_solid_from_traditional_face_vertex_maps(
    maps,
):
    """Create a CadQuery solid from a face-vertex map.

    Args:
        maps: A mapping with ``"vertexes"`` and ``"faces"`` entries. The vertex
            data may be provided as either a sequence (ordered by index) or a
            mapping whose keys can be converted to integers. Each vertex value
            is interpreted as an ``(x, y, z)`` coordinate triple. Face data can
            likewise be a sequence or mapping of integer-convertible keys to a
            sequence of vertex indices that define the perimeter of the face.

    Returns:
        ``cadquery.Solid`` constructed from the supplied topology.

    Raises:
        KeyError: if required keys are missing.
        ValueError: if the topology is invalid or does not describe a closed
            volume.
    """
    return adapter_create_solid_from_traditional_face_vertex_maps(maps)


def apply_fillet_to_edges(solid, fillet_radius, edges):
    """Apply fillet to specific edges of a solid.

    Args:
        solid: CadQuery solid
        fillet_radius: Radius of the fillet
        edges: List of edges to fillet

    Returns:
        Filleted solid
    """
    return adapter_apply_fillet_to_edges(solid, fillet_radius, edges)


def filter_edges_by_function(solid, filter_function):
    """Filter edges of a solid based on a user-defined function.

    Args:
        solid: CadQuery solid
        filter_function: Function that takes an edge and returns True if it should be included

    Returns:
        List of edges that match the filter criteria
    """
    return adapter_filter_edges_by_function(solid, filter_function)


def create_cone(
    radius1,
    radius2,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
):
    """Create a cone with base ``radius1`` and top ``radius2``."""

    return adapter_create_cone(
        radius1,
        radius2,
        height,
        origin=origin,
        direction=direction,
    )


def create_sphere(
    radius,
    origin=(0.0, 0.0, 0.0),
):
    """Create a sphere centered at ``origin``."""
    return adapter_create_sphere(radius, origin=origin)


def get_adapter_id() -> str:
    """Get the ID of the currently selected CAD adapter."""
    return adapter_get_adapter_id()
