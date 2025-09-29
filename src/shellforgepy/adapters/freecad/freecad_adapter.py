from typing import Optional

import numpy as np
from shellforgepy.construct.alignment import Alignment

# FreeCAD imports will be available when run in FreeCAD environment
try:
    import FreeCAD
    import Part
    from FreeCAD import Base
except ImportError:
    # When not in FreeCAD environment, these will be None
    Part = None
    FreeCAD = None
    Base = None


import logging

_logger = logging.getLogger(__name__)

# FreeCAD specific adapter implementations
# Here, cad-backend-specific should be implemented
# Any code that can be implemented backend-agnostic should go in geometry/ or construct/ or produce/ or similar
# To test this functionality, run in a FreeCAD Python environment, such as
# ./freecad_python.sh -m pytest tests/unit/adapters/freecad/test_freecad_basic_transformations.py


def get_adapter_id():
    """Return a string identifying this adapter."""
    return "freecad"


def create_solid_from_traditional_face_vertex_maps(maps):
    """
    Create a solid from traditional face-vertex maps.

    Args:
    maps (dict): A dictionary containing vertexes and faces keys.

    The vertexes key should map vertex indexes to 3-tuple coordinates.
    The faces key should map face indexes to lists of vertex indexes.

    Example:
    A tetrahedron with vertices at (0, 0, 0), (1, 0, 0), (0, 1, 0), and (0, 0, 1).
    The faces are defined by the vertex indexes in a counter-clockwise order.
    {
        "faces": {"0": [0, 1, 2], "1": [0, 1, 3], "2": [0, 2, 3], "3": [1, 2, 3]},
        "vertexes": {
            "0": [0.0, 1.0, 0],
            "1": [0.87, -0.5, 0],
            "2": [-0.87, -0.5, 0],
            "3": [0, 0, 1],
        },
    }

    Returns:
    Part.Shape: The solid shape, if successful.
    """

    import Part
    from FreeCAD import Base

    def vec_from_tuple(t):
        return Base.Vector(t[0], t[1], t[2])

    vertexes = maps["vertexes"]
    faces = maps["faces"]

    face_edges = set()
    for f in faces.values():
        for i in range(len(f)):
            edge = (f[i], f[(i + 1) % len(f)])
            face_edges.add(edge)

    for e in face_edges:
        if (e[1], e[0]) not in face_edges:
            raise ValueError(
                f"The face-vertex maps do not form a closed solid. edge {e} is in one direction only"
            )

    fc_faces = []
    for f in faces.values():
        vertex_points = [vertexes[v] for v in f]
        vertex_points.append(vertex_points[0])
        vertex_points = [vec_from_tuple(v) for v in vertex_points]

        wire = Part.makePolygon(vertex_points)
        face = Part.Face(wire)

        fc_faces.append(face)

    shell = Part.makeShell(fc_faces)

    return Part.makeSolid(shell)


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
    bbox = obj.BoundBox
    min_point = (bbox.XMin, bbox.YMin, bbox.ZMin)
    max_point = (bbox.XMax, bbox.YMax, bbox.ZMax)
    return min_point, max_point


def get_bounding_box_center(obj):
    """
    Get the center point of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (x, y, z) coordinates of the center
    """
    min_point, max_point = get_bounding_box(obj)
    center = (
        (min_point[0] + max_point[0]) / 2,
        (min_point[1] + max_point[1]) / 2,
        (min_point[2] + max_point[2]) / 2,
    )
    return center


def get_bounding_box_size(obj):
    """
    Get the size (dimensions) of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (width, height, depth) - the size in x, y, z directions
    """
    min_point, max_point = get_bounding_box(obj)
    size = (
        max_point[0] - min_point[0],
        max_point[1] - min_point[1],
        max_point[2] - min_point[2],
    )
    return size


def get_bounding_box_min(obj):
    """
    Get the minimum point of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (x_min, y_min, z_min)
    """
    min_point, _ = get_bounding_box(obj)
    return min_point


def get_bounding_box_max(obj):
    """
    Get the maximum point of the bounding box.

    Args:
        obj: A CadQuery geometry object

    Returns:
        Tuple of (x_max, y_max, z_max)
    """
    _, max_point = get_bounding_box(obj)
    return max_point


def get_z_min(obj):
    """
    Get the minimum Z coordinate of the object.

    Args:
        obj: A CadQuery geometry object

    Returns:
        The minimum Z coordinate
    """
    min_point, _ = get_bounding_box(obj)
    return min_point[2]


def get_z_max(obj):
    """
    Get the maximum Z coordinate of the object.

    Args:
        obj: A CadQuery geometry object

    Returns:
        The maximum Z coordinate
    """
    _, max_point = get_bounding_box(obj)
    return max_point[2]


# Convenience functions that return numpy arrays for easier computation
def get_bounding_box_center_np(obj):
    """
    Get the center point of the bounding box as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of [x, y, z] coordinates of the center
    """
    return np.array(get_bounding_box_center(obj))


def get_bounding_box_min_np(obj):
    """
    Get the minimum point of the bounding box as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of [x_min, y_min, z_min]
    """
    return np.array(get_bounding_box_min(obj))


def get_bounding_box_max_np(obj):
    """
    Get the maximum point of the bounding box as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of [x_max, y_max, z_max]
    """
    return np.array(get_bounding_box_max(obj))


def get_bounding_box_size_np(obj):
    """
    Get the size of the bounding box as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of [width, height, depth]
    """
    return np.array(get_bounding_box_size(obj))


def get_vertices(obj):
    """
    Get vertices from a FreeCAD geometry object.

    Args:
        obj: A FreeCAD Part object (Shape, Compound, etc.)

    Returns:
        List of vertex coordinate tuples
    """
    if hasattr(obj, "Vertexes"):
        return [tuple([v.Point.x, v.Point.y, v.Point.z]) for v in obj.Vertexes]
    else:
        return []


def get_vertex_coordinates(obj) -> list:
    """
    Get all vertex coordinates from a geometry object.

    Args:
        obj: A CadQuery geometry object

    Returns:
        List of (x, y, z) tuples representing vertex coordinates
    """
    vertices = get_vertices(obj)
    coordinates = []

    for vertex in vertices:
        # CadQuery vertices have different coordinate access patterns
        if hasattr(vertex, "X") and hasattr(vertex, "Y") and hasattr(vertex, "Z"):
            # CadQuery Vector-like interface
            coordinates.append((vertex.X, vertex.Y, vertex.Z))
        elif hasattr(vertex, "Point"):
            # CadQuery Vertex with Point attribute
            point = vertex.Point
            if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                coordinates.append((point.x, point.y, point.z))
            elif hasattr(point, "X") and hasattr(point, "Y") and hasattr(point, "Z"):
                coordinates.append((point.X, point.Y, point.Z))
            else:
                # Try to treat as tuple/list
                coordinates.append((point[0], point[1], point[2]))
        else:
            # Try to treat vertex as coordinate directly
            coordinates.append((vertex[0], vertex[1], vertex[2]))

    return coordinates


def get_vertex_coordinates_np(obj):
    """
    Get all vertex coordinates from a geometry object as a numpy array.

    Args:
        obj: A CadQuery geometry object

    Returns:
        numpy array of shape (n_vertices, 3) with coordinates
    """
    coordinates = get_vertex_coordinates(obj)
    return np.array(coordinates)


def calc_center(part):
    """
    Calculate the center of a Part object.
    Returns a tuple representing the center.
    """
    bb = part.BoundBox
    center = (
        (bb.XMin + bb.XMax) / 2,
        (bb.YMin + bb.YMax) / 2,
        (bb.ZMin + bb.ZMax) / 2,
    )
    return center


def get_vertex_points(obj) -> list:
    """
    Get vertex Point objects from a geometry object (for FreeCAD compatibility).

    Args:
        obj: A CadQuery geometry object

    Returns:
        List of Point objects
    """
    vertices = get_vertices(obj)
    points = []

    for vertex in vertices:
        if hasattr(vertex, "Point"):
            points.append(vertex.Point)
        else:
            # For future FreeCAD compatibility, might need different handling
            points.append(vertex)

    return points


def create_text_object(
    text: str,
    size,
    thickness,
    font_path=None,
    *,
    padding=0.0,
):
    """Create an extruded text solid using FreeCAD Draft.

    The resulting solid is translated so its minimum X/Y lie ``padding``
    millimetres from the origin and its minimum Z sits on ``Z = 0``.
    """
    try:
        import Draft
    except ImportError:
        raise RuntimeError("FreeCAD Draft module not available")

    if not text:
        raise ValueError("Text must be a non-empty string")
    if size <= 0:
        raise ValueError("Size must be positive")
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if padding < 0:
        raise ValueError("Padding cannot be negative")

    if font_path is None:
        font_path = "/Users/mege/Library/Fonts/lintsec.regular.ttf"

    S1 = Draft.make_shapestring(text, font_path, size)
    extr = S1.Shape.extrude(Base.Vector(0, 0, thickness))

    # Apply padding offset
    if padding > 0:
        bbox = extr.BoundBox
        offset = Base.Vector(-bbox.XMin + padding, -bbox.YMin + padding, -bbox.ZMin)
        extr.translate(offset)

    return extr


def create_basic_box(
    length,
    width,
    height,
    origin=(0.0, 0.0, 0.0),
):
    """Create an axis-aligned box with its minimum corner at ``origin``."""
    box = Part.makeBox(length, width, height)
    if origin != (0.0, 0.0, 0.0):
        box.translate(Base.Vector(origin[0], origin[1], origin[2]))
    return box


def create_basic_cylinder(
    radius,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    angle: Optional[float] = None,
):
    """Create a cylinder, optionally using ``angle`` for partial segments."""
    origin_vec = Base.Vector(origin[0], origin[1], origin[2])
    dir_vec = Base.Vector(direction[0], direction[1], direction[2])

    if angle is not None:
        return Part.makeCylinder(radius, height, origin_vec, dir_vec, angle)
    else:
        return Part.makeCylinder(radius, height, origin_vec, dir_vec)


def create_basic_sphere(
    radius,
    origin=(0.0, 0.0, 0.0),
):
    """Create a sphere centered at ``origin``."""
    sphere = Part.makeSphere(radius)
    if origin != (0.0, 0.0, 0.0):
        sphere.translate(Base.Vector(origin[0], origin[1], origin[2]))
    return sphere


def create_basic_cone(
    radius1,
    radius2,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
):
    """Create a cone with base ``radius1`` and top ``radius2``."""
    origin_vec = Base.Vector(origin[0], origin[1], origin[2])
    dir_vec = Base.Vector(direction[0], direction[1], direction[2])
    return Part.makeCone(radius1, radius2, height, origin_vec, dir_vec)


def export_solid_to_stl(
    solid,
    destination: str,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
) -> None:
    """Export a FreeCAD solid to an STL file.

    Args:
        solid: FreeCAD Part.Shape to export.
        destination: Path to write the STL file to.
        tolerance: Linear deflection tolerance in model units (defaults to
            0.1 mm, suitable for most 3D printing previews).
        angular_tolerance: Angular deflection tolerance in radians.
    """
    import os

    if os.path.exists(destination):
        os.remove(destination)
    solid.exportStl(destination)


def copy_part(part):
    """Create a copy of a FreeCAD part."""
    # There are NO FRAMEWORK SPECIFIC CALLS allowed here! Use adapter functions only!
    # isinstance(x, NamedPart) or similar ARE FORBIDDEN here!
    # if something is needed like this, do it in reconstruct

    return part.copy()


def translate_part(part, vector):
    """Translate a FreeCAD part by the given vector."""
    # There are NO FRAMEWORK SPECIFIC CALLS allowed here! Use adapter functions only!
    # isinstance(x, NamedPart) or similar ARE FORBIDDEN here!
    # if something is needed like this, do it in reconstruct

    _logger.info(f"Translating part by vector {vector}, part={part} , id={id(part)}")
    vec = Base.Vector(vector[0], vector[1], vector[2])

    # FreeCAD modifies in-place, so create a copy first
    translate_retval = part.copy()
    translate_retval.translate(vec)

    if hasattr(part, "reconstruct"):
        return part.reconstruct(translate_retval)
    else:
        return translate_retval


def rotate_part(part, angle, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """Rotate a FreeCAD part around the given axis.

    Args:
        part: FreeCAD part to rotate
        angle: Rotation angle in degrees
        center: Center point of rotation as (x, y, z) tuple
        axis: Rotation axis as (x, y, z) tuple
    """
    # There are NO FRAMEWORK SPECIFIC CALLS allowed here! Use adapter functions only!
    # isinstance(x, NamedPart) or similar ARE FORBIDDEN here!
    # if something is needed like this, do it in reconstruct

    if center is None:
        center = (0.0, 0.0, 0.0)
    if axis is None:
        axis = (0.0, 0.0, 1.0)

    center_vec = Base.Vector(center[0], center[1], center[2])
    axis_vec = Base.Vector(axis[0], axis[1], axis[2])

    # FreeCAD modifies in-place, so create a copy first
    rotate_retval = part.copy()
    rotate_retval.rotate(center_vec, axis_vec, angle)

    if hasattr(part, "reconstruct"):
        return part.reconstruct(rotate_retval)
    else:
        return rotate_retval


def translate_part_native(part, *args):
    """Translate using native method. Must use the underlying cad system such that it creates a new object, and passes it to the reconstruct method if available, or returns it outright."""
    _logger.info(
        f"Native translating part by vector {args}, part={part} , id={id(part)}"
    )

    # we copy the part
    translated = part.copy()
    # the underlying translate must translate in-place
    translated.translate(*args)

    if hasattr(part, "reconstruct"):
        _logger.info(
            f"Reconstructing part {part} , id={id(part)}, translated id={id(translated)}"
        )
        return part.reconstruct(translated)
    else:
        _logger.info(
            f"Not reconstructing part {part} , id={id(part)}, translated id={id(translated)}"
        )
        return translated


def rotate_part_native(part, base, dir, degree):
    """Rotate using native FreeCAD signature: rotate(base, dir, degree). Other adapters may have different signatures."""
    rotated = part.copy()
    rotated.rotate(base, dir, degree)

    if hasattr(part, "reconstruct"):
        return part.reconstruct(rotated)
    else:
        return rotated


def fuse_parts(part1, part2):
    """Fuse two FreeCAD parts together."""
    return part1.fuse(part2)


def cut_parts(part1, part2):
    """Cut part2 from part1."""
    return part1.cut(part2)


def create_extruded_polygon(points, thickness):
    """
    Create an extruded polygon from a list of 2-tuple (x, y) coordinates in the active document.

    Args:
    points (list of tuple): List of (x, y) coordinates for the polygon.
    thickness (float): The extrusion thickness along the z-axis.

    Returns:
    Part.Shape: The extruded polygon shape, if successful.
    """

    # Convert list of tuples into FreeCAD Vectors, adding the first point at the end to close the polygon
    points = [Base.Vector(x, y, 0) for x, y in points]
    # check if it is already closed

    if points[0].distanceToPoint(points[-1]) > 1e-6:
        points = points + [Base.Vector(points[0][0], points[0][1], 0)]

    # Create a polygon (wire) from the points and close it
    wire = Part.makePolygon(points)

    # Convert the wire to a face
    face = Part.Face(wire)

    # Extrude the face to create a solid
    extruded = face.extrude(Base.Vector(0, 0, thickness))

    return extruded


def get_volume(solid):
    """Get the volume of a FreeCAD solid."""
    return solid.Volume


def create_filleted_box(
    length, width, height, fillet_radius, fillets_at=None, no_fillets_at=None
):
    box = Part.makeBox(length, width, height)

    def edge_is_at(v0, v1, alignment):
        if alignment == Alignment.TOP:
            return (abs(v0.z - height) < 1e-3) and (abs(v1.z - height) < 1e-3)
        elif alignment == Alignment.BOTTOM:
            return (abs(v0.z) < 1e-3) and (abs(v1.z) < 1e-3)
        elif alignment == Alignment.LEFT:
            return (abs(v0.x) < 1e-3) and (abs(v1.x) < 1e-3)
        elif alignment == Alignment.RIGHT:
            return (abs(v0.x - length) < 1e-3) and (abs(v1.x - length) < 1e-3)
        elif alignment == Alignment.FRONT:
            return (abs(v0.y) < 1e-3) and (abs(v1.y) < 1e-3)
        elif alignment == Alignment.BACK:
            return (abs(v0.y - width) < 1e-3) and (abs(v1.y - width) < 1e-3)
        else:
            raise Exception(f"Invalid alignment: {alignment}")

    def edge_is_at_one_of(v0, v1, alignments):
        for alignment in alignments:
            if edge_is_at(v0, v1, alignment):
                return True
        return False

    edges = []
    for i in box.Edges:
        v0 = i.Vertexes[0].Point
        v1 = i.Vertexes[1].Point

        if fillets_at is None or edge_is_at_one_of(v0, v1, fillets_at):
            if no_fillets_at is None or not edge_is_at_one_of(v0, v1, no_fillets_at):
                edges.append(i)

    box = box.makeFillet(fillet_radius, edges)

    return box


def filter_edges_by_z_position(solid, z_threshold, below=True):
    """Filter edges based on their Z position.

    Args:
        solid: FreeCAD solid
        z_threshold: Z coordinate threshold
        below: If True, return edges with all vertices <= threshold;
               if False, return edges with all vertices >= threshold

    Returns:
        List of edges that meet the criteria
    """
    edges = []
    try:
        for edge in solid.Edges:
            # Get edge's vertices
            vertices = edge.Vertexes
            if vertices:
                # Check if all vertices meet the criteria
                vertex_z_coords = [v.Point.z for v in vertices]

                if below:
                    # All vertices must be at or below the threshold
                    if all(z <= z_threshold for z in vertex_z_coords):
                        edges.append(edge)
                else:
                    # All vertices must be at or above the threshold
                    if all(z >= z_threshold for z in vertex_z_coords):
                        edges.append(edge)

    except Exception as e:
        _logger.warning(f"Error filtering edges: {e}")

    return edges


def filter_edges_by_alignment(solid, fillets_at=None, no_fillets_at=None):
    """Filter edges based on alignment positions (top, bottom, left, right, front, back).

    Args:
        solid: FreeCAD solid
        fillets_at: List of Alignment values indicating which faces/edges to include
        no_fillets_at: List of Alignment values indicating which faces/edges to exclude

    Returns:
        List of edges that meet the criteria
    """
    from shellforgepy.construct.alignment import Alignment

    # Get bounding box for alignment calculations
    bbox = solid.BoundBox
    length = bbox.XLength
    width = bbox.YLength
    height = bbox.ZLength
    x_min, y_min, z_min = bbox.XMin, bbox.YMin, bbox.ZMin

    def edge_is_at(edge, alignment):
        """Check if an edge is at a specific alignment position."""
        tolerance = 1e-3

        # Get edge bounding box for circular edges
        edge_bbox = edge.BoundBox

        # For circular edges, check if the edge is at constant Z/X/Y within tolerance
        if alignment == Alignment.TOP:
            # Edge is at top if its Z range is at the top of the solid
            return (
                abs(edge_bbox.ZMax - (z_min + height)) < tolerance
                and abs(edge_bbox.ZMin - (z_min + height)) < tolerance
            )
        elif alignment == Alignment.BOTTOM:
            # Edge is at bottom if its Z range is at the bottom of the solid
            return (
                abs(edge_bbox.ZMax - z_min) < tolerance
                and abs(edge_bbox.ZMin - z_min) < tolerance
            )
        elif alignment == Alignment.LEFT:
            # Edge is at left if its X range is at the left of the solid
            return (
                abs(edge_bbox.XMax - x_min) < tolerance
                and abs(edge_bbox.XMin - x_min) < tolerance
            )
        elif alignment == Alignment.RIGHT:
            # Edge is at right if its X range is at the right of the solid
            return (
                abs(edge_bbox.XMax - (x_min + length)) < tolerance
                and abs(edge_bbox.XMin - (x_min + length)) < tolerance
            )
        elif alignment == Alignment.FRONT:
            # Edge is at front if its Y range is at the front of the solid
            return (
                abs(edge_bbox.YMax - y_min) < tolerance
                and abs(edge_bbox.YMin - y_min) < tolerance
            )
        elif alignment == Alignment.BACK:
            # Edge is at back if its Y range is at the back of the solid
            return (
                abs(edge_bbox.YMax - (y_min + width)) < tolerance
                and abs(edge_bbox.YMin - (y_min + width)) < tolerance
            )
        else:
            return False

    def edge_is_at_one_of(edge, alignments):
        """Check if an edge is at any of the specified alignments."""
        for alignment in alignments:
            if edge_is_at(edge, alignment):
                return True
        return False

    edges = []
    try:
        for edge in solid.Edges:
            # Include edge if it matches fillets_at criteria
            include_edge = True
            if fillets_at is not None:
                include_edge = edge_is_at_one_of(edge, fillets_at)

            # Exclude edge if it matches no_fillets_at criteria
            if include_edge and no_fillets_at is not None:
                include_edge = not edge_is_at_one_of(edge, no_fillets_at)

            if include_edge:
                edges.append(edge)

    except Exception as e:
        _logger.warning(f"Error filtering edges by alignment: {e}")

    return edges


def filter_edges_by_function(solid, edge_filter_func):
    """Filter edges using a custom function.

    Args:
        solid: FreeCAD solid
        edge_filter_func: Function that takes (bbox, v0_point, v1_point) and returns bool
                         bbox: bounding box as (min_point, max_point) tuples
                         v0_point: first vertex as (x, y, z) tuple
                         v1_point: second vertex as (x, y, z) tuple

    Returns:
        List of edges that meet the criteria
    """
    edges = []
    try:
        # Get bounding box
        bbox = get_bounding_box(solid)

        for edge in solid.Edges:
            vertices = edge.Vertexes
            if len(vertices) >= 2:
                v0 = vertices[0].Point
                v1 = vertices[1].Point

                # Convert vertices to tuples
                v0_point = (v0.x, v0.y, v0.z)
                v1_point = (v1.x, v1.y, v1.z)

                # Call user's filter function
                if edge_filter_func(bbox, v0_point, v1_point):
                    edges.append(edge)

    except Exception as e:
        _logger.warning(f"Error filtering edges by function: {e}")

    return edges


def apply_fillet_to_edges(solid, fillet_radius, edges):
    """Apply fillet to specific edges of a solid.

    Args:
        solid: FreeCAD solid
        fillet_radius: Radius of the fillet
        edges: List of edges to fillet

    Returns:
        Filleted solid
    """
    if not edges:
        return solid

    try:
        return solid.makeFillet(fillet_radius, edges)
    except Exception as e:
        _logger.warning(f"Error applying fillet: {e}")
        return solid


def apply_fillet_by_alignment(
    solid, fillet_radius, fillets_at=None, no_fillets_at=None
):
    """Apply fillet to edges based on alignment positions.

    Args:
        solid: FreeCAD solid
        fillet_radius: Radius of the fillet
        fillets_at: List of Alignment values indicating which edges to fillet
        no_fillets_at: List of Alignment values indicating which edges NOT to fillet

    Returns:
        Filleted solid
    """
    edges = filter_edges_by_alignment(solid, fillets_at, no_fillets_at)
    return apply_fillet_to_edges(solid, fillet_radius, edges)


def apply_fillet_by_function(solid, fillet_radius, edge_filter_func):
    """Apply fillet to edges selected by a custom function.

    Args:
        solid: FreeCAD solid
        fillet_radius: Radius of the fillet
        edge_filter_func: Function that takes (bbox, v0_point, v1_point) and returns bool

    Returns:
        Filleted solid
    """
    edges = filter_edges_by_function(solid, edge_filter_func)
    return apply_fillet_to_edges(solid, fillet_radius, edges)
