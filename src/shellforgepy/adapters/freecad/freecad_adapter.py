from typing import Optional

import numpy as np

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
    if hasattr(part, "copy"):
        return part.copy()
    else:
        return part


def translate_part(part, vector):
    """Translate a FreeCAD part by the given vector."""
    if len(vector) != 3:
        raise ValueError("Vector must contain exactly three components")
    if np.linalg.norm(np.array(vector)) < 1e-8:
        return part.copy()  # No translation needed
    translated = part.copy()
    translated.translate(Base.Vector(vector[0], vector[1], vector[2]))
    return translated


def rotate_part(part, angle, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """Rotate a FreeCAD part around the given axis.

    Args:
        part: FreeCAD part to rotate
        angle: Rotation angle in degrees
        center: Center point of rotation as (x, y, z) tuple
        axis: Rotation axis as (x, y, z) tuple
    """
    center_vec = Base.Vector(center[0], center[1], center[2])
    axis_vec = Base.Vector(axis[0], axis[1], axis[2])
    rotated = part.copy()
    # FreeCAD's rotate method expects angle in degrees, not radians
    rotated.rotate(center_vec, axis_vec, angle)
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
