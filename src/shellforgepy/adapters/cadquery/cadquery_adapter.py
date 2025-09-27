import math
from typing import List, Optional, Tuple, Union

import cadquery as cq
import numpy as np

from ...construct.alignment import Alignment


def _as_cq_vector(value) -> cq.Vector:
    if isinstance(value, cq.Vector):
        return value
    if len(value) != 3:
        raise ValueError("Vector value must provide exactly three components")
    return cq.Vector(float(value[0]), float(value[1]), float(value[2]))


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
    # CadQuery objects use BoundingBox() method
    bbox = obj.BoundingBox()
    min_point = (bbox.xmin, bbox.ymin, bbox.zmin)
    max_point = (bbox.xmax, bbox.ymax, bbox.zmax)
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
    Get vertices from a geometry object in a portable way.

    Args:
        obj: A CadQuery geometry object (Shape, Compound, etc.)

    Returns:
        List of vertex objects that have coordinate access
    """
    if hasattr(obj, "Vertices"):
        # CadQuery objects use Vertices() method
        vertices = obj.Vertices()
        return vertices if vertices is not None else []
    elif hasattr(obj, "Vertexes"):
        # FreeCAD objects use Vertexes property (for future compatibility)
        return obj.Vertexes
    else:
        raise AttributeError(
            f"Object of type {type(obj)} does not have a recognized vertices interface"
        )


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


def translate(x, y, z):
    """
    Create a translation function that can be applied to CadQuery objects.

    Args:
        x, y, z: Translation distances along respective axes

    Returns:
        A function that applies the translation to a CadQuery object
    """

    def retval(body):
        if isinstance(body, (list, tuple)):
            return [retval(b) for b in body]

        if isinstance(body, cq.Workplane):
            return body.translate(cq.Vector(x, y, z))
        elif isinstance(body, cq.Shape):
            return body.translate(cq.Vector(x, y, z))
        else:
            raise TypeError(f"Unsupported type for translation: {type(body)}")

    return retval


def rotate(
    angle,
    center=None,
    axis=None,
):
    """
    Create a rotation function that can be applied to CadQuery objects.

    Args:
        angle: Rotation angle in degrees
        center: Center point of rotation (x, y, z). Defaults to origin.
        axis: Rotation axis (x, y, z). Defaults to Z-axis.

    Returns:
        A function that applies the rotation to a CadQuery object
    """
    if center is None:
        center = (0, 0, 0)

    if axis is None:
        axis = (0, 0, 1)

    def retval(body):
        if isinstance(body, list):
            return [retval(b) for b in body]

        if isinstance(body, cq.Workplane):
            return body.rotate(
                axisStartPoint=cq.Vector(*center),
                axisEndPoint=cq.Vector(*center) + cq.Vector(*axis),
                angleDegrees=angle,
            )
        elif isinstance(body, cq.Shape):
            axis_start = cq.Vector(*center)
            axis_end = axis_start + cq.Vector(*axis)
            return body.rotate(axis_start, axis_end, angle)
        else:
            raise TypeError(f"Unsupported type for rotation: {type(body)}")

    return retval


def _normalize_vertex_map(vertexes):
    """Normalize vertex data to a dictionary of int -> (x, y, z)."""
    if isinstance(vertexes, dict):
        return {int(k): tuple(v) for k, v in vertexes.items()}
    elif isinstance(vertexes, (list, tuple)):
        return {i: tuple(v) for i, v in enumerate(vertexes)}
    else:
        raise ValueError("Vertexes must be dict, list, or tuple")


def _normalize_face_map(faces):
    """Normalize face data to a list of vertex index lists."""
    if isinstance(faces, dict):
        return [list(face) for face in faces.values()]
    elif isinstance(faces, (list, tuple)):
        return [list(face) for face in faces]
    else:
        raise ValueError("Faces must be dict, list, or tuple")


def _validate_closed_mesh(vertexes, faces) -> None:
    edge_set = set()
    for face in faces:
        count = len(face)
        for i in range(count):
            edge = (face[i], face[(i + 1) % count])
            if edge[0] == edge[1]:
                raise ValueError(f"Degenerate edge detected in face {face}: {edge}")
            edge_set.add(edge)

    for start, end in edge_set:
        if (end, start) not in edge_set:
            raise ValueError(
                "The face-vertex maps do not form a closed solid. "
                f"Missing opposing edge for ({start}, {end})."
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

    if "vertexes" not in maps or "faces" not in maps:
        raise KeyError("maps must contain 'vertexes' and 'faces' entries")

    vertex_lookup = _normalize_vertex_map(maps["vertexes"])  # type: ignore[arg-type]
    face_list = _normalize_face_map(maps["faces"])  # type: ignore[arg-type]

    _validate_closed_mesh(vertex_lookup, face_list)

    cq_faces: List[cq.Face] = []
    for face_indices in face_list:
        points = [cq.Vector(*vertex_lookup[index]) for index in face_indices]
        wire = cq.Wire.makePolygon(points, close=True)
        cq_face = cq.Face.makeFromWires(wire)
        if cq_face is None or cq_face.isNull():
            raise ValueError(f"Failed to build face from indices {face_indices}")
        cq_faces.append(cq_face)

    shell = cq.Shell.makeShell(cq_faces)
    if shell is None or shell.isNull():
        raise ValueError("Failed to build shell from faces")
    shell_closed: bool
    if hasattr(shell, "isClosed"):
        shell_closed = shell.isClosed()  # type: ignore[call-arg]
    elif hasattr(shell, "Closed"):
        shell_closed = bool(shell.Closed)
    else:
        shell_closed = True

    if not shell_closed:
        raise ValueError("The generated shell is not closed")

    solid = cq.Solid.makeSolid(shell)
    if solid is None or solid.isNull():
        raise ValueError("Failed to build solid from shell")

    return solid


def directed_cylinder_at(
    base_point,
    direction,
    radius,
    height,
):
    """Create a cylinder oriented along ``direction`` starting at ``base_point``.

    Args:
        base_point: XYZ coordinates of the cylinder's base centre in millimetres.
        direction: Vector indicating the extrusion direction. Must be non-zero.
        radius: Cylinder radius.
        height: Cylinder height measured along ``direction``.

    Returns:
        ``cadquery.Solid`` positioned and oriented as requested.
    """

    direction_vec = _as_cq_vector(direction)
    if direction_vec.Length <= 1e-9:
        raise ValueError("Direction vector cannot be zero-length")

    base_shape = cq.Workplane("XY").circle(radius).extrude(height).val()

    default_dir = cq.Vector(0.0, 0.0, 1.0)
    target_dir = direction_vec.normalized()

    axis = default_dir.cross(target_dir)
    if axis.Length > 1e-9:
        angle_deg = math.degrees(default_dir.getAngle(target_dir))
        axis_start = cq.Vector(0.0, 0.0, 0.0)
        axis_end = axis_start + axis.normalized()
        base_shape = base_shape.rotate(axis_start, axis_end, angle_deg)
    else:
        dot = default_dir.dot(target_dir)
        if dot < 0:
            axis_start = cq.Vector(0.0, 0.0, 0.0)
            axis_end = axis_start + cq.Vector(1.0, 0.0, 0.0)
            base_shape = base_shape.rotate(axis_start, axis_end, 180.0)

    base_shape = base_shape.translate(_as_cq_vector(base_point))
    return base_shape


def create_text_object(
    text: str,
    size,
    thickness,
    font=None,
    *,
    padding=0.0,
):
    """Create an extruded text solid anchored to the XY origin.

    The resulting solid is translated so its minimum X/Y lie ``padding``
    millimetres from the origin and its minimum Z sits on ``Z = 0``.
    """

    if not text:
        raise ValueError("Text must be a non-empty string")
    if size <= 0:
        raise ValueError("Size must be positive")
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    if padding < 0:
        raise ValueError("Padding cannot be negative")

    text_kwargs = {
        "combine": True,
        "clean": True,
        "halign": "left",
        "valign": "baseline",
    }
    if font:
        text_kwargs["font"] = font

    text_wp = cq.Workplane("XY").text(text, size, thickness, **text_kwargs)
    solid = text_wp.val()
    if solid is None:
        raise RuntimeError("CadQuery text generation returned no solid")

    bbox = solid.BoundingBox()
    offset = cq.Vector(-bbox.xmin + padding, -bbox.ymin + padding, -bbox.zmin)
    return solid.translate(offset)


def create_basic_box(
    length,
    width,
    height,
    origin=(0.0, 0.0, 0.0),
):
    """Create an axis-aligned box with its minimum corner at ``origin``."""

    return cq.Solid.makeBox(length, width, height, _as_cq_vector(origin))


def create_basic_cylinder(
    radius,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    angle: Optional[float] = None,
):
    """Create a cylinder, optionally using ``angle`` for partial segments."""

    base = _as_cq_vector(origin)
    axis = _as_cq_vector(direction)
    if angle is not None:
        return cq.Solid.makeCylinder(radius, height, base, axis, angle)
    return cq.Solid.makeCylinder(radius, height, base, axis)


def create_basic_sphere(
    radius,
    origin=(0.0, 0.0, 0.0),
):
    """Create a sphere centered at ``origin``."""
    sphere = cq.Workplane("XY").sphere(radius).val()
    offset = _as_cq_vector(origin)
    if offset.Length > 0:
        sphere = sphere.translate(offset)
    return sphere


def create_basic_cone(
    radius1,
    radius2,
    height,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
):
    """Create a cone with base ``radius1`` and top ``radius2``."""

    return cq.Solid.makeCone(
        radius1,
        radius2,
        height,
        _as_cq_vector(origin),
        _as_cq_vector(direction),
    )


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

    cq.exporters.export(
        solid,
        destination,
        tolerance=tolerance,
        angularTolerance=angular_tolerance,
    )


def copy_part(part):
    """Create a copy of a CadQuery part."""
    if hasattr(part, "copy"):
        return part.copy()
    else:
        return part


def translate_part(part, vector):
    """Translate a CadQuery part by the given vector."""
    if len(vector) != 3:
        raise ValueError("Vector must contain exactly three components")
    vec = cq.Vector(*map(float, vector))
    return part.translate(vec)


def rotate_part(part, angle, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """Rotate a CadQuery part around the given axis."""
    center_vec = cq.Vector(*center)
    axis_vec = cq.Vector(*axis)
    return part.rotate(center_vec, center_vec + axis_vec, angle)


def fuse_parts(part1, part2):
    """Fuse two CadQuery parts together."""
    return part1.fuse(part2)


def align(part, to, alignment, axes=None):
    """Align a part to another part according to alignment."""
    return align_translation(part, to, alignment, axes=axes)(part.copy())


def align_translation(
    part,
    to,
    alignment: Alignment,
    axes: Optional[List[int]] = None,
):
    """
    Create a translation function that aligns one object to another.

    Args:
        part: The object to be aligned
        to: The target object to align to
        alignment: The type of alignment to perform
        axes: Optional list of axes to constrain alignment to (0=X, 1=Y, 2=Z)

    Returns:
        A function that applies the alignment translation
    """
    # Extract the solid from workplane if needed
    part_obj = part.val() if hasattr(part, "val") else part
    to_obj = to.val() if hasattr(to, "val") else to

    bb = part_obj.BoundingBox()
    to_bb = to_obj.BoundingBox()

    part_width = bb.xlen
    part_length = bb.ylen
    part_height = bb.zlen

    def project_to_axes(x: float, y: float, z: float) -> Tuple[float, float, float]:
        if axes is None:
            return x, y, z

        return (x if 0 in axes else 0, y if 1 in axes else 0, z if 2 in axes else 0)

    if alignment == Alignment.LEFT:
        return translate(*project_to_axes(to_bb.xmin - bb.xmin, 0, 0))
    elif alignment == Alignment.RIGHT:
        return translate(*project_to_axes(to_bb.xmax - bb.xmax, 0, 0))
    elif alignment == Alignment.BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb.ymax, 0))
    elif alignment == Alignment.FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb.ymin, 0))
    elif alignment == Alignment.TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb.zmax))
    elif alignment == Alignment.BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb.zmin))
    elif alignment == Alignment.CENTER:
        return translate(
            *project_to_axes(
                (to_bb.xmin + to_bb.xmax) / 2 - (bb.xmin + bb.xmax) / 2,
                (to_bb.ymin + to_bb.ymax) / 2 - (bb.ymin + bb.ymax) / 2,
                (to_bb.zmin + to_bb.zmax) / 2 - (bb.zmin + bb.zmax) / 2,
            )
        )
    elif alignment == Alignment.STACK_LEFT:
        return translate(*project_to_axes(to_bb.xmin - bb.xmin - part_width, 0, 0))
    elif alignment == Alignment.STACK_RIGHT:
        return translate(*project_to_axes(to_bb.xmax - bb.xmax + part_width, 0, 0))
    elif alignment == Alignment.STACK_BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb.ymax + part_length, 0))
    elif alignment == Alignment.STACK_FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb.ymin - part_length, 0))
    elif alignment == Alignment.STACK_TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb.zmax + part_height))
    elif alignment == Alignment.STACK_BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb.zmin - part_height))
    else:
        raise ValueError(f"Unknown alignment: {alignment}")


def align(
    part,
    to,
    alignment: Alignment,
    axes: Optional[List[int]] = None,
) -> Union[cq.Workplane, cq.Shape]:
    """
    Align one object to another and return the aligned copy.

    Args:
        part: The object to be aligned (will be copied)
        to: The target object to align to
        alignment: The type of alignment to perform
        axes: Optional list of axes to constrain alignment to (0=X, 1=Y, 2=Z)

    Returns:
        A copy of the part object aligned to the target
    """
    if isinstance(part, cq.Workplane):
        part_copy = part
    elif isinstance(part, cq.Shape):
        part_copy = part
    else:
        raise TypeError(f"Unsupported type for alignment: {type(part)}")

    return align_translation(part, to, alignment, axes=axes)(part_copy)
