import logging
from typing import List, Optional

import cadquery as cq
import numpy as np

_logger = logging.getLogger(__name__)


# cadquery specific adapter implementations
# Here, cad-backend-specific should be implemented
# Any code that can be implemented backend-agnostic should go in geometry/ or construct/ or produce/ or similar


def get_adapter_id():
    """Return a string identifying this adapter."""
    return "cadquery"


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
    return part.copy()


def translate_part(part, vector):
    """Translate a CadQuery part by the given vector."""
    _logger.info(f"Translating part by vector {vector}, part={part} , id={id(part)}")
    vec = cq.Vector(*map(float, vector))

    retval = part.translate(vec)
    _logger.info(f"Translated part id={id(retval)}")
    return retval


def rotate_part(part, angle, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!"
    # if something is needed like this, do it in reconstruct

    """Rotate a CadQuery part around the given axis."""

    if center is None:
        center = (0.0, 0.0, 0.0)
    if axis is None:
        axis = (0.0, 0.0, 1.0)
    center_vec = cq.Vector(*center)
    axis_vec = cq.Vector(*axis)
    rotate_retval = part.rotate(center_vec, center_vec + axis_vec, angle)
    if hasattr(part, "reconstruct"):
        return part.reconstruct(rotate_retval)
    else:
        return rotate_retval


def translate_part_native(part, *args):
    """Translate using native CadQuery signature. Used by composite objects."""
    _logger.info(
        f"Native translating part by vector {args}, part={part} , id={id(part)}"
    )
    translate_retval = part.translate(*args)
    if hasattr(part, "reconstruct"):
        _logger.info(
            f"Reconstructing part {part} , id={id(part)}, translated id={id(translate_retval)}"
        )
        return part.reconstruct(translate_retval)
    else:
        _logger.info(
            f"Not reconstructing part {part} , id={id(part)}, translated id={id(translate_retval)}"
        )
        return translate_retval


def rotate_part_native(part, v1, v2, angle):
    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!
    # if something is needed like this, do it in reconstruct

    rotation_retval = part.rotate(v1, v2, angle)
    if hasattr(part, "reconstruct"):
        return part.reconstruct(rotation_retval)
    else:
        return rotation_retval


def fuse_parts(part1, part2):
    """Fuse two CadQuery parts together."""
    return part1.fuse(part2)


def cut_parts(part1, part2):
    """Cut part2 from part1."""
    return part1.cut(part2)


def create_hex_prism(diameter, height, origin=(0, 0, 0)):
    """Create a hexagonal prism."""
    hex_prism = cq.Workplane("XY").polygon(6, diameter).extrude(height).val()
    return translate_part(hex_prism, origin)


def create_extruded_polygon(points, thickness):
    """Create an extruded polygon from a list of (x, y) coordinates."""
    # Convert to CadQuery points and create wire
    cq_points = [(x, y) for x, y in points]
    workplane = cq.Workplane("XY")
    wire = workplane.polyline(cq_points).close()
    return wire.extrude(thickness).val()


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
    from shellforgepy.construct.alignment import Alignment

    # Create the basic box workplane
    box_wp = cq.Workplane("XY").box(length, width, height)

    # If no specific alignment is given, fillet all edges
    if fillets_at is None and no_fillets_at is None:
        try:
            return box_wp.edges().fillet(fillet_radius).val()
        except Exception:
            return box_wp.val()

    # If empty fillets_at list, return unfilleted box
    if fillets_at is not None and len(fillets_at) == 0:
        return box_wp.val()

    try:
        result_wp = box_wp

        # Handle fillets_at case - only fillet specified alignments
        if fillets_at is not None:
            for alignment in fillets_at:
                try:
                    if alignment == Alignment.TOP:
                        # Fillet edges on the top face (+Z direction)
                        result_wp = result_wp.faces("+Z").edges().fillet(fillet_radius)
                    elif alignment == Alignment.BOTTOM:
                        # Fillet edges on the bottom face (-Z direction)
                        result_wp = result_wp.faces("-Z").edges().fillet(fillet_radius)
                    elif alignment == Alignment.LEFT:
                        # Fillet edges on the left face (-X direction)
                        result_wp = result_wp.faces("-X").edges().fillet(fillet_radius)
                    elif alignment == Alignment.RIGHT:
                        # Fillet edges on the right face (+X direction)
                        result_wp = result_wp.faces("+X").edges().fillet(fillet_radius)
                    elif alignment == Alignment.FRONT:
                        # Fillet edges on the front face (-Y direction)
                        result_wp = result_wp.faces("-Y").edges().fillet(fillet_radius)
                    elif alignment == Alignment.BACK:
                        # Fillet edges on the back face (+Y direction)
                        result_wp = result_wp.faces("+Y").edges().fillet(fillet_radius)
                except Exception:
                    # Continue if this alignment fails
                    continue

            return result_wp.val()

        # Handle no_fillets_at case - this is complex with CadQuery
        # For now, just do all edges (can be improved later)
        elif no_fillets_at is not None:
            # For simplicity, fillet all edges for now
            # TODO: Implement proper exclusion logic
            try:
                return box_wp.edges().fillet(fillet_radius).val()
            except Exception:
                return box_wp.val()

        else:
            # Default case - fillet all edges
            return box_wp.edges().fillet(fillet_radius).val()

    except Exception:
        # If all filleting fails, return the original box
        return box_wp.val()


def get_volume(solid):
    """Get the volume of a CadQuery solid."""
    return solid.Volume()


def filter_edges_by_z_position(solid, z_threshold, below=True):
    """Filter edges based on their Z position.

    Args:
        solid: CadQuery solid
        z_threshold: Z coordinate threshold
        below: If True, return edges with all vertices <= threshold;
               if False, return edges with all vertices >= threshold

    Returns:
        List of edges that meet the criteria
    """
    edges = []
    try:
        for edge in solid.Edges():
            # Get edge's vertices
            vertices = edge.Vertices()
            if vertices:
                # Check if all vertices meet the criteria
                vertex_z_coords = [v.Z for v in vertices]

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
        solid: CadQuery solid
        fillets_at: List of Alignment values indicating which faces/edges to include
        no_fillets_at: List of Alignment values indicating which faces/edges to exclude

    Returns:
        List of edges that meet the criteria
    """
    from shellforgepy.construct.alignment import Alignment

    # Get bounding box for alignment calculations
    bbox = solid.BoundingBox()
    length = bbox.xlen
    width = bbox.ylen
    height = bbox.zlen
    x_min, y_min, z_min = bbox.xmin, bbox.ymin, bbox.zmin

    def edge_is_at(edge, alignment):
        """Check if an edge is at a specific alignment position."""
        tolerance = 1e-3

        # Get edge bounding box for circular edges
        edge_bbox = edge.BoundingBox()

        # For circular edges, check if the edge is at constant Z/X/Y within tolerance
        if alignment == Alignment.TOP:
            # Edge is at top if its Z range is at the top of the solid
            return (
                abs(edge_bbox.zmax - (z_min + height)) < tolerance
                and abs(edge_bbox.zmin - (z_min + height)) < tolerance
            )
        elif alignment == Alignment.BOTTOM:
            # Edge is at bottom if its Z range is at the bottom of the solid
            return (
                abs(edge_bbox.zmax - z_min) < tolerance
                and abs(edge_bbox.zmin - z_min) < tolerance
            )
        elif alignment == Alignment.LEFT:
            # Edge is at left if its X range is at the left of the solid
            return (
                abs(edge_bbox.xmax - x_min) < tolerance
                and abs(edge_bbox.xmin - x_min) < tolerance
            )
        elif alignment == Alignment.RIGHT:
            # Edge is at right if its X range is at the right of the solid
            return (
                abs(edge_bbox.xmax - (x_min + length)) < tolerance
                and abs(edge_bbox.xmin - (x_min + length)) < tolerance
            )
        elif alignment == Alignment.FRONT:
            # Edge is at front if its Y range is at the front of the solid
            return (
                abs(edge_bbox.ymax - y_min) < tolerance
                and abs(edge_bbox.ymin - y_min) < tolerance
            )
        elif alignment == Alignment.BACK:
            # Edge is at back if its Y range is at the back of the solid
            return (
                abs(edge_bbox.ymax - (y_min + width)) < tolerance
                and abs(edge_bbox.ymin - (y_min + width)) < tolerance
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
        for edge in solid.Edges():
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
        solid: CadQuery solid
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

        for edge in solid.Edges():
            vertices = edge.Vertices()
            if len(vertices) >= 2:
                v0 = vertices[0]
                v1 = vertices[1]

                # Convert vertices to tuples
                v0_point = (v0.X, v0.Y, v0.Z)
                v1_point = (v1.X, v1.Y, v1.Z)

                # Call user's filter function
                if edge_filter_func(bbox, v0_point, v1_point):
                    edges.append(edge)

    except Exception as e:
        _logger.warning(f"Error filtering edges by function: {e}")

    return edges


def apply_fillet_to_edges(solid, fillet_radius, edges):
    """Apply fillet to specific edges of a solid.

    Args:
        solid: CadQuery solid
        fillet_radius: Radius of the fillet
        edges: List of edges to fillet

    Returns:
        Filleted solid
    """
    if not edges:
        return solid

    try:
        # Convert to CadQuery workplane and apply fillet
        wp = cq.Workplane().add(solid)
        # Use the edges directly for filleting
        result = solid.fillet(fillet_radius, edges)
        return result
    except Exception as e:
        _logger.warning(f"Error applying fillet: {e}")
        return solid


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
    edges = filter_edges_by_alignment(solid, fillets_at, no_fillets_at)
    return apply_fillet_to_edges(solid, fillet_radius, edges)


def apply_fillet_by_function(solid, fillet_radius, edge_filter_func):
    """Apply fillet to edges selected by a custom function.

    Args:
        solid: CadQuery solid
        fillet_radius: Radius of the fillet
        edge_filter_func: Function that takes (bbox, v0_point, v1_point) and returns bool

    Returns:
        Filleted solid
    """
    edges = filter_edges_by_function(solid, edge_filter_func)
    return apply_fillet_to_edges(solid, fillet_radius, edges)
