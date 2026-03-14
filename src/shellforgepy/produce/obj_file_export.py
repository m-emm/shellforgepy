from typing import Dict, Optional, Tuple


def _vertex_to_xyz(vertex) -> tuple[float, float, float]:
    """Convert supported vertex representations to an XYZ float triple."""
    if hasattr(vertex, "toTuple"):
        coords = vertex.toTuple()
    elif hasattr(vertex, "x") and hasattr(vertex, "y") and hasattr(vertex, "z"):
        coords = (vertex.x, vertex.y, vertex.z)
    elif hasattr(vertex, "X") and hasattr(vertex, "Y") and hasattr(vertex, "Z"):
        coords = (vertex.X, vertex.Y, vertex.Z)
    else:
        coords = vertex

    return (float(coords[0]), float(coords[1]), float(coords[2]))


def _triangle_to_indices(triangle) -> tuple[int, int, int]:
    """Convert supported triangle representations to integer indices."""
    return (int(triangle[0]), int(triangle[1]), int(triangle[2]))


def _color_to_rgb(color) -> tuple[float, float, float]:
    """Normalize material color inputs to RGB float triples."""
    return (float(color[0]), float(color[1]), float(color[2]))


def _animation_to_xyz(vector) -> tuple[float, float, float]:
    """Normalize animation vectors to XYZ float triples."""
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _write_animation_comments(file_obj, animation) -> None:
    if not animation:
        return

    for animation_key, vector in animation.items():
        x, y, z = _animation_to_xyz(vector)
        file_obj.write(f"# shellforgepy_anim {animation_key} {x} {y} {z}\n")


def _write_mtl_file(
    path: str,
    materials: Dict[str, Tuple[float, float, float]],
) -> None:
    """Write an MTL material library file.

    Args:
        path: Path to write the MTL file to.
        materials: Dictionary mapping material names to RGB color tuples (0.0-1.0).
    """
    with open(path, "w") as f:
        f.write("# MTL file exported by ShellForgePy\n\n")

        for mat_name, color in materials.items():
            r, g, b = color
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Ka {r*0.2:.6f} {g*0.2:.6f} {b*0.2:.6f}\n")  # Ambient
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")  # Diffuse
            f.write(f"Ks 0.500000 0.500000 0.500000\n")  # Specular
            f.write(f"Ns 96.078431\n")  # Specular exponent
            f.write(f"Ni 1.000000\n")  # Optical density
            f.write(f"d 1.000000\n")  # Dissolve (opacity)
            f.write(f"illum 2\n\n")  # Illumination model


def export_mesh_to_obj(
    vertices,
    triangles,
    destination: str,
    *,
    color: Optional[Tuple[float, float, float]] = None,
    material_name: str = "material_0",
) -> None:
    """Export a mesh to an OBJ file with optional color via MTL.

    Args:
        vertices: Vertex rows or vertex objects with coordinate access.
        triangles: Triangles as integer index triplets.
        destination: Path to write the OBJ file to.
        color: Optional RGB color tuple (0.0-1.0 range). If provided, creates an MTL file.
        material_name: Name of the material in the MTL file.
    """
    import os

    destination = str(destination)

    # Determine MTL file path if color is provided
    mtl_path = None
    mtl_filename = None
    if color is not None:
        base, _ = os.path.splitext(destination)
        mtl_path = base + ".mtl"
        mtl_filename = os.path.basename(mtl_path)

    # Write OBJ file
    with open(destination, "w") as f:
        f.write("# OBJ file exported by ShellForgePy\n")

        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")

        # Write vertices
        for v in vertices:
            x, y, z = _vertex_to_xyz(v)
            f.write(f"v {x} {y} {z}\n")

        # Use material if provided
        if color is not None:
            f.write(f"usemtl {material_name}\n")

        # Write faces (OBJ uses 1-based indexing)
        for tri in triangles:
            i0, i1, i2 = _triangle_to_indices(tri)
            f.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")

    # Write MTL file if color is provided
    if mtl_path and color is not None:
        _write_mtl_file(mtl_path, {material_name: _color_to_rgb(color)})


def export_colored_meshes_to_obj(
    meshes,
    destination: str,
) -> None:
    """Export multiple parts with different colors to a single OBJ file.

    Args:
        meshes: List of tuples `(vertices, triangles, name, color)` or
            `(vertices, triangles, name, color, animation)` where:
            - vertices: Vertex rows or vertex objects with coordinate access
            - triangles: Triangles as integer index triplets
            - name: Part/material name (used as material identifier)
            - color: RGB tuple (0.0-1.0 range)
            - animation: Optional dict mapping animation key to XYZ vector
        destination: Path to write the OBJ file to.
    """
    import os

    destination = str(destination)
    base, _ = os.path.splitext(destination)
    mtl_path = base + ".mtl"
    mtl_filename = os.path.basename(mtl_path)

    materials = {}
    vertex_offset = 0

    with open(destination, "w") as f:
        f.write("# OBJ file exported by ShellForgePy\n")
        f.write(f"mtllib {mtl_filename}\n\n")

        for mesh in meshes:
            if len(mesh) == 4:
                vertices, triangles, name, color = mesh
                animation = None
            elif len(mesh) == 5:
                vertices, triangles, name, color, animation = mesh
            else:
                raise ValueError(
                    "Each mesh entry must contain 4 or 5 values: "
                    "(vertices, triangles, name, color[, animation])"
                )

            # Sanitize material name (OBJ material names shouldn't have spaces)
            mat_name = name.replace(" ", "_").replace("/", "_")
            materials[mat_name] = _color_to_rgb(color)

            f.write(f"# Object: {name}\n")
            f.write(f"o {mat_name}\n")
            _write_animation_comments(f, animation)

            # Write vertices
            normalized_vertices = [_vertex_to_xyz(v) for v in vertices]
            normalized_triangles = [_triangle_to_indices(tri) for tri in triangles]

            for x, y, z in normalized_vertices:
                f.write(f"v {x} {y} {z}\n")

            # Use material
            f.write(f"usemtl {mat_name}\n")

            # Write faces (OBJ uses 1-based indexing, offset by previous vertices)
            for i0, i1, i2 in normalized_triangles:
                f.write(
                    f"f {i0 + 1 + vertex_offset} {i1 + 1 + vertex_offset} {i2 + 1 + vertex_offset}\n"
                )

            vertex_offset += len(normalized_vertices)
            f.write("\n")

    # Write MTL file
    _write_mtl_file(mtl_path, materials)
