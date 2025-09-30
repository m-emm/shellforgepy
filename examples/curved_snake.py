#!/usr/bin/env python3
"""
Curved Snake Example

Creates a curved channel following a sine wave pattern.
Demonstrates path-following capabilities with proper coordinate transformation.

Usage:
    python examples/curved_snake.py

Output:
    output/curved_snake.stl
"""

import os

import numpy as np
from shellforgepy.simple import create_trapezoidal_snake_geometry, write_stl_binary


def combine_segments(meshes):
    """
    Combine multiple mesh segments into a single mesh with proper indexing.

    Args:
        meshes: List of mesh dictionaries with 'vertexes' and 'faces' keys

    Returns:
        Tuple of (all_vertices, all_faces) dictionaries
    """
    all_vertices = {}
    all_faces = {}
    vertex_offset = 0
    face_offset = 0

    for mesh in meshes:
        # Add vertices with offset
        for vertex_id, vertex_pos in mesh["vertexes"].items():
            all_vertices[vertex_offset + vertex_id] = vertex_pos

        # Add faces with vertex offset
        for face_id, face_verts in mesh["faces"].items():
            offset_face_verts = [v + vertex_offset for v in face_verts]
            all_faces[face_offset + face_id] = offset_face_verts

        vertex_offset += len(mesh["vertexes"])
        face_offset += len(mesh["faces"])

    return all_vertices, all_faces


def main():
    """Create a curved snake following a sine wave pattern."""
    print("Creating curved sine wave snake...")

    # Trapezoidal cross-section for 3D printing
    cross_section = np.array(
        [
            [-5.0, 0.0],  # Bottom left (10mm wide)
            [5.0, 0.0],  # Bottom right (10mm wide)
            [2.5, 5.0],  # Top right (5mm wide, 5mm tall)
            [-2.5, 5.0],  # Top left (5mm wide, 5mm tall)
        ]
    )

    # Create sine wave path in X-Y plane
    num_points = 20
    x_values = np.linspace(0, 100, num_points)  # 100mm total length
    y_values = 15 * np.sin(2 * np.pi * x_values / 50)  # 15mm amplitude, 50mm wavelength
    z_values = np.zeros_like(x_values)  # Keep Z=0 for planar snake

    base_points = np.column_stack([x_values, y_values, z_values])

    # All normals point up in Z direction
    normals = np.zeros_like(base_points)
    normals[:, 2] = 1.0

    # Generate curved snake geometry
    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Combine all segments into one mesh
    all_vertices, all_faces = combine_segments(meshes)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "curved_snake.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"✅ Exported curved snake: {output_path}")
    print(f"   Length: 100mm sine wave, Amplitude: 15mm")
    print(f"   Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("\nGreat for decorative elements or organic-shaped channels!")


if __name__ == "__main__":
    main()
