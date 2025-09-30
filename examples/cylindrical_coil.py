#!/usr/bin/env python3
"""
Cylindrical Coil Example

Creates a cylindrical helical coil with trapezoidal cross-section.
Perfect for LED strip coils, spring-like structures, or decorative spirals.

Usage:
    python examples/cylindrical_coil.py

Output:
    output/cylindrical_coil.stl
"""

import os

import numpy as np
from shellforgepy.simple import create_trapezoidal_snake_geometry, write_stl_binary


def combine_segments(meshes):
    """Combine multiple mesh segments into a single mesh with proper indexing."""
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
    """Create a cylindrical helical coil."""
    print("Creating cylindrical helical coil...")

    # Coil parameters
    coil_radius = 30.0  # mm
    pitch_per_turn = 15.0  # mm
    num_turns = 3
    points_per_turn = 16
    total_points = num_turns * points_per_turn

    # Trapezoidal cross-section for LED strip channel
    cross_section = np.array(
        [
            [-3.0, 0.0],  # Bottom left (6mm wide)
            [3.0, 0.0],  # Bottom right (6mm wide)
            [2.0, 4.0],  # Top right (4mm wide, 4mm tall)
            [-2.0, 4.0],  # Top left (4mm wide, 4mm tall)
        ]
    )

    # Generate helical path
    theta_values = np.linspace(0, 2 * np.pi * num_turns, total_points)
    x_values = coil_radius * np.cos(theta_values)
    y_values = coil_radius * np.sin(theta_values)
    z_values = (pitch_per_turn / (2 * np.pi)) * theta_values

    base_points = np.column_stack([x_values, y_values, z_values])

    # Calculate outward-pointing normals (radial direction from coil axis)
    normals = np.zeros_like(base_points)
    for i in range(len(base_points)):
        normals[i, 0] = np.cos(theta_values[i])  # X component
        normals[i, 1] = np.sin(theta_values[i])  # Y component
        normals[i, 2] = 0.0  # Z component (purely radial)

    # Generate coil geometry
    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Combine all segments
    all_vertices, all_faces = combine_segments(meshes)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "cylindrical_coil.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"✅ Exported cylindrical coil: {output_path}")
    print(f"   Radius: {coil_radius}mm, Height: {num_turns * pitch_per_turn}mm")
    print(f"   Turns: {num_turns}, Pitch: {pitch_per_turn}mm")
    print(f"   Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("\nPerfect for LED strip coils or decorative spirals!")


if __name__ == "__main__":
    main()
