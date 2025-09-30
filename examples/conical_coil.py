#!/usr/bin/env python3
"""
Conical Coil Example

Creates a conical helical coil with varying radius.
Demonstrates advanced helical path generation with changing radius.

Usage:
    python examples/conical_coil.py

Output:
    output/conical_coil.stl
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
    """Create a conical helical coil with varying radius."""
    print("Creating conical helical coil...")

    # Conical coil parameters
    base_radius = 40.0  # mm (bottom)
    top_radius = 20.0  # mm (top)
    pitch_per_turn = 20.0  # mm
    num_turns = 4
    points_per_turn = 20
    total_points = num_turns * points_per_turn

    # Trapezoidal cross-section optimized for 3D printing
    cross_section = np.array(
        [
            [-4.0, 0.0],  # Bottom left (8mm wide)
            [4.0, 0.0],  # Bottom right (8mm wide)
            [3.0, 6.0],  # Top right (6mm wide, 6mm tall)
            [-3.0, 6.0],  # Top left (6mm wide, 6mm tall)
        ]
    )

    # Generate helical path with varying radius (conical)
    theta_values = np.linspace(0, 2 * np.pi * num_turns, total_points)
    z_values = (pitch_per_turn / (2 * np.pi)) * theta_values

    # Linear interpolation of radius from base to top
    radius_values = base_radius + (top_radius - base_radius) * (z_values / z_values[-1])

    x_values = radius_values * np.cos(theta_values)
    y_values = radius_values * np.sin(theta_values)

    base_points = np.column_stack([x_values, y_values, z_values])

    # Calculate outward-pointing normals (radial from Z-axis)
    normals = np.zeros_like(base_points)
    for i in range(len(base_points)):
        normals[i, 0] = np.cos(theta_values[i])  # X component
        normals[i, 1] = np.sin(theta_values[i])  # Y component
        normals[i, 2] = 0.0  # Z component (purely radial)

    # Generate conical coil geometry
    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Combine all segments
    all_vertices, all_faces = combine_segments(meshes)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "conical_coil.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"✅ Exported conical coil: {output_path}")
    print(f"   Base radius: {base_radius}mm, Top radius: {top_radius}mm")
    print(
        f"   Height: {num_turns * pitch_per_turn}mm, Taper: {base_radius - top_radius}mm"
    )
    print(f"   Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("\nAdvanced geometry impossible with traditional CAD!")


if __name__ == "__main__":
    main()
