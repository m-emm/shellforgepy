#!/usr/bin/env python3
"""
Möbius Strip Example

Creates a mathematical Möbius strip - a surface with only one side!
The ultimate demonstration of coordinate transformation capabilities.

Usage:
    python examples/mobius_strip.py

Output:
    output/mobius_strip.stl
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
    """Create a Möbius strip - a surface with only one side!"""
    print("Creating Möbius strip - topological marvel!")

    # Möbius strip parameters
    radius = 40.0  # mm - radius of the circular path
    num_points = 80  # Many points for smooth curves and proper closure

    # CRITICAL: Cross-section MUST be centered at (0,0) for proper closure!
    # Thin rectangular strip - 20mm wide, 2mm thick, centered on origin
    cross_section = np.array(
        [
            [-10.0, -1.0],  # Bottom left
            [10.0, -1.0],  # Bottom right
            [10.0, 1.0],  # Top right
            [-10.0, 1.0],  # Top left
        ]
    )

    print("Möbius strip cross-section (CENTERED at origin):")
    for i, pt in enumerate(cross_section):
        print(f"  Point {i}: {pt}")

    # Generate circular path in X-Y plane
    theta_values = np.linspace(
        0, 2 * np.pi, num_points, endpoint=False
    )  # Don't duplicate endpoint
    x_values = radius * np.cos(theta_values)
    y_values = radius * np.sin(theta_values)
    z_values = np.zeros_like(theta_values)  # Keep in Z=0 plane

    base_points = np.column_stack([x_values, y_values, z_values])

    print(f"Created circular path with {num_points} points")
    print(f"Circle radius: {radius}mm")

    # THE MAGIC: Normals rotate by 180° over one full circle!
    # This creates the Möbius twist - after one revolution, the normal
    # has flipped, creating a surface with only one side!
    normals = np.zeros_like(base_points)

    for i in range(len(base_points)):
        # Normal rotation angle: 180° over full circle (π radians)
        normal_rotation = (
            theta_values[i] / (2 * np.pi)
        ) * np.pi  # 0 to π over full circle

        # Rotate the Z-normal around the tangent axis by normal_rotation angle
        # This creates the Möbius twist!
        normals[i, 0] = 0  # X component (will be calculated by coordinate transform)
        normals[i, 1] = 0  # Y component (will be calculated by coordinate transform)
        normals[i, 2] = np.cos(normal_rotation)  # Z component varies from 1 to -1

        # Add the radial component that varies with the twist
        radial_component = np.sin(normal_rotation)
        normals[i, 0] = radial_component * np.cos(theta_values[i])  # Radial X
        normals[i, 1] = radial_component * np.sin(theta_values[i])  # Radial Y

    print("Möbius twist: Normals rotate 180° over one revolution")
    print("This creates a surface with only ONE SIDE!")

    # Generate Möbius strip geometry with loop closure
    meshes = create_trapezoidal_snake_geometry(
        cross_section, base_points, normals, close_loop=True
    )

    # Combine all segments
    all_vertices, all_faces = combine_segments(meshes)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "mobius_strip.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"✅ Exported Möbius strip: {output_path}")
    print(f"   Radius: {radius}mm, Width: 20mm, Thickness: 2mm")
    print(f"   Points: {num_points} (ensures smooth closure)")
    print(f"   Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("\n🔥 Mathematical marvel: ONE-SIDED SURFACE!")
    print("   PROPERLY CLOSED LOOP: No gaps at the seam!")
    print(
        "   Try tracing the surface with your finger - you'll end up where you started!"
    )


if __name__ == "__main__":
    main()
