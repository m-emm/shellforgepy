#!/usr/bin/env python3
"""
Straight Snake Example

Creates a simple straight channel with trapezoidal cross-section.
Perfect for LED strip channels or cable management.

Usage:
    python examples/straight_snake.py

Output:
    output/straight_snake.stl
"""

import os

import numpy as np
from shellforgepy.simple import create_trapezoidal_snake_geometry, write_stl_binary


def main():
    """Create a straight snake with trapezoidal cross-section."""
    print("Creating straight trapezoidal snake...")

    # Define trapezoidal cross-section (wider at bottom, narrower at top)
    cross_section = np.array(
        [
            [-5.0, 0.0],  # Bottom left (10mm wide)
            [5.0, 0.0],  # Bottom right (10mm wide)
            [2.5, 5.0],  # Top right (5mm wide, 5mm tall)
            [-2.5, 5.0],  # Top left (5mm wide, 5mm tall)
        ]
    )

    # Create straight path along X-axis
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ]
    )

    # Z normals (pointing up)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Generate the geometry
    meshes = create_trapezoidal_snake_geometry(cross_section, base_points, normals)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Export STL
    mesh = meshes[0]
    vertices_list = [mesh["vertexes"][i] for i in range(len(mesh["vertexes"]))]
    triangles_list = [tuple(face_verts) for face_verts in mesh["faces"].values()]

    output_path = os.path.join(output_dir, "straight_snake.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"✅ Exported straight snake: {output_path}")
    print(f"   Length: 50mm, Cross-section: 10mm x 5mm trapezoid")
    print(f"   Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("\nPerfect for LED strip channels or cable management!")


if __name__ == "__main__":
    main()
