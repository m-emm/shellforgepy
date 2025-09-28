#!/usr/bin/env python3
"""
Example script demonstrating trapezoidal snake geometry creation.

This script demonstrates the complete workflow for creating complex 3D geometries
using the trapezoidal snake system:
1. Create straight snakes with trapezoidal cross-sections
2. Generate curved snakes following sine waves
3. Build helical coils (cylindrical and conical)
4. Export all geometries as STL files ready for 3D printing

The trapezoidal snake system excels at creating LED strip channels, cable ducts,
decorative moldings, and any geometry that follows a path with consistent
cross-section orientation.
"""

import os

import numpy as np
from shellforgepy.geometry.mesh_utils import write_stl_binary
from shellforgepy.geometry.treapezoidal_snake_geometry import (
    create_trapezoidal_snake_geometry,
)


def create_straight_snake_stl(output_dir="output"):
    """
    Create a straight snake with trapezoidal cross-section.

    Demonstrates basic usage of the trapezoidal snake system.
    Perfect for LED strip channels or cable management.

    Args:
        output_dir: Directory to save STL files
    """
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

    # Export STL
    mesh = meshes[0]
    vertices_list = [mesh["vertexes"][i] for i in range(len(mesh["vertexes"]))]
    triangles_list = [tuple(face_verts) for face_verts in mesh["faces"].values()]

    output_path = os.path.join(output_dir, "straight_snake.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"Exported straight snake: {output_path}")
    print(f"  - Length: 50mm, Cross-section: 10mm x 5mm trapezoid")
    print(f"  - Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")


def create_curved_snake_stl(output_dir="output"):
    """
    Create a curved snake following a sine wave pattern.

    Demonstrates path-following capabilities with proper coordinate transformation.
    Great for decorative elements or organic-shaped channels.

    Args:
        output_dir: Directory to save STL files
    """
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

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "curved_snake.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"Exported curved snake: {output_path}")
    print(f"  - Length: 100mm sine wave, Amplitude: 15mm")
    print(f"  - Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")


def create_cylindrical_coil_stl(output_dir="output"):
    """
    Create a cylindrical helical coil.

    Demonstrates helical path generation with outward-pointing normals.
    Perfect for LED strip coils, spring-like structures, or decorative spirals.

    Args:
        output_dir: Directory to save STL files
    """
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

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "cylindrical_coil.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"Exported cylindrical coil: {output_path}")
    print(f"  - Radius: {coil_radius}mm, Height: {num_turns * pitch_per_turn}mm")
    print(f"  - Turns: {num_turns}, Pitch: {pitch_per_turn}mm")
    print(f"  - Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")


def create_conical_coil_stl(output_dir="output"):
    """
    Create a conical helical coil with varying radius.

    Demonstrates advanced capability: helical path with changing radius.

    Args:
        output_dir: Directory to save STL files
    """
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

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "conical_coil.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"Exported conical coil: {output_path}")
    print(f"  - Base radius: {base_radius}mm, Top radius: {top_radius}mm")
    print(
        f"  - Height: {num_turns * pitch_per_turn}mm, Taper: {base_radius - top_radius}mm"
    )
    print(f"  - Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("  - This geometry was IMPOSSIBLE with previous implementations!")


def create_mobius_strip_stl(output_dir="output"):
    """
    Create a Möbius strip - the ultimate demonstration of coordinate transformation!

    This creates a mathematical Möbius strip by following a circle while rotating
    the normal vector by 180 degrees over one complete revolution. The result is
    a surface with only one side and one edge - a topological marvel!

    CRITICAL: Cross-section must be centered at (0,0) so that after one full
    rotation (180° normal twist), the geometry closes perfectly on itself.

    Args:
        output_dir: Directory to save STL files
    """
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

    # Export STL
    vertices_list = [all_vertices[i] for i in range(len(all_vertices))]
    triangles_list = [tuple(face_verts) for face_verts in all_faces.values()]

    output_path = os.path.join(output_dir, "mobius_strip.stl")
    write_stl_binary(output_path, vertices_list, triangles_list)

    print(f"Exported Möbius strip: {output_path}")
    print(f"  - Radius: {radius}mm, Width: 20mm, Thickness: 2mm")
    print(f"  - Points: {num_points} (ensures smooth closure)")
    print(f"  - Vertices: {len(vertices_list)}, Triangles: {len(triangles_list)}")
    print("  - Mathematical marvel: ONE-SIDED SURFACE!")
    print("  - PROPERLY CLOSED LOOP: No gaps at the seam!")
    print(
        "  - Try tracing the surface with your finger - you'll end up where you started!"
    )
    print("  - This demonstrates the ultimate power of coordinate transformation!")


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


def create_all_examples(output_dir="output"):
    """
    Create all trapezoidal snake examples and export as STL files.

    This function demonstrates the full capabilities of the trapezoidal snake
    geometry system, from simple straight channels to complex conical coils.

    Args:
        output_dir: Directory to save all STL files
    """
    print("=== Trapezoidal Snake Geometry Examples ===")
    print("Creating comprehensive set of 3D printable examples...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate all examples
    print("\n1. Straight Snake (Basic Example)")
    create_straight_snake_stl(output_dir)

    print("\n2. Curved Snake (Sine Wave Path)")
    create_curved_snake_stl(output_dir)

    print("\n3. Cylindrical Coil (Classic Helix)")
    create_cylindrical_coil_stl(output_dir)

    print("\n4. Conical Coil (Advanced Geometry)")
    create_conical_coil_stl(output_dir)

    print("\n5. Möbius Strip (Mathematical Marvel)")
    create_mobius_strip_stl(output_dir)

    print("\n=== Summary ===")
    print("All examples exported successfully!")
    print("Key capabilities demonstrated:")
    print("  ✓ Straight paths with trapezoidal cross-sections")
    print("  ✓ Curved paths following mathematical functions")
    print("  ✓ Helical coils with radial normal orientation")
    print("  ✓ Advanced conical coils with varying radius")
    print("  ✓ Möbius strips with rotating normals (TOPOLOGICAL MARVEL!)")
    print("  ✓ Proper coordinate transformation for all geometries")
    print("\nPerfect for:")
    print("  - LED strip channels and mounting systems")
    print("  - Cable management and wire routing")
    print("  - Decorative moldings and trim pieces")
    print("  - Custom coils and spiral structures")
    print("  - Mathematical models and educational demonstrations")
    print("  - Any geometry following a path with consistent cross-section")
    print(f"\nAll STL files are ready for 3D printing and saved in: {output_dir}")


if __name__ == "__main__":
    create_all_examples()
