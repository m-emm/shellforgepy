#!/usr/bin/env python3
"""
Cylinder Mesh Example

Creates cylinder point clouds, triangulated meshes, and exports as STL.
Demonstrates basic mesh generation workflow with 4 cylinders arranged in a row.

Usage:
    python examples/create_cylinder_stl.py

Output:
    output/cylinder_mesh.stl
"""

import os

import numpy as np
from shellforgepy.shells.transformed_region_view import TransformedRegionView
from shellforgepy.simple import (
    MeshPartition,
    PartitionableSpheroidTriangleMesh,
    merge_meshes,
    write_stl_binary,
)


def generate_cylinder_point_cloud(
    radius=50.0, height=100.0, angular_resolution=20, num_vertical=30, num_top_bottom=10
):
    """
    Generate a point cloud representing a cylinder.

    Args:
        radius: Cylinder radius in mm
        height: Cylinder height in mm
        num_radial: Number of points around the circumference
        num_vertical: Number of vertical layers on the side
        num_top_bottom: Number of points on top and bottom faces

    Returns:
        numpy array of 3D points
    """
    points = []

    # Generate points on the cylindrical surface
    for i in range(num_vertical):
        z = (i / (num_vertical - 1)) * height  # From 0 to height
        for j in range(angular_resolution):
            theta = (j / angular_resolution) * 2 * np.pi
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])

    # Generate points on the top face (z = height)
    for i in range(num_top_bottom):
        for j in range(angular_resolution):
            # Create concentric circles on the top face
            r = ((i + 1) / num_top_bottom) * radius
            theta = (j / angular_resolution) * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = height
            points.append([x, y, z])

    # Add center point on top
    points.append([0, 0, height])

    # Generate points on the bottom face (z = 0)
    for i in range(num_top_bottom):
        for j in range(angular_resolution):
            # Create concentric circles on the bottom face
            r = ((i + 1) / num_top_bottom) * radius
            theta = (j / angular_resolution) * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0
            points.append([x, y, z])

    # Add center point on bottom
    points.append([0, 0, 0])

    return np.array(points)


def create_cylinder_mesh_stl(output_dir="output"):
    """
    Create a cylinder mesh and export as STL file.

    Args:
        output_dir: Directory to save STL file
    """
    print("Creating cylinder mesh...")

    # Step 1: Generate cylinder point cloud
    print("Step 1: Generating cylinder point cloud...")
    radius = 5.0
    height = 60.0

    points = generate_cylinder_point_cloud(
        radius, height, num_vertical=2, angular_resolution=80, num_top_bottom=2
    )
    print(
        f"Generated {len(points)} points for cylinder (radius={radius}mm, height={height}mm)"
    )

    # Step 2: Create triangulated mesh from point cloud
    print("Step 2: Creating triangulated mesh...")
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(points)
    print(
        f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
    )

    # Step 3: Create a single partition covering the whole mesh
    print("Step 3: Creating mesh partition...")

    # Create a simple partition where all faces belong to region 0
    face_to_region_map = {face_idx: 0 for face_idx in range(len(mesh.faces))}

    # Create mesh partition
    partition = MeshPartition(mesh, face_to_region_map)

    view = TransformedRegionView(
        partition=partition, region_id=0, transform=np.eye(4)  # Identity transform
    )

    num_cylinders = 4

    meshes = []

    for c in range(num_cylinders):
        transformed_view = view.translated((c * (radius * 2 + 10), 0, 0))
        v, f, e = transformed_view.get_transformed_vertices_faces_boundary_edges()

        meshes.append((v, f))

    current_mesh = None

    for v, f in meshes:
        if current_mesh is None:
            current_mesh = (v, f)
        else:
            current_mesh = merge_meshes(current_mesh[0], current_mesh[1], v, f, 1e-6)

    print(f"Created partition with {len(face_to_region_map)} faces in region 0")

    # Step 4: Export the complete mesh as STL
    print("Step 4: Exporting mesh as STL...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get the mesh vertices and faces directly
    vertices = current_mesh[0]
    faces = current_mesh[1]

    # Export to STL
    output_path = os.path.join(output_dir, "cylinder_mesh.stl")
    write_stl_binary(
        path=output_path,
        vertices=vertices,
        triangles=faces,
        header_text="Cylinder mesh created by shellforgepy",
    )

    print(f"Successfully exported cylinder mesh to: {output_path}")
    print(f"Mesh statistics:")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Triangles: {len(faces)}")

    # Calculate and print mesh bounds
    vertices_np = np.array(vertices)
    min_coords = np.min(vertices_np, axis=0)
    max_coords = np.max(vertices_np, axis=0)
    dimensions = max_coords - min_coords

    print(
        f"  - Dimensions: {dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f} mm"
    )
    print(
        f"  - Bounds: X[{min_coords[0]:.1f}, {max_coords[0]:.1f}] Y[{min_coords[1]:.1f}, {max_coords[1]:.1f}] Z[{min_coords[2]:.1f}, {max_coords[2]:.1f}]"
    )


if __name__ == "__main__":
    create_cylinder_mesh_stl()
