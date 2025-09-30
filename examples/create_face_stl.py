#!/usr/bin/env python3
"""
Face Mesh Example

Creates complex organic face shapes with mesh partitioning for multi-part printing.
Demonstrates advanced mesh processing with front/back splitting and shell generation.

Usage:
    python examples/create_face_stl.py

Output:
    face_stl_output/face_m_front.stl
    face_stl_output/face_m_back.stl
    face_stl_output/face_m_complete.stl
"""

import os

import numpy as np
from shellforgepy.simple import (
    MeshPartition,
    PartitionableSpheroidTriangleMesh,
    TransformedRegionView,
    face_point_cloud,
    write_shell_maps_to_stl,
)


def create_face_mesh_stl(face_key="m", output_dir="output"):
    """
    Create 3D face mesh and export as STL files.

    Args:
        face_key: Face type to generate ("m" or "n")
        output_dir: Directory to save STL files
    """
    print(f"Creating face mesh for face type '{face_key}'...")

    # Step 1: Generate face point cloud
    print("Step 1: Generating face point cloud...")
    points, labels = face_point_cloud(face_key)
    print(f"Generated {len(points)} points")

    # Step 2: Create triangulated mesh from point cloud
    print("Step 2: Creating triangulated mesh...")
    mesh = PartitionableSpheroidTriangleMesh.from_point_cloud(
        points, vertex_labels=labels
    )
    print(
        f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
    )

    # Step 2.5: Scale mesh to reasonable 3D printing size
    print("Step 2.5: Scaling mesh for 3D printing...")

    # Calculate bounding box
    min_coords = np.min(mesh.vertices, axis=0)
    max_coords = np.max(mesh.vertices, axis=0)
    dimensions = max_coords - min_coords

    print(
        f"Original dimensions: {dimensions[0]:.4f} x {dimensions[1]:.4f} x {dimensions[2]:.4f} units"
    )

    # Find the longest dimension
    max_dimension = np.max(dimensions)

    # Scale to make the longest side 200mm
    target_size_mm = 200.0  # 200mm
    scale_factor = target_size_mm / max_dimension

    print(
        f"Scaling by factor {scale_factor:.2f} to make longest side {target_size_mm}mm"
    )

    # Apply scaling to mesh vertices
    mesh.vertices = mesh.vertices * scale_factor

    # Recalculate dimensions after scaling
    min_coords_scaled = np.min(mesh.vertices, axis=0)
    max_coords_scaled = np.max(mesh.vertices, axis=0)
    dimensions_scaled = max_coords_scaled - min_coords_scaled

    print(
        f"Scaled dimensions: {dimensions_scaled[0]:.1f} x {dimensions_scaled[1]:.1f} x {dimensions_scaled[2]:.1f} mm"
    )

    # Step 3: Partition the mesh into front and back halves (for mask-like splitting)
    print("Step 3: Partitioning mesh...")

    # Find the center point for partitioning along Z-axis (front/back)
    center_z = np.mean(mesh.vertices[:, 2])

    # Create a vertical plane at the center to split the face front/back
    plane_point = np.array([0.0, 0.0, center_z])
    plane_normal = np.array([0.0, 0.0, 1.0])  # Vertical plane (splits at Z)

    print(f"Partitioning at Z = {center_z:.2f}mm (front/back split for mask)")

    # Partition the mesh
    perforated_mesh, face_mapping = mesh.perforate_along_plane(
        plane_point, plane_normal
    )

    print(
        f"Perforated mesh has {len(perforated_mesh.vertices)} vertices and {len(perforated_mesh.faces)} faces"
    )

    # Step 4: Create mesh partition and region views
    print("Step 4: Creating mesh partition...")

    # Identify faces in front and back of the partition plane
    face_to_region_map = {}

    for face_idx, face in enumerate(perforated_mesh.faces):
        # Calculate face centroid
        face_vertices = perforated_mesh.vertices[face]
        centroid_z = np.mean(face_vertices[:, 2])

        if centroid_z > center_z:
            face_to_region_map[face_idx] = 0  # Front region (outer face/mask exterior)
        else:
            face_to_region_map[face_idx] = 1  # Back region (inner face/mask interior)

    # Create mesh partition
    partition = MeshPartition(perforated_mesh, face_to_region_map)

    # Count faces in each region
    front_face_count = sum(
        1 for region_id in face_to_region_map.values() if region_id == 0
    )
    back_face_count = sum(
        1 for region_id in face_to_region_map.values() if region_id == 1
    )

    print(f"Front region (ID 0): {front_face_count} faces (mask exterior)")
    print(f"Back region (ID 1): {back_face_count} faces (mask interior)")

    # Create region views for front and back halves
    front_region = TransformedRegionView(
        partition=partition, region_id=0, transform=np.eye(4)  # Identity transform
    )

    back_region = TransformedRegionView(
        partition=partition, region_id=1, transform=np.eye(4)  # Identity transform
    )

    # Step 5: Generate 3D printable shells
    print("Step 5: Generating 3D printable shells...")

    # Calculate appropriate shell thickness based on scale
    # For 200mm object, 2-3mm shell thickness is reasonable
    shell_thickness_mm = 2.5  # 2.5mm thick shell for 3D printing

    print(f"Using shell thickness: {shell_thickness_mm}mm")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and export front half
    if front_face_count > 0:
        print("  Processing front half...")
        front_shell_maps, _ = front_region.get_transformed_materialized_shell_maps(
            shell_thickness=shell_thickness_mm, shrinkage=0.0, smooth_inside=True
        )

        front_stl_path = os.path.join(output_dir, f"face_{face_key}_front.stl")
        write_shell_maps_to_stl(
            front_stl_path,
            front_shell_maps,
            header_text=f"Face {face_key} Front Half - 3D Printable",
            remove_inner_faces=True,
            merge_duplicate_vertices=True,
        )
        print(f"  Exported front half: {front_stl_path}")

    # Generate and export back half
    if back_face_count > 0:
        print("  Processing back half...")
        back_shell_maps, _ = back_region.get_transformed_materialized_shell_maps(
            shell_thickness=shell_thickness_mm,
            shrinkage=0.05,  # 5% shrinkage for better printing
            smooth_inside=True,
        )

        back_stl_path = os.path.join(output_dir, f"face_{face_key}_back.stl")
        write_shell_maps_to_stl(
            back_stl_path,
            back_shell_maps,
            header_text=f"Face {face_key} Back Half - 3D Printable",
            remove_inner_faces=True,
            merge_duplicate_vertices=True,
        )
        print(f"  Exported back half: {back_stl_path}")

    # Step 6: Also export the complete mesh for reference
    print("Step 6: Exporting complete mesh...")
    complete_shell_maps, _ = mesh.calculate_materialized_shell_maps(
        shell_thickness=shell_thickness_mm, shrinkage=0.05, smooth_inside=True
    )

    complete_stl_path = os.path.join(output_dir, f"face_{face_key}_complete.stl")
    write_shell_maps_to_stl(
        complete_stl_path,
        complete_shell_maps,
        header_text=f"Face {face_key} Complete - 3D Printable",
        remove_inner_faces=True,
        merge_duplicate_vertices=True,
    )
    print(f"  Exported complete mesh: {complete_stl_path}")

    print(f"\n✅ Success! Generated STL files in '{output_dir}' directory:")
    if front_face_count > 0:
        print(f"   - face_{face_key}_front.stl (front half)")
    if back_face_count > 0:
        print(f"   - face_{face_key}_back.stl (back half)")
    print(f"   - face_{face_key}_complete.stl (complete)")
    print("\nThese files can be:")
    print("  • Viewed in STL viewers (MeshLab, 3D Builder, online viewers)")
    print("  • 3D printed (shell thickness: 2mm)")
    print("  • Imported into CAD software for further editing")


def main():
    """Main function to demonstrate face mesh STL export."""
    print("🎭 Face Mesh STL Export Example")
    print("================================")

    # Define constants
    target_size_mm = 200.0  # 200mm
    shell_thickness_mm = 2.5  # 2.5mm thick shell

    try:
        # Create STL files for face type "m"
        create_face_mesh_stl(face_key="m", output_dir="face_stl_output")

        print(f"\n📊 Mesh Statistics:")

        # Load and analyze one of the generated files for stats
        points, labels = face_point_cloud("m")
        mesh_stats = PartitionableSpheroidTriangleMesh.from_point_cloud(
            points, vertex_labels=labels
        )

        # Apply the same scaling for consistent stats
        min_coords = np.min(mesh_stats.vertices, axis=0)
        max_coords = np.max(mesh_stats.vertices, axis=0)
        dimensions = max_coords - min_coords
        max_dimension = np.max(dimensions)
        scale_factor = target_size_mm / max_dimension
        mesh_stats.vertices = mesh_stats.vertices * scale_factor

        # Calculate final dimensions
        min_coords_final = np.min(mesh_stats.vertices, axis=0)
        max_coords_final = np.max(mesh_stats.vertices, axis=0)
        dimensions_final = max_coords_final - min_coords_final

        print(f"   • Original point cloud: {len(points)} points")
        print(
            f"   • Triangulated mesh: {len(mesh_stats.vertices)} vertices, {len(mesh_stats.faces)} faces"
        )
        print(
            f"   • Final dimensions: {dimensions_final[0]:.1f} x {dimensions_final[1]:.1f} x {dimensions_final[2]:.1f} mm"
        )
        print(f"   • Total surface area: {mesh_stats.total_area():.1f} mm²")
        print(f"   • Shell thickness: {shell_thickness_mm}mm")

        # Calculate approximate volume (for a closed mesh)
        center = np.mean(mesh_stats.vertices, axis=0)
        distances = [np.linalg.norm(v - center) for v in mesh_stats.vertices]
        avg_radius = np.mean(distances)
        approx_volume = (4 / 3) * np.pi * avg_radius**3
        print(f"   • Approximate volume: {approx_volume:.0f} mm³")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required dependencies are available.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
