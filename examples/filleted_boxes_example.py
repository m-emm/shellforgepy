#!/usr/bin/env python3
"""
Filleted Boxes Example

Demonstrates create_filleted_box functionality with 12 different fillet configurations
and arranges them on a 200x200mm build plate for 3D printing.

Usage:
    python examples/filleted_boxes_example.py

Output:
    Individual STL files: filleted_boxes_example_*.stl
    Combined layout: filleted_boxes_example.stl
    Process data: filleted_boxes_example_process.json
"""

import time

from shellforgepy.simple import (
    Alignment,
    PartList,
    arrange_and_export_parts,
    create_filleted_box,
)


def main():
    start_time = time.time()

    # Create a list to hold all our parts
    parts = PartList()

    # Build plate dimensions
    bed_width = 200  # mm
    part_gap = 5  # mm gap between parts

    # Box dimensions
    box_size = 20  # mm
    fillet_radius = 3  # mm

    print("Creating filleted box examples...")

    # 1. Basic box with all edges filleted
    all_filleted = create_filleted_box(box_size, box_size, box_size, fillet_radius)
    parts.add(all_filleted, "all_edges_filleted")

    # 2. Box with no fillets (for comparison)
    no_fillets = create_filleted_box(
        box_size, box_size, box_size, fillet_radius, fillets_at=[]
    )
    parts.add(no_fillets, "no_fillets")

    # 3. Box with only top edges filleted
    top_only = create_filleted_box(
        box_size, box_size, box_size, fillet_radius, fillets_at=[Alignment.TOP]
    )
    parts.add(top_only, "top_edges_only")

    # 4. Box with only bottom edges filleted
    bottom_only = create_filleted_box(
        box_size, box_size, box_size, fillet_radius, fillets_at=[Alignment.BOTTOM]
    )
    parts.add(bottom_only, "bottom_edges_only")

    # 5. Box with left and right edges filleted
    left_right = create_filleted_box(
        box_size,
        box_size,
        box_size,
        fillet_radius,
        fillets_at=[Alignment.LEFT, Alignment.RIGHT],
    )
    parts.add(left_right, "left_right_edges")

    # 6. Box with front and back edges filleted
    front_back = create_filleted_box(
        box_size,
        box_size,
        box_size,
        fillet_radius,
        fillets_at=[Alignment.FRONT, Alignment.BACK],
    )
    parts.add(front_back, "front_back_edges")

    # 7. Box with top, front, and right edges filleted
    top_front_right = create_filleted_box(
        box_size,
        box_size,
        box_size,
        fillet_radius,
        fillets_at=[Alignment.TOP, Alignment.FRONT, Alignment.RIGHT],
    )
    parts.add(top_front_right, "top_front_right")

    # 8. Large box with small fillet
    large_small_fillet = create_filleted_box(
        box_size * 1.5, box_size * 1.5, box_size, 1.0
    )
    parts.add(large_small_fillet, "large_small_fillet")

    # 9. Small box with large fillet
    small_large_fillet = create_filleted_box(
        box_size * 0.7, box_size * 0.7, box_size, 4.0
    )
    parts.add(small_large_fillet, "small_large_fillet")

    # 10. Rectangular box with selective fillets
    rectangular = create_filleted_box(
        box_size * 1.8,
        box_size * 0.8,
        box_size,
        fillet_radius,
        fillets_at=[Alignment.TOP, Alignment.BOTTOM],
    )
    parts.add(rectangular, "rectangular_top_bottom")

    # 11. Tall box with side fillets only
    tall_box = create_filleted_box(
        box_size,
        box_size,
        box_size * 1.5,
        fillet_radius,
        fillets_at=[Alignment.LEFT, Alignment.RIGHT, Alignment.FRONT, Alignment.BACK],
    )
    parts.add(tall_box, "tall_sides_only")

    # 12. Demonstration using no_fillets_at parameter
    # (This fillets all edges except the bottom ones)
    all_except_bottom = create_filleted_box(
        box_size, box_size, box_size, fillet_radius, no_fillets_at=[Alignment.BOTTOM]
    )
    parts.add(all_except_bottom, "all_except_bottom")

    print(f"Created {len(parts.as_list())} different filleted box examples")

    # Define process data for 3D printing
    process_data = {
        "filament": "PLA",
        "process_overrides": {
            "layer_height": "0.2",
            "nozzle_temperature": "210",
            "hot_plate_temp_initial_layer": "60",
            "sparse_infill_density": "20%",
            "wall_loops": "2",
            "top_shell_layers": "3",
            "bottom_shell_layers": "3",
        },
    }

    # Arrange and export all parts
    print(f"Arranging parts on {bed_width}x{bed_width}mm build plate...")
    arrange_and_export_parts(
        parts.as_list(),
        part_gap,
        bed_width,
        __file__,
        prod=True,  # Enable production mode for clean STL export
        process_data=process_data,
        max_build_height=200,  # mm
    )

    elapsed_time = time.time() - start_time
    print(f"Example completed in {elapsed_time:.1f} seconds")
    print("STL files have been exported to the output directory")


if __name__ == "__main__":
    main()
