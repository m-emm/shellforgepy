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


import numpy as np
from shellforgepy.simple import *

PRODUCTION = True


#### Process settings for this part

# Overrides master settings with these values


quality_speed = 55  # Increased for PLA capability
inner_speed = 85  # Higher speeds for PLA

quality_acceleration = 2500  # Higher for PLA
inner_acceleration = 8000  # Higher for PLA

quality_jerk = 6  # Slightly higher for PLA
inner_jerk = 12  # Higher for PLA
nozzle_diameter = 0.6

min_layer_height = nozzle_diameter * 0.25
max_layer_height = nozzle_diameter * 0.75  # Slightly lower ratio for .6mm

##########
layer_height_factor = 0.6
##########

vertical_layers = 2

layer_height = np.round(
    min_layer_height + (max_layer_height - min_layer_height) * layer_height_factor, 2
)
print(
    f"Layer height: {layer_height} (nozzle_diameter: {nozzle_diameter}, min_layer_height: {min_layer_height}, max_layer_height: {max_layer_height})"
)

PROCESS_DATA_06_PLA = {
    "filament": "FilamentPLAExample",
    "process_overrides": {
        ######### .6MM NOZZLE SETTINGS #########
        "nozzle_diameter": "0.6",
        "max_layer_height": f"{max_layer_height}",
        "min_layer_height": f"{min_layer_height}",
        "layer_height": f"{layer_height}",
        "line_width": "0.65",  # ~108% of nozzle diameter for good flow
        "inner_wall_line_width": "0.65",
        "outer_wall_line_width": "0.6",  # Slightly smaller for quality
        "sparse_infill_line_width": "0.7",  # Wider for faster infill
        "initial_layer_line_width": "0.7",
        "solid_infill_line_width": "0.65",
        "support_interface_line_width": "0.6",
        "internal_solid_infill_line_width": "0.65",
        "support_line_width": "0.65",
        "bridge_line_width": "0.6",
        "thin_wall_line_width": "0.6",
        "gap_fill_line_width": "0.6",
        "top_surface_line_width": "0.6",  # Finer for surface quality
        ##### END .6MM NOZZLE SETTINGS #####
        # Basic setup
        "adaptive_layer_height": "0",
        "enable_arc_fitting": "1",
        # Layer and shell structure - adjusted for PLA strength requirements
        "bottom_shell_layers": "3",  # More layers for strength
        "top_shell_layers": "3",
        "wall_loops": "2",  # Keep 2 walls for strength
        "initial_layer_print_height": "0.3",
        # Infill - balanced for PLA
        "sparse_infill_density": "80%",
        "sparse_infill_pattern": "cubic",
        # Temperature settings for PLA
        "nozzle_temperature": "230",
        "nozzle_temperature_initial_layer": "235",
        "hot_plate_temp_initial_layer": "65",
        # Cooling
        "fan_cooling_layer_time": "30",
        "fan_max_speed": "100",
        "fan_min_speed": "80",
        "overhang_fan_speed": "100",
        "slow_down_for_layer_cooling": "1",
        "min_layer_time": "8",
        # Speed settings
        "external_perimeter_speed": f"{quality_speed}",
        "initial_layer_infill_speed": f"{quality_speed}",
        "initial_layer_speed": f"{quality_speed}",
        "inner_wall_speed": f"{inner_speed}",
        "internal_solid_infill_speed": f"{inner_speed}",
        "gap_fill_speed": f"{inner_speed}",
        "gap_infill_speed": f"{inner_speed}",
        "solid_infill_speed": f"{inner_speed}",
        "sparse_infill_speed": f"{inner_speed}",
        "support_interface_speed": f"{inner_speed}",
        "support_speed": f"{inner_speed}",
        "top_surface_speed": f"{quality_speed}",
        "outer_wall_speed": f"{quality_speed}",
        "bridge_speed": "30",
        # Acceleration settings
        "initial_layer_acceleration": f"{quality_acceleration}",
        "outer_wall_acceleration": f"{quality_acceleration}",
        "top_surface_acceleration": f"{quality_acceleration}",
        "inner_wall_acceleration": f"{inner_acceleration}",
        "solid_infill_acceleration": f"{inner_acceleration}",
        "sparse_infill_acceleration": f"{inner_acceleration}",
        "support_acceleration": f"{inner_acceleration}",
        "support_interface_acceleration": f"{inner_acceleration}",
        "gap_fill_acceleration": f"{inner_acceleration}",
        "bridge_acceleration": f"{inner_acceleration}",
        # Jerk settings
        "initial_layer_jerk": f"{quality_jerk}",
        "outer_wall_jerk": f"{quality_jerk}",
        "top_surface_jerk": f"{quality_jerk}",
        "inner_wall_jerk": f"{inner_jerk}",
        "solid_infill_jerk": f"{inner_jerk}",
        "sparse_infill_jerk": f"{inner_jerk}",
        "support_interface_jerk": f"{inner_jerk}",
        "support_jerk": f"{inner_jerk}",
        "gap_fill_jerk": f"{inner_jerk}",
        # Retraction
        "filament_retraction_length": "1.2",
        "filament_retraction_speed": "40",
        "filament_deretraction_speed": "30",
        "filament_flow_ratio": "1.0",
        # Support settings
        "enable_support": "0",
        "bridge_no_support": "1",
        "support_threshold_angle": "50",
        # Brim settings
        "brim_type": "outer_and_inner",
        "brim_width": "4",
        "brim_ears_detection_length": "1",
        "brim_ears_max_angle": "125",
        "brim_object_gap": "0",
        # Surface quality and compensation
        "elefant_foot_compensation": "0.1",
        "xy_contour_compensation": "0",
        "xy_hole_compensation": "0.05",
        "infill_wall_overlap": "25%",
        "resolution": "0.05",
    },
}


PROCESS_DATA = PROCESS_DATA_06_PLA

### End of process settings ###


def create_geometry(output_dir="output"):
    """
    Create a cylinder mesh and export as STL file.

    Args:
        output_dir: Directory to save STL file
    """

    radius = 20
    height = 60.0
    parts = PartList()

    parts.add(create_basic_cylinder(radius=radius, height=height), "example_cylinder_1")

    # Export everything as either assembly or production layout
    arrange_and_export_parts(
        parts,
        prod_gap=3,  # 3mm gap between parts
        bed_width=200,  # 200mm bed width
        script_file=__file__,
        export_directory=output_dir,
        prod=PRODUCTION,  # Production mode controlled by global flag
        process_data=PROCESS_DATA,
    )


if __name__ == "__main__":
    create_geometry()
