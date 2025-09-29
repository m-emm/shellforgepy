"""
Hawaii Bottle Cap Example

"""

from shellforgepy.simple import (
    Alignment,
    PartList,
    align,
    apply_fillet_by_alignment,
    arrange_and_export_parts,
    create_basic_cylinder,
    create_ring,
    create_screw_thread,
    rotate,
    translate,
)

# ---- PROD/Export controls ----
PROD = True
PROD_GAP = 2
BED_WIDTH = 220
MAX_BUILD_HEIGHT = 245

# Optional slicer process overrides
PROCESS_DATA = {
    "filament": "FilamentPETGMegeMaster",
    "process_overrides": {
        "nozzle_diameter": "0.8",
        "layer_height": "0.2",
        "line_width": "0.6",
        "outer_wall_speed": "20",
        "inner_wall_speed": "45",
        "solid_infill_speed": "45",
        "sparse_infill_speed": "60",
        "xy_hole_compensation": "0.2",
    },
}


def main():
    """Create the Hawaii bottle cap."""
    parts = PartList()

    try:
        # Design parameters
        resolution = 180
        inner_radius = 15 / 2
        outer_radius = inner_radius + 1.4
        outer_thickness = 0.2
        core_height = 12
        cap_inner_diameter = 27.7
        cap_holder_outer_diameter = 30.9
        cap_holder_height = 7.48
        cap_rim_outer_diameter = 33.7
        cap_rim_height = 1.35
        cap_depth_to_gasket_bottom = 20
        cap_height = 24.5
        pitch = 4.3  # calibrated to 4.3
        cap_cover_thickness = cap_height - cap_depth_to_gasket_bottom
        cap_fillet = 1

        ripple_depth = 0.9
        ripple_height = 0.9 * (
            cap_height - cap_cover_thickness - cap_rim_height - cap_holder_height
        )
        num_ripples = 24

        screw_top_from_top = 8.3  # screw top from the top of the rim
        screw_top_from_top_of_cover = (
            cap_height - cap_cover_thickness - screw_top_from_top
        )
        core_overlength = 5
        num_turns = (screw_top_from_top_of_cover - core_overlength) / pitch

        inner_thickness = pitch - 0.2
        core_height = screw_top_from_top_of_cover
        core_offset = 0.5

        # Create the cap cover (top cylinder) - NO FILLET YET!
        # Keep clean geometry for reliable alignments
        cap_cover_clean = create_basic_cylinder(
            radius=cap_rim_outer_diameter / 2, height=cap_cover_thickness
        )

        # Create the main cap body (ring)
        cap = create_ring(
            outer_radius=cap_rim_outer_diameter / 2,
            inner_radius=cap_inner_diameter / 2,
            height=cap_height - cap_cover_thickness,
        )

        # Stack the cover on top of the cap using alignment with clean geometry
        cap = align(cap, cap_cover_clean, Alignment.STACK_TOP)
        cap = cap.fuse(cap_cover_clean)

        # Add ripples around the cap - align to CLEAN geometry for reliable positioning
        # Ripple cutters should be positioned so half cuts into the cap (creating the groove)
        for i in range(num_ripples):
            # Create cylinder for ripple cutting
            ripple_cutter = create_basic_cylinder(
                radius=ripple_depth, height=ripple_height
            )

            # Align to clean, unfilleted geometry for reliable bounding boxes
            ripple_cutter = align(ripple_cutter, cap, Alignment.CENTER)
            ripple_cutter = align(ripple_cutter, cap_cover_clean, Alignment.STACK_TOP)
            ripple_cutter = align(ripple_cutter, cap_cover_clean, Alignment.RIGHT)

            # Move ripple cutter outward by its radius so half of it cuts into the cap, half is outside
            # This creates the proper ripple groove geometry
            ripple_cutter = translate(ripple_depth, 0, 0)(ripple_cutter)
            ripple_cutter = rotate(
                360 / num_ripples * i,
                center=(0, 0, 0),
                axis=(0, 0, 1),
            )(ripple_cutter)

            # Cut each ripple individually from cap
            cap = cap.cut(ripple_cutter)

        # Create the screw thread
        screw = create_screw_thread(
            pitch=pitch,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            outer_thickness=outer_thickness,
            num_turns=num_turns,
            resolution=resolution,
            inner_thickness=inner_thickness,
            optimize_start=True,
            core_height=core_height,
            core_offset=core_offset,
        )

        # Flip the screw (rotate 180 degrees around Y axis)
        screw = rotate(180, center=(0, 0, 0), axis=(0, 1, 0))(screw)

        # Position the screw using alignment with clean geometry
        screw = align(screw, cap, Alignment.CENTER)
        screw = align(screw, cap_cover_clean, Alignment.STACK_TOP)

        # Cut out the cap holder area using exact FCMacro alignment approach
        cap_holder_cutter = create_ring(
            outer_radius=cap_rim_outer_diameter / 2 * 2,  # Large outer radius
            inner_radius=cap_holder_outer_diameter / 2,
            height=cap_holder_height,
        )

        # Position using alignment exactly like FCMacro:
        # 1. Center with cap
        cap_holder_cutter = align(cap_holder_cutter, cap, Alignment.CENTER)
        # 2. Align to top of cap
        cap_holder_cutter = align(cap_holder_cutter, cap, Alignment.TOP)
        # 3. Translate down by cap_rim_height
        cap_holder_cutter = translate(0, 0, -cap_rim_height)(cap_holder_cutter)

        cap = cap.cut(cap_holder_cutter)

        # Fuse the screw to the cap
        cap = cap.fuse(screw)

        # APPLY FILLETS ONLY AT THE VERY END - after all alignments and cuts
        # This ensures reliable geometry throughout the construction process
        # There is a FreeCAD bug which leads to wrong bounding boxes after filleting
        print("Applying fillets to final geometry...")
        cap = apply_fillet_by_alignment(cap, cap_fillet, fillets_at=[Alignment.BOTTOM])

        # Add the cap to the parts list
        parts.add(cap, "cap", flip=False)

        # ---- Arrange and export ----
        arrange_and_export_parts(
            parts.as_list(),
            PROD_GAP,
            BED_WIDTH,
            __file__,
            prod=PROD,
            process_data=PROCESS_DATA,
            max_build_height=MAX_BUILD_HEIGHT,
        )

        print("Hawaii bottle cap created successfully!")

    except Exception as e:
        print(f"Error creating hawaii bottle cap: {e}")
        import traceback

        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
