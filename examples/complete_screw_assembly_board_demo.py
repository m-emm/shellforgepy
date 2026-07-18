#!/usr/bin/env python3
"""
Complete Screw Assembly Board Demo

Shows how ``create_complete_screw_assembly()`` keeps a screw, its hole cutters,
and its reference geometry together while the headless shaft remains the
alignment leader.

Every mounted position is produced with ``align()``. The two bottom-entry
examples first rotate the complete assembly by 180 degrees; no placement uses
coordinate translations. A smaller invisible placement guide provides three
columns and two rows inset from the board edges.

Usage:
    python examples/complete_screw_assembly_board_demo.py

Output:
    output/complete_screw_assembly_board_demo.obj
    output/complete_screw_assembly_board_demo.stl

Change ``mount_specs`` in ``main()`` to try other screw sizes, head types, hole
types, clearances, or mounting faces.
"""

from shellforgepy.simple import (
    Alignment,
    HoleType,
    MScrew,
    PartList,
    ScrewType,
    align,
    arrange_and_export_parts,
    create_box,
    create_complete_screw_assembly,
    rotate,
)

BOARD_WIDTH = 160.0
BOARD_DEPTH = 100.0
BOARD_THICKNESS = 18.0
GUIDE_WIDTH = 130.0
GUIDE_DEPTH = 70.0


def align_mount(
    assembly,
    board,
    placement_guide,
    x_alignment,
    y_alignment,
    z_alignment,
    *,
    from_bottom=False,
):
    """Orient and align a complete assembly without placement translations."""
    if from_bottom:
        assembly = rotate(180, axis=(1, 0, 0))(assembly)

    assembly = align(assembly, placement_guide, x_alignment)
    assembly = align(assembly, placement_guide, y_alignment)
    assembly = align(assembly, board, z_alignment)
    return assembly


def main():
    board = create_box(BOARD_WIDTH, BOARD_DEPTH, BOARD_THICKNESS)

    # This helper is only an alignment target. It is deliberately smaller than
    # the board and is never added to the exported PartList.
    placement_guide = create_box(GUIDE_WIDTH, GUIDE_DEPTH, BOARD_THICKNESS)
    placement_guide = align(placement_guide, board, Alignment.CENTER, axes=[0, 1])

    recessed_size = "M5"
    recessed_head_height = MScrew.from_size(recessed_size).cylinder_head_height
    recess_below_top = 3.0
    recessed_shaft_length = BOARD_THICKNESS - recessed_head_height - recess_below_top

    self_threading_gap = 5.0
    mount_specs = [
        {
            "name": "M3_conical_flush_top",
            "description": "M3 conical head flush with the board top",
            "assembly_kwargs": {
                "size": "M3",
                "length": BOARD_THICKNESS,
                "screw_type": ScrewType.CONICAL_HEAD,
                "hole_type": HoleType.CLEARANCE,
                "with_access_hole": True,
            },
            "x_alignment": Alignment.LEFT,
            "y_alignment": Alignment.BACK,
            "z_alignment": Alignment.BOTTOM,
            "color": (0.90, 0.62, 0.18),
        },
        {
            "name": "M4_cylinder_above_top",
            "description": "M4 cylinder head standing above the board",
            "assembly_kwargs": {
                "size": "M4",
                "length": BOARD_THICKNESS,
                "screw_type": ScrewType.CYLINDER_HEAD,
                "hole_type": HoleType.CLEARANCE,
            },
            "x_alignment": Alignment.CENTER,
            "y_alignment": Alignment.BACK,
            "z_alignment": Alignment.BOTTOM,
            "color": (0.25, 0.55, 0.92),
        },
        {
            "name": "M5_cylinder_recessed_with_access",
            "description": "M5 cylinder head recessed below a top access hole",
            "assembly_kwargs": {
                "size": recessed_size,
                "length": recessed_shaft_length,
                "screw_type": ScrewType.CYLINDER_HEAD,
                "hole_type": HoleType.CLEARANCE,
                "with_access_hole": True,
                "extra_access_hole_length": recess_below_top,
            },
            "x_alignment": Alignment.RIGHT,
            "y_alignment": Alignment.BACK,
            "z_alignment": Alignment.BOTTOM,
            "color": (0.72, 0.32, 0.82),
        },
        {
            "name": "M3_cylinder_from_bottom",
            "description": "M3 cylinder-head screw entering from below",
            "assembly_kwargs": {
                "size": "M3",
                "length": BOARD_THICKNESS,
                "screw_type": ScrewType.CYLINDER_HEAD,
                "hole_type": HoleType.CORE,
            },
            "x_alignment": Alignment.LEFT,
            "y_alignment": Alignment.FRONT,
            "z_alignment": Alignment.BOTTOM,
            "from_bottom": True,
            "color": (0.24, 0.75, 0.45),
        },
        {
            "name": "M4_conical_flush_bottom",
            "description": "M4 conical head flush with the board bottom",
            "assembly_kwargs": {
                "size": "M4",
                "length": BOARD_THICKNESS,
                "screw_type": ScrewType.CONICAL_HEAD,
                "hole_type": HoleType.CLEARANCE,
                "with_access_hole": True,
            },
            "x_alignment": Alignment.CENTER,
            "y_alignment": Alignment.FRONT,
            "z_alignment": Alignment.TOP,
            "from_bottom": True,
            "color": (0.90, 0.34, 0.32),
        },
        {
            "name": "M5_self_threading_with_head_gap",
            "description": (
                "M5 cylinder head above a gap, with the self-threading lead-in "
                "at the board top"
            ),
            "assembly_kwargs": {
                "size": "M5",
                "length": BOARD_THICKNESS + self_threading_gap,
                "screw_type": ScrewType.CYLINDER_HEAD,
                "hole_type": HoleType.SELF_THREADING,
                "hole_distance_from_head": self_threading_gap,
                "core_radius_adjustment": -0.1,
                "lead_in": True,
            },
            "x_alignment": Alignment.RIGHT,
            "y_alignment": Alignment.FRONT,
            "z_alignment": Alignment.BOTTOM,
            "color": (0.20, 0.72, 0.76),
        },
    ]

    mounted_screws = []
    for mount_spec in mount_specs:
        assembly = create_complete_screw_assembly(
            with_thread=False, **mount_spec["assembly_kwargs"]
        )
        assembly = align_mount(
            assembly,
            board,
            placement_guide,
            mount_spec["x_alignment"],
            mount_spec["y_alignment"],
            mount_spec["z_alignment"],
            from_bottom=mount_spec.get("from_bottom", False),
        )

        board = assembly.use_as_cutter_on(board)
        mounted_screws.append(
            (
                assembly.get_named_non_production_part("complete_screw"),
                mount_spec,
            )
        )
        print(f"Mounted: {mount_spec['description']}")

    parts = PartList()
    parts.add(board, "board_with_complete_screw_mounts", color=(0.72, 0.58, 0.38))
    for screw, mount_spec in mounted_screws:
        parts.add(screw, mount_spec["name"], color=mount_spec["color"])

    arrange_and_export_parts(
        parts,
        prod_gap=5,
        bed_width=200,
        script_file=__file__,
        export_directory="output",
        prod=False,
        export_obj=True,
        export_stl=True,
        export_individual_parts=False,
        preserve_model_coordinates=True,
    )
    print("Exported complete screw assembly board demo to output/")


if __name__ == "__main__":
    main()
