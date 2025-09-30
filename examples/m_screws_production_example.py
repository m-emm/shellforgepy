#!/usr/bin/env python3
"""
M-Screws Production Assembly Example for shellforgepy.

This example shows how to create nuts, bolts, and threaded components
using the comprehensive m_screws module.

- production=False: Creates a beautiful assembly view with nuts aligned on bolts at different heights
- production=True: Automatically arranges all components flat on a single plate for 3D printing
"""

import os

from shellforgepy.simple import (
    Alignment,
    PartList,
    align,
    arrange_and_export_parts,
    create_bolt_thread,
    create_cylinder_screw,
    create_nut,
    get_clearance_hole_diameter,
    get_screw_info,
    list_supported_sizes,
    translate,
)

# Production mode toggle
PRODUCTION = False  # Set to False for assembly view, True for 3D printing layout


def main():
    print("M-Screws Example for shellforgepy - Assembly & Production")
    print("=" * 60)
    print(
        f"Mode: {'PRODUCTION (3D printing layout)' if PRODUCTION else 'ASSEMBLY (beautiful aligned view)'}"
    )
    print("=" * 60)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Production parts collector
    parts = PartList()

    # Show supported sizes
    print("Supported screw sizes:", list_supported_sizes())

    # Get information about M8 screws
    print("\nM8 Screw specifications:")
    m8_info = get_screw_info("M8")
    for key, value in m8_info.items():
        print(f"  {key}: {value}")

    if PRODUCTION:
        print("\n" + "=" * 60)
        print("CREATING PRODUCTION LAYOUT (optimized for 3D printing)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("CREATING ASSEMBLY VIEW (beautiful aligned demonstration)")
        print("=" * 60)

    # Create screws first (they'll be the reference for alignment)
    screw_specs = [
        ("M8", 25, 0),  # M8 at ground level
        ("M10", 30, 40),  # M10 at 40mm height
        ("M12", 35, 80),  # M12 at 80mm height
    ]

    screws = {}
    for size, length, height in screw_specs:
        print(f"Creating {size} cylinder screw ({length}mm long)...")
        screw = create_cylinder_screw(size, length=length, with_thread=False)

        # Position screws at different heights for assembly view
        screw = translate(0, height, height)(screw)
        print(f"  → Positioned at height {height}mm for assembly view")

        screws[size] = screw
        parts.add(
            screw, f"{size}_screw_{length}mm", flip=True
        )  # Flip for production printing

    # Create nuts and align them with screws
    print("\nCreating nuts...")
    for size, _, height in screw_specs:
        print(f"Creating {size} nut...")
        nut = create_nut(size)

        # Center the nut on the screw shaft for assembly view
        screw = screws[size]
        nut = align(nut, screw, Alignment.CENTER)  # Center horizontally
        nut = align(nut, screw, Alignment.BOTTOM)  # Align to screw bottom
        nut = translate(0, 0, 10)(nut)  # Lift nut 10mm up the screw shaft
        print(f"  → Aligned and positioned on {size} screw shaft")

        parts.add(nut, f"{size}_nut")

    # Create 3D printing optimized nuts with slack
    print("\nCreating 3D print optimized nuts...")
    for size in ["M8", "M10", "M12"]:
        nut_slack = create_nut(size, slack=0.3)
        parts.add(nut_slack, f"{size}_nut_3D_print")

    # Create solid hex pieces for reference
    print("\nCreating solid hex pieces...")
    for i, size in enumerate(["M8", "M10", "M12"]):
        hex_piece = create_nut(size, no_hole=True)

        # Space them out in assembly view
        hex_piece = translate(50 + i * 30, 0, 0)(hex_piece)
        print(f"  → {size} hex piece positioned at x={50 + i * 30}")

        parts.add(hex_piece, f"{size}_hex_solid")

    # Create threaded components for demonstration
    print("\nCreating threaded demonstration components...")

    # M8 threaded bolt - positioned separately in assembly
    m8_thread = create_bolt_thread("M8", length=8)
    m8_thread = translate(-40, 0, 20)(m8_thread)  # Position to the left, elevated
    print("  → M8 threaded bolt positioned at (-40, 0, 20)")
    parts.add(m8_thread, "M8_thread_8mm")

    # M10 threaded screw - positioned separately in assembly
    m10_threaded_screw = create_cylinder_screw(
        "M10", length=15, with_thread=True, only_minimal_thread=True
    )
    m10_threaded_screw = translate(-40, 30, 35)(
        m10_threaded_screw
    )  # Position to the left, elevated
    print("  → M10 threaded screw positioned at (-40, 30, 35)")
    parts.add(
        m10_threaded_screw,
        "M10_screw_threaded_15mm",
        flip=True,  # Flip in production for printability
    )

    print(f"\nTotal parts in assembly: {len(parts.as_list())}")

    print("\n" + "=" * 60)
    print("TECHNICAL INFORMATION")
    print("=" * 60)

    # Show clearance hole information
    print(f"M8 clearance hole diameters:")
    print(f"  Close fit: {get_clearance_hole_diameter('M8', 'close')}mm")
    print(f"  Normal fit: {get_clearance_hole_diameter('M8', 'normal')}mm")
    print(f"  Loose fit: {get_clearance_hole_diameter('M8', 'loose')}mm")

    print("\n" + "=" * 60)
    print("EXPORTING ASSEMBLY")
    print("=" * 60)

    # Export everything as either assembly or production layout
    production_file = arrange_and_export_parts(
        parts,
        prod_gap=3,  # 3mm gap between parts
        bed_width=200,  # 200mm bed width
        script_file=__file__,
        export_directory=output_dir,
        prod=PRODUCTION,  # Production mode controlled by global flag
    )

    if PRODUCTION:
        print(f"\n🎉 Production layout created!")
        print(f"📁 All parts flipped and arranged for 3D printing")
        print(f"📁 Screw heads on bottom, nuts flat - ready to print!")
    else:
        print(f"\n🎨 Assembly view created!")
        print(f"📁 Beautiful 3D arrangement showing how parts fit together")
        print(f"📁 Nuts aligned on screw shafts at different heights")
        print(f"📁 Threaded components positioned for demonstration")

    print(f"📁 Main file: {production_file}")
    print(f"📁 Individual parts in: {output_dir}/")

    print("\n" + "=" * 60)
    print("MODE INSTRUCTIONS")
    print("=" * 60)
    print("🔧 To switch modes, edit the PRODUCTION variable at the top:")
    print("   PRODUCTION = True  → 3D printing layout (flat arrangement)")
    print("   PRODUCTION = False → Assembly view (beautiful 3D positioning)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
