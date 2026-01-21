"""Example demonstrating OBJ export with colors."""

from shellforgepy.adapters.cadquery.cadquery_adapter import (
    create_box,
    create_cylinder,
    create_sphere,
    export_colored_parts_to_obj,
    export_solid_to_obj,
    translate_part,
)


def main():
    # Example 1: Single colored part
    box = create_box(20, 20, 20)
    export_solid_to_obj(
        box,
        "output/single_red_box.obj",
        color=(0.9, 0.2, 0.2),  # Red
    )
    print("Created: output/single_red_box.obj + .mtl")

    # Example 2: Multi-colored assembly
    parts = [
        # Red box at origin
        (create_box(15, 15, 15), "red_cube", (0.9, 0.2, 0.2)),
        # Green cylinder to the right
        (
            translate_part(create_cylinder(8, 25), (30, 0, 0)),
            "green_cylinder",
            (0.2, 0.8, 0.2),
        ),
        # Blue sphere above
        (translate_part(create_sphere(10), (0, 0, 35)), "blue_sphere", (0.2, 0.4, 0.9)),
        # Yellow box behind
        (
            translate_part(create_box(10, 10, 10), (0, 30, 0)),
            "yellow_cube",
            (0.95, 0.85, 0.2),
        ),
    ]

    export_colored_parts_to_obj(parts, "output/colored_assembly.obj")
    print("Created: output/colored_assembly.obj + .mtl")

    print("\nYou can open these OBJ files in:")
    print("  - Blender (File > Import > Wavefront (.obj))")
    print("  - MeshLab")
    print("  - macOS Preview/Quick Look")
    print("  - Online viewers like https://3dviewer.net/")


if __name__ == "__main__":
    main()
