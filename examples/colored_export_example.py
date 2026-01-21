"""Example demonstrating colored part export with viewer URL."""

import os
import tempfile

from shellforgepy.adapters.cadquery.cadquery_adapter import (
    create_box,
    create_cylinder,
    create_sphere,
    translate_part,
)
from shellforgepy.produce.arrange_and_export import arrange_and_export
from shellforgepy.produce.production_parts_model import PartList


def main():
    parts = PartList()

    # Add parts with explicit colors
    parts.add(
        create_box(20, 20, 10),
        "red_base",
        color=(0.9, 0.2, 0.2),
    )
    parts.add(
        translate_part(create_cylinder(8, 25), (30, 0, 0)),
        "green_pillar",
        color=(0.2, 0.8, 0.2),
    )
    parts.add(
        translate_part(create_sphere(12), (0, 30, 6)),
        "blue_dome",
        color=(0.2, 0.4, 0.9),
    )
    # Part without color - will get default from palette
    parts.add(
        translate_part(create_box(10, 10, 15), (30, 30, 0)),
        "auto_color_tower",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Exporting to: {tmpdir}")

        # Export with viewer URL
        result = arrange_and_export(
            parts,
            script_file=__file__,
            export_directory=tmpdir,
            prod=False,
            export_obj=True,
            viewer_base_url="http://localhost:5173",
        )

        print(f"\nExported assembly STL: {result}")

        # List generated files
        print("\nGenerated files:")
        for f in sorted(os.listdir(tmpdir)):
            fpath = os.path.join(tmpdir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size} bytes)")

        # Show OBJ content sample
        obj_file = os.path.join(tmpdir, "colored_export_example.obj")
        if os.path.exists(obj_file):
            print(f"\nOBJ file preview:")
            with open(obj_file) as f:
                for i, line in enumerate(f):
                    if i < 20:
                        print(f"  {line.rstrip()}")
                    else:
                        print("  ...")
                        break

        # Show MTL content
        mtl_file = os.path.join(tmpdir, "colored_export_example.mtl")
        if os.path.exists(mtl_file):
            print(f"\nMTL file content:")
            with open(mtl_file) as f:
                print(f.read())


if __name__ == "__main__":
    main()
