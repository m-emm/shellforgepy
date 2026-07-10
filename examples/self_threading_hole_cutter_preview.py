"""Generate preview images for the self-threading hole cutter.

Run from the repository root:

    python examples/self_threading_hole_cutter_preview.py

Outputs are written to:

    output/self_threading_hole_cutter_preview/
"""

from pathlib import Path

from shellforgepy.adapters._adapter import cut_parts
from shellforgepy.produce.arrange_and_export import arrange_and_export
from shellforgepy.produce.production_parts_model import PartList
from shellforgepy.render.api import render_obj_views
from shellforgepy.render.image import preferred_image_suffix
from shellforgepy.simple import (
    Alignment,
    align,
    create_box,
    create_self_threading_hole_cutter,
    translate,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "self_threading_hole_cutter_preview"
)
PREVIEW_VIEWS = ("top", "front_angle")


def export_obj_scene(part, name, color):
    parts = PartList()
    parts.add(part, name, color=color)
    return arrange_and_export(
        parts,
        script_file=__file__,
        export_base_name=name,
        export_directory=OUTPUT_DIR,
        export_stl=False,
        export_step=False,
        export_obj=True,
        export_individual_parts=False,
        preserve_model_coordinates=True,
    )


def render_scene(obj_path, filename_prefix):
    paths = render_obj_views(
        obj_path,
        output_dir=OUTPUT_DIR,
        views=PREVIEW_VIEWS,
        width=768,
        height=768,
        filename_prefix=filename_prefix,
    )

    for path in paths:
        if path.suffix.lower() != ".png":
            raise RuntimeError(
                "PNG preview generation requires Pillow; "
                f"rendered {path.name} instead"
            )

    return paths


def main():
    if preferred_image_suffix() != ".png":
        raise RuntimeError("PNG preview generation requires Pillow")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cutter_length = 8
    block_height = 6

    default_cutter = create_self_threading_hole_cutter("M3", cutter_length)
    tightened_cutter = create_self_threading_hole_cutter(
        "M3",
        cutter_length,
        core_radius_adjustment=-0.15,
    )
    tightened_lead_in_cutter = create_self_threading_hole_cutter(
        "M3",
        cutter_length,
        core_radius_adjustment=-0.15,
        lead_in=True,
    )
    tightened_lead_in_cutter_cutaway = cut_parts(
        tightened_lead_in_cutter,
        create_box(4, 4, cutter_length + 2, origin=(-2, 0, -1)),
    )

    block = create_box(16, 16, block_height)
    drilling_cutter = align(
        tightened_lead_in_cutter, block, Alignment.CENTER, axes=[0, 1]
    )
    drilling_cutter = translate(0, 0, block_height - cutter_length)(drilling_cutter)
    drilled_block = cut_parts(block, drilling_cutter)
    drilled_block_cutaway = cut_parts(
        drilled_block,
        create_box(20, 10, 8, origin=(-2, -2, -1)),
    )

    preview_paths = []
    for part, name, color in [
        (
            default_cutter,
            "self_threading_hole_cutter_default",
            (0.95, 0.38, 0.18),
        ),
        (
            tightened_cutter,
            "self_threading_hole_cutter_tightened",
            (0.95, 0.52, 0.18),
        ),
        (
            tightened_lead_in_cutter,
            "self_threading_hole_cutter_tightened_lead_in",
            (0.25, 0.62, 0.95),
        ),
        (
            tightened_lead_in_cutter_cutaway,
            "self_threading_hole_cutter_tightened_lead_in_cutaway",
            (0.25, 0.62, 0.95),
        ),
        (
            drilled_block,
            "self_threading_hole_drilled_block_lead_in",
            (0.86, 0.90, 0.78),
        ),
        (
            drilled_block_cutaway,
            "self_threading_hole_drilled_block_lead_in_cutaway",
            (0.86, 0.90, 0.78),
        ),
    ]:
        obj_path = export_obj_scene(part, name, color=color)
        preview_paths.extend(render_scene(obj_path, name))

    print("Generated self-threading hole cutter previews:")
    for path in preview_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
