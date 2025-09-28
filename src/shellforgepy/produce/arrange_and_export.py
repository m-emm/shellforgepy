"""CAD-agnostic part arrangement and STL export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from shellforgepy.adapters.simple import (
    export_solid_to_stl as adapter_export_solid_to_stl,
)
from shellforgepy.adapters.simple import get_bounding_box
from shellforgepy.construct.alignment_operations import translate

# Import the adapter to delegate CAD-specific operations
from shellforgepy.construct.part_collector import PartCollector
from shellforgepy.produce.production_parts_model import PartList


def export_solid_to_stl(
    solid,
    destination,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
):
    """Export a CAD solid to an STL file using the appropriate"""

    adapter_export_solid_to_stl(
        solid,
        str(destination),
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _arrange_parts_in_rows(
    parts,
    *,
    gap,
    bed_width,
):

    arranged = []
    x_cursor = 0.0
    y_cursor = 0.0
    row_depth = 0.0

    for shape in parts:
        min_point, max_point = get_bounding_box(shape)
        width = max_point[0] - min_point[0]
        depth = max_point[1] - min_point[1]

        if bed_width is not None and arranged:
            projected_width = x_cursor + width
            if projected_width > bed_width:
                x_cursor = 0.0
                y_cursor += row_depth + gap
                row_depth = 0.0

        move_vector = (
            x_cursor - min_point[0],
            y_cursor - min_point[1],
            -min_point[2],
        )
        arranged_shape = translate(*move_vector)(shape)
        arranged.append(arranged_shape)

        x_cursor += width + gap
        row_depth = max(row_depth, depth)

    return arranged


def arrange_and_export_parts(
    parts,
    prod_gap,
    bed_with,
    script_file,
    export_directory=None,
    *,
    prod=False,
    process_data=None,
    max_build_height=None,
):
    """Arrange named parts, export individual STLs, and a fused assembly."""

    if isinstance(parts, PartList):
        parts_iterable = parts.as_list()
    else:
        parts_iterable = parts

    parts_list = [dict(item) for item in parts_iterable]
    if prod:
        parts_list = [p for p in parts_list if not p.get("skip_in_production", False)]
        print("Arranging for production")

    if not parts_list:
        raise ValueError("No parts provided for arrangement and export")

    shapes = []
    names = []
    for entry in parts_list:
        if "name" not in entry or "part" not in entry:
            raise KeyError("Each part mapping must include 'name' and 'part'")
        shape = entry["part"]
        min_point, max_point = get_bounding_box(shape)
        if (
            prod
            and max_build_height is not None
            and max_point[2] - min_point[2] > max_build_height
        ):
            raise ValueError(
                f"Part {entry['name']} exceeds max_build_height ({max_build_height} mm)"
            )
        shapes.append(shape)
        names.append(str(entry["name"]))

    arranged_shapes = _arrange_parts_in_rows(shapes, gap=prod_gap, bed_width=bed_with)

    export_dir = Path(export_directory) if export_directory is not None else Path.home()
    export_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(script_file).stem or "cadquery_parts"
    fused_collector = PartCollector()

    print("Fusing parts")

    for name, arranged_shape in zip(names, arranged_shapes):
        fused_collector.fuse(arranged_shape)
        part_filename = export_dir / f"{base_name}_{_safe_name(name)}.stl"
        print(f"Exporting {name} to {part_filename}")
        export_solid_to_stl(arranged_shape, part_filename)
        print(f"Exported {name} to {part_filename}")

    fused_shape = fused_collector.part
    assert fused_shape is not None  # fused_collector received at least one part

    assembly_path = export_dir / f"{base_name}.stl"
    export_solid_to_stl(fused_shape, assembly_path)
    print(f"Exported whole part to {assembly_path}")

    if process_data is not None:
        process_data["part_file"] = assembly_path.resolve().as_posix()
        process_filename = assembly_path.with_name(f"{assembly_path.stem}_process.json")
        with process_filename.open("w", encoding="utf-8") as handle:
            json.dump(process_data, handle, indent=4)
        print(f"Exported process data to {process_filename}")

    return assembly_path
