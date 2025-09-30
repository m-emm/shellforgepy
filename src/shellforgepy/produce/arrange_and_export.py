"""CAD-agnostic part arrangement and STL export helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from shellforgepy.adapters._adapter import (
    export_solid_to_stl as adapter_export_solid_to_stl,
)
from shellforgepy.adapters._adapter import get_bounding_box
from shellforgepy.construct.alignment_operations import rotate_part, translate
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


def _arrange_parts_for_production(
    parts_list,
    *,
    gap,
    bed_width,
    bed_depth=None,
    verbose=False,
    max_build_height=None,
):
    """
    Arrange parts for production with proper flipping and rotation support.
    Based on the sophisticated FreeCAD arrange_for_production function.
    """
    if bed_depth is None:
        bed_depth = bed_width  # assume square bed

    def _print(msg):
        if verbose:
            print(msg)

    # Prepare parts as rectangles with dimensions, applying production transformations
    rects = []
    for part_entry in parts_list:
        shape = part_entry["part"]

        # Apply flip transformation if needed (180° rotation around Y-axis)
        if part_entry.get("flip", False):
            _print(f"Flipping part '{part_entry['name']}'")
            # Rotate 180° around Y-axis to flip for printing
            shape = rotate_part(shape, angle=180, axis=(0, 1, 0))

        # Apply production rotation if specified
        if (
            part_entry.get("prod_rotation_angle") is not None
            and part_entry.get("prod_rotation_axis") is not None
        ):
            angle = part_entry["prod_rotation_angle"]
            axis = part_entry["prod_rotation_axis"]
            _print(
                f"Rotating part '{part_entry['name']}' by {angle}° around axis {axis}"
            )
            shape = rotate_part(shape, angle=angle, axis=axis)
        else:
            _print(f"No production rotation for part '{part_entry['name']}'")

        # Get bounding box after transformations
        min_point, max_point = get_bounding_box(shape)
        width = max_point[0] - min_point[0]
        height = max_point[1] - min_point[1]
        depth = max_point[2] - min_point[2]

        # Check build height constraints
        if max_build_height is not None and depth > max_build_height:
            raise ValueError(
                f"Part {part_entry['name']} exceeds max_build_height ({max_build_height} mm)"
            )

        rects.append(
            {
                "name": part_entry["name"],
                "shape": shape,
                "width": width,
                "height": height,
                "depth": depth,
                "min_point": min_point,
                "max_point": max_point,
                "original": part_entry,
            }
        )

    # Sort parts by area descending (largest first for better packing)
    rects.sort(key=lambda r: -(r["width"] * r["height"]))

    # Simple shelf-based arrangement algorithm
    arranged = []
    x_cursor = 0.0
    y_cursor = 0.0
    row_depth = 0.0

    for rect in rects:
        width = rect["width"]
        height = rect["height"]

        if width > bed_width:
            raise ValueError(
                f"Part '{rect['name']}' too wide for bed ({width:.1f}mm > {bed_width}mm)"
            )

        # Check if we need a new row
        if arranged and x_cursor + width > bed_width:
            y_cursor += row_depth + gap
            x_cursor = 0.0
            row_depth = 0.0

        _print(f"Placing '{rect['name']}' at ({x_cursor:.1f}, {y_cursor:.1f})")

        # Position the part: move to origin first, then to final position
        shape = rect["shape"]
        min_point = rect["min_point"]

        # Move so bottom-left-back corner is at origin
        shape = translate(-min_point[0], -min_point[1], -min_point[2])(shape)

        # Move to final position on the bed
        shape = translate(x_cursor, y_cursor, 0)(shape)

        arranged.append(
            {
                "name": rect["name"],
                "shape": shape,
                "x": x_cursor,
                "y": y_cursor,
                "width": width,
                "height": height,
            }
        )

        x_cursor += width + gap
        row_depth = max(row_depth, height)

    # Center the arrangement on the bed
    if arranged:
        # Calculate total bounds
        min_x = min(item["x"] for item in arranged)
        max_x = max(item["x"] + item["width"] for item in arranged)
        min_y = min(item["y"] for item in arranged)
        max_y = max(item["y"] + item["height"] for item in arranged)

        total_width = max_x - min_x
        total_height = max_y - min_y

        # Calculate centering offset
        offset_x = (bed_width - total_width) / 2 - min_x
        offset_y = (bed_depth - total_height) / 2 - min_y

        # Apply centering offset to all parts
        for item in arranged:
            item["shape"] = translate(offset_x, offset_y, 0)(item["shape"])

    _print(f"Arranged {len(arranged)} parts for production")
    return [{"name": item["name"], "part": item["shape"]} for item in arranged]


def arrange_and_export_parts(
    parts,
    prod_gap,
    bed_width,
    script_file,
    *,
    export_directory=None,
    prod=False,
    process_data=None,
    max_build_height=None,
    verbose=False,
):
    """Arrange named parts with production support, export individual STLs, and a fused assembly."""

    env_export_dir = os.environ.get("SHELLFORGEPY_EXPORT_DIR")
    manifest_path_env = os.environ.get("SHELLFORGEPY_WORKFLOW_MANIFEST")
    manifest_path: Path | None = (
        Path(manifest_path_env).expanduser() if manifest_path_env else None
    )
    manifest_data: dict[str, object] | None = None

    if manifest_path is not None:
        manifest_data = {
            "run_id": os.environ.get("SHELLFORGEPY_RUN_ID"),
            "script_file": str(script_file),
            "parts": [],
        }

    if env_export_dir:
        export_directory = env_export_dir

    if isinstance(parts, PartList):
        parts_iterable = parts.as_list()
    else:
        parts_iterable = parts

    parts_list = [dict(item) for item in parts_iterable]

    # Filter out parts that should be skipped in production
    if prod:
        parts_list = [p for p in parts_list if not p.get("skip_in_production", False)]
        print("Arranging for production")
    else:
        print("Leaving parts where they are")

    if not parts_list:
        raise ValueError("No parts provided for arrangement and export")

    # Use production arrangement or simple positioning
    if prod:
        # Use sophisticated production arrangement with flipping and rotation
        arranged_parts = _arrange_parts_for_production(
            parts_list,
            gap=prod_gap,
            bed_width=bed_width,
            max_build_height=max_build_height,
            verbose=verbose,
        )
        arranged_shapes = [item["part"] for item in arranged_parts]
        names = [item["name"] for item in arranged_parts]
    else:
        # Simple arrangement - just extract shapes and names
        shapes = []
        names = []
        for entry in parts_list:
            if "name" not in entry or "part" not in entry:
                raise KeyError("Each part mapping must include 'name' and 'part'")
            shape = entry["part"]
            # Check build height even in non-production mode
            if max_build_height is not None:
                min_point, max_point = get_bounding_box(shape)
                if max_point[2] - min_point[2] > max_build_height:
                    raise ValueError(
                        f"Part {entry['name']} exceeds max_build_height ({max_build_height} mm)"
                    )
            shapes.append(shape)
            names.append(str(entry["name"]))

        arranged_shapes = shapes

    export_dir = Path(export_directory) if export_directory is not None else Path.home()
    export_dir = export_dir.expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    if manifest_data is not None:
        manifest_data["export_dir"] = str(export_dir.resolve())

    base_name = Path(script_file).stem or "cadquery_parts"
    fused_collector = PartCollector()

    print("Fusing parts")

    for name, arranged_shape in zip(names, arranged_shapes):
        fused_collector.fuse(arranged_shape)
        part_filename = export_dir / f"{base_name}_{_safe_name(name)}.stl"
        print(f"Exporting {name} to {part_filename}")
        export_solid_to_stl(arranged_shape, part_filename)
        print(f"Exported {name} to {part_filename}")

        if manifest_data is not None:
            manifest_parts = manifest_data.setdefault("part_files", [])
            if isinstance(manifest_parts, list):
                manifest_parts.append(str(part_filename.resolve()))

    fused_shape = fused_collector.part
    assert fused_shape is not None  # fused_collector received at least one part

    assembly_path = export_dir / f"{base_name}.stl"
    export_solid_to_stl(fused_shape, assembly_path)
    print(f"Exported whole part to {assembly_path}")

    if manifest_data is not None:
        manifest_data["assembly_path"] = str(assembly_path.resolve())

    if process_data is not None:
        process_data["part_file"] = assembly_path.resolve().as_posix()
        process_filename = assembly_path.with_name(f"{assembly_path.stem}_process.json")
        with process_filename.open("w", encoding="utf-8") as handle:
            json.dump(process_data, handle, indent=4)
        print(f"Exported process data to {process_filename}")

        if manifest_data is not None:
            manifest_data["process_data_path"] = str(process_filename.resolve())

    if manifest_path is not None and manifest_data is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest_data, handle, indent=2, sort_keys=True)
        print(f"Wrote workflow manifest to {manifest_path}")

    return assembly_path
