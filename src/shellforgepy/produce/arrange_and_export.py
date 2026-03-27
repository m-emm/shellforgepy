"""CAD-agnostic part arrangement and STL export helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from shellforgepy.adapters._adapter import (
    export_colored_parts_to_obj as adapter_export_colored_parts_to_obj,
)
from shellforgepy.adapters._adapter import (
    export_solid_to_step as adapter_export_solid_to_step,
)
from shellforgepy.adapters._adapter import (
    export_solid_to_stl as adapter_export_solid_to_stl,
)
from shellforgepy.adapters._adapter import get_bounding_box
from shellforgepy.construct.alignment_operations import rotate_part, translate
from shellforgepy.construct.part_collector import PartCollector
from shellforgepy.produce.production_parts_model import PartList

_logger = logging.getLogger(__name__)

# Default colors for parts when not specified.
# Uses standard ColorBrewer qualitative palettes from matplotlib
# (Set2, Set3, Pastel2). All colors are in the lighter half of HLS
# color space (lightness > 0.5) for better 3D visualization.
DEFAULT_PART_COLORS = [
    # Set2
    (0.4000, 0.7608, 0.6471),  # Set2[0]
    (0.9882, 0.5529, 0.3843),  # Set2[1]
    (0.5529, 0.6275, 0.7961),  # Set2[2]
    (0.9059, 0.5412, 0.7647),  # Set2[3]
    (0.6510, 0.8471, 0.3294),  # Set2[4]
    (1.0000, 0.8510, 0.1843),  # Set2[5]
    (0.8980, 0.7686, 0.5804),  # Set2[6]
    (0.7020, 0.7020, 0.7020),  # Set2[7]
    # Set3
    (0.5529, 0.8275, 0.7804),  # Set3[0]
    (1.0000, 1.0000, 0.7020),  # Set3[1]
    (0.7451, 0.7294, 0.8549),  # Set3[2]
    (0.9843, 0.5020, 0.4471),  # Set3[3]
    (0.5020, 0.6941, 0.8275),  # Set3[4]
    (0.9922, 0.7059, 0.3843),  # Set3[5]
    (0.7020, 0.8706, 0.4118),  # Set3[6]
    (0.9882, 0.8039, 0.8980),  # Set3[7]
    (0.8510, 0.8510, 0.8510),  # Set3[8]
    (0.7373, 0.5020, 0.7412),  # Set3[9]
    (0.8000, 0.9216, 0.7725),  # Set3[10]
    (1.0000, 0.9294, 0.4353),  # Set3[11]
    # Pastel2
    (0.7020, 0.8863, 0.8039),  # Pastel2[0]
    (0.9922, 0.8039, 0.6745),  # Pastel2[1]
    (0.7961, 0.8353, 0.9098),  # Pastel2[2]
    (0.9569, 0.7922, 0.8941),  # Pastel2[3]
    (0.9020, 0.9608, 0.7882),  # Pastel2[4]
    (1.0000, 0.9490, 0.6824),  # Pastel2[5]
    (0.9451, 0.8863, 0.8000),  # Pastel2[6]
    (0.8000, 0.8000, 0.8000),  # Pastel2[7]
]


def export_solid_to_stl(
    solid,
    destination,
    *,
    tolerance=0.1,
    angular_tolerance=0.1,
):
    """Export a CAD solid to an STL file using the appropriate adapter."""

    adapter_export_solid_to_stl(
        solid,
        str(destination),
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def export_solid_to_step(
    solid,
    destination,
):
    """Export a CAD solid to a STEP file using the appropriate adapter."""

    adapter_export_solid_to_step(
        solid,
        str(destination),
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

    # Prepare parts as rectangles with dimensions, applying production transformations
    rects = []
    for part_entry in parts_list:
        shape = part_entry["part"]

        # Apply flip transformation if needed (180° rotation around Y-axis)
        if part_entry.get("flip", False):
            _logger.info(f"Flipping part '{part_entry['name']}'")
            # Rotate 180° around Y-axis to flip for printing
            shape = rotate_part(shape, angle=180, axis=(0, 1, 0))

        # Apply production rotation if specified
        if (
            part_entry.get("prod_rotation_angle") is not None
            and part_entry.get("prod_rotation_axis") is not None
        ):
            angle = part_entry["prod_rotation_angle"]
            axis = part_entry["prod_rotation_axis"]
            _logger.info(
                f"Rotating part '{part_entry['name']}' by {angle}° around axis {axis}"
            )
            shape = rotate_part(shape, angle=angle, axis=axis)
        else:
            _logger.info(f"No production rotation for part '{part_entry['name']}'")

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
        else:
            _logger.info(
                f"Part '{rect['name']}' fits width-wise: {width:.1f}mm <= {bed_width}mm"
            )

        if height > bed_depth:
            raise ValueError(
                f"Part '{rect['name']}' too deep for bed ({height:.1f}mm > {bed_depth}mm)"
            )
        else:
            _logger.info(
                f"Part '{rect['name']}' fits depth-wise: {height:.1f}mm <= {bed_depth}mm"
            )

        # Check if we need a new row
        if arranged and x_cursor + width > bed_width:
            y_cursor += row_depth + gap
            x_cursor = 0.0
            row_depth = 0.0

        _logger.info(f"Placing '{rect['name']}' at ({x_cursor:.1f}, {y_cursor:.1f})")

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
                "color": rect["original"].get("color"),
                "animation": rect["original"].get("animation"),
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

    _logger.info(f"Arranged {len(arranged)} parts for production")
    return [
        {
            "name": item["name"],
            "part": item["shape"],
            "color": item.get("color"),
            "animation": item.get("animation"),
        }
        for item in arranged
    ]


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
    export_step=False,
    export_obj=True,
    viewer_base_url=None,
    export_individual_parts=True,
    export_stl=True,
):
    """Arrange named parts with production support and export requested formats.

    Args:
        export_stl: If True (default), export STL files for individual parts and the assembly.
        export_step: If True, also export STEP files alongside STL files.
        export_obj: If True (default), export OBJ files with colors/materials.
        viewer_base_url: Base URL for the 3D viewer. If set, viewer URLs are added to the manifest.
        export_individual_parts: If True (default), export individual part files alongside the fused assembly.
    """

    env_export_dir = os.environ.get("SHELLFORGEPY_EXPORT_DIR")
    env_viewer_url = os.environ.get("SHELLFORGEPY_VIEWER_BASE_URL")
    if env_viewer_url:
        viewer_base_url = env_viewer_url
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

    # Override prod flag with environment variable if set by workflow
    env_prod = os.environ.get("SHELLFORGEPY_PRODUCTION")
    if env_prod is not None:
        prod = env_prod == "1"

    if isinstance(parts, PartList):
        parts_iterable = parts.as_list()
    else:
        parts_iterable = parts

    parts_list = [dict(item) for item in parts_iterable]

    # Filter out parts that should be skipped in production
    if prod:
        parts_list = [p for p in parts_list if not p.get("skip_in_production", False)]
        print(f"Arranging for production; skipped {len(parts) - len(parts_list)} parts")
    else:
        print("Leaving parts where they are")

    if not parts_list:
        raise ValueError("No parts provided for arrangement and export")

    if not export_stl and not export_step and not export_obj:
        raise ValueError("At least one export format must be enabled")

    if process_data is not None and prod and not export_stl:
        raise ValueError(
            "process_data requires export_stl=True in production because slicer settings "
            "need a generated STL part_file"
        )

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
        # Colors are preserved through arrangement
        colors = [item.get("color") for item in arranged_parts]
        animations = [item.get("animation") for item in arranged_parts]
    else:
        # Simple arrangement - just extract shapes and export metadata
        shapes = []
        names = []
        colors = []
        animations = []
        for entry in parts_list:
            if "name" not in entry or "part" not in entry:
                raise KeyError("Each part mapping must include 'name' and 'part'")
            shape = entry["part"]
            shapes.append(shape)
            names.append(str(entry["name"]))
            colors.append(entry.get("color"))
            animations.append(entry.get("animation"))

        arranged_shapes = shapes

    export_dir = Path(export_directory) if export_directory is not None else Path.home()
    export_dir = export_dir.expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    if manifest_data is not None:
        manifest_data["export_dir"] = str(export_dir.resolve())

    base_name = Path(script_file).stem or "cadquery_parts"
    fused_shape = None
    assembly_path = None
    assembly_step_path = None
    obj_path = None
    needs_fused_export = export_stl or export_step

    fused_collector = PartCollector() if needs_fused_export else None

    if needs_fused_export:
        _logger.info("Fusing parts")

    for name, arranged_shape in zip(names, arranged_shapes):
        try:
            if fused_collector is not None:
                fused_collector.fuse(arranged_shape)

            part_filename = export_dir / f"{base_name}_{_safe_name(name)}.stl"
            if export_stl and export_individual_parts:
                _logger.info("Exporting %s to %s", name, part_filename)
                export_solid_to_stl(arranged_shape, part_filename)
                _logger.info("Exported %s to %s", name, part_filename)
            elif export_stl:
                _logger.info(
                    "Skipping individual export for %s due to export_individual_parts=False",
                    name,
                )

            if export_step and export_individual_parts:
                step_filename = export_dir / f"{base_name}_{_safe_name(name)}.step"
                _logger.info("Exporting %s to %s", name, step_filename)
                export_solid_to_step(arranged_shape, step_filename)
                _logger.info("Exported %s to %s", name, step_filename)
            elif export_step:
                _logger.info(
                    "Skipping individual STEP export for %s due to export_individual_parts=False",
                    name,
                )

            if manifest_data is not None and export_stl and export_individual_parts:
                manifest_parts = manifest_data.setdefault("part_files", [])
                if isinstance(manifest_parts, list):
                    manifest_parts.append(str(part_filename.resolve()))
        except Exception as e:
            raise RuntimeError(f"Failed to export part '{name}': {e}") from e

    if fused_collector is not None:
        fused_shape = fused_collector.part
        assert fused_shape is not None  # fused_collector received at least one part

    if export_stl:
        assert fused_shape is not None
        assembly_path = export_dir / f"{base_name}.stl"
        export_solid_to_stl(fused_shape, assembly_path)
        _logger.info("Exported whole part to %s", assembly_path)

    if export_step:
        assert fused_shape is not None
        assembly_step_path = export_dir / f"{base_name}.step"
        export_solid_to_step(fused_shape, assembly_step_path)
        _logger.info("Exported whole part to %s", assembly_step_path)

    # Export colored OBJ file
    if export_obj:
        obj_path = export_dir / f"{base_name}.obj"
        # Build parts list with colors (assign default colors if not specified)
        colored_parts = []
        for i, (name, shape, color, animation) in enumerate(
            zip(names, arranged_shapes, colors, animations)
        ):
            if color is None:
                # Assign a default color from the palette
                color = DEFAULT_PART_COLORS[i % len(DEFAULT_PART_COLORS)]
            colored_parts.append((shape, name, tuple(color), animation))

        _logger.info("Exporting colored OBJ to %s", obj_path)
        adapter_export_colored_parts_to_obj(colored_parts, str(obj_path))
        _logger.info("Exported colored OBJ to %s", obj_path)

        if manifest_data is not None:
            manifest_data["obj_path"] = str(obj_path.resolve())
            mtl_path = obj_path.with_suffix(".mtl")
            manifest_data["mtl_path"] = str(mtl_path.resolve())

            # Generate viewer URL if base URL is configured
            if viewer_base_url:
                obj_filename = obj_path.name
                viewer_url = f"{viewer_base_url.rstrip('/')}/?file={obj_filename}"
                manifest_data["viewer_url"] = viewer_url
                _logger.info("Viewer URL: %s", viewer_url)

    if manifest_data is not None and assembly_path is not None:
        manifest_data["assembly_path"] = str(assembly_path.resolve())
    if manifest_data is not None and assembly_step_path is not None:
        manifest_data["assembly_step_path"] = str(assembly_step_path.resolve())

    if process_data is not None and export_stl:
        assert assembly_path is not None
        process_data["part_file"] = assembly_path.resolve().as_posix()
        process_filename = assembly_path.with_name(f"{assembly_path.stem}_process.json")
        with process_filename.open("w", encoding="utf-8") as handle:
            json.dump(process_data, handle, indent=4)
        _logger.info("Exported process data to %s", process_filename)

        if manifest_data is not None:
            manifest_data["process_data_path"] = str(process_filename.resolve())
    elif process_data is not None:
        _logger.info("Skipping process data export because export_stl=False")

    if manifest_path is not None and manifest_data is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest_data, handle, indent=2, sort_keys=True)
        _logger.info("Wrote workflow manifest to %s", manifest_path)

    if assembly_path is not None:
        return assembly_path
    if assembly_step_path is not None:
        return assembly_step_path
    assert obj_path is not None
    return obj_path


def arrange_and_export(
    parts,
    *,
    prod_gap=1.0,
    bed_width=200.0,
    script_file=None,
    export_directory=None,
    prod=False,
    process_data=None,
    max_build_height=None,
    verbose=False,
    export_step=False,
    export_obj=True,
    viewer_base_url=None,
    export_individual_parts=True,
    export_stl=True,
):
    """Arrange and export a single part with production support.

    Args:
        export_stl: If True (default), export STL files for individual parts and the assembly.
        export_step: If True, also export STEP files alongside STL files.
        export_obj: If True (default), export OBJ files with colors/materials.
        viewer_base_url: Base URL for the 3D viewer. If set, viewer URLs are added to the manifest.
        export_individual_parts: If True (default), export individual part files alongside the fused assembly.
    """

    if script_file is None:
        # get the call stack
        import inspect

        stack = inspect.stack()
        # find the first frame that is not this function
        for frame_info in stack:
            if frame_info.function != "arrange_and_export":
                script_file = frame_info.filename
                break

    if script_file is None:
        script_file = "unknown_script"

    return arrange_and_export_parts(
        parts,
        prod_gap=prod_gap,
        bed_width=bed_width,
        script_file=script_file,
        export_directory=export_directory,
        prod=prod,
        process_data=process_data,
        max_build_height=max_build_height,
        verbose=verbose,
        export_step=export_step,
        export_obj=export_obj,
        viewer_base_url=viewer_base_url,
        export_individual_parts=export_individual_parts,
        export_stl=export_stl,
    )
