"""Build the Stage 0 plate artifact and resolve its drawing request.

This command intentionally contains no SVG geometry code.  The plate generator
is shared by the builder resource and the request-resolution smoke check.
"""

from pathlib import Path

from construction_drawing_demo.plate_generator import (
    construction_drawing_request,
    create_plate_assembly,
)
from shellforgepy.builder.builder import main as builder_main
from shellforgepy.drawing import (
    create_svg_document,
    drawing_bounds_from_model_bounds,
    resolve_view_frame,
)
from shellforgepy.simple import get_bounding_box


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = (
        repo_root / "examples" / "construction_drawing_demo" / "assemblies.yaml"
    )
    output_root = repo_root / "output" / "construction_drawing_stage0"

    result = builder_main(
        [
            str(config_file),
            "--repository-dir",
            str(output_root / "repository"),
            "--runs-dir",
            str(output_root / "runs"),
            "--run-id",
            "stage0",
        ]
    )

    fixture = create_plate_assembly()
    request = construction_drawing_request()
    model_bounds = get_bounding_box(fixture.leader)
    frame = resolve_view_frame(request, model_bounds)
    drawing_bounds = drawing_bounds_from_model_bounds(model_bounds, frame)
    _, geometry_group = create_svg_document(request, frame, drawing_bounds)

    print(f"Built Stage 0 plate artifact from {config_file}")
    print(f"Drawing request: {request['name']} ({request['view']})")
    print(f"Resolved frame origin: {frame['origin']}")
    print(f"SVG geometry insertion group: {geometry_group.attrib['id']}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
