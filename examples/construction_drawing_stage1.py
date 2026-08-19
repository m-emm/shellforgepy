"""Build the Stage 1 plate artifact and emit its exact top-section SVG."""

from pathlib import Path

from construction_drawing_demo.plate_generator import (
    construction_drawing_request,
    create_plate_assembly,
)
from shellforgepy.builder.builder import main as builder_main
from shellforgepy.drawing import render_construction_drawing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = (
        repo_root / "examples" / "construction_drawing_demo" / "assemblies.yaml"
    )
    output_root = repo_root / "output" / "construction_drawing_stage1"

    result = builder_main(
        [
            str(config_file),
            "--repository-dir",
            str(output_root / "repository"),
            "--runs-dir",
            str(output_root / "runs"),
            "--run-id",
            "stage1",
        ]
    )
    request = construction_drawing_request()
    drawing_path = render_construction_drawing(
        create_plate_assembly().leader,
        request,
        output_root / "plate_top.svg",
        part_identity="plate",
        source="plate.leader",
    )
    print(f"Built Stage 1 plate artifact from {config_file}")
    print(f"Exact construction drawing: {drawing_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
