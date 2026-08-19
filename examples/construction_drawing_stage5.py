"""Build the independent Stage 5 M3 tapped-hole technical drawing example."""

from pathlib import Path

from shellforgepy.builder.builder import main as builder_main


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = (
        repo_root / "examples" / "construction_drawing_stage5_demo" / "assemblies.yaml"
    )
    output_root = repo_root / "output" / "construction_drawing_stage5"
    result = builder_main(
        [
            str(config_file),
            "--repository-dir",
            str(output_root / "repository"),
            "--runs-dir",
            str(output_root / "runs"),
            "--run-id",
            "stage5",
            "--visualize",
            "--assembly",
            "m3_threaded_plate",
        ]
    )
    drawing_path = (
        output_root
        / "runs"
        / "m3_threaded_plate_run_stage5"
        / "construction_drawings"
        / "m3_threaded_plate_top.svg"
    )
    print(f"Built Stage 5 M3 threaded-hole drawing: {drawing_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
