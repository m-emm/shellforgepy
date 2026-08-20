"""Build a three-view M3 tapped-hole technical drawing with exact projections."""

from pathlib import Path

from shellforgepy.builder.builder import main as builder_main


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = (
        repo_root
        / "examples"
        / "construction_drawing_m3_tapped_holes_demo"
        / "assemblies.yaml"
    )
    output_root = repo_root / "output" / "construction_drawing_m3_tapped_holes"
    result = builder_main(
        [
            str(config_file),
            "--repository-dir",
            str(output_root / "repository"),
            "--runs-dir",
            str(output_root / "runs"),
            "--run-id",
            "latest",
            "--visualize",
            "--assembly",
            "m3_threaded_plate",
        ]
    )
    drawing_path = (
        output_root
        / "runs"
        / "m3_threaded_plate_run_latest"
        / "construction_drawings"
        / "m3_threaded_plate.svg"
    )
    print(f"Built M3 tapped-hole drawing: {drawing_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
