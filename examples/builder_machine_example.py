"""Build and visualize the declarative inspection-gantry example.

The geometry generators live beside the YAML so the complete example can be
read without jumping through the ShellForgePy source tree.  The orchestration
itself is in ``builder_machine_demo/assemblies.yaml``.
"""

from pathlib import Path

from shellforgepy.builder.builder import main as builder_main


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = repo_root / "examples" / "builder_machine_demo" / "assemblies.yaml"
    repository_dir = repo_root / "output" / "builder_machine_demo_repository"
    runs_dir = repo_root / "output" / "builder_machine_demo_runs"

    print("Building an adaptive inspection gantry from declarative YAML.")
    print("The carriage measures the generated bridge and sizes itself to fit.")
    print(f"Configuration: {config_file}")

    result = builder_main(
        [
            str(config_file),
            "--assembly",
            "machine_demo",
            "--visualize",
            "--repository-dir",
            str(repository_dir),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "latest",
        ]
    )

    print(
        "\nScene written to "
        "output/builder_machine_demo_runs/machine_demo_run_latest/."
    )
    print("Run this example again to see unchanged assemblies come from the cache.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
