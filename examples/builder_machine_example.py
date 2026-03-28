"""Illustrative declarative builder example for a small machine assembly.

This example uses YAML files in ``examples/builder_machine_demo`` to build three
assemblies declaratively:

1. machine_base
2. guide_column (stacked on base)
3. spindle_head (stacked on column)

Run:
    python examples/builder_machine_example.py
"""

from pathlib import Path

from shellforgepy.builder import build_from_file


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = repo_root / "examples" / "builder_machine_demo" / "assemblies.yaml"
    repository_dir = repo_root / "output" / "builder_machine_demo_repo"

    print("Building declarative machine demo...")
    print(f"  config: {config_file}")
    print(f"  repository_dir: {repository_dir}")

    results = build_from_file(config_file, repository_dir=repository_dir)

    print("\nBuild results:")
    for result in results:
        artifacts = result.get("artifacts", {})
        print(f"- assembly: {result['assembly_name']}")
        print(f"  hash: {result['parameter_hash']}")
        print(f"  cache_hit: {result['cache_hit']}")
        print(f"  artifact_dir: {result['artifact_dir']}")
        if artifacts.get("leader_step"):
            print(f"  leader_step: {artifacts['leader_step']}")

    print("\nDone. Inspect output files under output/builder_machine_demo_repo/.")


if __name__ == "__main__":
    main()
