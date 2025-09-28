#!/usr/bin/env python3
"""
Local container hash calculation utility for content-based versioning.
Calculates hash of local container source code for consistent versioning.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_container_source_files(container_dir: str) -> List[str]:
    """Get all source files for a local container."""
    files = []
    container_path = Path(container_dir)

    # Get all Python files in the container directory
    for file_path in container_path.rglob("*.py"):
        if "__pycache__" not in str(file_path):
            files.append(str(file_path))

    # Include Dockerfile
    dockerfile = container_path / "Dockerfile"
    if dockerfile.exists():
        files.append(str(dockerfile))

    # Include requirements.txt if it exists
    req_file = container_path / "requirements.txt"
    if req_file.exists():
        files.append(str(req_file))

    # Include any shell scripts
    for file_path in container_path.glob("*.sh"):
        files.append(str(file_path))

    # Include any YAML/JSON config files
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        for file_path in container_path.glob(pattern):
            files.append(str(file_path))

    return sorted(files)  # Sort for consistent hashing


def calculate_container_hash(container_dir: str) -> str:
    """
    Calculate hash for a local container including all source files.

    Args:
        container_dir: Path to the container directory

    Returns:
        12-character hash string
    """
    files_to_hash = get_container_source_files(container_dir)

    # Calculate combined hash
    hash_sha256 = hashlib.sha256()

    # Add file paths and contents to hash
    for file_path in files_to_hash:
        # Add relative path to hash for consistency
        rel_path = os.path.relpath(file_path, container_dir)
        hash_sha256.update(rel_path.encode("utf-8"))

        # Add file content to hash
        try:
            file_hash = calculate_file_hash(file_path)
            hash_sha256.update(file_hash.encode("utf-8"))
        except Exception as e:
            print(f"Warning: Could not hash file {file_path}: {e}", file=sys.stderr)
            # Use filename as fallback
            hash_sha256.update(os.path.basename(file_path).encode("utf-8"))

    # Return first 12 characters of hex digest
    return hash_sha256.hexdigest()[:12]


def get_all_container_hashes(containers_root: str) -> Dict[str, str]:
    """Get hashes for all local containers in the project."""
    container_hashes = {}
    containers_dir = Path(containers_root)

    if not containers_dir.exists():
        return container_hashes

    for container_path in containers_dir.iterdir():
        if container_path.is_dir() and not container_path.name.startswith("."):
            # Check if it has a Dockerfile or main.py to identify it as a container
            dockerfile = container_path / "Dockerfile"
            main_py = container_path / "main.py"

            if dockerfile.exists() or main_py.exists():
                container_name = container_path.name
                try:
                    container_hash = calculate_container_hash(str(container_path))
                    container_hashes[container_name] = container_hash
                except Exception as e:
                    print(
                        f"Error calculating hash for {container_name}: {e}",
                        file=sys.stderr,
                    )
                    # Use a fallback hash based on directory name
                    container_hashes[container_name] = hashlib.sha256(
                        container_name.encode()
                    ).hexdigest()[:12]

    return container_hashes


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python container_hash_calculator.py <container_dir_or_containers_root> [container_name]"
        )
        print("Examples:")
        print("  python container_hash_calculator.py /path/to/local_containers")
        print(
            "  python container_hash_calculator.py /path/to/local_containers local_lambda_runtime"
        )
        print(
            "  python container_hash_calculator.py /path/to/local_containers/local_lambda_runtime"
        )
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) == 3:
        # Calculate hash for specific container by name
        container_name = sys.argv[2]
        containers_root = input_path
        container_dir = os.path.join(containers_root, container_name)

        if not os.path.exists(container_dir):
            print(f"Container directory not found: {container_dir}", file=sys.stderr)
            sys.exit(1)

        container_hash = calculate_container_hash(container_dir)
        print(container_hash)

    elif os.path.isdir(input_path) and (
        os.path.exists(os.path.join(input_path, "Dockerfile"))
        or os.path.exists(os.path.join(input_path, "main.py"))
    ):
        # Calculate hash for specific container directory
        container_hash = calculate_container_hash(input_path)
        print(container_hash)

    else:
        # Calculate hashes for all containers in the root directory
        all_hashes = get_all_container_hashes(input_path)

        if not all_hashes:
            print("No containers found in the specified directory", file=sys.stderr)
            sys.exit(1)

        for container_name, container_hash in all_hashes.items():
            print(f"{container_name}={container_hash}")


if __name__ == "__main__":
    main()
