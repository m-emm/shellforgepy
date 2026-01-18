#!/usr/bin/env python3
"""
Create a new ShellForgePy part file with proper boilerplate.

Usage:
    python $SHELLFORGEPY_HOME/tools/new_part_file.py my_part_name
    
Creates my_part_name.py in the current directory, ready to run and export.
"""

import argparse
import os
import sys
from pathlib import Path

TEMPLATE = '''\
"""
{description}

Usage:
    cd <project_root> && ./run.sh path/to/{filename}
    # or with production mode:
    cd <project_root> && SHELLFORGEPY_PRODUCTION=1 ./run.sh path/to/{filename}
"""

import logging
import os

from shellforgepy.simple import *

_logger = logging.getLogger(__name__)

# Production mode from environment variable
PROD = os.environ.get("SHELLFORGEPY_PRODUCTION", "0") == "1"

# Optional slicer process overrides
PROCESS_DATA = {{
    "filament": "FilamentPLAMegeMaster",
    "process_overrides": {{
        "nozzle_diameter": "0.6",
        "layer_height": "0.2",
    }},
}}


def create_{func_name}():
    """Create the {part_name} part."""
    # Example: simple box with a cylindrical hole
    width = 30
    depth = 20
    height = 10
    hole_radius = 4

    # Create base box
    part = create_box(width, depth, height)

    # Create a hole cutter
    hole = create_cylinder(hole_radius, height + 2)
    hole = align(hole, part, Alignment.CENTER)
    hole = translate(0, 0, -1)(hole)

    # Cut the hole
    part = part.cut(hole)

    return part


def main():
    logging.basicConfig(level=logging.INFO)
    parts = PartList()

    # Create the part
    part = create_{func_name}()
    parts.add(part, "{part_name}", flip=False)

    # Arrange and export
    arrange_and_export(
        parts.as_list(),
        script_file=__file__,
        prod=PROD,
        process_data=PROCESS_DATA,
    )

    _logger.info("{part_name} created successfully!")


if __name__ == "__main__":
    main()
'''


def to_snake_case(name):
    """Convert a name to snake_case."""
    # Replace hyphens and spaces with underscores
    name = name.replace("-", "_").replace(" ", "_")
    # Remove any non-alphanumeric characters except underscores
    name = "".join(c if c.isalnum() or c == "_" else "" for c in name)
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = "_" + name
    return name.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Create a new ShellForgePy part file with boilerplate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python new_part_file.py iphone_sleeve
    python new_part_file.py "Cable Holder"
    python new_part_file.py my-bracket --output /path/to/designs/
        """,
    )
    parser.add_argument(
        "name",
        help="Name of the part (used for filename and function name)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing file",
    )

    args = parser.parse_args()

    # Convert name to snake_case for filename and function
    func_name = to_snake_case(args.name)
    filename = f"{func_name}.py"

    # Determine output path
    output_dir = Path(args.output).resolve()
    output_path = output_dir / filename

    # Check if file exists
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    # Generate description from name
    description = args.name.replace("_", " ").replace("-", " ").title()

    # Generate file content
    content = TEMPLATE.format(
        description=description,
        filename=filename,
        func_name=func_name,
        part_name=func_name,
    )

    # Write the file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    # Make executable
    os.chmod(output_path, 0o755)

    print(f"Created: {output_path}")
    print(f"Run with: python {filename}")


if __name__ == "__main__":
    main()
