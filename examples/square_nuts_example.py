#!/usr/bin/env python3
"""Display DIN 562 square nuts and their matching hidden pocket cutters."""

import os

from shellforgepy.simple import (
    PartList,
    arrange_and_export_parts,
    create_hidden_nut_pocket_cutter,
    create_square_nut,
    square_nuts_table,
    translate,
)


def main():
    """Build a labelled-by-filename gallery of supported square nut sizes."""
    output_directory = "output/square_nuts"
    os.makedirs(output_directory, exist_ok=True)
    parts = PartList()

    for index, size in enumerate(square_nuts_table):
        nut = create_square_nut(size)
        parts.add(translate(index * 18, 0, 0)(nut), f"{size}_DIN_562_square_nut")

        pocket = create_hidden_nut_pocket_cutter(
            size,
            top_cutter_length=8,
            slack=0.2,
            square_nut=True,
        )
        parts.add(
            translate(index * 18, 18, 0)(pocket.cutters[0]),
            f"{size}_square_nut_pocket_cutter",
        )

    output_file = arrange_and_export_parts(
        parts,
        prod=False,
        script_file=__file__,
        export_directory=output_directory,
    )
    print(f"Exported DIN 562 square-nut gallery to {output_file}")


if __name__ == "__main__":
    main()
