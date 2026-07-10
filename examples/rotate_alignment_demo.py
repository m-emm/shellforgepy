#!/usr/bin/env python3
"""
Rotated Alignment Demo

Shows how rotate_alignment() lets one master-coordinate placement recipe work
in all four rotated corner systems.

Usage:
    python examples/rotate_alignment_demo.py

Output:
    output/rotate_alignment_demo.stl
    output/rotate_alignment_demo.obj
"""

from shellforgepy.simple import *


def create_f_marker():
    stroke = 2.0
    width = 9.0
    middle_width = 6.0
    height = 12.0
    thickness = 1.5

    spine = create_box(stroke, height, thickness)

    top_bar = create_box(width, stroke, thickness)
    top_bar = align(top_bar, spine, Alignment.LEFT)
    top_bar = align(top_bar, spine, Alignment.BACK)

    middle_bar = create_box(middle_width, stroke, thickness)
    middle_bar = align(middle_bar, spine, Alignment.CENTER)
    middle_bar = align(middle_bar, spine, Alignment.LEFT)

    marker = spine.fuse(top_bar)
    marker = marker.fuse(middle_bar)
    marker = rotate(10)(marker)
    return marker


def main():
    rectangle = create_box(40, 60, 1.5)
    master_marker = create_f_marker()

    parts = PartList()
    parts.add(rectangle, "rectangle_40x60", color=(0.75, 0.75, 0.75))

    for i in range(4):
        angle = i * 90
        turn_alignment = rotate_alignment(angle)

        marker = rotate(angle)(master_marker)
        marker = align(
            marker,
            rectangle,
            turn_alignment(Alignment.STACK_RIGHT),
            stack_gap=3,
        )
        marker = align(marker, rectangle, turn_alignment(Alignment.BACK))

        parts.add(marker, f"f_marker_{angle}", color=(0.1, 0.35 + 0.13 * i, 0.9))

    arrange_and_export_parts(
        parts,
        prod_gap=5,
        bed_width=120,
        script_file=__file__,
        export_directory="output",
        prod=False,
        export_individual_parts=False,
    )

    print("Exported rotated alignment demo to output/rotate_alignment_demo.stl")


if __name__ == "__main__":
    main()
