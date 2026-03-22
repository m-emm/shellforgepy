"""Helpers for clipping parts to simple keep-volume envelopes."""

from shellforgepy.adapters._adapter import create_box
from shellforgepy.construct.alignment import Alignment
from shellforgepy.construct.alignment_operations import align
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.part_collector import PartCollector

DEFAULT_BOX_HOLE_CUTTER_SIZE = 500.0

_BOX_HOLE_CUTTER_ALIGNMENTS = (
    Alignment.STACK_TOP,
    Alignment.STACK_BOTTOM,
    Alignment.STACK_FRONT,
    Alignment.STACK_BACK,
    Alignment.STACK_LEFT,
    Alignment.STACK_RIGHT,
)


def _require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def create_box_hole_cutter(
    box_width,
    box_length,
    box_height,
    *,
    cutter_size=DEFAULT_BOX_HOLE_CUTTER_SIZE,
) -> LeaderFollowersCuttersPart:
    """Create a box-shaped clipping assembly that keeps only what is inside the box.

    The returned :class:`LeaderFollowersCuttersPart` uses the keep-volume box as its
    leader and a single fused cutter that surrounds it on all six sides. After
    aligning the assembly to another part, ``use_as_cutter_on(...)`` removes
    everything outside the box, as long as the outside geometry lies within
    ``cutter_size`` of the box on each side.

    This is convenient for:
    - trimming a large part down to a test window
    - extracting a rectangular region from a bigger solid
    - building box-shaped "keep only this area" workflows without hand-placing
      six separate cutters

    Args:
        box_width: Box size along the X axis.
        box_length: Box size along the Y axis.
        box_height: Box size along the Z axis.
        cutter_size: Size of each surrounding cutter box. The default ``500`` is
            the usual design-script ``BIG_THING`` value. Increase it when the
            geometry you want to remove extends farther away from the keep-volume.

    Returns:
        ``LeaderFollowersCuttersPart`` whose leader is the keep-volume box and whose
        single cutter removes everything around it.

    Example:
        keep_volume = create_box_hole_cutter(60, 40, 20)
        keep_volume = align(keep_volume, part, Alignment.CENTER)
        trimmed_part = keep_volume.use_as_cutter_on(part)
    """

    _require_positive("box_width", box_width)
    _require_positive("box_length", box_length)
    _require_positive("box_height", box_height)
    _require_positive("cutter_size", cutter_size)

    box_to_leave_free = create_box(box_width, box_length, box_height)

    box_hole_cutter = PartCollector()
    for alignment in _BOX_HOLE_CUTTER_ALIGNMENTS:
        cutter = create_box(cutter_size, cutter_size, cutter_size)
        cutter = align(cutter, box_to_leave_free, Alignment.CENTER)
        cutter = align(cutter, box_to_leave_free, alignment)
        box_hole_cutter = box_hole_cutter.fuse(cutter)

    return LeaderFollowersCuttersPart(
        leader=box_to_leave_free,
        cutters=[box_hole_cutter],
    )


__all__ = [
    "create_box_hole_cutter",
]
