from types import SimpleNamespace

import numpy as np
from shellforgepy.adapters._adapter import (
    copy_part,
    get_bounding_box,
    rotate_part,
    translate_part,
)
from shellforgepy.construct.alignment import Alignment
from shellforgepy.construct.bounding_box_helpers import (
    get_xlen,
    get_xmax,
    get_xmin,
    get_ylen,
    get_ymax,
    get_ymin,
    get_zlen,
    get_zmax,
    get_zmin,
)


def translate(x, y, z):
    """Create a translation transformation function."""

    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!"

    def retval(body):
        body_copy = copy_part(body)
        return translate_part(body_copy, (x, y, z))

    return retval


def rotate(angle, center=None, axis=None):
    """Create a rotation transformation function."""

    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!"

    def retval(body):
        body_copy = copy_part(body)

        return rotate_part(body_copy, angle, center=center, axis=axis)

    return retval


def stack_alignment_of(alignment):
    stack_map = {
        Alignment.LEFT: Alignment.STACK_LEFT,
        Alignment.RIGHT: Alignment.STACK_RIGHT,
        Alignment.TOP: Alignment.STACK_TOP,
        Alignment.BOTTOM: Alignment.STACK_BOTTOM,
        Alignment.FRONT: Alignment.STACK_FRONT,
        Alignment.BACK: Alignment.STACK_BACK,
    }
    if alignment not in stack_map:
        raise ValueError(f"Aligmment {alignment} has no corresponding stack alignment")

    return stack_map[alignment]


def align_translation(part, to, alignment: Alignment, axes=None):
    """
    Create a translation function that aligns one object to another.

    Args:
        part: The object to be aligned
        to: The target object to align to
        alignment: The type of alignment to perform
        axes: Optional list of axes to constrain alignment to (0=X, 1=Y, 2=Z)

    Returns:
        A function that applies the alignment translation
    """
    # Extract the solid from workplane if needed

    bb = get_bounding_box(part)
    to_bb = get_bounding_box(to)

    part_width = get_xlen(bb)
    part_length = get_ylen(bb)
    part_height = get_zlen(bb)

    min_bb_np = np.array(bb[0])
    max_bb_np = np.array(bb[1])

    min_to_bb_np = np.array(to_bb[0])
    max_to_bb_np = np.array(to_bb[1])

    bb = SimpleNamespace(
        xmin=get_xmin(bb),
        xmax=get_xmax(bb),
        ymin=get_ymin(bb),
        ymax=get_ymax(bb),
        zmin=get_zmin(bb),
        zmax=get_zmax(bb),
    )
    to_bb = SimpleNamespace(
        xmin=get_xmin(to_bb),
        xmax=get_xmax(to_bb),
        ymin=get_ymin(to_bb),
        ymax=get_ymax(to_bb),
        zmin=get_zmin(to_bb),
        zmax=get_zmax(to_bb),
    )

    def project_to_axes(x: float, y: float, z: float):
        if axes is None:
            return x, y, z

        return (x if 0 in axes else 0, y if 1 in axes else 0, z if 2 in axes else 0)

    if alignment == Alignment.LEFT:
        return translate(*project_to_axes(to_bb.xmin - bb.xmin, 0, 0))
    elif alignment == Alignment.RIGHT:
        return translate(*project_to_axes(to_bb.xmax - bb.xmax, 0, 0))
    elif alignment == Alignment.BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb.ymax, 0))
    elif alignment == Alignment.FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb.ymin, 0))
    elif alignment == Alignment.TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb.zmax))
    elif alignment == Alignment.BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb.zmin))
    elif alignment == Alignment.CENTER:
        return translate(
            *project_to_axes(
                *(max_to_bb_np + min_to_bb_np) / 2 - (max_bb_np + min_bb_np) / 2
            )
        )
    elif alignment == Alignment.STACK_LEFT:
        return translate(*project_to_axes(to_bb.xmin - bb.xmin - part_width, 0, 0))
    elif alignment == Alignment.STACK_RIGHT:
        return translate(*project_to_axes(to_bb.xmax - bb.xmax + part_width, 0, 0))
    elif alignment == Alignment.STACK_BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb.ymax + part_length, 0))
    elif alignment == Alignment.STACK_FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb.ymin - part_length, 0))
    elif alignment == Alignment.STACK_TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb.zmax + part_height))
    elif alignment == Alignment.STACK_BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb.zmin - part_height))
    else:
        raise ValueError(f"Unknown alignment: {alignment}")


def alignment_signs(aligmment_list):

    if isinstance(aligmment_list, Alignment):
        aligmment_list = [aligmment_list]

    signs = {
        Alignment.LEFT: (-1, 0, 0),
        Alignment.RIGHT: (1, 0, 0),
        Alignment.TOP: (0, 0, 1),
        Alignment.BOTTOM: (0, 0, -1),
        Alignment.FRONT: (0, -1, 0),
        Alignment.BACK: (0, 1, 0),
        Alignment.CENTER: (0, 0, 0),
    }

    vectors = np.array(
        [signs[alignment] for alignment in aligmment_list if alignment in signs]
    )

    # Handle empty list case
    if vectors.size == 0:
        return (0, 0, 0)

    return tuple(np.sum(vectors, axis=0))


def chain_translations(*translations):
    """
    Chain multiple translation functions together.

    Args:
        *translations: Variable number of translation functions

    Returns:
        A function that applies all translations in sequence
    """

    def retval(part):
        result = part
        for translation in translations:
            result = translation(result)
        return result

    return retval


def align(part, to, alignment, axes=None):
    """
    Align one object to another and return the aligned copy.

    This is a wrapper that delegates to the CAD adapter's align function.
    """
    return align_translation(part, to, alignment, axes)(part)
