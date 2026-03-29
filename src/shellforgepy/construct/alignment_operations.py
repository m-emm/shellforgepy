from types import SimpleNamespace

import numpy as np
from shellforgepy.adapters._adapter import (
    copy_part,
    get_bounding_box,
    mirror_part,
    rotate_part,
    scale_part,
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


def mirror(normal=(1, 0, 0), point=(0, 0, 0)):
    """Create a mirroring transformation function."""

    def retval(body):
        body_copy = copy_part(body)
        return mirror_part(body_copy, normal=normal, point=point)

    return retval


def scale(factor, center=None):
    """Create a scaling transformation function."""

    # There are NO FRAMEWORK SPECIFC CALLS allowed here! Use adapter functions only!
    #  isinstance(x, NamedPart)  or similar ARE FORBIDDEN here!"

    def retval(body):
        body_copy = copy_part(body)
        return scale_part(body_copy, factor, center=center)

    return retval


def stack_alignment_of(alignment):
    stack_alignment = alignment.stack_alignment
    if stack_alignment is None:
        raise ValueError(f"Aligmment {alignment} has no corresponding stack alignment")

    return stack_alignment


def _calc_stack_translation_vector(
    alignment,
    bb,
    to_bb,
    part_width,
    part_length,
    part_height,
    stack_gap,
    project_to_axes,
):
    retval = None
    if alignment == Alignment.STACK_LEFT:
        retval = (to_bb.xmin - bb.xmin - part_width, 0, 0)
    elif alignment == Alignment.STACK_RIGHT:
        retval = (to_bb.xmax - bb.xmax + part_width, 0, 0)
    elif alignment == Alignment.STACK_BACK:
        retval = (0, to_bb.ymax - bb.ymax + part_length, 0)
    elif alignment == Alignment.STACK_FRONT:
        retval = (0, to_bb.ymin - bb.ymin - part_length, 0)
    elif alignment == Alignment.STACK_TOP:
        retval = (0, 0, to_bb.zmax - bb.zmax + part_height)
    elif alignment == Alignment.STACK_BOTTOM:
        retval = (0, 0, to_bb.zmin - bb.zmin - part_height)
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    retval = project_to_axes(*retval)

    if stack_gap != 0:
        offset = [0, 0, 0]
        offset[alignment.axis] = alignment.sign * stack_gap
        offset = project_to_axes(*offset)
        retval = tuple(np.array(retval) + np.array(offset))
    return retval


def align_translation(part, to, alignment: Alignment, axes=None, stack_gap=0):
    """
    Return a reusable translation function computed from an alignment setup.

    This is the lower-level companion to ``align()``. In most design code you
    should prefer ``align()`` because it is more direct and reads more clearly
    at the call site. See ``align()`` for the full explanation of alignment
    families, idiomatic chaining, ``axes``, and ``stack_gap`` semantics.

    The difference is that ``align_translation()`` does not immediately move
    ``part``. Instead, it calculates the translation implied by aligning
    ``part`` relative to ``to`` and returns that translation as a callable.
    You can then apply the same movement to one or more parts.

    This is useful when a single placement decision must be reused for a whole
    group of related parts, helper solids, or cutter geometry.

    Example
    -------
    Compute the translation once and reuse it:

    .. code-block:: python

        move_to_carriage = align_translation(tool_head, carriage, Alignment.CENTER)

        tool_head = move_to_carriage(tool_head)
        nozzle = move_to_carriage(nozzle)
        cable_relief = move_to_carriage(cable_relief)

    This keeps several parts in the same relative arrangement while moving the
    whole group according to the placement computed from ``tool_head``.

    A common advanced pattern is to compute the transform from one
    representative part and then apply it to associated geometry that should
    travel with it.

    Parameters
    ----------
    part:
        The reference part whose bounding box is used to compute the
        translation.
    to:
        The target reference object. As with ``align()``, ``None`` is only
        valid together with ``Alignment.CENTER`` and means centering around
        the origin.
    alignment:
        The alignment mode used to compute the translation.
    axes:
        Optional iterable of constrained axes ``[0, 1, 2]`` for
        ``X, Y, Z``. Only supported for ``Alignment.CENTER``.
    stack_gap:
        Additional offset used for ``STACK_*`` alignments.

    Returns
    -------
    A transformation function equivalent to ``translate(dx, dy, dz)`` for the
    computed alignment.

    Notes
    -----
    - The returned callable copies the part it is applied to, just like
      ``translate()`` and ``align()``.
    - The translation is computed from the bounding boxes of ``part`` and
      ``to`` at the time ``align_translation()`` is called.
    - If you only need to move one part once, prefer ``align()``.
    """

    bb = get_bounding_box(part)

    if to is None:
        if alignment == Alignment.CENTER:
            translation_vector = (
                (get_xmin(bb) + get_xmax(bb)) / -2,
                (get_ymin(bb) + get_ymax(bb)) / -2,
                (get_zmin(bb) + get_zmax(bb)) / -2,
            )
            translation_vector = [
                0 if axes is not None and i not in axes else v
                for i, v in enumerate(translation_vector)
            ]

            return translate(*translation_vector)
        else:
            raise ValueError(
                "If 'to' is None, only CENTER alignment is supported and will center at origin."
            )

    if axes is not None and alignment != Alignment.CENTER:
        raise ValueError("Axes constraint is only supported for CENTER alignment.")

    to_bb = get_bounding_box(to)

    part_width = get_xlen(bb)
    part_length = get_ylen(bb)
    part_height = get_zlen(bb)

    min_bb_np = np.array(bb[0])
    max_bb_np = np.array(bb[1])
    bb_center_np = (max_bb_np + min_bb_np) / 2

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
    elif alignment == Alignment.EDGE_LEFT:
        return translate(*project_to_axes(to_bb.xmin - bb_center_np[0], 0, 0))
    elif alignment == Alignment.EDGE_RIGHT:
        return translate(*project_to_axes(to_bb.xmax - bb_center_np[0], 0, 0))
    elif alignment == Alignment.BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb.ymax, 0))
    elif alignment == Alignment.FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb.ymin, 0))
    elif alignment == Alignment.EDGE_BACK:
        return translate(*project_to_axes(0, to_bb.ymax - bb_center_np[1], 0))
    elif alignment == Alignment.EDGE_FRONT:
        return translate(*project_to_axes(0, to_bb.ymin - bb_center_np[1], 0))
    elif alignment == Alignment.TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb.zmax))
    elif alignment == Alignment.BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb.zmin))
    elif alignment == Alignment.EDGE_TOP:
        return translate(*project_to_axes(0, 0, to_bb.zmax - bb_center_np[2]))
    elif alignment == Alignment.EDGE_BOTTOM:
        return translate(*project_to_axes(0, 0, to_bb.zmin - bb_center_np[2]))
    elif alignment == Alignment.CENTER:
        return translate(
            *project_to_axes(
                *(max_to_bb_np + min_to_bb_np) / 2 - (max_bb_np + min_bb_np) / 2
            )
        )
    elif alignment in [
        Alignment.STACK_LEFT,
        Alignment.STACK_RIGHT,
        Alignment.STACK_BACK,
        Alignment.STACK_FRONT,
        Alignment.STACK_TOP,
        Alignment.STACK_BOTTOM,
    ]:
        translation_vector = _calc_stack_translation_vector(
            alignment,
            bb,
            to_bb,
            part_width,
            part_length,
            part_height,
            stack_gap,
            project_to_axes,
        )
        return translate(*translation_vector)

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
        Alignment.EDGE_LEFT: (-1, 0, 0),
        Alignment.EDGE_RIGHT: (1, 0, 0),
        Alignment.EDGE_TOP: (0, 0, 1),
        Alignment.EDGE_BOTTOM: (0, 0, -1),
        Alignment.EDGE_FRONT: (0, -1, 0),
        Alignment.EDGE_BACK: (0, 1, 0),
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


def align(part, to, alignment, axes=None, stack_gap=0):
    """
    Return a translated copy of ``part`` aligned relative to ``to``.

    This is one of the core placement APIs in ShellForgePy. It computes a
    bounding-box based translation and applies it to a copy of ``part``;
    the original input object is not modified.

    The function is intentionally simple and idiomatic usage relies on
    chaining multiple calls to express placement step by step. In practice,
    most designs start with a coarse placement such as ``Alignment.CENTER``
    and then refine it with one or two directional alignments:

    .. code-block:: python

        mount = align(mount, motor, Alignment.CENTER)
        mount = align(mount, motor, Alignment.STACK_TOP)
        mount = align(mount, frame, Alignment.BACK)

    This style is used throughout the codebase because it is easy to read,
    easy to tweak, and avoids manual offset arithmetic.

    Alignment families
    ------------------
    ``Alignment`` values fall into three practical groups:

    - Face alignments: ``LEFT``, ``RIGHT``, ``FRONT``, ``BACK``, ``TOP``,
      ``BOTTOM``, ``CENTER``.
      These make the corresponding face, or the full center, coincide with
      the target.
    - Edge alignments: ``EDGE_LEFT``, ``EDGE_RIGHT``, ``EDGE_FRONT``,
      ``EDGE_BACK``, ``EDGE_TOP``, ``EDGE_BOTTOM``.
      These place the center of ``part`` onto the corresponding target face
      plane. They are useful for cutters, holes, and helper geometry that
      should sit on a boundary rather than span across the full part.
    - Stack alignments: ``STACK_LEFT``, ``STACK_RIGHT``, ``STACK_FRONT``,
      ``STACK_BACK``, ``STACK_TOP``, ``STACK_BOTTOM``.
      These place ``part`` immediately outside the target so the touching
      faces butt against each other. ``stack_gap`` then moves the part
      farther apart or slightly into the target.

    Parameters
    ----------
    part:
        The part to move. Any object supported by the active adapter is
        accepted as long as it has a bounding box.
    to:
        The reference part to align against. When ``None``, only
        ``Alignment.CENTER`` is allowed; in that special case the part is
        centered around the origin instead of another object.
    alignment:
        The requested relative placement, typically an ``Alignment`` enum
        value.
    axes:
        Optional iterable of constrained axes ``[0, 1, 2]`` for
        ``X, Y, Z``. Only supported with ``Alignment.CENTER``.
        This is commonly used for partial centering, for example keeping a
        drill centered in ``X`` and ``Y`` while preserving a separately
        chosen ``Z`` placement, or centering a model around the origin only
        in the print bed plane.
    stack_gap:
        Extra offset used with ``STACK_*`` alignments. Positive values add
        clearance between the parts. Negative values intentionally create
        overlap or inset placement and are used frequently for cutters,
        sockets, and print helpers.

    Returns
    -------
    The translated copy of ``part``.

    Idiomatic usage
    ---------------
    Common patterns found across ShellForgePy projects:

    1. Center, then choose a face.

       .. code-block:: python

           cutter = align(cutter, plate, Alignment.CENTER)
           cutter = align(cutter, plate, Alignment.TOP)

       This is the most common pattern for placing holes, bosses, and
       cutters relative to a host body.

    2. Build compound placement from orthogonal constraints.

       .. code-block:: python

           wall = align(wall, base, Alignment.CENTER)
           wall = align(wall, base, Alignment.STACK_TOP)
           wall = align(wall, base, Alignment.LEFT)

       Each call constrains one aspect of the final position and the result
       reads like assembly intent instead of coordinate math.

    3. Use stack alignments for adjacency.

       .. code-block:: python

           cover = align(cover, housing, Alignment.CENTER)
           cover = align(cover, housing, Alignment.STACK_TOP, stack_gap=0.2)

       Without a gap, the faces touch exactly. With a positive gap, the
       parts are separated. With a negative gap, the part is pushed into the
       reference volume.

    4. Use axis-restricted centering to preserve an earlier decision.

       .. code-block:: python

           drill = align(drill, screw_hole, Alignment.CENTER)
           drill = align(drill, jig, Alignment.CENTER, axes=[2])

       Here the second call only adjusts ``Z`` while keeping the previous
       ``X`` and ``Y`` placement intact.

    5. Center around the origin for export or setup geometry.

       .. code-block:: python

           body = align(body, None, Alignment.CENTER)
           plate = align(plate, None, Alignment.CENTER, axes=[0, 1])

    Notes
    -----
    - Alignment is based on bounding boxes, not mating features or local
      coordinate systems.
    - Repeated calls are expected and cheap to read; do not try to compress
      normal placement logic into manual translation formulas unless there is
      a specific geometric reason.
    - ``axes`` with non-``CENTER`` alignments raises ``ValueError``.
    - ``to=None`` with anything other than ``CENTER`` raises ``ValueError``.
    """
    return align_translation(part, to, alignment, axes, stack_gap)(part)
