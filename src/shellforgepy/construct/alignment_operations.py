import numpy as np
from shellforgepy.construct.alignment import Alignment

from shellforgepy.adapters.adapter_chooser import get_cad_adapter

# Get the adapter dynamically
_cad_adapter = get_cad_adapter()

# Import adapter functions if they exist, otherwise create stubs
try:
    translate = _cad_adapter.translate
    rotate = _cad_adapter.rotate
    get_bounding_box = _cad_adapter.get_bounding_box
except AttributeError as e:

    def _missing_function(name):
        def stub(*args, **kwargs):
            raise AttributeError(
                f"Function {name} not available in current CAD adapter"
            )

        return stub

    translate = getattr(_cad_adapter, "translate", _missing_function("translate"))
    rotate = getattr(_cad_adapter, "rotate", _missing_function("rotate"))
    get_bounding_box = getattr(
        _cad_adapter, "get_bounding_box", _missing_function("get_bounding_box")
    )


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


def aligment_signs(aligmment_list):

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


def align_translation(part, to, alignment, axes=None):
    """
    Create a translation function that aligns one object to another.

    This is a wrapper that delegates to the CAD adapter's align_translation function.
    """
    return _cad_adapter.align_translation(part, to, alignment, axes)


def align(part, to, alignment, axes=None):
    """
    Align one object to another and return the aligned copy.

    This is a wrapper that delegates to the CAD adapter's align function.
    """
    return _cad_adapter.align(part, to, alignment, axes)
