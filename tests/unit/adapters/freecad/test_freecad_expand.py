import pytest
from shellforgepy.adapters.freecad import freecad_adapter


def test_expand_part_is_not_implemented_for_freecad():
    with pytest.raises(
        NotImplementedError,
        match="expand is not implemented for the FreeCAD adapter yet",
    ):
        freecad_adapter.expand_part(None, 1.0)
