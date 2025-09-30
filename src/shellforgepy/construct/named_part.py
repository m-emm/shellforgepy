import logging

from shellforgepy.adapters._adapter import (
    copy_part,
    rotate_part_native,
    translate_part_native,
)

_logger = logging.getLogger(__name__)


class NamedPart:
    """A part with a name, useful for tracking individual parts in assemblies."""

    def __init__(self, name, part):
        self.name = name
        self.part = part

    def copy(self):
        """Create a copy of this named part."""
        return NamedPart(self.name, copy_part(self.part))

    def translate(self, *args):
        """We mimick most cad-sytems in-place translation. We must do an in-place update of self.part."""

        _logger.info(
            f"Translating NamedPart: {self.name} with args: {args}, type of part: {type(self.part)}"
        )
        translated_part = translate_part_native(self.part, *args)
        self.part = translated_part
        return NamedPart(self.name, translated_part)

    def rotate(self, *args):
        """We mimick most cad-systems in-place rotation. We must do an in-place update of self.part."""
        rotated_part = rotate_part_native(self.part, *args)
        self.part = rotated_part
        return NamedPart(self.name, rotated_part)

    def reconstruct(self, transformed_result=None):
        """Reconstruct this NamedPart after transformation."""
        transformed_result_id = (
            id(transformed_result) if transformed_result is not None else "None"
        )
        part_id = id(self.part)
        _logger.info(
            f"Reconstructing NamedPart: {self.name} with transformed_result: {transformed_result}, self.part.id {part_id}, transformed_result.id {transformed_result_id}"
        )
        if transformed_result is not None:
            # Use the transformation result if provided
            return NamedPart(self.name, transformed_result)
        else:
            # Fallback for backward compatibility
            return NamedPart(self.name, copy_part(self.part))

    def fuse(self, other):
        """Fuse this part with another part - duck-types as native CAD object."""
        if isinstance(other, NamedPart):
            other_part = other.part
        else:
            other_part = other
        fused_part = self.part.fuse(other_part)
        return NamedPart(self.name, fused_part)

    def __getattr__(self, name):
        """Delegate any other method calls to the underlying part."""
        return getattr(self.part, name)
