from shellforgepy.adapters.simple import copy_part, rotate_part_native, translate_part


class NamedPart:
    """A part with a name, useful for tracking individual parts in assemblies."""

    def __init__(self, name, part):
        self.name = name
        self.part = part

    def copy(self):
        """Create a copy of this named part."""
        return NamedPart(self.name, copy_part(self.part))

    def translate(self, vector):
        """Translate this part by a vector."""
        translated_part = translate_part(self.part, vector)
        return NamedPart(self.name, translated_part)

    def rotate(self, *args):
        """Rotate this part using adapter function for consistent interface."""
        rotated_part = rotate_part_native(self.part, *args)
        return rotated_part

    def reconstruct(self):
        """Reconstruct this NamedPart after in-place transformation."""
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
