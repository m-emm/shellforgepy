from dataclasses import dataclass
from typing import Any

from shellforgepy.construct.alignment_operations import copy_part, rotate, translate


@dataclass
class NamedPart:
    """A CAD part with a name."""

    name: str
    part: Any  # CAD object type depends on the adapter

    def copy(self):
        """Create a copy of this named part."""

        return NamedPart(self.name, copy_part(self.part))

    def translate(self, vector):
        """Translate this part by the given vector."""

        translated_part = translate(*vector)(self.part)
        return NamedPart(self.name, translated_part)

    def rotate(
        self,
        angle,
        center=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
    ):
        """Rotate this part around the given axis."""

        rotated_part = rotate(angle, center=center, axis=axis)(self.part)
        return NamedPart(self.name, rotated_part)
