from dataclasses import dataclass


@dataclass
class PartCollector:
    """Accumulates CAD parts and fuses them into a single shape."""

    part = None

    def fuse(self, other):
        """Fuse this part with another part using the appropriate CAD"""
        if self.part is None:
            self.part = other
        else:
            self.part = self.part.fuse(other)
        return self.part

    def cut(self, other):
        """Cut another part from this part using the appropriate CAD adapter"""
        if self.part is None:
            raise ValueError("Cannot cut from None part")
        else:
            self.part = self.part.cut(other)
        return self.part
