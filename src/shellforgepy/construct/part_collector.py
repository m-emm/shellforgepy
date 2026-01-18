from dataclasses import dataclass


@dataclass
class PartCollector:
    """Accumulates CAD parts and fuses them into a single shape.

    Usage:
        collector = PartCollector()
        for i in range(5):
            part = create_box(10, 10, 10)
            part = translate(i * 10, 0, 0)(part)
            collector = collector.fuse(part)

    Note: Do NOT USE .part of this class - it means you are not using it as intended! After the first fuse/cut operation, you will use the part that was returned for futher collecting or cutting into it.
    At the end, the collector will be a regular, single CAD part like one constructed by the primitive geneartion functions. If you use
    loops which may not add any parts, check this before and handle it.
    """

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
