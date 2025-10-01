from types import SimpleNamespace

from shellforgepy.adapters._adapter import (
    copy_part,
    get_bounding_box,
    rotate_part_native,
    translate_part_native,
)
from shellforgepy.adapters.freecad.freecad_adapter import get_vertex_points
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.construct.part_collector import PartCollector


def _ensure_list(items):
    """Ensure items is a list, converting None to empty list."""
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        return [items]
    return list(items)


class LeaderFollowersCuttersPart:
    """Group a leader part with follower, cutter, and non-production parts."""

    def __init__(
        self,
        leader,
        followers=None,
        cutters=None,
        non_production_parts=None,
    ):
        self.leader = leader
        # Store raw parts directly for convenience during construction
        self.followers = _ensure_list(followers)
        self.cutters = _ensure_list(cutters)
        self.non_production_parts = _ensure_list(non_production_parts)

    def use_as_cutter_on(self, part):

        retval = copy_part(part)
        for cutter in self.cutters:
            retval = retval.cut(cutter)

        return retval

    def get_leader_as_part(self):
        return self.leader

    def get_non_production_parts_fused(self):
        collector = PartCollector()
        for part in self.non_production_parts:
            collector.fuse(_unwrap_named_part(part))
        return collector.part if collector.part is not None else collector

    def leaders_followers_fused(self):
        collector = PartCollector()
        for part in [_unwrap_named_part(self.leader)] + [
            _unwrap_named_part(follower) for follower in self.followers
        ]:
            collector.fuse(part)
        return collector.part if collector.part is not None else collector

    def copy(self):

        return LeaderFollowersCuttersPart(
            _clone_part(self.leader),
            [_clone_part(follower) for follower in self.followers],
            [_clone_part(cutter) for cutter in self.cutters],
            [_clone_part(non_prod) for non_prod in self.non_production_parts],
        )

    def BoundingBox(self):
        leader_bb = get_bounding_box(_unwrap_named_part(self.leader))
        return SimpleNamespace(
            xmin=leader_bb[0][0],
            xmax=leader_bb[1][0],
            ymin=leader_bb[0][1],
            ymax=leader_bb[1][1],
            zmin=leader_bb[0][2],
            zmax=leader_bb[1][2],
        )

    def Vertices(self):
        return get_vertex_points(_unwrap_named_part(self.leader))

    def Vertexes(self):
        return get_vertex_points(_unwrap_named_part(self.leader))

    def BoundBox(self):
        leader_bb = get_bounding_box(_unwrap_named_part(self.leader))

        return SimpleNamespace(
            XMin=leader_bb[0][0],
            YMin=leader_bb[0][1],
            ZMin=leader_bb[0][2],
            XMax=leader_bb[1][0],
            YMax=leader_bb[1][1],
            ZMax=leader_bb[1][2],
        )

    def fuse(
        self,
        other,
    ):

        if isinstance(other, LeaderFollowersCuttersPart):
            new_leader = _unwrap_named_part(self.leader).fuse(
                _unwrap_named_part(other.leader)
            )
            new_followers = [_clone_part(f) for f in (self.followers + other.followers)]
            new_cutters = [_clone_part(c) for c in (self.cutters + other.cutters)]
            new_non_prod = [
                _clone_part(n)
                for n in (self.non_production_parts + other.non_production_parts)
            ]
            return LeaderFollowersCuttersPart(
                new_leader, new_followers, new_cutters, new_non_prod
            )

        other_shape = _unwrap_named_part(other)
        new_leader = _unwrap_named_part(self.leader).fuse(other_shape)
        return LeaderFollowersCuttersPart(
            new_leader,
            [_clone_part(f) for f in self.followers],
            [_clone_part(c) for c in self.cutters],
            [_clone_part(n) for n in self.non_production_parts],
        )

    def translate(self, *args):
        """Translate all parts in this composite."""
        self.leader = translate_part_native(self.leader, *args)
        self.followers = [follower.translate(*args) for follower in self.followers]
        self.cutters = [cutter.translate(*args) for cutter in self.cutters]
        self.non_production_parts = [
            part.translate(*args) for part in self.non_production_parts
        ]
        return self

    def rotate(self, *args):
        """Rotate all parts in this composite."""
        self.leader = rotate_part_native(self.leader, *args)
        self.followers = [follower.rotate(*args) for follower in self.followers]
        self.cutters = [cutter.rotate(*args) for cutter in self.cutters]
        self.non_production_parts = [
            part.rotate(*args) for part in self.non_production_parts
        ]
        return self

    def reconstruct(self, transformed_result=None):
        """Reconstruct this composite after in-place transformation."""

        if transformed_result is not None:
            # Use the transformation result if provided
            return LeaderFollowersCuttersPart(
                transformed_result.leader,
                [follower for follower in transformed_result.followers],
                [cutter for cutter in transformed_result.cutters],
                [part for part in transformed_result.non_production_parts],
            )

        else:

            return LeaderFollowersCuttersPart(
                _clone_part(self.leader),
                [_clone_part(follower) for follower in self.followers],
                [_clone_part(cutter) for cutter in self.cutters],
                [_clone_part(part) for part in self.non_production_parts],
            )


def _unwrap_named_part(part):
    if isinstance(part, NamedPart):
        return part.part
    return part


def _clone_part(part):
    if isinstance(part, NamedPart):
        return part.copy()
    return copy_part(part)
