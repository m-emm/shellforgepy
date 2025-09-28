
from typing import Mapping
from shellforgepy.construct.alignment_operations import (
 translate, rotate
)

from shellforgepy.construct.named_part import NamedPart

def _normalize_named_parts(
    parts,
):
    if not parts:
        return []

    normalized = []
    for idx, item in enumerate(parts):
        if isinstance(item, NamedPart):
            normalized.append(item)
            continue
        if isinstance(item, Mapping):
            if "name" not in item or "part" not in item:
                raise KeyError("Named parts mappings must contain 'name' and 'part'")
            normalized.append(NamedPart(str(item["name"]), item["part"]))
            continue
        if isinstance(item, tuple) and len(item) == 2:
            name, part = item
            normalized.append(NamedPart(str(name), part))
            continue
        raise TypeError(
            f"Unsupported item in named parts sequence at index {idx}: {item!r}"
        )
    return normalized


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
        self.followers = _normalize_named_parts(followers)
        self.cutters = _normalize_named_parts(cutters)
        self.non_production_parts = _normalize_named_parts(non_production_parts)

    def get_leader_as_part(self):
        return self.leader

    def get_non_production_parts_fused(self):
        if not self.non_production_parts:
            return None
        collector = PartCollector()
        for part in self.non_production_parts:
            collector.fuse(part.part)
        return collector.part

    def leaders_followers_fused(self):
        collector = PartCollector()
        collector.fuse(self.leader)
        for follower in self.followers:
            collector.fuse(follower.part)
        assert collector.part is not None
        return collector.part

    def copy(self):

        return LeaderFollowersCuttersPart(
            copy_part(self.leader),
            [follower.copy() for follower in self.followers],
            [cutter.copy() for cutter in self.cutters],
            [non_prod.copy() for non_prod in self.non_production_parts],
        )

    def fuse(
        self,
        other,
    ):

        if isinstance(other, LeaderFollowersCuttersPart):
            new_leader = fuse_parts(self.leader, other.leader)
            new_followers = [f.copy() for f in (self.followers + other.followers)]
            new_cutters = [c.copy() for c in (self.cutters + other.cutters)]
            new_non_prod = [
                n.copy()
                for n in (self.non_production_parts + other.non_production_parts)
            ]
            return LeaderFollowersCuttersPart(
                new_leader, new_followers, new_cutters, new_non_prod
            )

        other_shape = other
        new_leader = fuse_parts(self.leader, other_shape)
        return LeaderFollowersCuttersPart(
            new_leader,
            [f.copy() for f in self.followers],
            [c.copy() for c in self.cutters],
            [n.copy() for n in self.non_production_parts],
        )

    def translate(self, vector):

        vec = vector
        self.leader = translate_part(self.leader, vec)
        self.followers = [follower.translate(vec) for follower in self.followers]
        self.cutters = [cutter.translate(vec) for cutter in self.cutters]
        self.non_production_parts = [
            part.translate(vec) for part in self.non_production_parts
        ]
        return self

    def rotate(
        self,
        angle,
        center=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
    ):

        center_vec = center
        axis_vec = axis
        self.leader = rotate_part(self.leader, angle, center_vec, axis_vec)
        self.followers = [
            follower.rotate(angle, center_vec, axis_vec) for follower in self.followers
        ]
        self.cutters = [
            cutter.rotate(angle, center_vec, axis_vec) for cutter in self.cutters
        ]
        self.non_production_parts = [
            part.rotate(angle, center_vec, axis_vec)
            for part in self.non_production_parts
        ]
        return self

    @property
    def BoundBox(self):

        return get_bounding_box(self.leader)

