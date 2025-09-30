from typing import Mapping

from shellforgepy.adapters._adapter import (
    copy_part,
    rotate_part_native,
    translate_part_native,
)
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.construct.part_collector import PartCollector


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
            new_leader = self.leader.fuse(other.leader)
            new_followers = [copy_part(f) for f in (self.followers + other.followers)]
            new_cutters = [copy_part(c) for c in (self.cutters + other.cutters)]
            new_non_prod = [
                copy_part(n)
                for n in (self.non_production_parts + other.non_production_parts)
            ]
            return LeaderFollowersCuttersPart(
                new_leader, new_followers, new_cutters, new_non_prod
            )

        other_shape = other
        new_leader = self.leader.fuse(other_shape)
        return LeaderFollowersCuttersPart(
            new_leader,
            [copy_part(f) for f in self.followers],
            [copy_part(c) for c in self.cutters],
            [copy_part(n) for n in self.non_production_parts],
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
                copy_part(self.leader),
                [copy_part(follower) for follower in self.followers],
                [copy_part(cutter) for cutter in self.cutters],
                [copy_part(part) for part in self.non_production_parts],
            )
