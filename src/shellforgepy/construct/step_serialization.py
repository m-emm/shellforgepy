from typing import Dict, List, Tuple

from shellforgepy.adapters._adapter_bridge import export_structured_step
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.simple import *


def _unwrap(part):
    """
    Extract the underlying CAD object from a NamedPart or return as-is.
    """
    if hasattr(part, "part"):
        return part.part
    return part


def serialize_to_step(
    part: LeaderFollowersCuttersPart,
    file_path: str,
) -> None:
    """
    Serialize a LeaderFollowersCuttersPart into a structured STEP file,
    preserving semantic groupings and part names.
    """

    def _with_names(parts, name_map):
        """Return list of (name, part) tuples."""
        result = []
        for idx, p in enumerate(parts):
            name = None
            for n, i in name_map.items():
                if i == idx:
                    name = n
                    break
            result.append((name, _unwrap(p)))
        return result

    structure: Dict[str, List[Tuple[str | None, object]]] = {}

    # Leader
    if part.leader is not None:
        structure["LEADER"] = [("leader", _unwrap(part.leader))]

    # Followers
    if part.followers:
        structure["FOLLOWERS"] = _with_names(
            part.followers,
            part.follower_indices_by_name,
        )

    # Cutters
    if part.cutters:
        structure["CUTTERS"] = _with_names(
            part.cutters,
            part.cutter_indices_by_name,
        )

    # Non-production parts
    if part.non_production_parts:
        structure["NON_PRODUCTION"] = _with_names(
            part.non_production_parts,
            part.non_production_indices_by_name,
        )

    export_structured_step(structure, file_path)


def deserialize_to_leader_followers_cutters_part(
    path: str,
) -> LeaderFollowersCuttersPart:
    """
    Deserialize a structured STEP file into a LeaderFollowersCuttersPart.
    """

    structure = deserialize_structured_step(path)

    # --- Leader ---
    leader = None
    if "LEADER" in structure and structure["LEADER"]:
        leader = structure["LEADER"][0][1]
    elif "ROOT" in structure and structure["ROOT"]:
        leader = structure["ROOT"][0][1]

    def _extract_parts(entries):
        return [part for _, part in entries]

    def _names_or_none(entries):
        if not entries:
            return None
        if all(name is not None for name, _ in entries):
            return [name for name, _ in entries]
        return None

    # --- Followers ---
    follower_entries = structure.get("FOLLOWERS", [])
    followers = _extract_parts(follower_entries)
    follower_names = _names_or_none(follower_entries)

    # --- Cutters ---
    cutter_entries = structure.get("CUTTERS", [])
    cutters = _extract_parts(cutter_entries)
    cutter_names = _names_or_none(cutter_entries)

    # --- Non-production parts ---
    non_production_entries = structure.get("NON_PRODUCTION", [])
    non_production = _extract_parts(non_production_entries)
    non_production_names = _names_or_none(non_production_entries)

    return LeaderFollowersCuttersPart(
        leader=leader,
        followers=followers,
        cutters=cutters,
        non_production_parts=non_production,
        follower_names=follower_names,
        cutter_names=cutter_names,
        non_production_names=non_production_names,
    )
