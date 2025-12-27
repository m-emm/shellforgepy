"""
STEP serialization for LeaderFollowersCuttersPart.

Since CadQuery's STEP import doesn't preserve assembly structure, we use a
two-file approach:
1. A single STEP file containing all solids (stacked with Z offsets to separate them)
2. A JSON sidecar file storing the structure metadata (group names, part names, Z offsets)

On deserialization, we read the metadata, extract solids from the STEP file,
and reconstruct the LeaderFollowersCuttersPart with proper groupings.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from shellforgepy.adapters._adapter import (
    export_solid_to_step,
    fuse_parts,
    import_solid_from_step,
)
from shellforgepy.adapters._adapter_bridge import get_bounding_box, translate_part
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)

# Spacing between parts in the stacked STEP file
_STACK_SPACING = 1000.0


def _unwrap(part):
    """
    Extract the underlying CAD object from a NamedPart or return as-is.
    """
    if hasattr(part, "part"):
        return part.part
    return part


def _get_metadata_path(step_path: str) -> str:
    """Get the path to the metadata JSON file for a STEP file."""
    step_path = str(step_path)
    base = os.path.splitext(step_path)[0]
    return base + ".lfcp.json"


def serialize_to_step(
    part: LeaderFollowersCuttersPart,
    file_path: str,
) -> None:
    """
    Serialize a LeaderFollowersCuttersPart into a STEP file with sidecar metadata.

    Creates two files:
    - {file_path}: STEP file with all solids stacked vertically
    - {file_path}.lfcp.json: Metadata file with structure information

    Args:
        part: The LeaderFollowersCuttersPart to serialize
        file_path: Path to the STEP file (str or Path-like)
    """
    file_path = str(file_path)

    if not isinstance(part, LeaderFollowersCuttersPart):
        part = LeaderFollowersCuttersPart(leader=part)

    def _get_name_for_index(name_map: Dict[str, int], idx: int) -> Optional[str]:
        """Get name for a given index from a name map."""
        for name, i in name_map.items():
            if i == idx:
                return name
        return None

    # Build metadata and collect parts with Z offsets
    metadata = {"version": 1, "groups": {}}

    all_parts = []  # List of translated parts
    current_z_offset = 0.0

    # Helper to process a group of parts
    def process_group(group_name: str, parts: List, name_map: Dict[str, int]):
        nonlocal current_z_offset

        if not parts:
            return

        group_metadata = []

        for idx, p in enumerate(parts):
            unwrapped = _unwrap(p)
            name = _get_name_for_index(name_map, idx)

            # Get part bounding box for height calculation
            bbox = get_bounding_box(unwrapped)
            part_height = bbox[1][2] - bbox[0][2]
            z_min = bbox[0][2]

            # Offset to place part at current_z_offset
            z_translation = current_z_offset - z_min
            translated = translate_part(unwrapped, (0, 0, z_translation))

            group_metadata.append(
                {
                    "name": name,
                    "z_offset": current_z_offset,
                    "z_min_original": z_min,
                    "height": part_height,
                }
            )

            all_parts.append(translated)
            current_z_offset += part_height + _STACK_SPACING

        metadata["groups"][group_name] = group_metadata

    # Process all groups
    if part.leader is not None:
        process_group("LEADER", [part.leader], {})

    if part.followers:
        process_group("FOLLOWERS", part.followers, part.follower_indices_by_name)

    if part.cutters:
        process_group("CUTTERS", part.cutters, part.cutter_indices_by_name)

    if part.non_production_parts:
        process_group(
            "NON_PRODUCTION",
            part.non_production_parts,
            part.non_production_indices_by_name,
        )

    # Fuse all parts into a single compound for export
    if all_parts:
        combined = all_parts[0]
        for p in all_parts[1:]:
            combined = fuse_parts(combined, p)

        export_solid_to_step(combined, file_path)

    # Write metadata
    metadata_path = _get_metadata_path(file_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def deserialize_to_leader_followers_cutters_part(
    path: str,
) -> LeaderFollowersCuttersPart:
    """
    Deserialize a STEP file with sidecar metadata into a LeaderFollowersCuttersPart.

    Reads the metadata file to understand the structure, then extracts and
    repositions parts from the STEP file.

    Args:
        path: Path to the STEP file (str or Path-like)

    Returns:
        Reconstructed LeaderFollowersCuttersPart
    """
    path = str(path)
    metadata_path = _get_metadata_path(path)

    # Check if metadata exists - if not, fall back to simple import
    if not os.path.exists(metadata_path):
        # Legacy mode: import as flat leader
        imported = import_solid_from_step(path)
        return LeaderFollowersCuttersPart(leader=imported)

    # Read metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Import the combined STEP file
    combined = import_solid_from_step(path)

    # Extract individual solids from the compound
    from shellforgepy.adapters.adapter_chooser import get_cad_adapter

    adapter = get_cad_adapter()
    solids = adapter.extract_solids(combined)

    # Sort solids by their Z position (bounding box minimum Z)
    def get_z_min(solid):
        bbox = adapter.get_bounding_box(solid)
        return bbox[0][2]

    solids_sorted = sorted(solids, key=get_z_min)

    # Now match solids to metadata groups
    def extract_parts_for_group(group_name: str) -> Tuple[List, List[Optional[str]]]:
        if group_name not in metadata["groups"]:
            return [], []

        group_meta = metadata["groups"][group_name]
        parts = []
        names = []

        for item_meta in group_meta:
            z_offset = item_meta["z_offset"]
            z_min_original = item_meta["z_min_original"]

            # Find the solid at this Z offset
            target_z = z_offset
            best_solid = None
            best_diff = float("inf")

            for solid in solids_sorted:
                solid_z = get_z_min(solid)
                diff = abs(solid_z - target_z)
                if diff < best_diff:
                    best_diff = diff
                    best_solid = solid

            if best_solid is not None and best_diff < _STACK_SPACING / 2:
                # Translate back to original position
                current_z = get_z_min(best_solid)
                z_translation = z_min_original - current_z
                restored = translate_part(best_solid, (0, 0, z_translation))
                parts.append(restored)
                names.append(item_meta.get("name"))
                # Remove from list to avoid reuse
                solids_sorted.remove(best_solid)

        return parts, names

    # Extract all groups
    leader_parts, _ = extract_parts_for_group("LEADER")
    leader = leader_parts[0] if leader_parts else None

    followers, follower_names = extract_parts_for_group("FOLLOWERS")
    cutters, cutter_names = extract_parts_for_group("CUTTERS")
    non_production, non_production_names = extract_parts_for_group("NON_PRODUCTION")

    # Clean up None names
    follower_names = (
        follower_names if all(n is not None for n in follower_names) else None
    )
    cutter_names = cutter_names if all(n is not None for n in cutter_names) else None
    non_production_names = (
        non_production_names
        if all(n is not None for n in non_production_names)
        else None
    )

    return LeaderFollowersCuttersPart(
        leader=leader,
        followers=followers if followers else None,
        cutters=cutters if cutters else None,
        non_production_parts=non_production if non_production else None,
        follower_names=follower_names,
        cutter_names=cutter_names,
        non_production_names=non_production_names,
    )
