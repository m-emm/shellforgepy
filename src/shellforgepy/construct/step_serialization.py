"""
STEP serialization for LeaderFollowersCuttersPart.

Uses CadQuery's Assembly feature to preserve structure in STEP files.
Each part is added to the assembly with a structured name encoding its
group (LEADER, FOLLOWERS, CUTTERS, NON_PRODUCTION) and index.

A JSON sidecar file stores additional metadata like user-defined names
and additional_data dict that cannot be encoded in the STEP file's object names.
"""

import functools
import hashlib
import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cadquery as cq
import numpy as np
from shellforgepy.construct.leader_followers_cutters_part import (
    LeaderFollowersCuttersPart,
)
from shellforgepy.construct.part_parameters import PartParameters

_logger = logging.getLogger(__name__)

# Metadata version - increment when format changes
_METADATA_VERSION = 3
_STEP_CACHE_ENV_VAR = "SHELLFORGEPY_STEP_CACHE_DIR"
_STEP_CACHE_PLAIN_PART_KEY = "_step_cache_plain_part"


def _convert_for_json(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.

    Handles numpy arrays, numpy scalars, and nested dicts/lists.
    """
    if isinstance(obj, np.ndarray):
        return {"__numpy_array__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def _convert_from_json(obj: Any) -> Any:
    """
    Recursively convert JSON-deserialized objects back to original types.

    Restores numpy arrays from their serialized form.
    """
    if isinstance(obj, dict):
        if obj.get("__numpy_array__"):
            return np.array(obj["data"], dtype=obj.get("dtype", None))
        return {k: _convert_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_from_json(v) for v in obj]
    return obj


def _unwrap(part):
    """
    Extract the underlying CAD object from a NamedPart or return as-is.
    """
    if hasattr(part, "part"):
        return part.part
    return part


def _to_cq_shape(part):
    """Convert a part to a CadQuery shape suitable for assembly."""
    unwrapped = _unwrap(part)

    # If it's already a CadQuery shape, return as-is
    if isinstance(unwrapped, (cq.Shape, cq.Solid, cq.Compound)):
        return unwrapped

    # If it's a Workplane, extract the solid
    if isinstance(unwrapped, cq.Workplane):
        return unwrapped.val()

    return unwrapped


def _get_metadata_path(step_path: str) -> str:
    """Get the path to the metadata JSON file for a STEP file."""
    step_path = str(step_path)
    base = os.path.splitext(step_path)[0]
    return base + ".lfcp.json"


def _make_assembly_name(group: str, index: int) -> str:
    """Create a structured name for an assembly part."""
    return f"{group}_{index:04d}"


def _parse_assembly_name(name: str) -> Optional[Tuple[str, int]]:
    """Parse a structured assembly name back to group and index."""
    try:
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            group = parts[0]
            index = int(parts[1])
            return (group, index)
    except (ValueError, IndexError):
        pass
    return None


def serialize_to_step(
    part: LeaderFollowersCuttersPart,
    file_path: str,
) -> None:
    """
    Serialize a LeaderFollowersCuttersPart into a STEP file with sidecar metadata.

    Uses CadQuery's Assembly to preserve part structure in the STEP file.
    Each part is added with a structured name encoding its group and index.

    Creates two files:
    - {file_path}: STEP file with assembly structure
    - {file_path}.lfcp.json: Metadata file with user-defined names

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

    # Create assembly
    assy = cq.Assembly(name="LeaderFollowersCuttersPart")

    # Build metadata for user-defined names and additional_data
    metadata = {"version": _METADATA_VERSION, "groups": {}}

    # Serialize additional_data if present
    if part.additional_data:
        metadata["additional_data"] = _convert_for_json(part.additional_data)

    def add_group_to_assembly(group_name: str, parts: List, name_map: Dict[str, int]):
        """Add all parts from a group to the assembly."""
        if not parts:
            return

        group_metadata = []

        for idx, p in enumerate(parts):
            shape = _to_cq_shape(p)
            assy_name = _make_assembly_name(group_name, idx)
            user_name = _get_name_for_index(name_map, idx)

            # Add to assembly with structured name
            assy.add(shape, name=assy_name)

            # Record user-defined name in metadata
            group_metadata.append({"name": user_name, "index": idx})

        metadata["groups"][group_name] = group_metadata

    # Process all groups
    if part.leader is not None:
        add_group_to_assembly("LEADER", [part.leader], {})

    if part.followers:
        add_group_to_assembly(
            "FOLLOWERS", part.followers, part.follower_indices_by_name
        )

    if part.cutters:
        add_group_to_assembly("CUTTERS", part.cutters, part.cutter_indices_by_name)

    if part.non_production_parts:
        add_group_to_assembly(
            "NON_PRODUCTION",
            part.non_production_parts,
            part.non_production_indices_by_name,
        )

    # Export assembly to STEP
    assy.save(file_path)

    # Write metadata
    metadata_path = _get_metadata_path(file_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _extract_parts_from_assembly(
    assy: cq.Assembly,
) -> Dict[str, List[Tuple[int, cq.Shape]]]:
    """
    Extract parts from an assembly, organized by group.

    Returns a dict mapping group names to lists of (index, shape) tuples.
    """
    groups: Dict[str, List[Tuple[int, cq.Shape]]] = {}

    def process_node(node: cq.Assembly):
        """Recursively process assembly nodes using children list."""
        # Try to parse this node's name as a structured name
        parsed = _parse_assembly_name(node.name)
        if parsed:
            group, index = parsed
            if group not in groups:
                groups[group] = []

            # Get the shape from this assembly node
            if node.obj is not None:
                groups[group].append((index, node.obj))

        # Process children (not objects, which includes self)
        for child in node.children:
            process_node(child)

    # Process all children of the root assembly
    for child in assy.children:
        process_node(child)

    # Sort each group by index
    for group in groups:
        groups[group].sort(key=lambda x: x[0])

    return groups


def deserialize_to_leader_followers_cutters_part(
    path: str,
) -> LeaderFollowersCuttersPart:
    """
    Deserialize a STEP file with sidecar metadata into a LeaderFollowersCuttersPart.

    Loads the assembly from the STEP file and extracts parts by their
    structured names.

    Args:
        path: Path to the STEP file (str or Path-like)

    Returns:
        Reconstructed LeaderFollowersCuttersPart
    """
    path = str(path)
    metadata_path = _get_metadata_path(path)

    # Require metadata file to exist
    if not os.path.exists(metadata_path):
        raise ValueError(
            f"Metadata file not found: {metadata_path}. "
            "STEP files must have an accompanying .lfcp.json metadata file."
        )

    # Read metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    version = metadata.get("version", 1)

    # Require version 2+ format (v2 = assembly-based, v3 = assembly + additional_data)
    if version < 2:
        raise ValueError(
            f"STEP file {path} uses legacy v1 format which is no longer supported. "
            "Please regenerate the STEP file with the current version."
        )

    # Extract additional_data if present (v3+)
    additional_data = None
    if "additional_data" in metadata:
        additional_data = _convert_from_json(metadata["additional_data"])

    # Load assembly from STEP
    assy = cq.Assembly.load(path)

    # Extract parts from assembly
    groups = _extract_parts_from_assembly(assy)

    def get_parts_and_names(group_name: str) -> Tuple[List, List[Optional[str]]]:
        """Get parts and their user-defined names for a group."""
        if group_name not in groups:
            return [], []

        indexed_parts = groups[group_name]
        parts = [shape for _, shape in indexed_parts]

        # Get user-defined names from metadata
        names = []
        group_meta = metadata.get("groups", {}).get(group_name, [])
        name_by_index = {item["index"]: item.get("name") for item in group_meta}

        for idx, _ in indexed_parts:
            names.append(name_by_index.get(idx))

        return parts, names

    # Extract all groups
    leader_parts, _ = get_parts_and_names("LEADER")
    leader = leader_parts[0] if leader_parts else None

    followers, follower_names = get_parts_and_names("FOLLOWERS")
    cutters, cutter_names = get_parts_and_names("CUTTERS")
    non_production, non_production_names = get_parts_and_names("NON_PRODUCTION")

    # Clean up None names
    follower_names = (
        follower_names
        if follower_names and all(n is not None for n in follower_names)
        else None
    )
    cutter_names = (
        cutter_names
        if cutter_names and all(n is not None for n in cutter_names)
        else None
    )
    non_production_names = (
        non_production_names
        if non_production_names and all(n is not None for n in non_production_names)
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
        additional_data=additional_data,
    )


def step_cached(func=None, *, include_source_hash=False):
    """
    Cache parts to STEP files based on PartParameters.

    When SHELLFORGEPY_STEP_CACHE_DIR is set, a STEP file named by the parameter
    hash will be used for cache hits. On misses, the function is called and the
    result serialized to the cache.

    Set include_source_hash=True to invalidate the cache when the function
    source changes.
    """
    if func is not None:
        return _step_cached_impl(func, include_source_hash=include_source_hash)

    def decorator(inner_func):
        return _step_cached_impl(inner_func, include_source_hash=include_source_hash)

    return decorator


def _get_function_source_hash(func) -> Optional[str]:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        _logger.warning("Unable to read source for %s; skipping source hash.", func)
        return None
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return digest


def _step_cached_impl(func, include_source_hash: bool):
    @functools.wraps(func)
    def wrapper(parameters: PartParameters, *args, **kwargs):
        if not isinstance(parameters, PartParameters):
            raise TypeError("step_cached requires a PartParameters argument.")

        cache_dir = os.environ.get(_STEP_CACHE_ENV_VAR)
        if not cache_dir:
            return func(parameters, *args, **kwargs)

        os.makedirs(cache_dir, exist_ok=True)
        cache_key = parameters.parameters_hash()
        if include_source_hash:
            source_hash = _get_function_source_hash(func)
            if source_hash:
                cache_key = f"{cache_key}-{source_hash}"
        step_path = os.path.join(cache_dir, f"{cache_key}.step")

        if os.path.exists(step_path):
            _logger.info("Using cached STEP file: %s", step_path)
            restored = deserialize_to_leader_followers_cutters_part(step_path)
            additional_data = getattr(restored, "additional_data", None)
            if additional_data and additional_data.get(_STEP_CACHE_PLAIN_PART_KEY):
                return restored.leader
            return restored

        part = func(parameters, *args, **kwargs)
        if isinstance(part, LeaderFollowersCuttersPart):
            serialize_to_step(part, step_path)
        else:
            cached = LeaderFollowersCuttersPart(
                leader=part,
                additional_data={_STEP_CACHE_PLAIN_PART_KEY: True},
            )
            serialize_to_step(cached, step_path)
        return part

    return wrapper
