"""Explicit graph model for declarative builder dependency reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx
from shellforgepy.builder.errors import BuilderError


@dataclass(frozen=True)
class PlacementStep:
    index: int
    spec: Dict[str, Any]
    moving_reference: Optional[str]
    target_reference: Optional[str]
    moving_assembly_name: Optional[str]
    target_assembly_name: Optional[str]
    rigid_attach: bool
    rigid_group_assembly_names: tuple[str, ...] = ()
    affected_assembly_names: tuple[str, ...] = ()

    @property
    def is_rigid_group(self) -> bool:
        return bool(self.rigid_group_assembly_names)


@dataclass(frozen=True)
class BuilderGraphModel:
    assemblies: List[Dict[str, Any]]
    assemblies_by_name: Dict[str, Dict[str, Any]]
    assembly_dependency_graph: nx.DiGraph
    injection_graph: nx.MultiDiGraph
    placement_steps: List[PlacementStep]
    placement_execution_dag: nx.DiGraph
    first_moving_alignment_index: Dict[str, int]
    first_involved_alignment_index: Dict[str, int]
    placement_build_dependencies: Dict[str, List[str]]
    build_items_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    join_operations_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    join_output_to_operation: Dict[str, str] = field(default_factory=dict)
    join_outputs_by_operation: Dict[str, Dict[str, str]] = field(default_factory=dict)


JOIN_NODE_PREFIX = "join:"


def join_node_name(operation_name: str) -> str:
    return f"{JOIN_NODE_PREFIX}{operation_name}"


def is_join_node_name(name: str) -> bool:
    return str(name).startswith(JOIN_NODE_PREFIX)


def join_operation_name_from_node(node_name: str) -> str:
    normalized = str(node_name)
    if not is_join_node_name(normalized):
        raise BuilderError(f"Not a join operation node: {node_name}")
    return normalized[len(JOIN_NODE_PREFIX) :]


def _entry_kind(entry: Mapping[str, Any]) -> str:
    return str(entry.get("kind", "assembly")).strip().lower()


def _is_join_entry(entry: Mapping[str, Any]) -> bool:
    return _entry_kind(entry) == "join"


def _is_joined_output_entry(entry: Mapping[str, Any]) -> bool:
    return _entry_kind(entry) == "joined_output"


def dependency_names(entry: Mapping[str, Any]) -> List[str]:
    dependencies = entry.get("depends_on")
    if dependencies is None:
        dependencies = entry.get("dependencies", [])
    if dependencies is None:
        return []
    if not isinstance(dependencies, list):
        raise BuilderError(
            f"Dependencies for assembly '{entry.get('name')}' must be a list"
        )
    return [str(item) for item in dependencies]


def _entry_name(entry: Mapping[str, Any]) -> str:
    name = entry.get("name")
    if not name:
        raise BuilderError("Each assembly entry must define a name")
    return str(name)


def _validate_output_alias(alias: Any, operation_name: str) -> str:
    normalized = str(alias).strip()
    if not normalized:
        raise BuilderError(
            f"Join operation '{operation_name}' has an empty output alias"
        )
    return normalized


def _validate_output_assembly_name(value: Any, operation_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise BuilderError(
            f"Join operation '{operation_name}' has an empty output assembly name"
        )
    return normalized


def _normalize_join_declarations(
    assemblies: Sequence[Mapping[str, Any]],
) -> tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    Dict[str, Dict[str, str]],
]:
    normal_entries_by_name: Dict[str, Dict[str, Any]] = {}
    join_entries_by_name: Dict[str, Dict[str, Any]] = {}
    output_entries_by_name: Dict[str, Dict[str, Any]] = {}
    join_output_to_operation: Dict[str, str] = {}
    join_outputs_by_operation: Dict[str, Dict[str, str]] = {}
    resource_keys = (
        "assembly_file",
        "assembly_resource",
        "resource_file",
        "template_file",
    )

    for raw_entry in assemblies:
        if not isinstance(raw_entry, Mapping):
            raise BuilderError("Each assembly entry must be a mapping")
        entry = dict(raw_entry)
        name = entry.get("name")
        if not name:
            raise BuilderError("Each assembly entry must define a name")
        name = str(name)

        if _is_join_entry(entry):
            if name in join_entries_by_name:
                raise BuilderError(f"Duplicate join operation name '{name}'")
            if name in normal_entries_by_name:
                raise BuilderError(
                    f"Join operation name '{name}' collides with an assembly name"
                )
            if name in output_entries_by_name:
                raise BuilderError(
                    f"Join operation name '{name}' collides with a join output assembly name"
                )
            if not any(entry.get(resource_key) for resource_key in resource_keys):
                raise BuilderError(f"Join operation '{name}' must define resource_file")

            raw_outputs = entry.get("outputs")
            if not isinstance(raw_outputs, Mapping):
                raise BuilderError(
                    f"Join operation '{name}' must define outputs as a mapping"
                )
            if len(raw_outputs) < 2:
                raise BuilderError(
                    f"Join operation '{name}' must define at least two outputs"
                )

            outputs: Dict[str, str] = {}
            for raw_alias, raw_output_name in raw_outputs.items():
                alias = _validate_output_alias(raw_alias, name)
                output_name = _validate_output_assembly_name(raw_output_name, name)
                if alias in outputs:
                    raise BuilderError(
                        f"Join operation '{name}' repeats output alias '{alias}'"
                    )
                if output_name in normal_entries_by_name:
                    raise BuilderError(
                        f"Join output assembly name '{output_name}' collides with a normal assembly name"
                    )
                if output_name in join_entries_by_name or output_name == name:
                    raise BuilderError(
                        f"Join output assembly name '{output_name}' collides with a join operation name"
                    )
                if output_name in output_entries_by_name:
                    raise BuilderError(
                        f"Duplicate join output assembly name '{output_name}'"
                    )
                outputs[alias] = output_name
                output_entry = {
                    "name": output_name,
                    "kind": "joined_output",
                    "join_operation": name,
                    "join_output_alias": alias,
                    "_join_outputs": dict(outputs),
                    "_join_operation_entry": dict(entry),
                }
                for resource_key in resource_keys:
                    if resource_key in entry:
                        output_entry[resource_key] = entry[resource_key]
                output_entries_by_name[output_name] = output_entry
                join_output_to_operation[output_name] = name

            join_entry = dict(entry)
            join_entry["_join_outputs"] = dict(outputs)
            join_entries_by_name[name] = join_entry
            join_outputs_by_operation[name] = dict(outputs)
            for alias, output_name in outputs.items():
                output_entries_by_name[output_name]["_join_outputs"] = dict(outputs)
            continue

        if name in normal_entries_by_name:
            raise BuilderError(f"Duplicate assembly name '{name}'")
        if name in join_entries_by_name:
            raise BuilderError(
                f"Assembly name '{name}' collides with a join operation name"
            )
        if name in output_entries_by_name:
            raise BuilderError(
                f"Assembly name '{name}' collides with a join output assembly name"
            )
        normal_entries_by_name[name] = entry

    output_collisions = sorted(set(output_entries_by_name) & set(join_entries_by_name))
    if output_collisions:
        raise BuilderError(
            "Join output assembly name(s) collide with join operation name(s): "
            + ", ".join(output_collisions)
        )

    for operation_name, outputs in join_outputs_by_operation.items():
        join_entry = join_entries_by_name[operation_name]
        for alias, output_name in outputs.items():
            output_entries_by_name[output_name] = {
                **output_entries_by_name[output_name],
                "_join_outputs": dict(outputs),
                "_join_operation_entry": dict(join_entry),
            }

    public_entries_by_name: Dict[str, Dict[str, Any]] = {}
    public_entries_by_name.update(normal_entries_by_name)
    public_entries_by_name.update(output_entries_by_name)

    for operation_name, outputs in join_outputs_by_operation.items():
        join_entry = join_entries_by_name[operation_name]
        injections = resolve_injected_parts_config(join_entry)
        if not injections:
            raise BuilderError(
                f"Join operation '{operation_name}' must define inject_parts"
            )
        for kwarg_name, spec in injections.items():
            if spec["artifact"] != "assembly":
                raise BuilderError(
                    f"Join operation '{operation_name}' inject_parts.{kwarg_name} must inject the full assembly"
                )
            provider = spec["assembly"]
            if provider not in public_entries_by_name:
                raise BuilderError(
                    f"Join operation '{operation_name}' injects unknown assembly '{provider}'"
                )

        own_outputs = set(outputs.values())
        for dependency in dependency_names(join_entry):
            if dependency not in public_entries_by_name:
                raise BuilderError(
                    f"Join operation '{operation_name}' depends on unknown assembly '{dependency}'"
                )
            if dependency in own_outputs:
                raise BuilderError(
                    f"Join operation '{operation_name}' must not depend on its own output '{dependency}'"
                )

    public_entries = [dict(entry) for entry in normal_entries_by_name.values()] + [
        dict(entry) for entry in output_entries_by_name.values()
    ]
    return (
        public_entries,
        public_entries_by_name,
        join_entries_by_name,
        join_output_to_operation,
        join_outputs_by_operation,
    )


def assemblies_by_name(
    assemblies: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    _, by_name, _, _, _ = _normalize_join_declarations(assemblies)
    return by_name


def resolve_injected_parts_config(
    entry: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    config = entry.get("inject_parts", {})
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise BuilderError(
            f"inject_parts for assembly '{entry.get('name')}' must be a mapping"
        )

    normalized: Dict[str, Dict[str, str]] = {}
    for kwarg_name, raw_spec in config.items():
        if not isinstance(raw_spec, str):
            raise BuilderError(
                f"inject_parts.{kwarg_name} for assembly '{entry.get('name')}' must be a string reference"
            )

        assembly_name = raw_spec
        artifact = "assembly"
        if "." in raw_spec:
            assembly_name, artifact = raw_spec.split(".", 1)
        if not assembly_name or not artifact:
            raise BuilderError(
                f"inject_parts.{kwarg_name} for assembly '{entry.get('name')}' must be '<assembly>' or '<assembly>.<artifact>'"
            )

        normalized[str(kwarg_name)] = {
            "assembly": str(assembly_name),
            "artifact": str(artifact),
        }
    return normalized


def _merge_mapping(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(existing, value)
        else:
            merged[key] = value
    return merged


def _builder_defaults(config_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not config_data:
        return {}

    raw_defaults = config_data.get("builder_defaults")
    if raw_defaults is None:
        raw_defaults = config_data.get("BuilderDefaults", {})
    if raw_defaults is None:
        return {}
    if not isinstance(raw_defaults, Mapping):
        raise BuilderError("builder_defaults must be a mapping")
    return dict(raw_defaults)


def builder_section(
    resource_data: Mapping[str, Any],
    section_name: str,
    config_data: Optional[Mapping[str, Any]] = None,
    *,
    output_alias: Optional[str] = None,
) -> Dict[str, Any]:
    defaults = _builder_defaults(config_data)
    default_section = defaults.get(section_name, {})
    if default_section is None:
        default_section = {}
    if not isinstance(default_section, Mapping):
        raise BuilderError(f"builder_defaults.{section_name} must be a mapping")

    resource_builder_data = resource_data.get("Builder", {})
    if resource_builder_data is None:
        resource_builder_data = {}
    if not isinstance(resource_builder_data, Mapping):
        raise BuilderError("Builder section must be a mapping")

    if output_alias is not None:
        raw_outputs = resource_builder_data.get("Outputs", {})
        if raw_outputs is None:
            raw_outputs = {}
        if not isinstance(raw_outputs, Mapping):
            raise BuilderError("Builder.Outputs must be a mapping")
        raw_output = raw_outputs.get(output_alias)
        if raw_output is not None:
            if not isinstance(raw_output, Mapping):
                raise BuilderError(f"Builder.Outputs.{output_alias} must be a mapping")
            output_section = raw_output.get(section_name)
            if output_section is not None:
                if not isinstance(output_section, Mapping):
                    raise BuilderError(
                        f"Builder.Outputs.{output_alias}.{section_name} must be a mapping"
                    )
                return _merge_mapping(default_section, output_section)

    section = resource_builder_data.get(section_name, {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise BuilderError(f"Builder.{section_name} must be a mapping")
    return _merge_mapping(default_section, section)


def scene_dependency_names(
    resource_data: Mapping[str, Any],
    mode: str,
    config_data: Optional[Mapping[str, Any]],
    *,
    output_alias: Optional[str] = None,
    join_outputs: Optional[Mapping[str, str]] = None,
    join_inputs: Optional[Mapping[str, str]] = None,
) -> List[str]:
    section_name = "Visualization" if mode == "visualization" else "Production"
    section = builder_section(
        resource_data,
        section_name,
        config_data,
        output_alias=output_alias,
    )
    parts = section.get("parts")
    if not isinstance(parts, list):
        return []

    dependency_names_set: set[str] = set()
    for rule in parts:
        if not isinstance(rule, Mapping):
            raise BuilderError(f"Builder.{section_name}.parts items must be mappings")
        source = str(rule.get("source", "self"))
        dependency_name = rule.get("assembly")
        if source in {"dependencies", "dependency"}:
            if dependency_name:
                dependency_names_set.add(str(dependency_name))
            continue
        if source == "output":
            if dependency_name:
                output_name = (join_outputs or {}).get(str(dependency_name))
                if output_name:
                    dependency_names_set.add(str(output_name))
            else:
                dependency_names_set.update(
                    str(name) for name in (join_outputs or {}).values()
                )
            continue
        if source in {"injected", "inject"}:
            if dependency_name:
                input_name = (join_inputs or {}).get(str(dependency_name))
                if input_name:
                    dependency_names_set.add(str(input_name))
            else:
                dependency_names_set.update(
                    str(name) for name in (join_inputs or {}).values()
                )
    return sorted(dependency_names_set)


def placement_section(config_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not config_data:
        return {}

    placement = config_data.get("placement")
    if placement is None:
        placement = config_data.get("Placement", {})
    if placement is None:
        return {}
    if not isinstance(placement, Mapping):
        raise BuilderError("placement section must be a mapping")
    return dict(placement)


def placement_alignments(
    config_data: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    placement = placement_section(config_data)
    alignments = placement.get("alignments", [])
    if alignments is None:
        return []
    if not isinstance(alignments, list):
        raise BuilderError("placement.alignments must be a list")
    return [dict(item) for item in alignments]


def _placement_step_error(
    index: int,
    alignment: Mapping[str, Any],
    message: str,
) -> BuilderError:
    keys = sorted(str(key) for key in alignment)
    details = [f"placement.alignments[{index}]: {message}"]
    if keys:
        details.append("keys present: " + ", ".join(f"'{key}'" for key in keys))
    else:
        details.append("entry is empty")

    if "rigd_group" in alignment and "rigid_group" not in alignment:
        details.append("did you mean 'rigid_group' instead of 'rigd_group'?")

    return BuilderError("; ".join(details))


def _placement_step_summary(step: PlacementStep) -> str:
    if step.is_rigid_group:
        return (
            f"rigid_group {list(step.rigid_group_assembly_names)} "
            f"to {step.target_assembly_name}"
        )
    if step.target_reference is None:
        return str(step.moving_reference)
    return f"{step.moving_reference} to {step.target_reference}"


def _placement_rigid_motion_error(
    step: PlacementStep,
    moving_assembly_name: str,
    target_assembly_name: str,
    moving_group: Iterable[str],
) -> BuilderError:
    return _placement_step_error(
        step.index,
        step.spec,
        (
            f"{_placement_step_summary(step)} attempts to move rigidly attached "
            f"assemblies '{moving_assembly_name}' and '{target_assembly_name}' "
            "relative to each other; "
            f"placement spec: {step.spec}; "
            f"'{moving_assembly_name}' is already rigidly connected with "
            f"{sorted(moving_group)}"
        ),
    )


def parse_part_reference(
    reference: str,
    known_assembly_names: Sequence[str],
) -> tuple[str, str]:
    normalized = str(reference).strip()
    if not normalized:
        raise BuilderError("Placement part reference must not be empty")
    for assembly_name in known_assembly_names:
        prefix = f"{assembly_name}."
        if normalized == assembly_name:
            return assembly_name, "leader"
        if normalized.startswith(prefix):
            selector = normalized[len(prefix) :]
            if not selector:
                return assembly_name, "leader"
            return assembly_name, selector
    raise BuilderError(f"Unknown assembly reference '{reference}' in placement")


def _add_edge_kind(
    graph: nx.DiGraph,
    source: str,
    target: str,
    kind: str,
) -> None:
    if graph.has_edge(source, target):
        existing = set(graph.edges[source, target].get("kinds", set()))
        existing.add(kind)
        graph.edges[source, target]["kinds"] = existing
        return
    graph.add_edge(source, target, kinds={kind})


def _build_placement_steps(
    by_name: Mapping[str, Mapping[str, Any]],
    config_data: Optional[Mapping[str, Any]],
) -> List[PlacementStep]:
    known_assembly_names = sorted(by_name, key=len, reverse=True)
    steps: List[PlacementStep] = []
    for index, alignment in enumerate(placement_alignments(config_data)):
        rigid_group = alignment.get("rigid_group")
        if rigid_group is not None:
            forbidden_keys = []
            for key in (
                "part",
                "alignment",
                "post_rotation",
                "post_translation",
                "stack_gap",
                "axes",
                "rigid_attach",
            ):
                if key == "rigid_attach":
                    if key in alignment:
                        forbidden_keys.append(key)
                    continue
                if alignment.get(key) is not None:
                    forbidden_keys.append(key)
            if forbidden_keys:
                raise BuilderError(
                    "Placement steps with 'rigid_group' must not also define "
                    + ", ".join(f"'{key}'" for key in forbidden_keys)
                )
            target_reference = alignment.get("to")
            if target_reference is None:
                raise BuilderError(
                    "Placement steps with 'rigid_group' also require 'to'"
                )
            if not isinstance(rigid_group, list):
                raise BuilderError("placement rigid_group must be a list")
            if len(rigid_group) < 1:
                raise BuilderError(
                    "placement rigid_group must list at least one assembly"
                )

            target_assembly_name, _ = parse_part_reference(
                str(target_reference), known_assembly_names
            )

            rigid_group_assembly_names: List[str] = []
            seen_group_members: set[str] = set()
            for raw_reference in rigid_group:
                if not isinstance(raw_reference, str):
                    raise BuilderError(
                        "placement rigid_group entries must be assembly names"
                    )
                assembly_name, selector = parse_part_reference(
                    raw_reference, known_assembly_names
                )
                if raw_reference.strip() != assembly_name or selector != "leader":
                    raise BuilderError(
                        "placement rigid_group entries must reference assemblies, not sub-parts"
                    )
                if assembly_name in seen_group_members:
                    raise BuilderError(
                        "placement rigid_group must not repeat assemblies"
                    )
                if assembly_name == target_assembly_name:
                    raise BuilderError(
                        "placement rigid_group must not include the same assembly as 'to'"
                    )
                seen_group_members.add(assembly_name)
                rigid_group_assembly_names.append(assembly_name)

            steps.append(
                PlacementStep(
                    index=index,
                    spec=dict(alignment),
                    moving_reference=None,
                    target_reference=str(target_reference),
                    moving_assembly_name=None,
                    target_assembly_name=target_assembly_name,
                    rigid_attach=False,
                    rigid_group_assembly_names=tuple(rigid_group_assembly_names),
                )
            )
            continue

        moving_reference = alignment.get("part")
        target_reference = alignment.get("to")
        has_alignment = alignment.get("alignment") is not None
        has_post_rotation = alignment.get("post_rotation") is not None
        has_post_translation = alignment.get("post_translation") is not None
        if "rigid_attach" in alignment:
            raise _placement_step_error(
                index,
                alignment,
                "placement rigid_attach is no longer supported; use a separate "
                "'rigid_group' placement step",
            )
        if not moving_reference:
            raise _placement_step_error(
                index,
                alignment,
                "placement alignment requires 'part' or 'rigid_group'",
            )
        if has_alignment and not target_reference:
            raise _placement_step_error(
                index,
                alignment,
                "Placement steps with 'alignment' also require 'to'",
            )
        if not has_alignment and not has_post_rotation and not has_post_translation:
            raise _placement_step_error(
                index,
                alignment,
                "Each placement step requires 'alignment', 'post_rotation', or 'post_translation'",
            )
        moving_assembly_name, _ = parse_part_reference(
            str(moving_reference), known_assembly_names
        )
        target_assembly_name = None
        if target_reference:
            target_assembly_name, _ = parse_part_reference(
                str(target_reference), known_assembly_names
            )
        steps.append(
            PlacementStep(
                index=index,
                spec=dict(alignment),
                moving_reference=str(moving_reference),
                target_reference=(
                    str(target_reference) if target_reference is not None else None
                ),
                moving_assembly_name=moving_assembly_name,
                target_assembly_name=target_assembly_name,
                rigid_attach=False,
            )
        )
    return steps


def _rigid_group_members(
    rigidity_graph: nx.Graph,
    assembly_name: str,
) -> set[str]:
    if assembly_name not in rigidity_graph:
        rigidity_graph.add_node(assembly_name)
    return set(nx.node_connected_component(rigidity_graph, assembly_name))


def _rigid_group_dependency_pairs(
    assembly_names: Sequence[str],
    target_assembly_name: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (assembly_name, target_assembly_name) for assembly_name in assembly_names
    )


def _rigid_group_source_members(
    assembly_names: Sequence[str],
) -> set[str]:
    return set(assembly_names)


def _build_placement_execution_dag(
    placement_steps: Sequence[PlacementStep],
) -> tuple[nx.DiGraph, Dict[str, int], Dict[str, int], List[PlacementStep]]:
    graph = nx.DiGraph()
    first_moving_alignment_index: Dict[str, int] = {}
    first_involved_alignment_index: Dict[str, int] = {}
    latest_effect_by_assembly: Dict[str, int] = {}
    rigidity_graph = nx.Graph()
    enriched_steps: List[PlacementStep] = []

    for step in placement_steps:
        if step.is_rigid_group:
            rigidity_graph.add_nodes_from(step.rigid_group_assembly_names)
            if step.target_assembly_name is not None:
                rigidity_graph.add_node(step.target_assembly_name)
            continue
        if step.moving_assembly_name is not None:
            rigidity_graph.add_node(step.moving_assembly_name)
        if step.target_assembly_name is not None:
            rigidity_graph.add_node(step.target_assembly_name)

    for step in placement_steps:
        if step.is_rigid_group:
            if step.target_assembly_name is None:
                raise BuilderError(
                    "Placement steps with 'rigid_group' also require 'to'"
                )

            graph.add_node(
                step.index,
                moving_assembly_name=None,
                target_assembly_name=step.target_assembly_name,
            )

            target_group = _rigid_group_members(
                rigidity_graph, step.target_assembly_name
            )
            group_members = set(step.rigid_group_assembly_names)
            source_members = _rigid_group_source_members(
                step.rigid_group_assembly_names
            )
            affected_members: set[str] = set()
            for assembly_name in sorted(source_members):
                moving_group = _rigid_group_members(rigidity_graph, assembly_name)
                if step.target_assembly_name in moving_group:
                    raise _placement_rigid_motion_error(
                        step,
                        assembly_name,
                        step.target_assembly_name,
                        moving_group,
                    )
                affected_members.update(moving_group)

            dependencies = {
                predecessor
                for assembly_name in sorted(affected_members | target_group)
                for predecessor in [latest_effect_by_assembly.get(assembly_name)]
                if predecessor is not None
            }
            for predecessor in sorted(dependencies):
                graph.add_edge(predecessor, step.index)

            for assembly_name in sorted(affected_members):
                latest_effect_by_assembly[assembly_name] = step.index

            for assembly_name in sorted(group_members):
                first_involved_alignment_index.setdefault(assembly_name, step.index)
            first_involved_alignment_index.setdefault(
                step.target_assembly_name, step.index
            )

            enriched_steps.append(
                PlacementStep(
                    index=step.index,
                    spec=step.spec,
                    moving_reference=step.moving_reference,
                    target_reference=step.target_reference,
                    moving_assembly_name=step.moving_assembly_name,
                    target_assembly_name=step.target_assembly_name,
                    rigid_attach=step.rigid_attach,
                    rigid_group_assembly_names=step.rigid_group_assembly_names,
                    affected_assembly_names=tuple(sorted(affected_members)),
                )
            )

            for source, target in _rigid_group_dependency_pairs(
                step.rigid_group_assembly_names,
                step.target_assembly_name,
            ):
                rigidity_graph.add_edge(source, target)
            continue

        moving_group = _rigid_group_members(rigidity_graph, step.moving_assembly_name)
        target_group: set[str] = set()
        if step.target_assembly_name is not None:
            target_group = _rigid_group_members(
                rigidity_graph, step.target_assembly_name
            )
            if step.target_assembly_name in moving_group:
                raise _placement_rigid_motion_error(
                    step,
                    step.moving_assembly_name,
                    step.target_assembly_name,
                    moving_group,
                )

        graph.add_node(
            step.index,
            moving_assembly_name=step.moving_assembly_name,
            target_assembly_name=step.target_assembly_name,
        )

        dependencies: set[int] = set()
        for assembly_name in sorted(moving_group | target_group):
            predecessor = latest_effect_by_assembly.get(assembly_name)
            if predecessor is not None:
                dependencies.add(predecessor)

        for predecessor in sorted(dependencies):
            graph.add_edge(predecessor, step.index)

        for assembly_name in sorted(moving_group):
            latest_effect_by_assembly[assembly_name] = step.index

        first_moving_alignment_index.setdefault(step.moving_assembly_name, step.index)
        first_involved_alignment_index.setdefault(step.moving_assembly_name, step.index)
        if step.target_assembly_name is not None:
            first_involved_alignment_index.setdefault(
                step.target_assembly_name, step.index
            )

        enriched_steps.append(
            PlacementStep(
                index=step.index,
                spec=step.spec,
                moving_reference=step.moving_reference,
                target_reference=step.target_reference,
                moving_assembly_name=step.moving_assembly_name,
                target_assembly_name=step.target_assembly_name,
                rigid_attach=step.rigid_attach,
                rigid_group_assembly_names=step.rigid_group_assembly_names,
                affected_assembly_names=tuple(sorted(moving_group)),
            )
        )
    return (
        graph,
        first_moving_alignment_index,
        first_involved_alignment_index,
        enriched_steps,
    )


def relevant_placement_alignment_indices(
    model: BuilderGraphModel,
    root_assemblies: Iterable[str],
    *,
    stop_index: Optional[int] = None,
    include_rigid_effects: bool = False,
) -> List[int]:
    roots = {str(name) for name in root_assemblies}
    if not roots:
        return []

    root_steps = [
        step.index
        for step in model.placement_steps
        if (
            roots.intersection(step.affected_assembly_names)
            if include_rigid_effects
            else (
                any(
                    assembly_name in roots
                    for assembly_name in step.rigid_group_assembly_names
                )
                if step.is_rigid_group
                else step.moving_assembly_name in roots
            )
        )
        and (stop_index is None or step.index < stop_index)
    ]
    if not root_steps:
        return []

    relevant: set[int] = set(root_steps)
    for step_index in root_steps:
        relevant.update(nx.ancestors(model.placement_execution_dag, step_index))
    return sorted(relevant)


def placement_dependency_closure(
    model: BuilderGraphModel,
    root_assemblies: Iterable[str],
    *,
    stop_index: Optional[int] = None,
    include_rigid_effects: bool = False,
) -> List[str]:
    needed = {str(name) for name in root_assemblies}
    steps_by_index = {step.index: step for step in model.placement_steps}
    for step_index in relevant_placement_alignment_indices(
        model,
        needed,
        stop_index=stop_index,
        include_rigid_effects=include_rigid_effects,
    ):
        step = steps_by_index[step_index]
        if include_rigid_effects:
            needed.update(step.affected_assembly_names)
        else:
            needed_before_step = set(needed)
            if step.is_rigid_group:
                if step.target_assembly_name is None:
                    raise BuilderError(
                        "Placement steps with 'rigid_group' also require 'to'"
                    )
                for source, target in _rigid_group_dependency_pairs(
                    step.rigid_group_assembly_names,
                    step.target_assembly_name,
                ):
                    if source in needed_before_step:
                        needed.add(source)
                        needed.add(target)
            elif step.moving_assembly_name is not None:
                needed.add(step.moving_assembly_name)
        if step.target_assembly_name is not None:
            needed.add(step.target_assembly_name)
    return sorted(needed)


def _step_referenced_assemblies(step: PlacementStep) -> set[str]:
    referenced: set[str] = set()
    if step.is_rigid_group:
        referenced.update(step.rigid_group_assembly_names)
    elif step.moving_assembly_name is not None:
        referenced.add(step.moving_assembly_name)
    if step.target_assembly_name is not None:
        referenced.add(step.target_assembly_name)
    return referenced


def _consumer_placement_boundary_index(
    consumer_name: str,
    entry: Mapping[str, Any],
    placement_steps: Sequence[PlacementStep],
    model: BuilderGraphModel,
) -> Optional[int]:
    if _is_join_entry(entry):
        output_names: Sequence[str] = tuple(
            str(name)
            for name in model.join_outputs_by_operation.get(
                str(entry["name"]),
                {},
            ).values()
        )
    else:
        output_names = (consumer_name,)

    boundary_candidates = [
        index
        for output_name in output_names
        for index in (
            model.first_moving_alignment_index.get(output_name),
            model.first_involved_alignment_index.get(output_name),
        )
        if index is not None
    ]
    if boundary_candidates:
        return min(boundary_candidates)
    if _is_join_entry(entry):
        return max((step.index for step in placement_steps), default=-1) + 1
    return None


def _required_injected_provider_placement_step_indices(
    model: BuilderGraphModel,
    consumer_name: str,
    entry: Mapping[str, Any],
    provider_assemblies: Iterable[str],
) -> List[int]:
    providers = {str(name) for name in provider_assemblies}
    if not providers:
        return []

    boundary_index = _consumer_placement_boundary_index(
        consumer_name,
        entry,
        model.placement_steps,
        model,
    )
    if boundary_index is None:
        return []

    root_indices = {
        step.index
        for step in model.placement_steps
        if step.index < boundary_index
        and providers.intersection(step.affected_assembly_names)
    }
    if not root_indices:
        return []

    required_indices: set[int] = set(root_indices)
    for step_index in root_indices:
        required_indices.update(nx.ancestors(model.placement_execution_dag, step_index))
    return sorted(
        step_index for step_index in required_indices if step_index < boundary_index
    )


def required_injected_provider_placement_step_indices(
    model: BuilderGraphModel,
    consumer_name: str,
    provider_assembly_name: str,
) -> List[int]:
    entry = model.build_items_by_name.get(str(consumer_name))
    if entry is None:
        return []
    return _required_injected_provider_placement_step_indices(
        model,
        str(consumer_name),
        entry,
        [str(provider_assembly_name)],
    )


def _compute_placement_build_dependencies(
    assemblies: Mapping[str, Mapping[str, Any]],
    placement_steps: Sequence[PlacementStep],
    model: BuilderGraphModel | None = None,
) -> Dict[str, List[str]]:
    if model is None:
        raise BuilderError("Builder graph model is required")

    implicit_dependencies: Dict[str, List[str]] = {name: [] for name in assemblies}
    steps_by_index = {step.index: step for step in placement_steps}
    for assembly_name, entry in assemblies.items():
        injected_assemblies = {
            spec["assembly"]
            for spec in resolve_injected_parts_config(entry).values()
            if spec.get("assembly")
        }
        if not injected_assemblies:
            continue

        required_indices = _required_injected_provider_placement_step_indices(
            model,
            assembly_name,
            entry,
            injected_assemblies,
        )
        if not required_indices:
            continue

        required: set[str] = set()
        for step_index in required_indices:
            required.update(_step_referenced_assemblies(steps_by_index[step_index]))
        required.discard(assembly_name)
        implicit_dependencies[assembly_name] = sorted(required)

    return implicit_dependencies


def build_graph_model(
    assemblies: Sequence[Mapping[str, Any]],
    config_data: Optional[Mapping[str, Any]] = None,
) -> BuilderGraphModel:
    (
        public_assemblies,
        by_name,
        join_operations_by_name,
        join_output_to_operation,
        join_outputs_by_operation,
    ) = _normalize_join_declarations(assemblies)

    build_items_by_name: Dict[str, Dict[str, Any]] = {
        name: dict(entry) for name, entry in by_name.items()
    }
    for operation_name, entry in join_operations_by_name.items():
        build_items_by_name[join_node_name(operation_name)] = dict(entry)

    assembly_dependency_graph = nx.DiGraph()
    injection_graph = nx.MultiDiGraph()
    for assembly_name in sorted(by_name):
        assembly_dependency_graph.add_node(assembly_name)
        injection_graph.add_node(assembly_name)
    for operation_name in sorted(join_operations_by_name):
        assembly_dependency_graph.add_node(join_node_name(operation_name))

    for assembly_name, entry in by_name.items():
        if _is_joined_output_entry(entry):
            operation_name = str(entry["join_operation"])
            _add_edge_kind(
                assembly_dependency_graph,
                join_node_name(operation_name),
                assembly_name,
                "join_output",
            )
            continue
        for dependency in dependency_names(entry):
            if dependency not in by_name:
                raise BuilderError(
                    f"Assembly '{assembly_name}' depends on unknown assembly '{dependency}'"
                )
            _add_edge_kind(
                assembly_dependency_graph,
                dependency,
                assembly_name,
                "declared_dependency",
            )
        for kwarg_name, spec in resolve_injected_parts_config(entry).items():
            provider = spec["assembly"]
            if provider not in by_name:
                raise BuilderError(
                    f"Assembly '{assembly_name}' injects unknown assembly '{provider}'"
                )
            _add_edge_kind(
                assembly_dependency_graph,
                provider,
                assembly_name,
                "injection",
            )
            injection_graph.add_edge(
                provider,
                assembly_name,
                kwarg_name=kwarg_name,
                artifact=spec["artifact"],
            )

    for operation_name, entry in join_operations_by_name.items():
        operation_node = join_node_name(operation_name)
        for dependency in dependency_names(entry):
            _add_edge_kind(
                assembly_dependency_graph,
                dependency,
                operation_node,
                "declared_dependency",
            )
        for kwarg_name, spec in resolve_injected_parts_config(entry).items():
            provider = spec["assembly"]
            _add_edge_kind(
                assembly_dependency_graph,
                provider,
                operation_node,
                "join_injection",
            )
            injection_graph.add_edge(
                provider,
                operation_node,
                kwarg_name=kwarg_name,
                artifact=spec["artifact"],
            )

    placement_steps = _build_placement_steps(by_name, config_data)
    (
        placement_execution_dag,
        first_moving_alignment_index,
        first_involved_alignment_index,
        placement_steps,
    ) = _build_placement_execution_dag(placement_steps)

    partial_model = BuilderGraphModel(
        assemblies=[dict(item) for item in public_assemblies],
        assemblies_by_name=by_name,
        assembly_dependency_graph=assembly_dependency_graph,
        injection_graph=injection_graph,
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies={},
        build_items_by_name=build_items_by_name,
        join_operations_by_name=join_operations_by_name,
        join_output_to_operation=join_output_to_operation,
        join_outputs_by_operation=join_outputs_by_operation,
    )
    placement_build_dependencies = _compute_placement_build_dependencies(
        build_items_by_name,
        placement_steps,
        model=partial_model,
    )
    return BuilderGraphModel(
        assemblies=[dict(item) for item in public_assemblies],
        assemblies_by_name=by_name,
        assembly_dependency_graph=assembly_dependency_graph,
        injection_graph=injection_graph,
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies=placement_build_dependencies,
        build_items_by_name=build_items_by_name,
        join_operations_by_name=join_operations_by_name,
        join_output_to_operation=join_output_to_operation,
        join_outputs_by_operation=join_outputs_by_operation,
    )


def _replace_placement_steps(
    model: BuilderGraphModel,
    placement_steps: Sequence[PlacementStep],
) -> BuilderGraphModel:
    (
        placement_execution_dag,
        first_moving_alignment_index,
        first_involved_alignment_index,
        placement_steps,
    ) = _build_placement_execution_dag(placement_steps)
    partial_model = BuilderGraphModel(
        assemblies=[dict(item) for item in model.assemblies],
        assemblies_by_name={
            name: dict(entry) for name, entry in model.assemblies_by_name.items()
        },
        assembly_dependency_graph=model.assembly_dependency_graph.copy(),
        injection_graph=model.injection_graph.copy(),
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies={},
        build_items_by_name={
            name: dict(entry) for name, entry in model.build_items_by_name.items()
        },
        join_operations_by_name={
            name: dict(entry) for name, entry in model.join_operations_by_name.items()
        },
        join_output_to_operation=dict(model.join_output_to_operation),
        join_outputs_by_operation={
            name: dict(outputs)
            for name, outputs in model.join_outputs_by_operation.items()
        },
    )
    placement_build_dependencies = _compute_placement_build_dependencies(
        partial_model.build_items_by_name,
        placement_steps,
        model=partial_model,
    )
    return BuilderGraphModel(
        assemblies=[dict(item) for item in model.assemblies],
        assemblies_by_name={
            name: dict(entry) for name, entry in model.assemblies_by_name.items()
        },
        assembly_dependency_graph=model.assembly_dependency_graph.copy(),
        injection_graph=model.injection_graph.copy(),
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies=placement_build_dependencies,
        build_items_by_name={
            name: dict(entry) for name, entry in model.build_items_by_name.items()
        },
        join_operations_by_name={
            name: dict(entry) for name, entry in model.join_operations_by_name.items()
        },
        join_output_to_operation=dict(model.join_output_to_operation),
        join_outputs_by_operation={
            name: dict(outputs)
            for name, outputs in model.join_outputs_by_operation.items()
        },
    )


def without_rigid_attach_effects(model: BuilderGraphModel) -> BuilderGraphModel:
    if not any(step.rigid_attach for step in model.placement_steps):
        return model

    stripped_steps = []
    for step in model.placement_steps:
        stripped_spec = dict(step.spec)
        if "rigid_attach" in stripped_spec:
            stripped_spec["rigid_attach"] = False
        stripped_steps.append(
            PlacementStep(
                index=step.index,
                spec=stripped_spec,
                moving_reference=step.moving_reference,
                target_reference=step.target_reference,
                moving_assembly_name=step.moving_assembly_name,
                target_assembly_name=step.target_assembly_name,
                rigid_attach=False,
                rigid_group_assembly_names=step.rigid_group_assembly_names,
            )
        )
    return _replace_placement_steps(model, stripped_steps)


def restrict_to_active_assemblies(
    model: BuilderGraphModel,
    active_assemblies: Iterable[str],
) -> BuilderGraphModel:
    requested_names = {str(name) for name in active_assemblies}
    active_build_item_names = {
        name for name in requested_names if name in model.build_items_by_name
    }
    active_public_names = {
        name for name in active_build_item_names if name in model.assemblies_by_name
    }
    for output_name in tuple(active_public_names):
        operation_name = model.join_output_to_operation.get(output_name)
        if operation_name is not None:
            active_build_item_names.add(join_node_name(operation_name))

    if active_build_item_names == set(model.build_items_by_name):
        return model

    filtered_steps = []
    for step in model.placement_steps:
        if step.is_rigid_group:
            if step.target_assembly_name not in active_public_names:
                continue
            active_group = tuple(
                assembly_name
                for assembly_name in step.rigid_group_assembly_names
                if assembly_name in active_public_names
            )
            if len(active_group) < 1:
                continue
            filtered_spec = dict(step.spec)
            filtered_spec["rigid_group"] = list(active_group)
            filtered_steps.append(
                PlacementStep(
                    index=step.index,
                    spec=filtered_spec,
                    moving_reference=None,
                    target_reference=step.target_reference,
                    moving_assembly_name=None,
                    target_assembly_name=step.target_assembly_name,
                    rigid_attach=False,
                    rigid_group_assembly_names=active_group,
                )
            )
            continue
        if step.moving_assembly_name not in active_public_names:
            continue
        if (
            step.target_assembly_name is not None
            and step.target_assembly_name not in active_public_names
        ):
            continue
        filtered_steps.append(step)
    (
        placement_execution_dag,
        first_moving_alignment_index,
        first_involved_alignment_index,
        placement_steps,
    ) = _build_placement_execution_dag(filtered_steps)
    partial_model = BuilderGraphModel(
        assemblies=[dict(item) for item in model.assemblies],
        assemblies_by_name={
            name: dict(entry) for name, entry in model.assemblies_by_name.items()
        },
        assembly_dependency_graph=model.assembly_dependency_graph.copy(),
        injection_graph=model.injection_graph.copy(),
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies={},
        build_items_by_name={
            name: dict(entry)
            for name, entry in model.build_items_by_name.items()
            if name in active_build_item_names
        },
        join_operations_by_name={
            name: dict(entry)
            for name, entry in model.join_operations_by_name.items()
            if join_node_name(name) in active_build_item_names
        },
        join_output_to_operation=dict(model.join_output_to_operation),
        join_outputs_by_operation={
            name: dict(outputs)
            for name, outputs in model.join_outputs_by_operation.items()
        },
    )
    recomputed_placement_build_dependencies = _compute_placement_build_dependencies(
        partial_model.build_items_by_name,
        placement_steps,
        model=partial_model,
    )
    filtered_placement_build_dependencies = {
        name: [
            dependency
            for dependency in dependencies
            if dependency in active_public_names
        ]
        for name, dependencies in recomputed_placement_build_dependencies.items()
        if name in active_build_item_names
    }

    return BuilderGraphModel(
        assemblies=[dict(item) for item in model.assemblies],
        assemblies_by_name={
            name: dict(entry) for name, entry in model.assemblies_by_name.items()
        },
        assembly_dependency_graph=model.assembly_dependency_graph.copy(),
        injection_graph=model.injection_graph.copy(),
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies=filtered_placement_build_dependencies,
        build_items_by_name={
            name: dict(entry) for name, entry in model.build_items_by_name.items()
        },
        join_operations_by_name={
            name: dict(entry) for name, entry in model.join_operations_by_name.items()
        },
        join_output_to_operation=dict(model.join_output_to_operation),
        join_outputs_by_operation={
            name: dict(outputs)
            for name, outputs in model.join_outputs_by_operation.items()
        },
    )


def expand_with_dependents(
    model: BuilderGraphModel,
    selected_names: Sequence[str],
) -> List[str]:
    missing = sorted(
        name for name in selected_names if name not in model.assemblies_by_name
    )
    if missing:
        raise BuilderError(f"Unknown assembly name(s): {', '.join(missing)}")

    expanded: set[str] = set()
    pending = list(selected_names)
    while pending:
        current = pending.pop()
        if current in expanded:
            continue
        expanded.add(current)
        pending.extend(sorted(model.assembly_dependency_graph.successors(current)))

    return sorted(name for name in expanded if name in model.assemblies_by_name)


def _canonicalize_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in cycle)
    if not normalized:
        return ()
    start_index = min(range(len(normalized)), key=lambda index: normalized[index])
    return tuple(
        normalized[(start_index + offset) % len(normalized)]
        for offset in range(len(normalized))
    )


def _format_cycle(graph: nx.DiGraph, cycle: Sequence[str]) -> str:
    ordered_cycle = _canonicalize_cycle(cycle)
    if not ordered_cycle:
        return "<empty cycle>"

    formatted = [ordered_cycle[0]]
    for index, source in enumerate(ordered_cycle):
        target = ordered_cycle[(index + 1) % len(ordered_cycle)]
        edge_data = graph.edges[source, target]
        kinds = edge_data.get("kinds", set())
        if isinstance(kinds, str):
            kind_text = kinds
        else:
            kind_text = ", ".join(sorted(str(kind) for kind in kinds)) or "unknown"
        formatted.append(f"-[{kind_text}]-> {target}")
    return " ".join(formatted)


def _dependency_cycle_debug_lines(
    graph: nx.DiGraph,
    *,
    max_cycles: int = 10,
) -> List[str]:
    raw_cycles = list(islice(nx.simple_cycles(graph), max_cycles + 1))
    if not raw_cycles:
        return ["unable to enumerate concrete cycles"]

    has_more = len(raw_cycles) > max_cycles
    cycles = raw_cycles[:max_cycles]
    unique_cycles = sorted({_canonicalize_cycle(cycle) for cycle in cycles})
    debug_lines = [
        f"cycle {index}: {_format_cycle(graph, cycle)}"
        for index, cycle in enumerate(unique_cycles, start=1)
    ]
    if has_more:
        debug_lines.append(
            f"showing first {len(cycles)} cycles; additional cycles omitted"
        )
    return debug_lines


def resolve_build_generation_names(
    model: BuilderGraphModel,
    selected_names: Optional[Iterable[str]] = None,
) -> List[List[str]]:
    requested = list(selected_names or model.assemblies_by_name.keys())
    missing = sorted(name for name in requested if name not in model.assemblies_by_name)
    if missing:
        raise BuilderError(f"Unknown assembly name(s): {', '.join(missing)}")

    expanded_requested: set[str] = set(requested)
    for assembly_name in requested:
        operation_name = model.join_output_to_operation.get(assembly_name)
        if operation_name is None:
            continue
        expanded_requested.update(
            model.join_outputs_by_operation[operation_name].values()
        )

    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(model.assembly_dependency_graph.nodes)
    dependency_graph.add_edges_from(model.assembly_dependency_graph.edges(data=True))
    for assembly_name, dependencies in model.placement_build_dependencies.items():
        for dependency in dependencies:
            if dependency not in model.assemblies_by_name:
                raise BuilderError(
                    f"Assembly '{assembly_name}' depends on unknown assembly '{dependency}'"
                )
            _add_edge_kind(
                dependency_graph,
                dependency,
                assembly_name,
                "placement_build",
            )

    required_names: set[str] = set()
    for assembly_name in sorted(expanded_requested):
        required_names.add(assembly_name)
        required_names.update(nx.ancestors(dependency_graph, assembly_name))

    for node_name in list(required_names):
        if not is_join_node_name(node_name):
            continue
        operation_name = join_operation_name_from_node(node_name)
        required_names.update(model.join_outputs_by_operation[operation_name].values())

    subgraph = dependency_graph.subgraph(sorted(required_names)).copy()
    try:
        generations = list(nx.topological_generations(subgraph))
    except nx.NetworkXUnfeasible as exc:
        debug_lines = _dependency_cycle_debug_lines(subgraph)
        raise BuilderError(
            "Cyclic dependency detected in assembly graph:\n" + "\n".join(debug_lines)
        ) from exc

    return [sorted(str(name) for name in generation) for generation in generations]
