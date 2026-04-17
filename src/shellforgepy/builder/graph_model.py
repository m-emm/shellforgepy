"""Explicit graph model for declarative builder dependency reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx

from .errors import BuilderError


@dataclass(frozen=True)
class PlacementStep:
    index: int
    spec: Dict[str, Any]
    moving_reference: str
    target_reference: Optional[str]
    moving_assembly_name: str
    target_assembly_name: Optional[str]
    rigid_attach: bool
    affected_assembly_names: tuple[str, ...] = ()


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


def assemblies_by_name(
    assemblies: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in assemblies:
        name = entry.get("name")
        if not name:
            raise BuilderError("Each assembly entry must define a name")
        if name in by_name:
            raise BuilderError(f"Duplicate assembly name '{name}'")
        by_name[str(name)] = dict(entry)
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
) -> List[str]:
    section_name = "Visualization" if mode == "visualization" else "Production"
    section = builder_section(resource_data, section_name, config_data)
    parts = section.get("parts")
    if not isinstance(parts, list):
        return []

    dependency_names_set: set[str] = set()
    for rule in parts:
        if not isinstance(rule, Mapping):
            raise BuilderError(f"Builder.{section_name}.parts items must be mappings")
        source = str(rule.get("source", "self"))
        if source not in {"dependencies", "dependency"}:
            continue
        dependency_name = rule.get("assembly")
        if dependency_name:
            dependency_names_set.add(str(dependency_name))
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
        moving_reference = alignment.get("part")
        target_reference = alignment.get("to")
        has_alignment = alignment.get("alignment") is not None
        has_post_rotation = alignment.get("post_rotation") is not None
        rigid_attach = alignment.get("rigid_attach", False)
        if not moving_reference:
            raise BuilderError("Each placement alignment requires 'part'")
        if has_alignment and not target_reference:
            raise BuilderError("Placement steps with 'alignment' also require 'to'")
        if not has_alignment and not has_post_rotation:
            raise BuilderError(
                "Each placement step requires either 'alignment' or 'post_rotation'"
            )
        if rigid_attach and target_reference is None:
            raise BuilderError("Placement steps with 'rigid_attach' also require 'to'")
        if not isinstance(rigid_attach, bool):
            raise BuilderError("placement alignment rigid_attach must be a boolean")
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
                rigid_attach=rigid_attach,
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
        rigidity_graph.add_node(step.moving_assembly_name)
        if step.target_assembly_name is not None:
            rigidity_graph.add_node(step.target_assembly_name)

    for step in placement_steps:
        moving_group = _rigid_group_members(rigidity_graph, step.moving_assembly_name)
        target_group: set[str] = set()
        if step.target_assembly_name is not None:
            target_group = _rigid_group_members(
                rigidity_graph, step.target_assembly_name
            )
            if step.target_assembly_name in moving_group:
                raise BuilderError(
                    "Placement step "
                    f"{step.index} attempts to move rigidly attached assemblies "
                    f"'{step.moving_assembly_name}' and '{step.target_assembly_name}' "
                    "relative to each other"
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
                affected_assembly_names=tuple(sorted(moving_group)),
            )
        )

        if step.rigid_attach and step.target_assembly_name is not None:
            rigidity_graph.add_edge(
                step.moving_assembly_name,
                step.target_assembly_name,
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
            else step.moving_assembly_name in roots
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
            needed.add(step.moving_assembly_name)
        if step.target_assembly_name is not None:
            needed.add(step.target_assembly_name)
    return sorted(needed)


def _compute_placement_build_dependencies(
    assemblies: Mapping[str, Mapping[str, Any]],
    placement_steps: Sequence[PlacementStep],
    model: BuilderGraphModel | None = None,
) -> Dict[str, List[str]]:
    if model is None:
        raise BuilderError("Builder graph model is required")

    implicit_dependencies: Dict[str, List[str]] = {name: [] for name in assemblies}
    for assembly_name, entry in assemblies.items():
        injected_assemblies = {
            spec["assembly"]
            for spec in resolve_injected_parts_config(entry).values()
            if spec.get("assembly")
        }
        if not injected_assemblies:
            continue

        boundary_index = model.first_moving_alignment_index.get(assembly_name)
        required = set(
            placement_dependency_closure(
                model,
                injected_assemblies,
                stop_index=boundary_index,
                include_rigid_effects=False,
            )
        )
        required.discard(assembly_name)
        implicit_dependencies[assembly_name] = sorted(required)

    return implicit_dependencies


def build_graph_model(
    assemblies: Sequence[Mapping[str, Any]],
    config_data: Optional[Mapping[str, Any]] = None,
) -> BuilderGraphModel:
    by_name = assemblies_by_name(assemblies)

    assembly_dependency_graph = nx.DiGraph()
    injection_graph = nx.MultiDiGraph()
    for assembly_name in sorted(by_name):
        assembly_dependency_graph.add_node(assembly_name)
        injection_graph.add_node(assembly_name)

    for assembly_name, entry in by_name.items():
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

    placement_steps = _build_placement_steps(by_name, config_data)
    (
        placement_execution_dag,
        first_moving_alignment_index,
        first_involved_alignment_index,
        placement_steps,
    ) = _build_placement_execution_dag(placement_steps)

    partial_model = BuilderGraphModel(
        assemblies=[dict(item) for item in assemblies],
        assemblies_by_name=by_name,
        assembly_dependency_graph=assembly_dependency_graph,
        injection_graph=injection_graph,
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies={},
    )
    placement_build_dependencies = _compute_placement_build_dependencies(
        by_name,
        placement_steps,
        model=partial_model,
    )
    return BuilderGraphModel(
        assemblies=[dict(item) for item in assemblies],
        assemblies_by_name=by_name,
        assembly_dependency_graph=assembly_dependency_graph,
        injection_graph=injection_graph,
        placement_steps=placement_steps,
        placement_execution_dag=placement_execution_dag,
        first_moving_alignment_index=first_moving_alignment_index,
        first_involved_alignment_index=first_involved_alignment_index,
        placement_build_dependencies=placement_build_dependencies,
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
    )
    placement_build_dependencies = _compute_placement_build_dependencies(
        partial_model.assemblies_by_name,
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
            )
        )
    return _replace_placement_steps(model, stripped_steps)


def restrict_to_active_assemblies(
    model: BuilderGraphModel,
    active_assemblies: Iterable[str],
) -> BuilderGraphModel:
    active_names = {
        str(name) for name in active_assemblies if str(name) in model.assemblies_by_name
    }
    if active_names == set(model.assemblies_by_name):
        return model

    filtered_steps = [
        step
        for step in model.placement_steps
        if step.moving_assembly_name in active_names
        and (
            step.target_assembly_name is None
            or step.target_assembly_name in active_names
        )
    ]
    (
        placement_execution_dag,
        first_moving_alignment_index,
        first_involved_alignment_index,
        placement_steps,
    ) = _build_placement_execution_dag(filtered_steps)
    filtered_placement_build_dependencies = {
        name: [dependency for dependency in dependencies if dependency in active_names]
        for name, dependencies in model.placement_build_dependencies.items()
        if name in active_names
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

    return sorted(expanded)


def resolve_build_generation_names(
    model: BuilderGraphModel,
    selected_names: Optional[Iterable[str]] = None,
) -> List[List[str]]:
    requested = list(selected_names or model.assemblies_by_name.keys())
    missing = sorted(name for name in requested if name not in model.assemblies_by_name)
    if missing:
        raise BuilderError(f"Unknown assembly name(s): {', '.join(missing)}")

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
    for assembly_name in requested:
        required_names.add(assembly_name)
        required_names.update(nx.ancestors(dependency_graph, assembly_name))

    subgraph = dependency_graph.subgraph(sorted(required_names)).copy()
    try:
        generations = list(nx.topological_generations(subgraph))
    except nx.NetworkXUnfeasible as exc:
        raise BuilderError("Cyclic dependency detected in assembly graph") from exc

    return [sorted(str(name) for name in generation) for generation in generations]
