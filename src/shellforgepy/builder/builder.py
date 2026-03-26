"""Declarative assembly builder for ShellForgePy."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx
import yaml
from shellforgepy.construct.part_parameters import PartParameters

_logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r"\$\{([A-Za-z0-9_./:-]+)\}")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class BuilderError(Exception):
    """Raised when declarative build resolution or execution fails."""


class _BuilderLoader(yaml.SafeLoader):
    """YAML loader supporting small CloudFormation-style tags."""


def _construct_ref(loader: yaml.SafeLoader, node: yaml.Node) -> Dict[str, str]:
    return {"$ref": loader.construct_scalar(node)}


def _construct_sub(loader: yaml.SafeLoader, node: yaml.Node) -> Dict[str, Any]:
    if isinstance(node, yaml.ScalarNode):
        return {"$sub": loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {"$sub": loader.construct_sequence(node)}
    return {"$sub": loader.construct_mapping(node)}


def _construct_expr(loader: yaml.SafeLoader, node: yaml.Node) -> Dict[str, Any]:
    if isinstance(node, yaml.ScalarNode):
        return {"$expr": loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {"$expr": loader.construct_sequence(node)}
    return {"$expr": loader.construct_mapping(node)}


for _tag, _constructor in {
    "!Ref": _construct_ref,
    "!Sub": _construct_sub,
    "!Expr": _construct_expr,
}.items():
    _BuilderLoader.add_constructor(_tag, _constructor)


@dataclass(frozen=True)
class _ResolvedAssembly:
    name: str
    entry: Dict[str, Any]
    resource_path: Path
    resource_data: Dict[str, Any]
    public_parameters: Dict[str, Any]
    generator_kwargs: Dict[str, Any]
    generator_path: str
    logical_part_name: str
    parameter_hash: str


@dataclass(frozen=True)
class _DependencyInjection:
    kwarg_name: str
    assembly_name: str
    artifact: str
    source_parameter_hash: str
    step_path: Path
    part: Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "item"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=_BuilderLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise BuilderError(
            f"Expected YAML mapping at {path}, got {type(loaded).__name__}"
        )
    return loaded


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if any(
            (candidate / marker).exists()
            for marker in ("pyproject.toml", "setup.cfg", ".git")
        ):
            return candidate
    return start.resolve()


def _ensure_import_paths(
    project_root: Path, extra_paths: Optional[Sequence[str]] = None
) -> None:
    candidate_paths = [project_root, project_root / "src"]
    for extra_path in extra_paths or []:
        candidate = Path(extra_path).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        candidate_paths.append(candidate)

    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _substitute_template(template: str, resolver: Callable[[str], Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = resolver(key)
        if value is None:
            raise BuilderError(
                f"Reference '{key}' resolved to None inside template '{template}'"
            )
        return str(value)

    return _REF_PATTERN.sub(replace, template)


def _eval_expression(expression: str) -> Any:
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise BuilderError(f"Invalid expression '{expression}': {exc}") from exc

    def evaluate(current: ast.AST) -> Any:
        if isinstance(current, ast.Expression):
            return evaluate(current.body)
        if isinstance(current, ast.Constant):
            if isinstance(current.value, (int, float, bool)):
                return current.value
            raise BuilderError(
                f"Unsupported constant in expression '{expression}': {current.value!r}"
            )
        if isinstance(current, ast.UnaryOp):
            operand = evaluate(current.operand)
            if isinstance(current.op, ast.UAdd):
                return +operand
            if isinstance(current.op, ast.USub):
                return -operand
            raise BuilderError(
                f"Unsupported unary operator in expression '{expression}'"
            )
        if isinstance(current, ast.BinOp):
            left = evaluate(current.left)
            right = evaluate(current.right)
            operations = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a**b,
            }
            for operation_type, handler in operations.items():
                if isinstance(current.op, operation_type):
                    return handler(left, right)
            raise BuilderError(
                f"Unsupported binary operator in expression '{expression}'"
            )
        raise BuilderError(
            f"Unsupported expression node {type(current).__name__} in '{expression}'"
        )

    return evaluate(node)


def _resolve_mapping(
    mapping: Mapping[str, Any],
    base_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = dict(base_context or {})
    resolving: set[str] = set()

    def resolve_name(name: str) -> Any:
        if name in resolved:
            return resolved[name]
        if name not in mapping:
            raise BuilderError(f"Unknown reference '{name}'")
        if name in resolving:
            raise BuilderError(f"Cyclic reference detected while resolving '{name}'")
        resolving.add(name)
        try:
            value = _resolve_value(mapping[name], resolve_name)
            resolved[name] = value
            return value
        finally:
            resolving.remove(name)

    for key in mapping:
        resolve_name(key)

    return {key: resolved[key] for key in mapping}


def _resolve_value(value: Any, resolver: Callable[[str], Any]) -> Any:
    if isinstance(value, dict):
        if "$ref" in value and len(value) == 1:
            return resolver(str(value["$ref"]))
        if "$sub" in value and len(value) == 1:
            template = value["$sub"]
            if not isinstance(template, str):
                raise BuilderError(
                    f"$sub expects a string template, got {type(template).__name__}"
                )
            return _substitute_template(template, resolver)
        if "$expr" in value and len(value) == 1:
            expression_source = value["$expr"]
            if isinstance(expression_source, dict) and "$sub" in expression_source:
                expression = _resolve_value(expression_source, resolver)
            elif isinstance(expression_source, str):
                expression = expression_source
            else:
                expression = str(_resolve_value(expression_source, resolver))
            return _eval_expression(expression)
        return {key: _resolve_value(item, resolver) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, resolver) for item in value]
    return value


def resolve_globals(globals_mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not globals_mapping:
        return {}
    return _resolve_mapping(globals_mapping)


def _resolve_inline_mapping(
    mapping: Optional[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    if not mapping:
        return {}
    local = dict(context)

    def resolver(name: str) -> Any:
        if name in local:
            return local[name]
        raise BuilderError(f"Unknown reference '{name}'")

    resolved: Dict[str, Any] = {}
    for key, raw_value in mapping.items():
        value = _resolve_value(raw_value, resolver)
        resolved[key] = value
        local[key] = value
    return resolved


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise BuilderError(f"Cannot coerce {value!r} to bool")


def _coerce_parameter_value(
    parameter_name: str, definition: Mapping[str, Any], value: Any
) -> Any:
    param_type = str(definition.get("Type", "String")).lower()

    try:
        if param_type in {"float", "number"}:
            return float(value)
        if param_type in {"integer", "int"}:
            return int(value)
        if param_type in {"bool", "boolean"}:
            return _normalize_bool(value)
        if param_type == "string":
            return str(value)
    except (TypeError, ValueError) as exc:
        raise BuilderError(
            f"Could not coerce parameter '{parameter_name}' with value {value!r} to {definition.get('Type')}"
        ) from exc

    return value


def _resolve_public_parameters(
    parameter_definitions: Mapping[str, Any],
    parameter_overrides: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved_overrides = _resolve_inline_mapping(parameter_overrides, context)
    public_parameters: Dict[str, Any] = {}
    for parameter_name, definition in parameter_definitions.items():
        if not isinstance(definition, Mapping):
            definition = {}
        if parameter_name in resolved_overrides:
            raw_value = resolved_overrides[parameter_name]
        elif "Default" in definition:
            raw_value = definition["Default"]
        else:
            raise BuilderError(f"Missing required parameter '{parameter_name}'")
        public_parameters[parameter_name] = _coerce_parameter_value(
            parameter_name,
            definition,
            raw_value,
        )
    extra_parameters = sorted(set(resolved_overrides) - set(parameter_definitions))
    if extra_parameters:
        raise BuilderError(
            f"Unknown parameter override(s): {', '.join(extra_parameters)}"
        )
    return public_parameters


def _get_assemblies(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    assemblies = config.get("assemblies")
    if assemblies is None:
        raise BuilderError("Expected top-level 'assemblies' list")
    if not isinstance(assemblies, list):
        raise BuilderError("Assemblies definition must be a list")
    return [dict(item) for item in assemblies]


def _dependency_names(entry: Mapping[str, Any]) -> List[str]:
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


def _resolve_build_generations(
    assemblies: Sequence[Mapping[str, Any]],
    selected_names: Optional[Iterable[str]] = None,
) -> List[List[Dict[str, Any]]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in assemblies:
        name = entry.get("name")
        if not name:
            raise BuilderError("Each assembly entry must define a name")
        if name in by_name:
            raise BuilderError(f"Duplicate assembly name '{name}'")
        by_name[str(name)] = dict(entry)

    requested = list(selected_names or by_name.keys())
    missing = sorted(name for name in requested if name not in by_name)
    if missing:
        raise BuilderError(f"Unknown assembly name(s): {', '.join(missing)}")

    required_names: set[str] = set()

    def collect(name: str) -> None:
        if name in required_names:
            return
        required_names.add(name)
        entry = by_name[name]
        for dependency in _dependency_names(entry):
            if dependency not in by_name:
                raise BuilderError(
                    f"Assembly '{name}' depends on unknown assembly '{dependency}'"
                )
            collect(dependency)

    for name in requested:
        collect(name)

    dependency_graph = nx.DiGraph()
    for name in sorted(required_names):
        dependency_graph.add_node(name)

    for name in sorted(required_names):
        for dependency in _dependency_names(by_name[name]):
            dependency_graph.add_edge(dependency, name)

    try:
        generations = list(nx.topological_generations(dependency_graph))
    except nx.NetworkXUnfeasible as exc:
        raise BuilderError("Cyclic dependency detected in assembly graph") from exc

    resolved_generations: List[List[Dict[str, Any]]] = []
    for generation_index, generation_names in enumerate(generations):
        entries_in_generation: List[Dict[str, Any]] = []
        for assembly_name in sorted(generation_names):
            entry = by_name[assembly_name]
            if entry.get("disabled"):
                _logger.info("Skipping disabled assembly '%s'", assembly_name)
                continue
            entries_in_generation.append(entry)
        if entries_in_generation:
            resolved_generations.append(entries_in_generation)
            _logger.info(
                "Build generation %s: %s",
                generation_index,
                ", ".join(entry["name"] for entry in entries_in_generation),
            )

    return resolved_generations


def _assemblies_by_name(
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


def _expand_with_dependents(
    assemblies: Sequence[Mapping[str, Any]],
    selected_names: Sequence[str],
) -> List[str]:
    by_name = _assemblies_by_name(assemblies)
    missing = sorted(name for name in selected_names if name not in by_name)
    if missing:
        raise BuilderError(f"Unknown assembly name(s): {', '.join(missing)}")

    reverse_dependencies: Dict[str, set[str]] = {name: set() for name in by_name}
    for name, entry in by_name.items():
        for dependency in _dependency_names(entry):
            if dependency not in by_name:
                raise BuilderError(
                    f"Assembly '{name}' depends on unknown assembly '{dependency}'"
                )
            reverse_dependencies[dependency].add(name)

    expanded: set[str] = set()
    pending = list(selected_names)
    while pending:
        current = pending.pop()
        if current in expanded:
            continue
        expanded.add(current)
        pending.extend(sorted(reverse_dependencies[current]))

    return sorted(expanded)


def _default_repository_dir(config_path: Path) -> Path:
    base_dir = config_path.parent
    if base_dir.name == "assemblies":
        return base_dir.parent / "repository"
    return base_dir / "repository"


def _resolve_repository_dir(
    config_path: Path,
    config_data: Mapping[str, Any],
    override: Optional[str],
) -> Path:
    raw_value = override or config_data.get("repository_dir")
    if raw_value:
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        return candidate
    return _default_repository_dir(config_path).resolve()


def _discover_resource_file(config_path: Path, entry: Mapping[str, Any]) -> Path:
    config_dir = config_path.parent
    explicit_keys = (
        "assembly_file",
        "assembly_resource",
        "resource_file",
        "template_file",
    )
    for key in explicit_keys:
        raw_value = entry.get(key)
        if not raw_value:
            continue
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (config_dir / candidate).resolve()
        if candidate.exists():
            return candidate
        raise BuilderError(
            f"Assembly resource file for '{entry['name']}' not found: {candidate}"
        )

    target_name = str(entry["name"])
    search_dirs = [config_dir]
    for directory in search_dirs:
        for extension in (".yaml", ".yml"):
            direct = directory / f"{target_name}{extension}"
            if direct.exists():
                return direct.resolve()

    matches = sorted(
        candidate.resolve()
        for candidate in config_dir.glob("*.y*ml")
        if candidate.stem == target_name
    )
    if matches:
        return matches[0]

    raise BuilderError(
        f"Could not find assembly resource file for '{target_name}' in {config_dir}"
    )


def _select_part_definition(
    resource_path: Path,
    resource_data: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    parts = resource_data.get("Parts")
    if not isinstance(parts, Mapping) or not parts:
        raise BuilderError(
            f"Assembly resource {resource_path} does not define any Parts"
        )

    explicit_name = (
        entry.get("logical_part_name")
        or entry.get("part_name")
        or entry.get("resource_name")
    )
    if explicit_name:
        if explicit_name not in parts:
            raise BuilderError(
                f"Assembly resource {resource_path} has no part named '{explicit_name}'"
            )
        return str(explicit_name), dict(parts[explicit_name])

    if len(parts) == 1:
        logical_name, definition = next(iter(parts.items()))
        return str(logical_name), dict(definition)

    matching_keys = [
        key
        for key in parts
        if _safe_name(str(key)).lower() == _safe_name(str(entry["name"])).lower()
    ]
    if len(matching_keys) == 1:
        key = matching_keys[0]
        return str(key), dict(parts[key])

    raise BuilderError(
        f"Assembly resource {resource_path} defines multiple parts; specify logical_part_name for '{entry['name']}'"
    )


def _resolve_assembly(
    config_path: Path,
    entry: Mapping[str, Any],
    global_context: Mapping[str, Any],
) -> _ResolvedAssembly:
    resource_path = _discover_resource_file(config_path, entry)
    resource_data = _load_yaml(resource_path)
    logical_part_name, part_definition = _select_part_definition(
        resource_path, resource_data, entry
    )

    parameter_definitions = resource_data.get("Parameters", {})
    if not isinstance(parameter_definitions, Mapping):
        raise BuilderError(f"Parameters section in {resource_path} must be a mapping")

    parameter_overrides = entry.get("parameters", {})
    if not isinstance(parameter_overrides, Mapping):
        raise BuilderError(
            f"Parameters for assembly '{entry['name']}' must be a mapping"
        )

    public_parameters = _resolve_public_parameters(
        parameter_definitions,
        parameter_overrides,
        global_context,
    )

    properties = part_definition.get("Properties", {})
    if not isinstance(properties, Mapping):
        raise BuilderError(
            f"Part '{logical_part_name}' in {resource_path} has invalid Properties"
        )

    generator_path = properties.get("Generator")
    if not generator_path:
        raise BuilderError(
            f"Part '{logical_part_name}' in {resource_path} is missing Properties.Generator"
        )

    generator_properties = properties.get("Properties", public_parameters)
    if not isinstance(generator_properties, Mapping):
        raise BuilderError(
            f"Part '{logical_part_name}' in {resource_path} has invalid Properties.Properties section"
        )

    generator_kwargs = _resolve_inline_mapping(generator_properties, public_parameters)
    parameter_hash = PartParameters(generator_kwargs).parameters_hash()

    return _ResolvedAssembly(
        name=str(entry["name"]),
        entry=dict(entry),
        resource_path=resource_path,
        resource_data=resource_data,
        public_parameters=public_parameters,
        generator_kwargs=generator_kwargs,
        generator_path=str(generator_path),
        logical_part_name=logical_part_name,
        parameter_hash=parameter_hash,
    )


def _load_generator_callable(generator_path: str) -> Callable[..., Any]:
    if "." not in generator_path:
        raise BuilderError(
            f"Generator path must be module.function, got '{generator_path}'"
        )

    module_name, function_name = generator_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BuilderError(
            f"Could not import generator module '{module_name}': {exc}"
        ) from exc

    try:
        generator = getattr(module, function_name)
    except AttributeError as exc:
        raise BuilderError(
            f"Generator function '{function_name}' not found in module '{module_name}'"
        ) from exc

    if not callable(generator):
        raise BuilderError(f"Resolved generator '{generator_path}' is not callable")
    return generator


def _group_named_parts(
    parts: Sequence[Any], name_map: Mapping[str, int]
) -> List[Dict[str, Any]]:
    names_by_index = {index: name for name, index in name_map.items()}
    grouped = []
    for index, part in enumerate(parts):
        grouped.append(
            {
                "index": index,
                "name": names_by_index.get(index),
                "part": part,
            }
        )
    return grouped


def _normalize_generated_part(generated_part: Any) -> Dict[str, Any]:
    if all(
        hasattr(generated_part, attribute)
        for attribute in ("leader", "followers", "cutters", "non_production_parts")
    ):
        return {
            "leader": generated_part.leader,
            "followers": _group_named_parts(
                list(getattr(generated_part, "followers", [])),
                getattr(generated_part, "follower_indices_by_name", {}),
            ),
            "cutters": _group_named_parts(
                list(getattr(generated_part, "cutters", [])),
                getattr(generated_part, "cutter_indices_by_name", {}),
            ),
            "non_production_parts": _group_named_parts(
                list(getattr(generated_part, "non_production_parts", [])),
                getattr(generated_part, "non_production_indices_by_name", {}),
            ),
            "fused": (
                generated_part.leaders_followers_fused()
                if hasattr(generated_part, "leaders_followers_fused")
                else None
            ),
        }

    return {
        "leader": generated_part,
        "followers": [],
        "cutters": [],
        "non_production_parts": [],
        "fused": None,
    }


def _export_part_to_step(part: Any, destination: Path) -> None:
    from shellforgepy.produce.arrange_and_export import export_solid_to_step

    destination.parent.mkdir(parents=True, exist_ok=True)
    export_solid_to_step(part, destination)


def _artifact_filename(
    assembly_name: str,
    parameter_hash: str,
    label: str,
    index: Optional[int] = None,
    name: Optional[str] = None,
) -> str:
    parts = [assembly_name, parameter_hash, label]
    if index is not None:
        parts.append(f"{index:04d}")
    if name:
        parts.append(_safe_name(name))
    return "__".join(parts) + ".step"


def _write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_path(artifact_dir: Path, assembly_name: str, parameter_hash: str) -> Path:
    return artifact_dir / f"{assembly_name}__{parameter_hash}__metadata.json"


def _import_dependency_part(step_path: Path) -> Any:
    from shellforgepy.simple import import_solid_from_step

    return import_solid_from_step(str(step_path))


def _resolve_injected_parts_config(
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
    dependencies = _dependency_names(entry)
    for kwarg_name, raw_spec in config.items():
        if isinstance(raw_spec, str):
            assembly_name = raw_spec
            artifact = "leader"
        elif isinstance(raw_spec, Mapping):
            assembly_name = raw_spec.get("assembly")
            if assembly_name is None:
                if len(dependencies) == 1:
                    assembly_name = dependencies[0]
                else:
                    raise BuilderError(
                        f"inject_parts.{kwarg_name} for assembly '{entry.get('name')}' must specify an assembly"
                    )
            artifact = raw_spec.get("artifact", "leader")
        else:
            raise BuilderError(
                f"inject_parts.{kwarg_name} for assembly '{entry.get('name')}' must be a string or mapping"
            )
        normalized[str(kwarg_name)] = {
            "assembly": str(assembly_name),
            "artifact": str(artifact),
        }
    return normalized


def _find_latest_metadata_for_assembly(
    repository_dir: Path, assembly_name: str
) -> Dict[str, Any]:
    assembly_dir = repository_dir / assembly_name
    if not assembly_dir.exists():
        raise BuilderError(
            f"Dependency assembly '{assembly_name}' has no repository directory at {assembly_dir}"
        )

    metadata_files = sorted(
        assembly_dir.glob(f"{assembly_name}__*/{assembly_name}__*__metadata.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not metadata_files:
        raise BuilderError(
            f"Dependency assembly '{assembly_name}' has no cached metadata in {assembly_dir}"
        )
    return _read_metadata(metadata_files[0])


def _dependency_metadata_for_assembly(
    assembly_name: str,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
) -> Dict[str, Any]:
    metadata = built_results_by_name.get(assembly_name)
    if metadata is not None:
        return metadata
    return _find_latest_metadata_for_assembly(repository_dir, assembly_name)


def _dependency_step_path(metadata: Mapping[str, Any], artifact: str) -> Path:
    artifacts = metadata.get("artifacts", {})
    if artifact == "leader":
        step_path = artifacts.get("leader_step")
    elif artifact == "fused":
        step_path = artifacts.get("fused_step")
    elif "." in artifact:
        artifact_group, target_name = artifact.split(".", 1)
        if artifact_group not in {"followers", "cutters", "non_production_parts"}:
            raise BuilderError(f"Unsupported dependency artifact selector '{artifact}'")
        step_path = None
        for item in artifacts.get(artifact_group, []):
            if item.get("name") == target_name:
                step_path = item.get("path")
                break
    else:
        raise BuilderError(f"Unsupported dependency artifact selector '{artifact}'")

    if not step_path:
        raise BuilderError(
            f"Could not resolve dependency artifact '{artifact}' for assembly '{metadata.get('assembly_name')}'"
        )
    return Path(step_path).expanduser().resolve()


def _resolve_dependency_injections(
    entry: Mapping[str, Any],
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
) -> List[_DependencyInjection]:
    injections: List[_DependencyInjection] = []
    for kwarg_name, spec in _resolve_injected_parts_config(entry).items():
        assembly_name = spec["assembly"]
        metadata = _dependency_metadata_for_assembly(
            assembly_name,
            built_results_by_name,
            repository_dir,
        )
        step_path = _dependency_step_path(metadata, spec["artifact"])
        injections.append(
            _DependencyInjection(
                kwarg_name=kwarg_name,
                assembly_name=assembly_name,
                artifact=spec["artifact"],
                source_parameter_hash=str(metadata["parameter_hash"]),
                step_path=step_path,
                part=_import_dependency_part(step_path),
            )
        )
    return injections


def _hash_with_dependencies(
    generator_kwargs: Mapping[str, Any],
    injections: Sequence[_DependencyInjection],
) -> str:
    hash_inputs = dict(generator_kwargs)
    for injection in injections:
        hash_inputs[f"__dependency__{injection.kwarg_name}"] = (
            f"{injection.assembly_name}:{injection.artifact}:{injection.source_parameter_hash}"
        )
    return PartParameters(hash_inputs).parameters_hash()


def _export_artifacts(
    assembly_name: str,
    parameter_hash: str,
    generated_part: Any,
    artifact_dir: Path,
) -> Dict[str, Any]:
    normalized = _normalize_generated_part(generated_part)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Any] = {
        "leader_step": None,
        "fused_step": None,
        "followers": [],
        "cutters": [],
        "non_production_parts": [],
    }

    leader_path = artifact_dir / _artifact_filename(
        assembly_name,
        parameter_hash,
        "leader",
    )
    _export_part_to_step(normalized["leader"], leader_path)
    artifacts["leader_step"] = str(leader_path)

    fused = normalized.get("fused")
    if fused is not None:
        fused_path = artifact_dir / _artifact_filename(
            assembly_name,
            parameter_hash,
            "fused",
        )
        _export_part_to_step(fused, fused_path)
        artifacts["fused_step"] = str(fused_path)

    for group_key, label in (
        ("followers", "follower"),
        ("cutters", "cutter"),
        ("non_production_parts", "non_production"),
    ):
        for item in normalized[group_key]:
            component_path = artifact_dir / _artifact_filename(
                assembly_name,
                parameter_hash,
                label,
                index=item["index"],
                name=item["name"],
            )
            _export_part_to_step(item["part"], component_path)
            artifacts[group_key].append(
                {
                    "index": item["index"],
                    "name": item["name"],
                    "path": str(component_path),
                }
            )

    return artifacts


def _load_attribute(dotted_path: str) -> Any:
    if "." not in dotted_path:
        raise BuilderError(f"Expected dotted import path, got '{dotted_path}'")
    module_name, attribute_name = dotted_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BuilderError(f"Could not import module '{module_name}': {exc}") from exc
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise BuilderError(
            f"Attribute '{attribute_name}' not found in module '{module_name}'"
        ) from exc


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


def _builder_section(
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


def _default_scene_rules(
    metadata: Mapping[str, Any], mode: str
) -> List[Dict[str, Any]]:
    artifacts = metadata.get("artifacts", {})
    followers = artifacts.get("followers", [])
    rules: List[Dict[str, Any]] = []
    if followers:
        rules.append({"source": "self", "artifact": "followers"})
    else:
        rules.append({"source": "self", "artifact": "leader"})
    if mode == "visualization" and artifacts.get("non_production_parts"):
        rules.append({"source": "self", "artifact": "non_production_parts"})
    return rules


def _scene_rules_for_mode(
    metadata: Mapping[str, Any],
    resource_data: Mapping[str, Any],
    mode: str,
    config_data: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    section_name = "Visualization" if mode == "visualization" else "Production"
    section = _builder_section(resource_data, section_name, config_data)
    parts = section.get("parts")
    if parts is None:
        return _default_scene_rules(metadata, mode)
    if not isinstance(parts, list):
        raise BuilderError(f"Builder.{section_name}.parts must be a list")
    return [dict(item) for item in parts]


def _artifact_entries_for_selector(
    metadata: Mapping[str, Any],
    artifact: str,
) -> List[Dict[str, Any]]:
    artifacts = metadata.get("artifacts", {})
    assembly_name = str(metadata.get("assembly_name"))

    if artifact == "leader":
        path = artifacts.get("leader_step")
        return (
            [
                {
                    "assembly_name": assembly_name,
                    "artifact": artifact,
                    "path": str(path),
                    "name": None,
                    "index": None,
                }
            ]
            if path
            else []
        )
    if artifact == "fused":
        path = artifacts.get("fused_step")
        return (
            [
                {
                    "assembly_name": assembly_name,
                    "artifact": artifact,
                    "path": str(path),
                    "name": None,
                    "index": None,
                }
            ]
            if path
            else []
        )
    if artifact in {"followers", "cutters", "non_production_parts"}:
        return [
            {
                "assembly_name": assembly_name,
                "artifact": artifact,
                "path": str(item["path"]),
                "name": item.get("name"),
                "index": item.get("index"),
            }
            for item in artifacts.get(artifact, [])
        ]
    raise BuilderError(f"Unsupported scene artifact selector '{artifact}'")


def _dependency_metadata_map(
    metadata: Mapping[str, Any],
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    resolved: Dict[str, Dict[str, Any]] = {}
    for dependency in metadata.get("dependencies", []):
        assembly_name = str(dependency["assembly_name"])
        if assembly_name in resolved:
            continue
        resolved[assembly_name] = _dependency_metadata_for_assembly(
            assembly_name,
            built_results_by_name,
            repository_dir,
        )
    return resolved


def _format_scene_name(
    item: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> str:
    template = rule.get("name_template")
    metadata_name = item.get("name")
    assembly_name = str(item["assembly_name"])
    artifact = str(item["artifact"])
    default_name = (
        str(metadata_name)
        if metadata_name
        else (assembly_name if artifact == "leader" else f"{assembly_name}_{artifact}")
    )
    if template:
        return str(
            template.format(
                assembly_name=assembly_name,
                artifact=artifact,
                name=metadata_name or "",
                index="" if item.get("index") is None else item["index"],
                default_name=default_name,
            )
        )
    return str(rule.get("name") or default_name)


def _apply_scene_transform(
    part: Any,
    transform: Optional[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Any:
    if not transform:
        return part
    if not isinstance(transform, Mapping):
        raise BuilderError("Scene transform must be a mapping")
    function_path = transform.get("function")
    if not function_path:
        raise BuilderError("Scene transform requires a function")
    function_name = str(function_path)
    if "." not in function_name:
        function_name = f"shellforgepy.simple.{function_name}"
    function = _load_attribute(function_name)
    kwargs = transform.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise BuilderError("Scene transform kwargs must be a mapping")
    resolved_kwargs = _resolve_inline_mapping(kwargs, context)
    return function(part, **resolved_kwargs)


def _materialize_rule_parts(
    metadata: Mapping[str, Any],
    resource_data: Mapping[str, Any],
    mode: str,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    config_data: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rules = _scene_rules_for_mode(metadata, resource_data, mode, config_data)
    dependency_metadata = _dependency_metadata_map(
        metadata, built_results_by_name, repository_dir
    )
    parameter_context = dict(metadata.get("public_parameters", {}))
    parameter_context.update(metadata.get("generator_kwargs", {}))

    scene_parts: List[Dict[str, Any]] = []
    for rule in rules:
        source = str(rule.get("source", "self"))
        artifact = str(rule.get("artifact", "leader"))

        if source in {"self", "assembly"}:
            targets = [metadata]
        elif source in {"dependencies", "dependency"}:
            dependency_name = rule.get("assembly")
            if dependency_name:
                dependency_key = str(dependency_name)
                if dependency_key not in dependency_metadata:
                    raise BuilderError(
                        f"Assembly '{metadata.get('assembly_name')}' has no dependency '{dependency_key}'"
                    )
                targets = [dependency_metadata[dependency_key]]
            else:
                targets = [
                    dependency_metadata[name] for name in sorted(dependency_metadata)
                ]
        else:
            raise BuilderError(f"Unsupported scene source '{source}'")

        for target in targets:
            entries = _artifact_entries_for_selector(target, artifact)
            for entry in entries:
                part = _import_dependency_part(
                    Path(entry["path"]).expanduser().resolve()
                )
                part_context = dict(parameter_context)
                part_context.update(
                    {
                        "assembly_name": entry["assembly_name"],
                        "artifact": entry["artifact"],
                        "part_name": entry.get("name"),
                        "part_index": entry.get("index"),
                    }
                )
                part = _apply_scene_transform(part, rule.get("transform"), part_context)
                scene_parts.append(
                    {
                        "name": _format_scene_name(entry, rule),
                        "part": part,
                        "source_path": entry["path"],
                        "flip": bool(rule.get("flip", False)),
                        "skip_in_production": bool(
                            rule.get("skip_in_production", False)
                        ),
                        "prod_rotation_angle": rule.get("prod_rotation_angle"),
                        "prod_rotation_axis": rule.get("prod_rotation_axis"),
                        "color": rule.get("color"),
                        "animation": rule.get("animation"),
                    }
                )
    return scene_parts


def _resolve_process_data(
    metadata: Mapping[str, Any],
    resource_data: Mapping[str, Any],
    config_data: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    production_section = _builder_section(resource_data, "Production", config_data)
    process_data_spec = production_section.get("process_data")
    if process_data_spec is None:
        return None
    if not isinstance(process_data_spec, Mapping):
        raise BuilderError("Builder.Production.process_data must be a mapping")
    source = process_data_spec.get("source")
    if not source:
        raise BuilderError("Builder.Production.process_data.source is required")

    loaded = _load_attribute(str(source))
    if not isinstance(loaded, Mapping):
        raise BuilderError(
            f"Resolved process_data source '{source}' must be a mapping, got {type(loaded).__name__}"
        )
    resolved = deepcopy(dict(loaded))
    overrides = process_data_spec.get("overrides", {})
    if overrides:
        if not isinstance(overrides, Mapping):
            raise BuilderError(
                "Builder.Production.process_data.overrides must be a mapping"
            )
        context = dict(metadata.get("public_parameters", {}))
        context.update(metadata.get("generator_kwargs", {}))
        resolved_overrides = _resolve_inline_mapping(overrides, context)
        for key, value in resolved_overrides.items():
            if isinstance(value, Mapping) and isinstance(resolved.get(key), Mapping):
                merged = dict(resolved[key])
                merged.update(value)
                resolved[key] = merged
            else:
                resolved[key] = value
    return resolved


def _resolve_export_options(
    resource_data: Mapping[str, Any],
    mode: str,
    config_data: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    section_name = "Visualization" if mode == "visualization" else "Production"
    section = _builder_section(resource_data, section_name, config_data)
    arrange = section.get("arrange", {})
    if arrange is None:
        arrange = {}
    if not isinstance(arrange, Mapping):
        raise BuilderError(f"Builder.{section_name}.arrange must be a mapping")

    if mode == "visualization":
        defaults = {
            "prod_gap": 1.0,
            "bed_width": 200.0,
            "max_build_height": None,
            "export_step": True,
            "export_obj": True,
            "export_individual_parts": False,
            "export_stl": False,
        }
    else:
        defaults = {
            "prod_gap": 1.0,
            "bed_width": 200.0,
            "max_build_height": None,
            "export_step": True,
            "export_obj": True,
            "export_individual_parts": True,
            "export_stl": True,
        }

    resolved = dict(defaults)
    resolved.update(dict(arrange))
    return resolved


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in values.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_from_file(
    config_path: str | os.PathLike[str],
    *,
    assembly_names: Optional[Sequence[str]] = None,
    repository_dir: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise BuilderError(f"Assembly config file does not exist: {config_file}")

    config_data = _load_yaml(config_file)
    project_root = _find_project_root(config_file.parent)
    extra_paths = config_data.get("python_paths")
    if extra_paths is not None and not isinstance(extra_paths, list):
        raise BuilderError("python_paths must be a list when provided")
    _ensure_import_paths(project_root, extra_paths)

    global_context = resolve_globals(config_data.get("globals", {}))
    repository_path = _resolve_repository_dir(config_file, config_data, repository_dir)
    repository_path.mkdir(parents=True, exist_ok=True)

    assemblies = _get_assemblies(config_data)
    build_generations = _resolve_build_generations(assemblies, assembly_names)

    results: List[Dict[str, Any]] = []
    built_results_by_name: Dict[str, Dict[str, Any]] = {}
    for generation_index, generation_entries in enumerate(build_generations):
        for assembly_index, entry in enumerate(generation_entries):
            resolved = _resolve_assembly(config_file, entry, global_context)
            dependency_injections = _resolve_dependency_injections(
                entry,
                built_results_by_name,
                repository_path,
            )
            parameter_hash = _hash_with_dependencies(
                resolved.generator_kwargs,
                dependency_injections,
            )
            artifact_dir = (
                repository_path / resolved.name / f"{resolved.name}__{parameter_hash}"
            )
            metadata_path = _metadata_path(artifact_dir, resolved.name, parameter_hash)

            if metadata_path.exists() and not force:
                cached_metadata = _read_metadata(metadata_path)
                cached_metadata["cache_hit"] = True
                results.append(cached_metadata)
                built_results_by_name[resolved.name] = cached_metadata
                _logger.info(
                    "Using cached build for '%s' from %s",
                    resolved.name,
                    metadata_path,
                )
                continue

            generator = _load_generator_callable(resolved.generator_path)
            _logger.info(
                "Building assembly '%s' with %s",
                resolved.name,
                resolved.generator_path,
            )
            generator_kwargs = deepcopy(resolved.generator_kwargs)
            for injection in dependency_injections:
                generator_kwargs[injection.kwarg_name] = injection.part
            generated_part = generator(**generator_kwargs)
            artifacts = _export_artifacts(
                resolved.name,
                parameter_hash,
                generated_part,
                artifact_dir,
            )

            metadata = {
                "schema_version": 1,
                "cache_hit": False,
                "built_at": _utc_now_iso(),
                "project_root": str(project_root),
                "repository_dir": str(repository_path),
                "artifact_dir": str(artifact_dir),
                "assembly_name": resolved.name,
                "logical_part_name": resolved.logical_part_name,
                "resource_file": str(resolved.resource_path),
                "generator": resolved.generator_path,
                "parameter_hash": parameter_hash,
                "public_parameters": resolved.public_parameters,
                "generator_kwargs": resolved.generator_kwargs,
                "generation": generation_index,
                "assembly_in_generation": assembly_index,
                "dependencies": [
                    {
                        "kwarg_name": injection.kwarg_name,
                        "assembly_name": injection.assembly_name,
                        "artifact": injection.artifact,
                        "source_parameter_hash": injection.source_parameter_hash,
                        "step_path": str(injection.step_path),
                    }
                    for injection in dependency_injections
                ],
                "artifacts": artifacts,
            }
            _write_metadata(metadata_path, metadata)
            results.append(metadata)
            built_results_by_name[resolved.name] = metadata

    return results


def _resolve_runs_base_dir(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    from shellforgepy.workflow.workflow import (
        DEFAULT_RUNS_DIR_NAME,
        _resolve_config_key_value,
    )

    runs_base = Path(
        getattr(args, "runs_dir", None)
        or _resolve_config_key_value(dict(config), "runs_dir")
        or (Path.cwd() / DEFAULT_RUNS_DIR_NAME)
    ).expanduser()
    if not runs_base.is_absolute():
        runs_base = Path.cwd() / runs_base
    runs_base = runs_base.resolve()
    runs_base.mkdir(parents=True, exist_ok=True)
    return runs_base


def _export_scene_for_assembly(
    *,
    args: argparse.Namespace,
    config_data: Mapping[str, Any],
    build_results: Sequence[Dict[str, Any]],
    selected_assembly: str,
    scene_assembly_names: Sequence[str],
) -> int:
    from shellforgepy.produce.arrange_and_export import arrange_and_export_parts
    from shellforgepy.workflow.workflow import (
        EXPORT_DIR_ENV,
        MANIFEST_ENV,
        MANIFEST_FILENAME,
        RUN_DIR_ENV,
        RUN_ID_ENV,
        _resolve_config_key_value,
        complete_workflow_run,
        get_config_path,
        load_config,
    )

    built_results_by_name = {
        result["assembly_name"]: dict(result) for result in build_results
    }
    if selected_assembly not in built_results_by_name:
        raise BuilderError(f"Selected assembly '{selected_assembly}' was not built")

    config_file_for_workflow = get_config_path(getattr(args, "config", None))
    workflow_config = load_config(config_file_for_workflow)
    run_id = getattr(args, "run_id", None) or datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S"
    )
    runs_base = _resolve_runs_base_dir(args, workflow_config)
    run_directory = (runs_base / f"{selected_assembly}_run_{run_id}").resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()

    viewer_base_url = _resolve_config_key_value(workflow_config, "viewer_base_url")
    repository_dir = Path(
        built_results_by_name[selected_assembly]["repository_dir"]
    ).resolve()

    production_mode = bool(
        getattr(args, "production", False)
        or getattr(args, "slice", False)
        or getattr(args, "upload", False)
    )
    mode = "production" if production_mode else "visualization"
    selected_metadata = built_results_by_name[selected_assembly]

    scene_parts: List[Dict[str, Any]] = []
    seen_source_paths: set[str] = set()
    for assembly_name in scene_assembly_names:
        metadata = built_results_by_name.get(assembly_name)
        if metadata is None:
            metadata = _dependency_metadata_for_assembly(
                assembly_name,
                built_results_by_name,
                repository_dir,
            )
        resource_data = _load_yaml(
            Path(metadata["resource_file"]).expanduser().resolve()
        )
        for part in _materialize_rule_parts(
            metadata,
            resource_data,
            mode,
            built_results_by_name,
            repository_dir,
            config_data,
        ):
            source_path = str(part.get("source_path") or "")
            if source_path and source_path in seen_source_paths:
                continue
            if source_path:
                seen_source_paths.add(source_path)
            part.pop("source_path", None)
            scene_parts.append(part)

    if not scene_parts:
        raise BuilderError(
            f"No scene parts resolved for assembly '{selected_assembly}' in {mode} mode"
        )

    process_data = None
    if production_mode:
        selected_resource_data = _load_yaml(
            Path(selected_metadata["resource_file"]).expanduser().resolve()
        )
        process_data = _resolve_process_data(
            selected_metadata,
            selected_resource_data,
            config_data,
        )
        if process_data is None:
            raise BuilderError(
                f"Assembly '{selected_assembly}' does not declare Builder.Production.process_data"
            )

    export_options = _resolve_export_options(
        _load_yaml(Path(selected_metadata["resource_file"]).expanduser().resolve()),
        mode,
        config_data,
    )
    env = {
        RUN_ID_ENV: run_id,
        RUN_DIR_ENV: str(run_directory),
        EXPORT_DIR_ENV: str(run_directory),
        MANIFEST_ENV: str(manifest_path),
    }
    if viewer_base_url:
        env["SHELLFORGEPY_VIEWER_BASE_URL"] = str(viewer_base_url)
    if production_mode:
        env["SHELLFORGEPY_PRODUCTION"] = "1"

    with _temporary_environment(env):
        arrange_and_export_parts(
            scene_parts,
            prod_gap=float(export_options["prod_gap"]),
            bed_width=float(export_options["bed_width"]),
            script_file=str(Path(selected_metadata["resource_file"]).resolve()),
            export_directory=run_directory,
            prod=production_mode,
            process_data=process_data,
            max_build_height=export_options.get("max_build_height"),
            verbose=bool(getattr(args, "verbose", False)),
            export_step=bool(export_options.get("export_step", True)),
            export_obj=bool(export_options.get("export_obj", True)),
            export_individual_parts=bool(
                export_options.get("export_individual_parts", True)
            ),
            export_stl=bool(export_options.get("export_stl", True)),
        )

    if not manifest_path.exists():
        raise BuilderError(f"Expected workflow manifest at {manifest_path}")
    manifest = _read_metadata(manifest_path)
    return complete_workflow_run(
        args,
        config=workflow_config,
        run_directory=run_directory,
        manifest=manifest,
        target_label=selected_assembly,
    )


def run_builder(args: argparse.Namespace) -> int:
    config_file = Path(args.config_file).expanduser().resolve()
    config_data = _load_yaml(config_file)
    assemblies = _get_assemblies(config_data)

    requested_assemblies = list(args.assembly or [])
    build_assemblies: Optional[List[str]] = requested_assemblies or None
    if requested_assemblies and getattr(args, "with_dependents", False):
        build_assemblies = _expand_with_dependents(assemblies, requested_assemblies)

    results = build_from_file(
        config_file,
        assembly_names=build_assemblies,
        repository_dir=args.repository_dir,
        force=bool(args.force),
    )
    for result in results:
        status = "cache" if result.get("cache_hit") else "built"
        _logger.info(
            "%s: %s -> %s",
            status,
            result["assembly_name"],
            result["artifact_dir"],
        )

    production_mode = bool(
        getattr(args, "production", False)
        or getattr(args, "slice", False)
        or getattr(args, "upload", False)
    )
    visualization_mode = bool(getattr(args, "visualize", False) or production_mode)
    if not visualization_mode:
        return 0

    if len(requested_assemblies) != 1:
        raise BuilderError(
            "Visualization and production builder runs require exactly one --assembly"
        )

    scene_assemblies = list(requested_assemblies)
    if getattr(args, "with_dependents", False):
        expanded_scene_set = set(
            _expand_with_dependents(assemblies, requested_assemblies)
        )
        scene_assemblies = [
            result["assembly_name"]
            for result in results
            if result["assembly_name"] in expanded_scene_set
        ]

    return _export_scene_for_assembly(
        args=args,
        config_data=config_data,
        build_results=results,
        selected_assembly=requested_assemblies[0],
        scene_assembly_names=scene_assemblies,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ShellForgePy declarative builder")
    parser.add_argument("config_file", help="Path to assemblies YAML file")
    parser.add_argument(
        "--assembly",
        action="append",
        help="Build only the named assembly. Repeat for multiple assemblies.",
    )
    parser.add_argument(
        "--repository-dir",
        help="Override the repository directory used for cached STEP artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when a matching hashed metadata file already exists.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export a scene for the selected assembly using Builder.Visualization rules.",
    )
    parser.add_argument(
        "--with-dependents",
        action="store_true",
        help="Include downstream dependent assemblies in the build and exported scene.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Export a production scene using Builder.Production rules.",
    )
    parser.add_argument(
        "--slice",
        action="store_true",
        help="Run OrcaSlicer after exporting a production scene.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated G-code files after slicing (implies --slice).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open OrcaSlicer with the generated 3MF project after slicing.",
    )
    parser.add_argument(
        "--run-id", help="Override automatically generated run identifier"
    )
    parser.add_argument("--runs-dir", help="Directory to store exported run artifacts")
    parser.add_argument(
        "--master-settings-dir",
        help="Path to Orca master settings directory for production slicing",
    )
    parser.add_argument(
        "--orca-executable",
        help="Path to the OrcaSlicer executable",
    )
    parser.add_argument(
        "--orca-debug",
        type=int,
        help="Debug level for OrcaSlicer",
    )
    parser.add_argument(
        "--printer",
        help="Printer destination to use for --upload",
    )
    parser.add_argument(
        "--part-file",
        help="Override the generated STL path for downstream slicer steps",
    )
    parser.add_argument(
        "--process-file",
        help="Override the generated process JSON path for downstream slicer steps",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_builder(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
