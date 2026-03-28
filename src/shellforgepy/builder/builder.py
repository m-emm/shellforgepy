"""Declarative assembly builder for ShellForgePy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
from shellforgepy.construct.part_parameters import PartParameters
from shellforgepy.metrics import (
    reset_metrics,
    snapshot_has_metrics,
    snapshot_metrics,
    using_metrics_snapshot,
    write_metrics_report,
)

from . import graph_model as builder_graph_model
from .errors import BuilderError

_logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r"\$\{([A-Za-z0-9_./:-]+)\}")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
BUILD_METADATA_SCHEMA_VERSION = 2


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
    source_version_inputs: Dict[str, Any]
    step_path: Path
    part: Any
    placement_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class _PlacementExecutionState:
    alignments: List[Dict[str, Any]]
    known_assembly_names: List[str]
    placement_context: Dict[str, Any]
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None
    cursor: int = 0
    executed_alignment_indices: set[int] = field(default_factory=set)
    translation_history: Dict[str, List[Callable[[Any], Any]]] = field(
        default_factory=dict
    )
    placement_offsets: Dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _resolve_inline_value(value: Any, context: Mapping[str, Any]) -> Any:
    local = dict(context)

    def resolver(name: str) -> Any:
        if name in local:
            return local[name]
        raise BuilderError(f"Unknown reference '{name}'")

    return _resolve_value(value, resolver)


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
        elif parameter_name in context:
            raw_value = context[parameter_name]
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
    return builder_graph_model.dependency_names(entry)


def _placement_build_dependencies(
    assemblies: Sequence[Mapping[str, Any]],
    config_data: Optional[Mapping[str, Any]] = None,
    *,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> Dict[str, List[str]]:
    model = graph_model or builder_graph_model.build_graph_model(
        assemblies, config_data
    )
    return {
        name: list(dependencies)
        for name, dependencies in model.placement_build_dependencies.items()
    }


def _resolve_build_generations(
    assemblies: Sequence[Mapping[str, Any]],
    selected_names: Optional[Iterable[str]] = None,
    config_data: Optional[Mapping[str, Any]] = None,
    *,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> List[List[Dict[str, Any]]]:
    model = graph_model or builder_graph_model.build_graph_model(
        assemblies, config_data
    )
    by_name = model.assemblies_by_name
    generations = builder_graph_model.resolve_build_generation_names(
        model, selected_names
    )
    resolved_generations: List[List[Dict[str, Any]]] = []
    for generation_index, generation_names in enumerate(generations):
        entries_in_generation: List[Dict[str, Any]] = []
        for assembly_name in generation_names:
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
    return builder_graph_model.assemblies_by_name(assemblies)


def _expand_with_dependents(
    assemblies: Sequence[Mapping[str, Any]],
    selected_names: Sequence[str],
    *,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> List[str]:
    model = graph_model or builder_graph_model.build_graph_model(assemblies)
    return builder_graph_model.expand_with_dependents(model, selected_names)


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
    return _ResolvedAssembly(
        name=str(entry["name"]),
        entry=dict(entry),
        resource_path=resource_path,
        resource_data=resource_data,
        public_parameters=public_parameters,
        generator_kwargs=generator_kwargs,
        generator_path=str(generator_path),
        logical_part_name=logical_part_name,
        parameter_hash="",
    )


def _load_generator_callable(generator_path: str) -> Callable[..., Any]:
    if "." not in generator_path:
        raise BuilderError(
            f"Generator path must be module.function, got '{generator_path}'"
        )

    module_name, function_name = generator_path.rsplit(".", 1)
    try:
        importlib.invalidate_caches()
        module_spec = importlib.util.find_spec(module_name)
        if module_spec and module_spec.origin and module_spec.origin.endswith(".py"):
            pyc_path = Path(importlib.util.cache_from_source(module_spec.origin))
            pyc_path.unlink(missing_ok=True)
        sys.modules.pop(module_name, None)
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


def _generator_accepts_keyword_argument(
    generator: Callable[..., Any], argument_name: str
) -> bool:
    try:
        signature = inspect.signature(generator)
    except (TypeError, ValueError):
        return False

    if argument_name in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _generator_context_payload(global_context: Mapping[str, Any]) -> Dict[str, Any]:
    return deepcopy(dict(global_context))


def _generator_source_path(
    generator: Callable[..., Any], generator_path: str
) -> Optional[Path]:
    source_file = inspect.getsourcefile(generator)
    if source_file is None:
        try:
            source_file = inspect.getfile(generator)
        except TypeError:
            source_file = None
    if source_file is None:
        module_name = generator_path.rsplit(".", 1)[0]
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            source_file = module_file
    if not source_file:
        return None
    candidate = Path(source_file).expanduser().resolve()
    if not candidate.exists():
        return None
    return candidate


def _version_inputs(
    *,
    generator: Callable[..., Any],
    generator_path: str,
    resource_path: Path,
) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "generator_path": generator_path,
        "resource_path": str(resource_path.resolve()),
        "resource_sha256": _file_sha256(resource_path.resolve()),
    }
    generator_source_path = _generator_source_path(generator, generator_path)
    if generator_source_path is not None:
        inputs["generator_source_path"] = str(generator_source_path)
        inputs["generator_source_sha256"] = _file_sha256(generator_source_path)
    return inputs


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


def _metadata_supports_current_schema(metadata: Mapping[str, Any]) -> bool:
    if int(metadata.get("schema_version", 0)) < BUILD_METADATA_SCHEMA_VERSION:
        return False
    metrics = metadata.get("metrics")
    return isinstance(metrics, Mapping) and "snapshot" in metrics


def _import_dependency_part(step_path: Path) -> Any:
    from shellforgepy.simple import import_solid_from_step

    return import_solid_from_step(str(step_path))


def _import_dependency_assembly(metadata: Mapping[str, Any]) -> Any:
    from shellforgepy.construct.leader_followers_cutters_part import (
        LeaderFollowersCuttersPart,
    )

    artifacts = metadata.get("artifacts", {})
    leader_path = artifacts.get("leader_step")
    if not leader_path:
        raise BuilderError(
            f"Dependency assembly '{metadata.get('assembly_name')}' has no leader artifact"
        )

    assembly = LeaderFollowersCuttersPart(
        _import_dependency_part(Path(leader_path).expanduser().resolve())
    )

    for item in artifacts.get("followers", []):
        follower = _import_dependency_part(Path(item["path"]).expanduser().resolve())
        name = item.get("name")
        if name:
            assembly.add_named_follower(follower, str(name))
        else:
            assembly.followers.append(follower)

    for item in artifacts.get("cutters", []):
        cutter = _import_dependency_part(Path(item["path"]).expanduser().resolve())
        name = item.get("name")
        if name:
            assembly.add_named_cutter(cutter, str(name))
        else:
            assembly.cutters.append(cutter)

    for item in artifacts.get("non_production_parts", []):
        non_production_part = _import_dependency_part(
            Path(item["path"]).expanduser().resolve()
        )
        name = item.get("name")
        if name:
            assembly.add_named_non_production_part(non_production_part, str(name))
        else:
            assembly.non_production_parts.append(non_production_part)

    return assembly


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


def _declared_dependency_names_for_metadata(
    metadata: Mapping[str, Any],
    config_data: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    declared = metadata.get("declared_dependencies")
    if isinstance(declared, list):
        return [str(item) for item in declared]

    if config_data is None:
        return []

    assembly_name = metadata.get("assembly_name")
    if not assembly_name:
        return []

    for entry in _get_assemblies(config_data):
        if str(entry.get("name")) == str(assembly_name):
            return _dependency_names(entry)
    return []


def _metrics_aggregation_closure(
    build_results: Sequence[Mapping[str, Any]],
    root_assembly_names: Sequence[str],
) -> List[Mapping[str, Any]]:
    by_name = {
        str(result["assembly_name"]): result
        for result in build_results
        if result.get("assembly_name")
    }
    ordered_results: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    stack = list(root_assembly_names)

    while stack:
        assembly_name = stack.pop()
        if assembly_name in seen:
            continue
        seen.add(assembly_name)

        metadata = by_name.get(assembly_name)
        if metadata is None:
            continue

        ordered_results.append(metadata)
        declared_dependencies = list(metadata.get("declared_dependencies") or [])
        injected_dependencies = [
            dependency.get("assembly_name")
            for dependency in metadata.get("dependencies", [])
            if dependency.get("assembly_name")
        ]
        for dependency_name in reversed(
            [*declared_dependencies, *injected_dependencies]
        ):
            stack.append(str(dependency_name))

    return ordered_results


def _combined_metrics_snapshot_for_results(
    build_results: Sequence[Mapping[str, Any]],
    root_assembly_names: Sequence[str],
) -> dict[str, Any] | None:
    from shellforgepy.metrics import merge_metrics_snapshot

    matching_results = _metrics_aggregation_closure(build_results, root_assembly_names)
    if not matching_results:
        return None

    reset_metrics()
    try:
        for metadata in reversed(matching_results):
            metrics = metadata.get("metrics") or {}
            snapshot = metrics.get("snapshot")
            if snapshot:
                merge_metrics_snapshot(snapshot)

        combined_snapshot = snapshot_metrics()
        if not snapshot_has_metrics(combined_snapshot):
            return None
        return combined_snapshot
    finally:
        reset_metrics()


def _scene_metrics_assembly_names(
    *,
    seed_assemblies: Sequence[str],
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    mode: str,
    config_data: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    resolved: List[str] = []
    seen: set[str] = set()
    pending = list(seed_assemblies)

    while pending:
        assembly_name = pending.pop()
        if assembly_name in seen:
            continue
        seen.add(assembly_name)
        resolved.append(assembly_name)

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
        for dependency_name in reversed(
            _scene_dependency_names(resource_data, mode, config_data)
        ):
            if dependency_name not in seen:
                pending.append(dependency_name)

    return resolved


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
    placement_state: Optional[_PlacementExecutionState] = None,
) -> List[_DependencyInjection]:
    injections: List[_DependencyInjection] = []
    for kwarg_name, spec in _resolve_injected_parts_config(entry).items():
        assembly_name = spec["assembly"]
        metadata = _dependency_metadata_for_assembly(
            assembly_name,
            built_results_by_name,
            repository_dir,
        )
        if spec["artifact"] == "assembly":
            step_path = (
                Path(metadata["artifacts"]["leader_step"]).expanduser().resolve()
            )
            imported_part = _import_dependency_assembly(metadata)
        else:
            step_path = _dependency_step_path(metadata, spec["artifact"])
            imported_part = _import_dependency_part(step_path)
        imported_part = _apply_placement_state_to_part(
            imported_part, assembly_name, placement_state
        )
        injections.append(
            _DependencyInjection(
                kwarg_name=kwarg_name,
                assembly_name=assembly_name,
                artifact=spec["artifact"],
                source_parameter_hash=str(metadata["parameter_hash"]),
                source_version_inputs=dict(metadata.get("version_inputs", {})),
                step_path=step_path,
                part=imported_part,
                placement_offset=_placement_offset_for_assembly(
                    assembly_name, placement_state
                ),
            )
        )
    return injections


def _hash_with_dependencies(
    generator_kwargs: Mapping[str, Any],
    injections: Sequence[_DependencyInjection],
    generator_context: Optional[Mapping[str, Any]] = None,
    version_inputs: Optional[Mapping[str, Any]] = None,
) -> str:
    def normalize_hash_value(value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)):
            return value
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    hash_inputs = {
        key: normalize_hash_value(value) for key, value in generator_kwargs.items()
    }
    if generator_context:
        for key, value in generator_context.items():
            hash_inputs[f"__context__{key}"] = normalize_hash_value(value)
    if version_inputs:
        for key, value in version_inputs.items():
            hash_inputs[f"__version__{key}"] = value
    for injection in injections:
        hash_inputs[f"__dependency__{injection.kwarg_name}"] = (
            f"{injection.assembly_name}:{injection.artifact}:{injection.source_parameter_hash}:{json.dumps(injection.placement_offset)}"
        )
    return PartParameters(hash_inputs).parameters_hash()


def _metadata_resolution_context(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    context = dict(metadata.get("public_parameters") or {})
    context.update(metadata.get("generator_kwargs") or {})
    context.update(metadata.get("generator_context") or {})
    return context


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
    if "." in artifact:
        artifact_group, target_name = artifact.split(".", 1)
        if artifact_group not in {"followers", "cutters", "non_production_parts"}:
            raise BuilderError(f"Unsupported scene artifact selector '{artifact}'")
        for item in artifacts.get(artifact_group, []):
            if item.get("name") == target_name:
                return [
                    {
                        "assembly_name": assembly_name,
                        "artifact": artifact_group,
                        "path": str(item["path"]),
                        "name": item.get("name"),
                        "index": item.get("index"),
                    }
                ]
        return []
    raise BuilderError(f"Unsupported scene artifact selector '{artifact}'")


def _filter_artifact_entries(
    entries: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    include_names = rule.get("names")
    exclude_names = rule.get("exclude_names")
    if include_names is not None:
        if not isinstance(include_names, list):
            raise BuilderError("Scene rule names must be a list")
        include_set = {str(item) for item in include_names}
    else:
        include_set = None

    if exclude_names is not None:
        if not isinstance(exclude_names, list):
            raise BuilderError("Scene rule exclude_names must be a list")
        exclude_set = {str(item) for item in exclude_names}
    else:
        exclude_set = set()

    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        name = entry.get("name")
        if include_set is not None and str(name) not in include_set:
            continue
        if name is not None and str(name) in exclude_set:
            continue
        filtered.append(dict(entry))
    return filtered


def _placement_section(config_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return builder_graph_model.placement_section(config_data)


def _placement_alignments(
    config_data: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return builder_graph_model.placement_alignments(config_data)


def _known_assembly_names(
    config_data: Optional[Mapping[str, Any]],
    built_results_by_name: Mapping[str, Dict[str, Any]],
) -> List[str]:
    names = set(str(name) for name in built_results_by_name)
    if config_data is not None:
        names.update(str(entry["name"]) for entry in _get_assemblies(config_data))
    return sorted(names, key=len, reverse=True)


def _parse_part_reference(
    reference: str,
    known_assembly_names: Sequence[str],
) -> tuple[str, str]:
    return builder_graph_model.parse_part_reference(reference, known_assembly_names)


def _resolve_alignment_enum(raw_alignment: Any) -> Any:
    from shellforgepy.construct.alignment import Alignment

    if isinstance(raw_alignment, Alignment):
        return raw_alignment

    try:
        return getattr(Alignment, str(raw_alignment))
    except AttributeError as exc:
        raise BuilderError(f"Unknown placement alignment '{raw_alignment}'") from exc


def _make_placement_translation(
    moving_anchor: Any,
    target_anchor: Any,
    *,
    alignment: Any,
    axes: Optional[Sequence[int]] = None,
    stack_gap: float = 0,
) -> Callable[[Any], Any]:
    from shellforgepy.construct.alignment_operations import align_translation

    return align_translation(
        moving_anchor,
        target_anchor,
        alignment,
        axes=axes,
        stack_gap=stack_gap,
    )


def _resolve_post_translation(
    alignment_spec: Mapping[str, Any],
    placement_context: Mapping[str, Any],
) -> Optional[tuple[float, float, float]]:
    raw_post_translation = alignment_spec.get("post_translation")
    if raw_post_translation is None:
        return None

    resolved_post_translation = _resolve_inline_value(
        raw_post_translation,
        placement_context,
    )
    if not isinstance(resolved_post_translation, (list, tuple)):
        raise BuilderError(
            "placement alignment post_translation must resolve to a 3-item list"
        )
    if len(resolved_post_translation) != 3:
        raise BuilderError(
            "placement alignment post_translation must resolve to exactly 3 values"
        )
    return tuple(float(value) for value in resolved_post_translation)


def _make_post_translation(delta: Sequence[float]) -> Callable[[Any], Any]:
    from shellforgepy.simple import translate

    return translate(*delta)


def _compose_translations(
    *translations: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    def apply(part: Any) -> Any:
        transformed = part
        for translation in translations:
            transformed = translation(transformed)
        return transformed

    return apply


def _placement_reference_names(
    config_data: Optional[Mapping[str, Any]],
    built_results_by_name: Mapping[str, Dict[str, Any]],
) -> List[str]:
    known_assembly_names = _known_assembly_names(config_data, built_results_by_name)
    referenced: set[str] = set()
    for moving_assembly, target_assembly in _placement_alignment_references(
        config_data, known_assembly_names
    ):
        referenced.add(moving_assembly)
        referenced.add(target_assembly)
    return sorted(referenced)


def _placement_alignment_references(
    config_data: Optional[Mapping[str, Any]],
    known_assembly_names: Sequence[str],
    *,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> List[tuple[str, str]]:
    model = graph_model
    if model is None:
        assemblies = [{"name": name} for name in sorted(set(known_assembly_names))]
        model = builder_graph_model.build_graph_model(assemblies, config_data)
    return [
        (step.moving_assembly_name, step.target_assembly_name)
        for step in model.placement_steps
    ]


def _placement_dependency_closure(
    config_data: Optional[Mapping[str, Any]],
    known_assembly_names: Sequence[str],
    root_assemblies: Iterable[str],
    *,
    stop_index: Optional[int] = None,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> set[str]:
    model = graph_model
    if model is None:
        assemblies = [{"name": name} for name in sorted(set(known_assembly_names))]
        model = builder_graph_model.build_graph_model(assemblies, config_data)
    return set(
        builder_graph_model.placement_dependency_closure(
            model,
            root_assemblies,
            stop_index=stop_index,
        )
    )


def _relevant_placement_alignment_indices(
    config_data: Optional[Mapping[str, Any]],
    known_assembly_names: Sequence[str],
    root_assemblies: Iterable[str],
    *,
    stop_index: Optional[int] = None,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> List[int]:
    model = graph_model
    if model is None:
        assemblies = [{"name": name} for name in sorted(set(known_assembly_names))]
        model = builder_graph_model.build_graph_model(assemblies, config_data)
    return builder_graph_model.relevant_placement_alignment_indices(
        model,
        root_assemblies,
        stop_index=stop_index,
    )


def _scene_dependency_names(
    resource_data: Mapping[str, Any],
    mode: str,
    config_data: Optional[Mapping[str, Any]],
) -> List[str]:
    return builder_graph_model.scene_dependency_names(resource_data, mode, config_data)


def _dependency_metadata_map(
    metadata: Mapping[str, Any],
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    config_data: Optional[Mapping[str, Any]] = None,
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
    for assembly_name in _declared_dependency_names_for_metadata(metadata, config_data):
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


def _scene_transform_record(
    function_name: str, resolved_kwargs: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "kind": "scene_transform",
        "function": function_name,
        "kwargs": deepcopy(dict(resolved_kwargs)),
    }


def _apply_scene_transform(
    part: Any,
    transform: Optional[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> tuple[Any, Optional[Dict[str, Any]]]:
    if not transform:
        return part, None
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
    return function(part, **resolved_kwargs), _scene_transform_record(
        function_name, resolved_kwargs
    )


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
        metadata, built_results_by_name, repository_dir, config_data
    )
    parameter_context = _metadata_resolution_context(metadata)

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
                dependency_target = dependency_metadata.get(dependency_key)
                if dependency_target is None:
                    dependency_target = _dependency_metadata_for_assembly(
                        dependency_key,
                        built_results_by_name,
                        repository_dir,
                    )
                    dependency_metadata[dependency_key] = dependency_target
                targets = [dependency_target]
            else:
                targets = [
                    dependency_metadata[name] for name in sorted(dependency_metadata)
                ]
        else:
            raise BuilderError(f"Unsupported scene source '{source}'")

        for target in targets:
            entries = _filter_artifact_entries(
                _artifact_entries_for_selector(target, artifact),
                rule,
            )
            for entry in entries:
                part = _import_dependency_part(
                    Path(entry["path"]).expanduser().resolve()
                )
                transform_history: List[Dict[str, Any]] = []
                part_context = dict(parameter_context)
                part_context.update(
                    {
                        "assembly_name": entry["assembly_name"],
                        "artifact": entry["artifact"],
                        "part_name": entry.get("name"),
                        "part_index": entry.get("index"),
                    }
                )
                part, transform_record = _apply_scene_transform(
                    part, rule.get("transform"), part_context
                )
                if transform_record is not None:
                    transform_history.append(transform_record)
                scene_parts.append(
                    {
                        "name": _format_scene_name(entry, rule),
                        "part": part,
                        "assembly_name": entry["assembly_name"],
                        "source_path": entry["path"],
                        "source_parameter_hash": target.get("parameter_hash"),
                        "source_version_inputs": deepcopy(
                            dict(target.get("version_inputs", {}))
                        ),
                        "transform_history": transform_history,
                        "flip": bool(rule.get("flip", False)),
                        "skip_in_production": bool(
                            rule.get("skip_in_production", False)
                        ),
                        "prod_rotation_angle": rule.get("prod_rotation_angle"),
                        "prod_rotation_axis": rule.get("prod_rotation_axis"),
                        "color": rule.get("color"),
                        "animation": (
                            _resolve_inline_value(rule.get("animation"), part_context)
                            if rule.get("animation") is not None
                            else None
                        ),
                    }
                )
    return scene_parts


def _apply_translation_to_scene_parts(
    scene_parts: List[Dict[str, Any]],
    assembly_name: str,
    translation: Callable[[Any], Any],
    transform_record: Optional[Mapping[str, Any]] = None,
) -> None:
    for item in scene_parts:
        if str(item.get("assembly_name")) != assembly_name:
            continue
        item["part"] = translation(item["part"])
        if transform_record is not None:
            history = list(item.get("transform_history") or [])
            history.append(deepcopy(dict(transform_record)))
            item["transform_history"] = history


def _part_center(part: Any) -> tuple[float, float, float]:
    from shellforgepy.simple import get_bounding_box_center

    center = get_bounding_box_center(part)
    return tuple(float(value) for value in center)


def _format_point(point: Sequence[float]) -> str:
    from shellforgepy.simple import point_string

    return point_string(point)


def _translation_delta(
    before_center: Sequence[float], after_center: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        float(after_value - before_value)
        for before_value, after_value in zip(before_center, after_center)
    )


def _add_translation_delta(
    existing: Sequence[float], delta: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        float(existing_value + delta_value)
        for existing_value, delta_value in zip(existing, delta)
    )


def _initialize_placement_execution_state(
    config_data: Optional[Mapping[str, Any]],
    built_results_by_name: Mapping[str, Dict[str, Any]],
    *,
    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None,
) -> _PlacementExecutionState:
    if graph_model is None:
        if config_data is not None and config_data.get("assemblies") is not None:
            graph_model = builder_graph_model.build_graph_model(
                _get_assemblies(config_data),
                config_data,
            )
        else:
            graph_model = builder_graph_model.build_graph_model(
                [{"name": str(name)} for name in sorted(built_results_by_name)],
                config_data,
            )
    return _PlacementExecutionState(
        alignments=_placement_alignments(config_data),
        known_assembly_names=_known_assembly_names(config_data, built_results_by_name),
        placement_context=(
            resolve_globals(config_data.get("globals", {})) if config_data else {}
        ),
        graph_model=graph_model,
    )


def _apply_translation_sequence(
    part: Any, translations: Sequence[Callable[[Any], Any]]
) -> Any:
    transformed = part
    for translation in translations:
        transformed = translation(transformed)
    return transformed


def _apply_placement_state_to_part(
    part: Any,
    assembly_name: str,
    placement_state: Optional[_PlacementExecutionState],
) -> Any:
    if placement_state is None:
        return part
    translations = placement_state.translation_history.get(assembly_name, [])
    if not translations:
        return part
    return _apply_translation_sequence(part, translations)


def _placement_offset_for_assembly(
    assembly_name: str,
    placement_state: Optional[_PlacementExecutionState],
) -> tuple[float, float, float]:
    if placement_state is None:
        return (0.0, 0.0, 0.0)
    return placement_state.placement_offsets.get(assembly_name, (0.0, 0.0, 0.0))


def _resolve_placement_anchor(
    reference: str,
    *,
    placement_state: _PlacementExecutionState,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
) -> tuple[str, str, Any]:
    assembly_name, selector = _parse_part_reference(
        reference, placement_state.known_assembly_names
    )
    metadata = _dependency_metadata_for_assembly(
        assembly_name,
        built_results_by_name,
        repository_dir,
    )
    entries = _artifact_entries_for_selector(metadata, selector)
    if len(entries) != 1:
        raise BuilderError(
            f"Placement reference '{reference}' did not resolve to exactly one artifact"
        )
    anchor = _import_dependency_part(Path(entries[0]["path"]).expanduser().resolve())
    anchor = _apply_placement_state_to_part(anchor, assembly_name, placement_state)
    return assembly_name, selector, anchor


def _advance_placement_execution(
    placement_state: _PlacementExecutionState,
    *,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
) -> _PlacementExecutionState:
    placement_steps = (
        placement_state.graph_model.placement_steps
        if placement_state.graph_model is not None
        else [
            builder_graph_model.PlacementStep(
                index=index,
                spec=dict(alignment),
                moving_reference=str(alignment["part"]),
                target_reference=str(alignment["to"]),
                moving_assembly_name=_parse_part_reference(
                    str(alignment["part"]), placement_state.known_assembly_names
                )[0],
                target_assembly_name=_parse_part_reference(
                    str(alignment["to"]), placement_state.known_assembly_names
                )[0],
            )
            for index, alignment in enumerate(placement_state.alignments)
        ]
    )
    placement_dag = (
        placement_state.graph_model.placement_execution_dag
        if placement_state.graph_model is not None
        else builder_graph_model.build_graph_model(
            [{"name": name} for name in placement_state.known_assembly_names],
            {"placement": {"alignments": placement_state.alignments}},
        ).placement_execution_dag
    )

    while True:
        progress = False
        for placement_step in placement_steps:
            if placement_step.index in placement_state.executed_alignment_indices:
                continue
            predecessors = set(placement_dag.predecessors(placement_step.index))
            if not predecessors.issubset(placement_state.executed_alignment_indices):
                continue

            alignment_spec = placement_step.spec
            moving_reference_str = placement_step.moving_reference
            target_reference_str = placement_step.target_reference
            moving_assembly_name = placement_step.moving_assembly_name
            target_assembly_name = placement_step.target_assembly_name

            if moving_assembly_name not in built_results_by_name:
                continue
            if target_assembly_name not in built_results_by_name:
                continue

            _, _, moving_anchor = _resolve_placement_anchor(
                moving_reference_str,
                placement_state=placement_state,
                built_results_by_name=built_results_by_name,
                repository_dir=repository_dir,
            )
            _, _, target_anchor = _resolve_placement_anchor(
                target_reference_str,
                placement_state=placement_state,
                built_results_by_name=built_results_by_name,
                repository_dir=repository_dir,
            )
            _, _, moving_part = _resolve_placement_anchor(
                moving_assembly_name,
                placement_state=placement_state,
                built_results_by_name=built_results_by_name,
                repository_dir=repository_dir,
            )
            _, _, target_part = _resolve_placement_anchor(
                target_assembly_name,
                placement_state=placement_state,
                built_results_by_name=built_results_by_name,
                repository_dir=repository_dir,
            )

            resolved_alignment = _resolve_alignment_enum(
                alignment_spec.get("alignment")
            )
            raw_axes = alignment_spec.get("axes")
            axes = None
            if raw_axes is not None:
                resolved_axes = _resolve_inline_value(
                    raw_axes, placement_state.placement_context
                )
                if not isinstance(resolved_axes, list):
                    raise BuilderError(
                        "placement alignment axes must resolve to a list"
                    )
                axes = [int(axis) for axis in resolved_axes]

            stack_gap = alignment_spec.get("stack_gap", 0)
            resolved_stack_gap = _resolve_inline_value(
                stack_gap, placement_state.placement_context
            )
            resolved_post_translation = _resolve_post_translation(
                alignment_spec,
                placement_state.placement_context,
            )

            moving_center_before = _part_center(moving_anchor)
            target_center_before = _part_center(target_anchor)
            moving_part_center_before = _part_center(moving_part)
            target_part_center_before = _part_center(target_part)

            translation = _make_placement_translation(
                moving_anchor,
                target_anchor,
                alignment=resolved_alignment,
                axes=axes,
                stack_gap=resolved_stack_gap,
            )
            if resolved_post_translation is not None:
                translation = _compose_translations(
                    translation,
                    _make_post_translation(resolved_post_translation),
                )
            moved_anchor = translation(moving_anchor)
            moving_center_after = _part_center(moved_anchor)
            delta = _translation_delta(moving_center_before, moving_center_after)

            _logger.info(
                "Placement step: %s aligned to %s via %s; moving_anchor_center=%s; target_anchor_center=%s; moving_part_position=%s; target_part_position=%s; shift=%s",
                moving_reference_str,
                target_reference_str,
                resolved_alignment.name,
                _format_point(moving_center_before),
                _format_point(target_center_before),
                _format_point(moving_part_center_before),
                _format_point(target_part_center_before),
                _format_point(delta),
            )

            placement_state.translation_history.setdefault(
                moving_assembly_name, []
            ).append(translation)
            placement_state.placement_offsets[moving_assembly_name] = (
                _add_translation_delta(
                    placement_state.placement_offsets.get(
                        moving_assembly_name, (0.0, 0.0, 0.0)
                    ),
                    delta,
                )
            )
            placement_state.executed_alignment_indices.add(placement_step.index)
            placement_state.cursor = len(placement_state.executed_alignment_indices)
            progress = True

        if not progress:
            break

    return placement_state


def _apply_placement_alignments(
    scene_parts: List[Dict[str, Any]],
    *,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    config_data: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    alignments = _placement_alignments(config_data)
    if not alignments:
        return scene_parts

    graph_model: Optional[builder_graph_model.BuilderGraphModel] = None
    if config_data is not None and config_data.get("assemblies") is not None:
        graph_model = builder_graph_model.build_graph_model(
            _get_assemblies(config_data),
            config_data,
        )
    known_assembly_names = _known_assembly_names(config_data, built_results_by_name)
    placement_context = (
        resolve_globals(config_data.get("globals", {})) if config_data else {}
    )
    relevant_assemblies = {
        str(item["assembly_name"])
        for item in scene_parts
        if item.get("assembly_name") is not None
    }
    relevant_alignment_indices = set(
        _relevant_placement_alignment_indices(
            config_data,
            known_assembly_names,
            relevant_assemblies,
            graph_model=graph_model,
        )
    )

    metadata_by_name: Dict[str, Dict[str, Any]] = {
        str(name): dict(value) for name, value in built_results_by_name.items()
    }

    anchor_cache: Dict[tuple[str, str], Any] = {}
    translation_history: Dict[str, List[Callable[[Any], Any]]] = {}

    def resolve_metadata(assembly_name: str) -> Dict[str, Any]:
        metadata = metadata_by_name.get(assembly_name)
        if metadata is None:
            metadata = _dependency_metadata_for_assembly(
                assembly_name,
                built_results_by_name,
                repository_dir,
            )
            metadata_by_name[assembly_name] = metadata
        return metadata

    def resolve_anchor(reference: str) -> tuple[str, str, Any]:
        assembly_name, selector = _parse_part_reference(reference, known_assembly_names)
        cache_key = (assembly_name, selector)
        if cache_key not in anchor_cache:
            metadata = resolve_metadata(assembly_name)
            entries = _artifact_entries_for_selector(metadata, selector)
            if len(entries) != 1:
                raise BuilderError(
                    f"Placement reference '{reference}' did not resolve to exactly one artifact"
                )
            imported_anchor = _import_dependency_part(
                Path(entries[0]["path"]).expanduser().resolve()
            )
            for translation in translation_history.get(assembly_name, []):
                imported_anchor = translation(imported_anchor)
            anchor_cache[cache_key] = imported_anchor
        return assembly_name, selector, anchor_cache[cache_key]

    for placement_index, alignment_spec in enumerate(alignments):
        if placement_index not in relevant_alignment_indices:
            continue
        moving_reference = alignment_spec.get("part")
        target_reference = alignment_spec.get("to")
        if not moving_reference or not target_reference:
            raise BuilderError("Each placement alignment requires 'part' and 'to'")

        moving_reference_str = str(moving_reference)
        target_reference_str = str(target_reference)

        moving_assembly_name, _, moving_anchor = resolve_anchor(moving_reference_str)
        target_assembly_name, _, target_anchor = resolve_anchor(target_reference_str)

        resolved_alignment = _resolve_alignment_enum(alignment_spec.get("alignment"))
        raw_axes = alignment_spec.get("axes")
        axes = None
        if raw_axes is not None:
            resolved_axes = _resolve_inline_value(raw_axes, placement_context)
            if not isinstance(resolved_axes, list):
                raise BuilderError("placement alignment axes must resolve to a list")
            axes = [int(axis) for axis in resolved_axes]

        stack_gap = alignment_spec.get("stack_gap", 0)
        resolved_stack_gap = _resolve_inline_value(stack_gap, placement_context)
        resolved_post_translation = _resolve_post_translation(
            alignment_spec,
            placement_context,
        )

        moving_center_before = _part_center(moving_anchor)
        target_center_before = _part_center(target_anchor)
        _, _, moving_part = resolve_anchor(moving_assembly_name)
        _, _, target_part = resolve_anchor(target_assembly_name)
        moving_part_center_before = _part_center(moving_part)
        target_part_center_before = _part_center(target_part)

        translation = _make_placement_translation(
            moving_anchor,
            target_anchor,
            alignment=resolved_alignment,
            axes=axes,
            stack_gap=resolved_stack_gap,
        )
        if resolved_post_translation is not None:
            translation = _compose_translations(
                translation,
                _make_post_translation(resolved_post_translation),
            )

        moved_anchor = translation(moving_anchor)
        moving_center_after = _part_center(moved_anchor)
        translation_delta = _translation_delta(
            moving_center_before, moving_center_after
        )

        _logger.info(
            "Placement step: %s aligned to %s via %s; moving_anchor_center=%s; target_anchor_center=%s; moving_part_position=%s; target_part_position=%s; shift=%s",
            moving_reference_str,
            target_reference_str,
            resolved_alignment.name,
            _format_point(moving_center_before),
            _format_point(target_center_before),
            _format_point(moving_part_center_before),
            _format_point(target_part_center_before),
            _format_point(translation_delta),
        )

        _apply_translation_to_scene_parts(
            scene_parts,
            moving_assembly_name,
            translation,
            transform_record={
                "kind": "translate",
                "vector": [float(value) for value in translation_delta],
                "placement_step": placement_index,
            },
        )
        translation_history.setdefault(moving_assembly_name, []).append(translation)
        for (assembly_name, selector), anchor_part in list(anchor_cache.items()):
            if assembly_name != moving_assembly_name:
                continue
            if anchor_part is moving_anchor:
                anchor_cache[(assembly_name, selector)] = moved_anchor
                continue
            anchor_cache[(assembly_name, selector)] = translation(anchor_part)

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
        context = _metadata_resolution_context(metadata)
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
            "export_step": False,
            "export_obj": True,
            "export_individual_parts": False,
            "export_stl": False,
            "plates": None,
            "auto_assign_plates": False,
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
            "plates": None,
            "auto_assign_plates": False,
        }

    resolved = dict(defaults)
    resolved.update(dict(arrange))
    return resolved


def _apply_prototype_arrange_overrides(
    export_options: Mapping[str, Any],
    *,
    selected_metadata: Mapping[str, Any],
    selected_resource_data: Mapping[str, Any],
    config_data: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    production_section = _builder_section(
        selected_resource_data,
        "Production",
        config_data,
    )
    prototype = production_section.get("prototype")
    if prototype is None:
        return dict(export_options)
    if not isinstance(prototype, Mapping):
        raise BuilderError("Builder.Production.prototype must be a mapping")

    arrange = prototype.get("arrange")
    if arrange is None:
        return dict(export_options)
    if not isinstance(arrange, Mapping):
        raise BuilderError("Builder.Production.prototype.arrange must be a mapping")

    resolved = dict(export_options)
    resolved.update(
        _resolve_inline_mapping(
            arrange, _metadata_resolution_context(selected_metadata)
        )
    )
    return resolved


def _resolve_prototype_reference(reference: str, selected_assembly: str) -> str:
    normalized = str(reference).strip()
    if not normalized:
        raise BuilderError("Prototype reference must not be empty")
    if normalized.startswith("self."):
        return f"{selected_assembly}.{normalized[len('self.'):]}"
    if "." not in normalized or normalized.startswith(
        ("leader", "fused", "followers.", "cutters.", "non_production_parts.")
    ):
        return f"{selected_assembly}.{normalized}"
    return normalized


def _resolve_prototype_part_names(
    raw_names: Any,
    context: Mapping[str, Any],
    *,
    field_name: str,
) -> Optional[set[str]]:
    if raw_names is None:
        return None
    resolved_names = _resolve_inline_value(raw_names, context)
    if not isinstance(resolved_names, list):
        raise BuilderError(f"Builder.Production.prototype.{field_name} must be a list")
    return {str(item) for item in resolved_names}


def _resolve_prototype_anchor_part(
    reference: str,
    *,
    selected_assembly: str,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    placement_state: Optional[_PlacementExecutionState],
    config_data: Optional[Mapping[str, Any]],
) -> Any:
    resolved_reference = _resolve_prototype_reference(reference, selected_assembly)
    known_assembly_names = _known_assembly_names(config_data, built_results_by_name)
    assembly_name, selector = _parse_part_reference(
        resolved_reference, known_assembly_names
    )
    metadata = _dependency_metadata_for_assembly(
        assembly_name,
        built_results_by_name,
        repository_dir,
    )
    entries = _artifact_entries_for_selector(metadata, selector)
    if len(entries) != 1:
        raise BuilderError(
            f"Prototype anchor '{reference}' did not resolve to exactly one artifact"
        )
    anchor = _import_dependency_part(Path(entries[0]["path"]).expanduser().resolve())
    return _apply_placement_state_to_part(anchor, assembly_name, placement_state)


def _apply_prototype_configuration(
    scene_parts: List[Dict[str, Any]],
    *,
    selected_metadata: Mapping[str, Any],
    selected_resource_data: Mapping[str, Any],
    selected_assembly: str,
    built_results_by_name: Mapping[str, Dict[str, Any]],
    repository_dir: Path,
    placement_state: Optional[_PlacementExecutionState],
    config_data: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    from shellforgepy.construct.alignment import Alignment
    from shellforgepy.geometry.keepouts import create_box_hole_cutter
    from shellforgepy.simple import align, translate

    production_section = _builder_section(
        selected_resource_data,
        "Production",
        config_data,
    )
    prototype = production_section.get("prototype")
    if prototype is None:
        raise BuilderError(
            f"Assembly '{selected_assembly}' does not declare Builder.Production.prototype"
        )
    if not isinstance(prototype, Mapping):
        raise BuilderError("Builder.Production.prototype must be a mapping")

    context = _metadata_resolution_context(selected_metadata)
    include_parts = _resolve_prototype_part_names(
        prototype.get("include_parts"),
        context,
        field_name="include_parts",
    )
    exclude_parts = (
        _resolve_prototype_part_names(
            prototype.get("exclude_parts"),
            context,
            field_name="exclude_parts",
        )
        or set()
    )

    filtered_scene_parts: List[Dict[str, Any]] = []
    for item in scene_parts:
        name = str(item.get("name"))
        if include_parts is not None and name not in include_parts:
            continue
        if name in exclude_parts:
            continue
        filtered_scene_parts.append(dict(item))

    if not filtered_scene_parts:
        raise BuilderError(
            f"Prototype selection for assembly '{selected_assembly}' resolved to no parts"
        )

    by_name = {str(item["name"]): item for item in filtered_scene_parts}
    box_cutters = prototype.get("box_cutters", [])
    if box_cutters is None:
        box_cutters = []
    if not isinstance(box_cutters, list):
        raise BuilderError("Builder.Production.prototype.box_cutters must be a list")

    for index, cutter_spec in enumerate(box_cutters):
        if not isinstance(cutter_spec, Mapping):
            raise BuilderError("Prototype box cutter entries must be mappings")

        target_names: list[str]
        if "part" in cutter_spec and "parts" in cutter_spec:
            raise BuilderError(
                "Prototype box cutter may specify either 'part' or 'parts', not both"
            )
        if "part" in cutter_spec:
            target_names = [str(_resolve_inline_value(cutter_spec["part"], context))]
        elif "parts" in cutter_spec:
            resolved_parts = _resolve_inline_value(cutter_spec["parts"], context)
            if not isinstance(resolved_parts, list):
                raise BuilderError("Prototype box cutter parts must resolve to a list")
            target_names = [str(item) for item in resolved_parts]
        else:
            raise BuilderError("Prototype box cutter requires 'part' or 'parts'")

        around = cutter_spec.get("around") or cutter_spec.get("center_on")
        if around is None:
            raise BuilderError("Prototype box cutter requires 'around'")
        around_reference = str(_resolve_inline_value(around, context))
        anchor_part = _resolve_prototype_anchor_part(
            around_reference,
            selected_assembly=selected_assembly,
            built_results_by_name=built_results_by_name,
            repository_dir=repository_dir,
            placement_state=placement_state,
            config_data=config_data,
        )

        resolved_size = _resolve_inline_value(cutter_spec.get("size"), context)
        if not isinstance(resolved_size, (list, tuple)) or len(resolved_size) != 3:
            raise BuilderError("Prototype box cutter size must resolve to three values")
        size = tuple(float(value) for value in resolved_size)

        resolved_cutter_size = cutter_spec.get(
            "cutter_size",
            context.get("BIG_THING", 500.0),
        )
        cutter_size = float(_resolve_inline_value(resolved_cutter_size, context))

        keep_volume = create_box_hole_cutter(
            size[0],
            size[1],
            size[2],
            cutter_size=cutter_size,
        )
        keep_volume = align(keep_volume, anchor_part, Alignment.CENTER)
        if cutter_spec.get("offset") is not None:
            resolved_offset = _resolve_inline_value(cutter_spec.get("offset"), context)
            if (
                not isinstance(resolved_offset, (list, tuple))
                or len(resolved_offset) != 3
            ):
                raise BuilderError(
                    "Prototype box cutter offset must resolve to three values"
                )
            keep_volume = translate(
                float(resolved_offset[0]),
                float(resolved_offset[1]),
                float(resolved_offset[2]),
            )(keep_volume)

        for target_name in target_names:
            target_part = by_name.get(target_name)
            if target_part is None:
                raise BuilderError(
                    f"Prototype box cutter #{index + 1} targets unknown part '{target_name}'"
                )
            target_part["part"] = keep_volume.use_as_cutter_on(target_part["part"])
            transform_history = list(target_part.get("transform_history") or [])
            transform_history.append(
                {
                    "kind": "prototype_box_cutter",
                    "part": target_name,
                    "around": around_reference,
                    "size": [float(value) for value in size],
                    "cutter_size": cutter_size,
                    **(
                        {
                            "offset": [
                                float(resolved_offset[0]),
                                float(resolved_offset[1]),
                                float(resolved_offset[2]),
                            ]
                        }
                        if cutter_spec.get("offset") is not None
                        else {}
                    ),
                }
            )
            target_part["transform_history"] = transform_history

    return filtered_scene_parts


def _filter_declared_plates_for_scene_parts(
    declared_plates: Any,
    scene_parts: Sequence[Mapping[str, Any]],
) -> Any:
    if not declared_plates:
        return declared_plates

    valid_names = {str(item["name"]) for item in scene_parts}

    if isinstance(declared_plates, dict):
        filtered: Dict[str, List[str]] = {}
        for plate_name, part_names in declared_plates.items():
            filtered_parts = [
                str(name) for name in list(part_names or []) if str(name) in valid_names
            ]
            if filtered_parts:
                filtered[str(plate_name)] = filtered_parts
        return filtered

    if isinstance(declared_plates, list):
        filtered_list: List[Dict[str, Any]] = []
        for index, entry in enumerate(declared_plates, start=1):
            if not isinstance(entry, Mapping):
                raise BuilderError("Each declared plate must be a mapping")
            filtered_parts = [
                str(name)
                for name in list(entry.get("parts", []) or [])
                if str(name) in valid_names
            ]
            if not filtered_parts:
                continue
            filtered_entry = dict(entry)
            filtered_entry["name"] = str(entry.get("name") or f"plate_{index}")
            filtered_entry["parts"] = filtered_parts
            filtered_list.append(filtered_entry)
        return filtered_list

    raise BuilderError("plates must be a mapping or a list")


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
    graph_model = builder_graph_model.build_graph_model(assemblies, config_data)
    build_generations = _resolve_build_generations(
        assemblies,
        assembly_names,
        config_data,
        graph_model=graph_model,
    )

    results: List[Dict[str, Any]] = []
    built_results_by_name: Dict[str, Dict[str, Any]] = {}
    placement_state = _initialize_placement_execution_state(
        config_data,
        built_results_by_name,
        graph_model=graph_model,
    )
    for generation_index, generation_entries in enumerate(build_generations):
        for assembly_index, entry in enumerate(generation_entries):
            resolved = _resolve_assembly(config_file, entry, global_context)
            dependency_injections = _resolve_dependency_injections(
                entry,
                built_results_by_name,
                repository_path,
                placement_state,
            )
            generator = _load_generator_callable(resolved.generator_path)
            generator_context = (
                _generator_context_payload(global_context)
                if _generator_accepts_keyword_argument(generator, "context")
                else None
            )
            version_inputs = _version_inputs(
                generator=generator,
                generator_path=resolved.generator_path,
                resource_path=resolved.resource_path,
            )
            parameter_hash = _hash_with_dependencies(
                resolved.generator_kwargs,
                dependency_injections,
                generator_context,
                version_inputs,
            )
            artifact_dir = (
                repository_path / resolved.name / f"{resolved.name}__{parameter_hash}"
            )
            metadata_path = _metadata_path(artifact_dir, resolved.name, parameter_hash)

            if metadata_path.exists() and not force:
                cached_metadata = _read_metadata(metadata_path)
                if _metadata_supports_current_schema(cached_metadata):
                    cached_metadata["cache_hit"] = True
                    results.append(cached_metadata)
                    built_results_by_name[resolved.name] = cached_metadata
                    placement_state = _advance_placement_execution(
                        placement_state,
                        built_results_by_name=built_results_by_name,
                        repository_dir=repository_path,
                    )
                    _logger.info(
                        "Using cached build for '%s' from %s",
                        resolved.name,
                        metadata_path,
                    )
                    continue
                _logger.info(
                    "Rebuilding assembly '%s' because cached metadata at %s uses an older schema",
                    resolved.name,
                    metadata_path,
                )

            _logger.info(
                "Building assembly '%s' with %s",
                resolved.name,
                resolved.generator_path,
            )
            generator_kwargs = deepcopy(resolved.generator_kwargs)
            for injection in dependency_injections:
                generator_kwargs[injection.kwarg_name] = injection.part
            if generator_context is not None:
                generator_kwargs["context"] = deepcopy(generator_context)
            reset_metrics()
            try:
                generated_part = generator(**generator_kwargs)
                metrics_snapshot = snapshot_metrics()
            finally:
                reset_metrics()
            artifacts = _export_artifacts(
                resolved.name,
                parameter_hash,
                generated_part,
                artifact_dir,
            )
            metrics_report_path = write_metrics_report(
                artifact_dir,
                base_name=f"{resolved.name}__{parameter_hash}__metrics_report",
                snapshot=metrics_snapshot,
            )

            metadata = {
                "schema_version": BUILD_METADATA_SCHEMA_VERSION,
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
                "generator_context": generator_context,
                "version_inputs": version_inputs,
                "generation": generation_index,
                "assembly_in_generation": assembly_index,
                "declared_dependencies": _dependency_names(entry),
                "dependencies": [
                    {
                        "kwarg_name": injection.kwarg_name,
                        "assembly_name": injection.assembly_name,
                        "artifact": injection.artifact,
                        "source_parameter_hash": injection.source_parameter_hash,
                        "source_assembly_hash": injection.source_parameter_hash,
                        "source_version_inputs": injection.source_version_inputs,
                        "step_path": str(injection.step_path),
                        **(
                            {
                                "source_placement_offset": list(
                                    injection.placement_offset
                                )
                            }
                            if injection.placement_offset != (0.0, 0.0, 0.0)
                            else {}
                        ),
                    }
                    for injection in dependency_injections
                ],
                "artifacts": artifacts,
                "metrics": {
                    "has_metrics": snapshot_has_metrics(metrics_snapshot),
                    "snapshot": metrics_snapshot,
                    "report_path": (
                        None
                        if metrics_report_path is None
                        else str(metrics_report_path.resolve())
                    ),
                },
            }
            _write_metadata(metadata_path, metadata)
            results.append(metadata)
            built_results_by_name[resolved.name] = metadata
            placement_state = _advance_placement_execution(
                placement_state,
                built_results_by_name=built_results_by_name,
                repository_dir=repository_path,
            )

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
    selected_resource_data = _load_yaml(
        Path(selected_metadata["resource_file"]).expanduser().resolve()
    )

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
            scene_parts.append(part)

    scene_parts = _apply_placement_alignments(
        scene_parts,
        built_results_by_name=built_results_by_name,
        repository_dir=repository_dir,
        config_data=config_data,
    )

    if bool(getattr(args, "prototype", False)):
        placement_state = _initialize_placement_execution_state(
            config_data,
            built_results_by_name,
        )
        placement_state = _advance_placement_execution(
            placement_state,
            built_results_by_name=built_results_by_name,
            repository_dir=repository_dir,
        )
        scene_parts = _apply_prototype_configuration(
            scene_parts,
            selected_metadata=selected_metadata,
            selected_resource_data=selected_resource_data,
            selected_assembly=selected_assembly,
            built_results_by_name=built_results_by_name,
            repository_dir=repository_dir,
            placement_state=placement_state,
            config_data=config_data,
        )

    if not scene_parts:
        raise BuilderError(
            f"No scene parts resolved for assembly '{selected_assembly}' in {mode} mode"
        )

    process_data = None
    if production_mode:
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
        selected_resource_data,
        mode,
        config_data,
    )
    if bool(getattr(args, "prototype", False)):
        export_options = _apply_prototype_arrange_overrides(
            export_options,
            selected_metadata=selected_metadata,
            selected_resource_data=selected_resource_data,
            config_data=config_data,
        )
        export_options["plates"] = _filter_declared_plates_for_scene_parts(
            export_options.get("plates"),
            scene_parts,
        )
    metrics_assembly_names = _scene_metrics_assembly_names(
        seed_assemblies=scene_assembly_names,
        built_results_by_name=built_results_by_name,
        repository_dir=repository_dir,
        mode=mode,
        config_data=config_data,
    )
    combined_metrics_snapshot = _combined_metrics_snapshot_for_results(
        build_results,
        metrics_assembly_names,
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
        exportable_scene_parts: List[Dict[str, Any]] = []
        for part in scene_parts:
            export_part = dict(part)
            export_part.pop("assembly_name", None)
            exportable_scene_parts.append(export_part)
        enforce_bed_size = not (
            bool(getattr(args, "visualize", False))
            and production_mode
            and not bool(getattr(args, "slice", False))
            and not bool(getattr(args, "upload", False))
        )
        with using_metrics_snapshot(combined_metrics_snapshot):
            arrange_and_export_parts(
                exportable_scene_parts,
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
                mesh_cache_dir=repository_dir / "__mesh_cache__" / "obj",
                plates=export_options.get("plates"),
                auto_assign_plates=bool(
                    export_options.get("auto_assign_plates", False)
                ),
                enforce_bed_size=enforce_bed_size,
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
    graph_model = builder_graph_model.build_graph_model(assemblies, config_data)
    assemblies_by_name = graph_model.assemblies_by_name

    requested_assemblies = list(args.assembly or [])
    build_assemblies: Optional[List[str]] = requested_assemblies or None
    if requested_assemblies and getattr(args, "with_dependents", False):
        build_assemblies = _expand_with_dependents(
            assemblies,
            requested_assemblies,
            graph_model=graph_model,
        )

    production_mode = bool(
        getattr(args, "production", False)
        or getattr(args, "prototype", False)
        or getattr(args, "slice", False)
        or getattr(args, "upload", False)
    )
    visualization_mode = bool(getattr(args, "visualize", False) or production_mode)

    if visualization_mode and len(requested_assemblies) == 1:
        selected_assembly = requested_assemblies[0]
        selected_entry = assemblies_by_name.get(selected_assembly)
        if selected_entry is None:
            raise BuilderError(f"Unknown assembly name '{selected_assembly}'")
        selected_resource_data = _load_yaml(
            _discover_resource_file(config_file, selected_entry)
        )
        scene_mode = "production" if production_mode else "visualization"
        implicit_scene_dependencies = set(
            _scene_dependency_names(selected_resource_data, scene_mode, config_data)
        )
        implicit_scene_dependencies.update(
            _placement_dependency_closure(
                config_data,
                sorted(assemblies_by_name),
                set(requested_assemblies) | implicit_scene_dependencies,
                graph_model=graph_model,
            )
        )
        if implicit_scene_dependencies:
            build_selection = set(build_assemblies or [])
            build_selection.update(requested_assemblies)
            build_selection.update(implicit_scene_dependencies)
            build_assemblies = sorted(build_selection)

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

    if not visualization_mode:
        return 0

    if len(requested_assemblies) != 1:
        raise BuilderError(
            "Visualization and production builder runs require exactly one --assembly"
        )

    scene_assemblies = list(requested_assemblies)
    if getattr(args, "with_dependents", False):
        expanded_scene_set = set(
            _expand_with_dependents(
                assemblies,
                requested_assemblies,
                graph_model=graph_model,
            )
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
        "--prototype",
        action="store_true",
        help="Apply Builder.Production.prototype filtering and clipping before export.",
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
