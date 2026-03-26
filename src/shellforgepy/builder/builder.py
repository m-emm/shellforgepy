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
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

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
        raise BuilderError(f"Expected YAML mapping at {path}, got {type(loaded).__name__}")
    return loaded


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if any((candidate / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
            return candidate
    return start.resolve()


def _ensure_import_paths(project_root: Path, extra_paths: Optional[Sequence[str]] = None) -> None:
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
            raise BuilderError(f"Reference '{key}' resolved to None inside template '{template}'")
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
            raise BuilderError(f"Unsupported constant in expression '{expression}': {current.value!r}")
        if isinstance(current, ast.UnaryOp):
            operand = evaluate(current.operand)
            if isinstance(current.op, ast.UAdd):
                return +operand
            if isinstance(current.op, ast.USub):
                return -operand
            raise BuilderError(f"Unsupported unary operator in expression '{expression}'")
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
            raise BuilderError(f"Unsupported binary operator in expression '{expression}'")
        raise BuilderError(f"Unsupported expression node {type(current).__name__} in '{expression}'")

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
                raise BuilderError(f"$sub expects a string template, got {type(template).__name__}")
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


def _coerce_parameter_value(parameter_name: str, definition: Mapping[str, Any], value: Any) -> Any:
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
        raise BuilderError(f"Unknown parameter override(s): {', '.join(extra_parameters)}")
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
        raise BuilderError(f"Dependencies for assembly '{entry.get('name')}' must be a list")
    return [str(item) for item in dependencies]


def _resolve_build_order(
    assemblies: Sequence[Mapping[str, Any]],
    selected_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
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

    order: List[Dict[str, Any]] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise BuilderError(f"Cyclic dependency detected at assembly '{name}'")
        temporary.add(name)
        entry = by_name[name]
        for dependency in _dependency_names(entry):
            if dependency not in by_name:
                raise BuilderError(f"Assembly '{name}' depends on unknown assembly '{dependency}'")
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)
        if entry.get("disabled"):
            _logger.info("Skipping disabled assembly '%s'", name)
            return
        order.append(entry)

    for name in requested:
        visit(name)

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in order:
        name = str(entry["name"])
        if name not in seen:
            deduped.append(entry)
            seen.add(name)
    return deduped


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
    explicit_keys = ("assembly_file", "assembly_resource", "resource_file", "template_file")
    for key in explicit_keys:
        raw_value = entry.get(key)
        if not raw_value:
            continue
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (config_dir / candidate).resolve()
        if candidate.exists():
            return candidate
        raise BuilderError(f"Assembly resource file for '{entry['name']}' not found: {candidate}")

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

    raise BuilderError(f"Could not find assembly resource file for '{target_name}' in {config_dir}")


def _select_part_definition(
    resource_path: Path,
    resource_data: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    parts = resource_data.get("Parts")
    if not isinstance(parts, Mapping) or not parts:
        raise BuilderError(f"Assembly resource {resource_path} does not define any Parts")

    explicit_name = entry.get("logical_part_name") or entry.get("part_name") or entry.get("resource_name")
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
        key for key in parts if _safe_name(str(key)).lower() == _safe_name(str(entry["name"])).lower()
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
    logical_part_name, part_definition = _select_part_definition(resource_path, resource_data, entry)

    parameter_definitions = resource_data.get("Parameters", {})
    if not isinstance(parameter_definitions, Mapping):
        raise BuilderError(f"Parameters section in {resource_path} must be a mapping")

    parameter_overrides = entry.get("parameters", {})
    if not isinstance(parameter_overrides, Mapping):
        raise BuilderError(f"Parameters for assembly '{entry['name']}' must be a mapping")

    public_parameters = _resolve_public_parameters(
        parameter_definitions,
        parameter_overrides,
        global_context,
    )

    properties = part_definition.get("Properties", {})
    if not isinstance(properties, Mapping):
        raise BuilderError(f"Part '{logical_part_name}' in {resource_path} has invalid Properties")

    generator_path = properties.get("Generator")
    if not generator_path:
        raise BuilderError(f"Part '{logical_part_name}' in {resource_path} is missing Properties.Generator")

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
        raise BuilderError(f"Generator path must be module.function, got '{generator_path}'")

    module_name, function_name = generator_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BuilderError(f"Could not import generator module '{module_name}': {exc}") from exc

    try:
        generator = getattr(module, function_name)
    except AttributeError as exc:
        raise BuilderError(
            f"Generator function '{function_name}' not found in module '{module_name}'"
        ) from exc

    if not callable(generator):
        raise BuilderError(f"Resolved generator '{generator_path}' is not callable")
    return generator


def _group_named_parts(parts: Sequence[Any], name_map: Mapping[str, int]) -> List[Dict[str, Any]]:
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
            "fused": generated_part.leaders_followers_fused()
            if hasattr(generated_part, "leaders_followers_fused")
            else None,
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


def _export_artifacts(
    resolved: _ResolvedAssembly,
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
        resolved.name,
        resolved.parameter_hash,
        "leader",
    )
    _export_part_to_step(normalized["leader"], leader_path)
    artifacts["leader_step"] = str(leader_path)

    fused = normalized.get("fused")
    if fused is not None:
        fused_path = artifact_dir / _artifact_filename(
            resolved.name,
            resolved.parameter_hash,
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
                resolved.name,
                resolved.parameter_hash,
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
    ordered_entries = _resolve_build_order(assemblies, assembly_names)

    results: List[Dict[str, Any]] = []
    for entry in ordered_entries:
        resolved = _resolve_assembly(config_file, entry, global_context)
        artifact_dir = repository_path / resolved.name / f"{resolved.name}__{resolved.parameter_hash}"
        metadata_path = _metadata_path(artifact_dir, resolved.name, resolved.parameter_hash)

        if metadata_path.exists() and not force:
            cached_metadata = _read_metadata(metadata_path)
            cached_metadata["cache_hit"] = True
            results.append(cached_metadata)
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
        generated_part = generator(**deepcopy(resolved.generator_kwargs))
        artifacts = _export_artifacts(resolved, generated_part, artifact_dir)

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
            "parameter_hash": resolved.parameter_hash,
            "public_parameters": resolved.public_parameters,
            "generator_kwargs": resolved.generator_kwargs,
            "artifacts": artifacts,
        }
        _write_metadata(metadata_path, metadata)
        results.append(metadata)

    return results


def run_builder(args: argparse.Namespace) -> int:
    results = build_from_file(
        args.config_file,
        assembly_names=args.assembly,
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
    return 0


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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_builder(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
