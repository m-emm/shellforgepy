"""First-class metrics collection and reporting for ShellForgePy."""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping


class Material(Enum):
    ALUMINUM = "ALUMINUM"
    PLA = "PLA"
    PLA_CF = "PLA_CF"
    PETG = "PETG"
    PETG_CF = "PETG_CF"
    STEEL = "STEEL"
    TPU = "TPU"

    @property
    def density_kg_per_m3(self) -> float:
        return {
            Material.ALUMINUM: 2.70e3,
            Material.PLA: 1.24e3,
            Material.PLA_CF: 1.30e3,
            Material.PETG: 1.27e3,
            Material.PETG_CF: 1.30e3,
            Material.STEEL: 7.85e3,
            Material.TPU: 1.21e3,
        }[self]

    @property
    def density_g_per_cm3(self) -> float:
        return self.density_kg_per_m3 / 1000.0


class BuildingKind(Enum):
    GARAGE = "garage"
    APPARTMENT = "appartment"
    REFURBISHMENT = "refurbishment"


@dataclass(frozen=True)
class LengthMetric:
    category: str
    stock_type: str
    part_name: str
    length_mm: float


@dataclass(frozen=True)
class MarkMetric:
    stock_type: str
    part_name: str
    stock_length_mm: float
    mark_name: str
    position_mm: float


@dataclass(frozen=True)
class WeightMetric:
    assembly_id: str
    material: Material
    volume_mm3: float | None = None
    mass_kg: float | None = None
    part_id: str | None = None


@dataclass(frozen=True)
class BuildingCostMetric:
    assembly_id: str
    building_kind: BuildingKind
    cubic_meters: float
    cost_per_m3: float
    part_id: str | None = None


@dataclass(frozen=True)
class LivingSpaceMetric:
    assembly_id: str
    square_meters: float
    part_id: str | None = None


METRICS_SNAPSHOT_SCHEMA_VERSION = 2

_length_metrics: list[LengthMetric] = []
_mark_metrics: list[MarkMetric] = []
_weight_metrics: list[WeightMetric] = []
_building_cost_per_m3_map: dict[BuildingKind, float] = {}
_building_cost_metrics: list[BuildingCostMetric] = []
_living_space_metrics: list[LivingSpaceMetric] = []
_metrics_report_logged = False


def _set_report_logged(value: bool) -> None:
    global _metrics_report_logged
    _metrics_report_logged = value


def has_metrics() -> bool:
    return bool(
        _length_metrics
        or _mark_metrics
        or _weight_metrics
        or _building_cost_metrics
        or _living_space_metrics
    )


def was_metrics_report_logged() -> bool:
    return _metrics_report_logged


def reset_metrics() -> None:
    _length_metrics.clear()
    _mark_metrics.clear()
    _weight_metrics.clear()
    _building_cost_per_m3_map.clear()
    _building_cost_metrics.clear()
    _living_space_metrics.clear()
    _set_report_logged(False)


def _coerce_building_kind(building_kind: BuildingKind | str) -> BuildingKind:
    if isinstance(building_kind, BuildingKind):
        return building_kind
    if not isinstance(building_kind, str):
        raise TypeError("building_kind must be an instance of BuildingKind or a string")

    normalized = building_kind.strip()
    if not normalized:
        raise ValueError("building_kind must be a non-empty string")

    try:
        return BuildingKind[normalized.upper()]
    except KeyError:
        pass

    try:
        return BuildingKind(normalized.lower())
    except ValueError as exc:
        valid_values = ", ".join(kind.value for kind in BuildingKind)
        raise ValueError(
            f"unsupported building_kind {building_kind!r}; expected one of {valid_values}"
        ) from exc


def configure_building_cost_per_m3_map(
    building_cost_per_m3_map: Mapping[BuildingKind | str, float],
) -> None:
    normalized_map: dict[BuildingKind, float] = {}
    for building_kind, cost_per_m3 in building_cost_per_m3_map.items():
        kind = _coerce_building_kind(building_kind)
        normalized_cost_per_m3 = round(float(cost_per_m3), 6)
        if normalized_cost_per_m3 < 0:
            raise ValueError("building cost per m3 values must be non-negative")
        normalized_map[kind] = normalized_cost_per_m3

    _building_cost_per_m3_map.clear()
    _building_cost_per_m3_map.update(normalized_map)
    _set_report_logged(False)


def record_length_metric(
    category: str, stock_type: str, part_name: str, length_mm: float
) -> None:
    _set_report_logged(False)
    _length_metrics.append(
        LengthMetric(
            category=category,
            stock_type=stock_type,
            part_name=part_name,
            length_mm=round(float(length_mm), 3),
        )
    )


def record_mark_metric(
    stock_type: str,
    part_name: str,
    stock_length_mm: float,
    mark_name: str,
    position_mm: float,
) -> None:
    _set_report_logged(False)
    _mark_metrics.append(
        MarkMetric(
            stock_type=stock_type,
            part_name=part_name,
            stock_length_mm=round(float(stock_length_mm), 3),
            mark_name=mark_name,
            position_mm=round(float(position_mm), 3),
        )
    )


def calculate_mass_kg(material: Material, volume_mm3: float) -> float:
    return round(float(material.density_kg_per_m3) * float(volume_mm3) * 1e-9, 6)


def calculate_weight_g(material: Material, volume_mm3: float) -> float:
    return round(calculate_mass_kg(material, volume_mm3) * 1e3, 3)


def record_weight_metric(
    assembly_id: str,
    material: Material,
    volume_mm3: float,
    part_id: str | None = None,
) -> None:
    if not assembly_id:
        raise ValueError("assembly_id must be a non-empty string")
    if not isinstance(material, Material):
        raise TypeError("material must be an instance of Material")

    normalized_volume_mm3 = round(float(volume_mm3), 3)
    if normalized_volume_mm3 < 0:
        raise ValueError("volume_mm3 must be non-negative")

    _set_report_logged(False)
    _weight_metrics.append(
        WeightMetric(
            assembly_id=assembly_id,
            material=material,
            volume_mm3=normalized_volume_mm3,
            mass_kg=None,
            part_id=part_id,
        )
    )


def record_measured_mass_metric(
    assembly_id: str,
    material: Material,
    mass_kg: float,
    part_id: str | None = None,
) -> None:
    if not assembly_id:
        raise ValueError("assembly_id must be a non-empty string")
    if not isinstance(material, Material):
        raise TypeError("material must be an instance of Material")

    normalized_mass_kg = round(float(mass_kg), 6)
    if normalized_mass_kg < 0:
        raise ValueError("mass_kg must be non-negative")

    _set_report_logged(False)
    _weight_metrics.append(
        WeightMetric(
            assembly_id=assembly_id,
            material=material,
            volume_mm3=None,
            mass_kg=normalized_mass_kg,
            part_id=part_id,
        )
    )


def record_building_cost_metric(
    assembly_id: str,
    building_kind: BuildingKind | str,
    cubic_meters: float,
    part_id: str | None = None,
) -> None:
    if not assembly_id:
        raise ValueError("assembly_id must be a non-empty string")
    kind = _coerce_building_kind(building_kind)
    if kind not in _building_cost_per_m3_map:
        raise ValueError(f"building cost per m3 is not configured for {kind.value!r}")

    normalized_cubic_meters = round(float(cubic_meters), 6)
    if normalized_cubic_meters < 0:
        raise ValueError("cubic_meters must be non-negative")

    _set_report_logged(False)
    _building_cost_metrics.append(
        BuildingCostMetric(
            assembly_id=assembly_id,
            building_kind=kind,
            cubic_meters=normalized_cubic_meters,
            cost_per_m3=_building_cost_per_m3_map[kind],
            part_id=part_id,
        )
    )


def record_living_space_metric(
    assembly_id: str,
    square_meters: float,
    part_id: str | None = None,
) -> None:
    if not assembly_id:
        raise ValueError("assembly_id must be a non-empty string")

    normalized_square_meters = round(float(square_meters), 6)
    if normalized_square_meters < 0:
        raise ValueError("square_meters must be non-negative")

    _set_report_logged(False)
    _living_space_metrics.append(
        LivingSpaceMetric(
            assembly_id=assembly_id,
            square_meters=normalized_square_meters,
            part_id=part_id,
        )
    )


def get_metric_mass_kg(metric: WeightMetric) -> float:
    if metric.mass_kg is not None:
        return metric.mass_kg
    if metric.volume_mm3 is None:
        raise ValueError("weight metric must define either mass_kg or volume_mm3")
    return calculate_mass_kg(metric.material, metric.volume_mm3)


def build_weight_totals_by_assembly_id() -> dict[str, float]:
    totals_by_assembly_id = defaultdict(float)
    for metric in _weight_metrics:
        totals_by_assembly_id[metric.assembly_id] += get_metric_mass_kg(metric)

    return {
        assembly_id: round(total_mass_kg, 6)
        for assembly_id, total_mass_kg in sorted(totals_by_assembly_id.items())
    }


def get_building_metric_cost(metric: BuildingCostMetric) -> float:
    return round(metric.cubic_meters * metric.cost_per_m3, 2)


def build_building_cost_totals_by_assembly_id() -> dict[str, float]:
    totals_by_assembly_id = defaultdict(float)
    for metric in _building_cost_metrics:
        totals_by_assembly_id[metric.assembly_id] += get_building_metric_cost(metric)

    return {
        assembly_id: round(total_cost, 2)
        for assembly_id, total_cost in sorted(totals_by_assembly_id.items())
    }


def build_living_space_totals_by_assembly_id() -> dict[str, float]:
    totals_by_assembly_id = defaultdict(float)
    for metric in _living_space_metrics:
        totals_by_assembly_id[metric.assembly_id] += metric.square_meters

    return {
        assembly_id: round(total_square_meters, 3)
        for assembly_id, total_square_meters in sorted(totals_by_assembly_id.items())
    }


def _round_length_mm(length_mm: float) -> int:
    return int(float(length_mm) + 0.5)


def _format_length(length_mm: float) -> str:
    return str(_round_length_mm(length_mm))


def _format_mass_kg(mass_kg: float) -> str:
    return f"{float(mass_kg):.6f} kg"


def _format_cubic_meters(cubic_meters: float) -> str:
    return f"{float(cubic_meters):.3f} m3"


def _format_square_meters(square_meters: float) -> str:
    return f"{float(square_meters):.3f} m2"


def _format_building_cost(cost: float) -> str:
    return f"{round(float(cost) / 1000.0):.0f}k"


def _format_cost_per_m3(cost_per_m3: float) -> str:
    return f"{float(cost_per_m3):.0f}"


def build_cut_stock_report_lines() -> list[str]:
    if not _length_metrics and not _mark_metrics:
        return ["Cut stock metrics: no metrics recorded."]

    grouped_metrics = defaultdict(list)
    for metric in _length_metrics:
        grouped_metrics[(metric.category, metric.stock_type)].append(metric)

    lines = ["Cut stock metrics:"]
    for category, stock_type in sorted(grouped_metrics):
        lines.append(f"{category} {stock_type}:")

        lengths_grouped = defaultdict(list)
        for metric in grouped_metrics[(category, stock_type)]:
            lengths_grouped[_round_length_mm(metric.length_mm)].append(metric.part_name)

        for length_mm in sorted(lengths_grouped):
            part_names = sorted(lengths_grouped[length_mm])
            lines.append(f"  {_format_length(length_mm)} mm x{len(part_names)}")
            for part_name in part_names:
                lines.append(f"    - {part_name}")

    if _mark_metrics:
        lines.append("Stock marks:")

        grouped_marks = defaultdict(list)
        for mark_metric in _mark_metrics:
            key = (
                mark_metric.stock_type,
                mark_metric.part_name,
                _round_length_mm(mark_metric.stock_length_mm),
            )
            grouped_marks[key].append(mark_metric)

        for stock_type, part_name, stock_length_mm in sorted(grouped_marks):
            lines.append(
                f"{part_name} ({stock_type}, {_format_length(stock_length_mm)} mm):"
            )
            for mark_metric in sorted(
                grouped_marks[(stock_type, part_name, stock_length_mm)],
                key=lambda current_mark: (
                    _round_length_mm(current_mark.position_mm),
                    current_mark.mark_name,
                ),
            ):
                lines.append(
                    f"  mark at {_format_length(mark_metric.position_mm)} mm - {mark_metric.mark_name}"
                )

    return lines


def build_building_cost_report_lines() -> list[str]:
    if not _building_cost_metrics:
        return ["Building cost metrics: no metrics recorded."]

    total_cubic_meters = round(
        sum(metric.cubic_meters for metric in _building_cost_metrics), 3
    )
    total_cost = round(
        sum(get_building_metric_cost(metric) for metric in _building_cost_metrics), 2
    )
    lines = [
        "Building cost metrics:",
        f"Overview: {_format_cubic_meters(total_cubic_meters)}, {_format_building_cost(total_cost)}",
    ]

    assembly_metrics = defaultdict(list)
    for metric in _building_cost_metrics:
        assembly_metrics[metric.assembly_id].append(metric)

    for assembly_id in sorted(assembly_metrics):
        metrics = assembly_metrics[assembly_id]
        assembly_cubic_meters = round(sum(metric.cubic_meters for metric in metrics), 3)
        assembly_cost = round(
            sum(get_building_metric_cost(metric) for metric in metrics), 2
        )
        lines.append(
            f"{assembly_id}: {_format_cubic_meters(assembly_cubic_meters)}, {_format_building_cost(assembly_cost)}"
        )

        kind_totals: dict[BuildingKind, dict[str, float]] = defaultdict(
            lambda: {"cubic_meters": 0.0, "cost": 0.0}
        )
        part_totals: dict[tuple[str, BuildingKind, float], dict[str, float]] = (
            defaultdict(lambda: {"cubic_meters": 0.0, "cost": 0.0})
        )
        for metric in metrics:
            metric_cost = get_building_metric_cost(metric)
            kind_totals[metric.building_kind]["cubic_meters"] += metric.cubic_meters
            kind_totals[metric.building_kind]["cost"] += metric_cost
            if metric.part_id:
                part_key = (metric.part_id, metric.building_kind, metric.cost_per_m3)
                part_totals[part_key]["cubic_meters"] += metric.cubic_meters
                part_totals[part_key]["cost"] += metric_cost

        for building_kind in sorted(kind_totals, key=lambda kind: kind.name):
            totals = kind_totals[building_kind]
            lines.append(
                f"  {building_kind.name}: {_format_cubic_meters(round(totals['cubic_meters'], 3))}, {_format_building_cost(round(totals['cost'], 2))}"
            )

        for part_id, building_kind, cost_per_m3 in sorted(
            part_totals,
            key=lambda current_part: (
                current_part[0],
                current_part[1].name,
                current_part[2],
            ),
        ):
            totals = part_totals[(part_id, building_kind, cost_per_m3)]
            lines.append(
                f"  {part_id} ({building_kind.name}): {_format_cubic_meters(round(totals['cubic_meters'], 3))} @ {_format_cost_per_m3(cost_per_m3)}/m3 = {_format_building_cost(round(totals['cost'], 2))}"
            )

    return lines


def build_living_space_report_lines() -> list[str]:
    if not _living_space_metrics:
        return ["Living space metrics: no metrics recorded."]

    total_square_meters = round(
        sum(metric.square_meters for metric in _living_space_metrics), 3
    )
    lines = [
        "Living space metrics:",
        f"Overview: {_format_square_meters(total_square_meters)}",
    ]

    assembly_metrics = defaultdict(list)
    for metric in _living_space_metrics:
        assembly_metrics[metric.assembly_id].append(metric)

    for assembly_id in sorted(assembly_metrics):
        metrics = assembly_metrics[assembly_id]
        assembly_square_meters = round(
            sum(metric.square_meters for metric in metrics), 3
        )
        lines.append(f"{assembly_id}: {_format_square_meters(assembly_square_meters)}")

        part_totals = defaultdict(float)
        for metric in metrics:
            if metric.part_id:
                part_totals[metric.part_id] += metric.square_meters

        for part_id in sorted(part_totals):
            lines.append(
                f"  {part_id}: {_format_square_meters(round(part_totals[part_id], 3))}"
            )

    return lines


def build_weight_report_lines() -> list[str]:
    if not _weight_metrics:
        return ["Weight metrics: no metrics recorded."]

    assembly_metrics = defaultdict(list)
    for metric in _weight_metrics:
        assembly_metrics[metric.assembly_id].append(metric)

    lines = ["Weight metrics:"]
    for assembly_id in sorted(assembly_metrics):
        material_totals = defaultdict(float)
        part_totals = defaultdict(float)
        for metric in assembly_metrics[assembly_id]:
            mass_kg = get_metric_mass_kg(metric)
            material_totals[metric.material] += mass_kg
            if metric.part_id:
                part_totals[(metric.part_id, metric.material)] += mass_kg

        total_mass_kg = round(sum(material_totals.values()), 6)
        lines.append(f"{assembly_id}: {_format_mass_kg(total_mass_kg)}")

        for material in sorted(
            material_totals, key=lambda current_material: current_material.name
        ):
            lines.append(
                f"  {material.name}: {_format_mass_kg(round(material_totals[material], 6))}"
            )

        for part_id, material in sorted(
            part_totals,
            key=lambda current_part: (current_part[0], current_part[1].name),
        ):
            lines.append(
                f"  {part_id} ({material.name}): {_format_mass_kg(round(part_totals[(part_id, material)], 6))}"
            )

    return lines


def build_metrics_report_lines() -> list[str]:
    if not has_metrics():
        return ["Cut stock metrics: no metrics recorded."]

    lines: list[str] = []
    report_sections = []
    if _length_metrics or _mark_metrics:
        report_sections.append(build_cut_stock_report_lines())
    if _weight_metrics:
        report_sections.append(build_weight_report_lines())
    if _building_cost_metrics:
        report_sections.append(build_building_cost_report_lines())
    if _living_space_metrics:
        report_sections.append(build_living_space_report_lines())

    for section in report_sections:
        if lines:
            lines.append("")
        lines.extend(section)
    return lines


def build_metrics_report_text() -> str:
    return "\n".join(build_metrics_report_lines()) + "\n"


def log_metrics_report(logger: logging.Logger) -> None:
    for line in build_metrics_report_lines():
        logger.info(line)
    _set_report_logged(True)


def snapshot_metrics() -> dict[str, Any]:
    return {
        "schema_version": METRICS_SNAPSHOT_SCHEMA_VERSION,
        "length_metrics": [asdict(metric) for metric in _length_metrics],
        "mark_metrics": [asdict(metric) for metric in _mark_metrics],
        "weight_metrics": [
            {
                **asdict(metric),
                "material": metric.material.name,
            }
            for metric in _weight_metrics
        ],
        "building_cost_per_m3_map": {
            building_kind.value: cost_per_m3
            for building_kind, cost_per_m3 in sorted(
                _building_cost_per_m3_map.items(), key=lambda item: item[0].name
            )
        },
        "building_cost_metrics": [
            {
                **asdict(metric),
                "building_kind": metric.building_kind.name,
            }
            for metric in _building_cost_metrics
        ],
        "living_space_metrics": [asdict(metric) for metric in _living_space_metrics],
    }


def _normalize_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {
            "schema_version": METRICS_SNAPSHOT_SCHEMA_VERSION,
            "length_metrics": [],
            "mark_metrics": [],
            "weight_metrics": [],
            "building_cost_per_m3_map": {},
            "building_cost_metrics": [],
            "living_space_metrics": [],
        }

    return {
        "schema_version": int(
            snapshot.get("schema_version", METRICS_SNAPSHOT_SCHEMA_VERSION)
        ),
        "length_metrics": list(snapshot.get("length_metrics") or []),
        "mark_metrics": list(snapshot.get("mark_metrics") or []),
        "weight_metrics": list(snapshot.get("weight_metrics") or []),
        "building_cost_per_m3_map": dict(
            snapshot.get("building_cost_per_m3_map") or {}
        ),
        "building_cost_metrics": list(snapshot.get("building_cost_metrics") or []),
        "living_space_metrics": list(snapshot.get("living_space_metrics") or []),
    }


def snapshot_has_metrics(snapshot: Mapping[str, Any] | None) -> bool:
    normalized = _normalize_snapshot(snapshot)
    return bool(
        normalized["length_metrics"]
        or normalized["mark_metrics"]
        or normalized["weight_metrics"]
        or normalized["building_cost_metrics"]
        or normalized["living_space_metrics"]
    )


def restore_metrics(
    snapshot: Mapping[str, Any] | None, *, append: bool = False
) -> None:
    normalized = _normalize_snapshot(snapshot)
    if not append:
        reset_metrics()

    for item in normalized["length_metrics"]:
        _length_metrics.append(
            LengthMetric(
                category=str(item["category"]),
                stock_type=str(item["stock_type"]),
                part_name=str(item["part_name"]),
                length_mm=round(float(item["length_mm"]), 3),
            )
        )

    for item in normalized["mark_metrics"]:
        _mark_metrics.append(
            MarkMetric(
                stock_type=str(item["stock_type"]),
                part_name=str(item["part_name"]),
                stock_length_mm=round(float(item["stock_length_mm"]), 3),
                mark_name=str(item["mark_name"]),
                position_mm=round(float(item["position_mm"]), 3),
            )
        )

    for item in normalized["weight_metrics"]:
        material_value = item["material"]
        if isinstance(material_value, Material):
            material = material_value
        else:
            material = Material[str(material_value)]
        _weight_metrics.append(
            WeightMetric(
                assembly_id=str(item["assembly_id"]),
                material=material,
                volume_mm3=(
                    None
                    if item.get("volume_mm3") is None
                    else round(float(item["volume_mm3"]), 3)
                ),
                mass_kg=(
                    None
                    if item.get("mass_kg") is None
                    else round(float(item["mass_kg"]), 6)
                ),
                part_id=(
                    None if item.get("part_id") is None else str(item.get("part_id"))
                ),
            )
        )

    for building_kind_value, cost_per_m3 in normalized[
        "building_cost_per_m3_map"
    ].items():
        _building_cost_per_m3_map[_coerce_building_kind(building_kind_value)] = round(
            float(cost_per_m3), 6
        )

    for item in normalized["building_cost_metrics"]:
        building_kind = _coerce_building_kind(item["building_kind"])
        _building_cost_metrics.append(
            BuildingCostMetric(
                assembly_id=str(item["assembly_id"]),
                building_kind=building_kind,
                cubic_meters=round(float(item["cubic_meters"]), 6),
                cost_per_m3=round(float(item["cost_per_m3"]), 6),
                part_id=(
                    None if item.get("part_id") is None else str(item.get("part_id"))
                ),
            )
        )

    for item in normalized["living_space_metrics"]:
        _living_space_metrics.append(
            LivingSpaceMetric(
                assembly_id=str(item["assembly_id"]),
                square_meters=round(float(item["square_meters"]), 6),
                part_id=(
                    None if item.get("part_id") is None else str(item.get("part_id"))
                ),
            )
        )

    _set_report_logged(False)


def merge_metrics_snapshot(snapshot: Mapping[str, Any] | None) -> None:
    restore_metrics(snapshot, append=True)


@contextmanager
def using_metrics_snapshot(snapshot: Mapping[str, Any] | None) -> Iterator[None]:
    previous_snapshot = snapshot_metrics()
    previous_logged = was_metrics_report_logged()
    try:
        restore_metrics(snapshot, append=False)
        yield
    finally:
        restore_metrics(previous_snapshot, append=False)
        _set_report_logged(previous_logged)


def write_metrics_report(
    output_dir: str | Path,
    *,
    base_name: str = "metrics_report",
    snapshot: Mapping[str, Any] | None = None,
) -> Path | None:
    active_snapshot = snapshot if snapshot is not None else snapshot_metrics()
    if not snapshot_has_metrics(active_snapshot):
        return None

    output_path = (Path(output_dir).expanduser() / f"{base_name}.txt").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with using_metrics_snapshot(active_snapshot):
        output_path.write_text(build_metrics_report_text(), encoding="utf-8")

    return output_path


__all__ = [
    "BuildingCostMetric",
    "BuildingKind",
    "LengthMetric",
    "LivingSpaceMetric",
    "MarkMetric",
    "Material",
    "METRICS_SNAPSHOT_SCHEMA_VERSION",
    "WeightMetric",
    "build_building_cost_report_lines",
    "build_building_cost_totals_by_assembly_id",
    "build_cut_stock_report_lines",
    "build_living_space_report_lines",
    "build_living_space_totals_by_assembly_id",
    "build_metrics_report_lines",
    "build_metrics_report_text",
    "build_weight_report_lines",
    "build_weight_totals_by_assembly_id",
    "calculate_mass_kg",
    "calculate_weight_g",
    "configure_building_cost_per_m3_map",
    "get_building_metric_cost",
    "get_metric_mass_kg",
    "has_metrics",
    "log_metrics_report",
    "merge_metrics_snapshot",
    "record_building_cost_metric",
    "record_length_metric",
    "record_living_space_metric",
    "record_mark_metric",
    "record_measured_mass_metric",
    "record_weight_metric",
    "reset_metrics",
    "restore_metrics",
    "snapshot_has_metrics",
    "snapshot_metrics",
    "using_metrics_snapshot",
    "was_metrics_report_logged",
    "write_metrics_report",
]
