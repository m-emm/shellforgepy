"""CAD-agnostic part arrangement and STL export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Import the adapter to delegate CAD-specific operations
from ..adapters.adapter_chooser import get_cad_adapter

# Type alias for CAD objects - actual type depends on the adapter
CadQueryObject = Any


def _to_shape(obj: Any) -> Any:
    """Convert a CAD object to a shape using the appropriate adapter."""
    adapter = get_cad_adapter()
    # For now, assume the adapter can handle the object directly
    # This may need to be expanded based on adapter-specific logic
    return obj


def _as_vector(vector: Union[Sequence[float], Any]) -> Any:
    """Convert a vector to the appropriate CAD vector type using the adapter."""
    if hasattr(vector, "__len__") and len(vector) == 3:
        return tuple(float(v) for v in vector)
    return vector


@dataclass
class PartCollector:
    """Accumulates CAD parts and fuses them into a single shape."""

    part: Optional[Any] = None

    def fuse(self, other: Any) -> Any:
        """Fuse this part with another part using the appropriate CAD adapter."""
        adapter = get_cad_adapter()
        if self.part is None:
            self.part = other
        else:
            # Delegate to adapter for fusing - this would need to be implemented in adapters
            self.part = adapter.fuse_parts(self.part, other)
        return self.part


@dataclass
class PartInfo:
    """Information about a part for production arrangement."""

    name: str
    part: Any  # CAD object type depends on the adapter
    flip: bool = False
    skip_in_production: bool = False
    prod_rotation_angle: Optional[float] = None
    prod_rotation_axis: Optional[Tuple[float, float, float]] = None


@dataclass
class NamedPart:
    """A CAD part with a name."""

    name: str
    part: Any  # CAD object type depends on the adapter

    def copy(self) -> "NamedPart":
        """Create a copy of this named part."""
        adapter = get_cad_adapter()
        return NamedPart(self.name, adapter.copy_part(self.part))

    def translate(self, vector: Sequence[float]) -> "NamedPart":
        """Translate this part by the given vector."""
        adapter = get_cad_adapter()
        translated_part = adapter.translate_part(self.part, vector)
        return NamedPart(self.name, translated_part)

    def rotate(
        self,
        angle: float,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        axis: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "NamedPart":
        """Rotate this part around the given axis."""
        adapter = get_cad_adapter()
        rotated_part = adapter.rotate_part(self.part, angle, center, axis)
        return NamedPart(self.name, rotated_part)


def _normalize_named_parts(
    parts: Optional[
        Sequence[Union[NamedPart, Mapping[str, object], Tuple[str, CadQueryObject]]]
    ],
) -> List[NamedPart]:
    if not parts:
        return []

    normalized: List[NamedPart] = []
    for idx, item in enumerate(parts):
        if isinstance(item, NamedPart):
            normalized.append(item)
            continue
        if isinstance(item, Mapping):
            if "name" not in item or "part" not in item:
                raise KeyError("Named parts mappings must contain 'name' and 'part'")
            normalized.append(NamedPart(str(item["name"]), _to_shape(item["part"])))
            continue
        if isinstance(item, tuple) and len(item) == 2:
            name, part = item
            normalized.append(NamedPart(str(name), _to_shape(part)))
            continue
        raise TypeError(
            f"Unsupported item in named parts sequence at index {idx}: {item!r}"
        )
    return normalized


class PartList:
    """Container for managing named CadQuery parts."""

    def __init__(self) -> None:
        self.parts: List[PartInfo] = []

    def add(
        self,
        part: Union[CadQueryObject, "LeaderFollowersCuttersPart"],
        name: str,
        *,
        flip: bool = False,
        skip_in_production: bool = False,
        prod_rotation_angle: Optional[float] = None,
        prod_rotation_axis: Optional[Sequence[float]] = None,
    ) -> None:
        if isinstance(part, LeaderFollowersCuttersPart):
            shape = part.get_leader_as_part()
        else:
            shape = _to_shape(part)

        if any(existing.name == name for existing in self.parts):
            raise ValueError(f"Part with name '{name}' already exists")

        axis_tuple: Optional[Tuple[float, float, float]] = None
        if prod_rotation_axis is not None:
            if len(prod_rotation_axis) != 3:
                raise ValueError("prod_rotation_axis must contain exactly three values")
            axis_tuple = tuple(float(component) for component in prod_rotation_axis)

        self.parts.append(
            PartInfo(
                name=name,
                part=shape,
                flip=flip,
                skip_in_production=skip_in_production,
                prod_rotation_angle=prod_rotation_angle,
                prod_rotation_axis=axis_tuple,
            )
        )

    def as_list(self) -> List[Dict[str, object]]:
        return [
            {
                "name": info.name,
                "part": info.part,
                "flip": info.flip,
                "skip_in_production": info.skip_in_production,
                "prod_rotation_angle": info.prod_rotation_angle,
                "prod_rotation_axis": (
                    list(info.prod_rotation_axis)
                    if info.prod_rotation_axis is not None
                    else None
                ),
            }
            for info in self.parts
        ]

    def __iter__(self) -> Iterator[PartInfo]:
        return iter(self.parts)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.parts)

    def __getitem__(self, key: int) -> PartInfo:  # pragma: no cover - trivial
        return self.parts[key]


class LeaderFollowersCuttersPart:
    """Group a leader part with follower, cutter, and non-production parts."""

    def __init__(
        self,
        leader: CadQueryObject,
        followers: Optional[
            Sequence[Union[NamedPart, Mapping[str, object], Tuple[str, CadQueryObject]]]
        ] = None,
        cutters: Optional[
            Sequence[Union[NamedPart, Mapping[str, object], Tuple[str, CadQueryObject]]]
        ] = None,
        non_production_parts: Optional[
            Sequence[Union[NamedPart, Mapping[str, object], Tuple[str, CadQueryObject]]]
        ] = None,
    ) -> None:
        self.leader: Any = _to_shape(leader)
        self.followers: List[NamedPart] = _normalize_named_parts(followers)
        self.cutters: List[NamedPart] = _normalize_named_parts(cutters)
        self.non_production_parts: List[NamedPart] = _normalize_named_parts(
            non_production_parts
        )

    def get_leader_as_part(self) -> Any:
        return self.leader

    def get_non_production_parts_fused(self) -> Optional[Any]:
        if not self.non_production_parts:
            return None
        collector = PartCollector()
        for part in self.non_production_parts:
            collector.fuse(part.part)
        return collector.part

    def leaders_followers_fused(self) -> Any:
        collector = PartCollector()
        collector.fuse(self.leader)
        for follower in self.followers:
            collector.fuse(follower.part)
        assert collector.part is not None
        return collector.part

    def copy(self) -> "LeaderFollowersCuttersPart":
        adapter = get_cad_adapter()
        return LeaderFollowersCuttersPart(
            adapter.copy_part(self.leader),
            [follower.copy() for follower in self.followers],
            [cutter.copy() for cutter in self.cutters],
            [non_prod.copy() for non_prod in self.non_production_parts],
        )

    def fuse(
        self,
        other: Union[CadQueryObject, "LeaderFollowersCuttersPart"],
    ) -> "LeaderFollowersCuttersPart":
        adapter = get_cad_adapter()
        if isinstance(other, LeaderFollowersCuttersPart):
            new_leader = adapter.fuse_parts(self.leader, other.leader)
            new_followers = [f.copy() for f in (self.followers + other.followers)]
            new_cutters = [c.copy() for c in (self.cutters + other.cutters)]
            new_non_prod = [
                n.copy()
                for n in (self.non_production_parts + other.non_production_parts)
            ]
            return LeaderFollowersCuttersPart(
                new_leader, new_followers, new_cutters, new_non_prod
            )

        other_shape = _to_shape(other)
        new_leader = adapter.fuse_parts(self.leader, other_shape)
        return LeaderFollowersCuttersPart(
            new_leader,
            [f.copy() for f in self.followers],
            [c.copy() for c in self.cutters],
            [n.copy() for n in self.non_production_parts],
        )

    def translate(self, vector: Sequence[float]) -> "LeaderFollowersCuttersPart":
        adapter = get_cad_adapter()
        vec = _as_vector(vector)
        self.leader = adapter.translate_part(self.leader, vec)
        self.followers = [follower.translate(vec) for follower in self.followers]
        self.cutters = [cutter.translate(vec) for cutter in self.cutters]
        self.non_production_parts = [
            part.translate(vec) for part in self.non_production_parts
        ]
        return self

    def rotate(
        self,
        angle: float,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        axis: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "LeaderFollowersCuttersPart":
        adapter = get_cad_adapter()
        center_vec = _as_vector(center)
        axis_vec = _as_vector(axis)
        self.leader = adapter.rotate_part(self.leader, angle, center_vec, axis_vec)
        self.followers = [
            follower.rotate(angle, center_vec, axis_vec) for follower in self.followers
        ]
        self.cutters = [
            cutter.rotate(angle, center_vec, axis_vec) for cutter in self.cutters
        ]
        self.non_production_parts = [
            part.rotate(angle, center_vec, axis_vec)
            for part in self.non_production_parts
        ]
        return self

    @property
    def BoundBox(self) -> Any:
        adapter = get_cad_adapter()
        return adapter.get_bounding_box(self.leader)


def export_solid_to_stl(
    solid: Any,
    destination: Union[str, Path],
    *,
    tolerance: float = 0.1,
    angular_tolerance: float = 0.1,
) -> None:
    """Export a CAD solid to an STL file using the appropriate adapter."""
    adapter = get_cad_adapter()
    adapter.export_solid_to_stl(
        solid,
        str(destination),
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _arrange_parts_in_rows(
    parts: Sequence[Any],
    *,
    gap: float,
    bed_width: Optional[float],
) -> List[Any]:
    adapter = get_cad_adapter()
    arranged: List[Any] = []
    x_cursor = 0.0
    y_cursor = 0.0
    row_depth = 0.0

    for shape in parts:
        min_point, max_point = adapter.get_bounding_box(shape)
        width = max_point[0] - min_point[0]
        depth = max_point[1] - min_point[1]

        if bed_width is not None and arranged:
            projected_width = x_cursor + width
            if projected_width > bed_width:
                x_cursor = 0.0
                y_cursor += row_depth + gap
                row_depth = 0.0

        move_vector = (
            x_cursor - min_point[0],
            y_cursor - min_point[1],
            -min_point[2],
        )
        arranged_shape = adapter.translate_part(shape, move_vector)
        arranged.append(arranged_shape)

        x_cursor += width + gap
        row_depth = max(row_depth, depth)

    return arranged


def arrange_and_export_parts(
    parts: Union[Iterable[Mapping[str, object]], PartList],
    prod_gap: float,
    bed_with: float,
    script_file: str,
    export_directory: Optional[Union[str, Path]] = None,
    *,
    prod: bool = False,
    process_data: Optional[MutableMapping[str, object]] = None,
    max_build_height: Optional[float] = None,
) -> Path:
    """Arrange named parts, export individual STLs, and a fused assembly."""

    if isinstance(parts, PartList):
        parts_iterable: Iterable[Mapping[str, object]] = parts.as_list()
    else:
        parts_iterable = parts

    parts_list: List[Dict[str, object]] = [dict(item) for item in parts_iterable]
    if prod:
        parts_list = [p for p in parts_list if not p.get("skip_in_production", False)]
        print("Arranging for production")

    if not parts_list:
        raise ValueError("No parts provided for arrangement and export")

    adapter = get_cad_adapter()
    shapes: List[Any] = []
    names: List[str] = []
    for entry in parts_list:
        if "name" not in entry or "part" not in entry:
            raise KeyError("Each part mapping must include 'name' and 'part'")
        shape = _to_shape(entry["part"])  # type: ignore[arg-type]
        min_point, max_point = adapter.get_bounding_box(shape)
        if (
            prod
            and max_build_height is not None
            and max_point[2] - min_point[2] > max_build_height
        ):
            raise ValueError(
                f"Part {entry['name']} exceeds max_build_height ({max_build_height} mm)"
            )
        shapes.append(shape)
        names.append(str(entry["name"]))

    arranged_shapes = _arrange_parts_in_rows(shapes, gap=prod_gap, bed_width=bed_with)

    export_dir = Path(export_directory) if export_directory is not None else Path.home()
    export_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(script_file).stem or "cadquery_parts"
    fused_collector = PartCollector()

    print("Fusing parts")

    for name, arranged_shape in zip(names, arranged_shapes):
        fused_collector.fuse(arranged_shape)
        part_filename = export_dir / f"{base_name}_{_safe_name(name)}.stl"
        print(f"Exporting {name} to {part_filename}")
        export_solid_to_stl(arranged_shape, part_filename)
        print(f"Exported {name} to {part_filename}")

    fused_shape = fused_collector.part
    assert fused_shape is not None  # fused_collector received at least one part

    assembly_path = export_dir / f"{base_name}.stl"
    export_solid_to_stl(fused_shape, assembly_path)
    print(f"Exported whole part to {assembly_path}")

    if process_data is not None:
        process_data["part_file"] = assembly_path.resolve().as_posix()
        process_filename = assembly_path.with_name(f"{assembly_path.stem}_process.json")
        with process_filename.open("w", encoding="utf-8") as handle:
            json.dump(process_data, handle, indent=4)
        print(f"Exported process data to {process_filename}")

    return assembly_path
