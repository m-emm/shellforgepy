from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from shellforgepy.render.image import preferred_image_suffix, write_image
from shellforgepy.render.obj_loader import load_obj_scene
from shellforgepy.render.presets import DEFAULT_PREVIEW_VIEWS
from shellforgepy.render.rasterizer import render_scene


def _normalize_views(views: Sequence[str] | None) -> tuple[str, ...]:
    if views is None:
        return tuple(DEFAULT_PREVIEW_VIEWS)
    normalized = tuple(str(view).strip() for view in views if str(view).strip())
    return normalized or tuple(DEFAULT_PREVIEW_VIEWS)


@dataclass(frozen=True, slots=True)
class PreviewRenderResult:
    view: str
    path: Path
    width: int
    height: int
    triangle_count: int
    vertex_count: int
    object_count: int
    render_seconds: float


@dataclass(frozen=True, slots=True)
class PreviewRenderBatchResult:
    obj_path: Path
    scene_load_seconds: float
    total_seconds: float
    triangle_count: int
    vertex_count: int
    object_count: int
    results: tuple[PreviewRenderResult, ...]


def render_obj_views_with_stats(
    obj_path: str | Path,
    *,
    output_dir: str | Path,
    views: Sequence[str] | None = None,
    width: int = 512,
    height: int = 512,
    background_color: tuple[int, int, int] = (250, 250, 250),
    filename_prefix: str | None = None,
) -> PreviewRenderBatchResult:
    """Render one or more named views of an OBJ scene and collect timing stats."""

    batch_start = time.perf_counter()
    scene_load_start = batch_start
    scene = load_obj_scene(obj_path)
    scene_load_seconds = time.perf_counter() - scene_load_start

    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = filename_prefix or obj_path.stem
    actual_views = _normalize_views(views)
    suffix = preferred_image_suffix()
    triangle_count = scene.triangle_count()
    vertex_count = scene.vertex_count()
    object_count = scene.object_count()
    results: list[PreviewRenderResult] = []

    for view_name in actual_views:
        render_start = time.perf_counter()
        image = render_scene(
            scene,
            view_name=view_name,
            width=width,
            height=height,
            background_color=background_color,
        )
        destination = output_dir / f"{base_name}_{view_name}{suffix}"
        output_path = write_image(image, destination)
        render_seconds = time.perf_counter() - render_start
        results.append(
            PreviewRenderResult(
                view=view_name,
                path=output_path,
                width=width,
                height=height,
                triangle_count=triangle_count,
                vertex_count=vertex_count,
                object_count=object_count,
                render_seconds=render_seconds,
            )
        )

    total_seconds = time.perf_counter() - batch_start
    return PreviewRenderBatchResult(
        obj_path=obj_path,
        scene_load_seconds=scene_load_seconds,
        total_seconds=total_seconds,
        triangle_count=triangle_count,
        vertex_count=vertex_count,
        object_count=object_count,
        results=tuple(results),
    )


def render_obj_views(
    obj_path: str | Path,
    *,
    output_dir: str | Path,
    views: Sequence[str] | None = None,
    width: int = 512,
    height: int = 512,
    background_color: tuple[int, int, int] = (250, 250, 250),
    filename_prefix: str | None = None,
) -> list[Path]:
    """Render one or more named views of an OBJ scene."""

    batch = render_obj_views_with_stats(
        obj_path,
        output_dir=output_dir,
        views=views,
        width=width,
        height=height,
        background_color=background_color,
        filename_prefix=filename_prefix,
    )
    return [result.path for result in batch.results]


__all__ = [
    "DEFAULT_PREVIEW_VIEWS",
    "PreviewRenderBatchResult",
    "PreviewRenderResult",
    "render_obj_views",
    "render_obj_views_with_stats",
]
