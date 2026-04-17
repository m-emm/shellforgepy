import logging
import os
import time
from math import ceil, sqrt

import numpy as np
import pytest
from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj
from shellforgepy.render.api import (
    DEFAULT_PREVIEW_VIEWS,
    render_obj_view_to_image_with_stats,
    render_obj_views,
    render_obj_views_with_stats,
)
from shellforgepy.render.image import preferred_image_suffix, write_image
from shellforgepy.render.model import MeshObject, Scene
from shellforgepy.render.rasterizer import (
    numba_renderer_available,
    render_backend_name,
    render_scene,
)

_logger = logging.getLogger(__name__)

_RUN_RENDER_PERF_ENV = "SHELLFORGEPY_RUN_RENDER_PERF"
_RENDER_PERF_CUBE_COUNTS = (100, 500, 2000)
_RENDER_PERF_RESOLUTIONS = (256, 512, 1024)


def _render_backend_params() -> list:
    params = [pytest.param(True, id="numpy")]
    if numba_renderer_available():
        params.append(pytest.param(False, id="numba"))
    return params


@pytest.fixture(params=_render_backend_params())
def disable_numba(request) -> bool:
    return bool(request.param)


def _create_cube_mesh_object(
    *,
    name: str,
    origin: tuple[float, float, float],
    size: float,
    color: tuple[float, float, float],
) -> MeshObject:
    ox, oy, oz = origin
    vertices = np.asarray(
        [
            (ox, oy, oz),
            (ox + size, oy, oz),
            (ox + size, oy + size, oz),
            (ox, oy + size, oz),
            (ox, oy, oz + size),
            (ox + size, oy, oz + size),
            (ox + size, oy + size, oz + size),
            (ox, oy + size, oz + size),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    return MeshObject(name=name, vertices=vertices, faces=faces, color=color)


def _build_staggered_cube_scene(
    *,
    cube_count: int = 500,
    columns: int | None = None,
    cube_size: float = 8.0,
    gap: float = 2.0,
) -> Scene:
    if columns is None:
        columns = max(1, ceil(sqrt(cube_count)))
    pitch = cube_size + gap
    objects: list[MeshObject] = []

    for cube_index in range(cube_count):
        row = cube_index // columns
        column = cube_index % columns
        x_offset = column * pitch + (pitch * 0.5 if row % 2 else 0.0)
        y_offset = row * pitch
        z_offset = float((cube_index % 7) * 1.5 + (row % 3) * 0.75)
        color = (
            0.25 + 0.55 * ((column % 5) / 4.0),
            0.30 + 0.45 * ((row % 6) / 5.0),
            0.35 + 0.40 * ((cube_index % 9) / 8.0),
        )
        objects.append(
            _create_cube_mesh_object(
                name=f"cube_{cube_index:03d}",
                origin=(x_offset, y_offset, z_offset),
                size=cube_size,
                color=color,
            )
        )

    return Scene(objects)


def _dominant_channel_centroid_x(image: np.ndarray, channel_index: int) -> float:
    positions = _dominant_channel_positions(image, channel_index)
    return float(np.mean(positions[:, 1]))


def _dominant_channel_positions(image: np.ndarray, channel_index: int) -> np.ndarray:
    other_channels = [index for index in range(3) if index != channel_index]
    channel = image[:, :, channel_index].astype(np.int16)
    mask = (
        (channel >= 20)
        & (channel > image[:, :, other_channels[0]].astype(np.int16) + 20)
        & (channel > image[:, :, other_channels[1]].astype(np.int16) + 20)
    )
    positions = np.argwhere(mask)
    assert len(positions) > 0
    return positions


def _dominant_channel_bbox_size(
    image: np.ndarray, channel_index: int
) -> tuple[float, float]:
    positions = _dominant_channel_positions(image, channel_index)
    min_y, min_x = positions.min(axis=0)
    max_y, max_x = positions.max(axis=0)
    return float(max_x - min_x + 1), float(max_y - min_y + 1)


def _run_render_perf_case(
    *,
    cube_count: int,
    resolution: int,
    tmp_path,
    disable_numba: bool,
    background_color: tuple[int, int, int] = (250, 250, 250),
) -> dict[str, int | float | str]:
    total_start = time.perf_counter()
    backend_name = render_backend_name(disable_numba=disable_numba)

    scene_build_start = time.perf_counter()
    scene = _build_staggered_cube_scene(cube_count=cube_count)
    scene_build_seconds = time.perf_counter() - scene_build_start
    bounds_min, bounds_max = scene.bounds()
    triangle_count = scene.triangle_count()
    vertex_count = scene.vertex_count()
    object_count = scene.object_count()

    _logger.info(
        "Render perf scene built: backend=%s objects=%d vertices=%d triangles=%d "
        "bounds_min=%s bounds_max=%s build_seconds=%.4f",
        backend_name,
        object_count,
        vertex_count,
        triangle_count,
        bounds_min.tolist(),
        bounds_max.tolist(),
        scene_build_seconds,
    )

    render_start = time.perf_counter()
    image = render_scene(
        scene,
        view_name="front_angle",
        width=resolution,
        height=resolution,
        background_color=background_color,
        disable_numba=disable_numba,
    )
    render_seconds = time.perf_counter() - render_start

    background = np.asarray(background_color, dtype=np.uint8)
    non_background_pixels = int(np.count_nonzero(np.any(image != background, axis=2)))
    coverage = non_background_pixels / float(resolution * resolution)
    megapixels_per_second = (
        (resolution * resolution) / max(render_seconds, 1e-12) / 1_000_000.0
    )

    _logger.info(
        "Render perf rasterized: cubes=%d view=%s size=%dx%d "
        "backend=%s non_background_pixels=%d coverage=%.4f render_seconds=%.4f "
        "megapixels_per_second=%.2f",
        cube_count,
        "front_angle",
        resolution,
        resolution,
        backend_name,
        non_background_pixels,
        coverage,
        render_seconds,
        megapixels_per_second,
    )

    output_path = write_image(
        image,
        tmp_path
        / f"staggered_cube_perf_{cube_count}_{resolution}{preferred_image_suffix()}",
    )
    total_seconds = time.perf_counter() - total_start

    _logger.info(
        "Render perf artifact written: cubes=%d size=%dx%d path=%s bytes=%d "
        "backend=%s total_seconds=%.4f",
        cube_count,
        resolution,
        resolution,
        output_path,
        output_path.stat().st_size,
        backend_name,
        total_seconds,
    )

    assert object_count == cube_count
    assert triangle_count == cube_count * 12
    assert vertex_count == cube_count * 8
    assert image.shape == (resolution, resolution, 3)
    assert non_background_pixels > 0
    assert output_path.exists()

    return {
        "backend": backend_name,
        "cube_count": cube_count,
        "resolution": resolution,
        "build_ms": int(round(scene_build_seconds * 1000)),
        "render_ms": int(round(render_seconds * 1000)),
        "total_ms": int(round(total_seconds * 1000)),
        "triangle_count": triangle_count,
        "vertex_count": vertex_count,
        "coverage_percent": coverage * 100.0,
        "output_path": str(output_path),
    }


def _format_render_perf_table(rows: list[dict[str, int | float | str]]) -> str:
    lines = [
        "| backend | cubes | resolution | build_ms | render_ms | total_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['cube_count']} | {row['resolution']} | "
            f"{row['build_ms']} | {row['render_ms']} | {row['total_ms']} |"
        )
    return "\n".join(lines)


def _format_render_perf_comparison_table(
    numpy_rows: list[dict[str, int | float | str]],
    numba_rows: list[dict[str, int | float | str]],
) -> str:
    numba_by_case = {
        (int(row["cube_count"]), int(row["resolution"])): row for row in numba_rows
    }
    lines = [
        "| cubes | resolution | numpy_render_ms | numba_render_ms | speedup_x | delta_ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for numpy_row in numpy_rows:
        key = (int(numpy_row["cube_count"]), int(numpy_row["resolution"]))
        numba_row = numba_by_case[key]
        numpy_render_ms = int(numpy_row["render_ms"])
        numba_render_ms = int(numba_row["render_ms"])
        speedup = numpy_render_ms / max(numba_render_ms, 1)
        delta_ms = numpy_render_ms - numba_render_ms
        lines.append(
            f"| {key[0]} | {key[1]} | {numpy_render_ms} | {numba_render_ms} | "
            f"{speedup:.2f} | {delta_ms} |"
        )
    return "\n".join(lines)


def _warm_numba_renderer() -> None:
    warmup_scene = _build_staggered_cube_scene(cube_count=2)
    render_scene(
        warmup_scene,
        view_name="front_angle",
        width=32,
        height=32,
        disable_numba=False,
    )


def test_render_obj_views_uses_default_front_angle_view(tmp_path, disable_numba):
    obj_path = tmp_path / "box.obj"
    vertices = [
        (0.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (20.0, 20.0, 0.0),
        (0.0, 20.0, 0.0),
        (0.0, 0.0, 10.0),
        (20.0, 0.0, 10.0),
        (20.0, 20.0, 10.0),
        (0.0, 20.0, 10.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    export_colored_meshes_to_obj(
        [(vertices, faces, "box", (0.8, 0.2, 0.1))],
        obj_path,
    )

    preview_paths = render_obj_views(
        obj_path,
        output_dir=tmp_path / "previews",
        disable_numba=disable_numba,
    )

    assert len(preview_paths) == len(DEFAULT_PREVIEW_VIEWS)
    assert preview_paths[0].stem.endswith("_front_angle")
    assert preview_paths[0].exists()
    assert preview_paths[0].stat().st_size > 0


def test_render_obj_views_writes_multiple_requested_views(tmp_path, disable_numba):
    obj_path = tmp_path / "assembly.obj"
    vertices = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
    ]
    faces = [(0, 1, 2)]
    export_colored_meshes_to_obj(
        [(vertices, faces, "triangle", (0.2, 0.6, 0.9))],
        obj_path,
    )

    preview_paths = render_obj_views(
        obj_path,
        output_dir=tmp_path / "views",
        views=["top", "front"],
        disable_numba=disable_numba,
    )

    assert [path.stem for path in preview_paths] == ["assembly_top", "assembly_front"]
    assert all(path.exists() for path in preview_paths)


def test_render_obj_views_with_stats_reports_geometry_and_dimensions(
    tmp_path, disable_numba
):
    obj_path = tmp_path / "stats.obj"
    vertices = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)]
    faces = [(0, 1, 2)]
    export_colored_meshes_to_obj(
        [(vertices, faces, "triangle", (0.9, 0.3, 0.2))],
        obj_path,
    )

    batch = render_obj_views_with_stats(
        obj_path,
        output_dir=tmp_path / "stats_previews",
        width=512,
        height=512,
        disable_numba=disable_numba,
    )

    assert batch.triangle_count == 1
    assert batch.vertex_count == 3
    assert batch.object_count == 1
    assert len(batch.results) == len(DEFAULT_PREVIEW_VIEWS)
    assert batch.results[0].width == 512
    assert batch.results[0].height == 512
    assert batch.results[0].render_seconds >= 0.0


@pytest.mark.parametrize("view_name", ["front", "isometric"])
def test_render_scene_preserves_left_to_right_order(view_name, disable_numba):
    scene = Scene(
        [
            _create_cube_mesh_object(
                name="left_cube",
                origin=(0.0, 0.0, 0.0),
                size=10.0,
                color=(0.95, 0.10, 0.10),
            ),
            _create_cube_mesh_object(
                name="right_cube",
                origin=(20.0, 0.0, 0.0),
                size=10.0,
                color=(0.10, 0.10, 0.95),
            ),
        ]
    )

    image = render_scene(
        scene,
        view_name=view_name,
        width=256,
        height=256,
        disable_numba=disable_numba,
    )

    red_centroid_x = _dominant_channel_centroid_x(image, 0)
    blue_centroid_x = _dominant_channel_centroid_x(image, 2)

    assert red_centroid_x < blue_centroid_x


def test_render_scene_preserves_cube_proportions_in_wide_front_view(disable_numba):
    scene = Scene(
        [
            _create_cube_mesh_object(
                name="reference_cube",
                origin=(0.0, 0.0, 0.0),
                size=10.0,
                color=(0.95, 0.10, 0.10),
            ),
            _create_cube_mesh_object(
                name="far_anchor",
                origin=(90.0, 0.0, 0.0),
                size=2.0,
                color=(0.10, 0.10, 0.95),
            ),
        ]
    )

    image = render_scene(
        scene,
        view_name="front",
        width=320,
        height=320,
        disable_numba=disable_numba,
    )

    cube_width, cube_height = _dominant_channel_bbox_size(image, 0)

    assert cube_height / cube_width == pytest.approx(1.0, rel=0.15)


def test_render_obj_view_to_image_with_stats_can_exclude_named_objects(
    tmp_path, disable_numba
):
    obj_path = tmp_path / "filtered.obj"
    vertices = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)]
    faces = [(0, 1, 2)]
    export_colored_meshes_to_obj(
        [
            (vertices, faces, "triangle", (0.9, 0.3, 0.2)),
            (vertices, faces, "plate_boundary_plate_a", (0.8, 0.8, 0.8)),
        ],
        obj_path,
    )

    result = render_obj_view_to_image_with_stats(
        obj_path,
        destination=tmp_path / "filtered.png",
        disable_numba=disable_numba,
        exclude_object_name_prefixes=("plate_boundary_",),
    )

    assert result.object_count == 1
    assert result.triangle_count == 1
    assert result.vertex_count == 3
    assert result.path.exists()


def test_render_scene_prefers_numba_when_available():
    expected_backend = "numba" if numba_renderer_available() else "numpy"

    assert render_backend_name() == expected_backend


@pytest.mark.skipif(
    not numba_renderer_available(),
    reason="numba renderer backend is not available",
)
def test_render_scene_disable_numba_matches_numba_output():
    scene = _build_staggered_cube_scene(cube_count=12)

    numpy_image = render_scene(
        scene,
        view_name="front_angle",
        width=256,
        height=256,
        disable_numba=True,
    )
    numba_image = render_scene(
        scene,
        view_name="front_angle",
        width=256,
        height=256,
        disable_numba=False,
    )

    assert render_backend_name(disable_numba=True) == "numpy"
    assert render_backend_name(disable_numba=False) == "numba"
    assert np.array_equal(numba_image, numpy_image)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_RUN_RENDER_PERF_ENV) != "1",
    reason=f"Set {_RUN_RENDER_PERF_ENV}=1 to run render performance logging.",
)
def test_render_scene_logs_performance_matrix_for_staggered_cubes(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    numpy_rows: list[dict[str, int | float | str]] = []
    numba_rows: list[dict[str, int | float | str]] = []

    for cube_count in _RENDER_PERF_CUBE_COUNTS:
        for resolution in _RENDER_PERF_RESOLUTIONS:
            numpy_rows.append(
                _run_render_perf_case(
                    cube_count=cube_count,
                    resolution=resolution,
                    tmp_path=tmp_path,
                    disable_numba=True,
                )
            )

    if numba_renderer_available():
        _logger.info("Render perf numba warmup starting")
        _warm_numba_renderer()
        _logger.info("Render perf numba warmup finished")

        for cube_count in _RENDER_PERF_CUBE_COUNTS:
            for resolution in _RENDER_PERF_RESOLUTIONS:
                numba_rows.append(
                    _run_render_perf_case(
                        cube_count=cube_count,
                        resolution=resolution,
                        tmp_path=tmp_path,
                        disable_numba=False,
                    )
                )

    _logger.info(
        "Render perf numpy results:\n%s", _format_render_perf_table(numpy_rows)
    )
    if numba_rows:
        _logger.info(
            "Render perf numba results:\n%s", _format_render_perf_table(numba_rows)
        )
        _logger.info(
            "Render perf comparison results:\n%s",
            _format_render_perf_comparison_table(numpy_rows, numba_rows),
        )

    assert len(numpy_rows) == len(_RENDER_PERF_CUBE_COUNTS) * len(
        _RENDER_PERF_RESOLUTIONS
    )
    if numba_renderer_available():
        assert len(numba_rows) == len(numpy_rows)
        assert "Render perf numba warmup finished" in caplog.text
        assert "Render perf numba results" in caplog.text
        assert "Render perf comparison results" in caplog.text
    assert "Render perf scene built" in caplog.text
    assert "Render perf rasterized" in caplog.text
    assert "Render perf numpy results" in caplog.text
