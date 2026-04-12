from shellforgepy.produce.obj_file_export import export_colored_meshes_to_obj
from shellforgepy.render.api import (
    DEFAULT_PREVIEW_VIEWS,
    render_obj_views,
    render_obj_views_with_stats,
)


def test_render_obj_views_uses_default_front_angle_view(tmp_path):
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

    preview_paths = render_obj_views(obj_path, output_dir=tmp_path / "previews")

    assert len(preview_paths) == len(DEFAULT_PREVIEW_VIEWS)
    assert preview_paths[0].stem.endswith("_front_angle")
    assert preview_paths[0].exists()
    assert preview_paths[0].stat().st_size > 0


def test_render_obj_views_writes_multiple_requested_views(tmp_path):
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
    )

    assert [path.stem for path in preview_paths] == ["assembly_top", "assembly_front"]
    assert all(path.exists() for path in preview_paths)


def test_render_obj_views_with_stats_reports_geometry_and_dimensions(tmp_path):
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
    )

    assert batch.triangle_count == 1
    assert batch.vertex_count == 3
    assert batch.object_count == 1
    assert len(batch.results) == len(DEFAULT_PREVIEW_VIEWS)
    assert batch.results[0].width == 512
    assert batch.results[0].height == 512
    assert batch.results[0].render_seconds >= 0.0
