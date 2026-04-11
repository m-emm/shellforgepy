from __future__ import annotations

import math

import numpy as np
from shellforgepy.render.model import Scene
from shellforgepy.render.presets import view_direction_for_name


def _pick_stable_up(direction: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(direction, up)) > 0.95:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return up


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _ortho_matrix(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    proj = np.eye(4, dtype=np.float32)
    proj[0, 0] = 2.0 / max(right - left, 1e-6)
    proj[1, 1] = 2.0 / max(top - bottom, 1e-6)
    proj[2, 2] = -2.0 / max(far - near, 1e-6)
    proj[0, 3] = -(right + left) / max(right - left, 1e-6)
    proj[1, 3] = -(top + bottom) / max(top - bottom, 1e-6)
    proj[2, 3] = -(far + near) / max(far - near, 1e-6)
    return proj


def _scene_triangles(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    triangles = []
    colors = []

    for obj in scene.objects:
        object_vertices = np.asarray(obj.vertices, dtype=np.float32)
        object_faces = np.asarray(obj.faces, dtype=np.int32)
        if len(object_faces) == 0:
            continue
        triangles.append(object_vertices[object_faces])
        colors.append(
            np.repeat(
                np.asarray([obj.color], dtype=np.float32), len(object_faces), axis=0
            )
        )

    if not triangles:
        return (
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    return np.concatenate(triangles, axis=0), np.concatenate(colors, axis=0)


def _camera_mvp_for_scene(
    scene: Scene,
    view_name: str,
    *,
    margin_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min, bounds_max = scene.bounds()
    center = 0.5 * (bounds_min + bounds_max)
    extent = np.maximum(bounds_max - bounds_min, 1e-3)
    radius = float(np.linalg.norm(extent))
    direction = view_direction_for_name(view_name)
    eye = center - direction * max(radius * 2.5, 1.0)
    up = _pick_stable_up(direction)
    view = _look_at(eye, center, up)

    corners = np.array(
        [
            [x, y, z]
            for x in (bounds_min[0], bounds_max[0])
            for y in (bounds_min[1], bounds_max[1])
            for z in (bounds_min[2], bounds_max[2])
        ],
        dtype=np.float32,
    )
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1), dtype=np.float32)])
    view_space = (view @ corners_h.T).T[:, :3]
    view_min = view_space.min(axis=0)
    view_max = view_space.max(axis=0)
    center_view = 0.5 * (view_min + view_max)
    half_extent = 0.5 * (view_max - view_min)
    half_extent *= 1.0 + margin_ratio
    half_extent = np.maximum(half_extent, 1e-3)
    view_min = center_view - half_extent
    view_max = center_view + half_extent

    proj = _ortho_matrix(
        float(view_min[0]),
        float(view_max[0]),
        float(view_min[1]),
        float(view_max[1]),
        float(view_min[2]) - radius,
        float(view_max[2]) + radius,
    )
    return proj @ view, view


def _ndc_to_screen(triangles_ndc: np.ndarray, width: int, height: int) -> np.ndarray:
    x = (triangles_ndc[..., 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (triangles_ndc[..., 1] * 0.5 + 0.5)) * (height - 1)
    z = triangles_ndc[..., 2]
    return np.stack([x, y, z], axis=-1)


def _face_normals(triangles: np.ndarray) -> np.ndarray:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals.astype(np.float32)


def _shade_colors(face_colors: np.ndarray, face_normals_view: np.ndarray) -> np.ndarray:
    light_dir = np.array([0.35, -0.45, -1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-12
    diffuse = np.maximum(0.0, face_normals_view @ -light_dir)
    intensity = 0.45 + 0.55 * diffuse
    shaded = face_colors * intensity[:, None]
    return np.clip(shaded, 0.0, 1.0)


def _rasterize_triangles(
    screen_triangles: np.ndarray,
    face_colors: np.ndarray,
    width: int,
    height: int,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    depth = np.full((height, width), np.inf, dtype=np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = np.asarray(background_color, dtype=np.uint8)

    for triangle, color in zip(screen_triangles, face_colors):
        x = triangle[:, 0]
        y = triangle[:, 1]
        z = triangle[:, 2]

        min_x = max(int(math.floor(float(np.min(x)))), 0)
        max_x = min(int(math.ceil(float(np.max(x)))), width - 1)
        min_y = max(int(math.floor(float(np.min(y)))), 0)
        max_y = min(int(math.ceil(float(np.max(y)))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        denom = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(float(denom)) < 1e-12:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        w0 = ((y[1] - y[2]) * (xx - x[2]) + (x[2] - x[1]) * (yy - y[2])) / denom
        w1 = ((y[2] - y[0]) * (xx - x[2]) + (x[0] - x[2]) * (yy - y[2])) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(mask):
            continue

        z_box = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        closer = (z_box < sub_depth) & mask
        if not np.any(closer):
            continue

        sub_depth[closer] = z_box[closer]
        rgb[min_y : max_y + 1, min_x : max_x + 1][closer] = np.clip(
            color * 255.0 + 0.5, 0, 255
        ).astype(np.uint8)

    return rgb


def render_scene(
    scene: Scene,
    *,
    view_name: str,
    width: int = 1024,
    height: int = 1024,
    background_color: tuple[int, int, int] = (250, 250, 250),
) -> np.ndarray:
    """Render a mesh scene to an RGB numpy image."""

    triangles, face_colors = _scene_triangles(scene)
    if len(triangles) == 0:
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = np.asarray(background_color, dtype=np.uint8)
        return image

    mvp, view = _camera_mvp_for_scene(scene, view_name)
    triangles_h = np.concatenate(
        [triangles, np.ones((triangles.shape[0], 3, 1), dtype=np.float32)], axis=2
    )
    triangles_clip = (mvp @ triangles_h.transpose(0, 2, 1)).transpose(0, 2, 1)
    triangles_ndc = triangles_clip[..., :3] / np.clip(
        triangles_clip[..., 3:4], 1e-12, None
    )
    screen_triangles = _ndc_to_screen(triangles_ndc, width, height)

    view_rotation = view[:3, :3]
    normals_view = (_face_normals(triangles) @ view_rotation.T).astype(np.float32)
    normals_view /= np.linalg.norm(normals_view, axis=1, keepdims=True) + 1e-12
    shaded_colors = _shade_colors(face_colors, normals_view)

    return _rasterize_triangles(
        screen_triangles,
        shaded_colors,
        width,
        height,
        background_color,
    )
