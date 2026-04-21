from __future__ import annotations

import numpy as np

DEFAULT_PREVIEW_VIEWS = ("front_angle",)


def view_direction_for_name(name: str) -> np.ndarray:
    """Return the normalized camera direction for a named orthographic view."""

    directions = {
        "top": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "bottom": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "front": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "back": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "isometric": np.array([1.0, -1.0, 1.0], dtype=np.float32),
        "front_angle": np.array([0.5, -1.0, 0.35], dtype=np.float32),
    }
    try:
        direction = directions[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown preview view: {name} - legal views are: {', '.join(directions.keys())}"
        ) from exc
    return direction / (np.linalg.norm(direction) + 1e-12)
