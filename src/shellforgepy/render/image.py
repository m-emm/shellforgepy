from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from PIL import Image

    HAVE_PIL = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PIL = False


def preferred_image_suffix() -> str:
    """Return the preferred output extension for rendered previews."""

    return ".png" if HAVE_PIL else ".ppm"


def write_image(image: np.ndarray, destination: str | Path) -> Path:
    """Write an RGB image to PNG when Pillow is available, otherwise PPM."""

    image_array = np.asarray(image, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Rendered image must have shape (height, width, 3)")

    requested_path = Path(destination)
    actual_path = requested_path
    if requested_path.suffix.lower() == ".png" and not HAVE_PIL:
        actual_path = requested_path.with_suffix(".ppm")

    actual_path.parent.mkdir(parents=True, exist_ok=True)

    if actual_path.suffix.lower() == ".png":
        if not HAVE_PIL:
            raise RuntimeError("Pillow is required to write PNG files")
        Image.fromarray(image_array).save(actual_path)
        return actual_path

    with actual_path.open("wb") as handle:
        header = f"P6\n{image_array.shape[1]} {image_array.shape[0]}\n255\n"
        handle.write(header.encode("ascii"))
        handle.write(image_array.tobytes())
    return actual_path
