"""Utilities to upload G-code files to a printer."""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 4409
HEADER_END_MARKER = "; HEADER_BLOCK_END"
THUMBNAIL_MARKER = "; THUMBNAIL_BLOCK_START"


class UploadError(RuntimeError):
    """Raised when an upload or preprocessing step fails."""


@dataclass
class PrinterTarget:
    host: str
    port: int = DEFAULT_PORT

    def as_tuple(self) -> Tuple[str, int]:
        return self.host, self.port


def parse_host_port(host_string: str) -> PrinterTarget:
    """Parse a host string of the form 'host[:port]' into a PrinterTarget."""

    if ":" in host_string:
        host, port = host_string.split(":", 1)
        return PrinterTarget(host.strip(), int(port.strip()))
    return PrinterTarget(host_string.strip(), DEFAULT_PORT)


def read_print_host(gcode_path: Path) -> Optional[PrinterTarget]:
    """Read printer host information from a neighbouring print_host.txt file."""

    candidate = gcode_path.parent / "print_host.txt"
    if not candidate.is_file():
        return None
    content = candidate.read_text(encoding="utf-8").strip()
    if not content:
        return None
    try:
        return parse_host_port(content)
    except ValueError as exc:  # pragma: no cover - defensive coding
        LOGGER.warning("Failed to parse print_host.txt: %s", exc)
        return None


def _wrap_base64(data: str, line_length: int = 78) -> Iterable[str]:
    return (data[i : i + line_length] for i in range(0, len(data), line_length))


def embed_preview_image(
    gcode_path: Path,
    image_path: Path,
    resolutions: Iterable[Tuple[int, int]] = ((96, 96), (300, 300)),
) -> None:
    """Embed thumbnail previews into a G-code file if an image is available."""

    from PIL import Image

    if not image_path.is_file():
        LOGGER.debug("No preview image at %s; skipping thumbnail embed.", image_path)
        return

    try:
        gcode_lines = gcode_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:  # pragma: no cover - defensive
        raise UploadError(f"Failed to read G-code file {gcode_path}: {exc}") from exc

    insert_index = (
        next(
            (
                idx
                for idx, line in enumerate(gcode_lines)
                if line.strip() == HEADER_END_MARKER
            ),
            0,
        )
        + 1
    )

    blocks: list[str] = []
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            for width, height in resolutions:
                thumb = img.copy()
                thumb.thumbnail((width, height))
                buffer = BytesIO()
                thumb.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                block_lines = list(_wrap_base64(encoded))

                blocks.extend(
                    [
                        f"{THUMBNAIL_MARKER}\n",
                        ";\n",
                        f"; thumbnail begin {width}x{height} {len(encoded)}\n",
                        *[f"; {line}\n" for line in block_lines],
                        "; thumbnail end\n",
                        "; THUMBNAIL_BLOCK_END\n\n",
                    ]
                )

        patched_lines = gcode_lines[:insert_index] + blocks + gcode_lines[insert_index:]
        gcode_path.write_text("".join(patched_lines), encoding="utf-8")
        LOGGER.info(
            "Embedded thumbnails (%s) into %s",
            ", ".join(f"{w}x{h}" for w, h in resolutions),
            gcode_path,
        )
    except Exception as exc:  # pragma: no cover - image processing best effort
        LOGGER.warning("Failed to embed thumbnails into %s: %s", gcode_path, exc)


def parse_gcode_metadata(gcode_path: Path) -> Dict[str, Optional[float | str]]:
    metadata: Dict[str, Optional[float | str]] = {
        "model_name": None,
        "estimated_time": None,
        "estimated_seconds": None,
        "filament_used_mm": None,
        "object_height_mm": None,
    }

    with gcode_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()

            if (
                line.startswith("EXCLUDE_OBJECT_DEFINE NAME=")
                and metadata["model_name"] is None
            ):
                match = re.search(r"NAME=([^\s]+)", line)
                if match:
                    raw_name = match.group(1)
                    metadata["model_name"] = raw_name.split(".stl")[0]

            elif (
                "estimated printing time" in line and metadata["estimated_time"] is None
            ):
                parts = line.split("=")
                if len(parts) >= 2:
                    raw_time = parts[1].strip()
                    metadata["estimated_time"] = raw_time.replace(" ", "")
                    time_match = re.search(r"(\d+)h\s*(\d+)m\s*(\d+)s", raw_time)
                    if time_match:
                        hours, minutes, seconds = (
                            int(time_match.group(1)),
                            int(time_match.group(2)),
                            int(time_match.group(3)),
                        )
                        metadata["estimated_seconds"] = (
                            hours * 3600 + minutes * 60 + seconds
                        )

            elif "filament used [mm]" in line and metadata["filament_used_mm"] is None:
                match = re.search(r"([\d\.]+)", line)
                if match:
                    metadata["filament_used_mm"] = float(match.group(1))

            elif "max_z_height" in line and metadata["object_height_mm"] is None:
                match = re.search(r"([\d\.]+)", line)
                if match:
                    metadata["object_height_mm"] = float(match.group(1))

    return metadata


def patch_generated_by_line(gcode_path: Path) -> None:
    lines = gcode_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    modified = False
    new_lines = []
    for line in lines:
        if line.lower().startswith("; generated by orcaslicer"):
            line = line.replace("OrcaSlicer", "PrusaSlicer")
            modified = True
        new_lines.append(line)

    if modified:
        stat = gcode_path.stat()
        gcode_path.write_text("".join(new_lines), encoding="utf-8")
        os.utime(gcode_path, (stat.st_atime, stat.st_mtime))
        LOGGER.info("Patched 'generated by' line to PrusaSlicer in %s", gcode_path)


def patch_gcode_for_moonraker(
    gcode_path: Path, metadata: Dict[str, Optional[float | str]]
) -> None:
    original_lines = gcode_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )

    meta_lines = []
    if metadata.get("estimated_seconds"):
        meta_lines.append(f";TIME:{int(metadata['estimated_seconds'])}\n")
    if metadata.get("filament_used_mm"):
        meta_lines.append(f";FILAMENT_USED={float(metadata['filament_used_mm']):.2f}\n")
    if metadata.get("object_height_mm"):
        meta_lines.append(f";HEIGHT:{float(metadata['object_height_mm']):.2f}\n")

    if not meta_lines:
        LOGGER.debug("No Moonraker metadata discovered for %s", gcode_path)
        return

    patched_lines = []
    inserted = False
    for line in original_lines:
        if not inserted and line.strip() == HEADER_END_MARKER:
            patched_lines.extend(meta_lines)
            inserted = True
        patched_lines.append(line)

    if not inserted:
        patched_lines = meta_lines + original_lines

    stat = gcode_path.stat()
    gcode_path.write_text("".join(patched_lines), encoding="utf-8")
    os.utime(gcode_path, (stat.st_atime, stat.st_mtime))
    LOGGER.info("Inserted Moonraker metadata into %s", gcode_path)


def upload_gcode_file(
    gcode_path: Path, target: PrinterTarget, remote_filename: str
) -> None:
    url = f"http://{target.host}:{target.port}/server/files/upload"

    import requests
    from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
    from tqdm import tqdm

    with gcode_path.open("rb") as handle:

        encoder = MultipartEncoder(
            fields={"file": (remote_filename, handle, "application/octet-stream")}
        )
        progress = tqdm(total=encoder.len, unit="B", unit_scale=True, desc="Uploading")

        def monitor_callback(monitor: MultipartEncoderMonitor) -> None:
            progress.update(monitor.bytes_read - progress.n)

        monitor = MultipartEncoderMonitor(encoder, monitor_callback)

        try:
            response = requests.post(
                url,
                data=monitor,
                headers={"Content-Type": monitor.content_type},
                timeout=300,
            )
        finally:
            progress.close()

    if response.status_code not in {200, 201}:
        raise UploadError(
            f"Upload failed with status {response.status_code}: {response.text}"
        )

    LOGGER.info("Upload successful: %s -> %s", remote_filename, response.text)


def upload_to_printer(gcode_file: Path | str, printer: Optional[str] = None) -> None:
    """Prepare a G-code file and upload it to the printer."""

    gcode_path = Path(gcode_file).expanduser()
    if not gcode_path.is_file():
        raise UploadError(f"G-code file does not exist: {gcode_path}")

    if printer:
        target = parse_host_port(printer)
    else:
        target = read_print_host(gcode_path)
        if target is None:
            raise UploadError(
                "Printer host not provided and no print_host.txt found near the G-code file."
            )

    metadata = parse_gcode_metadata(gcode_path)
    model_name = metadata.get("model_name") or gcode_path.stem
    estimated_time = metadata.get("estimated_time") or "unknown"

    patch_generated_by_line(gcode_path)
    patch_gcode_for_moonraker(gcode_path, metadata)

    preview_path = gcode_path.with_name(f"{gcode_path.stem}_preview.png")
    embed_preview_image(gcode_path, preview_path)

    remote_filename = f"{model_name}_{estimated_time}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gcode"

    LOGGER.info(
        "Uploading %s as %s to %s:%s",
        gcode_path,
        remote_filename,
        target.host,
        target.port,
    )

    upload_gcode_file(gcode_path, target, remote_filename)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Upload a G-code file to a printer")
    parser.add_argument("gcode", help="Path to the G-code file")
    parser.add_argument(
        "--printer",
        help="Printer host (ip[:port]); falls back to print_host.txt if omitted",
    )
    args = parser.parse_args()
    upload_to_printer(args.gcode, args.printer)
