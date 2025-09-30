import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)


def generate_settings(
    process_data_file: Path, output_dir: Path, master_settings_dir: Path
) -> None:
    """Generate OrcaSlicer settings into ``output_dir``.

    Raises:
        FileNotFoundError: If any required input file or directory is missing
        ValueError: If the filament specified in process_data_file is not found
    """
    if not process_data_file.exists():
        raise FileNotFoundError(f"File {process_data_file} does not exist.")

    with process_data_file.open("r", encoding="utf-8") as file_handle:
        process_data = json.load(file_handle)

    _logger.info(
        f"Loaded {process_data_file}, data:\n{json.dumps(process_data, indent=2)}"
    )

    filament = process_data["filament"]

    if not output_dir.exists():
        raise FileNotFoundError(f"Directory {output_dir} does not exist.")

    if not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a directory.")

    (output_dir / "filaments").mkdir(parents=True, exist_ok=True)

    if not master_settings_dir.exists():
        raise FileNotFoundError(f"Directory {master_settings_dir} does not exist.")

    filament_found = False
    for config_file in master_settings_dir.glob("*.yaml"):
        with config_file.open("r", encoding="utf-8") as file_handle:
            master_data = yaml.safe_load(file_handle)

        name = master_data["name"]

        path_part = ""
        if master_data["type"] == "filament":
            path_part = "filaments/"

            if master_data["name"] != filament:
                _logger.info(
                    f"Skipping {name} config, not matching filament name {filament}"
                )
                continue
            filament_found = True

        if master_data["type"] == "machine":
            print_host = master_data.get("print_host")
            if print_host is not None:
                with (output_dir / "print_host.txt").open(
                    "w", encoding="utf-8"
                ) as file_handle:
                    file_handle.write(print_host)
                _logger.info(f"Saved print host to {output_dir / 'print_host.txt'}")

        for key, value in process_data["process_overrides"].items():
            if key in master_data:
                _logger.info(f"Overriding {key} with {value}")
                master_data[key] = str(value)

        with (output_dir / f"{path_part}{name}.json").open(
            "w", encoding="utf-8"
        ) as file_handle:
            json.dump(master_data, file_handle, indent=2)
        _logger.info(f"Saved {name} config to {path_part}{name}.json")

        info_text = (
            " "
            "\nsync_info = update"
            "\nuser_id = "
            "\nsetting_id = "
            "\nbase_id ="
            "\nupdated_time = 1713556125\n"
        )

        with (output_dir / f"{path_part}{name}.info").open(
            "w", encoding="utf-8"
        ) as file_handle:
            file_handle.write(info_text)
        _logger.info(f"Saved {name} config to {path_part}{name}.info")

    if not filament_found:
        raise ValueError(f"Filament {filament} not found in {master_settings_dir}.")

    part_path = Path(process_data["part_file"]).expanduser()
    if not part_path.exists():
        raise FileNotFoundError(f"File {part_path} does not exist.")

    destination = (output_dir / part_path.name).resolve()
    if part_path.resolve() == destination:
        _logger.debug("Part file already present in output directory: %s", destination)
    else:
        shutil.copy(part_path, destination)
        _logger.info("Copied part file to %s", destination)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate OrcaSlicer settings from master YAML configurations."
    )
    parser.add_argument(
        "process_data_file",
        type=Path,
        help="JSON file containing overrides and metadata for the print.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where generated configuration files are written.",
    )
    parser.add_argument(
        "master_settings_dir",
        type=Path,
        help="Directory containing master .yaml configuration files.",
    )
    return parser.parse_args(args)


def main(argv=None):

    args = parse_args(argv)

    try:
        generate_settings(
            process_data_file=args.process_data_file,
            output_dir=args.output_dir,
            master_settings_dir=args.master_settings_dir,
        )
    except Exception as exc:  # pragma: no cover - CLI safeguard
        _logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
