import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)


def _first_list_value(value):
    if isinstance(value, list) and value:
        return value[0]
    return value


def _sync_machine_default_filament(machine_data, filament_data):
    """Keep the machine preset's default filament metadata aligned with the export.

    Orca persists these machine-level defaults into the 3MF project. If they are left
    at the static machine master values, the GUI can show a stale material label even
    when the actual filament profile embedded for the plate is correct.
    """

    machine_data["default_filament_profile"] = [str(filament_data["name"])]

    filament_colour = _first_list_value(filament_data.get("default_filament_colour"))
    if filament_colour is not None:
        machine_data["default_filament_colour"] = [str(filament_colour)]


def _resolve_process_override_targets(process_overrides, used_master_configs):
    missing_keys = []
    ambiguous_keys = {}
    resolved_targets = {}

    for key in process_overrides:
        matching_configs = [
            config for config in used_master_configs if key in config["master_data"]
        ]

        if not matching_configs:
            missing_keys.append(key)
            continue

        if len(matching_configs) > 1:
            ambiguous_keys[key] = [
                config["config_file"].absolute().as_posix()
                for config in matching_configs
            ]
            continue

        resolved_targets[key] = matching_configs[0]

    if not missing_keys and not ambiguous_keys:
        return resolved_targets

    used_files = "\n".join(
        config["config_file"].absolute().as_posix() for config in used_master_configs
    )
    error_lines = [
        "The following flat process_overrides keys could not be resolved uniquely "
        "against the loaded OrcaSlicer master settings."
    ]

    if missing_keys:
        error_lines.append(
            "Missing keys (would be silently dropped): "
            + ", ".join(sorted(missing_keys))
        )
        error_lines.append(
            "Add each missing key to the correct master YAML before overriding it."
        )

    if ambiguous_keys:
        error_lines.append("Ambiguous keys (match more than one master YAML):")
        for key in sorted(ambiguous_keys):
            error_lines.append(f"{key}:")
            for config_file in ambiguous_keys[key]:
                error_lines.append(config_file)

    error_lines.append("Loaded master settings:")
    error_lines.append(used_files)
    raise ValueError("\n".join(error_lines))


def generate_settings(
    process_data_file: Path, output_dir: Path, master_settings_dir: Path
) -> dict[str, object]:
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
    process_overrides = dict(process_data.get("process_overrides", {}))

    if not output_dir.exists():
        raise FileNotFoundError(f"Directory {output_dir} does not exist.")

    if not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a directory.")

    (output_dir / "filaments").mkdir(parents=True, exist_ok=True)

    if not master_settings_dir.exists():
        raise FileNotFoundError(f"Directory {master_settings_dir} does not exist.")

    filament_found = False
    filament_filename = None
    selected_filament_master = None
    settings_files_used = []
    used_master_configs = []
    generated_artifacts: dict[str, object] = {
        "machine_settings_path": None,
        "process_settings_path": None,
        "filament_settings_paths": [],
        "print_host_path": None,
        "used_master_settings_files": [],
    }
    for config_file in sorted(master_settings_dir.glob("*.yaml")):
        with config_file.open("r", encoding="utf-8") as file_handle:
            master_data = yaml.safe_load(file_handle)

        name = master_data["name"]

        path_part = ""
        if master_data["type"] == "filament":
            path_part = "filaments/"

            if master_data["name"] != filament:
                _logger.info(
                    f"Skipping {name} config, in {config_file.absolute().as_posix()} matching filament name {filament}"
                )
                continue
            filament_found = True
            filament_filename = config_file.absolute().as_posix()
            selected_filament_master = master_data

        settings_files_used.append(config_file.absolute().as_posix())
        used_master_configs.append(
            {
                "config_file": config_file,
                "master_data": master_data,
                "path_part": path_part,
            }
        )

    if not filament_found:
        raise ValueError(f"Filament {filament} not found in {master_settings_dir}.")
    else:
        _logger.info(f"Filament {filament} found in {filament_filename} and processed.")

    override_targets = _resolve_process_override_targets(
        process_overrides, used_master_configs
    )

    for config in used_master_configs:
        master_data = config["master_data"]
        path_part = config["path_part"]
        name = master_data["name"]

        if master_data["type"] == "machine":
            _sync_machine_default_filament(master_data, selected_filament_master)

            print_host = master_data.get("print_host")
            if print_host is not None:

                with (output_dir / "print_host.txt").open(
                    "w", encoding="utf-8"
                ) as file_handle:
                    file_handle.write(print_host)
                _logger.info(f"Saved print host to {output_dir / 'print_host.txt'}")
                generated_artifacts["print_host_path"] = str(
                    (output_dir / "print_host.txt").resolve()
                )

        for key, value in process_overrides.items():
            if override_targets.get(key) != config:
                continue

            _logger.info(f"Overriding {key} in {name} with {value}")
            master_data[key] = str(value)

        json_path = (output_dir / f"{path_part}{name}.json").resolve()
        with json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(master_data, file_handle, indent=2)
        _logger.info(f"Saved {name} config to {path_part}{name}.json")
        if master_data["type"] == "machine":
            generated_artifacts["machine_settings_path"] = str(json_path)
        elif master_data["type"] == "process":
            generated_artifacts["process_settings_path"] = str(json_path)
        elif master_data["type"] == "filament":
            cast_list = generated_artifacts["filament_settings_paths"]
            assert isinstance(cast_list, list)
            cast_list.append(str(json_path))

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

    _logger.info(
        f"Used the following master settings files:\n{'\n'.join(settings_files_used)}"
    )
    generated_artifacts["used_master_settings_files"] = settings_files_used

    part_path = Path(process_data["part_file"]).expanduser()
    if not part_path.exists():
        raise FileNotFoundError(f"File {part_path} does not exist.")

    destination = (output_dir / part_path.name).resolve()
    if part_path.resolve() == destination:
        _logger.debug("Part file already present in output directory: %s", destination)
    else:
        shutil.copy(part_path, destination)
        _logger.info("Copied part file to %s", destination)

    return generated_artifacts


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
