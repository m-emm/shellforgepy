# ShellforgePy Slicer Integration

This guide explains how to use `shellforgepy.workflow.workflow` to turn a geometry script into production-ready print artifacts with OrcaSlicer. The workflow CLI orchestrates three stages:

1. **Geometry generation** – run any shellforgepy example or your own script to produce STL(s) and process metadata.
2. **Settings materialisation** – combine the generated process data with an OrcaSlicer master settings directory to emit machine/filament JSON files.
3. **Slicing & publishing** – call OrcaSlicer in CLI mode, optionally copy the results to a viewer location, and upload the G-code if desired.

The example script `examples/process_and_workflow.py` demonstrates the full pipeline and is referenced throughout this document.

---

## 1. Install requirements

- ShellforgePy and its dependencies
- OrcaSlicer 2.3.0 (or compatible) installed locally
- Optional: FreeCAD/CadQuery if your geometry script depends on them

Ensure you can execute OrcaSlicer from the path you intend to configure (e.g. `/Applications/OrcaSlicer 2_3_0.app/Contents/MacOS/OrcaSlicer`).

---

## 2. Prepare OrcaSlicer master settings

The workflow consumes a *master settings directory* containing machine/filament YAML definitions. You can:

- Use the bundled examples in `examples/example_settings_master/` (contains `ExampleMachine.yaml`, `ExampleProcess.yaml`, `FilamentPLAExample.yaml`)
- If you have a different printer than what is in the examples, you need to hand-craft your own master files. It's best to extract the json config from your GUI settings and convert them to yaml for easy editing. To use the system, you have to do this once for your printer. Place these master file under version control in your workspace.

The generator writes JSON/INFO files based on the YAML input and the `process_data` overrides emitted by your geometry script.

---

## 3. Configure the workflow CLI

The CLI reads configuration from `~/.shellforgepy/config.json` (override with `--config`). Initialise the file once:

```bash
python src/shellforgepy/workflow/workflow.py config init
```

Then edit/add the relevant entries. A minimal configuration targeting the bundled example settings looks like this:

```json
{
  "runs_dir": "runs",
  "python": {
    "runner": "/Users/mege/.pyenv/versions/cadquery/bin/python"
  },
  "orca": {
    "executable": "/Applications/OrcaSlicer 2_3_0.app/Contents/MacOS/OrcaSlicer",
    "master_settings_dir": "/Users/mege/git/shellforgepy/examples/example_settings_master",
    "debug_level": 6
  },
  "viewer": {
    "default_stl_file": "/Users/mege/git/mege-stl-viewer/public/current.stl"
  },
  "upload": {
    "printer": "192.168.178.85:4409"
  }
}
```

### Key settings

| Key | Purpose |
| --- | --- |
| `runs_dir` | Folder where per-run artifacts are stored; defaults to `./runs` |
| `python.runner` | Interpreter (or `freecad_python.sh`) used to run geometry scripts |
| `orca.executable` | OrcaSlicer CLI binary |
| `orca.master_settings_dir` | Directory containing machine/filament YAML masters |
| `viewer.default_stl_file` | Optional: path updated with the latest combined STL for STL viewers |
| `upload.printer` | Optional: Moonraker printer endpoint (`ip[:port]`) for automatic uploads |

All paths may use `~` and will be expanded/resolved by the workflow.

---

## 4. Run the full workflow

With configuration in place, execute the example pipeline:

```bash
python src/shellforgepy/workflow/workflow.py --verbose run --slice examples/process_and_workflow.py
```

What happens:

1. `process_and_workflow.py` generates the geometry, writes the combined STL, and emits `<script>_process.json` metadata.
2. The workflow reads the manifest left by `arrange_and_export_parts`, copies the STL to `viewer.default_stl_file` (if configured), and materialises OrcaSlicer JSON settings.
3. OrcaSlicer slices the STL using the generated machine/process/filament configs, producing:
   - `runs/<script>_run_<timestamp>/plate_1.gcode`
   - `runs/<script>_run_<timestamp>/<script>.3mf`
   - `runs/<script>_run_<timestamp>/slicedata/` (raw slicing data)
4. Optional: the workflow uploads any generated G-code files to the configured printer.

Verbose mode streams OrcaSlicer’s CLI output, which includes progress, warnings, and shader messages (safe to ignore when running headless on macOS).

---


## Why use standalone YAML masters?

OrcaSlicer ships with a large web of preset fragments that inherit, override, and merge each other. Importing those presets directly makes it hard to understand which parameter is actually taking effect because the CLI replays the entire inheritance chain at run time.

The approach taken in `examples/example_settings_master/` is therefore:

1. Start from the presets you like inside OrcaSlicer once.
2. Export the relevant machine, process, and filament JSON files.
3. Convert them to YAML (or keep them as JSON) and clean them up so each profile has a unique name and contains only the values you want to own.
4. Store the cleaned files in version control and edit them like normal config data.
5. Let `generate_settings` convert those YAML masters back into the flat JSON/INFO files that OrcaSlicer CLI expects.

By treating the YAML files as the single source of truth you get repeatable, reviewable configuration without OrcaSlicer’s preset inheritance interfering. The workflow CLI never touches Orca’s built‑in presets – it only feeds the JSON generated from your master directory to the CLI, so the slicer will not quietly pull in defaults you did not track.

## Per-design overrides

Per-design overrides only work for parameters that exist in your master configuration. If a setting is missing from the master files, the workflow has no way of knowing whether it belongs to the machine, process, or filament profile. Keep your masters as complete as possible; per-design tweaks can then safely override any of those values.


Your master directory captures the printer-, process-, and filament-level defaults, but individual designs often need tweaks (for example, a finer layer height or extra walls). ShellforgePy handles this by letting each geometry script supply its own `process_data` dictionary. Whatever keys you set there – whether they logically belong to machine, process, or filament – are written into `<script>_process.json` and layered on top of the masters when OrcaSlicer runs.

Example snippet from `examples/process_and_workflow.py`:

```python
PROCESS_DATA = {
    "filament": "FilamentPLAExample",
    "process_overrides": {
        "layer_height": "0.18",      # print this part fine
        "top_shell_layers": "5",
        "wall_loops": "3",
        "printable_height": "230"
    }

```

You do not need to remember which parameter lives in which profile inside Orca. Simply place overrides under the appropriate section (`process_overrides`, `filament_overrides`, or `machine_overrides`) and the workflow will merge them onto the generated JSON. All of these settings sit next to your design code, so they are versioned with the geometry.

When the workflow runs:

1. `arrange_and_export_parts` writes the combined STL and the process JSON containing your overrides.
2. `generate_settings` replays those overrides on top of the master YAML files.
3. OrcaSlicer receives the updated JSON and uses exactly the values you provided for this run.

Use this mechanism for per-part tweaks such as:

- Lower layer heights for higher surface detail in a decorative part
- Use a higher infill percentage, a more mechanically stable infill pattern and more walls to produce a rugged functional part
- Reduce infill percentage and walls and use a fast running infill pattern to use less filament to quickly print a decorative part
- Print faster for a draft part, and slower and finer for production part

## 5. Tips & customisation

- **Viewer mirroring:** Set `viewer.default_stl_file` to keep an STL viewer pointed at the latest combined geometry.
- **Alternate runners:** Swap `python.runner` for `freecad_python.sh` if your script expects the FreeCAD Python environment.
- **Multiple printers/filaments:** Provide additional master YAMLs in the same directory; the workflow picks the ones referenced by the process JSON (`filament`, `type: machine`, etc.).
- **Uploads & previews:** Configure `upload.*` for Moonraker uploads and `render.*` for a post-processing preview pipeline; both hooks are optional and safe to leave unset.

For reference integrations, inspect:

- `examples/process_and_workflow.py`
- `examples/example_settings_master/ExampleMachine.yaml`
- `examples/example_settings_master/ExampleProcess.yaml`
- `examples/example_settings_master/FilamentPLAExample.yaml`

Happy slicing!

