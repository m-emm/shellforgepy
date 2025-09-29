# Agent Orientation

This repository (`shellforgepy`) hosts the Python runtime for the ShellForge toolchain. It offers a geometry-first core (`geometry/`), alignment and construction helpers (`construct/`), production utilities (`produce/`), and pluggable CAD adapters (`adapters/`) for CadQuery and FreeCAD. The goal is backend-agnostic modelling with optional materialisation in real CAD kernels.

## Code Style
- The codebase targets Python 3.12. Follow the existing conventions: Black (line length 88) and isort are the canonical formatters.
- Prefer descriptive names over comments. Add docstrings for user-facing functions or adapters.
- Type hints are used where they improve readability or are required by adapters, but do not blanket-annotate obvious primitives.
- Implementing algorithms in cad-library agnostic way is preferred, so try this first.
- Keep adapter-specific dependencies inside their respective modules so core layers stay dependency-light.

## Testing
- Prefer creating proper pytest tests, not ad-hoc scripts. Avoid running quick debug/test scripts.
- Pytest lives under `tests/unit`. Add `test_*.py` functions (no classes needed) and keep adapter-dependent tests behind the relevant extras if they require CadQuery/FreeCAD.
- Run `python -m pytest` from the repo root before sending patches
- For full testing, test with both adapters, that is, also run `./freecad_python.sh -m pytest` in the root dir. This will run in a freecad python interpreter and use the freecad engine.

## Repository Layout
- `src/shellforgepy/`
  - `geometry/`: NumPy/SciPy-based primitives (no CAD dependencies).
  - `construct/`: alignment and composition helpers used across adapters.
  - `produce/`: arrangement/export logic for fabrication.
  - `adapters/`: `cadquery/`, `freecad/`, and adapter chooser glue.
  - `simple.py`: convenience facade that auto-selects an installed adapter.
- `tests/`: unit tests mirroring the structure above.
- ShellForge-specific scripts (e.g., `freecad_python.sh`, `freecad_python.FCMacro`) live at the repo root for adapter integration.

## Environment Notes
- The project expects Python ≥3.11. Create a virtual environment and install with `pip install -e ".[testing]"` for development.
- Optional extras: `[cadquery]`, `[freecad]`, or `[all]` pull in adapter dependencies. Install them if you need the corresponding backend.
- FreeCAD integration assumes the system FreeCAD Python modules are discoverable (see `freecad_python.sh` for guidance).

## Common Tasks
- Run tests: `pytest`
- Format code: `black src/ tests/` and `isort src/ tests/`
- Verify functionality: run examples as documented in README.md: `python examples/filleted_boxes_example.py`
- Export example: use helpers in `produce/arrange_and_export.py` together with primitives from `construct` and `geometry`.

## Integration Points
- Geometry routines reuse utilities mirrored from `py_3d_construct_lib`. Keep shared logic consistent across repositories.
- Adapters rely on CadQuery/FreeCAD APIs; when updating them, check `tests/unit/adapters/*` for coverage and update both adapters for feature parity when possible.
- `shellforgepy.simple` is the public entry point; ensure new capabilities surface there with clean imports. Users should import specific functions they need rather than accessing internal adapter details.

Keep this guide current as the ShellForge pipeline evolves.
