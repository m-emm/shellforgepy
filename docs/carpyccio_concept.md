# carpyccio Architecture & Concept Document

Status: draft for repository bootstrap  
Audience: developers creating the new `carpyccio` repository  
Goal: reach first usable deterministic G-code output that can print a small cube

---

## 1) Project intent

`carpyccio` is a **headless Python slicer backend** that converts 3D model files into G-code using a deterministic files-to-files workflow.

It is designed as a companion project to `shellforgepy`, and should share its style:

- explicit inputs/outputs
- modular pipeline stages
- test-first development
- minimal, CI-friendly dependencies

### 1.1 Product goals

1. **Deterministic output**: same inputs + same version ⇒ byte-identical G-code.
2. **Simple operation in CI**: no GUI, no OpenGL/X11, no hidden config files.
3. **Schema-first configuration**: one validated JSON/YAML parameter file.
4. **Debuggable pipeline**: stage artifacts can be dumped and inspected.
5. **Incremental architecture**: production-ready core for FDM, extensible later.

### 1.2 Explicit non-goals for initial repo

- No GUI and no printer host control (OctoPrint upload, etc.) in v0.
- No support generation in the first usable release.
- No adaptive layers in the first usable release.
- No variable line width, pressure advance tuning, or advanced acceleration planners in v0.

---

## 2) First usable milestone definition (v0.1)

This document defines **v0.1** as “print a small cube reliably.”

### 2.1 v0.1 scope

Required in first implementation:

1. Load one STL mesh.
2. Slice at constant layer height.
3. Generate:
   - outer walls (2 perimeters)
   - rectilinear infill
   - solid top and bottom layers
4. Emit Marlin-compatible G-code with:
   - start/end blocks
   - temperatures
   - fan control (basic)
   - travel and extrusion moves
   - retraction and Z-hop (simple fixed model)
5. Deterministic output formatting and ordering.
6. Command-line usage with one params file and one output directory.

### 2.2 v0.1 acceptance target

Given `cube_20mm.stl` and a reference `params.yaml`, running:

```bash
carpyccio slice --input cube_20mm.stl --params params.yaml --outdir out/
```

must produce:

- `out/result.gcode`
- `out/result.meta.json`
- `out/canonical.params`
- `out/manifest.json`

and `result.gcode` must pass deterministic golden comparison in CI.

---

## 3) Dependency policy and concrete dependency set

## 3.1 Hard constraints

Allowed dependencies must be:

- pip-installable on Linux/macOS/Windows
- broadly maintained
- free of GUI/display stack requirements

Forbidden in core:

- Qt/GTK/wx
- OpenGL/X11 dependencies
- PyTorch/TensorFlow

## 3.2 Proposed dependencies by layer

### Runtime core (required)

- `numpy`: vector math, coordinate arrays
- `shapely>=2`: robust 2D polygon ops for layer regions/toolpath prep
- `trimesh`: mesh loading and plane intersection for slicing
- `jsonschema`: strict config validation
- `PyYAML`: YAML input support (JSON via stdlib)
- `typing-extensions` (only if needed for forward compatibility)

### Optional runtime extras

- `orjson` as optional faster JSON for artifact writing
- `zstandard` for compressed debug artifacts

### Dev/test dependencies

- `pytest`
- `pytest-cov`
- `hypothesis` (optional but recommended for geometry property tests)
- `black`, `isort`, `ruff`

## 3.3 Why this set is minimal-enough

- `trimesh + shapely + numpy` gives sufficient geometry primitives for v0.1.
- No C++ CAD kernels needed for first print path.
- Installation remains simple compared to full GUI slicers.

---

## 4) Architecture overview

The system is a strict stage pipeline:

1. Resolve inputs and build run manifest
2. Parse and validate parameters
3. Canonicalize parameters
4. Load/normalize geometry
5. Plan layers
6. Extract per-layer cross-sections
7. Build print regions
8. Generate toolpaths
9. Convert toolpaths to machine motion commands
10. Emit deterministic G-code and metadata

Each stage:

- receives a typed input object
- returns a typed output object
- is pure (or side effects isolated in I/O wrappers)

This keeps unit tests focused and supports artifact dumps.

---

## 5) Data contracts (high-level structures)

Use `dataclasses` for pipeline payloads (avoid heavy frameworks in v0.1).

## 5.1 Common primitives

- `Point2(x: float, y: float)`
- `Point3(x: float, y: float, z: float)`
- `Polyline2(points: np.ndarray[N,2], closed: bool)`
- `Polygon2(exterior: np.ndarray[M,2], holes: list[np.ndarray[K,2]])`

## 5.2 Manifest and configuration

```python
@dataclass(frozen=True)
class InputFileRecord:
    path: str
    sha256: str
    size_bytes: int

@dataclass(frozen=True)
class RunManifest:
    carpyccio_version: str
    schema_version: str
    inputs: list[InputFileRecord]
    canonical_params_hash: str
    job_hash: str
```

```python
@dataclass(frozen=True)
class ValidatedParams:
    machine: MachineParams
    material: MaterialParams
    process: ProcessParams
    output: OutputParams
```

## 5.3 Geometry and layers

```python
@dataclass(frozen=True)
class NormalizedModel:
    model_id: str
    source_path: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    mesh: object  # backend handle, hidden behind protocol
```

```python
@dataclass(frozen=True)
class LayerDef:
    index: int
    z_bottom: float
    z_top: float
    z_mid: float

@dataclass(frozen=True)
class LayerPlan:
    layers: list[LayerDef]
```

```python
@dataclass(frozen=True)
class LayerSlice:
    layer_index: int
    islands: list[Polygon2]
    warnings: list[str]

@dataclass(frozen=True)
class SliceSet:
    slices: list[LayerSlice]
```

## 5.4 Regions and toolpaths

```python
@dataclass(frozen=True)
class LayerRegions:
    layer_index: int
    perimeter_region: list[Polygon2]
    solid_region: list[Polygon2]
    infill_region: list[Polygon2]
```

```python
@dataclass(frozen=True)
class PathSegment:
    points: np.ndarray  # [N,2]
    role: str           # wall|infill|solid|travel
    speed_mm_s: float
    extrusion_width: float | None

@dataclass(frozen=True)
class LayerToolpaths:
    layer_index: int
    extrusion_paths: list[PathSegment]
    travel_paths: list[PathSegment]
```

## 5.5 Motion and G-code

```python
@dataclass(frozen=True)
class MotionCmd:
    op: str  # e.g. G0, G1, M104
    x: float | None = None
    y: float | None = None
    z: float | None = None
    e: float | None = None
    f: float | None = None
    comment: str | None = None
```

```python
@dataclass(frozen=True)
class MotionProgram:
    commands: list[MotionCmd]
```

---

## 6) Pipeline stage specification for v0.1

## Stage 00: `resolve_inputs`

Input: CLI args (`--input`, `--params`, `--outdir`)  
Output: `RunManifest` skeleton + raw bytes/hash table

Rules:

- fail if more than one model for v0.1 (keep simple)
- hash file contents with SHA-256

## Stage 01: `validate_params`

Input: params file bytes  
Output: `ValidatedParams`

Rules:

- support YAML and JSON
- validate against one shipped JSON schema
- reject unknown keys (`additionalProperties: false` for strictness)

## Stage 02: `canonicalize_params`

Input: `ValidatedParams`  
Output: canonical text + hash

Canonical text format:

- flattened dotted keys
- `key=value` per line
- UTF-8, `\n` newline
- lexicographically sorted keys
- normalized float string format (fixed precision policy)

## Stage 03: `load_and_normalize_model`

Input: model file path + validated process settings  
Output: `NormalizedModel`

Rules:

- load STL via `trimesh`
- ensure model sits on bed (`z_min -> 0` translation)
- optional user transform support can be postponed; keep minimal

## Stage 04: `plan_layers`

Input: model z bounds + `layer_height`, `first_layer_height`  
Output: `LayerPlan`

Rules:

- deterministic z values, avoid cumulative float drift (derive from integer layer index)

## Stage 05: `slice_layers`

Input: `NormalizedModel`, `LayerPlan`  
Output: `SliceSet`

Rules:

- intersect mesh with each slice plane
- polygonize and repair invalid rings with shapely cleaning path

## Stage 06: `classify_regions`

Input: `SliceSet`, process params  
Output: per-layer `LayerRegions`

v0.1 behavior:

- generate perimeter offsets for wall lines
- classify top/bottom solid layers by index from top/bottom
- remaining interior goes to infill region

## Stage 07: `build_toolpaths`

Input: `LayerRegions`, pattern params  
Output: `LayerToolpaths`

v0.1 behavior:

- perimeters: contour-following paths
- infill: 0/90 alternating rectilinear
- top/bottom: solid hatch with same spacing as line width
- deterministic path ordering: lowest X/Y start, then clockwise/counterclockwise normalization

## Stage 08: `plan_motion`

Input: toolpaths + machine/material params  
Output: `MotionProgram`

v0.1 behavior:

- convert XY paths to G0/G1
- compute extrusion using volumetric model: `E = volume / filament_area`
- use absolute E mode for simplicity (`M82`)
- simple fixed retraction between disjoint extrusion islands

## Stage 09: `emit_gcode`

Input: `MotionProgram`, start/end templates  
Output: `result.gcode`, `result.meta.json`

Rules:

- fixed float precision and deterministic command formatting
- comments only from deterministic sources
- include summary stats (layer count, filament estimate, hashes)

---

## 7) CLI contract (draft, implementation-ready)

## 7.1 Commands

```bash
carpyccio slice --input MODEL.stl --params PARAMS.yaml --outdir OUTDIR [--debug-dump all|failed|none]
carpyccio validate --params PARAMS.yaml
carpyccio canonicalize --params PARAMS.yaml --out canonical.params
carpyccio schema --out schema.json
```

## 7.2 CLI behavior rules

- No implicit profiles; all needed print settings must come from params file.
- Exit code non-zero on validation/slicing failure.
- Human-readable stderr + optional machine-readable error JSON.

## 7.3 Output layout

```text
OUTDIR/
  result.gcode
  result.meta.json
  canonical.params
  manifest.json
  debug/
    00-resolved-inputs.json
    01-validated-params.json
    02-canonical-params.txt
    03-normalized-model.json
    04-layer-plan.json
    05-slices.json
    06-regions.json
    07-toolpaths.json
    08-motion.json
```

---

## 8) Repository and module structure (new project)

Recommended initial tree:

```text
carpyccio/
  pyproject.toml
  README.md
  LICENSE
  src/carpyccio/
    __init__.py
    __main__.py
    cli.py
    version.py

    config/
      schema.py
      loader.py
      validate.py
      canonicalize.py
      models.py

    io/
      files.py
      hashing.py
      artifact_dump.py
      manifest_io.py

    geometry/
      backend_protocol.py
      trimesh_backend.py
      polygon_ops.py

    pipeline/
      context.py
      run_pipeline.py
      stage_00_resolve_inputs.py
      stage_01_validate_params.py
      stage_02_canonicalize_params.py
      stage_03_load_model.py
      stage_04_plan_layers.py
      stage_05_slice_layers.py
      stage_06_classify_regions.py
      stage_07_build_toolpaths.py
      stage_08_plan_motion.py
      stage_09_emit_gcode.py

    toolpath/
      perimeters.py
      infill_rectilinear.py
      solids.py
      ordering.py

    motion/
      extrusion.py
      planner.py
      retraction.py

    gcode/
      commands.py
      marlin.py
      format.py

    models/
      common.py
      manifest.py
      layers.py
      slices.py
      regions.py
      toolpaths.py
      motion.py

  tests/
    unit/
      config/
      geometry/
      toolpath/
      motion/
      pipeline/
      gcode/
    integration/
      test_cube_end_to_end.py
      fixtures/
        cube_20mm.stl
        params_reference.yaml
        expected_result.gcode
```

---

## 9) Parameter schema (v0.1 minimum)

Schema should be versioned and shipped in package, e.g. `schema_version = 1`.

Minimum keys to support first print:

- `machine.bed_size.{x,y,z}`
- `machine.nozzle_diameter`
- `machine.filament_diameter`
- `machine.gcode_flavor` (only `marlin` in v0.1)
- `material.nozzle_temp_c`
- `material.bed_temp_c`
- `material.fan_percent`
- `process.layer_height`
- `process.first_layer_height`
- `process.line_width`
- `process.wall_count`
- `process.top_layers`
- `process.bottom_layers`
- `process.infill_density`
- `process.infill_pattern` (`rectilinear` only in v0.1)
- `process.print_speed_mm_s`
- `process.travel_speed_mm_s`
- `process.retract_length_mm`
- `process.retract_speed_mm_s`
- `output.precision_decimals`

### 9.1 Validation principles

- range constraints for all numeric values
- enums for bounded string values
- descriptions/examples in schema for future UI use
- strict additional-properties rejection

---

## 10) Determinism and cacheability contract

## 10.1 Deterministic inputs

`job_hash` should be computed from:

1. carpyccio version
2. schema version
3. canonical params bytes
4. ordered `(relative_path, file_hash)` tuples for model inputs

## 10.2 Deterministic outputs

- fixed float renderer (e.g., `%.5f` or configurable but pinned)
- stable ordering of islands, rings, paths, and command emission
- no wall-clock timestamps in `result.gcode`
- any metadata timestamp must be outside hash-critical output or explicitly excluded

## 10.3 Golden testing policy

- `expected_result.gcode` is committed for reference fixture(s)
- CI compares exact file bytes
- if algorithm changes intentionally, golden update requires explicit PR justification

---

## 11) Testing strategy (detailed and actionable)

## 11.1 Unit tests by module

- `tests/unit/config/test_validate.py`
  - valid params pass
  - unknown keys fail
  - out-of-range values fail
- `tests/unit/config/test_canonicalize.py`
  - sorted keys stable
  - equivalent YAML/JSON produce identical canonical file

- `tests/unit/geometry/test_trimesh_backend.py`
  - STL loads
  - bounds normalized to bed origin

- `tests/unit/pipeline/test_layer_planning.py`
  - layer counts and z values expected for simple bounds

- `tests/unit/toolpath/test_perimeters.py`
  - perimeter offset count and path closure
- `tests/unit/toolpath/test_rectilinear.py`
  - deterministic hatch lines for known polygon

- `tests/unit/motion/test_extrusion.py`
  - E computation against known path length
- `tests/unit/gcode/test_marlin_format.py`
  - command formatting exactness and precision

## 11.2 Integration tests

- `test_cube_end_to_end.py`:
  - run full `slice` command on `cube_20mm.stl`
  - assert output files exist
  - assert `result.gcode` equals golden
  - assert `manifest` hashes are present and stable

## 11.3 Repeatability test

- run same command 3 times in temp dirs
- assert identical `result.gcode` SHA-256 each run

## 11.4 CI matrix

For bootstrap:

- Linux + Python 3.12 (required gate)
- Optional expansion later: macOS/Windows determinism checks

---

## 12) Implementation plan for first iterations

## Iteration A: bootstrap and config core

Deliverables:

- repo scaffold, packaging, CLI skeleton
- schema + validation + canonicalization
- manifest and hashing utilities

Done criteria:

- `validate`, `canonicalize`, `schema` commands working
- unit tests green

## Iteration B: geometry and layer slicing

Deliverables:

- STL loading and normalization
- layer planning
- polygon extraction for each layer

Done criteria:

- debug artifacts for stages 03–05
- tests for cube cross-sections pass

## Iteration C: toolpath + motion + G-code (first printable cube)

Deliverables:

- perimeters, infill, top/bottom solids
- motion conversion and Marlin emission
- end-to-end cube golden test

Done criteria:

- first usable `result.gcode` for cube fixture
- deterministic repeatability test passes

## Iteration D: hardening

Deliverables:

- improve validation errors and diagnostics
- optimize path ordering and travel minimization (deterministically)
- docs cleanup and examples

Done criteria:

- stable CI, documented known limitations

---

## 13) Known limitations after v0.1 (intentional)

- no supports
- no bridging specialization
- no multi-material
- no tree supports or organic infill patterns
- no arc fitting

These are deferred to keep first release focused and shippable.

---

## 14) Developer kickoff checklist

When creating the new repo, start in this order:

1. Scaffold package and CI with strict lint/test steps.
2. Implement schema + canonicalization + manifest path.
3. Add pipeline skeleton and artifact dumping.
4. Implement STL -> slices for cube fixture.
5. Implement minimal toolpath generation.
6. Implement motion + Marlin emitter.
7. Lock golden cube test and determinism test.
8. Document exactly which params are supported in v0.1.

If this order is followed, developers can quickly reach a first printable cube while keeping architecture clean for future slicer growth.
