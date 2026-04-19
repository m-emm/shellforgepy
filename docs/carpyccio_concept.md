# carpyccio — Concept Document (v0.1 draft)

## 1) Vision and Scope

`carpyccio` is a companion project to `shellforgepy`: a **headless, deterministic, CI-friendly Python slicer** that transforms 3D geometry files into G-code.

Core goals:

- Replace GUI-centric slicing workflows with a **files-in / files-out** pipeline.
- Keep dependencies lightweight and easy to install from PyPI (no GUI/OpenGL/X11 stacks).
- Make slicing behavior explicit, reproducible, hashable, and introspectable.
- Encode all user-facing parameters in a schema-driven config model suitable for docs and future UI layers.

Out of scope for initial versions:

- Native interactive GUI.
- Hardware acceleration requirements.
- Printer host management beyond optional static output artifacts.

---

## 2) Non-Functional Requirements

### 2.1 Installation and Dependency Constraints

- Python package installable with plain `pip install carpyccio`.
- Dependencies limited to widely available, manylinux/macos/windows-friendly packages.
- Avoid heavy runtime dependencies such as:
  - GUI frameworks (Qt, GTK, wx)
  - OpenGL/X11 libraries
  - ML stacks (PyTorch, TensorFlow)

### 2.2 Determinism and Reproducibility

Given:

- identical geometry inputs,
- identical parameter file contents,
- identical `carpyccio` version,
- identical platform-relevant numeric settings,

the slicer should produce **byte-for-byte identical G-code**.

### 2.3 Configuration Simplicity

- Exactly one user parameter file per slicing run (JSON or YAML).
- No hidden defaults from user home directories.
- No inheritance layers, profile merge chains, or environment-driven overrides.
- Effective config is always materialized and written to output.

### 2.4 Traceability

- Internal stage outputs can be dumped as JSON/CBOR/NPZ-like artifacts for debugging.
- Every output bundle includes provenance metadata: tool version, schema version, input file hashes.

---

## 3) Minimal Dependency Strategy

## 3.1 Recommended Geometry and Core Math Stack

A practical minimal stack for robust slicing:

- **numpy**: vectorized math and array storage.
- **shapely** (2.x): 2D polygon operations (union, offset, intersection, clipping) for per-layer processing.
- **trimesh** (optional but strongly recommended): mesh loading and slicing utilities for STL/OBJ/3MF-like workflows.
- **pyyaml** (optional): YAML input support; JSON always available via stdlib.
- **jsonschema**: config validation against a first-class schema.
- **networkx** (optional): path ordering / travel optimization graph algorithms.

Potentially optional, behind extras:

- `scipy`: advanced spatial acceleration or optimization.
- `lz4/zstandard`: faster compression for debug dumps.

## 3.2 Why This Stack

- All are pip-installable and common in CI.
- No GUI/system display requirements.
- Clean separation between 3D mesh ingest and 2D per-layer toolpath planning.

## 3.3 Adapter-Friendly Geometry Backend Abstraction

To align with `shellforgepy` philosophy, `carpyccio` should define an internal geometry protocol:

- `MeshSource` interface (load triangles, bounds, metadata)
- `PlanarOps` interface (offset/boolean/repair polygons)

Default implementation can use `trimesh + shapely`, while alternative backends can be swapped later.

---

## 4) Workflow Stages and Intermediary Data Structures

The slicer should use explicit, named stages with typed dataclasses (or pydantic models only where beneficial).

## 4.1 Stage 0: Run Manifest / Input Resolution

**Inputs**:

- geometry files (e.g., STL/OBJ/3MF)
- params file (`params.json` or `params.yaml`)

**Outputs**:

- `RunManifest`
  - `tool_version`
  - `schema_version`
  - `input_files[]` with SHA-256 hashes
  - `timestamp_utc` (for metadata only; excluded from deterministic content hash)
  - `canonical_params_hash`

## 4.2 Stage 1: Parameter Parse + Schema Validation

**Actions**:

- Parse JSON/YAML into structured object.
- Validate against JSON Schema.
- Emit actionable errors with JSONPointer paths.

**Outputs**:

- `ValidatedParams` (typed in-memory model)
- `ValidationReport`

## 4.3 Stage 2: Canonicalization

**Actions**:

- Flatten nested parameters into dotted keys (`print.layer_height=0.2`).
- Normalize units and enums.
- Sort keys lexicographically.

**Outputs**:

- `CanonicalParamsFile` (text lines `key=value` sorted)
- `canonical_params_hash` (SHA-256 of canonical text)

## 4.4 Stage 3: Geometry Ingest + Normalization

**Actions**:

- Load and triangulate as needed.
- Validate manifoldness where possible.
- Apply transforms (scale, rotate, translate to bed coordinates).

**Outputs**:

- `NormalizedModel`
  - `model_id`
  - `triangles` / mesh handle
  - `bbox`
  - `transform_applied`
  - `quality_flags`

## 4.5 Stage 4: Layer Planning

**Actions**:

- Resolve Z-layer boundaries from layer strategy (uniform/adaptive).
- Compute per-layer slice planes.

**Outputs**:

- `LayerPlan`
  - `layers[]` each with `index`, `z_bottom`, `z_top`, `z_mid`
  - global `layer_height_strategy`

## 4.6 Stage 5: Cross-Section Extraction

**Actions**:

- Intersect mesh with each layer plane or slab.
- Build closed 2D polygons (outer boundaries + holes).
- Repair invalid rings where possible.

**Outputs**:

- `LayerSlices`
  - `LayerSlice` per layer:
    - `islands[]` polygons
    - `holes[]`
    - `slice_warnings[]`

## 4.7 Stage 6: Region Classification

**Actions**:

- Partition each layer into print regions:
  - walls/perimeters
  - top/bottom solid
  - infill region
  - support region (if enabled)

**Outputs**:

- `LayerRegions`
  - per layer region polygons and semantic tags

## 4.8 Stage 7: Toolpath Generation

**Actions**:

- Generate ordered polylines for:
  - perimeters (N walls)
  - infill (pattern + density)
  - supports
  - skirt/brim/raft
- Add seam and travel strategy decisions.

**Outputs**:

- `ToolpathPlan`
  - per-layer `extrusion_paths[]`
  - `travel_paths[]`
  - path metadata (`speed_class`, `feature_type`, `extrusion_role`)

## 4.9 Stage 8: Motion + Extrusion Planning

**Actions**:

- Convert geometric toolpaths to machine commands with printer profile:
  - feedrates
  - acceleration hints
  - extrusion E values (absolute/relative)
  - retractions, z-hop

**Outputs**:

- `MotionProgram`
  - ordered command primitives (`Move`, `ExtrudeMove`, `SetTemp`, etc.)

## 4.10 Stage 9: G-code Emission

**Actions**:

- Render deterministic textual G-code.
- Ensure stable formatting (float precision, ordering, newline policy).

**Outputs**:

- `output.gcode`
- `output.meta.json` (hashes, counts, timing, warnings)

## 4.11 Stage 10: Optional Debug Artifact Dump

At each stage, if debug dumping enabled:

- Write `NN-stage-name.json.zst` (or plain `.json`) files.
- These artifacts are pure function outputs from previous stage + params.

This makes pipeline behavior inspectable and supports golden-file testing.

---

## 5) Draft CLI Shape

## 5.1 Primary Command

```bash
carpyccio slice \
  --input model.stl \
  --params params.yaml \
  --outdir out/run_001
```

## 5.2 Suggested Commands

- `carpyccio slice ...` → full pipeline to G-code
- `carpyccio validate --params params.yaml` → schema validation only
- `carpyccio canonicalize --params params.yaml --out canonical.params` → canonical key=value output
- `carpyccio schema --out schema.json` → export active JSON schema
- `carpyccio inspect --artifact out/run_001/debug/05-layer-slices.json` → summarize internal artifact

## 5.3 Suggested Flags

- `--input` (repeatable) for one/many models
- `--params` single required config
- `--outdir` output directory
- `--debug-dump [none|failed|all]`
- `--deterministic` (strict mode; fail on non-deterministic settings)
- `--threads N` (defaults to deterministic-safe value)
- `--format-version` (pins output format)

---

## 6) Input/Output File and Directory Structure

## 6.1 Input

```text
job/
  models/
    part_a.stl
    part_b.stl
  params.yaml
```

## 6.2 Output

```text
out/run_001/
  result.gcode
  result.meta.json
  canonical.params
  manifest.json
  debug/
    00-manifest.json
    01-validated-params.json
    02-canonical-params.txt
    03-normalized-models.json
    04-layer-plan.json
    05-layer-slices.json
    06-layer-regions.json
    07-toolpaths.json
    08-motion-program.json
```

`manifest.json` should include:

- `carpyccio_version`
- `schema_version`
- input file list + hashes
- `canonical_params_hash`
- `gcode_hash`
- stage timings

---

## 7) Draft Python Module Structure

```text
src/carpyccio/
  __init__.py
  cli.py
  version.py

  config/
    schema.py              # JSON schema definition/export
    loader.py              # parse json/yaml
    validate.py            # jsonschema validation
    canonicalize.py        # flatten + stable key=value serializer
    model.py               # typed parameter model/dataclasses

  io/
    inputs.py              # discover/load input files
    outputs.py             # output bundle writing
    hashing.py             # file/content hash helpers
    artifacts.py           # debug artifact dump/load

  geometry/
    protocol.py            # backend interfaces
    trimesh_backend.py     # default mesh ingest/slice support
    planar_ops.py          # shapely helpers
    repair.py              # mesh/polygon cleanup

  pipeline/
    stages.py              # stage orchestration and step registry
    context.py             # run context and deterministic controls
    manifest.py            # RunManifest builder

    stage_01_validate.py
    stage_02_canonicalize.py
    stage_03_ingest.py
    stage_04_layer_plan.py
    stage_05_slice.py
    stage_06_regions.py
    stage_07_toolpaths.py
    stage_08_motion.py
    stage_09_emit.py

  toolpath/
    perimeters.py
    infill.py
    supports.py
    travel.py
    skirts_brims.py

  gcode/
    commands.py            # command dataclasses
    emitter.py             # deterministic renderer
    flavor_marlin.py
    flavor_klipper.py

  models/
    manifest.py
    layers.py
    regions.py
    toolpaths.py
    motion.py

  testing/
    golden.py              # stable output assertion helpers
    fixtures.py
```

---

## 8) Determinism Design Notes

To make outputs cacheable and hashable:

- canonical float formatting (fixed precision policy)
- stable sort order everywhere (islands, paths, keys)
- controlled random seeds (or no randomness)
- deterministic parallelism strategy (or single-thread deterministic core)
- explicit handling of NaN/inf rejection in params
- explicit locale-independent formatting (`.` decimal separator)

Recommended deterministic hash inputs:

1. canonical params file bytes
2. ordered tuple of `(relative_path, sha256)` for input geometry files
3. carpyccio semantic version + format version

Then compute `job_hash = sha256(concatenated_inputs)`.

---

## 9) Schema-First Parameter UX

The JSON schema should be the single source of truth for:

- accepted parameter keys
- value types and ranges
- defaults (if any; ideally explicit)
- descriptions and examples
- deprecations/migrations

This enables:

- strict CLI validation
- generated docs pages
- future GUI form generation/tooltips without changing backend

---

## 10) Alignment with shellforgepy Philosophy

Shared principles:

- explicit data flow over hidden state
- backend abstraction with practical defaults
- reproducible, automation-friendly outputs
- strong testability with unit-level artifacts

Although separate repositories, `carpyccio` should mirror `shellforgepy`’s clarity and modularity:

- pure-python orchestration
- small focused modules
- data-first pipeline boundaries

---

## 11) Proposed Milestones

1. **M0 — Skeleton**: CLI, schema validation, canonicalization, manifest I/O.
2. **M1 — Basic slicing**: STL ingest, fixed layer slicing, perimeters + simple infill, Marlin G-code.
3. **M2 — Determinism hardening**: golden tests, stable hash contract, debug artifacts.
4. **M3 — Feature parity runway**: supports, adaptive layers, multiple flavors.
5. **M4 — UI-ready backend**: schema docs generation and artifact inspector tooling.

---

## 12) Testing Strategy (High-Level)

- Unit tests for every stage transformation.
- Golden-file tests asserting byte-identical `result.gcode` for pinned inputs.
- Schema validation tests for both valid and invalid configs.
- Determinism tests repeating same run N times and comparing hashes.
- Cross-platform CI matrix (Linux/macOS/Windows) to catch floating-point/ordering differences.

This keeps `carpyccio` reliable as a long-term, automation-native slicer backend.
