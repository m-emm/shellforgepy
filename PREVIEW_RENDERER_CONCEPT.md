# Low-Dependency Preview Renderer Concept

## Goal

Add a proper headless preview-rendering subsystem to ShellForgePy that can render exported geometry such as OBJ scenes into PNG images without requiring:

- a GUI
- OpenGL
- Qt
- Blender
- VTK
- heavyweight mesh/graphics stacks
- native C++ build steps during normal installation

The renderer should be:

- batch-friendly
- CI-friendly
- deterministic enough for tests
- fast enough for routine preview generation
- useful both for G-code thumbnails and for general geometry inspection

Typical outputs should include:

- isometric preview images
- top/front/side orthographic views
- multi-view contact sheets
- per-plate production previews
- inspection renders for builder/export workflows

## Why this belongs in ShellForgePy

ShellForgePy already produces tessellated mesh data and already exports colored OBJ scenes.
It also already has a simple STL preview rasterizer for workflow preview generation.

So the missing piece is not "can we get geometry into a mesh form?".
The missing piece is:

- a reusable mesh-scene renderer
- a small OBJ/MTL loader for ShellForgePy-exported scenes
- a stable API that other subsystems can call

Bringing that into ShellForgePy proper would let us render previews for:

- exported OBJ scenes from `arrange_and_export`
- production plate arrangements
- builder-generated assemblies
- workflow-generated assets during slicing/upload
- test fixtures and documentation artifacts

## Non-goals

This renderer should not try to become a full real-time graphics engine.

It should not target:

- interactive viewport manipulation
- GPU rendering
- physically based rendering
- complex material systems
- texture mapping
- transparency-heavy scenes
- animation playback

The focus is high-signal inspection images for CAD/export workflows.

## Design principles

### 1. NumPy-first baseline

The baseline implementation should run with dependencies ShellForgePy effectively already has:

- `numpy`
- optionally `Pillow` for PNG output

This baseline should produce correct images even if it is not maximally fast.

### 2. Optional acceleration

`numba` should be supported as an optional accelerator, not as a required install dependency.

That gives us:

- good out-of-the-box portability
- no hard JIT/runtime dependency for all users
- a clear performance upgrade path for large scenes

Recommended packaging direction:

- core install: no `numba`
- extra: `shellforgepy[render]` or `shellforgepy[preview]` installs `Pillow`
- extra: `shellforgepy[render-fast]` installs `Pillow` and `numba`

### 3. Headless by construction

The renderer should never depend on a display server, windowing toolkit, or graphics context.

### 4. Mesh-scene focused

The rendering input should be mesh scene data, not CAD kernel objects.

That keeps the renderer:

- backend-agnostic
- simple to test
- usable on exported OBJ files
- usable from cached tessellation payloads

### 5. Deterministic enough for CI

Rendering should be stable enough that tests can validate:

- output dimensions
- visibility
- approximate color distribution
- camera framing
- metadata-driven scene selection

We should avoid highly nondeterministic shading behavior.

## Proposed scope

### Phase 1

Provide a small, solid rendering core for triangle meshes:

- orthographic cameras
- simple perspective camera support if needed later
- flat or simple diffuse shading
- solid per-object color
- z-buffer rasterization
- PNG output
- white or configurable background
- top/front/side/isometric camera presets

Input sources:

- ShellForgePy-exported OBJ + MTL
- direct mesh arrays `(vertices, faces, color)`

### Phase 2

Add scene-level convenience features:

- auto framing from scene bounding box
- shadows as optional higher-quality mode
- contact sheets of multiple views
- render selected subassemblies from OBJ metadata
- batch rendering helpers for whole run directories

### Phase 3

Integrate more deeply with existing workflows:

- workflow preview generation from OBJ instead of STL when available
- plate-specific previews in manifests
- builder/export inspection tools
- docs/test snapshot utilities

## Why OBJ should be the main scene input

ShellForgePy already writes:

- one or more objects
- per-object materials
- diffuse material colors via `Kd`
- metadata comments such as hierarchy and builder selectors

This makes OBJ a strong preview source because it is:

- already generated
- backend-independent
- lightweight to parse
- human-inspectable
- suitable for caching and reuse

For preview rendering, OBJ is a better scene interchange format than trying to render CAD solids directly inside the renderer.

## Proposed package layout

Suggested new package:

```text
src/shellforgepy/render/
    __init__.py
    model.py
    camera.py
    scene.py
    lights.py
    obj_loader.py
    mtl_loader.py
    rasterizer.py
    image.py
    presets.py
    api.py
```

Possible responsibilities:

- `model.py`
  - lightweight mesh and scene dataclasses
- `camera.py`
  - orthographic camera math
- `scene.py`
  - scene bounding boxes and object grouping
- `lights.py`
  - simple directional and point light models
- `obj_loader.py`
  - parse ShellForgePy-style OBJ files
- `mtl_loader.py`
  - parse minimal MTL subset needed for color previews
- `rasterizer.py`
  - z-buffer triangle rasterization
- `image.py`
  - PNG writing and contact sheet helpers
- `presets.py`
  - top/front/side/isometric presets
- `api.py`
  - high-level public helpers

## Proposed public API

High-level functions should be very simple to call.

Examples:

```python
from shellforgepy.render.api import render_obj_to_png

render_obj_to_png(
    obj_path="assembly.obj",
    output_path="assembly_iso.png",
    view="isometric",
    width=1024,
    height=1024,
)
```

```python
from shellforgepy.render.api import render_obj_views

render_obj_views(
    obj_path="assembly.obj",
    output_dir="previews",
    views=["top", "front", "side", "isometric"],
    width=768,
    height=768,
)
```

```python
from shellforgepy.render.api import render_mesh_scene_to_png

render_mesh_scene_to_png(
    scene=my_scene,
    output_path="preview.png",
    view="top",
)
```

Potential lower-level API:

```python
scene = load_obj_scene("assembly.obj")
camera = make_camera_preset("isometric", scene.bounds())
image = render_scene(scene, camera, width=1024, height=1024)
image.save("assembly.png")
```

## Scene model

The renderer should operate on a simple internal scene graph:

```python
Scene(
    objects=[
        MeshObject(
            name="part_a",
            vertices=np.ndarray,   # (n, 3)
            faces=np.ndarray,      # (m, 3)
            color=(r, g, b),
            metadata={...},
        ),
    ]
)
```

Key point:

- one mesh object corresponds naturally to one OBJ object/material section
- metadata from ShellForgePy OBJ comments can be preserved for filtering and view selection

Useful metadata to carry through:

- object name
- material name
- hierarchy path
- hierarchy labels
- builder selector

## OBJ/MTL support target

We do not need a fully general OBJ implementation at first.
We only need a robust parser for the subset that ShellForgePy exports.

### OBJ subset

- `mtllib`
- `o`
- `v`
- `usemtl`
- `f`
- comments

Initially, face support can be limited to:

- triangles
- triangle-only scenes written by ShellForgePy

If we later need broader compatibility, we can add:

- polygon triangulation
- `g`
- `vn`
- `vt`

### MTL subset

- `newmtl`
- `Kd`
- optionally `Ka`
- ignore unsupported fields safely

This is enough for material color reproduction in previews.

## Rasterization strategy

### Baseline path

Use a pure-NumPy implementation for:

- camera transform
- projection
- screen mapping
- z-buffer setup
- triangle rasterization
- flat or interpolated color

This should be optimized for simplicity and reliability first.

### Accelerated path

If `numba` is installed:

- compile the triangle raster kernel
- reuse the same API and data model
- accelerate the hottest loops only

The important architectural rule is:

- one code path in terms of features
- two execution modes underneath

That avoids feature drift between "slow" and "fast" renderers.

## Shading model

The shading model should stay intentionally simple.

Recommended initial modes:

- `flat`
  - color only, no lighting
- `diffuse`
  - ambient + one or two directional lights
- `diffuse_shadows`
  - optional higher-quality mode if worth the cost

For many inspection tasks, even `flat` or lightly shaded `diffuse` is enough.
The main value comes from:

- correct silhouette
- part separation
- color preservation
- repeatable framing

## Camera model

Orthographic rendering should be the default because it is ideal for technical previews.

Recommended built-in presets:

- `top`
- `bottom`
- `front`
- `back`
- `left`
- `right`
- `isometric`

Auto-framing should:

- fit the full scene bounding box
- add configurable margin
- preserve aspect ratio
- keep output stable across resolutions

Optional later support:

- perspective isometric
- custom eye/target/up camera creation

## Background and image conventions

Suggested defaults:

- white or light neutral background
- no alpha by default
- sRGB 8-bit PNG output
- fixed deterministic tone mapping

Optional additions:

- transparent background
- bed outline overlay
- axis indicator
- simple scale bar
- object labels for debugging

## Integration with current workflow code

Today ShellForgePy already contains:

- OBJ export in `produce/obj_file_export.py`
- a simple STL preview renderer in `workflow/preview_generator.py`
- G-code thumbnail embedding in `workflow/upload_to_printer.py`

The clean direction is to evolve `workflow/preview_generator.py` into a higher-level preview entry point backed by the new renderer package.

Possible end state:

- if OBJ exists, render OBJ preview
- otherwise, fall back to STL preview
- if a custom render script is configured, let it override both

That gives a compatibility ladder rather than a disruptive replacement.

## Proposed implementation path

### Step 1: establish renderer package

Add a small renderer package under `src/shellforgepy/render/` with:

- scene dataclasses
- basic camera presets
- minimal OBJ/MTL loader
- pure-NumPy z-buffer renderer
- Pillow-based PNG writer

### Step 2: support ShellForgePy-exported OBJ scenes

Render a ShellForgePy OBJ into:

- one PNG
- or a set of named views

This is the main milestone.

### Step 3: add optional `numba`

Wrap the hot rasterization kernel so it uses:

- pure Python/NumPy fallback if `numba` is absent
- JIT-accelerated execution if `numba` is present

Import pattern should be defensive:

```python
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
```

The module should remain importable without `numba`.

### Step 4: migrate workflow preview generation

Update preview generation so it can:

- render from OBJ when `manifest["obj_path"]` is available
- preserve the existing STL fallback
- write the same `*_preview.png` artifact used by G-code embedding

### Step 5: add multi-view helpers

Provide helpers for:

- top/front/side/isometric batches
- contact sheets
- CI artifact generation

### Step 6: expose as CLI utility

Potential CLI:

```bash
python -m shellforgepy.render.render_obj assembly.obj out.png --view isometric
python -m shellforgepy.render.render_obj assembly.obj previews/ --views top front side isometric
```

This would be very useful for CI and scripting.

## Dependency recommendation

### Core recommendation

Keep the preview renderer low-dependency by default:

- require `numpy`
- do not require `numba`
- use `Pillow` for PNG output, ideally as a dedicated render extra if we want to keep base installs lean

### Suggested extras

```ini
[options.extras_require]
render =
    Pillow

render-fast =
    Pillow
    numba
```

If we want image output to be universally available, `Pillow` could also be promoted to a normal dependency.
That is a project-level tradeoff:

- adding `Pillow` to base install makes rendering and thumbnail embedding simpler
- keeping it optional keeps the base package smaller

Given that `upload_to_printer.py` already uses Pillow for thumbnail embedding, there is a good argument that Pillow is already part of the practical workflow surface.

## Quality bar

The renderer does not need photorealism.
It does need to be good enough that a user can quickly answer:

- Did the part export correctly?
- Is the production arrangement correct?
- Are all parts present?
- Is the orientation right?
- Are the colors/material groups correct?
- Did the builder produce the expected subassembly?

A good preview renderer for ShellForgePy should optimize for:

- legibility
- consistency
- speed
- low operational friction

## Testing strategy

Tests should avoid brittle pixel-perfect snapshots as the only validation.

Recommended coverage:

- OBJ parser reads ShellForgePy-exported files correctly
- MTL colors map correctly
- camera presets frame simple scenes correctly
- renderer produces non-empty images of expected dimensions
- selected objects appear in expected regions of the image
- optional `numba` mode matches baseline mode within a tolerance

Useful fixture scenes:

- single cube
- two-color two-object assembly
- production plate arrangement
- tall object for front/side view coverage

For a small number of canonical fixtures, image-based golden tests may still be worthwhile if they use tolerant comparisons.

## Risks and tradeoffs

### 1. Performance on very large scenes

Pure-NumPy may be slower than desired for large triangle counts.

Mitigation:

- start with correct low-dependency baseline
- add optional `numba`
- reuse tessellation caches where available

### 2. Scope creep

A renderer can easily grow into a mini graphics engine.

Mitigation:

- stay focused on preview rendering
- keep materials simple
- prefer orthographic and deterministic outputs

### 3. Feature drift from workshop renderer

The workshop renderer already has interesting capabilities such as deferred shading and shadows.

Mitigation:

- treat it as inspiration, not a drop-in transplant
- import the ideas that improve output quality
- simplify the API and dependency surface for ShellForgePy proper

### 4. Dependency creep

Adding too much rendering infrastructure would undermine the goal.

Mitigation:

- make `numba` optional
- avoid GUI or OpenGL dependencies entirely
- keep the file-format support narrow and practical

## Recommendation

ShellForgePy should add a dedicated headless preview renderer package with the following stance:

- mesh-scene based
- OBJ-first for exported scene previews
- NumPy baseline
- Pillow for PNG output
- optional `numba` acceleration
- orthographic multi-view support as a first-class feature

This would give ShellForgePy a powerful new inspection capability that is:

- lightweight
- automation-friendly
- useful in workflows beyond G-code thumbnails
- aligned with the existing OBJ export architecture

## Concrete first milestone

The best first milestone is modest and high value:

1. Add `shellforgepy.render` with scene model, OBJ/MTL loader, and simple orthographic renderer.
2. Render ShellForgePy-exported OBJ files into `top`, `front`, `side`, and `isometric` PNG previews.
3. Wire workflow preview generation to prefer OBJ previews when available.
4. Keep STL preview generation as fallback.
5. Add optional `numba` acceleration only after correctness and API shape are settled.

If this milestone is successful, ShellForgePy gains a general-purpose, fully headless preview pipeline that can be used in local scripts, batch jobs, CI, and printer-upload workflows without dragging in heavyweight graphics tooling.
