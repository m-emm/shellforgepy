# Declarative Builder Guide

The **builder** feature lets you define complex assemblies as data-first YAML
configuration instead of imperative Python orchestration code. You define:

- which assemblies exist,
- which resource file generates each assembly,
- parameter values (including computed values),
- inter-assembly dependencies,
- and post-build placement alignments.

This is useful for large machinery where many parts are reused and assembled in a
specific order.

## Why use the builder?

- **Declarative topology**: capture assembly relationships in YAML.
- **Deterministic caching**: parameter/dependency hashes skip unchanged rebuilds.
- **Composable pipelines**: couple build output to visualization/production workflows.
- **Readable reviews**: assembly diffs are mostly config changes.

## Core file layout

A typical project layout looks like this:

```text
my_project/
├── assemblies/
│   ├── assemblies.yaml
│   ├── machine_base.yaml
│   ├── guide_column.yaml
│   └── spindle_head.yaml
└── src/
    └── my_project/
        └── generators.py
```

`assemblies.yaml` is the orchestration file. Individual `*.yaml` resource files
define generators and their parameter contracts.

## Minimal orchestration file

```yaml
globals:
  machine_name: "mini-mill"

assemblies:
  - name: machine_base
    parameters:
      length: 260
      width: 180
      height: 24

  - name: guide_column
    depends_on: [machine_base]
    parameters:
      radius: 20
      height: 220

  - name: spindle_head
    depends_on: [guide_column]
    parameters:
      length: 90
      width: 70
      height: 55

placement:
  alignments:
    - part: guide_column
      to: machine_base
      alignment: CENTER
      axes: [0, 1]

    - part: guide_column
      to: machine_base
      alignment: STACK_TOP
      stack_gap: 0

    - part: spindle_head
      to: guide_column
      alignment: CENTER
      axes: [0, 1]

    - part: spindle_head
      to: guide_column
      alignment: STACK_TOP
      stack_gap: 0
```

Useful pattern for machine assemblies: chain multiple alignments for the same
part, e.g. first `CENTER` on `axes: [0, 1]` (X/Y centering), then `STACK_TOP`
for deterministic Z stacking.

## Dependency graph, injections, and placement execution

The builder coordinates three related mechanisms:

1. **Build dependency graph (`depends_on`)**
   - `depends_on` declares which assemblies must be built first.
   - This defines artifact availability order and hash dependency propagation.
   - If `B` depends on `A`, `A` is guaranteed to exist before `B` is generated.

2. **Part injection into generators (`inject_parts`)**
   - `inject_parts` passes already-built dependency artifacts into a generator
     function (for example `frame`, `bracket`, `leader`, or named followers).
   - This enables shape-aware generation: create features that fit/mate with
     upstream geometry (clearances, bolt patterns, matching interfaces).
   - Injection uses the dependency graph, so referenced provider assemblies must
     be resolvable from previously built artifacts.

3. **Placement sequence (`placement.alignments`)**
   - Placement actions are evaluated in configured order.
   - Each action executes as soon as the moving and target assemblies referenced
     by that step are available.
   - This allows incremental scene composition across dependency boundaries
     rather than requiring one monolithic final pass.

In practice, these mechanisms are complementary:
- `depends_on` controls **when** an assembly can be built,
- `inject_parts` controls **what geometry context** a generator can consume,
- `placement.alignments` controls **where built artifacts end up** in the shared
  scene.

### Multi-round alignment pattern

Multiple rounds of placement are valid and often necessary for machinery:

1. Roughly align Part A to Part B.
2. Build/inject Part C using A/B context so C becomes a joiner/bracket.
3. Reposition A+B+C together relative to Part D using later placement steps.

Because placement steps are ordered and dependency-aware, you can stage these
transformations declaratively instead of hard-coding a single final transform.

## Minimal assembly resource file

Each assembly resource file contains a generator declaration plus parameter
schema.

```yaml
ShellforgepyBuilderVersion: "2026-03-27"

Parameters:
  length:
    Type: Float
  width:
    Type: Float
  height:
    Type: Float

Parts:
  Main:
    Type: Shellforgepy::Assembly
    Properties:
      Generator: shellforgepy.simple.create_box
      Properties:
        length:
          $ref: length
        width:
          $ref: width
        height:
          $ref: height
```

## Running the builder

From your project root:

```bash
python -m shellforgepy.builder.builder assemblies/assemblies.yaml --verbose
```

Useful options:

- `--assembly <name>` build only selected assemblies.
- `--repository-dir <path>` choose artifact/cache location.
- `--force` rebuild even when hash-matched cache exists.
- `--visualize` / `--production` export scene outputs for a selected assembly.

## Production plates (optional)

When `Builder.Production.arrange` is used, you can now split output into
multiple **plates**. Each plate gets its own STL and process JSON, and the
workflow slices each plate independently (resulting in separate G-code files).

```yaml
Builder:
  Production:
    arrange:
      bed_width: 220
      prod_gap: 3
      # Optional: explicit/manual plate declaration
      plates:
        - name: plate_frame
          parts: [frame, frame_clamps]
        - name: plate_motion
          parts: [y_axis, x_axis]
      # Optional: place any parts not listed above on auto plates
      auto_assign_plates: true
```

Behavior notes:

- `plates` is optional.
- `auto_assign_plates` defaults to `false`.
- If `plates` is present and `auto_assign_plates` is `false`, every production
  part must be listed in a plate.
- If `auto_assign_plates` is `true`, remaining parts are assigned to additional
  auto-generated plates in build order.
- If neither option is used, behavior remains unchanged (single plate export).

## Working example in this repository

See the complete demo under:

- `examples/builder_machine_demo/assemblies.yaml`
- `examples/builder_machine_demo/machine_base.yaml`
- `examples/builder_machine_demo/guide_column.yaml`
- `examples/builder_machine_demo/spindle_head.yaml`
- `examples/builder_machine_example.py`

Run it:

```bash
python examples/builder_machine_example.py
```

The script builds a small machine-like stack (base + column + spindle head) via
builder YAML and prints generated artifact locations.
