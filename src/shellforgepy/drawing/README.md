# Construction drawing Stages 0-7

Stage 0 established the request and coordinate boundary for the SVG
construction-drawing pipeline. Stage 1 adds the first exact section
pinch-through for the plate fixture: four `<line>` elements and two analytic
`<circle>` elements. The real-assembly integration also emits exact circular
arcs as SVG arc commands where the machined mount requires them; unsupported
curve types remain rejected rather than tessellated. Stage 5 adds explicit
dimensions targeted by canonical ShellForgePy part references; a named cutter
can therefore provide a hole diameter without being visible in the drawing.
Circle-diameter callouts use an angled arrow, elbow, and horizontal text
landing so they remain readable on a technical sheet. Their measurement target
may remain hidden; their label layout always follows the visible section.
Stage 7 adds exact orthographic projections, including dashed hidden feature
edges, alongside the existing exact-section representation.

## Coordinate and metadata contract

- Model coordinates are millimetres.
- The default view is `top`: section normal `(0, 0, 1)`, drawing up `(0, 1, 0)`,
  and derived drawing right `(1, 0, 0)`.
- If no explicit origin is supplied, the origin is the combined selected-part
  bounding-box centre after assembly placement.
- Explicit views provide orthogonal `normal` and `up` vectors. `right` is
  derived as `up x normal`.
- The canonical in-memory and serialized representation is standard
  `xml.etree.ElementTree` SVG. Geometry belongs below the
  `g[data-shellforgepy-role="geometry"]` group. Each selected part gets a
  `section-contour` or `projection` group according to its representation.
- Application metadata uses `data-shellforgepy-*` attributes. Backend objects
  never cross this boundary.
- The geometry group applies one Y inversion transform so model-space positive
  Y remains drawing-up while SVG remains valid in its native coordinate system.

## Stage 0 backend matrix

| Capability | CadQuery | FreeCAD |
| --- | ---: | ---: |
| Build the plate fixture through the adapter bridge | yes | yes |
| Produce the existing STEP artifact | yes | yes |
| Construct and resolve a drawing request/view frame | backend independent | backend independent |
| Extract exact section edges and circles | implemented | implemented |
| Extract exact Stage 7 HLR projection | implemented | implemented |

The extractor uses the existing adapter seam and preserves exact geometry
provenance. It rejects unsupported curves instead of silently tessellating
them.

## Orthographic projections

Use an explicit per-view representation for an ordinary model projection:

```yaml
views:
  - id: top
    view: top
    representation:
      mode: projection
      include:
        - visible_outline
        - visible_feature_edges
        - hidden_feature_edges
  - id: front
    view: front
    representation:
      mode: projection
      include: [visible_outline, visible_feature_edges, hidden_feature_edges]
```

`visible_outline` is required. The default include list is
`visible_outline, visible_feature_edges`; hidden edges and tangent edges are
opt-in. Projection works from the final selected B-rep, not the original
cutters: a blind or through cut therefore appears wherever its resulting
feature faces are visible or requested as hidden geometry. Hidden primitives
are dashed and carry `data-shellforgepy-visibility="hidden"`; every projection
primitive also carries its role, source kind, and stable extractor-local source
reference. A projection rejects a nonzero `section_thickness`.

## Technical drawing sheets

Set `frame: technical` on a builder construction-drawing rule and provide an
optional A4 sheet specification:

```yaml
frame: technical
sheet:
  format: A4
  orientation: landscape
  margin: 10
  border: true
  title_block:
    title: Tool Head Mount (Machined)
    drawing_number: THM-MACHINED-BOTTOM
    revision: A
    material: Aluminum
    units: mm
    scale: "1:1"
    source: tool_head_mount_machined_bottom_assembly
```

The sheet renderer adds an A4 border, a framed view viewport, a metadata
frame, and a structured title block. Exact section elements remain in their
original drawing coordinates below the geometry group; only the group transform
fits and centers them in the sheet viewport. The source, view, units, scale,
and revision are also retained in the manifest/request data.

## Explicit dimensions

Construction-drawing annotations use the existing canonical part-reference
paths exposed by the builder and visualization GUI. The target selects the
source object; the operation selects the measurement. Stage 5 supports outer
X/Y extents and a circle-or-arc-only diameter callout:

```yaml
precision: 2
annotations:
  - id: overall_width
    operation: bounding_box_x_dimension
    target: tool_head_mount_machined_bottom_assembly.leader
    placement:
      alignments:
        - alignment: STACK_FRONT
          stack_gap: 8
  - id: left_front_hole_diameter
    operation: circle_diameter
    target: tool_head_mount_machined_bottom_assembly.cutters.hole_drill_LEFT_FRONT
    quantity: 14
    placement:
      alignments:
        - alignment: STACK_LEFT
          stack_gap: 6
```

The canonical target is measured and, for a circle, supplies the arrow
attachment point. Placement uses a small CAD-free 2D layout helper against the
exact bounds of the *visible* section geometry. It supports the planar
`Alignment` subset: `LEFT`, `RIGHT`, `FRONT`, `BACK`, `CENTER`, their `EDGE_*`
forms, and `STACK_LEFT`, `STACK_RIGHT`, `STACK_FRONT`, and `STACK_BACK`.
`FRONT` is drawing-down and `BACK` drawing-up in the default top view.
`TOP`/`BOTTOM` alignments are model-Z operations and are rejected here.

When placement is omitted, outer X dimensions go `STACK_FRONT`, outer Y
dimensions go `STACK_RIGHT`, and circle callouts go `STACK_BACK`, each with a
5 mm gap. This places measurement text outside the visible part by default.

`circle_diameter` can additionally describe an explicitly selected threaded or
through hole. These fields only control the displayed callout; the target still
provides the exact diameter from its 2D section geometry.

```yaml
  - id: blind_m3_thread
    operation: circle_diameter
    target: m3_threaded_plate.cutters.m3_blind_left
    quantity: 2
    diameter_tolerance: "±0.05"
    thread_size: M3
    thread_tolerance_class: 6H
    depth: 6
    placement:
      alignments:
        - alignment: STACK_BACK
          stack_gap: 8
        - alignment: STACK_LEFT
          stack_gap: 8
```

This renders the supplier-style two-line label `2 X M3 - 6H ↧ 6.00` and
`⌀2.50 ±0.05`. Use `through: true` instead of `depth` to render `THRU`.
`quantity` is author-supplied and does not infer a hole pattern.

Every `circle_diameter` callout uses one horizontal and one vertical standard
alignment. Horizontal choices are `LEFT`, `RIGHT`, their `EDGE_*` forms, and
their `STACK_*` forms; vertical choices are the matching `FRONT`/`BACK` forms.
The order is irrelevant. Missing directions default to `STACK_BACK` and
`STACK_RIGHT`, each with a 5 mm gap. A circle callout requires at least one
`STACK_*` direction, so its envelope remains outside the part on one axis.
This allows compact mixed placement such as right-aligned within the part span
and stacked below it:

```yaml
placement:
  alignments:
    - alignment: RIGHT
    - alignment: STACK_FRONT
      stack_gap: 8
```

`stack_gap` controls clearance for the stacking axis. Duplicate or conflicting
directions on either axis, and `CENTER`, are rejected.

The horizontal callout rule runs from the elbow outward. The elbow is always
one endpoint, never an intermediate point. Threaded callouts place the thread
specification above the rule and the diameter below it with a fixed 0.8 mm
visible-ink gap. Diameter-only and `THRU` callouts retain the same layout but
render only the lower diameter row. `*_LEFT` callouts extend their rule and
text left from the elbow; `*_RIGHT` callouts mirror this. This makes the
diagonal leader leave the text envelope toward the hole without crossing it.
The complete real-or-virtual two-row envelope is placed outside the visible
part bounds on every stacking axis. A normal or edge-aligned axis may remain
within the part span because the other stacking axis prevents overlap.

Circle arrow tips terminate on the measured rim along the radial line through
the hole centre. Arc callouts retain their analytic arc-contour attachment.
The marker reference is its triangle tip, so it does not extend through the
line. Linear dimension witness lines stop 0.8 mm short of the part, continue 1.5 mm past
the dimension line, and use a 0.12 mm stroke (the visible part contour uses
0.2 mm), keeping the drafting aids distinct from manufactured geometry.

`depth` and `through` are mutually exclusive, and
`thread_tolerance_class` requires `thread_size`. Callout fields are invalid on
the bounding-box dimension operations. A target remains valid when its exact
section is one circle, one arc, or CAD-split co-circular arc edges; mixed or
non-circular geometry is rejected.

The SVG dimension group and workflow manifest both retain the canonical target
path, operation, exact value, target bounds, visible layout bounds, and placed
annotation bounds.

## Linear dimensions between named projected parts

Stage 6 adds `linear_dimension` for two explicitly named projected envelopes.
It is deliberately not a CAD-edge selector. `RIGHT` accepts `EDGE_LEFT` and
`EDGE_RIGHT`, measures `to_x - from_x`, and requires the endpoints to be
left-to-right. `BACK` accepts `EDGE_FRONT` and `EDGE_BACK`, measures
`from_y - to_y`, and requires the endpoints to be back-to-front. The renderer
rejects a reversed pair instead of silently swapping it.

```yaml
  - id: left_web_width
    operation: linear_dimension
    from:
      target: tool_head_mount_machined_bottom_assembly.leader
      edge: EDGE_LEFT
    to:
      target: tool_head_mount_machined_bottom_assembly.non_production_parts.extruder_cutout_reference
      edge: EDGE_LEFT
    dimension_direction: RIGHT
    placement:
      alignments:
        - alignment: STACK_FRONT
          stack_gap: 6
```

Named non-production parts are valid endpoints even when their viewer metadata
marks them hidden by default; they are not added to the visible drawing
geometry. For horizontal dimensions on the same `STACK_FRONT` or `STACK_BACK`
side, shorter spans are nearest the visible part and longer spans move outward
by a fixed drawing-style pitch. This includes explicit outer X dimensions, so
an overall width remains outside the shorter web dimensions. SVG metadata and
the workflow manifest retain both endpoint paths, edge selectors, projected
coordinates, visible layout bounds, and final rule/label bounds.

Run the M3 tapped-hole technical-sheet example with:

```bash
python examples/construction_drawing_m3_tapped_holes.py
```

## Multiple views on one sheet

Use `views` instead of the legacy root-level `view` and `annotations` fields
when one drawing contains multiple sections. Each view has a stable `id`, its
own section frame, and only the annotations that belong to that section:

```yaml
name: m3_threaded_plate
parts:
  - source: self
    artifact: leader
    name: m3_threaded_plate
views:
  - id: top
    view:
      normal: [0, 0, 1]
      up: [0, 1, 0]
      origin: [40, 25, 6]
    annotations: []
  - id: front
    view:
      normal: [0, -1, 0]
      up: [0, 0, 1]
      origin: [40, 0, 4]
    annotations:
      - id: front_tap
        operation: circle_diameter
        target: m3_threaded_plate.cutters.m3_front_center
        thread_size: M3
        thread_tolerance_class: 6H
        depth: 5
```

The first view is centred as the drawing origin. Each later view is placed
relative to the preceding view with the same 2D `placement.alignments` syntax
as annotations. The default is `STACK_RIGHT` with a 12 mm gap, so views lay
out left to right. Use `STACK_LEFT`, `STACK_FRONT`, `STACK_BACK`, or the other
planar alignments when another orthographic relationship is needed:

```yaml
- id: front
  placement:
    alignments:
      - alignment: STACK_RIGHT
        stack_gap: 16
  view: front
```

Views share one technical-sheet scale and each annotation is measured and laid
out only in its owning view's section frame. Technical sheets also reserve a
5 mm white `drawing_margin` between all drawing content and the viewport frame;
override `sheet.drawing_margin` only when a different clear border is needed.

Technical sheets select the largest fitting scale from the `1:1`, `1:2`,
`1:5`, `1:10`, `1:20`, `1:50`, `1:100`, … series. The SVG metadata, workflow
manifest, sheet metadata, and title block all report the effective scale as
`1:n (1 mm = n mm)`; a manually supplied title-block scale is not used.
