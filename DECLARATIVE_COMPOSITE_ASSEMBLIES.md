# Declarative Composite Assemblies

## Goal

Add a third assembly mode to the builder:

- Python generator assemblies: full freedom, but require Python glue code
- `Builder.Collection: true` assemblies: declarative scene aggregations, mainly for visualization/export
- declarative composite assemblies: declarative `LeaderFollowersCuttersPart` builders assembled from injected assemblies

The new mode should let a resource file describe a real composite assembly in YAML, without a custom generator function, while still producing a normal build artifact that:

- is cached like any other assembly
- enters the build memory pool like any other assembly
- can be injected into downstream assemblies
- participates in global placement like any other assembly

The current prototype in `tool_head_assembly.yaml` is directionally correct. The main missing pieces are selector normalization, forwarding semantics for all artifact groups, and explicit integration with the existing dependency and placement model.

## Motivation

There is currently a gap between two useful builder patterns.

### 1. Python generators are powerful but too heavy for simple assembly glue

Many subassemblies are not algorithmic geometry problems. They are just:

- fuse these leaders
- forward these followers
- keep these cutters
- forward these non-production parts with safe names

Writing a Python function for that is repetitive and obscures the intent.

### 2. `Collection: true` is useful, but it is scene-only

`whole_printer_assembly.yaml` is a good example of the current collection pattern:

- it gathers already-built assemblies
- it is excellent for visualization and export composition
- but it does not define a canonical `LeaderFollowersCuttersPart`

That means a collection assembly is not the right tool when downstream geometry needs to inject the combined result.

### 3. Real assembly composition should be declarative too

Assemblies such as `tool_head_left_assembly` and `tool_head_right_assembly` are structurally composed from already-built assemblies. They should be expressible as:

- dependency declarations in top-level `assemblies.yaml`
- injected source assemblies
- a YAML description of how the resulting composite is formed

without introducing a new Python file whose only purpose is "fuse A and B, then forward C".

## Proposed YAML Model

### Resource-file syntax

The resource file keeps the normal `Parts.<LogicalName>.Type: Shellforgepy::Assembly` shape, but replaces `Properties.Generator` with `Properties.Composite`.

```yaml
Parts:
  ToolHeadAssembly:
    Type: Shellforgepy::Assembly

    Properties:
      Composite:
        Leader:
          Fused:
            - source: injected
              assembly: nitehawk_holder
              artifact: leader
            - source: injected
              assembly: part_fans
              artifact: leader

        Followers:
          - source: injected
            assembly: part_fans
            artifact: followers
            names:
              - blower_ducts

        Cutters: []

        NonProductionParts:
          - source: injected
            assembly: nitehawk_holder
            artifact: non_production_parts
            name_template: "nitehawk_holder_{name}"
          - source: injected
            assembly: part_fans
            artifact: non_production_parts
            name_template: "part_fans_{name}"
```

This is close to the current draft and can stay the v1 syntax.

### Top-level `assemblies.yaml` stays authoritative for dependencies

Declarative composites should still use top-level `depends_on` and `inject_parts`.

```yaml
- name: tool_head_left_assembly
  resource_file: tool_head_assembly.yaml
  depends_on:
    - sprite_extruder_left_assembly
    - nitehawk_holder_left_assembly
    - part_fan_left_assembly
  inject_parts:
    sprite_extruder: sprite_extruder_left_assembly
    nitehawk_holder: nitehawk_holder_left_assembly
    part_fans: part_fan_left_assembly
```

This is important. The resource file describes how to compose the result; the top-level assembly entry still describes where the inputs come from in the global build graph.

### Selector object

Each entry inside `Leader.Fused`, `Followers`, `Cutters`, and `NonProductionParts` is a selector mapping.

### Required fields

- `source`: only `injected` in v1
- `assembly`: the injected alias from top-level `inject_parts`
- `artifact`: what to read from that injected assembly

### Supported `artifact` values in v1

- `leader`
- `fused`
- `followers`
- `followers.<name>`
- `cutters`
- `cutters.<name>`
- `non_production_parts`
- `non_production_parts.<name>`

`all` should not be supported in v1 for composites. It is convenient for scene rules, but too ambiguous for canonical assembly construction because it flattens multiple semantic groups and omits cutters.

### Optional selector fields

- `names`: include only these named members from a multi-entry artifact group
- `exclude_names`: exclude these named members from a multi-entry artifact group
- `name`: rename a single selected part
- `name_template`: rename each selected named part

Recommended `name_template` variables:

- `{inject_name}`: the alias from `inject_parts`
- `{assembly_name}`: the real provider assembly name
- `{artifact}`: the artifact group name
- `{name}`: the original source part name
- `{index}`: source index within that artifact group

The important difference between `{inject_name}` and `{assembly_name}` is reuse. For `tool_head_left_assembly` and `tool_head_right_assembly`, `{inject_name}` stays stable while the provider assembly name changes.

### Channel semantics

### `Leader.Fused`

- all resolved parts are copied and fused into the new leader
- order is the YAML order after selector expansion
- the resolved set must not be empty

### `Followers`

- all resolved parts are appended to the composite's `followers` list in selector-expansion order
- if a source follower has a name, that name is also registered in `follower_indices_by_name`
- if a source follower is unnamed, it remains addressable only by position, just like today
- `name` may be used when exactly one part is selected and assigns a new addressable follower name
- `name_template` rewrites each selected source name; it is intended for named parts and should be an error for unnamed entries unless `{name}` is not used and the builder can still produce a deterministic name
- final follower names must be unique within followers and must also not collide with cutter or non-production names, matching current `LeaderFollowersCuttersPart` rules

### `Cutters`

- all resolved parts are appended to the composite's `cutters` list in selector-expansion order
- if a source cutter has a name, that name is also registered in `cutter_indices_by_name`
- if a source cutter is unnamed, it remains positional only
- `name` and `name_template` follow the same rules as for followers
- final cutter names must be unique within cutters and must not collide with follower or non-production names

### `NonProductionParts`

- all resolved parts are appended to the composite's `non_production_parts` list in selector-expansion order
- if a source non-production part has a name, that name is also registered in `non_production_indices_by_name`
- if a source non-production part is unnamed, it remains positional only
- `name` and `name_template` follow the same rules as for followers
- final non-production names must be unique within non-production parts and must not collide with follower or cutter names

### Validation rules

V1 should keep the model strict.

- `Composite` and `Generator` are mutually exclusive
- `Composite` and `Builder.Collection: true` are mutually exclusive
- every selector must use `source: injected`
- every referenced injected alias must exist in top-level `inject_parts`
- v1 `inject_parts` entries for composite resources should inject full assemblies, not preselected artifacts
- `name` is only valid when the selector resolves exactly one part
- unresolved selectors are errors, not silent no-ops
- name collisions after rename/prefix processing are errors

The "inject full assemblies" rule is important. The resource file is where canonical part selection now lives. Top-level `inject_parts.foo: some_assembly.leader` would split that responsibility across two layers and make reuse much harder.

## Suggested implementation approach

### 1. Parse and normalize `Properties.Composite` in `builder.py`

`_resolve_assembly(...)` already detects `Properties.Composite`, but the current prototype only extracts a `_to_fuse` mapping for leader inputs.

That should be replaced by a normalized composite spec, for example:

- `_composite_spec["leader_fused"]`
- `_composite_spec["followers"]`
- `_composite_spec["cutters"]`
- `_composite_spec["non_production_parts"]`

Each channel should contain the same normalized selector objects so the runtime builder does not need to reason about raw YAML shapes.

Recommended changes:

- add a `_normalize_composite_spec(...)` helper
- validate mutual exclusion with `Generator`
- validate mutual exclusion with `Builder.Collection: true`
- keep the internal generator path as `_build_composite_assembly`
- store the normalized spec in `generator_kwargs`, for hashing and metadata

### 2. Replace the `_build_composite_assembly(...)` stub with a real builder

The current stub only checks `_to_fuse` and returns nothing.

The real implementation should:

1. read `_composite_spec`
2. resolve each selector against the injected runtime assemblies passed in `kwargs`
3. build a new `LeaderFollowersCuttersPart`
4. fuse leader parts into the new leader
5. forward followers, cutters, and non-production parts
6. apply rename rules
7. return the finished composite

This should operate on the injected runtime assemblies that already came through `_resolve_dependency_injections(...)`, not on raw metadata files. That way the composite sees exactly the same build-time placement state that a Python generator would see.

### 3. Add a builder-side selector resolver for injected assemblies

The builder already has good metadata-based helpers for scene materialization:

- `_artifact_entries_for_selector(...)`
- `_filter_artifact_entries(...)`

Declarative composites need the runtime equivalent, but for injected `LeaderFollowersCuttersPart` objects rather than STEP metadata. A good shape would be something like:

- `_composite_entries_for_injected_assembly(...)`
- `_filter_composite_entries(...)`
- `_rename_composite_entries(...)`

Each resolved entry should carry:

- `part`
- `inject_name`
- `assembly_name`
- `artifact`
- `name`
- `index`

so renaming and diagnostics stay informative.

### 4. Add a small helper surface to `leader_followers_cutters_part.py`

This feature can be implemented entirely in the builder, but a cleaner implementation would expose small builder-facing helpers from `LeaderFollowersCuttersPart` so the builder does not keep reaching into:

- `.followers`
- `.cutters`
- `.non_production_parts`
- `.follower_indices_by_name`
- `.cutter_indices_by_name`
- `.non_production_indices_by_name`

Helpful additions would be:

- a way to iterate a named artifact group as `(index, name, part)` entries
- a way to copy selected entries from one group
- possibly a generic "append to group with optional name" helper

`prefixed_copy(...)` is already useful for whole-assembly renaming, but declarative composites need finer-grained selection and forwarding than whole-assembly prefixing.

### 5. Keep artifact export and metadata format unchanged

Once `_build_composite_assembly(...)` returns a `LeaderFollowersCuttersPart`, the existing pipeline already knows how to:

- normalize it
- export leader/fused/follower/cutter/non-production artifacts
- write metadata
- insert the result into the build pool

So the export schema does not need a new artifact type. The new feature should fit into the existing canonical artifact layout.

The metadata should naturally record:

- `generator = shellforgepy.builder.builder._build_composite_assembly`
- normalized composite spec inside `generator_kwargs`

That keeps caching and debugging consistent with normal generators.

### 6. Keep dependency graph ownership in top-level config

No new graph category is required if top-level `inject_parts` remains authoritative.

That means `graph_model.py` can continue to infer:

- declared dependency edges
- injection edges
- placement-build edges

without understanding composite internals in detail.

The one recommended addition is validation: when a resolved assembly uses `Properties.Composite`, the builder should verify that every referenced injected alias is actually present in the top-level assembly entry's `inject_parts`.

### 7. Add focused tests

The implementation should come with tests for:

- parsing and validation of `Properties.Composite`
- leader fusion from multiple injected assemblies
- forwarding followers/cutters/non-production parts
- `name` and `name_template` behavior
- collision detection
- composite assemblies being injectable downstream
- placement-sensitive rebuilds when injected providers move before consumer build

## Interplay with sequential placement

Declarative composites should not introduce a second placement system.

They should reuse the existing build-time placement model exactly as-is:

- dependencies are built in graph order
- the placement executor advances as a sequential blocking cursor
- injected provider assemblies are imported through `_resolve_dependency_injections(...)`
- those injected providers already carry the current build-time placement prefix

So the rule is simple:

> a declarative composite sees the exact same placed state of its injected inputs that a Python generator would see at the same build moment

This is the correct behavior.

If the global placement cursor is currently blocked before some later transform of an injected provider, the composite must not "look ahead" and use that future placement. It should only consume the maximal executable prefix.

That keeps declarative composites aligned with the pure sequential placement model described in `PURE_SEQUENTIAL_PLACEMENT_CONCEPT.md`.

## Interplay with dependency mechanics

### Injection edges

Because declarative composites still use top-level `inject_parts`, they automatically participate in the same injection graph as Python generators.

That gives us:

- provider-before-consumer build ordering
- explicit provenance in metadata
- the same downstream cache invalidation semantics

### Placement-derived build dependencies

The existing placement-build dependency logic keys off injected provider assemblies and the consumer's first placement involvement.

That is exactly what declarative composites need.

No special composite-only placement graph is required, because the builder only needs to know:

- which assemblies are injected into the consumer
- which placement prefix must have executed before the consumer builds

The fact that the consumer is assembled by YAML rather than Python should be irrelevant at graph level.

### Build memory pool

After a declarative composite is built and exported, it should be inserted into the build pool exactly like a Python-generated assembly.

That means downstream consumers can inject it using the normal forms:

- `<assembly>`
- `<assembly>.leader`
- `<assembly>.fused`
- `<assembly>.followers.<name>`
- `<assembly>.cutters.<name>`
- `<assembly>.non_production_parts.<name>`

This is one of the main reasons declarative composites are not just "better collections". They produce canonical build artifacts, not only scene composition.

### Cache keys

The current dependency hash already includes:

- the effective generator identity
- injected source parameter hash
- injected source placement offset
- injected source placement transform records

That means declarative composites should automatically rebuild when:

- the composite definition itself changes
- one of their injected providers changes
- or a provider's effective build-time placement prefix changes

The subtle point is the resource file.

For normal Python-backed assemblies, using the full resource file hash is reasonable because the resource usually defines the generator path and the generator arguments. For declarative composites, that is too broad. A change to:

- `Builder.Visualization`
- `Builder.Production`
- preview settings
- process-data presets or overrides
- arrange/export options

should not invalidate the canonical built composite artifact.

So for declarative composites, the cache key should not be driven by the full resource YAML. It should be driven only by the build-relevant declarative program.

The correct model is to treat the normalized `Parts.<LogicalPart>.Properties.Composite` subtree as the "generator program" for cache purposes.

That means the cache key for a declarative composite should include:

- the internal generator identity, for example `shellforgepy.builder.builder._build_composite_assembly`
- the normalized composite spec
- any resolved parameter/context values that are actually consumed while normalizing that composite spec
- the injected dependency hashes and placement records, as today

It should explicitly not include unrelated resource-file sections such as:

- `Builder.Visualization`
- `Builder.Production`
- process-data declarations
- preview configuration
- scene-only export and arrangement rules

The important detail is normalization. The hash should not depend on raw YAML formatting, comments, or key ordering. It should depend on semantic content after validation and normalization, for example:

- leader fused selectors
- follower selectors
- cutter selectors
- non-production selectors
- referenced injected aliases
- rename rules such as `name` and `name_template`
- include/exclude filters such as `names` and `exclude_names`

In implementation terms, this argues for a composite-specific version input instead of the generic full-resource SHA.

Recommended approach:

1. Parse `Properties.Composite`.
2. Resolve any inline references that are allowed there.
3. Normalize the result into a canonical in-memory spec.
4. Serialize that normalized spec in a stable form, for example sorted JSON.
5. Hash that stable serialization into something like `composite_spec_sha256`.
6. Feed that `composite_spec_sha256` into the assembly cache key.

That gives the correct rebuild behavior:

- changing `Composite.Leader.Fused` rebuilds
- changing selector filters or renaming rules rebuilds
- changing only visualization or production/export rules does not rebuild
- changing an injected upstream assembly rebuilds
- changing an injected upstream placement history rebuilds

So the design should distinguish two concepts:

- resource file: one YAML document that may contain both build rules and scene/export rules
- composite program: only the normalized `Composite` subtree that defines the built artifact

Only the second one belongs in the declarative composite cache key.

### Interplay with collection assemblies and scene rules

Declarative composites should not replace `Collection: true`.

The two features serve different purposes:

- collection assembly: scene composition and export orchestration
- declarative composite assembly: canonical composite artifact generation

`whole_printer_assembly.yaml` should remain a collection-style resource.

`tool_head_assembly.yaml` is a good example of the new composite-style resource, because it wants to create a real build artifact while still keeping independent scene rules for:

- visualization
- production arrangement
- scene-only injected extras such as `sprite_extruder`

That separation is important:

- `Properties.Composite` defines the canonical built assembly
- `Builder.Visualization` and `Builder.Production` still define how that assembly is shown or exported in scenes

## Recommended v1 constraints

To keep the first version understandable, the feature should intentionally not try to solve everything.

Recommended v1 limits:

- `source: injected` only
- no `artifact: all`
- no per-selector local transform mini-language
- no nested placement program inside `Composite`
- no automatic dependency inference from resource-file selectors
- no implicit collision resolution beyond explicit rename rules

This keeps declarative composites a thin, predictable layer on top of the existing builder architecture.

## Summary

The feature should be treated as "declarative generator wiring", not as a scene helper.

The design that best fits the current builder is:

- keep top-level `depends_on` and `inject_parts` authoritative
- add a normalized `Properties.Composite` spec in assembly resource files
- resolve selectors against injected runtime assemblies
- build a real `LeaderFollowersCuttersPart`
- let the existing caching, placement, export, and build-pool systems handle the rest

If implemented that way, declarative composites become a natural bridge between:

- low-ceremony YAML composition
- real canonical build artifacts
- and the sequential placement/dependency model already being established in the builder
