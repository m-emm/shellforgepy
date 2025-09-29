# shellforgepy

Python tooling for the **ShellForge** workflow: model geometry in pure Python, pick a
CAD backend at runtime, and export parts ready for fabrication. The package
provides a layered architecture — from NumPy-based geometry utilities through
alignment-centric construction helpers and production-focused exporters — with
optional adapters for CadQuery and FreeCAD.

## 🚀 **Try the Examples!**

**Get started in 30 seconds:**

```bash
# Beginner-friendly parametric CAD
python examples/filleted_boxes_example.py

# Amazing mathematical surfaces
python examples/mobius_strip.py

# All 8 examples ready to run!
```

**[👉 See All Examples →](examples/README.md)** | **[🎯 Quick Examples Guide →](#examples)**

---

## Why ShellForgePy is Awesome? 🤯

### 🔥 **Unique Value Propositions**

**🚀 Mathematical Surfaces Made Easy**
- Create **Möbius strips** (one-sided surfaces!) with coordinate transformations
- Generate **helical coils**, **conical spirals**, and **sine wave channels**
- Build **organic face meshes** with automatic partitioning for multi-part printing
- All with pure Python math - no CAD expertise required!

**🎯 Smart CAD Independence**
- **Write once, run anywhere**: Your geometry code works with CadQuery OR FreeCAD
- **No vendor lock-in**: Switch CAD backends without changing your design logic
- **Graceful degradation**: Auto-detects available backends with helpful error messages
- **Future-proof**: Add new CAD adapters without breaking existing code

**⚡ Production-Ready Workflows**
- **Automatic part arrangement** on build plates (200x200mm, custom sizes)
- **STL export with process data** for 3D printing (PLA settings, layer heights)
- **Multi-part splitting** for large models that exceed printer bed sizes
- **Shell generation** with customizable thickness for hollow prints

**🧠 Advanced Mesh Processing**
- **Partitionable spheroid meshes** for complex organic shapes
- **Mesh partitioning** along arbitrary planes (front/back mask splitting)
- **Connector hint generation** for automatic assembly joints
- **Face-vertex maps** with proper STL export and mesh merging

### 🔬 **Technical Superpowers**

**📐 Alignment-First Design Philosophy**
- Position parts with semantic alignment (`CENTER`, `TOP`, `LEFT`) instead of coordinates
- **Predictable transformations**: Translate, rotate, and fuse with mathematical precision
- **Named parts tracking** for complex assemblies with automatic reconstruction
- **Leader-follower-cutter** hierarchies for advanced part relationships

**🛠️ Extensible Architecture**
```
geometry/     ← Pure NumPy/SciPy math (fibonacci spheres, trapezoidal snakes)
construct/    ← Alignment operations & part composition
produce/      ← Fabrication-ready export & arrangement
adapters/     ← Pluggable CAD backends (CadQuery, FreeCAD, future...)
```

**🎨 From Simple to Mind-Bending**
- **Beginner**: Filleted boxes with parametric control
- **Intermediate**: Path-following geometries with proper normal calculations
- **Advanced**: One-sided Möbius surfaces with 180° normal rotation
- **Expert**: Multi-region mesh partitioning with shell materialization

### 🌍 **Real-World Applications**

**🏭 Manufacturing & Prototyping**
```python
# Parametric enclosures with perfect fillets
create_filleted_box(80, 60, 20, fillet_radius=3,
                    fillets_at=[Alignment.TOP])  # Only top edges filleted
```

**🎨 Art & Design**
```python
# Mathematical art pieces (Möbius strips, Klein bottles)
# Organic sculptures with mesh partitioning
# Custom jewelry with helical patterns
```

**🔧 Engineering Applications**
```python
# Fluid channels with sine wave patterns
# Spring coils with conical tapering
# Assembly joints with automatic connector hints
```

**🎓 Education & Research**
```python
# Topology demonstrations (one-sided surfaces)
# Computational geometry prototypes
# CAD-independent algorithm development
```

---

## Installation

Base package (geometry + construct + arrange/produce layers):

```bash
pip install shellforgepy
```

Optional extras:

```bash
# CadQuery adapter
pip install shellforgepy[cadquery]

# FreeCAD adapter (requires a system FreeCAD installation)
pip install shellforgepy[freecad]

# Everything
pip install shellforgepy[all]
```

### Development install

```bash
git clone git@github.com:m-emm/shellforgepy.git
cd shellforgepy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[testing]"
```

---

## Using FreeCAD

ShellForgePy can use an existing FreeCAD installation as a CAD backend. This requires some additional setup since FreeCAD uses its own Python environment.

### FreeCAD Python wrapper script

The repository includes `freecad_python.sh` (macOS) that allows running Python scripts within FreeCAD's environment:

```bash
# Interactive FreeCAD Python REPL
./freecad_python.sh

# Run pytest with FreeCAD modules
./freecad_python.sh -m pytest tests/unit/adapters/freecad/ -v

# Execute code
./freecad_python.sh -c "import FreeCAD; print('FreeCAD available!')"

# Run a script
./freecad_python.sh examples/filleted_boxes_example.py
```

### Platform-specific setup

**macOS:** The script assumes FreeCAD is installed at `/Applications/FreeCAD.app`. Modify the path in `freecad_python.sh` if needed.

**Linux/Windows:** Adapt the script by:
1. Changing the FreeCAD executable path (typically `/usr/bin/freecad` on Linux)
2. Adjusting the macro file path as needed

### Requirements

- System FreeCAD installation
- Python environment with `shellforgepy[freecad]` installed
- FreeCAD's Python modules discoverable by the wrapper script

The wrapper script handles environment setup and provides a standard Python-like interface while giving access to FreeCAD modules (`FreeCAD`, `Part`, etc.).

---

## Quick start

```python
from shellforgepy.simple import (
    Alignment,
    align,
    arrange_and_export_parts,
    create_basic_box,
    create_basic_cylinder,
)

# Model a simple assembly (pure Python)
base = create_basic_box(80, 60, 5)
post = create_basic_cylinder(radius=5, height=40)
post = align(post, base, Alignment.CENTER)
assembly = [
    {"name": "base", "part": base},
    {"name": "post", "part": post},
]

# Lay out parts for printing and export to STL
arrange_and_export_parts(
    parts=assembly,
    prod_gap=5.0,
    bed_width=200.0,
    script_file="examples/pedestal.py",
    export_directory="output",
)
```

If a CadQuery or FreeCAD adapter is available, the exporter will use it
transparently. Otherwise you get a helpful error telling you which dependency is
missing.

---

## Project layout

```
src/shellforgepy/
├── geometry/        # Pure NumPy/ SciPy helpers
├── construct/       # Alignment and composition primitives
├── produce/         # Arrangement + export helpers
├── adapters/        # Optional cadquery/ and freecad/ backends
└── simple.py        # Convenience facade with auto-selected adapter
```

---

## 🎯 Examples - From Zero to Mind-Blown 🤯

**8 working examples** that showcase everything from basic CAD to mathematical wizardry!

### 🔰 **Beginner-Friendly**
```bash
python examples/filleted_boxes_example.py    # 12 parametric boxes with selective fillets
python examples/create_cylinder_stl.py       # Mesh generation fundamentals
python examples/straight_snake.py            # Path-following basics
```
*Perfect for learning CAD fundamentals and ShellForgePy concepts*

### 🚀 **Mind-Bending Mathematics**
```bash
python examples/curved_snake.py              # Sine wave fluid channels
python examples/cylindrical_coil.py          # Perfect helical springs
python examples/conical_coil.py              # Tapering coil geometries
python examples/mobius_strip.py              # One-sided surfaces! 🤯
```
*Advanced coordinate transformations that would be nightmare in traditional CAD*

### 🧠 **Professional-Grade Mesh Processing**
```bash
python examples/create_face_stl.py           # Organic face models with front/back splitting
```
*200mm organic sculptures with automatic partitioning and 2.5mm shell thickness*

### ⚡ **What Makes These Special**

- **Instant gratification**: All examples run in 30 seconds with zero setup
- **Production ready**: Generate STL files with proper 3D printing parameters
- **Educational**: Each example teaches specific ShellForgePy capabilities
- **Scalable**: From 20mm test pieces to 200mm art sculptures
- **Mathematical**: Implement complex surfaces that are impossible in traditional CAD

**🎯 Try the Möbius strip example** - it creates a mathematically perfect one-sided surface by rotating normals 180° over a circular path. Good luck doing that in Fusion 360! 😉

**[📖 Complete Examples Guide →](examples/README.md)** - Detailed descriptions, features, and outputs for all examples.

---

## Contributing & Development

Run linters/tests before pushing:

```bash
pytest
black src/ tests/
isort src/ tests/
```

Bug reports and pull requests are welcome! Please document new APIs in docstrings
and keep adapter-specific code isolated so ShellForgePy stays backend-agnostic by
default.

---

## Code Nutrition Facts
```
Serving size: 1 pip install

Amount per serving:
- Human effort .................. 72%
- AI seasoning .................. 18%
- PLA dust & failed prints ....... 7%
- Duct tape & hotfix glue ........ 2%
- Mystery ingredients ............ 1%

* Percentages are approximate and may vary between commits.

Allergen Information:
⚠️ May contain traces of AI-generated code.
Sensitive developers could experience
mild irritation, spontaneous refactoring urges,
or existential dread.

````

## License

MIT — see [LICENSE.txt](LICENSE.txt).
