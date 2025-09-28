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

## Why ShellForgePy?

- **Backend‑agnostic modeling** – Keep your design logic independent of any CAD
  kernel. Only when you need to materialise a shape do you pick an adapter
  (CadQuery, FreeCAD, …).
- **Alignment‑first construction** – Use the `construct` helpers to position and
  combine parts predictably. Translating ShellForge design ideas into code stays
  tidy and explicit.
- **Production utilities** – `produce` offers layout helpers, STL export, and
  other fabrication conveniences targeted at 3D printing and similar workflows.
- **Composable adapters** – Import `shellforgepy.simple` to auto-select an
  available CAD backend at import time, while still giving you informative error
  messages if nothing is installed.

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
    bed_with=200.0,
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

## 🎯 Examples

**8 working examples** that demonstrate shellforgepy's full capabilities - from beginner CAD to advanced mathematical surfaces!

### 🔰 **Beginner Examples**
```bash
python examples/filleted_boxes_example.py    # Parametric CAD with fillets
python examples/create_cylinder_stl.py       # Basic mesh generation
python examples/straight_snake.py            # Simple path following
```

### 🔥 **Advanced Path-Following**
```bash
python examples/curved_snake.py              # Sine wave channels
python examples/cylindrical_coil.py          # Helical coils
python examples/conical_coil.py              # Tapering coils
python examples/mobius_strip.py              # One-sided surfaces! 🤯
```

### 🧠 **Complex Meshes**
```bash
python examples/create_face_stl.py           # Organic shapes + partitioning
```

**📁 Output:** All examples generate STL files ready for 3D printing!

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

## License

MIT — see [LICENSE.txt](LICENSE.txt).
