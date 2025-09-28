# 🚀 ShellForgePy Examples

This directory contains working examples demonstrating shellforgepy's capabilities for creating 3D printable geometries. Each example generates real STL files ready for 3D printing!

## 🎯 Quick Start

```bash
# Try the beginner-friendly example first
python examples/filleted_boxes_example.py

# Or dive into mesh generation
python examples/create_cylinder_stl.py
```

## 📋 Available Examples

### 🔰 Beginner Examples

#### **Filleted Boxes** (`filleted_boxes_example.py`)
Parametric CAD modeling with selective edge filleting.

```bash
python examples/filleted_boxes_example.py
```

**Output:**
- Individual STL files: `filleted_boxes_example_*.stl`
- Combined layout: `filleted_boxes_example.stl`
- Process data: `filleted_boxes_example_process.json`

**Features:**
- 12 different fillet configurations
- Production-ready part arrangement
- Automatic build plate layout
- 3D printing process parameters

---

#### **Cylinder Mesh** (`create_cylinder_stl.py`)
Basic mesh generation from point clouds.

```bash
python examples/create_cylinder_stl.py
```

**Output:**
- `output/cylinder_mesh.stl`

**Features:**
- Point cloud generation for cylinders
- Mesh triangulation and partitioning
- Multi-object positioning
- STL export

![Cylinders Example](cylinders.png)

---

### 🔥 Path-Following Geometries

#### **Straight Snake** (`straight_snake.py`)
Simple straight channel with trapezoidal cross-section.

```bash
python examples/straight_snake.py
```

**Output:**
- `output/straight_snake.stl`

Perfect for LED strip channels or cable management.

---

#### **Curved Snake** (`curved_snake.py`)
Curved channel following a sine wave pattern.

```bash
python examples/curved_snake.py
```

**Output:**
- `output/curved_snake.stl`

Great for decorative elements or organic-shaped channels.

---

#### **Cylindrical Coil** (`cylindrical_coil.py`)
Helical coil with constant radius.

```bash
python examples/cylindrical_coil.py
```

**Output:**
- `output/cylindrical_coil.stl`

Perfect for LED strip coils or decorative spirals.

---

#### **Conical Coil** (`conical_coil.py`)
Advanced helical coil with varying radius.

```bash
python examples/conical_coil.py
```

**Output:**
- `output/conical_coil.stl`

![Conical Coil Example](ConicalCoil.png)

Demonstrates advanced geometry impossible with traditional CAD!

---

#### **Möbius Strip** (`mobius_strip.py`)
Mathematical marvel - a surface with only one side!

```bash
python examples/mobius_strip.py
```

**Output:**
- `output/mobius_strip.stl`

![Möbius Strip Example](Mobius.png)

The ultimate demonstration of coordinate transformation capabilities.

---

### 🧠 Advanced Examples

#### **Face Mesh** (`create_face_stl.py`)
Complex organic shapes with mesh partitioning.

```bash
python examples/create_face_stl.py
```

**Output:**
- `face_stl_output/face_m_front.stl`
- `face_stl_output/face_m_back.stl`
- `face_stl_output/face_m_complete.stl`

**Features:**
- Organic shape point cloud generation
- Mesh partitioning (front/back splitting)
- Shell creation for hollow parts
- Multiple STL outputs for different regions

![Face Example](Face.png)

---

## 🎲 Run All Examples

Want to see everything in action?

```bash
# Run each example individually
python examples/filleted_boxes_example.py
python examples/create_cylinder_stl.py
python examples/straight_snake.py
python examples/curved_snake.py
python examples/cylindrical_coil.py
python examples/conical_coil.py
python examples/mobius_strip.py
python examples/create_face_stl.py
```

## 📁 Output Files

Examples create STL files in these locations:
```
├── output/                          # Most examples
│   ├── cylinder_mesh.stl
│   ├── straight_snake.stl
│   ├── curved_snake.stl
│   ├── cylindrical_coil.stl
│   ├── conical_coil.stl
│   └── mobius_strip.stl
├── face_stl_output/                # Face example
│   ├── face_m_front.stl
│   ├── face_m_back.stl
│   └── face_m_complete.stl
└── filleted_boxes_example_*.stl    # Filleted boxes (current directory)
```

## 📊 Example Complexity

| Example | Complexity | Focus | Output Files |
|---------|------------|-------|--------------|
| `filleted_boxes_example.py` | 🔰 Beginner | CAD adapter usage, production workflow | 13 STL files |
| `create_cylinder_stl.py` | 🔰 Beginner | Basic mesh workflows | 1 STL file |
| `straight_snake.py` | 🔰 Beginner | Path-following basics | 1 STL file |
| `curved_snake.py` | 🔶 Intermediate | Curved path following | 1 STL file |
| `cylindrical_coil.py` | 🔶 Intermediate | Helical geometries | 1 STL file |
| `conical_coil.py` | 🔶 Intermediate | Advanced helical paths | 1 STL file |
| `mobius_strip.py` | 🔶 Intermediate | Topological surfaces | 1 STL file |
| `create_face_stl.py` | 🔴 Advanced | Organic shapes, mesh partitioning | 3 STL files |

## 🛠️ Technologies Demonstrated

### Core Features:
- ✅ CAD adapter system (CadQuery/FreeCAD backend selection)
- ✅ Parametric solid modeling with filleted edges
- ✅ Point cloud generation for various geometries
- ✅ Mesh triangulation and conversion to printable meshes
- ✅ Mesh partitioning for multi-part printing
- ✅ Coordinate transformation for path-following geometries
- ✅ Production-ready part arrangement and export
- ✅ Binary STL export

### Path-Following Capabilities:
- ✅ Following 3D curves with consistent cross-sections
- ✅ Surface normal direction control
- ✅ Multi-segment assembly and connection
- ✅ Loop closure with vertex correspondence detection
- ✅ Mathematical surface generation (Möbius strips)

## 🎯 Applications

These examples are perfect for:

- **LED strip channels and mounting systems**
- **Cable management and wire routing**
- **Decorative moldings and trim pieces**
- **Custom coils and spiral structures**
- **Mathematical models and educational demonstrations**
- **Rapid prototyping and 3D printing projects**

## 🖨️ 3D Printing Ready

All examples generate STL files optimized for 3D printing:

- ✅ Dimensions in millimeters
- ✅ Appropriate wall thickness for FDM/SLA printing
- ✅ Manifold meshes (watertight geometry)
- ✅ Optimized triangle counts

## 🚀 Next Steps

1. **Start with `filleted_boxes_example.py`** for CAD adapter basics
2. **Try `create_cylinder_stl.py`** for mesh fundamentals
3. **Explore path-following** with snake and coil examples
4. **Challenge yourself** with the advanced face mesh example
5. **Modify the examples** for your own projects!

---

Ready to create amazing 3D geometries? Pick an example and start building! 🎯