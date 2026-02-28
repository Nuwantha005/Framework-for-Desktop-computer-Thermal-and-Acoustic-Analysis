# Geometry Module

The geometry module (`core.geometry`) provides the data structures and generators for panel mesh creation and multi-body scene assembly.

## Key Classes

### Mesh

The central data structure holding panel discretization data.

```python
from core.geometry import Mesh

# Mesh fields (populated on construction + compute_geometry())
mesh.nodes          # (N_nodes, 3) float64 — vertex coordinates
mesh.panels         # (N_panels, 2) int32 — node index pairs
mesh.dimension      # 2 or 3
mesh.component_ids  # (N_panels,) int32 — which component each panel belongs to
mesh.centers        # (N_panels, 3) float64 — panel midpoints
mesh.normals        # (N_panels, 3) float64 — outward unit normals
mesh.tangents       # (N_panels, 3) float64 — unit tangent vectors
mesh.areas          # (N_panels,) float64 — panel lengths (2D) or areas (3D)
mesh.cell_data      # dict — solver results (source_strength, Cp, etc.)
```

All coordinate arrays are shape `(N, 3)` with `z=0` for 2D. This convention simplifies future 3D extension.

### Component and Transform

A `Component` wraps a mesh with a spatial transform and boundary condition:

```python
from core.geometry import Component, Transform

transform = Transform.from_2d(tx=1.0, ty=0.5, angle_deg=45.0)
component = Component(
    name="obstacle",
    local_mesh=mesh,
    transform=transform,
    bc_type="wall"
)
global_mesh = component.get_global_mesh(component_id=0)
```

### Scene

A `Scene` holds multiple components and assembles them into a single mesh:

```python
from core.geometry import Scene
import numpy as np

scene = Scene(
    name="test_case",
    components=[comp1, comp2],
    freestream=np.array([1.0, 0.0, 0.0])
)
merged_mesh = scene.assemble()
```

## Parametric Generators

Three built-in generators create meshes from parameters:

```python
from core.geometry.generators import generate_circle, generate_rectangle, generate_rounded_rectangle

# Circle with 64 panels, radius 0.5
circle = generate_circle(64, radius=0.5, center=(0.0, 0.0))

# Rectangle with 8 panels per side (x) and 6 per side (y)
rect = generate_rectangle(8, 6, width=2.0, height=1.0)

# Rounded rectangle: 10 panels per straight side, 5 per corner arc
rrect = generate_rounded_rectangle(10, 5, width=2.0, height=1.0, corner_radius=0.2)
```

## GeometryFactory

The factory maps type strings (from case YAML) to generator functions:

```python
from core.geometry.factory import GeometryFactory

# Create from case-style definition
mesh = GeometryFactory.create(
    geometry_definition={"type": "circle", "parameters": {"radius": 0.5}},
    resolution=[64]
)

# List registered types
print(GeometryFactory.list_types())  # ['circle', 'rectangle', 'rounded_rectangle']
```

## File Layout

| File | Contents |
|------|----------|
| `primitives.py` | `Point3D`, `Vector3D` dataclasses, rotation matrix helpers |
| `mesh.py` | `Mesh` dataclass with `compute_geometry()` |
| `component.py` | `Transform`, `Component` dataclasses |
| `scene.py` | `Scene` dataclass with `assemble()` |
| `generators.py` | `generate_circle()`, `generate_rectangle()`, `generate_rounded_rectangle()` |
| `factory.py` | `GeometryFactory` registry |
