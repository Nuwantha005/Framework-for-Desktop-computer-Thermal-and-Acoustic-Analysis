# Case Files

The solver reads simulation definitions from YAML case files. Each case lives in its own directory under `cases/`.

## Directory Structure

```
cases/
└── cylinder_flow/
    ├── case.yaml          # Simulation definition
    ├── shapes/            # Optional: JSON geometry files
    │   └── custom.json
    └── out/               # Generated outputs (gitignored)
        ├── combined.png
        └── velocity_contours.png
```

## Case YAML Format

A complete case file defines freestream conditions, geometry components, solver settings, visualization domain, and fluid properties.

### Example: Cylinder Flow

```yaml
name: "Cylinder Flow"
case_type: "parametric_2d"
description: "Flow around cylinder - analytical validation case"

freestream:
  velocity: [1.0, 0.0, 0.0]    # [Vx, Vy, Vz] — Vz=0 for 2D

components:
  - name: "cylinder"
    geometry:
      type: "circle"             # Parametric type (circle, rectangle, rounded_rectangle)
      parameters:
        radius: 0.5
        center: [0.0, 0.0]
    mesh_levels:                 # Multiple resolution levels for convergence studies
      - [16]                     # Level 0: 16 panels (coarse)
      - [32]                     # Level 1: 32 panels
      - [64]                     # Level 2: 64 panels
      - [128]                    # Level 3: 128 panels
      - [256]                    # Level 4: 256 panels (fine)
    transform:
      translation: [0.0, 0.0, 0.0]
      rotation_deg: 0.0
    boundary_condition:
      type: "wall"

solver:
  type: "constant_source"        # Currently the only implemented solver
  tolerance: 1.0e-10

visualization:
  domain:
    x_range: [-2.0, 3.0]
    y_range: [-2.0, 2.0]
  resolution: [200, 160]         # Grid points [nx, ny]

fluid:
  density: 1.225                 # Required (kg/m³)
  viscosity: 1.789e-5            # Optional (Pa·s)
  gravity: 0.0                   # Optional (m/s²)
  reference_pressure: 101325.0   # Optional (Pa)
  reference_type: freestream     # "freestream" or "stagnation"
```

### Sections

#### `freestream`
Defines the undisturbed flow velocity as a 3D vector. For 2D cases, set `Vz = 0`.

#### `components`
List of geometry components. Each component has:

- **`name`**: Identifier string
- **`geometry.type`**: One of `"circle"`, `"rectangle"`, `"rounded_rectangle"`
- **`geometry.parameters`**: Shape-specific parameters:
    - Circle: `radius`, `center`
    - Rectangle: `width`, `height`, `center`
    - Rounded rectangle: `width`, `height`, `corner_radius`, `center`
- **`mesh_levels`**: List of resolution tuples. Each tuple unpacks as arguments to the generator:
    - Circle: `[num_panels]`
    - Rectangle: `[panels_x, panels_y]`
    - Rounded rectangle: `[panels_per_side, panels_per_arc]`
- **`transform`**: Translation and rotation applied to the component
- **`boundary_condition`**: Currently only `"wall"` (no-penetration) is supported

#### `solver`
- **`type`**: Solver type string. Maps to factory key: `"constant_source"` → `SourcePanelSolver`
- **`tolerance`**: Solver tolerance (not currently used for direct solve)

#### `visualization`
- **`domain`**: Bounding box for field computation (`x_range`, `y_range`)
- **`resolution`**: Grid resolution `[nx, ny]`

#### `fluid`
Physical fluid properties. Only `density` is required; others are optional and used by post-processing.

## Geometry JSON Format

Raw geometry can also be specified as JSON files with explicit node coordinates and panel connectivity:

```json
{
  "format": "panels_2d",
  "nodes": [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0]
  ],
  "panels": [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
  ],
  "normal_direction": "outward"
}
```

- **`nodes`**: List of `[x, y, z]` coordinates (z=0 for 2D). Shape `(N, 3)`.
- **`panels`**: List of `[start_node, end_node]` index pairs. CCW ordering for outward normals.
- **`normal_direction`**: `"outward"` (default)

## Loading Cases in Code

```python
from core.io import CaseLoader

# Load at default mesh level (level 0)
case = CaseLoader.load_case("cases/cylinder_flow")

# Load at a specific mesh refinement level
case = CaseLoader.load_case("cases/cylinder_flow", mesh_level_index=3)

# Switch mesh level on an existing case
finer_case = case.reload_at_level(4)
```
