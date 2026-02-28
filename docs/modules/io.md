# I/O Module

The I/O module (`core.io` and `core.config`) handles loading case definitions, parsing geometry files, and exporting results.

## Case Loading

### CaseLoader

The primary entry point for loading simulation definitions:

```python
from core.io import CaseLoader

# Load a case (default: mesh level 0)
case = CaseLoader.load_case("cases/cylinder_flow")

# Load at a specific mesh refinement level
case = CaseLoader.load_case("cases/cylinder_flow", mesh_level_index=3)

# Validate a case file without loading
is_valid = CaseLoader.validate("cases/cylinder_flow/case.yaml")
```

### Case Wrapper

`Case` is a dataclass that unifies the `Scene`, `SimulationConfig`, and case directory into a single object with convenience accessors:

```python
case.name                # "Cylinder Flow"
case.num_panels          # Number of panels at current mesh level
case.num_components      # Number of geometry components
case.v_inf               # Freestream velocity magnitude
case.aoa                 # Angle of attack (degrees)
case.x_range             # Visualization x-domain
case.y_range             # Visualization y-domain
case.resolution          # Grid resolution [nx, ny]
case.output_dir          # Path to cases/<name>/out/

# Create solver from case config
solver = case.create_solver()

# Get fluid state for post-processing
fluid = case.get_fluid_state()

# Switch mesh level
finer = case.reload_at_level(4)
```

## Geometry I/O

### GeometryReader

Reads geometry from JSON or XY files:

```python
from core.io import GeometryReader

# Read from JSON (nodes + panels)
mesh = GeometryReader.read_json("data/geometries/square_unit.json")

# Read from XY coordinate file
mesh = GeometryReader.read_xy("path/to/coords.xy")

# Auto-detect format
mesh = GeometryReader.read("some_geometry.json")
```

### CaseExporter

Exports mesh and scene data:

```python
from core.io import CaseExporter

CaseExporter.from_mesh(mesh, "output/geometry.json")
CaseExporter.from_scene(scene, "output/case_dir")
```

## Configuration Schemas

The `core.config.schemas` module defines Pydantic models for validation:

| Schema | Purpose |
|--------|---------|
| `SimulationConfig` | Top-level config (wraps all sections) |
| `ComponentConfig` | Per-component geometry + transform + BC |
| `GeometryConfig` | Geometry type and parameters |
| `TransformConfig` | Translation and rotation |
| `SolverConfig` | Solver type, tolerance |
| `FluidConfig` | Density, viscosity, gravity, reference pressure |
| `VisualizationConfig` | Domain and resolution |
| `OutputConfig` | Output settings |
| `BoundaryConditionType` | Enum of BC types |

## File Layout

| File | Contents |
|------|----------|
| `core/io/case_loader.py` | `CaseLoader` — YAML parsing, Scene creation |
| `core/io/case.py` | `Case` dataclass — unified wrapper |
| `core/io/case_exporter.py` | `CaseExporter` — JSON/YAML export |
| `core/io/geometry_io.py` | `GeometryReader` — JSON/XY mesh reading |
| `core/config/schemas.py` | Pydantic validation models (317 lines) |
