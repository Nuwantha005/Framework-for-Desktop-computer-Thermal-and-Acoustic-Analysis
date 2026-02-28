# Panel Method Solver

A 2D panel method solver for potential flow analysis, designed as part of a framework for desktop computer thermal and acoustic analysis. The solver implements constant-strength source panels with Neumann boundary conditions and includes an OpenFOAM-based validation pipeline for comparing results against CFD.

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| Geometry | ✅ Stable | Parametric shape generation (circle, rectangle, rounded rectangle), mesh assembly |
| Constant Source Solver | ✅ Working | Constant-strength source panel method (Katz & Plotkin) |
| I/O | ✅ Working | YAML case files, JSON geometry, case loading and export |
| Post-processing | ✅ Working | Pressure, velocity potential, stream function, vorticity pipeline |
| Visualization | ✅ Working | Contours, streamlines, Cp plots, surface envelopes, comparison |
| Validation Pipeline | ✅ Working | OpenFOAM case generation, meshing, grid independence, comparison |
| Vortex Panels | 🔲 Planned | Lifting bodies with Kutta condition |
| Higher-Order Panels | 🔲 Planned | Linear and quadratic strength distributions |
| Viscous BL Solver | 🔲 Planned | Von Kármán momentum integral method |
| Thermal BL Solver | 🔲 Planned | Thermal boundary layer via BDIM |
| 3D Panel Method | 🔲 Planned | Extension to 3D quadrilateral panels |

## Quick Start

### Installation

```bash
mamba create -n fyp python=3.13
mamba activate fyp
pip install -e .
```

### Running a Simulation

```python
from core.io import CaseLoader
from visualization import Visualizer
from visualization.field2d import VelocityField2D

# Load a case
case = CaseLoader.load_case("cases/cylinder_flow")
print(f"Case: {case.name}, Panels: {case.num_panels}")

# Solve
solver = case.create_solver()
solver.solve()
print(f"Cp range: [{solver.Cp.min():.4f}, {solver.Cp.max():.4f}]")

# Compute velocity field
field = VelocityField2D(solver)
XX, YY, Vx, Vy = field.compute(case.x_range, case.y_range, case.resolution)

# Visualize
viz = Visualizer(output_dir=case.output_dir)
viz.create_figure(subplots=(1, 2), figsize=(14, 6), title=case.name)
viz.plot_contours(XX, YY, Vx, Vy, case.mesh, ax_index=0, title="Velocity")
viz.plot_streamlines(XX, YY, Vx, Vy, case.mesh, ax_index=1, title="Streamlines")
viz.finalize(save="results.png", show=False)
```

### Running Demos

```bash
cd demos
python demo_combined.py ../cases/cylinder_flow --show
python demo_combined.py ../cases/single_square --save
```

## Project Structure

```
panel-method-solver/
├── src/
│   ├── core/
│   │   ├── config/schemas.py          # Pydantic config models
│   │   ├── geometry/                   # Mesh, Component, Scene, generators
│   │   └── io/                         # CaseLoader, Case, CaseExporter
│   ├── solvers/
│   │   ├── base.py                     # Solver ABC
│   │   ├── factory.py                  # SolverFactory registry
│   │   └── panel2d/                    # 2D panel method implementations
│   ├── postprocessing/                 # FieldData, ProcessorPipeline, processors
│   └── visualization/                  # Visualizer, VelocityField2D, comparison
├── validation/                         # OpenFOAM validation pipeline
├── cases/                              # Case definitions (YAML + geometry)
├── demos/                              # Example scripts
├── docs/                               # This documentation
└── pyproject.toml                      # Package configuration
```
