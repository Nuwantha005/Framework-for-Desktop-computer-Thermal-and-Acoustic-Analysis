# Getting Started

## Prerequisites

- Python 3.10+ (tested with 3.13)
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or Miniconda

## Environment Setup

```bash
# Create and activate the environment
mamba create -n fyp python=3.13
mamba activate fyp

# Clone the repository
git clone <repository-url> panel-method-solver
cd panel-method-solver

# Install in editable mode (includes all core dependencies)
pip install -e .

# Install development tools (optional)
pip install -e ".[dev]"

# Install documentation tools (optional)
pip install -e ".[docs]"
```

## Verify Installation

```bash
python -c "from core.geometry import Mesh; from solvers import SourcePanelSolver; print('OK')"
```

## First Simulation

### Using the Demo Scripts

The fastest way to run a simulation is through the demo scripts:

```bash
cd demos

# Cylinder flow (analytical validation case)
python demo_combined.py ../cases/cylinder_flow --show

# Single square body
python demo_combined.py ../cases/single_square --save

# Two rounded rectangles
python demo_combined.py ../cases/two_rounded_rects --save --cores 6
```

Demo options:

- `--show`: Display the plot interactively
- `--save`: Save the plot to `cases/<name>/out/`
- `--cores N`: Number of CPU cores for field computation (default: 6)
- `--protect`: Save to a timestamped subfolder to avoid overwriting

### Using the Python API

```python
from core.io import CaseLoader
from visualization import Visualizer
from visualization.field2d import VelocityField2D

# Load a case definition
case = CaseLoader.load_case("cases/cylinder_flow")
print(f"Case: {case.name}")
print(f"  Components: {case.num_components}")
print(f"  Panels: {case.num_panels}")
print(f"  Freestream: V={case.v_inf}, AoA={case.aoa}°")

# Create and run the solver
solver = case.create_solver()
solver.solve()

# Access results
print(f"  Cp range: [{solver.Cp.min():.4f}, {solver.Cp.max():.4f}]")
print(f"  Source strength sum: {solver.sigma.sum():.2e}")

# Compute the velocity field on a grid
field = VelocityField2D(solver)
XX, YY, Vx, Vy = field.compute(
    x_range=case.x_range,
    y_range=case.y_range,
    resolution=case.resolution
)

# Create a visualization
viz = Visualizer(output_dir=case.output_dir)
viz.create_figure(subplots=(1, 1), figsize=(10, 8), title=case.name)
viz.plot_contours(XX, YY, Vx, Vy, case.mesh, ax_index=0, levels=25)
viz.finalize(save="velocity_contours.png", show=False)
```

## Available Cases

| Case | Directory | Description |
|------|-----------|-------------|
| Cylinder Flow | `cases/cylinder_flow/` | Single cylinder, analytical $C_p = 1 - 4\sin^2\theta$ available |
| Single Square | `cases/single_square/` | Rounded rectangle, OpenFOAM validation available |
| Rounded Square | `cases/rounded_square/` | Rounded rectangle variant |
| Two Rounded Rects | `cases/two_rounded_rects/` | Multi-body case with two rounded rectangles |
| Cylinder and Square | `cases/cylinder_and_square/` | Multi-body: circle + rectangle |

## Running Tests

```bash
pytest src/test/ -v
```
