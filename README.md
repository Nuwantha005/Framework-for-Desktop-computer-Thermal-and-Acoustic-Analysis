# Panel Method Solver & Thermal/Acoustic Analysis Framework

A Python framework for 2D and 3D potential flow panel method solving, boundary layer analysis, and ducted actuator disk fan modeling. Developed as part of a Final Year Project for desktop thermal and acoustic analysis.

---

## Capabilities

### 1. 2D Panel Method Solvers
- **Constant-Strength Source Panel Method** (Katz & Plotkin formulation)
- **Linear-Strength Source Panel Method** (Node-based continuous formulation)
- **Linear-Strength Vortex Panel Method** (Zero net circulation closure for bluff bodies)
- **Constant-Strength Dirichlet Doublet Panel Method** (Morino formulation)
- **Linear-Strength Source/Doublet Panel Method** (Morino linear Dirichlet BCs)

### 2. Viscous & Thermal Boundary Layer Solvers
- **Von Kármán Momentum Integral BL Solver**: Supports Thwaites, Pohlhausen, Falkner-Skan, and Blasius profiles.
- Automatic stagnation point detection via sign-change interpolation and analytical patching.
- Reconstructed 2D boundary layer velocity fields and boundary layer displacement thickness envelopes.

### 3. 3D Panel Solver & Actuator Disk Model (ADM)
- **3D Constant-Source Panel Method**: Quad and triangle surface discretization, vectorized Numba JIT influence matrix calculation.
- **Gmsh CAD Importer**: Direct import and surface meshing of `.step` and `.stl` geometries.
- **Actuator Disk Model (ADM)**: Coupled potential flow + ducted fan pressure-jump iteration matching empirical P-Q fan curves with multi-fan support.
- Direct static pressure field reconstruction across field points and duct cross-sections.

### 4. Post-Processing & Visualization
- **2D Visualizations**: Contour plots (velocity, pressure coefficient $C_p$, stream function $\psi$), surface envelope distributions, OpenFOAM CFD comparison plots.
- **3D Visualizations**: PyVista VTK volume field rendering, 3D surface streamlines, vector glyphs, cut-plane slices, and ParaView export.

---

## Quick Start

### 1. Installation & Environment

The framework uses standard scientific Python dependencies and `pyproject.toml` for editable installation:

```bash
# Clone the repository
git clone https://github.com/Nuwantha005/Framework-for-Desktop-computer-Thermal-and-Acoustic-Analysis.git
cd panel-method-solver

# Install in editable mode
pip install -e .
```

### 2. Running 2D Demos

```bash
# Run 2D rounded rectangle demo
python demos/demo_rounded_rectangle.py

# Run 2D solver comparison (Constant vs Linear vs Vortex vs Doublet)
python demos/demo_solver_comparison.py

# Run Von Kármán viscous boundary layer demo
python demos/demo_boundary_layer.py
```

### 3. Running 3D Demos & Actuator Disk Model

```bash
# 3D Sphere Potential Flow Demo
python demos/demo_sphere_3d.py

# 3D Ducted Actuator Disk Model (Circular Vent case)
python demos/demo_actuator_disk.py --case cases/cicular_vent

# PyVista 3D Streamlines Demo
python demos/demo_streamlines_3d_pyvista.py
```

---

## Project Architecture

```
src/
├── core/
│   ├── config/          # Pydantic configuration schemas (schemas.py)
│   ├── geometry/        # Mesh2D/Mesh3D, Component, Scene graph, Gmsh CAD reader
│   └── io/              # YAML case loader, JSON geometry I/O, case exporter
├── solvers/
│   ├── panel2d/         # Source, Linear Source, Vortex, Doublet 2D solvers & influences
│   ├── panel3d/         # 3D Constant Source panel solver with Numba JIT acceleration
│   ├── boundary_layer/  # Von Kármán momentum integral BL solver & profiles
│   └── actuator/        # ADM coupled solver, doublet sheets, fan curve interpolation
├── postprocessing/      # FieldData, FluidState, ProcessorPipeline (Cp, phi, psi, omega)
└── visualization/       # Visualizer, ComparisonVisualizer, VelocityField2D, PyVista/Matplotlib plotters
```

---

## Case File & Folder Structure

Case definitions are self-contained inside dedicated subdirectories under `cases/`. Each case folder contains its simulation specification (`case.yaml`), raw geometry files (JSON/STEP/STL), empirical fan curves or input data, and an automatically created `out/` directory for generated plots, CSV exports, and VTK fields.

### Case Directory Layout Example

```
cases/cicular_vent/
├── case.yaml                  # Primary simulation & solver config
├── shapes/                    # CAD geometry files or JSON panel specs
│   └── duct_casing.STEP       # (Optional) STEP/STL or JSON boundary geometry
├── data/                      # Input datasets and fan performance curves
│   └── fan_curve.csv          # P-Q performance data (Flow rate vs Static Pressure)
└── out/                       # Generated outputs (Gitignored)
    ├── mesh.png               # Visualized boundary mesh
    ├── adm/                   # Actuator disk convergence plots & CSVs
    │   ├── doublet_iterations.csv
    │   └── doublet_iterations.png
    └── validation/            # Extracted surface metrics & Fluent comparison plots
        ├── cut_plane_comparison.png
        └── axis_line_samples.csv
```

### Example `case.yaml`

```yaml
name: "Ducted Fan Vent Simulation"
case_type: "actuator_disk_3d"

freestream: [0.0, 0.0, 0.0]  # Freestream velocity vector [U_x, U_y, U_z] in m/s

fluid:
  density: 1.225              # Fluid density (kg/m³)
  kinematic_viscosity: 1.5e-5  # Kinematic viscosity (m²/s)

components:
  - name: "duct_wall"
    geometry_file: "shapes/duct_casing.STEP"
    mesh_levels: [0.025, 0.012]

inlets:
  - name: "inlet_disk"
    center: [0.0, 0.0, -0.5]
    radius: 0.06
    normal: [0.0, 0.0, 1.0]

outlets:
  - name: "outlet_disk"
    center: [0.0, 0.0, 0.5]
    radius: 0.06
    normal: [0.0, 0.0, 1.0]

actuator_disks:
  - name: "main_fan"
    center: [0.0, 0.0, 0.0]
    axis: [0.0, 0.0, 1.0]
    radius: 0.06
    curve_file: "data/fan_curve.csv"
    curve_type: "linear"       # "linear" or "spline"
    relaxation: 0.3
    tolerance: 1.0e-4
    max_iterations: 50
```

---

## License & Project Context

Developed as part of a Final Year Project in Thermal and Acoustic Analysis of Desktop Computers.
