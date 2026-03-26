# Running Validation

Validation currently uses two tracks:

- Active: Fluent-based boundary-layer (BL) comparison
- Legacy: OpenFOAM-based panel-method comparison

## Fluent BL Validation (Active)

### Prerequisites

- Fluent export files available under `cases/<case>/fluent_case/export/viscous_bl/`
- BL reconstruction enabled during solve (handled by the scripts)

### Commands

From repository root:

```bash
python validation/scripts/plot_bl_fluent_difference.py cases/cylinder_flow
python validation/scripts/plot_bl_fluent_side_by_side.py cases/cylinder_flow
```

Common options:

- `--compare-profile thwaites`
- `--profiles blasius thwaites`
- `--mesh-level -1`
- `--solver-type linear_source`
- `--output-dir <path>`
- `--show-plots`

Default output directories:

- `cases/<case>/out/boundary_layer/fluent_comparison/difference/`
- `cases/<case>/out/boundary_layer/fluent_comparison/side_by_side/`

## OpenFOAM Panel Validation (Legacy)

The OpenFOAM pipeline remains for historical panel-method validation.

### Prerequisites

- **OpenFOAM**: v2312 or later (ESI version) installed and sourced
- **foamlib**: `pip install foamlib` (OpenFOAM case generation)
- **trimesh**: `pip install trimesh` (STL geometry export)
- Both should be installed in the `fyp` mamba environment

## Legacy Pipeline Overview

The validation workflow consists of five sequential scripts in `validation/scripts/`:

```
1. generate_base_case.py    → Create OpenFOAM case from panel method case YAML
2. run_of_convergence.py    → Run OpenFOAM at multiple mesh refinement levels
3. run_panel_convergence.py → Run panel method at multiple mesh levels
4. visualize.py             → Generate comparison plots
5. compare_surface.py       → Quantitative surface comparison with error metrics
```

## Step-by-Step

### 1. Generate Base OpenFOAM Case

```bash
cd validation/scripts
python generate_base_case.py ../../cases/cylinder_flow
```

This creates an OpenFOAM case directory (`cases/cylinder_flow/of_case/`) with:

- `0/` — Initial/boundary conditions (U, p)
- `constant/` — Physical properties, geometry (STL in `triSurface/`)
- `system/` — Mesh and solver settings (blockMeshDict, snappyHexMeshDict, fvSchemes, fvSolution)

The geometry is exported as STL via `GeometryConverter`, which uses trimesh to convert the panel mesh to a triangulated surface.

### 2. Run OpenFOAM Mesh Convergence

```bash
python run_of_convergence.py ../../cases/cylinder_flow
```

This runs the meshing and solving pipeline at multiple refinement levels:

1. **blockMesh** — Creates the background hex mesh
2. **snappyHexMesh** — Refines around the geometry STL
3. **potentialFoam** — Solves the potential flow equation

Each refinement level produces results that are stored for later comparison. The Grid Convergence Index (GCI) is computed to assess mesh independence.

### 3. Run Panel Method Convergence

```bash
python run_panel_convergence.py ../../cases/cylinder_flow
```

Runs the panel method at each `mesh_levels` defined in `case.yaml` and records surface velocities and pressure coefficients.

### 4. Visualize Results

```bash
python visualize.py ../../cases/cylinder_flow
```

Generates comparison plots including:

- Panel method vs OpenFOAM contour fields
- Difference maps
- Error distribution plots

### 5. Surface Comparison

```bash
python compare_surface.py ../../cases/cylinder_flow
```

Performs quantitative comparison of surface quantities ($C_p$, tangential velocity) between panel method and OpenFOAM. Reports error metrics:

- **$L_2$ norm**: Root mean square of pointwise differences
- **$L_\infty$ norm**: Maximum absolute pointwise error
- **RMS**: Root mean square error
- **MAE**: Mean absolute error
- **Relative errors**: Normalized by reference values

## Error Metrics

The `ErrorMetrics` dataclass provides:

```python
from validation.convergence.metrics import compute_error_metrics

metrics = compute_error_metrics(reference_values, test_values)
print(f"L2:  {metrics.l2:.6f}")
print(f"Linf: {metrics.linf:.6f}")
print(f"RMS: {metrics.rms:.6f}")
print(f"MAE: {metrics.mae:.6f}")
```

## Grid Convergence Index (GCI)

The GCI assesses whether the solution is in the asymptotic convergence range:

```python
from validation.convergence.metrics import compute_gci

gci = compute_gci(solutions=[coarse, medium, fine], spacings=[h1, h2, h3])
print(f"GCI_fine: {gci.gci_fine:.4f}")
print(f"Asymptotic range: {gci.in_asymptotic_range}")
```

## Output Structure

Validation results are saved under `validation_results/<case_name>/`:

```
validation_results/
└── cylinder_flow/
    ├── panel_convergence/
    │   ├── cp_comparison.png
    │   └── convergence_metrics.csv
    ├── of_convergence/
    │   ├── mesh_level_0/
    │   ├── mesh_level_1/
    │   └── gci_results.csv
    └── surface_comparison/
        ├── cp_comparison.png
        ├── vt_comparison.png
        └── error_metrics.csv
```
