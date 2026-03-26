# Validation Module

The validation module (`validation/`) now has two tracks:

- Current: Fluent-based boundary-layer (BL) validation
- Legacy: OpenFOAM-based panel-method validation

The Fluent BL track is the active comparison workflow.

## Pipeline Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Case YAML       │────▶│  OpenFOAM Case    │────▶│  Mesh & Solve    │
│  (panel method)  │     │  Generation       │     │  (multiple       │
│                  │     │  (STL + dicts)    │     │   refinements)   │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
┌──────────────────┐     ┌──────────────────┐              │
│  Error Metrics   │◀────│  Surface         │◀─────────────┘
│  L2, L∞, GCI    │     │  Extraction      │
└──────────────────┘     └──────────────────┘
```

## Key Components

### Fluent BL Comparison (Current)

Fluent BL comparison is implemented under `src/validation/adapters/fluent/` and compares reconstructed BL fields against Fluent exports:

- `ascii_reader.py` — loads `fluent_case/export/viscous_bl/{filed_data,wall_data}`
- `bl_extractor.py` — extracts wall/edge quantities from Fluent data
- `interpolator.py` — interpolates Fluent velocity onto BL solver `(s, y)` grids
- `comparison.py` — `BLComparisonRunner`, metrics (`L2`, `Linf`, `RMS`, `MAE`, relative)

Plot generation scripts:

- `validation/scripts/plot_bl_fluent_difference.py` — difference contours/envelopes + metrics report
- `validation/scripts/plot_bl_fluent_side_by_side.py` — absolute BL-vs-Fluent side-by-side plots

### Case Generation

`FoamlibCaseGenerator` creates a complete OpenFOAM case from a panel method `Case` object:

- Generates `0/`, `constant/`, `system/` directories
- Exports body geometry as STL via `GeometryConverter` (uses trimesh)
- Configures blockMesh background mesh, snappyHexMesh refinement, and potentialFoam solver

### Mesh Convergence

The convergence module supports both panel method and OpenFOAM mesh convergence studies:

- `run_panel_convergence()` — Solves at each `mesh_levels` from case YAML
- `run_openfoam_convergence()` — Runs OpenFOAM at multiple refinement levels
- `compute_gci()` — Grid Convergence Index with asymptotic range checking

### Surface Extraction

`OpenFOAMSurfaceExtractor` reads VTP files from OpenFOAM's `postProcessing/` output and extracts wall boundary data (tangential velocity, pressure coefficient).

`GeometryMapper` handles arc-length projection for consistent comparison between meshes with different discretizations.

### Error Metrics

```python
from validation.convergence.metrics import ErrorMetrics, compute_error_metrics

metrics = compute_error_metrics(reference, test)
# metrics.l2, metrics.linf, metrics.rms, metrics.mae, metrics.relative_*
```

## Scripts

The validation workflow is executed through scripts in `validation/scripts/`:

| Script | Step | Description |
|--------|------|-------------|
| `plot_bl_fluent_difference.py` | Current-1 | BL Fluent difference plots |
| `plot_bl_fluent_side_by_side.py` | Current-2 | BL Fluent side-by-side absolute plots |
| `generate_base_case.py` | 1 | Create OpenFOAM case from panel case YAML |
| `run_of_convergence.py` | 2 | Run OpenFOAM mesh convergence study |
| `run_panel_convergence.py` | 3 | Run panel method at multiple mesh levels |
| `visualize.py` | 4 | Generate comparison plots |
| `compare_surface.py` | 5 | Quantitative surface comparison |

See the [validation user guide](../user_guide/validation.md) for step-by-step instructions.

## File Layout

| Directory | Key Files |
|-----------|-----------|
| `validation/` | `__init__.py`, `geometry_mapper.py` |
| `validation/adapters/openfoam/` | `case_generator.py`, `foamlib_generator.py`, `runner.py`, `geometry_converter.py`, `surface_extractor.py` |
| `validation/comparison/` | `grid.py`, `surface.py`, `probe.py` |
| `validation/convergence/` | `metrics.py`, `panel.py`, `openfoam.py` |
| `validation/scripts/` | Pipeline scripts + `utils/` helpers |
