# Validation Module

The validation module (`validation/`) provides an OpenFOAM-based pipeline for comparing panel method results against CFD simulations.

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
