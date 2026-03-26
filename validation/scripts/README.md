# Validation Scripts

This folder contains validation utilities for two tracks:

- Legacy OpenFOAM pipeline (panel-method validation)
- Current Fluent boundary-layer (BL) validation pipeline

## Fluent BL Validation (Current)

Run from repository root.

Difference-style plots:

```bash
python validation/scripts/plot_bl_fluent_difference.py cases/rounded_square
```

Side-by-side absolute plots:

```bash
python validation/scripts/plot_bl_fluent_side_by_side.py cases/rounded_square
```

Useful options (both scripts):

- `--compare-profile thwaites` (single profile)
- `--profiles blasius thwaites` (BL solver profiles)
- `--mesh-level -1`
- `--solver-type linear_source`
- `--output-dir <path>`
- `--show-plots`

Default output folders:

- `cases/<case>/out/boundary_layer/fluent_comparison/difference/`
- `cases/<case>/out/boundary_layer/fluent_comparison/side_by_side/`

## Boundary-Layer Single Solve Demo

`demos/demo_boundary_layer.py` now contains only single-solve BL activities.
It no longer runs Fluent comparison plotting.

## OpenFOAM Validation (Legacy)

These scripts remain available for legacy panel-vs-OpenFOAM workflows:

- `generate_base_case.py`
- `run_of_convergence.py`
- `run_panel_convergence.py`
- `visualize.py`
- `compare_surface.py`
