# Validation Scripts

This folder contains validation utilities for two tracks:

- Legacy OpenFOAM pipeline (panel-method validation)
- Current Fluent boundary-layer (BL) validation pipeline

## Fluent BL Validation (Current)

Run from repository root.

### Velocity Field Plots

Difference-style plots (velocity contours, normalized contours, wrapped envelopes):

```bash
python validation/scripts/plot_bl_fluent_difference.py cases/rounded_square
```

Side-by-side absolute plots:

```bash
python validation/scripts/plot_bl_fluent_side_by_side.py cases/rounded_square
```

### Wall Quantity Envelope Plots

Wall quantity envelopes (Ue, Cf, delta, Cp) wrapped around the body:

```bash
# Both side-by-side and overlay plots (default)
python validation/scripts/plot_bl_fluent_wall_envelopes.py cases/rounded_square

# Overlay only (both results on same body)
python validation/scripts/plot_bl_fluent_wall_envelopes.py cases/rounded_square --mode overlay

# Side-by-side only
python validation/scripts/plot_bl_fluent_wall_envelopes.py cases/rounded_square --mode side_by_side

# Specific quantities
python validation/scripts/plot_bl_fluent_wall_envelopes.py cases/rounded_square --quantities Ue Cf

# 2x2 grid of all quantities
python validation/scripts/plot_bl_fluent_wall_envelopes.py cases/rounded_square --mode grid
```

### Common Options

Useful options (all scripts):

- `--compare-profile thwaites` (single profile for comparison)
- `--profiles blasius thwaites` (BL solver profiles to solve)
- `--mesh-level -1` (mesh refinement level index)
- `--solver-type linear_source` (panel solver type)
- `--output-dir <path>` (custom output directory)
- `--show-plots` (display plots interactively)

Default output folders:

- `cases/<case>/out/boundary_layer/fluent_comparison/difference/`
- `cases/<case>/out/boundary_layer/fluent_comparison/side_by_side/`
- `cases/<case>/out/boundary_layer/fluent_comparison/wall_envelopes/`

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
