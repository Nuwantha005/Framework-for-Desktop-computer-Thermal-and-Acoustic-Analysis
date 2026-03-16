# Visualization Module State
**Last modified**: 2026-03-16

## Files
- `visualizer.py` (654 lines) — `OutputManager` (directory/timestamp handling); `Visualizer` facade: `create_figure()`, `plot_mesh()`, `plot_scene()`, `plot_contours()`, `plot_streamlines()`, `plot_cp()`, `finalize()`
- `field2d.py` (204 lines) — `VelocityField2D`: `compute(x_range, y_range, resolution)` → (XX, YY, Vx, Vy); per-component body masking; result caching
- `comparison.py` (978 lines) — `FieldSeries`, `LineSeries`, `ComparisonMetrics` dataclasses; `ComparisonVisualizer`: `compare_contours()`, `plot_difference()`, `compare_lines()`, `compute_metrics()`
- `solver_comparison.py` — `SolverComparisonVisualizer`: generates Vt/Cp envelope overlays, arc-length line charts, difference plots, metrics tables, and ranking charts for inter-solver + OF reference comparison. Outputs to `<case>/out/solver_comparison/`. OF reference drawn as dashed dark-grey line.
- `bl_plots.py` (1027 lines) — Boundary layer visualization. Contains:
  - **Line plots**: `plot_bl_line()`, `plot_bl_lines_multi()`, `plot_bl_two_sides()` — BL quantities (cf, δ*, θ, H, Re_θ) vs arc length, single or two-sided
  - **Envelope plots**: `plot_bl_envelope()`, `plot_bl_envelope_comparison()` — BL quantity wrapped around body surface (single or multi-profile overlay)
  - **Full comparison**: `plot_bl_comparison()` — composite figure with two-sided lines + envelope + Ue reference
  - **Velocity contour (Phase 5)**: `plot_bl_velocity_contour()` — s-y pcolormesh of reconstructed u(s,y) with δ(s) overlay
  - **Normalised contour (Phase 5)**: `plot_bl_velocity_contour_normalized()` — s-(y/δ) normalised rectangle contour
  - **Velocity envelope (Phase 5)**: `plot_bl_velocity_envelope()` — velocity-coloured quads wrapped around body geometry using outward normals
  - **Two-sided wrappers (Phase 5)**: `plot_bl_velocity_contour_two_sides()`, `plot_bl_velocity_contour_normalized_two_sides()`, `plot_bl_velocity_envelope_two_sides()` — upper/lower in one figure
  - **OF comparison placeholder (Phase 5)**: `plot_bl_of_comparison()` — shows panel-method contour with annotation when of_field=None; future: RMS difference contour vs OpenFOAM
  - **Helper**: `_cell_edges()` — cell-edge coordinates from centres for pcolormesh
- `panel2d.py` (184 lines) — `PanelVisualizer2D` (legacy): `compute_field()`, `plot_streamlines()`, `plot_contours()`
- `mesh_plot.py` (364 lines) — `MeshPlotter`; quick helpers: `quick_plot_mesh()`, `quick_plot_component()`, `quick_plot_scene()`
- `surface_envelope.py` (509 lines) — `compute_outward_normals()`, `plot_surface_envelope()`, `plot_surface_envelope_comparison()`, `plot_dual_surface_envelope()`
- `streamlines.py` (346 lines) — `StreamlineVisualizer` (multiprocessing-based, legacy)
- `plotters/contours.py` (109 lines) — `ContourPlotter` with `plot_velocity_magnitude()`, `plot_pressure_coefficient()`
- `plotters/streamlines.py` (133 lines) — `StreamlinePlotter` with `plot()`

## Public API

### Core visualizers
- `Visualizer(output_dir, protect_overwrite)` → `.create_figure()` → `.plot_contours()` → `.finalize(save, show)`
- `VelocityField2D(solver)` → `.compute(x_range, y_range, resolution)` → (XX, YY, Vx, Vy)
- `ComparisonVisualizer(output_dir)` → `.compare_contours(fields, mesh)` → Figure
- `SolverComparisonVisualizer(result, output_dir, subfolder="solver_comparison")` → `.plot_all(show, save)` — plots: `vt_envelope`, `cp_envelope`, `vt_arc_length`, `cp_arc_length`, `vt_dual`, `vt_difference`, `metrics_table`, `ranking`

### BL integral-quantity plots (bl_plots.py)
- `plot_bl_line(path_result, quantity, ax, title, output_path)` → `(fig, ax)` — single-path line plot of BL quantity vs s
- `plot_bl_lines_multi(path_result, quantities, title, output_path)` → `(fig, axes)` — multi-panel line plots
- `plot_bl_two_sides(case_result, quantities, title, output_path)` → `(fig, axes)` — upper|lower side-by-side line plots
- `plot_bl_envelope(case_result, quantity, profile_name, scale, colormap, ax, title, output_path)` → `(fig, ax)` — single-profile envelope on body
- `plot_bl_envelope_comparison(case_result, quantity, scale, ax, title, output_path)` → `(fig, ax)` — multi-profile overlay envelope
- `plot_bl_comparison(case_result, quantities, envelope_quantity, envelope_scale, title, output_path, show)` → `Figure` — full composite comparison figure

### BL velocity-field plots (bl_plots.py, Phase 5)
All require `BLFieldData` from `reconstruct_bl_field()` (set `reconstruct=True` in runner).

- `plot_bl_velocity_contour(field, ax, cmap, show_delta, title, output_path, n_levels)` → `(fig, ax)` — s-y pcolormesh with δ(s) overlay
- `plot_bl_velocity_contour_normalized(field, ax, cmap, title, output_path)` → `(fig, ax)` — s-(y/δ) normalised rectangle contour
- `plot_bl_velocity_envelope(field, surface_x, surface_y, panel_indices, scale, cmap, ax, show_body, title, output_path, n_y_vis)` → `(fig, ax)` — velocity-coloured quads wrapped around body
- `plot_bl_velocity_contour_two_sides(field_upper, field_lower, cmap, show_delta, title, output_path)` → `(fig, (ax_u, ax_l))` — two-panel s-y contour
- `plot_bl_velocity_contour_normalized_two_sides(field_upper, field_lower, cmap, title, output_path)` → `(fig, (ax_u, ax_l))` — two-panel normalised contour
- `plot_bl_velocity_envelope_two_sides(field_upper, field_lower, case_result, scale, cmap, title, output_path, n_y_vis)` → `(fig, ax)` — both paths on one body
- `plot_bl_of_comparison(field, of_field, cmap, ax, title, output_path)` → `(fig, ax)` — OpenFOAM comparison placeholder (shows annotation when of_field=None)

## Data Flow

### Panel solver → field visualization
Solver → `VelocityField2D.compute()` → grid data → `Visualizer.plot_contours/streamlines()` → matplotlib figure → `finalize(save/show)`

`ComparisonResult` → `SolverComparisonVisualizer.plot_all()` → envelope/line/diff/table/ranking PNGs under `<case>/out/solver_comparison/solver_cmp_*.png`

### BL solver → BL visualization
`BoundaryLayerRunner.run(reconstruct=True)` → `BoundaryLayerCaseResult` (with `.upper.fields[name]` / `.lower.fields[name]` → `BLFieldData`) → `plot_bl_velocity_contour()` / `plot_bl_velocity_envelope()` / etc.

For integral-quantity plots (cf, δ*, θ, H): `BoundaryLayerCaseResult` → `plot_bl_line()` / `plot_bl_two_sides()` / `plot_bl_envelope()` / `plot_bl_comparison()`

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `solvers.base.Solver`, `solvers.comparison.ComparisonResult`, `solvers.boundary_layer.field.BLFieldData`, `visualization.surface_envelope.compute_outward_normals`
- External: numpy, matplotlib (including `matplotlib.colors`, `matplotlib.collections.LineCollection`), multiprocessing

## What's Next
- PyVista 3D visualization for 3D panel results
- Interactive plotting (ipywidgets or panel)
- OpenFOAM BL field extraction for `plot_bl_of_comparison()` (currently placeholder)

## Known Issues
- `PanelVisualizer2D` and `StreamlineVisualizer` are legacy; prefer `Visualizer` + `VelocityField2D`
- `plot_bl_velocity_envelope()` uses per-quad `ax.fill()` which can be slow for dense grids; consider `PolyCollection` if performance becomes an issue
