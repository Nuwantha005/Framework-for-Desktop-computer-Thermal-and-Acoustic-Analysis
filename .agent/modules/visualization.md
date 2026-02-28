# Visualization Module State
**Last modified**: 2026-02-28

## Files
- `visualizer.py` (654 lines) — `OutputManager` (directory/timestamp handling); `Visualizer` facade: `create_figure()`, `plot_mesh()`, `plot_scene()`, `plot_contours()`, `plot_streamlines()`, `plot_cp()`, `finalize()`
- `field2d.py` (204 lines) — `VelocityField2D`: `compute(x_range, y_range, resolution)` → (XX, YY, Vx, Vy); per-component body masking; result caching
- `comparison.py` (978 lines) — `FieldSeries`, `LineSeries`, `ComparisonMetrics` dataclasses; `ComparisonVisualizer`: `compare_contours()`, `plot_difference()`, `compare_lines()`, `compute_metrics()`
- `solver_comparison.py` — `SolverComparisonVisualizer`: generates Vt/Cp envelope overlays, arc-length line charts, difference plots, and metrics tables for inter-solver comparison
- `panel2d.py` (184 lines) — `PanelVisualizer2D` (legacy): `compute_field()`, `plot_streamlines()`, `plot_contours()`
- `mesh_plot.py` (364 lines) — `MeshPlotter`; quick helpers: `quick_plot_mesh()`, `quick_plot_component()`, `quick_plot_scene()`
- `surface_envelope.py` (509 lines) — `compute_outward_normals()`, `plot_surface_envelope()`, `plot_surface_envelope_comparison()`, `plot_dual_surface_envelope()`
- `streamlines.py` (346 lines) — `StreamlineVisualizer` (multiprocessing-based, legacy)
- `plotters/contours.py` (109 lines) — `ContourPlotter` with `plot_velocity_magnitude()`, `plot_pressure_coefficient()`
- `plotters/streamlines.py` (133 lines) — `StreamlinePlotter` with `plot()`

## Public API
- `Visualizer(output_dir, protect_overwrite)` → `.create_figure()` → `.plot_contours()` → `.finalize(save, show)`
- `VelocityField2D(solver)` → `.compute(x_range, y_range, resolution)` → (XX, YY, Vx, Vy)
- `ComparisonVisualizer(output_dir)` → `.compare_contours(fields, mesh)` → Figure
- `SolverComparisonVisualizer(result, output_dir)` → `.plot_all(show, save)` — plots: `vt_envelope`, `cp_envelope`, `vt_arc_length`, `cp_arc_length`, `vt_dual`, `vt_difference`, `metrics_table`

## Data Flow
Solver → `VelocityField2D.compute()` → grid data → `Visualizer.plot_contours/streamlines()` → matplotlib figure → `finalize(save/show)`

`ComparisonResult` → `SolverComparisonVisualizer.plot_all()` → envelope/line/diff/table PNGs under `<case>/out/solver_cmp_*.png`

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `solvers.base.Solver`, `solvers.comparison.ComparisonResult`
- External: numpy, matplotlib, multiprocessing

## What's Next
- PyVista 3D visualization for 3D panel results
- Interactive plotting (ipywidgets or panel)

## Known Issues
- `PanelVisualizer2D` and `StreamlineVisualizer` are legacy; prefer `Visualizer` + `VelocityField2D`
