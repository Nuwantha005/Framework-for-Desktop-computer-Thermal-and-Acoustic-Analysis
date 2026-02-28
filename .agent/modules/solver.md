# Solver Module State
**Last modified**: 2026-02-28

## Files
- `solvers/base.py` — `Solver` ABC: `solve()`, `surface_velocity`, `velocity_at(points)`, `is_solved`, `mesh`
- `solvers/factory.py` — `SolverFactory`: registry `(singularity, order, geometry) → class`; `register()`, `create()`, `create_panel_solver()`, `available()`, `is_registered()`
- `solvers/comparison.py` — `SolverComparisonRunner`, `ComparisonResult`, `SolverResult`; run N solvers on the same case and collect Vt/Cp metrics
- `solvers/panel2d/base.py` — `PanelMethodConfig` (frozen dataclass); `PanelSolver2D(Solver)` ABC with template method: init → `_compute_influence_matrices()` → `_solve_linear_system()` → `_compute_surface_velocity()` → done
- `solvers/panel2d/spm.py` — `SourcePanelSolver(PanelSolver2D)`: constant-strength source panels, Katz & Plotkin formulation; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_source_solver.py` — `LinearSourcePanelSolver(PanelSolver2D)`: linear-strength source panels; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/influences/source.py` — `compute_source_influence_matrices(mesh) → (I, J)`; `compute_source_velocity_influence(point, ...) → (Mx, My)`; `compute_source_potential_influence(point, ...) → coeff`

## Public API
- `SourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearSourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `SolverFactory.create(config, mesh, v_inf, aoa) -> PanelSolver2D`
- `Case.create_solver(solver_type=None)` — now accepts optional `solver_type` override (e.g. `"linear_source"`)
- `SolverComparisonRunner(case).run(["constant", "linear"]) -> ComparisonResult`
- `ComparisonResult.compute_metrics()` — pairwise Vt/Cp error metrics (L∞, RMS, MAE, rel%)

## Solver Comparison Workflow
```
Case → SolverComparisonRunner.run(["constant", "linear"])
       → for each type: case.create_solver(solver_type=...) → solve() → SurfaceDataExtractor
       → ComparisonResult (with SolverResult per solver + pairwise metrics)
       → SolverComparisonVisualizer.plot_all() → envelope, line, diff, metrics plots
```

## Data Flow
`Mesh` + flow conditions → `PanelSolver2D.__init__()` → `.solve()` computes influence matrices (I, J) → solves Aσ = b → surface velocity from potential gradient → results cached

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `core.io.case.Case`, `postprocessing.surface.SurfaceDataExtractor`
- External: numpy, scipy.linalg (symmetry check)

## Registered Solvers
- `("source", "constant", "flat")` → `SourcePanelSolver`
- `("source", "linear", "flat")` → `LinearSourcePanelSolver`

## What's Next
- Constant vortex panels → lifting bodies + Kutta condition
- Linear vortex panels
- Curved panel geometry

## Known Issues
- Debug `print()` + `scipy.linalg.issymmetric` calls left in `compute_source_influence_matrices()`
- Double `@property sigma` definition in spm.py (lines 48 and 215)
- Inner loops in `_velocity_at_points()` and influence computation — not yet vectorized
