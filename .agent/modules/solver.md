# Solver Module State
**Last modified**: 2026-02-28

## Files
- `solvers/base.py` — `Solver` ABC: `solve()`, `surface_velocity`, `velocity_at(points)`, `is_solved`, `mesh`
- `solvers/factory.py` — `SolverFactory`: registry `(singularity, order, geometry) → class`; `register()`, `create()`, `create_panel_solver()`, `available()`, `is_registered()`
- `solvers/comparison.py` — `SolverComparisonRunner`, `ComparisonResult`, `SolverResult`, `extract_openfoam_reference()`; run N solvers on the same case, optionally include OpenFOAM CFD reference, compute metrics, and rank solvers
- `solvers/panel2d/base.py` — `PanelMethodConfig` (frozen dataclass); `PanelSolver2D(Solver)` ABC with template method: init → `_compute_influence_matrices()` → `_solve_linear_system()` → `_compute_surface_velocity()` → done
- `solvers/panel2d/spm.py` — `SourcePanelSolver(PanelSolver2D)`: constant-strength source panels, Katz & Plotkin formulation; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_source_solver.py` — `LinearSourcePanelSolver(PanelSolver2D)`: linear-strength source panels; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_vortex_solver.py` — `LinearVortexPanelSolver(PanelSolver2D)`: linear-strength vortex panels with zero-circulation closure; properties: `gamma`, `Vt`, `Cp`
- `solvers/panel2d/influences/source.py` — `compute_source_influence_matrices(mesh) → (I, J)`; `compute_source_velocity_influence(point, ...) → (Mx, My)`; `compute_source_potential_influence(point, ...) → coeff`
- `solvers/panel2d/influences/linear_source.py` — `compute_linear_source_influence_matrices(mesh) → (I, J)`; `compute_linear_source_velocity_field(points, mesh, strengths)`
- `solvers/panel2d/influences/linear_vortex.py` — `compute_linear_vortex_influence_matrices(mesh) → (I, J)`; `compute_linear_vortex_velocity_influence(point, ...) → ((Mx_a, My_a), (Mx_b, My_b))`; `compute_linear_vortex_velocity_field(points, mesh, gamma)`

## Public API
- `SourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearSourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearVortexPanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.gamma`, `.surface_velocity`, `.velocity_at(points)`
- `SolverFactory.create(config, mesh, v_inf, aoa) -> PanelSolver2D`
- `Case.create_solver(solver_type=None)` — accepts optional `solver_type` override (e.g. `"linear_source"`)
- `SolverComparisonRunner(case).run(["constant", "linear"], of_case_dir=...) -> ComparisonResult`
- `ComparisonResult.compute_metrics()` — pairwise Vt/Cp error metrics (L∞, RMS, MAE, rel%); when OF reference present, metrics are computed vs reference with interpolation
- `ComparisonResult.ranking` — `List[Tuple[str, float]]` solvers sorted by Vt_rel_L2_pct ascending (best first)
- `ComparisonResult.reference` — returns the OF `SolverResult` if present
- `ComparisonResult.solver_results` — returns only panel-method results
- `extract_openfoam_reference(case, of_case_dir)` → `SolverResult(is_reference=True)` with OF surface data

## Solver Comparison Workflow
```
Case → SolverComparisonRunner.run(["constant", "linear"], of_case_dir=...)
       → extract_openfoam_reference() → SolverResult(is_reference=True)
       → for each panel type: case.create_solver(solver_type=...) → solve() → SurfaceDataExtractor
       → ComparisonResult (OF ref + SolverResult per solver + metrics + ranking)
       → SolverComparisonVisualizer.plot_all()
           → envelope, line, diff, metrics, ranking plots under <case>/out/solver_comparison/
```

## Data Flow
`Mesh` + flow conditions → `PanelSolver2D.__init__()` → `.solve()` computes influence matrices (I, J) → solves Aσ = b → surface velocity from potential gradient → results cached

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `core.io.case.Case`, `postprocessing.surface.SurfaceDataExtractor`
- External: numpy, scipy.linalg (symmetry check), scipy.interpolate (for OF comparison)

## Registered Solvers
- `("source", "constant", "flat")` → `SourcePanelSolver`
- `("source", "linear", "flat")` → `LinearSourcePanelSolver`
- `("vortex", "linear", "flat")` → `LinearVortexPanelSolver`

## Solver Aliases (comparison framework)
- `"constant"` → `"constant_source"`
- `"linear"` → `"linear_source"`
- `"vortex"` → `"linear_vortex"`

## What's Next
- Quadratic strength panels
- Curved panel geometry
- Viscous boundary layer solver (Von Kármán integral) consuming Vt from linear vortex solver

## Known Issues
- Debug `print()` + `scipy.linalg.issymmetric` calls left in `compute_source_influence_matrices()`
- Double `@property sigma` definition in spm.py (lines 48 and 215)
- Inner loops in `_velocity_at_points()` and influence computation — not yet vectorized
