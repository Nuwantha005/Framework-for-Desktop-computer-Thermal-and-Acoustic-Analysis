# Solver Module State
**Last modified**: 2026-02-25

## Files
- `solvers/base.py` — `Solver` ABC: `solve()`, `surface_velocity`, `velocity_at(points)`, `is_solved`, `mesh`
- `solvers/factory.py` — `SolverFactory`: registry `(singularity, order, geometry) → class`; `register()`, `create()`, `create_panel_solver()`, `available()`, `is_registered()`
- `solvers/panel2d/base.py` — `PanelMethodConfig` (frozen dataclass); `PanelSolver2D(Solver)` ABC with template method: init → `_compute_influence_matrices()` → `_solve_linear_system()` → `_compute_surface_velocity()` → done
- `solvers/panel2d/spm.py` — `SourcePanelSolver(PanelSolver2D)`: constant-strength source panels, Katz & Plotkin formulation; properties: `sigma`, `Vt`, `Cp`; 424 lines
- `solvers/panel2d/influences/source.py` — `compute_source_influence_matrices(mesh) → (I, J)`; `compute_source_velocity_influence(point, ...) → (Mx, My)`; `compute_source_potential_influence(point, ...) → coeff`

## Public API
- `SourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `SolverFactory.create(config, mesh, v_inf, aoa) -> PanelSolver2D`

## Data Flow
`Mesh` + flow conditions → `SourcePanelSolver.__init__()` → `.solve()` computes influence matrices (I, J) → solves Aσ = b → surface velocity from potential gradient → results cached

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `visualization.surface_envelope` (for validation plots)
- External: numpy, scipy.linalg (symmetry check)

## Registered Solvers
- `("source", "constant", "flat")` → `SourcePanelSolver`

## What's Next
- Constant vortex panels → lifting bodies + Kutta condition
- Linear strength source/vortex panels
- Curved panel geometry

## Known Issues
- Debug `print()` + `scipy.linalg.issymmetric` calls left in `compute_source_influence_matrices()`
- Double `@property sigma` definition in spm.py (lines 48 and 215)
- Inner loops in `_velocity_at_points()` and influence computation — not yet vectorized
