# Solver Module State
**Last modified**: 2026-03-28 (Thermal BL solver integration)

## ADM Update
- `solvers/actuator/` — simple actuator disk model for 3D panel coupling.
- `ActuatorDiskCoupledSolver3D` wraps the configured 3D panel solver from
  `SolverFactory`; it is not hard-wired to `SourcePanelSolver3D`.
- 3D solvers can support ADM by honoring
  `solve(normal_velocity_disturbance=...)` in the body-panel RHS.
- Cases without `actuator_disks` follow the existing solver path unchanged.
- Generic 3D scripts should call `Case.create_solver()` rather than
  `SolverFactory.create()` directly, otherwise ADM coupling is bypassed.

## Files
- `solvers/base.py` — `Solver` ABC: `solve()`, `surface_velocity`, `velocity_at(points)`, `is_solved`, `mesh`
- `solvers/factory.py` — `SolverFactory`: registry `(singularity, order, geometry) → class`; `register()`, `create()`, `create_panel_solver()`, `available()`, `is_registered()`
- `solvers/comparison.py` — `SolverComparisonRunner`, `ComparisonResult`, `SolverResult`, `extract_openfoam_reference()`; run N solvers on the same case, optionally include OpenFOAM CFD reference, compute metrics, and rank solvers
- `solvers/panel2d/base.py` — `PanelMethodConfig` (frozen dataclass); `PanelSolver2D(Solver)` ABC with template method: init → `_compute_influence_matrices()` → `_solve_linear_system()` → `_compute_surface_velocity()` → done
- `solvers/panel2d/spm.py` — `SourcePanelSolver(PanelSolver2D)`: constant-strength source panels, Katz & Plotkin formulation; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_source_solver.py` — `LinearSourcePanelSolver(PanelSolver2D)`: linear-strength source panels; properties: `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_vortex_solver.py` — `LinearVortexPanelSolver(PanelSolver2D)`: linear-strength vortex panels with zero-circulation closure; properties: `gamma`, `Vt`, `Cp`
- `solvers/panel2d/dirichlet_doublet_solver.py` — `DirichletDoubletSolver(PanelSolver2D)`: Morino source+doublet, Dirichlet internal-potential BC, μ₁=0 gauge fix; properties: `mu`, `sigma`, `Vt`, `Cp`
- `solvers/panel2d/linear_source_doublet_solver.py` — `LinearSourceDoubletSolver(PanelSolver2D)`: linear-strength doublet + linear-strength source, Dirichlet internal-potential BC (Morino), lstsq with gauge fix for rank-deficient symmetric meshes; surface Vt via dμ/ds central differences on node-averaged μ; properties: `mu`, `sigma`, `Vt`, `Cp`
- `solvers/panel2d/influences/source.py` — `compute_source_influence_matrices(mesh) → (I, J)`; `compute_source_velocity_influence(point, ...) → (Mx, My)`; `compute_source_potential_influence(point, ...) → coeff`
- `solvers/panel2d/influences/linear_source.py` — `compute_linear_source_influence_matrices(mesh) → (I, J)`; `compute_linear_source_velocity_field(points, mesh, strengths)`
- `solvers/panel2d/influences/linear_vortex.py` — `compute_linear_vortex_influence_matrices(mesh) → (I, J)`; `compute_linear_vortex_velocity_influence(point, ...) → ((Mx_a, My_a), (Mx_b, My_b))`; `compute_linear_vortex_velocity_field(points, mesh, gamma)`
- `solvers/panel2d/influences/doublet.py` — `compute_doublet_potential_influence(point, start, end) → coeff`; `compute_doublet_influence_matrix(mesh) → C`; `compute_source_potential_matrix(mesh) → B`; `compute_doublet_velocity_influence(point, start, end) → (u, w)`
- `solvers/panel3d/base.py` — `PanelSolver3D` ABC: 3D counterpart to PanelSolver2D.
- `solvers/panel3d/source_panel_solver.py` — `SourcePanelSolver3D(PanelSolver3D)`: 3D constant-strength source panel method (Hess-Smith/Katz & Plotkin); properties: `sigma`, `Vt`, `Cp`.
- `solvers/panel3d/influences/source3d.py` — `compute_source_influence_matrix()`, `compute_all_velocities_influence()`: Highly optimized (Numba JIT, parallelized) functions evaluating 3D quad source panel influences.
- `solvers/boundary_layer/__init__.py` — package exports for BL solver and field reconstruction
- `solvers/boundary_layer/base.py` — `BoundaryLayerSolver` (Von Kármán integral ODE solver), `BoundaryLayerResult` (dataclass container for θ, δ*, C_f∞, H, Re_θ); accepts optional `K` for stagnation patching
- `solvers/boundary_layer/runner.py` — `BoundaryLayerRunner` (orchestration), `BoundaryLayerPathResult`, `BoundaryLayerCaseResult`; exact stagnation detection via `_interpolate_stagnation()`, velocity gradient `_compute_K()`, optional velocity-field reconstruction
- `solvers/boundary_layer/field.py` — `BLFieldData` (reconstructed 2-D velocity field), `reconstruct_bl_field()` (batch reconstruction from integral results + profile)
- `solvers/boundary_layer/transition.py` — `michel_criterion()`, `en_criterion()`, `TransitionResult` (frozen dataclass)
- `solvers/boundary_layer/profiles/base.py` — `VelocityProfile` ABC (`compute_closure()`, `initial_theta()`, `stagnation_theta()`, `compute_delta()`, `reconstruct_velocity()`, `name`), `ProfileClosureData` (H, cf_2, optional ratios)
- `solvers/boundary_layer/profiles/blasius.py` — `BlasiusProfile`: flat-plate, H=2.591, stagnation_theta via Blasius ODE constant, reconstruction via Blasius table
- `solvers/boundary_layer/profiles/pohlhausen.py` — `PohlhausenProfile`: 4th-order polynomial, Λ parameter, −12≤Λ≤12, reconstruction via H→Λ inversion
- `solvers/boundary_layer/profiles/falkner_skan.py` — `FalknerSkanProfile`: wedge-flow similarity, tabulated H(β) and S(β)/Re_θ, reconstruction via H→β inversion + F-S table
- `solvers/boundary_layer/profiles/power_law.py` — `PowerLawProfile(n=7)`: turbulent 1/n-th law, H=(n+2)/n, reconstruction via algebraic (y/δ)^(1/n)
- `solvers/boundary_layer/profiles/thwaites.py` — `ThwaitesProfile`: one-param laminar correlation, `quadrature_theta()` utility, configurable reconstruction pairing (falkner_skan or pohlhausen)
- `solvers/boundary_layer/profiles/tables.py` — `BlasiusTable`, `FalknerSkanTable` (lazy singleton loaders with interpolation and ODE fallback)
- `solvers/thermal/__init__.py` — package exports for thermal BL solver
- `solvers/thermal/base.py` — `ThermalBLInput` (input from viscous BL), `ThermalResult` (output), `ThermalSolverConfig`, `ThermalSolver` ABC, `extract_thermal_input()` helper
- `solvers/thermal/reynolds_analogy.py` — `ReynoldsAnalogyThermal(ThermalSolver)`: Chilton-Colburn analogy for laminar/turbulent BLs
- `solvers/thermal/factory.py` — `ThermalSolverFactory`, `create_thermal_solver()`: factory for creating solvers from case config
- `solvers/thermal/bdim/solver.py` — `BDIMThermalSolver`, `BDIMInput`, `BDIMConfig`: boundary-domain integral method (advanced, requires domain mesh)

## Public API
- `SourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearSourcePanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearVortexPanelSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.gamma`, `.surface_velocity`, `.velocity_at(points)`
- `DirichletDoubletSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.mu`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `LinearSourceDoubletSolver(mesh, v_inf=1.0, aoa=0.0)` → `.solve()` → `.Cp`, `.Vt`, `.mu`, `.sigma`, `.surface_velocity`, `.velocity_at(points)`
- `SolverFactory.create(config, mesh, v_inf, aoa) -> PanelSolver2D`
- `Case.create_solver(solver_type=None)` — accepts optional `solver_type` override (e.g. `"linear_source"`)
- `SolverComparisonRunner(case).run(["constant", "linear"], of_case_dir=...) -> ComparisonResult`
- `ComparisonResult.compute_metrics()` — pairwise Vt/Cp error metrics (L∞, RMS, MAE, rel%); when OF reference present, metrics are computed vs reference with interpolation
- `ComparisonResult.ranking` — `List[Tuple[str, float]]` solvers sorted by Vt_rel_L2_pct ascending (best first)
- `ComparisonResult.reference` — returns the OF `SolverResult` if present
- `ComparisonResult.solver_results` — returns only panel-method results
- `extract_openfoam_reference(case, of_case_dir)` → `SolverResult(is_reference=True)` with OF surface data
- `BoundaryLayerSolver(edge_velocity, arc_length, nu, profile)` → `.solve(K=None)` → `BoundaryLayerResult`
- `BoundaryLayerResult`: dataclass with `.s`, `.theta`, `.delta_star`, `.cf` (freestream-normalized `C_f∞`), `.H`, `.Re_theta`, `.Ue`, `.transition_s`, `.profile_name`, `.converged`
- `BoundaryLayerRunner(case, solver)` → `.run(profiles, reconstruct=False, ...)` → `BoundaryLayerCaseResult`
- `BoundaryLayerPathResult`: `.results` (profile→result), `.fields` (profile→BLFieldData), `.K`, `.transitions`
- `BLFieldData`: reconstructed 2-D field (`.s`, `.y`, `.u`, `.delta`, `.Ue`, `.theta`, `.H`, `.profile_name`)
- `reconstruct_bl_field(result, profile, n_y=80, extend=1.0)` → `BLFieldData`
- `VelocityProfile` ABC → `compute_closure()`, `initial_theta()`, `stagnation_theta(nu, K)`, `compute_delta(theta, H)`, `reconstruct_velocity(y, theta, H, Ue)`
- Profiles: `BlasiusProfile()`, `ThwaitesProfile(reconstruction="falkner_skan"|"pohlhausen")`, `PohlhausenProfile()`, `FalknerSkanProfile()`, `PowerLawProfile(n=7)`
- Tables: `blasius_table()`, `falkner_skan_table()` (module-level singleton accessors)
- Transition: `michel_criterion(s, Ue, theta, nu) → TransitionResult`, `en_criterion(s, Ue, theta, nu, n_crit=9) → TransitionResult`
- `ThwaitesProfile.quadrature_theta(s, Ue, nu)` — direct θ(s) via Thwaites' closed-form integral
- `extract_thermal_input(bl_path_result, profile_name) → ThermalBLInput` — extracts thermal solver input from viscous BL
- `ThermalBLInput` — common interface for viscous BL → thermal solver (arc_length, Ue, cf, coordinates)
- `ThermalResult` — thermal solver output (T_w, h, Nu, q_w, δ_T, total_heat_rate)
- `ThermalSolverConfig` — configuration (T_inf, Pr, k, rho, cp, q_wall or T_wall BC)
- `ReynoldsAnalogyThermal(bl_input, config)` → `.solve()` → `ThermalResult`
- `ThermalSolverFactory.create_from_case(case, bl_input)` → thermal solver from case config
- `create_thermal_solver(type, bl_input, config)` → instantiate by type name

## Solver Comparison Workflow
```
Case → SolverComparisonRunner.run(["constant", "linear"], of_case_dir=...)
       → extract_openfoam_reference() → SolverResult(is_reference=True)
       → for each panel type: case.create_solver(solver_type=...) → solve() → SurfaceDataExtractor
       → ComparisonResult (OF ref + SolverResult per solver + metrics + ranking)
       → SolverComparisonVisualizer.plot_all()
           → envelope, line, diff, metrics, ranking plots under <case>/out/solver_comparison/
```

## Thermal BL Workflow
```
Viscous BL (BoundaryLayerPathResult)
  → extract_thermal_input(path, "thwaites")
  → ThermalBLInput (s, Ue, cf, x, y)
  → ThermalSolverConfig (T_inf, Pr, k, q_wall or T_wall)
  → ReynoldsAnalogyThermal(input, config).solve()
  → ThermalResult (T_w, h, Nu, q_w, δ_T per station)
  → plot_thermal_two_sides(), plot_thermal_envelope_two_sides()
```

Case YAML thermal config example:
```yaml
thermal:
  enabled: true
  solver: "reynolds_analogy"
  envelope_scale: 0.15

fluid:
  freestream_temperature: 300.0
  thermal_conductivity: 0.026
  specific_heat_cp: 1005.0

components:
  - name: "body"
    boundary_condition:
      type: "wall"
      heat_flux: 500.0  # W/m² uniform
```

## Data Flow
`Mesh` + flow conditions → `PanelSolver2D.__init__()` → `.solve()` computes influence matrices (I, J) → solves Aσ = b → surface velocity from potential gradient → results cached

## Dependencies
- Internal: `core.geometry.mesh.Mesh`, `core.io.case.Case`, `postprocessing.surface.SurfaceDataExtractor`
- External: numpy, scipy.linalg (symmetry check), scipy.interpolate (for OF comparison, BL solver), scipy.integrate (BL ODE)

## Registered Solvers
- `("source", "constant", "flat")` → `SourcePanelSolver`
- `("source", "linear", "flat")` → `LinearSourcePanelSolver`
- `("vortex", "linear", "flat")` → `LinearVortexPanelSolver`
- `("source_doublet", "constant", "flat")` → `DirichletDoubletSolver`
- `("source_doublet", "linear", "flat")` → `LinearSourceDoubletSolver`
- `("source_3d", "constant", "flat")` → `SourcePanelSolver3D`

## Solver Aliases (comparison framework)
- `"constant"` → `"constant_source"`
- `"linear"` → `"linear_source"`
- `"vortex"` → `"linear_vortex"`
- `"doublet"` → `"source_doublet"`
- `"linear_doublet"` → `"linear_source_doublet"`

## What's Next
- Quadratic strength panels
- Curved panel geometry
- Viscous-inviscid coupling: feed BL δ* back as transpiration velocity to panel method
- BDIM thermal solver for full domain thermal field (requires domain mesh setup)
- Validate thermal BL against OpenFOAM or analytical solutions
- Validate BL solver against OpenFOAM wallShearStress on existing cases

## Known Issues
- Debug `print()` + `scipy.linalg.issymmetric` calls left in `compute_source_influence_matrices()`
- Double `@property sigma` definition in spm.py (lines 48 and 215)
- Inner loops in `_velocity_at_points()` and influence computation — not yet vectorized
- BL Falkner-Skan S(β) table values for β > 0.5 are approximate (derived from correlations)
- BL e^N transition model uses simplified growth rate — adequate for engineering estimates
- BDIM thermal solver requires manual domain setup — not integrated with case loader yet
