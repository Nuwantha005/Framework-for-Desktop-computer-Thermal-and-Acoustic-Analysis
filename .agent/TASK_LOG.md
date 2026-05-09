# Task Log

## 2026-02-25
### Documentation system setup
- **What was done**: Created `.agent/` context structure and MkDocs documentation site
- **Files created**:
  - `pyproject.toml` — PEP 621 package config for editable install
  - `.agent/PROJECT_CONTEXT.md` — full project context
  - `.agent/CONVENTIONS.md` — coding conventions
  - `.agent/TASK_LOG.md` — this file
  - `.agent/modules/{solver,geometry,io,visualization,postprocessing,validation}.md`
  - `mkdocs.yml` — MkDocs Material configuration
  - `docs/index.md` — home page with status table
  - `docs/architecture.md` — module connection diagram
  - `docs/user_guide/{getting_started,case_files,validation}.md`
  - `docs/theory/{panel_methods,boundary_layers}.md`
  - `docs/modules/{solver,geometry,io,visualization,validation}.md`
  - `docs/api/index.md` — mkdocstrings auto-generated API reference
  - `docs/dev/{roadmap,decisions}.md`
- **Other changes**: `notes/` renamed to `notes_archived/`; module docstring added to `spm.py`
- **Status**: Complete

## 2026-02-26
### Agent infrastructure setup
- **What was done**: Set up Copilot custom instructions, reusable task prompts, decision log, and updated all agent context files for the multi-phase roadmap.
- **Files created**:
  - `.github/copilot-instructions.md` — Copilot auto-loaded project instructions
  - `.agent/prompts/implement-solver.md` — 6-step solver implementation guide
  - `.agent/prompts/validate-vt.md` — tangential velocity debugging loop
  - `.agent/prompts/implement-bl-solver.md` — Von Kármán BL solver prompt
  - `.agent/prompts/implement-thermal-bl.md` — BDIM thermal BL solver prompt
  - `.agent/prompts/port-to-3d.md` — 3D panel method + PyVista prompt
  - `.agent/decisions/README.md` — decision record template
- **Files modified**:
  - `.agent/CONVENTIONS.md` — added MCP documentation, agent workflow pattern, prompt index
  - `.agent/PROJECT_CONTEXT.md` — added Current Focus section, agent infrastructure section, roadmap
  - `AGENTS.md` — added Pylance MCP, agent context system, prompts index, current focus, limitations update
- **Status**: Complete

## 2026-02-28
### Implement Linear Source Solver
- **What was done**: Implemented linear source panel method following Katz & Plotkin formulas.
- **Files created**: 
  - `src/solvers/panel2d/influences/linear_source.py`
  - `src/solvers/panel2d/linear_source_solver.py`
  - `docs/theory/linear_source_panels.md`
- **Files modified**:
  - `src/solvers/__init__.py`
  - `.agent/modules/solver.md`
- **Status**: Complete

### Implement Linear Vortex Solver
- **What was done**: Implemented linear-strength vortex panel method adapted for non-lifting bluff bodies using Zero Net Circulation closure. Derived vortex influence coefficients from linear source via rotation identity. Validated against OpenFOAM on rounded_square (3.78% Vt rel RMS). Registered solver in factory, comparison framework, and config schemas.
- **Files created**:
  - `src/solvers/panel2d/influences/linear_vortex.py` — influence coefficient functions
  - `src/solvers/panel2d/linear_vortex_solver.py` — solver class with lstsq overdetermined solve
  - `docs/theory/linear_vortex_panels.md` — full theory documentation
- **Files modified**:
  - `src/solvers/__init__.py` — register `("vortex", "linear", "flat")`
  - `src/solvers/panel2d/__init__.py` — export `LinearVortexPanelSolver`
  - `src/solvers/comparison.py` — add `"vortex"` / `"linear_vortex"` aliases
  - `src/core/config/schemas.py` — add `"linear_vortex"` to legacy type Literal
  - `demos/demo_solver_comparison.py` — add vortex to default solvers list
  - `mkdocs.yml` — add Linear Vortex Panels to nav
  - `docs/theory/panel_methods_overview.md` — add link to vortex page
  - `docs/modules/solver.md` — update architecture, factory, influences, file layout
  - `.agent/modules/solver.md` — add vortex solver to registered solvers, files, API
  - `.agent/PROJECT_CONTEXT.md` — mark vortex done, update focus to BL solver phase
  - `.agent/decisions/README.md` — record zero-circulation closure decision
  - `.agent/TASK_LOG.md` — this entry
- **Status**: Complete

### Implement Dirichlet Doublet (Morino) Solver
- **What was done**: Implemented constant-strength Dirichlet doublet panel method (Morino source+doublet formulation) adapted for non-lifting bluff bodies. Sources prescribed (σ = n̂·V∞), doublets solved from Dirichlet internal-potential BC (C·μ = −B·σ). Null space cured by pinning μ₁=0 per component. Surface velocity via dμ/ds + V∞·t̂. Validated against OpenFOAM on rounded_square (50.87% Vt rel RMS — expected for constant-order elements, matches constant source at 50.74%).
- **Files created**:
  - `src/solvers/panel2d/influences/doublet.py` — doublet potential & velocity influence functions
  - `src/solvers/panel2d/dirichlet_doublet_solver.py` — solver class with Morino formulation
  - `docs/theory/dirichlet_doublet_panels.md` — full theory documentation
- **Files modified**:
  - `src/solvers/__init__.py` — register `("source_doublet", "constant", "flat")`
  - `src/solvers/panel2d/__init__.py` — export `DirichletDoubletSolver`
  - `src/solvers/panel2d/influences/__init__.py` — export doublet functions
  - `src/solvers/comparison.py` — add `"doublet"` / `"source_doublet"` aliases
  - `src/core/config/schemas.py` — add `"source_doublet"` to legacy type Literal + type_mapping
  - `mkdocs.yml` — add Dirichlet Doublet Panels to nav
  - `docs/theory/panel_methods_overview.md` — add link to doublet page
  - `docs/modules/solver.md` — update architecture, factory, influences, file layout, planned table
  - `.agent/modules/solver.md` — add doublet solver to registered solvers, files, API, aliases
  - `.agent/PROJECT_CONTEXT.md` — mark doublet done, update focus, module map, source structure
  - `.agent/decisions/README.md` — record Morino bluff-body adaptation decision
  - `.agent/TASK_LOG.md` — this entry
- **Status**: Complete

## 2026-03-01
### Implement Linear Source/Doublet Solver (Dirichlet, K&P §11.5.1)
- **What was done**: Implemented linear-strength doublet + linear-strength source panel method (Morino/Dirichlet internal-potential BC) following Katz & Plotkin §11.5.1. Node-based unknowns (μ at N nodes on closed bodies). Source strengths σ prescribed from averaged panel normals. Surface velocity via dμ/ds central differences on node-interpolated μ plus V∞·t̂. Uses lstsq with gauge fix for robustness on symmetric meshes with rank deficiency > 1. Validated on rounded_square: 50.83% Vt error vs OpenFOAM — matches constant-order Dirichlet methods (50.87% for constant doublet) as expected since dμ/ds extraction is constant-order for velocity.
- **Files created**:
  - `src/solvers/panel2d/influences/linear_doublet.py` — 6 functions: linear doublet/source potential influences (K&P 11.114/11.115), node-accumulation influence matrices, velocity influences, batch velocity field
  - `src/solvers/panel2d/linear_source_doublet_solver.py` — `LinearSourceDoubletSolver(PanelSolver2D)`
- **Files modified**:
  - `src/solvers/__init__.py` — register `("source_doublet", "linear", "flat")`
  - `src/solvers/panel2d/__init__.py` — export `LinearSourceDoubletSolver`
  - `src/solvers/panel2d/influences/__init__.py` — export linear_doublet functions
  - `src/solvers/comparison.py` — add `"linear_doublet"` / `"linear_source_doublet"` aliases
  - `src/core/config/schemas.py` — add `"linear_source_doublet"` to legacy type Literal + type_mapping
- **Documentation updated**:
  - `.agent/modules/solver.md` — added solver file, influence module, API, factory key, alias
  - `.agent/PROJECT_CONTEXT.md` — added solver to checklist, module map, source tree; updated focus
  - `docs/modules/solver.md` — added architecture entry, factory listing, influence docs, file layout, planned table
  - `.agent/TASK_LOG.md` — this entry
- **Status**: Complete

## 2026-03-26
### Add Wall Quantity Envelope Plots for BL-Fluent Comparison
- **What was done**: Added wall quantity envelope plotting functions (Ue, Cf, δ, Cp) for comparing BL solver results against Fluent CFD. Created a new dedicated module to avoid bloating existing files. Two plot variants: side-by-side (BL left, Fluent right) and overlay (both on same body). Also added a 2x2 grid option for all four quantities.
- **Files created**:
  - `src/visualization/bl_wall_envelope_plots.py` — wall quantity envelope plot functions
  - `validation/scripts/plot_bl_fluent_wall_envelopes.py` — validation script
- **Files modified**:
  - `src/visualization/__init__.py` — added exports for new functions
  - `validation/scripts/README.md` — added usage documentation
  - `.agent/TASK_LOG.md` — this entry
- **New functions**:
  - `plot_wall_quantity_envelope_side_by_side()` — side-by-side comparison
  - `plot_wall_quantity_envelope_overlay()` — overlay on same body
  - `plot_wall_quantity_envelopes_grid()` — 2x2 grid of all quantities
- **Status**: Complete

## 2026-03-31
### 3D Panel Solver Implementation - Session Start
- **What was done**: Planned 3D panel solver extension, documented architecture decisions, created implementation plan.
- **Files created**:
  - `.agent/plans/3d-panel-solver.md` — comprehensive 4-phase implementation plan
  - `.agent/decisions/002-3d-panel-solver-architecture.md` — ADR for mesh hierarchy, panel type, visualization choices
- **Key decisions**:
  - Refactor mesh to `MeshBase`/`Mesh2D`/`Mesh3D` hierarchy
  - Quad panels only (not triangles)
  - pygmsh for sphere generation, meshio for STL import
  - VTK export for ParaView visualization
  - Phase 0 refactoring on `main`, Phases 1-3 on `3d-panel-solver-implemention` branch
- **Branches**:
  - `main` — will receive mesh refactoring
  - `3d-panel-solver-implemention` — feature branch for 3D solver
  - `boundary-layer-experimentation` — pushed earlier (thermal BL WIP)
## 2026-04-01
### 3D Panel Solver Implementation - Performance Optimization
- **What was done**: Profiled and optimized the 3D source panel method solver. Identified the primary bottleneck in the $O(N^2)$ influence matrix and velocity computations running in pure Python. Implemented extensive performance optimizations using Numba JIT compilation (`@njit(fastmath=True, cache=True)`) and parallelization (`prange`) across both influence matrix construction and full-domain induced velocity calculations.
- **Files modified**:
  - `src/solvers/panel3d/influences/source3d.py` — JIT-compiled all math functions and added `compute_all_velocities_influence` for parallel multi-point velocity evaluation.
  - `src/solvers/panel3d/source_panel_solver.py` — Updated to consume the new vectorized/parallelized influence functions for surface velocity calculations.
  - `src/solvers/panel3d/influences/__init__.py` — Exported new functions.
  - `profile_solver.py` — Updated testing script to measure and prove timing improvements.
  - `requirements.txt` — Added `numba>=0.58.0` dependency.
- **Results**: Build time for a ~2000 panel mesh (level 1) decreased from `>120` seconds to `~0.5` seconds for the influence matrix, and surface velocity computation down to `~0.5` seconds. Tested successfully up to `level 2` (8192 panels, ~26s total solve time on single machine). Validation sphere tests run successfully with zero precision regression.
- **Status**: Complete

## 2026-04-29
### Add Cylinder Generator + Circular Vent Case
- **What was done**: Added an open cylinder (shell) parametric generator for 3D cases and created a circular vent case definition (1.0 m length, 120 mm diameter).
- **Files modified**:
  - `src/core/geometry/io/sphere_generator.py` — added `generate_cylinder()`
  - `src/core/geometry/io/__init__.py` — export cylinder generator
  - `src/core/geometry/factory.py` — register `"cylinder"` type
  - `src/core/config/schemas.py` — update geometry type description
- **Files created**:
  - `cases/cicular_vent/case.yaml` — circular vent case definition
- **Status**: Complete

## 2026-04-30
### Actuator Disk Model Planning
- **What was done**: Reviewed ADM implementation prompt, ADM theory notes, existing 3D solver architecture, solver factory, circular vent case, and current solver module documentation. Defined the ADM architecture as a solver-agnostic 3D coupling layer that creates the configured body solver through `SolverFactory` and applies actuator disk influence as a known RHS disturbance.
- **Files created**:
  - `.agent/plans/adm-implementation.md` — implementation plan for ADM schema, fan curves, disk mesh generation, 3D solver coupling, persistence, visualization/export, docs, and tests.
- **Key decisions**:
  - Use `normal` in `case.yaml` to define disk orientation and positive flow direction.
  - Keep ADM coupled to the `PanelSolver3D`/`SolverFactory` path instead of hard-coding `SourcePanelSolver3D`.
  - Add a minimal 3D solver hook for external normal-velocity disturbances so future 3D singularity solvers can participate without major rewrites.
  - Skip Fluent validation execution until Fluent exports are available.
- **Status**: Planned

### Simple Actuator Disk Model Implementation
- **What was done**: Implemented the first simple ADM slice for 3D panel coupling. Added optional case schema support, fan-curve loading/interpolation, polar actuator disk mesh generation, point-doublet disk influence, a solver-agnostic `ActuatorDiskCoupledSolver3D` wrapper, persistence/plotting outputs, and a 3D body-solver RHS disturbance hook. Cases without `actuator_disks` continue to create the configured solver directly.
- **Files created**:
  - `src/solvers/actuator/` — ADM package (`fan_curve`, `disk_mesh`, `doublet_influence`, `coupling`, `models`, `persistence`, `plotting`)
  - `docs/theory/actuator_disk_model.md` — theory, sign convention, P-Q loop, limitations
  - `src/test/test_actuator_disk.py` — focused tests for curve loading, disk mesh, doublet mapping, and solver factory paths
- **Files modified**:
  - `src/core/config/schemas.py` — optional `actuator_disks` schema
  - `src/core/io/case.py` — create ADM coupled solver only when 3D actuator disks are present
  - `src/solvers/factory.py` — preserve true 3D freestream vectors for 3D solvers
  - `src/solvers/panel3d/base.py` and `source_panel_solver.py` — optional normal-velocity disturbance hook
  - `demos/demo_case_mesh_export.py` — export actuator disk meshes alongside body mesh
  - `cases/cicular_vent/case.yaml` — placeholder 120 mm fan config and z-axis freestream
  - `docs/modules/solver.md`, `.agent/modules/solver.md` — ADM architecture notes
- **Checks run**:
  - `python -m pytest src/test/test_actuator_disk.py -q` — 5 passed
  - `python -m pytest src/test/test_geometry_foundation.py src/test/test_actuator_disk.py -q` — 19 passed
  - `python -m pytest src/test -q` — 23 passed
  - Circular vent ADM smoke solve completed and wrote `out/adm/` + `out/solverRuns/`; placeholder fan data did not converge within configured iterations because the operating point sits near the fan-curve tail.
  - `python demos/demo_case_mesh_export.py cases/cicular_vent --mesh-level 0 --output-name mesh_level_0_with_adm.vtp` — exported body and disk meshes.
- **Status**: Initial implementation complete; Fluent validation intentionally not run.

### ADM Demo and Fan-Curve Bounds Handling
- **What was done**: Added a runnable ADM demo script with `--case`, made direct execution of `src/test/test_actuator_disk.py` explain pytest/demo usage, added fan-curve progression plotting, and stopped ADM iterations immediately when evaluated flow rate leaves the supplied fan-curve bounds.
- **Files created**:
  - `demos/demo_actuator_disk.py` — runs a case, prints ADM iteration output, and exports body/disk artifacts.
- **Files modified**:
  - `src/solvers/actuator/fan_curve.py` — exposed fan-curve bounds and range checking.
  - `src/solvers/actuator/coupling.py` — early stop + warning on out-of-bounds fan operating point.
  - `src/solvers/actuator/plotting.py` — added `adm_fan_curve_progression.png`.
  - `src/solvers/actuator/models.py`, `demos/README.md`, `docs/modules/solver.md`, `docs/theory/actuator_disk_model.md`, `src/test/test_actuator_disk.py`.
- **Checks run**:
  - `python demos/demo_actuator_disk.py --case cases/cicular_vent --mesh-level 0`
  - `python demos/demo_actuator_disk.py --case cases/cicular_vent --mesh-level -1 --no-surface-export`
  - `python -m pytest src/test/test_actuator_disk.py -q` — 5 passed
  - `python -m pytest src/test -q` — 23 passed
- **Status**: Complete

### ADM Stationary Ambient Operating Mode
- **What was done**: Updated the circular vent setup to use zero freestream for fan-driven flow, changed ADM pressure-jump scaling to use a disk-radius length scale and fan-curve velocity scale when freestream is stationary, and sampled disk flow on offset planes so the disk-induced field contributes without evaluating on the singular sheet.
- **Rationale**: A prescribed `1 m/s` freestream was acting as an artificial inlet velocity and forcing the flow rate outside the fan operating range. For the P12 PWM curve, the max-speed operating condition is already represented by the supplied P-Q data; RPM/PWM only needs explicit handling when scaling to a different speed.
- **Checks run**:
  - `python demos/demo_actuator_disk.py --case cases/cicular_vent --mesh-level 0` — converged in 19 iterations
  - `python demos/demo_actuator_disk.py --case cases/cicular_vent --mesh-level -1 --no-surface-export` — converged in 13 iterations
  - `python -m pytest src/test/test_actuator_disk.py -q` — 5 passed
  - `python -m pytest src/test -q` — 23 passed
- **Status**: Complete

### Route 3D Export Scripts Through Case Solver
- **What was done**: Updated generic 3D visualization/export scripts to use `CaseLoader.load_case(...).create_solver()` so cases with `actuator_disks` automatically run the ADM-coupled solver instead of bypassing it with direct `SolverFactory` calls.
- **Files modified**:
  - `validation/scripts/3d/surface_streamlines.py`
  - `validation/scripts/3d/vector_glyphs.py`
  - `validation/scripts/3d/plot_cut_plane.py`
  - `validation/scripts/3d/compare_surface.py`
  - `validation/scripts/3d/convergence_study.py`
  - `demos/demo_case_paraview_export.py`
  - `src/solvers/panel3d/base.py` — zero-freestream Cp guard
  - `docs/modules/solver.md`, `.agent/modules/solver.md`
- **Checks run**:
  - `python validation/scripts/3d/surface_streamlines.py cases/cicular_vent` — ADM-coupled solve and VTP export
  - `python validation/scripts/3d/plot_cut_plane.py cases/cicular_vent --mesh-level 0 --resolution 30 --axis z --offset 0.0`
  - `python demos/demo_case_paraview_export.py cases/cicular_vent --mesh-level 0 --resolution 8 8 8 --output-dir cases/cicular_vent/out/panel_solver/adm_smoke`
  - `python validation/scripts/3d/compare_surface.py cases/cicular_vent --mesh-level 0`
  - `python -m pytest src/test/test_actuator_disk.py -q` — 5 passed
  - `python -m pytest src/test -q` — 23 passed
- **Status**: Complete

### ADM Constant Strength Doublet Integration
- **What was done**: Replaced point-doublet approximations with mathematically correct constant-strength doublet panels for the Actuator Disk Model (ADM). Point doublets suffered from tip-leakage where flow looped backwards inside the duct; constant doublets (equivalent to vortex rings around panel perimeters) enforce a contiguous pressure jump sheet, channeling flow properly to the duct ends.
- **Files created**:
  - `src/solvers/actuator/doublet_influence.py` — implemented exact quad doublet velocity using `compute_all_doublet_velocities` from `doublet3d.py`.
- **Files deleted**:
  - `src/solvers/actuator/source_influence.py` — removed abandoned source panel test file.
- **Files modified**:
  - `docs/theory/actuator_disk_model.md` — updated theory docs reflecting the shift to contiguous doublet sheets (vortex rings).
- **Checks run**:
  - `pytest src/test/test_actuator_disk.py` — 5 passed.
- **Status**: Complete

### ADM Sign Convention Fix
- **What was done**: Fixed a sign issue in the exact quad doublet velocity calculation. `Mesh3D` uses `(p4-p2)x(p3-p1)` for normal generation, meaning geometry generators produce clockwise-ordered panels to point outward. A standard positive CCW vortex ring implies that CW panels induce velocity *opposite* to their normal. Subtracted the Biot-Savart segment contributions so that a positive doublet (a dipole pointing along the normal) correctly induces flow in the `+n` direction. This resolves the negative flow rate seen during the first ADM iteration.
- **Files modified**:
  - `src/solvers/panel3d/influences/doublet3d.py` — reversed signs in `compute_quad_doublet_velocity` and `compute_all_doublet_velocities`.
- **Checks run**:
  - `python demos/demo_actuator_disk.py --case cases/cicular_vent --mesh-level -1` — ADM correctly yields positive flow rate and converges successfully in 13 iterations.
  - `pytest src/test -q` — 23 passed.
- **Status**: Complete

### Fixed Thin-Shell Duct Transparency
- **What was done**: The previous open-cylinder geometry was mathematically a "thin shell". In a source panel method, thin source sheets cannot sustain a pressure difference and act transparently to cross-flow. The doublet's dipole field was simply passing through the duct walls, creating a local unconfined recirculation zone (the "magnetic field" effect).
- **Solution**: Upgraded the duct geometry generator to create a **Thick Cylinder** (inner wall, outer wall, and end lips). By giving the duct finite thickness, it forms a completely closed 3D body. The source panel method perfectly enforces the solid boundary, forcing the doublet's momentum entirely through the internal pipe and preventing radial tip-leakage.
- **Files modified**:
  - `src/core/geometry/io/sphere_generator.py` — added `generate_thick_cylinder()`
  - `src/core/geometry/factory.py` — registered `"thick_cylinder"` geometry
  - `cases/cicular_vent/case.yaml` — updated to use `thick_cylinder` with `radius_inner=0.06` and `radius_outer=0.065`.
- **Status**: Complete

### ADM Inlet and Outlet Boundaries
- **What was done**: Re-architected the system to strictly enforce internal pipe flow and completely prevent "magnetic-field" tip leakage through boundary kinematics. Added `inlets` and `outlets` to the configuration schema as independent geometric source/sink disks. The ADM iteration loop was upgraded to iterate the *System Flow Rate* ($Q_{sys}$). At each iteration, $Q_{sys}$ is prescribed to the inlet sources and outlet sinks, guaranteeing mass perfectly enters and exits the duct. The internal doublet continues to add the correct fan $\Delta P$ pressure jump. By letting the sources and sinks define the kinematics, the internal flow is perfectly channeled and cannot loop around the exterior of the pipe.
- **Files modified**:
  - `src/core/config/schemas.py` — added `BoundaryRegionConfig`, appended `inlets` and `outlets` to `SimulationConfig`.
  - `src/solvers/actuator/coupling.py` — introduced `BoundaryRegionRuntime` to parse and build independent source meshes. Refactored the ADM loop to iterate `system_q`, apply $\pm 2Q/A$ source strengths to inlets/outlets, measure resulting internal velocity field, and compute the residual flow error against the system target.
  - `cases/cicular_vent/case.yaml` — Added an inlet disk at $z=-0.5$ and an outlet disk at $z=+0.5$ to perfectly seal the vent flow.
- **Status**: Complete

### Fix 3D Source Panel Sign Convention Bug
- **What was done**: Investigated and fixed a spatial offset and unphysical velocity field bug in ParaView/VTK outputs for 3D cases (like `sphere_flow`). The 3D source influence matrix previously hardcoded `A[i,i] = -0.5`. This enforced the zero-normal-velocity boundary condition on the *interior* side of the panels (making the problem an internal flow rather than external potential flow), yielding correct interior flow (zero) but completely incorrect exterior flow. Changed to `A[i,i] = 0.5` to correctly enforce the external flow boundary condition.
- **Files modified**:
  - `src/solvers/panel3d/influences/source3d.py`
- **Checks run**:
  - Validated upstream velocity axis profile matches theoretical expectations (drops from freestream to 0 at stagnation point).
  - Run `plot_cut_plane.py` and `demo_case_paraview_export.py` on `sphere_flow` with successful outputs.
- **Status**: Complete

## 2026-05-09
### ADM Pressure Reconstruction and Gauge-Pressure Validation
- **What was done**: Added direct pressure reconstruction to the ADM coupled solver for arbitrary field points. The solver now reconstructs static pressure from Bernoulli plus actuator-disk pressure-jump offsets, with half-jump treatment on the disk plane and gauge-pressure access for Fluent comparison. Updated 3D ParaView export and ADM validation helpers to use this path.
- **Files modified**:
  - `src/solvers/actuator/coupling.py`
  - `demos/demo_case_paraview_export.py`
  - `demos/demo_actuator_disk.py`
  - `validation/scripts/3d/adm/common.py`
  - `src/test/test_actuator_disk.py`
- **Checks run**:
  - `/home/nuwa/miniforge3/envs/fyp/bin/python -m py_compile src/solvers/actuator/coupling.py demos/demo_case_paraview_export.py demos/demo_actuator_disk.py validation/scripts/3d/adm/common.py src/test/test_actuator_disk.py`
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python -m pytest src/test/test_actuator_disk.py -q` — 6 passed
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python validation/scripts/3d/adm/compare_axis_line.py cases/cicular_vent --quantities pressure`
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python validation/scripts/3d/adm/compare_cut_plane.py cases/cicular_vent --quantities pressure`
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python demos/demo_case_paraview_export.py cases/cicular_vent --mesh-level 0 --resolution 60 60 60 --output-dir cases/cicular_vent/out/panel_solver_pressure_tmp`
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python validation/scripts/3d/adm/compare_cut_plane.py cases/cicular_vent --quantities pressure --volume-field cases/cicular_vent/out/panel_solver_pressure_tmp/volume_fields.vts --output-dir cases/cicular_vent/out/validation/adm/cut_plane_pressure_tmp`
  - `env MPLCONFIGDIR=/tmp/matplotlib-codex /home/nuwa/miniforge3/envs/fyp/bin/python validation/scripts/3d/adm/compare_axis_line.py cases/cicular_vent --quantities pressure --volume-field cases/cicular_vent/out/panel_solver_pressure_tmp/volume_fields.vts --output-dir cases/cicular_vent/out/validation/adm/axis_line_pressure_tmp`
- **Notes**:
  - Old `out/panel_solver/volume_fields.vts` files generated before this change still carry invalid pressure for zero-freestream ADM cases; rerun the export script once to refresh them.
