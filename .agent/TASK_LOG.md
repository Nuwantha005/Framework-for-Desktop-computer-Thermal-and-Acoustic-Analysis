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
