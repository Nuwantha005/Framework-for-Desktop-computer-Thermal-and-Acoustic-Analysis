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
