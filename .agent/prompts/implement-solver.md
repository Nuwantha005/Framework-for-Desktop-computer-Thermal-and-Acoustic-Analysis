# Prompt: Implement a New Panel Solver

## Context
This prompt guides implementation of a new 2D panel solver variant following the established
6-step pattern. The solver architecture uses a template method pattern:
`PanelSolver2D` ABC → concrete solver (e.g., `SourcePanelSolver`).

## Reference Materials
- **6-step checklist**: `notes_archived/solver_implementation/00_overview.md`
- **Influence coefficient theory**: `notes_archived/solver_implementation/01_influence_coefficients.md`
- **Singularity types (source/doublet/vortex)**: `notes_archived/solver_implementation/02_singularity_types.md`
- **Panel orders (constant/linear/quadratic)**: `notes_archived/solver_implementation/03_panel_orders.md`
- **Panel geometry (flat/curved)**: `notes_archived/solver_implementation/04_panel_geometry.md`
- **K&P Chapter 10 (influence coefficients)**: `notes_archived/Low-Speed Aerodynamics 2nd edition*/chapter 10*.md`
- **K&P Chapter 11 (2D solutions)**: `notes_archived/Low-Speed Aerodynamics 2nd edition*/chapter 11*.md`

## Existing Implementation to Follow
- **Influence functions**: `src/solvers/panel2d/influences/source.py` (271 lines)
- **Solver class**: `src/solvers/panel2d/spm.py` (430 lines, `SourcePanelSolver`)
- **Solver ABC**: `src/solvers/panel2d/base.py` (417 lines, `PanelSolver2D`)
- **Factory**: `src/solvers/factory.py` — register via `SolverFactory.register(singularity, order, geometry, cls)`
- **Config schema**: `src/core/config/schemas.py` — `SolverConfig` Literal types

## Implementation Steps

### Step 1: Influence Coefficients
Create `src/solvers/panel2d/influences/<singularity>_<order>.py`:
```python
def compute_<name>_influence_matrices(mesh: Mesh) -> tuple[NDArray, NDArray]:
    """Compute NxN influence coefficient matrices.
    
    Follow K&P notation. Verify formulas against:
    - K&P equations (cite specific equation numbers)
    - Reference code in notes_archived/code-snippets-2d/
    
    Returns: Tuple of influence matrices (normal, tangential or I, J).
    """
```
**Key rules:**
- Vectorize with NumPy — avoid Python-level double loops (current spm.py has this as tech debt)
- Handle self-influence terms analytically (e.g., π for constant source)
- Use `np.float64` throughout; handle numerical singularities (r→0) with epsilon guard
- For higher-order panels: split into shape-function-weighted sub-matrices (see 03_panel_orders.md)

### Step 2: Export Influences
Add to `src/solvers/panel2d/influences/__init__.py`.

### Step 3: Solver Class
Create `src/solvers/panel2d/<name>_solver.py`:
- Extend `PanelSolver2D`
- Implement 4 abstract methods: `_compute_influence_matrices`, `_solve_linear_system`,
  `_compute_surface_velocity`, `_velocity_at_points`
- Define `config` property returning `PanelMethodConfig(singularity, order, geometry)`
- Compute surface velocity via potential differentiation (preferred for accuracy) or
  direct influence summation

### Step 4: Export Solver
Add to `src/solvers/panel2d/__init__.py`.

### Step 5: Register with Factory
In `src/solvers/__init__.py`: `SolverFactory.register(singularity, order, geometry, NewSolver)`

### Step 6: Update Config Schema
If adding new singularity type or order, update Literals in `src/core/config/schemas.py`.

## Validation Gate (must pass before marking complete)
1. **BC check**: `solver.validate()` — Vn RMS < 1e-10
2. **Mass conservation**: Σσ ≈ 0 (or equivalent for new singularity)
3. **Baseline comparison**: Run on cylinder case, compare Vt and Cp against constant-source baseline
4. **OF comparison**: Run `demos/demo_surface_comparison.py` against OpenFOAM — report L2/L∞/RMS metrics
5. **Convergence**: Verify error decreases with mesh refinement (run at N=32, 64, 128)

## Specific Variants

### Linear-Strength Source Panels (priority)
- Registry key: `("source", "linear", "flat")`
- Design: `notes_archived/solver_implementation/03_panel_orders.md` → "Linear Strength" section
- Continuous formulation preferred (N+1 unknowns for N panels, shared node strengths)
- Two sub-matrices per influence: weight by ξ position along panel
- 2-point Gaussian quadrature for integration
- Expected accuracy: ~10x improvement over constant for same N

### Quadratic-Strength Source Panels (later)
- Registry key: `("source", "quadratic", "flat")`
- Three shape functions per panel, 3-point Gauss quadrature
- Best paired with curved panel geometry

## Post-Implementation
- Update `.agent/modules/solver.md` with new solver entry
- Record design decisions in `.agent/decisions/`
- Update `.agent/PROJECT_CONTEXT.md` status checkboxes
- Update the docs with complete description of formulas
