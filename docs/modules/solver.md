# Solver Module

The solver module (`solvers`) implements panel method solvers using an abstract base class hierarchy with a factory registry for config-driven creation.

## Architecture

```
Solver (ABC)                     # Interface: solve(), velocity_at(), surface_velocity
└── PanelSolver2D (ABC)          # Template method: solve() orchestrates 4 abstract steps
    └── SourcePanelSolver        # Constant-strength source panels (Katz & Plotkin)
```

### Solver ABC

Defines the minimal interface all solvers must implement:

- `solve()` — Execute the solver
- `surface_velocity` — Property returning `(N, 3)` velocity at panel centers
- `velocity_at(points)` — Compute velocity at arbitrary `(M, 3)` field points
- `is_solved` — Whether `solve()` has been called
- `mesh` — Access to the panel mesh

### PanelSolver2D

Implements the **template method** pattern for 2D panel methods. The `solve()` method executes these steps in order:

1. `_compute_influence_matrices()` → geometric integrals
2. `_solve_linear_system(matrices)` → singularity strengths
3. `_compute_surface_velocity(matrices, strengths)` → surface velocities

Subclasses implement each step for their specific singularity type.

### SourcePanelSolver

The currently implemented solver. Key properties after calling `solve()`:

```python
from core.io import CaseLoader

case = CaseLoader.load_case("cases/cylinder_flow")
solver = case.create_solver()
solver.solve()

solver.sigma              # (N,) source strengths
solver.Vt                 # (N,) tangential velocity at panel centers
solver.Cp                 # (N,) pressure coefficient (Bernoulli)
solver.surface_velocity   # (N, 3) full velocity vector at panel centers
```

## Mathematical Formulation

The solver assembles and solves the linear system:

$$A_{ij}\sigma_j = b_i$$

where $A_{ij} = I_{ij}$ for $i \neq j$ (influence coefficients) and $A_{ii} = \pi$ (self-influence), and:

$$b_i = -V_\infty \cdot 2\pi \cos\beta_i$$

Surface velocity is recovered by differentiating the velocity potential along the surface:

$$V_t = \frac{d\phi}{ds}$$

Pressure coefficient from Bernoulli:

$$C_p = 1 - \left(\frac{V_t}{V_\infty}\right)^2$$

See the [Theory page](../theory/panel_methods.md) for the full derivation.

## SolverFactory

Solvers are created through a registry-based factory:

```python
from solvers import SolverFactory

# List available configurations
print(SolverFactory.available())
# {('source', 'constant', 'flat'): 'SourcePanelSolver'}

# Create from config (used by Case.create_solver())
solver = SolverFactory.create_panel_solver(
    singularity="source",
    mesh=mesh,
    v_inf=1.0,
    aoa=0.0
)
```

## Influence Coefficients

The `influences/source.py` module computes the geometric integrals analytically:

- `compute_source_influence_matrices(mesh)` → `(I, J)` matrices of shape `(N, N)`
- `compute_source_velocity_influence(point, ...)` → `(Mx, My)` velocity coefficients
- `compute_source_potential_influence(point, ...)` → potential coefficient

## Planned Extensions

| Solver | Singularity | Status |
|--------|-------------|--------|
| Constant source | Source, constant strength, flat | ✅ Implemented |
| Constant vortex | Vortex, constant strength, flat | 🔲 Planned |
| Linear source | Source, linear strength, flat | 🔲 Planned |
| Source + vortex | Combined with Kutta condition | 🔲 Planned |

## File Layout

| File | Contents |
|------|----------|
| `base.py` | `Solver` ABC |
| `factory.py` | `SolverFactory` registry |
| `panel2d/base.py` | `PanelMethodConfig`, `PanelSolver2D` ABC |
| `panel2d/spm.py` | `SourcePanelSolver` implementation (424 lines) |
| `panel2d/influences/source.py` | Source influence coefficient functions |
