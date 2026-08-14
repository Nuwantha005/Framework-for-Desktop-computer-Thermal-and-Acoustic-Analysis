# Solver Module

## Actuator Disk Coupling

The 3D solver path supports optional actuator disks through
`ActuatorDiskCoupledSolver3D`. Cases without `actuator_disks` continue to create
the configured body solver directly via the factory. The ADM implementation
adds actuator disk constant-strength doublet influence (vortex rings) as a known
normal-velocity disturbance on the body-panel RHS. It also supports `inlets` and `outlets`
which act as independent source and sink boundary meshes whose strengths are automatically
matched to the system flow rate ($Q$) by the ADM loop, guaranteeing perfectly sealed
internal flow kinematics without tip-leakage.

The first supported body solver is the registered constant-source 3D solver.
Future 3D singularity solvers should honor the same
`normal_velocity_disturbance` solve hook to participate in ADM coupling.

Actuator disk outputs are written to `case/out/adm/`; reusable solve bundles are
written to `case/out/solverRuns/`.

ADM plotting currently writes both `adm_convergence.png` and
`adm_fan_curve_progression.png`. The coupled iteration stops early with a
warning if an evaluated fan flow rate leaves the tabulated P-Q curve bounds.
For fan-driven quiescent cases, set the case freestream to zero; ADM initializes
its velocity scale from the fan curve rather than imposing an inlet velocity.
Generic 3D export/visualization scripts should create solvers through
`Case.create_solver()` so actuator disks in the case config are automatically
coupled.

The solver module (`solvers`) implements panel method solvers using an abstract base class hierarchy with a factory registry for config-driven creation.

## Architecture

```
Solver (ABC)                     # Interface: solve(), velocity_at(), surface_velocity
└── PanelSolver2D (ABC)          # Template method: solve() orchestrates 4 abstract steps
    ├── SourcePanelSolver        # Constant-strength source panels (Katz & Plotkin)
    ├── LinearSourcePanelSolver  # Linear-strength source panels (continuous nodes)
    ├── LinearVortexPanelSolver  # Linear-strength vortex panels (zero-circulation closure)
    ├── DirichletDoubletSolver   # Morino source+doublet, Dirichlet internal-potential BC
    └── LinearSourceDoubletSolver # Linear source+doublet, Dirichlet BC (K&P §11.5.1)
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

See the [Theory page](../theory/panel_methods_overview.md) for the full derivation.

## SolverFactory

Solvers are created through a registry-based factory:

```python
from solvers import SolverFactory

# List available configurations
print(SolverFactory.available())
# {('source', 'constant', 'flat'): 'SourcePanelSolver',
#  ('source', 'linear', 'flat'): 'LinearSourcePanelSolver',
#  ('vortex', 'linear', 'flat'): 'LinearVortexPanelSolver',
#  ('source_doublet', 'constant', 'flat'): 'DirichletDoubletSolver',
#  ('source_doublet', 'linear', 'flat'): 'LinearSourceDoubletSolver'}

# Create from config (used by Case.create_solver())
solver = SolverFactory.create_panel_solver(
    singularity="vortex",
    mesh=mesh,
    v_inf=1.0,
    aoa=0.0
)
```

## Influence Coefficients

The `influences/` subpackage computes geometric integrals analytically for each singularity type:

### Source influences (`influences/source.py`)
- `compute_source_influence_matrices(mesh)` → `(I, J)` matrices of shape `(N, N)`
- `compute_source_velocity_influence(point, ...)` → `(Mx, My)` velocity coefficients
- `compute_source_potential_influence(point, ...)` → potential coefficient

### Linear source influences (`influences/linear_source.py`)
- `compute_linear_source_influence_matrices(mesh)` → `(I, J)` of shape `(N, N+1)`
- `compute_linear_source_velocity_field(points, mesh, strengths)` → `(M, 2)` velocity vectors

### Linear vortex influences (`influences/linear_vortex.py`)
- `compute_linear_vortex_influence_matrices(mesh)` → `(I, J)` of shape `(N, N+1)`
- `compute_linear_vortex_velocity_influence(point, ...)` → `((Mx_a, My_a), (Mx_b, My_b))`
- `compute_linear_vortex_velocity_field(points, mesh, gamma)` → `(M, 2)` velocity vectors

### Doublet influences (`influences/doublet.py`)
- `compute_doublet_potential_influence(point, panel_start, panel_end)` → potential coefficient
- `compute_doublet_influence_matrix(mesh)` → `(N, N)` doublet potential matrix $C$
- `compute_source_potential_matrix(mesh)` → `(N, N)` source potential matrix $B$
- `compute_doublet_velocity_influence(point, panel_start, panel_end)` → `(u, w)` velocity coefficients

### Linear doublet influences (`influences/linear_doublet.py`)
- `compute_linear_doublet_potential_influence(point, start, end)` → `(Φ_a, Φ_b)` — K&P Eqs. 11.114/11.115
- `compute_linear_source_potential_influence(point, start, end)` → `(B_a, B_b)` — full integration with $-2S$ constant
- `compute_linear_doublet_influence_matrix(mesh)` → `(N, N)` node-accumulation doublet potential matrix $C$
- `compute_linear_source_potential_matrix(mesh)` → `(N, N)` node-accumulation source potential matrix $B$
- `compute_linear_doublet_velocity_influence(point, start, end)` → `((u_a, w_a), (u_b, w_b))` off-body velocity
- `compute_linear_doublet_velocity_field(points, mesh, mu)` → `(M, 2)` batch off-body velocity field

The vortex influences are derived from the source influences via the rotation identity $u_{\text{vortex}} = w_{\text{source}}$, $w_{\text{vortex}} = -u_{\text{source}}$.

## Planned Extensions

| Solver | Singularity | Status |
|--------|-------------|--------|
| Constant source | Source, constant strength, flat | ✅ Implemented |
| Linear source | Source, linear strength, flat | ✅ Implemented |
| Linear vortex | Vortex, linear strength, flat | ✅ Implemented |
| Dirichlet doublet | Source+doublet, constant strength, flat | ✅ Implemented |
| Linear source/doublet | Source+doublet, linear strength, flat (K&P §11.5.1) | ✅ Implemented |
| Quadratic source | Source, quadratic strength, flat | 🔲 Planned |
| Source + vortex | Combined with Kutta condition | 🔲 Planned |

## File Layout

| File | Contents |
|------|----------|
| `base.py` | `Solver` ABC |
| `factory.py` | `SolverFactory` registry |
| `comparison.py` | `SolverComparisonRunner`, `ComparisonResult`, metrics, ranking |
| `panel2d/base.py` | `PanelMethodConfig`, `PanelSolver2D` ABC |
| `panel2d/spm.py` | `SourcePanelSolver` implementation |
| `panel2d/linear_source_solver.py` | `LinearSourcePanelSolver` implementation |
| `panel2d/linear_vortex_solver.py` | `LinearVortexPanelSolver` implementation |
| `panel2d/dirichlet_doublet_solver.py` | `DirichletDoubletSolver` implementation (Morino) |
| `panel2d/linear_source_doublet_solver.py` | `LinearSourceDoubletSolver` implementation (linear Morino, K&P §11.5.1) |
| `panel2d/influences/source.py` | Constant source influence coefficient functions |
| `panel2d/influences/linear_source.py` | Linear source influence coefficient functions |
| `panel2d/influences/linear_vortex.py` | Linear vortex influence coefficient functions |
| `panel2d/influences/doublet.py` | Constant doublet potential & velocity influence functions |
| `panel2d/influences/linear_doublet.py` | Linear doublet + source potential & velocity influence functions |
