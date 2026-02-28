# Dirichlet Doublet Panels (Morino Formulation)

## 1. Introduction

The Dirichlet doublet method (also known as the **Morino formulation**) is a combined source + doublet panel method that uses an *internal-potential* boundary condition rather than the classical Neumann (no-penetration) condition. Sources are prescribed from the freestream and only doublet strengths are solved as unknowns. This makes it mathematically elegant and the basis for production codes like PANAIR and VSAERO.

For non-lifting bluff bodies (no wake, no Kutta condition), the method requires a gauge fix — pinning one doublet strength to zero — to remove a rank-1 deficiency caused by the absence of wake panels. Surface velocity is recovered from the doublet-strength gradient: $V_t = d\mu/ds + V_{\infty,t}$.

## 2. Theoretical Formulation

### Boundary Condition

The Dirichlet condition sets the **perturbation potential to zero inside the body** (K&P §11.3.1):

$$
\phi_{\text{perturbation}}^{(\text{internal})} = 0
$$

This replaces the Neumann condition $\partial\phi/\partial n = 0$ used by source-only methods.

### Doublet Potential Influence

A constant-strength doublet panel induces a potential at a field point $P$ (K&P Eq. 10.28):

$$
\phi = \frac{-\mu}{2\pi}\,\Delta\theta
$$

where $\Delta\theta = \theta_2 - \theta_1$ is the angle subtended by the panel endpoints at $P$, computed in panel-local coordinates:

$$
\Delta\theta = \arctan\!\left(\frac{z}{x - x_2}\right) - \arctan\!\left(\frac{z}{x - x_1}\right)
$$

The **self-influence** at the panel centre (from the interior side) is $c_{ii} = \tfrac{1}{2}$ (K&P Eq. 11.69).

### Source Potential Influence

The source potential at $P$ from a constant-strength source panel of length $S$ is (K&P Eq. 10.22):

$$
\phi_\sigma = \frac{\sigma}{4\pi}\, B_{ij}
$$

where $B_{ij}$ contains the logarithmic integral terms. Self-influence at the panel midpoint uses the analytical result $B_{ii} = S \ln(S/2)^2$.

### Morino Linear System

Applying the Dirichlet BC at each panel centre collocation point yields the linear system:

$$
C \mathbf{\mu} = -B \mathbf{\sigma}
$$

where:

- $C$ is the $N \times N$ doublet potential influence matrix ($C_{ii} = 1/2$)
- $B$ is the $N \times N$ source potential influence matrix
- Source strengths are **prescribed**: $\sigma_j = \hat{n}_j \cdot \mathbf{V}_\infty$

### Bluff-Body Adaptation

For non-lifting bodies without wake panels ($\mu_W = 0$), the matrix $C$ is rank-deficient by 1 per connected component — a uniform doublet distribution produces zero perturbation potential inside any closed body.

We cure this by **pinning** $\mu_1 = 0$ (replacing the first equation of each component with $\mu_{k} = 0$). This selects the unique solution that satisfies the Dirichlet BC with the smallest doublet variation.

## 3. Surface Velocity

The perturbation potential on the **exterior** surface is (from the doublet jump condition with $\phi_{\text{int}} = 0$):

$$
\phi_{\text{perturbation}}^{(\text{ext})} = \mu
$$

Total surface velocity is obtained by differentiating the total potential along the surface arc length $s$ (K&P Eq. 11.76):

$$
V_t = \frac{d\mu}{ds} + \mathbf{V}_\infty \cdot \hat{t}
$$

The derivative $d\mu/ds$ is computed numerically using central differences with periodic wrap-around for each closed component.

## 4. Field Velocity

At off-body points, the velocity is the sum of freestream, doublet, and source contributions:

$$
\mathbf{V}(P) = \mathbf{V}_\infty + \sum_{j=1}^{N} \mu_j \, \mathbf{v}_{\text{doublet},j}(P) + \sum_{j=1}^{N} \frac{\sigma_j}{2\pi} \, \mathbf{v}_{\text{source},j}(P)
$$

A constant doublet panel is equivalent to two opposite point vortices at the panel endpoints (K&P Ch. 10), giving:

$$
u = \frac{-\mu}{2\pi}\left[\frac{z}{r_1^2} - \frac{z}{r_2^2}\right], \qquad
w = \frac{\mu}{2\pi}\left[\frac{x - x_1}{r_1^2} - \frac{x - x_2}{r_2^2}\right]
$$

## 5. Implementation

**Solver class**: `DirichletDoubletSolver` in `solvers/panel2d/dirichlet_doublet_solver.py`

**Influence functions**: `solvers/panel2d/influences/doublet.py`

- `compute_doublet_potential_influence()` — single panel→point potential coefficient
- `compute_doublet_influence_matrix()` — full $N \times N$ matrix $C$
- `compute_source_potential_matrix()` — full $N \times N$ matrix $B$
- `compute_doublet_velocity_influence()` — off-body velocity coefficients

**Factory key**: `("source_doublet", "constant", "flat")`

**Comparison alias**: `"doublet"` or `"source_doublet"`

## 6. References

- Katz & Plotkin, *Low-Speed Aerodynamics*, 2nd ed., §10.2.2, §11.3.1, §11.5.1
- Morino, L. (1974). A general formulation for potential aerodynamics with applications. *AIAA Journal*, 12(2), 180–186.
