# Linear Source/Doublet Panels (Higher-Order Morino)

## 1. Introduction

The linear source/doublet method extends the constant Dirichlet doublet (Morino) formulation to **linearly varying** singularity distributions. Both source and doublet strengths are defined at **panel nodes** and vary linearly across each panel, giving continuous distributions over the body surface. The method follows K&P §10.3.2 and §11.5.1.

As with the constant Morino method, the Dirichlet internal-potential boundary condition is used: the perturbation potential is set to zero inside the body, and only the doublet strengths are solved as unknowns. Source strengths are prescribed from the freestream and surface normals.

For non-lifting bluff bodies (no wake, no Kutta condition), the same gauge-fix strategy applies — pinning one doublet node to zero to remove the rank-1 deficiency.

## 2. Theoretical Formulation

### Singularity Distributions

On a panel of length $S$ with nodes A (at $\xi = 0$) and B (at $\xi = S$), the doublet and source strengths vary linearly:

$$
\mu(\xi) = \mu_A \left(1 - \frac{\xi}{S}\right) + \mu_B \frac{\xi}{S}
$$

$$
\sigma(\xi) = \sigma_A \left(1 - \frac{\xi}{S}\right) + \sigma_B \frac{\xi}{S}
$$

where $\xi$ is the panel-local coordinate measured along the panel tangent from node A.

### Decomposition into Constant and Linear Parts

Following K&P Eqs. 11.108–11.115, the linear distribution is decomposed as:

$$
\mu(\xi) = \mu_0 + \mu_1 \xi, \qquad \mu_0 = \mu_A, \quad \mu_1 = \frac{\mu_B - \mu_A}{S}
$$

The potential at a field point is the integral of the singularity kernel weighted by the linear distribution. This integral is evaluated analytically and split into per-node contributions:

$$
\phi = \Phi^a \cdot \mu_A \;+\; \Phi^b \cdot \mu_B
$$

### Doublet Potential Influence

The per-node doublet potential influences at a point $(x, z)$ in panel-local coordinates are (K&P Eqs. 11.114–11.115):

$$
\Phi^a = -\frac{1}{2\pi}\left[\Delta\theta - \frac{x \,\Delta\theta + \frac{z}{2}\ln\!\left(\frac{r_2^2}{r_1^2}\right)}{S}\right]
$$

$$
\Phi^b = -\frac{1}{2\pi S}\left[x \,\Delta\theta + \frac{z}{2}\ln\!\left(\frac{r_2^2}{r_1^2}\right)\right]
$$

where:

- $r_1^2 = x^2 + z^2$ — squared distance from node A
- $r_2^2 = (x - S)^2 + z^2$ — squared distance from node B
- $\Delta\theta = \theta_2 - \theta_1 = \arctan\!\left(\frac{z}{x - S}\right) - \arctan\!\left(\frac{z}{x}\right)$

**Self-influence** (field point on the panel interior, $z \to 0^-$):

$$
\Phi^a_{\text{self}} = \frac{1}{2}\left(1 - \frac{x}{S}\right), \qquad \Phi^b_{\text{self}} = \frac{1}{2}\cdot\frac{x}{S}
$$

At the panel midpoint ($x = S/2$): $\Phi^a_{\text{self}} = \Phi^b_{\text{self}} = \frac{1}{4}$.

### Source Potential Influence

The source potential is built from the constant-strength result (K&P Eq. 10.22) plus a linear correction (K&P Eq. 10.47). The **constant source** potential at $(x, z)$ is:

$$
\phi_0 = \frac{1}{4\pi}\left[x \ln r_1^2 - (x - S)\ln r_2^2 + 2z\,\Delta\theta - 2S\right]
$$

The **linear source** addition involves integrals of $\xi \cdot \text{kernel}$:

$$
\phi_1 = \frac{1}{4\pi}\left[\frac{1}{2}(x^2 - z^2)(\ln r_1^2 - \ln r_2^2) + 2xz\,\Delta\theta - xS - \frac{S^2}{2}\right]
$$

Combined and split into per-node contributions:

$$
B_a = \phi_0 - \frac{\phi_1}{S}, \qquad B_b = \frac{\phi_1}{S}
$$

so that $\phi_\sigma = \sigma_A \cdot B_a + \sigma_B \cdot B_b$.

### Boundary Condition and Matrix Assembly

The Dirichlet condition (zero perturbation potential inside the body) at the $N$ panel midpoints yields the linear system:

$$
C \,\mathbf{\mu} = -B \,\mathbf{\sigma}
$$

where:

- $C$ is the $N \times N$ doublet potential influence matrix, assembled by **node accumulation**: panel $j$ connects nodes $n_1$ and $n_2$, so the panel's $\Phi^a$ contributes to column $n_1$ and $\Phi^b$ to column $n_2$. For closed bodies, $N_{\text{nodes}} = N_{\text{panels}}$.
- $B$ is the $N \times N$ source potential matrix, assembled via the same node-accumulation scheme.
- $\mathbf{\sigma}$ is the vector of prescribed source strengths at nodes.
- $\mathbf{\mu}$ is the vector of unknown doublet strengths at nodes.

### Source Strength Prescription

Source strengths at nodes are prescribed from the freestream and body normals. Since normals are defined per-panel but unknowns live at nodes, we average adjacent panel normals:

$$
\sigma_k = \frac{1}{|\mathcal{P}_k|}\sum_{j \in \mathcal{P}_k} \hat{n}_j \cdot \mathbf{V}_\infty
$$

where $\mathcal{P}_k$ is the set of panels adjacent to node $k$.

### Bluff-Body Adaptation

For non-lifting bodies without wake panels ($\mu_W = 0$), the doublet matrix $C$ is rank-deficient by at least 1 per connected component. As with the constant formulation, the fix is to **pin** $\mu_1 = 0$ per component (replacing the first equation). The system is solved using `lstsq` for robustness against additional null modes that arise on highly symmetric meshes (e.g., a circular cylinder has rank-2 deficiency from the alternating mode).

## 3. Surface Velocity

The perturbation potential on the exterior surface equals $\mu$ (from the doublet jump condition with $\phi_{\text{int}} = 0$):

$$
\phi_{\text{ext}} = \mu
$$

Total surface tangential velocity is obtained by differentiating along the arc length $s$:

$$
V_t = \frac{d\mu}{ds} + \mathbf{V}_\infty \cdot \hat{t}
$$

The derivative $d\mu/ds$ is computed using **central differences** with periodic wrap-around for each closed component. Doublet strengths (defined at nodes) are first averaged to panel centres:

$$
\mu_j^{\text{panel}} = \frac{\mu_{n_1} + \mu_{n_2}}{2}
$$

then the central-difference stencil is applied per component.

## 4. Field Velocity

At off-body points, the velocity from a single linear doublet panel is decomposed into per-node contributions (K&P §10.3.2):

$$
(u, w)_{\text{doublet}} = \mu_A \,(u_a, w_a) + \mu_B \,(u_b, w_b)
$$

where:

$$
u_a = \frac{-1}{2\pi}\left[\frac{z}{r_1^2} - \frac{1}{S}\left(\frac{z\,x}{r_1^2} - \frac{z(x-S)}{r_2^2}\right)\right]
$$

$$
u_b = \frac{-1}{2\pi S}\left[\frac{z\,x}{r_1^2} - \frac{z(x-S)}{r_2^2}\right]
$$

$$
w_a = \frac{1}{2\pi}\left[\frac{x}{r_1^2} - \frac{1}{S}\left(\frac{x^2}{r_1^2} - \frac{(x-S)^2}{r_2^2}\right)\right]
$$

$$
w_b = \frac{1}{2\pi S}\left[\frac{x^2}{r_1^2} - \frac{(x-S)^2}{r_2^2}\right]
$$

The total field velocity sums freestream, linear doublet, and linear source contributions from all panels:

$$
\mathbf{V}(P) = \mathbf{V}_\infty + \sum_{j=1}^{N}\left[\mu_{A_j}\,\mathbf{v}^a_j + \mu_{B_j}\,\mathbf{v}^b_j\right]_{\text{doublet}} + \sum_{j=1}^{N}\left[\sigma_{A_j}\,\mathbf{v}^a_j + \sigma_{B_j}\,\mathbf{v}^b_j\right]_{\text{source}}
$$

## 5. Comparison with Constant Dirichlet

| Aspect | Constant Dirichlet | Linear Source/Doublet |
|--------|-------------------|-----------------------|
| **Singularity order** | Constant per panel | Linear (node-based) |
| **Unknowns** | $N$ panel centres | $N$ nodes |
| **Self-influence** | $C_{ii} = 1/2$ | $\Phi^a = \Phi^b = 1/4$ at midpoint |
| **Source prescription** | Panel normals | Averaged adjacent panel normals |
| **Matrix assembly** | Direct panel-to-panel | Node accumulation |
| **Surface velocity** | Central differences on $\mu$ | Central differences on node-averaged $\mu$ |
| **Continuity** | Discontinuous $\sigma$, $\mu$ | Continuous $\sigma$, $\mu$ across panels |

Both methods share the same theoretical limitation for bluff-body surface velocity extraction: the $d\mu/ds$ derivative is sensitive to mesh quality and cannot capture the stagnation-region physics that viscous CFD resolves. Validation shows comparable accuracy (~50% Vt RMS error vs OpenFOAM for the rounded square case).

## 6. Implementation

**Solver class**: `LinearSourceDoubletSolver` in `solvers/panel2d/linear_source_doublet_solver.py`

**Influence functions**: `solvers/panel2d/influences/linear_doublet.py`

- `compute_linear_doublet_potential_influence()` — single panel→point $(\Phi^a, \Phi^b)$
- `compute_linear_source_potential_influence()` — single panel→point $(B_a, B_b)$
- `compute_linear_doublet_influence_matrix()` — full $N \times N$ matrix $C$ (node accumulation)
- `compute_linear_source_potential_matrix()` — full $N \times N$ matrix $B$ (node accumulation)
- `compute_linear_doublet_velocity_influence()` — off-body per-node velocity coefficients
- `compute_linear_doublet_velocity_field()` — batch off-body velocity field

**Factory key**: `("source_doublet", "linear", "flat")`

**Comparison alias**: `"linear_doublet"` or `"linear_source_doublet"`

## 7. References

- Katz & Plotkin, *Low-Speed Aerodynamics*, 2nd ed., §10.3.2, §10.4, §11.5.1
- Morino, L. (1974). A general formulation for potential aerodynamics with applications. *AIAA Journal*, 12(2), 180–186.
