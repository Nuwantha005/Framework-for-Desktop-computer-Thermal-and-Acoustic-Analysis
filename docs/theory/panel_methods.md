# Panel Method Theory

This page describes the mathematical formulation implemented in the solver.

## Potential Flow

The solver assumes incompressible, irrotational, inviscid flow. Under these assumptions, the velocity field can be expressed as the gradient of a scalar potential:

$$\mathbf{V} = \nabla\phi$$

where the velocity potential $\phi$ satisfies Laplace's equation everywhere outside the body:

$$\nabla^2 \phi = 0$$

## Panel Method Concept

Instead of solving Laplace's equation on a volume mesh, panel methods reformulate the problem as a boundary integral equation. The body surface is discretized into $N$ flat **panels**, and a singularity distribution (source, vortex, or doublet) is placed on each panel. The strengths of these singularities are determined by enforcing boundary conditions at **control points** (panel midpoints).

This reduces the problem from a 2D/3D field solve to a system of $N$ linear equations — a significant computational advantage.

## Constant-Strength Source Panel Method

The current implementation uses **constant-strength source panels** following the formulation in Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), Chapter 10.

### Singularity Distribution

Each panel $j$ carries a uniform source distribution of strength $\sigma_j$ (units: $\mathrm{m/s}$). The velocity potential induced by panel $j$ at a field point $P$ is:

$$\phi_j(P) = \frac{\sigma_j}{4\pi} \int_{\text{panel}_j} \ln r \, dl$$

where $r$ is the distance from the integration point on the panel to $P$.

### Influence Coefficients

The integrals are evaluated analytically for flat panels. For the normal velocity at control point $i$ due to panel $j$:

$$V_{n_{ij}} = \frac{\sigma_j}{2\pi} I_{ij}$$

where $I_{ij}$ is the **normal influence coefficient** computed from the panel geometry:

$$I_{ij} = \frac{1}{2} C_n \ln\frac{S_j^2 + 2A_{ij}S_j + B_{ij}}{B_{ij}} + \frac{D_n - A_{ij}C_n}{E_{ij}} \left[\arctan\frac{S_j + A_{ij}}{E_{ij}} - \arctan\frac{A_{ij}}{E_{ij}}\right]$$

The geometric quantities $A_{ij}$, $B_{ij}$, $C_n$, $D_n$, $E_{ij}$ depend on the relative position and orientation of panel $i$'s control point with respect to panel $j$. Similarly, the **tangential influence coefficient** $J_{ij}$ is computed with the tangential direction substituted.

### Boundary Condition

The **Neumann boundary condition** (no-penetration) requires that the total normal velocity vanishes at each control point:

$$V_{n_i}^{\text{total}} = 0 \quad \text{for } i = 1, \ldots, N$$

The total normal velocity at control point $i$ consists of the freestream contribution and the induced velocity from all panels:

$$\sum_{j=1}^{N} \frac{\sigma_j}{2\pi} I_{ij} + \frac{\sigma_i}{2} + \mathbf{V}_\infty \cdot \hat{n}_i = 0$$

The $\sigma_i / 2$ term is the self-influence of panel $i$ (a source sheet induces a velocity jump of $\sigma/2$ across itself).

### Linear System Assembly

Rearranging the boundary condition gives the linear system:

$$A\sigma = b$$

where:

$$A_{ij} = \begin{cases} \pi & \text{if } i = j \\ I_{ij} & \text{if } i \neq j \end{cases}$$

$$b_i = -V_\infty \cdot 2\pi \cos\beta_i$$

and $\beta_i$ is the angle between the freestream direction and panel $i$'s outward normal.

### Surface Velocity Recovery

After solving for $\sigma$, the tangential velocity at each panel center is recovered. The implementation uses a **potential-based approach**: the total velocity potential $\phi$ is computed at each panel center, then differentiated numerically along the surface arc length:

$$V_{t_i} = \frac{d\phi}{ds}\bigg|_i \approx \frac{\phi_{i+1} - \phi_{i-1}}{s_{i+1} - s_{i-1}}$$

This is more robust than direct summation of tangential influences, particularly at corners where curvature changes rapidly.

### Pressure Coefficient

The pressure coefficient follows directly from Bernoulli's equation:

$$C_p = 1 - \left(\frac{V_t}{V_\infty}\right)^2$$

### Mass Conservation Check

For a closed body, mass conservation requires:

$$\sum_{j=1}^{N} \sigma_j S_j = 0$$

where $S_j$ is the length of panel $j$. This serves as a useful validation check.

## Current Limitations

- **Source panels only**: No lift generation (no circulation). Suitable for non-lifting bodies.
- **Constant strength**: Accuracy limited by panel density, especially at high curvature regions.
- **Inviscid**: No boundary layer, no separation, no wake.
- **2D only**: 3D panel methods planned for future phases.

## References

1. Katz, J. and Plotkin, A., *Low-Speed Aerodynamics* (2nd ed.), Cambridge University Press, 2001. Chapters 9–11.
