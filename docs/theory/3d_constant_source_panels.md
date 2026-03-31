# 3D Constant-Strength Source Panels

This page describes the mathematical formulation implemented in the solver for the 3D constant-strength source panel method.

## Formulation Concept

The implementation for 3D flows uses **constant-strength source quadrilateral panels** following the formulation in Katz & Plotkin, *Low-Speed Aerodynamics* (2nd ed.), Chapter 10. This method is suitable for simulating potential flow over non-lifting, fully three-dimensional bluff bodies (e.g. spheres, fuselages).

### Singularity Distribution

Each quadrilateral panel $j$ carries a uniform surface source distribution of strength $\sigma_j$ (units: $\mathrm{m/s}$). The velocity potential induced by panel $j$ at an arbitrary field point $P(x, y, z)$ is:

$$\phi(x,y,z) = -\frac{\sigma}{4\pi} \int \int_{\text{panel}} \frac{1}{r} \, dS$$

where $r = \sqrt{(x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2}$ is the distance from the integration point $(x_0, y_0, 0)$ on the panel to the field point $P$. This integral is evaluated in a local panel coordinate system where the panel lies flat on the $z=0$ plane.

### Influence Coefficients

The surface integral can be evaluated analytically by transforming to a local coordinate system $(x^*, y^*, z^*)$ aligned with the panel. For a flat quadrilateral panel with four vertices (1, 2, 3, 4), the potential and velocity influences are assembled by superimposing the effects of the four edges.

The induced velocity vector $(u, v, w)$ in local panel coordinates is given by:

$$ u = \frac{\sigma}{4\pi} \sum_{k=1}^4 \frac{\Delta y_k}{d_k} \ln \left( \frac{r_k + r_{k+1} - d_k}{r_k + r_{k+1} + d_k} \right) $$

$$ v = \frac{\sigma}{4\pi} \sum_{k=1}^4 \frac{-\Delta x_k}{d_k} \ln \left( \frac{r_k + r_{k+1} - d_k}{r_k + r_{k+1} + d_k} \right) $$

$$ w = \frac{\sigma}{4\pi} \sum_{k=1}^4 \left( \arctan \frac{m_k e_k - h_k}{z r_k} - \arctan \frac{m_k e_{k+1} - h_{k+1}}{z r_{k+1}} \right) $$

where $d_k$ is the length of edge $k$, $m_k$ is the slope of the edge, $r_k$ is the distance from vertex $k$ to the field point, and $e_k, h_k$ are auxiliary geometric quantities (Katz & Plotkin Eq. 10.95-10.97).

### Boundary Condition

The **Neumann boundary condition** (no-penetration) requires that the total normal velocity vanishes at each panel center:

$$V_{n_i}^{\text{total}} = 0 \quad \text{for } i = 1, \ldots, N$$

The total normal velocity at control point $i$ consists of the freestream normal velocity and the induced velocity from all panels. In the global coordinate frame:

$$\sum_{j=1}^{N} A_{ij} \sigma_j = -\mathbf{V}_\infty \cdot \hat{n}_i$$

where $A_{ij} = (\mathbf{V}_{ij} \cdot \hat{n}_i)$ is the **normal influence coefficient**: the normal velocity induced at the center of panel $i$ by a unit-strength source on panel $j$. 

For self-influence ($i = j$), the normal velocity jump across a source sheet yields exactly:

$$A_{ii} = -\frac{1}{2}$$

### Velocity and Pressure Recovery

Once the linear system $A\sigma = b$ is solved for the source strengths $\sigma$, the velocity at any point in the flow can be computed by summing the freestream velocity and the induced velocities from all panels:

$$\mathbf{V}_i = \mathbf{V}_\infty + \sum_{j=1}^{N} \mathbf{V}_{ij}(\sigma_j)$$

For points located on the surface (at the panel centers), this total velocity represents the tangential slip velocity $\mathbf{V}_t$, since the normal component is zero by definition of the boundary condition.

The pressure coefficient is then computed directly using Bernoulli's equation:

$$C_p = 1 - \frac{|\mathbf{V}_t|^2}{|\mathbf{V}_\infty|^2}$$

### Performance Optimization

Computing the dense $N \times N$ influence matrix requires $O(N^2)$ complex logarithmic and arctangent operations. To make this computationally feasible for high-resolution 3D meshes (e.g. 5,000+ panels), the 3D solver utilizes **JIT compilation (Numba)** and multi-core parallelization.

## References

1. Katz, J. and Plotkin, A., *Low-Speed Aerodynamics* (2nd ed.), Cambridge University Press, 2001. Section 10.4: Three-Dimensional Source Panel Method.
2. Hess, J. L., and Smith, A. M. O., "Calculation of Potential Flow About Arbitrary Bodies," *Progress in Aeronautical Sciences*, Vol. 8, Pergamon Press, New York, 1967.