# BDIM Thermal Solver: Stabilization and Implementation Details

This document covers the specific numerical stabilization techniques and implementation details required to make the Boundary-Domain Integral Method (BDIM) robust for realistic thermal boundary layer flows.

## 1. Near-Wall Singularities: Exact Analytical Integration

### The Problem
The standard BDIM formulation relies on the fundamental solutions for 2D diffusion:
$$ T^*(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln \frac{1}{r} $$
$$ T_{,n}^*(\mathbf{x}, \mathbf{y}) = \frac{-1}{2\pi r^2}(y_i - x_i) n_i $$

When evaluated numerically using a midpoint approximation, the gradients diverge as $r \to 0$. For domain points located inside the boundary layer immediately adjacent to the wall (e.g., $y \approx 10^{-5}$ m), the geometric distance to the boundary panel midpoint is extremely small. This causes the $[H_I]$ matrix elements to blow up, leading to massive localized negative temperature artifacts in the domain field.

### The Solution
To resolve this, the numerical midpoint approximation for points near the boundary must be replaced with exact analytical integration over constant-strength panels. For a panel of length $L$ with local tangent $t$ and normal $n$, the local coordinates of an evaluation point are $(x_{loc}, y_{loc})$. 

The exact integrals over the panel yield:
$$ H = -\frac{\Delta\theta}{2\pi} $$
$$ G = -\frac{1}{2\pi} \left( x_{loc} \ln(r_1) - (x_{loc} - L) \ln(r_2) + y_{loc} \Delta\theta - L \right) $$
Where $\Delta\theta = \theta_2 - \theta_1$ is the angle subtended by the panel at the evaluation point, and $r_1, r_2$ are the distances to the panel endpoints. This analytical treatment completely avoids the $1/r$ singularity and provides smooth, bounded evaluations up to the boundary surface ($H \to 0.5$ as $y_{loc} \to 0$).

## 2. Convective Instability: Local Artificial Diffusion

### The Problem
In strongly convective flows (high Péclet numbers), standard central-difference or standard BEM formulations suffer from numerical oscillations. The BDIM solver would globally destabilize, yielding non-physical wall temperatures on the order of $\pm 10^5$ K.

### The Solution
An upwinding-equivalent stabilization is introduced via **Artificial Diffusion**. We define a local cell Péclet number:
$$ Pe_{cell} = \frac{\rho c_p |\mathbf{u}| \Delta x}{k} $$

To maintain numerical stability, we constrain the effective cell Péclet number to be $\le 0.5$. This is achieved by dynamically scaling the local thermal conductivity $k_{eff}$ on a per-cell and per-boundary-panel basis:
$$ k_{eff} = \max\left(k, \frac{\rho c_p |\mathbf{u}| \Delta x}{0.5}\right) $$
By substituting $k_{eff}$ into the specific heat capacity mapping term ($\frac{\rho c_p}{k_{eff}} \mathbf{u}$), we naturally introduce sufficient artificial diffusion to suppress convective oscillations without altering the fundamental boundary integral structure.

## 3. Truncated Domain Boundary Conditions

### The Problem
The theoretical BDIM formulation assumes a closed boundary $\Gamma$ encompassing the domain $\Omega$. However, for boundary layer calculations, we are solving on a truncated open domain (the fluid edge $y \to \infty$ and the trailing edge wake).

### The Solution
1. **Mechanical Dissipation:** Because the domain is not closed, the divergence trick used to map constant mechanical dissipation $\{c_w\}$ into a boundary integral creates massive artificial fluxes. We set $\{c_w\} = 0$, which is physically justifiable as viscous dissipation is negligible for low-speed incompressible flows.
2. **Matrix Forcing:** We explicitly enforce the freestream temperature (implicitly $\theta = 0$) at the inflow and the outer edge of the boundary layer domain grid by zeroing out the respective rows in the global matrix and setting the diagonal to 1.0.

## 4. Geometric Mapping and Visualization Fixes

Two critical visual and physical mapping artifacts were resolved during stabilization:
1. **Tangent Vector Extraction:** The lower boundary layer field was originally plotted backward (from trailing edge to stagnation point). This occurred because tangent vectors were derived by rotating outward-flipped normals. The fix strictly extracts the tangent based on the physical flow direction *before* any normal-flipping logic is applied.
2. **Wake Bridging:** When mapping separated wake regions (which contain `NaN` or invalid data), Matplotlib's `plot_surface` equivalent would bridge polygons across the void. This is mitigated using masked arrays and `np.ma.clump_unmasked` to plot strictly contiguous, valid envelope segments.
