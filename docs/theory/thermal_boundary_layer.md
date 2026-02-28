# Thermal Boundary Layer: Boundary-Domain Integral Method (BDIM)

This document details the formulation and algorithmic implementation for solving the thermal boundary layer and calculating the surface heat transfer distribution. The methodology strictly follows the Boundary-Domain Integral Method (BDIM) proposed by Gao et al. (2013) for steady incompressible flows.

## 1. Governing Formulations

For steady, two-dimensional, incompressible viscous fluid flow with constant thermophysical properties, the energy equation is decoupled from the continuity and momentum equations. It can be formulated with respect to the total energy per unit mass $E$:

$$
\frac{\partial \rho E}{\partial t} = \frac{\partial}{\partial y_i} \left( k \frac{\partial T}{\partial y_i} \right) + \frac{\partial w_i}{\partial y_i} + \rho b_i u_i
$$

For steady conditions ($\partial / \partial t = 0$), the first term vanishes. Here, $k$ is the constant fluid thermal conductivity, and the variable $w_i$ encapsulates the convective transfer of internal and kinetic energy, along with the viscous dissipation effects:

$$
w_i = \sigma_{ij} u_j - \rho E u_i
$$

By enforcing the constitutive relationship for Newtonian fluids and expressing energy $E = c_v T + u_k u_k / 2$, the auxiliary vector $w_i$ is explicitly:

$$
w_i(\mathbf{y}) = \mu \left( \frac{\partial u_i}{\partial y_j} + \frac{\partial u_j}{\partial y_i} \right) u_j - \left(p + \frac{1}{2}\rho u_k u_k\right) u_i + \rho c_v u_i T
$$

where $c_v$ is the specific heat capacity at constant volume (for an incompressible liquid $c_v \approx c_p$). Notice that $w_i$ incorporates the temperature linearly due to the specific heat term $\rho c_v u_i T$.

## 2. Boundary-Domain Integral Equation

By applying a weighted residual formulation with the fundamental solution $T^*$ of the 2D diffusion operator, the differential equation transforms into the continuous Boundary-Domain Integral Equation (BDIE):

$$
c(\mathbf{x}) k T(\mathbf{x}) + \int_{\Gamma} T_{,n}^*(\mathbf{x}, \mathbf{y}) k T(\mathbf{y}) d\Gamma(\mathbf{y}) = -\int_{\Gamma} T^*(\mathbf{x}, \mathbf{y}) q(\mathbf{y}) d\Gamma(\mathbf{y}) + \int_{\Gamma} T^*(\mathbf{x}, \mathbf{y}) n_i(\mathbf{y}) w_i(\mathbf{y}) d\Gamma(\mathbf{y}) - \int_{\Omega} T_{,i}^*(\mathbf{x}, \mathbf{y}) w_i(\mathbf{y}) d\Omega(\mathbf{y})
$$

### Fundamental Solutions (Green's Kernels)

For a two-dimensional domain, the fundamental temperature solution and its derivatives are:

$$
T^*(\mathbf{x}, \mathbf{y}) = \frac{1}{2\pi} \ln \frac{1}{r}
$$

$$
T_{,i}^*(\mathbf{x}, \mathbf{y}) = \frac{\partial T^*}{\partial y_i} = \frac{-1}{2\pi r} r_{,i} = \frac{-1}{2\pi r^2}(y_i - x_i)
$$

$$
T_{,n}^*(\mathbf{x}, \mathbf{y}) = T_{,i}^* n_i = \frac{-1}{2\pi r^2}(y_i - x_i) n_i
$$

where $r = \|\mathbf{y} - \mathbf{x}\|$ is the distance from the source point $\mathbf{x}$ to the field point $\mathbf{y}$, $n_i$ represents the local outward-facing normal vector component, and $q(\mathbf{y}) = -k \frac{\partial T(\mathbf{y})}{\partial n}$ is the boundary heat flux. For points exactly on a smooth boundary, the geometric coefficient is $c(\mathbf{x}) = \frac{1}{2}$, whereas for internal points, $c(\mathbf{x}) = 1$.

## 3. Discretization and Matrix Assembly

To compute the temperature profile when subject to an arbitrary heat flux boundary condition, the boundary $\Gamma$ and the internal field domain $\Omega$ must be formally discretized.

1. **Boundary Elements**: Discretize $\Gamma$ discretely using panels or line elements.
2. **Internal Cells**: Although BDIM eliminates standard domain meshing for the diffusive terms, the $w_i$ component necessitates internal cell discretization covering the flow field (or strongly restricted to the viscous boundary layer).

By substituting nodal interpolation functions $N^\eta$ such that $T = \sum N^\eta T^\eta$, $q = \sum N^\eta q^\eta$, and evaluating exactly at all discrete nodes, the global algebraic system reduces to:

$$
[H] \{T_b\} = [G] \{q_b\} + [E_b] \{w\}
$$

Where:
- $\{T_b\}$ and $\{q_b\}$ are boundary temperature and heat flux vectors.
- $\{w\}$ is evaluated at all boundary and internal cell nodes.
- $[H]$, $[G]$, and $[E_b]$ are boundary-element characteristic matrices derived from integrating $T_{,n}^*$, $T^*$ and domain operators.

To handle the implicit temperature dependence embedded in $w_i$, $w_i$ is decomposed into its temperature-independent (convective kinetic and viscous dissipation sources) and temperature-dependent components:

$$
\{w\} = \{c_w\} + [B] \{T_b\} + [D] \{T_I\}
$$

where $\{c_w\}$ contains the known local evaluated variables $\left[\mu(u_{i,j} + u_{j,i})u_j - (p + \frac{1}{2}\rho u^2) u_i \right]$, while $[B]$ and $[D]$ are matrices populated systematically by the $\rho c_v u_i$ local mapping for boundary and internal points, respectively.

## 4. Solving for the Surface Temperature

### Imposing the Heat Flux Boundary Condition

Consider a scenario where the heat flux $q_w$ is strictly provided as the boundary condition across the surface. Let $\{X\}$ represent the vector of entirely unknown system boundary variables. Since heat flux is known everywhere on the surface, $\{X\} = \{T_b\}$.

1. Construct the grouped knowns vector, integrating the given heat fluxes:

$$[A_b] \{X\} = \{Y_b\} + [E_b] \{w\}$$

Since all $T_b$ are unknown, $[A_b] = [H]$ and $\{Y_b\} = [G] \{q_{\text{known}}\} + \text{BC contributions}$.

1. Substituting the expanded decomposition of $w$:

$$[A_b] \{X\} = \{Y_b\} + [E_b] \Big( \{c_w\} + [B] \{X\} + [D] \{T_I\} \Big)$$

3. To isolate the unknowns, this process must similarly be repeated for the internal domain evaluation resulting in $\{T_I\}$ dependencies:

$$\{T_I\} = [A_I] \{X\} + \{Y_I\} + [E_I] \Big( \{c_w\} + [B] \{X\} + [D] \{T_I\} \Big)$$

By algebraically rearranging these boundary and internal constraints, a completely linear, uncoupled matrix system arises that allows simultaneous resolution without iterative processes:

$$
\begin{bmatrix}
[A^b] & [\widetilde{E}^b] \\
[A^I] & [\widetilde{E}^I]
\end{bmatrix}
\begin{Bmatrix}
\{X\} \\
\{T_I\}
\end{Bmatrix}
=
\begin{Bmatrix}
\{\widetilde{Y}^b\} \\
\{\widetilde{Y}^I\}
\end{Bmatrix}
$$

Where the blocks absorb explicit dependencies mapped over from $[H]$ and the diagonal mappings inside $[B]$ and $[D]$ representing local physical enthalpy advection. The inversion of this linear matrix immediately resolves $\{X\}$, yielding the direct boundary temperature distribution $T_w(s)$ across the discretized surface perimeter points natively linked to to heat flux constraints.

## 5. Heat Transfer Formulation Algorithm

For a programmatic pipeline in the solver module, the procedure operates as follows:

1. **Prerequisite Invocation**:
   Accept the discrete velocity gradients $u_{i,j}$, fluid pressure $p$, and density/viscosity mappings transferred independently from previous boundary layer or inviscid executions.
   Define the boundary vector arrays where $q(s)$ is enforced directly. Let $q>0$ represent flux into the fluid system.

2. **Matrix Construction**:
   Assembly of Green's functions kernels $T^*, T_{,n}^*, T_{,i}^*$. Compute dense characteristic matrices $[H], [G], [E_b], [E_I]$.
   Assign arrays evaluating $c_w$ directly from current $u_i, p, \mu$.

3. **System Solution**:
   Solve the algebraic matrix system delineated above. Slice the resulting state vector to acquire purely boundary node quantities $\{T_w\}$.

4. **Surface Heat Rate Integration**:
   Using the recovered analytical wall temperature trajectory $T_w(s)$ relative to established fluid asymptotic freestream temperature $T_\infty$, standard Newton transfer relationships dictate the localized coefficient:

$$
   h(s) = \frac{q_w(s)}{T_w(s) - T_\infty}
$$

   Derived directly into non-dimensional similarity metrics across characteristic length $L$:

$$
   \text{Local Nusselt Number, } Nu(s) = \frac{h(s) \cdot L}{k} = \frac{q_w(s) \cdot L}{k\left( T_w(s) - T_\infty \right)}
$$

   Note that because $q(\mathbf{x})$ acts directly on boundary, the explicit resolution of these fractions precisely completes the target task output structure defining local Nusselt distributions along any generalized immersed arbitrary curvature. The total surface integrated heat transfer rate per unit span ($[W/m]$) effectively mirrors integrating the fixed applied $q_w$: 
   
$$
   Q_{total} = \int_{0}^{s_{max}} q_w(s) ds
$$

---
**References:**
[1] Gao, X.-W., Peng, H.-F., & Liu, J. (2013). A boundary-domain integral equation method for solving convective heat transfer problems. *International Journal of Heat and Mass Transfer*.
