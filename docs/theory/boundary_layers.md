# Boundary Layer Theory

## Von Kármán Momentum Integral Method

The solver uses the Von Kármán momentum integral equation to compute
boundary layer quantities along a body surface given the edge velocity
$U_e(s)$ from the inviscid panel method.

### Governing Equation

Through integration of the boundary layer equations across the layer
thickness, the Navier–Stokes PDEs reduce to a single ODE in the
arc-length coordinate $s$:

$$\frac{d\theta}{ds} + \frac{\theta}{U_e}\frac{dU_e}{ds}(2 + H) = \frac{c_f}{2}$$

where:

| Symbol | Quantity |
|--------|----------|
| $\theta(s)$ | Momentum thickness |
| $\delta^*(s) = H\theta$ | Displacement thickness |
| $H = \delta^*/\theta$ | Shape factor |
| $c_f/2$ | Half skin-friction coefficient |
| $U_e(s)$ | Edge velocity from the panel solver |

### Closure Relations

The ODE has three unknowns ($\theta$, $H$, $c_f$) but only one equation.
**Closure** is provided by a velocity profile assumption that relates $H$
and $c_f/2$ to the local state ($Re_\theta$, pressure-gradient parameter).

Each profile is implemented as a `VelocityProfile` subclass returning
`ProfileClosureData(H, cf_2)`.

### Implemented Velocity Profiles

| Profile | Type | $H$ | $c_f/2$ | Pressure gradient | Details |
|---------|------|-----|---------|-------------------|---------|
| [**Blasius**](bl_profiles/blasius.md) | Laminar flat-plate | 2.591 (const) | $0.2205 / Re_\theta$ | Ignored | Exact similarity ODE |
| [**Falkner–Skan**](bl_profiles/falkner_skan.md) | Laminar similarity | $H(\beta)$ table | $S(\beta)/Re_\theta$ table | $\beta = 2m/(m+1)$ | Tabulated ODE solutions |
| [**Pohlhausen**](bl_profiles/pohlhausen.md) | Laminar polynomial | $f(\Lambda)$ | $f(\Lambda, Re_\theta)$ | $\Lambda = \delta^2/\nu \cdot dU_e/ds$ | 4th-degree polynomial |
| [**Thwaites**](bl_profiles/thwaites.md) | Laminar correlation | $H(\lambda)$ table | $S(\lambda)/Re_\theta$ table | $\lambda = \theta^2/\nu \cdot dU_e/ds$ | Quadrature + correlations |
| [**Power-law $1/n$**](bl_profiles/power_law.md) | Turbulent | $(n+2)/n$ | $a/Re_\theta^{1/(n+1)}$ | Ignored | Empirical |

---

## Stagnation Point Handling

### The Stagnation Singularity

At a stagnation point the edge velocity $U_e = 0$, causing the momentum
integral equation to become singular — the $dU_e/ds$ term diverges while
$c_f/2$ involves division by $U_e$.  The standard approach of simply
"starting the integration a few panels downstream" is ad-hoc and
introduces errors.

### Exact Stagnation Detection

The solver detects the stagnation point by finding where the signed
tangential velocity $V_t$ changes sign along each boundary layer path:

1. **Sign-change interpolation**: Scan along the path for adjacent panels
   where $V_t$ changes sign. The exact stagnation location $s_0$ is found
   by linear interpolation between the two panels.
2. **Fallback**: If no sign change exists (e.g. the body has no true
   forward stagnation point), the panel with minimum $|V_t|$ is used.
3. **Arc-length re-zeroing**: The arc-length coordinate is re-zeroed so
   that $s = 0$ at the stagnation point. Panels with $s < 0$ lie on the
   opposite branch and are excluded from the BL integration.

### Velocity Gradient $K$

Near the stagnation point, the edge velocity is linear:

$$U_e(s) \approx K s$$

The velocity gradient $K = dU_e/ds|_{s=0}$ is computed by linear
regression of $U_e$ vs $s$ over near-stagnation panels, forced through
the origin. This value is then used for analytical stagnation patching.

### Analytical Stagnation Patching

Rather than starting the ODE from an arbitrary small $\theta$, each
profile provides an **analytical initial momentum thickness** at the
stagnation point by solving the equilibrium form of the momentum integral
equation at $s \to 0$.

With $U_e = Ks$ the momentum integral equation reduces to:

$$\frac{(2 + H)\,\theta}{s} = \frac{c_f}{2}$$

Substituting each profile's closure relations yields a profile-specific
equilibrium:

| Profile | $\theta^2_\text{stag}$ | Physical basis |
|---------|----------------------|----------------|
| [Blasius](bl_profiles/blasius.md#stagnation-patching) | $0.04803\;\nu/K$ | Self-consistent but ignores pressure gradient |
| [Falkner–Skan](bl_profiles/falkner_skan.md#stagnation-patching) | $0.08547\;\nu/K$ | **Exact** (Hiemenz flow, $\beta = 1$) |
| [Pohlhausen](bl_profiles/pohlhausen.md#stagnation-patching) | $0.0770\;\nu/K$ | Polynomial equilibrium at $\Lambda \approx 7.05$ |
| [Thwaites](bl_profiles/thwaites.md#stagnation-patching) | $0.075\;\nu/K$ | Quadrature formula limit |
| [Power-law](bl_profiles/power_law.md#stagnation-patching) | N/A | Turbulent — no stagnation patch |

The Falkner–Skan value is the exact reference solution. All other methods
approximate it to within 5–12%.

---

## Integration Procedure

1. **Stagnation detection**: Find exact stagnation point via sign-change
   interpolation on $V_t$. Compute velocity gradient $K$.
2. **Initial condition**: Use the profile's analytical
   `stagnation_theta(ν, K)` if available; fall back to
   `initial_theta(ν, Ue₀)` otherwise.
3. **Start index**: Begin integration at the first panel with $s > 0$
   (i.e. just downstream of the stagnation point).
4. **ODE integration**: `scipy.integrate.solve_ivp` (RK45) marches
   $\theta(s)$ forward from the start index. At each station the profile
   closure provides $H$ and $c_f/2$.
5. **Derived quantities**: At each station, $\delta^*$, $c_f$, $Re_\theta$
   are computed from the profile closure and the current $\theta$.

### Thwaites' Quadrature

Thwaites' method also provides a direct closed-form solution bypassing
the ODE entirely:

$$\theta^2(s) = \frac{0.45\,\nu}{U_e(s)^6} \int_0^s U_e(s')^5\,ds'$$

Available via `ThwaitesProfile.quadrature_theta(s, Ue, nu)`.
See [Thwaites Method](bl_profiles/thwaites.md) for the derivation.

---

## Velocity Field Reconstruction

After the integral solver produces $\theta(s)$, $H(s)$, $U_e(s)$, the
full velocity field $u(s, y)$ within the boundary layer can be
**reconstructed** by reversing the integral process.

### Core Principle

The integral method *assumes* a velocity profile shape and integrates it
to obtain $\theta$ and $\delta^*$. To recover the velocity field:

1. From the solver output ($\theta$, $H$), determine the **free
   parameter** of the assumed profile ($\Lambda$, $\beta$, or $n$)
2. Recover the **boundary layer thickness** $\delta$ using the known
   analytical ratio $\theta/\delta$ for that profile
3. Evaluate the **profile function** $u/U_e = g(y/\delta)$ at desired $y$
   locations

!!! note "Accuracy caveat"
    The reconstructed velocity field is only as accurate as the profile
    assumption. It is *not* the true Navier–Stokes solution — it is the
    best representation consistent with the integral method's assumptions.

### Profile Categories

Profiles fall into two categories for reconstruction:

**Finite-domain profiles** (Pohlhausen, Power-law): defined on
$y \in [0, \delta]$ with $u = U_e$ at $y = \delta$. The boundary layer
thickness $\delta$ is exact and well-defined.

$$u(s, y) = \begin{cases}
U_e(s) \cdot g\!\left(\dfrac{y}{\delta(s)}\right) & 0 \le y \le \delta \\
U_e(s) & y > \delta
\end{cases}$$

**Similarity profiles** (Blasius, Falkner–Skan): defined on
$y \in [0, \infty)$ with $u \to U_e$ asymptotically. There is no finite
$\delta$; we use $\delta_{99}$ where $u/U_e = 0.99$.

$$u(s, y) = U_e(s) \cdot f'\!\left(\frac{y}{L(s)}\right), \qquad
L = \frac{\theta}{I_2}$$

**Correlation methods** (Thwaites): no explicit profile shape. Must be
paired with Falkner–Skan or Pohlhausen for reconstruction. Configurable
via `thwaites_reconstruction` in the case YAML.

### Boundary Layer Thickness Summary

| Profile | $\delta/\theta$ formula | Typical value |
|---------|------------------------|---------------|
| Pohlhausen ($\Lambda = 0$) | $1/\Phi(\Lambda)$ | $8.51$ |
| Power-law $n=7$ | $(n+1)(n+2)/n$ | $10.29$ |
| Blasius | $\eta_{99}/I_2$ | $7.53$ |
| Falkner–Skan $\beta=1$ | $\eta_{99}(\beta)/I_2(\beta)$ | $\approx 8.2$ |
| Thwaites | Delegates to paired profile | — |

Each profile's reconstruction details are documented on their individual
pages.

---

## Transition Prediction

Two laminar-turbulent transition criteria are implemented:

### Michel's Criterion

Empirical correlation:

$$Re_\theta \geq 1.174\,Re_x^{0.46}$$

### Simplified $e^N$ Method

Single-parameter amplification model. Instability onset at
$Re_\theta \approx 150$ (Tollmien–Schlichting waves). Transition when
the amplification factor $n \geq N_\text{crit}$ (default 9.0).

---

## Viscous-Inviscid Coupling

!!! note "Planned"
    The displacement thickness $\delta^*(s)$ from the BL solver will be
    used to compute a transpiration velocity that modifies the panel
    method Neumann boundary condition, enabling iterative viscous-inviscid
    coupling.

---

## Thermal Boundary Layer

The thermal boundary layer solver is formulated using the Boundary-Domain
Integral Method (BDIM) to solve the viscous energy equation explicitly
without iterative processes. With the fluid velocity parameters resolved,
the methodology extracts precise surface temperature or heat flux
distributions depending on known inputs.

For full mathematical descriptions, refer to the
[Thermal Boundary Layer (BDIM)](thermal_boundary_layer.md) documentation.

---

## References

1. Katz, J. and Plotkin, A., *Low-Speed Aerodynamics* (2nd ed.), Chapter 14.
2. Drela, M., *Flight Vehicle Aerodynamics*, MIT Press, 2014.
3. White, F.M., *Viscous Fluid Flow* (3rd ed.), McGraw-Hill, §4-3 to §4-6, §6-5.
4. Schlichting, H. and Gersten, K., *Boundary-Layer Theory* (8th ed.), Springer, §7–9, §21.
5. Cebeci, T. and Bradshaw, P., *Physical and Computational Aspects of Convective Heat Transfer*, Springer, 1984.
6. Thwaites, A., "Approximate calculation of the laminar boundary layer," *Aero. Quarterly*, Vol. 1, 1949, pp. 245–280.
7. Hartree, D.R., "On an equation occurring in Falkner and Skan's approximate treatment of the equations of the boundary layer," *Proc. Cambridge Phil. Soc.*, 33(2), 1937, pp. 223–239.
