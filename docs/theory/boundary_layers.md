# Boundary Layer Theory

## Von Kármán Momentum Integral Method

The solver uses the Von Kármán momentum integral equation to compute
boundary layer quantities along a body surface given the edge velocity
$U_e(s)$ from the inviscid panel method.

### Governing Equation

The 2-D steady incompressible momentum integral equation:

$$\frac{d\theta}{ds} + \frac{\theta}{U_e}\frac{dU_e}{ds}(2 + H) = \frac{c_f}{2}$$

where:

- $\theta(s)$ — momentum thickness
- $\delta^*(s) = H \theta$ — displacement thickness
- $H = \delta^*/\theta$ — shape factor
- $c_f/2$ — half skin-friction coefficient
- $U_e(s)$ — edge velocity from the panel solver

### Closure Relations

The ODE has three unknowns ($\theta$, $H$, $c_f$) but only one equation.
**Closure** is provided by a velocity profile assumption that relates $H$
and $c_f/2$ to the local state ($Re_\theta$, pressure-gradient parameter $\lambda$).

Each profile is implemented as a `VelocityProfile` subclass returning
`ProfileClosureData(H, cf_2)`.

### Implemented Velocity Profiles

| Profile | Type | $H$ | $c_f/2$ | Pressure gradient |
|---------|------|-----|---------|-------------------|
| **Blasius** | Laminar flat-plate | 2.591 (const) | $0.2205 / Re_\theta$ | Ignored |
| **Pohlhausen** | Laminar polynomial | $f(\Lambda)$ | $f(\Lambda, Re_\theta)$ | Λ = δ²/ν · dUe/ds |
| **Falkner-Skan** | Laminar similarity | $H(\beta)$ table | $S(\beta)/Re_\theta$ table | β ≈ 2λ |
| **Thwaites** | Laminar correlation | $H(\lambda)$ table | $S(\lambda)/Re_\theta$ table | λ = θ²/ν · dUe/ds |
| **Power-law 1/n** | Turbulent | $(n+2)/n$ | $a/Re_\theta^{1/(n+1)}$ | Ignored |

### Integration Procedure

1. **Locate start**: Find the first station where $|U_e| > 0.01 \max|U_e|$
   (skips stagnation singularity).
2. **Initial condition**: Each profile provides `initial_theta(ν, Ue₀)` for
   bootstrapping (e.g. Thwaites: $\theta_0 = \sqrt{0.45 \nu s_0 / U_{e0}}$).
3. **ODE integration**: `scipy.integrate.solve_ivp` (RK45) marches $\theta(s)$
   forward and backward from the start index.
4. **Derived quantities**: At each station, $H$, $\delta^*$, $c_f$, $Re_\theta$
   are computed from the profile closure.

### Thwaites' Quadrature

Thwaites' method also provides a direct closed-form solution bypassing
the ODE:

$$\theta^2(s) = \frac{0.45\,\nu}{U_e(s)^6} \int_0^s U_e(s')^5\,ds'$$

Available via `ThwaitesProfile.quadrature_theta(s, Ue, nu)`.

## Transition Prediction

Two laminar–turbulent transition criteria are implemented:

### Michel's Criterion

Empirical correlation:

$$Re_\theta \geq 1.174\,Re_x^{0.46}$$

### Simplified e^N Method

Single-parameter amplification model. Instability onset at $Re_\theta \approx 150$
(Tollmien-Schlichting waves). Transition when the amplification factor
$n \geq N_{crit}$ (default 9.0).

## Viscous-Inviscid Coupling

!!! note "Planned"
    The displacement thickness $\delta^*(s)$ from the BL solver will be
    used to compute a transpiration velocity that modifies the panel
    method Neumann boundary condition, enabling iterative viscous-inviscid
    coupling.

## Thermal Boundary Layer

The thermal boundary layer solver is formulated using the Boundary-Domain Integral Method (BDIM) to solve the viscous energy equation explicitly without iterative processes. With the fluid velocity parameters resolved, the methodology extracts precise surface temperature or heat flux distributions depending on known inputs.

For full mathematical descriptions, refer to the [Thermal Boundary Layer (BDIM)](thermal_boundary_layer.md) documentation.

## References

1. Katz, J. and Plotkin, A., *Low-Speed Aerodynamics* (2nd ed.), Chapter 14.
2. Drela, M., *Flight Vehicle Aerodynamics*, MIT Press, 2014.
3. Cebeci, T. and Bradshaw, P., *Physical and Computational Aspects of Convective Heat Transfer*, Springer, 1984.
4. White, F.M., *Viscous Fluid Flow* (3rd ed.), McGraw-Hill.
5. Schlichting, H., *Boundary-Layer Theory* (8th ed.), Springer.
