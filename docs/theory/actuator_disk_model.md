# Actuator Disk Model

The actuator disk model represents a cooling fan as an infinitesimally thin
surface that adds static pressure to the flow. The disk is not a solid wall:
normal velocity is allowed through it, and the fan operating point is found
from the intersection of the panel-solver flow rate and the supplied P-Q curve.

For a disk with configured unit normal `n`, positive flow rate is

```text
Q = integral(V dot n dS)
```

Positive pressure rise follows the same normal direction. A reversed flow gives
a negative `V dot n`; the current implementation clamps fan-curve lookup to the
tabulated flow range.

## Potential-Flow Coupling

In potential theory, a doublet sheet represents a jump in velocity potential
across a surface. The ADM uses this equivalence: each disk panel carries a
prescribed doublet strength, and the body-panel system treats the disk influence
as a known disturbance rather than enforcing no-through-flow on the disk.

The body-panel Neumann condition becomes

```text
(V_inf + V_body + V_adm) dot n_body = 0
```

so the right-hand side is modified as

```text
A sigma = -(V_inf dot n_body + V_adm dot n_body)
```

The first implementation uses a point-doublet approximation for each disk panel:
the panel moment is `mu * area * n_disk`, evaluated at the panel center. This is
simple, modular, and adequate for early fan-system coupling studies, but it can
be replaced later with exact quadrilateral doublet influence coefficients.

## Pressure Jump Convention

A steady pressure jump is a fan energy input, while a potential jump has units
of `m^2/s`. For this simple ADM, the mapping is isolated as

```text
mu = Delta p * R / (rho * U_ref)
```

where `R` is the disk radius and `U_ref` is the freestream velocity component in
the disk-normal direction, or a fan-curve velocity scale when the ambient flow
is stationary. The stationary case uses the midpoint tabulated flow rate divided
by disk area for the initial velocity scale, so the fan curve, not an artificial
inlet velocity, sets the operating condition. This expresses the pressure rise
as a specific-energy jump scaled by a disk-length convective scale. The mapping
is intentionally localized in code so higher-fidelity conventions can replace
it without changing the coupling architecture.

## P-Q Iteration

Each fan loads `data/fan_curve.csv` from its case directory. At every iteration:

1. Convert current pressure rise to disk doublet strength.
2. Solve the configured 3D panel solver with the ADM disturbance on the RHS.
3. Evaluate velocity on offset disk sampling planes so the disk's own induced
   field contributes without sampling exactly on the singular sheet.
4. Integrate `Q = sum((V dot n) area)`.
5. Interpolate the fan curve to get a new pressure rise.
6. Under-relax the update until the pressure residual converges.

If the evaluated disk flow rate leaves the tabulated P-Q curve range, the ADM
iteration stops immediately and reports a warning. This avoids continuing with
clamped fan-curve values after the operating point has left the supplied data.

The P-Q curve is assumed to correspond to the fan speed being simulated. For a
manufacturer maximum-speed curve, the RPM/PWM setting is already implicit in the
data. RPM becomes an explicit input only when scaling to another speed or PWM
setting, typically using fan affinity laws as a first approximation
(`Q proportional to N`, `Delta p proportional to N^2`).

Convergence plots are saved under `out/adm/`, and reusable solver data is saved
under `out/solverRuns/`.

## Limitations

- No swirl.
- Uniform pressure loading over each disk.
- Point-doublet panel approximation.
- No Kutta condition for duct trailing edges yet.
- No branch-wake or disk-rim singularity mitigation yet.
- Fluent comparison is scaffolded separately and should be run only when Fluent
  cut-plane exports are available.
