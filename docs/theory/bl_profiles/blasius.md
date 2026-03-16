# Blasius Profile

The Blasius profile is a classic exact similarity solution to the 2D,
steady, laminar, incompressible boundary layer equations for flow over a
flat plate at zero incidence.

## Assumptions

- Steady, laminar, incompressible flow
- Zero pressure gradient: $dU_e/dx = 0$

## Governing ODE

The Blasius equation:

$$f''' + \tfrac{1}{2} f\, f'' = 0$$

with boundary conditions:

| Condition | Meaning |
|-----------|---------|
| $f(0) = 0$ | No flow through the wall |
| $f'(0) = 0$ | No-slip at the wall |
| $\lim_{\eta \to \infty} f'(\eta) = 1$ | Velocity approaches freestream |

The similarity variable is:

$$\eta = y \sqrt{\frac{U_e}{2\nu s}} = \frac{y}{L}, \qquad
L = \sqrt{\frac{2\nu s}{U_e}}$$

and the normalised velocity is $u/U_e = f'(\eta)$.

!!! warning "Blasius vs Falkner–Skan at $\beta = 0$"
    The Blasius ODE ($f''' + \frac{1}{2}ff'' = 0$) and the Falkner–Skan
    ODE at $\beta = 0$ ($f''' + ff'' = 0$) are **different equations**
    with a $\sqrt{2}$ scaling difference. They describe the same physical
    flow but use different similarity variable definitions. Separate
    tabulated data is required for each.

## Key Constants

The shooting solution (Topfer scaling trick: solve with $f''(0) = 1$,
get $f'(\infty) = A$, then correct $f''(0) = 1/A^{3/2}$) yields:

| Quantity | Symbol | Value |
|----------|--------|-------|
| Wall shear parameter | $f''(0)$ | $0.33206$ |
| Displacement integral | $I_1 = \int_0^\infty(1-f')\,d\eta$ | $1.72080$ |
| Momentum integral | $I_2 = \int_0^\infty f'(1-f')\,d\eta$ | $0.66411$ |
| Shape factor | $H = I_1/I_2$ | $2.5911$ |
| 99% thickness | $\eta_{99}$ | $\approx 5.0$ |

**Origin of the closure constants:**

- $H = 2.591 = 1.7208/0.6641$: ratio of the displacement and momentum
  integrals from the Blasius profile.
- $c_f/2 = 0.2205/Re_\theta$: arises from $c_f/2 = f''(0) \cdot \nu /
  (U_e L)$, converting to $Re_\theta$ using $Re_\theta = I_2 \sqrt{2 Re_s}$
  gives $c_f/2 = f''(0) \cdot I_2 / Re_\theta = 0.33206 \times 0.66411 =
  0.2205$.

## Closure Relations

$$H = 2.591 \quad (\text{constant})$$

$$\frac{c_f}{2} = \frac{0.2205}{Re_\theta}$$

Since both $H$ and $c_f/2$ depend only on $\theta$ and $U_e$, the
Von Kármán momentum integral equation becomes a single solvable ODE
for $\theta(s)$.

## Tabulated Profile

A selection of values from the numerical solution:

| $\eta$ | $f'(\eta)$ |
|--------|-----------|
| 0.0 | 0.0000 |
| 0.4 | 0.1328 |
| 0.8 | 0.2647 |
| 1.2 | 0.3938 |
| 1.6 | 0.5168 |
| 2.0 | 0.6298 |
| 2.4 | 0.7290 |
| 2.8 | 0.8115 |
| 3.2 | 0.8761 |
| 3.6 | 0.9233 |
| 4.0 | 0.9555 |
| 4.4 | 0.9759 |
| 4.8 | 0.9878 |
| 5.0 | 0.9916 |

The full tabulated solution (200 points, $\eta \in [0, 10]$) is stored
in `data/bl-solver-profiles/blasius.json` and loaded at runtime by
`BlasiusTable`.

## Stagnation Patching

Substituting the Blasius closure into the equilibrium form of the
momentum integral equation at $U_e = Ks$:

$$\frac{(2 + H)\,\theta}{s} = \frac{c_f}{2}$$

$$\frac{4.591\,\theta}{s} = \frac{0.2205\,\nu}{\theta \cdot Ks}$$

The $1/s$ cancels:

$$4.591\,\theta^2 K = 0.2205\,\nu$$

$$\boxed{\theta^2_\text{stag} = \frac{0.2205}{4.591}\,\frac{\nu}{K}
= 0.04803\,\frac{\nu}{K}}$$

**Derived quantities:**

$$\delta^*_\text{stag} = 2.591\,\theta_\text{stag}, \qquad
\frac{c_f}{2}\bigg|_{s_1} = \frac{0.2205\,\nu}{\theta_\text{stag}
\cdot K s_1}$$

!!! note "Physical caveat"
    This is internally self-consistent (finite equilibrium $\theta$) but
    physically inconsistent because the Blasius profile ignores the
    pressure gradient that defines stagnation-point flow. Use this only
    when no better laminar profile is available.

## Velocity Field Reconstruction

### Recovering $\delta_{99}$

The length scale $L$ is recovered from the integral solver's $\theta$:

$$L = \frac{\theta}{I_2} = \frac{\theta}{0.6641}$$

$$\delta_{99} = \eta_{99} \cdot L = \frac{5.0}{0.6641}\,\theta
\approx 7.53\,\theta$$

### Reconstruction Procedure

At each station $s_i$:

1. $L_i = \theta_i / 0.6641$
2. $\delta_{99,i} = 5.0 \cdot L_i$
3. For any wall-normal distance $y$: compute $\eta = y / L_i$, then
   $u_i(y) = U_{e,i} \cdot f'(\eta)$ by interpolation from the
   tabulated Blasius solution.

The profile $f'(\eta)$ is stored in `blasius.json` and accessed through
the `BlasiusTable` class, which provides `fprime(eta)` interpolation.

## References

1. White, F.M., *Viscous Fluid Flow* (3rd ed.), §4-3, Table 4-1.
2. Schlichting, H. and Gersten, K., *Boundary-Layer Theory* (8th ed.), §7.3.
