# Pohlhausen Polynomial Profile

The Pohlhausen method uses a 4th-degree polynomial to approximate the
velocity profile within the boundary layer. It is a classical integral
method that incorporates the pressure gradient through the shape
parameter $\Lambda$.

## Profile Function

$$\frac{u}{U_e} = g(\eta) = 2\eta - 2\eta^3 + \eta^4
+ \frac{\Lambda}{6}\,\eta\,(1-\eta)^3$$

where $\eta = y/\delta$ and $\Lambda$ is the Pohlhausen pressure-gradient
parameter:

$$\Lambda = \frac{\delta^2}{\nu}\,\frac{dU_e}{ds}$$

**Valid range**: $-12 \le \Lambda \le 12$.

- At $\Lambda = -12$: zero wall shear (incipient separation)
- At $\Lambda = 0$: zero pressure gradient
- At $\Lambda = 12$: profile overshoots $u/U_e > 1$ near the edge

## Coefficient Determination

The polynomial coefficients are determined from boundary conditions:

- **At the wall** ($\eta = 0$): $u = 0$, $du/dy$ determined by the
  pressure gradient
- **At the BL edge** ($\eta = 1$): $u = U_e$, $du/dy = 0$

The additional condition from the momentum equation at the wall
introduces the pressure gradient dependence through $\Lambda$.

## Integral Ratios

Direct polynomial integration yields exact analytical expressions:

$$G(\Lambda) = \frac{\delta^*}{\delta}
= \frac{3}{10} - \frac{\Lambda}{120}$$

$$\Phi(\Lambda) = \frac{\theta}{\delta}
= \frac{37}{315} - \frac{\Lambda}{945} - \frac{\Lambda^2}{9072}$$

$$\tau(\Lambda) = \frac{\delta}{\nu}\,\frac{\tau_w}{\rho U_e}
= 2 + \frac{\Lambda}{6}$$

The shape factor is $H = G/\Phi$.

??? info "Derivation of $G(\Lambda)$"
    $$G = \int_0^1 \left[1 - 2\eta + 2\eta^3 - \eta^4
    - \frac{\Lambda}{6}\eta(1-\eta)^3\right] d\eta$$

    $$= 1 - 1 + \frac{1}{2} - \frac{1}{5}
    - \frac{\Lambda}{6}\int_0^1 \eta(1 - 3\eta + 3\eta^2 - \eta^3)\,d\eta$$

    $$= \frac{3}{10} - \frac{\Lambda}{6}\left(\frac{1}{2} - 1
    + \frac{3}{4} - \frac{1}{5}\right)
    = \frac{3}{10} - \frac{\Lambda}{6} \cdot \frac{1}{20}
    = \frac{3}{10} - \frac{\Lambda}{120}$$

??? info "Derivation of $\Phi(\Lambda)$"
    The momentum-thickness integral $\int_0^1 g(1-g)\,d\eta$ expands to
    a sum of polynomial integrals containing terms up to $\eta^8$.
    Collecting powers of $\Lambda$:

    $$\Phi = \frac{37}{315} - \frac{\Lambda}{945}
    - \frac{\Lambda^2}{9072}$$

    This is tedious but purely mechanical — expand $g^2$, integrate term
    by term, collect powers of $\Lambda$. Verifiable with any CAS.

## Closure Relations

$$H = \frac{G(\Lambda)}{\Phi(\Lambda)}$$

$$\frac{c_f}{2} = \frac{\nu}{\delta\,U_e}\left(2 + \frac{\Lambda}{6}\right)
= \frac{\nu\,\tau(\Lambda)}{\delta\,U_e}$$

Since $\delta = \theta/\Phi(\Lambda)$ and $\Lambda$ is determined by
the local $dU_e/ds$, the system is closed.

## Stagnation Patching

Writing $\theta = \Phi\delta$, $H = G/\Phi$, and substituting into the
equilibrium form of the momentum integral equation at $U_e = Ks$:

$$(2\Phi + G)\,\delta^2 K = \tau\,\nu$$

Since $\Lambda = \delta^2 K/\nu$:

$$\boxed{\Lambda\bigl[2\Phi(\Lambda) + G(\Lambda)\bigr] = \tau(\Lambda)}$$

This is a single nonlinear equation in $\Lambda$, solvable by Newton's
method or bisection.

### Numerical Solution

| $\Lambda$ | $\Phi$ | $G$ | $2\Phi+G$ | $\tau$ | $\Lambda(2\Phi+G)$ | Residual |
|-----------|--------|------|-----------|--------|---------------------|----------|
| 6.0 | 0.1108 | 0.2500 | 0.4716 | 3.000 | 2.830 | $-0.170$ |
| 7.0 | 0.1047 | 0.2417 | 0.4510 | 3.167 | 3.157 | $-0.010$ |
| **7.052** | **0.10452** | **0.2412** | **0.4503** | **3.175** | **3.175** | $\approx 0$ |
| 8.0 | 0.0981 | 0.2333 | 0.4296 | 3.333 | 3.437 | $+0.104$ |

$$\boxed{\Lambda_\text{stag} \approx 7.052}$$

**Derived quantities:**

$$\delta_\text{stag} = \sqrt{\frac{7.052\,\nu}{K}}, \qquad
\theta_\text{stag} = 0.1045\,\delta_\text{stag}, \qquad
\theta^2_\text{stag} = 0.0770\,\frac{\nu}{K}$$

$$H_\text{stag} = \frac{G}{\Phi} = \frac{0.2412}{0.1045} = 2.308$$

## Velocity Field Reconstruction

### Recovering $\Lambda$ from $H$

The solver stores $H$ at each station. The profile parameter $\Lambda$
is found by solving $H = G(\Lambda)/\Phi(\Lambda)$, which rearranges to
the quadratic:

$$\frac{H}{9072}\,\Lambda^2
+ \left(\frac{H}{945} - \frac{1}{120}\right)\Lambda
+ \left(\frac{3}{10} - \frac{37H}{315}\right) = 0$$

Take the root satisfying $-12 \le \Lambda \le 12$.

### Recovering $\delta$

$$\delta = \frac{\theta}{\Phi(\Lambda)}$$

Or equivalently $\delta = \delta^*/G(\Lambda)$. Both give the same answer
as a consistency check.

### Reconstruction Procedure

At each station $s_i$:

1. Read $\theta_i$, $H_i$, $U_{e,i}$ from the solver
2. Solve the quadratic for $\Lambda_i$
3. Compute $\delta_i = \theta_i / \Phi(\Lambda_i)$
4. For any wall-normal distance $y$:
   $u_i(y) = U_{e,i} \cdot g(y/\delta_i;\,\Lambda_i)$

## References

1. Schlichting, H. and Gersten, K., *Boundary-Layer Theory* (8th ed.), §8.3.
2. Anderson, J.D., *Fundamentals of Aerodynamics* (6th ed.), §4.5.
