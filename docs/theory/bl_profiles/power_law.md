# Power-Law Turbulent Profile

For turbulent boundary layers, a commonly used empirical velocity
profile is the power-law:

$$\frac{u}{U_e} = \left(\frac{y}{\delta}\right)^{1/n}$$

with $n \approx 7$ being the typical choice (the "1/7th power law").

## Profile Function

$$g(\eta) = \eta^{1/n}, \qquad \eta = y/\delta \in [0, 1]$$

!!! warning "Wall singularity"
    The derivative $g'(0) = \frac{1}{n}\eta^{1/n-1} \to \infty$ as
    $\eta \to 0$. The wall shear stress from the profile itself is
    singular. In practice, the power-law is not valid in the viscous
    sublayer ($y^+ \lesssim 30$). The skin friction closure
    $c_f/2 = a/Re_\theta^{1/(n+1)}$ is an *empirical correlation*, not
    derived from differentiating the profile at the wall. For
    visualization, this singularity affects only the first fraction of
    a percent of $\delta$ and is inconsequential.

## Integral Ratios

Exact integration of the power-law profile:

$$G = \frac{\delta^*}{\delta} = \int_0^1(1 - \eta^{1/n})\,d\eta
= \frac{1}{n+1}$$

$$\Phi = \frac{\theta}{\delta}
= \int_0^1 \eta^{1/n}(1 - \eta^{1/n})\,d\eta
= \frac{n}{(n+1)(n+2)}$$

$$H = \frac{G}{\Phi} = \frac{n+2}{n}$$

??? info "Derivation of $\Phi$"
    $$\int_0^1 \eta^{1/n}\,d\eta = \frac{n}{n+1}$$

    $$\int_0^1 \eta^{2/n}\,d\eta = \frac{n}{n+2}$$

    $$\Phi = \frac{n}{n+1} - \frac{n}{n+2}
    = \frac{n(n+2) - n(n+1)}{(n+1)(n+2)}
    = \frac{n}{(n+1)(n+2)}$$

### Values for $n = 7$

| Ratio | Formula | Value |
|-------|---------|-------|
| $\delta^*/\delta$ | $1/(n+1)$ | $1/8 = 0.125$ |
| $\theta/\delta$ | $n/[(n+1)(n+2)]$ | $7/72 = 0.09722$ |
| $H$ | $(n+2)/n$ | $9/7 = 1.2857$ |
| $\delta/\theta$ | $(n+1)(n+2)/n$ | $72/7 = 10.286$ |
| $\delta/\delta^*$ | $n+1$ | $8$ |

## Closure Relations

$$H = \frac{n+2}{n} \quad (\text{constant for fixed } n)$$

$$\frac{c_f}{2} = \frac{a}{Re_\theta^{1/(n+1)}}$$

For $n = 7$: $H = 9/7 \approx 1.286$, $a \approx 0.0128$,
exponent $= 1/8$.

## Stagnation Patching

Substituting the power-law closure into the equilibrium equation with
$U_e = Ks$:

$$\frac{(2+H)\,\theta}{s}
= a\left(\frac{\nu}{\theta\,Ks}\right)^{1/(n+1)}$$

The left side scales as $\theta/s$. The right side scales as
$s^{-1/(n+1)}$. For balance:

$$\frac{\theta}{s} \propto s^{-1/(n+1)}
\quad\Rightarrow\quad \theta \propto s^{n/(n+1)}$$

This means $\theta \to 0$ as $s \to 0$ — there is **no finite
equilibrium**. The boundary layer thickness vanishes at the stagnation
point under a turbulent profile.

!!! important "Resolution"
    The stagnation-point boundary layer is **always laminar**. The
    turbulent power-law profile should only be applied **after
    transition**:

    1. Start at stagnation with a laminar profile (Thwaites, Falkner–Skan,
       or Pohlhausen)
    2. March downstream using the laminar closure
    3. At the predicted transition point (Michel criterion or $e^N$),
       switch to the turbulent power-law profile
    4. The laminar $\theta$ at transition becomes the initial condition
       for the turbulent march

    **No stagnation patch is needed for the power-law profile.** The
    implementation raises `NotImplementedError` from `stagnation_theta()`.

## Velocity Field Reconstruction

### Recovering $\delta$

$$\boxed{\delta = \frac{(n+1)(n+2)}{n}\,\theta}$$

For $n = 7$: $\delta = (72/7)\,\theta \approx 10.286\,\theta$.

### Reconstruction Procedure

At each station $s_i$:

1. $\delta_i = \frac{(n+1)(n+2)}{n}\,\theta_i$
2. For any wall-normal distance $y$:

$$u_i(y) = \begin{cases}
U_{e,i} \cdot \left(\dfrac{y}{\delta_i}\right)^{1/n} & y \le \delta_i \\
U_{e,i} & y > \delta_i
\end{cases}$$

### Consistency Check

From the solver's $H$, the exponent can be recovered:

$$n = \frac{2}{H - 1}$$

For $H = 1.2857$: $n = 2/0.2857 = 7.0$. This should match the fixed
input $n$.

## References

1. Schlichting, H. and Gersten, K., *Boundary-Layer Theory* (8th ed.), §21.2.
2. White, F.M., *Viscous Fluid Flow* (3rd ed.), §6-5.
