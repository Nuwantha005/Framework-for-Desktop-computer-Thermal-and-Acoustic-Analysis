# Thwaites' Method

Thwaites' method is a simplified integral method for laminar boundary
layers. Instead of solving the Von Kármán ODE directly, it provides a
**closed-form quadrature** for $\theta(s)$ and uses empirical
correlations for $H$ and $c_f$.

## Quadrature Formula

$$\theta^2(s) = \frac{0.45\,\nu}{U_e(s)^6}
\int_0^s U_e(s')^5\,ds'$$

This computes $\theta(s)$ **without solving the ODE at all**, making it
very popular in panel-method boundary layer solvers.

### Origin of the Constants

Thwaites showed that the momentum integral equation for any Falkner–Skan
profile can be written as:

$$U_e\,\frac{d(\theta^2)}{ds} + a\,U_e'\,\theta^2 = b\,\nu$$

where $a$ and $b$ are not truly universal constants — they vary slightly
between profiles. Thwaites found the best average values $a = 6$,
$b = 0.45$ by fitting all Falkner–Skan solutions simultaneously. The
method is accurate to about 3–5% across the range, which is acceptable
for engineering purposes.

## Thwaites Parameter

$$\lambda = \frac{\theta^2}{\nu}\,\frac{dU_e}{ds}$$

The closure correlations are functions of $\lambda$.

## Closure Relations

### White's Correlation

$$S(\lambda) = (\lambda + 0.09)^{0.62}$$

### Cebeci–Bradshaw Correlations

$$S(\lambda) \approx 0.22 + 1.57\lambda - 1.80\lambda^2$$

$$H(\lambda) \approx 2.61 - 3.75\lambda - 5.24\lambda^2$$

These were obtained by fitting data from many Falkner–Skan solutions.

## Stagnation Patching

With $U_e = Ks$, the quadrature formula gives:

$$\theta^2 = \frac{0.45\,\nu}{K^6 s^6}
\int_0^s K^5 s'^5\,ds'
= \frac{0.45\,\nu}{K^6 s^6} \cdot \frac{K^5 s^6}{6}
= \frac{0.45\,\nu}{6K}$$

$$\boxed{\theta^2_\text{stag} = \frac{0.075\,\nu}{K}}$$

This is constant (independent of $s$). At $s = 0$ exactly, the formula
is $0/0$; applying L'Hopital's rule confirms the limit:

$$\lim_{s\to 0}\frac{0.45\nu\int_0^s U_e^5\,ds'}{U_e^6}
= \lim_{s\to 0}\frac{0.45\nu\,U_e^5}{6\,U_e^5\,U_e'}
= \frac{0.45\nu}{6K} = \frac{0.075\nu}{K} \;\checkmark$$

**Thwaites parameter at stagnation:**

$$\lambda = \frac{\theta^2}{\nu}\,\frac{dU_e}{ds}
= \frac{0.075\nu}{K\nu}\cdot K = 0.075$$

**Closure values at $\lambda = 0.075$:**

Using White's correlation: $S(0.075) = (0.165)^{0.62} \approx 0.327$

Using Cebeci–Bradshaw: $S(0.075) \approx 0.328$, $H(0.075) \approx 2.300$

### Comparison with Exact Stagnation Values

| Quantity | Falkner–Skan exact ($\beta = 1$) | Thwaites at $\lambda = 0.075$ |
|----------|----------------------------------|-------------------------------|
| $\theta^2/(\nu/K)$ | 0.0855 | 0.0750 |
| $H$ | 2.216 | $\approx 2.30$ |
| $S$ | 0.360 | $\approx 0.33$ |

The discrepancy exists because the constants 0.45 and 6 are best-fit
values over the *entire* range of Falkner–Skan profiles — they are not
exact at any single point.

## Velocity Field Reconstruction

Thwaites' method provides only integral quantities ($\theta$, $H$, $S$)
— it does **not** define an explicit velocity profile shape.

To reconstruct the velocity field, Thwaites must be **paired with a
profile family**. Two options are implemented:

### Option A: Falkner–Skan Pairing (Recommended)

Since Thwaites' correlations are *derived from* Falkner–Skan solutions,
the natural reconstruction is:

1. From the solver's $H_i$, use the Falkner–Skan table to find the
   equivalent $\beta_i$ via $\beta = H^{-1}(H)$
2. Use the Falkner–Skan profile for that $\beta_i$ as described in the
   [Falkner–Skan reconstruction](falkner_skan.md#velocity-field-reconstruction)

This is the default pairing (`thwaites_reconstruction: "falkner_skan"`
in the case YAML).

### Option B: Pohlhausen Pairing

Given $H_i$ from Thwaites:

1. Solve the quadratic for $\Lambda_i$ as described in the
   [Pohlhausen reconstruction](pohlhausen.md#velocity-field-reconstruction)
2. Use the Pohlhausen polynomial

This is less physically motivated but requires no ODE solutions — only
polynomial evaluation. Selected via `thwaites_reconstruction: "pohlhausen"`.

## References

1. Thwaites, A., "Approximate calculation of the laminar boundary layer," *Aero. Quarterly*, Vol. 1, 1949, pp. 245–280.
2. White, F.M., *Viscous Fluid Flow* (3rd ed.), §4-6.3, Table 4-4.
3. Cebeci, T. and Bradshaw, P., *Physical and Computational Aspects of Convective Heat Transfer*, §4.3.
