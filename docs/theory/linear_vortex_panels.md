# Linear Vortex Panels

## 1. Introduction
The linear-strength vortex panel method is a higher-order numerical technique for potential flow. While predominantly employed for lifting bodies making use of a Kutta condition, it can be seamlessly adapted for non-lifting closed bluff bodies (such as cylinders or sharp-edged rectangles) where flow is presumed completely attached, wake phenomena are neglected, and no Kutta shedding condition is applied.

By enforcing a continuous vortex distribution $\gamma(s)$ along the geometry, the abrupt discontinuity inherently present at panel boundaries with constant-strength elements is fundamentally resolved. This continuity notably improves velocity distribution accuracy specifically near adjacent panel borders. For a standard discretization configured with $N$ flat panels, the sequential segment endpoints (nodes) directly constitute the domain boundary. Because panel limits share nodes, evaluating a completely interconnected boundary mandates $N+1$ unknown nodal vortex strengths: $\gamma_1, \gamma_2, \dots, \gamma_{N+1}$.

## 2. Theoretical Formulation

### Singularity Element Strength
In the localized panel reference frame (with its geometric origin situated at the first leading node, aligning the $x$-axis physically along the panel itself), the vortex intensity fluctuates linearly completely across the panel's chordal length $S$:

$$
\gamma(x_{loc}) = \gamma_A + \frac{\gamma_B - \gamma_A}{S} x_{loc} = \gamma_A \left( 1 - \frac{x_{loc}}{S} \right) + \gamma_B \left( \frac{x_{loc}}{S} \right)
$$

where:
* $\gamma_A$ = Vortex strength exactly situated at the beginning, leading node ($x_{loc} = 0$)
* $\gamma_B$ = Vortex strength positioned smoothly at the trailing node ($x_{loc} = S$)

### Induced Velocity Components
Through foundational Biot-Savart integration outlined in Katz & Plotkin (*Low-Speed Aerodynamics*, Equations 10.72 and 10.73), the analytical induced velocity components $(u_{loc}, w_{loc})$ evaluated at an arbitrary fluid point $P(x_{loc}, y_{loc})$ can be linearly split separating effects induced distinctly by the leading ($\gamma_A$) and trailing ($\gamma_B$) nodal intensities:

$$
u_{loc} = u_{loc}^a \gamma_A + u_{loc}^b \gamma_B
$$

$$
w_{loc} = w_{loc}^a \gamma_A + w_{loc}^b \gamma_B
$$

Synthesizing both the underlying constant aspect alongside linear ramping factors translates into exact fractional influence components:

**For node A (leading node influence):**

$$
u_{loc}^a = \frac{S - x_{loc}}{2\pi S} \Delta\theta + \frac{y_{loc}}{2\pi S} \ln \frac{r_1}{r_2}
$$

$$
w_{loc}^a = -\frac{S - x_{loc}}{2\pi S} \ln \frac{r_1}{r_2} - \frac{1}{2\pi} + \frac{y_{loc}}{2\pi S} \Delta\theta
$$

**For node B (trailing node influence):**

$$
u_{loc}^b = \frac{x_{loc}}{2\pi S} \Delta\theta - \frac{y_{loc}}{2\pi S} \ln \frac{r_1}{r_2}
$$

$$
w_{loc}^b = - \frac{x_{loc}}{2\pi S} \ln \frac{r_1}{r_2} + \frac{1}{2\pi} - \frac{y_{loc}}{2\pi S} \Delta\theta
$$

**Geometric Variables:**
* $r_1 = \sqrt{x_{loc}^2 + y_{loc}^2}$
* $r_2 = \sqrt{(x_{loc} - S)^2 + y_{loc}^2}$
* $\theta_1 = \tan^{-1}\left(\frac{y_{loc}}{x_{loc}}\right)$
* $\theta_2 = \tan^{-1}\left(\frac{y_{loc}}{x_{loc} - S}\right)$
* $\Delta\theta = \theta_2 - \theta_1$

### Global Coordinates Transformation
Velocity influences must be rotated from the panel's aligned $(x_{loc}, y_{loc})$ reference structure back to the global $(x, y)$ arrangement to satisfy collective normal/tangential boundaries. Using standard rotational projection for an analyzed angle $\phi$:

$$
\begin{pmatrix} u \\ w \end{pmatrix}^a_{global} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} u_{loc}^a \\ w_{loc}^a \end{pmatrix}
$$

*(Analogous rotation applies simultaneously for the separated $b$ element metrics).*

## 3. Assembling the System Matrix Constraints

For an assembled completely closed loop containing $N$ bounded flat panels forming an impermeable structure, fluid physics dictate a zero normal flux boundary essentially mapping at $N$ distinct collocation points conventionally localized at individual panel midpoints. Mathematically, this establishes $N$ linear boundary equations cross-linking the $N+1$ unknown strengths ($\gamma_1, \dots, \gamma_{N+1}$).

The zero boundary normal velocity fundamental condition dictates:

$$
\sum_{k=1}^{N+1} A_{i,k} \, \gamma_k = - \vec{V}_{\infty} \cdot \vec{n}_i \quad \text{for each collocation point } i
$$

### Extracting System Closure (Zero Net Circulation vs. Geometric Overlap)

Matrix viability definitively requires equating unknown degrees with governing algebraic dimensions. Structurally, we require closure constraints addressing $N$ normal equations matched against $N+1$ interrelated unknown coefficients:

1. **Geometric Identity Overlap Constraint:**
   By logical definition representing a closed body geometry, the ultimate node directly maps the starting node coordinates structurally enforcing strictly:
   
   $$
   \gamma_1 = \gamma_{N+1}
   $$

   However, exploiting this single structural constraint directly substituting elements drops our variable unknowns completely to $N$ independent node points against $N$ defined Neumann evaluations resulting in formulating an unmodified $N \times N$ matrix. Uniquely localized solving linear combinations of completely enclosed pure vortex sheets intrinsically generates null spaces since uniformly blanketed vortex layers effectively induce pure zero normal velocity throughout the enclosed surface. The resultant unmodified $N \times N$ formulation inevitably produces a mathematically singular ill-posed matrix lacking single resolution uniqueness.

2. **Zero Net Circulation Constraint (Physical Closure):**
   A physically robust deterministic approach inherently must eliminate structural null spaces substituting a strictly unambiguous non-lifting circulation constraint bypassing wake-induced Kuttas. Total integral circulation inherently spans:
   
   $$
   \Gamma = \sum_{j=1}^N \frac{\gamma_j + \gamma_{j+1}}{2} S_j = 0
   $$

**Summary of the Definitive Matrix Closure:**
Between these constraining options, effectively assigning the **Zero Net Circulation** ($\Gamma = 0$) physical metric uniquely dictates absolute determinable framework closure eliminating arbitrary constants. For reliable programmable solvers mimicking closed-loop boundary geometries, we explicitly incorporate geometric connectivity mapping $\gamma_1 \equiv \gamma_{N+1}$, collapsing functional geometric influences. Consequently, to mathematically stabilize the collapsed redundant singular $N \times N$ matrix, solvers definitively bypass/replace one redundant localized panel normal condition constraint—or systematically append it projecting numerically deterministic least squares equations—structurally constrained exclusively around validating the integral non-lifting circulation requirement $\Gamma = 0$.