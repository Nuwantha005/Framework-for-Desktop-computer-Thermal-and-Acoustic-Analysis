# Linear Source Panels

## 1. Introduction
The linear-strength source panel method is a higher-order numerical technique for potential flow over non-lifting configurations. Modifying the constant source element formulation, it enforces a completely continuous source distribution $\sigma(s)$ along the structural geometry, largely bypassing the discontinuity at panel boundaries that typically induces localized errors in tangential velocity ($V_t$).

For a discretization spanning $N$ flat panels, the points separating them (nodes) constitute the fluid boundary. Because of enforced continuity, instead of $N$ unknown panel-wise constants, there are $N+1$ unknown nodal source strengths: $\sigma_1, \sigma_2, \dots, \sigma_{N+1}$.

## 2. Theoretical Formulation

### Singularity Element
In the local panel reference frame (origin centered at the leading node, with the positive local x-axis aligned with the panel itself), the source strength naturally varies linearly across the panel's length $S$:

$$
\sigma(x_{loc}) = \sigma_A + \frac{\sigma_B - \sigma_A}{S} x_{loc}
$$

where:
* $\sigma_A$ = Source strength exactly at the first node ($x_{loc} = 0$)
* $\sigma_B$ = Source strength exactly at the second node ($x_{loc} = S$)

### Induced Velocity Components
Through fundamental integration (following the Katz & Plotkin "Low-Speed Aerodynamics" derivations from Chapters 10 and 11), the analytically determined induced velocity $(u_{loc}, w_{loc})$ evaluated at a generic geometric point $P(x_{loc}, y_{loc})$ can be linearly split into distinct combinations of effects from the leading ($\sigma_A$) and trailing ($\sigma_B$) nodal coefficients:

$$
u_{loc} = u_{loc}^a \sigma_A + u_{loc}^b \sigma_B
$$

$$
w_{loc} = w_{loc}^a \sigma_A + w_{loc}^b \sigma_B
$$

The rigorous algebraic expansions for these partial influence coefficients are derived as follows:

**For node A (leading node influence):**

$$
u_{loc}^a = \frac{S - x_{loc}}{2\pi S} \ln \frac{r_1}{r_2} + \frac{1}{2\pi} - \frac{y_{loc}}{2\pi S} \Delta\theta
$$

$$
w_{loc}^a = \frac{S - x_{loc}}{2\pi S} \Delta\theta + \frac{y_{loc}}{2\pi S} \ln \frac{r_1}{r_2}
$$

**For node B (trailing node influence):**

$$
u_{loc}^b = \frac{x_{loc}}{2\pi S} \ln \frac{r_1}{r_2} - \frac{1}{2\pi} + \frac{y_{loc}}{2\pi S} \Delta\theta
$$

$$
w_{loc}^b = \frac{x_{loc}}{2\pi S} \Delta\theta - \frac{y_{loc}}{2\pi S} \ln \frac{r_1}{r_2}
$$

**Geometric Variables:**
* $r_1 = \sqrt{x_{loc}^2 + y_{loc}^2}$
* $r_2 = \sqrt{(x_{loc} - S)^2 + y_{loc}^2}$
* $\theta_1 = \tan^{-1}\left(\frac{y_{loc}}{x_{loc}}\right)$
* $\theta_2 = \tan^{-1}\left(\frac{y_{loc}}{x_{loc} - S}\right)$
* $\Delta\theta = \theta_2 - \theta_1$

### Global Coordinates Transformation
Velocity influences must be rotated from the target panel's aligned frame back to the global $(x, y)$ coordinate system before they are projected across boundary normals/tangents. For an evaluated given orientation angle $\phi$:

$$
\begin{pmatrix} u \\ v \end{pmatrix}^a_{global} = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} u_{loc}^a \\ w_{loc}^a \end{pmatrix}
$$

*(Identical rotation logic simultaneously applies for the $b$ element matrices).*

## 3. Assembling the System Matrix
For an assembly of $N$ contiguous panels, defining an impermeable flow boundary dictates zero normal fluxes at $N$ distinct collocation points (ordinarily the geometrical center of each successive panel). This mathematically yields $N$ linear equations referencing $N+1$ variables ($\sigma_1, \dots, \sigma_{N+1}$).

The zero normal velocity condition essentially constructs:

$$
\sum_{k=1}^{N+1} A_{i,k} \, \sigma_k = - \vec{V}_{\infty} \cdot \vec{n}_i \quad \text{for each collocation point } i
$$

The composite coefficient $A_{i,k}$, signifying the complete aerodynamic influence of continuous node $k$ on fixed collocation point $i$, is formed structurally by accumulating trailing and leading elements simultaneously:

* **Node 1 ($k=1$):** Takes contribution solely from the leading sub-influence ($a$-component) of Panel 1.
* **Inner Nodes ($1 < k \leq N$):** Intercepts combining influence from the trailing segment ($b$-component) of preceding Panel $k-1$ overlaid with the leading segment ($a$-component) of current Panel $k$.
* **Final Node ($k=N+1$):** Comprised structurally from the trailing effect ($b$-component) of the ultimate terminal Panel $N$.

### Boundary Condition Constraint Equation
Matrix solution mandates matching equation density ($N \times N+1 \rightarrow \text{unsolvable}$). A secondary supplementary criteria provides strict system closure:

1. **Stagnating Fix:** Setting $\sigma_1 + \sigma_{N+1} = 0$, guaranteeing a balanced aerodynamic stagnation point mimicking theoretical shedding for a definitive wedge structure.
2. **Looped Mesh Equivalent Constraint:** If perfectly evaluating a completely continuous loop outline (e.g. cylinder), ensuring $\sigma_1 \equiv \sigma_{N+1}$ strictly overlaps boundaries and organically reduces solving density strictly equivalent to an integrated $N \times N$ form.
