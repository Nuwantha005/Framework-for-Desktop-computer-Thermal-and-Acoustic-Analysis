# Prompt: Refactorize the Boundary Layer Solver


## Content
Several adjustments related to how the boundary layer solver works near the stagnation point are outlined here. The workings of the current implementation are as follows.

### Current Structure

1. Identify the panels with smallest tangential velocity using the criterion $|U_e| > 0.01 \max|U_e|$
2. Of those, identify the flow facing stagnation point using the criterion $\mathbf{U}_\infty \cdot \mathbf{n} < 0$
3. Start the marching from this point onwards along each streamline. For current 2D cases, only 2 streamlines exists and the are reffered as _upper_ and _lower_ in the code.

However, this results in several panels being missing and no velocity filed evaluation for those panels. To fix this, it has been decided to do following changes.

### Pinpointing the Exact stagnation point
Instead of just gathering all panels where $U_e < 0.01 U_{max}$, look for where the surface tangential velocity **changes sign** (or where $\mathbf{U}_\infty \cdot \mathbf{n}$ reaches its lowest/negative value and streamlines diverge). We can interpolate between the two adjacent panels to find the exact point where $s=0$ and $U_e = 0$. Note that the way panels are arranged in the mesh affects how this should be handled. Sometimes, the panels may start and end at the stagnation point.

### Analytical Stagnation Patching
Near the stagnation point, the velocity grows linearly,
$$U_e(s) \approx K \cdot s \quad \text{where} \quad K = \left. \frac{dU_e}{ds} \right|_{stagnation}$$

Now what we need to do is apply this linear velocity profile into the Von Kármán Momentum Integral and apply L'Hôpital's rule as $s\to 0$ to find the exact value of Momentum Thickness at the stagnation point. How this is done is dependent on the velocity profile we use.

The relevant procedures to handle stagnation point for each velocity profile has been outlined in its relevant note at `/notes_archived/bl_solvers/velocity_profiles/`. In each note, see the section **Limit at the Stagnation Point** for specific instructions.

### Velocity Gradient across the surface
We need the value of $K$ for downstream calculations. For that, we can use all the panels that matches our condition ($U_e < 0.01 U_{max}$) and run a linear regression on them to find the value of $K$. If we use linear regression,
$$K = \frac{\displaystyle\sum_i U_{e,i}\,(s_i - s_\text{stag})}{\displaystyle\sum_i (s_i - s_\text{stag})^2}$$
Fit each branch separately. You can use standard numerical libraries for this. For a body at angle of attack, the velocity gradients on the upper and lower surfaces differ. We will get $K_{upper}$ and $K_{lower}$ independently. But more generally, $K$ should be calculated for each streamline seperately.

##  Boundary Layer Post Processing
The results from viscous solvers are given in the form of `BoundaryLayerResult` dataclass, which contains integral specific data such as momentum and displacement thickness, but not ones we are interested in such as boundary layer thickness velocity values. To obtain them, we need to post proces the boundary layer solver results and the way it works depends on the profile we used. 

The profile specific processes and other relevant details are located in the boundary layer solver notes at `/notes_archived/bl_solvers/velocity_profiles/`. See the section **Post Processing** in each note for specific instructions.

### Tabulated values
In several of the velocity profiles, a specific ODE is involved and the solutions to them are given in tabular format in classic physics textbooks. Here what we are going to do is write one time run scripts that generates these tables. After generation we can store them in a directory such as `data/bl-solver-profiles` in a format like JSON. 

The specific functions that generates these tables should be stored somewhere in the specific profile file(s) and one time run script(s) should be implemented to generate these scripts. Then during runtime, these tables can be looked up and interpolated to find these specific values. If the look-up value is out of the range in the table, then the function that was used to generate the table can be solved in on the fly and get an answer (failsafe method).

## Visualization

It was decided to use the velocity magnitute as the validation comparison metric because its what is being passed down to the thermal solver. Now, if i have an result from openfoam, i can compare the values. I need following visualization techniques,

1) Boundary layer thickness envelope plot
2) Take boundary layer velocity filed and transform it into a 2d contour plot where x axis is the arc length and the y axis is the coordinate normal to the the surface. Then we can plot the velocity value as a contour plot, with upper bound of the contour plot being the layer height.
3) We can get RMS difference between our bl solver result and simpleFOAM and plot the difference in plot (2) and get an idea of where the most error is.
4) We can normalize the y coordinate of the the plot mentioned in (2) and get a uniform rectangle looking contour plot, which can be used to visualize the thin parts of the BL better.
5) We can wrap the (2) plot around the object and create a envelope plot, which is just (1) but with envelope coloured based on the velocity values.
6) For plot (2) and related ones, we can add a parameter to show some fraction of free stream values as well like like 1.2δ, to see the blend into the freestream.
7) We need to do all these plottings for each streamline, and in this 2D case - upper and lower streamlines



