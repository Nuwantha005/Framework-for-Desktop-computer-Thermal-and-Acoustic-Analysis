# Promp : Implement the fluent comparison pipeline

The current validation pipeleine with code in the folder `validation/` was used to compare the results from panel method solver to poentialFOAM. Than includes automatic case generation, mesh convergence test, and final comparison. This folder, and this pipline is deprecated now.

From now on, we use the results exported from Ansys fluent as the ground truth to compare. Panel method comparison is complete. Now the comparison should be done for the boundary layer solver.

## Comparison Metrics
1. Velocity Maginitude
2. Edge Velocity $U_e$
3. Boundary Layer thickness $\delta(s)$
4. Skin Friction Coefficint $C_f$
5. Seperation Point

The specfic calculation methods for each of these are described in the sections below.

## Visualizations
To see the current visualization capabilities of the boundary layer solver, check the old prompt `.agent/prompts/bl-solver-refactorization.md` line 41-51. These features are now implemented. So visualization capabilities for comparison are already there, but th code uses 'OpenFOAM Comparison' instead. Use those functions, for example the function `plot_bl_of_comparison` at line 970 on `src/visualization/bl_plots.py`. These are the final endpoints of this pipeline.

After the implementation, you can directly use the file `demos/demo_boundary_layer.py` for a demo, which already utilizes this function.

## Data Flow
All the files related to fluent are placed at `cases/case_name/fluent_case` folder. The location of the export files are as follows.

```
cases/cylinder_flow/fluent_case
├── export
│   ├── thermal_bl
│   └── viscous_bl
│       ├── filed_data
│       └── wall_data
└── spaceclaim_model.scdoc
```
Both files are exported as ASCII files. Example for their contens are given below.

**filed_data:**
```
nodenumber,    x-coordinate,    y-coordinate,        pressure,      x-velocity,      y-velocity
         1, 8.427639012E-17,-7.500000000E-01,-3.011805813E-01, 1.228020079E+00,-2.323784826E-01
         2,-2.179031490E-02,-7.496833880E-01,-3.005723495E-01, 1.226030280E+00,-2.405907942E-01
         3, 3.256109368E-02,-7.492928501E-01,-3.016362810E-01, 1.230227082E+00,-2.211259240E-01
```
**wall_data:**
```
nodenumber,    x-coordinate,    y-coordinate,        pressure,      wall-shear
         1, 2.220446049E-16,-5.000000000E-01,-4.862246001E-01, 1.522855383E-03
         2, 1.560226457E-02,-4.997565101E-01,-4.530828855E-01, 1.778663425E-03
         3,-1.595709488E-02,-4.997453063E-01,-5.283129445E-01, 8.964168236E-04
```
## Calculations
Note that all the boundary layer sovlers ends before reaching rear stagnation point. Therefore, we only compare the velocities and other parameters only in the places where the BL solver results exist.

### Velocity Maginitude
The boundary layer solver outpus the dataClass `BLFieldData` which is located in `src/solvers/boundary_layer/field.py` file. This returns the values in (s,y) local coordinate system. We need to transform these to global system and then use those coordinates as the interpolation inputs from fluent data.

Critical Note: We need to use the mesh points from boundary layer solver as the reference and interpolate fluent results for those points, not the other way around.

The boundary layer solver returns the **Tangential Velocity**. Fluent gives the velocity vector. We have to use the specific panel, related to the boundary layer point $s$ and convert that fluent velocity vector into the tangential component. Ideally, it should be tangential anyway. We can use the normal vector of the panel in question for this purpose.

> The next few calculations are specific about how to extract quantities from fluent dataset.
### Edge velocity Calculation
We assume incompressible flow, and use the static pressure at the wall to calculate the edge velocity using,
$$U_e = \sqrt{\frac{2(P_{0,\infty} - P_{wall})}{\rho}}$$
Where $P_{0,\infty}$ is the freestream pressure that can be fount in the case file.

### Boundary Layer Thickness
For the selected panel, we march along the normal of the panel untill the tangential velocity magnitude becomes,
$$\delta(s)\approx 0.99 U_e$$

### Skin Friction Coefficint
Since we have data for wall shear stress at wall from fluent, we can calculate
$$C_f = \frac{\tau_w}{\frac{1}{2} \rho U_\infty^2}$$

### Seperation Point
Seperation occurs when,
$$\tau_w = 0$$
So we can march along the wall and find out the arc length at which this happens. All the profiles in the boundary layer solver ends at the seperation point.

## Arc Length Calculations
The boundary layer solver has 2 seperate streamlines for these 2D cases, and it finds the respective panels, divides them into correct streamline, solvers seperately and gives results seperately. Therefore, when dealing with the wall dataset from fluent, we have to use the coordinates of those panels and find the correct streamlines a given data point belongs to. Only after that we can do the calculations.