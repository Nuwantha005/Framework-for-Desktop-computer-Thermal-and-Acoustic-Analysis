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
         1,-4.158276732E-16,-8.000000000E+00,-2.188333742E-02, 1.000000000E+00, 0.000000000E+00
         2, 2.770342647E-02,-8.000000000E+00,-2.206055188E-02, 1.005299157E+00, 0.000000000E+00
         3, 5.580152667E-02,-8.000000000E+00,-2.223045663E-02, 1.005445430E+00, 0.000000000E+00
```
**wall_data:**
```
nodenumber,    x-coordinate,    y-coordinate,      wall-shear
         1,-7.753743520E-03,-4.999398758E-01, 1.348020168E-03
         2, 8.036596938E-03,-4.999354089E-01, 1.711057191E-03
         3,-2.361364641E-02,-4.994420844E-01, 5.989121072E-04
```
## Calculations
Note that all the boundary layer sovlers ends before reaching rear stagnation point. Therefore, we only compare the velocities and other parameters only in the places where the BL solver results exist.

### Velocity Maginitude
The boundary layer solver outpus the dataClass `BLFieldData` which is located in `src/solvers/boundary_layer/field.py` file. This returns the values in (s,y) local coordinate system. We need to transform these to global system and then use those coordinates as the interpolation inputs from fluent data.

Critical Note: We need to use the mesh points from boundary layer solver as the reference and interpolate fluent results for those points, not the other way around.

The boundary layer solver returns the **Tangential Velocity**. Fluent gives the velocity vector. We have to use the specific panel, related to the boundary layer point $s$ and convert that fluent velocity vector into the tangential component. Ideally, it should be tangential anyway. We can use the normal vector of the panel in question for this purpose.

###