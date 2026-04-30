# Prompt: Implement a simple actuator disk model (ADM) to work alongside panel method
## Introduction
A simple actuator disk model that simulates the pressure jump of an cooling fan has to be implemented to work alongside the 3D panel method sovler. The solver should be able to place multiple actuator disks with different performance and dimensions on arbritary locations.
## Theoritical Background
A Basic guide on how this should be implemented is noted at `notes_archived/ADM_Details/Potential Flow ADM Coupling Search.md` file.
## Data Extraction
- For each case, the file `data/fan_curve.csv` contains the P-Q curve of the fan in question, with the format, `Flow rate (m^3/s),	Ps Static Pressure (Pa)` excluding top row which containts the headings.
- Fan data needes to be interpolated during the operation. Use optins piecewise linear and cubic spline, which can be specified in the case file.
- The other relevant details of each fan should be extracted from `case.yaml` file. Since multiple fans can be incoperated, use a sophisticated format with fan dimensions, operation conditions,  position and orientation and other details.
## Validation
The primary test cases is `cases/cicular_vent` where a fan should be located in the middle of the duct. This case has a P-Q curve in the relevant directory and the diameter is 120mm. Fill the other details as placehoders related to fan for now. I will update correct ones later.
## Tasks
- Create a sophisticated, multi step plan, and save it to `.agent/plans` folder.
- Execture the plan one by one, and mark the progress in `.agent/TASK_LOG.md`.
- Document the theritical aspects on the implementation in `docs/theory` folder.
- Document the technical details in `docs/modules` and anywhere else needed.
- Add short descriptions in `.agent/modules/solver.md` for future agentic references.
---
## ADM Formulation (Potential Flow Coupling)
- Treat the actuator disk as an infinitesimally thin surface with a **prescribed pressure jump** across it.
- Implement the disk as a **doublet sheet** representing a **potential jump** across the disk.
- In steady state, treat the doublet strength (potential jump) as prescribed by Δp.
- Disk panels are **not** enforced with zero normal flow; their influence is moved to the RHS of the body-panel system as a known disturbance.
## Disk Discretization
- Use a **polar grid** (radial + azimuthal subdivisions) for the disk surface.
- Allow per‑fan resolution control (n_r, n_theta) in case.yaml.
## Fan Curve Coupling (P‑Q Iteration)
- Use the fan curve directly (no manufacturer rig corrections).
- Iterative coupling loop:
  1) Initialize Δp (from curve midpoint or shut‑off).
  2) Solve the panel flow with that Δp.
  3) Compute Q = ∫ V·n dS over the disk.
  4) Lookup Δp_new from the P‑Q curve at Q.
  5) Relax: Δp_next = Δp_old + ω(Δp_new − Δp_old).
  6) Converge on Δp or Q.
- Make it modular so that we can test other convergence techniques later (like gradient based). 
## Fan Configuration in case.yaml
Define a structured list of fans. Example fields to include:
- name
- center (x, y, z)
- normal or axis (x/y/z)
- radius (m)
- n_r, n_theta
- curve_file (relative path to data/fan_curve.csv)
- dp_initial, relaxation, tolerance, max_iterations

> Note: Since these calculations would consume large amount of time, we should have "save" files that would allow us to run visualization and validation scripts without running the solver. for that, create a subdirectory `out/solverRuns`, and save solver run minimum results that can be used to reporoduce data. i suppose these would be singularity values, and anything else thats needed to produce the velocity filed. Then the validation and visualizaton scripts can accept these files and run them. 

## Visualization and Export
- As with generic 3D cases, the scripts in `validation/scripts/3d/` should be used to visualize the results - they will output .vtk files.
- `demos/demo_case_mesh_export.py` should export the actuator disk meshes alongside other surface meshes
- The output results should be modular so that they could be used for other purposes, such as values on the actuator disk, streamlines on cut planes, and animated streamlines using vtk.
- When the iterative solver is running for P-Q curve convergence of the fans, the error and other metrics should be printed to the command line. Furthermore, a convergence plot should be saved in `out/adm` folder.
- All the plots should be saved with clear titles, axis lables, units and legends. Use latex for those when required.
## Validation (Fluent)
- Use Fluent exports as ground truth (see `.agent/prompts/fluent_comparison.md`).
- Add a new validation script under `validation/scripts/3d/` to compare **cut‑plane sections**:
  - Generate the same cut plane in panel+ADM.
  - Load Fluent cut‑plane export (CSV with x,y,z,p,u,v,w).
  - Interpolate Fluent values onto the panel cut grid.
  - Plot **pressure/velocity along duct axis** on a single plot (Fluent vs panel+ADM).
  - Validation plots and exports should be saved to `out/validation` folder, and when needed, plots / data should be saved on sub folders for better clarity.  

- Primary metrics:
  - Max pressure rise across the disk
  - Max velocity along axis
  - Airflow rate through disk (Q)
  - Overall result of these values comparison should be printed to the termial.

## Potential Issues (Mention Only)
- **Kutta condition** at duct trailing edge may be needed for correct circulation in ducted fans.
- **Branch wake / edge singularity** at disk rim can cause non‑physical spikes; mitigation may be needed later.