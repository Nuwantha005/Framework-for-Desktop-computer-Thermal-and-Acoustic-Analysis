# Actuator Disk Model (ADM) Implementation Plan

**Created**: 2026-04-30  
**Status**: Planned  
**Scope**: Simple pressure-jump actuator disk model coupled to configurable 3D panel solvers.

## Goals

- Add one or more cooling-fan actuator disks to 3D cases.
- Read fan P-Q curves from per-case CSV files.
- Iterate each fan operating point until pressure rise and flow rate converge.
- Couple ADM to the configured 3D panel solver without hard-coding a specific singularity type.
- Save reusable solver-run artifacts so visualization and validation can run without recomputing.
- Prepare Fluent comparison tooling, but do not run validation until Fluent data is available.

## Key Architectural Decision

ADM will be implemented as a 3D solver coupling layer, not as a special case inside
`SourcePanelSolver3D`.

The coupling layer should create the body panel solver through `SolverFactory` using
the case `solver` config:

```text
case.yaml solver config
  -> SolverFactory.create(...)
  -> configured PanelSolver3D subclass
  -> ActuatorDiskCoupledSolver3D wraps/couples it
```

This keeps the ADM compatible with future 3D solvers such as linear source,
doublet, or source-doublet variants. The first concrete implementation will be
verified with the existing constant-source 3D solver.

## Physical Model

- The disk is an infinitesimally thin pressure-jump surface.
- The disk orientation is defined by `normal` in `case.yaml`.
- Positive flow rate is:

```text
Q = integral(V dot n_disk dS)
```

- Positive pressure rise follows the same normal direction.
- The actuator disk is represented by a prescribed doublet sheet because a
  doublet distribution represents a potential jump across a surface.
- The steady pressure-jump to potential-jump mapping must be documented
  explicitly in `docs/theory/actuator_disk_model.md`.
- The implementation should isolate this mapping in one function so it can be
  replaced if the theoretical convention is refined.
- Disk panels are not solved with zero-normal-flow boundary conditions. Their
  influence is treated as a known disturbance on the body-panel system RHS.

## Case YAML Schema

Add optional top-level `actuator_disks` section:

```yaml
actuator_disks:
  - name: "fan_120mm"
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    radius: 0.06
    n_r: 6
    n_theta: 48
    curve_file: "data/fan_curve.csv"
    interpolation: "linear"  # linear | cubic
    dp_initial: null
    relaxation: 0.4
    tolerance: 1.0e-3
    max_iterations: 50
```

Notes:
- `normal` defines both orientation and positive flow direction.
- `curve_file` is relative to the case directory.
- If `dp_initial` is omitted, initialize from a robust fan-curve value
  such as shutoff pressure or curve midpoint.
- Use Pydantic validation for positive radius, valid resolution, normalized
  nonzero normal, and interpolation choice.

## Module Layout

Create:

```text
src/solvers/actuator/
  __init__.py
  fan_curve.py
  disk_mesh.py
  models.py
  doublet_influence.py
  coupling.py
  persistence.py
  plotting.py
```

Responsibilities:

- `fan_curve.py`
  - Load `Flow rate (m^3/s), Ps Static Pressure (Pa)` CSV.
  - Support piecewise-linear and cubic interpolation.
  - Clamp or warn outside tabulated flow-rate range.

- `disk_mesh.py`
  - Generate polar quad disk meshes.
  - Support arbitrary center and normal.
  - Preserve `(N, 3)` coordinates and quad connectivity.
  - Compute panel centers, normals, and areas.

- `models.py`
  - Dataclasses for disk config runtime state, convergence history, and results.

- `doublet_influence.py`
  - Compute induced velocity from constant-strength disk doublet panels.
  - Compute normal-velocity disturbance at body panel control points.
  - Keep formulas and approximations documented.

- `coupling.py`
  - `ActuatorDiskCoupledSolver3D`.
  - Creates configured body solver through `SolverFactory`.
  - Runs P-Q iteration.
  - Exposes `surface_velocity`, `velocity_at(points)`, `mesh`, and ADM results.

- `persistence.py`
  - Save/load minimal solver-run artifacts under `case/out/solverRuns/`.
  - Include panel strengths, disk mesh, disk doublet strengths, final Q, final
    pressure rise, freestream, density, and convergence history.

- `plotting.py`
  - Save convergence plots under `case/out/adm/`.

## Required 3D Solver Interface Work

The existing `PanelSolver3D` interface is good for downstream visualization but
does not yet expose a generic way to add known normal-velocity disturbances to
the RHS.

Add a minimal extension point to `PanelSolver3D`:

```text
solve(normal_velocity_disturbance: NDArray | None = None)
```

or an equivalent protected hook:

```text
_external_normal_velocity: Optional[NDArray]
```

Then body solvers can assemble:

```text
A * strengths = -(V_inf dot n_body + V_adm dot n_body)
```

For the first implementation, update `SourcePanelSolver3D` to honor this hook.
Future 3D solvers should use the same hook, so ADM remains solver-agnostic.

## P-Q Iteration

For each coupled solve:

1. Load fan curves and generate disk meshes.
2. Initialize each disk pressure rise.
3. Convert pressure rise to prescribed disk doublet strengths.
4. Compute disk-induced normal velocity at body panel centers.
5. Solve the configured body panel solver with ADM disturbance on RHS.
6. Compute velocity on disk panels from freestream, body singularities, and disk doublets.
7. Integrate flow rate:

```text
Q = sum((V_disk_center dot n_disk) * area_disk_panel)
```

8. Interpolate fan curve to get updated pressure rise.
9. Apply relaxation:

```text
dp_next = dp_old + relaxation * (dp_curve(Q) - dp_old)
```

10. Print iteration metrics and stop when pressure rise and/or flow rate converges.
11. Save convergence history, final solution bundle, disk VTK, and convergence plot.

## Circular Vent Case

Update `cases/cicular_vent/case.yaml` with one placeholder 120 mm fan:

- `radius: 0.06`
- `center`: middle of the duct
- `normal`: aligned with intended duct flow direction
- `curve_file: data/fan_curve.csv`
- conservative initial ADM resolution such as `n_r: 6`, `n_theta: 48`

The user will later refine fan details.

## Mesh Export and Visualization

- Update `demos/demo_case_mesh_export.py` to optionally export actuator disk meshes.
- Save body mesh under the existing output location.
- Save disk meshes under `case/out/adm/`.
- Disk VTK fields should include:
  - fan name or fan id
  - panel area
  - normal velocity
  - pressure rise
  - doublet strength

## Fluent Validation Preparation

Do not run Fluent validation yet.

Add a future-facing script under `validation/scripts/3d/` for cut-plane comparison:

- Load a saved ADM solver run.
- Generate panel+ADM cut-plane data.
- Load Fluent CSV when available with columns such as `x,y,z,p,u,v,w`.
- Interpolate Fluent data onto panel cut grid.
- Plot pressure and velocity along the duct axis.
- Save plots and metrics under `case/out/validation/`.

Primary metrics:

- pressure rise across disk
- maximum axial velocity
- disk flow rate Q

## Documentation

Create/update:

- `docs/theory/actuator_disk_model.md`
  - pressure jump, doublet sheet, orientation convention, P-Q iteration,
    limitations, and pressure-to-potential convention.
- `docs/modules/solver.md`
  - ADM module layout and coupled solver workflow.
- `.agent/modules/solver.md`
  - compact future-agent notes.
- `.agent/TASK_LOG.md`
  - append progress after each implementation session.

## Tests and Checks

Add focused tests:

- fan curve CSV loading and interpolation.
- disk mesh panel count and total area close to `pi * radius^2`.
- disk normal orientation from arbitrary input normal.
- single-fan synthetic P-Q convergence smoke test.
- `SourcePanelSolver3D` still solves existing sphere/cylinder cases without ADM.

Run local tests only; skip Fluent validation until data exists.

## Implementation Sequence

1. Add schema models and config loading for `actuator_disks`.
2. Implement fan curve loader/interpolator.
3. Implement disk mesh generator.
4. Add ADM result dataclasses.
5. Add 3D solver RHS disturbance hook and update `SourcePanelSolver3D`.
6. Implement doublet influence helpers.
7. Implement `ActuatorDiskCoupledSolver3D`.
8. Add persistence and convergence plotting.
9. Update circular vent case.
10. Update mesh export demo.
11. Add docs and module notes.
12. Add focused tests and run non-Fluent checks.

## Deferred Items

- Kutta condition at duct trailing edge.
- Branch wake / rim singularity mitigation.
- Nonuniform radial loading.
- Swirl.
- Wake roll-up or slipstream contraction.
- Fluent validation execution.
