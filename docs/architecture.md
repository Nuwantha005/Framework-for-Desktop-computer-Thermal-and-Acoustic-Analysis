# Architecture

## Module Overview

The solver is organized into five major modules, each with a clear responsibility boundary.

```
┌─────────────────────────────────────────────────────┐
│                    Case YAML                        │
│              cases/<name>/case.yaml                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   core.io                           │
│  CaseLoader → parses YAML → creates Scene + Config │
│  Case → unified wrapper with convenience methods   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               core.geometry                         │
│  GeometryFactory → generators → Mesh               │
│  Component(Mesh + Transform) → Scene.assemble()    │
└────────────────────┬────────────────────────────────┘
                     │ Mesh
                     ▼
┌─────────────────────────────────────────────────────┐
│                  solvers                            │
│  SolverFactory → SourcePanelSolver                  │
│  solve() → influence matrices → Aσ = b → Vt, Cp   │
└────────┬───────────────────────────────┬────────────┘
         │ surface_velocity              │ velocity_at(points)
         ▼                               ▼
┌─────────────────────┐   ┌──────────────────────────┐
│   postprocessing    │   │   visualization          │
│  ProcessorPipeline  │   │  VelocityField2D         │
│  → pressure, Cp,    │   │  → grid computation      │
│    φ, ψ, ω          │   │  Visualizer              │
└─────────────────────┘   │  → contours, streamlines │
                          │  ComparisonVisualizer    │
                          └──────────────────────────┘
```

## Data Flow

1. **Case Loading**: A YAML file defines the simulation (freestream velocity, components with parametric geometry, solver type, visualization domain, fluid properties). `CaseLoader` parses this and creates a `Scene` containing `Component` objects, each wrapping a `Mesh` with a `Transform`.

2. **Mesh Assembly**: `Scene.assemble()` applies transforms and merges all component meshes into a single `Mesh` with `component_ids` tracking which panels belong to which body.

3. **Solving**: The assembled `Mesh` is passed to a solver (currently `SourcePanelSolver`). The solver computes influence coefficient matrices, assembles and solves the linear system $A\sigma = b$, then recovers surface velocities and pressure coefficients.

4. **Field Computation**: `VelocityField2D` creates a structured grid over the visualization domain and evaluates the velocity at each grid point using the solver's `velocity_at()` method. Per-component body masking prevents artifacts inside solid bodies.

5. **Post-processing**: `ProcessorPipeline` takes the velocity field stored in a `FieldData` container and runs processors in dependency order (topological sort) to compute derived quantities: pressure, $C_p$, velocity potential $\phi$, stream function $\psi$, and vorticity $\omega$.

6. **Visualization**: `Visualizer` provides a matplotlib facade for creating multi-panel figures. `ComparisonVisualizer` adds difference plots and error metrics ($L_2$, $L_\infty$, RMS, MAE).

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| Dataclass containers | Mesh, Component, Scene, Case, FieldData | Immutable-like data holders with computed properties |
| Abstract Base Class | Solver, PanelSolver2D, PostProcessor | Define interface contracts for extensibility |
| Template Method | PanelSolver2D.solve() | Common solve workflow, subclass-specific steps |
| Factory Registry | SolverFactory, GeometryFactory | String → class mapping for config-driven creation |
| Pipeline | ProcessorPipeline | Topological sort of processors by requires/produces |
| Facade | Visualizer, Case | Simplify complex subsystem interactions |

## Array Conventions

All coordinate arrays use shape `(N, 3)` with `z=0` for 2D problems:

- **Nodes**: `float64`, shape `(N_nodes, 3)` — vertex coordinates
- **Panels**: `int32`, shape `(N_panels, 2)` for 2D — index pairs into nodes array
- **Centers/Normals/Tangents**: `float64`, shape `(N_panels, 3)` — computed panel geometry
- **Areas/Lengths**: `float64`, shape `(N_panels,)` — panel sizes
- **Freestream**: `float64`, shape `(3,)` — velocity vector

This convention simplifies future extension to 3D while keeping 2D code consistent.
